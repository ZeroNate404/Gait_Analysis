"""
Data layer for the gait event editor.
 
Deliberately free of any Qt import so it can be unit-tested, scripted and
reused headlessly.  Everything the UI needs to know about the ``*_GaitEvents.npz``
file lives here:
 
  * discovery of marker trajectories / scalar metadata / 1-D signals,
  * the mapping between *absolute frame numbers* (how LHS/RHS/LTO/RTO are
    stored) and *array indices* (how the trajectories are stored),
  * an undoable edit model (add / delete / move),
  * validation of the resulting event sequence,
  * saving back to .npz with a timestamped backup, preserving every key,
    dtype and the original compression setting.
"""
 
from __future__ import annotations
 
import bisect
import datetime as _dt
import os
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
 
import numpy as np
 
# --------------------------------------------------------------------------
# Event taxonomy
# --------------------------------------------------------------------------
 
EVENT_KEYS: Tuple[str, ...] = ("LHS", "LTO", "RHS", "RTO")
 
 
@dataclass(frozen=True)
class EventSpec:
    key: str
    side: str  # "L" | "R"
    kind: str  # "HS" | "TO"
    label: str
    short: str
    symbol: str  # pyqtgraph scatter symbol
 
 
#: ``symbol`` follows the physical gesture: heel strike points *down* into the
#: floor, toe off points *up* away from it.
EVENT_SPECS: Dict[str, EventSpec] = {
    "LHS": EventSpec("LHS", "L", "HS", "Left Heel Strike", "L-HS", "t"),
    "LTO": EventSpec("LTO", "L", "TO", "Left Toe Off", "L-TO", "t1"),
    "RHS": EventSpec("RHS", "R", "HS", "Right Heel Strike", "R-HS", "t"),
    "RTO": EventSpec("RTO", "R", "TO", "Right Toe Off", "R-TO", "t1"),
}
 
SIDES: Tuple[str, ...] = ("L", "R")
SIDE_LABEL = {"L": "LEFT", "R": "RIGHT"}
 
 
def keys_for_side(side: str) -> Tuple[str, ...]:
    return tuple(k for k, s in EVENT_SPECS.items() if s.side == side)
 
 
# --------------------------------------------------------------------------
# Marker naming
# --------------------------------------------------------------------------
 
#: Maps the keys used by the upstream pipeline onto conventional marker labels.
MARKER_ALIASES: Dict[str, str] = {
    "sacrum_arr": "SACR",
    "SACR_arr": "SACR",
    "LHarr": "LHEE",
    "LTarr": "LTOE",
    "RHarr": "RHEE",
    "RTarr": "RTOE",
    "LHEE_arr": "LHEE",
    "LTOE_arr": "LTOE",
    "RHEE_arr": "RHEE",
    "RTOE_arr": "RTOE",
}
 
_HEEL_PAT = re.compile(r"(HEE|HEEL|_H_|^LH|^RH)", re.IGNORECASE)
_TOE_PAT = re.compile(r"(TOE|_T_|^LT|^RT)", re.IGNORECASE)
_PELVIS_PAT = re.compile(r"(SACR|PELV|SACRUM|ASIS|PSIS)", re.IGNORECASE)
 
 
@dataclass
class Marker:
    """One 3-D trajectory, shape ``(n_samples, 3)``."""
 
    name: str  # display name, e.g. "LHEE"
    source_key: str  # original npz key, e.g. "LHarr"
    xyz: np.ndarray
    side: str = "C"  # "L" | "R" | "C" (centre / unsided)
    part: str = "other"  # "heel" | "toe" | "pelvis" | "other"
 
    @property
    def n_samples(self) -> int:
        return int(self.xyz.shape[0])
 
 
def _pretty_marker_name(key: str) -> str:
    if key in MARKER_ALIASES:
        return MARKER_ALIASES[key]
    name = re.sub(r"(_arr|_ARR|arr|Arr)$", "", key)
    return name or key
 
 
def _classify_marker(display_name: str, source_key: str) -> Tuple[str, str]:
    """Return ``(side, part)`` inferred from a marker name."""
    probe = f"{display_name} {source_key}"
    if _PELVIS_PAT.search(probe):
        return "C", "pelvis"
 
    side = "C"
    head = display_name.upper()[:1]
    if head == "L":
        side = "L"
    elif head == "R":
        side = "R"
 
    part = "other"
    if _HEEL_PAT.search(display_name) or _HEEL_PAT.search(source_key):
        part = "heel"
    elif _TOE_PAT.search(display_name) or _TOE_PAT.search(source_key):
        part = "toe"
    return side, part
 
 
# --------------------------------------------------------------------------
# Frame <-> index mapping
# --------------------------------------------------------------------------
 
 
@dataclass
class FrameMapping:
    """``index = frame - offset``.
 
    Event frames are stored as *absolute* frame numbers; the trajectory arrays
    are plain 0-based arrays.  ``offset`` is the absolute frame number that
    array index 0 corresponds to.
    """
 
    offset: int
    n_samples: int
    inferred_from: str = "explicit"
    scores: Dict[int, float] = field(default_factory=dict)
 
    @property
    def first_frame(self) -> int:
        return self.offset
 
    @property
    def last_frame(self) -> int:
        return self.offset + self.n_samples - 1
 
    def to_index(self, frame: int) -> int:
        return int(frame) - self.offset
 
    def to_frame(self, index: int) -> int:
        return int(index) + self.offset
 
    def clamp_frame(self, frame: int) -> int:
        return int(min(max(int(frame), self.first_frame), self.last_frame))
 
    def contains(self, frame: int) -> bool:
        return self.first_frame <= int(frame) <= self.last_frame
 
    def describe(self) -> str:
        return (
            f"array[0] = frame {self.offset}  ·  frames "
            f"{self.first_frame}–{self.last_frame}  ({self.inferred_from})"
        )
 
 
def infer_frame_mapping(
    n_samples: int,
    start_frame: Optional[int],
    end_frame: Optional[int],
    events: Dict[str, Sequence[int]],
) -> FrameMapping:
    """Work out which absolute frame array index 0 corresponds to.
 
    Three conventions are plausible and all appear in the wild:
      * ``offset = start_frame``  – arrays cover the analysed bout only,
      * ``offset = 1``            – arrays cover the whole trial, 1-based,
      * ``offset = 0``            – arrays cover the whole trial, 0-based.
 
    Each candidate is scored on (a) whether the implied last frame agrees with
    ``end_frame`` and (b) how many events land inside the resulting window.
    Containment is weighted highest: an offset that puts events outside the
    data is definitionally wrong, whatever the metadata claims.
    """
    all_events = [int(f) for key in EVENT_KEYS for f in events.get(key, ())]
 
    candidates: List[int] = []
    for cand in (start_frame, 1, 0):
        if cand is None:
            continue
        cand = int(cand)
        if cand not in candidates:
            candidates.append(cand)
    if not candidates:
        candidates = [0]
 
    scores: Dict[int, float] = {}
    for cand in candidates:
        lo, hi = cand, cand + n_samples - 1
        score = 0.0
        if all_events:
            inside = sum(1 for f in all_events if lo <= f <= hi)
            score += 10.0 * inside / len(all_events)
        if end_frame is not None:
            delta = abs(hi - int(end_frame))
            if delta == 0:
                score += 3.0
            elif delta <= 1:
                score += 2.0
            elif delta <= 5:
                score += 1.0
        scores[cand] = score
 
    best = max(candidates, key=lambda c: (scores[c], c == (start_frame or -10**9)))
    if len(candidates) == 1:
        why = "only candidate"
    elif all_events:
        inside = sum(1 for f in all_events if best <= f <= best + n_samples - 1)
        why = f"auto: {inside}/{len(all_events)} events in range"
    else:
        why = "auto: from start/end_frame"
    return FrameMapping(offset=best, n_samples=n_samples, inferred_from=why, scores=scores)
 
 
# --------------------------------------------------------------------------
# Axis inference
# --------------------------------------------------------------------------
 
AXIS_NAMES = ("X", "Y", "Z")
 
 
@dataclass
class AxisFrame:
    forward: int = 0
    vertical: int = 2
    lateral: int = 1
    sign: int = 1  # +1 if the subject walks along +forward, -1 otherwise
    inferred_from: str = "default"
 
    def describe(self) -> str:
        s = "+" if self.sign > 0 else "-"
        return (
            f"forward {s}{AXIS_NAMES[self.forward]} · up {AXIS_NAMES[self.vertical]} "
            f"({self.inferred_from})"
        )
 
 
def parse_walk_dir(value) -> Optional[Tuple[int, int]]:
    """Best-effort parse of the ``walk_dir`` field -> ``(axis, sign)``."""
    if value is None:
        return None
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8", "ignore")
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        sign = -1 if text.startswith("-") or "neg" in text or "minus" in text else 1
        for axis, name in enumerate("xyz"):
            if name in text:
                return axis, sign
        return None
    arr = np.asarray(value).ravel()
    if arr.size == 1:
        v = arr[0]
        try:
            v = int(v)
        except (TypeError, ValueError):
            return None
        if v in (0, 1, 2):  # bare axis index, direction unknown
            return v, 1
        if v in (-1, -2, -3):
            return abs(v) - 1, -1
        return None
    if arr.size == 3 and np.issubdtype(arr.dtype, np.number):
        axis = int(np.argmax(np.abs(arr)))
        return axis, (1 if arr[axis] >= 0 else -1)
    return None
 
 
def infer_axis_frame(
    markers: Dict[str, Marker], walk_dir=None
) -> AxisFrame:
    """Infer forward / vertical / lateral axes from the marker cloud.
 
    The forward axis is the one along which the pelvis travels furthest.  The
    vertical axis is then whichever of the two remaining axes puts the pelvis
    highest above the feet — a far more reliable cue than assuming Z-up,
    because it is a fact about the recorded geometry rather than a convention.
    """
    pelvis = next((m for m in markers.values() if m.part == "pelvis"), None)
    feet = [m for m in markers.values() if m.part in ("heel", "toe")]
    if pelvis is None and markers:
        pelvis = next(iter(markers.values()))
    if pelvis is None:
        return AxisFrame()
 
    p = np.asarray(pelvis.xyz, dtype=float)
    finite = p[np.isfinite(p).all(axis=1)]
    if finite.shape[0] < 2:
        return AxisFrame()
 
    span = np.nanmax(finite, axis=0) - np.nanmin(finite, axis=0)
 
    parsed = parse_walk_dir(walk_dir)
    if parsed is not None:
        forward, sign = parsed
        source = "walk_dir"
    else:
        forward = int(np.argmax(span))
        k = max(1, finite.shape[0] // 10)
        net = float(np.median(finite[-k:, forward]) - np.median(finite[:k, forward]))
        sign = 1 if net >= 0 else -1
        source = "auto (pelvis travel)"
 
    remaining = [a for a in range(3) if a != forward]
    if feet:
        foot_low = {
            a: float(np.nanmin([np.nanmin(m.xyz[:, a]) for m in feet])) for a in remaining
        }
        pelvis_mid = {a: float(np.nanmean(finite[:, a])) for a in remaining}
        vertical = max(remaining, key=lambda a: abs(pelvis_mid[a] - foot_low[a]))
    else:
        vertical = min(remaining, key=lambda a: span[a])
    lateral = [a for a in remaining if a != vertical][0]
    return AxisFrame(forward, vertical, lateral, sign, source)
 
 
# --------------------------------------------------------------------------
# Undoable commands
# --------------------------------------------------------------------------
 
 
class Command:
    label = "edit"
 
    def apply(self, events: Dict[str, List[int]]) -> None:  # pragma: no cover
        raise NotImplementedError
 
    def revert(self, events: Dict[str, List[int]]) -> None:  # pragma: no cover
        raise NotImplementedError
 
    def touched_frames(self) -> List[int]:
        return []
 
 
def _insert(events: Dict[str, List[int]], key: str, frame: int) -> None:
    lst = events[key]
    idx = bisect.bisect_left(lst, frame)
    if idx < len(lst) and lst[idx] == frame:
        return
    lst.insert(idx, int(frame))
 
 
def _remove(events: Dict[str, List[int]], key: str, frame: int) -> None:
    lst = events[key]
    idx = bisect.bisect_left(lst, frame)
    if idx < len(lst) and lst[idx] == frame:
        lst.pop(idx)
 
 
@dataclass
class AddEvent(Command):
    key: str
    frame: int
 
    def __post_init__(self):
        self.label = f"add {EVENT_SPECS[self.key].short} @ {self.frame}"
 
    def apply(self, events):
        _insert(events, self.key, self.frame)
 
    def revert(self, events):
        _remove(events, self.key, self.frame)
 
    def touched_frames(self):
        return [self.frame]
 
 
@dataclass
class DeleteEvent(Command):
    key: str
    frame: int
 
    def __post_init__(self):
        self.label = f"delete {EVENT_SPECS[self.key].short} @ {self.frame}"
 
    def apply(self, events):
        _remove(events, self.key, self.frame)
 
    def revert(self, events):
        _insert(events, self.key, self.frame)
 
    def touched_frames(self):
        return [self.frame]
 
 
@dataclass
class MoveEvent(Command):
    key: str
    old_frame: int
    new_frame: int
 
    def __post_init__(self):
        self.label = (
            f"move {EVENT_SPECS[self.key].short} {self.old_frame} → {self.new_frame}"
        )
 
    def apply(self, events):
        _remove(events, self.key, self.old_frame)
        _insert(events, self.key, self.new_frame)
 
    def revert(self, events):
        _remove(events, self.key, self.new_frame)
        _insert(events, self.key, self.old_frame)
 
    def touched_frames(self):
        return [self.old_frame, self.new_frame]
 
 
# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
 
 
@dataclass
class Issue:
    severity: str  # "critical" | "warning" | "info"
    message: str
    frame: Optional[int] = None
    key: Optional[str] = None
 
 
def _outlier_bounds(values: Sequence[float]) -> Optional[Tuple[float, float]]:
    """Robust bounds via the median; needs a few cycles to mean anything."""
    if len(values) < 4:
        return None
    med = float(np.median(values))
    if med <= 0:
        return None
    return 0.45 * med, 2.2 * med
 
 
# --------------------------------------------------------------------------
# The document
# --------------------------------------------------------------------------
 
 
class GaitEventDocument:
    """Loads, edits and saves one ``*_GaitEvents.npz``."""
 
    def __init__(self, path: Optional[Path] = None):
        self.path: Optional[Path] = Path(path) if path else None
        self.raw: Dict[str, np.ndarray] = {}
        self.markers: Dict[str, Marker] = {}
        self.signals: Dict[str, np.ndarray] = {}
        self.meta: Dict[str, object] = {}
        self.events: Dict[str, List[int]] = {k: [] for k in EVENT_KEYS}
 
        self.frame_rate: float = 100.0
        self.start_frame: Optional[int] = None
        self.end_frame: Optional[int] = None
        self.walk_dir = None
        self.n_samples: int = 0
 
        self.mapping: FrameMapping = FrameMapping(0, 0)
        self.axes: AxisFrame = AxisFrame()
 
        self._event_dtypes: Dict[str, np.dtype] = {}
        self._compressed: bool = False
        self._undo: List[Command] = []
        self._redo: List[Command] = []
        self._clean_depth: int = 0
        self._listeners: List[Callable[[str], None]] = []
        self.load_notes: List[str] = []
 
    # -- observers ---------------------------------------------------------
 
    def subscribe(self, fn: Callable[[str], None]) -> None:
        self._listeners.append(fn)
 
    def _notify(self, what: str = "events") -> None:
        for fn in list(self._listeners):
            fn(what)
 
    # -- loading -----------------------------------------------------------
 
    @classmethod
    def load(cls, path) -> "GaitEventDocument":
        doc = cls(Path(path))
        doc._load()
        return doc
 
    def _load(self) -> None:
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"No such gait-events file: {path}")
 
        # allow_pickle is required because upstream may store None / str / dict
        # scalars alongside the numeric arrays.
        with np.load(path, allow_pickle=True) as z:
            self.raw = {k: z[k] for k in z.files}
        self._compressed = _npz_is_compressed(path)
 
        # --- events (needed early: they inform the frame mapping) ---------
        for key in EVENT_KEYS:
            arr = self.raw.get(key)
            if arr is None:
                self.load_notes.append(f"'{key}' missing from file — starting empty.")
                self.events[key] = []
                self._event_dtypes[key] = np.dtype(np.int64)
                continue
            a = np.asarray(arr).ravel()
            self._event_dtypes[key] = (
                a.dtype if np.issubdtype(a.dtype, np.integer) else np.dtype(np.int64)
            )
            if a.size == 0:
                self.events[key] = []
                continue
            if not np.issubdtype(a.dtype, np.number):
                raise ValueError(f"Event array '{key}' is not numeric (dtype {a.dtype})")
            rounded = np.rint(a.astype(float))
            if not np.allclose(rounded, a.astype(float), atol=1e-6):
                self.load_notes.append(
                    f"'{key}' held non-integer frames — rounded to nearest frame."
                )
            self.events[key] = sorted({int(v) for v in rounded})
 
        # --- scalars -------------------------------------------------------
        self.frame_rate = float(_scalar(self.raw.get("frame_rate"), 100.0) or 100.0)
        self.start_frame = _int_or_none(_scalar(self.raw.get("start_frame")))
        self.end_frame = _int_or_none(_scalar(self.raw.get("end_frame")))
        self.walk_dir = _scalar(self.raw.get("walk_dir"))
        for k in ("Date", "date"):
            if k in self.raw:
                self.meta["Date"] = _scalar(self.raw[k])
 
        # --- markers -------------------------------------------------------
        for key, value in self.raw.items():
            if key in EVENT_KEYS:
                continue
            xyz = _as_marker_array(value)
            if xyz is None:
                continue
            name = _pretty_marker_name(key)
            side, part = _classify_marker(name, key)
            self.markers[name] = Marker(name, key, xyz, side, part)
 
        if self.markers:
            # The trajectory length is the one most candidates agree on, not the
            # longest.  Files routinely carry small (M,3) arrays alongside the
            # real markers — force-plate corners, a lab origin, segment
            # definitions — and those are shaped exactly like a trajectory.
            lengths = [m.n_samples for m in self.markers.values()]
            counts = {n: lengths.count(n) for n in set(lengths)}
            self.n_samples = max(counts, key=lambda n: (counts[n], n))
 
            tolerance = max(2, int(self.n_samples * 0.02))
            rejected = {
                name: m.n_samples
                for name, m in self.markers.items()
                if abs(m.n_samples - self.n_samples) > tolerance
            }
            for name in rejected:
                del self.markers[name]
            if rejected:
                self.load_notes.append(
                    "Ignored as non-trajectory (wrong length for "
                    f"{self.n_samples} frames): "
                    + ", ".join(f"{k}={v}" for k, v in rejected.items())
                )
            ragged = {m.name: m.n_samples for m in self.markers.values()
                      if m.n_samples != self.n_samples}
            if ragged:
                self.load_notes.append(
                    "Marker arrays have differing lengths: "
                    + ", ".join(f"{k}={v}" for k, v in ragged.items())
                )
        elif self.start_frame is not None and self.end_frame is not None:
            self.n_samples = int(self.end_frame) - int(self.start_frame) + 1
            self.load_notes.append(
                "No (N,3) marker arrays found — 3-D view will be empty."
            )
        else:
            self.n_samples = 0
 
        # --- 1-D signals ---------------------------------------------------
        for key, value in self.raw.items():
            if key in EVENT_KEYS:
                continue
            a = np.asarray(value)
            if a.ndim == 1 and np.issubdtype(a.dtype, np.number) and a.size > 3:
                if abs(a.size - self.n_samples) <= 2:
                    self.signals[key] = a.astype(float)
 
        if self.n_samples <= 0:
            raise ValueError(
                f"Could not determine a frame count from {path.name}. "
                "Expected at least one (N,3) marker array or start/end_frame."
            )
 
        self.mapping = infer_frame_mapping(
            self.n_samples, self.start_frame, self.end_frame, self.events
        )
        self.axes = infer_axis_frame(self.markers, self.walk_dir)
 
        stray = [
            (k, f)
            for k in EVENT_KEYS
            for f in self.events[k]
            if not self.mapping.contains(f)
        ]
        if stray:
            self.load_notes.append(
                f"{len(stray)} event(s) fall outside frames "
                f"{self.mapping.first_frame}–{self.mapping.last_frame}; "
                "check the frame-offset setting in the toolbar."
            )
 
    # -- convenience -------------------------------------------------------
 
    @property
    def first_frame(self) -> int:
        return self.mapping.first_frame
 
    @property
    def last_frame(self) -> int:
        return self.mapping.last_frame
 
    def frame_to_time(self, frame: int) -> float:
        return (frame - self.mapping.first_frame) / max(self.frame_rate, 1e-9)
 
    def set_offset(self, offset: int) -> None:
        """Re-interpret the array<->frame alignment without touching the data."""
        if offset == self.mapping.offset:
            return
        self.mapping = FrameMapping(int(offset), self.n_samples, "manual override")
        self._notify("mapping")
 
    def marker_at(self, name: str, frame: int) -> Optional[np.ndarray]:
        m = self.markers.get(name)
        if m is None:
            return None
        idx = self.mapping.to_index(frame)
        if 0 <= idx < m.n_samples:
            return m.xyz[idx]
        return None
 
    def pelvis_marker(self) -> Optional[Marker]:
        return next((m for m in self.markers.values() if m.part == "pelvis"), None)
 
    def foot_markers(self, side: str) -> Dict[str, Marker]:
        return {
            m.part: m
            for m in self.markers.values()
            if m.side == side and m.part in ("heel", "toe")
        }
 
    def progression_signal(self, marker_name: str) -> Optional[np.ndarray]:
        """Marker position along the walking axis, relative to the pelvis.
 
        This is the signal the Zeni event detector thresholds: heel strike sits
        at its maximum, toe off at its minimum.  Plotting it under the timeline
        turns "is this event on the right frame?" into a visual check.
        """
        m = self.markers.get(marker_name)
        if m is None:
            return None
        ax, sign = self.axes.forward, self.axes.sign
        series = m.xyz[:, ax].astype(float) * sign
        pelvis = self.pelvis_marker()
        if pelvis is not None and pelvis.n_samples == m.n_samples:
            series = series - pelvis.xyz[:, ax].astype(float) * sign
        return series
 
    def vertical_signal(self, marker_name: str) -> Optional[np.ndarray]:
        m = self.markers.get(marker_name)
        if m is None:
            return None
        return m.xyz[:, self.axes.vertical].astype(float)
 
    def stored_signal(self, marker_name: str) -> Optional[np.ndarray]:
        """The pre-computed velocity trace shipped in the file, if there is one."""
        m = self.markers.get(marker_name)
        if m is None:
            return None
        stem = m.source_key.replace("arr", "").replace("_", "")
        for key, arr in self.signals.items():
            k = key.replace("_", "")
            if k.lower().startswith(stem.lower()) and "v" in k.lower():
                return arr
        return None
 
    # -- editing -----------------------------------------------------------
 
    def can_add(self, key: str, frame: int) -> bool:
        return self.mapping.contains(frame) and int(frame) not in self.events[key]
 
    def do(self, cmd: Command) -> None:
        cmd.apply(self.events)
        self._undo.append(cmd)
        self._redo.clear()
        self._notify("events")
 
    def add_event(self, key: str, frame: int) -> bool:
        frame = int(frame)
        if not self.can_add(key, frame):
            return False
        self.do(AddEvent(key, frame))
        return True
 
    def delete_event(self, key: str, frame: int) -> bool:
        if int(frame) not in self.events[key]:
            return False
        self.do(DeleteEvent(key, int(frame)))
        return True
 
    def move_event(self, key: str, old_frame: int, new_frame: int) -> bool:
        old_frame, new_frame = int(old_frame), int(new_frame)
        if old_frame == new_frame:
            return False
        if old_frame not in self.events[key]:
            return False
        if not self.mapping.contains(new_frame) or new_frame in self.events[key]:
            return False
        self.do(MoveEvent(key, old_frame, new_frame))
        return True
 
    @property
    def can_undo(self) -> bool:
        return bool(self._undo)
 
    @property
    def can_redo(self) -> bool:
        return bool(self._redo)
 
    def undo_label(self) -> str:
        return self._undo[-1].label if self._undo else ""
 
    def redo_label(self) -> str:
        return self._redo[-1].label if self._redo else ""
 
    def undo(self) -> Optional[Command]:
        if not self._undo:
            return None
        cmd = self._undo.pop()
        cmd.revert(self.events)
        self._redo.append(cmd)
        self._notify("events")
        return cmd
 
    def redo(self) -> Optional[Command]:
        if not self._redo:
            return None
        cmd = self._redo.pop()
        cmd.apply(self.events)
        self._undo.append(cmd)
        self._notify("events")
        return cmd
 
    @property
    def is_dirty(self) -> bool:
        return len(self._undo) != self._clean_depth
 
    def mark_clean(self) -> None:
        self._clean_depth = len(self._undo)
        self._notify("clean")
 
    # -- queries used by the widgets ---------------------------------------
 
    def all_events_sorted(self) -> List[Tuple[int, str]]:
        out = [(f, k) for k in EVENT_KEYS for f in self.events[k]]
        out.sort(key=lambda t: (t[0], t[1]))
        return out
 
    def nearest_event(
        self, frame: int, direction: int = 0, side: Optional[str] = None
    ) -> Optional[Tuple[int, str]]:
        items = [
            (f, k)
            for f, k in self.all_events_sorted()
            if side is None or EVENT_SPECS[k].side == side
        ]
        if not items:
            return None
        if direction > 0:
            later = [t for t in items if t[0] > frame]
            return later[0] if later else None
        if direction < 0:
            earlier = [t for t in items if t[0] < frame]
            return earlier[-1] if earlier else None
        return min(items, key=lambda t: abs(t[0] - frame))
 
    def stance_intervals(self, side: str) -> List[Tuple[int, int, bool]]:
        """``(heel_strike, toe_off, closed)`` pairs for shading the timeline."""
        hs = self.events[f"{side}HS"]
        to = self.events[f"{side}TO"]
        out: List[Tuple[int, int, bool]] = []
        for f in hs:
            idx = bisect.bisect_right(to, f)
            if idx < len(to):
                out.append((f, to[idx], True))
            else:
                out.append((f, self.last_frame, False))
        return out
 
    def side_in_stance(self, side: str, frame: int) -> bool:
        return any(a <= frame < b for a, b, _ in self.stance_intervals(side))
 
    # -- validation --------------------------------------------------------
 
    def validate(self) -> List[Issue]:
        issues: List[Issue] = []
        lo, hi = self.mapping.first_frame, self.mapping.last_frame
 
        for key in EVENT_KEYS:
            spec = EVENT_SPECS[key]
            for f in self.events[key]:
                if not (lo <= f <= hi):
                    issues.append(
                        Issue(
                            "critical",
                            f"{spec.short} at frame {f} is outside the data "
                            f"({lo}–{hi})",
                            f,
                            key,
                        )
                    )
 
        for side in SIDES:
            hs_key, to_key = f"{side}HS", f"{side}TO"
            clash = set(self.events[hs_key]) & set(self.events[to_key])
            for f in sorted(clash):
                issues.append(
                    Issue(
                        "critical",
                        f"{SIDE_LABEL[side]}: heel strike and toe off both at "
                        f"frame {f}",
                        f,
                        hs_key,
                    )
                )
 
            merged = sorted(
                [(f, "HS", hs_key) for f in self.events[hs_key]]
                + [(f, "TO", to_key) for f in self.events[to_key]]
            )
            for (f1, k1, key1), (f2, k2, _) in zip(merged, merged[1:]):
                if k1 == k2:
                    missing = "toe off" if k1 == "HS" else "heel strike"
                    issues.append(
                        Issue(
                            "warning",
                            f"{SIDE_LABEL[side]}: two consecutive "
                            f"{'heel strikes' if k1 == 'HS' else 'toe offs'} at "
                            f"{f1} and {f2} — a {missing} is missing between them",
                            f2,
                            key1,
                        )
                    )
 
            # duration outliers, judged against this trial's own median
            stance = [
                (a, b - a) for a, b, closed in self.stance_intervals(side) if closed
            ]
            bounds = _outlier_bounds([d for _, d in stance])
            if bounds:
                lo_d, hi_d = bounds
                for start, dur in stance:
                    if dur < lo_d or dur > hi_d:
                        issues.append(
                            Issue(
                                "warning",
                                f"{SIDE_LABEL[side]}: stance starting at frame "
                                f"{start} lasts {dur} frames "
                                f"({dur / self.frame_rate:.2f}s) — well off this "
                                f"trial's median of {int(np.median([d for _, d in stance]))}",
                                start,
                                hs_key,
                            )
                        )
 
            hs = self.events[hs_key]
            strides = [(a, b - a) for a, b in zip(hs, hs[1:])]
            bounds = _outlier_bounds([d for _, d in strides])
            if bounds:
                lo_d, hi_d = bounds
                for start, dur in strides:
                    if dur < lo_d or dur > hi_d:
                        issues.append(
                            Issue(
                                "warning",
                                f"{SIDE_LABEL[side]}: stride starting at frame "
                                f"{start} lasts {dur} frames "
                                f"({dur / self.frame_rate:.2f}s) — an outlier for "
                                f"this trial",
                                start,
                                hs_key,
                            )
                        )
 
        empties = [EVENT_SPECS[k].short for k in EVENT_KEYS if not self.events[k]]
        if empties:
            issues.append(
                Issue("info", "No events of type: " + ", ".join(empties))
            )
 
        issues.sort(key=lambda i: ({"critical": 0, "warning": 1, "info": 2}[i.severity],
                                   i.frame if i.frame is not None else -1))
        return issues
 
    # -- saving ------------------------------------------------------------
 
    def save(
        self,
        path=None,
        backup: bool = True,
        write_provenance: bool = True,
    ) -> Tuple[Path, Optional[Path]]:
        """Write the file back, returning ``(saved_path, backup_path)``.
 
        Every original key is preserved byte-for-byte apart from the four event
        arrays, which are rewritten sorted and in their original integer dtype.
        The write goes to a temporary file and is then atomically renamed, so an
        interrupted save can never leave a half-written .npz behind.
        """
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("No path to save to")
        target = Path(target)
 
        out: Dict[str, np.ndarray] = dict(self.raw)
        for key in EVENT_KEYS:
            dtype = self._event_dtypes.get(key, np.dtype(np.int64))
            out[key] = np.asarray(sorted(self.events[key]), dtype=dtype)
 
        if write_provenance:
            out["Date_edited"] = np.asarray(
                _dt.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            )
 
        backup_path: Optional[Path] = None
        if backup and target.exists():
            stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = target.with_name(f"{target.name}.bak_{stamp}")
            shutil.copy2(target, backup_path)
 
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        saver = np.savez_compressed if self._compressed else np.savez
        try:
            # Pass a file handle: np.savez appends ".npz" when given a name.
            with open(tmp, "wb") as fh:
                saver(fh, **out)
            os.replace(tmp, target)
        except Exception:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            raise
 
        self.raw = out
        self.path = target
        self.mark_clean()
        return target, backup_path
 
 
# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
 
 
def _scalar(value, default=None):
    if value is None:
        return default
    a = np.asarray(value)
    if a.ndim == 0:
        try:
            return a.item()
        except (ValueError, AttributeError):
            return default
    if a.size == 1:
        try:
            return a.reshape(-1)[0].item()
        except (ValueError, AttributeError):
            return default
    return value
 
 
def _int_or_none(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
 
 
def _as_marker_array(value) -> Optional[np.ndarray]:
    """Return an ``(N,3)`` float array if ``value`` looks like a trajectory."""
    a = np.asarray(value)
    if a.ndim != 2 or not np.issubdtype(a.dtype, np.number):
        return None
    rows, cols = a.shape
    if cols == 3 and rows >= 2:
        return a.astype(float)
    if rows == 3 and cols >= 4:  # stored transposed
        return a.T.astype(float)
    return None
 
 
def _npz_is_compressed(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return any(i.compress_type != zipfile.ZIP_STORED for i in z.infolist())
    except (OSError, zipfile.BadZipFile):
        return False