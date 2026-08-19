
"""
3-D marker viewer.
 
The recorded data is rotated once, at load time, into a canonical frame where
+X is the direction of travel, +Z is up and the floor sits at Z=0.  Everything
downstream — the ground grid, the camera presets, the drop lines — can then
assume Z-up instead of guessing per file, and the sagittal/frontal presets mean
the same thing whatever axis convention the lab used.
 
If OpenGL is unavailable the widget degrades to a 2-D sagittal projection of
the same canonical data rather than failing; every other feature of the editor
keeps working.
"""
 
from __future__ import annotations
 
from typing import Dict, List, Optional
 
import numpy as np
import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets
 
from . import theme
from .event_data import GaitEventDocument, Marker
 
try:  # OpenGL is optional
    import pyqtgraph.opengl as gl
 
    HAVE_GL = True
    _GL_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on the host
    gl = None
    HAVE_GL = False
    _GL_ERROR = str(exc)
 
 
def gl_is_usable() -> bool:
    """Can this machine actually create a GL context, not just import the module?
 
    Importing ``pyqtgraph.opengl`` succeeds on plenty of machines that then fail
    to create a context — headless servers, remote sessions without GLX, VMs
    with no driver.  Probing an offscreen surface up front means those users get
    the 2-D fallback instead of a black rectangle.
    """
    if not HAVE_GL:
        return False
    if QtWidgets.QApplication.instance() is None:
        return True  # nothing to probe against yet; assume the best
    try:
        context = QtGui.QOpenGLContext()
        context.setFormat(QtGui.QSurfaceFormat.defaultFormat())
        if not context.create():
            return False
        surface = QtGui.QOffscreenSurface()
        surface.setFormat(context.format())
        surface.create()
        if not surface.isValid() or not context.makeCurrent(surface):
            return False
        version = context.format().version()
        context.doneCurrent()
        # The same test GLViewWidget.initializeGL applies.  Checking it here
        # means a context Qt negotiated down to 2.0 (common under Xvfb, VMs and
        # some remote-desktop stacks, even when the driver itself reports 4.x)
        # sends us to the 2-D fallback instead of raising mid-paint.
        return tuple(version) >= (2, 1)
    except Exception:
        return False
 
 
CAMERA_PRESETS: Dict[str, Dict[str, float]] = {
    # azimuth is measured in the floor plane from +X (the walking direction)
    "Oblique": {"azimuth": 60.0, "elevation": 18.0},
    "Sagittal": {"azimuth": 90.0, "elevation": 0.0},
    "Frontal": {"azimuth": 0.0, "elevation": 0.0},
    "Top": {"azimuth": 90.0, "elevation": 89.0},
}
 
 
def _nice_step(span: float) -> float:
    """A round grid spacing about a twelfth of ``span``."""
    if span <= 0 or not np.isfinite(span):
        return 1.0
    raw = span / 12.0
    mag = 10.0 ** np.floor(np.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        if raw <= mult * mag:
            return float(mult * mag)
    return float(10 * mag)
 
 
def _segment_pairs(points: np.ndarray) -> np.ndarray:
    """Consecutive finite pairs of ``(N,3)`` -> ``(2M,3)`` for ``mode='lines'``.
 
    Splitting at gaps rather than interpolating means a dropped marker shows up
    as a break in the trail, which is information the user wants to see.
    """
    if points.shape[0] < 2:
        return np.zeros((0, 3))
    ok = np.isfinite(points).all(axis=1)
    valid = ok[:-1] & ok[1:]
    if not valid.any():
        return np.zeros((0, 3))
    starts = points[:-1][valid]
    ends = points[1:][valid]
    out = np.empty((starts.shape[0] * 2, 3), dtype=float)
    out[0::2] = starts
    out[1::2] = ends
    return out
 
 
class Viewer3D(QtWidgets.QWidget):
    """Point-cloud view of every marker discovered in the file."""
 
    def __init__(self, parent=None):
        super().__init__(parent)
        self.doc: Optional[GaitEventDocument] = None
        self.frame = 0
        self.trail_len = 40
        self.follow = True
        self.color_mode = "side"  # "side" | "segment"
        self.show_labels = True
        self.show_drops = True
 
        self._names: List[str] = []
        self._canon: Dict[str, np.ndarray] = {}
        self._floor = 0.0
        self._extent = 1000.0
        self._body_scale = 1000.0
        self._focus_z = 0.0
        self._mid = np.zeros(3)
        self._preset = "Oblique"
 
        self.use_gl = gl_is_usable()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        if self.use_gl:
            try:
                self._build_gl()
            except Exception:  # a driver that fails later than the probe
                self.use_gl = False
        if not self.use_gl:
            self._build_fallback()
        layout.addWidget(self.canvas)
 
    # -- construction ------------------------------------------------------
 
    def _build_gl(self) -> None:
        self.canvas = gl.GLViewWidget()
        self.canvas.setBackgroundColor(pg.mkColor(theme.SURFACE))
        self.canvas.setMinimumSize(320, 240)
 
        self.grid = gl.GLGridItem()
        self.grid.setColor(pg.mkColor(theme.rgba(theme.INK, 0.16)))
        self.canvas.addItem(self.grid)
 
        self.scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)), size=12, pxMode=True
        )
        self.scatter.setGLOptions("translucent")
        self.canvas.addItem(self.scatter)
 
        self.trails: Dict[str, "gl.GLLinePlotItem"] = {}
        self.segments = gl.GLLinePlotItem(mode="lines", width=2.0, antialias=True)
        self.canvas.addItem(self.segments)
        self.drops = gl.GLLinePlotItem(mode="lines", width=1.0, antialias=True)
        self.canvas.addItem(self.drops)
        self.labels: Dict[str, object] = {}
 
    def _build_fallback(self) -> None:
        self.canvas = pg.PlotWidget(background=theme.SURFACE)
        self.canvas.setMinimumSize(320, 240)
        self.canvas.setAspectLocked(True)
        self.canvas.showGrid(x=True, y=True, alpha=0.15)
        for name in ("left", "bottom"):
            ax = self.canvas.getAxis(name)
            ax.setPen(pg.mkPen(theme.AXIS))
            ax.setTextPen(pg.mkPen(theme.MUTED))
        self.canvas.getAxis("bottom").setLabel("progression", color=theme.MUTED)
        self.canvas.getAxis("left").setLabel("vertical", color=theme.MUTED)
 
        self._fb_floor = pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen(theme.AXIS, width=2)
        )
        self.canvas.addItem(self._fb_floor)
        self._fb_trails = pg.PlotCurveItem(
            pen=pg.mkPen(theme.rgba(theme.INK_2, 0.35), width=1), connect="pairs"
        )
        self.canvas.addItem(self._fb_trails)
        self._fb_segments = pg.PlotCurveItem(
            pen=pg.mkPen(theme.rgba(theme.INK, 0.55), width=2), connect="pairs"
        )
        self.canvas.addItem(self._fb_segments)
        self._fb_scatter = pg.ScatterPlotItem(size=13, pen=pg.mkPen(theme.SURFACE, width=2))
        self.canvas.addItem(self._fb_scatter)
        self._fb_labels: List[pg.TextItem] = []
 
        why = (
            "PyOpenGL is not installed"
            if not HAVE_GL
            else "this display cannot create an OpenGL context"
        )
        banner = QtWidgets.QLabel(
            f"3-D view unavailable ({why}) — showing a 2-D sagittal "
            "projection instead. Every editing feature still works."
        )
        banner.setStyleSheet(
            f"color:{theme.WARNING}; background:{theme.RAISED}; padding:4px 8px;"
        )
        banner.setWordWrap(True)
        self.layout().addWidget(banner)
 
    # -- document ----------------------------------------------------------
 
    def set_document(self, doc: Optional[GaitEventDocument]) -> None:
        self.doc = doc
        self._canon.clear()
        self._names = []
        if doc is None or not doc.markers:
            return
 
        ax = doc.axes
        for name, marker in doc.markers.items():
            xyz = np.asarray(marker.xyz, dtype=float)
            canon = np.column_stack(
                [
                    xyz[:, ax.forward] * ax.sign,
                    xyz[:, ax.lateral],
                    xyz[:, ax.vertical],
                ]
            )
            self._canon[name] = canon
            self._names.append(name)
 
        stacked = np.concatenate([v for v in self._canon.values()], axis=0)
        finite = stacked[np.isfinite(stacked).all(axis=1)]
        if finite.size == 0:
            return
        lo, hi = finite.min(axis=0), finite.max(axis=0)
        self._floor = float(lo[2])
        for name in self._canon:
            self._canon[name][:, 2] -= self._floor
        lo[2] -= self._floor
        hi[2] -= self._floor
        self._mid = (lo + hi) / 2.0
        self._extent = float(max(hi - lo).item()) or 1000.0
        # Framing the whole walkway makes the subject a speck.  When the camera
        # follows the pelvis the useful scale is the body: its height and width,
        # with enough floor around it to see a stride land.
        self._body_scale = float(max(hi[2] - lo[2], hi[1] - lo[1], 1.0)) * 1.35
        # look slightly above the mid-height so the pelvis label is not clipped
        self._focus_z = float(lo[2] + 0.62 * (hi[2] - lo[2]))
 
        if self.use_gl:
            self._setup_gl_scene(lo, hi)
        self.set_camera(self._preset)
        self.update_frame(self.frame)
 
    def _setup_gl_scene(self, lo, hi) -> None:
        step = _nice_step(self._body_scale * 2.0)
        gx = max(step * 4, (hi[0] - lo[0]) * 1.2)
        gy = max(step * 4, (hi[1] - lo[1]) * 1.6)
        self.grid.setSize(x=gx, y=gy)
        self.grid.setSpacing(x=step, y=step)
        self.grid.resetTransform()
        self.grid.translate(self._mid[0], self._mid[1], 0.0)
 
        for item in self.trails.values():
            self.canvas.removeItem(item)
        self.trails.clear()
        for item in self.labels.values():
            self.canvas.removeItem(item)
        self.labels.clear()
 
        for name in self._names:
            trail = gl.GLLinePlotItem(mode="lines", width=1.4, antialias=True)
            trail.setGLOptions("translucent")
            self.canvas.addItem(trail)
            self.trails[name] = trail
            if hasattr(gl, "GLTextItem"):
                try:
                    label = gl.GLTextItem(
                        pos=np.zeros(3),
                        text=name,
                        color=pg.mkColor(theme.INK_2),
                        font=QtGui.QFont("sans", 9),
                    )
                    self.canvas.addItem(label)
                    self.labels[name] = label
                except Exception:
                    pass
 
    # -- appearance --------------------------------------------------------
 
    def _marker_color(self, marker: Marker, stance: bool) -> tuple:
        if marker.part == "pelvis" or marker.side == "C":
            base = theme.MUTED
        elif self.color_mode == "side":
            base = theme.LEFT if marker.side == "L" else (
                theme.RIGHT if marker.side == "R" else theme.MUTED
            )
        else:
            base = (
                theme.HEEL if marker.part == "heel"
                else theme.TOE if marker.part == "toe"
                else theme.MUTED
            )
        c = pg.mkColor(base)
        alpha = 1.0 if stance else 0.80
        return (c.redF(), c.greenF(), c.blueF(), alpha)
 
    def set_color_mode(self, mode: str) -> None:
        self.color_mode = mode
        self.update_frame(self.frame)
 
    def set_trail(self, n: int) -> None:
        self.trail_len = max(0, int(n))
        self.update_frame(self.frame)
 
    def set_follow(self, on: bool) -> None:
        self.follow = bool(on)
        self.set_camera(self._preset)  # the useful distance changes with it
        self.update_frame(self.frame)
 
    def set_labels(self, on: bool) -> None:
        self.show_labels = bool(on)
        self.update_frame(self.frame)
 
    def set_camera(self, preset: str) -> None:
        self._preset = preset
        if not self.use_gl or self.doc is None:
            return
        cfg = CAMERA_PRESETS.get(preset, CAMERA_PRESETS["Oblique"])
        distance = self._body_scale * 2.9 if self.follow else self._extent * 1.15
        centre = pg.Vector(self._mid[0], self._mid[1],
                           self._focus_z if self.follow else self._mid[2])
        self.canvas.setCameraPosition(
            pos=centre,
            distance=distance,
            azimuth=cfg["azimuth"],
            elevation=cfg["elevation"],
        )
 
    # -- per-frame update --------------------------------------------------
 
    def update_frame(self, frame: int) -> None:
        self.frame = int(frame)
        if self.doc is None or not self._canon:
            return
        idx = self.doc.mapping.to_index(self.frame)
        # Clamp per marker, not against the shortest array in the file: one
        # stray short (M,3) array — force-plate corners, a lab origin — used to
        # pin the index for *every* marker, freezing the whole scene while the
        # timeline and signal panel carried on working.
        idx = max(0, idx)
        stance = {s: self.doc.side_in_stance(s, self.frame) for s in ("L", "R")}
 
        positions, colors, sizes, names = [], [], [], []
        for name in self._names:
            track = self._canon[name]
            p = track[min(idx, track.shape[0] - 1)]
            if not np.isfinite(p).all():
                continue
            marker = self.doc.markers[name]
            in_stance = stance.get(marker.side, True)
            positions.append(p)
            colors.append(self._marker_color(marker, in_stance))
            sizes.append(16.0 if (in_stance and marker.part in ("heel", "toe")) else 11.0)
            names.append(name)
 
        if self.use_gl:
            self._update_gl(idx, positions, colors, sizes, names, stance)
        else:
            self._update_fallback(idx, positions, colors, names)
 
    # -- GL path -----------------------------------------------------------
 
    def _update_gl(self, idx, positions, colors, sizes, names, stance) -> None:
        if positions:
            pos = np.asarray(positions, dtype=float)
            self.scatter.setData(
                pos=pos,
                color=np.asarray(colors, dtype=float),
                size=np.asarray(sizes, dtype=float),
            )
        else:
            self.scatter.setData(pos=np.zeros((0, 3)))
 
        lo = max(0, idx - self.trail_len)
        for name, item in self.trails.items():
            if self.trail_len <= 0:
                item.setData(pos=np.zeros((0, 3)))
                continue
            pts = _segment_pairs(self._canon[name][lo: idx + 1])
            if pts.shape[0] == 0:
                item.setData(pos=np.zeros((0, 3)))
                continue
            marker = self.doc.markers[name]
            r, g, b, _ = self._marker_color(marker, True)
            fade = np.linspace(0.12, 0.75, pts.shape[0])
            item.setData(
                pos=pts,
                color=np.column_stack(
                    [np.full_like(fade, r), np.full_like(fade, g),
                     np.full_like(fade, b), fade]
                ),
            )
 
        self.segments.setData(pos=self._skeleton(idx))
 
        if self.show_drops and positions:
            pos = np.asarray(positions, dtype=float)
            drops = np.empty((pos.shape[0] * 2, 3))
            drops[0::2] = pos
            drops[1::2] = np.column_stack([pos[:, 0], pos[:, 1], np.zeros(len(pos))])
            self.drops.setData(
                pos=drops, color=pg.mkColor(theme.rgba(theme.INK, 0.18))
            )
        else:
            self.drops.setData(pos=np.zeros((0, 3)))
 
        # Stagger the label heights: the four foot markers sit within a few
        # centimetres of each other and their labels would otherwise overprint.
        shown = set(names)
        stagger = {"heel": 0.11, "toe": 0.20, "pelvis": 0.09}
        for name, label in self.labels.items():
            if self.show_labels and name in shown:
                marker = self.doc.markers[name]
                lift = self._body_scale * stagger.get(marker.part, 0.15)
                track = self._canon[name]
                here = track[min(idx, track.shape[0] - 1)]
                label.setData(pos=here + np.array([0.0, 0.0, lift]))
                label.setVisible(True)
            else:
                label.setVisible(False)
 
        if self.follow:
            pelvis = self.doc.pelvis_marker()
            if pelvis is not None and pelvis.name in self._canon:
                track = self._canon[pelvis.name]
                p = track[min(idx, track.shape[0] - 1)]
                if np.isfinite(p).all():
                    self.canvas.opts["center"] = pg.Vector(p[0], p[1], self._focus_z)
                    self.canvas.update()
 
    def _skeleton(self, idx: int) -> np.ndarray:
        """Heel–toe per foot, plus pelvis–heel, when those markers exist."""
        pairs: List[tuple] = []
        by_role = {
            (m.side, m.part): name
            for name, m in self.doc.markers.items()
            if name in self._canon
        }
        pelvis = next(
            (n for (s, p), n in by_role.items() if p == "pelvis"), None
        )
        for side in ("L", "R"):
            heel = by_role.get((side, "heel"))
            toe = by_role.get((side, "toe"))
            if heel and toe:
                pairs.append((heel, toe))
            if pelvis and heel:
                pairs.append((pelvis, heel))
        out: List[np.ndarray] = []
        for a, b in pairs:
            ta, tb = self._canon[a], self._canon[b]
            pa = ta[min(idx, ta.shape[0] - 1)]
            pb = tb[min(idx, tb.shape[0] - 1)]
            if np.isfinite(pa).all() and np.isfinite(pb).all():
                out.extend([pa, pb])
        return np.asarray(out) if out else np.zeros((0, 3))
 
    # -- 2-D fallback path -------------------------------------------------
 
    def _update_fallback(self, idx, positions, colors, names) -> None:
        if positions:
            pos = np.asarray(positions)
            brushes = [pg.mkBrush(pg.mkColor([int(c * 255) for c in col]))
                       for col in colors]
            self._fb_scatter.setData(x=pos[:, 0], y=pos[:, 2], brush=brushes)
        else:
            self._fb_scatter.setData(x=[], y=[])
 
        lo = max(0, idx - self.trail_len)
        tx, ty = [], []
        for name in self._names:
            pts = _segment_pairs(self._canon[name][lo: idx + 1])
            tx.extend(pts[:, 0].tolist())
            ty.extend(pts[:, 2].tolist())
        self._fb_trails.setData(np.asarray(tx), np.asarray(ty))
 
        seg = self._skeleton(idx)
        self._fb_segments.setData(
            seg[:, 0] if seg.size else np.array([]),
            seg[:, 2] if seg.size else np.array([]),
        )
 
        while len(self._fb_labels) < len(names):
            item = pg.TextItem(color=theme.MUTED, anchor=(0.5, 1.3))
            self.canvas.addItem(item)
            self._fb_labels.append(item)
        for i, item in enumerate(self._fb_labels):
            if self.show_labels and i < len(names):
                item.setText(names[i])
                item.setPos(positions[i][0], positions[i][2])
                item.show()
            else:
                item.hide()
 
        if self.follow and positions:
            span = self._extent * 0.9
            cx = float(np.mean([p[0] for p in positions]))
            self.canvas.setXRange(cx - span / 2, cx + span / 2, padding=0)