"""
compare_trajectory.py
 
Overlay the marker trajectories of two already-aligned trials
(*_GaitEvents.npz: sacrum at the origin, +X = walking direction, mm).
 
Two views, both immune to the frame-rate difference:
  * visualize_paths       -- one coordinate against another; sampling density
                             changes, trajectory geometry does not.
  * visualize_timeseries  -- x axis in SECONDS, zeroed on a reference gait
                             event, so two trials at 100 Hz and 150 Hz line up
                             without any resampling.
 
NOTE ON NAMES: the four markers have ONE canonical name each (LHEE, LTOE,
RHEE, RTOE) used everywhere in this module. What they happen to be called
inside the .npz is a separate concern, handled once in MARKER_KEYS. Do not
rename MARKERS to match a file -- add the file's spelling to MARKER_KEYS.
"""
 
import warnings
from pathlib import Path
 
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
 
from gait_Analysis.utils.find_project_root import find_project_root
 
OUTLINE = [pe.withStroke(linewidth=3, foreground="white")]
 
# Canonical marker names -- the vocabulary of this module.
MARKERS = ("LHEE", "LTOE", "RHEE", "RTOE")
EVENT_KEYS = ("LHS", "LTO", "RHS", "RTO")
 
# Candidate .npz keys per canonical marker, tried in order.
# Add new spellings HERE; never rename MARKERS.
MARKER_KEYS = {
    "LHEE": ("LHEE_arr", "LHEE", "LHarr", "LH_arr"),
    "LTOE": ("LTOE_arr", "LTOE", "LTarr", "LT_arr"),
    "RHEE": ("RHEE_arr", "RHEE", "RHarr", "RH_arr"),
    "RTOE": ("RTOE_arr", "RTOE", "RTarr", "RT_arr"),
}
 
# Which events belong on which marker's curve
MARKER_EVENTS = {"LHEE": ("LHS", "LTO"), "LTOE": ("LHS", "LTO"),
                 "RHEE": ("RHS", "RTO"), "RTOE": ("RHS", "RTO")}
 
MARKER_STYLE = {"LHEE": ("tab:blue",   "Left Heel"),
                "LTOE": ("tab:green",  "Left Toe"),
                "RHEE": ("tab:orange", "Right Heel"),
                "RTOE": ("tab:red",    "Right Toe")}
 
# Target frame after alignment: X(front) | Y(right) | Z(up)
AXIS_LABEL = {0: "X fore-aft", 1: "Y medio-lateral", 2: "Z vertical"}
PLANES = {"sagittal": (0, 2), "transverse": (0, 1), "frontal": (1, 2)}
 
# Trial 0 solid, trial 1 dashed -- deliberately the ONLY difference between
# them, so colour stays free to encode the marker.
TRIAL_STYLE = ({"linestyle": "-",  "alpha": 1.00, "linewidth": 1.7},
               {"linestyle": "--", "alpha": 0.75, "linewidth": 1.5})
# Event markers are black for trial 0, grey for trial 1, so an event can still
# be attributed to a trial where the two curves overlap.
TRIAL_EVENT_COLOR = ("black", "dimgray")
 
 
class InvalidConfigError(Exception):
    def __init__(self, error_msgs):
        self.error_msgs = list(error_msgs)
        super().__init__("\n".join(self.error_msgs))
 
 
# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def gait_events_path(project_root, src_type, trial_name, session_name=None):
    """Path of one *_GaitEvents.npz. Only vicon nests under a session folder."""
    directory = Path(project_root) / "data" / "GaitEvents" / src_type
    if src_type == "vicon":
        if not session_name:
            raise InvalidConfigError(
                [f"'{trial_name}' is a vicon trial and needs a session name"])
        directory = directory / session_name
    return directory / f"{trial_name}_GaitEvents.npz"
 
 
def _find_marker(data, name, overrides=None):
    """Resolve canonical `name` to whatever key the file actually uses."""
    candidates = MARKER_KEYS[name]
    if overrides and name in overrides:
        candidates = (overrides[name],) + tuple(candidates)
    for key in candidates:
        if key in data.files:
            return np.asarray(data[key], dtype=float), key
    raise InvalidConfigError(
        [f"no key for marker {name}: tried {list(candidates)}",
         f"  file contains: {sorted(data.files)}",
         f"  add the right spelling to MARKER_KEYS['{name}']"])
 
 
def load_trial(path, label, key_overrides=None):
    """Return one trial as a uniform dict, keyed by CANONICAL marker names.
 
    Marker arrays become (n_frames, n_axes) with n_axes 2 or 3; events become
    0-based frame indices, clipped to the trial and sorted.
    """
    path = Path(path)
    if not path.exists():
        raise InvalidConfigError([f"GaitEvents file not found: {path}"])
 
    with np.load(path, allow_pickle=False) as data:
        markers, used_keys = {}, {}
        for name in MARKERS:
            arr, key = _find_marker(data, name, key_overrides)
            if arr.ndim != 2:
                raise InvalidConfigError(
                    [f"{path.name}: {key} has shape {arr.shape}, expected 2-D"])
            # tolerate a stored (n_axes, n_frames) layout
            if arr.shape[0] in (2, 3) and arr.shape[1] not in (2, 3):
                arr = arr.T
            markers[name] = arr
            used_keys[name] = key
 
        sacrum = np.asarray(data["sacrum_arr"], dtype=float)
        frame_rate = float(np.asarray(data["frame_rate"]).ravel()[0])
        raw_events = {k: np.asarray(data[k]).astype(int).ravel() - 1  # 1 -> 0
                      for k in EVENT_KEYS}
 
    ref = MARKERS[0]
    n_frames = len(markers[ref])
    for name, arr in markers.items():
        if len(arr) != n_frames:
            raise InvalidConfigError(
                [f"{path.name}: {name} has {len(arr)} frames, "
                 f"{ref} has {n_frames}"])
 
    n_axes = markers[ref].shape[1]
    if n_axes not in (2, 3):
        raise InvalidConfigError(
            [f"{path.name}: markers have {n_axes} columns, expected 2 or 3"])
 
    events = {}
    for key, ev in raw_events.items():
        keep = ev[(ev >= 0) & (ev < n_frames)]
        if len(keep) != len(ev):
            warnings.warn(f"{path.name}: dropped {len(ev) - len(keep)} "
                          f"{key} event(s) outside frames 0..{n_frames - 1}",
                          RuntimeWarning)
        events[key] = np.sort(keep)
 
    if len(events["LHS"]) == 0 or len(events["RHS"]) == 0:
        raise InvalidConfigError(
            [f"{path.name}: no heel strikes on one or both feet"])
 
    return {"label": label, "path": path, "markers": markers, "sacrum": sacrum,
            "events": events, "frame_rate": frame_rate, "n_frames": n_frames,
            "n_axes": n_axes, "keys": used_keys}
 
 
# --------------------------------------------------------------------------
# Capability query
# --------------------------------------------------------------------------
def available_planes(trials):
    """Planes that every trial has the columns for, in preferred order."""
    n_axes = min(t["n_axes"] for t in trials)
    return [p for p in ("sagittal", "transverse", "frontal")
            if max(PLANES[p]) < n_axes]
 
 
# --------------------------------------------------------------------------
# Time base
# --------------------------------------------------------------------------
def time_vector(trial, t_zero="first_LHS"):
    """Seconds for every frame of a trial.
 
    t_zero:
      "first_LHS"  -- t=0 at the first LEFT heel strike (default). Using the
                      same named event in both trials guarantees they are
                      phase-matched; "first heel strike of either foot" does
                      NOT, because one trial may lead with the other leg.
      "first_RHS"  -- same, right foot.
      "start"      -- t=0 at frame 0. Only meaningful if the two recordings
                      were started together, which they were not.
    """
    t = np.arange(trial["n_frames"]) / trial["frame_rate"]
    if t_zero == "start":
        return t
    key = {"first_LHS": "LHS", "first_RHS": "RHS"}.get(t_zero)
    if key is None:
        raise ValueError(f"Unknown t_zero '{t_zero}'")
    return t - t[trial["events"][key][0]]
 
 
# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def _draw_events(ax, trial, marker, x, y, show, trial_index=0):
    """Heel strike = filled circle, toe off = open square."""
    if not show:
        return
    ec = TRIAL_EVENT_COLOR[trial_index % len(TRIAL_EVENT_COLOR)]
    hs_key, to_key = MARKER_EVENTS[marker]
    for key, style in ((hs_key, dict(marker="o", markersize=6, mfc=ec)),
                       (to_key, dict(marker="s", markersize=6, mfc="none"))):
        ev = trial["events"][key]
        if ev.size:
            ax.plot(x[ev], y[ev], linestyle="none", mec=ec,
                    markeredgewidth=1.2, zorder=6, **style)
 
 
def visualize_paths(trials, plane="sagittal", combined=False,
                    show_events=True, equal_aspect=True):
    """Spatial overlay of both trials -- frame-rate independent.
 
    Plots one coordinate against another rather than against time, so a 100 Hz
    and a 150 Hz trial trace the same curve with no resampling.
    """
    if plane not in PLANES:
        raise ValueError(f"plane must be one of {list(PLANES)}")
    h, v = PLANES[plane]
 
    n_axes = min(t["n_axes"] for t in trials)
    if max(h, v) >= n_axes:
        usable = available_planes(trials) or ["none"]
        raise ValueError(
            f"plane '{plane}' needs axis {max(h, v)} ({AXIS_LABEL[max(h, v)]}) "
            f"but the saved markers have only {n_axes} columns. "
            f"Plottable now: {usable}. To get the sagittal view, re-export the "
            f"GaitEvents files with the vertical axis kept in generalize().")
 
    if combined:
        fig, ax = plt.subplots(figsize=(9.5, 7.5))
        axes = {name: ax for name in MARKERS}
    else:
        fig, axs = plt.subplots(2, 2, figsize=(13, 6.5), sharex=True, sharey=True)
        axes = dict(zip(MARKERS, axs.flat))
 
    for name in MARKERS:
        ax = axes[name]
        color, title = MARKER_STYLE[name]
        for i, trial in enumerate(trials):
            xy = trial["markers"][name]
            lab = f"{title} - {trial['label']}" if combined else trial["label"]
            ax.plot(xy[:, h], xy[:, v], color=color, label=lab, **TRIAL_STYLE[i])
            _draw_events(ax, trial, name, xy[:, h], xy[:, v], show_events, i)
        if not combined:
            ax.set_title(title)
 
    for ax in dict.fromkeys(axes.values()):
        ax.plot(0, 0, "k+", markersize=12, markeredgewidth=2, zorder=7)
        ax.annotate("sacrum", (0, 0), textcoords="offset points",
                    xytext=(7, 7), fontsize=8, path_effects=OUTLINE, zorder=7)
        if equal_aspect:
            # "box", not "datalim" -- datalim is illegal on shared axes.
            ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.set_xlabel(f"{AXIS_LABEL[h]} (mm)")
        ax.set_ylabel(f"{AXIS_LABEL[v]} (mm)")
        ax.legend(fontsize=7, loc="best")
 
    note = "  (o heel strike, square toe off)" if show_events else ""
    fig.suptitle(f"Sacrum-relative foot paths - {plane} plane{note}", fontsize=12)
    fig.tight_layout()
    return fig
 
 
def visualize_timeseries(trials, components=None, t_zero="first_LHS",
                         show_events=True, xlim=None):
    """Grid of markers (rows) against coordinates (columns), overlaid.
 
    The x axis is seconds relative to `t_zero`, not frames, so the two trials
    are directly comparable despite different sampling rates.
    """
    n_axes = min(t["n_axes"] for t in trials)
    components = tuple(range(n_axes)) if components is None else tuple(components)
    if max(components) >= n_axes:
        raise ValueError(
            f"component {max(components)} requested but the saved markers "
            f"have only {n_axes} columns")
 
    times = [time_vector(t, t_zero) for t in trials]
 
    fig, axs = plt.subplots(len(MARKERS), len(components),
                            figsize=(4.8 * len(components), 2.5 * len(MARKERS)),
                            sharex=True, squeeze=False)
 
    for r, name in enumerate(MARKERS):
        color, title = MARKER_STYLE[name]
        for c, comp in enumerate(components):
            ax = axs[r, c]
            for i, trial in enumerate(trials):
                y = trial["markers"][name][:, comp]
                ax.plot(times[i], y, color=color, label=trial["label"],
                        **TRIAL_STYLE[i])
                _draw_events(ax, trial, name, times[i], y, show_events, i)
            ax.axhline(0, linestyle=":", color="black", linewidth=0.8)
            ax.axvline(0, linestyle=":", color="black", linewidth=0.8)
            ax.grid(alpha=0.25)
            if r == 0:
                ax.set_title(AXIS_LABEL[comp])
            if c == 0:
                ax.set_ylabel(f"{title}\n(mm)")
            if r == len(MARKERS) - 1:
                ax.set_xlabel(f"Time (s), 0 = {t_zero.replace('first_', 'first ')}")
            if xlim is not None:
                ax.set_xlim(*xlim)
 
    axs[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Sacrum-relative marker trajectories"
                 + ("  (o heel strike, square toe off)" if show_events else ""),
                 fontsize=12)
    fig.tight_layout()
    return fig
 
 
# --------------------------------------------------------------------------
# Descriptive comparison (printed, not plotted)
# --------------------------------------------------------------------------
def _stride_times(trial, key="LHS"):
    ev = trial["events"][key]
    return np.diff(ev) / trial["frame_rate"] if len(ev) > 1 else np.array([])
 
 
def describe(trials, outlier_factor=1.5):
    head = (f"{'trial':<34}{'frames':>8}{'Hz':>6}{'dur s':>8}{'LHS':>5}{'RHS':>5}"
            f"{'stride s':>10}{'SD':>7}{'min':>7}{'max':>7}")
    print(head)
    print("-" * len(head))
 
    means = []
    for t in trials:
        st = _stride_times(t)
        m = st.mean() if st.size else np.nan
        means.append(m)
        print(f"{t['label'][:33]:<34}{t['n_frames']:>8}{t['frame_rate']:>6.0f}"
              f"{t['n_frames'] / t['frame_rate']:>8.2f}"
              f"{len(t['events']['LHS']):>5}{len(t['events']['RHS']):>5}"
              f"{m:>10.3f}{st.std() if st.size else np.nan:>7.3f}"
              f"{st.min() if st.size else np.nan:>7.3f}"
              f"{st.max() if st.size else np.nan:>7.3f}")
 
    # Per-trial stride-time outliers: a stride far from the median usually
    # means a missed or spurious heel strike, or a turn / pause inside the
    # trial. Either way the "mean stride time" above is not describing steady
    # walking, and neither is any overlay built on it.
    for t in trials:
        st = _stride_times(t)
        if st.size < 3:
            continue
        med = np.median(st)
        bad = np.flatnonzero((st > outlier_factor * med) |
                             (st < med / outlier_factor))
        if bad.size:
            lhs = t["events"]["LHS"]
            detail = ", ".join(f"{st[b]:.2f}s between frames {lhs[b]}-{lhs[b+1]}"
                               for b in bad)
            warnings.warn(
                f"{t['label']}: {bad.size} stride(s) off the median "
                f"({med:.2f}s) by more than {outlier_factor}x -- {detail}. "
                "Likely a missed/extra heel strike, a turn, or a pause.",
                RuntimeWarning)
 
    if all(np.isfinite(means)) and abs(means[0] - means[1]) > 0.15 * max(means):
        warnings.warn(
            "Mean stride time differs by more than 15% between trials. The "
            "curves will drift apart after the first stride no matter how "
            "well they are aligned -- that is a real cadence difference, not "
            "a plotting artefact.", RuntimeWarning)
 
 
# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(config):
    error_msgs = []
    trial = config.get("trial", {}) or {}
    inp = config.get("input", {}) or {}
    cmp_ = config.get("compare", {}) or {}
    plot_cfg = config.get("plot", {}) or {}
 
    session_name = trial.get("session")
    trial_name = trial.get("name")
    input_type = inp.get("type")
    compare_name = cmp_.get("name")
    compare_type = cmp_.get("type")
    # the comparison trial may come from a different session
    compare_session = cmp_.get("session", session_name)
 
    if not trial_name:
        error_msgs.append("!! Missing trial.name !!")
    if not input_type:
        error_msgs.append("!! Missing input.type !!")
    if not compare_name:
        error_msgs.append("!! Missing compare.name !!")
    if not compare_type:
        error_msgs.append("!! Missing compare.type !!")
    if input_type == "vicon" and not session_name:
        error_msgs.append("!! Missing trial.session (required for vicon) !!")
    if compare_type == "vicon" and not compare_session:
        error_msgs.append("!! Missing compare.session (required for vicon) !!")
    if error_msgs:
        raise InvalidConfigError(error_msgs)
 
    PROJECT_ROOT = find_project_root()
    overrides = plot_cfg.get("marker_keys")  # optional {canonical: npz_key}
    # NOTE: the comparison path is built from compare_type, not input_type.
    trials = [
        load_trial(gait_events_path(PROJECT_ROOT, input_type, trial_name,
                                    session_name),
                   f"{input_type}: {trial_name}", overrides),
        load_trial(gait_events_path(PROJECT_ROOT, compare_type, compare_name,
                                    compare_session),
                   f"{compare_type}: {compare_name}", overrides),
    ]
    for t in trials:
        print(f"Loaded {t['path'].name}  {t['n_frames']} frames x "
              f"{t['n_axes']} axes  keys={list(t['keys'].values())}")
    print()
    describe(trials)
    print()
 
    usable = available_planes(trials)
    # frontal is available whenever sagittal is, but adds little for foot
    # trajectories -- request it explicitly via plot.planes if you want it.
    wanted = plot_cfg.get("planes") or [p for p in usable if p != "frontal"]
    planes = [p for p in wanted if p in usable]
    if set(wanted) - set(usable):
        print(f"NOTE: requested {sorted(set(wanted) - set(usable))} but the "
              f"data does not have those axes.")
    if "sagittal" not in planes:
        print("NOTE: the saved markers are 2-column (ground plane only), so the "
              "sagittal view is unavailable. Re-export the GaitEvents files "
              "keeping the vertical axis to enable it.")
    print(f"Plotting planes: {planes}\n")
 
    t_zero = plot_cfg.get("t_zero", "first_LHS")
    for plane in planes:
        visualize_paths(trials, plane=plane, combined=(plane == "transverse"))
    visualize_timeseries(trials, t_zero=t_zero)
    plt.show()
 
 
if __name__ == "__main__":
    CONFIG_PATH = (find_project_root() / "src" / "gait_Analysis" /
                   "config" / "input_config.yaml")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    try:
        types = {(config.get("input") or {}).get("type"),
                 (config.get("compare") or {}).get("type")}
        if "vicon" in types:
            from viconnexusapi import ViconNexus  # noqa: F401
        main(config)
    except InvalidConfigError as e:
        for msg in e.error_msgs:
            print(msg)