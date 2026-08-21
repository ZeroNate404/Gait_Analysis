"""Visualisation of computed gait parameters.
 
Expects the per-cycle structure written by compute_gait_params():
 
    gait_params = [
        {"stepping_foot": "left" | "right",
         "cycle":         <per-foot cycle index>,
         "flags":         [<str>, ...],
         "params": {
             "step_order":            <foot>,
             "distance":              {foot: value},   # mm, per stride
             "step_length":           {foot: value},   # mm
             "step_width":            {foot: value},   # mm
             "step_frames":           {foot: value},   # frames
             "stride_length":         {foot: value},   # mm
             "stride_frames":         {foot: value},
             "stance_frames":         {foot: value},
             "swing_frames":          {foot: value},
             "single_support_frames": {foot: value},
             "double_support_frames": {foot: value},
         }},
        ...
    ]
 
Each entry is one *step event*, so a parameter dict holds exactly one key --
the stepping foot. Series for a given side are recovered by filtering on
"stepping_foot" while keeping the global cycle index for the x-axis, which
keeps the two feet correctly positioned relative to each other in time.
 
Temporal parameters are read as either "<base>_frames" (converted to seconds
here using frame_rate) or "<base>_time" (assumed already in seconds), so this
module keeps working if compute_cycle() switches units.
"""
 
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from gait_Analysis.utils.find_project_root import find_project_root
 
 
FEET = ("left", "right")
FOOT_COLOR = {"left": "tab:blue", "right": "tab:red"}
FOOT_LABEL = {"left": "Left", "right": "Right"}
 
# (kind, key, title, unit) -- kind "value" reads the parameter directly,
# kind "time" resolves <key>_frames / <key>_time and returns seconds.
TREND_SPECS = [
    ("value", "step_length",   "Step Length",   "mm"),
    ("value", "step_width",    "Step Width",    "mm"),
    ("time",  "step",          "Step Time",     "s"),
    ("value", "stride_length", "Stride Length", "mm"),
    ("time",  "stride",        "Stride Time",   "s"),
]
 
# (base name in params, display label)
PHASE_SPECS = [
    ("stance",         "Stance"),
    ("swing",          "Swing"),
    ("single_support", "Single\nSupport"),
    ("double_support", "Double\nSupport"),
]
 
 
# --------------------------------------------------------------------------- #
# Paths / loading
# --------------------------------------------------------------------------- #
def _params_path(session, trial_name, input_type):
    d = find_project_root() / "data" / "GaitParams" / input_type
    if(input_type == "vicon"):
        d = d / session
    return d / f"{trial_name}_GaitParams.npz"
 
 
def _events_path(session, trial_name, input_type):
    d = find_project_root() / "data" / "GaitEvents" / input_type
    if(input_type == "vicon"):
        d = d / session
    return d / f"{trial_name}_GaitEvents.npz"
 
 
def _load_gait_params(session, trial_name, input_type):
    """Load a saved *_GaitParams.npz file.
 
    The per-cycle dicts are stored as a 1-D object array under "gait_params";
    any other entries are scalars/arrays saved alongside (e.g. frame_rate).
 
    Returns (cycles, meta).
    """
    path = _params_path(session, trial_name, input_type)
    with np.load(path, allow_pickle=True) as data:
        if "gait_params" not in data.files:
            raise KeyError(
                f"{path.name} has no 'gait_params' key (found: {list(data.files)}). "
                "This file was written by an older version of compute_gait_params(); "
                "re-run it to regenerate."
            )
        cycles = [dict(c) for c in data["gait_params"]]
        meta = {
            k: (data[k].item() if data[k].ndim == 0 else data[k])
            for k in data.files if k != "gait_params"
        }
 
    if not cycles:
        raise ValueError(f"{path.name} contains no gait cycles.")
    return cycles, meta
 
 
def _resolve_frame_rate(meta, session, trial_name, input_type):
    """frame_rate, preferring the GaitParams file, falling back to GaitEvents.
 
    Adding `frame_rate=frame_rate` to the np.savez() call in
    compute_gait_params() makes the GaitParams file self-describing and stops
    the fallback below from ever being needed.
    """
    if "frame_rate" in meta:
        return float(meta["frame_rate"])
 
    events = _events_path(session, trial_name, input_type)
    if events.exists():
        with np.load(events) as d:
            if "frame_rate" in d.files:
                return float(d["frame_rate"])
 
    raise KeyError(
        "frame_rate not found in the GaitParams file, and it could not be "
        f"recovered from {events}. Every temporal parameter is stored in frames, "
        "so it cannot be converted to seconds without it. Save it alongside:\n"
        "    np.savez(SAVE_PATH, gait_params=..., frame_rate=frame_rate)"
    )
 
 
# --------------------------------------------------------------------------- #
# Series extraction
# --------------------------------------------------------------------------- #
def _find_key(cycles, *candidates):
    """First candidate parameter name that appears in any cycle. None if absent."""
    for name in candidates:
        if any(name in cd.get("params", {}) for cd in cycles):
            return name
    return None
 
 
def _empty():
    return np.empty(0, dtype=int), np.empty(0, dtype=float)
 
 
def _series(cycles, name, foot):
    """(cycle_indices, values) for parameter `name` on `foot`, in stepping order.
 
    Only cycles whose stepping_foot is `foot` are returned, so the caller gets a
    dense array of that side's values together with their *global* cycle index.
    """
    if name is None:
        return _empty()
    idx, vals = [], []
    for i, cd in enumerate(cycles):
        if cd.get("stepping_foot") != foot:
            continue
        raw = cd.get("params", {}).get(name, np.nan)
        if isinstance(raw, dict):                 # sparse per-foot parameter
            raw = raw.get(foot, np.nan)
        idx.append(i)
        vals.append(np.nan if raw is None else float(raw))
    return np.asarray(idx, dtype=int), np.asarray(vals, dtype=float)
 
 
def _time_series(cycles, base, foot, frame_rate):
    """Duration in seconds, from either <base>_frames or <base>_time."""
    key = _find_key(cycles, f"{base}_frames", f"{base}_time")
    if key is None:
        return _empty()
    idx, vals = _series(cycles, key, foot)
    if key.endswith("_frames"):
        vals = vals / frame_rate
    return idx, vals
 
 
def _trend_series(cycles, kind, key, foot, frame_rate):
    if kind == "time":
        return _time_series(cycles, key, foot, frame_rate)
    return _series(cycles, key, foot)
 
 
def _phase_percent(cycles, base, foot, frame_rate):
    """Phase duration as a percentage of that cycle's stride."""
    idx, sec = _time_series(cycles, base, foot, frame_rate)
    if idx.size == 0:
        return _empty()
    idx_stride, stride = _time_series(cycles, "stride", foot, frame_rate)
    if idx_stride.size != idx.size:
        return _empty()
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(stride > 0, sec / stride * 100.0, np.nan)
    return idx, pct
 
 
# --------------------------------------------------------------------------- #
# Aggregates / formatting
# --------------------------------------------------------------------------- #
def _nanmean(a):
    a = np.asarray(a, dtype=float)
    return np.nan if a.size == 0 or np.isnan(a).all() else float(np.nanmean(a))
 
 
def _nanstd(a):
    a = np.asarray(a, dtype=float)
    return np.nan if a.size == 0 or np.isnan(a).all() else float(np.nanstd(a))
 
 
def _fmt(v, decimals=1, suffix=""):
    return "n/a" if v is None or np.isnan(v) else f"{v:.{decimals}f}{suffix}"
 
 
def _flagged(cycles):
    """(indices of flagged cycles, {flag: count})."""
    idx, counts = [], {}
    for i, cd in enumerate(cycles):
        flags = cd.get("flags") or []
        if not flags:
            continue
        idx.append(i)
        for f in flags:
            counts[f] = counts.get(f, 0) + 1
    return idx, counts
 
 
def _speed_cadence(cycles, frame_rate):
    """Trial speed (mm/s), cadence (steps/min) and walking time (s).
 
    step_time tiles the trial exactly -- each value is the gap between
    consecutive heel strikes in the merged L/R sequence -- so summing the valid
    ones gives (last HS - first HS) without double counting.
 
    Speed uses per-cycle sacrum path distance divided by that cycle's stride
    time, then averages. Summing the distances directly would double count,
    because left and right strides overlap in time.
    """
    dist_key = _find_key(cycles, "stride_distance", "distance")
 
    step_times, cycle_speeds = [], []
    for foot in FEET:
        _, st = _time_series(cycles, "step", foot, frame_rate)
        step_times.extend(st[~np.isnan(st)].tolist())
 
        _, strt = _time_series(cycles, "stride", foot, frame_rate)
        _, dist = _series(cycles, dist_key, foot)
        if strt.size == dist.size and strt.size:
            with np.errstate(invalid="ignore", divide="ignore"):
                sp = np.where(strt > 0, dist / strt, np.nan)
            cycle_speeds.extend(sp[~np.isnan(sp)].tolist())
 
    total_time = float(np.sum(step_times)) if step_times else np.nan
    cadence = (len(step_times) / total_time * 60.0) if step_times and total_time > 0 else np.nan
    return _nanmean(cycle_speeds), cadence, total_time
 
 
# --------------------------------------------------------------------------- #
# Figure 1 -- trends
# --------------------------------------------------------------------------- #
def plot_gait_summary(config):
    """Figure 1: step/stride length, width, and time trends (Left vs Right), with
    per-side average reference lines, plus a single-trial speed/cadence summary panel.
    Saved as {trial}_GaitSummary.png."""
    session = config["trial"]["session"]
    trial_name = config["trial"]["name"]
    input_type = config["input"]["type"]
    cycles, meta = _load_gait_params(session, trial_name, input_type)
    frame_rate = _resolve_frame_rate(meta, session, trial_name, input_type)
    flag_idx, flag_counts = _flagged(cycles)
 
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Gait Summary — {trial_name}", fontsize=14, fontweight="bold")
 
    flat_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]
 
    for ax, (kind, key, title, unit) in zip(flat_axes, TREND_SPECS):
        # flagged cycles marked behind the data
        for i in flag_idx:
            ax.axvline(i, color="0.85", linewidth=1.2, zorder=0)
 
        title_bits = []
        for foot in FEET:
            idx, vals = _trend_series(cycles, kind, key, foot, frame_rate)
            valid = ~np.isnan(vals) if vals.size else np.zeros(0, dtype=bool)
 
            ax.plot(idx[valid], vals[valid], "o-", color=FOOT_COLOR[foot],
                    label=FOOT_LABEL[foot], markersize=4, zorder=2)
 
            avg = _nanmean(vals)
            if not np.isnan(avg):
                ax.axhline(avg, color=FOOT_COLOR[foot], linestyle="--",
                           alpha=0.5, linewidth=1, zorder=1)
            title_bits.append(f"{FOOT_LABEL[foot][0]} avg: {_fmt(avg, 1, unit)}")
 
        ax.set_title(f"{title}\n" + "   ".join(title_bits), fontsize=10)
        ax.set_xlabel("Cycle Index (merged L/R stepping order)")
        ax.set_ylabel(f"{title} ({unit})")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
 
    # --- Scalar summary panel: speed & cadence (single-trial only) ---
    # NOTE: cross-trial comparison for speed/cadence (and other params) is a
    # separate, deferred feature — this panel just reports this trial's values.
    # Both are derived from the per-cycle values; they are not stored as
    # scalars in the GaitParams file.
    ax_scalar = axes[1, 2]
    ax_scalar.axis("off")
    speed, cadence, total_time = _speed_cadence(cycles, frame_rate)
 
    ax_scalar.text(0.5, 0.82, _fmt(speed, 1), ha="center", va="center",
                   fontsize=28, fontweight="bold", color="tab:blue", transform=ax_scalar.transAxes)
    ax_scalar.text(0.5, 0.71, "Speed (mm/s)", ha="center", va="center",
                   fontsize=11, color="dimgray", transform=ax_scalar.transAxes)
    ax_scalar.text(0.5, 0.55, _fmt(cadence, 1), ha="center", va="center",
                   fontsize=28, fontweight="bold", color="tab:purple", transform=ax_scalar.transAxes)
    ax_scalar.text(0.5, 0.44, "Cadence (steps/min)", ha="center", va="center",
                   fontsize=11, color="dimgray", transform=ax_scalar.transAxes)
    ax_scalar.text(0.5, 0.29, f"{len(cycles)} steps over {_fmt(total_time, 2, ' s')}",
                   ha="center", va="center", fontsize=10, color="dimgray",
                   transform=ax_scalar.transAxes)
 
    if flag_counts:
        detail = ", ".join(f"{k} x{v}" for k, v in sorted(flag_counts.items()))
        ax_scalar.text(0.5, 0.18, f"{len(flag_idx)} flagged cycle(s)", ha="center",
                       va="center", fontsize=10, color="tab:orange", fontweight="bold",
                       transform=ax_scalar.transAxes)
        ax_scalar.text(0.5, 0.12, detail, ha="center", va="center", fontsize=8,
                       color="tab:orange", transform=ax_scalar.transAxes, wrap=True)
    else:
        ax_scalar.text(0.5, 0.16, "no flagged cycles", ha="center", va="center",
                       fontsize=9, color="darkseagreen", transform=ax_scalar.transAxes)
 
    ax_scalar.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.88, fill=False,
                                        edgecolor="lightgray", linewidth=1.2,
                                        transform=ax_scalar.transAxes))
 
    plt.tight_layout(rect=(0, 0, 1, 0.96))
 
    # Save visualization graphs
    PROJECT_ROOT = find_project_root()
    SAVE_DIR = PROJECT_ROOT / "data" / "GaitReports" / input_type
    if(input_type == "vicon"):
        SAVE_DIR = SAVE_DIR / session
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_PATH = SAVE_DIR / f"{trial_name}_GaitSummary.png"
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return SAVE_PATH
 
 
# --------------------------------------------------------------------------- #
# Figure 2 -- cycle phases
# --------------------------------------------------------------------------- #
def plot_gait_cycle_phases(config):
    """Figure 2: stance/swing and single/double support, per side, shown in
    seconds and as a percentage of the stride, side by side. Bars are means
    across cycles with +/-1 SD error bars. Saved as {trial}_GaitCyclePhases.png."""
    session = config["trial"]["session"]
    trial_name = config["trial"]["name"]
    input_type = config["input"]["type"]
    cycles, meta = _load_gait_params(session, trial_name, input_type)
    frame_rate = _resolve_frame_rate(meta, session, trial_name, input_type)
 
    labels = [label for _, label in PHASE_SPECS]
    x = np.arange(len(PHASE_SPECS))
    width = 0.38
 
    # stats[foot][unit] -> (means, sds)
    stats = {}
    for foot in FEET:
        sec_m, sec_s, pct_m, pct_s = [], [], [], []
        for base, _ in PHASE_SPECS:
            _, sec = _time_series(cycles, base, foot, frame_rate)
            _, pct = _phase_percent(cycles, base, foot, frame_rate)
            sec_m.append(_nanmean(sec));  sec_s.append(_nanstd(sec))
            pct_m.append(_nanmean(pct));  pct_s.append(_nanstd(pct))
        stats[foot] = {"s": (sec_m, sec_s), "%": (pct_m, pct_s)}
 
    fig, (ax_s, ax_pct) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Gait Cycle Phases — {trial_name}", fontsize=14, fontweight="bold")
 
    for ax, unit, ylabel, panel_title, decimals, suffix in (
        (ax_s,   "s", "Mean Time across Cycles (s) ± 1 SD",            "Seconds", 2, ""),
        (ax_pct, "%", "Mean Percent of Gait Cycle (%) ± 1 SD",         "Percent", 1, "%"),
    ):
        for k, foot in enumerate(FEET):
            means, sds = stats[foot][unit]
            offset = (k - 0.5) * width
            bars = ax.bar(x + offset, means, width,
                          yerr=np.nan_to_num(sds, nan=0.0), capsize=3,
                          color=FOOT_COLOR[foot], label=FOOT_LABEL[foot])
            for rect, v in zip(bars, means):
                if np.isnan(v):
                    continue
                ax.text(rect.get_x() + rect.get_width() / 2, v,
                        f"{v:.{decimals}f}{suffix}", ha="center", va="bottom", fontsize=8)
 
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.legend(fontsize=9, loc="best")
        ax.grid(axis="y", alpha=0.3)
 
    fig.text(0.5, 0.005,
             "Stance + Swing = one stride; Single + Double Support = Stance by "
             "construction (double_support = stance - single_support). "
             "Percentages are computed per cycle against that cycle's own stride, "
             "then averaged. Error bars: ± 1 SD across cycles.",
             ha="center", fontsize=8, color="dimgray")
 
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
 
    # Save visualization graphs
    PROJECT_ROOT = find_project_root()
    SAVE_DIR = PROJECT_ROOT / "data" / "GaitReports" / input_type
    if(input_type == "vicon"):
        SAVE_DIR = SAVE_DIR / session
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_PATH = SAVE_DIR / f"{trial_name}_GaitCyclePhases.png"
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return SAVE_PATH
 
 
class InvalidConfigError(Exception):
    def __init__(self, error_msgs):
        self.error_msgs = error_msgs
        super().__init__("\n".join(error_msgs))
 
 
if __name__ == "__main__":
    # Example usage
    CONFIG_PATH = find_project_root() / "src" / "gait_Analysis" / "config" / "input_config.yaml"
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    try:
        print(plot_gait_summary(config))
        print(plot_gait_cycle_phases(config))
    except InvalidConfigError as e:
        for msg in e.error_msgs:
            print(msg)