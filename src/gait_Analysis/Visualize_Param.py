import os, yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from gait_Analysis.utils.find_project_root import find_project_root


def _load_gait_params(session, trial_name, input_type):
    """Load a saved *_GaitParams.npz file back into a plain nested-dict structure.
    Nested dicts (step_lengths, stance_time, etc.) are saved by np.savez as 0-d
    object arrays, so they need .item() to unwrap back into real dicts."""

    PROJECT_ROOT = find_project_root()
    INPUT_DIR = PROJECT_ROOT / "data" / "GaitParams" / input_type
    if(input_type == "vicon"):
        INPUT_DIR = INPUT_DIR / session
    INPUT_PATH = INPUT_DIR / f"{trial_name}_GaitParams.npz"
    data = np.load(INPUT_PATH, allow_pickle=True)

    gait_params = {}
    for key in data.files:
        val = data[key]
        gait_params[key] = val.item() if val.ndim == 0 else val
    data.close()
    return gait_params


def plot_gait_summary(config):
    """Figure 1: step/stride length, width, and time trends (Left vs Right), with
    per-side average reference lines, plus a single-trial speed/cadence summary panel.
    Saved as {trial}_GaitSummary.png."""
    session = config["trial"]["session"]
    trial_name = config["trial"]["name"]
    input_type = config["input"]["type"]
    gait_params = _load_gait_params(session, trial_name, input_type)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Gait Summary — {trial_name}", fontsize=14, fontweight="bold")

    panel_specs = [
        (axes[0, 0], "step_lengths", "Step Length", "mm"),
        (axes[0, 1], "step_widths", "Step Width", "mm"),
        (axes[0, 2], "step_times", "Step Time", "s"),
        (axes[1, 0], "stride_lengths", "Stride Length", "mm"),
        (axes[1, 1], "stride_times", "Stride Time", "s"),
    ]

    for ax, key, title, unit in panel_specs:
        param = gait_params[key]
        L, R = np.asarray(param["left"], dtype=float), np.asarray(param["right"], dtype=float)

        ax.plot(range(len(L)), L, "o-", color="tab:blue", label="Left", markersize=4)
        ax.plot(range(len(R)), R, "o-", color="tab:red", label="Right", markersize=4)

        L_avg, R_avg = np.nanmean(L), np.nanmean(R)
        ax.axhline(L_avg, color="tab:blue", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(R_avg, color="tab:red", linestyle="--", alpha=0.5, linewidth=1)

        ax.set_title(f"{title}\nL avg: {L_avg:.1f}{unit}   R avg: {R_avg:.1f}{unit}", fontsize=10)
        ax.set_xlabel("Step / Stride Index")
        ax.set_ylabel(f"{title} ({unit})")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    # --- Scalar summary panel: speed & cadence (single-trial only) ---
    # NOTE: cross-trial comparison for speed/cadence (and other params) is a
    # separate, deferred feature — this panel just reports this trial's values.
    ax_scalar = axes[1, 2]
    ax_scalar.axis("off")
    speed, cadence = gait_params["speed"], gait_params["cadence"]

    ax_scalar.text(0.5, 0.72, f"{speed:.1f}", ha="center", va="center",
                   fontsize=30, fontweight="bold", color="tab:blue", transform=ax_scalar.transAxes)
    ax_scalar.text(0.5, 0.58, "Speed (mm/s)", ha="center", va="center",
                   fontsize=11, color="dimgray", transform=ax_scalar.transAxes)
    ax_scalar.text(0.5, 0.36, f"{cadence:.1f}", ha="center", va="center",
                   fontsize=30, fontweight="bold", color="tab:purple", transform=ax_scalar.transAxes)
    ax_scalar.text(0.5, 0.22, "Cadence (steps/min)", ha="center", va="center",
                   fontsize=11, color="dimgray", transform=ax_scalar.transAxes)
    ax_scalar.add_patch(plt.Rectangle((0.05, 0.08), 0.9, 0.84, fill=False,
                                        edgecolor="lightgray", linewidth=1.2,
                                        transform=ax_scalar.transAxes))

    plt.tight_layout()

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


def plot_gait_cycle_phases(config):
    """Figure 2: stance/swing (per side) and single/double support (trial-level),
    shown in seconds and percent side by side. Saved as {trial}_GaitCyclePhases.png."""
    session = config["trial"]["session"]
    trial_name = config["trial"]["name"]
    input_type = config["input"]["type"]
    gait_params = _load_gait_params(session, trial_name, input_type)

    stance, swing = gait_params["stance_time"], gait_params["swing_time"]
    single_support, double_support = gait_params["single_support_time"], gait_params["double_support_time"]

    labels = ["L Stance", "L Swing", "R Stance", "R Swing", "Single\nSupport", "Double\nSupport"]
    seconds_vals = [stance["left"]["s"], swing["left"]["s"], stance["right"]["s"], swing["right"]["s"],
                     single_support["s"], double_support["s"]]
    percent_vals = [stance["left"]["%"], swing["left"]["%"], stance["right"]["%"], swing["right"]["%"],
                     single_support["%"], double_support["%"]]
    # per-side bars colored by foot; trial-level (single/double support) bars get distinct colors
    colors = ["tab:blue", "tab:blue", "tab:red", "tab:red", "tab:green", "tab:orange"]

    fig, (ax_s, ax_pct) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Gait Cycle Phases — {trial_name}", fontsize=14, fontweight="bold")

    ax_s.bar(labels, seconds_vals, color=colors)
    ax_s.set_ylabel("Time (s)")
    ax_s.set_title("Seconds")
    ax_s.grid(axis="y", alpha=0.3)
    for i, v in enumerate(seconds_vals):
        ax_s.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax_pct.bar(labels, percent_vals, color=colors)
    ax_pct.set_ylabel("Percent of Gait Cycle (%)")
    ax_pct.set_title("Percent")
    ax_pct.grid(axis="y", alpha=0.3)
    for i, v in enumerate(percent_vals):
        ax_pct.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

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
    with open("src/gait_analysis/config/input_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    try:
        plot_gait_summary(config)
        plot_gait_cycle_phases(config)
    except InvalidConfigError as e:
        for msg in e.error_msgs:
            print(msg)