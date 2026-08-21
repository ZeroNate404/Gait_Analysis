import os, yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from gait_Analysis.Visualize_Param import plot_gait_summary, plot_gait_cycle_phases
from gait_Analysis.utils.find_project_root import find_project_root
from gait_Analysis.utils.save_to_csv import save_gait_params_csv

def get_stride_distance(data, stepping_frame, prev_frame):
    if(prev_frame is None): return np.nan # First Step
    SACR_arr = data['sacrum_arr']
    stride_distance = np.sum(np.linalg.norm(np.diff(SACR_arr[prev_frame:stepping_frame+1], axis=0), axis=1))
    return stride_distance

def get_step_length_width(HEE_step_arr, HEE_anchor_arr, stepping_frame, anchor_frame, prev_frame):
    if(prev_frame is None): return np.nan, np.nan  # First step without previous ipsi foot
    # Calculate step length and width using the law of cosines
    c = np.linalg.norm(HEE_step_arr[stepping_frame] - HEE_step_arr[prev_frame])
    a = np.linalg.norm(HEE_anchor_arr[anchor_frame] - HEE_step_arr[prev_frame])
    b = np.linalg.norm(HEE_step_arr[stepping_frame] - HEE_anchor_arr[anchor_frame])
    step_length = (b**2 + c**2 - a**2)/(2*c)
    step_width = np.sqrt(max(0.0,b**2 - step_length**2))
    return step_length, step_width

def get_step_time(stepping_frame, anchor_frame):
    if(anchor_frame is None): return np.nan  # First Step
    step_frames = stepping_frame - anchor_frame
    return step_frames

def get_stride_length(HEE_step_arr, stepping_frame, prev_frame):
    if(prev_frame is None): return np.nan  # First stride
    stride_length = np.linalg.norm(HEE_step_arr[stepping_frame] - HEE_step_arr[prev_frame])
    return stride_length

def get_stride_time(stepping_frame, prev_frame):
    if(prev_frame is None): return np.nan  # First stride
    stride_frames = stepping_frame - prev_frame
    return stride_frames

def get_swing_stance_time(stepping_frame, lift_frame, anchor_frame, prev_frame):
    if(anchor_frame is None): return np.nan, np.nan  # First Step
    swing_frames = stepping_frame - lift_frame
    if(prev_frame is None): stance_frames = np.nan
    else: stance_frames = lift_frame - prev_frame
    return swing_frames, stance_frames

def get_single_double_support_time(anchor_frame, anchor_lift_frame, stance_frame):
    if(anchor_frame is None or anchor_lift_frame is None): return np.nan, np.nan  # First Step
    single_support_frames = anchor_frame - anchor_lift_frame
    double_support_frames = stance_frame - single_support_frames
    return single_support_frames, double_support_frames

def compute_cycle(data:dict, stepping_data:tuple, lift_data:tuple, anchor_data:tuple, anchor_lift_data:tuple, prev_ipsi:tuple) -> dict:
    stepping_foot, stepping_frame= stepping_data
    lift_foot, lift_frame = lift_data
    anchor_foot, anchor_frame = anchor_data
    anchor_lift_foot, anchor_lift_frame = anchor_lift_data
    prev_foot, prev_frame = prev_ipsi
    HEE_step_arr = data['LHarr'] if stepping_foot == "left" else data['RHarr']
    HEE_anchor_arr = data['RHarr'] if anchor_foot == "right" else data['LHarr']

    stride_distance = get_stride_distance(data, stepping_frame, prev_frame)
    step_length, step_width = get_step_length_width(HEE_step_arr, HEE_anchor_arr, stepping_frame, anchor_frame, prev_frame)
    step_frames = get_step_time(stepping_frame, anchor_frame)
    stride_length = get_stride_length(HEE_step_arr, stepping_frame, prev_frame)
    stride_frames = get_stride_time(stepping_frame, prev_frame)
    swing_frames, stance_frames = get_swing_stance_time(stepping_frame, lift_frame, anchor_frame, prev_frame)
    single_support_frames, double_support_frames = get_single_double_support_time(anchor_frame, anchor_lift_frame, stance_frames)

    # Save to dict
    cycle_params = {}
    cycle_params["step_order"] = stepping_foot
    cycle_params.setdefault("distance", {})[stepping_foot] = stride_distance
    cycle_params.setdefault("step_width", {})[stepping_foot] = step_width
    cycle_params.setdefault("step_length", {})[stepping_foot] = step_length
    cycle_params.setdefault("step_frames", {})[stepping_foot] = step_frames
    cycle_params.setdefault("stride_length", {})[stepping_foot] = stride_length
    cycle_params.setdefault("stride_frames", {})[stepping_foot] = stride_frames
    cycle_params.setdefault("stance_frames", {})[stepping_foot] = stance_frames
    cycle_params.setdefault("swing_frames", {})[stepping_foot] = swing_frames
    cycle_params.setdefault("single_support_frames", {})[stepping_foot] = single_support_frames
    cycle_params.setdefault("double_support_frames", {})[stepping_foot] = double_support_frames

    return cycle_params

def build_step_sequence(LHS, RHS):
    """Merge the two heel-strike series into one chronological step sequence.

    Every index question is answered once, here. Each entry carries the four
    reference frames the per-cycle computation needs.
    """
    events = sorted(
        [("left", f) for f in LHS] + [("right", f) for f in RHS],
        key=lambda e: e[1],
    )

    steps, seen = [], {"left": [], "right": []}
    for i, (foot, frame) in enumerate(events):
        anchor_foot, anchor_frame = events[i - 1] if i > 0 else (None, None)

        # previous ipsilateral heel strike
        prev_frame = seen[foot][-1] if seen[foot] else None

        # the anchor foot's own previous heel strike:
        # seen[anchor_foot][-1] IS anchor_frame (appended last iteration), so
        # the one before it is [-2].
        hist = seen[anchor_foot] if anchor_foot else []
        anchor_prev_frame = hist[-2] if len(hist) >= 2 else None

        steps.append({
            "index":             i,
            "foot_cycle":        len(seen[foot]),
            "stepping_foot":     foot,
            "stepping_frame":    frame,
            "anchor_foot":       anchor_foot,
            "anchor_frame":      anchor_frame,
            "prev_frame":        prev_frame,
            "anchor_prev_frame": anchor_prev_frame,
        })
        seen[foot].append(frame)

    return steps

def last_before(toe_offs, upper, lower=None):
    """Largest toe-off strictly below `upper` (and strictly above `lower`, if
    given). None when nothing qualifies. `toe_offs` must be sorted ascending."""
    if upper is None:
        return None
    cand = toe_offs[toe_offs < upper]
    if lower is not None:
        cand = cand[cand > lower]
    return int(cand[-1]) if cand.size else None

def compute_gait_params(config):
    # Get INPUT
    error_msgs = []
    session_name = config.get("trial", {}).get("session")
    trial_name = config.get("trial", {}).get("name")
    input_type = config.get("input", {}).get("type")
    if not session_name : error_msgs.append("!! Missing trial session !!")
    if not trial_name : error_msgs.append("!! Missing trial name !!")
    if not input_type : error_msgs.append("!! Missing input type !!")
    if(error_msgs): raise InvalidConfigError(error_msgs)

    # Extract GaitEvents file
    PROJECT_ROOT = find_project_root()
    INPUT_DIR = PROJECT_ROOT / "data" / "GaitEvents" / input_type
    if input_type == "vicon": INPUT_DIR = INPUT_DIR / session_name / f"{trial_name}_GaitEvents.npz"
    else: INPUT_DIR = INPUT_DIR / f"{trial_name}_GaitEvents.npz"
    with np.load(INPUT_DIR) as data:
        # Get gait events and other relevant data from the loaded file
        LHS, LTO, RHS, RTO = (a-1 for a in [data['LHS'], data['LTO'], data['RHS'], data['RTO']]) # 1 -> 0 Indexing
        LHS = np.delete(LHS, 4)
        sacrum_arr = data['sacrum_arr']
        frame_rate = data['frame_rate']
        if(len(LHS) ==0 or len(RHS) == 0): raise ValueError("No heel strikes detected for one or both feet. Cannot compute gait parameters.")

        # Total time between first step and last
        first_stepping_foot = "left" if (LHS[0] < RHS[0]) else "right"
        starting_IC = min(LHS[0], RHS[0])
        ending_IC = max(LHS[-1], RHS[-1])
        total_time = (ending_IC - starting_IC) / frame_rate
        LTO = LTO[(LTO >= starting_IC) & (LTO <= ending_IC)]
        RTO = RTO[(RTO >= starting_IC) & (RTO <= ending_IC)]
        arrays = {
            "sacrum_arr": data["sacrum_arr"],
            "LHarr":      data["LHarr"],
            "RHarr":      data["RHarr"],
            "frame_rate": float(data["frame_rate"]),
        }

    # Gait parameters to compute
    gait_params = []

    # Compute gait parameters for each cycle
    TO = {"left": np.sort(LTO), "right": np.sort(RTO)}

    for s in build_step_sequence(LHS, RHS):
        stepping_foot, anchor_foot = s["stepping_foot"], s["anchor_foot"]
        lift_frame = last_before(TO[stepping_foot], s["stepping_frame"], s["prev_frame"])
        anchor_lift_frame = (
            last_before(TO[anchor_foot], s["anchor_frame"], s["anchor_prev_frame"])
            if anchor_foot else None
        )

        flags = []
        if anchor_foot == stepping_foot:
            flags.append("ipsilateral_anchor")          # two same-side strikes adjacent
        if lift_frame is not None and s["anchor_frame"] is not None \
                and lift_frame < s["anchor_frame"]:
            flags.append("toe_off_before_anchor_hs")    # flight phase or bad event

        # COMPUTE GAIT PARAMETERS FOR THIS CYCLE
        
        cycle_params = compute_cycle(
                        arrays,
                        (stepping_foot, s["stepping_frame"]),
                        (stepping_foot, lift_frame),
                        (anchor_foot, s["anchor_frame"]),
                        (anchor_foot, anchor_lift_frame),
                        (stepping_foot, s["prev_frame"]),
                    )
        cycle_data = {
            "stepping_foot": stepping_foot,
            "cycle": s["foot_cycle"],
            "flags": flags,
            "params": cycle_params
        }
        gait_params.append(cycle_data)

    # Save the Computed Gait Parameters to NPZ
    SAVE_DIR = PROJECT_ROOT / "data" / "GaitParams" / input_type
    if input_type == "vicon": SAVE_DIR = SAVE_DIR / session_name
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_PATH = SAVE_DIR / f"{trial_name}_GaitParams.npz"
    np.savez(SAVE_PATH, gait_params=np.array(gait_params, dtype=object))

    # Save the Computed Gait Parameters to CSV
    SAVE_PATH = SAVE_DIR / f"{trial_name}_GaitParams.csv"
    save_gait_params_csv(gait_params, SAVE_PATH)

    # Visualize the acquired gait parameters
    plot_gait_summary(config)
    plot_gait_cycle_phases(config)

class InvalidConfigError(Exception):
    def __init__(self, error_msgs):
        self.error_msgs = error_msgs
        super().__init__("\n".join(error_msgs))

with open("src/gait_Analysis/config/input_config.yaml", "r") as f:
    config = yaml.safe_load(f)
try:
    compute_gait_params(config)
except InvalidConfigError as e:
    for msg in e.error_msgs:
        print(msg)