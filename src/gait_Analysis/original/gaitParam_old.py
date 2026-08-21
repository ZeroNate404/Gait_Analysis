import os, yaml, sys, csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import zip_longest
from gait_Analysis.Visualize_Param import plot_gait_summary, plot_gait_cycle_phases
from gait_Analysis.utils.find_project_root import find_project_root

def get_speed(SACR_arr, total_time):
    # Calculate the total cumulative sacrum displacements as path length
    total_distance = np.sum(np.linalg.norm(np.diff(SACR_arr, axis=0), axis=1))
    speed = total_distance / total_time
    return speed

def get_cadence(LHS, RHS, total_time):
    num_of_steps = len(LHS) + len(RHS) - 1  # NOT INCLUDING THE FIRST STEP
    cadence = num_of_steps / total_time * 60.0  # steps per minute NOT INCLUDING THE FIRST STEP
    return cadence

def get_step_length_width(LHEE_arr, RHEE_arr, LHS, RHS):
    L_step_lengths, R_step_lengths = [], []
    L_step_widths, R_step_widths = [], []
    total_step_length, total_step_width = 0, 0
    l_idx, r_idx = 0,0

    while (l_idx < len(LHS) or r_idx < len(RHS)):
        # Which foot is stepping next
        stepping_foot = "left" if (l_idx < len(LHS) and (r_idx >= len(RHS) or LHS[l_idx] < RHS[r_idx])) else "right"
        if(stepping_foot == "left"):
            if(l_idx == 0):  # First step length is undefined
                L_step_lengths.append(np.nan)
                L_step_widths.append(np.nan)
            else:
                c = np.linalg.norm(LHEE_arr[LHS[l_idx]] - LHEE_arr[LHS[l_idx-1]])
                a = np.linalg.norm(RHEE_arr[RHS[r_idx-1]] - LHEE_arr[LHS[l_idx-1]])
                b = np.linalg.norm(LHEE_arr[LHS[l_idx]] - RHEE_arr[RHS[r_idx-1]])
                step_length = (b**2 + c**2 - a**2)/(2*c)
                L_step_lengths.append(step_length)
                total_step_length += step_length

                step_width = np.sqrt(b**2 - step_length**2)
                L_step_widths.append(step_width)
                total_step_width += step_width
            l_idx += 1
        else:  # stepping_foot == "right"
            if(r_idx == 0):  # First step length is undefined
                R_step_lengths.append(np.nan)
                R_step_widths.append(np.nan)
            else:
                c = np.linalg.norm(RHEE_arr[RHS[r_idx]] - RHEE_arr[RHS[r_idx-1]])
                a = np.linalg.norm(LHEE_arr[LHS[l_idx-1]] - RHEE_arr[RHS[r_idx-1]])
                b = np.linalg.norm(RHEE_arr[RHS[r_idx]] - LHEE_arr[LHS[l_idx-1]])
                step_length = (b**2 + c**2 - a**2)/(2*c)
                R_step_lengths.append(step_length)
                total_step_length += step_length

                step_width = np.sqrt(b**2 - step_length**2)
                R_step_widths.append(step_width)
                total_step_width += step_width
            r_idx += 1
    avg_step_length = total_step_length / (len(LHS)-1 + len(RHS)-1)  # NOT INCLUDING THE FIRST STEP OF EACH FOOT
    avg_step_width = total_step_width / (len(LHS)-1 + len(RHS)-1) # NOT INCLUDING THE FIRST STEP OF EACH FOOT
    return avg_step_length, L_step_lengths, R_step_lengths, avg_step_width, L_step_widths, R_step_widths

def get_step_time(LHS, RHS, frame_rate=100):
    L_step_times, R_step_times = [], []
    total_step_time = 0
    l_idx, r_idx = 0,0
    stepping_foot = "left" if (l_idx < len(LHS) and (r_idx >= len(RHS) or LHS[l_idx] < RHS[r_idx])) else "right"
    if(stepping_foot == "left"):
        L_step_times.append(np.nan)  # First step time is undefined
        l_idx += 1
    else:
        R_step_times.append(np.nan)  # First step time is undefined
        r_idx += 1

    while(l_idx < len(LHS) or r_idx < len(RHS)):
        # Which foot is stepping next
        stepping_foot = "left" if (l_idx < len(LHS) and (r_idx >= len(RHS) or LHS[l_idx] < RHS[r_idx])) else "right"
        if(stepping_foot == "left"):
            step_before = max(LHS[l_idx-1], RHS[r_idx-1]) if (l_idx > 0 and r_idx > 0) else (LHS[l_idx-1] if l_idx > 0 else RHS[r_idx-1])
            step_time = (LHS[l_idx] - step_before)/frame_rate  # Convert to seconds
            L_step_times.append(step_time)
            total_step_time += step_time
            l_idx += 1
        else:  # stepping_foot == "right"
            step_before = max(LHS[l_idx-1], RHS[r_idx-1]) if (l_idx > 0 and r_idx > 0) else (LHS[l_idx-1] if l_idx > 0 else RHS[r_idx-1])
            step_time = (RHS[r_idx] - step_before)/frame_rate  # Convert to seconds
            R_step_times.append(step_time)
            total_step_time += step_time
            r_idx += 1

    avg_step_time = total_step_time / (len(LHS) + len(RHS)-1)  # NOT INCLUDING THE FIRST STEP
    return avg_step_time, L_step_times, R_step_times

def get_stride_length(LHEE_arr, RHEE_arr, LHS, RHS):
    L_stride_lengths, R_stride_lengths = np.array([np.nan]), np.array([np.nan])
    total_stride_length = 0

    for i in range (1, len(LHS)):
        stride_length = np.linalg.norm(LHEE_arr[LHS[i]] - LHEE_arr[LHS[i-1]])
        L_stride_lengths = np.append(L_stride_lengths, stride_length)
        total_stride_length += stride_length
    for i in range (1, len(RHS)):
        stride_length = np.linalg.norm(RHEE_arr[RHS[i]] - RHEE_arr[RHS[i-1]])
        R_stride_lengths = np.append(R_stride_lengths, stride_length)
        total_stride_length += stride_length
    avg_stride_length = total_stride_length / (len(LHS)-1 + len(RHS)-1)  # NOT INCLUDING THE FIRST STRIDE OF EACH FOOT
    return avg_stride_length, L_stride_lengths, R_stride_lengths

def get_stride_time(LHS, RHS, frame_rate=100):
    L_stride_times, R_stride_times = np.array([np.nan]), np.array([np.nan])
    total_stride_time = 0
    for i in range(1, len(LHS)):
        stride_time = (LHS[i] - LHS[i-1])/frame_rate  # Convert to seconds
        L_stride_times = np.append(L_stride_times, stride_time)
        total_stride_time += stride_time    
    for i in range(1, len(RHS)):
        stride_time = (RHS[i] - RHS[i-1])/frame_rate  # Convert to seconds
        R_stride_times = np.append(R_stride_times, stride_time)
        total_stride_time += stride_time
    avg_stride_time = total_stride_time / (len(LHS)-1 + len(RHS)-1)  # NOT INCLUDING THE FIRST STRIDE OF EACH FOOT

    return avg_stride_time, L_stride_times, R_stride_times

def get_swing_time(LHS, LTO, RHS, RTO, frame_rate=100):
    L_swing_times = np.array([])
    R_swing_times = np.array([])

    # At this point, there must be at least 1 heel strike for each foot
    for i in range(0, len(LHS)):
        # Find the toe-off that occured before
        valid_to = None
        for toe_off in reversed(LTO):
            if(toe_off < LHS[i]):
                if(i==0):
                    valid_to = toe_off
                    break
                else:
                    # Ensure that the toe-off is after the previous heel strike
                    if(toe_off > LHS[i-1]):
                        valid_to = toe_off
                        break
        if valid_to is not None: L_swing_times = np.append(L_swing_times, (LHS[i] - valid_to) / frame_rate)  # Convert to seconds
        else: L_swing_times = np.append(L_swing_times, np.nan)  # If no valid toe-off is found, append NaN

    for i in range(0, len(RHS)):
        # Find the toe-off that occured before
        valid_to = None
        for toe_off in reversed(RTO):
            if(toe_off < RHS[i]):
                if(i==0):
                    valid_to = toe_off
                    break
                else:
                    # Ensure that the toe-off is after the previous heel strike
                    if(toe_off > RHS[i-1]):
                        valid_to = toe_off
                        break
        if valid_to is not None: R_swing_times = np.append(R_swing_times, (RHS[i] - valid_to) / frame_rate)  # Convert to seconds
        else: R_swing_times = np.append(R_swing_times, np.nan)  # If no valid toe-off is found, append NaN

    return L_swing_times, R_swing_times

def get_support_time(L_stride_times, R_stride_times, L_swing_times, R_swing_times, first_stepping_foot):
    L_double_support_times_s = np.array([np.nan])
    R_double_support_times_s = np.array([np.nan])
    L_single_support_times_s = np.array([np.nan])
    R_single_support_times_s = np.array([np.nan])
    l,r = 0,0
    if first_stepping_foot == "left": l+=1
    else: r+=1
    
    for i in range(1, len(L_stride_times)):
        double_support_time = L_stride_times[i] - L_swing_times[i] - R_swing_times[i-l]
        L_double_support_times_s = np.append(L_double_support_times_s, double_support_time)
        single_support_time = L_stride_times[i] - R_swing_times[i-l]
        L_single_support_times_s = np.append(L_single_support_times_s, single_support_time)

    for i in range(1, len(R_stride_times)):
        double_support_time = R_stride_times[i] - R_swing_times[i] - L_swing_times[i-r]
        R_double_support_times_s = np.append(R_double_support_times_s, double_support_time)
        single_support_time = R_stride_times[i] - L_swing_times[i-r]
        R_single_support_times_s = np.append(R_single_support_times_s, single_support_time)

    return L_double_support_times_s, R_double_support_times_s, L_single_support_times_s, R_single_support_times_s



def compute_gait_params(config):
    # Get INPUT
    error_msgs = []
    session_name = config["trial"]["session"]
    trial_name = config["trial"]["name"]
    input_type = config["input"]["type"]
    if not session_name : error_msgs.append("!! Missing trial session !!")
    if not trial_name : error_msgs.append("!! Missing trial name !!")
    if not input_type : error_msgs.append("!! Missing input type !!")
    if(error_msgs): raise InvalidConfigError

    # Extract GaitEvents file
    PROJECT_ROOT = find_project_root()
    INPUT_DIR = PROJECT_ROOT / "data" / "GaitEvents" / input_type
    if input_type == "vicon": INPUT_DIR = INPUT_DIR / session_name / f"{trial_name}_GaitEvents.npz"
    else: INPUT_DIR = INPUT_DIR / f"{trial_name}_GaitEvents.npz"
    data = np.load(INPUT_DIR)

    # Get gait events and other relevant data from the loaded file
    LHEE_arr, LTOE_arr, RHEE_arr, RTOE_arr = data['LHarr'], data['LTarr'], data['RHarr'], data['RTarr']
    LHS, LTO, RHS, RTO = data['LHS'], data['LTO'], data['RHS'], data['RTO']
    sacrum_arr = data['sacrum_arr']
    frame_rate = data['frame_rate']
    data.close()  # Close the file after loading the data

    if(len(LHS) ==0 or len(RHS) == 0): raise ValueError("No heel strikes detected for one or both feet. Cannot compute gait parameters.")

    # Total time between first step and last (with 100Hz frame rate)
    first_stepping_foot = "left" if (LHS[0] < RHS[0]) else "right"
    starting_IC = min(LHS[0], RHS[0])
    ending_IC = max(LHS[-1], RHS[-1])
    total_time = (ending_IC - starting_IC) / frame_rate
    LTO = LTO[(LTO >= starting_IC) & (LTO <= ending_IC)]
    RTO = RTO[(RTO >= starting_IC) & (RTO <= ending_IC)]
    LHS = np.delete(LHS,4)

    # Compute each gait parameter
    '''
    ======================HOW GAIT PARAMETERS ARE CALCULATED==========================
    (1-based)       starting_IC         : frame of the first heel strike
    (1-based)       ending_IC           : frame of the last heel strike
    (s)             total_time          : time between the first and last heel strikes

    ===== 1. SPEED =====
    (mm)            total_distance      : cumulative sacrum discplacements between the first and last frame of the GAIT CYCLE (mm)
    (mm/s)          speed               : total_distance / total_time (mm/s)

    ===== 2. CADENCE =====
    (N)             num_of_steps        : total number of steps (left + right) - 1 excluding the first step
    (N/min)         cadence             : num_of_steps / total_time * 60 (steps/min)

    ===== 3. STEP LENGTHS & WIDTHS =====
    (list)          X_step_lengths      : list of distances between the heel strike of one foot and the heel strike of the opposite foot
    (mm)            X_step_lengths[i]   : distance between the heel strike of one foot and the heel strike of the opposite foot in the direction of progression  
    (list)          X_step_widths       : list of distances between the heel strike of one foot and the heel strike of the opposite foot in the lateral direction
    (mm)            X_step_widths[i]    : distance between the heel strike of one foot and the heel strike of the opposite foot in the lateral direction
    (mm)            avg_step_length     : average of L_step_lengths & R_step_lengths
    (mm)            avg_step_width      : average of L_step_widths & R_step_widths

    The first X_step_length and X_step_width (represented by i=0) are ommitted. 

    ### FOR i between (1,len(X_heel_strikes)): ###
    X_step_lengths are calculated using the law of cosines.
    X_step_length[i] defines the projection of the "b" onto "c" and is calculated as:
    X_step_length[i] = (b^2 + c^2 - a^2)/(2*c)
    where:
        c = distance between the heel strikes of the current and previous ipsilateral foot
        a = distance between the heel strikes of the previous ipsilateral foot and the contralateral foot
        b = distance between the heel strikes of the current foot and the contralateral foot

    X_step_widths are calculated using the Pythagorean theorem.
    X_step_width[i] defines the vertical distance between the contralateral foot and "c"
    X_step_width[i] = sqrt(b^2 - X_step_length[i]^2)
    where:
        b = distance between the heel strikes of the current foot and the contralateral foot
        X_step_length[i] = the previously calculated step length

        

    
    ===== 4) STEP TIMES =====
    (list)          X_step_times        : list of times between the heel strike of one foot and the heel strike of the opposite foot
    (s)             X_step_times[i]     : time between the heel strike of one foot and the heel strike of the opposite foot
    (s)             avg_step_time       : average of L_step_times & R_step_times

    The first X_step_time (represented by i=0) is ommitted.

    ### FOR i between (1,len(X_heel_strikes)): ###
    X_step_time[i] is calculated as the time between the heel strike of one foot and the heel strike of the opposite foot.

    
    ===== 5) STRIDE LENGTHS & TIMES =====
    (list)          X_stride_lengths    : list of distances between the heel strikes of the same foot
    (mm)            X_stride_lengths[i] : distance between the heel strikes of the same foot
    (list)          X_stride_times      : list of times between the heel strikes of the same foot
    (s)             X_stride_times[i]   : time between the heel strikes of the same foot
    (mm)            avg_stride_length   : average of L_stride_lengths & R_stride_lengths
    (s)             avg_stride_time     : average of L_stride_times & R_stride_times

    The first X_stride_length and X_stride_time (represented by i=0) are ommitted.

    ### FOR i between (1,len(X_heel_strikes)): ###
    X_stride_length[i] is calculated as the distance between the heel strikes of the same foot
        distance between X_heel_arr[X_heel_strikes[i]] - X_heel_arr[X_heel_strikes[i-1]])

    X_stride_time[i] is calculated as the time between the heel strikes of the same foot
        time between (X_heel_strikes[i] - X_heel_strikes[i-1])/frame_rate

    ===== 6) STANCE & SWING TIMES =====
    (s)             X_swing_time        : total time the foot is in the air (between toe-off and heel strike)
    (s)             X_stance_time       : total time the foot is on the ground

    X_swing_time is calculated as the sum of the time between each X_toe_off and the subsequent X_heel_strike
    X_stance_time is calculated as the rest of total_time_X deducted by X_swing_time

    ===== 7) SINGLE & DOUBLE SUPPORT TIMES =====
    (s)             single_support_time : total time one foot is in contact with the ground
    (s)             double_support_time : total time both feet are in contact with the ground

    single_support_time is calculated as the sum of L_swing_time and R_swing_time
    double_support_time is calculated as the rest of total_time deducted by single_support_time

    '''
    speed                                                   = get_speed(sacrum_arr[starting_IC-1:ending_IC], total_time) # Accounting for 0-based indexing
    cadence                                                 = get_cadence(LHS, RHS, total_time)
    avg_step_length, L_step_lengths, R_step_lengths, avg_step_width, L_step_widths, R_step_widths = get_step_length_width(LHEE_arr, RHEE_arr, LHS-1, RHS-1)
    avg_step_time, L_step_times, R_step_times               = get_step_time(LHS, RHS, frame_rate)
    avg_stride_length, L_stride_lengths, R_stride_lengths   = get_stride_length(LHEE_arr, RHEE_arr, LHS-1, RHS-1)
    avg_stride_time, L_stride_times, R_stride_times         = get_stride_time(LHS, RHS, frame_rate)

    L_swing_times, R_swing_times                            = get_swing_time(LHS, LTO, RHS, RTO, frame_rate)
    L_stance_times, R_stance_times                          = L_stride_times - L_swing_times, R_stride_times - R_swing_times
    L_double_support_times_s, R_double_support_times_s, L_single_support_times_s, R_single_support_times_s = get_support_time(L_stride_times, R_stride_times, L_swing_times, R_swing_times, first_stepping_foot)


    gait_params = {
        "speed": speed,
        "cadence": cadence,
        "step_widths": {
            "left": L_step_widths,
            "right": R_step_widths,
            "average": avg_step_width
        },
        "step_lengths": {
            "left": L_step_lengths,
            "right": R_step_lengths,
            "average": avg_step_length
        },
        "step_times": {
            "left": L_step_times,
            "right": R_step_times,
            "average": avg_step_time
        },
        "stride_lengths": {
            "left": L_stride_lengths,
            "right": R_stride_lengths,
            "average": avg_stride_length
        },
        "stride_times": {
            "left": L_stride_times,
            "right": R_stride_times,
            "average": avg_stride_time
        },
        "stance_time": {
            "left": L_stance_times,
            "right": R_stance_times,
        },
        "swing_time": {
            "left": L_swing_times,
            "right": R_swing_times,
        },
        "single_support_time": {
            "left": L_single_support_times_s,
            "right": R_single_support_times_s,
        },
        "double_support_time": {
            "left": L_double_support_times_s,
            "right": R_double_support_times_s,
        }
    }
    # Save the Computed Gait Parameters to NPZ
    SAVE_DIR = PROJECT_ROOT / "data" / "GaitParams" / input_type
    if input_type == "vicon": SAVE_DIR = SAVE_DIR / session_name
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_PATH = SAVE_DIR / f"{trial_name}_GaitParams.npz"
    np.savez(SAVE_PATH, **gait_params)

    # Save the Computed Gait Parameters to CSV
    columns = {
        "speed": [speed],
        "cadence": [cadence],

        "step_width_left": L_step_widths,
        "step_width_right": R_step_widths,
        "step_width_average": [avg_step_width],

        "step_length_left": L_step_lengths,
        "step_length_right": R_step_lengths,
        "step_length_average": [avg_step_length],

        "step_time_left": L_step_times,
        "step_time_right": R_step_times,
        "step_time_average": [avg_step_time],

        "stride_length_left": L_stride_lengths,
        "stride_length_right": R_stride_lengths,
        "stride_length_average": [avg_stride_length],
        "stride_time_left": L_stride_times,
        "stride_time_right": R_stride_times,
        "stride_time_average": [avg_stride_time],

        "stance_time_left": L_stance_times,
        "stance_time_right": R_stance_times,
        "swing_time_left": L_swing_times,
        "swing_time_right": R_swing_times,
        "single_support_time_left": L_single_support_times_s,
        "single_support_time_right": R_single_support_times_s,
        "double_support_time_left": L_double_support_times_s,
        "double_support_time_right": R_double_support_times_s
    }

    SAVE_PATH = SAVE_DIR / f"{trial_name}_GaitParams.csv"
    with open(SAVE_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns.keys())
        for row in zip_longest(*columns.values(), fillvalue=""):
            writer.writerow(row)

    # Visualize the acquired gait parameters
    # plot_gait_summary(config)
    # plot_gait_cycle_phases(config)

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