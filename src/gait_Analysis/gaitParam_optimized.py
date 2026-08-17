import os, yaml
import numpy as np
import matplotlib.pyplot as plt
from Visualize_Param import plot_gait_summary, plot_gait_cycle_phases

def get_speed(sacrum_arr, total_time):
    # Calculate the total cumulative sacrum displacements as path length
    total_distance = np.sum(np.linalg.norm(np.diff(sacrum_arr, axis=0), axis=1))
    speed = total_distance / total_time
    return speed

def get_cadence(L_heel_strikes, R_heel_strikes, total_time):
    num_of_steps = len(L_heel_strikes) + len(R_heel_strikes) - 1  # NOT INCLUDING THE FIRST STEP
    cadence = num_of_steps / total_time * 60.0  # steps per minute NOT INCLUDING THE FIRST STEP
    return cadence

def get_step_length_width(L_heel_arr, R_heel_arr, LHS, RHS):
    L_step_lengths, R_step_lengths = [], []
    L_step_widths, R_step_widths = [], []
    total_step_length, total_step_width = 0, 0
    l_next, r_next = 1, 1
    L_step_lengths.append(np.nan)  # First step length is undefined
    R_step_lengths.append(np.nan)  # First step length is undefined
    L_step_widths.append(np.nan)  # First step width is undefined
    R_step_widths.append(np.nan)  # First step width is undefined
    
    # Starting our step lengths&widths analysis from LHS[1] and RHS[1]
    if(LHS[0] < RHS[0]): next_foot = "left"
    else : next_foot = "right"

    while (l_next < len(LHS) or r_next < len(RHS)):
        if next_foot == "right": # Do right foot step
            c = np.linalg.norm(R_heel_arr[RHS[r_next]] - R_heel_arr[RHS[r_next-1]])
            a = np.linalg.norm(L_heel_arr[LHS[l_next-1]] - R_heel_arr[RHS[r_next-1]])
            b = np.linalg.norm(R_heel_arr[RHS[r_next]] - L_heel_arr[LHS[l_next-1]])
            step_length = (b**2 + c**2 - a**2)/(2*c)
            R_step_lengths.append(step_length)
            total_step_length += step_length

            step_width = np.sqrt(b**2 - step_length**2)
            R_step_widths.append(step_width)
            total_step_width += step_width

            r_next += 1
            next_foot = "left"
        elif (next_foot == "left"): # Do left foot step
            c = np.linalg.norm(L_heel_arr[LHS[l_next]] - L_heel_arr[LHS[l_next-1]])
            a = np.linalg.norm(R_heel_arr[RHS[r_next-1]] - L_heel_arr[LHS[l_next-1]])
            b = np.linalg.norm(L_heel_arr[LHS[l_next]] - R_heel_arr[RHS[r_next-1]])
            step_length = (b**2 + c**2 - a**2)/(2*c)
            L_step_lengths.append(step_length)
            total_step_length += step_length

            step_width = np.sqrt(b**2 - step_length**2)
            L_step_widths.append(step_width)
            total_step_width += step_width

            l_next += 1
            next_foot = "right"

    avg_step_length = total_step_length / (len(LHS)-1 + len(RHS)-1)  # NOT INCLUDING THE FIRST STEP OF EACH FOOT
    avg_step_width = total_step_width / (len(LHS)-1 + len(RHS)-1)  # NOT INCLUDING THE FIRST STEP OF EACH FOOT

    return avg_step_length, L_step_lengths, R_step_lengths, avg_step_width, L_step_widths, R_step_widths

def get_step_time(LHS, RHS, frame_rate=100):
    L_step_times, R_step_times = [], []
    total_step_time = 0
    L_step_times.append(np.nan)  # First step time is undefined
    R_step_times.append(np.nan)  # First step time is undefined
    l_next, r_next = 0,0
    if(LHS[0] < RHS[0]): 
        next_foot = "left"
        l_next += 1
    else : 
        next_foot = "right"
        r_next += 1

    while (l_next < len(LHS) and r_next < len(RHS)):
        if next_foot == "right":
            step_time = (RHS[r_next] - LHS[l_next])/frame_rate  # Convert to seconds
            R_step_times.append(step_time)
            total_step_time += step_time
            l_next += 1
            next_foot = "left"
            
        elif (next_foot == "left"):
            step_time = (LHS[l_next] - RHS[r_next])/frame_rate  # Convert to seconds
            L_step_times.append(step_time)
            total_step_time += step_time
            r_next += 1
            next_foot = "right"

    avg_step_time = total_step_time / (len(LHS)-1 + len(RHS)-1)  # NOT INCLUDING THE FIRST STEP
    return avg_step_time, L_step_times, R_step_times

def get_stride_length(L_heel_arr, R_heel_arr, L_heel_strikes, R_heel_strikes):
    L_stride_lengths, R_stride_lengths = [], []
    total_stride_length = 0
    L_stride_lengths.append(np.nan)  # First stride length is undefined
    R_stride_lengths.append(np.nan)  # First stride length is undefined

    for i in range (1, len(L_heel_strikes)):
        stride_length = np.linalg.norm(L_heel_arr[L_heel_strikes[i]] - L_heel_arr[L_heel_strikes[i-1]])
        L_stride_lengths.append(stride_length)
        total_stride_length += stride_length
    for i in range (1, len(R_heel_strikes)):
        stride_length = np.linalg.norm(R_heel_arr[R_heel_strikes[i]] - R_heel_arr[R_heel_strikes[i-1]])
        R_stride_lengths.append(stride_length)
        total_stride_length += stride_length
    avg_stride_length = total_stride_length / (len(L_heel_strikes)-1 + len(R_heel_strikes)-1)  # NOT INCLUDING THE FIRST STRIDE OF EACH FOOT
    return avg_stride_length, L_stride_lengths, R_stride_lengths

def get_stride_time(LHS, RHS, frame_rate=100):
    L_stride_times, R_stride_times = [], []
    total_stride_time = 0
    L_stride_times.append(np.nan)  # First stride time is undefined
    R_stride_times.append(np.nan)  # First stride time is undefined
    for i in range(1, len(LHS)):
        stride_time = (LHS[i] - LHS[i-1])/frame_rate  # Convert to seconds
        L_stride_times.append(stride_time)
        total_stride_time += stride_time
    for i in range(1, len(RHS)):
        stride_time = (RHS[i] - RHS[i-1])/frame_rate  # Convert to seconds
        R_stride_times.append(stride_time)
        total_stride_time += stride_time
    avg_stride_time = total_stride_time / (len(LHS)-1 + len(RHS)-1)  # NOT INCLUDING THE FIRST STRIDE OF EACH FOOT

    return avg_stride_time, L_stride_times, R_stride_times

# Consider how you properly assess the starting IC and ending IC of the gait cycle to calculate stance and swing times.
def get_swing_time(L_heel_strikes, L_toe_offs, R_heel_strikes, R_toe_offs, frame_rate=100):
    L_swing_time = 0
    R_swing_time = 0
    for i in range (1,len(L_heel_strikes)):
        L_swing_time += (L_heel_strikes[i] - L_toe_offs[i-1]) / frame_rate  # Convert to seconds
    for i in range (1,len(R_heel_strikes)):
        R_swing_time += (R_heel_strikes[i] - R_toe_offs[i-1]) / frame_rate  # Convert to seconds
    return L_swing_time, R_swing_time

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
    if(input_type == "vicon"):
        data = np.load(f"data\\GaitEvents\\{input_type}\\{session_name}\\{trial_name}_GaitEvents.npz")
    else:
        data = np.load(f"data\\GaitEvents\\{input_type}\\{trial_name}_GaitEvents.npz")
    L_heel_arr, L_toe_arr, R_heel_arr, R_toe_arr = data['LHarr'], data['LTarr'], data['RHarr'], data['RTarr']
    L_heel_strikes, L_toe_offs, R_heel_strikes, R_toe_offs = data['LHS'], data['LTO'], data['RHS'], data['RTO']
    sacrum_arr = data['sacrum_arr']
    frame_rate = data['frame_rate']
    start_frame = data['start_frame']
    data.close()  # Close the file after loading the data

    # Total time between first step and last (with 100Hz frame rate)
    starting_IC = min(L_heel_strikes[0], R_heel_strikes[0])
    ending_IC = max(L_heel_strikes[-1], R_heel_strikes[-1])
    total_time = (ending_IC - starting_IC) / frame_rate
    total_time_left = (L_heel_strikes[-1] - L_heel_strikes[0]) / frame_rate
    total_time_right = (R_heel_strikes[-1] - R_heel_strikes[0]) / frame_rate

    # Compute each gait parameter
    '''
    ======================HOW GAIT PARAMETERS ARE CALCULATED==========================
    (1-based)       starting_IC         : frame of the first heel strike
    (1-based)       ending_IC           : frame of the last heel strike
    (s)             total_time          : time between the first and last heel strikes
    (s)             total_time_left     : time between the first and last left heel strikes
    (s)             total_time_right    : time between the first and last right heel strikes

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
    cadence                                                 = get_cadence(L_heel_strikes, R_heel_strikes, total_time)
    avg_step_length, L_step_lengths, R_step_lengths, avg_step_width, L_step_widths, R_step_widths = get_step_length_width(L_heel_arr, R_heel_arr, L_heel_strikes-1, R_heel_strikes-1)
    avg_step_time, L_step_times, R_step_times               = get_step_time(L_heel_strikes, R_heel_strikes, frame_rate)
    avg_stride_length, L_stride_lengths, R_stride_lengths   = get_stride_length(L_heel_arr, R_heel_arr, L_heel_strikes-1, R_heel_strikes-1)
    avg_stride_time, L_stride_times, R_stride_times         = get_stride_time(L_heel_strikes, R_heel_strikes, frame_rate)

    L_swing_time, R_swing_time                              = get_swing_time(L_heel_strikes, L_toe_offs, R_heel_strikes, R_toe_offs, frame_rate)
    L_stance_time, R_stance_time                            = total_time_left - L_swing_time, total_time_right - R_swing_time
    single_support_time                                     = L_swing_time + R_swing_time
    double_support_time                                     = total_time - single_support_time


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
            "left": {
                "s" : L_stance_time,
                "%" : (L_stance_time / total_time_left) * 100
            },
            "right": {
                "s" : R_stance_time,
                "%" : (R_stance_time / total_time_right) * 100
            }
        },
        "swing_time": {
            "left": {
                "s" : L_swing_time,
                "%" : (L_swing_time / total_time_left) * 100
            },
            "right": {
                "s" : R_swing_time,
                "%" : (R_swing_time / total_time_right) * 100
            }
        },
        "single_support_time": {
            "s" : single_support_time,
            "%" : (single_support_time / total_time) * 100
        },
        "double_support_time": {
            "s" : double_support_time,
            "%" : (double_support_time / total_time) * 100
        }
    }
    if(input_type == "vicon"):
        save_dir = rf"data\GaitParams\{input_type}\{session_name}"
    else:
        save_dir = rf"data\GaitParams\{input_type}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{trial_name}_GaitParams.npz")
    np.savez(save_path, **gait_params)

    plot_gait_summary(config)
    plot_gait_cycle_phases(config)

class InvalidConfigError(Exception):
    def __init__(self, error_msgs):
        self.error_msgs = error_msgs
        super().__init__("\n".join(error_msgs))

with open("src/gait_analysis/config/input_config.yaml", "r") as f:
    config = yaml.safe_load(f)
try:
    compute_gait_params(config)
except InvalidConfigError as e:
    for msg in e.error_msgs:
        print(msg)