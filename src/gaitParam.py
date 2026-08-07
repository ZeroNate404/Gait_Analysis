import os
from matplotlib.pyplot import step
import numpy as np


def get_speed(sacrum_arr, total_time):
    # Calculate the total distance covered by both legs
    total_distance = np.sum(np.linalg.norm(np.diff(sacrum_arr, axis=0), axis=1))
    speed = total_distance / total_time
    return speed

def get_cadence(L_heel_strikes, R_heel_strikes, total_time):
    cadence = (len(L_heel_strikes) + len(R_heel_strikes) - 1) / total_time * 60.0  # steps per minute NOT INCLUDING THE FIRST STEP
    return cadence

def get_step_length_width(L_heel_arr, R_heel_arr, LHS, RHS):
    L_step_lengths, R_step_lengths = [], []
    L_step_widths, R_step_widths = [], []
    total_step_length, total_step_width = 0, 0
    if (LHS[0] < RHS[0]): next_foot = "right"
    else : next_foot = "left"

    l_next, r_next = 1, 1
    # Calculate first step length
    if(next_foot == "right"):
        c = L_heel_arr[LHS[l_next]] - L_heel_arr[LHS[l_next-1]]
        a = R_heel_arr[RHS[r_next-1]] - L_heel_arr[LHS[r_next-1]]
        b = L_heel_arr[LHS[l_next]] - R_heel_arr[RHS[r_next-1]]

        step_length = np.linalg.norm((np.dot(a,c)/np.linalg.norm(c)**2)*c)
        R_step_lengths.append(step_length)
        total_step_length += step_length

        step_width = np.sqrt(np.linalg.norm(b)**2 - step_length**2)
        R_step_widths.append(step_width)
        total_step_width += step_width

        r_next += 1
        next_foot = "left"
    else:
        c = R_heel_arr[RHS[r_next]] - R_heel_arr[RHS[r_next-1]]
        a = L_heel_arr[LHS[l_next-1]] - R_heel_arr[RHS[r_next-1]]
        b = R_heel_arr[RHS[r_next]] - L_heel_arr[LHS[l_next-1]]
        step_length = np.linalg.norm((np.dot(a,c)/np.linalg.norm(c)**2)*c)
        L_step_lengths.append(step_length)
        total_step_length += step_length

        step_width = np.sqrt(np.linalg.norm(b)**2 - step_length**2)
        L_step_widths.append(step_width)
        total_step_width += step_width

        l_next += 1
        next_foot = "right"

    while (l_next < len(LHS) or r_next < len(RHS)):
        if next_foot == "right":
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
        elif (next_foot == "left"):
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

    avg_step_length = total_step_length / (len(LHS) + len(RHS)-1)  # NOT INCLUDING THE FIRST STEP
    avg_step_width = total_step_width / (len(LHS) + len(RHS)-1)  # NOT INCLUDING THE FIRST STEP

    return avg_step_length, L_step_lengths, R_step_lengths, avg_step_width, L_step_widths, R_step_widths

def get_step_time(LHS, RHS, frame_rate=100):
    L_step_times, R_step_times = [], []
    total_step_time = 0
    if(LHS[0] < RHS[0]): next_foot = "right"
    else : next_foot = "left"

    l_idx, r_idx = 0, 0
    while (l_idx < len(LHS) or r_idx < len(RHS)):
        if next_foot == "right":
            step_time = (RHS[r_idx] - LHS[l_idx])/frame_rate  # Convert to seconds
            R_step_times.append(step_time)
            total_step_time += step_time
            l_idx += 1
            next_foot = "left"
            
        elif (next_foot == "left"):
            step_time = (LHS[l_idx] - RHS[r_idx])/frame_rate  # Convert to seconds
            L_step_times.append(step_time)
            total_step_time += step_time
            r_idx += 1
            next_foot = "right"

    avg_step_time = total_step_time / (len(LHS) + len(RHS)-1)  # NOT INCLUDING THE FIRST STEP
    return avg_step_time, L_step_times, R_step_times

def get_stride_length(L_heel_arr, R_heel_arr, L_heel_strikes, R_heel_strikes):
    L_stride_lengths, R_stride_lengths = [], []
    total_stride_length = 0
    for i in range (1, len(L_heel_strikes)):
        stride_length = np.linalg.norm(L_heel_arr[L_heel_strikes[i]] - L_heel_arr[L_heel_strikes[i-1]])
        L_stride_lengths.append(stride_length)
        total_stride_length += stride_length
    for i in range (1, len(R_heel_strikes)):
        stride_length = np.linalg.norm(R_heel_arr[R_heel_strikes[i]] - R_heel_arr[R_heel_strikes[i-1]])
        R_stride_lengths.append(stride_length)
        total_stride_length += stride_length
    avg_stride_length = total_stride_length / (len(L_heel_strikes) + len(R_heel_strikes)-2)  # NOT INCLUDING THE FIRST STRIDE OF EACH FOOT
    return avg_stride_length, L_stride_lengths, R_stride_lengths

def get_stride_time(LHS, RHS, frame_rate=100):
    L_stride_times, R_stride_times = [], []
    total_stride_time = 0
    for i in range(1, len(LHS)):
        stride_time = (LHS[i] - LHS[i-1])/frame_rate  # Convert to seconds
        L_stride_times.append(stride_time)
        total_stride_time += stride_time
    for i in range(1, len(RHS)):
        stride_time = (RHS[i] - RHS[i-1])/frame_rate  # Convert to seconds
        R_stride_times.append(stride_time)
        total_stride_time += stride_time
    avg_stride_time = total_stride_time / (len(LHS) + len(RHS)-2)  # NOT INCLUDING THE FIRST STRIDE OF EACH FOOT

    return avg_stride_time, L_stride_times, R_stride_times

def get_stance_time(L_heel_strikes, L_toe_offs, R_heel_strikes, R_toe_offs, frame_rate=100):
    L_stance_time = 0
    R_stance_time = 0
    for i in range (len(L_toe_offs)):
        L_stance_time += (L_toe_offs[i] - L_heel_strikes[i]) / frame_rate  # Convert to seconds 
    for i in range (len(R_toe_offs)):
        R_stance_time += (R_toe_offs[i] - R_heel_strikes[i]) / frame_rate  # Convert to seconds
    return L_stance_time, R_stance_time

def get_swing_time(L_heel_strikes, L_toe_offs, R_heel_strikes, R_toe_offs, frame_rate=100):
    L_swing_time = 0
    R_swing_time = 0
    for i in range (1,len(L_heel_strikes)):
        L_swing_time += (L_heel_strikes[i] - L_toe_offs[i-1]) / frame_rate  # Convert to seconds
    for i in range (1,len(R_heel_strikes)):
        R_swing_time += (R_heel_strikes[i] - R_toe_offs[i-1]) / frame_rate  # Convert to seconds
    return L_swing_time, R_swing_time

def compute_gait_params(trial_name, session_name):
    data = np.load(f"D:\\python_scripts\\Gait_Analysis\\data\\GaitEvents\\{session_name}\\{trial_name}_GaitEvents.npz")
    L_heel_arr, L_toe_arr, R_heel_arr, R_toe_arr = data['LHarr'], data['LTarr'], data['RHarr'], data['RTarr']
    L_heel_vX, L_toe_vX, R_heel_vX, R_toe_vX = data['LHvX'], data['LTvX'], data['RHvX'], data['RTvX']
    L_heel_strikes, L_toe_offs, R_heel_strikes, R_toe_offs = data['LHS'], data['LTO'], data['RHS'], data['RTO']
    sacrum_arr = data['sacrum_arr']
    frame_rate = data['frame_rate']
    start_frame = data['start_frame']

    # Total time between first step and last (with 100Hz frame rate)
    total_time = ((max(L_heel_strikes[-1], R_heel_strikes[-1]) - min(L_heel_strikes[0], R_heel_strikes[0])) + 1) / frame_rate
    total_time_left = ((L_heel_strikes[-1] - L_heel_strikes[0]) + 1) / frame_rate
    total_time_right = ((R_heel_strikes[-1] - R_heel_strikes[0]) + 1) / frame_rate

    # Compute each gait parameter
    speed                                                   = get_speed(sacrum_arr, total_time)
    cadence                                                 = get_cadence(L_heel_strikes, R_heel_strikes, total_time)
    avg_step_length, L_step_lengths, R_step_lengths, avg_step_width, L_step_widths, R_step_widths = get_step_length_width(L_heel_arr-start_frame, R_heel_arr-start_frame, L_heel_strikes, R_heel_strikes)
    avg_step_time, L_step_times, R_step_times               = get_step_time(L_heel_strikes, R_heel_strikes, frame_rate)
    avg_stride_length, L_stride_lengths, R_stride_lengths   = get_stride_length(L_heel_arr-start_frame, R_heel_arr-start_frame, L_heel_strikes, R_heel_strikes)
    avg_stride_time, L_stride_times, R_stride_times         = get_stride_time(L_heel_strikes, R_heel_strikes, frame_rate)
    L_stance_time, R_stance_time                            = get_stance_time(L_heel_strikes, L_toe_offs, R_heel_strikes, R_toe_offs, frame_rate)
    L_swing_time, R_swing_time                              = get_swing_time(L_heel_strikes, L_toe_offs, R_heel_strikes, R_toe_offs, frame_rate)
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
    save_dir = rf"D:\python_scripts\Gait_Analysis\data\GaitParams\{session_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{trial_name}_GaitParams.npz")
    np.savez(save_path, **gait_params)