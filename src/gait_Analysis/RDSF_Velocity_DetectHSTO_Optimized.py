import sys, datetime, os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import gaitParam_optimized as gp
from scipy.signal import butter, filtfilt
from viconnexusapi import ViconNexus

'''
===================INPUT===================================
--INFO---------------------DataType-------------Default
Subject_name             --String               Default: first subject detected in the session
starting_frame           --Integer [1,N]        Default: 1
ending_frame             --Integer [1,N]        Default: Last frame of session

where N is the total frames of the motion capture session

===================OUTPUT==================================
Writes an EVENT onto the vicon Nexus session timeline
and prints the frame number
for each HEEL STRIKE and TOE OFF detected for both left and right foot.
'''

def get_heel_toe_arr(vicon, subject):
    # 2. Get Marker Trajectories
    L_heel_X, L_heel_Y, _, exist = vicon.GetTrajectory(subject, "LHEE")
    L_toe_X, L_toe_Y, _, _  = vicon.GetTrajectory(subject, "LTOE")
    R_heel_X, R_heel_Y, _, _ = vicon.GetTrajectory(subject, "RHEE")
    R_toe_X, R_toe_Y, _, _  = vicon.GetTrajectory(subject, "RTOE")
    L_PSI_X, L_PSI_Y, _, _ = vicon.GetTrajectory(subject, "LPSI")
    R_PSI_X, R_PSI_Y, _, _ = vicon.GetTrajectory(subject, "RPSI")
    # Stack 1D X and Y arrays into a 2D (N, 2) array
    L_heel_arr = np.column_stack((L_heel_X, L_heel_Y))
    L_toe_arr  = np.column_stack((L_toe_X, L_toe_Y))
    R_heel_arr = np.column_stack((R_heel_X, R_heel_Y))
    R_toe_arr  = np.column_stack((R_toe_X, R_toe_Y))
    L_PSI_arr = np.column_stack((L_PSI_X, L_PSI_Y))
    R_PSI_arr = np.column_stack((R_PSI_X, R_PSI_Y))
    # Calculate Sacrum as the midpoint: S = (LPSI + RPSI) / 2
    sacrum_arr = (L_PSI_arr + R_PSI_arr) / 2.0

    return L_heel_arr, L_toe_arr, R_heel_arr, R_toe_arr, sacrum_arr

def compute_walk_dir(sacrum_arr, frame_rate):
    dt = 1.0 / frame_rate
    sacrum_vel = np.gradient(sacrum_arr, dt, axis=0)
    # Align avg sacrum velocity with the x-axis to define the walking direction
    sacrum_avg_vel = np.mean(sacrum_vel, axis=0) # [vx, vy] : determines the direction of rotation transformation to align this direction with the x-axis
    '''
    This method introduces small floating point errors
    # Normalize direction vectors to unit vectors (length = 1)
    sacrum_speed = np.linalg.norm(sacrum_avg_vel)
    if sacrum_speed > 1e-8: walk_dir = sacrum_avg_vel / sacrum_speed
    else:  walk_dir = np.array([0.0, 0.0])  # or handle however you prefer
    '''
    speed = np.linalg.norm(sacrum_avg_vel)
    if speed > 1e-8:
        walk_dir = sacrum_avg_vel / speed
        R = np.array([
            [ walk_dir[0],  walk_dir[1]],
            [-walk_dir[1],  walk_dir[0]]
        ])
    else:
        warnings.warn(
            f"Sacrum speed ({speed:.2e}) near zero — likely a treadmill trial. "
            "Defaulting walk_dir to lab X-axis; no rotation applied.",
            RuntimeWarning
        )
        walk_dir = np.array([1.0, 0.0])
        R = np.eye(2)

    return R, walk_dir

def align_with_walkdir(L_heel_arr, L_toe_arr, R_heel_arr, R_toe_arr, sacrum_arr, R):
    # Set Sacrum as Origin
    L_heel_arr -= sacrum_arr
    L_toe_arr -= sacrum_arr
    R_heel_arr -= sacrum_arr
    R_toe_arr -= sacrum_arr

    # Rotate coordinates by R to align with the walking direction
    L_heel_arr = L_heel_arr @ R.T
    L_toe_arr  = L_toe_arr @ R.T
    R_heel_arr = R_heel_arr @ R.T
    R_toe_arr  = R_toe_arr @ R.T

    return L_heel_arr, L_toe_arr, R_heel_arr, R_toe_arr

def compute_projected_velocity(L_heel_arr, L_toe_arr, R_heel_arr, R_toe_arr, frame_rate):
    # Obtain velocity vectors (mm/s)
    dt = 1.0 / frame_rate
    L_heel_vel = np.gradient(L_heel_arr, dt, axis=0)
    L_toe_vel = np.gradient(L_toe_arr, dt, axis=0)
    R_heel_vel = np.gradient(R_heel_arr, dt, axis=0)
    R_toe_vel = np.gradient(R_toe_arr, dt, axis=0)

    # Track vX (velocity in the walking direction) for each marker
    L_heel_vX = L_heel_vel[:, 0]
    L_toe_vX = L_toe_vel[:, 0]
    R_heel_vX = R_heel_vel[:, 0]
    R_toe_vX = R_toe_vel[:, 0]

    return L_heel_vX, L_toe_vX, R_heel_vX, R_toe_vX

def filter_data(L_heel_vX, L_toe_vX, R_heel_vX, R_toe_vX, frame_rate):
    # 4. Filter data (2nd order Butterworth, 6Hz cutoff)
    b, a = butter(2, 6.0 / (0.5 * frame_rate), btype='low')
    L_heel_filt = filtfilt(b, a, L_heel_vX)
    L_toe_filt  = filtfilt(b, a, L_toe_vX)
    R_heel_filt = filtfilt(b, a, R_heel_vX)
    R_toe_filt  = filtfilt(b, a, R_toe_vX)
    return L_heel_filt, L_toe_filt, R_heel_filt, R_toe_filt

def detect_events(L_heel_vX, L_toe_vX, R_heel_vX, R_toe_vX, start_frame=1):
    L_heel_sign = np.sign(L_heel_vX)
    L_heel_valid = np.where(L_heel_sign != 0)[0]
    L_heel_changes = np.where(np.diff(L_heel_sign[L_heel_valid]) == -2)[0]
    L_heel_strikes = L_heel_valid[L_heel_changes + 1]

    L_toe_sign = np.sign(L_toe_vX)
    L_toe_valid = np.where(L_toe_sign != 0)[0]
    L_toe_changes = np.where(np.diff(L_toe_sign[L_toe_valid]) == 2)[0]
    L_toe_offs = L_toe_valid[L_toe_changes + 1]

    R_heel_sign = np.sign(R_heel_vX)
    R_heel_valid = np.where(R_heel_sign != 0)[0]
    R_heel_changes = np.where(np.diff(R_heel_sign[R_heel_valid]) == -2)[0]
    R_heel_strikes = R_heel_valid[R_heel_changes + 1]

    R_toe_sign = np.sign(R_toe_vX)
    R_toe_valid = np.where(R_toe_sign != 0)[0]
    R_toe_changes = np.where(np.diff(R_toe_sign[R_toe_valid]) == 2)[0]
    R_toe_offs = R_toe_valid[R_toe_changes + 1]

    # Adjust for 0-based indexing and the start frame offset
    L_heel_strikes += start_frame  
    R_heel_strikes += start_frame
    L_toe_offs += start_frame
    R_toe_offs += start_frame

    return L_heel_strikes, L_toe_offs, R_heel_strikes, R_toe_offs

def visualize_matplotlib(L_heel_vX, L_toe_vX, R_heel_vX, R_toe_vX, L_heel_strikes, L_toe_offs, R_heel_strikes, R_toe_offs, start_frame, end_frame):
    # Matplotlib visualization of Velocities and detected events
    x_frames = np.arange(len(L_heel_vX)) + start_frame
    plt.plot(x_frames, L_heel_vX, color="blue")
    plt.plot(x_frames, L_toe_vX, color="green")
    plt.plot(x_frames, R_heel_vX, color="orange")
    plt.plot(x_frames, R_toe_vX, color="red")
    plt.plot(L_heel_strikes, L_heel_vX[L_heel_strikes-start_frame], "x", color="darkblue")
    plt.plot(R_heel_strikes, R_heel_vX[R_heel_strikes-start_frame], "x", color="darkred")
    plt.plot(x_frames, np.full_like(x_frames, 0), "--", color="black")
    plt.xlim(left=start_frame,right=end_frame)
    plt.show()

# Write events to Vicon Nexus
def nexus_write_events(vicon, subject, L_heel_strikes, L_toe_offs, R_heel_strikes, R_toe_offs):
    for frame in L_heel_strikes: 
        print(f"Left Heel Strike frame : {frame}")
        vicon.CreateAnEvent(subject, "Left", "Foot Strike", int(frame), 0.0)
    for frame in R_heel_strikes:
        print(f"Right Heel Strike frame : {frame}")
        vicon.CreateAnEvent(subject, "Right", "Foot Strike", int(frame), 0.0)
    for frame in L_toe_offs: 
        print(f"Left Toe Off frame : {frame}")
        vicon.CreateAnEvent(subject, "Left", "Foot Off", int(frame), 0.0)
    for frame in R_toe_offs:
        print(f"Right Toe Off frame : {frame}")
        vicon.CreateAnEvent(subject, "Right", "Foot Off", int(frame), 0.0)
    print(f"Successfully created {len(L_heel_strikes)} Left Heel Strikes and {len(R_heel_strikes)} Right Heel Strikes.")
    print(f"Successfully created {len(L_toe_offs)} Left Toe Offs and {len(R_toe_offs)} Right Toe Offs.")

def save_events_npz(trial_name, session_name, gait_events):
    save_dir = rf"D:\python_scripts\Gait_Analysis\data\GaitEvents\{session_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{trial_name}_GaitEvents.npz")
    np.savez(save_path, **gait_events)

def main():
    # 1. Connect to active Vicon Nexus session
    vicon = ViconNexus.ViconNexus()
    frame_rate = vicon.GetFrameRate()
    # Command-line arguments (INPUT)
    # subject = sys.argv[1] if len(sys.argv) > 1 else vicon.GetSubjectNames()[0]
    subject = vicon.GetSubjectNames()[0]  # Default to the first subject detected in the session
    start_frame = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    end_frame = int(sys.argv[3]) if len(sys.argv) > 3 else vicon.GetFrameCount()

    # 2. Get Marker Trajectories
    L_heel_arr, L_toe_arr, R_heel_arr, R_toe_arr, sacrum_arr = get_heel_toe_arr(vicon, subject)

    # 3. Compute Walking Direction 
    sacrum_arr_cropped = sacrum_arr[start_frame-1:end_frame] # Align 1-indexing to 0-indexing. Note numpy slicing is [start,end)
    R, walk_dir = compute_walk_dir(sacrum_arr_cropped, frame_rate)

    # 4. Transform Coordinates to align with walking direction
    L_heel_arr, L_toe_arr, R_heel_arr, R_toe_arr = align_with_walkdir(L_heel_arr, L_toe_arr, R_heel_arr, R_toe_arr, sacrum_arr, R)

    # 4. Determine velocity/walking direction
    L_heel_vX, L_toe_vX, R_heel_vX, R_toe_vX = compute_projected_velocity(L_heel_arr, L_toe_arr, R_heel_arr, R_toe_arr, frame_rate)

    # 5. Filter data (2nd order Butterworth, 6Hz cutoff)(Optional)
    # L_heel_vX, L_toe_vX, R_heel_vX, R_toe_vX = filter_data(L_heel_vX, L_toe_vX, R_heel_vX, R_toe_vX, frame_rate)

    # 6. Detect Heel Strike (+ to - velocity) and Toe Off (- to + velocity) events
    L_heel_strikes, L_toe_offs, R_heel_strikes, R_toe_offs = detect_events(L_heel_vX[start_frame-1:end_frame], # Crop to specified frame range
                                                                           L_toe_vX[start_frame-1:end_frame], 
                                                                           R_heel_vX[start_frame-1:end_frame], 
                                                                           R_toe_vX[start_frame-1:end_frame], 
                                                                           start_frame)

    # 7. Visualize Events to Matplotlib
    visualize_matplotlib(L_heel_vX[start_frame-1:end_frame], 
                         L_toe_vX[start_frame-1:end_frame], 
                         R_heel_vX[start_frame-1:end_frame], 
                         R_toe_vX[start_frame-1:end_frame],
                         L_heel_strikes, L_toe_offs, R_heel_strikes, R_toe_offs, start_frame, end_frame)
    # 8. Write Events to Vicon Nexus
    nexus_write_events(vicon, subject, L_heel_strikes, L_toe_offs, R_heel_strikes, R_toe_offs)

    # 9. Save Events to NPZ file
    trial_path, trial_name = vicon.GetTrialName()
    session_name = os.path.basename(os.path.normpath(trial_path))
    subject_name = vicon.GetSubjectNames()[0]
    gait_events = {
        "Date" : datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        "Run_name" : trial_name,
        "Session_name" : session_name,
        "Subject_name" : subject_name,
        "frame_rate": frame_rate,
        "start_frame": start_frame,
        "end_frame": end_frame,

        "walk_dir" : walk_dir,
        "sacrum_arr": sacrum_arr,

        "LHarr": L_heel_arr, # frames [1,N]
        "LTarr": L_toe_arr,
        "RHarr": R_heel_arr,
        "RTarr": R_toe_arr,

        "LHvX": L_heel_vX, # frames [1,N]
        "LTvX": L_toe_vX,
        "RHvX": R_heel_vX,
        "RTvX": R_toe_vX,  
        
        "LHS": L_heel_strikes,
        "RHS": R_heel_strikes,
        "LTO": L_toe_offs,
        "RTO": R_toe_offs,
    }
    save_events_npz(trial_name, session_name, gait_events)

    # 9. Compute Gait Parameters (Optional)
    # gp.compute_gait_params(trial_name, session_name)


main()