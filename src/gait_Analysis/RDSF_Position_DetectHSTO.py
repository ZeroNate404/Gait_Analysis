import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import datetime
from viconnexusapi import ViconNexus

'''
===================INPUT=======================
Subject_name             --String               Default: first subject detected in the session
starting_frame           --Integer [1,N]        Default: 1
ending_frame             --Integer [1,N]        Default: Last frame of session

where N is the total frames of the motion capture session

===================OUTPUT======================
Writes an EVENT onto the vicon Nexus session timeline
and prints the frame number
for each HEEL STRIKE and TOE OFF detected for both left and right foot.
'''

# 1. Connect to active Vicon Nexus session
vicon = ViconNexus.ViconNexus()
subject = "Oli 2"
# subject = sys.argv[1] if len(sys.argv) > 1 else vicon.GetSubjectNames()[0]

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

# Cut the motiont tracking to a specific range of frames
start_frame = int(sys.argv[2]) if len(sys.argv) > 2 else 1
end_frame = int(sys.argv[3]) if len(sys.argv) > 3 else len(L_heel_arr)
L_heel_arr = L_heel_arr[start_frame-1:end_frame-1] # to align with 0-indexing
L_toe_arr  = L_toe_arr[start_frame-1:end_frame-1]
R_heel_arr = R_heel_arr[start_frame-1:end_frame-1]
R_toe_arr  = R_toe_arr[start_frame-1:end_frame-1]
L_PSI_arr = L_PSI_arr[start_frame-1:end_frame-1]
R_PSI_arr = R_PSI_arr[start_frame-1:end_frame-1]

# Calculate Sacrum as the midpoint: S = (LPSI + RPSI) / 2
sacrum_arr = (L_PSI_arr + R_PSI_arr) / 2.0

# 3. Compute Axis-Independent Progression
# Calculate relative position vectors in the horizontal plane
L_rel_heel_vec = L_heel_arr - sacrum_arr
L_rel_toe_vec  = L_toe_arr - sacrum_arr
R_rel_heel_vec = R_heel_arr - sacrum_arr
R_rel_toe_vec  = R_toe_arr - sacrum_arr

# Calculate the Instantaneous Walking Direction Vector from Sacrum movement
sacrum_vel = np.gradient(sacrum_arr, axis=0)
# Normalize direction vectors to unit vectors (length = 1)
sacrum_speed = np.linalg.norm(sacrum_vel, axis=1, keepdims=True)
# Avoid division by zero when subject is standing still
sacrum_speed[sacrum_speed == 0] = 1e-6  
walk_dir = sacrum_vel / sacrum_speed  # Unit vector along line of progression (N, 2)

# Project 2D relative foot vectors onto the instantaneous walking direction
# Dot product frame-by-frame: (X_rel * Dir_x) + (Y_rel * Dir_y)
l_heel_projected = np.sum(L_rel_heel_vec * walk_dir, axis=1)
l_toe_projected  = np.sum(L_rel_toe_vec * walk_dir, axis=1)
r_heel_projected = np.sum(R_rel_heel_vec * walk_dir, axis=1)
r_toe_projected  = np.sum(R_rel_toe_vec * walk_dir, axis=1)

# 4. Filter data (2nd order Butterworth, 6Hz cutoff)
frame_rate = vicon.GetFrameRate()
b, a = butter(2, 6.0 / (0.5 * frame_rate), btype='low')
L_heel_filt = filtfilt(b, a, l_heel_projected)
L_toe_filt  = filtfilt(b, a, l_toe_projected)
R_heel_filt = filtfilt(b, a, r_heel_projected)
R_toe_filt  = filtfilt(b, a, r_toe_projected)

# 5. Find local maxima for Heel Strike, local minima for Toe Off
# Calulate graph mean line
L_heel_mean = np.mean(L_heel_filt)
R_heel_mean = np.mean(R_heel_filt)
L_toe_mean = np.mean(L_toe_filt)
R_toe_mean = np.mean(R_toe_filt)
# Adjust 'distance' based on min frames between steps (e.g., 0.5 sec * frame_rate)
min_dist = int(0.5 * frame_rate)
L_heel_strikes, _ = find_peaks(L_heel_filt, distance=min_dist, height=L_heel_mean)
R_heel_strikes, _ = find_peaks(R_heel_filt, distance=min_dist, height=R_heel_mean)
L_toe_offs, _     = find_peaks(-L_toe_filt, distance=min_dist, height=L_toe_mean)
R_toe_offs, _     = find_peaks(-R_toe_filt, distance=min_dist, height=R_toe_mean)

# Adjust for 0-based indexing and the start frame offset
L_heel_strikes += start_frame  
R_heel_strikes += start_frame
L_toe_offs += start_frame
R_toe_offs += start_frame

# Matplotlib visualization of Relative Distance trajectories and detected events
x_frames = np.arange(len(l_heel_projected)) + start_frame
plt.plot(x_frames, l_heel_projected, color="blue")
plt.plot(x_frames, r_heel_projected, color="orange")
plt.plot(L_heel_strikes, l_heel_projected[L_heel_strikes-start_frame], "x", color="darkblue")
plt.plot(R_heel_strikes, r_heel_projected[R_heel_strikes-start_frame], "x", color="red")
plt.plot(x_frames, np.full_like(x_frames, L_heel_mean), "--", color="black")
plt.plot(x_frames, np.full_like(x_frames, R_heel_mean), "--", color="black")
plt.xlim(left=start_frame,right=end_frame)
plt.show()

# 6. Write events back into Vicon Nexus session
with open("D:\\python scripts\\GaitEvents.txt", "a") as file:
    file.write(f"Test run time {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    for frame in L_heel_strikes:
        file.write(f"Left Heel Strike frame : {frame}  | distance : {L_heel_filt[frame-start_frame]}\n")
        print("==============================================")
        print(f"distance before:")
        for i in range(3): print(f"{(-3+i)}: {L_heel_filt[frame-start_frame + (-3+i)]}")
        print(f"Left Heel Strike frame : {frame}  | distance : {L_heel_filt[frame-start_frame]}")
        print(f"distance after :")
        for i in range(3): print(f"{(1+i)}: {L_heel_filt[frame-start_frame + (1+i)]}")- 
        vicon.CreateAnEvent(subject, "Left", "Heel Strike", int(frame), 0.0)
        print()
    # for frame in R_heel_strikes:
    #     print(f"Right Heel Strike frame : {frame}")
    #     vicon.CreateAnEvent(subject, "Right", "Heel Strike", int(frame), 0.0)
    for frame in L_toe_offs:
        file.write(f"Left Toe Off frame : {frame}  | distance : {L_toe_filt[frame-start_frame]}\n")
        print("==============================================")
        print(f"distance before:")
        for i in range(3): print(f"{(-3+i)}: {L_toe_filt[frame-start_frame + (-3+i)]}")
        print(f"Left Toe Off frame : {frame}  | distance : {L_toe_filt[frame-start_frame]}")
        print(f"distance after :")
        for i in range(3): print(f"{(1+i)}: {L_toe_filt[frame-start_frame + (1+i)]}")
        vicon.CreateAnEvent(subject, "Left", "Toe Off", int(frame), 0.0)
        print()
    # for frame in R_toe_offs:
    #     print(f"Right Toe Off frame : {frame}")
    #     vicon.CreateAnEvent(subject, "Right", "Toe Off", int(frame), 0.0)
print(f"Successfully created {len(L_heel_strikes)} Left Heel Strikes and {len(R_heel_strikes)} Right Heel Strikes.")
print(f"Successfully created {len(L_toe_offs)} Left Toe Offs and {len(R_toe_offs)} Right Toe Offs.")