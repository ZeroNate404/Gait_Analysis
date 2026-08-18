import sys, datetime, os, yaml
import ezc3d
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.signal import butter, filtfilt
from pathlib import Path
from gait_Analysis.utils.find_project_root import find_project_root

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
error_msgs = []

def generalize(metadata, LHEE, LTOE, RHEE, RTOE, SACR):
    # GENERALIZE DATA
    adjust_units = {"km": 1e6,
                    "hm": 1e5,
                    "dam": 1e4,
                    "m": 1000,
                    "dm": 100,
                    "cm": 10,
                    "mm": 1}

    # Generalize measurement units
    if (metadata["units"] != "mm"):
        LHEE *= adjust_units[metadata["units"]]
        LTOE *= adjust_units[metadata["units"]]
        RHEE *= adjust_units[metadata["units"]]
        RTOE *= adjust_units[metadata["units"]]
        SACR *= adjust_units[metadata["units"]]

    # Generalize coordinate system orientation
    if(metadata["input_type"] == "bmclab"):
        # BMCLab: X(front) | Y(up) | Z(left)
        # X' = X
        # Y' = -Z 
        # Z' = Y
        LHEE = np.column_stack((LHEE[0], -LHEE[2])) 
        LTOE = np.column_stack((LTOE[0], -LTOE[2]))
        RHEE = np.column_stack((RHEE[0], -RHEE[2]))
        RTOE = np.column_stack((RTOE[0], -RTOE[2]))
        SACR = np.column_stack((SACR[0], -SACR[2]))

    # Generalize frame rate
    '''Not yet written'''

    # Return XY trajectories
    return LHEE[:, :2], LTOE[:, :2], RHEE[:, :2], RTOE[:, :2], SACR[:, :2] # Take only XY

def get_vicon_heel_toe_arr(vicon, subject):
    # 2. Get Marker Trajectories
    LHEE_X, LHEE_Y, LHEE_Z, _ = vicon.GetTrajectory(subject, "LHEE")
    LTOE_X, LTOE_Y, LTOE_Z, _ = vicon.GetTrajectory(subject, "LTOE")
    RHEE_X, RHEE_Y, RHEE_Z, _ = vicon.GetTrajectory(subject, "RHEE")
    RTOE_X, RTOE_Y, RTOE_Z, _ = vicon.GetTrajectory(subject, "RTOE")
    LPSI_X, LPSI_Y, LPSI_Z, _ = vicon.GetTrajectory(subject, "LPSI")
    RPSI_X, RPSI_Y, RPSI_Z, _ = vicon.GetTrajectory(subject, "RPSI")
    # Stack 1D X, Y, and Z arrays into a 3D (N, 3) array
    LHEE_arr = np.column_stack((LHEE_X, LHEE_Y, LHEE_Z))
    LTOE_arr  = np.column_stack((LTOE_X, LTOE_Y, LTOE_Z))
    RHEE_arr = np.column_stack((RHEE_X, RHEE_Y, RHEE_Z))
    RTOE_arr  = np.column_stack((RTOE_X, RTOE_Y, RTOE_Z))
    LPSI_arr = np.column_stack((LPSI_X, LPSI_Y, LPSI_Z))
    RPSI_arr = np.column_stack((RPSI_X, RPSI_Y, RPSI_Z))
    # Calculate Sacrum as the midpoint: S = (LPSI + RPSI) / 2
    SACR_arr = (LPSI_arr + RPSI_arr) / 2.0

    return LHEE_arr, LTOE_arr, RHEE_arr, RTOE_arr, SACR_arr

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

def align_with_walkdir(LHEE_arr, LTOE_arr, RHEE_arr, RTOE_arr, sacrum_arr, R):
    # Set Sacrum as Origin
    LHEE_arr -= sacrum_arr
    LTOE_arr -= sacrum_arr
    RHEE_arr -= sacrum_arr
    RTOE_arr -= sacrum_arr

    # Rotate coordinates by R to align with the walking direction
    LHEE_arr = LHEE_arr @ R.T
    LTOE_arr  = LTOE_arr @ R.T
    RHEE_arr = RHEE_arr @ R.T
    RTOE_arr  = RTOE_arr @ R.T

    return LHEE_arr, LTOE_arr, RHEE_arr, RTOE_arr

def compute_projected_velocity(LHEE_arr, LTOE_arr, RHEE_arr, RTOE_arr, frame_rate):
    # Obtain velocity vectors (mm/s)
    dt = 1.0 / frame_rate
    L_heel_vel = np.gradient(LHEE_arr, dt, axis=0)
    L_toe_vel = np.gradient(LTOE_arr, dt, axis=0)
    R_heel_vel = np.gradient(RHEE_arr, dt, axis=0)
    R_toe_vel = np.gradient(RTOE_arr, dt, axis=0)

    # Track vX (velocity in the walking direction) for each marker
    LHEE_vX = L_heel_vel[:, 0]
    LTOE_vX = L_toe_vel[:, 0]
    RHEE_vX = R_heel_vel[:, 0]
    RTOE_vX = R_toe_vel[:, 0]

    return LHEE_vX, LTOE_vX, RHEE_vX, RTOE_vX

def filter_data(LHEE_vX, LTOE_vX, RHEE_vX, RTOE_vX, frame_rate):
    # 4. Filter data (2nd order Butterworth, 6Hz cutoff)
    b, a = butter(2, 6.0 / (0.5 * frame_rate), btype='low')
    LHEE_filt = filtfilt(b, a, LHEE_vX)
    LTOE_filt  = filtfilt(b, a, LTOE_vX)
    RHEE_filt = filtfilt(b, a, RHEE_vX)
    RTOE_filt  = filtfilt(b, a, RTOE_vX)
    return LHEE_filt, LTOE_filt, RHEE_filt, RTOE_filt

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

def save_events_npz(input_type, trial_name, session_name, gait_events):
    PROJECT_ROOT = find_project_root()
    SAVE_DIR = PROJECT_ROOT / "data" / "GaitEvents" / input_type
    if input_type == "vicon":
        SAVE_DIR = SAVE_DIR / session_name
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_PATH = SAVE_DIR / f"{trial_name}_GaitEvents.npz"
    np.savez(SAVE_PATH, **gait_events)

def main(config):
    # 1. Config arguments (INPUT)
    input_type = config["input"]["type"].lower()
    input_dir = config["input"]["directory"]
    session_name = config["trial"]["session"]       # May be Overwritten
    trial_name = config["trial"]["name"]            # May be Overwritten
    frame_rate = 150                                # May be Overwritten
    start_frame = config["trial"]["start_frame"]    # May be overwritten
    end_frame = config["trial"]["end_frame"]        # May be overwritten
    units = ""                                      # Will be Overwritten
    if not input_type or not input_dir: error_msgs.append("Input type and directory must be specified in the configuration.")
    if not start_frame or not end_frame: error_msgs.append("Start frame and end frame must be specified in the configuration.")
    if not frame_rate: error_msgs.append("Frame rate must be specified in the configuration.")
    if(error_msgs): raise InvalidConfigError(error_msgs)

    # 2. Get Marker Trajectories
    if(input_type == "vicon"):
        from viconnexusapi import ViconNexus # Vicon's proprietary motion-capture SDK
        # Connect to active Vicon Nexus session
        vicon = ViconNexus.ViconNexus()
        # Overwrite necessary variables
        subject = vicon.getSubjectNames()[0]
        trial_path, trial_name = vicon.GetTrialName()
        session_name = os.path.basename(os.path.normpath(trial_path))
        frame_rate = vicon.GetFrameRate()
        if(len(sys.argv)>1): start_frame = sys.argv[1]
        if(len(sys.argv)>2): end_frame = sys.argv[2]
        units = "mm"
        xyz_orient = {"X": "front", "Y": "right", "Z": "up"}

        # Marker Trajectories (XYZ)
        LHEE_arr, LTOE_arr, RHEE_arr, RTOE_arr, SACR_arr = get_vicon_heel_toe_arr(vicon, subject)

    elif(input_type == "bmclab"):
        # Load C3D
        c3d_file = ezc3d.c3d(input_dir)
        # Overwrite necessary variables
        frame_rate = c3d_file["header"]["points"]["frame_rate"]
        units = c3d_file["parameters"]["POINT"]["UNITS"]["value"][0]
        xyz_orient = {"X": "front", "Y": "up", "Z": "left"}
        points = c3d_file["data"]["points"]
        labels = c3d_file["parameters"]["POINT"]["LABELS"]["value"]
        marker_dict = {label:i for i,label in enumerate(labels)}

        # Marker Trajectories (XYZ)
        LHEE_arr = np.array(points[:3,marker_dict["L.Heel"],start_frame-1:end_frame]) # 1 --> 0 Indexing
        LTOE_arr = np.array(points[:3,marker_dict["L.MT2"],start_frame-1:end_frame])
        RHEE_arr = np.array(points[:3,marker_dict["R.Heel"],start_frame-1:end_frame])
        RTOE_arr = np.array(points[:3,marker_dict["R.MT2"],start_frame-1:end_frame])
        RPSI_arr = np.array(points[:3,marker_dict["R.PSIS"],start_frame-1:end_frame])
        LPSI_arr = np.array(points[:3,marker_dict["L.PSIS"],start_frame-1:end_frame])
        SACR_arr = (RPSI_arr + LPSI_arr) / 2

    elif(input_type == "carepd"):
        # Load NPZ
        data = np.load(input_dir, allow_pickle=True)
        # Overwrite necessary variables
        frame_rate = data["fps"]
        units = data["unit"]
        xyz_orient = {"X": "front", "Y": "right", "Z": "up"}
        points = data["positions"]
        labels = data["joint_names"]
        marker_dict = {label:i for i,label in enumerate(labels)}
        start_frame, end_frame = 1, points.shape[0]  # Use all frames in the NPZ file

        # Marker Trajectories (XYZ)
        LHEE_arr = np.array(points[:,marker_dict["left_heel"],:])
        LTOE_arr = np.array(points[:,marker_dict["left_big_toe"],:])
        RHEE_arr = np.array(points[:,marker_dict["right_heel"],:])
        RTOE_arr = np.array(points[:,marker_dict["right_big_toe"],:])
        SACR_arr = np.array(points[:,marker_dict["sacrum"],:])

    else: error_msgs.append("Input Type not Specified")
    if(error_msgs): raise InvalidConfigError(error_msgs)
    
    '''
    GENERALIZE INPUT METADATA
    Measurement unit        : mm
    World XYZ               : X(front) | Y(right) | Z(up)
    Frame_rate              : 100 Hz
    Time alignment          : First Heel Strike
    '''
    metadata = {"units": units,
                "frame_rate": frame_rate,
                "xyz_orientation": xyz_orient,
                "input_type": input_type
                }
    LHEE_arr, LTOE_arr, RHEE_arr, RTOE_arr, SACR_arr = generalize(metadata, LHEE_arr, LTOE_arr, RHEE_arr, RTOE_arr, SACR_arr)

    # 3. Compute Walking Direction 
    SACR_arr_cropped = SACR_arr[start_frame-1:end_frame] # 1 -> 0 Indexing
    R, walk_dir = compute_walk_dir(SACR_arr_cropped, frame_rate)

    # 4. Transform Coordinates to align with walking direction
    LHEE_arr, LTOE_arr, RHEE_arr, RTOE_arr = align_with_walkdir(LHEE_arr, LTOE_arr, RHEE_arr, RTOE_arr, SACR_arr, R)

    # 4. Determine velocity/walking direction
    LHEE_vX, LTOE_vX, RHEE_vX, RTOE_vX = compute_projected_velocity(LHEE_arr, LTOE_arr, RHEE_arr, RTOE_arr, frame_rate)

    # 5. Filter data (2nd order Butterworth, 6Hz cutoff)(Optional)
    # LHEE_vX, LTOE_vX, RHEE_vX, RTOE_vX = filter_data(LHEE_vX, LTOE_vX, RHEE_vX, RTOE_vX, frame_rate)

    # 6. Detect Heel Strike (+ to - velocity) and Toe Off (- to + velocity) events
    L_heel_strikes, L_toe_offs, R_heel_strikes, R_toe_offs = detect_events(LHEE_vX[start_frame-1:end_frame], # Crop to specified frame range
                                                                           LTOE_vX[start_frame-1:end_frame], 
                                                                           RHEE_vX[start_frame-1:end_frame], 
                                                                           RTOE_vX[start_frame-1:end_frame], 
                                                                           start_frame)

    # 7. Visualize Events to Matplotlib
    visualize_matplotlib(LHEE_vX[start_frame-1:end_frame], 
                         LTOE_vX[start_frame-1:end_frame], 
                         RHEE_vX[start_frame-1:end_frame], 
                         RTOE_vX[start_frame-1:end_frame],
                         L_heel_strikes, L_toe_offs, R_heel_strikes, R_toe_offs, start_frame, end_frame)
    # 8. Write Events to Vicon Nexus
    if(input_type == "vicon"):
        nexus_write_events(vicon, subject, L_heel_strikes, L_toe_offs, R_heel_strikes, R_toe_offs)

    # 9. Save Events to NPZ file
    gait_events = {
        "Date" : datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        "frame_rate": frame_rate,
        "start_frame": start_frame,
        "end_frame": end_frame,

        "walk_dir" : walk_dir,
        "sacrum_arr": SACR_arr,

        "LHarr": LHEE_arr, # frames [1,N]
        "LTarr": LTOE_arr,
        "RHarr": RHEE_arr,
        "RTarr": RTOE_arr,

        "LHvX": LHEE_vX, # frames [1,N]
        "LTvX": LTOE_vX,
        "RHvX": RHEE_vX,
        "RTvX": RTOE_vX,  
        
        "LHS": L_heel_strikes,
        "RHS": R_heel_strikes,
        "LTO": L_toe_offs,
        "RTO": R_toe_offs,
    }
    if(input_type == "vicon"):
        gait_events["Run_name"] = trial_name
        gait_events["Session_name"] = session_name
        gait_events["Subject_name"] = subject

    save_events_npz(input_type, trial_name, session_name, gait_events)

class InvalidConfigError(Exception):
    def __init__(self, error_msgs):
        self.error_msgs = error_msgs
        super().__init__("\n".join(error_msgs))

with open("src/gait_analysis/config/input_config.yaml", "r") as f:
    config = yaml.safe_load(f)
try:
    main(config)
except InvalidConfigError as e:
    for msg in e.error_msgs:
        print(msg)