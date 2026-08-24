import os
import numpy as np
from viconnexusapi import ViconNexus
from gait_Analysis.utils.find_project_root import find_project_root


def main():
    # 1. Connect to active Vicon Nexus session
    vicon = ViconNexus.ViconNexus()
    frame_rate = vicon.GetFrameRate()
    subject = vicon.GetSubjectNames()[0]  # Default to the first subject detected in the session
    trial_path, trial_name = vicon.GetTrialName()
    input_type = "vicon"
    session_name = os.path.basename(os.path.normpath(trial_path))

    PROJECT_ROOT = find_project_root()
    INPUT_DIR = PROJECT_ROOT / "data" / "GaitEvents" / input_type
    if input_type == "vicon":
        INPUT_DIR = INPUT_DIR / session_name
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_PATH = INPUT_DIR / f"{trial_name}_GaitEvents.npz"
    
    # 2. Load existing file into a plain dict, then close it
    npz_file = np.load(INPUT_PATH, allow_pickle=True)
    data = {key: npz_file[key] for key in npz_file.files}
    npz_file.close()

    LHS = vicon.GetEvents(subject, "Left", "Foot Strike")
    LTO = vicon.GetEvents(subject, "Left", "Foot Off")
    RHS = vicon.GetEvents(subject, "Right", "Foot Strike")
    RTO = vicon.GetEvents(subject, "Right", "Foot Off") 
    data['LHS'] = np.array(LHS[0],dtype=int)
    data['LTO'] = np.array(LTO[0],dtype=int)
    data['RHS'] = np.array(RHS[0],dtype=int)
    data['RTO'] = np.array(RTO[0],dtype=int)
    print(f"LHS: {data['LHS']}\nLTO: {data['LTO']}\nRHS: {data['RHS']}\nRTO: {data['RTO']}")
    np.savez(INPUT_PATH, **data)


main()