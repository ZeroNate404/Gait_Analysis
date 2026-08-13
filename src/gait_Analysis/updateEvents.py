import os
import numpy as np
import gaitParam_optimized as gp
from viconnexusapi import ViconNexus


def main():
    # 1. Connect to active Vicon Nexus session
    vicon = ViconNexus.ViconNexus()
    frame_rate = vicon.GetFrameRate()
    subject = vicon.GetSubjectNames()[0]  # Default to the first subject detected in the session
    trial_path, trial_name = vicon.GetTrialName()
    session_name = os.path.basename(os.path.normpath(trial_path))
 
    file_path = f"D:\\python_scripts\\Gait_Analysis\\data\\GaitEvents\\{session_name}\\{trial_name}_GaitEvents.npz"
    # 2. Load existing file into a plain dict, then close it
    npz_file = np.load(file_path, allow_pickle=True)
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

    np.savez(file_path, **data)

    gp.compute_gait_params(trial_name, session_name)  # Compute gait parameters after updating events


main()