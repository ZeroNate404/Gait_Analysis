import numpy as np


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