
import numpy as np


file_path = "data\CarePD\BMCLab_SUB05_off_walk_1_canonical_keypoints.npz"

data = np.load(file_path, allow_pickle=True)
for key in data.keys(): print(key)
points = data["positions"]
labels = data["joint_names"]
marker_dict = {label: i for i,label in enumerate(labels)}
L_heel_arr = points[:,28,:]
print(L_heel_arr)