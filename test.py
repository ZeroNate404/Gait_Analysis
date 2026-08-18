import numpy as np

data = np.load("data\\CarePD\\BMCLab_SUB05_off_walk_1_canonical_keypoints_mm.npz")
for key in data[joint_names]:
    print(key)