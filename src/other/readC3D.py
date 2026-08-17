import ezc3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load C3D
c3d = ezc3d.c3d(r"data\BMCLab\SUB05_off_walk_1.c3d")

# Marker trajectories
frame_rate = c3d["header"]["points"]["frame_rate"]
units = c3d["parameters"]["POINT"]["UNITS"]["value"][0]
points = c3d["data"]["points"]
labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
marker_dict = {label: i for i, label in enumerate(labels)}

print(f"Frame rate: {frame_rate}")
print(f"XYZ Coordinates units : {units}")
print(f"Marker labels: {labels}")

print(f"XYZ Coordinates shape : {points.shape}")
print(f"Number of markers: {points.shape[1]}")
print(f"Number of frames: {points.shape[2]}")

# Create figure
n_frames = points.shape[2]
xyz = points[:3, :, :]
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Initial frame
scatter = ax.scatter(
    xyz[0, :, 0],
    xyz[1, :, 0],
    xyz[2, :, 0]
)

# Set fixed limits so the camera doesn't jump around
ax.set_xlim(np.nanmin(xyz[0])-100, np.nanmax(xyz[0])+100)
ax.set_ylim(np.nanmin(xyz[1])-100, np.nanmax(xyz[1])+100)
ax.set_zlim(np.nanmin(xyz[2])-100, np.nanmax(xyz[2])+100)

ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")


def update(frame):
    scatter._offsets3d = (
        xyz[0, :, frame],
        xyz[1, :, frame],
        xyz[2, :, frame]
    )
    ax.set_title(f"Frame {frame}")

    return scatter,


animation = FuncAnimation(
    fig,
    update,
    frames=n_frames,
    interval=1000 / frame_rate,
)

plt.show()