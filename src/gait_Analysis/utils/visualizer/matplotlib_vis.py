
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

OUTLINE = [pe.withStroke(linewidth=3, foreground="white")]

def visualize_matplotlib(LHEE_vX, LHEE_aligned, LTOE_vX, LTOE_aligned, RHEE_vX, RHEE_aligned, RTOE_vX, 
                         RTOE_aligned, LHS=[], LTO=[], RHS=[], RTO=[], start_frame=1, end_frame=3000):
    # Matplotlib visualization of Velocities and detected events
    x_frames = np.arange(len(LHEE_vX)) + start_frame
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=False)

    axs[0, 0].plot(x_frames, LHEE_vX, color="blue", marker="o", markersize=3)
    axs[0, 0].plot(LHS, LHEE_vX[LHS-start_frame], "x", markersize=8, color="darkblue")
    for f in LHS: axs[0, 0].annotate(str(f), (f, LHEE_vX[f-start_frame]),
                            textcoords="offset points", xytext=(-15, -10),
                            ha="center", fontsize=8, color="darkblue",
                            path_effects=OUTLINE, zorder=5)
    axs[0, 0].axhline(0, linestyle="--", color="black")
    axs[0, 0].set_title("Left Heel")
    axs[0, 0].text(0.5, 0.98, f"Detections at frames = {LHS.tolist()}",
                   transform=axs[0, 0].transAxes, ha="center", va="top", fontsize=8)

    axs[0, 1].plot(x_frames, LTOE_vX, color="green", marker="o", markersize=3)
    axs[0, 1].plot(LTO, LTOE_vX[LTO-start_frame], "x", markersize=8, color="darkgreen")
    for f in LTO: axs[0, 1].annotate(str(f), (f, LTOE_vX[f-start_frame]),
                            textcoords="offset points", xytext=(-15, 10),
                            ha="center", fontsize=8, color="darkgreen",
                            path_effects=OUTLINE, zorder=5)
    axs[0, 1].axhline(0, linestyle="--", color="black")
    axs[0, 1].set_title("Left Toe")
    axs[0, 1].text(0.5, 0.98, f"Detections at frames = {LTO.tolist()}",
                   transform=axs[0, 1].transAxes, ha="center", va="top", fontsize=8)

    axs[1, 0].plot(x_frames, RHEE_vX, color="orange", marker="o", markersize=3)
    axs[1, 0].plot(RHS, RHEE_vX[RHS-start_frame], "x", markersize=8, color="darkred")
    for f in RHS: axs[1, 0].annotate(str(f), (f, RHEE_vX[f-start_frame]),
                            textcoords="offset points", xytext=(-15, -10),
                            ha="center", fontsize=8, color="darkred",
                            path_effects=OUTLINE, zorder=5)
    axs[1, 0].axhline(0, linestyle="--", color="black")
    axs[1, 0].set_title("Right Heel")
    axs[1, 0].text(0.5, 0.98, f"Detections at frames = {RHS.tolist()}",
                   transform=axs[1, 0].transAxes, ha="center", va="top", fontsize=8)

    axs[1, 1].plot(x_frames, RTOE_vX, color="red", marker="o", markersize=3)
    axs[1, 1].plot(RTO, RTOE_vX[RTO-start_frame], "x", markersize=8, color="darkred")
    for f in RTO: axs[1, 1].annotate(str(f), (f, RTOE_vX[f-start_frame]),
                            textcoords="offset points", xytext=(-15, 10),
                            ha="center", fontsize=8, color="darkred",
                            path_effects=OUTLINE, zorder=5)
    axs[1, 1].axhline(0, linestyle="--", color="black")
    axs[1, 1].set_title("Right Toe")
    axs[1, 1].text(0.5, 0.98, f"Detections at frames = {RTO.tolist()}",
                   transform=axs[1, 1].transAxes, ha="center", va="top", fontsize=8)

    for ax in axs.flat:
        ax.set_xlim(left=start_frame, right=end_frame)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Velocity (mm/s)")

    plt.tight_layout()
    plt.show()