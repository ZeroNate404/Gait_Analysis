import numpy as np
import matplotlib.pyplot as plt

def visualize_histo(hist, edges, threshold):
    hist, edges = np.asarray(hist), np.asarray(edges, float)
    widths  = np.diff(edges)
    centres = (edges[:-1] + edges[1:]) / 2.0
    below   = centres < threshold if np.isfinite(threshold) else np.ones(hist.size, bool)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(edges[:-1][below],  hist[below],  width=widths[below],
           align="edge", color="#8a8983", label="below threshold")
    ax.bar(edges[:-1][~below], hist[~below], width=widths[~below],
           align="edge", color="#2a78d6", label="at / above threshold")

    # One noise bin holds hundreds of counts, the real cluster holds single digits.
    # On a linear axis the real cluster is invisible.
    occupied = hist[hist > 0]
    if occupied.size and occupied.max() / occupied.min() >= 20:
        ax.set_yscale("log"); ax.set_ylim(bottom=0.7)

    if np.isfinite(threshold):
        x = min(max(threshold, edges[0]), edges[-1])       # clamp so the label stays on canvas
        off = "  (OFF-SCALE)" if x != threshold else ""
        ax.axvline(x, color="#e34948", ls="--", lw=2, zorder=5)
        ax.annotate(f"threshold = {threshold:.4g}{off}", xy=(x, 1.0),
                    xycoords=("data", "axes fraction"), va="top", fontsize=9,
                    color="#e34948", textcoords="offset points",
                    xytext=(6, -8) if x < centres.mean() else (-6, -8),
                    ha="left" if x < centres.mean() else "right")

        lo, hi = np.nonzero(hist * below)[0], np.nonzero(hist * ~below)[0]
        msg = f"{int(hist[below].sum())} below / {int(hist[~below].sum())} above"
        if lo.size and hi.size:
            empty = int(hi[0] - lo[-1] - 1)
            msg += f"\ngap {edges[lo[-1]+1]:.4g} … {edges[hi[0]]:.4g}  ({empty} empty bins)"
            if empty == 0:
                msg += "\nNO VALLEY — clusters touch"
        ax.text(0.98, 0.95, msg, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="#52514e", linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#d9d8d2"))

    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel("value"); ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25, lw=0.8); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    plt.show()

def otsu_threshold(values, nbins=256):
    """Left-prominence threshold separating real and fake events.
 
    Pass the result to valid_event. Compute one threshold per signal - do not
    share a threshold between feet.

    speed : optional pelvis speed, same length as vX (see sacrum_speed).
            Removes standing-still frames before thresholding. If you pass it
            here, pass the same array to valid_event.
 
    Returns nan when there is too little to threshold; 
    """
    hist, edges = np.histogram(values, bins=nbins)
    # print(f"Histogram: {hist}\nEdges: {edges}")
    centres = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres)
    m1 = m0[-1] - m0
    const_weight = 0.5
    with np.errstate(invalid="ignore", divide="ignore"):
        between = w0 * w1 * (m0 / np.where(w0 == 0, 1, w0) - m1 / np.where(w1 == 0, 1, w1)) ** 2
        between = between * (1.0 - const_weight * (centres / centres.max()))    # <-- the left pull    
    # print(f"Between-class variance: {between}")
    best_var = int(np.nanargmax(between))  # Index of the best variance
    THRESHOLD = float(centres[best_var])  # The threshold is the centre of that bin
    lo, hi = values[values < THRESHOLD], values[values >= THRESHOLD]
    if lo.size and hi.size: THRESHOLD = float(np.sqrt(lo.max() * hi.min()))   # geometric centre of the gap

    # Visualize the histogram and threshold
    visualize_histo(hist, edges, THRESHOLD)

    return THRESHOLD

def pass_otsu(event, values):
    threshold = otsu_threshold(values)
    print(f"Otsu threshold: {threshold}")
    for ev in event:
        if values[ev] < threshold:
            print(f"Event {ev} with value {values[ev]} is below threshold {threshold}")
        else:
            print(f"Event {ev} with value {values[ev]} is above threshold {threshold}")
    return event[values[event] >= threshold]