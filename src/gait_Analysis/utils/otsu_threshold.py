"""
gait_event_screen.py
====================
Throw away fake gait events using the left prominence of the swing peak.
 
    LHEE_Threshold = otsu_threshold(LHEE_vX)
    L_heel_strikes = valid_event(L_heel_strikes, LHEE_vX, LHEE_Threshold)
 
    RHEE_Threshold = otsu_threshold(RHEE_vX)
    R_heel_strikes = valid_event(R_heel_strikes, RHEE_vX, RHEE_Threshold)
 
    LTOE_Threshold = otsu_threshold(LTOE_vX)
    L_toe_offs     = valid_event(L_toe_offs, LTOE_vX, LTOE_Threshold)
 
    RTOE_Threshold = otsu_threshold(RTOE_vX)
    R_toe_offs     = valid_event(R_toe_offs, RTOE_vX, RTOE_Threshold)
 
INDEXING
--------
Events are PLAIN INDICES into vX. That is, `vX[event]` must be the velocity
value at that event. Nothing else is assumed and no offset is applied
anywhere - whatever slicing you did to vX, do the same to the events.
 
In gaitParam_revised.py that means using them straight after your existing
`a-1` conversion, together with the full-length vX arrays from the npz.
 
If you pass 1-based frame numbers by mistake, valid_event raises instead of
silently shifting everything by one frame.
 
Heel strike vs toe off is detected automatically from the sign change at the
events, so there is no direction argument to get wrong.
 
WHAT IT DOES
------------
Each foot's velocity has one big positive hump per cycle (the swing). Toe off
sits on its rising flank, heel strike on its falling flank. The LEFT prominence
of that hump - summit minus the lowest point reached walking leftward - tells
real from fake:
 
    normal step    rise huge,  drop huge
    scuff (real)   rise huge,  drop tiny   <- kept
    standing sway  rise tiny,  drop tiny   <- dropped
 
Otsu picks the cut. Nothing is filtered; no sample value or frame index is
modified. The output is always a subset of the events you passed in.
"""
 
import numpy as np
from scipy.signal import find_peaks, peak_prominences
 
__all__ = ["otsu_threshold", "valid_event", "pelvis_speed"]
 
 
# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------
 
def _moving_mask(n, speed, frac):
    """True where the pelvis is travelling. All True when speed is None."""
    if speed is None:
        return np.ones(n, bool)
    s = np.asarray(speed, float).ravel()
    if s.size != n:
        raise ValueError(f"speed has length {s.size} but vX has length {n}; "
                         f"slice them the same way.")
    ref = np.percentile(s, 75)
    if not np.isfinite(ref) or ref <= 0:
        return np.ones(n, bool)
    return s > frac * ref
 
 
def _peak_left_prominences(v, speed, frac):
    """Every local maximum of v, and the left-flank prominence of each."""
    peaks, _ = find_peaks(v)
    if peaks.size:
        peaks = peaks[_moving_mask(v.size, speed, frac)[peaks]]
    if peaks.size == 0:
        return peaks, np.array([])
    # peak_prominences' FIRST return value measures down to the HIGHER of the
    # left and right bases. We want the left one only, so use the base index.
    _, left_bases, _ = peak_prominences(v, peaks)
    return peaks, v[peaks] - v[left_bases]
 
 
def _direction(v, events):
    """-1 if these events are downward crossings (heel strike),
       +1 if upward (toe off). Decided by majority vote."""
    e = events[events > 0]
    if e.size == 0:
        return -1
    down = np.count_nonzero((v[e] < 0) & (v[e - 1] >= 0))
    up = np.count_nonzero((v[e] > 0) & (v[e - 1] <= 0))
    return -1 if down >= up else 1
 
 
def _as_index_array(events, v, name="events"):
    ev = np.asarray(events, dtype=np.int64).ravel()
    if ev.size and (ev.min() < 0 or ev.max() >= v.size):
        raise ValueError(
            f"{name} span [{ev.min()}, {ev.max()}] but vX has length {v.size}. "
            f"Events must be plain indices into vX. If yours are 1-based frame "
            f"numbers, subtract 1 first (or subtract start_frame if vX is "
            f"sliced to the analysis window)."
        )
    return ev
 
 
# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
 
def otsu_threshold(vX, speed=None, frac=0.30, nbins=256):
    """Left-prominence threshold separating real swing peaks from noise.
 
    Pass the result to valid_event. Compute one threshold per signal - do not
    share a threshold between feet: in asymmetric gait the weaker side's swing
    peaks are genuinely smaller, and a shared cut would classify that entire
    side as noise.
 
    speed : optional pelvis speed, same length as vX (see pelvis_speed).
            Removes standing-still frames before thresholding. If you pass it
            here, pass the same array to valid_event.
 
    Returns nan when there is too little to threshold; valid_event treats a
    nan threshold as "keep everything".
    """
    v = np.asarray(vX, float).ravel()
    _, lprom = _peak_left_prominences(v, speed, frac)
    lprom = lprom[np.isfinite(lprom)]
    if lprom.size < 3:
        return float("nan")
 
    # Otsu on a LINEAR histogram. Do not log-transform first: in log space the
    # noise cluster is wide and the real cluster is a handful of points, so
    # Otsu splits the noise instead (115 false events vs 0 in testing).
    hist, edges = np.histogram(lprom, bins=nbins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres)
    m1 = m0[-1] - m0
    with np.errstate(invalid="ignore", divide="ignore"):
        between = w0 * w1 * (m0 / np.where(w0 == 0, 1, w0)
                             - m1 / np.where(w1 == 0, 1, w1)) ** 2
    return float(centres[int(np.nanargmax(between))])
 
 
def valid_event(events, vX, threshold, speed=None, frac=0.30):
    """Keep only the events whose swing peak clears `threshold`.
 
    events    : indices into vX (see INDEXING above)
    vX        : the velocity signal those events were detected in
    threshold : from otsu_threshold(vX)
    speed     : optional, must match what you gave otsu_threshold
 
    Returns a subset of `events`, in ascending order. Heel strike vs toe off
    is detected automatically.
    """
    v = np.asarray(vX, float).ravel()
    ev = _as_index_array(events, v)
    if ev.size == 0:
        return ev
    if not np.isfinite(threshold):
        return np.sort(ev)                      # nothing to threshold: keep all
 
    direction = _direction(v, ev)
    peaks, lprom = _peak_left_prominences(v, speed, frac)
    if peaks.size < 3:
        return np.sort(ev)                      # too few peaks: keep all
 
    ev = ev[_moving_mask(v.size, speed, frac)[ev]]
    good = peaks[lprom >= threshold]
    if good.size == 0 or ev.size == 0:
        return np.array([], dtype=ev.dtype)
 
    # Each qualifying peak claims exactly ONE event. Looking up "which event
    # belongs to this peak" rather than "which peak belongs to this event" is
    # what stops chatter attaching itself to a real swing peak - and it lets a
    # scuff cycle keep both heel strikes, because it has two qualifying peaks.
    # The fences (nxt / prv) stop two peaks claiming the same event.
    kept = []
    for i, pk in enumerate(good):
        if direction < 0:                                   # heel strike
            nxt = good[i + 1] if i + 1 < good.size else v.size
            window = ev[(ev > pk) & (ev < nxt)]
            if window.size:
                kept.append(int(window[0]))                 # first after peak
        else:                                               # toe off
            prv = good[i - 1] if i > 0 else -1
            window = ev[(ev < pk) & (ev > prv)]
            if window.size:
                kept.append(int(window[-1]))                # last before peak
 
    return np.array(sorted(kept), dtype=ev.dtype)
 
 
def pelvis_speed(sacrum_arr, frame_rate):
    """Instantaneous pelvis speed from the sacrum trajectory, for the optional
    standing-still gate. OVERGROUND ONLY - on a treadmill the pelvis does not
    translate during walking either, and the gate would mask the whole trial.
 
    Slice the result exactly as you sliced vX before passing it on.
    """
    dt = 1.0 / float(np.asarray(frame_rate).ravel()[0])
    sac = np.asarray(sacrum_arr, float)
    return np.linalg.norm(np.gradient(sac, dt, axis=0), axis=1)