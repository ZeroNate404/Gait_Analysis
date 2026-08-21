"""Write computed gait parameters to a wide CSV: one row per step event,
one column per (parameter x side).
 
Expects the structure produced by compute_gait_params():
 
    gait_params = [
        {"stepping_foot": "left" | "right",
         "cycle":         <per-foot cycle index>,
         "flags":         [<str>, ...],
         "params": {
             "step_order":   <foot>,
             "distance":     {foot: value},   # sparse: only the stepping foot
             "step_length":  {foot: value},
             ...
         }},
        ...
    ]
 
Each entry holds exactly one foot's value, so the contralateral cell is left
blank. Which side a value belongs to is read from "stepping_foot", which is
also cross-checked against the key of each per-foot dict.
"""
 
import csv
import numpy as np
from pathlib import Path
 
 
FOOT_PREFIX = {"left": "L", "right": "R"}
FEET = ("left", "right")
 
# Row identity / quality columns, always first.
META_FIELDS = ("cycle_idx", "stepping_foot", "foot_cycle", "flags")
 
# Redundant with the stepping_foot column.
SKIP_PARAMS = {"step_order"}
 
# Pins the column layout so downstream scripts can rely on it even if the
# setdefault() order in compute_cycle() changes. Parameters not listed here are
# appended in first-appearance order, so adding a new one needs no edit.
PARAM_ORDER = (
    "distance", "stride_distance",
    "step_length", "step_width",
    "step_frames", "step_time",
    "stride_length",
    "stride_frames", "stride_time",
    "stance_frames", "stance_time",
    "swing_frames", "swing_time",
    "single_support_frames", "single_support_time",
    "double_support_frames", "double_support_time",
)
 
 
# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _is_missing(v):
    """True for None or NaN (handles python floats and numpy scalars)."""
    if v is None:
        return True
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False
 
 
def _out_name(name, frame_rate):
    """Column stem: <phase>_frames becomes <phase>_time when converting."""
    if frame_rate and name.endswith("_frames"):
        return name[: -len("_frames")] + "_time"
    return name
 
 
def _out_value(name, value, frame_rate, decimals):
    """Value after optional frames -> seconds conversion and rounding."""
    if _is_missing(value):
        return value
    if frame_rate and name.endswith("_frames"):
        value = float(value) / float(frame_rate)
    if decimals is not None and isinstance(value, (float, np.floating)):
        value = round(float(value), decimals)
    return value
 
 
def _discover_params(gait_params):
    """[(param_name, is_per_foot)] across all cycles, in a stable order.
 
    Known parameters sort by PARAM_ORDER; unknown ones keep first-appearance
    order and land at the end (sorted() is stable).
    """
    found, seen = [], set()
    for cd in gait_params:
        for name, val in cd.get("params", {}).items():
            if name in SKIP_PARAMS or name in seen:
                continue
            seen.add(name)
            found.append((name, isinstance(val, dict)))
 
    rank = {n: i for i, n in enumerate(PARAM_ORDER)}
    return sorted(found, key=lambda p: rank.get(p[0], len(PARAM_ORDER)))
 
 
def _cycle_columns(cd, index, frame_rate, decimals):
    """Yield (column_name, value) for one cycle's parameters."""
    foot = cd["stepping_foot"]
    for name, val in cd.get("params", {}).items():
        if name in SKIP_PARAMS:
            continue
        if isinstance(val, dict):                      # per-foot parameter
            for f, v in val.items():
                if f != foot:                          # sanity guard
                    raise ValueError(
                        f"cycle {index}: '{name}' is keyed on '{f}' but "
                        f"stepping_foot is '{foot}'"
                    )
                yield (f"{FOOT_PREFIX[f]}_{_out_name(name, frame_rate)}",
                       _out_value(name, v, frame_rate, decimals))
        else:                                          # trial-wide scalar
            yield (_out_name(name, frame_rate),
                   _out_value(name, val, frame_rate, decimals))
 
 
# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_gait_param_rows(gait_params, nan_as_blank=True,
                          frame_rate=None, decimals=None):
    """Pivot gait_params (one entry per step event) into wide rows.
 
    Returns (fieldnames, rows), rows being a list of dicts -- ready for
    csv.DictWriter, or for pd.DataFrame(rows, columns=fieldnames).
 
    nan_as_blank : write "" for None/NaN instead of the literal "nan".
    frame_rate   : if given, <phase>_frames columns are emitted as
                   <phase>_time in seconds.
    decimals     : if given, round float values to this many decimal places.
    """
    if not gait_params:
        return list(META_FIELDS), []
 
    # --- Pass 1: column layout -------------------------------------------- #
    params = _discover_params(gait_params)
    fieldnames = list(META_FIELDS)
    for name, per_foot in params:
        stem = _out_name(name, frame_rate)
        if per_foot:
            fieldnames += [f"{FOOT_PREFIX[f]}_{stem}" for f in FEET]
        else:
            fieldnames.append(stem)
 
    # --- Pass 2: one row per step event ------------------------------------ #
    rows = []
    for i, cd in enumerate(gait_params):
        row = {k: "" for k in fieldnames}
        row["cycle_idx"] = i                                # global step index
        row["stepping_foot"] = cd["stepping_foot"]
        row["foot_cycle"] = cd.get("cycle", "")             # per-foot index
        row["flags"] = ";".join(cd.get("flags", []))        # "" when clean
 
        for col, val in _cycle_columns(cd, i, frame_rate, decimals):
            if col not in row:
                raise ValueError(
                    f"cycle {i}: parameter column '{col}' was not discovered "
                    "in pass 1 -- gait_params is inconsistent between cycles"
                )
            row[col] = "" if (nan_as_blank and _is_missing(val)) else val
        rows.append(row)
 
    return fieldnames, rows
 
 
def save_gait_params_csv(gait_params, save_path, nan_as_blank=True,
                         frame_rate=None, decimals=None):
    """Write gait_params to `save_path` as a wide CSV. Returns the path."""
    fieldnames, rows = build_gait_param_rows(
        gait_params, nan_as_blank=nan_as_blank,
        frame_rate=frame_rate, decimals=decimals,
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return save_path