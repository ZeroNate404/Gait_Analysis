# Gait Event Editor

A desktop editor for the `*_GaitEvents.npz` files your detection step writes. Play the
3-D marker cloud, scrub frames, and correct `LHS` / `RHS` / `LTO` / `RTO` by dragging
marks on a two-row timeline.

![the editor](docs/window.png)

---

## Install

```bash
pip install pyside6 pyqtgraph PyOpenGL PyOpenGL_accelerate numpy
```

`PyOpenGL` is optional. Without it — or on a machine that cannot create a GL
context — the 3-D pane falls back to a 2-D sagittal projection and every editing
feature still works.

## Run

```bash
# straight at a file
python -m gait_event_editor.app  data/GaitEvents/vicon/S01/trial03_GaitEvents.npz

# or through your config, as the original editEvents.py intended
python editEvents.py
python editEvents.py --npz path/to/trial_GaitEvents.npz
```

From your own code:

```python
from gait_Analysis.ui.gait_event_editor import launch_editor
launch_editor(events_path)
```

## Where the files go

Drop the package wherever your `gait_Analysis` package lives, e.g.

```
src/gait_Analysis/
    ui/
        gait_event_editor/       <- this package
            __init__.py  app.py  event_data.py  theme.py
            timeline_widget.py   signal_panel.py  viewer3d.py  editor_window.py
    ...
editEvents.py                    <- replaces your stub
```

`editEvents.py` tries `gait_Analysis.ui.gait_event_editor` first and falls back to a
`gait_event_editor` package sitting beside it, so it works either way.

---

## Reading the display

Two encodings, used consistently in every panel:

| channel | means |
|---|---|
| **row / panel position** | limb — LEFT above, RIGHT below |
| **hue + symbol** | ▼ blue = heel strike / heel marker · ▲ orange = toe off / toe marker |
| shaded bar | stance, from a heel strike to the next toe off on that side |
| grey bar | a heel strike with **no** toe off after it — an unterminated stance |
| white ring | the selected event |

The palette is a validated categorical pair (worst-pair CVD ΔE 26.8, normal-vision
ΔE 31.8, both slots ≥ 3:1 against the surface), and every colour distinction is
doubled by a symbol, so nothing depends on colour alone.

**The signal panel is the point.** It plots each foot marker's position along the
walking axis *relative to the pelvis* — the quantity coordinate-based detectors
threshold. Heel strike belongs at that curve's maximum, toe off at its minimum, and
the events are drawn as marks **on the curve**. A misplaced event is visible as a
mark sitting off the peak, so "is frame 312 really heel strike?" becomes something
you answer by looking.

---

## Editing

| | |
|---|---|
| **select** | click a mark |
| **move** | drag it — a live ghost and a `Δ+7` readout follow the cursor; snapped to whole frames |
| **nudge** | `[` / `]` for ∓1 frame, `Shift+[` / `Shift+]` for ∓5, or the `−1` / `+1` buttons |
| **snap** | `S` moves the selected event to the playhead |
| **add** | `1` `2` `3` `4` at the playhead, or right-click a row for a context menu |
| **delete** | `Del`, the Delete button, or the right-click menu |
| **undo** | `Ctrl+Z` / `Ctrl+Shift+Z`, unlimited, labelled (`Undo move R-HS 545 → 552`) |

Full list under **Help → Keyboard shortcuts** (`F1`).

Two events of the same type can never land on one frame — the drag refuses, rather
than silently merging them.

## Saving

`Ctrl+S` overwrites the original `.npz` and first copies it to
`trial_GaitEvents.npz.bak_YYYYmmdd_HHMMSS`. Downstream code needs no path changes.

Everything except the four event arrays is written back **byte-for-byte**, including
the original integer dtype of the event arrays and whether the archive was
compressed. The write goes to a temp file and is atomically renamed, so an
interrupted save cannot leave a half-written `.npz`. One key is added,
`Date_edited`; pass `write_provenance=False` to `save()` to suppress it.

## Checks

The **Checks** panel re-runs on every edit. Click an entry to jump to it.

- an event outside the data range — *error*
- a heel strike and toe off on the same frame, same side — *error*
- two consecutive heel strikes (or toe offs) on one side — a missing event between them
- a stance or stride whose duration is a strong outlier **against this trial's own
  median**, rather than against a fixed physiological threshold — so a slow or
  pathological gait does not generate a wall of false positives

Deliberately *not* checked: the cross-limb ordering of double support. It is
routinely violated in pathological populations and would be noise.

---

## Frame numbers

You confirmed events are stored as absolute frame numbers, so the only thing left to
resolve is which absolute frame array index 0 corresponds to. That is inferred on
load from `start_frame` / `end_frame` and, weighted highest, from how many events
actually land inside each candidate window — an offset that puts events outside the
data is wrong whatever the metadata says.

The result is shown in the **File** panel as `array[0] =` and is editable. If the
event marks look shifted against the kinematics in the signal panel, change it
there; saving always writes absolute frames back, in the same convention they came
in.

Marker discovery is generic: any `(N,3)` array in the file becomes a marker
(`(3,N)` is transposed), `sacrum_arr` / `LHarr` / `LTarr` / `RHarr` / `RTarr` are
renamed to `SACR` / `LHEE` / `LTOE` / `RHEE` / `RTOE`, and anything else is picked up
under its own name — so adding markers upstream needs no change here.

The walking and vertical axes are inferred too: forward is the axis the pelvis
travels furthest along (`walk_dir` is used when parseable), and vertical is
whichever remaining axis puts the pelvis highest above the feet — a fact about the
recorded geometry rather than an assumption that Z is up. The 3-D scene is rotated
into that frame once at load, so the Sagittal / Frontal / Top presets mean the same
thing whatever convention the lab used.

---

## What changed in `editEvents.py`

Five latent bugs, not style:

1. `raise InvalidConfigError` was raised bare, but `__init__` requires `error_msgs` —
   the guard clause would have died with a `TypeError` instead of reporting the
   configuration problem it had just detected.
2. `config["trial"]["session"]` raises `KeyError` when the key is absent, so the
   `!! Missing ... !!` messages could only ever fire for keys that were present but
   empty. Now `.get()`.
3. The module-level `open(...)` / `edit_events(...)` block ran on **import**, not just
   on execution. Now under `if __name__ == "__main__":`.
4. The config path was relative to the working directory, so the script only worked
   when launched from the repository root. Now resolved against `find_project_root()`.
5. `np.load` left the archive handle open and would reject object arrays; the editor
   reads it fully, with `allow_pickle`, and closes it.

---

## Tests

```bash
python tests/make_synthetic.py     # writes fixtures
python tests/test_editor.py        # 121 checks
xvfb-run -a python tests/test_editor.py   # headless
```

`tests/make_synthetic.py` builds a physically-shaped trial: the foot travels at
exactly pelvis speed at both ends of swing, which puts the kinematic extrema
*exactly* on the event frames. The suite then asserts the events land there — so an
off-by-one anywhere in the axis inference, the pelvis subtraction or the frame
mapping fails the run. It also covers the round trip through `.npz` (dtype,
compression, untouched keys, backup contents), empty and missing event arrays,
transposed marker arrays, Y-up data, drag hit-testing, undo/redo, the keyboard
guard, and — on a real display — that the 3-D scene draws without GL errors.

`event_data.py` imports no Qt, so the load/edit/validate/save model can be scripted
or tested without a display.
