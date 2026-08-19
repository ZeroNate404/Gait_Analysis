"""
Entry point: resolve the trial's GaitEvents file from the config and open the
editor on it.

Changes from the original stub, all of them latent bugs rather than style:

1. ``raise InvalidConfigError`` was raised bare, but ``__init__`` requires
   ``error_msgs`` — the guard clause would have died with a TypeError instead
   of reporting the actual configuration problem.
2. ``config["trial"]["session"]`` raises KeyError when the key is absent, so
   the "!! Missing ... !!" messages could only ever fire for keys that were
   present but empty.  ``.get()`` makes the intended check work.
3. The module-level ``open(...)`` / ``edit_events(...)`` block ran on *import*,
   not just when the file was executed.  It is now under ``__main__``.
4. The config path was relative to the current working directory, so the script
   only worked when launched from the repository root.  It now resolves against
   ``find_project_root()`` like everything else.
5. ``np.load`` kept the .npz file handle open and refused object arrays; the
   editor reads the archive fully and closes it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional
import yaml
from gait_Analysis.utils.find_project_root import find_project_root

try:
    from gait_Analysis.ui.gait_event_editor import launch_editor
except ImportError:  # editor package sitting next to this file
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gait_event_editor import launch_editor


CONFIG_RELPATH = Path("src") / "gait_Analysis" / "config" / "input_config.yaml"
PROJECT_ROOT = find_project_root()


class InvalidConfigError(Exception):
    def __init__(self, error_msgs):
        self.error_msgs = list(error_msgs)
        super().__init__("\n".join(self.error_msgs))


def resolve_events_path(config: dict) -> Path:
    """Turn a config dict into the path of its ``*_GaitEvents.npz``."""
    trial = (config or {}).get("trial") or {}
    inp = (config or {}).get("input") or {}
    session_name = trial.get("session")
    trial_name = trial.get("name")
    input_type = inp.get("type")

    error_msgs = []
    if not session_name:
        error_msgs.append("!! Missing trial session !!")
    if not trial_name:
        error_msgs.append("!! Missing trial name !!")
    if not input_type:
        error_msgs.append("!! Missing input type !!")
    if error_msgs:
        raise InvalidConfigError(error_msgs)

    PROJECT_ROOT = Path(find_project_root())
    INPUT_DIR = PROJECT_ROOT / "data" / "GaitEvents" / input_type
    if input_type == "vicon": INPUT_DIR = INPUT_DIR / session_name
    return INPUT_DIR / f"{trial_name}_GaitEvents.npz"


def open_config(config: dict, launch: bool = True) -> Path:
    """Resolve the events file from ``config`` and open the editor on it."""
    events_path = resolve_events_path(config)
    if not events_path.exists():
        raise FileNotFoundError(
            f"No gait-events file at {events_path}\n"
            "Run the event-detection step for this trial first, or point "
            "--npz at the file directly."
        )
    if launch:
        launch_editor(events_path, block=True)
    return events_path


def load_config(path: Optional[Path] = None) -> dict:
    config_path = Path(path) if path else Path(find_project_root()) / CONFIG_RELPATH
    if not config_path.exists():
        raise FileNotFoundError(f"No config file at {config_path}")
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh) or {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Open the gait event editor for a trial."
    )
    parser.add_argument("--config", type=Path, help="path to input_config.yaml")
    parser.add_argument(
        "--npz", type=Path, help="open this file directly and ignore the config"
    )
    args = parser.parse_args(argv)

    if args.npz:
        if not args.npz.exists():
            print(f"!! No such file: {args.npz} !!", file=sys.stderr)
            return 2
        launch_editor(args.npz, block=True)
        return 0

    try:
        open_config(load_config(args.config))
    except InvalidConfigError as exc:
        for msg in exc.error_msgs:
            print(msg, file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"!! {exc} !!", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
