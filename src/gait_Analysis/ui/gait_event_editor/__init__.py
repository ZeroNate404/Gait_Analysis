"""
Gait event editor — a viewer and editor for ``*_GaitEvents.npz`` files.

Typical use from your pipeline::

    from gait_Analysis.ui.gait_event_editor import launch_editor
    launch_editor(path_to_npz)

or from a shell::

    python -m gait_Analysis.ui.gait_event_editor.app trial_GaitEvents.npz

``event_data`` is deliberately Qt-free, so the load / edit / validate / save
model can be scripted or unit-tested without a display.
"""

from .event_data import (
    EVENT_KEYS,
    EVENT_SPECS,
    AddEvent,
    DeleteEvent,
    GaitEventDocument,
    Issue,
    MoveEvent,
)

__all__ = [
    "EVENT_KEYS",
    "EVENT_SPECS",
    "AddEvent",
    "DeleteEvent",
    "MoveEvent",
    "GaitEventDocument",
    "Issue",
    "launch_editor",
    "main",
]

__version__ = "1.0.0"


def launch_editor(path=None, block: bool = True):
    """Open the editor window (imports Qt lazily)."""
    from .app import launch_editor as _launch

    return _launch(path, block=block)


def main(argv=None) -> int:
    from .app import main as _main

    return _main(argv)
