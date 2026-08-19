"""Application bootstrap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from PySide6 import QtGui, QtWidgets

from . import theme


def _configure_gl_format() -> None:
    """Ask for a context pyqtgraph.opengl will accept, before Qt makes one.

    Qt otherwise negotiates a bare-minimum surface format, which on several
    common stacks (Xvfb, VMs, remote desktops) reports OpenGL 2.0 even where the
    driver offers 4.x — and GLViewWidget then raises from inside initializeGL.
    Requesting 2.1 compatibility up front avoids that; if the request cannot be
    met, viewer3d's probe sees it and uses the 2-D fallback.
    """
    fmt = QtGui.QSurfaceFormat()
    # 3.3, not pyqtgraph's stated 2.1 floor: its line items compile GLSL 1.30+
    # shaders, which a 2.1 context rejects at glUseProgram.  Drivers that cannot
    # give 3.3 negotiate down, and viewer3d's probe then decides what to build.
    fmt.setVersion(3, 3)
    fmt.setProfile(QtGui.QSurfaceFormat.CompatibilityProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)  # MSAA, so marker trails do not alias
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)


def _application() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        _configure_gl_format()  # must precede QApplication construction
        app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("Gait Event Editor")
    app.setOrganizationName("gait_Analysis")
    app.setStyle("Fusion")
    app.setStyleSheet(theme.APP_STYLESHEET)

    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(theme.PANEL))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(theme.SURFACE))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(theme.RAISED))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(theme.INK_2))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(theme.INK))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(theme.RAISED))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(theme.INK_2))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(theme.HEEL))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(theme.INK))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(theme.RAISED))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(theme.INK))
    app.setPalette(palette)
    return app


def launch_editor(path: Optional[Path] = None, block: bool = True):
    """Open the editor on ``path``.

    With ``block=True`` this runs the Qt event loop and returns its exit code,
    which is what you want from a script.  With ``block=False`` it returns the
    window so it can be driven from a REPL or a test.
    """
    from .editor_window import EditorWindow

    app = _application()
    window = EditorWindow(Path(path) if path else None)
    window.show()
    if not block:
        return window
    app.exec()
    return window


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gait-event-editor",
        description="Inspect and correct gait events stored in a *_GaitEvents.npz file.",
    )
    parser.add_argument(
        "npz", nargs="?", type=Path, help="path to a *_GaitEvents.npz file"
    )
    args = parser.parse_args(argv)
    if args.npz is not None and not args.npz.exists():
        parser.error(f"no such file: {args.npz}")
    launch_editor(args.npz, block=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
