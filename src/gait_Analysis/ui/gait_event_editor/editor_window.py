"""Main window: layout, transport, editing commands and validation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from . import theme
from .event_data import (
    EVENT_KEYS,
    EVENT_SPECS,
    GaitEventDocument,
    Issue,
)
from .signal_panel import MODES, LegendStrip, SignalPanel
from .timeline_widget import TimelineWidget
from .viewer3d import CAMERA_PRESETS, Viewer3D

pg.setConfigOptions(antialias=True, background=theme.SURFACE, foreground=theme.INK_2)

SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)


# --------------------------------------------------------------------------
# Side panels
# --------------------------------------------------------------------------


class EventTable(QtWidgets.QWidget):
    """Every event in one sortable list; the keyboard-free way to edit."""

    selected = QtCore.Signal(object)  # (key, frame) or None
    deleteRequested = QtCore.Signal(str, int)
    nudgeRequested = QtCore.Signal(int)

    FILTERS = ("All", "Left only", "Right only", "Heel strikes", "Toe offs")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.doc: Optional[GaitEventDocument] = None
        self._syncing = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(6)
        self.filter = QtWidgets.QComboBox()
        self.filter.addItems(self.FILTERS)
        self.filter.currentIndexChanged.connect(self.refresh)
        top.addWidget(QtWidgets.QLabel("Show"))
        top.addWidget(self.filter, 1)
        self.count_label = QtWidgets.QLabel("")
        self.count_label.setProperty("role", "hint")
        top.addWidget(self.count_label)
        layout.addLayout(top)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Event", "Frame", "Time (s)"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.table.sortByColumn(1, QtCore.Qt.AscendingOrder)
        self.table.itemSelectionChanged.connect(self._on_selection)
        self.table.customContextMenuRequested.connect(self._on_context)
        layout.addWidget(self.table, 1)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(5)
        self.btn_minus = QtWidgets.QPushButton("−1")
        self.btn_minus.setToolTip("Move the selected event one frame earlier  ([)")
        self.btn_minus.clicked.connect(lambda: self.nudgeRequested.emit(-1))
        self.btn_plus = QtWidgets.QPushButton("+1")
        self.btn_plus.setToolTip("Move the selected event one frame later  (])")
        self.btn_plus.clicked.connect(lambda: self.nudgeRequested.emit(1))
        self.btn_delete = QtWidgets.QPushButton("Delete")
        self.btn_delete.setToolTip("Delete the selected event  (Del)")
        self.btn_delete.clicked.connect(self._delete_selected)
        row.addWidget(self.btn_minus)
        row.addWidget(self.btn_plus)
        row.addStretch(1)
        row.addWidget(self.btn_delete)
        layout.addLayout(row)

    def set_document(self, doc: Optional[GaitEventDocument]) -> None:
        self.doc = doc
        self.refresh()

    def _passes(self, key: str) -> bool:
        spec = EVENT_SPECS[key]
        mode = self.filter.currentText()
        return {
            "All": True,
            "Left only": spec.side == "L",
            "Right only": spec.side == "R",
            "Heel strikes": spec.kind == "HS",
            "Toe offs": spec.kind == "TO",
        }[mode]

    def refresh(self) -> None:
        keep = self.current_selection()
        self._syncing = True
        try:
            self.table.setSortingEnabled(False)
            self.table.setRowCount(0)
            if self.doc is None:
                return
            rows = [(f, k) for f, k in self.doc.all_events_sorted() if self._passes(k)]
            self.table.setRowCount(len(rows))
            for r, (frame, key) in enumerate(rows):
                spec = EVENT_SPECS[key]
                glyph = "▼" if spec.kind == "HS" else "▲"
                name = QtWidgets.QTableWidgetItem(f"{glyph}  {spec.short}")
                name.setForeground(QtGui.QColor(theme.event_color(spec.kind)))
                name.setToolTip(spec.label)
                name.setData(QtCore.Qt.UserRole, (key, frame))
                num = QtWidgets.QTableWidgetItem()
                num.setData(QtCore.Qt.DisplayRole, int(frame))
                num.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                secs = QtWidgets.QTableWidgetItem()
                secs.setData(
                    QtCore.Qt.DisplayRole,
                    round(self.doc.frame_to_time(frame), 3),
                )
                secs.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.table.setItem(r, 0, name)
                self.table.setItem(r, 1, num)
                self.table.setItem(r, 2, secs)
            self.table.setSortingEnabled(True)
            total = sum(len(self.doc.events[k]) for k in EVENT_KEYS)
            self.count_label.setText(
                f"{len(rows)} of {total}" if len(rows) != total else f"{total} events"
            )
        finally:
            self._syncing = False
        if keep:
            self.select(keep)

    def current_selection(self) -> Optional[Tuple[str, int]]:
        items = self.table.selectedItems()
        if not items:
            return None
        return self.table.item(items[0].row(), 0).data(QtCore.Qt.UserRole)

    def select(self, selection: Optional[Tuple[str, int]]) -> None:
        self._syncing = True
        try:
            if selection is None:
                self.table.clearSelection()
                return
            for r in range(self.table.rowCount()):
                if self.table.item(r, 0).data(QtCore.Qt.UserRole) == tuple(selection):
                    self.table.selectRow(r)
                    self.table.scrollToItem(self.table.item(r, 0))
                    return
            self.table.clearSelection()
        finally:
            self._syncing = False

    def _on_selection(self) -> None:
        if not self._syncing:
            self.selected.emit(self.current_selection())

    def _delete_selected(self) -> None:
        sel = self.current_selection()
        if sel:
            self.deleteRequested.emit(sel[0], sel[1])

    def _on_context(self, pos) -> None:
        sel = self.current_selection()
        if not sel:
            return
        key, frame = sel
        menu = QtWidgets.QMenu(self)
        act = menu.addAction(f"Delete {EVENT_SPECS[key].short} @ {frame}")
        act.triggered.connect(lambda: self.deleteRequested.emit(key, frame))
        menu.exec(self.table.viewport().mapToGlobal(pos))


class IssueList(QtWidgets.QWidget):
    """Sequence checks — the things a detector gets wrong that a human can see."""

    jumpTo = QtCore.Signal(object)  # (key, frame)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.summary = QtWidgets.QLabel("—")
        self.summary.setProperty("role", "hint")
        layout.addWidget(self.summary)
        self.list = QtWidgets.QListWidget()
        self.list.setWordWrap(True)
        self.list.itemActivated.connect(self._activate)
        self.list.itemClicked.connect(self._activate)
        layout.addWidget(self.list, 1)

    def set_issues(self, issues: List[Issue]) -> None:
        self.list.clear()
        counts = {"critical": 0, "warning": 0, "info": 0}
        for issue in issues:
            counts[issue.severity] += 1
            mark = theme.SEVERITY_MARK[issue.severity]
            item = QtWidgets.QListWidgetItem(f"{mark}  {issue.message}")
            item.setForeground(QtGui.QColor(theme.SEVERITY_COLOR[issue.severity]))
            if issue.frame is not None and issue.key is not None:
                item.setData(QtCore.Qt.UserRole, (issue.key, issue.frame))
            self.list.addItem(item)
        if counts["critical"] or counts["warning"]:
            self.summary.setText(
                f"{counts['critical']} error(s), {counts['warning']} warning(s)"
            )
            self.summary.setStyleSheet(
                f"color: {theme.CRITICAL if counts['critical'] else theme.WARNING};"
            )
        else:
            self.summary.setText("Sequence looks consistent")
            self.summary.setStyleSheet(f"color: {theme.GOOD};")

    def _activate(self, item) -> None:
        target = item.data(QtCore.Qt.UserRole)
        if target:
            self.jumpTo.emit(tuple(target))


class TypingGuard(QtCore.QObject):
    """Stop single-key shortcuts from eating keystrokes meant for a field.

    The editor binds bare keys — ``1``-``4`` to add events, ``[``/``]`` to
    nudge, ``S`` to snap — which would otherwise fire while the user is typing a
    frame number into a spin box, because Qt resolves shortcuts before it
    delivers key presses.  Accepting the ShortcutOverride event is the
    documented way to say "the focused widget wants this key", and it is scoped
    to unmodified keys so Ctrl+S still saves from anywhere.

    Item views are deliberately *not* guarded: Up/Down still walk the event
    table while Left/Right keep stepping frames.
    """

    TEXT_WIDGETS = (
        QtWidgets.QAbstractSpinBox,
        QtWidgets.QLineEdit,
        QtWidgets.QComboBox,
        QtWidgets.QTextEdit,
        QtWidgets.QPlainTextEdit,
    )
    _COMMAND_KEYS = (
        QtCore.Qt.ControlModifier
        | QtCore.Qt.AltModifier
        | QtCore.Qt.MetaModifier
    )

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.ShortcutOverride:
            if not (event.modifiers() & self._COMMAND_KEYS) and self.is_typing():
                event.accept()
                return True
        return False

    @classmethod
    def is_typing(cls, widget=None) -> bool:
        if widget is None:
            widget = QtWidgets.QApplication.focusWidget()
        for _ in range(4):  # a spin box's own QLineEdit holds the focus
            if widget is None:
                return False
            if isinstance(widget, cls.TEXT_WIDGETS):
                return True
            widget = widget.parentWidget()
        return False


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------


class EditorWindow(QtWidgets.QMainWindow):
    def __init__(self, path: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self.doc: Optional[GaitEventDocument] = None
        self.frame = 0
        self.selection: Optional[Tuple[str, int]] = None
        self._playing = False

        self.setWindowTitle("Gait Event Editor")
        self.resize(1500, 940)
        self.setStyleSheet(theme.APP_STYLESHEET)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._play_step = 1

        self._build_ui()
        self._build_actions()
        self._connect()
        self._update_actions()

        self._typing_guard = TypingGuard(self)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self._typing_guard)

        if path:
            self.open_path(Path(path))

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        self.viewer = Viewer3D()
        self.signals = SignalPanel()
        self.timeline = TimelineWidget()
        self.legend = LegendStrip()

        bottom = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        bl.addWidget(self.signals, 3)
        bl.addWidget(self.legend)
        bl.addWidget(self.timeline, 4)
        bl.addWidget(self._build_transport())

        left = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left.addWidget(self.viewer)
        left.addWidget(bottom)
        left.setStretchFactor(0, 5)
        left.setStretchFactor(1, 6)
        left.setChildrenCollapsible(False)
        left.setSizes([430, 550])

        root = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(left)
        root.addWidget(self._build_side_panel())
        root.setStretchFactor(0, 5)
        root.setStretchFactor(1, 0)
        root.setSizes([1120, 340])
        self.setCentralWidget(root)

        self.status_left = QtWidgets.QLabel("No file loaded")
        self.status_right = QtWidgets.QLabel("")
        self.status_right.setProperty("role", "hint")
        bar = self.statusBar()
        bar.addWidget(self.status_left, 1)
        bar.addPermanentWidget(self.status_right)

    def _build_transport(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        bar.setStyleSheet(
            f"background:{theme.RAISED}; border-top:1px solid {theme.BORDER};"
        )
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(4)

        def button(text, tip, slot, width=34):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.setFixedWidth(width)
            b.clicked.connect(slot)
            return b

        row.addWidget(button("|◀", "First frame  (Home)", lambda: self.goto_first()))
        row.addWidget(button("◀◀", "Previous event  (Ctrl+←)", lambda: self.step_event(-1)))
        row.addWidget(button("◀", "Previous frame  (←)", lambda: self.step_frame(-1)))
        self.btn_play = button("▶", "Play / pause  (Space)", self.toggle_play, 44)
        row.addWidget(self.btn_play)
        row.addWidget(button("▶", "Next frame  (→)", lambda: self.step_frame(1)))
        row.addWidget(button("▶▶", "Next event  (Ctrl+→)", lambda: self.step_event(1)))
        row.addWidget(button("▶|", "Last frame  (End)", lambda: self.goto_last()))

        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("Frame"))
        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_spin.setFixedWidth(84)
        self.frame_spin.setKeyboardTracking(False)
        self.frame_spin.valueChanged.connect(self.set_frame)
        row.addWidget(self.frame_spin)
        self.time_label = QtWidgets.QLabel("0.000 s")
        self.time_label.setProperty("role", "hint")
        self.time_label.setFixedWidth(76)
        row.addWidget(self.time_label)

        row.addSpacing(10)
        row.addWidget(QtWidgets.QLabel("Speed"))
        self.speed_combo = QtWidgets.QComboBox()
        for s in SPEEDS:
            self.speed_combo.addItem(f"{s:g}×", s)
        self.speed_combo.setCurrentIndex(SPEEDS.index(1.0))
        self.speed_combo.currentIndexChanged.connect(self._restart_timer)
        row.addWidget(self.speed_combo)
        self.loop_check = QtWidgets.QCheckBox("Loop")
        self.loop_check.setChecked(True)
        row.addWidget(self.loop_check)

        row.addStretch(1)
        self.selection_label = QtWidgets.QLabel("Nothing selected")
        self.selection_label.setProperty("role", "hint")
        row.addWidget(self.selection_label)
        return bar

    def _build_side_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        info = QtWidgets.QGroupBox("File")
        form = QtWidgets.QFormLayout(info)
        form.setContentsMargins(8, 4, 8, 8)
        form.setSpacing(4)
        self.info_name = QtWidgets.QLabel("—")
        self.info_name.setWordWrap(True)
        self.info_rate = QtWidgets.QLabel("—")
        self.info_range = QtWidgets.QLabel("—")
        self.info_axes = QtWidgets.QLabel("—")
        self.info_markers = QtWidgets.QLabel("—")
        for label, widget in (
            ("Trial", self.info_name),
            ("Rate", self.info_rate),
            ("Frames", self.info_range),
            ("Axes", self.info_axes),
            ("Markers", self.info_markers),
        ):
            form.addRow(label, widget)

        self.offset_spin = QtWidgets.QSpinBox()
        self.offset_spin.setRange(-10_000_000, 10_000_000)
        self.offset_spin.setKeyboardTracking(False)
        self.offset_spin.setToolTip(
            "Absolute frame number of array index 0.\n"
            "Inferred on load; change it if the events look shifted "
            "against the kinematics."
        )
        self.offset_spin.valueChanged.connect(self._on_offset_changed)
        form.addRow("array[0] =", self.offset_spin)
        layout.addWidget(info)

        self.table = EventTable()
        events_box = QtWidgets.QGroupBox("Events")
        el = QtWidgets.QVBoxLayout(events_box)
        el.setContentsMargins(8, 4, 8, 8)
        el.addWidget(self.table)
        layout.addWidget(events_box, 3)

        self.issues = IssueList()
        issues_box = QtWidgets.QGroupBox("Checks")
        il = QtWidgets.QVBoxLayout(issues_box)
        il.setContentsMargins(8, 4, 8, 8)
        il.addWidget(self.issues)
        layout.addWidget(issues_box, 2)
        return panel

    # -- actions -----------------------------------------------------------

    def _build_actions(self) -> None:
        def action(text, shortcut=None, slot=None, tip=None, checkable=False):
            a = QtGui.QAction(text, self)
            if shortcut:
                a.setShortcut(QtGui.QKeySequence(shortcut))
            if tip:
                a.setToolTip(tip)
                a.setStatusTip(tip)
            if checkable:
                a.setCheckable(True)
            if slot:
                a.triggered.connect(slot)
            return a

        self.act_open = action("Open…", "Ctrl+O", self.on_open)
        self.act_reload = action("Reload", "Ctrl+R", self.on_reload)
        self.act_save = action("Save", "Ctrl+S", self.on_save,
                               "Overwrite the .npz, keeping a timestamped backup")
        self.act_save_as = action("Save As…", "Ctrl+Shift+S", self.on_save_as)
        self.act_quit = action("Quit", "Ctrl+Q", self.close)

        self.act_undo = action("Undo", "Ctrl+Z", self.on_undo)
        self.act_redo = action("Redo", "Ctrl+Shift+Z", self.on_redo)
        self.act_redo_alt = action("", "Ctrl+Y", self.on_redo)
        self.addAction(self.act_redo_alt)
        self.act_delete = action("Delete event", "Del", self.delete_selected)
        self.act_delete_alt = action("", "Backspace", self.delete_selected)
        self.addAction(self.act_delete_alt)
        self.act_snap = action("Snap to playhead", "S", self.snap_selected_to_playhead)
        self.act_fit = action("Fit timeline", "F", self.timeline.zoom_to_fit)
        self.act_zoom = action("Zoom to playhead", "Z",
                               lambda: self.timeline.zoom_around_playhead(120))
        self.act_clear = action("Clear selection", "Esc",
                                lambda: self.set_selection(None))
        self.addAction(self.act_clear)

        self.add_actions = {}
        for i, key in enumerate(("LHS", "LTO", "RHS", "RTO"), start=1):
            spec = EVENT_SPECS[key]
            self.add_actions[key] = action(
                f"Add {spec.label} at playhead", str(i),
                lambda _=False, k=key: self.add_at_playhead(k)
            )

        self.act_play = action("Play / pause", "Space", self.toggle_play)
        self.addAction(self.act_play)
        for seq, delta in (("Left", -1), ("Right", 1),
                           ("Shift+Left", -10), ("Shift+Right", 10)):
            self.addAction(action("", seq, lambda _=False, d=delta: self.step_frame(d)))
        for seq, delta in (("Ctrl+Left", -1), ("Ctrl+Right", 1),
                           (",", -1), (".", 1)):
            self.addAction(action("", seq, lambda _=False, d=delta: self.step_event(d)))
        self.addAction(action("", "Home", self.goto_first))
        self.addAction(action("", "End", self.goto_last))
        for seq, delta in (("[", -1), ("]", 1),
                           ("Shift+[", -5), ("Shift+]", 5)):
            self.addAction(action("", seq, lambda _=False, d=delta: self.nudge_selected(d)))

        self.act_shortcuts = action("Keyboard shortcuts…", "F1", self.show_shortcuts)

        bar = self.menuBar()
        m = bar.addMenu("&File")
        for a in (self.act_open, self.act_reload, None, self.act_save,
                  self.act_save_as, None, self.act_quit):
            m.addSeparator() if a is None else m.addAction(a)
        m = bar.addMenu("&Edit")
        for a in (self.act_undo, self.act_redo, None, self.act_delete,
                  self.act_snap, None):
            m.addSeparator() if a is None else m.addAction(a)
        for a in self.add_actions.values():
            m.addAction(a)
        m = bar.addMenu("&View")
        for a in (self.act_fit, self.act_zoom):
            m.addAction(a)
        bar.addMenu("&Help").addAction(self.act_shortcuts)

        tb = QtWidgets.QToolBar("Main")
        tb.setMovable(False)
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.addToolBar(tb)
        tb.addAction(self.act_open)
        tb.addAction(self.act_save)
        tb.addSeparator()
        self.tb_undo = QtGui.QAction("Undo", self)
        self.tb_undo.triggered.connect(self.on_undo)
        self.tb_redo = QtGui.QAction("Redo", self)
        self.tb_redo.triggered.connect(self.on_redo)
        tb.addAction(self.tb_undo)
        tb.addAction(self.tb_redo)
        tb.addSeparator()
        tb.addAction(self.act_fit)
        tb.addAction(self.act_zoom)
        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel("  Signal  "))
        self.mode_combo = QtWidgets.QComboBox()
        for value, label in MODES:
            self.mode_combo.addItem(label, value)
        self.mode_combo.currentIndexChanged.connect(
            lambda: self.signals.set_mode(self.mode_combo.currentData())
        )
        tb.addWidget(self.mode_combo)
        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel("  3-D  "))
        self.camera_combo = QtWidgets.QComboBox()
        self.camera_combo.addItems(list(CAMERA_PRESETS))
        self.camera_combo.setEnabled(self.viewer.use_gl)
        self.camera_combo.currentTextChanged.connect(self.viewer.set_camera)
        tb.addWidget(self.camera_combo)

        self.color_combo = QtWidgets.QComboBox()
        self.color_combo.addItem("Colour: side", "side")
        self.color_combo.addItem("Colour: segment", "segment")
        self.color_combo.currentIndexChanged.connect(
            lambda: self.viewer.set_color_mode(self.color_combo.currentData())
        )
        tb.addWidget(self.color_combo)

        self.follow_check = QtWidgets.QCheckBox("Follow")
        self.follow_check.setChecked(True)
        self.follow_check.toggled.connect(self.viewer.set_follow)
        tb.addWidget(self.follow_check)

        self.labels_check = QtWidgets.QCheckBox("Labels")
        self.labels_check.setChecked(True)
        self.labels_check.toggled.connect(self.viewer.set_labels)
        tb.addWidget(self.labels_check)

        tb.addWidget(QtWidgets.QLabel("   Trail  "))
        self.trail_spin = QtWidgets.QSpinBox()
        self.trail_spin.setRange(0, 500)
        self.trail_spin.setValue(40)
        self.trail_spin.setFixedWidth(64)
        self.trail_spin.valueChanged.connect(self.viewer.set_trail)
        tb.addWidget(self.trail_spin)

    def _connect(self) -> None:
        self.timeline.frameChanged.connect(self.set_frame)
        self.timeline.selectionChanged.connect(self.set_selection)
        self.timeline.addRequested.connect(self.add_event)
        self.timeline.deleteRequested.connect(self.delete_event)
        self.timeline.moveRequested.connect(self.move_event)
        self.signals.frameChanged.connect(self.set_frame)
        self.table.selected.connect(self._on_table_selection)
        self.table.deleteRequested.connect(self.delete_event)
        self.table.nudgeRequested.connect(self.nudge_selected)
        self.issues.jumpTo.connect(self._on_table_selection)
        self.signals.link_x_to(self.timeline.vb)

    # -- file --------------------------------------------------------------

    def open_path(self, path: Path) -> bool:
        try:
            doc = GaitEventDocument.load(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Could not open file", f"{path}\n\n{type(exc).__name__}: {exc}"
            )
            return False
        self.doc = doc
        doc.subscribe(self._on_doc_changed)

        self.frame_spin.blockSignals(True)
        self.frame_spin.setRange(doc.first_frame, doc.last_frame)
        self.frame_spin.blockSignals(False)
        self.offset_spin.blockSignals(True)
        self.offset_spin.setValue(doc.mapping.offset)
        self.offset_spin.blockSignals(False)

        self.timeline.set_document(doc)
        self.signals.set_document(doc)
        self.viewer.set_document(doc)
        self.table.set_document(doc)

        self.selection = None
        self.set_frame(doc.first_frame)
        self._refresh_info()
        self._on_doc_changed("loaded")

        if doc.load_notes:
            self.statusBar().showMessage(" · ".join(doc.load_notes), 15000)
        return True

    def on_open(self) -> None:
        if not self._confirm_discard():
            return
        start = str(self.doc.path.parent) if self.doc and self.doc.path else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open gait events", start, "Gait events (*.npz);;All files (*)"
        )
        if path:
            self.open_path(Path(path))

    def on_reload(self) -> None:
        if self.doc and self.doc.path and self._confirm_discard():
            self.open_path(self.doc.path)

    def on_save(self) -> None:
        if self.doc is None:
            return
        try:
            saved, backup = self.doc.save(backup=True)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Save failed", f"{type(exc).__name__}: {exc}"
            )
            return
        msg = f"Saved {saved.name}"
        if backup:
            msg += f"   ·   backup: {backup.name}"
        self.statusBar().showMessage(msg, 8000)
        self._refresh_title()

    def on_save_as(self) -> None:
        if self.doc is None:
            return
        start = str(self.doc.path) if self.doc.path else ""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save gait events as", start, "Gait events (*.npz)"
        )
        if not path:
            return
        try:
            saved, backup = self.doc.save(Path(path), backup=True)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Save failed", f"{type(exc).__name__}: {exc}"
            )
            return
        self.statusBar().showMessage(f"Saved {saved.name}", 8000)
        self._refresh_title()

    def _confirm_discard(self) -> bool:
        if self.doc is None or not self.doc.is_dirty:
            return True
        choice = QtWidgets.QMessageBox.warning(
            self,
            "Unsaved edits",
            "This trial has unsaved event edits.",
            QtWidgets.QMessageBox.Save
            | QtWidgets.QMessageBox.Discard
            | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Save,
        )
        if choice == QtWidgets.QMessageBox.Cancel:
            return False
        if choice == QtWidgets.QMessageBox.Save:
            self.on_save()
            return not self.doc.is_dirty
        return True

    def closeEvent(self, event) -> None:
        self.timer.stop()
        event.accept() if self._confirm_discard() else event.ignore()

    # -- transport ---------------------------------------------------------

    def set_frame(self, frame: int) -> None:
        if self.doc is None:
            return
        frame = self.doc.mapping.clamp_frame(int(frame))
        self.frame = frame
        if self.frame_spin.value() != frame:
            self.frame_spin.blockSignals(True)
            self.frame_spin.setValue(frame)
            self.frame_spin.blockSignals(False)
        self.time_label.setText(f"{self.doc.frame_to_time(frame):.3f} s")
        self.timeline.set_frame(frame)
        self.signals.set_frame(frame)
        self.viewer.update_frame(frame)

    def step_frame(self, delta: int) -> None:
        if self.doc:
            self.set_frame(self.frame + delta)

    def goto_first(self) -> None:
        if self.doc:
            self.set_frame(self.doc.first_frame)

    def goto_last(self) -> None:
        if self.doc:
            self.set_frame(self.doc.last_frame)

    def step_event(self, direction: int) -> None:
        if self.doc is None:
            return
        found = self.doc.nearest_event(self.frame, direction)
        if found:
            frame, key = found
            self.set_frame(frame)
            self.set_selection((key, frame))

    def toggle_play(self) -> None:
        if self.doc is None:
            return
        self._playing = not self._playing
        self.btn_play.setText("❚❚" if self._playing else "▶")
        if self._playing:
            self._restart_timer()
        else:
            self.timer.stop()

    def _restart_timer(self) -> None:
        if self.doc is None or not self._playing:
            return
        fps = max(self.doc.frame_rate * self.speed_combo.currentData(), 0.1)
        # never ask Qt for more than ~60 repaints/s; skip frames instead
        self._play_step = max(1, int(round(fps / 60.0)))
        self.timer.start(max(8, int(round(1000.0 * self._play_step / fps))))

    def _tick(self) -> None:
        if self.doc is None:
            return
        nxt = self.frame + self._play_step
        if nxt > self.doc.last_frame:
            if self.loop_check.isChecked():
                nxt = self.doc.first_frame
            else:
                self.toggle_play()
                return
        self.set_frame(nxt)

    # -- selection ---------------------------------------------------------

    def set_selection(self, selection) -> None:
        selection = tuple(selection) if selection else None
        self.selection = selection
        self.timeline.set_selection(selection)
        self.table.select(selection)
        if selection:
            key, frame = selection
            spec = EVENT_SPECS[key]
            self.selection_label.setText(f"Selected  {spec.label} @ frame {frame}")
        else:
            self.selection_label.setText("Nothing selected")
        self._update_actions()

    def _on_table_selection(self, selection) -> None:
        if selection:
            self.set_selection(selection)
            self.set_frame(selection[1])
        else:
            self.set_selection(None)

    # -- editing -----------------------------------------------------------

    def add_event(self, key: str, frame: int) -> None:
        if self.doc is None:
            return
        if self.doc.add_event(key, frame):
            self.set_selection((key, int(frame)))
        else:
            self.statusBar().showMessage(
                f"{EVENT_SPECS[key].short} already exists at frame {frame}", 4000
            )

    def add_at_playhead(self, key: str) -> None:
        self.add_event(key, self.frame)

    def delete_event(self, key: str, frame: int) -> None:
        if self.doc and self.doc.delete_event(key, frame):
            if self.selection == (key, frame):
                self.set_selection(None)

    def delete_selected(self) -> None:
        if self.selection:
            self.delete_event(*self.selection)

    def move_event(self, key: str, old_frame: int, new_frame: int) -> None:
        if self.doc is None:
            return
        if self.doc.move_event(key, old_frame, new_frame):
            self.set_selection((key, int(new_frame)))
        else:
            self.timeline.refresh()
            self.statusBar().showMessage(
                f"Cannot move {EVENT_SPECS[key].short} to frame {new_frame}", 4000
            )

    def nudge_selected(self, delta: int) -> None:
        if self.selection:
            key, frame = self.selection
            self.move_event(key, frame, frame + delta)

    def snap_selected_to_playhead(self) -> None:
        if self.selection:
            key, frame = self.selection
            self.move_event(key, frame, self.frame)

    def on_undo(self) -> None:
        if self.doc and self.doc.can_undo:
            cmd = self.doc.undo()
            self.statusBar().showMessage(f"Undid: {cmd.label}", 4000)
            self._reselect_after(cmd, undo=True)

    def on_redo(self) -> None:
        if self.doc and self.doc.can_redo:
            cmd = self.doc.redo()
            self.statusBar().showMessage(f"Redid: {cmd.label}", 4000)
            self._reselect_after(cmd, undo=False)

    def _reselect_after(self, cmd, undo: bool) -> None:
        """Put the caret back where the user's attention already is."""
        frames = cmd.touched_frames()
        if not frames:
            return
        key = getattr(cmd, "key", None)
        frame = frames[0] if undo else frames[-1]
        if key and frame in self.doc.events.get(key, ()):
            self.set_selection((key, frame))
            self.set_frame(frame)
        else:
            self.set_selection(None)
            self.set_frame(frame)

    def _on_offset_changed(self, value: int) -> None:
        if self.doc is None:
            return
        self.doc.set_offset(int(value))
        self.frame_spin.blockSignals(True)
        self.frame_spin.setRange(self.doc.first_frame, self.doc.last_frame)
        self.frame_spin.blockSignals(False)
        self.timeline.set_document(self.doc)
        self.signals.refresh()
        self.set_frame(self.doc.mapping.clamp_frame(self.frame))
        self._refresh_info()

    # -- refresh -----------------------------------------------------------

    def _on_doc_changed(self, what: str = "events") -> None:
        if self.doc is None:
            return
        self.timeline.refresh()
        self.signals.refresh()
        self.table.refresh()
        self.issues.set_issues(self.doc.validate())
        self.viewer.update_frame(self.frame)
        self._refresh_title()
        self._update_actions()
        counts = "   ".join(
            f"{EVENT_SPECS[k].short} {len(self.doc.events[k])}" for k in EVENT_KEYS
        )
        self.status_right.setText(counts)

    def _refresh_info(self) -> None:
        doc = self.doc
        if doc is None:
            return
        self.info_name.setText(doc.path.name if doc.path else "—")
        self.info_rate.setText(f"{doc.frame_rate:g} Hz")
        self.info_range.setText(
            f"{doc.first_frame}–{doc.last_frame}  ({doc.n_samples} samples)"
        )
        self.info_axes.setText(doc.axes.describe())
        self.info_markers.setText(", ".join(doc.markers) or "none found")
        self.status_left.setText(str(doc.path) if doc.path else "—")

    def _refresh_title(self) -> None:
        name = self.doc.path.name if (self.doc and self.doc.path) else "no file"
        dirty = "• " if (self.doc and self.doc.is_dirty) else ""
        self.setWindowTitle(f"{dirty}{name} — Gait Event Editor")

    def _update_actions(self) -> None:
        has_doc = self.doc is not None
        self.act_save.setEnabled(has_doc)
        self.act_save_as.setEnabled(has_doc)
        self.act_reload.setEnabled(has_doc and self.doc.path is not None)
        self.act_undo.setEnabled(has_doc and self.doc.can_undo)
        self.act_redo.setEnabled(has_doc and self.doc.can_redo)
        undo_txt = f"Undo {self.doc.undo_label()}" if has_doc and self.doc.can_undo else "Undo"
        redo_txt = f"Redo {self.doc.redo_label()}" if has_doc and self.doc.can_redo else "Redo"
        self.act_undo.setText(undo_txt)
        self.act_redo.setText(redo_txt)
        for act, txt, enabled in (
            (self.tb_undo, undo_txt, has_doc and self.doc.can_undo),
            (self.tb_redo, redo_txt, has_doc and self.doc.can_redo),
        ):
            act.setEnabled(enabled)
            act.setToolTip(txt)
        has_sel = self.selection is not None
        self.act_delete.setEnabled(has_sel)
        self.act_snap.setEnabled(has_sel)
        self.table.btn_delete.setEnabled(has_sel)
        self.table.btn_minus.setEnabled(has_sel)
        self.table.btn_plus.setEnabled(has_sel)

    # -- help --------------------------------------------------------------

    def show_shortcuts(self) -> None:
        rows = [
            ("Space", "play / pause"),
            ("← →", "step one frame"),
            ("Shift + ← →", "step ten frames"),
            ("Ctrl + ← →   or   , .", "jump to previous / next event"),
            ("Home / End", "first / last frame"),
            ("1 2 3 4", "add L-HS / L-TO / R-HS / R-TO at the playhead"),
            ("click a mark", "select it"),
            ("drag a mark", "move it to another frame"),
            ("[ ]", "nudge the selected event ∓1 frame"),
            ("Shift + [ ]", "nudge the selected event ∓5 frames"),
            ("S", "snap the selected event to the playhead"),
            ("Del / Backspace", "delete the selected event"),
            ("Esc", "clear the selection"),
            ("right click", "add / delete / snap menu"),
            ("wheel", "zoom the timeline · middle-drag pans"),
            ("F / Z", "fit timeline / zoom to playhead"),
            ("Ctrl+Z / Ctrl+Shift+Z", "undo / redo"),
            ("Ctrl+S", "save (overwrites, keeps a timestamped backup)"),
        ]
        body = "".join(
            f"<tr><td style='padding:2px 16px 2px 0; color:{theme.INK}; "
            f"white-space:nowrap;'><b>{k}</b></td>"
            f"<td style='padding:2px 0; color:{theme.INK_2};'>{v}</td></tr>"
            for k, v in rows
        )
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Keyboard shortcuts")
        box.setTextFormat(QtCore.Qt.RichText)
        box.setText(f"<table>{body}</table>")
        box.exec()
