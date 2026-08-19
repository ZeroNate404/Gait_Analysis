"""
The two-row, directly-editable gait event timeline.

Encoding
--------
    row (position)   ->  limb          LEFT on top, RIGHT below
    symbol + hue     ->  event type    heel strike = blue triangle pointing
                                       *down* into the floor, toe off = orange
                                       triangle pointing *up* off it
    shaded bar       ->  stance        heel strike to the next toe off
    white ring       ->  selection

Interaction
-----------
    left click         move the playhead / select the event under the cursor
    left drag (empty)  scrub
    left drag (event)  move that event, with a live ghost and Δ readout
    right click        context menu: add here, delete, snap to playhead
    wheel              zoom the frame axis;  middle drag  pan
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from . import theme
from .event_data import EVENT_SPECS, SIDE_LABEL, SIDES, GaitEventDocument

# -- row geometry (view coordinates; the y range is pinned to [-1, 1]) -----
ROW_Y: Dict[str, float] = {"L": 0.5, "R": -0.5}
ROW_HALF = 0.44
MARK_DY = 0.17
STANCE_HALF = 0.055
HIT_TOL_PX = 13


# --------------------------------------------------------------------------


class SecondsAxis(pg.AxisItem):
    """Top axis: the same frames, labelled in seconds."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_rate = 100.0
        self.first_frame = 0

    def tickStrings(self, values, scale, spacing):
        fr = max(self.frame_rate, 1e-9)
        # Decimals come from the visible range, not from `spacing`: pyqtgraph
        # calls this once per tick level, and a per-level choice renders the
        # same instant as both "2" and "2.0" on the one axis.
        try:
            span = abs(self.range[1] - self.range[0]) / fr
        except (TypeError, IndexError):
            span = 10.0
        decimals = 0 if span >= 20 else 1 if span >= 3 else 2
        return [f"{(v - self.first_frame) / fr:.{decimals}f}" for v in values]


class RowAxis(pg.AxisItem):
    """Left axis: one bold label per limb instead of numeric ticks."""

    def tickValues(self, minVal, maxVal, size):
        return [(1.0, [ROW_Y["L"], ROW_Y["R"]])]

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            if abs(v - ROW_Y["L"]) < 1e-6:
                out.append("LEFT")
            elif abs(v - ROW_Y["R"]) < 1e-6:
                out.append("RIGHT")
            else:
                out.append("")
        return out


class StanceBars(pg.GraphicsObject):
    """One cached QPicture holding every stance rectangle."""

    def __init__(self):
        super().__init__()
        self._picture = QtGui.QPicture()
        self.setZValue(-5)

    def set_bars(self, bars: List[Tuple[float, float, float, bool]]) -> None:
        """``bars`` = ``(x0, x1, y_centre, closed)``."""
        picture = QtGui.QPicture()
        painter = QtGui.QPainter(picture)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        closed_brush = pg.mkBrush(theme.rgba(theme.HEEL, 0.20))
        open_brush = pg.mkBrush(theme.rgba(theme.MUTED, 0.16))
        painter.setPen(QtCore.Qt.NoPen)
        for x0, x1, yc, closed in bars:
            painter.setBrush(closed_brush if closed else open_brush)
            painter.drawRect(
                QtCore.QRectF(x0, yc - STANCE_HALF, max(x1 - x0, 1e-6), 2 * STANCE_HALF)
            )
        painter.end()
        self.prepareGeometryChange()
        self._picture = picture
        self.informViewBoundsChanged()
        self.update()

    def paint(self, painter, *args):
        painter.drawPicture(0, 0, self._picture)

    def boundingRect(self):
        return QtCore.QRectF(self._picture.boundingRect())


class TimelineViewBox(pg.ViewBox):
    """ViewBox with gait-editing mouse semantics instead of pan/zoom defaults."""

    sigClicked = QtCore.Signal(object, object)  # view QPointF, modifiers
    sigDragStart = QtCore.Signal(object, object)
    sigDragMove = QtCore.Signal(object, object)
    sigDragEnd = QtCore.Signal(object, object)
    sigContext = QtCore.Signal(object, object)  # view QPointF, screen QPoint

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setMouseEnabled(x=True, y=False)
        self.setMenuEnabled(False)

    def mouseClickEvent(self, ev):
        pos = self.mapSceneToView(ev.scenePos())
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            self.sigContext.emit(pos, ev.screenPos().toPoint())
            return
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            self.sigClicked.emit(pos, ev.modifiers())
            return
        super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            super().mouseDragEvent(ev, axis=axis)  # middle = pan, right = scale
            return
        ev.accept()
        pos = self.mapSceneToView(ev.scenePos())
        if ev.isStart():
            self.sigDragStart.emit(pos, ev.modifiers())
        elif ev.isFinish():
            self.sigDragEnd.emit(pos, ev.modifiers())
        else:
            self.sigDragMove.emit(pos, ev.modifiers())


# --------------------------------------------------------------------------


class TimelineWidget(QtWidgets.QWidget):
    """Main editable timeline plus a full-trial overview strip."""

    frameChanged = QtCore.Signal(int)
    selectionChanged = QtCore.Signal(object)  # (key, frame) or None
    addRequested = QtCore.Signal(str, int)
    deleteRequested = QtCore.Signal(str, int)
    moveRequested = QtCore.Signal(str, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.doc: Optional[GaitEventDocument] = None
        self.frame: int = 0
        self.selection: Optional[Tuple[str, int]] = None

        self._drag_mode: Optional[str] = None
        self._drag_key: Optional[str] = None
        self._drag_from: int = 0
        self._drag_to: int = 0
        self._preview: Optional[Tuple[str, int, int]] = None
        self._points: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._syncing = False

        self._build()

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.seconds_axis = SecondsAxis(orientation="top")
        self.plot = pg.PlotWidget(
            viewBox=TimelineViewBox(),
            axisItems={"left": RowAxis(orientation="left"),
                       "top": self.seconds_axis},
            background=theme.SURFACE,
        )
        self.vb: TimelineViewBox = self.plot.getViewBox()
        self.plot.setMinimumHeight(140)
        self.plot.showGrid(x=True, y=False, alpha=0.16)
        self.plot.setYRange(-1, 1, padding=0)
        self.plot.hideButtons()

        for name in ("left", "bottom", "top"):
            ax = self.plot.getAxis(name)
            ax.setPen(pg.mkPen(theme.AXIS))
            ax.setTextPen(pg.mkPen(theme.MUTED))
        self.plot.getAxis("left").setWidth(52)
        self.plot.getAxis("left").setTextPen(pg.mkPen(theme.INK_2))
        self.plot.getAxis("bottom").setLabel("frame", color=theme.MUTED)
        self.plot.getAxis("top").setLabel("time (s)", color=theme.MUTED)
        self.plot.getAxis("top").setStyle(maxTickLevel=0)  # major ticks only
        self.plot.getAxis("left").setStyle(tickLength=0)

        # row backgrounds + the divider between limbs
        left_bg = pg.LinearRegionItem(
            values=(ROW_Y["L"] - ROW_HALF, ROW_Y["L"] + ROW_HALF),
            orientation="horizontal",
            brush=pg.mkBrush(theme.rgba(theme.INK, 0.035)),
            pen=pg.mkPen(None),
            movable=False,
        )
        left_bg.setZValue(-30)
        self.plot.addItem(left_bg)
        divider = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(theme.BORDER, width=1))
        divider.setZValue(-20)
        self.plot.addItem(divider)

        self.stance_bars = StanceBars()
        self.plot.addItem(self.stance_bars)

        # stems joining each mark to its row centre line
        self.stems = pg.PlotCurveItem(
            pen=pg.mkPen(theme.rgba(theme.INK, 0.30), width=1), connect="pairs"
        )
        self.stems.setZValue(-2)
        self.plot.addItem(self.stems)

        self.scatters: Dict[str, pg.ScatterPlotItem] = {}
        for key, spec in EVENT_SPECS.items():
            color = theme.event_color(spec.kind)
            item = pg.ScatterPlotItem(
                symbol=spec.symbol,
                size=13,
                brush=pg.mkBrush(color),
                pen=pg.mkPen(theme.SURFACE, width=2),  # 2px surface ring
                hoverable=False,
            )
            item.setZValue(10)
            self.scatters[key] = item
            self.plot.addItem(item)

        self.selection_ring = pg.ScatterPlotItem(
            size=24, brush=pg.mkBrush(None), pen=pg.mkPen(theme.SELECT, width=2)
        )
        self.selection_ring.setZValue(20)
        self.plot.addItem(self.selection_ring)

        self.hover_ring = pg.ScatterPlotItem(
            size=20, brush=pg.mkBrush(None),
            pen=pg.mkPen(theme.rgba(theme.SELECT, 0.55), width=1),
        )
        self.hover_ring.setZValue(19)
        self.plot.addItem(self.hover_ring)

        self.playhead = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen(theme.PLAYHEAD, width=1)
        )
        self.playhead.setZValue(30)
        self.plot.addItem(self.playhead)

        self.drag_guide = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen(theme.SELECT, width=1, style=QtCore.Qt.DashLine),
        )
        self.drag_guide.setZValue(29)
        self.drag_guide.hide()
        self.plot.addItem(self.drag_guide)

        self.tip = pg.TextItem(color=theme.INK, anchor=(0.5, 1.4),
                               fill=pg.mkBrush(theme.rgba("#000000", 0.80)))
        self.tip.setZValue(40)
        self.tip.hide()
        self.plot.addItem(self.tip)

        layout.addWidget(self.plot, 1)
        layout.addWidget(self._build_overview())

        self.vb.sigClicked.connect(self._on_click)
        self.vb.sigDragStart.connect(self._on_drag_start)
        self.vb.sigDragMove.connect(self._on_drag_move)
        self.vb.sigDragEnd.connect(self._on_drag_end)
        self.vb.sigContext.connect(self._on_context)
        self.plot.scene().sigMouseMoved.connect(self._on_hover)
        self.vb.sigXRangeChanged.connect(self._push_range_to_overview)

    def _build_overview(self) -> QtWidgets.QWidget:
        self.overview = pg.PlotWidget(background=theme.SURFACE)
        self.overview.setFixedHeight(30)
        self.overview.setYRange(-1, 1, padding=0)
        self.overview.hideAxis("left")
        self.overview.hideButtons()
        self.overview.setMouseEnabled(x=False, y=False)
        self.overview.setMenuEnabled(False)
        self.overview.setToolTip(
            "Whole trial. Drag the shaded window to pan the timeline above; "
            "drag its edges to zoom."
        )
        # the frame axis of the timeline directly above already labels this range
        ax = self.overview.getAxis("bottom")
        ax.setPen(pg.mkPen(theme.AXIS))
        ax.setStyle(showValues=False, tickLength=0)
        ax.setHeight(4)
        self.overview.getPlotItem().layout.setContentsMargins(52, 2, 10, 0)

        self.overview_marks = pg.PlotCurveItem(
            pen=pg.mkPen(theme.rgba(theme.INK_2, 0.75), width=1), connect="pairs"
        )
        self.overview.addItem(self.overview_marks)
        self.overview_playhead = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen(theme.PLAYHEAD, width=1)
        )
        self.overview.addItem(self.overview_playhead)

        self.range_region = pg.LinearRegionItem(
            brush=pg.mkBrush(theme.rgba(theme.INK, 0.10)),
            pen=pg.mkPen(theme.rgba(theme.INK, 0.35)),
            hoverBrush=pg.mkBrush(theme.rgba(theme.INK, 0.16)),
        )
        self.range_region.setZValue(-10)
        self.overview.addItem(self.range_region)
        self.range_region.sigRegionChanged.connect(self._pull_range_from_overview)
        return self.overview

    # -- document ----------------------------------------------------------

    def set_document(self, doc: Optional[GaitEventDocument]) -> None:
        self.doc = doc
        self.selection = None
        self._preview = None
        if doc is None:
            for item in self.scatters.values():
                item.setData([], [])
            return
        self.seconds_axis.frame_rate = doc.frame_rate
        self.seconds_axis.first_frame = doc.first_frame
        lo, hi = doc.first_frame, doc.last_frame
        pad = max(1, (hi - lo) * 0.01)
        self.vb.setLimits(xMin=lo - pad, xMax=hi + pad, yMin=-1, yMax=1)
        self.overview.setXRange(lo, hi, padding=0.005)
        self.range_region.setBounds([lo, hi])
        self.set_x_range(lo, hi)
        self.frame = doc.mapping.clamp_frame(self.frame)
        self.refresh()

    def refresh(self) -> None:
        if self.doc is None:
            return
        self._rebuild_marks()
        self._rebuild_stance()
        self._rebuild_overview()
        self._refresh_selection_ring()
        self.playhead.setPos(self.frame)
        self.overview_playhead.setPos(self.frame)

    def _rebuild_marks(self) -> None:
        doc = self.doc
        stem_x: List[float] = []
        stem_y: List[float] = []
        self._points = {}
        for key, spec in EVENT_SPECS.items():
            frames = list(doc.events[key])
            if self._preview and self._preview[0] == key:
                _, old, new = self._preview
                frames = sorted([f for f in frames if f != old] + [new])
            y_centre = ROW_Y[spec.side]
            y = y_centre + (MARK_DY if spec.kind == "HS" else -MARK_DY)
            xs = np.asarray(frames, dtype=float)
            ys = np.full(xs.shape, y, dtype=float)
            self.scatters[key].setData(xs, ys)
            self._points[key] = (xs, ys)
            for f in frames:
                stem_x.extend([f, f])
                stem_y.extend([y_centre, y])
        self.stems.setData(np.asarray(stem_x), np.asarray(stem_y))

    def _rebuild_stance(self) -> None:
        bars: List[Tuple[float, float, float, bool]] = []
        for side in SIDES:
            for a, b, closed in self.doc.stance_intervals(side):
                bars.append((float(a), float(b), ROW_Y[side], closed))
        self.stance_bars.set_bars(bars)

    def _rebuild_overview(self) -> None:
        xs: List[float] = []
        ys: List[float] = []
        for key, spec in EVENT_SPECS.items():
            base = 0.15 if spec.side == "L" else -0.95
            top = 0.95 if spec.side == "L" else -0.15
            lo, hi = (base, top) if spec.kind == "HS" else (base, (base + top) / 2)
            for f in self.doc.events[key]:
                xs.extend([f, f])
                ys.extend([lo, hi])
        self.overview_marks.setData(np.asarray(xs), np.asarray(ys))

    # -- external state ----------------------------------------------------

    def set_frame(self, frame: int) -> None:
        self.frame = int(frame)
        self.playhead.setPos(self.frame)
        self.overview_playhead.setPos(self.frame)
        self._keep_playhead_visible()

    def set_selection(self, selection: Optional[Tuple[str, int]]) -> None:
        self.selection = selection
        self._refresh_selection_ring()

    def _refresh_selection_ring(self) -> None:
        if not self.selection or self.doc is None:
            self.selection_ring.setData([], [])
            return
        key, frame = self.selection
        spec = EVENT_SPECS[key]
        y = ROW_Y[spec.side] + (MARK_DY if spec.kind == "HS" else -MARK_DY)
        self.selection_ring.setData([frame], [y])

    def set_x_range(self, lo: float, hi: float) -> None:
        self.vb.setXRange(lo, hi, padding=0)

    def _keep_playhead_visible(self) -> None:
        lo, hi = self.vb.viewRange()[0]
        if lo <= self.frame <= hi:
            return
        span = hi - lo
        new_lo = self.frame - span / 2
        if self.doc is not None:
            new_lo = max(self.doc.first_frame, min(new_lo, self.doc.last_frame - span))
        self.vb.setXRange(new_lo, new_lo + span, padding=0)

    # -- overview sync -----------------------------------------------------

    def _push_range_to_overview(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self.range_region.setRegion(self.vb.viewRange()[0])
        finally:
            self._syncing = False

    def _pull_range_from_overview(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self.vb.setXRange(*self.range_region.getRegion(), padding=0)
        finally:
            self._syncing = False

    # -- hit testing -------------------------------------------------------

    def _hit_test(self, pos: QtCore.QPointF) -> Optional[Tuple[str, int]]:
        if self.doc is None:
            return None
        px, py = self.vb.viewPixelSize()
        px, py = max(px, 1e-12), max(py, 1e-12)
        best: Optional[Tuple[str, int]] = None
        best_d = float(HIT_TOL_PX)
        for key, (xs, ys) in self._points.items():
            if xs.size == 0:
                continue
            d = np.hypot((xs - pos.x()) / px, (ys - pos.y()) / py)
            i = int(np.argmin(d))
            if d[i] < best_d:
                best_d = float(d[i])
                best = (key, int(round(xs[i])))
        return best

    @staticmethod
    def _side_at(pos: QtCore.QPointF) -> str:
        return "L" if pos.y() >= 0 else "R"

    def _snap(self, x: float) -> int:
        frame = int(round(x))
        return self.doc.mapping.clamp_frame(frame) if self.doc else frame

    # -- mouse -------------------------------------------------------------

    def _on_click(self, pos, _modifiers) -> None:
        if self.doc is None:
            return
        hit = self._hit_test(pos)
        if hit:
            self.set_selection(hit)
            self.selectionChanged.emit(hit)
        else:
            self.set_selection(None)
            self.selectionChanged.emit(None)
            self.frameChanged.emit(self._snap(pos.x()))

    def _on_drag_start(self, pos, _modifiers) -> None:
        if self.doc is None:
            return
        hit = self._hit_test(pos)
        if hit:
            self._drag_mode = "move"
            self._drag_key, self._drag_from = hit
            self._drag_to = self._drag_from
            self.set_selection(hit)
            self.selectionChanged.emit(hit)
            self.drag_guide.setPos(self._drag_from)
            self.drag_guide.show()
        else:
            self._drag_mode = "scrub"
            self.frameChanged.emit(self._snap(pos.x()))

    def _on_drag_move(self, pos, _modifiers) -> None:
        if self._drag_mode == "scrub":
            self.frameChanged.emit(self._snap(pos.x()))
            return
        if self._drag_mode != "move" or self._drag_key is None:
            return
        target = self._snap(pos.x())
        taken = set(self.doc.events[self._drag_key]) - {self._drag_from}
        if target in taken:  # never stack two events of one type on one frame
            return
        self._drag_to = target
        self._preview = (self._drag_key, self._drag_from, target)
        self._rebuild_marks()
        self.set_selection((self._drag_key, target))
        spec = EVENT_SPECS[self._drag_key]
        delta = target - self._drag_from
        y = ROW_Y[spec.side] + (MARK_DY if spec.kind == "HS" else -MARK_DY)
        self.tip.setText(f"{spec.short}  {target}   Δ{delta:+d}")
        self.tip.setPos(target, y)
        self.tip.show()

    def _on_drag_end(self, pos, _modifiers) -> None:
        mode, key, src, dst = self._drag_mode, self._drag_key, self._drag_from, self._drag_to
        self._drag_mode = None
        self._drag_key = None
        self._preview = None
        self.drag_guide.hide()
        self.tip.hide()
        if mode == "scrub":
            self.frameChanged.emit(self._snap(pos.x()))
            return
        if mode == "move" and key is not None:
            self._rebuild_marks()
            if dst != src:
                self.moveRequested.emit(key, src, dst)
            else:
                self.refresh()

    def _on_hover(self, scene_pos) -> None:
        if self.doc is None or self._drag_mode is not None:
            return
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            self.hover_ring.setData([], [])
            self.tip.hide()
            return
        pos = self.vb.mapSceneToView(scene_pos)
        hit = self._hit_test(pos)
        if not hit:
            self.hover_ring.setData([], [])
            self.tip.hide()
            self.plot.setCursor(QtCore.Qt.ArrowCursor)
            return
        key, frame = hit
        spec = EVENT_SPECS[key]
        y = ROW_Y[spec.side] + (MARK_DY if spec.kind == "HS" else -MARK_DY)
        self.hover_ring.setData([frame], [y])
        self.tip.setText(
            f"{spec.label}\nframe {frame}   t={self.doc.frame_to_time(frame):.3f}s"
        )
        self.tip.setPos(frame, y)
        self.tip.show()
        self.plot.setCursor(QtCore.Qt.SizeHorCursor)

    def _on_context(self, pos, screen_pos) -> None:
        if self.doc is None:
            return
        frame = self._snap(pos.x())
        side = self._side_at(pos)
        hit = self._hit_test(pos)
        menu = QtWidgets.QMenu(self)

        if hit:
            key, hit_frame = hit
            spec = EVENT_SPECS[key]
            header = menu.addAction(f"{spec.label} @ frame {hit_frame}")
            header.setEnabled(False)
            menu.addSeparator()
            act_del = menu.addAction(f"Delete  {spec.short} @ {hit_frame}")
            act_del.triggered.connect(
                lambda _=False, k=key, f=hit_frame: self.deleteRequested.emit(k, f)
            )
            if self.frame != hit_frame:
                act_snap = menu.addAction(f"Snap to playhead (frame {self.frame})")
                act_snap.triggered.connect(
                    lambda _=False, k=key, f=hit_frame: self.moveRequested.emit(
                        k, f, self.frame
                    )
                )
            menu.addSeparator()

        header = menu.addAction(f"{SIDE_LABEL[side]} row · frame {frame}")
        header.setEnabled(False)
        for key in (f"{side}HS", f"{side}TO"):
            spec = EVENT_SPECS[key]
            act = menu.addAction(f"Add {spec.label} here")
            act.setEnabled(self.doc.can_add(key, frame))
            act.triggered.connect(
                lambda _=False, k=key, f=frame: self.addRequested.emit(k, f)
            )
        menu.addSeparator()
        act_here = menu.addAction(f"Move playhead to frame {frame}")
        act_here.triggered.connect(
            lambda _=False, f=frame: self.frameChanged.emit(f)
        )
        menu.exec(QtGui.QCursor.pos() if screen_pos is None
                  else QtCore.QPoint(screen_pos))

    # -- view helpers ------------------------------------------------------

    def zoom_to_fit(self) -> None:
        if self.doc:
            self.set_x_range(self.doc.first_frame, self.doc.last_frame)

    def zoom_around_playhead(self, span: int) -> None:
        if self.doc is None:
            return
        half = max(4, span // 2)
        lo = max(self.doc.first_frame, self.frame - half)
        self.set_x_range(lo, min(self.doc.last_frame, lo + 2 * half))
