"""
Kinematic signal panel — the evidence for whether an event sits on the right frame.

One sub-plot per limb, stacked so each sits directly above/below its timeline
row, and x-linked to the timeline so zoom and pan stay in step.

The default signal is the marker's position along the walking axis *relative to
the pelvis*.  That is the quantity coordinate-based detectors threshold: heel
strike falls at its maximum, toe off at its minimum.  Drawing the stored events
as marks **on the curve** turns "is frame 312 really heel strike?" into a
question you answer by looking, not by trusting the detector.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from . import theme
from .event_data import EVENT_SPECS, SIDE_LABEL, SIDES, GaitEventDocument

MODES = (
    ("progression", "Progression (marker − pelvis)"),
    ("vertical", "Vertical position"),
    ("stored", "Stored velocity from file"),
)


class SignalPanel(QtWidgets.QWidget):
    frameChanged = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.doc: Optional[GaitEventDocument] = None
        self.mode = "progression"
        self.frame = 0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground(theme.SURFACE)
        self.graphics.setMinimumHeight(150)
        layout.addWidget(self.graphics)

        self.plots: Dict[str, pg.PlotItem] = {}
        self.curves: Dict[str, Dict[str, pg.PlotDataItem]] = {}
        self.marks: Dict[str, pg.ScatterPlotItem] = {}
        self.playheads: Dict[str, pg.InfiniteLine] = {}
        self.corner_labels: Dict[str, pg.TextItem] = {}

        for row, side in enumerate(SIDES):
            plot = self.graphics.addPlot(row=row, col=0)
            plot.setMenuEnabled(False)
            plot.showGrid(x=True, y=True, alpha=0.13)
            plot.hideButtons()
            plot.setMouseEnabled(x=True, y=False)
            # No rotated axis label: it collides with the tick values in a plot
            # this short.  The limb name goes in the corner instead, pinned in
            # pixel coordinates so it never moves with the data.
            plot.getAxis("left").setWidth(52)
            for name in ("left", "bottom"):
                ax = plot.getAxis(name)
                ax.setPen(pg.mkPen(theme.AXIS))
                ax.setTextPen(pg.mkPen(theme.MUTED))
            if row == 0:
                plot.getAxis("bottom").setStyle(showValues=False)
            else:
                plot.setXLink(self.plots[SIDES[0]])

            # connect="finite" so a dropped marker reads as a gap in the trace
            # rather than a straight line drawn across missing data
            self.curves[side] = {
                "heel": plot.plot(
                    pen=pg.mkPen(theme.HEEL, width=2), connect="finite",
                    name=f"{side} heel",
                ),
                "toe": plot.plot(
                    pen=pg.mkPen(theme.TOE, width=2, style=QtCore.Qt.DashLine),
                    connect="finite", name=f"{side} toe",
                ),
            }
            marks = pg.ScatterPlotItem(size=12, pen=pg.mkPen(theme.SURFACE, width=2))
            marks.setZValue(10)
            plot.addItem(marks)
            self.marks[side] = marks

            head = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(theme.PLAYHEAD, width=1))
            head.setZValue(20)
            plot.addItem(head)
            self.playheads[side] = head

            corner = pg.TextItem(
                color=theme.INK_2, anchor=(0, 0),
                fill=pg.mkBrush(theme.rgba(theme.SURFACE, 0.82)),
            )
            corner.setParentItem(plot.vb)
            corner.setPos(7, 2)
            corner.setZValue(30)
            self.corner_labels[side] = corner
            self.plots[side] = plot

        self.graphics.ci.layout.setSpacing(2)
        self.graphics.scene().sigMouseClicked.connect(self._on_click)

    # -- wiring ------------------------------------------------------------

    def link_x_to(self, view_box) -> None:
        self.plots[SIDES[0]].setXLink(view_box)

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.refresh()

    def set_document(self, doc: Optional[GaitEventDocument]) -> None:
        self.doc = doc
        self.refresh()

    def set_frame(self, frame: int) -> None:
        self.frame = int(frame)
        for head in self.playheads.values():
            head.setPos(self.frame)

    # -- drawing -----------------------------------------------------------

    def _series(self, marker_name: str) -> Optional[np.ndarray]:
        if self.mode == "vertical":
            return self.doc.vertical_signal(marker_name)
        if self.mode == "stored":
            return self.doc.stored_signal(marker_name)
        return self.doc.progression_signal(marker_name)

    def refresh(self) -> None:
        doc = self.doc
        if doc is None:
            for side in SIDES:
                for curve in self.curves[side].values():
                    curve.setData([], [])
                self.marks[side].setData([], [])
            return

        unit = {"progression": "along progression",
                "vertical": "height",
                "stored": "velocity"}[self.mode]

        for side in SIDES:
            feet = doc.foot_markers(side)
            plot = self.plots[side]
            self.corner_labels[side].setHtml(
                f'<span style="color:{theme.INK}; font-weight:600;">'
                f'{SIDE_LABEL[side]}</span>'
                f'<span style="color:{theme.MUTED};">&nbsp;&nbsp;{unit}</span>'
            )
            mark_x: List[float] = []
            mark_y: List[float] = []
            mark_sym: List[str] = []
            mark_brush = []

            for part, kind in (("heel", "HS"), ("toe", "TO")):
                curve = self.curves[side][part]
                marker = feet.get(part)
                series = self._series(marker.name) if marker else None
                if series is None or series.size == 0:
                    curve.setData([], [])
                    continue
                x = np.arange(series.size) + doc.mapping.offset
                curve.setData(x, series)

                spec = EVENT_SPECS[f"{side}{kind}"]
                for f in doc.events[spec.key]:
                    i = doc.mapping.to_index(f)
                    if 0 <= i < series.size and np.isfinite(series[i]):
                        mark_x.append(f)
                        mark_y.append(float(series[i]))
                        mark_sym.append(spec.symbol)
                        mark_brush.append(pg.mkBrush(theme.event_color(spec.kind)))

            self.marks[side].setData(
                x=mark_x, y=mark_y, symbol=mark_sym, brush=mark_brush
            )
            plot.enableAutoRange(axis="y")

    # -- interaction -------------------------------------------------------

    def _on_click(self, ev) -> None:
        if self.doc is None or ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        for plot in self.plots.values():
            if plot.sceneBoundingRect().contains(ev.scenePos()):
                x = plot.vb.mapSceneToView(ev.scenePos()).x()
                self.frameChanged.emit(self.doc.mapping.clamp_frame(int(round(x))))
                ev.accept()
                return


class LegendStrip(QtWidgets.QWidget):
    """One shared legend for the timeline and the signal panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(56, 2, 8, 2)
        row.setSpacing(16)
        entries = (
            ("▼", theme.HEEL, "heel strike / heel marker"),
            ("▲", theme.TOE, "toe off / toe marker"),
            ("▬", theme.rgba(theme.HEEL, 0.55), "stance phase"),
            ("○", theme.SELECT, "selected"),
        )
        for glyph, color, text in entries:
            label = QtWidgets.QLabel(
                f'<span style="color:{color}; font-size:13px;">{glyph}</span>'
                f'<span style="color:{theme.MUTED};">&nbsp;{text}</span>'
            )
            row.addWidget(label)
        row.addStretch(1)
        self.setFixedHeight(20)
