"""
Colour and styling constants.

The palette is a validated categorical set (worst-pair CVD ΔE 26.8, normal
vision ΔE 31.8, all slots inside the dark lightness band and above 3:1 against
the chart surface).  It carries exactly two hues, because the encoding scheme
only ever needs two:

    position  ->  limb      (LEFT row / RIGHT row; LEFT panel / RIGHT panel)
    hue       ->  segment   (heel / heel-strike = blue, toe / toe-off = orange)
    symbol    ->  segment   (redundant with hue, so colour is never load-bearing)
    white     ->  selection (primary ink; never used for a data series)

Status colours are reserved for the validation list and always ship with a
text label, never colour alone.
"""

from __future__ import annotations

# -- surfaces & ink --------------------------------------------------------
SURFACE = "#1a1a19"   # plot surface
PANEL = "#0d0d0d"     # window / page plane
RAISED = "#232322"    # toolbars, table headers
INK = "#ffffff"       # primary text
INK_2 = "#c3c2b7"     # secondary text
MUTED = "#898781"     # axis labels, unsided markers
GRID = "#2c2c2a"
AXIS = "#383835"
BORDER = "#3a3a37"

# -- data hues (categorical slots 1 & 2) -----------------------------------
HEEL = "#3987e5"      # heel marker / heel strike
TOE = "#d95926"       # toe marker / toe off
HEEL_DIM = "#1f4c80"
TOE_DIM = "#7a3216"

# -- limb hues, used only where position cannot carry limb (the 3-D view) --
LEFT = "#3987e5"
RIGHT = "#d95926"

SELECT = "#ffffff"    # selection ring
PLAYHEAD = "#ffffff"

# -- status (validation list only; always paired with a text label) --------
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"

SEVERITY_COLOR = {"critical": CRITICAL, "warning": WARNING, "info": MUTED}
SEVERITY_MARK = {"critical": "!!", "warning": "!", "info": "i"}


def event_color(kind: str) -> str:
    """``kind`` is ``"HS"`` or ``"TO"``."""
    return HEEL if kind == "HS" else TOE


def rgba(hex_color: str, alpha: float) -> str:
    """``"#3987e5", 0.2`` -> ``"#3987e533"`` (pyqtgraph accepts 8-digit hex)."""
    a = max(0, min(255, int(round(alpha * 255))))
    return f"{hex_color}{a:02x}"


APP_STYLESHEET = f"""
QWidget {{
    background-color: {PANEL};
    color: {INK_2};
    font-size: 12px;
}}
QMainWindow::separator {{ background: {BORDER}; width: 1px; height: 1px; }}
QToolBar {{
    background: {RAISED};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 3px;
    spacing: 3px;
}}
QToolBar QToolButton {{
    color: {INK_2};
    padding: 4px 9px;
    border: 1px solid transparent;
    border-radius: 4px;
}}
QToolBar QToolButton:hover {{ background: #303030; color: {INK}; }}
QToolBar QToolButton:pressed, QToolBar QToolButton:checked {{
    background: #3a3a38; color: {INK}; border-color: {BORDER};
}}
QToolBar QToolButton:disabled {{ color: #5a5a57; }}
QToolBar::separator {{ background: {BORDER}; width: 1px; margin: 4px 5px; }}
QLabel {{ color: {INK_2}; }}
QLabel[role="hint"] {{ color: {MUTED}; }}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 5px;
    margin-top: 9px;
    padding-top: 7px;
    color: {MUTED};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: {MUTED};
}}
QPushButton {{
    background: {RAISED};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 11px;
    color: {INK_2};
}}
QPushButton:hover {{ background: #303030; color: {INK}; }}
QPushButton:pressed {{ background: #3a3a38; }}
QPushButton:disabled {{ color: #5a5a57; border-color: #2a2a28; }}
QPushButton:checked {{ background: #3a3a38; color: {INK}; }}
QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {RAISED};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 3px 6px;
    color: {INK};
    selection-background-color: {HEEL};
}}
QComboBox::drop-down {{ border: none; width: 16px; }}
QComboBox QAbstractItemView {{
    background: {RAISED};
    border: 1px solid {BORDER};
    selection-background-color: {HEEL};
    color: {INK};
}}
QCheckBox {{ color: {INK_2}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 13px; height: 13px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: {RAISED};
}}
QCheckBox::indicator:checked {{ background: {HEEL}; border-color: {HEEL}; }}
QSlider::groove:horizontal {{ height: 3px; background: {AXIS}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {INK_2}; width: 11px; margin: -5px 0; border-radius: 5px;
}}
QTableWidget, QListWidget, QTreeWidget {{
    background: {SURFACE};
    alternate-background-color: #201f1e;
    border: 1px solid {BORDER};
    border-radius: 5px;
    gridline-color: {GRID};
    color: {INK_2};
    selection-background-color: #2c4f7c;
    selection-color: {INK};
    outline: none;
}}
QHeaderView::section {{
    background: {RAISED};
    color: {MUTED};
    border: none;
    border-bottom: 1px solid {BORDER};
    border-right: 1px solid {BORDER};
    padding: 4px 6px;
}}
QTableWidget::item {{ padding: 2px 4px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: #45443f; border-radius: 5px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: #57564f; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{
    background: #45443f; border-radius: 5px; min-width: 24px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QStatusBar {{
    background: {RAISED};
    border-top: 1px solid {BORDER};
    color: {MUTED};
}}
QStatusBar::item {{ border: none; }}
QMenu {{
    background: {RAISED};
    border: 1px solid {BORDER};
    padding: 4px;
    color: {INK_2};
}}
QMenu::item {{ padding: 5px 22px 5px 14px; border-radius: 3px; }}
QMenu::item:selected {{ background: {HEEL}; color: {INK}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}
QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QToolTip {{
    background: {RAISED};
    color: {INK};
    border: 1px solid {BORDER};
    padding: 4px 6px;
}}
"""
