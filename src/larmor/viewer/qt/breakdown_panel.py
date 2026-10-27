"""The Signal breakdown panel, which shows one voxel's signal of the phantom."""

from typing import Any, override

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, QObject, QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QHideEvent, QPalette, QShowEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from larmor.plot import plot_signal_breakdown
from larmor.viewer.model.breakdown import CardRow, card_rows, heading, title
from larmor.viewer.model.events import BreakdownChanged, Event, PhantomChanged
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.qt.style import form_layout

# The narrowest the panel gets: the chart's two-column legend still fits.
MIN_WIDTH = 340  # px
# The b-values are listed rounded to the nearest 10 s/mm^2, so a shell is one entry.
SHELL_ROUNDING = -1  # decimals
ANGLE_TIP = (
    "The angle between the bundle's fibres and the gradient of the volume chosen "
    "in the Direction list, which it follows. At 0 deg the bundle loses the most "
    "signal to diffusion, at 90 deg the least. Blank at b = 0."
)
VOLUME_TIP = (
    "The volume number is its position in the simulated image, counted from 0 "
    "with the b = 0 volumes included."
)


class BreakdownPanel(QWidget):
    """Shows `viewer.breakdown` as the voxel's table, the volume choice and a chart.

    It keeps no state of its own: the voxel and the volume are the scene's, and
    the chart is `plot_signal_breakdown`'s. While it shows, the breakdown follows
    the pointer.
    """

    title_changed = Signal(str)  # the dock's title, naming the voxel

    def __init__(self, viewer: Viewer, parent: QWidget | None = None) -> None:
        """Builds the voxel table, the volume choice and the chart."""
        super().__init__(parent)
        self.viewer = viewer
        self.setObjectName("breakdownPanel")
        self.setMinimumWidth(MIN_WIDTH)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._build_table(layout)
        self._build_choices(layout)
        self._build_chart(layout)

        # The viewer can outlive the panel: stop listening when it goes.
        unsubscribe = viewer.scene.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: unsubscribe())
        self.redraw()

    def _build_table(self, layout: QVBoxLayout) -> None:
        """Adds the heading, the voxel's table and the hint."""
        self.heading = QLabel(self)
        self.heading.setObjectName("formTitle")
        self.heading.setContentsMargins(14, 10, 14, 4)
        # What fills the voxel: a row per tissue and an indented one per bundle.
        self.table = QWidget(self)
        self.rows = QGridLayout(self.table)
        self.rows.setContentsMargins(14, 0, 14, 0)
        self.rows.setHorizontalSpacing(16)
        self.rows.setVerticalSpacing(2)
        self.hint = QLabel(self)
        self.hint.setObjectName("panelHint")
        self.hint.setWordWrap(True)
        self.hint.setContentsMargins(14, 10, 14, 0)
        for widget in (self.heading, self.table, self.hint):
            layout.addWidget(widget)

    def _build_choices(self, layout: QVBoxLayout) -> None:
        """Adds the b-value and direction choices."""
        choices = QWidget(self)
        form: QFormLayout = form_layout(choices)
        self.b_value = QComboBox(choices)
        self.b_value.setToolTip("The shell whose signal the chart shows")
        self.direction = QComboBox(choices)
        self.direction.setToolTip(
            "The gradient direction of the volume shown. " + VOLUME_TIP
        )
        form.addRow("b-value", self.b_value)
        form.addRow("Direction", self.direction)
        layout.addWidget(choices)
        self.b_value.activated.connect(self._choose_shell)
        self.direction.activated.connect(self._choose_volume)

    def _build_chart(self, layout: QVBoxLayout) -> None:
        """Adds the chart's canvas, drawn again after each resize."""
        self.figure = Figure(figsize=(4.5, 4.0), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)  # type: ignore[no-untyped-call]
        self.canvas.setParent(self)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.canvas.setMinimumHeight(260)
        # A timer of the panel's own, so a pending redraw dies with the panel.
        self._relayout = QTimer(self)
        self._relayout.setSingleShot(True)
        self._relayout.setInterval(0)
        self._relayout.timeout.connect(self._draw_chart)
        self.canvas.installEventFilter(self)
        layout.addWidget(self.canvas, stretch=1)
        # Takes the room the chart leaves while it is hidden, so the choices stay up.
        layout.addStretch(0)
        self.figure.set_facecolor(self._theme()["figure.facecolor"])

    @override
    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        """Starts following the phantom's signal."""
        super().showEvent(event)
        self.viewer.show_breakdown(True)

    @override
    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        """Draws the chart again in the new colours when the theme changes."""
        super().changeEvent(event)
        changes = (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange)
        if event.type() in changes and hasattr(self, "_relayout"):
            self.figure.set_facecolor(self._theme()["figure.facecolor"])
            self._relayout.start()

    @override
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Draws the chart again when the canvas changes size, legend included."""
        if watched is self.canvas and event.type() == QEvent.Type.Resize:
            self._relayout.start()  # after the canvas has resized its figure
        return super().eventFilter(watched, event)

    @override
    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        """Stops following it: a hidden panel needs no breakdown."""
        super().hideEvent(event)
        if not event.spontaneous():  # not when the window is minimised
            self.viewer.show_breakdown(False)

    def _on_event(self, event: Event) -> None:
        """Shows the breakdown again, or the choices after a phantom change."""
        if isinstance(event, BreakdownChanged):
            self.redraw()
        elif isinstance(event, PhantomChanged):
            self._show_choices()

    def redraw(self) -> None:
        """Shows the breakdown as the scene holds it: table, choices and chart.

        With no voxel, only the hint and the empty choices show.
        """
        state = self.viewer.breakdown
        self.title_changed.emit(title(state))
        breakdown = state.breakdown if state is not None else None
        self.figure.clear()
        if state is None or breakdown is None:
            self.heading.hide()
            self.table.hide()
            self.hint.setText("" if state is None else state.problem or "")
            self.hint.show()
            self._show_choices()
            self.canvas.hide()
            return
        self.hint.hide()
        self.canvas.show()
        self.heading.setText(heading(state))
        self.heading.show()
        self._show_rows(card_rows(breakdown))
        self.table.show()
        self._show_choices()
        self._draw_chart()

    def _draw_chart(self) -> None:
        """Draws the chart alone, at the canvas's present size."""
        self.figure.clear()
        state = self.viewer.breakdown
        breakdown = state.breakdown if state is not None else None
        if breakdown is not None:
            theme: Any = self._theme()  # rc_context types its keys as literals
            with rc_context(theme):
                self.figure.set_facecolor(theme["figure.facecolor"])
                ax = self.figure.add_subplot()
                plot_signal_breakdown(breakdown, ax)
        self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _show_rows(self, rows: list[CardRow]) -> None:
        """Fills the table: the column names, then a row per tissue and bundle."""
        while self.rows.count():
            item = self.rows.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.hide()  # at once: deleteLater waits for the event loop
                widget.deleteLater()
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        for column, text in enumerate(("", "Share", "Fibre angle to gradient")):
            label = QLabel(text, self.table)
            label.setObjectName("panelHint")
            if column == 2:
                label.setToolTip(ANGLE_TIP)
            self.rows.addWidget(label, 0, column, alignment=right if column else left)
        for number, row in enumerate(rows, start=1):
            name = QLabel(("    " if row.bundle else "") + row.name, self.table)
            self.rows.addWidget(name, number, 0)
            self.rows.addWidget(
                QLabel(row.share, self.table), number, 1, alignment=right
            )
            self.rows.addWidget(
                QLabel(row.angle, self.table), number, 2, alignment=right
            )
        self.rows.setColumnStretch(0, 1)

    def _theme(self) -> dict[str, Any]:
        """Returns Matplotlib's colours for the theme, from the panel's palette."""
        palette = self.palette()
        window = palette.color(QPalette.ColorRole.Window).name()
        text = palette.color(QPalette.ColorRole.WindowText).name()
        muted = palette.color(QPalette.ColorRole.PlaceholderText).name()
        return {
            "figure.facecolor": window,
            "axes.facecolor": window,
            "axes.edgecolor": muted,
            "axes.labelcolor": text,
            "text.color": text,
            "xtick.color": muted,
            "ytick.color": muted,
            "grid.color": muted,
            "legend.labelcolor": text,
            "font.size": 9,
        }

    def _show_choices(self) -> None:
        """Lists the scan's b-values and the directions of the one shown."""
        gradients = self.viewer.breakdown_gradients()
        state = self.viewer.breakdown
        breakdown = state.breakdown if state is not None else None
        enabled = gradients is not None and breakdown is not None
        self.b_value.setEnabled(enabled)
        self.direction.setEnabled(enabled)
        with QSignalBlocker(self.b_value), QSignalBlocker(self.direction):
            self.b_value.clear()
            self.direction.clear()
            if gradients is None or breakdown is None:
                return
            shells = np.round(gradients.bvals, SHELL_ROUNDING)
            for b in np.unique(shells):
                self.b_value.addItem(f"{b:g} s/mm^2", float(b))
            shell = shells[breakdown.volume]
            self.b_value.setCurrentIndex(self.b_value.findData(float(shell)))
            for volume in np.flatnonzero(shells == shell):
                x, y, z = gradients.bvecs[volume]
                self.direction.addItem(
                    f"Volume {volume}: ({x:+.2f}, {y:+.2f}, {z:+.2f})", int(volume)
                )
                self.direction.setItemData(
                    self.direction.count() - 1, VOLUME_TIP, Qt.ItemDataRole.ToolTipRole
                )
            self.direction.setCurrentIndex(self.direction.findData(breakdown.volume))

    def _choose_shell(self, index: int) -> None:
        """Shows the first volume of the shell chosen."""
        gradients = self.viewer.breakdown_gradients()
        if gradients is None or self.viewer.breakdown is None:
            return
        shells = np.round(gradients.bvals, SHELL_ROUNDING)
        volumes = np.flatnonzero(shells == self.b_value.itemData(index))
        if len(volumes):
            self.viewer.set_breakdown_volume(int(volumes[0]))

    def _choose_volume(self, index: int) -> None:
        """Shows the volume chosen."""
        if self.viewer.breakdown is not None:
            self.viewer.set_breakdown_volume(int(self.direction.itemData(index)))
