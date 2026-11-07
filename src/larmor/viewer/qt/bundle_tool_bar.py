"""The bar of the drawing tools' options, and the slider fields of a bundle's fill."""

from typing import override

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QToolButton,
    QWidget,
)

from larmor.viewer.model.bundle_tools import ToolChoice, ToolSlider
from larmor.viewer.model.events import Event, ViewStateChanged
from larmor.viewer.model.phantom import (
    DEFAULT_FULLNESS,
    slider_of_tangle,
    tangle_of_slider,
)
from larmor.viewer.model.view_state import ViewField, Workspace
from larmor.viewer.model.viewer import Viewer

BAR_HEIGHT = 34  # px
SLIDER_WIDTH = 160  # px, on the bar
SLIDER_STEPS = 1000  # of a tool's slider, which shows values between its steps
VALUE_WIDTH = 64  # px, of the value beside a slider in the panel
# The range of the Fullness slider, in multiples of the section's share.
FULLNESS_RANGE = (0.2, 10.0)
FULLNESS_STEP = 0.1


class BundleToolBar(QWidget):
    """Shows the options of the drawing tool chosen, such as its reach or radius."""

    def __init__(self, viewer: Viewer, parent: QWidget | None = None) -> None:
        """Creates the bar, following the tool chosen and what the tools change."""
        super().__init__(parent)
        self.viewer = viewer
        self.setObjectName("bundleToolBar")
        self.setFixedHeight(BAR_HEIGHT)
        self.row = QHBoxLayout(self)
        self.row.setContentsMargins(10, 2, 10, 2)
        self.row.setSpacing(8)
        self.sliders: dict[str, tuple[QSlider, QLabel, ToolSlider]] = {}
        self.choices: dict[str, dict[str, QToolButton]] = {}
        self._shown: tuple[object, ...] = ()
        unsubscribe = viewer.scene.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: unsubscribe())
        stop = viewer.bundle_tools.subscribe(self.refresh)
        self.destroyed.connect(lambda *_: stop())
        self.refresh()

    def wanted(self) -> bool:
        """Tells whether the tool chosen has options to show."""
        in_symdwi = self.viewer.scene.view.workspace == Workspace.SIMULATE
        return in_symdwi and bool(self.viewer.bundle_tools.options())

    def refresh(self) -> None:
        """Shows the tool's options, built again for another tool and else updated."""
        options = self.viewer.bundle_tools.options() if self.wanted() else ()
        layout = tuple(
            (o.key, o.low, o.high) if isinstance(o, ToolSlider) else (o.key, o.choices)
            for o in options
        )
        if layout != self._shown:
            self._build(options)
            self._shown = layout
        for option in options:
            if isinstance(option, ToolSlider):
                slider, value, _ = self.sliders[option.key]
                self.sliders[option.key] = (slider, value, option)
                with QSignalBlocker(slider):
                    slider.setValue(_position(option, option.value))
                value.setText(option.text)
            else:
                for word, button in self.choices[option.key].items():
                    button.setChecked(word == option.value)

    def _build(self, options: tuple[ToolSlider | ToolChoice, ...]) -> None:
        """Puts a slider or a row of buttons on the bar for each option."""
        while self.row.count():
            taken = self.row.takeAt(0)
            child = taken.widget() if taken is not None else None
            if child is not None:
                child.setParent(None)
                child.deleteLater()
        self.sliders, self.choices = {}, {}
        for option in options:
            if isinstance(option, ToolSlider):
                self._add_slider(option)
            else:
                self._add_choice(option)
        self.row.addStretch(1)

    def _add_slider(self, option: ToolSlider) -> None:
        """Adds a labelled slider."""
        label = QLabel(option.label, self)
        slider = QSlider(Qt.Orientation.Horizontal, self)
        slider.setRange(0, SLIDER_STEPS)
        slider.setFixedWidth(SLIDER_WIDTH)
        value = QLabel(option.text, self)
        slider.valueChanged.connect(lambda at, key=option.key: self._moved(key, at))
        # A drag of the slider is one step to undo.
        slider.sliderPressed.connect(self.viewer.begin_bundle_step)
        slider.sliderReleased.connect(self.viewer.end_bundle_step)
        self.row.addWidget(label)
        self.row.addWidget(slider)
        self.row.addWidget(value)
        self.sliders[option.key] = (slider, value, option)

    def _add_choice(self, option: ToolChoice) -> None:
        """Adds a row of buttons, one of them checked."""
        group = QButtonGroup(self)
        buttons = {}
        for word in option.choices:
            button = QToolButton(self)
            button.setObjectName("toolButton")
            button.setText(word)
            button.setCheckable(True)
            button.clicked.connect(
                lambda _, key=option.key, chosen=word: (
                    self.viewer.bundle_tools.set_option(key, chosen)
                )
            )
            group.addButton(button)
            self.row.addWidget(button)
            buttons[word] = button
        self.choices[option.key] = buttons

    def _moved(self, key: str, position: int) -> None:
        """Sets an option from its slider, at the steps the option has."""
        option = self.sliders[key][2]
        share = position / SLIDER_STEPS
        value = option.low + share * (option.high - option.low)
        value = option.low + round((value - option.low) / option.step) * option.step
        self.viewer.bundle_tools.set_option(key, value)

    def _on_event(self, event: Event) -> None:
        """Shows the bar again for another tool or workspace."""
        if isinstance(event, ViewStateChanged) and event.name in (
            ViewField.TOOL,
            ViewField.WORKSPACE,
        ):
            self.refresh()


def _position(option: ToolSlider, value: float) -> int:
    """Returns where a value lies on an option's slider."""
    span = max(option.high - option.low, 1e-9)
    return round((value - option.low) / span * SLIDER_STEPS)


class _Note(QLabel):
    """A wrapped line of text that always takes the height it needs at its width.

    A form lays a row out once for a word-wrapped label, and does not make it taller
    when a longer text comes.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Creates the note, with no text."""
        super().__init__(parent)
        self.setObjectName("panelHint")
        self.setWordWrap(True)

    def say(self, text: str) -> None:
        """Shows a text, at the height it needs."""
        self.setText(text)
        self._fit()

    @override
    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        """Takes the height the text needs at the new width."""
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        """Asks the layout for the height the text needs at the width it has."""
        self.setMinimumHeight(max(self.heightForWidth(self.width()), 0))


class FullnessField(QWidget):
    """Sets how full a bundle is, and says plainly whether it fills by its tube.

    A plain tube fills the voxels inside it until the slider moves, and this is the
    thing a reader can most easily misread, so `below`, a row for under this one,
    says it in words next to a button that goes back to the default.
    """

    # The fullness chosen; None for the default.
    chosen = Signal(object)
    # A drag of the slider begins and ends.
    drag_begun = Signal()
    drag_ended = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Creates the slider and its value, and the row below."""
        super().__init__(parent)
        low, high = FULLNESS_RANGE
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(round(low / FULLNESS_STEP), round(high / FULLNESS_STEP))
        self.value = QLabel(self)
        self.value.setMinimumWidth(VALUE_WIDTH)
        self.slider.valueChanged.connect(
            lambda at: self.chosen.emit(at * FULLNESS_STEP)
        )
        self.slider.sliderPressed.connect(self.drag_begun)
        self.slider.sliderReleased.connect(self.drag_ended)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.slider, 1)
        row.addWidget(self.value)
        self.below = QWidget(parent)
        self.note = _Note(self.below)
        self.reset = QPushButton("Default", self.below)
        self.reset.setToolTip("Goes back to the default fill of this bundle")
        self.reset.clicked.connect(lambda: self.chosen.emit(None))
        below = QHBoxLayout(self.below)
        below.setContentsMargins(0, 0, 0, 0)
        below.addWidget(self.note, 1)
        below.addWidget(self.reset, 0, Qt.AlignmentFlag.AlignTop)

    def show_fullness(self, fullness: float | None, is_tube: bool) -> None:
        """Shows a bundle's fullness; None if it has no area of its own."""
        shown = fullness if fullness is not None else DEFAULT_FULLNESS
        with QSignalBlocker(self.slider):
            self.slider.setValue(round(shown / FULLNESS_STEP))
        self.reset.setEnabled(fullness is not None)
        if fullness is None and is_tube:
            self.value.setText("Tube")
            self.note.say(
                "This plain tube fills the voxels inside it. Moving the slider makes "
                "it fill by streamline length instead, with a soft edge."
            )
        elif is_tube:
            self.value.setText(f"{shown:.1f}")
            self.note.say(
                "This tube fills by streamline length, not as a tube. Default goes "
                "back to filling its tube."
            )
        else:
            self.value.setText(f"{shown:.1f}")
            default = " at the default" if fullness is None else ""
            self.note.say(f"It fills by streamline length{default}.")


class TangleField(QWidget):
    """Sets how far along a bundle each streamline slides to another place across it."""

    # The tangle chosen, in mm; 0 for none.
    chosen = Signal(float)
    # A drag of the slider begins and ends.
    drag_begun = Signal()
    drag_ended = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Creates the slider and its value."""
        super().__init__(parent)
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, 100)
        self.value = QLabel(self)
        self.value.setMinimumWidth(VALUE_WIDTH)
        self.slider.valueChanged.connect(
            lambda at: self.chosen.emit(tangle_of_slider(at))
        )
        self.slider.sliderPressed.connect(self.drag_begun)
        self.slider.sliderReleased.connect(self.drag_ended)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.slider, 1)
        row.addWidget(self.value)

    def show_tangle(self, tangle_mm: float, least: float) -> None:
        """Shows a bundle's tangle, and says if it is the shortest it can show."""
        kept = max(tangle_mm, least) if tangle_mm > 0 else 0.0
        with QSignalBlocker(self.slider):
            self.slider.setValue(slider_of_tangle(kept))
        if kept <= 0:
            self.value.setText("Off")
        elif kept <= least * (1 + 1e-6):
            self.value.setText(f"{kept:.1f} mm min")
        else:
            self.value.setText(f"{kept:.0f} mm")
