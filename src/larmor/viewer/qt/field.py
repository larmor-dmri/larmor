"""The fields that edit options and settings, one widget per kind of value."""

from dataclasses import Field
from typing import Any, override

from PySide6.QtCore import QEvent, QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QWidget,
)

from larmor.viewer.model.options import Colour, OptionInfo
from larmor.viewer.model.viewer import MIXED
from larmor.viewer.qt.style import icon_path

SLIDER_MIN_WIDTH = 80  # px


class SliderField(QWidget):
    """Edits a number with a slider and a text field kept in step."""

    edited = Signal(object)

    def __init__(
        self,
        info: OptionInfo,
        parent: QWidget | None = None,
        bounds: tuple[float, float] | None = None,
        field_type: type[int] | type[float] = float,
    ) -> None:
        """Creates the field; `bounds` stand in where the option declares none."""
        super().__init__(parent)
        self.info = info
        self.field_type = field_type
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.set_bounds(bounds)
        # Narrower, a form puts the field below its label rather than squeeze it.
        self.slider.setMinimumWidth(SLIDER_MIN_WIDTH)
        self.text = QLineEdit(self)
        self.text.setFixedWidth(60)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.text)

        self.slider.valueChanged.connect(
            lambda i: self.edited.emit(self._from_slider(i))
        )
        self.text.editingFinished.connect(
            lambda: self.edited.emit(self._parse(self.text.text()))
        )

    def set_bounds(self, bounds: tuple[float, float] | None) -> None:
        """Spans the slider over `bounds` where the option declares none."""
        info = self.info
        lo = info.minimum if info.minimum is not None else bounds and bounds[0]
        hi = info.maximum if info.maximum is not None else bounds and bounds[1]
        if lo is None or hi is None:
            raise TypeError(f"{info.label} needs a minimum and a maximum for a slider.")
        self.minimum = lo
        # A thousand steps across a range from the data; 1 for a constant image.
        self.step = info.step or (hi - lo) / 1000 or 1.0
        with QSignalBlocker(self.slider):
            self.slider.setRange(0, round((hi - lo) / self.step))

    def _from_slider(self, i: int) -> int | float:
        """Returns the value at a slider position."""
        return self.field_type(round(self.minimum + i * self.step, 10))

    def _parse(self, text: str) -> object:
        """Returns the text as a number, or the text itself if it is not one."""
        try:
            return self.field_type(text)
        except ValueError:
            return text

    def show_value(self, value: object) -> None:
        """Shows a value from the model without sending `edited`."""
        with QSignalBlocker(self.slider):
            if value is MIXED:
                self.text.clear()
                self.text.setPlaceholderText("mixed")
            else:
                assert isinstance(value, (int, float))
                self.slider.setValue(round((value - self.minimum) / self.step))
                self.text.setText(f"{value:g}")


class CheckField(QCheckBox):
    """Edits an on/off option."""

    edited = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Creates the field."""
        super().__init__(parent)
        self.clicked.connect(lambda checked: self.edited.emit(checked))

    def show_value(self, value: object) -> None:
        """Shows a value from the model; MIXED shows as partially checked."""
        self.setTristate(value is MIXED)
        if value is MIXED:
            self.setCheckState(Qt.CheckState.PartiallyChecked)
        else:
            self.setCheckState(
                Qt.CheckState.Checked if value else Qt.CheckState.Unchecked
            )


class ChoiceField(QComboBox):
    """Edits an option with a fixed set of choices, in a drop-down."""

    edited = Signal(object)

    def __init__(self, choices: tuple[str, ...], parent: QWidget | None = None) -> None:
        """Creates the field with its choices."""
        super().__init__(parent)
        self.addItems(list(choices))
        self.activated.connect(lambda i: self.edited.emit(self.itemText(i)))

    def show_value(self, value: object) -> None:
        """Shows a value from the model; MIXED leaves the box blank."""
        self.setCurrentIndex(-1 if value is MIXED else self.findText(str(value)))


class ItemField(QComboBox):
    """Edits an option that names an item or its data, in a drop-down with "None"."""

    edited = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Creates the field, with no item to choose until `set_choices`."""
        super().__init__(parent)
        self._ids: list[object] = []  # an item's id or a data name, by row
        self.set_choices([])
        self.activated.connect(lambda i: self.edited.emit(self._ids[i]))

    def set_choices(self, items: list[tuple[object, str]]) -> None:
        """Offers these (value, name) after "None", choosing none of them."""
        with QSignalBlocker(self):
            self.clear()
            self._ids = [None, *(item_id for item_id, _ in items)]
            self.addItems(["None", *(name for _, name in items)])

    def show_value(self, value: object) -> None:
        """Shows a value, or None, from the model; MIXED leaves it blank."""
        index = self._ids.index(value) if value in self._ids else -1
        with QSignalBlocker(self):
            self.setCurrentIndex(index)


class PathField(QWidget):
    """Edits a file name in a text field, with a button that opens a file chooser."""

    edited = Signal(object)

    def __init__(self, info: OptionInfo, parent: QWidget | None = None) -> None:
        """Creates the field; an empty value reads as the built-in default."""
        super().__init__(parent)
        self.label = info.label
        self.text = QLineEdit(self)
        self.text.setPlaceholderText("built-in")
        self.button = QPushButton("Browse...", self)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.text, 1)
        layout.addWidget(self.button)

        self.text.editingFinished.connect(lambda: self.edited.emit(self.text.text()))
        self.button.clicked.connect(self._browse)

    def _browse(self) -> None:
        """Asks for a file and sends it as the new value."""
        path, _ = QFileDialog.getOpenFileName(self, self.label)
        if path:  # empty when the user cancels
            self.text.setText(path)
            self.edited.emit(path)

    def show_value(self, value: object) -> None:
        """Shows a value from the model; MIXED leaves the field empty."""
        self.text.setPlaceholderText("mixed" if value is MIXED else "built-in")
        self.text.setText("" if value is MIXED else str(value))


class ColourField(QPushButton):
    """Edits a colour with a button filled with it, which opens a colour chooser."""

    edited = Signal(object)

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        """Creates the field."""
        super().__init__(parent)
        self.setObjectName("colourField")
        self.label = label
        self.colour = ""
        self.clicked.connect(self._choose)

    def _choose(self) -> None:
        """Asks for a colour and sends it as the new value."""
        chosen = QColorDialog.getColor(QColor(self.colour), self, self.label)
        if chosen.isValid():  # invalid when the user cancels
            self.edited.emit(chosen.name())

    def show_value(self, value: object) -> None:
        """Shows a colour from the model; MIXED shows as a plain button."""
        if value is MIXED:
            self.colour = ""
            self.setText("mixed")
            self.setStyleSheet("")
            return
        self.colour = str(value)
        self.setText(self.colour)
        text = "#000000" if QColor(self.colour).lightness() > 127 else "#ffffff"
        self.setStyleSheet(f"background-color: {self.colour}; color: {text}")


type OptionField = (
    CheckField | ChoiceField | ColourField | ItemField | PathField | SliderField
)


def field_for(
    f: Field[Any],
    info: OptionInfo,
    parent: QWidget,
    bounds: tuple[float, float] | None = None,
) -> OptionField:
    """Creates the field that edits an option, chosen by its choices or its default.

    Returns:
        OptionField: The field.

    Raises:
        TypeError: If no field edits the option's type.
    """
    if info.choices is not None:
        return ChoiceField(info.choices, parent)
    if info.item_ref or info.data_key:
        return ItemField(parent)  # the panel gives it the names it can take
    if info.path:
        return PathField(info, parent)
    if isinstance(f.default, Colour):
        return ColourField(info.label, parent)
    if isinstance(f.default, bool):
        return CheckField(parent)
    if isinstance(f.default, (float, int)):
        return SliderField(info, parent, bounds, field_type=type(f.default))
    kind = type(f.default).__name__
    raise TypeError(f"No field for option {f.name} of type {kind}.")


class FieldLabel(QLabel):
    """Shows a field's label; with help, as a tooltip and a muted info icon after it."""

    GAP = 4  # pixels between the text and the icon

    def __init__(
        self, text: str, help: str = "", parent: QWidget | None = None
    ) -> None:
        """Creates the label; with no help, a plain label."""
        super().__init__(text, parent)
        self.icon: QLabel | None = None
        if help:
            self.setToolTip(help)
            self.icon = QLabel(self)
            self.icon.setObjectName("helpIcon")
            self.icon.setToolTip(help)
            row = QHBoxLayout(self)  # the icon after the text
            row.setSpacing(0)
            row.addWidget(self.icon, 0, Qt.AlignmentFlag.AlignVCenter)
            row.addStretch()
            self._draw_icon()

    @override
    def sizeHint(self) -> QSize:
        """Returns the text's size, with room for the icon after it."""
        # The minimum size stays the text's: in a narrow panel the icon is cut first.
        hint = super().sizeHint()
        if self.icon is not None:
            hint.setWidth(hint.width() + self.GAP + self.icon.width())
        return hint

    @override
    def changeEvent(self, event: QEvent) -> None:
        """Redraws the icon for a new font or theme."""
        super().changeEvent(event)
        kinds = (QEvent.Type.FontChange, QEvent.Type.StyleChange)
        if self.icon is not None and event.type() in kinds:
            self._draw_icon()

    def _draw_icon(self) -> None:
        """Draws the info icon at the text's height, just after the text."""
        assert self.icon is not None
        metrics = self.fontMetrics()
        size = metrics.height() - 2  # about the text's height
        pixmap = QIcon(icon_path("info")).pixmap(
            QSize(size, size), self.devicePixelRatioF()
        )
        self.icon.setPixmap(pixmap)
        self.icon.setFixedSize(size, size)
        left = metrics.horizontalAdvance(self.text()) + self.GAP
        row = self.layout()
        assert row is not None
        row.setContentsMargins(left, 0, 0, 0)


def add_field_row(
    layout: QFormLayout, label: str, help: str, field: QWidget, parent: QWidget
) -> None:
    """Adds a row of a field and its label; help is their tooltip, if any."""
    if help:
        field.setToolTip(help)
    layout.addRow(FieldLabel(label, help, parent), field)
