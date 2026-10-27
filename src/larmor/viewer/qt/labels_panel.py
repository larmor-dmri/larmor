"""The labels list, which chooses the labels of a parcellation the 3D view draws."""

from typing import override

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QCloseEvent, QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from larmor.viewer.model.events import DataChanged, Event, ItemRemoved, OptionChanged
from larmor.viewer.model.item import ItemId
from larmor.viewer.model.labels import label_name, label_rgb, lookup_table
from larmor.viewer.model.options import OptionName, OverlayOptions
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.qt.docks import PanelContent

SWATCH = 12  # px, the side of a label's colour square
LABEL = Qt.ItemDataRole.UserRole  # where a row keeps its label's number
# Options that change which labels are listed, or their names and colours.
REFILLING = (OptionName.VOLUME_INDEX, OptionName.LOOKUP_TABLE)


class LabelsList(PanelContent):
    """Lists an overlay's labels, each with a box that shows it in 3D.

    The search keeps the rows whose number or name holds the text; All and None
    show or hide the rows the search keeps. It closes when its item is removed;
    paused, it only notes the options that changed until it resumes.
    """

    closed = Signal(int)  # the item's id

    def __init__(
        self, viewer: Viewer, item_id: ItemId, parent: QWidget | None = None
    ) -> None:
        """Lists the labels in the overlay's shown volume."""
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.viewer = viewer
        self.item_id = item_id
        self.title = QLabel(viewer.scene.item(item_id).name, self)
        self.title.setObjectName("formTitle")
        self._paused = False
        self._missed: set[OptionName] = set()  # options changed while paused

        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Search by number or name")
        self.search.setClearButtonEnabled(True)
        self.list = QListWidget(self)
        self.all = QPushButton("All", self)
        self.all.setToolTip("Show the labels listed")
        self.none = QPushButton("None", self)
        self.none.setToolTip("Hide the labels listed")
        self.count = QLabel(self)

        buttons = QHBoxLayout()
        buttons.addWidget(self.count, stretch=1)
        buttons.addWidget(self.all)
        buttons.addWidget(self.none)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addWidget(self.search)
        layout.addWidget(self.list, stretch=1)
        layout.addLayout(buttons)

        self.search.textChanged.connect(self._filter)
        self.list.itemChanged.connect(self._toggled)
        self.all.clicked.connect(lambda: self._show_listed(True))
        self.none.clicked.connect(lambda: self._show_listed(False))
        # The viewer can outlive the list: stop listening when it closes, or when
        # it is deleted with its parent, which sends no close event.
        stop = viewer.scene.subscribe(self._on_event)
        stopped = [False]

        def unsubscribe() -> None:
            if not stopped[0]:
                stopped[0] = True
                stop()

        self._unsubscribe = unsubscribe
        self.destroyed.connect(lambda *_: unsubscribe())  # not `self` in the lambda
        self._fill()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Stops following the scene, then closes."""
        self._unsubscribe()
        self.closed.emit(self.item_id)
        super().closeEvent(event)

    @override
    def pause(self) -> None:
        """Only notes the options that change until it resumes."""
        self._paused = True

    @override
    def resume(self) -> None:
        """Lists the labels or checks the rows again if their options changed."""
        self._paused = False
        missed, self._missed = self._missed, set()
        if missed & set(REFILLING):
            self._fill()
        elif OptionName.SURFACE_LABELS in missed:
            self._show_checks()

    def _options(self) -> OverlayOptions:
        """Returns the overlay's options."""
        o = self.viewer.scene.item(self.item_id).options
        assert isinstance(o, OverlayOptions)
        return o

    def _on_event(self, event: Event) -> None:
        """Closes, lists the labels again or checks the rows, as the event requires."""
        if isinstance(event, ItemRemoved) and event.item.id == self.item_id:
            self.close()
        elif isinstance(event, DataChanged) and event.item.id == self.item_id:
            if self._paused:
                self._missed.add(OptionName.VOLUME_INDEX)  # refilled on resume
            else:
                self._fill()  # the new data's labels
        elif isinstance(event, OptionChanged) and event.item.id == self.item_id:
            if self._paused:
                self._missed.add(event.name)
            elif event.name in REFILLING:
                self._fill()  # other labels, or other names and colours
            elif event.name == OptionName.SURFACE_LABELS:
                self._show_checks()

    def _fill(self) -> None:
        """Lists the labels, with their names and colours, then their boxes."""
        table = lookup_table(self._options().lookup_table)
        with QSignalBlocker(self.list):
            self.list.clear()
            for label in self.viewer.labels(self.item_id):
                row = QListWidgetItem(f"{label}  {label_name(label, table)}")
                row.setData(LABEL, label)
                row.setIcon(_swatch(label_rgb(label, table)))
                row.setFlags(row.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                row.setCheckState(Qt.CheckState.Unchecked)
                self.list.addItem(row)
        self._filter(self.search.text())
        self._show_checks()

    def _rows(self) -> list[QListWidgetItem]:
        """Returns the list's rows."""
        return [self.list.item(i) for i in range(self.list.count())]

    def _show_checks(self) -> None:
        """Checks the rows of the labels shown, as the option holds them."""
        shown = set(self._options().surface_labels)
        with QSignalBlocker(self.list):
            for row in self._rows():
                on = row.data(LABEL) in shown
                row.setCheckState(
                    Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
                )
        self.count.setText(f"{len(shown)} of {self.list.count()} shown")

    def _filter(self, text: str) -> None:
        """Hides the rows whose number and name do not hold the text."""
        words = text.strip().lower()
        for row in self._rows():
            row.setHidden(words not in row.text().lower())

    def _toggled(self, row: QListWidgetItem) -> None:
        """Shows or hides a row's label in 3D, as its box was checked."""
        label = row.data(LABEL)
        shown = set(self._options().surface_labels)
        if row.checkState() == Qt.CheckState.Checked:
            shown.add(label)
        else:
            shown.discard(label)
        self.viewer.set(self.item_id, OptionName.SURFACE_LABELS, shown)

    def _show_listed(self, on: bool) -> None:
        """Shows or hides the labels of the rows the search keeps."""
        listed = {row.data(LABEL) for row in self._rows() if not row.isHidden()}
        shown = set(self._options().surface_labels)
        self.viewer.set(
            self.item_id,
            OptionName.SURFACE_LABELS,
            shown | listed if on else shown - listed,
        )


class LabelsButton(QPushButton):
    """Lists an overlay's labels in the inspector; enabled while it shows labels."""

    def __init__(self, viewer: Viewer, item_id: ItemId, parent: QWidget) -> None:
        """Creates the button, which follows the overlay's Labels option."""
        super().__init__("Choose labels", parent)
        self.setToolTip("Choose the labels drawn as surfaces in the 3D view")
        scene = viewer.scene

        def enable(event: Event | None = None) -> None:
            is_labels = isinstance(event, OptionChanged) and (
                event.item.id == item_id and event.name == OptionName.LABELS
            )
            if event is None or is_labels:
                o = scene.item(item_id).options
                self.setEnabled(isinstance(o, OverlayOptions) and o.labels)

        unsubscribe = scene.subscribe(enable)
        self.destroyed.connect(lambda *_: unsubscribe())
        enable()


def _swatch(rgb: tuple[float, float, float]) -> QIcon:
    """Returns a square of a label's colour."""
    pixmap = QPixmap(SWATCH, SWATCH)
    pixmap.fill(QColor.fromRgbF(*rgb))
    return QIcon(pixmap)
