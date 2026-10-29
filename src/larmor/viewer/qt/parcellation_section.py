"""The Parcellation section, which edits a label region of a parcellation."""

from typing import override

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from larmor.types import Volume
from larmor.viewer.model.events import Event, ItemRemoved, OptionChanged
from larmor.viewer.model.item import Category, ItemId
from larmor.viewer.model.labels import label_name, labels_in, lookup_table
from larmor.viewer.model.options import (
    LabelMode,
    LabelSet,
    OptionError,
    OptionName,
    OverlayOptions,
    RegionOptions,
    RegionShape,
)
from larmor.viewer.model.viewer import Viewer

LABEL = Qt.ItemDataRole.UserRole  # where a row of the label list keeps its label
MODE_NAMES = {
    LabelMode.FROM_A_TO_B: "From A to B",
    LabelMode.STAY_WITHIN: "Stay within",
}
# The options of a label region the section shows.
SHOWN = (
    OptionName.LABEL_MODE,
    OptionName.LABELS_A,
    OptionName.LABELS_B,
    OptionName.END_RADIUS,
)
SET_OPTIONS = {LabelSet.A: OptionName.LABELS_A, LabelSet.B: OptionName.LABELS_B}


class ParcellationSection(QWidget):
    """Edits the label region of one parcellation: its mode, label sets and radius.

    The section works on the parcellation's first label region, and adds one,
    applied to every tractogram, at the first change if there is none. Labels
    are added by name from a searchable list, or with the Label region tool in
    the set chosen as the one the tool edits; a chip's cross removes its label.
    """

    changed = Signal(int)  # the region's id, after its labels, mode or radius change
    editing_changed = Signal(object)  # the LabelSet the Label region tool edits

    def __init__(
        self, viewer: Viewer, parcellation_id: ItemId, parent: QWidget | None = None
    ) -> None:
        """Builds the section for a parcellation, a volume or overlay of one volume."""
        super().__init__(parent)
        self.viewer = viewer
        self.parcellation_id = parcellation_id
        self.region_id: ItemId | None = self._find_region()
        self.editing = LabelSet.A

        self.title = QLabel(viewer.scene.item(parcellation_id).name, self)
        self.title.setObjectName("formTitle")
        modes = self._build_modes()
        self._build_sets()
        self._build_radius()
        self.error = QLabel(self)
        self.error.setObjectName("fieldError")
        self.error.setWordWrap(True)
        self.error.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.title)
        layout.addLayout(modes)
        for row in self.sets.values():
            layout.addWidget(row)
        layout.addWidget(self.radius_row)
        layout.addWidget(self.error)
        # The viewer can outlive the section: stop listening when it goes.
        unsubscribe = viewer.scene.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: unsubscribe())
        self.refresh()

    def _build_modes(self) -> QHBoxLayout:
        """Builds the switch between From A to B and Stay within."""
        self.modes: dict[LabelMode, QToolButton] = {}
        modes = QHBoxLayout()
        modes.setSpacing(0)
        group = QButtonGroup(self)
        for mode, name in MODE_NAMES.items():
            button = QToolButton(self)
            button.setObjectName("segment")
            button.setText(name)
            button.setCheckable(True)
            button.clicked.connect(lambda _, m=mode: self.set_mode(m))
            group.addButton(button)
            modes.addWidget(button)
            self.modes[mode] = button
        modes.addStretch(1)
        return modes

    def _build_sets(self) -> None:
        """Builds the rows of sets A and B."""
        self.sets: dict[LabelSet, LabelSetRow] = {}
        edited = QButtonGroup(self)
        for which in LabelSet:
            row = LabelSetRow(which, self)
            row.add_button.clicked.connect(lambda _, w=which: self.ask_label(w))
            row.removed.connect(lambda label, w=which: self.remove_label(w, label))
            row.edit_button.clicked.connect(lambda _, w=which: self.set_editing(w))
            edited.addButton(row.edit_button)
            self.sets[which] = row

    def _build_radius(self) -> None:
        """Builds the End radius field."""
        self.radius = QDoubleSpinBox(self)
        self.radius.setRange(0.0, 50.0)
        self.radius.setSingleStep(0.5)
        self.radius.setSuffix(" mm")
        self.radius.setToolTip(
            "How far from a streamline's end a labelled voxel may be for the end "
            "to be in its label"
        )
        self.radius.editingFinished.connect(self._radius_edited)
        self.radius_row = QWidget(self)
        radius_layout = QHBoxLayout(self.radius_row)
        radius_layout.setContentsMargins(0, 0, 0, 0)
        radius_layout.addWidget(QLabel("End radius", self.radius_row))
        radius_layout.addWidget(self.radius)
        radius_layout.addStretch(1)

    # Reading

    def options(self) -> RegionOptions | None:
        """Returns the region's options; None while there is no region."""
        if self.region_id is None:
            return None
        o = self.viewer.scene.item(self.region_id).options
        assert isinstance(o, RegionOptions)
        return o

    def label_names(self) -> dict[int, str]:
        """Returns the parcellation's labels with their names, in label order."""
        item = self.viewer.scene.item(self.parcellation_id)
        o = item.options
        table = lookup_table(o.lookup_table) if isinstance(o, OverlayOptions) else {}
        if item.category == Category.OVERLAYS:
            labels = self.viewer.labels(item.id)
        else:
            assert isinstance(item.data, Volume)
            labels = labels_in(item.data, 0)
        return {label: label_name(label, table) for label in labels}

    # Editing

    def ensure_region(self) -> ItemId:
        """Returns the region's id, adding the region if there is none yet."""
        if self.region_id is None:
            self.region_id = self.viewer.add_label_region(
                self.parcellation_id, apply=True
            )
            self.viewer.edit_labels(self.region_id, self.editing)
        return self.region_id

    def set_mode(self, mode: LabelMode) -> None:
        """Sets what the region keeps; stay within edits set A."""
        self._set(OptionName.LABEL_MODE, mode)
        if mode == LabelMode.STAY_WITHIN:
            self.set_editing(LabelSet.A)

    def set_editing(self, which: LabelSet) -> None:
        """Chooses the set the Label region tool adds to and removes from."""
        self.editing = which
        self.viewer.edit_labels(self.ensure_region(), which)
        self.refresh()
        self.editing_changed.emit(which)

    def add_label(self, which: LabelSet, label: int) -> None:
        """Adds a label to a set."""
        o = self.options()
        current = () if o is None else getattr(o, SET_OPTIONS[which])
        self._set(SET_OPTIONS[which], (*current, label))

    def remove_label(self, which: LabelSet, label: int) -> None:
        """Removes a label from a set."""
        o = self.options()
        if o is not None:
            kept = tuple(v for v in getattr(o, SET_OPTIONS[which]) if v != label)
            self._set(SET_OPTIONS[which], kept)

    def ask_label(self, which: LabelSet) -> None:
        """Opens the searchable list of labels and adds the one chosen to a set."""
        picker = LabelPicker(self.label_names(), self)
        button = self.sets[which].add_button
        picker.move(button.mapToGlobal(QPoint(0, button.height())))
        if picker.exec() == QDialog.DialogCode.Accepted and picker.label is not None:
            self.add_label(which, picker.label)

    def _radius_edited(self) -> None:
        """Sets the end radius typed."""
        o = self.options()
        if o is None or self.radius.value() != o.end_radius:
            self._set(OptionName.END_RADIUS, self.radius.value())

    def _set(self, name: OptionName, value: object) -> None:
        """Sets an option of the region, adding it first if needed; shows a refusal."""
        region = self.ensure_region()
        try:
            self.viewer.set(region, name, value)
        except (OptionError, ValueError) as e:
            self.error.setText(str(e))
            self.error.show()
            return
        self.error.hide()

    # Following the scene

    def _find_region(self) -> ItemId | None:
        """Returns the first label region on the parcellation, if any."""
        for region in self.viewer.scene.in_category(Category.REGIONS):
            o = region.options
            if (
                isinstance(o, RegionOptions)
                and o.shape == RegionShape.LABELS
                and o.parcellation == self.parcellation_id
            ):
                return region.id
        return None

    def _on_event(self, event: Event) -> None:
        """Shows the region again when it changes or goes."""
        if isinstance(event, ItemRemoved):
            if event.item.id == self.region_id:
                self.region_id = None
                self.refresh()
            elif event.item.id == self.parcellation_id:
                self.setEnabled(False)
        elif (
            isinstance(event, OptionChanged)
            and event.item.id == self.region_id
            and event.name in SHOWN
        ):
            self.refresh()
            self.changed.emit(event.item.id)

    def refresh(self) -> None:
        """Shows the region's mode, sets, edited set and radius."""
        o = self.options()
        mode = LabelMode.FROM_A_TO_B if o is None else o.label_mode
        for m, button in self.modes.items():
            button.setChecked(m == mode)
        names = self.label_names() if self.isEnabled() else {}
        for which, row in self.sets.items():
            labels = () if o is None else getattr(o, SET_OPTIONS[which])
            row.show_labels([(v, names.get(v, f"Label {v}")) for v in labels])
            row.edit_button.setChecked(which == self.editing)
        within = mode == LabelMode.STAY_WITHIN
        self.sets[LabelSet.A].set_title("Labels" if within else "A")
        self.sets[LabelSet.B].setVisible(not within)
        self.radius_row.setVisible(not within)
        self.radius.setValue(4.0 if o is None else o.end_radius)


class LabelSetRow(QWidget):
    """One label set: its title, its labels as removable chips, and "+ label"."""

    removed = Signal(int)  # the label whose chip's cross was clicked

    def __init__(self, which: LabelSet, parent: QWidget | None = None) -> None:
        """Builds an empty row for set A or B."""
        super().__init__(parent)
        self.which = which
        self.title = QLabel(which.value.upper(), self)
        self.edit_button = QToolButton(self)
        self.edit_button.setObjectName("segment")
        self.edit_button.setText("L")
        self.edit_button.setCheckable(True)
        self.edit_button.setToolTip("Add and remove this set's labels with the L tool")
        self.add_button = QToolButton(self)
        self.add_button.setText("+ label")
        self.add_button.setToolTip("Add a label by name")
        self.chips: dict[int, QToolButton] = {}
        self.flow = FlowLayout()
        header = QHBoxLayout()
        header.addWidget(self.title)
        header.addStretch(1)
        header.addWidget(self.edit_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(header)
        layout.addLayout(self.flow)

    def set_title(self, text: str) -> None:
        """Names the row."""
        self.title.setText(text)

    def show_labels(self, labels: list[tuple[int, str]]) -> None:
        """Shows a chip per label, then "+ label"."""
        while self.flow.count():
            item = self.flow.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None and widget is not self.add_button:
                widget.deleteLater()
        self.chips = {}
        for label, name in labels:
            chip = QToolButton(self)
            chip.setObjectName("labelChip")
            chip.setText(f"{name}  \N{MULTIPLICATION SIGN}")
            chip.setToolTip(f"Label {label}; click to remove it")
            chip.clicked.connect(lambda _, v=label: self.removed.emit(v))
            self.flow.addWidget(chip)
            self.chips[label] = chip
        self.flow.addWidget(self.add_button)


class LabelPicker(QDialog):
    """Lists labels by number and name, with a search; the one chosen is `label`."""

    def __init__(self, names: dict[int, str], parent: QWidget | None = None) -> None:
        """Lists the labels; a double click or Enter chooses one."""
        super().__init__(parent, Qt.WindowType.Popup)
        self.label: int | None = None
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Search by number or name")
        self.search.setClearButtonEnabled(True)
        self.list = QListWidget(self)
        for label, name in names.items():
            row = QListWidgetItem(f"{label}  {name}")
            row.setData(LABEL, label)
            self.list.addItem(row)
        layout = QVBoxLayout(self)
        layout.addWidget(self.search)
        layout.addWidget(self.list)
        self.resize(260, 320)
        self.search.textChanged.connect(self.filter)
        self.search.returnPressed.connect(self._choose_first)
        self.list.itemActivated.connect(self.choose)

    def filter(self, text: str) -> None:
        """Hides the rows whose number and name do not hold the text."""
        words = text.strip().lower()
        for i in range(self.list.count()):
            row = self.list.item(i)
            row.setHidden(words not in row.text().lower())

    def choose(self, row: QListWidgetItem) -> None:
        """Chooses a row's label and closes."""
        self.label = int(row.data(LABEL))
        self.accept()

    def _choose_first(self) -> None:
        """Chooses the first row the search keeps."""
        for i in range(self.list.count()):
            row = self.list.item(i)
            if not row.isHidden():
                self.choose(row)
                return


class FlowLayout(QLayout):
    """Lays out widgets in rows, starting a new row when one is full."""

    def __init__(self, parent: QWidget | None = None, spacing: int = 4) -> None:
        """Creates an empty layout."""
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    @override
    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802
        """Adds an item at the end."""
        self._items.append(item)

    @override
    def count(self) -> int:
        """Returns the number of items."""
        return len(self._items)

    @override
    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802
        """Returns an item, or None past the end."""
        return self._items[index] if 0 <= index < len(self._items) else None

    @override
    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802
        """Removes an item and returns it, or None past the end."""
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    @override
    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802
        """Grows in neither direction."""
        return Qt.Orientation(0)

    @override
    def hasHeightForWidth(self) -> bool:  # noqa: N802
        """Its height depends on its width."""
        return True

    @override
    def heightForWidth(self, width: int) -> int:  # noqa: N802
        """Returns the height the rows take at a width."""
        return self._place(QRect(0, 0, width, 0), move=False)

    @override
    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        """Places the items in rows within rect."""
        super().setGeometry(rect)
        self._place(rect, move=True)

    @override
    def sizeHint(self) -> QSize:  # noqa: N802
        """Returns the smallest size."""
        return self.minimumSize()

    @override
    def minimumSize(self) -> QSize:  # noqa: N802
        """Returns the size of the largest item."""
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def _place(self, rect: QRect, move: bool) -> int:
        """Places the items in rows (if move) and returns the height they take."""
        x, y, row_height = rect.x(), rect.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            if x + hint.width() > rect.right() + 1 and row_height > 0:
                x, y = rect.x(), y + row_height + self._spacing
                row_height = 0
            if move:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._spacing
            row_height = max(row_height, hint.height())
        return y + row_height - rect.y()
