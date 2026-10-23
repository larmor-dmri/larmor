"""The editor of the regions a tractogram's streamlines are filtered by."""

from collections.abc import Callable
from functools import partial

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from larmor.viewer.model.events import (
    Event,
    ItemAdded,
    ItemRemoved,
    OptionChanged,
)
from larmor.viewer.model.item import ItemId
from larmor.viewer.model.options import OptionError, OptionName, RegionRole
from larmor.viewer.model.viewer import REGION_LISTS, Viewer

ENTRY = Qt.ItemDataRole.UserRole  # where a row keeps its (option, index)
SPHERE_RADIUS = 5.0  # mm, the radius a new sphere starts with
MASKS = (OptionName.INCLUDE_MASKS, OptionName.EXCLUDE_MASKS, OptionName.WITHIN_MASKS)
# The kind each list's regions are shown as, in the order the list shows them.
KINDS = {
    OptionName.INCLUDE_MASKS: "Include",
    OptionName.INCLUDE_SPHERES: "Include",
    OptionName.EXCLUDE_MASKS: "Exclude",
    OptionName.EXCLUDE_SPHERES: "Exclude",
    OptionName.WITHIN_MASKS: "Within",
    OptionName.WITHIN_SPHERES: "Within",
}
# The rows of the kind choice.
EXCLUDE_ROW = 1
WITHIN_ROW = 2


class RegionsEditor(QWidget):
    """Lists and edits a tractogram's include, exclude and within regions.

    A mask is chosen among the images loaded; a sphere is put at the cursor
    with the radius given. Each is added as an include, exclude or within region.
    """

    def __init__(
        self, viewer: Viewer, item_id: ItemId, parent: QWidget | None = None
    ) -> None:
        """Lists the item's regions."""
        super().__init__(parent)
        self.viewer = viewer
        self.item_id = item_id
        self._create_widgets()
        self._build_layout()
        self._connect()
        # The viewer can outlive the editor: stop listening when it goes.
        unsubscribe = viewer.scene.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: unsubscribe())
        self._fill_masks()
        self._fill()

    def _create_widgets(self) -> None:
        """Creates the list, the buttons, the kind and mask choices and the radius."""
        self.list = QListWidget(self)
        self.list.setMaximumHeight(110)
        self.remove_button = QPushButton("Remove", self)
        self.remove_button.setToolTip("Remove the regions chosen in the list")
        self.clear_button = QPushButton("Clear", self)
        self.clear_button.setToolTip("Remove every region: all streamlines are drawn")
        self.kind_choice = QComboBox(self)
        self.kind_choice.addItems(["Include", "Exclude", "Within"])
        self.kind_choice.setToolTip(
            "Include keeps the streamlines that pass through the region; exclude "
            "drops them; within keeps those that stay entirely inside it."
        )
        self.mask_choice = QComboBox(self)
        self.mask_choice.setToolTip("A loaded image of one volume, on any grid")
        self.add_mask = QPushButton("Add mask", self)
        self.radius = QDoubleSpinBox(self)
        self.radius.setRange(0.1, 100.0)
        self.radius.setSingleStep(0.5)
        self.radius.setValue(SPHERE_RADIUS)
        self.radius.setSuffix(" mm")
        self.radius.setToolTip("The radius of the next sphere")
        self.add_sphere = QPushButton("Add sphere at cursor", self)
        self.error = QLabel(self)
        self.error.setObjectName("fieldError")
        self.error.setWordWrap(True)
        self.error.hide()

    def _build_layout(self) -> None:
        """Lays out the list, the buttons and the choices."""
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(self.remove_button, 0, 0)
        grid.addWidget(self.clear_button, 0, 1)
        grid.addWidget(QLabel("Add as", self), 1, 0)
        grid.addWidget(self.kind_choice, 1, 1)
        grid.addWidget(self.mask_choice, 2, 0)
        grid.addWidget(self.add_mask, 2, 1)
        grid.addWidget(self.radius, 3, 0)
        grid.addWidget(self.add_sphere, 3, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.list)
        layout.addLayout(grid)
        layout.addWidget(self.error)

    def _connect(self) -> None:
        """Connects the buttons to the viewer's actions."""
        viewer, item_id = self.viewer, self.item_id
        self.remove_button.clicked.connect(self._remove_chosen)
        self.clear_button.clicked.connect(
            lambda: self._act(partial(viewer.clear_regions, item_id))
        )
        self.add_mask.clicked.connect(self._add_mask)
        self.add_sphere.clicked.connect(
            lambda: self._act(
                lambda: viewer.add_sphere(
                    item_id,
                    self.radius.value(),
                    role=self._role(),
                )
            )
        )

    def _on_event(self, event: Event) -> None:
        """Updates the mask choices or the list after a scene event about them."""
        if isinstance(event, (ItemAdded, ItemRemoved)):
            if event.item.id != self.item_id:
                self._fill_masks()
        elif (
            isinstance(event, OptionChanged)
            and event.item.id == self.item_id
            and event.name in REGION_LISTS
        ):
            self._fill()

    def _role(self) -> RegionRole:
        """Returns the role chosen for new regions."""
        row = self.kind_choice.currentIndex()
        if row == EXCLUDE_ROW:
            return RegionRole.EXCLUDE
        return RegionRole.WITHIN if row == WITHIN_ROW else RegionRole.INCLUDE

    def _act(self, action: Callable[[], None]) -> None:
        """Runs an action of the viewer; shows its error, if any, below."""
        try:
            action()
        except OptionError as error:
            self.error.setText(str(error))
            self.error.show()
            return
        self.error.hide()

    def _add_mask(self) -> None:
        """Adds the chosen mask as a region of the chosen role."""
        mask_id = self.mask_choice.currentData()
        if mask_id is None:
            return
        self._act(
            partial(
                self.viewer.add_mask,
                self.item_id,
                mask_id,
                role=self._role(),
            )
        )

    def _remove_chosen(self) -> None:
        """Removes the regions chosen in the list, all at once per option."""
        chosen: dict[OptionName, set[int]] = {}
        for row in self.list.selectedItems():
            name, index = row.data(ENTRY)
            chosen.setdefault(name, set()).add(index)
        o = self.viewer.scene.item(self.item_id).options
        for name, indices in chosen.items():
            entries = getattr(o, name)
            kept = tuple(e for i, e in enumerate(entries) if i not in indices)
            self._act(partial(self.viewer.set, self.item_id, name, kept))

    def _fill_masks(self) -> None:
        """Offers the images a mask can be chosen from."""
        choices = self.viewer.item_choices(self.item_id, OptionName.INCLUDE_MASKS)
        self.mask_choice.clear()
        for item in choices:
            self.mask_choice.addItem(item.name, item.id)
        self.add_mask.setEnabled(bool(choices))

    def _fill(self) -> None:
        """Lists the regions: include ones first, then exclude and within ones."""
        o = self.viewer.scene.item(self.item_id).options
        self.list.clear()
        for name, kind in KINDS.items():
            for index, entry in enumerate(getattr(o, name)):
                if name in MASKS:
                    text = f"{kind}: {self.viewer.scene.item(entry).name}"
                else:
                    x, y, z, radius = entry
                    text = f"{kind}: sphere ({x:g}, {y:g}, {z:g}), {radius:g} mm"
                row = QListWidgetItem(text)
                row.setData(ENTRY, (name, index))
                self.list.addItem(row)
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.clear_button.setEnabled(self.list.count() > 0)
