"""The list of regions a tractogram applies, at the end of its Regions group."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from larmor.viewer.model.events import Event, ItemAdded, ItemRemoved, OptionChanged
from larmor.viewer.model.item import Category, ItemId
from larmor.viewer.model.options import OptionError, OptionName, RegionOptions
from larmor.viewer.model.viewer import Viewer

REGION = Qt.ItemDataRole.UserRole  # where a row keeps its region's id
DOT = 10  # px, the colour dot before a region's name
# The options of a region its row shows.
SHOWN = (OptionName.ROLE, OptionName.COLOUR)
# The file types "Save kept..." offers.
TRACTOGRAM_FILTER = "Tractograms (*.tck *.trk *.vtk *.fib)"


class RegionsEditor(QWidget):
    """Lists every region of the scene with a tick box: ticked ones filter this item.

    Regions are added with the views' tools and right-click menu; "Save kept..."
    writes the streamlines the ticked regions keep.
    """

    def __init__(
        self, viewer: Viewer, item_id: ItemId, parent: QWidget | None = None
    ) -> None:
        """Lists the scene's regions, ticked where the tractogram applies them."""
        super().__init__(parent)
        self.viewer = viewer
        self.item_id = item_id
        self.list = QListWidget(self)
        self.list.setMaximumHeight(130)
        self.empty = QLabel("Add regions with the Sphere (O) and Box (B) tools.", self)
        self.empty.setWordWrap(True)
        self.save_button = QPushButton("Save kept...", self)
        self.save_button.setToolTip(
            "Write the streamlines the ticked regions keep to a tractogram file"
        )
        self.error = QLabel(self)
        self.error.setObjectName("fieldError")
        self.error.setWordWrap(True)
        self.error.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.list)
        layout.addWidget(self.empty)
        layout.addWidget(self.save_button)
        layout.addWidget(self.error)
        self.list.itemChanged.connect(self._ticked)
        self.save_button.clicked.connect(self._ask_save)
        # The viewer can outlive the editor: stop listening when it goes.
        unsubscribe = viewer.scene.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: unsubscribe())
        self._fill()

    def _on_event(self, event: Event) -> None:
        """Lists the regions again when one comes, goes or changes, or is ticked."""
        if isinstance(event, (ItemAdded, ItemRemoved)):
            if event.item.category == Category.REGIONS:
                self._fill()
        elif isinstance(event, OptionChanged) and (
            (event.item.id == self.item_id and event.name == OptionName.REGIONS)
            or (event.item.category == Category.REGIONS and event.name in SHOWN)
        ):
            self._fill()

    def _fill(self) -> None:
        """Lists every region, with its role and colour, ticked if applied."""
        applied = self.viewer.scene.item(self.item_id).options
        regions = self.viewer.scene.in_category(Category.REGIONS)
        self.list.blockSignals(True)
        self.list.clear()
        for region in regions:
            o = region.options
            assert isinstance(o, RegionOptions)
            row = QListWidgetItem(_dot(o.colour), f"{region.name} ({o.role})")
            row.setData(REGION, region.id)
            row.setFlags(row.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            ticked = region.id in getattr(applied, OptionName.REGIONS)
            row.setCheckState(
                Qt.CheckState.Checked if ticked else Qt.CheckState.Unchecked
            )
            self.list.addItem(row)
        self.list.blockSignals(False)
        self.list.setVisible(bool(regions))
        self.empty.setVisible(not regions)

    def _ticked(self, row: QListWidgetItem) -> None:
        """Applies or stops applying the row's region."""
        applied = row.checkState() == Qt.CheckState.Checked
        try:
            self.viewer.apply_region(self.item_id, row.data(REGION), applied)
        except OptionError as error:
            self._show_error(str(error))
            return
        self.error.hide()

    def _ask_save(self) -> None:
        """Asks for a file and writes the kept streamlines to it."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Kept Streamlines", "", TRACTOGRAM_FILTER
        )
        if path:
            self.save(path)

    def save(self, path: str) -> None:
        """Writes the kept streamlines to a file; shows the error below, if any."""
        try:
            self.viewer.save_kept(self.item_id, path)
        except (OSError, ValueError) as error:
            self._show_error(str(error))
            return
        self.error.hide()

    def _show_error(self, text: str) -> None:
        """Shows an error under the list."""
        self.error.setText(text)
        self.error.show()


def _dot(colour: str) -> QIcon:
    """Returns a round dot of a colour, as a row's icon."""
    pixmap = QPixmap(DOT, DOT)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(colour))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(0, 0, DOT, DOT)
    painter.end()
    return QIcon(pixmap)
