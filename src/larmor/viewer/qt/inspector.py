"""The inspector, with tabs for the selected item, the cursor and the scene."""

from collections.abc import Mapping
from enum import StrEnum

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from larmor.types import Volume
from larmor.viewer.model.camera import CameraPreset
from larmor.viewer.model.events import (
    DataChanged,
    Event,
    ItemAdded,
    ItemRemoved,
    ItemRenamed,
    OptionChanged,
    ViewStateChanged,
)
from larmor.viewer.model.item import Category, ViewKind
from larmor.viewer.model.labels import label_name, lookup_table
from larmor.viewer.model.options import (
    OptionGroup,
    OptionName,
    OverlayOptions,
    VolumeOptions,
)
from larmor.viewer.model.readout import read_volume
from larmor.viewer.model.settings import Settings
from larmor.viewer.model.view_state import Layout, ViewField
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.qt.docks import ItemPanel
from larmor.viewer.qt.fod_inspector_panel import FodInspectorView
from larmor.viewer.qt.labels_panel import LabelsList
from larmor.viewer.qt.options_panel import Extra, OptionsPanel
from larmor.viewer.qt.settings_panel import SCENE_SETTINGS, SettingsPanel

FOD_HEIGHT = 240  # px, the canvas of the FOD glyph at the cursor
# The layouts as the Scene tab and the toolbar name them.
LAYOUT_NAMES = {
    Layout.GRID: "2 x 2",
    Layout.ONE_PLUS_THREE: "One large and three small",
    Layout.ONE: "One view",
}
# The 3D camera's positions as the Scene tab's buttons name them.
CAMERA_NAMES = {
    CameraPreset.AXIAL: "Axial",
    CameraPreset.CORONAL: "Coronal",
    CameraPreset.SAGITTAL: "Sagittal",
}
# The options whose change can change the cursor's readout.
READOUT_OPTIONS = frozenset(
    {
        OptionName.VISIBLE,
        OptionName.VOLUME_INDEX,
        OptionName.INTERPOLATE,
        OptionName.LABELS,
        OptionName.LOOKUP_TABLE,
    }
)


class InspectorTab(StrEnum):
    """A tab of the inspector; each value is its title."""

    ITEM = "Item"
    CURSOR = "Cursor"
    SCENE = "Scene"


def layer_readout(viewer: Viewer) -> list[tuple[str, str]]:
    """Returns each visible item's name and what it shows at the cursor.

    An image gives its value, a parcellation its label and the label's name,
    and an image the cursor is outside of gives "outside". Other items give
    "" for no value.

    Returns:
        The names and texts, in the scene tree's order.
    """
    position = viewer.scene.view.cursor
    rows: list[tuple[str, str]] = []
    for item in viewer.scene.items:
        options = item.options
        if not options.visible:
            continue
        text = ""
        if isinstance(item.data, Volume) and isinstance(options, VolumeOptions):
            _, value = read_volume(
                item.data, position, options.volume_index, options.interpolate
            )
            if value is None:
                text = "outside"
            elif isinstance(options, OverlayOptions) and options.labels:
                _, label_value = read_volume(
                    item.data, position, options.volume_index, False
                )
                label = round(label_value or 0.0)
                table = lookup_table(options.lookup_table)
                text = f"{label} {label_name(label, table)}"
            else:
                text = f"{value:.4g}"
        rows.append((item.name, text))
    return rows


class CursorTab(QWidget):
    """Shows what every visible item holds at the cursor, an FOD and labels.

    The FOD glyph is the selected FOD's, or the first one's, and the labels are
    those of the selected overlay, or the first one, while it is shown as a
    parcellation.
    """

    def __init__(self, viewer: Viewer, parent: QWidget | None = None) -> None:
        """Builds the readout, the FOD glyph and the labels list."""
        super().__init__(parent)
        self.viewer = viewer
        self.readout = QTreeWidget(self)
        self.readout.setObjectName("cursorReadout")
        self.readout.setColumnCount(2)
        self.readout.setHeaderLabels(["Item", "At the cursor"])
        self.readout.setRootIsDecorated(False)
        self.position = QLabel(self)
        self.position.setObjectName("panelHint")
        self.fod = ItemPanel(
            viewer,
            Category.FODS,
            lambda item_id, panel: FodInspectorView(viewer, item_id, panel),
            "",
            self,
        )
        self.fod.setFixedHeight(FOD_HEIGHT)
        self.labels = ItemPanel(
            viewer,
            Category.OVERLAYS,
            lambda item_id, panel: LabelsList(viewer, item_id, panel),
            "",
            self,
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.position)
        layout.addWidget(self.readout, 1)
        layout.addWidget(self.fod)
        layout.addWidget(self.labels, 2)
        # The viewer can outlive the tab: stop listening when it goes.
        unsubscribe = viewer.scene.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: unsubscribe())
        self.refresh()

    def refresh(self) -> None:
        """Shows the readout, and the FOD and the labels when there are any."""
        x, y, z = self.viewer.scene.view.cursor
        self.position.setText(f"Cursor at ({x:.1f}, {y:.1f}, {z:.1f}) mm")
        self.readout.clear()
        for name, text in layer_readout(self.viewer):
            QTreeWidgetItem(self.readout, [name, text])
        self.readout.resizeColumnToContents(0)
        scene = self.viewer.scene
        self.fod.setVisible(bool(scene.in_category(Category.FODS)))
        shown = self.labels.item_id
        options = scene.item(shown).options if shown is not None else None
        self.labels.setVisible(isinstance(options, OverlayOptions) and options.labels)

    def _on_event(self, event: Event) -> None:
        """Shows the readout again after a change that can change it."""
        if isinstance(event, ViewStateChanged):
            if event.name in (ViewField.CURSOR, ViewField.SELECTED):
                self.refresh()
        elif isinstance(event, OptionChanged):
            if event.name in READOUT_OPTIONS:
                self.refresh()
        elif isinstance(event, (ItemAdded, ItemRemoved, DataChanged, ItemRenamed)):
            self.refresh()


class SceneTab(QScrollArea):
    """Edits how the views show the scene: convention, colours, camera, layout."""

    def __init__(
        self, viewer: Viewer, settings: Settings, parent: QWidget | None = None
    ) -> None:
        """Builds the scene's settings, the crosshair switch, cameras and layouts."""
        super().__init__(parent)
        self.viewer = viewer
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget(self)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        self.settings_form = SettingsPanel(settings, content, names=SCENE_SETTINGS)
        layout.addWidget(self.settings_form)

        self.crosshair = QCheckBox("Show the crosshair (X)", content)
        self.crosshair.clicked.connect(lambda _: viewer.toggle_crosshair())
        layout.addWidget(_section("Layout and camera", content))
        rows = QVBoxLayout()
        rows.setContentsMargins(14, 0, 14, 0)
        rows.addWidget(self.crosshair)
        self.layout_box = QComboBox(content)
        for value, name in LAYOUT_NAMES.items():
            self.layout_box.addItem(name, value)
        self.layout_box.activated.connect(
            lambda i: viewer.set_layout(Layout(self.layout_box.itemData(i)))
        )
        rows.addWidget(QLabel("Layout", content))
        rows.addWidget(self.layout_box)
        rows.addWidget(QLabel("3D camera", content))
        cameras = QHBoxLayout()
        self.camera_buttons: dict[CameraPreset | None, QPushButton] = {}
        for preset, name in CAMERA_NAMES.items():
            button = QPushButton(name, content)
            button.clicked.connect(
                lambda _, p=preset: viewer.set_camera(ViewKind.THREE_D, p)
            )
            self.camera_buttons[preset] = button
            cameras.addWidget(button)
        reset = QPushButton("Reset", content)
        reset.clicked.connect(lambda: viewer.reset_camera(ViewKind.THREE_D))
        self.camera_buttons[None] = reset
        cameras.addWidget(reset)
        rows.addLayout(cameras)
        layout.addLayout(rows)
        layout.addStretch(1)
        self.setWidget(content)
        unsubscribe = viewer.scene.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: unsubscribe())
        self._show_view()

    def _on_event(self, event: Event) -> None:
        """Shows the crosshair switch and the layout after they change."""
        if isinstance(event, ViewStateChanged) and event.name in (
            ViewField.CROSSHAIR_VISIBLE,
            ViewField.LAYOUT,
        ):
            self._show_view()

    def _show_view(self) -> None:
        """Shows the crosshair switch and the layout as the view state holds them."""
        state = self.viewer.scene.view
        self.crosshair.setChecked(state.crosshair_visible)
        self.layout_box.setCurrentIndex(self.layout_box.findData(state.layout))


class Inspector(QTabWidget):
    """Shows the Item, Cursor and Scene tabs on the right of the View workspace."""

    def __init__(
        self,
        viewer: Viewer,
        settings: Settings,
        parent: QWidget | None = None,
        extras: Mapping[Category, tuple[OptionGroup, Extra]] | None = None,
    ) -> None:
        """Creates the three tabs; extras end option groups, as in OptionsPanel."""
        super().__init__(parent)
        self.setObjectName("inspector")
        self.setDocumentMode(True)
        self.item = OptionsPanel(viewer, self, extras)
        self.cursor_tab = CursorTab(viewer, self)
        self.scene_tab = SceneTab(viewer, settings, self)
        self.tabs: dict[InspectorTab, QWidget] = {
            InspectorTab.ITEM: self.item,
            InspectorTab.CURSOR: self.cursor_tab,
            InspectorTab.SCENE: self.scene_tab,
        }
        for tab, widget in self.tabs.items():
            self.addTab(widget, tab.value)

    def show_tab(self, tab: InspectorTab) -> None:
        """Brings a tab to the front."""
        self.setCurrentWidget(self.tabs[tab])


def _section(text: str, parent: QWidget) -> QLabel:
    """Returns a section's heading, as the settings forms show them."""
    label = QLabel(text, parent)
    label.setObjectName("formSection")
    label.setContentsMargins(14, 8, 14, 0)
    return label
