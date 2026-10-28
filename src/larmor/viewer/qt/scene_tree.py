"""The scene tree, which lists the scene's items by category, and its panel."""

from pathlib import Path
from typing import override

from PySide6.QtCore import (
    QEvent,
    QMimeData,
    QModelIndex,
    QPoint,
    QSignalBlocker,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QDragEnterEvent,
    QDragLeaveEvent,
    QDragMoveEvent,
    QDropEvent,
    QIcon,
    QKeyEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QMenu,
    QMessageBox,
    QPushButton,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from larmor.types import Tractogram
from larmor.viewer.model.events import (
    DataChanged,
    Event,
    FilterChanged,
    ItemAdded,
    ItemRemoved,
    ItemRenamed,
    OptionChanged,
    SourceChanged,
)
from larmor.viewer.model.item import Category, Item, ItemId
from larmor.viewer.model.options import (
    FodOptions,
    OptionName,
    OverlayOptions,
    TractogramOptions,
)
from larmor.viewer.model.registry import REGISTRY
from larmor.viewer.model.viewer import SAVABLE, Viewer
from larmor.viewer.qt.style import icon_path

EYE_COLUMN = 0
NAME_COLUMN = 1
BADGE_COLUMN = 2
# The options whose change can change an item's badge.
BADGE_OPTIONS = frozenset(
    {
        OptionName.COLORMAP,
        OptionName.THRESHOLD,
        OptionName.LABELS,
        OptionName.BASIS,
        OptionName.REGION_FILTER,
        OptionName.ENDS_ONLY,
        OptionName.REGIONS,
    }
)
# The keys that remove the selected row; the Mac's delete key is Backspace.
REMOVE_KEYS = (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)


def _load_icons() -> tuple[QIcon, QIcon, QIcon]:
    """Loads the eyes, tinted for the theme: visible, hidden and mixed."""
    names = ("eye-open", "eye-closed", "eye-mixed")
    visible, invisible, mixed = (QIcon(icon_path(n)) for n in names)
    return visible, invisible, mixed


def badge(viewer: Viewer, item: Item) -> str:
    """Returns the short state a row shows after an item's name; "" for none.

    An overlay shows its colormap and threshold, or "labels"; an FOD its basis;
    a tractogram the streamlines its regions keep, when they keep fewer than all.
    """
    options = item.options
    if isinstance(options, OverlayOptions):
        if options.labels:
            return "labels"
        low = item.value_range[0] if item.value_range is not None else 0.0
        above = f" > {options.threshold:g}" if options.threshold > low else ""
        return f"{options.colormap.value}{above}"
    if isinstance(options, FodOptions):
        return options.basis.value
    if isinstance(options, TractogramOptions) and isinstance(item.data, Tractogram):
        try:
            kept = viewer.kept_streamlines(item.id)
        except ValueError:
            return ""
        total = item.data.num_streamlines
        return f"{kept:,} / {total:,}" if kept < total else ""
    return ""


class SceneTree(QTreeWidget):
    """Lists the scene's items, grouped by category; empty categories are hidden.

    While files are dragged over the tree, every category shows, so the files
    can be dropped onto the one that should read them.
    """

    # Files dropped onto a category: the paths (list[Path]) and the Category.
    files_dropped = Signal(list, object)
    # "Save As..." chosen in an item's context menu: the item's id.
    save_requested = Signal(int)
    # A row's name double-clicked.
    activated_name = Signal()

    def __init__(self, viewer: Viewer, parent: QWidget | None = None) -> None:
        """Creates the tree."""
        super().__init__(parent)
        self._ready = False  # style changes before the tree is built are ignored
        self._rows: dict[ItemId | Category, QTreeWidgetItem] = {}
        self.viewer = viewer
        self.dragging = False  # files are being dragged over the tree
        # The viewer can outlive the window: stop listening when the widget goes.
        unsubscribe = self.viewer.scene.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: unsubscribe())
        self.icon_visible, self.icon_invisible, self.icon_mixed = _load_icons()

        self.setObjectName("sceneTree")
        self.setHeaderHidden(True)
        self.setColumnCount(3)
        self.setTreePosition(NAME_COLUMN)  # the branches beside the names
        header = self.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(EYE_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(NAME_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(
            BADGE_COLUMN, QHeaderView.ResizeMode.ResizeToContents
        )
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self._drop_target: Category | None = None
        self.setExpandsOnDoubleClick(False)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.itemClicked.connect(self._on_clicked)
        self.itemDoubleClicked.connect(self._on_double_clicked)
        self.itemChanged.connect(self._on_renamed)
        self.currentItemChanged.connect(self._on_select)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)
        self._build_tree()
        self._update()
        self._ready = True

    @override
    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        """Tints the eyes again when the theme changes."""
        super().changeEvent(event)
        if event.type() == QEvent.Type.StyleChange and self._ready:
            self.icon_visible, self.icon_invisible, self.icon_mixed = _load_icons()
            self._update()

    def row(self, target: ItemId | Category) -> QTreeWidgetItem:
        """Returns the row of an item or category."""
        return self._rows[target]

    def _build_tree(self) -> None:
        """Rebuilds the rows of the categories and their items."""
        with QSignalBlocker(self):
            self.clear()
            self._rows.clear()
            self._drop_target = None  # new rows have no highlight
            for key, kind in REGISTRY.items():
                row = QTreeWidgetItem(self)
                row.setText(NAME_COLUMN, kind.label)
                row.setData(NAME_COLUMN, Qt.ItemDataRole.UserRole, key)
                self._rows[key] = row
                for item in self.viewer.scene.in_category(key):
                    item_row = QTreeWidgetItem(row)
                    item_row.setText(NAME_COLUMN, item.name)
                    item_row.setData(NAME_COLUMN, Qt.ItemDataRole.UserRole, item.id)
                    item_row.setToolTip(NAME_COLUMN, _source(item))
                    item_row.setFlags(item_row.flags() | Qt.ItemFlag.ItemIsEditable)
                    item_row.setText(BADGE_COLUMN, badge(self.viewer, item))
                    item_row.setForeground(
                        BADGE_COLUMN, QBrush(self.palette().placeholderText())
                    )
                    self._rows[item.id] = item_row
            self.expandAll()
            self._show_categories()

    def _show_categories(self) -> None:
        """Shows the categories with items, and every category during a drag."""
        for key, row in self._rows.items():
            if isinstance(key, Category):
                row.setHidden(not self.dragging and row.childCount() == 0)

    def _update(self) -> None:
        """Shows each row's visibility and the selection."""
        with QSignalBlocker(self):
            for target, row in self._rows.items():
                visible = self._visible(target)
                if isinstance(visible, bool):
                    icon = self.icon_visible if visible else self.icon_invisible
                    tip = "Visible" if visible else "Hidden"
                else:
                    icon, tip = self.icon_mixed, "Some visible"
                row.setIcon(EYE_COLUMN, icon)
                row.setToolTip(EYE_COLUMN, tip)
            selected = self.viewer.scene.view.selected
            if selected is None or selected not in self._rows:
                self.setCurrentIndex(QModelIndex())
            else:
                self.setCurrentItem(self._rows[selected], NAME_COLUMN)

    def refresh_badge(self, item_id: ItemId) -> None:
        """Shows an item's badge again, as after its regions kept other streamlines."""
        row = self._rows.get(item_id)
        if row is not None:
            with QSignalBlocker(self):
                item = self.viewer.scene.item(item_id)
                row.setText(BADGE_COLUMN, badge(self.viewer, item))

    def _on_event(self, event: Event) -> None:
        """Follows the items added, removed, renamed and changed, then the eyes."""
        if isinstance(event, (ItemAdded, ItemRemoved)):
            self._build_tree()
        elif isinstance(event, (DataChanged, SourceChanged)):
            self._rows[event.item.id].setToolTip(NAME_COLUMN, _source(event.item))
            self.refresh_badge(event.item.id)
        elif isinstance(event, ItemRenamed):
            with QSignalBlocker(self):
                self._rows[event.item.id].setText(NAME_COLUMN, event.item.name)
        elif isinstance(event, OptionChanged) and event.name in BADGE_OPTIONS:
            self.refresh_badge(event.item.id)
        elif isinstance(event, FilterChanged):
            self.refresh_badge(event.item.id)
        self._update()

    def _on_clicked(self, row: QTreeWidgetItem, column: int) -> None:
        """Shows or hides the row's item or category when its eye is clicked."""
        if column == EYE_COLUMN:
            target = _target(row)
            self.viewer.set(
                target, OptionName.VISIBLE, self._visible(target) is not True
            )

    def _on_double_clicked(self, row: QTreeWidgetItem, column: int) -> None:
        """Sends a double click on a name."""
        if column == NAME_COLUMN:
            self.activated_name.emit()

    def _on_renamed(self, row: QTreeWidgetItem, column: int) -> None:
        """Renames the item whose name was edited; an empty name is put back."""
        target = _target(row)
        if column != NAME_COLUMN or isinstance(target, Category):
            return
        try:
            self.viewer.rename(target, row.text(NAME_COLUMN))
        except ValueError:
            with QSignalBlocker(self):
                row.setText(NAME_COLUMN, self.viewer.scene.item(target).name)

    def rename(self, item_id: ItemId) -> None:
        """Starts editing an item's name in its row."""
        self.editItem(self._rows[item_id], NAME_COLUMN)

    def remove(self, target: ItemId | Category) -> None:
        """Removes an item, or a category's items after asking if there are several."""
        if isinstance(target, Category):
            n = len(self.viewer.scene.in_category(target))
            label = REGISTRY[target].label
            if n > 1 and not self._confirm_remove(f"Remove all {n} {label}?"):
                return
        self.viewer.remove(target)

    def remove_selected(self) -> None:
        """Removes the selected items, after asking if there are several.

        With one row selected, its item or category is removed as `remove` does.
        """
        targets = [_target(row) for row in self.selectedItems()]
        items = [t for t in targets if not isinstance(t, Category)]
        if len(items) > 1:
            if self._confirm_remove(f"Remove the {len(items)} selected items?"):
                for item_id in items:
                    self.viewer.remove(item_id)
            return
        current = self.currentItem()
        if current is not None:
            self.remove(_target(current))

    @override
    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        """Removes the selected items or category with Delete or Backspace."""
        if (
            event.key() in REMOVE_KEYS
            and self.state() != QAbstractItemView.State.EditingState
        ):
            self.remove_selected()
            return
        super().keyPressEvent(event)

    def context_menu(self, item_id: ItemId) -> QMenu:
        """Returns an item's context menu: Remove, Save As... and Rename."""
        menu = QMenu(self)
        remove = menu.addAction("Remove")
        remove.triggered.connect(lambda: self.remove(item_id))
        save = menu.addAction("Save As...")
        save.setEnabled(self.viewer.scene.item(item_id).category in SAVABLE)
        save.triggered.connect(lambda: self.save_requested.emit(item_id))
        rename = menu.addAction("Rename")
        rename.triggered.connect(lambda: self.rename(item_id))
        return menu

    def _context_menu(self, position: QPoint) -> None:
        """Shows an item row's context menu at position."""
        row = self.itemAt(position)
        if row is None:
            return
        target = _target(row)
        if isinstance(target, Category):
            return
        menu = self.context_menu(target)
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        menu.popup(self.viewport().mapToGlobal(position))

    def _visible(self, target: ItemId | Category) -> object:
        """Returns whether the target is visible; `MIXED` if its items differ."""
        if isinstance(target, Category):
            return self.viewer._category_value(target, OptionName.VISIBLE)
        return self.viewer.scene.item(target).options.visible

    def _on_select(
        self, current: QTreeWidgetItem | None, previous: QTreeWidgetItem | None
    ) -> None:
        """Selects the row's item or category in the viewer."""
        self.viewer.select(None if current is None else _target(current))

    def _confirm_remove(self, text: str) -> bool:
        """Returns whether the user confirms a removal."""
        q = QMessageBox.question(
            self,
            "Remove",
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return q == QMessageBox.StandardButton.Yes

    # Dropping files from the file manager

    def _set_dragging(self, dragging: bool) -> None:
        """Shows every category while files are dragged over the tree."""
        if dragging != self.dragging:
            self.dragging = dragging
            self._show_categories()

    @override
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Takes part in a drag that carries files, showing every category."""
        if _local_files(event.mimeData()):
            self._set_dragging(True)
            event.acceptProposedAction()
        else:
            event.ignore()

    @override
    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        """Highlights the category under the pointer, if it takes the files."""
        category = self._drop_category(event.mimeData(), event.position().toPoint())
        self._highlight(category)
        if category is None:
            event.ignore()
        else:
            event.acceptProposedAction()

    @override
    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        """Removes the highlight and hides the empty categories again."""
        self._highlight(None)
        self._set_dragging(False)
        event.accept()

    @override
    def dropEvent(self, event: QDropEvent) -> None:
        """Sends the files and their category to be loaded."""
        mime = event.mimeData()
        category = self._drop_category(mime, event.position().toPoint())
        self._highlight(None)
        self._set_dragging(False)
        if category is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self.files_dropped.emit(_local_files(mime), category)

    def _drop_category(self, mime: QMimeData, position: QPoint) -> Category | None:
        """Returns the category of the row at position, if it reads all the files."""
        row = self.itemAt(position)
        paths = _local_files(mime)
        if row is None or not paths:
            return None
        category = self._category(_target(row))
        kind = REGISTRY[category]
        return category if all(kind.reads(p) for p in paths) else None

    def _highlight(self, category: Category | None) -> None:
        """Marks the rows of category, or none if None, as where the files will go."""
        if category == self._drop_target:
            return
        self._drop_target = category
        colour = self.palette().highlight().color()
        colour.setAlpha(90)
        with QSignalBlocker(self):
            for key, row in self._rows.items():
                on = self._category(key) == category
                for column in range(self.columnCount()):
                    row.setBackground(column, QBrush(colour) if on else QBrush())

    def _category(self, target: ItemId | Category) -> Category:
        """Returns the category a row stands for, or that its item belongs to."""
        if isinstance(target, Category):
            return target
        return self.viewer.scene.item(target).category


class ScenePanel(QWidget):
    """Shows the scene tree above a footer with Open files... and Session."""

    open_requested = Signal()
    open_session_requested = Signal()
    save_session_requested = Signal()

    def __init__(self, viewer: Viewer, parent: QWidget | None = None) -> None:
        """Creates the tree and the footer."""
        super().__init__(parent)
        self.tree = SceneTree(viewer, self)
        self.open_button = QPushButton("Open files...", self)
        self.open_button.setToolTip("Open images, tractograms or surfaces (Ctrl+O)")
        self.open_button.clicked.connect(self.open_requested)
        self.session_button = QToolButton(self)
        self.session_button.setObjectName("menuButton")
        self.session_button.setText("Session")
        self.session_button.setToolTip("Open or save the items and the views")
        self.session_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.session_button)
        menu.addAction("Open Session...", self.open_session_requested.emit)
        menu.addAction("Save Session...", self.save_session_requested.emit)
        self.session_button.setMenu(menu)
        footer = QHBoxLayout()
        footer.setContentsMargins(8, 6, 8, 8)
        footer.addWidget(self.open_button, 1)
        footer.addWidget(self.session_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.tree, 1)
        layout.addLayout(footer)


def _local_files(mime: QMimeData) -> list[Path]:
    """Returns the files a drag carries; none if it carries anything else too."""
    urls = mime.urls()
    if not urls or not all(u.isLocalFile() for u in urls):
        return []
    paths = [Path(u.toLocalFile()) for u in urls]
    return paths if all(p.is_file() for p in paths) else []


def _source(item: Item) -> str:
    """Returns a row's tooltip, the item's file."""
    return str(item.source) if item.source is not None else "From Python"


def _target(row: QTreeWidgetItem) -> ItemId | Category:
    """Returns what a row stands for; Qt hands a stored Category back as text."""
    value = row.data(NAME_COLUMN, Qt.ItemDataRole.UserRole)
    return Category(value) if isinstance(value, str) else value
