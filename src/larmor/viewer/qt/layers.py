"""The Layers panel, which lists the scene's items by category."""

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
)
from PySide6.QtWidgets import (
    QHeaderView,
    QMenu,
    QMessageBox,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from larmor.viewer.model.events import (
    DataChanged,
    Event,
    ItemAdded,
    ItemRemoved,
    SourceChanged,
)
from larmor.viewer.model.item import Category, Item, ItemId
from larmor.viewer.model.options import OptionName
from larmor.viewer.model.registry import REGISTRY
from larmor.viewer.model.viewer import SAVABLE, Viewer
from larmor.viewer.qt.style import icon_path

NAME_COLUMN = 0
VISIBILITY_COLUMN = 1
DELETE_COLUMN = 2


def _load_icons() -> tuple[QIcon, QIcon, QIcon, QIcon, QIcon]:
    """Loads the eyes and the remove button, quiet and hovered, tinted for the theme."""
    names = ("eye-open", "eye-closed", "eye-mixed", "delete", "delete-hover")
    visible, invisible, mixed, delete, hover = (QIcon(icon_path(n)) for n in names)
    return visible, invisible, mixed, delete, hover


class LayersPanel(QTreeWidget):
    """Lists the scene's items, grouped by category."""

    # Files dropped onto a category: the paths (list[Path]) and the Category.
    files_dropped = Signal(list, object)
    # A row's name clicked once, or twice in a row; not its eye or remove button.
    name_clicked = Signal()
    name_double_clicked = Signal()
    # "Save As..." chosen in an item's context menu: the item's id.
    save_requested = Signal(int)

    def __init__(self, viewer: Viewer, parent: QWidget | None = None) -> None:
        """Creates the panel."""
        super().__init__(parent)
        self._ready = False  # style changes before the tree is built are ignored
        self._rows: dict[ItemId | Category, QTreeWidgetItem] = {}
        self.viewer = viewer
        # The viewer can outlive the window: stop listening when the widget goes.
        unsubscribe = self.viewer.scene.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: unsubscribe())

        (
            self.icon_visible,
            self.icon_invisible,
            self.icon_mixed,
            self.icon_delete,
            self.icon_delete_hover,
        ) = _load_icons()
        # The row whose remove button shows strongly: the one under the mouse.
        self._hovered: ItemId | Category | None = None

        self.setObjectName("layers")
        self.setMouseTracking(True)
        self.itemEntered.connect(self._on_hover)
        self.setHeaderHidden(True)
        self.setColumnCount(3)
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.header().setSectionResizeMode(
            VISIBILITY_COLUMN, QHeaderView.ResizeMode.ResizeToContents
        )
        self.header().setSectionResizeMode(
            DELETE_COLUMN, QHeaderView.ResizeMode.ResizeToContents
        )
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self._drop_target: Category | None = None
        # A double click asks for the row's options rather than folding it.
        self.setExpandsOnDoubleClick(False)
        self.itemClicked.connect(self._on_clicked)
        self.itemDoubleClicked.connect(self._on_double_clicked)
        self.currentItemChanged.connect(self._on_select)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)
        self._build_tree()
        self._update()
        self._ready = True

    @override
    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        """Tints the icons again when the theme changes."""
        super().changeEvent(event)
        if event.type() == QEvent.Type.StyleChange and self._ready:
            (
                self.icon_visible,
                self.icon_invisible,
                self.icon_mixed,
                self.icon_delete,
                self.icon_delete_hover,
            ) = _load_icons()
            self._build_tree()
            self._update()

    @override
    def leaveEvent(self, event: QEvent) -> None:
        """Quiets the remove button of the row the mouse leaves."""
        super().leaveEvent(event)
        self._on_hover(None, 0)

    def _on_hover(self, row: QTreeWidgetItem | None, column: int) -> None:
        """Shows the remove button of the row under the mouse strongly."""
        target = None if row is None else _target(row)
        if target == self._hovered:
            return
        for key, icon in (
            (self._hovered, self.icon_delete),
            (target, self.icon_delete_hover),
        ):
            if key is not None and key in self._rows:
                self._rows[key].setIcon(DELETE_COLUMN, icon)
        self._hovered = target

    def _build_tree(self) -> None:
        """Rebuilds the rows of the categories and their items."""
        with QSignalBlocker(self):
            self.clear()
            self._rows.clear()
            self._drop_target = None  # new rows have no highlight
            self._hovered = None  # nor a hovered remove button

            for key, kind in REGISTRY.items():
                text = kind.label
                row = QTreeWidgetItem(self, [text])
                row.setData(0, Qt.ItemDataRole.UserRole, key)
                row.setIcon(DELETE_COLUMN, self.icon_delete)
                row.setToolTip(DELETE_COLUMN, f"Remove all {text} from the viewer")
                self._rows[key] = row
                for item in self.viewer.scene.in_category(key):
                    item_row = QTreeWidgetItem(row, [item.name])
                    item_row.setData(0, Qt.ItemDataRole.UserRole, item.id)
                    item_row.setIcon(DELETE_COLUMN, self.icon_delete)
                    item_row.setToolTip(DELETE_COLUMN, "Remove from the viewer")
                    item_row.setToolTip(0, _source(item))
                    self._rows[item.id] = item_row

            self.expandAll()

    def _update(self) -> None:
        """Shows each row's visibility and the selection."""
        with QSignalBlocker(self):
            for target, row in self._rows.items():
                visible = self._visible(target)
                if isinstance(visible, bool):
                    row.setIcon(
                        VISIBILITY_COLUMN,
                        self.icon_visible if visible else self.icon_invisible,
                    )
                    row.setToolTip(
                        VISIBILITY_COLUMN, "Visible" if visible else "Hidden"
                    )
                else:
                    row.setIcon(
                        VISIBILITY_COLUMN,
                        self.icon_mixed,
                    )
                    row.setToolTip(VISIBILITY_COLUMN, "Some visible")
            selected = self.viewer.scene.view.selected
            if selected is None:
                self.setCurrentIndex(QModelIndex())
            else:
                self.setCurrentItem(self._rows[selected])

    def _on_event(self, event: Event) -> None:
        """Rebuilds the rows or updates a tooltip after a scene event, then the eyes."""
        if isinstance(event, (ItemAdded, ItemRemoved)):
            self._build_tree()
        elif isinstance(event, (DataChanged, SourceChanged)):
            self._rows[event.item.id].setToolTip(0, _source(event.item))
        self._update()

    def _on_clicked(self, row: QTreeWidgetItem, column: int) -> None:
        """Sends a name click, toggles visibility or removes, by the column clicked."""
        if column == NAME_COLUMN:
            self.name_clicked.emit()
        elif column == VISIBILITY_COLUMN:
            target = _target(row)
            self.viewer.set(
                target, OptionName.VISIBLE, self._visible(target) is not True
            )
        elif column == DELETE_COLUMN:
            target = _target(row)
            if isinstance(target, Category):
                lbl = REGISTRY[target].label
                n = len(self.viewer.scene.in_category(target))
                if n > 1:
                    if self._confirm_remove(f"Remove all {n} items under {lbl}?"):
                        self.viewer.remove(target)
                else:
                    self.viewer.remove(target)
            else:
                self.viewer.remove(target)

    def _on_double_clicked(self, row: QTreeWidgetItem, column: int) -> None:
        """Sends a double click on a name."""
        if column == NAME_COLUMN:
            self.name_double_clicked.emit()

    def context_menu(self, item_id: ItemId) -> QMenu:
        """Returns an item's context menu."""
        menu = QMenu(self)
        save = menu.addAction("Save As...")
        save.setEnabled(self.viewer.scene.item(item_id).category in SAVABLE)
        save.triggered.connect(lambda: self.save_requested.emit(item_id))
        return menu

    def _context_menu(self, position: QPoint) -> None:
        """Shows an item row's context menu at position."""
        row = self.itemAt(position)
        if row is None or isinstance(_target(row), Category):
            return
        target = _target(row)
        assert not isinstance(target, Category)
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

    @override
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Takes part in a drag that carries files."""
        if _local_files(event.mimeData()):
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
        """Removes the highlight."""
        self._highlight(None)
        event.accept()

    @override
    def dropEvent(self, event: QDropEvent) -> None:
        """Sends the files and their category to be loaded."""
        mime = event.mimeData()
        category = self._drop_category(mime, event.position().toPoint())
        self._highlight(None)
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
        for key, row in self._rows.items():
            on = self._category(key) == category
            for column in range(self.columnCount()):
                row.setBackground(column, QBrush(colour) if on else QBrush())

    def _category(self, target: ItemId | Category) -> Category:
        """Returns the category a row stands for, or that its item belongs to."""
        if isinstance(target, Category):
            return target
        return self.viewer.scene.item(target).category


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
    value = row.data(0, Qt.ItemDataRole.UserRole)
    return Category(value) if isinstance(value, str) else value
