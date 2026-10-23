"""The panels docked beside the views, and the stripes of icons that open them."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, TypeIs, override

import PySide6QtAds as QtAds
from PySide6.QtCore import QByteArray, QCoreApplication, QEvent, QObject, QSize, Qt
from PySide6.QtGui import QAction, QHideEvent, QIcon, QKeySequence, QShowEvent
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QSizePolicy,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from larmor._log import LarmorLogger
from larmor.viewer.model.events import (
    Event,
    ItemAdded,
    ItemRemoved,
    ViewStateChanged,
)
from larmor.viewer.model.item import Category, ItemId
from larmor.viewer.model.shortcuts import Command
from larmor.viewer.model.view_state import ViewField
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.qt.style import icon_path

LAYOUT_FILE = "layout.bin"  # next to the settings file
# Raised when the panels change, so a layout saved by an older viewer is ignored.
LAYOUT_VERSION = 3
SIDE_WIDTHS = (260, 300)  # px, the left and right panels at first start
STRIPE_ICON_SIZE = 20  # px
# Narrower, a restored or saved pinned panel counts as having no room.
MIN_PANEL_WIDTH = 100  # px
MIN_PANEL_HEIGHT = 100  # px; an opened panel lower than this gets a fair share


class Panel(StrEnum):
    """A panel of the window; the value names its dock in the saved layout."""

    LAYERS = "layers"
    LABELS = "labels"
    SCREENSHOTS = "screenshots"
    PHANTOM = "phantom"
    OPTIONS = "options"
    INSPECT_FOD = "inspect_fod"
    SIGNAL_BREAKDOWN = "signal_breakdown"
    SETTINGS = "settings"
    HELP = "help"


class Side(Enum):
    """The edge of the window a panel's stripe button and dock start on."""

    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class PanelInfo:
    """Holds a panel's title, stripe icon (a Lucide name), command and side."""

    title: str
    icon: str
    command: Command
    side: Side
    open_at_start: bool = False
    at_bottom: bool = False  # placed at the far end of its stripe
    # Opens as a window of its own this size (px), until it is docked.
    floating: tuple[int, int] | None = None


# In stripe order, top to bottom.
PANELS: dict[Panel, PanelInfo] = {
    Panel.LAYERS: PanelInfo("Layers", "layers", Command.LAYERS, Side.LEFT, True),
    Panel.LABELS: PanelInfo("Labels", "tags", Command.LABELS, Side.LEFT),
    Panel.SCREENSHOTS: PanelInfo(
        "Screenshots", "camera", Command.SCREENSHOT, Side.LEFT
    ),
    Panel.PHANTOM: PanelInfo(
        "Phantom", "spline", Command.PHANTOM, Side.LEFT, floating=(1420, 820)
    ),
    Panel.OPTIONS: PanelInfo(
        "Options", "sliders-horizontal", Command.OPTIONS, Side.RIGHT, True
    ),
    Panel.INSPECT_FOD: PanelInfo(
        "Inspect FOD", "scan-search", Command.INSPECT_FOD, Side.RIGHT
    ),
    Panel.SIGNAL_BREAKDOWN: PanelInfo(
        "Signal breakdown",
        "chart-column-stacked",
        Command.SIGNAL_BREAKDOWN,
        Side.RIGHT,
    ),
    Panel.SETTINGS: PanelInfo(
        "Settings", "settings", Command.SETTINGS, Side.RIGHT, at_bottom=True
    ),
    Panel.HELP: PanelInfo(
        "Help", "circle-help", Command.HELP, Side.RIGHT, at_bottom=True
    ),
}


def _configure() -> None:
    """Sets QtAds' global flags; they apply to the dock managers made after."""
    manager = QtAds.CDockManager
    flag = manager.eConfigFlag
    manager.setConfigFlags(
        flag.DisableStylesheet  # viewer.qss styles the docks
        | flag.DockAreaHasCloseButton
        | flag.DockAreaCloseButtonClosesTab
        | flag.OpaqueSplitterResize
        | flag.EqualSplitOnInsertion
        | flag.HideSingleCentralWidgetTitleBar
    )
    auto = manager.eAutoHideFlag
    manager.setAutoHideConfigFlags(
        auto.AutoHideFeatureEnabled
        | auto.DockAreaHasAutoHideButton
        | auto.AutoHideCloseButtonCollapsesDock
        | auto.AutoHideHasCloseButton
        | auto.AutoHideCloseOnOutsideMouseClick
    )


class Docks(QObject):
    """Docks the window's panels around the views, and opens them from two stripes.

    A stripe button, its menu item and its key are one QAction. A panel is open
    when it shows: docked and the current tab of its area, or auto-hidden and
    slid out. Panels are made the first time they open.
    """

    def __init__(
        self,
        window: QMainWindow,
        views: QWidget,
        makers: Mapping[Panel, Callable[[], QWidget]],
        actions: Mapping[Command, QAction],
    ) -> None:
        """Puts the views in the middle and a closed dock for each panel around them."""
        super().__init__(window)
        _configure()
        self.manager = QtAds.CDockManager(window)  # becomes the central widget
        # One stripe button is one panel: a drop splits beside an area, never
        # makes a tab in it, so no panel hides behind another.
        self.manager.dockAreaCreated.connect(_set_up_area)
        self._makers = dict(makers)
        self._made: dict[Panel, QWidget] = {}
        self._hidden: list[Panel] = []  # the panels the last Hide panels closed
        # The panels shown once, or placed by a restored layout: they open where
        # they were, not as a window of their own again.
        self._placed: set[Panel] = set()
        self.actions: dict[Panel, QAction] = {}  # the stripe buttons
        self._create_centre(views)
        self._create_docks()
        for panel in PANELS:
            if not PANELS[panel].open_at_start:
                self.docks[panel].toggleView(False)
        self.manager.setSplitterSizes(
            self._centre_area, [SIDE_WIDTHS[0], window.width(), SIDE_WIDTHS[1]]
        )
        self._create_stripes(window, actions)
        # The arrangement at first start, which Reset Layout goes back to.
        self._default = self.manager.saveState(LAYOUT_VERSION)
        self._sync()

    def _create_centre(self, views: QWidget) -> None:
        """Puts the views in the central dock, which takes no tabs."""
        centre = QtAds.CDockWidget(self.manager, "Views")
        centre.setObjectName("views")
        centre.setWidget(views, QtAds.CDockWidget.eInsertMode.ForceNoScrollArea)
        centre.setFeature(QtAds.CDockWidget.DockWidgetFeature.NoTab, True)
        centre_area = self.manager.setCentralWidget(centre)
        _no_tabs(centre_area)
        self._centre_area = centre_area

    def _create_docks(self) -> None:
        """Creates a dock for each panel, beside the views or below its side's first."""
        centre_area = self._centre_area
        features = QtAds.CDockWidget.DockWidgetFeature
        self.docks: dict[Panel, Any] = {}
        areas: dict[Side, Any] = {}
        for panel, info in PANELS.items():
            dock = QtAds.CDockWidget(self.manager, info.title)
            dock.setObjectName(panel.value)
            dock.setFeatures(
                features.DockWidgetClosable
                | features.DockWidgetMovable
                | features.DockWidgetPinnable
                | features.DockWidgetFocusable
            )
            if info.floating is not None:
                dock.setFeature(features.DockWidgetFloatable, True)
            dock.setMinimumSizeHintMode(
                QtAds.CDockWidget.eMinimumSizeHintMode.MinimumSizeHintFromDockWidget
            )
            # The first panel of a side docks beside the views, the others below it.
            previous = areas.get(info.side)
            if previous is None:
                where = (
                    QtAds.LeftDockWidgetArea
                    if info.side is Side.LEFT
                    else QtAds.RightDockWidgetArea
                )
                areas[info.side] = self.manager.addDockWidget(where, dock, centre_area)
            else:
                areas[info.side] = self.manager.addDockWidget(
                    QtAds.BottomDockWidgetArea, dock, previous
                )
            self.docks[panel] = dock
            # Made when it first shows; the dock reports each show and hide.
            dock.viewToggled.connect(lambda *_, p=panel: self._shown(p))
            dock.visibilityChanged.connect(lambda *_, p=panel: self._shown(p))

    def _create_stripes(
        self, window: QMainWindow, actions: Mapping[Command, QAction]
    ) -> None:
        """Creates the two stripes, with a button per panel from its action."""
        self.stripes = {side: Stripe(window, side) for side in Side}
        for panel, info in PANELS.items():
            action = actions[info.command]
            action.setText(info.title)
            action.setCheckable(True)
            keys = ", ".join(
                k.toString(QKeySequence.SequenceFormat.NativeText)
                for k in action.shortcuts()
            )
            action.setToolTip(f"{info.title} ({keys})")
            action.triggered.connect(lambda *_, p=panel: self.toggle(p))
            self.actions[panel] = action
            self.stripes[info.side].add(action, info.icon, at_bottom=info.at_bottom)
        actions[Command.HIDE_PANELS].setText("Hide all panels")
        actions[Command.HIDE_PANELS].triggered.connect(self.toggle_all)

    def widget(self, panel: Panel) -> QWidget:
        """Returns the panel's content, made now if it has not been yet."""
        made = self._made.get(panel)
        if made is None:
            made = self._makers[panel]()
            self._made[panel] = made
            dock = self.docks[panel]
            # Only Screenshots gets a scroll area; the Phantom editor scrolls its own.
            mode = (
                QtAds.CDockWidget.eInsertMode.ForceScrollArea
                if panel == Panel.SCREENSHOTS
                else QtAds.CDockWidget.eInsertMode.ForceNoScrollArea
            )
            dock.setWidget(made, mode)
            # The dock takes its minimum from itself, so a panel's own minimum
            # width, such as the Signal breakdown chart's, is passed on to it.
            dock.setMinimumWidth(max(made.minimumWidth(), MIN_PANEL_WIDTH))
        return made

    def set_side_widths(self, left: int, right: int) -> None:
        """Sets the widths of the panel columns left and right of the views, in px.

        The views take the rest of the window's width.
        """
        views = max(self.manager.width() - left - right, 0)
        self.manager.setSplitterSizes(self._centre_area, [left, views, right])

    def set_title(self, panel: Panel, text: str) -> None:
        """Names the panel's dock, as its title bar and tab show it."""
        dock = self.docks[panel]
        if dock.windowTitle() != text:
            dock.setWindowTitle(text)

    def is_open(self, panel: Panel) -> bool:
        """Returns whether the panel shows: docked as its area's tab, or slid out."""
        dock = self.docks[panel]
        if dock.isClosed():
            return False
        container = dock.autoHideDockContainer()
        if container is not None:
            return not container.isHidden()
        return bool(dock.isCurrentTab())

    def side_open(self, side: Side) -> bool:
        """Returns whether a panel of the side's stripe shows."""
        return any(self.is_open(p) for p, info in PANELS.items() if info.side is side)

    def is_pinned(self, panel: Panel) -> bool:
        """Returns whether the panel is docked beside the views, not auto-hidden."""
        return not self.docks[panel].isAutoHide()

    def open(self, panel: Panel) -> None:
        """Shows the panel: opens it, makes it its area's tab, or slides it out.

        A panel that opens as its own window does so the first time, unless a
        restored layout placed it.
        """
        self.widget(panel)
        dock = self.docks[panel]
        size = PANELS[panel].floating
        first = dock.isClosed() and panel not in self._placed
        if size is None and dock.dockAreaWidget() is None:
            self._dock_at_side(panel)
        dock.toggleView(True)
        if size is not None and first:
            dock.setFloating()
            container = dock.floatingDockContainer()
            if container is not None:
                container.resize(*size)
        self._placed.add(panel)
        self._make_room(panel)
        self._sync()

    def _dock_at_side(self, panel: Panel) -> None:
        """Docks a panel without a place: below a panel of its side, or by the views.

        A layout saved before the panel existed restores without it, and QtAds
        would open such a dock as a window of its own.
        """
        side = PANELS[panel].side
        order = [p for p, info in PANELS.items() if info.side is side and p != panel]
        areas = [self.docks[p].dockAreaWidget() for p in order]
        placed = [
            a for a in areas if a is not None and not a.dockContainer().isFloating()
        ]
        dock = self.docks[panel]
        if placed:
            self.manager.addDockWidget(QtAds.BottomDockWidgetArea, dock, placed[0])
        else:
            where = (
                QtAds.LeftDockWidgetArea
                if side is Side.LEFT
                else QtAds.RightDockWidgetArea
            )
            self.manager.addDockWidget(where, dock, self._centre_area)

    def _make_room(self, panel: Panel) -> None:
        """Shares a column equally if the panel opened in it gets almost no height.

        An area closed when the layout was saved comes back with no height.
        """
        dock = self.docks[panel]
        area = dock.dockAreaWidget()
        column = area.parentWidget() if area is not None else None
        if (
            dock.isAutoHide()
            or not isinstance(column, QSplitter)
            or column.orientation() != Qt.Orientation.Vertical
            or not dock.isVisible()
            or dock.height() >= MIN_PANEL_HEIGHT
        ):
            return
        areas = [column.widget(i) for i in range(column.count())]
        shown = [a is not None and not a.isHidden() for a in areas]
        share = sum(column.sizes()) // max(sum(shown), 1)
        column.setSizes([share if s else 0 for s in shown])

    def close(self, panel: Panel) -> None:
        """Hides the panel; an auto-hidden one slides back in."""
        dock = self.docks[panel]
        container = dock.autoHideDockContainer()
        if container is not None:
            container.collapseView(True)
        else:
            dock.toggleView(False)
        self._sync()

    def toggle(self, panel: Panel) -> None:
        """Hides the panel if it shows, else shows it."""
        if self.is_open(panel):
            self.close(panel)
        else:
            self.open(panel)

    def toggle_all(self) -> None:
        """Hides every panel shown, or shows again those this hid."""
        shown = [p for p in Panel if self.is_open(p)]
        if shown:
            self._hidden = shown
            for panel in shown:
                self.close(panel)
        else:
            again = self._hidden or [p for p, i in PANELS.items() if i.open_at_start]
            for panel in again:
                self.open(panel)

    def set_pinned(self, panel: Panel, pinned: bool) -> None:
        """Docks the panel beside the views, or leaves it to auto-hide at its side."""
        self.docks[panel].setAutoHide(not pinned)
        self._sync()

    def reset(self) -> None:
        """Goes back to the arrangement at first start.

        Every panel opens again where it first did, so the Phantom editor is a
        window of its own once more.
        """
        self.manager.restoreState(self._default, LAYOUT_VERSION)
        for panel, info in PANELS.items():
            dock = self.docks[panel]
            if dock.isInFloatingContainer():
                dock.toggleView(False)
            if info.open_at_start:
                self.open(panel)
            elif not dock.isClosed():
                dock.toggleView(False)
        self._placed.clear()
        self._sync()

    def save_state(self) -> bytes:
        """Returns the arrangement: open panels, sides, sizes, pinned or auto-hidden."""
        return bytes(self.manager.saveState(LAYOUT_VERSION).data())

    def restore_state(self, state: bytes) -> bool:
        """Arranges the panels as saved, in a shown window.

        Returns False, keeping the arrangement it had, if the state is unreadable,
        puts two panels in one area as tabs, leaves a pinned panel it opens
        hidden or without room, or makes the window too wide for its screen to
        show both stripes. The stripes show whatever the state.

        Returns:
            Whether the arrangement was restored.

        Raises:
            RuntimeError: If the window is not shown: whether a panel shows can
                only be checked on screen.
        """
        if not self.manager.isVisible():
            raise RuntimeError("The panel layout is restored once the window shows.")
        before = self.manager.saveState(LAYOUT_VERSION)
        restored = bool(self.manager.restoreState(QByteArray(state), LAYOUT_VERSION))
        if restored:
            self._show_open_areas()
            if (
                self._tabbed()
                or self._hidden_panel() is not None
                or not self._fits_screen()
            ):
                self.manager.restoreState(before, LAYOUT_VERSION)
                self._show_open_areas()
                restored = False
        for panel in Panel:
            if not self.docks[panel].isClosed() or self.docks[panel].isFloating():
                self._placed.add(panel)
            if not self.docks[panel].isClosed():
                self.widget(panel)
        for stripe in self.stripes.values():
            stripe.show()
        self._sync()
        return restored

    def save(self, path: Path) -> bool:
        """Writes the arrangement of a shown window to path.

        Returns False, writing nothing, if the window is not shown or a pinned
        panel it opens does not show: the state would restore panels unseen. A
        failure to write is only logged.

        Returns:
            Whether the file was written.
        """
        if not self.manager.isVisible():
            return False
        hidden = self._hidden_panel()
        if hidden is not None:
            LarmorLogger.debug(
                "Not saving the panel layout: the %s panel is open but hidden or "
                "narrower than %d px.",
                PANELS[hidden].title,
                MIN_PANEL_WIDTH,
            )
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.save_state())
        except OSError as error:
            LarmorLogger.warning("Cannot save the panel layout to %s: %s", path, error)
            return False
        return True

    def _pinned_open(self) -> list[Any]:
        """Returns the docks shown beside the views: open, pinned and current tab."""
        return [
            d
            for d in self.docks.values()
            if not d.isClosed() and not d.isAutoHide() and d.isCurrentTab()
        ]

    def _show_open_areas(self) -> None:
        """Shows the areas of the open docks a restored state leaves hidden.

        QtAds restores a dock that was closed before into an area it keeps
        hidden; closing and opening the dock shows the area.
        """
        for dock in self._pinned_open():
            if not dock.isVisible():
                dock.toggleView(False)
                dock.toggleView(True)

    def _hidden_panel(self) -> Panel | None:
        """Returns a pinned open panel that does not show or is too narrow, if any."""
        # The sizes a restore sets apply once the layouts run.
        QCoreApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest.value)
        for panel, dock in self.docks.items():
            if dock in self._pinned_open() and (
                not dock.isVisible() or dock.width() < MIN_PANEL_WIDTH
            ):
                return panel
        return None

    def _fits_screen(self) -> bool:
        """Returns whether the window's narrowest width, with stripes, fits its screen.

        Panels side by side can need more width than the screen has, which pushes
        the right stripe off it.

        Returns:
            True if it fits.
        """
        window: QWidget = self.manager.window()
        screen = window.screen()
        if screen is None:
            return True
        needed = window.minimumSizeHint().width()
        return needed <= screen.availableGeometry().width()

    def _tabbed(self) -> bool:
        """Returns whether an area holds two panels as tabs, as old layouts may."""
        areas = (self.manager.dockArea(i) for i in range(self.manager.dockAreaCount()))
        return any(area.dockWidgetsCount() > 1 for area in areas)

    def restore(self, path: Path) -> bool:
        """Arranges the panels as saved in path; False if missing or unreadable."""
        try:
            state = path.read_bytes()
        except OSError:
            return False
        return self.restore_state(state)

    def _shown(self, panel: Panel) -> None:
        """Makes a panel's content once it shows, and checks the stripe buttons."""
        if self.is_open(panel):
            self.widget(panel)
        self._sync()

    def _sync(self) -> None:
        """Checks the stripe buttons of the panels that show."""
        for panel, action in self.actions.items():
            action.setChecked(self.is_open(panel))


def _no_tabs(area: QWidget) -> None:
    """Lets drops split beside the area but not add a tab to it."""
    assert isinstance(area, QtAds.CDockAreaWidget)
    area.setAllowedAreas(QtAds.OuterDockAreas)


def _set_up_area(area: QWidget) -> None:
    """Sets up a panel's area: no tabs, and no narrower than a saved layout keeps."""
    _no_tabs(area)
    area.setMinimumWidth(MIN_PANEL_WIDTH)


class Stripe(QToolBar):
    """A thin column of panel buttons on one edge of the window."""

    def __init__(self, window: QMainWindow, side: Side) -> None:
        """Adds the stripe to the window's edge."""
        super().__init__(f"{side.value.capitalize()} panels", window)
        self.setObjectName("stripe")
        self.setProperty("side", side.value)
        self.setMovable(False)
        self.setFloatable(False)
        self.setOrientation(Qt.Orientation.Vertical)
        self.setIconSize(QSize(STRIPE_ICON_SIZE, STRIPE_ICON_SIZE))
        # No menu that would hide the stripe for good.
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self.toggleViewAction().setVisible(False)
        self._icons: dict[QAction, str] = {}
        self._bottom: QWidget | None = None
        area = (
            Qt.ToolBarArea.LeftToolBarArea
            if side is Side.LEFT
            else Qt.ToolBarArea.RightToolBarArea
        )
        window.addToolBar(area, self)

    def add(self, action: QAction, icon: str, at_bottom: bool = False) -> None:
        """Adds a panel's button, with its Lucide icon tinted for the theme."""
        if at_bottom and self._bottom is None:
            self._bottom = QWidget(self)
            self._bottom.setSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
            )
            self.addWidget(self._bottom)
        self._icons[action] = icon
        action.setIcon(_stripe_icon(icon))
        self.addAction(action)

    @override
    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        """Tints the icons again when the theme changes."""
        super().changeEvent(event)
        if event.type() == QEvent.Type.StyleChange:
            for action, icon in self._icons.items():
                action.setIcon(_stripe_icon(icon))


def _stripe_icon(name: str) -> QIcon:
    """Returns a stripe button's icon: muted, in the accent while its panel is open."""
    icon = QIcon(icon_path(f"stripe-{name}"))
    icon.addFile(icon_path(f"stripe-{name}-on"), state=QIcon.State.On)
    return icon


class PanelContent(QWidget):
    """An item panel's view of its item, paused while the panel is hidden."""

    def pause(self) -> None:
        """Stops redrawing for the scene's changes, all but its item's removal."""

    def resume(self) -> None:
        """Catches up once with the changes made while paused."""


class ItemPanel(QWidget):
    """Shows one item of a category, the selected one or else the last one shown.

    It shows the first item of the category when it has none yet, and a hint
    when the category is empty. The item's widget is made when the panel first
    shows it, and pauses while the panel is hidden.
    """

    def __init__(
        self,
        viewer: Viewer,
        category: Category,
        make: Callable[[ItemId, QWidget], PanelContent],
        hint: str,
        parent: QWidget | None = None,
    ) -> None:
        """Starts on the selected item of the category, else its first one."""
        super().__init__(parent)
        self.viewer = viewer
        self.category = category
        self._make = make
        self.item_id: ItemId | None = None
        self.content: PanelContent | None = None
        self._shown = False  # between a show event and a hide event
        self.hint = QLabel(hint, self)
        self.hint.setObjectName("panelHint")
        self.hint.setWordWrap(True)
        self.hint.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self.hint)
        # The viewer can outlive the panel: stop listening when it goes.
        unsubscribe = viewer.scene.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: unsubscribe())
        selected = viewer.scene.view.selected
        self.show_item(selected if self._fits(selected) else self._first())

    def show_item(self, item_id: ItemId | None) -> None:
        """Shows this item of the category, or the hint if None."""
        if item_id == self.item_id and (self.content is not None or item_id is None):
            return
        self.item_id = item_id
        self.drop()
        self.hint.setVisible(item_id is None)
        if item_id is not None and self._shown:
            self._build()

    def _build(self) -> None:
        """Makes the item's widget and adds it below the hint."""
        assert self.item_id is not None
        self.content = self._make(self.item_id, self)
        self._layout.addWidget(self.content, stretch=1)

    def drop(self) -> None:
        """Closes the item's widget, which stops it following the scene.

        The panel makes it again when it next shows.
        """
        if self.content is not None:
            self._layout.removeWidget(self.content)
            self.content.close()
            self.content.deleteLater()
            self.content = None

    @override
    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        """Makes the item's widget, or brings the paused one up to date."""
        super().showEvent(event)
        self._shown = True
        if self.content is not None:
            self.content.resume()
        elif self.item_id is not None:
            self._build()

    @override
    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        """Pauses the item's widget: a hidden panel draws nothing."""
        super().hideEvent(event)
        self._shown = False
        if self.content is not None:
            self.content.pause()

    def _fits(self, target: object) -> TypeIs[ItemId]:
        """Returns whether a selection is an item of the panel's category."""
        return isinstance(target, int) and (
            self.viewer.scene.item(target).category == self.category
        )

    def _first(self) -> ItemId | None:
        """Returns the category's first item; None if it is empty."""
        items = self.viewer.scene.in_category(self.category)
        return items[0].id if items else None

    def _on_event(self, event: Event) -> None:
        """Follows the selection, the first item added and the shown item's removal."""
        if isinstance(event, ViewStateChanged) and event.name == ViewField.SELECTED:
            if self._fits(event.new):
                self.show_item(event.new)
        elif isinstance(event, ItemAdded) and self.item_id is None:
            if event.item.category == self.category:
                self.show_item(event.item.id)
        elif isinstance(event, ItemRemoved) and event.item.id == self.item_id:
            self.show_item(self._first())
