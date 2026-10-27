"""The panels docked beside and below the views."""

from collections.abc import Callable
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any, TypeIs, override

import PySide6QtAds as QtAds
from PySide6.QtCore import QByteArray, QCoreApplication, QEvent, QObject, Qt
from PySide6.QtGui import QHideEvent, QShowEvent
from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from larmor._log import LarmorLogger
from larmor.viewer.model.events import (
    Event,
    ItemAdded,
    ItemRemoved,
    ViewStateChanged,
)
from larmor.viewer.model.item import Category, ItemId
from larmor.viewer.model.view_state import ViewField
from larmor.viewer.model.viewer import Viewer

LAYOUT_FILE = "layout.bin"  # next to the settings file
# Raised when the panels change, so a layout saved by an older viewer is ignored.
LAYOUT_VERSION = 4
SIDE_WIDTHS = (260, 320)  # px, the left and right panels at first start
BOTTOM_HEIGHT = 300  # px, the Signal breakdown when it first opens
# Narrower, a restored or saved panel counts as having no room.
MIN_PANEL_WIDTH = 100  # px


class Panel(StrEnum):
    """A dock of the window; the value names it in the saved layout."""

    LEFT = "left"  # the scene tree, or the phantom's steps
    RIGHT = "right"  # the inspector, or the page of the phantom's step
    SIGNAL_BREAKDOWN = "signal_breakdown"


# The docks either side of the views, which Ctrl+0 hides and shows.
SIDES = (Panel.LEFT, Panel.RIGHT)


def _configure() -> None:
    """Sets QtAds' global flags; they apply to the dock managers made after."""
    manager = QtAds.CDockManager
    flag = manager.eConfigFlag
    manager.setConfigFlags(
        flag.DisableStylesheet  # viewer.qss styles the docks
        | flag.DockAreaHasCloseButton
        | flag.DockAreaCloseButtonClosesTab
        | flag.DockAreaHideDisabledButtons
        | flag.OpaqueSplitterResize
        | flag.EqualSplitOnInsertion
        | flag.HideSingleCentralWidgetTitleBar
    )
    manager.setAutoHideConfigFlags(manager.eAutoHideFlag(0))


class Docks(QObject):
    """Docks a panel either side of the views and the Signal breakdown below them.

    The side panels are always docked; Ctrl+0 hides and shows them. The Signal
    breakdown is made the first time it opens.
    """

    def __init__(
        self,
        window: QMainWindow,
        views: QWidget,
        sides: dict[Panel, QWidget],
        make_breakdown: Callable[[], QWidget],
    ) -> None:
        """Puts the views in the middle, the side panels beside them, open."""
        super().__init__(window)
        _configure()
        self.manager = QtAds.CDockManager(window)  # becomes the central widget
        self.manager.dockAreaCreated.connect(_set_up_area)
        self._make_breakdown = make_breakdown
        self._breakdown: QWidget | None = None
        self._hidden: list[Panel] = []  # the sides the last Ctrl+0 hid
        self._create_centre(views)
        self.docks: dict[Panel, Any] = {}
        features = QtAds.CDockWidget.DockWidgetFeature
        for panel, where in (
            (Panel.LEFT, QtAds.LeftDockWidgetArea),
            (Panel.RIGHT, QtAds.RightDockWidgetArea),
        ):
            dock = self._dock(panel, features.DockWidgetMovable)
            dock.setWidget(
                sides[panel], QtAds.CDockWidget.eInsertMode.ForceNoScrollArea
            )
            self.manager.addDockWidget(where, dock, self._centre_area)
        breakdown = self._dock(
            Panel.SIGNAL_BREAKDOWN,
            features.DockWidgetClosable | features.DockWidgetMovable,
        )
        breakdown.setWindowTitle("Signal breakdown")
        self.manager.addDockWidget(
            QtAds.BottomDockWidgetArea, breakdown, self._centre_area
        )
        breakdown.viewToggled.connect(self._breakdown_toggled)
        breakdown.toggleView(False)
        self._hide_side_title_bars()
        self.set_side_widths(*SIDE_WIDTHS)
        # The arrangement at first start, which Reset goes back to.
        self._default = self.manager.saveState(LAYOUT_VERSION)

    def _create_centre(self, views: QWidget) -> None:
        """Puts the views in the central dock, which takes no tabs."""
        centre = QtAds.CDockWidget(self.manager, "Views")
        centre.setObjectName("views")
        centre.setWidget(views, QtAds.CDockWidget.eInsertMode.ForceNoScrollArea)
        centre.setFeature(QtAds.CDockWidget.DockWidgetFeature.NoTab, True)
        self._centre_area = self.manager.setCentralWidget(centre)
        _no_tabs(self._centre_area)

    def _dock(
        self, panel: Panel, features: QtAds.CDockWidget.DockWidgetFeature
    ) -> QtAds.CDockWidget:
        """Creates a panel's dock, named by the panel in the saved layout."""
        dock = QtAds.CDockWidget(self.manager, panel.value)
        dock.setObjectName(panel.value)
        dock.setFeatures(features | QtAds.CDockWidget.DockWidgetFeature.NoTab)
        dock.setMinimumSizeHintMode(
            QtAds.CDockWidget.eMinimumSizeHintMode.MinimumSizeHintFromDockWidget
        )
        self.docks[panel] = dock
        return dock

    def _hide_side_title_bars(self) -> None:
        """Hides the empty title bars of the side panels, which start under the bars.

        The Signal breakdown keeps its title bar for its close button.
        """
        flag = QtAds.CDockAreaWidget.eDockAreaFlag.HideSingleWidgetTitleBar
        for panel in SIDES:
            area = self.docks[panel].dockAreaWidget()
            if area is not None:
                area.setDockAreaFlag(flag, True)

    def breakdown(self) -> QWidget:
        """Returns the Signal breakdown's content, made now if it has not been yet."""
        if self._breakdown is None:
            self._breakdown = self._make_breakdown()
            dock = self.docks[Panel.SIGNAL_BREAKDOWN]
            dock.setWidget(
                self._breakdown, QtAds.CDockWidget.eInsertMode.ForceNoScrollArea
            )
            # A panel that names what it shows, as the Signal breakdown names
            # its voxel, titles its dock.
            titled = getattr(self._breakdown, "title_changed", None)
            if titled is not None:
                titled.connect(partial(self.set_title, Panel.SIGNAL_BREAKDOWN))
        return self._breakdown

    def _breakdown_toggled(self, shown: bool) -> None:
        """Makes the Signal breakdown's content the first time its dock shows."""
        if shown:
            self.breakdown()

    def set_title(self, panel: Panel, text: str) -> None:
        """Names a panel's dock, as its title bar shows it."""
        dock = self.docks[panel]
        if dock.windowTitle() != text:
            dock.setWindowTitle(text)

    def set_side_widths(self, left: int, right: int) -> None:
        """Sets the widths of the panels left and right of the views, in px."""
        views = max(self.manager.width() - left - right, 0)
        # The splitter of the side panels, which holds the views' column.
        area = self.docks[Panel.LEFT].dockAreaWidget()
        if area is not None:
            self.manager.setSplitterSizes(area, [left, views, right])

    def is_open(self, panel: Panel) -> bool:
        """Returns whether the panel's dock is open."""
        return not self.docks[panel].isClosed()

    def open(self, panel: Panel) -> None:
        """Opens a panel's dock where it was last."""
        first = panel == Panel.SIGNAL_BREAKDOWN and self._breakdown is None
        if panel == Panel.SIGNAL_BREAKDOWN:
            self.breakdown()
        self.docks[panel].toggleView(True)
        if first:
            self._set_breakdown_height(BOTTOM_HEIGHT)

    def _set_breakdown_height(self, height: int) -> None:
        """Sets the height of the Signal breakdown below the views, in px.

        It takes at most half the height, so the views keep the rest.
        """
        area = self.docks[Panel.SIGNAL_BREAKDOWN].dockAreaWidget()
        splitter = area.parentWidget() if area is not None else None
        if isinstance(splitter, QtAds.CDockSplitter) and splitter.count() == 2:
            total = sum(splitter.sizes())
            height = min(height, total // 2)
            self.manager.setSplitterSizes(area, [total - height, height])

    def close(self, panel: Panel) -> None:
        """Closes a panel's dock."""
        self.docks[panel].toggleView(False)

    def toggle(self, panel: Panel) -> None:
        """Closes the panel if it is open, else opens it."""
        if self.is_open(panel):
            self.close(panel)
        else:
            self.open(panel)

    def toggle_sides(self) -> None:
        """Hides the side panels that are open, or opens again those it hid."""
        shown = [p for p in SIDES if self.is_open(p)]
        if shown:
            self._hidden = shown
            for panel in shown:
                self.close(panel)
        else:
            for panel in self._hidden or list(SIDES):
                self.open(panel)

    def reset(self) -> None:
        """Goes back to the arrangement at first start."""
        self.manager.restoreState(self._default, LAYOUT_VERSION)
        self._hide_side_title_bars()
        for panel in SIDES:
            self.open(panel)

    def save_state(self) -> bytes:
        """Returns the arrangement: open panels, their sides and sizes."""
        return bytes(self.manager.saveState(LAYOUT_VERSION).data())

    def restore_state(self, state: bytes) -> bool:
        """Arranges the panels as saved, in a shown window; the Signal breakdown closes.

        Returns False, keeping the arrangement it had, if the state is unreadable
        or older, puts two panels in one area as tabs, leaves an open side panel
        hidden or without room, or makes the window too wide for its screen.

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
        if restored and (
            self._tabbed() or self._hidden_panel() is not None or not self._fits()
        ):
            self.manager.restoreState(before, LAYOUT_VERSION)
            restored = False
        self._hide_side_title_bars()
        # The Signal breakdown opens only when asked for, never from a saved layout.
        self.close(Panel.SIGNAL_BREAKDOWN)
        return restored

    def save(self, path: Path) -> bool:
        """Writes the arrangement of a shown window to path.

        Returns False, writing nothing, if the window is not shown or an open
        side panel does not show. A failure to write is only logged.

        Returns:
            Whether the file was written.
        """
        if not self.manager.isVisible() or self._hidden_panel() is not None:
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.save_state())
        except OSError as error:
            LarmorLogger.warning("Cannot save the panel layout to %s: %s", path, error)
            return False
        return True

    def restore(self, path: Path) -> bool:
        """Arranges the panels as saved in path; False if missing or unreadable."""
        try:
            state = path.read_bytes()
        except OSError:
            return False
        return self.restore_state(state)

    def _hidden_panel(self) -> Panel | None:
        """Returns an open side panel that does not show or is too narrow, if any."""
        # The sizes a restore sets apply once the layouts run.
        QCoreApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest.value)
        for panel in SIDES:
            dock = self.docks[panel]
            if not dock.isClosed() and (
                not dock.isVisible() or dock.width() < MIN_PANEL_WIDTH
            ):
                return panel
        return None

    def _fits(self) -> bool:
        """Returns whether the window's narrowest width fits its screen."""
        window: QWidget = self.manager.window()
        screen = window.screen()
        if screen is None:
            return True
        return window.minimumSizeHint().width() <= screen.availableGeometry().width()

    def _tabbed(self) -> bool:
        """Returns whether an area holds two panels as tabs."""
        areas = (self.manager.dockArea(i) for i in range(self.manager.dockAreaCount()))
        return any(area.dockWidgetsCount() > 1 for area in areas)


def _no_tabs(area: QWidget) -> None:
    """Lets drops split beside the area but not add a tab to it."""
    assert isinstance(area, QtAds.CDockAreaWidget)
    area.setAllowedAreas(QtAds.OuterDockAreas)


def _set_up_area(area: QWidget) -> None:
    """Sets up a panel's area: no tabs, and no narrower than a saved layout keeps."""
    _no_tabs(area)
    area.setMinimumWidth(MIN_PANEL_WIDTH)


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
