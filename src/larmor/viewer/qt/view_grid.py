"""The four views as the window lays them out: a grid, one large and three, or one."""

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QGridLayout, QWidget

from larmor.viewer.model.events import Event, ViewStateChanged
from larmor.viewer.model.item import VIEW_KINDS, ViewKind
from larmor.viewer.model.view_state import Layout, ViewField
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.qt.views import (
    FULL_RESOLUTION_DELAY,
    POSITIONS,
    ViewInput,
    ViewPanel,
)

# How many times wider the large view is than the column of small ones.
LARGE_STRETCH = 3
# The view state fields that change which views show and where.
ARRANGING = (ViewField.LAYOUT, ViewField.MAXIMISED, ViewField.SELECTED_VIEW)


class ViewGrid(QWidget):
    """Lays the four views out as the view state's layout says.

    A maximised view fills the grid whatever the layout. In the one large and
    three small layout, the selected view is the large one.
    """

    def __init__(self, viewer: Viewer, parent: QWidget | None = None) -> None:
        """Creates the four views and their mouse input."""
        super().__init__(parent)
        self.viewer = viewer
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        # Each panel's frame is half the gap between two views.
        self.grid.setSpacing(0)

        # While the user drags or scrolls, the views draw with less detail (and at
        # reduced resolution if the setting asks); this timer ends that once the
        # input stops.
        self.interaction_timer = QTimer(self)
        self.interaction_timer.setSingleShot(True)
        self.interaction_timer.setInterval(FULL_RESOLUTION_DELAY)
        self.interaction_timer.timeout.connect(self._end_interaction)

        self.panels: dict[ViewKind, ViewPanel] = {}
        self.inputs: dict[ViewKind, ViewInput] = {}
        for kind in VIEW_KINDS:
            panel = ViewPanel(kind, self)
            self.panels[kind] = panel
            self.inputs[kind] = ViewInput(viewer, panel)
            self.inputs[kind].interacted.connect(self._start_interaction)

        # The viewer can outlive the window: stop listening when the widget goes.
        unsubscribe = self.viewer.scene.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: unsubscribe())
        self._arrange()
        self._show_selected()

    def _on_event(self, event: Event) -> None:
        """Lays the views out again, or frames the selected one, after a change."""
        if not isinstance(event, ViewStateChanged):
            return
        if event.name in ARRANGING:
            self._arrange()
        if event.name == ViewField.SELECTED_VIEW:
            self._show_selected()

    def shown(self) -> list[ViewKind]:
        """Returns the views the layout shows, the large one first."""
        state = self.viewer.scene.view
        if state.maximised is not None:
            return [state.maximised]
        if state.layout == Layout.ONE:
            return [state.selected_view]
        if state.layout == Layout.ONE_PLUS_THREE:
            return [state.selected_view] + [
                k for k in VIEW_KINDS if k != state.selected_view
            ]
        return list(VIEW_KINDS)

    def _arrange(self) -> None:
        """Places the views the layout shows in the grid and hides the others."""
        state = self.viewer.scene.view
        shown = self.shown()
        for panel in self.panels.values():
            self.grid.removeWidget(panel)
        for row in range(3):
            self.grid.setRowStretch(row, 0)
        for column in range(2):
            self.grid.setColumnStretch(column, 0)
        if len(shown) == 1:
            self.grid.addWidget(self.panels[shown[0]], 0, 0)
            self.grid.setRowStretch(0, 1)
            self.grid.setColumnStretch(0, 1)
        elif state.layout == Layout.ONE_PLUS_THREE:
            large, *small = shown
            self.grid.addWidget(self.panels[large], 0, 0, len(small), 1)
            for row, kind in enumerate(small):
                self.grid.addWidget(self.panels[kind], row, 1)
                self.grid.setRowStretch(row, 1)
            self.grid.setColumnStretch(0, LARGE_STRETCH)
            self.grid.setColumnStretch(1, 1)
        else:
            for kind in shown:
                row, column = POSITIONS[kind]
                self.grid.addWidget(self.panels[kind], row, column)
                self.grid.setRowStretch(row, 1)
                self.grid.setColumnStretch(column, 1)
        for kind, panel in self.panels.items():
            panel.setVisible(kind in shown)
            panel.set_maximised(kind == state.maximised)

    def _start_interaction(self) -> None:
        """Draws every view with less detail until the input stops."""
        # Every view: dragging in one moves the slices and crosshairs of the others.
        for panel in self.panels.values():
            panel.view.start_interaction()
        self.interaction_timer.start()

    def _end_interaction(self) -> None:
        """Draws every view in full detail again."""
        for panel in self.panels.values():
            panel.view.end_interaction()

    def _show_selected(self) -> None:
        """Frames the selected view in the accent."""
        selected = self.viewer.scene.view.selected_view
        for kind, panel in self.panels.items():
            panel.set_selected(kind == selected)
