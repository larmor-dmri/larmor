"""The window's header and the toolbar of the views below it."""

from typing import override

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from larmor.viewer.model.events import (
    Event,
    ItemAdded,
    ItemRemoved,
    ItemRenamed,
    PhantomChanged,
    ViewStateChanged,
)
from larmor.viewer.model.phantom import run_problems
from larmor.viewer.model.phantom_editor import item_sources
from larmor.viewer.model.view_state import Layout, Tool, ViewField, Workspace
from larmor.viewer.model.viewer import WORKSPACE_TOOLS, Viewer
from larmor.viewer.qt.style import icon_path

HEADER_HEIGHT = 44  # px
TOOLBAR_HEIGHT = 36  # px
ICON_SIZE = 18  # px
NO_DATA = "No images open"  # the centre of the header with nothing to name
# The workspaces as the switch names them.
WORKSPACE_NAMES = {Workspace.VIEW: "View", Workspace.SIMULATE: "Simulate"}
# Each pointer tool's name and key, as its button shows them.
TOOL_NAMES = {
    Tool.CURSOR: ("Cursor", "V"),
    Tool.SPHERE: ("Sphere ROI", "O"),
    Tool.BOX: ("Box ROI", "B"),
    Tool.ZOOM_BOX: ("Zoom box", "Z"),
    Tool.DRAW_BUNDLE: ("Draw bundle", "N"),
}
# The layouts as the toolbar's switch names them, with their tooltips.
LAYOUT_BUTTONS = {
    Layout.GRID: ("2 x 2", "The four views in a grid"),
    Layout.ONE_PLUS_THREE: ("1 + 3", "The selected view large, the others beside it"),
    Layout.ONE: ("1", "The selected view alone"),
}


class Header(QWidget):
    """Shows the workspace switch, the dataset's name and the window's commands.

    The Run button shows only in the Simulate workspace. It is greyed out until
    the phantom can run, with the first problem as its tooltip, and while a
    simulation runs.
    """

    export_requested = Signal()
    settings_requested = Signal()
    help_requested = Signal()
    run_requested = Signal()

    def __init__(self, viewer: Viewer, parent: QWidget | None = None) -> None:
        """Creates the header, following the viewer's scene."""
        super().__init__(parent)
        self.viewer = viewer
        self.session: str | None = None  # the name of the session opened or saved
        self.running = False  # a simulation runs
        self.setObjectName("header")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setFixedHeight(HEADER_HEIGHT)

        switch = self._build_switch()
        self.title = QLabel(self)
        self.title.setObjectName("datasetName")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.run_button = QPushButton("Run", self)
        self.run_button.setObjectName("headerRun")
        self.run_button.clicked.connect(self.run_requested)
        self.export_button = QPushButton("Export...", self)
        self.export_button.setToolTip("Save the views as images or a video")
        self.export_button.clicked.connect(self.export_requested)
        self.settings_button = _icon_button("settings", "Settings (Ctrl+,)", self)
        self.settings_button.clicked.connect(self.settings_requested)
        self.help_button = _icon_button("help", "Help (F1)", self)
        self.help_button.clicked.connect(self.help_requested)

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 4, 10, 4)
        row.addLayout(switch)
        row.addWidget(self.title, 1)
        for widget in (
            self.run_button,
            self.export_button,
            self.settings_button,
            self.help_button,
        ):
            row.addWidget(widget)
        unsubscribe = viewer.scene.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: unsubscribe())
        self.refresh()

    def _build_switch(self) -> QHBoxLayout:
        """Builds the switch between the workspaces."""
        viewer = self.viewer
        self.workspaces: dict[Workspace, QToolButton] = {}
        group = QButtonGroup(self)
        switch = QHBoxLayout()
        switch.setSpacing(0)
        for workspace, name in WORKSPACE_NAMES.items():
            button = QToolButton(self)
            button.setObjectName("workspaceSwitch")
            button.setText(name)
            button.setCheckable(True)
            button.clicked.connect(lambda _, w=workspace: viewer.set_workspace(w))
            group.addButton(button)
            switch.addWidget(button)
            self.workspaces[workspace] = button
        return switch

    @override
    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        """Tints the icons again when the theme changes."""
        super().changeEvent(event)
        if event.type() == QEvent.Type.StyleChange:
            self.settings_button.setIcon(QIcon(icon_path("settings")))
            self.help_button.setIcon(QIcon(icon_path("help")))

    def set_session(self, name: str | None) -> None:
        """Names the session opened or saved, which the header shows from now on."""
        self.session = name
        self.refresh()

    def set_running(self, running: bool) -> None:
        """Greys out Run while a simulation runs."""
        self.running = running
        self.refresh()

    def refresh(self) -> None:
        """Shows the workspace, the dataset's name and whether the phantom can run."""
        state = self.viewer.scene.view
        for workspace, button in self.workspaces.items():
            button.setChecked(workspace == state.workspace)
        self.title.setText(self.dataset_name())
        self.run_button.setVisible(state.workspace == Workspace.SIMULATE)
        problem = self.run_problem()
        self.run_button.setEnabled(problem is None and not self.running)
        if self.running:
            self.run_button.setToolTip("A simulation is running")
        else:
            self.run_button.setToolTip(problem or "Simulate the phantom (Ctrl+R)")

    def dataset_name(self) -> str:
        """Returns the name the header shows: the session's, or the main volume's."""
        if self.session is not None:
            return self.session
        main = self.viewer.scene.view.main_volume
        if main is not None:
            return self.viewer.scene.item(main).name
        return NO_DATA

    def run_problem(self) -> str | None:
        """Returns why the phantom cannot run, the first problem; None if it can."""
        state = self.viewer.phantom
        if state is None:
            return "Start or open a phantom first"
        found = run_problems(state, item_sources(state, self.viewer.scene))
        return found[0].text if found else None

    def _on_event(self, event: Event) -> None:
        """Shows the header again after a change it shows."""
        if isinstance(event, ViewStateChanged) and event.name not in (
            ViewField.WORKSPACE,
            ViewField.MAIN_VOLUME,
        ):
            return
        if isinstance(
            event,
            (ViewStateChanged, PhantomChanged, ItemAdded, ItemRemoved, ItemRenamed),
        ):
            self.refresh()


class ViewToolbar(QWidget):
    """Shows the pointer tools of the workspace, the layouts and two switches."""

    # Why a tool could not start, for the status bar.
    refused = Signal(str)

    def __init__(self, viewer: Viewer, parent: QWidget | None = None) -> None:
        """Creates the toolbar, following the view state."""
        super().__init__(parent)
        self.viewer = viewer
        self.setObjectName("viewToolbar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setFixedHeight(TOOLBAR_HEIGHT)
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 2, 10, 2)
        row.setSpacing(2)
        self.tools: dict[Tool, QToolButton] = {}
        for tool, (name, key) in TOOL_NAMES.items():
            button = _toggle(f"{name}  {key}", f"{name} ({key})", self)
            button.clicked.connect(lambda _, t=tool: self.choose(t))
            row.addWidget(button)
            self.tools[tool] = button
        row.addSpacing(16)
        self.layouts: dict[Layout, QToolButton] = {}
        for layout, (name, tip) in LAYOUT_BUTTONS.items():
            button = _toggle(name, tip, self)
            button.clicked.connect(lambda _, lay=layout: viewer.set_layout(lay))
            row.addWidget(button)
            self.layouts[layout] = button
        row.addStretch(1)
        self.crosshair = _toggle("Crosshair  X", "Show the crosshair (X)", self)
        self.crosshair.clicked.connect(lambda _: viewer.toggle_crosshair())
        self.overlay = _toggle(
            "Overlay  I", "Show the text, labels and colour bars (I)", self
        )
        self.overlay.clicked.connect(lambda _: viewer.toggle_overlay())
        row.addWidget(self.crosshair)
        row.addWidget(self.overlay)
        unsubscribe = viewer.scene.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: unsubscribe())
        self.refresh()

    def choose(self, tool: Tool) -> None:
        """Chooses a pointer tool; Draw bundle also starts a bundle of the phantom."""
        if tool == Tool.DRAW_BUNDLE and not self.viewer.drawing:
            try:
                self.viewer.start_bundle()
            except ValueError as error:
                self.refused.emit(str(error))
                self.refresh()
                return
        self.viewer.set_tool(tool)

    def refresh(self) -> None:
        """Shows the workspace's tools, the tool, the layout and the switches."""
        state = self.viewer.scene.view
        offered = WORKSPACE_TOOLS[state.workspace]
        for tool, button in self.tools.items():
            button.setVisible(tool in offered)
            button.setChecked(tool == state.tool)
        for layout, button in self.layouts.items():
            button.setChecked(layout == state.layout)
        self.crosshair.setChecked(state.crosshair_visible)
        self.overlay.setChecked(state.overlay_visible)

    def _on_event(self, event: Event) -> None:
        """Shows the toolbar again after a change of the view state it shows."""
        if isinstance(event, ViewStateChanged) and event.name in (
            ViewField.TOOL,
            ViewField.WORKSPACE,
            ViewField.LAYOUT,
            ViewField.CROSSHAIR_VISIBLE,
            ViewField.OVERLAY_VISIBLE,
        ):
            self.refresh()


def _toggle(text: str, tip: str, parent: QWidget) -> QToolButton:
    """Returns a checkable button of the toolbar, which the view state checks."""
    button = QToolButton(parent)
    button.setObjectName("toolButton")
    button.setText(text)
    button.setToolTip(tip)
    button.setCheckable(True)
    button.setAutoRaise(True)
    return button


def _icon_button(icon: str, tip: str, parent: QWidget) -> QToolButton:
    """Returns a header button that shows an icon tinted for the theme."""
    button = QToolButton(parent)
    button.setObjectName("headerIcon")
    button.setIcon(QIcon(icon_path(icon)))
    button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
    button.setToolTip(tip)
    button.setAutoRaise(True)
    return button
