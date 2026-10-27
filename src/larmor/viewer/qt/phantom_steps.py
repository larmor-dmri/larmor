"""The Simulate workspace's panels: the steps of a phantom, and the page of one."""

from dataclasses import dataclass
from enum import StrEnum
from typing import override

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QIcon, QMouseEvent
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from larmor.viewer.model.events import (
    DataChanged,
    Event,
    ItemAdded,
    ItemRemoved,
    PhantomChanged,
    SourceChanged,
)
from larmor.viewer.model.phantom import (
    PhantomStep,
    ScanSettings,
    StepStatus,
    step_status,
)
from larmor.viewer.model.phantom_editor import item_sources
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.qt.phantom_panel import PhantomPanel
from larmor.viewer.qt.style import icon_path


class Extra(StrEnum):
    """A page of the steps that is not a step, open at any time."""

    TISSUE_MAPS = "tissue_maps"


type Page = PhantomStep | Extra


def default_scan_text() -> str:
    """Returns the sentence that describes the default scan, from the defaults."""
    scan = ScanSettings()
    kind = "single-shell" if len(scan.shells) == 1 else f"{len(scan.shells)}-shell"
    values = ", ".join(f"{b:g}" for b, _ in scan.shells)
    directions = sum(n for _, n in scan.shells)
    noise = "no noise" if scan.snr is None else f"noise at an SNR of {scan.snr:g}"
    return (
        f"The default scan is a {kind} acquisition at b = {values} s/mm^2 with "
        f"{directions} directions and {noise}."
    )


@dataclass(frozen=True)
class PageInfo:
    """Holds what a page of the steps shows: its name, its help and its section."""

    name: str
    text: str
    sections: tuple[str, ...]  # the panel's sections, by title


PAGES: dict[Page, PageInfo] = {
    PhantomStep.GRID: PageInfo(
        "Grid",
        "The grid is the box of voxels that the phantom is simulated on, and the "
        "box below works as it is. To line the simulation up with a loaded image, "
        "take the grid of that image instead.",
        ("Grid",),
    ),
    PhantomStep.BUNDLE: PageInfo(
        "Bundle",
        "Choose Draw bundle (N) and click in a 2D view to place the points of a "
        "bundle, one per click in that slice. Three or four points are enough, and "
        "Enter finishes the bundle.",
        ("Bundles",),
    ),
    PhantomStep.TISSUE: PageInfo(
        "Tissue",
        "Each bundle is filled with a tissue, and a new bundle gets WM, a standard "
        "model of white matter that works as it is. Change a tissue's parameters, "
        "or add one of your own, only if you want a different signal.",
        ("Tissues",),
    ),
    PhantomStep.SCAN: PageInfo(
        "Scan",
        default_scan_text()
        + " Add shell adds a b-value to the table, and More holds the timings, "
        "the sequence and the gradient files.",
        ("Scan",),
    ),
    PhantomStep.RUN: PageInfo(
        "Run",
        "Run starts the simulation in the background, once the other steps are "
        "ready. The images, the tissue masks and the true fibre directions open as "
        "items when it ends.",
        ("Run",),
    ),
    Extra.TISSUE_MAPS: PageInfo(
        "Tissue maps and background",
        "A tissue without a fibre direction, such as GM or CSF, can take its share "
        "of each voxel from a loaded overlay with values between 0 and 1. The "
        "background fills what the bundles and maps leave; a first phantom needs "
        "neither.",
        ("Grid",),
    ),
}
STEPS = tuple(PhantomStep)
MARK_SIZE = 16  # px, the check mark or circle of a step
ALL_SETTINGS_SIZE = (1420, 820)  # px, the All settings window when it opens
NO_PHANTOM = (
    "No phantom is open. Click New to start one on the grid of the main volume, "
    "or Open to read a phantom file or a simulation folder."
)


def _mark(ready: bool) -> QIcon:
    """Returns a step's mark, a check if it is ready, else a circle."""
    return QIcon(icon_path("step-ready" if ready else "step-waiting"))


class StepNavigator(QObject):
    """Holds the page of the steps shown, which the two panels of Simulate share.

    A new phantom starts again at the first step.
    """

    changed = Signal()

    def __init__(self, viewer: Viewer, parent: QObject | None = None) -> None:
        """Starts at the first step."""
        super().__init__(parent)
        self.viewer = viewer
        self.page: Page = PhantomStep.GRID
        self.last_step = PhantomStep.GRID  # where Back goes from an extra page
        state = viewer.phantom
        self._serial = state.serial if state is not None else None
        unsubscribe = viewer.scene.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: unsubscribe())

    def go_to(self, page: Page) -> None:
        """Shows a step, or an extra page; any of them at any time."""
        self.page = page
        if isinstance(page, PhantomStep):
            self.last_step = page
        self.changed.emit()

    def next(self) -> None:
        """Shows the step after the one shown."""
        if isinstance(self.page, PhantomStep) and self.page != STEPS[-1]:
            self.go_to(STEPS[STEPS.index(self.page) + 1])

    def back(self) -> None:
        """Shows the step before the one shown; from an extra page, the last step."""
        if isinstance(self.page, Extra):
            self.go_to(self.last_step)
        elif self.page != STEPS[0]:
            self.go_to(STEPS[STEPS.index(self.page) - 1])

    def _on_event(self, event: Event) -> None:
        """Goes back to the first step when a new phantom opens."""
        if isinstance(event, PhantomChanged):
            state = self.viewer.phantom
            serial = state.serial if state is not None else None
            if serial != self._serial:
                self._serial = serial
                self.go_to(PhantomStep.GRID)


class StepRow(QFrame):
    """Shows a step in the list with its mark, name and status; a click opens it."""

    clicked = Signal()

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        """Creates the row with its title, as "1. Grid"."""
        super().__init__(parent)
        self.setObjectName("stepRow")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mark = QLabel(self)
        self.mark.setFixedWidth(MARK_SIZE + 6)
        self.title = QLabel(title, self)
        self.status = QLabel(self)
        self.status.setObjectName("stepStatus")
        self.status.setWordWrap(True)
        text = QVBoxLayout()
        text.setSpacing(1)
        text.addWidget(self.title)
        text.addWidget(self.status)
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 6, 6, 6)
        row.addWidget(self.mark, 0, Qt.AlignmentFlag.AlignTop)
        row.addLayout(text, 1)

    def show_status(self, status: StepStatus, current: bool) -> None:
        """Shows where the step stands, and whether it is the page shown."""
        self.ready = status.ready
        self.mark.setPixmap(_mark(status.ready).pixmap(MARK_SIZE, MARK_SIZE))
        self.status.setText(status.text)
        font = self.title.font()
        font.setBold(current)
        self.title.setFont(font)
        if self.property("current") != current:
            self.setProperty("current", current)
            self.style().unpolish(self)  # the style sheet reads the property again
            self.style().polish(self)

    @override
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Opens the step."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class StepsPanel(QWidget):
    """Lists the five steps of a phantom with their status, and the optional pages.

    New, Open and Save sit at the top. The window carries out what needs a
    dialog: the signals ask it to.
    """

    new_requested = Signal()
    open_requested = Signal()
    open_simulation_requested = Signal()
    save_requested = Signal()
    breakdown_requested = Signal()
    all_settings_requested = Signal()

    def __init__(
        self, viewer: Viewer, steps: StepNavigator, parent: QWidget | None = None
    ) -> None:
        """Builds the file row, the steps and the links."""
        super().__init__(parent)
        self.viewer = viewer
        self.steps = steps
        self.setObjectName("stepRail")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 14)
        layout.setSpacing(4)
        layout.addLayout(self._build_files())
        self.hint = QLabel(NO_PHANTOM, self)
        self.hint.setObjectName("panelHint")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        title = QLabel("Steps", self)
        title.setObjectName("railTitle")
        layout.addWidget(title)
        self.rows: dict[PhantomStep, StepRow] = {}
        for number, step in enumerate(STEPS, 1):
            row = StepRow(f"{number}. {PAGES[step].name}", self)
            row.clicked.connect(lambda s=step: steps.go_to(s))
            self.rows[step] = row
            layout.addWidget(row)
        layout.addStretch(1)
        optional = QLabel("Optional", self)
        optional.setObjectName("railNote")
        layout.addWidget(optional)
        self.links: dict[str, QPushButton] = {}
        for key, text, slot in (
            ("breakdown", "Signal breakdown of a voxel", self.breakdown_requested.emit),
            (
                "maps",
                "Tissue maps and background",
                lambda: steps.go_to(Extra.TISSUE_MAPS),
            ),
            ("all", "All settings", self.all_settings_requested.emit),
        ):
            link = QPushButton(text, self)
            link.setObjectName("optionalStep")
            link.setCursor(Qt.CursorShape.PointingHandCursor)
            link.clicked.connect(slot)
            self.links[key] = link
            layout.addWidget(link)
        unsubscribe = viewer.scene.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: unsubscribe())
        steps.changed.connect(self.refresh)
        self.refresh()

    def _build_files(self) -> QHBoxLayout:
        """Builds the row of New, Open and Save."""
        row = QHBoxLayout()
        self.new_button = QPushButton("New", self)
        self.new_button.clicked.connect(self.new_requested)
        self.open_button = QToolButton(self)
        self.open_button.setObjectName("menuButton")
        self.open_button.setText("Open")
        self.open_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.open_button)
        menu.addAction("Phantom...", self.open_requested.emit)
        menu.addAction("Simulation folder...", self.open_simulation_requested.emit)
        self.open_button.setMenu(menu)
        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_requested)
        for widget in (self.new_button, self.open_button, self.save_button):
            row.addWidget(widget)
        row.addStretch(1)
        return row

    def refresh(self) -> None:
        """Shows where each step stands, and which page is open."""
        state = self.viewer.phantom
        self.hint.setVisible(state is None)
        self.save_button.setEnabled(state is not None)
        for row in self.rows.values():
            row.setVisible(state is not None)
        for link in self.links.values():
            link.setEnabled(state is not None)
        if state is None:
            return
        for status in step_status(state, item_sources(state, self.viewer.scene)):
            self.rows[status.step].show_status(status, status.step == self.steps.page)

    def _on_event(self, event: Event) -> None:
        """Shows the steps again after a change of the phantom or its items."""
        if isinstance(
            event,
            (PhantomChanged, ItemAdded, ItemRemoved, DataChanged, SourceChanged),
        ):
            self.refresh()


class StepPage(QWidget):
    """Shows the page of the step open: a heading, its help, its fields, Back, Next.

    The fields that matter show; the others are under a More fold.
    """

    def __init__(
        self, viewer: Viewer, steps: StepNavigator, parent: QWidget | None = None
    ) -> None:
        """Builds the page around a guided PhantomPanel."""
        super().__init__(parent)
        self.viewer = viewer
        self.steps = steps
        self.panel = PhantomPanel(viewer, self, guided=True)
        self.area = QScrollArea(self)
        self.area.setWidgetResizable(True)
        self.area.setFrameShape(QFrame.Shape.NoFrame)
        # The page fits the panel's width: its text wraps and its tables scroll.
        self.area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget(self.area)
        page = QVBoxLayout(content)
        page.setContentsMargins(12, 12, 12, 12)
        self.heading = QLabel(content)
        self.heading.setObjectName("stepHeading")
        self.text = QLabel(content)
        self.text.setObjectName("stepText")
        self.text.setWordWrap(True)
        page.addWidget(self.heading)
        page.addWidget(self.text)
        page.addWidget(self.panel, 1)
        self.area.setWidget(content)

        self._build_nav()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.area, 1)
        layout.addWidget(self.nav)
        steps.changed.connect(self.refresh)
        unsubscribe = viewer.scene.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: unsubscribe())
        self.refresh()

    def _build_nav(self) -> None:
        """Builds the bar of Back and Next."""
        steps = self.steps
        self.nav = QWidget(self)
        self.nav.setObjectName("stepNav")
        self.nav.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        row = QHBoxLayout(self.nav)
        row.setContentsMargins(12, 8, 12, 8)
        self.back_button = QPushButton("Back", self.nav)
        self.back_button.clicked.connect(steps.back)
        self.next_button = QPushButton("Next", self.nav)
        self.next_button.setObjectName("nextStep")
        self.next_button.clicked.connect(steps.next)
        row.addWidget(self.back_button)
        row.addStretch(1)
        row.addWidget(self.next_button)

    def refresh(self) -> None:
        """Shows the page open, its section, and where Back and Next go."""
        page = self.steps.page
        info = PAGES[page]
        shown = self.viewer.phantom is not None
        self.panel.show_sections(info.sections)
        if page == Extra.TISSUE_MAPS:
            self.panel.sections["Grid"].set_more_shown(True)
        for widget in (self.heading, self.text, self.nav):
            widget.setVisible(shown)
        self.text.setText(info.text)
        if isinstance(page, PhantomStep):
            number = STEPS.index(page)
            self.heading.setText(f"{number + 1}. {info.name}")
            first, last = number == 0, number == len(STEPS) - 1
            self.back_button.setText(
                "Back" if first else f"Back: {PAGES[STEPS[number - 1]].name}"
            )
            self.back_button.setEnabled(not first)
            self.next_button.setVisible(not last)
            if not last:
                self.next_button.setText(f"Next: {PAGES[STEPS[number + 1]].name}")
        else:
            self.heading.setText(info.name)
            self.back_button.setText(f"Back: {PAGES[self.steps.last_step].name}")
            self.back_button.setEnabled(True)
            self.next_button.setVisible(False)
        self.area.verticalScrollBar().setValue(0)

    def _on_event(self, event: Event) -> None:
        """Shows the page again when a phantom opens or closes."""
        if isinstance(event, PhantomChanged):
            shown = self.viewer.phantom is not None
            if self.heading.isVisible() != shown:
                self.refresh()


class AllSettingsWindow(QDialog):
    """Shows every section and field of the phantom editor in a window of its own."""

    def __init__(self, viewer: Viewer, parent: QWidget | None = None) -> None:
        """Creates the window around a PhantomPanel that shows everything."""
        super().__init__(parent)
        self.setWindowTitle("Phantom: all settings")
        self.setModal(False)
        self.panel = PhantomPanel(viewer, self)
        area = QScrollArea(self)
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setWidget(self.panel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(area)
        self.resize(*ALL_SETTINGS_SIZE)
