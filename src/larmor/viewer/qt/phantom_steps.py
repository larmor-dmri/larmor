"""The phantom editor, which shows the Phantom panel as guided steps or all at once."""

from dataclasses import dataclass
from enum import StrEnum
from typing import override

from PySide6.QtCore import QPoint, QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QMouseEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
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
from larmor.viewer.model.settings import (
    PhantomLayout,
    SettingChanged,
    SettingName,
    Settings,
)
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.qt.phantom_panel import PhantomPanel
from larmor.viewer.qt.style import icon_path


class Extra(StrEnum):
    """A page of the guided editor that is not a step, open at any time."""

    TISSUE_MAPS = "tissue_maps"
    PROXY_MAPS = "proxy_maps"


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
    """Holds what a page of the guided editor shows."""

    name: str
    text: str
    sections: tuple[str, ...]  # the panel's sections, by title


PAGES: dict[Page, PageInfo] = {
    PhantomStep.GRID: PageInfo(
        "Grid",
        "The grid is the box of voxels that the phantom is simulated on. The box "
        "below works as it is, so you can go on to the next step. To line the "
        "simulation up with a loaded image, take the grid of that image instead.",
        ("Grid",),
    ),
    PhantomStep.BUNDLE: PageInfo(
        "Bundle",
        "Click Draw (or press N), then click in a 2D view to place the points of "
        "the bundle; each click adds a point in that slice. Press Enter to finish "
        "the bundle, or Backspace to remove the last point. Three or four points "
        "are enough for a first bundle, and its shape settings work as they are.",
        ("Bundles",),
    ),
    PhantomStep.TISSUE: PageInfo(
        "Tissue",
        "Each bundle is filled with a tissue, which you pick for it on the Bundle "
        "step; this page edits the tissues themselves. A new bundle gets WM, a "
        "standard model of white matter, which works as it is. Change a tissue's "
        "parameters, or add a tissue of your own, only if you want a different "
        "signal.",
        ("Tissues",),
    ),
    PhantomStep.SCAN: PageInfo(
        "Scan",
        default_scan_text()
        + " To add a shell, click Add shell and type its b-value and number of "
        "directions in the table. The timings, the sequence and the noise are set "
        "in the fields above and below the table, or you can read the gradients "
        "from a pair of bvals and bvecs files.",
        ("Scan",),
    ),
    PhantomStep.RUN: PageInfo(
        "Run",
        "Once the steps are ready, Run (Ctrl+R) starts the simulation in the "
        "background. If no output folder is set, it first asks for a folder to "
        "save the results in. When the run ends, the images, the tissue masks and "
        "the true fibre directions open as items.",
        ("Run",),
    ),
    Extra.TISSUE_MAPS: PageInfo(
        "Tissue maps",
        "A tissue without a fibre direction, such as GM or CSF, can take its share "
        "of each voxel from a loaded overlay with values between 0 and 1 on the "
        "phantom's grid. Its map field is in the Grid section below, next to the "
        "Background, which fills what the bundles and maps leave. A first phantom "
        "needs neither.",
        ("Grid",),
    ),
    Extra.PROXY_MAPS: PageInfo(
        "Proxy maps",
        "Proxy maps are overlays that show what the phantom gives without running "
        "it, and they follow every edit. They let you check how a bundle fills "
        "the voxels, or a tissue's signal at b = 0, before a run.",
        ("Proxy maps",),
    ),
}
STEPS = tuple(PhantomStep)
GUIDED_WIDTH = 620  # px, the column of a step's sections
RAIL_WIDTH = 250  # px
ANY_WIDTH = 16_777_215  # Qt's largest widget size
MARK_SIZE = 16  # px, the check mark or circle of a step


def _mark(ready: bool) -> QIcon:
    """Returns a step's mark, a check if it is ready, else a circle."""
    return QIcon(icon_path("step-ready" if ready else "step-waiting"))


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


class PhantomEditor(QWidget):
    """Shows `PhantomPanel` as guided steps, one at a time, or every section at once.

    Guided lists the steps on the left and shows one step's sections; All settings
    shows every section under a row of the steps. The editor keeps only which page
    is open. The layout is the app setting PHANTOM_LAYOUT, changed by the switch
    at the top right and followed wherever it is changed.
    """

    def __init__(
        self, viewer: Viewer, settings: Settings, parent: QWidget | None = None
    ) -> None:
        """Builds the switch, the steps and the page around a new PhantomPanel."""
        super().__init__(parent)
        self.viewer = viewer
        self.settings = settings
        self.panel = PhantomPanel(viewer, self)
        self.page: Page = PhantomStep.GRID
        self._last_step = PhantomStep.GRID  # where Back goes from an extra page
        # The phantom shown: a new one starts again at the first step.
        state = viewer.phantom
        self._serial = state.serial if state is not None else None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addLayout(self._build_top())
        outer.addWidget(self._build_chips())
        body = QHBoxLayout()
        body.setContentsMargins(0, 8, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_rail())
        right = QVBoxLayout()
        right.setSpacing(0)
        right.addWidget(self._build_page(), 1)
        right.addWidget(self._build_nav())
        body.addLayout(right, 1)
        outer.addLayout(body, 1)

        # The viewer and the settings outlive the window: stop listening when it goes.
        unsubscribe = viewer.scene.subscribe(self._on_event)
        unsubscribe_settings = settings.subscribe(self._on_setting)
        self.destroyed.connect(lambda *_: (unsubscribe(), unsubscribe_settings()))
        self._lay_out()

    # Building

    def _build_top(self) -> QHBoxLayout:
        """Builds the top row: the panel's file buttons and the layout switch."""
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 14, 0)
        panel_layout = self.panel.layout()
        assert panel_layout is not None
        panel_layout.removeWidget(self.panel.file_row)
        top.addWidget(self.panel.file_row, 1)
        switch = QWidget(self)
        row = QHBoxLayout(switch)
        row.setContentsMargins(0, 8, 0, 0)
        row.setSpacing(0)
        self.guided_button = QToolButton(switch)
        self.guided_button.setText("Guided")
        self.guided_button.setToolTip("The steps to a first simulation, one at a time")
        self.all_button = QToolButton(switch)
        self.all_button.setText("All settings")
        self.all_button.setToolTip("Every section of the editor at once")
        group = QButtonGroup(switch)
        for button, layout in (
            (self.guided_button, PhantomLayout.GUIDED),
            (self.all_button, PhantomLayout.ALL),
        ):
            button.setObjectName("layoutSwitch")
            button.setCheckable(True)
            group.addButton(button)
            row.addWidget(button)
            button.clicked.connect(
                lambda _, chosen=layout: self.settings.set(
                    SettingName.PHANTOM_LAYOUT, chosen
                )
            )
        top.addWidget(switch, 0, Qt.AlignmentFlag.AlignTop)
        return top

    def _build_chips(self) -> QWidget:
        """Builds the row of steps above All settings; a click scrolls to one."""
        self.chips_row = QWidget(self)
        row = QHBoxLayout(self.chips_row)
        row.setContentsMargins(10, 6, 14, 2)
        row.setSpacing(4)
        self.chips: dict[PhantomStep, QToolButton] = {}
        for step in STEPS:
            chip = QToolButton(self.chips_row)
            chip.setObjectName("stepChip")
            chip.setText(PAGES[step].name)
            chip.setIconSize(QSize(MARK_SIZE - 2, MARK_SIZE - 2))
            chip.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.clicked.connect(lambda _, s=step: self.scroll_to(s))
            self.chips[step] = chip
            row.addWidget(chip)
        row.addStretch(1)
        note = QLabel("Click a step to go to its section", self.chips_row)
        note.setObjectName("railNote")
        row.addWidget(note)
        return self.chips_row

    def _build_rail(self) -> QWidget:
        """Builds the list of steps and extra pages on the left of Guided."""
        self.rail = QWidget(self)
        self.rail.setObjectName("stepRail")
        self.rail.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.rail.setFixedWidth(RAIL_WIDTH)
        layout = QVBoxLayout(self.rail)
        layout.setContentsMargins(10, 14, 10, 14)
        layout.setSpacing(4)
        title = QLabel("First simulation", self.rail)
        title.setObjectName("railTitle")
        layout.addWidget(title)
        note = QLabel(
            "A simulation setup consists of five steps. When done, a blue checkmark "
            "is shown. You can open any step at any time.",
            self.rail,
        )
        note.setObjectName("railNote")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addSpacing(8)
        self.rows: dict[PhantomStep, StepRow] = {}
        for number, step in enumerate(STEPS, 1):
            row = StepRow(f"{number}. {PAGES[step].name}", self.rail)
            row.clicked.connect(lambda s=step: self.go_to(s))
            self.rows[step] = row
            layout.addWidget(row)
        layout.addStretch(1)
        optional = QLabel("Optional, any time", self.rail)
        optional.setObjectName("railNote")
        layout.addWidget(optional)
        self.extras: dict[Extra, QPushButton] = {}
        for extra in Extra:
            link = QPushButton(PAGES[extra].name, self.rail)
            link.setObjectName("optionalStep")
            link.setCursor(Qt.CursorShape.PointingHandCursor)
            link.clicked.connect(lambda _, e=extra: self.go_to(e))
            self.extras[extra] = link
            layout.addWidget(link)
        return self.rail

    def _build_page(self) -> QScrollArea:
        """Builds the scroll area of the page: heading, text and the panel."""
        self.page_area = QScrollArea(self)
        self.page_area.setWidgetResizable(True)
        self.page_area.setFrameShape(QFrame.Shape.NoFrame)
        self.content = content = QWidget(self.page_area)
        self.page_layout = QVBoxLayout(content)
        self.heading = QLabel(content)
        self.heading.setObjectName("stepHeading")
        self.text = QLabel(content)
        self.text.setObjectName("stepText")
        self.text.setWordWrap(True)
        self.page_layout.addWidget(self.heading)
        self.page_layout.addWidget(self.text)
        # Up to GUIDED_WIDTH in Guided, as wide as the window in All settings.
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.panel, 1)
        row.addStretch(0)
        self.page_layout.addLayout(row, 1)
        self.page_area.setWidget(content)
        return self.page_area

    def _build_nav(self) -> QWidget:
        """Builds the Back and Next buttons."""
        self.nav = QWidget(self)
        self.nav.setObjectName("stepNav")
        self.nav.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        row = QHBoxLayout(self.nav)
        row.setContentsMargins(24, 8, 24, 8)
        self.back_button = QPushButton("Back", self.nav)
        self.back_button.clicked.connect(self.back)
        self.next_button = QPushButton("Next", self.nav)
        self.next_button.setObjectName("nextStep")
        self.next_button.clicked.connect(self.next)
        row.addWidget(self.back_button)
        row.addStretch(1)
        row.addWidget(self.next_button)
        return self.nav

    # Moving between the steps

    @property
    def guided(self) -> bool:
        """Returns whether the editor shows the guided steps."""
        return self.settings.values.phantom_layout == PhantomLayout.GUIDED

    def go_to(self, page: Page) -> None:
        """Shows a step, or an extra page, in Guided; any of them at any time."""
        self.page = page
        if isinstance(page, PhantomStep):
            self._last_step = page
        self._lay_out()
        self.page_area.verticalScrollBar().setValue(0)

    def next(self) -> None:
        """Shows the step after the one shown."""
        if isinstance(self.page, PhantomStep) and self.page != STEPS[-1]:
            self.go_to(STEPS[STEPS.index(self.page) + 1])

    def back(self) -> None:
        """Shows the step before the one shown; from an extra page, the last step."""
        if isinstance(self.page, Extra):
            self.go_to(self._last_step)
        elif self.page != STEPS[0]:
            self.go_to(STEPS[STEPS.index(self.page) - 1])

    def scroll_to(self, step: PhantomStep) -> None:
        """Scrolls All settings to the step's section, unfolded."""
        section = self.panel.sections[PAGES[step].sections[0]]
        section.header.setChecked(True)
        top = section.mapTo(self.content, QPoint(0, 0)).y()
        self.page_area.verticalScrollBar().setValue(top)

    # Showing

    def _on_event(self, event: Event) -> None:
        """Starts again at the first step for a new phantom; else shows the steps."""
        if isinstance(event, PhantomChanged):
            state = self.viewer.phantom
            serial = state.serial if state is not None else None
            if serial != self._serial:
                self._serial = serial
                self.page = self._last_step = PhantomStep.GRID  # a new phantom
                self._lay_out()
            else:
                self._show_steps()
        elif isinstance(event, (ItemAdded, ItemRemoved, DataChanged, SourceChanged)):
            self._show_steps()  # a loaded bundle or map may have gone or been saved

    def _on_setting(self, event: SettingChanged) -> None:
        """Shows the new layout when the setting changes."""
        if event.name == SettingName.PHANTOM_LAYOUT:
            self._lay_out()

    def _lay_out(self) -> None:
        """Shows the layout chosen, with the page open in Guided."""
        guided = self.guided
        with QSignalBlocker(self.guided_button), QSignalBlocker(self.all_button):
            self.guided_button.setChecked(guided)
            self.all_button.setChecked(not guided)
        self.panel.setMaximumWidth(GUIDED_WIDTH if guided else ANY_WIDTH)
        margins = (24, 12, 24, 12) if guided else (0, 0, 0, 0)
        self.page_layout.setContentsMargins(*margins)
        self.panel.show_sections(PAGES[self.page].sections if guided else None)
        self._show_steps()

    def _show_steps(self) -> None:
        """Shows where each step stands, and the parts of the layout chosen."""
        state = self.viewer.phantom
        guided = self.guided and state is not None
        self.rail.setVisible(guided)
        self.nav.setVisible(guided)
        self.heading.setVisible(guided)
        self.text.setVisible(guided)
        self.chips_row.setVisible(not self.guided and state is not None)
        if state is None:
            return
        for status in step_status(state, item_sources(state, self.viewer.scene)):
            self.rows[status.step].show_status(status, status.step == self.page)
            self.chips[status.step].setIcon(_mark(status.ready))
            self.chips[status.step].setToolTip(status.text)
        info = PAGES[self.page]
        if isinstance(self.page, PhantomStep):
            number = STEPS.index(self.page)
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
            self.back_button.setText(f"Back: {PAGES[self._last_step].name}")
            self.back_button.setEnabled(True)
            self.next_button.setVisible(False)
        self.text.setText(info.text)
