"""The Dissect workspace's panels: the steps to a bundle, and the page of one."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from larmor.types import Volume
from larmor.viewer.jobs import CANCELLING, JobStatus
from larmor.viewer.model.dissect import (
    CSF_DIFFUSIVITY,
    DISSECT_STEPS,
    FOD_OPTION_METHODS,
    GM_DIFFUSIVITY,
    STREAMLINES,
    Dissection,
    DissectStep,
    DwiFiles,
    FodMerge,
    FodMethod,
    FodSettings,
    FodSolver,
    FodSource,
    GreyMatter,
    Seeding,
    StartFrom,
    TrackAlgorithm,
    TrackSettings,
    dissect_status,
    start_items,
    two_fods,
)
from larmor.viewer.model.events import (
    DataChanged,
    Event,
    ItemAdded,
    ItemRemoved,
    ItemRenamed,
    OptionChanged,
    SourceChanged,
)
from larmor.viewer.model.item import Category, ItemId
from larmor.viewer.model.options import BASES, Basis, OptionName, OverlayOptions
from larmor.viewer.model.phantom import BvecFrame
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.qt.parcellation_section import ParcellationSection
from larmor.viewer.qt.phantom_panel import ChoiceBox, Section
from larmor.viewer.qt.phantom_steps import StepRow
from larmor.viewer.qt.regions_editor import RegionsEditor


class MaskTarget(StrEnum):
    """The mask field a "New mask..." link fills."""

    FIT = "fit"  # the FODs step's mask
    TRACK = "track"  # the Track step's tracking mask
    REGION = "region"  # a mask region of the Regions step


@dataclass(frozen=True)
class StepInfo:
    """Holds a step's name and the sentences its page starts with."""

    name: str
    text: str


STEP_INFO: dict[DissectStep, StepInfo] = {
    DissectStep.START: StepInfo(
        "Start",
        "Choose what you have. A tractogram goes straight to the regions, a FOD "
        "file is tracked, and a DWI first has its FODs estimated.",
    ),
    DissectStep.FODS: StepInfo(
        "FODs",
        "The FODs are what the tracking follows. Load a FOD file, or estimate the "
        "FODs from the DWI inside a brain mask.",
    ),
    DissectStep.TRACK: StepInfo(
        "Track",
        "Whole-brain tracking through the FODs, with the library's defaults. The "
        "run's settings and random seed are kept, so the Save step can write them.",
    ),
    DissectStep.REGIONS: StepInfo(
        "Regions",
        "Place regions with the Sphere ROI (O), Box ROI (B) and Label region (L) "
        "tools. Ticked regions keep the streamlines that pass through them.",
    ),
    DissectStep.SAVE: StepInfo(
        "Save",
        "Writes the kept streamlines, the regions, and the tracking run when the "
        "tractogram was tracked here, with the files they need.",
    ),
}
# A number field left empty takes the library's default, which it says.
AUTO = "auto"


class DissectNavigator(QObject):
    """Holds the step shown, which the two panels of Dissect share.

    Next and Back pass over the skipped steps.
    """

    changed = Signal()

    def __init__(self, state: Dissection, parent: QObject | None = None) -> None:
        """Starts at the Start step."""
        super().__init__(parent)
        self.state = state
        self.page = DissectStep.START

    def go_to(self, step: DissectStep) -> None:
        """Shows a step, unless it is skipped."""
        if not self.state.skipped(step):
            self.page = step
        self.changed.emit()

    def _neighbour(self, offset: int) -> DissectStep | None:
        """Returns the first step not skipped in a direction; None at an end."""
        index = DISSECT_STEPS.index(self.page) + offset
        while 0 <= index < len(DISSECT_STEPS):
            if not self.state.skipped(DISSECT_STEPS[index]):
                return DISSECT_STEPS[index]
            index += offset
        return None

    def next_step(self) -> DissectStep | None:
        """Returns the step Next goes to; None at the last."""
        return self._neighbour(1)

    def previous_step(self) -> DissectStep | None:
        """Returns the step Back goes to; None at the first."""
        return self._neighbour(-1)

    def next(self) -> None:
        """Shows the next step not skipped."""
        step = self.next_step()
        if step is not None:
            self.go_to(step)

    def back(self) -> None:
        """Shows the step before, not skipped."""
        step = self.previous_step()
        if step is not None:
            self.go_to(step)


class DissectStepsPanel(QWidget):
    """Lists the five steps of a dissection with their marks and status lines.

    A skipped step stays listed, greyed out.
    """

    def __init__(
        self,
        viewer: Viewer,
        steps: DissectNavigator,
        parent: QWidget | None = None,
    ) -> None:
        """Builds the list of steps."""
        super().__init__(parent)
        self.viewer = viewer
        self.steps = steps
        self.setObjectName("stepRail")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 14)
        layout.setSpacing(4)
        title = QLabel("Steps", self)
        title.setObjectName("railTitle")
        layout.addWidget(title)
        self.rows: dict[DissectStep, StepRow] = {}
        for number, step in enumerate(DISSECT_STEPS, 1):
            row = StepRow(f"{number}. {STEP_INFO[step].name}", self)
            row.clicked.connect(lambda s=step: steps.go_to(s))
            self.rows[step] = row
            layout.addWidget(row)
        layout.addStretch(1)
        unsubscribe = viewer.scene.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: unsubscribe())
        steps.changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        """Shows where each step stands, and which is open."""
        state = self.steps.state
        state.forget_missing(self.viewer.scene)
        for status in dissect_status(state, self.viewer.scene):
            row = self.rows[status.step]
            row.show_status(status, status.step == self.steps.page)
            row.setEnabled(not status.skipped)

    def _on_event(self, event: Event) -> None:
        """Shows the steps again after a change of the items they name."""
        if isinstance(
            event,
            (ItemAdded, ItemRemoved, ItemRenamed, DataChanged, SourceChanged),
        ) or _changes_regions(event):
            self.refresh()


class FileRow(QWidget):
    """Holds a file name in a text field, with a button that asks for a file."""

    def __init__(
        self, title: str, placeholder: str = "", parent: QWidget | None = None
    ) -> None:
        """Creates the row; title names the file dialog."""
        super().__init__(parent)
        self.title = title
        self.text = QLineEdit(self)
        self.text.setPlaceholderText(placeholder)
        self.button = QPushButton("Browse...", self)
        self.button.clicked.connect(self._browse)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.text, 1)
        row.addWidget(self.button)

    def _browse(self) -> None:
        """Asks for a file and shows it."""
        path, _ = QFileDialog.getOpenFileName(self, self.title)
        if path:
            self.set_path(Path(path))

    def set_path(self, path: Path | None) -> None:
        """Shows a file; None empties the field."""
        self.text.setText("" if path is None else str(path))

    def path(self) -> Path | None:
        """Returns the file shown; None if the field is empty."""
        text = self.text.text().strip()
        return Path(text) if text else None


class NumberField(QLineEdit):
    """Edits an optional number; empty means the library's default."""

    def __init__(self, parent: QWidget | None = None, *, whole: bool = False) -> None:
        """Creates the field; whole asks for a whole number."""
        super().__init__(parent)
        self.whole = whole
        self.setPlaceholderText(AUTO)

    def value(self) -> float | None:
        """Returns the number typed; None if the field is empty.

        Raises:
            ValueError: If the text is not a number.
        """
        text = self.text().strip()
        if not text:
            return None
        return int(text) if self.whole else float(text)


class JobBox(QWidget):
    """Shows a running job of a step, its progress and Cancel; hidden otherwise."""

    cancel_requested = Signal()

    def __init__(self, name: str, parent: QWidget | None = None) -> None:
        """Creates the box for the jobs called name."""
        super().__init__(parent)
        self.name = name
        self.label = QLabel(self)
        self.progress = QProgressBar(self)
        self.progress.setTextVisible(False)
        self.cancel = QPushButton("Cancel", self)
        self.cancel.clicked.connect(self.cancel_requested)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.addWidget(self.label)
        row = QHBoxLayout()
        row.addWidget(self.progress, 1)
        row.addWidget(self.cancel)
        layout.addLayout(row)
        self.hide()

    def show_job(self, status: JobStatus | None) -> None:
        """Shows how far this box's job is; hides the box for any other."""
        if status is None or status.name != self.name:
            self.hide()
            return
        self.label.setText(status.stage or status.name)
        self.cancel.setEnabled(status.stage != CANCELLING)
        if status.total > 0:
            self.progress.setRange(0, 1000)
            self.progress.setValue(round(status.fraction * 1000))
        else:
            self.progress.setRange(0, 0)
        self.show()


# The tractogram options that change which streamlines its regions keep.
REGION_OPTIONS = (OptionName.REGIONS, OptionName.REGION_FILTER, OptionName.ENDS_ONLY)


def _changes_regions(event: Event) -> bool:
    """Tells whether an event can change the streamlines the regions keep."""
    if not isinstance(event, OptionChanged):
        return False
    return event.item.category == Category.REGIONS or event.name in REGION_OPTIONS


def _link(text: str, parent: QWidget) -> QPushButton:
    """Returns a button drawn as a link, such as "New mask..."."""
    link = QPushButton(text, parent)
    link.setObjectName("optionalStep")
    link.setCursor(Qt.CursorShape.PointingHandCursor)
    return link


def image_choices(viewer: Viewer) -> list[tuple[object, str]]:
    """Returns the 3D images of the scene, overlays first, as (id, name) pairs."""
    found: list[tuple[object, str]] = []
    for category in (Category.OVERLAYS, Category.VOLUMES):
        for item in viewer.scene.in_category(category):
            if isinstance(item.data, Volume) and item.data.data.ndim == 3:
                found.append((item.id, item.name))
    return found


class StartPage(QWidget):
    """Offers the three starts: a tractogram, a FOD file, or a DWI."""

    start_requested = Signal(object, object)  # the StartFrom and the file
    item_chosen = Signal(object, object)  # the StartFrom and the item's id

    def __init__(self, viewer: Viewer, parent: QWidget | None = None) -> None:
        """Creates the three buttons."""
        super().__init__(parent)
        self.viewer = viewer
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.buttons: dict[StartFrom, QPushButton] = {}
        for start, text, tip in (
            (
                StartFrom.TRACTOGRAM,
                "A tractogram...",
                "Dissect a tractogram file; the FODs and Track steps are skipped",
            ),
            (StartFrom.FOD, "A FOD file...", "Track a FOD file, then dissect"),
            (
                StartFrom.DWI,
                "A DWI...",
                "Estimate FODs from a DWI, track them, then dissect",
            ),
        ):
            button = QPushButton(text, self)
            button.setToolTip(tip)
            button.clicked.connect(lambda _, s=start: self._ask(s))
            layout.addWidget(button)
            self.buttons[start] = button
        layout.addStretch(1)

    def _ask(self, start: StartFrom) -> None:
        """Offers the scene's items of a start, or asks for its file.

        With such items open, a menu lists them above "From a file...".
        """
        items = start_items(self.viewer.scene, start)
        if not items:
            self._ask_file(start)
            return
        menu = self.start_menu(start, items)
        button = self.buttons[start]
        menu.exec(button.mapToGlobal(button.rect().bottomLeft()))

    def start_menu(self, start: StartFrom, items: list[ItemId]) -> QMenu:
        """Returns the menu of a start: its items, then "From a file..."."""
        menu = QMenu(self)
        for item_id in items:
            name = self.viewer.scene.item(item_id).name
            action = menu.addAction(name)
            action.triggered.connect(
                lambda _=False, i=item_id: self.item_chosen.emit(start, i)
            )
        menu.addSeparator()
        menu.addAction("From a file...").triggered.connect(
            lambda _=False: self._ask_file(start)
        )
        return menu

    def _ask_file(self, start: StartFrom) -> None:
        """Asks for the file of a start, then asks the window to start from it."""
        path, _ = QFileDialog.getOpenFileName(self, self.buttons[start].text())
        if path:
            self.start_requested.emit(start, Path(path))


class FodsPage(QWidget):
    """Loads a FOD file, or estimates the FODs from the DWI, as the user chooses."""

    load_requested = Signal(object, object)  # the file and its Basis
    estimate_requested = Signal(object, object, object)  # DwiFiles, mask id, settings
    new_mask_requested = Signal(object)  # the MaskTarget
    save_requested = Signal(object)  # the FOD item's id
    refused = Signal(str)

    def __init__(
        self, viewer: Viewer, state: Dissection, parent: QWidget | None = None
    ) -> None:
        """Builds both routes; the dissection's choice shows one of them."""
        super().__init__(parent)
        self.viewer = viewer
        self.state = state
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        switch = QHBoxLayout()
        self.load_choice = QRadioButton("Load a FOD file", self)
        self.estimate_choice = QRadioButton("Estimate from the DWI", self)
        group = QButtonGroup(self)
        for button in (self.load_choice, self.estimate_choice):
            group.addButton(button)
            switch.addWidget(button)
        self.load_choice.toggled.connect(lambda _: self._show_route())
        layout.addLayout(switch)
        self.load_section = self._build_load()
        self.estimate_section = self._build_estimate()
        layout.addWidget(self.load_section)
        layout.addWidget(self.estimate_section)
        self.job = JobBox("Estimating FODs", self)
        layout.addWidget(self.job)
        self.result = QLabel(self)
        self.result.setWordWrap(True)
        self.save_button = QPushButton("Save...", self)
        self.save_button.setToolTip("Save the FOD to a file")
        self.save_button.clicked.connect(
            lambda: self.save_requested.emit(self.state.fod)
        )
        layout.addWidget(self.result)
        layout.addWidget(self.save_button)
        layout.addStretch(1)
        self.refresh()

    def _build_load(self) -> Section:
        """Builds the route that loads a FOD file and its basis."""
        section = Section("Load")
        section.set_header_visible(False)
        self.fod_file = FileRow("FOD file", parent=section)
        section.form.addRow("FOD file", self.fod_file)
        self.basis = ChoiceBox(section)
        self.basis.set_choices([(b, b.value) for b in BASES])
        self.basis.show_value(Basis.TOURNIER)
        section.form.addRow("SH basis", self.basis)
        self.load_button = QPushButton("Load", section)
        self.load_button.clicked.connect(self._load)
        section.form.addRow(self.load_button)
        return section

    def _build_estimate(self) -> Section:
        """Builds the route that fits FODs, the rarer choices under More."""
        section = Section("Estimate")
        section.set_header_visible(False)
        form = section.form
        self.dwi = FileRow("DWI", parent=section)
        self.bvals = FileRow("bvals", "from the header", section)
        self.bvecs = FileRow("bvecs", "from the header", section)
        form.addRow("DWI", self.dwi)
        form.addRow("bvals", self.bvals)
        form.addRow("bvecs", self.bvecs)
        mask = QHBoxLayout()
        self.mask_box = ChoiceBox(section)
        new_mask = _link("New mask...", section)
        new_mask.clicked.connect(lambda: self.new_mask_requested.emit(MaskTarget.FIT))
        mask.addWidget(self.mask_box, 1)
        mask.addWidget(new_mask)
        form.addRow("Mask", mask)
        self.method = ChoiceBox(section)
        self.method.set_choices([(m, m.value) for m in FodMethod])
        self.method.show_value(FodMethod.GRL)
        self.method.chosen.connect(lambda _: self._show_method_options())
        form.addRow("Method", self.method)
        self.lmax = QSpinBox(section)
        self.lmax.setRange(2, 16)
        self.lmax.setSingleStep(2)
        self.lmax.setValue(8)
        form.addRow("Lmax", self.lmax)
        self._build_estimate_more(section)
        self.estimate_button = QPushButton("Estimate", section)
        self.estimate_button.clicked.connect(self._estimate)
        form.addRow(self.estimate_button)
        return section

    def _build_estimate_more(self, section: Section) -> None:
        """Adds the responses, the solver and the bvecs frame under More."""
        more = self.more_form = section.more_form
        self.grey_matter = ChoiceBox(section)
        self.grey_matter.set_choices(
            [
                (GreyMatter.AUTO, "auto (NODDI for mFOD)"),
                (GreyMatter.ISOTROPIC, "isotropic"),
                (GreyMatter.NODDI, "NODDI"),
                (GreyMatter.NONE, "none"),
            ]
        )
        self.grey_matter.show_value(GreyMatter.AUTO)
        self.grey_matter.chosen.connect(lambda _: self._show_method_options())
        more.addRow("Grey matter", self.grey_matter)
        self.fod_merge = ChoiceBox(section)
        self.fod_merge.set_choices(
            [
                (FodMerge.WS, "weighted sum (mFOD-WS)"),
                (FodMerge.MAJORITY, "majority"),
                (FodMerge.NONE, "none: keep both"),
            ]
        )
        self.fod_merge.show_value(FodMerge.WS)
        self.fod_merge.setToolTip(
            "How the white and grey matter FODs of a NODDI fit are made one FOD to "
            "track: each times its tissue fraction and summed, or in each voxel the "
            "one of the larger fraction"
        )
        more.addRow("Merge FODs", self.fod_merge)
        self.gm_diffusivity = QLineEdit(f"{GM_DIFFUSIVITY:g}", section)
        more.addRow("GM diffusivity (mm^2/s)", self.gm_diffusivity)
        self.csf_diffusivity = QLineEdit(f"{CSF_DIFFUSIVITY:g}", section)
        self.csf_diffusivity.setPlaceholderText("none")
        more.addRow("CSF diffusivity (mm^2/s)", self.csf_diffusivity)
        self.response = FileRow("CSD response", "estimated", section)
        more.addRow("CSD response", self.response)
        self.tissue_responses = {
            name: FileRow(f"{name} response", "estimated", section)
            for name in ("WM", "GM", "CSF")
        }
        for name, row in self.tissue_responses.items():
            row.setToolTip(f"The {name} response of MSMT-CSD, a text file")
            more.addRow(f"{name} response", row)
        self.solver = ChoiceBox(section)
        self.solver.set_choices(
            [(None, "the method's own"), *[(s, s.value) for s in FodSolver]]
        )
        self.solver.show_value(None)
        more.addRow("Solver", self.solver)
        self.reg = NumberField(section)
        more.addRow("Penalty weight", self.reg)
        self.inner_shell_weight = NumberField(section)
        more.addRow("Inner-shell weight", self.inner_shell_weight)
        self.frame = ChoiceBox(section)
        self.frame.set_choices(
            [(BvecFrame.FSL, "FSL (grid axes)"), (BvecFrame.WORLD, "World (RAS)")]
        )
        self.frame.show_value(BvecFrame.FSL)
        more.addRow("bvecs frame", self.frame)
        self.method_rows: dict[str, QWidget] = {
            "grey_matter": self.grey_matter,
            "gm_diffusivity": self.gm_diffusivity,
            "csf_diffusivity": self.csf_diffusivity,
            "response": self.response,
            "wm_response": self.tissue_responses["WM"],
            "gm_response": self.tissue_responses["GM"],
            "csf_response": self.tissue_responses["CSF"],
            "solver": self.solver,
            "reg": self.reg,
            "inner_shell_weight": self.inner_shell_weight,
        }
        self._show_method_options()
        section.set_more_shown(False)

    def _applies(self, name: str) -> bool:
        """Tells whether a choice of the fit applies to the method chosen."""
        return self.method.value() in FOD_OPTION_METHODS[name][1]

    def _show_method_options(self) -> None:
        """Shows under More only the choices the method chosen takes."""
        for name, row in self.method_rows.items():
            self.more_form.setRowVisible(row, self._applies(name))
        self.more_form.setRowVisible(self.fod_merge, self._two_fods())

    def _two_fods(self) -> bool:
        """Tells whether the fit chosen has two FODs, a NODDI grey matter's."""
        method = self.method.value()
        assert isinstance(method, FodMethod)
        grey = self.grey_matter.value()
        assert isinstance(grey, GreyMatter)
        return two_fods(FodSettings(method, grey_matter=grey))

    def refresh(self) -> None:
        """Shows the route the dissection chose, its DWI, the masks and the result."""
        state = self.state
        loading = state.fod_source == FodSource.LOAD
        self.estimate_choice.setEnabled(state.start != StartFrom.FOD)
        button = self.load_choice if loading else self.estimate_choice
        button.blockSignals(True)
        button.setChecked(True)
        button.blockSignals(False)
        self._show_route()
        self.mask_box.set_choices(image_choices(self.viewer))
        if self.mask_box.value() is None and state.fit_mask is not None:
            self.mask_box.show_value(state.fit_mask)
        present = {item.id for item in self.viewer.scene.items}
        fod = state.fod if state.fod in present else None
        if fod is None:
            self.result.setText("No FOD yet.")
        else:
            item = self.viewer.scene.item(fod)
            where = "unsaved" if item.source is None else str(item.source)
            self.result.setText(f"The Track step uses {item.name} ({where}).")
        self.save_button.setVisible(fod is not None)

    def show_dwi(self, dwi: DwiFiles | None) -> None:
        """Fills the DWI's fields."""
        if dwi is None:
            return
        self.dwi.set_path(dwi.path)
        self.bvals.set_path(dwi.bvals)
        self.bvecs.set_path(dwi.bvecs)
        self.frame.show_value(dwi.frame)

    def _show_route(self) -> None:
        """Shows the fields of the route chosen."""
        loading = self.load_choice.isChecked()
        self.state.fod_source = FodSource.LOAD if loading else FodSource.ESTIMATE
        self.load_section.setVisible(loading)
        self.estimate_section.setVisible(not loading)

    def _load(self) -> None:
        """Asks the window to load the FOD file."""
        path = self.fod_file.path()
        if path is None:
            self.refused.emit("Choose a FOD file first.")
            return
        basis = self.basis.value()
        self.load_requested.emit(path, basis if basis is not None else Basis.TOURNIER)

    def settings(self) -> FodSettings:
        """Returns the choices of the fit.

        Raises:
            ValueError: If a number cannot be read.
        """
        method = self.method.value()
        assert isinstance(method, FodMethod)
        rows = self.tissue_responses
        readers: dict[str, Callable[[], Any]] = {
            "solver": self.solver.value,
            "reg": self.reg.value,
            "inner_shell_weight": self.inner_shell_weight.value,
            "grey_matter": self.grey_matter.value,
            "gm_diffusivity": lambda: float(self.gm_diffusivity.text()),
            "csf_diffusivity": self._csf_diffusivity,
            "response": self.response.path,
            "wm_response": rows["WM"].path,
            "gm_response": rows["GM"].path,
            "csf_response": rows["CSF"].path,
        }
        # A hidden row keeps the default, so it never reaches the fit.
        chosen = {n: read() for n, read in readers.items() if self._applies(n)}
        if self._two_fods():
            chosen["fod_merge"] = self.fod_merge.value()
        return replace(FodSettings(method, self.lmax.value()), **chosen)

    def _csf_diffusivity(self) -> float | None:
        """Returns the CSF diffusivity typed; None for none."""
        csf = self.csf_diffusivity.text().strip()
        return float(csf) if csf and csf != "none" else None

    def _estimate(self) -> None:
        """Asks the window to fit the FODs with the choices on the page."""
        path = self.dwi.path()
        mask = self.mask_box.value()
        if path is None:
            self.refused.emit("Choose the DWI first.")
            return
        if not isinstance(mask, int):
            self.refused.emit("Choose a brain mask first, or make one.")
            return
        try:
            settings = self.settings()
        except ValueError as error:
            self.refused.emit(f"A number cannot be read: {error}")
            return
        frame = self.frame.value()
        assert isinstance(frame, BvecFrame)
        dwi = DwiFiles(path, self.bvals.path(), self.bvecs.path(), frame)
        self.estimate_requested.emit(dwi, mask, settings)


class TrackPage(QWidget):
    """Sets up a whole-brain tracking run through the FODs, and starts it."""

    track_requested = Signal(object, object, object, object)  # FOD, seeds, mask, set
    fod_chosen = Signal(object)  # the FOD item's id
    new_mask_requested = Signal(object)  # the MaskTarget
    save_requested = Signal(object)  # the tractogram item's id
    refused = Signal(str)

    def __init__(
        self, viewer: Viewer, state: Dissection, parent: QWidget | None = None
    ) -> None:
        """Builds the fields, the seeding and the mask under More."""
        super().__init__(parent)
        self.viewer = viewer
        self.state = state
        self._defaulted: ItemId | None = None  # the fit mask last chosen for both
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.fod_label = QLabel(self)
        self.fod_label.setWordWrap(True)
        layout.addWidget(self.fod_label)
        # Shown when there are several FODs, such as both of a NODDI fit.
        self.fod_box = ChoiceBox(self)
        self.fod_box.setToolTip("The FOD to track")
        self.fod_box.chosen.connect(self._choose_fod)
        layout.addWidget(self.fod_box)
        section = Section("Track")
        section.set_header_visible(False)
        self._build_fields(section)
        self._build_more(section)
        self.track_button = QPushButton("Track", section)
        self.track_button.clicked.connect(self._track)
        section.form.addRow(self.track_button)
        layout.addWidget(section)
        self.job = JobBox("Tracking", self)
        layout.addWidget(self.job)
        self.result = QLabel(self)
        self.result.setWordWrap(True)
        self.save_button = QPushButton("Save...", self)
        self.save_button.setToolTip("Save the whole-brain tractogram to a file")
        self.save_button.clicked.connect(
            lambda: self.save_requested.emit(self.state.tractogram)
        )
        layout.addWidget(self.result)
        layout.addWidget(self.save_button)
        layout.addStretch(1)
        self.refresh()

    def _build_fields(self, section: Section) -> None:
        """Adds the algorithm, the number of streamlines, step, angle and lengths."""
        form = section.form
        self.algorithm = ChoiceBox(section)
        self.algorithm.set_choices(
            [
                (TrackAlgorithm.DET, "deterministic"),
                (TrackAlgorithm.PROB, "probabilistic"),
            ]
        )
        self.algorithm.show_value(TrackAlgorithm.DET)
        form.addRow("Algorithm", self.algorithm)
        self.streamlines = QSpinBox(section)
        self.streamlines.setRange(1, 100_000_000)
        self.streamlines.setGroupSeparatorShown(True)
        self.streamlines.setValue(STREAMLINES)
        form.addRow("Streamlines", self.streamlines)
        self.numbers: dict[str, NumberField] = {}
        for key, label in (
            ("step_size", "Step size (mm)"),
            ("angle", "Angle (degrees)"),
            ("cutoff", "Cutoff"),
            ("min_length", "Min length (mm)"),
            ("max_length", "Max length (mm)"),
        ):
            self.numbers[key] = NumberField(section)
            form.addRow(label, self.numbers[key])

    def _build_more(self, section: Section) -> None:
        """Adds the seeding, the masks and the random seed under More."""
        more = section.more_form
        self.seeding = ChoiceBox(section)
        self.seeding.set_choices(
            [
                (Seeding.WHOLE_MASK, "whole seed mask"),
                (Seeding.PER_VOXEL, "seeds per voxel"),
            ]
        )
        self.seeding.show_value(Seeding.WHOLE_MASK)
        more.addRow("Seeding", self.seeding)
        self.per_voxel = QSpinBox(section)
        self.per_voxel.setRange(1, 1000)
        more.addRow("Seeds per voxel", self.per_voxel)
        self.seed_mask = ChoiceBox(section)
        more.addRow("Seed mask", self.seed_mask)
        mask = QHBoxLayout()
        self.mask_box = ChoiceBox(section)
        new_mask = _link("New mask...", section)
        new_mask.clicked.connect(lambda: self.new_mask_requested.emit(MaskTarget.TRACK))
        mask.addWidget(self.mask_box, 1)
        mask.addWidget(new_mask)
        more.addRow("Tracking mask", mask)
        self.seed = NumberField(section, whole=True)
        self.seed.setPlaceholderText("random")
        more.addRow("Random seed", self.seed)
        section.set_more_shown(False)

    def refresh(self) -> None:
        """Shows the FOD tracked, the masks to choose from and the last run."""
        state = self.state
        present = {item.id for item in self.viewer.scene.items}
        fod = state.fod if state.fod in present else None
        fods = self.viewer.scene.in_category(Category.FODS)
        several = fod is not None and len(fods) > 1
        if fod is None:
            text = "No FOD yet: go back to the FODs step."
        elif several:
            text = "Tracks:"
        else:
            text = f"Tracks {self.viewer.scene.item(fod).name}."
        self.fod_label.setText(text)
        self.fod_box.setVisible(several)
        if several:
            self.fod_box.set_choices([(item.id, item.name) for item in fods])
            self.fod_box.show_value(fod)
        images = image_choices(self.viewer)
        self.seed_mask.set_choices([(None, "the FOD's support"), *images])
        self.mask_box.set_choices([(None, "none"), *images])
        # The fit's mask is the default of both, once per fit.
        if state.fit_mask != self._defaulted and state.fit_mask in present:
            self.seed_mask.show_value(state.fit_mask)
            self.mask_box.show_value(state.fit_mask)
            self._defaulted = state.fit_mask
        tract = state.tractogram if state.tractogram in present else None
        run = state.runs.get(tract) if tract is not None else None
        if run is None:
            self.result.setText("")
        else:
            seed = run.run.parameters.get("seed")
            self.result.setText(
                f"{run.run.num_streamlines:,} streamlines from "
                f"{run.run.num_seeds:,} seeds, random seed {seed}."
            )
        self.save_button.setVisible(run is not None)

    def _choose_fod(self, fod: object) -> None:
        """Makes the FOD chosen the one tracked, and tells the window."""
        if isinstance(fod, int):
            self.state.fod = fod
            self.fod_chosen.emit(fod)

    def settings(self) -> TrackSettings:
        """Returns the choices of the run.

        Raises:
            ValueError: If a number cannot be read.
        """
        algorithm = self.algorithm.value()
        seeding = self.seeding.value()
        assert isinstance(algorithm, TrackAlgorithm) and isinstance(seeding, Seeding)
        n = self.numbers
        seed = self.seed.value()
        return TrackSettings(
            algorithm=algorithm,
            streamlines=self.streamlines.value(),
            step_size=n["step_size"].value(),
            angle=n["angle"].value(),
            cutoff=n["cutoff"].value(),
            min_length=n["min_length"].value(),
            max_length=n["max_length"].value(),
            seeding=seeding,
            seeds_per_voxel=self.per_voxel.value(),
            seed=None if seed is None else int(seed),
        )

    def _track(self) -> None:
        """Asks the window to track with the choices on the page."""
        fod = self.state.fod
        if fod is None:
            self.refused.emit("There is no FOD to track: go back to the FODs step.")
            return
        try:
            settings = self.settings()
        except ValueError as error:
            self.refused.emit(f"A number cannot be read: {error}")
            return
        self.track_requested.emit(
            fod, self.seed_mask.value(), self.mask_box.value(), settings
        )


class RegionsPage(QWidget):
    """Lists the regions that dissect the tractogram, and how many streamlines stay.

    The top holds `parcellation_slot`, an empty layout where the regions of a
    parcellation's labels are chosen.
    """

    new_mask_requested = Signal(object)  # the MaskTarget
    open_requested = Signal()

    def __init__(
        self, viewer: Viewer, state: Dissection, parent: QWidget | None = None
    ) -> None:
        """Builds the page, its list made once a tractogram is set."""
        super().__init__(parent)
        self.viewer = viewer
        self.state = state
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.parcellation_slot = QVBoxLayout()
        self.layout_.addLayout(self.parcellation_slot)
        self.parcellation: ParcellationSection | None = None
        self.kept = QLabel(self)
        self.kept.setWordWrap(True)
        self.layout_.addWidget(self.kept)
        self.editor: RegionsEditor | None = None
        self.editor_slot = QVBoxLayout()
        self.layout_.addLayout(self.editor_slot)
        links = QHBoxLayout()
        new_mask = _link("New mask...", self)
        new_mask.clicked.connect(
            lambda: self.new_mask_requested.emit(MaskTarget.REGION)
        )
        self.open_button = QPushButton("Open regions...", self)
        self.open_button.setToolTip("Add the regions of a saved regions file")
        self.open_button.clicked.connect(self.open_requested)
        links.addWidget(new_mask)
        links.addStretch(1)
        links.addWidget(self.open_button)
        self.layout_.addLayout(links)
        self.layout_.addStretch(1)
        self.refresh()

    def refresh(self) -> None:
        """Shows the tractogram's regions and the count of streamlines kept."""
        self._refresh_parcellation()
        tract = self.state.tractogram
        present = {item.id for item in self.viewer.scene.items}
        if tract not in present:
            tract = None
        if self.editor is not None and (tract is None or self.editor.item_id != tract):
            self.editor.deleteLater()
            self.editor = None
        if tract is not None and self.editor is None:
            self.editor = RegionsEditor(self.viewer, tract, self)
            self.editor_slot.addWidget(self.editor)
        self.open_button.setEnabled(tract is not None)
        if tract is None:
            self.kept.setText("Start from a tractogram, or track one, first.")
            return
        kept = self.viewer.kept_streamlines(tract)
        data = self.viewer.scene.item(tract).data
        total = getattr(data, "num_streamlines", kept)
        self.kept.setText(f"{kept:,} of {total:,} streamlines kept.")

    def _refresh_parcellation(self) -> None:
        """Shows the label regions of the first overlay shown as labels, if any."""
        parcellation = next(
            (
                o.id
                for o in self.viewer.scene.in_category(Category.OVERLAYS)
                if isinstance(o.options, OverlayOptions)
                and o.options.labels
                and o.volume_count == 1
            ),
            None,
        )
        shown = self.parcellation
        if shown is not None and shown.parcellation_id == parcellation:
            return
        if shown is not None:
            shown.deleteLater()
            self.parcellation = None
        if parcellation is not None:
            self.parcellation = ParcellationSection(self.viewer, parcellation, self)
            self.parcellation.changed.connect(lambda _: self.refresh())
            self.parcellation_slot.addWidget(self.parcellation)


class SavePage(QWidget):
    """Names the files a save writes, with the unsaved items to write beside them."""

    save_requested = Signal(object, str, object)  # the folder, name and items

    def __init__(
        self, viewer: Viewer, state: Dissection, parent: QWidget | None = None
    ) -> None:
        """Builds the folder, the name, the list of files and Save."""
        super().__init__(parent)
        self.viewer = viewer
        self.state = state
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        folder = QHBoxLayout()
        self.folder = QLineEdit(self)
        browse = QPushButton("Folder...", self)
        browse.clicked.connect(self._ask_folder)
        folder.addWidget(self.folder, 1)
        folder.addWidget(browse)
        form.addRow("Folder", folder)
        self.name = QLineEdit("bundle", self)
        self.name.textChanged.connect(lambda _: self.refresh())
        self.folder.textChanged.connect(lambda _: self.refresh())
        form.addRow("Name", self.name)
        layout.addLayout(form)
        self.files = QLabel(self)
        self.files.setWordWrap(True)
        self.files.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.files)
        self.unsaved_title = QLabel("Unsaved items to save too", self)
        self.unsaved_title.setObjectName("railNote")
        layout.addWidget(self.unsaved_title)
        self.items_box = QVBoxLayout()
        layout.addLayout(self.items_box)
        self.checks: dict[ItemId, QCheckBox] = {}
        self.save_button = QPushButton("Save", self)
        self.save_button.setObjectName("nextStep")
        self.save_button.clicked.connect(self._save)
        layout.addWidget(self.save_button)
        layout.addStretch(1)
        self.refresh()

    def _ask_folder(self) -> None:
        """Asks for the folder to save into."""
        folder = QFileDialog.getExistingDirectory(self, "Save into")
        if folder:
            self.folder.setText(folder)

    def chosen(self) -> list[ItemId]:
        """Returns the unsaved items ticked."""
        return [i for i, box in self.checks.items() if box.isChecked()]

    def refresh(self) -> None:
        """Lists the unsaved items and the files a save would write."""
        from larmor.viewer.model.dissect import plan_save

        state, scene = self.state, self.viewer.scene
        unsaved = state.unsaved(scene)
        for item_id in list(self.checks):
            if item_id not in unsaved:
                self.checks.pop(item_id).deleteLater()
        for item_id in unsaved:
            if item_id not in self.checks:
                box = QCheckBox(scene.item(item_id).name, self)
                box.setChecked(True)
                box.toggled.connect(lambda _: self.refresh())
                self.items_box.addWidget(box)
                self.checks[item_id] = box
        self.unsaved_title.setVisible(bool(self.checks))
        folder = self.folder.text().strip()
        try:
            plan = plan_save(state, scene, Path(folder or "."), self.name.text(), [])
        except ValueError as error:
            self.files.setText(str(error))
            self.save_button.setEnabled(False)
            return
        # The items the bundle needs are saved whether ticked or not.
        for item_id, box in self.checks.items():
            needed = item_id in plan.items
            box.setEnabled(not needed)
            if needed:
                box.setToolTip("The bundle's run record or regions need this file")
        plan = plan_save(
            state, scene, Path(folder or "."), self.name.text(), self.chosen()
        )
        names = "\n".join(p.name for p in plan.paths)
        notes = "".join(f"\n\n{note}" for note in plan.notes)
        self.files.setText(f"Writes:\n{names}{notes}")
        self.save_button.setEnabled(bool(folder))

    def _save(self) -> None:
        """Asks the window to save into the folder."""
        self.save_requested.emit(
            Path(self.folder.text().strip()), self.name.text(), self.chosen()
        )


class DissectStepPage(QWidget):
    """Shows the page of the step open: a heading, its help, its fields, Back, Next."""

    def __init__(
        self,
        viewer: Viewer,
        steps: DissectNavigator,
        parent: QWidget | None = None,
    ) -> None:
        """Builds the five pages, one shown at a time."""
        super().__init__(parent)
        self.viewer = viewer
        self.steps = steps
        state = steps.state
        self.start = StartPage(viewer, self)
        self.fods = FodsPage(viewer, state, self)
        self.track = TrackPage(viewer, state, self)
        self.regions = RegionsPage(viewer, state, self)
        self.save = SavePage(viewer, state, self)
        self.pages: dict[DissectStep, QWidget] = {
            DissectStep.START: self.start,
            DissectStep.FODS: self.fods,
            DissectStep.TRACK: self.track,
            DissectStep.REGIONS: self.regions,
            DissectStep.SAVE: self.save,
        }
        area = self._build_area()
        self._build_nav()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(area, 1)
        layout.addWidget(self.nav)
        for widget in (self.fods, self.track):
            widget.refused.connect(self.show_error)
        steps.changed.connect(self.refresh)
        unsubscribe = viewer.scene.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: unsubscribe())
        self.refresh()

    def _build_area(self) -> QScrollArea:
        """Builds the scrolled heading, help, pages and error line."""
        self.stack = QStackedWidget(self)
        for widget in self.pages.values():
            self.stack.addWidget(widget)
        area = QScrollArea(self)
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget(area)
        page = QVBoxLayout(content)
        page.setContentsMargins(12, 12, 12, 12)
        self.heading = QLabel(content)
        self.heading.setObjectName("stepHeading")
        self.text = QLabel(content)
        self.text.setObjectName("stepText")
        self.text.setWordWrap(True)
        self.error = QLabel(content)
        self.error.setObjectName("fieldError")
        self.error.setWordWrap(True)
        self.error.hide()
        page.addWidget(self.heading)
        page.addWidget(self.text)
        page.addWidget(self.stack, 1)
        page.addWidget(self.error)
        area.setWidget(content)
        return area

    def _build_nav(self) -> None:
        """Builds the bar of Back and Next."""
        self.nav = QWidget(self)
        self.nav.setObjectName("stepNav")
        self.nav.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        row = QHBoxLayout(self.nav)
        row.setContentsMargins(12, 8, 12, 8)
        self.back_button = QPushButton("Back", self.nav)
        self.back_button.clicked.connect(self.steps.back)
        self.next_button = QPushButton("Next", self.nav)
        self.next_button.setObjectName("nextStep")
        self.next_button.clicked.connect(self.steps.next)
        row.addWidget(self.back_button)
        row.addStretch(1)
        row.addWidget(self.next_button)

    def show_error(self, text: str) -> None:
        """Shows why an action of the page cannot be done, below its fields."""
        self.error.setText(text)
        self.error.show()

    def refresh(self) -> None:
        """Shows the step open, its page, and where Back and Next go."""
        step = self.steps.page
        info = STEP_INFO[step]
        self.heading.setText(f"{DISSECT_STEPS.index(step) + 1}. {info.name}")
        self.text.setText(info.text)
        self.error.hide()
        self.stack.setCurrentWidget(self.pages[step])
        self.refresh_pages()
        before, after = self.steps.previous_step(), self.steps.next_step()
        self.back_button.setEnabled(before is not None)
        self.back_button.setText(
            "Back" if before is None else f"Back: {STEP_INFO[before].name}"
        )
        self.next_button.setVisible(after is not None)
        if after is not None:
            self.next_button.setText(f"Next: {STEP_INFO[after].name}")

    def refresh_pages(self) -> None:
        """Shows the steps' pages again from the dissection and the scene."""
        self.steps.state.forget_missing(self.viewer.scene)
        self.fods.refresh()
        self.track.refresh()
        self.regions.refresh()
        self.save.refresh()

    def show_job(self, status: JobStatus | None) -> None:
        """Shows a running job of the FODs or Track step on its page."""
        self.fods.job.show_job(status)
        self.track.job.show_job(status)
        running = status is not None
        self.fods.estimate_button.setEnabled(not running)
        self.track.track_button.setEnabled(not running)

    def _on_event(self, event: Event) -> None:
        """Shows the pages again after a change of the items they offer."""
        if isinstance(
            event, (ItemAdded, ItemRemoved, ItemRenamed, SourceChanged)
        ) or _changes_regions(event):
            self.refresh_pages()
