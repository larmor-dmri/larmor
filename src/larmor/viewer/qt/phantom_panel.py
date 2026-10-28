"""The phantom editor's panel, which edits the grid, tissues, bundles and scan."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast, override

from PySide6.QtCore import QEvent, QObject, QRectF, QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPaintEvent, QResizeEvent, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QStyleOptionToolButton,
    QStylePainter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from larmor.viewer.model.events import (
    Event,
    ItemAdded,
    ItemRemoved,
    PhantomChanged,
    ViewStateChanged,
)
from larmor.viewer.model.item import Category, ItemId
from larmor.viewer.model.options import OptionError
from larmor.viewer.model.phantom import (
    PROXY_NAMES,
    BundleField,
    BvecFrame,
    CompartmentKind,
    CompartmentSettings,
    EchoKind,
    PhantomState,
    ProxyMap,
    ScanField,
    TissuePreset,
    TissueSettings,
    build_tissue,
    preset_defaults,
    run_problems,
)
from larmor.viewer.model.phantom_editor import item_sources
from larmor.viewer.model.view_state import Tool, ViewField
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.qt.field import FieldLabel, ItemField, add_field_row
from larmor.viewer.qt.style import icon_path

# Shown above an empty phantom.
START_HINT = (
    "To start, set the grid below or take the grid of a loaded image. Then press N "
    "and click in a 2D view to place the points of a bundle, and press Enter to "
    "finish it. Pick the bundle's tissue in the Bundles section, and click Run to "
    "simulate the phantom."
)
# The sections, by column when the panel is wide; they share fewer columns when
# it is narrow.
COLUMN_GROUPS = (("Grid", "Proxy maps", "Run"), ("Tissues",), ("Bundles",), ("Scan",))
COLUMN_WIDTH = 420  # px, the width a column of fields needs

# What each part of a phantom is for, shown by the info icon after its title.
SECTION_HELP = {
    "Grid": "The voxel grid the phantom is simulated on, and what fills the parts of "
    "each voxel that no bundle fills.",
    "Tissues": "The tissues of the phantom. Each one is a sum of compartments with "
    "their own diffusion and relaxation, and bundles and maps refer to it by name.",
    "Bundles": "The fibre bundles. A drawn bundle is a tube of streamlines around a "
    "smooth line through the points you click, and it fills the voxels inside the "
    "tube with its tissue.",
    "Scan": "The scan the phantom is simulated with: the gradients, the timings and "
    "the noise.",
    "Proxy maps": "Overlays that show what the phantom gives without running it. They "
    "follow every edit.",
    "Run": "Simulates the phantom and writes the images and the ground truth into a "
    "folder, then opens them.",
}
BOX_FIELD_WIDTH = 64  # px, the least width of each of the box's three sizes
# What each section's More fold holds, as its button names it.
MORE = {
    "Grid": "More: grid box, background, tissue maps",
    "Tissues": "More: custom compartments",
    "Bundles": "More: shape, seed, editing, a loaded tractogram",
    "Scan": "More: sequence, timings, seeds, gradient files",
}
# The help of each field, shown by the info icon after its label.
HELP = {
    "Box (mm)": "The size of the grid along x, y and z, in mm. The box is centred on "
    "the origin.",
    "Voxel (mm)": "The size of one cubic voxel, in mm. Smaller voxels give a finer "
    "grid and a slower run.",
    "Or the grid of": "Use the voxel grid of a loaded image instead of a box, so that "
    "the simulated images line up with it.",
    "Background": "The tissue that fills what the bundles and maps leave of each "
    "voxel. With None, that part gives no signal.",
    "Made of": "A preset model of the tissue, whose parameters you can change, or "
    "custom compartments that you list yourself.",
    "Tissue": "The tissue that fills the bundle. Changing a tissue changes every "
    "bundle that uses it.",
    "Streamlines": "How many streamlines make up the bundle. They give the fibre "
    "directions in each voxel.",
    "Radius (mm)": "The radius of the bundle's tube, in mm.",
    "Points per streamline": "How many points each streamline has along the bundle.",
    "Fan at start": "How much wider the bundle is at its first point: 1 makes the "
    "radius twice as large there.",
    "Fan at end": "How much wider the bundle is at its last point: 1 makes the radius "
    "twice as large there.",
    "Fan sharpness": "How close to the ends the fanning stays. Larger values keep the "
    "middle of the bundle at its radius.",
    "Dispersion (mm)": "How far, in mm, the streamlines wander at random from their "
    "path at the inner points.",
    "Seed": "The seed of the bundle's random streamlines. The same seed always gives "
    "the same streamlines.",
    "Streamline area": "The cross-section in mm^2 that each streamline stands for. "
    "With default, a drawn bundle fills the voxels inside its tube.",
    "Densest fill": "For a loaded tractogram, the share of its densest voxels that "
    "the bundle fills, from 0 to 1. A tractogram carries no density of its own.",
    "Sequence": "A spin echo weights the signal with T2; a gradient echo weights it "
    "with T2*.",
    "TE (ms)": "The echo time in ms. Several times separated by commas give one image "
    "per echo.",
    "TR (ms)": "The repetition time in ms. It sets how much the T1 relaxation lowers "
    "the signal.",
    "delta (ms)": "The duration of each diffusion gradient pulse, in ms.",
    "Delta (ms)": "The time between the starts of the two diffusion gradient pulses, "
    "in ms. Delta plus delta must be shorter than the echo time.",
    "b = 0 volumes": "How many images without diffusion weighting come before the "
    "shells.",
    "Directions seed": "The seed of the gradient directions of the shells. The same "
    "seed gives the same directions.",
    "Noise seed": "The seed of the noise. The same seed gives the same images.",
    "Noise reference": "The tissue whose signal at b = 0 sets the noise: the noise "
    "level is the signal of a voxel full of it, divided by the SNR.",
    "bvecs frame": "The frame of the bvecs file. FSL, as FSL and dcm2niix write "
    "them, is along the axes of the phantom's grid with x flipped; world is RAS "
    "coordinates.",
}
# The help of the fields that have no label of their own.
NOISE_HELP = (
    "Adds Rician noise. The SNR is the signal at b = 0 of a voxel full of the noise "
    "reference tissue, WM unless chosen under More, divided by the noise level."
)
SHOW_BOX_HELP = (
    "Outlines the grid in the views. It is always outlined while you draw a bundle."
)
ADOPT_HELP = (
    "Makes a loaded tractogram a bundle of the phantom, with its streamlines as they "
    "are."
)
PROXY_HELP = {
    ProxyMap.FILL: "The share of each voxel that the bundles fill: the first volume "
    "is all bundles together, then one volume per bundle.",
    ProxyMap.PARTITION: "The share of each voxel that each tissue fills, one volume "
    "per tissue, with the background last.",
    ProxyMap.EXPECTED_B0: "The signal at b = 0 and the first echo time that a run "
    "without noise gives: each tissue's share times its relaxation weight.",
}


def _map_help(tissue: str) -> str:
    """Returns the help of a tissue map's field."""
    return (
        f"An overlay with the share of {tissue} in each voxel, from 0 to 1, on the "
        "phantom's grid. It fills what the bundles leave."
    )


# Labels and help of the preset parameters whose names alone are unclear,
# above all the fractions, which name a different compartment in each preset.
PARAMETER_TEXT: dict[tuple[TissuePreset, str], tuple[str, str]] = {
    (TissuePreset.STANDARD_MODEL, "intra_fraction"): (
        "Inside axons (fraction)",
        "The share of the tissue's water inside the axons, from 0 to 1. The water "
        "around the axons takes what the free water and this leave.",
    ),
    (TissuePreset.STANDARD_MODEL, "free_water_fraction"): (
        "Free water (fraction)",
        "The share of the tissue's water that diffuses freely, as in CSF, from 0 "
        "to 1. At 0 the tissue has none.",
    ),
    (TissuePreset.STANDARD_MODEL, "intra_diffusivity"): (
        "Inside axons, along (um^2/ms)",
        "How fast water diffuses along the axons, inside them. Across them it "
        "does not move.",
    ),
    (TissuePreset.STANDARD_MODEL, "extra_parallel_diffusivity"): (
        "Around axons, along (um^2/ms)",
        "How fast the water around the axons diffuses along them.",
    ),
    (TissuePreset.STANDARD_MODEL, "extra_perpendicular_diffusivity"): (
        "Around axons, across (um^2/ms)",
        "How fast the water around the axons diffuses across them.",
    ),
    (TissuePreset.STANDARD_MODEL, "intra_t2_ms"): (
        "T2 inside axons (ms)",
        "The T2 of the water inside the axons, which a spin echo uses.",
    ),
    (TissuePreset.STANDARD_MODEL, "extra_t2_ms"): (
        "T2 around axons (ms)",
        "The T2 of the water around the axons, which a spin echo uses.",
    ),
    (TissuePreset.NODDI, "intra_fraction"): (
        "Neurites (fraction)",
        "The neurites' share of the water that is not free water, from 0 to 1 "
        "(NODDI's v_ic). The rest of that water lies around the neurites.",
    ),
    (TissuePreset.NODDI, "free_water_fraction"): (
        "Free water (fraction)",
        "The share of the whole tissue's water that diffuses freely, as in CSF, "
        "from 0 to 1 (NODDI's v_iso).",
    ),
    (TissuePreset.NODDI, "diffusivity"): (
        "Along neurites (um^2/ms)",
        "How fast water diffuses along the neurites. The water around them "
        "follows from it and from kappa.",
    ),
    (TissuePreset.SANDI, "neurite_fraction"): (
        "Neurites (fraction)",
        "The share of the tissue's water inside neurites, from 0 to 1. The water "
        "outside cells takes what the somas and this leave.",
    ),
    (TissuePreset.SANDI, "soma_fraction"): (
        "Somas (fraction)",
        "The share of the tissue's water inside cell bodies, from 0 to 1.",
    ),
    (TissuePreset.SANDI, "extracellular_diffusivity"): (
        "Outside cells (um^2/ms)",
        "How fast the water outside cells diffuses, the same in every direction.",
    ),
}


def _parameter_label(preset: TissuePreset, key: str) -> str:
    """Returns a preset parameter's form label, with its unit."""
    text = PARAMETER_TEXT.get((preset, key))
    return text[0] if text is not None else _label(key)


def _parameter_help(preset: TissuePreset, key: str) -> str:
    """Returns the help of a preset parameter."""
    text = PARAMETER_TEXT.get((preset, key))
    if text is not None:
        return text[1]
    if "fraction" in key:
        return "The share of the tissue's volume in this compartment, from 0 to 1."
    if "diffusivity" in key:
        return "How fast water diffuses in this compartment, in um^2/ms."
    if key.startswith("t2star"):
        return "The T2* relaxation time in ms, which a gradient echo uses."
    if key.startswith("t2") or key.endswith("t2_ms"):
        return "The T2 relaxation time in ms, which a spin echo uses."
    if key.startswith("t1"):
        return "The T1 relaxation time in ms."
    if key == "proton_density":
        return "The water content relative to pure water, from 0 to 1."
    if "radius" in key:
        return "The radius of the soma spheres, in um."
    if key == "kappa":
        return (
            "How tightly the neurites follow the fibre direction: 0 spreads them "
            "evenly, larger values keep them close to it."
        )
    return "A parameter of the preset."


# Diffusivities are typed in um^2/ms (1e-3 mm^2/s), as papers quote them.
DIFFUSIVITY_SCALE = 1000.0
PRESET_NAMES = {
    TissuePreset.STANDARD_MODEL: "Standard model (WM)",
    TissuePreset.SANDI: "SANDI (GM)",
    TissuePreset.NODDI: "NODDI (WM)",
    TissuePreset.FREE_WATER: "Free water (CSF)",
    TissuePreset.CUSTOM: "Custom compartments",
}
# The columns of the custom compartments' table, after the model.
COMPARTMENT_COLUMNS = (
    ("fraction", "Fraction"),
    ("diffusivity", "D"),
    ("perpendicular_diffusivity", "D perp"),
    ("radius_um", "Radius um"),
    ("kappa", "kappa"),
    ("t1_ms", "T1"),
    ("t2_ms", "T2"),
    ("t2star_ms", "T2*"),
    ("proton_density", "PD"),
)
# Colours of the fraction bar's segments, one per compartment in turn.
FRACTION_COLOURS = ("#4c8bf5", "#f5a742", "#5cc46a", "#c45cc4", "#e05555", "#55c0e0")
# The geometry fields of the bundle form: label, minimum, maximum, decimals.
GEOMETRY_FORM: tuple[tuple[BundleField, str, float, float, int], ...] = (
    (BundleField.NUM_STREAMLINES, "Streamlines", 1, 100_000, 0),
    (BundleField.RADIUS, "Radius (mm)", 0, 100, 2),
    (BundleField.NUM_POINTS, "Points per streamline", 2, 10_000, 0),
    (BundleField.FAN_START, "Fan at start", 0, 20, 2),
    (BundleField.FAN_END, "Fan at end", 0, 20, 2),
    (BundleField.FAN_SHARPNESS, "Fan sharpness", 0.01, 20, 2),
    (BundleField.DISPERSION, "Dispersion (mm)", 0, 50, 2),
    (BundleField.SEED, "Seed", 0, 2**31 - 1, 0),
)


def _label(key: str) -> str:
    """Returns a preset parameter's name as a form label, with its unit."""
    words = key.replace("_ms", "").replace("_um", "").replace("_", " ")
    words = words.replace("t2star", "T2*").replace("t1", "T1").replace("t2", "T2")
    unit = " (um^2/ms)" if "diffusivity" in key else ""
    unit = unit or (" (ms)" if key.endswith("_ms") else "")
    unit = unit or (" (um)" if key.endswith("_um") else "")
    return words[:1].upper() + words[1:] + unit


def _spin(low: float, high: float, decimals: int = 2) -> QDoubleSpinBox:
    """Creates a spin box of decimals that edits once typing ends."""
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setKeyboardTracking(False)  # one edit when typing ends, not per key
    box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return box


def _int_spin(low: int, high: int) -> QSpinBox:
    """Creates an integer spin box that edits once typing ends."""
    box = QSpinBox()
    box.setRange(low, high)
    box.setKeyboardTracking(False)
    return box


def _show(box: QDoubleSpinBox | QSpinBox, value: float) -> None:
    """Shows a value from the model in a spin box without sending its signals."""
    with QSignalBlocker(box):
        if isinstance(box, QSpinBox):
            box.setValue(round(value))
        else:
            box.setValue(value)


class ChoiceBox(QComboBox):
    """A drop-down of values, each kept as the object given, never as its text.

    Qt would hand an enum given as item data back as plain text, so the values
    are kept here instead, by row.
    """

    chosen = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Creates the drop-down with no choices."""
        super().__init__(parent)
        self._values: list[object] = []
        self.activated.connect(lambda row: self.chosen.emit(self._values[row]))

    def set_choices(self, choices: list[tuple[object, str]]) -> None:
        """Offers these (value, text) pairs, keeping the value shown if offered."""
        if [v for v, _ in choices] == self._values and [
            self.itemText(i) for i in range(self.count())
        ] == [t for _, t in choices]:
            return
        shown = self.value()
        with QSignalBlocker(self):
            self.clear()
            self._values = [v for v, _ in choices]
            self.addItems([t for _, t in choices])
        self.show_value(shown)

    def value(self) -> object:
        """Returns the value chosen; None if none is."""
        row = self.currentIndex()
        return self._values[row] if 0 <= row < len(self._values) else None

    def show_value(self, value: object) -> None:
        """Shows a value from the model without sending `chosen`."""
        row = self._values.index(value) if value in self._values else -1
        with QSignalBlocker(self):
            self.setCurrentIndex(row)


class Section(QWidget):
    """A titled part of the panel that folds away, with a More fold at its end.

    The fields that matter go in `form`, the rest in `more_form`, which a
    button below the fields unfolds.
    """

    def __init__(
        self, title: str, help: str = "", parent: QWidget | None = None
    ) -> None:
        """Creates the section, unfolded, with an empty body; help explains it."""
        super().__init__(parent)
        self.header = QToolButton(self)
        self.header.setObjectName("optionGroup")
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(True)
        self.header.setAutoRaise(True)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.body = QWidget(self)
        self.form = _form(self.body)
        self.form.setContentsMargins(14, 4, 8, 8)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.help = help
        if help:
            self.header.setToolTip(help)
            top = QHBoxLayout()
            top.setContentsMargins(0, 0, 8, 0)
            top.addWidget(self.header, 1)
            top.addWidget(FieldLabel("", help, self))
            layout.addLayout(top)
        else:
            layout.addWidget(self.header)
        layout.addWidget(self.body)
        self._build_more(layout, title)
        self.header.toggled.connect(self._fold)
        self.more_button.toggled.connect(self._fold_more)
        self._fold(True)
        self._fold_more(False)

    def _build_more(self, layout: QVBoxLayout, title: str) -> None:
        """Adds the More button and the form of the fields it unfolds."""
        self.more_button = FoldButton(MORE.get(title, "More"), self)
        self.more_button.setObjectName("moreFold")
        self.more_button.setCheckable(True)
        self.more_button.setAutoRaise(True)
        self.more_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.more = QWidget(self)
        self.more_form = _form(self.more)
        self.more_form.setContentsMargins(14, 0, 8, 8)
        layout.addWidget(self.more_button)
        layout.addWidget(self.more)

    def _fold(self, shown: bool) -> None:
        """Shows or folds away the section's body."""
        self.body.setVisible(shown)
        self.header.setIcon(
            QIcon(icon_path("chevron-down" if shown else "chevron-right"))
        )
        self._show_more()

    def _fold_more(self, shown: bool) -> None:
        """Shows or folds away the fields under More."""
        self.more_button.setIcon(
            QIcon(icon_path("chevron-down" if shown else "chevron-right"))
        )
        self._show_more()

    def _show_more(self) -> None:
        """Shows the More button if it has fields, and them if it is unfolded."""
        body = self.header.isChecked()
        has_more = self.more_form.rowCount() > 0
        self.more_button.setVisible(body and has_more and self._more_button)
        self.more.setVisible(
            body
            and has_more
            and (self.more_button.isChecked() or not self._more_button)
        )

    _more_button = True  # False shows the fields under More without the button

    def set_more_button(self, shown: bool) -> None:
        """Shows the More button, or the fields under it without a button."""
        self._more_button = shown
        self._show_more()

    def set_more_shown(self, shown: bool) -> None:
        """Unfolds or folds the fields under More."""
        self.more_button.setChecked(shown)
        self._show_more()

    def set_header_visible(self, shown: bool) -> None:
        """Shows or hides the section's title, as a page that names it does."""
        self.header.setVisible(shown)
        for child in self.findChildren(FieldLabel):
            if child.parent() is self:
                child.setVisible(shown)


class FoldButton(QToolButton):
    """A fold's button that cuts its text short with an ellipsis when it is narrow."""

    NARROWEST = 80  # px, the least room it asks for

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        """Creates the button with its full text."""
        super().__init__(parent)
        self.setText(text)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    @override
    def minimumSizeHint(self) -> QSize:
        """Returns a narrow size: the text is cut short rather than widen the panel."""
        hint = super().minimumSizeHint()
        return QSize(min(hint.width(), self.NARROWEST), hint.height())

    @override
    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        """Draws the button with as much of its text as its width takes."""
        painter = QStylePainter(self)
        option = QStyleOptionToolButton()
        self.initStyleOption(option)
        metrics = self.fontMetrics()
        # The room the icon and padding take, then what is left for the text.
        frame = self.sizeHint().width() - metrics.horizontalAdvance(self.text())
        option.text = metrics.elidedText(
            self.text(), Qt.TextElideMode.ElideRight, self.width() - frame
        )
        painter.drawComplexControl(QStyle.ComplexControl.CC_ToolButton, option)


def _form(parent: QWidget | None = None) -> QFormLayout:
    """Creates a form whose fields go below their labels when the panel is narrow."""
    form = QFormLayout(parent)
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return form


class FractionBar(QWidget):
    """Draws the fractions of a tissue's compartments side by side.

    The bar is outlined in red unless the fractions sum to 1.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Creates an empty bar."""
        super().__init__(parent)
        self.fractions: list[float] = []
        self.setMinimumHeight(14)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_fractions(self, fractions: list[float]) -> None:
        """Shows these fractions."""
        self.fractions = fractions
        total = sum(fractions)
        self.setToolTip(f"The fractions sum to {total:.3g}; they must sum to 1.")
        self.update()

    @override
    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        """Draws a segment per compartment over the bar's width (1 is full)."""
        painter = QPainter(self)
        width, height = self.width(), self.height()
        painter.fillRect(0, 0, width, height, QColor("#80808040"))
        x = 0.0
        for index, fraction in enumerate(self.fractions):
            span = max(fraction, 0.0) * width
            colour = FRACTION_COLOURS[index % len(FRACTION_COLOURS)]
            painter.fillRect(QRectF(x, 0, span, height), QColor(colour))
            x += span
        if abs(sum(self.fractions) - 1) > 1e-6:
            painter.setPen(QColor("#d93025"))
            painter.drawRect(0, 0, width - 1, height - 1)
        painter.end()


class PhantomPanel(QWidget):
    """Edits the phantom through the viewer's actions; shows `viewer.phantom`.

    It keeps no copy of the phantom: every change comes back as an event and is
    shown from the scene. The window opens, saves and runs phantoms, as those
    need dialogs and the job runner.
    """

    new_requested = Signal()
    open_requested = Signal()
    save_requested = Signal()
    run_requested = Signal()
    cancel_requested = Signal()
    save_bundle_requested = Signal(int)

    def __init__(
        self, viewer: Viewer, parent: QWidget | None = None, *, guided: bool = False
    ) -> None:
        """Builds the sections and shows the phantom, if one is open.

        A guided panel shows one section at a time, without its title and the
        file buttons, and with the fields under More folded; the others show
        every field.
        """
        super().__init__(parent)
        self.viewer = viewer
        self.guided = guided
        self.running = False  # a simulation runs
        self.setObjectName("phantomPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._build_file_row(layout)
        self._build_hints(layout)
        self._build_sections(layout)
        for section in self.sections.values():
            section.set_more_button(guided)
            section.set_header_visible(not guided)
            section._show_more()
        self.file_row.setVisible(not guided)

        # The viewer can outlive the panel: stop listening when it goes.
        unsubscribe = viewer.scene.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: unsubscribe())
        self._reflow()
        self.refresh()

    def _build_file_row(self, layout: QVBoxLayout) -> None:
        """Adds the row of New, Open, Save and Close buttons."""
        # A widget of its own, so a window around the panel can show it elsewhere.
        self.file_row = QWidget(self)
        files = QHBoxLayout(self.file_row)
        files.setContentsMargins(14, 8, 8, 0)
        self.new_button = QPushButton("New", self)
        self.open_button = QPushButton("Open...", self)
        self.save_button = QPushButton("Save...", self)
        self.close_button = QPushButton("Close", self)
        for button in (self.new_button, self.open_button, self.save_button):
            files.addWidget(button)
        files.addWidget(self.close_button)
        files.addStretch(1)
        layout.addWidget(self.file_row)
        self.new_button.clicked.connect(self.new_requested)
        self.open_button.clicked.connect(self.open_requested)
        self.save_button.clicked.connect(self.save_requested)
        viewer = self.viewer
        self.close_button.clicked.connect(lambda: self._act(viewer.close_phantom))

    def _build_hints(self, layout: QVBoxLayout) -> None:
        """Adds the hint shown with no phantom, the error line and the start hint."""
        self.hint = QLabel(
            "No phantom is open. Click New to start one on the grid of the main "
            "volume, or Open to read a phantom file.",
            self,
        )
        self.hint.setObjectName("panelHint")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        self.error = QLabel(self)
        self.error.setObjectName("phantomError")
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #d93025;")
        self.error.hide()
        layout.addWidget(self.error)
        self.start_hint = QLabel(START_HINT, self)
        self.start_hint.setObjectName("panelHint")
        self.start_hint.setWordWrap(True)
        self.start_hint.setContentsMargins(14, 4, 14, 4)
        layout.addWidget(self.start_hint)

    def _build_sections(self, layout: QVBoxLayout) -> None:
        """Builds the sections and the columns they are laid out in."""
        self.sections: dict[str, Section] = {}
        for name in ("Grid", "Tissues", "Bundles", "Scan", "Proxy maps", "Run"):
            self.sections[name] = Section(name, SECTION_HELP[name], self)
        # The sections in columns, as many as the width takes (see `_reflow`).
        self.columns = QHBoxLayout()
        self.columns.setContentsMargins(0, 0, 0, 0)
        self.columns.setSpacing(12)
        layout.addLayout(self.columns)
        self._column_widgets: list[QWidget] = []
        layout.addStretch(1)
        self._column_count = 0
        self._only: tuple[str, ...] | None = None  # the sections shown, or all
        self._build_grid(self.sections["Grid"].form, self.sections["Grid"].more_form)
        tissues = self.sections["Tissues"]
        self._build_tissues(tissues.form, tissues.more_form)
        bundles = self.sections["Bundles"]
        self._build_bundles(bundles.form, bundles.more_form)
        self._build_scan(self.sections["Scan"].form, self.sections["Scan"].more_form)
        self._build_proxies(self.sections["Proxy maps"].form)
        self._build_run(self.sections["Run"].form)

    @override
    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        """Lays the sections out in as many columns as the width takes."""
        super().resizeEvent(event)
        self._reflow()

    @override
    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        """Follows the width of the scroll area it is shown in, if any."""
        super().showEvent(event)
        area = self._scroll_area()
        if area is not None:
            area.viewport().installEventFilter(self)
        self._reflow()

    @override
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Lays the sections out again when the scroll area's view resizes."""
        if event.type() == QEvent.Type.Resize:
            self._reflow()
        return False

    def _scroll_area(self) -> QScrollArea | None:
        """Returns the scroll area the panel is in, if any."""
        parent = self.parentWidget()
        while parent is not None and not isinstance(parent, QScrollArea):
            parent = parent.parentWidget()
        return parent

    def _room(self) -> int:
        """Returns the width the sections have, the scroll area's or the panel's."""
        area = self._scroll_area()
        return area.viewport().width() if area is not None else self.width()

    def show_sections(self, names: tuple[str, ...] | None) -> None:
        """Shows only these sections, in one column; every section if None."""
        self._only = names
        self._reflow()
        self.refresh()

    def _section_shown(self, name: str) -> bool:
        """Returns whether a section shows: a phantom is open and the section chosen."""
        if self.viewer.phantom is None:
            return False
        return self._only is None or name in self._only

    def _reflow(self) -> None:
        """Lays the sections out in as many columns as the width takes."""
        count = max(1, min(len(COLUMN_GROUPS), self._room() // COLUMN_WIDTH))
        if self._only is not None:
            count = 1
        if count == self._column_count:
            return
        self._column_count = count
        # Each column is a widget of its own, so its sections pack from the top
        # whatever the height of the others.
        for section in self.sections.values():
            section.setParent(self)
        for old in self._column_widgets:
            self.columns.removeWidget(old)
            old.deleteLater()
        self._column_widgets = []
        for _ in range(count):
            column = QWidget(self)
            stack = QVBoxLayout(column)
            stack.setContentsMargins(0, 0, 0, 0)
            stack.setSpacing(0)
            self.columns.addWidget(column, 1, Qt.AlignmentFlag.AlignTop)
            column.show()  # at once, not when the layout gets to it
            self._column_widgets.append(column)
        # The groups in order, split into `count` runs of about equal size.
        for index, group in enumerate(COLUMN_GROUPS):
            column = self._column_widgets[index * count // len(COLUMN_GROUPS)]
            layout = column.layout()
            assert isinstance(layout, QVBoxLayout)
            for name in group:
                layout.addWidget(self.sections[name])
                self.sections[name].setVisible(self._section_shown(name))

    # Building

    def _build_grid(self, form: QFormLayout, more: QFormLayout) -> None:
        """Builds the Grid section; the box switch, maps and background go in More."""
        box = QHBoxLayout()
        self.size_boxes = [_spin(1, 2000, 1) for _ in range(3)]
        for spin in self.size_boxes:
            spin.setValue(60.0)
            # Narrower than the widest value, so the three fit a narrow panel.
            spin.setMinimumWidth(BOX_FIELD_WIDTH)
            box.addWidget(spin)
        form.addRow(FieldLabel("Box (mm)", HELP["Box (mm)"], self), box)
        for spin in self.size_boxes:
            spin.setToolTip(HELP["Box (mm)"])
        self.voxel_box = _spin(0.05, 50, 2)
        self.voxel_box.setValue(2.0)
        add_field_row(form, "Voxel (mm)", HELP["Voxel (mm)"], self.voxel_box, self)
        self.apply_box = QPushButton("Use this box", self)
        self.apply_box.clicked.connect(self._apply_box)
        form.addRow(self.apply_box)
        self.show_box = QCheckBox("Show the grid box", self)
        self.show_box.setToolTip(SHOW_BOX_HELP)
        self.show_box.clicked.connect(
            lambda shown: self._act(self.viewer.show_grid_box, shown)
        )
        more.addRow(self.show_box)
        self.grid_of = ItemField(self)
        self.grid_of.edited.connect(self._apply_grid_of)
        add_field_row(
            form, "Or the grid of", HELP["Or the grid of"], self.grid_of, self
        )
        self.grid_label = QLabel(self)
        form.addRow(self.grid_label)
        self.map_form = _form()
        more.addRow(self.map_form)
        self.background = ChoiceBox(self)
        self.background.chosen.connect(
            lambda tissue: self._act(self.viewer.set_background_tissue, tissue)
        )
        add_field_row(more, "Background", HELP["Background"], self.background, self)

    def _build_tissues(self, form: QFormLayout, more: QFormLayout) -> None:
        """Builds the Tissues section; the custom compartments go in More."""
        self.tissue_list = QListWidget(self)
        self.tissue_list.setMinimumHeight(110)
        self.tissue_list.currentRowChanged.connect(lambda _: self._show_tissue())
        form.addRow(self.tissue_list)
        row = QHBoxLayout()
        self.add_tissue = QPushButton("Add...", self)
        self.add_tissue.clicked.connect(self._ask_tissue)
        self.remove_tissue = QPushButton("Remove", self)
        self.remove_tissue.clicked.connect(self._remove_tissue)
        row.addWidget(self.add_tissue)
        row.addWidget(self.remove_tissue)
        form.addRow(row)
        self.preset = ChoiceBox(self)
        self.preset.set_choices(list(PRESET_NAMES.items()))
        self.preset.chosen.connect(self._edit_preset)
        add_field_row(form, "Made of", HELP["Made of"], self.preset, self)
        self.parameters = QWidget(self)
        self.parameter_form = _form(self.parameters)
        self.parameter_form.setContentsMargins(0, 0, 0, 0)
        form.addRow(self.parameters)
        self.parameter_boxes: dict[str, QDoubleSpinBox] = {}
        self._parameter_preset: TissuePreset | None = None
        self._build_compartments()
        more.addRow(self.custom)

    def _build_compartments(self) -> None:
        """Builds the custom compartments' table, fraction bar and buttons."""
        self.custom = QWidget(self)
        custom = QVBoxLayout(self.custom)
        custom.setContentsMargins(0, 0, 0, 0)
        self.compartments = QTableWidget(0, 1 + len(COMPARTMENT_COLUMNS), self)
        self.compartments.setHorizontalHeaderLabels(
            ["Model", *(label for _, label in COMPARTMENT_COLUMNS)]
        )
        self.compartments.verticalHeader().setVisible(False)
        self.compartments.setMinimumHeight(110)
        self.compartments.itemChanged.connect(lambda _: self._edit_compartments())
        custom.addWidget(self.compartments)
        self.fraction_bar = FractionBar(self)
        custom.addWidget(self.fraction_bar)
        rows = QHBoxLayout()
        self.add_compartment = QPushButton("Add compartment", self)
        self.add_compartment.clicked.connect(self._add_compartment)
        self.remove_compartment = QPushButton("Remove", self)
        self.remove_compartment.clicked.connect(self._remove_compartment)
        rows.addWidget(self.add_compartment)
        rows.addWidget(self.remove_compartment)
        custom.addLayout(rows)

    def _build_bundles(self, form: QFormLayout, more: QFormLayout) -> None:
        """Builds the Bundles section; the shape's finer fields go in More."""
        self.bundle_table = QTableWidget(0, 3, self)
        self.bundle_table.setHorizontalHeaderLabels(["Bundle", "Tissue", "Points"])
        self.bundle_table.verticalHeader().setVisible(False)
        self.bundle_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.bundle_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.bundle_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.bundle_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.bundle_table.setMinimumHeight(90)
        self.bundle_table.itemSelectionChanged.connect(self._select_bundle)
        self._build_bundle_buttons(form, more)
        form.addRow(self.bundle_table)
        self.draft_tissue = ChoiceBox(self)
        self.draft_tissue.chosen.connect(
            lambda tissue: self._edit_bundle_value(BundleField.TISSUE, tissue)
        )
        add_field_row(form, "Tissue", HELP["Tissue"], self.draft_tissue, self)
        # Their rows show for the bundle drawn or selected.
        self._bundle_forms = (form, more)
        self._build_bundle_settings(form, more)
        self._build_points(more)

    def _build_bundle_buttons(self, form: QFormLayout, more: QFormLayout) -> None:
        """Adds Draw, the card of the bundle being drawn, and Edit, Save and Use."""
        self.draw_button = QPushButton("Draw (N)", self)
        self.draw_button.clicked.connect(self._draw)
        form.addRow(self.draw_button)
        self.draw_card = QFrame(self)
        self.draw_card.setObjectName("drawCard")
        card = QHBoxLayout(self.draw_card)
        card.setContentsMargins(10, 6, 6, 6)
        self.placed = QLabel(self.draw_card)
        self.finish_button = QPushButton("Finish (Enter)", self.draw_card)
        self.finish_button.clicked.connect(lambda: self._act(self.viewer.finish_bundle))
        self.cancel_button = QPushButton("Cancel", self.draw_card)
        self.cancel_button.clicked.connect(lambda: self._act(self.viewer.cancel_bundle))
        card.addWidget(self.placed, 1)
        card.addWidget(self.finish_button)
        card.addWidget(self.cancel_button)
        form.addRow(self.draw_card)
        row = QHBoxLayout()
        self.edit_button = QPushButton("Edit", self)
        self.edit_button.clicked.connect(self._edit_bundle)
        self.save_bundle = QPushButton("Save as...", self)
        self.save_bundle.clicked.connect(self._save_bundle)
        row.addWidget(self.edit_button)
        row.addWidget(self.save_bundle)
        more.addRow(row)
        self.adopt = ItemField(self)
        self.adopt.setToolTip(ADOPT_HELP)
        self.adopt.edited.connect(self._adopt)
        more.addRow(self.adopt)

    def _build_bundle_settings(self, form: QFormLayout, more: QFormLayout) -> None:
        """Adds the fields of a bundle's geometry, streamline area and fill.

        The tissue, streamlines and radius go in form, the rest in more.
        """
        self.geometry_boxes: dict[BundleField, QDoubleSpinBox | QSpinBox] = {}
        for name, label, low, high, decimals in GEOMETRY_FORM:
            box: QDoubleSpinBox | QSpinBox = (
                _int_spin(int(low), int(high))
                if decimals == 0
                else _spin(low, high, decimals)
            )
            box.valueChanged.connect(
                lambda value, n=name: self._edit_bundle_value(n, value)
            )
            self.geometry_boxes[name] = box
            main = name in (BundleField.NUM_STREAMLINES, BundleField.RADIUS)
            add_field_row(form if main else more, label, HELP[label], box, self)
        self.area_box = _spin(0, 1000, 4)
        self.area_box.setSpecialValueText("default")
        self.area_box.setToolTip(
            "mm^2 each streamline stands for; default: the bundle's cross-section "
            "shared by its streamlines"
        )
        self.area_box.valueChanged.connect(
            lambda value: self._edit_bundle_value(
                BundleField.STREAMLINE_AREA, value or None
            )
        )
        add_field_row(
            more, "Streamline area", HELP["Streamline area"], self.area_box, self
        )
        self.fill_box = _spin(0.01, 1, 2)
        self.fill_box.setToolTip("The fill of a loaded tractogram's dense voxels")
        self.fill_box.valueChanged.connect(
            lambda value: self._edit_bundle_value(BundleField.DENSEST_FILL, value)
        )
        add_field_row(more, "Densest fill", HELP["Densest fill"], self.fill_box, self)

    def _build_points(self, form: QFormLayout) -> None:
        """Adds the table of the drawn bundle's points and its buttons."""
        self.points = QTableWidget(0, 3, self)
        self.points.setHorizontalHeaderLabels(["x", "y", "z"])
        self.points.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.points.setMinimumHeight(100)
        self.points.itemChanged.connect(lambda _: self._edit_points())
        form.addRow(self.points)
        rows = QHBoxLayout()
        self.add_point = QPushButton("Add at crosshair", self)
        self.add_point.clicked.connect(
            lambda: self._act(
                self.viewer.add_bundle_point, self.viewer.scene.view.cursor
            )
        )
        self.remove_point = QPushButton("Remove last", self)
        self.remove_point.clicked.connect(
            lambda: self._act(self.viewer.remove_bundle_point)
        )
        rows.addWidget(self.add_point)
        rows.addWidget(self.remove_point)
        form.addRow(rows)
        self._point_buttons = rows

    def _build_scan(self, form: QFormLayout, more: QFormLayout) -> None:
        """Builds the Scan section; the sequence, timings and seeds go in More."""
        self.sequence = ChoiceBox(self)
        self.sequence.set_choices(
            [
                (EchoKind.SPIN_ECHO, "Spin echo (T2)"),
                (EchoKind.GRADIENT_ECHO, "Gradient echo (T2*)"),
            ]
        )
        self.sequence.chosen.connect(
            lambda sequence: self._scan(ScanField.SEQUENCE, sequence)
        )
        add_field_row(more, "Sequence", HELP["Sequence"], self.sequence, self)
        self.te = QLineEdit(self)
        self.te.setToolTip("One echo time, or several separated by commas")
        self.te.editingFinished.connect(self._edit_te)
        add_field_row(form, "TE (ms)", HELP["TE (ms)"], self.te, self)
        self.scan_boxes: dict[ScanField, QDoubleSpinBox | QSpinBox] = {
            ScanField.TR_MS: _spin(1, 100_000, 1),
            ScanField.SMALL_DELTA_MS: _spin(0.1, 1000, 2),
            ScanField.BIG_DELTA_MS: _spin(0.1, 1000, 2),
            ScanField.NUM_B0: _int_spin(0, 1000),
            ScanField.GRADIENT_SEED: _int_spin(0, 2**31 - 1),
            ScanField.SEED: _int_spin(0, 2**31 - 1),
        }
        labels = {
            ScanField.TR_MS: "TR (ms)",
            ScanField.SMALL_DELTA_MS: "delta (ms)",
            ScanField.BIG_DELTA_MS: "Delta (ms)",
            ScanField.NUM_B0: "b = 0 volumes",
            ScanField.GRADIENT_SEED: "Directions seed",
            ScanField.SEED: "Noise seed",
        }
        for name, box in self.scan_boxes.items():
            box.valueChanged.connect(lambda value, n=name: self._scan(n, value))
            add_field_row(more, labels[name], HELP[labels[name]], box, self)
        self._build_gradients(form, more)
        self._build_noise(form, more)

    def _build_gradients(self, form: QFormLayout, more: QFormLayout) -> None:
        """Adds the shells' table, and the choice of bvals and bvecs files to more."""
        self.shells = QTableWidget(0, 2, self)
        self.shells.setHorizontalHeaderLabels(["b (s/mm^2)", "Directions"])
        self.shells.verticalHeader().setVisible(False)
        self.shells.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.shells.setMinimumHeight(90)
        self.shells.itemChanged.connect(lambda _: self._edit_shells())
        form.addRow(self.shells)
        rows = QHBoxLayout()
        self.add_shell = QPushButton("Add shell", self)
        self.add_shell.clicked.connect(self._add_shell)
        self.remove_shell = QPushButton("Remove", self)
        self.remove_shell.clicked.connect(self._remove_shell)
        rows.addWidget(self.add_shell)
        rows.addWidget(self.remove_shell)
        form.addRow(rows)
        files = QHBoxLayout()
        self.gradient_label = QLabel(self)
        self.gradient_label.setWordWrap(True)
        self.gradient_files = QPushButton("bvals/bvecs...", self)
        self.gradient_files.clicked.connect(self._ask_gradients)
        self.clear_gradients = QPushButton("Use shells", self)
        self.clear_gradients.clicked.connect(
            lambda: self._scan(ScanField.GRADIENT_FILES, None)
        )
        files.addWidget(self.gradient_files)
        files.addWidget(self.clear_gradients)
        more.addRow(files)
        self.bvec_frame = ChoiceBox(self)
        self.bvec_frame.set_choices(
            [(BvecFrame.FSL, "FSL (grid axes)"), (BvecFrame.WORLD, "World (RAS)")]
        )
        self.bvec_frame.chosen.connect(
            lambda frame: self._scan(ScanField.BVEC_FRAME, frame)
        )
        add_field_row(more, "bvecs frame", HELP["bvecs frame"], self.bvec_frame, self)
        more.addRow(self.gradient_label)

    def _build_noise(self, form: QFormLayout, more: QFormLayout) -> None:
        """Adds the noise switch and its SNR, and its reference tissue to more."""
        noise = QHBoxLayout()
        self.noise = QCheckBox("Noise, SNR", self)
        self.noise.setToolTip(NOISE_HELP)
        self.noise.clicked.connect(self._edit_snr)
        self.snr = _spin(0.1, 10_000, 1)
        self.snr.setValue(20.0)
        self.snr.valueChanged.connect(lambda _: self._edit_snr())
        noise.addWidget(self.noise)
        noise.addWidget(self.snr)
        form.addRow(noise)
        self.noise_reference = ChoiceBox(self)
        self.noise_reference.chosen.connect(
            lambda tissue: self._scan(ScanField.NOISE_REFERENCE, tissue)
        )
        add_field_row(
            more,
            "Noise reference",
            HELP["Noise reference"],
            self.noise_reference,
            self,
        )

    def _build_proxies(self, form: QFormLayout) -> None:
        """Builds the Proxy maps section."""
        self.proxy_boxes: dict[ProxyMap, QCheckBox] = {}
        for kind, name in PROXY_NAMES.items():
            box = QCheckBox(name, self)
            box.setToolTip(PROXY_HELP[kind])
            box.clicked.connect(
                lambda shown, k=kind: self._act(self.viewer.show_proxy, k, shown)
            )
            self.proxy_boxes[kind] = box
            form.addRow(box)

    def _build_run(self, form: QFormLayout) -> None:
        """Builds the Run section."""
        folder = QHBoxLayout()
        self.output = QLabel(self)
        self.output.setWordWrap(True)
        self.choose_output = QPushButton("Folder...", self)
        self.choose_output.clicked.connect(self._ask_output)
        folder.addWidget(self.output, 1)
        folder.addWidget(self.choose_output)
        form.addRow(folder)
        buttons = QHBoxLayout()
        self.run_button = QPushButton("Run", self)
        self.run_button.setToolTip("Simulate and open the outputs (Ctrl+R)")
        self.run_button.clicked.connect(self.run_requested)
        self.stop_button = QPushButton("Cancel", self)
        self.stop_button.clicked.connect(self.cancel_requested)
        self.stop_button.hide()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.stop_button)
        form.addRow(buttons)

    # Showing the phantom

    def _on_event(self, event: Event) -> None:
        """Shows the phantom again after a scene event that concerns it."""
        if isinstance(event, PhantomChanged):
            self.refresh()
        elif isinstance(event, (ItemAdded, ItemRemoved)):
            self._refresh_choices(self.viewer.phantom)
        elif isinstance(event, ViewStateChanged) and event.name == ViewField.SELECTED:
            self._refresh_bundle(self.viewer.phantom)

    def refresh(self) -> None:
        """Shows the phantom as the scene holds it."""
        state = self.viewer.phantom
        self.hint.setVisible(state is None)
        self.start_hint.setVisible(False)
        for name, section in self.sections.items():
            section.setVisible(self._section_shown(name))
        self.save_button.setEnabled(state is not None)
        self.close_button.setEnabled(state is not None)
        if state is None:
            return
        self.error.hide()
        # The guided steps say how to start, in place of the hint.
        self.start_hint.setVisible(
            self._only is None
            and not state.bundles
            and not state.tissue_maps
            and state.draft is None
        )
        self.show_box.setChecked(state.show_box)
        self._refresh_grid(state)
        self._refresh_tissues(state)
        self._refresh_bundles(state)
        self._refresh_scan(state)
        self._refresh_choices(state)
        for kind, box in self.proxy_boxes.items():
            box.setChecked(state.proxy(kind) is not None)
        self.output.setText(str(state.output) if state.output else "No folder yet")
        self._show_run(state)

    def set_running(self, running: bool) -> None:
        """Shows Cancel and greys out Run while a simulation runs."""
        self.running = running
        self._show_run(self.viewer.phantom)

    def _show_run(self, state: PhantomState | None) -> None:
        """Enables Run once the phantom can run, with the first problem as its tip."""
        self.stop_button.setVisible(self.running)
        problems = (
            run_problems(state, item_sources(state, self.viewer.scene))
            if state is not None
            else []
        )
        self.run_button.setEnabled(
            state is not None and not problems and not self.running
        )
        if self.running:
            self.run_button.setToolTip("A simulation is running")
        elif problems:
            self.run_button.setToolTip(problems[0].text)
        else:
            self.run_button.setToolTip("Simulate and open the outputs (Ctrl+R)")

    def _refresh_grid(self, state: PhantomState) -> None:
        """Shows the grid, the tissue maps and the background."""
        grid = state.grid
        sizes = [float(abs(grid.affine[i, i])) for i in range(3)]
        x, y, z = grid.shape
        self.grid_label.setText(
            f"Grid: {x} x {y} x {z} voxels of "
            + " x ".join(f"{s:g}" for s in sizes)
            + " mm"
        )
        if not any(b.hasFocus() for b in (*self.size_boxes, self.voxel_box)):
            for box, count, size in zip(
                self.size_boxes, grid.shape, sizes, strict=True
            ):
                _show(box, count * size)
            _show(self.voxel_box, sizes[0])
        unoriented = [t.name for t in state.tissues if not _oriented(t)]
        while self.map_form.rowCount():
            self.map_form.removeRow(0)
        self.map_fields: dict[str, ItemField] = {}
        overlays = self.viewer.scene.in_category(Category.OVERLAYS)
        owned = state.owned()
        choices: list[tuple[object, str]] = [
            (i.id, i.name) for i in overlays if i.id not in owned
        ]
        maps = dict(state.tissue_maps)
        for name in unoriented:
            field = ItemField(self)
            field.set_choices(choices)
            field.show_value(maps.get(name))
            field.edited.connect(
                lambda value, n=name: self._act(self.viewer.set_tissue_map, n, value)
            )
            self.map_fields[name] = field
            add_field_row(self.map_form, f"{name} map", _map_help(name), field, self)
        self.background.set_choices(
            [(None, "None (no signal)"), *((name, name) for name in unoriented)]
        )
        self.background.show_value(state.background)

    def _refresh_tissues(self, state: PhantomState) -> None:
        """Shows the tissue list, keeping the tissue selected."""
        names = [t.name for t in state.tissues]
        current = self._tissue_name()
        with QSignalBlocker(self.tissue_list):
            if [
                self.tissue_list.item(i).text() for i in range(self.tissue_list.count())
            ] != names:
                self.tissue_list.clear()
                self.tissue_list.addItems(names)
            row = names.index(current) if current in names else 0
            self.tissue_list.setCurrentRow(row)
        self._show_tissue()

    def _tissue_name(self) -> str | None:
        """Returns the name of the tissue selected; None if none is."""
        row = self.tissue_list.currentRow()
        return self.tissue_list.item(row).text() if row >= 0 else None

    def _show_tissue(self) -> None:
        """Shows the selected tissue's preset parameters or custom compartments."""
        state = self.viewer.phantom
        name = self._tissue_name()
        if state is None or name is None:
            return
        tissue = state.tissue(name)
        self.preset.show_value(tissue.preset)
        custom = tissue.preset == TissuePreset.CUSTOM
        self.parameters.setVisible(not custom)
        self.custom.setVisible(custom)
        if custom:
            self.sections["Tissues"].set_more_shown(True)  # the table is under More
        if not custom:
            keys = list(preset_defaults(tissue.preset))
            if (
                list(self.parameter_boxes) != keys
                or self._parameter_preset != tissue.preset
            ):
                self._parameter_preset = tissue.preset
                while self.parameter_form.rowCount():
                    self.parameter_form.removeRow(0)
                self.parameter_boxes = {}
                for key in keys:
                    box = _spin(0, 100_000, 4)
                    box.valueChanged.connect(
                        lambda value, k=key: self._edit_parameter(k, value)
                    )
                    self.parameter_boxes[key] = box
                    add_field_row(
                        self.parameter_form,
                        _parameter_label(tissue.preset, key),
                        _parameter_help(tissue.preset, key),
                        box,
                        self,
                    )
            for key, box in self.parameter_boxes.items():
                scale = DIFFUSIVITY_SCALE if "diffusivity" in key else 1.0
                _show(box, tissue.value(key) * scale)
            return
        rows = tissue.compartments
        with QSignalBlocker(self.compartments):
            self.compartments.setRowCount(len(rows))
            for row, compartment in enumerate(rows):
                self._compartment_row(row, compartment)
        self.fraction_bar.set_fractions([c.fraction for c in rows])

    def _compartment_row(self, row: int, compartment: CompartmentSettings) -> None:
        """Fills a row of the compartments' table."""
        kind = ChoiceBox(self.compartments)
        kind.set_choices([(value, value.value) for value in CompartmentKind])
        kind.show_value(compartment.kind)
        kind.chosen.connect(lambda _: self._edit_compartments())
        self.compartments.setCellWidget(row, 0, kind)
        for column, (key, _) in enumerate(COMPARTMENT_COLUMNS, 1):
            value = getattr(compartment, key)
            if "diffusivity" in key:
                value *= DIFFUSIVITY_SCALE
            self.compartments.setItem(row, column, QTableWidgetItem(f"{value:g}"))

    def _refresh_bundles(self, state: PhantomState) -> None:
        """Shows the bundles' table, the drawn bundle's points and the bundle form."""
        scene = self.viewer.scene
        draft = state.draft
        rows: list[tuple[ItemId | None, str, str, str]] = []
        for bundle in state.bundles:
            name = scene.item(bundle.item_id).name
            points = (
                "file"
                if bundle.geometry is None
                else str(len(bundle.geometry.control_points))
            )
            if draft is not None and draft.item_id == bundle.item_id:
                points = f"{len(draft.points)} (editing)"
            rows.append((bundle.item_id, name, bundle.tissue, points))
        if draft is not None and draft.editing is None:
            rows.append(
                (draft.item_id, "(drawing)", draft.tissue, str(len(draft.points)))
            )
        names = [t.name for t in state.tissues]
        with QSignalBlocker(self.bundle_table):
            self.bundle_table.setRowCount(len(rows))
            for row, (item_id, name, tissue, points) in enumerate(rows):
                cell = QTableWidgetItem(name)
                cell.setData(Qt.ItemDataRole.UserRole, item_id)
                self.bundle_table.setItem(row, 0, cell)
                combo = ChoiceBox(self.bundle_table)
                combo.set_choices([(name, name) for name in names])
                combo.show_value(tissue)
                combo.chosen.connect(
                    lambda tissue, i=item_id: self._act(
                        self.viewer.set_bundle, i, BundleField.TISSUE, tissue
                    )
                )
                self.bundle_table.setCellWidget(row, 1, combo)
                self.bundle_table.setItem(row, 2, QTableWidgetItem(points))
        drawing = draft is not None
        self.draw_card.setVisible(drawing)
        if draft is not None:
            count = len(draft.points)
            self.placed.setText(f"{count} point{'' if count == 1 else 's'} placed")
        self.finish_button.setEnabled(drawing)
        self.cancel_button.setEnabled(drawing)
        self.add_point.setEnabled(drawing)
        self.remove_point.setEnabled(drawing)
        self.points.setEnabled(drawing)
        with QSignalBlocker(self.points):
            points_shown = draft.points if draft is not None else ()
            self.points.setRowCount(len(points_shown))
            for row, point in enumerate(points_shown):
                for column, value in enumerate(point):
                    self.points.setItem(row, column, QTableWidgetItem(f"{value:.2f}"))
        self.draft_tissue.set_choices([(name, name) for name in names])
        self._refresh_bundle(state)

    def _bundle_target(self, state: PhantomState | None) -> ItemId | None:
        """Returns the bundle the form edits, the one being drawn or else selected."""
        if state is None:
            return None
        if state.draft is not None:
            return state.draft.item_id
        selected = self.viewer.scene.view.selected
        if isinstance(selected, int) and any(
            b.item_id == selected for b in state.bundles
        ):
            return selected
        return None

    def _refresh_bundle(self, state: PhantomState | None) -> None:
        """Shows the settings of the bundle being drawn, or of the one selected."""
        if state is None:
            return
        draft = state.draft
        target = self._bundle_target(state)
        bundle = None
        if draft is None and target is not None:
            bundle = state.bundle(target)
        geometry = (
            draft.geometry
            if draft is not None
            else (bundle.geometry if bundle else None)
        )
        # Only the settings of a bundle drawn or selected show, so no field shows
        # a value that belongs to nothing.
        for name, box in self.geometry_boxes.items():
            self._set_row_visible(box, geometry is not None)
            if geometry is not None:
                _show(box, float(getattr(geometry, name.value)))
        settings = bundle if bundle is not None else (draft.editing if draft else None)
        self._set_row_visible(self.area_box, settings is not None)
        self._set_row_visible(
            self.fill_box, settings is not None and settings.geometry is None
        )
        self._set_row_visible(self.points, draft is not None)
        self._set_row_visible(self._point_buttons, draft is not None)
        if settings is not None:
            _show(self.area_box, settings.streamline_area or 0.0)
            _show(self.fill_box, settings.densest_fill)
        tissue = (
            draft.tissue if draft is not None else (bundle.tissue if bundle else None)
        )
        self._set_row_visible(self.draft_tissue, tissue is not None)
        if tissue is not None:
            self.draft_tissue.show_value(tissue)
        self.edit_button.setEnabled(
            draft is None and bundle is not None and bundle.geometry is not None
        )
        self.save_bundle.setEnabled(draft is None and bundle is not None)

    def _set_row_visible(self, row: QWidget | QHBoxLayout, shown: bool) -> None:
        """Shows or hides a row of the Bundles section, in its form or under More."""
        for form in self._bundle_forms:
            found = cast(
                tuple[int, object],
                form.getWidgetPosition(row)
                if isinstance(row, QWidget)
                else form.getLayoutPosition(row),
            )
            if found[0] >= 0:
                form.setRowVisible(row, shown)
                return

    def _draw(self) -> None:
        """Starts drawing a bundle with the Draw bundle tool."""
        if self._act(self.viewer.start_bundle):
            self.viewer.set_tool(Tool.DRAW_BUNDLE)

    def _refresh_scan(self, state: PhantomState) -> None:
        """Shows the scan."""
        scan = state.scan
        self.sequence.show_value(scan.sequence)
        if not self.te.hasFocus():
            self.te.setText(", ".join(f"{t:g}" for t in scan.te_ms))
        values = {
            ScanField.TR_MS: scan.tr_ms,
            ScanField.SMALL_DELTA_MS: scan.small_delta_ms,
            ScanField.BIG_DELTA_MS: scan.big_delta_ms,
            ScanField.NUM_B0: scan.num_b0,
            ScanField.GRADIENT_SEED: scan.gradient_seed,
            ScanField.SEED: scan.seed,
        }
        for name, box in self.scan_boxes.items():
            _show(box, float(values[name]))
        with QSignalBlocker(self.shells):
            self.shells.setRowCount(len(scan.shells))
            for row, (b, n) in enumerate(scan.shells):
                self.shells.setItem(row, 0, QTableWidgetItem(f"{b:g}"))
                self.shells.setItem(row, 1, QTableWidgetItem(str(n)))
        files = scan.gradient_files
        shells = files is None
        for widget in (self.shells, self.add_shell, self.remove_shell):
            widget.setEnabled(shells)
        self.scan_boxes[ScanField.NUM_B0].setEnabled(shells)
        self.scan_boxes[ScanField.GRADIENT_SEED].setEnabled(shells)
        self.bvec_frame.show_value(scan.bvec_frame)
        if files is not None:
            text = f"{files[0].name} and {files[1].name}"
        elif scan.table is not None:
            text = (
                f"The phantom file's {len(scan.table.bvals)} volumes, as the file "
                "has them. Editing the shells replaces them."
            )
        else:
            text = "Directions from the shells"
        self.gradient_label.setText(text)
        self.noise_reference.set_choices([(t.name, t.name) for t in state.tissues])
        self.noise_reference.show_value(scan.noise_reference)
        self.noise.setChecked(scan.snr is not None)
        self.snr.setEnabled(scan.snr is not None)
        if scan.snr is not None:
            _show(self.snr, scan.snr)

    def _refresh_choices(self, state: PhantomState | None) -> None:
        """Offers the loaded images and tractograms the phantom can use."""
        if state is None:
            return
        scene = self.viewer.scene
        images: list[tuple[object, str]] = [
            (i.id, i.name)
            for c in (Category.VOLUMES, Category.OVERLAYS)
            for i in scene.in_category(c)
        ]
        self.grid_of.set_choices(images)
        self.grid_of.show_value(None)
        owned = state.owned()
        tracts: list[tuple[object, str]] = [
            (i.id, i.name)
            for i in scene.in_category(Category.TRACTOGRAMS)
            if i.id not in owned
        ]
        self.adopt.set_choices(tracts)
        self.adopt.setItemText(0, "Use a tractogram...")
        self.adopt.show_value(None)

    # Edits: each through an action; a refusal is shown, and the field keeps the
    # value typed until it is fixed.

    def _act(self, action: Callable[..., object], *args: object) -> bool:
        """Calls a viewer action and shows its error; returns whether it succeeded."""
        try:
            action(*args)
        except (OptionError, OSError, TypeError, ValueError) as e:
            self.error.setText(str(e))
            self.error.show()
            return False
        self.error.hide()
        return True

    def _apply_box(self) -> None:
        """Starts a phantom on the box typed, or moves the open one to it."""
        sizes = tuple(b.value() for b in self.size_boxes)
        voxel = self.voxel_box.value()
        if self.viewer.phantom is None:
            self._act(self.viewer.new_phantom, sizes, voxel)
        else:
            self._act(self.viewer.set_phantom_grid, sizes, voxel)

    def _apply_grid_of(self, item_id: object) -> None:
        """Moves the phantom to the grid of a loaded image."""
        if isinstance(item_id, int):
            self._act(lambda: self.viewer.set_phantom_grid(grid_of=item_id))

    def _ask_tissue(self) -> None:
        """Asks for a name and adds a tissue of free water."""
        name, ok = QInputDialog.getText(self, "Add tissue", "Name of the new tissue:")
        if ok and name.strip():
            self._act(self.viewer.set_tissue, name.strip(), TissuePreset.FREE_WATER)

    def _remove_tissue(self) -> None:
        """Removes the selected tissue."""
        name = self._tissue_name()
        if name is not None:
            self._act(self.viewer.remove_tissue, name)

    def _edit_preset(self, preset: object) -> None:
        """Sets the selected tissue's preset."""
        name = self._tissue_name()
        if name is not None:
            self._act(self.viewer.set_tissue, name, preset)

    def _edit_parameter(self, key: str, value: float) -> None:
        """Sets a preset parameter of the selected tissue; diffusivities in um^2/ms."""
        name = self._tissue_name()
        state = self.viewer.phantom
        if name is None or state is None:
            return
        scale = DIFFUSIVITY_SCALE if "diffusivity" in key else 1.0
        preset = state.tissue(name).preset
        self._act(self.viewer.set_tissue, name, preset, {key: value / scale})

    def _compartment_values(self) -> list[dict[str, Any]]:
        """Reads the compartments' table.

        Returns:
            The values of each row, with text that is not a number as NaN, and the
            diffusivities, typed in um^2/ms, converted to mm^2/s.
        """
        rows = []
        for row in range(self.compartments.rowCount()):
            widget = self.compartments.cellWidget(row, 0)
            assert isinstance(widget, ChoiceBox)
            values: dict[str, Any] = {"kind": widget.value()}
            for column, (key, _) in enumerate(COMPARTMENT_COLUMNS, 1):
                cell = self.compartments.item(row, column)
                try:
                    number = float(cell.text()) if cell is not None else 0.0
                except ValueError:
                    number = float("nan")
                if "diffusivity" in key:
                    number /= DIFFUSIVITY_SCALE
                values[key] = number
            rows.append(values)
        return rows

    def _edit_compartments(self) -> None:
        """Sets the selected tissue's compartments from the table."""
        name = self._tissue_name()
        if name is None:
            return
        rows = self._compartment_values()
        self.fraction_bar.set_fractions([r["fraction"] for r in rows])
        self._act(self.viewer.set_compartments, name, rows)

    def _add_compartment(self) -> None:
        """Adds a ball compartment of zero fraction."""
        row = self.compartments.rowCount()
        with QSignalBlocker(self.compartments):
            self.compartments.insertRow(row)
            self._compartment_row(
                row,
                CompartmentSettings(CompartmentKind.BALL, 0.0, 3e-3, 4000, 2000, 400),
            )
        self._edit_compartments()

    def _remove_compartment(self) -> None:
        """Removes the compartment selected in the table."""
        row = self.compartments.currentRow()
        if row >= 0:
            self.compartments.removeRow(row)
            self._edit_compartments()

    def _select_bundle(self) -> None:
        """Selects the bundle chosen in the table."""
        rows = self.bundle_table.selectionModel().selectedRows()
        if not rows:
            return
        cell = self.bundle_table.item(rows[0].row(), 0)
        item_id = cell.data(Qt.ItemDataRole.UserRole) if cell is not None else None
        if isinstance(item_id, int):
            self._act(self.viewer.select, item_id)

    def _edit_bundle(self) -> None:
        """Starts editing the points of the bundle the form shows."""
        target = self._bundle_target(self.viewer.phantom)
        if target is not None:
            self._act(self.viewer.edit_bundle, target)

    def _save_bundle(self) -> None:
        """Asks the window to save the bundle the form shows."""
        target = self._bundle_target(self.viewer.phantom)
        if target is not None:
            self.save_bundle_requested.emit(target)

    def _adopt(self, item_id: object) -> None:
        """Makes a loaded tractogram a bundle of the phantom."""
        if isinstance(item_id, int):
            self._act(self.viewer.use_as_bundle, item_id)

    def _edit_bundle_value(self, name: BundleField, value: object) -> None:
        """Sets a field of the bundle being drawn, or of the one selected."""
        state = self.viewer.phantom
        if state is None:
            return
        target = None if state.draft is not None else self._bundle_target(state)
        if state.draft is None and target is None:
            return
        self._act(self.viewer.set_bundle, target, name, value)

    def _edit_points(self) -> None:
        """Sets the drawn bundle's points from the table."""
        points = []
        for row in range(self.points.rowCount()):
            try:
                x, y, z = (float(_cell(self.points, row, c)) for c in range(3))
            except AttributeError, ValueError:
                self.error.setText(f"Point {row + 1} needs three numbers.")
                self.error.show()
                return
            points.append((x, y, z))
        self._act(self.viewer.set_bundle_points, points)

    def _scan(self, name: ScanField, value: object) -> None:
        """Sets a field of the scan."""
        self._act(self.viewer.set_scan, name, value)

    def _edit_te(self) -> None:
        """Sets the echo times typed, separated by commas."""
        try:
            times = [
                float(t)
                for t in self.te.text().replace(";", ",").split(",")
                if t.strip()
            ]
        except ValueError:
            self.error.setText("TE: echo times are numbers in ms, separated by commas.")
            self.error.show()
            return
        self._scan(ScanField.TE_MS, times)

    def _edit_shells(self) -> None:
        """Sets the shells from the table."""
        shells = []
        for row in range(self.shells.rowCount()):
            try:
                b = float(_cell(self.shells, row, 0))
                n = int(_cell(self.shells, row, 1))
            except AttributeError, ValueError:
                self.error.setText(f"Shell {row + 1} needs a b-value and a count.")
                self.error.show()
                return
            shells.append((b, n))
        self._scan(ScanField.SHELLS, shells)

    def _add_shell(self) -> None:
        """Adds a shell of 32 directions, 1000 s/mm^2 above the last.

        A scan with no shells gets one at b = 1000 s/mm^2.
        """
        state = self.viewer.phantom
        if state is not None:
            shells = [list(s) for s in state.scan.shells]
            b = shells[-1][0] + 1000 if shells else 1000.0
            self._scan(ScanField.SHELLS, [*shells, [b, 32]])

    def _remove_shell(self) -> None:
        """Removes the shell selected in the table."""
        state = self.viewer.phantom
        row = self.shells.currentRow()
        if state is not None and row >= 0:
            shells = [s for i, s in enumerate(state.scan.shells) if i != row]
            self._scan(ScanField.SHELLS, shells)

    def _ask_gradients(self) -> None:
        """Asks for a bvals and a bvecs file and takes the directions from them."""
        bvals, _ = QFileDialog.getOpenFileName(
            self, "bvals file", "", "bvals (*.bval *.bvals);;All files (*)"
        )
        if not bvals:
            return
        bvecs, _ = QFileDialog.getOpenFileName(
            self,
            "bvecs file",
            str(Path(bvals).parent),
            "bvecs (*.bvec *.bvecs);;All files (*)",
        )
        if bvecs:
            self._scan(ScanField.GRADIENT_FILES, [bvals, bvecs])

    def _edit_snr(self) -> None:
        """Sets the noise's SNR, or no noise."""
        self._scan(ScanField.SNR, self.snr.value() if self.noise.isChecked() else None)

    def _ask_output(self) -> None:
        """Asks for the run's output folder."""
        folder = QFileDialog.getExistingDirectory(self, "Output folder")
        if folder:
            self._act(self.viewer.set_run_folder, folder)


def _cell(table: QTableWidget, row: int, column: int) -> str:
    """Returns a cell's text; empty for a cell never filled."""
    item = table.item(row, column)
    return item.text() if item is not None else ""


def _oriented(settings: TissueSettings) -> bool:
    """Returns whether a tissue needs a fibre direction; False if it cannot be built."""
    try:
        return build_tissue(settings).is_oriented
    except TypeError, ValueError:
        return False
