"""The MATLAB import dialog, which asks what to load from a .mat file."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from larmor.io import MatlabEntry, read_matlab
from larmor.viewer.model.item import Category, ItemId
from larmor.viewer.model.matlab_import import (
    LOAD_AS_LABELS,
    PRESET_LABELS,
    LoadAs,
    MatlabPreset,
    entry_problem,
    find_reference,
    needs_reference,
    preset_problem,
    reference_grid,
    suggest_preset,
)
from larmor.viewer.model.viewer import Viewer

_REFERENCE_FILTER = "Grids (*.mat *.nii *.nii.gz *.mif *.mif.gz *.mih)"
# Bytes per element of each MATLAB class, for the size column.
_BYTES = {
    "double": 8,
    "single": 4,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "logical": 1,
    "char": 2,
}
COLUMNS = ("Variable", "Size", "Class", "Bytes", "Load as", "")


@dataclass(frozen=True)
class MatlabChoice:
    """What the user chose to load from a MATLAB file."""

    path: Path
    preset: MatlabPreset
    reference: Path | None
    variables: dict[str, LoadAs]  # with the manual preset, what to load each as
    tractogram: ItemId | None  # the tractogram the values per point colour


class MatlabImportDialog(QDialog):
    """Asks what to load from a `.mat` file; never loads by itself.

    The preset that fits the file is selected, the others that do not are
    disabled with the reason as their tip. Load manually lists every variable,
    struct field and cell; a row whose shape cannot be the chosen type says why
    and keeps Load disabled.
    """

    def __init__(
        self, path: str | Path, viewer: Viewer, parent: QWidget | None = None
    ) -> None:
        """Reads the file's listing (not its data) and builds the dialog."""
        super().__init__(parent)
        self.path = Path(path).absolute()
        self.viewer = viewer
        self.file = read_matlab(self.path)
        self.entries = {e.path: e for e in self.file.entries()}
        self.setWindowTitle(f"Import {self.path.name}")
        self.resize(760, 520)
        layout = QVBoxLayout(self)
        group = self._build_presets(layout)
        self._build_reference_row(layout)
        self._build_table(layout)
        self._build_bottom(layout)

        group.buttonToggled.connect(lambda *_: self._update())
        self.reference.textChanged.connect(lambda *_: self._update())
        self._update()

    def _build_presets(self, layout: QVBoxLayout) -> QButtonGroup:
        """Adds the hint and a button per preset, the suggested one checked.

        Returns:
            QButtonGroup: The presets' buttons.
        """
        suggested = suggest_preset(self.file)
        self.hint = QLabel(f"{self.path.name}: suggested {PRESET_LABELS[suggested]}.")
        layout.addWidget(self.hint)
        self.presets: dict[MatlabPreset, QRadioButton] = {}
        group = QButtonGroup(self)
        for preset, label in PRESET_LABELS.items():
            button = QRadioButton(label, self)
            problem = (
                preset_problem(self.file, preset)
                if preset != MatlabPreset.MANUAL
                else None
            )
            button.setEnabled(problem is None)
            button.setToolTip(problem or "")
            group.addButton(button)
            layout.addWidget(button)
            self.presets[preset] = button
        self.presets[suggested].setChecked(True)
        return group

    def _build_reference_row(self, layout: QVBoxLayout) -> None:
        """Adds the reference grid's field, filled with one found next to the file."""
        row = QHBoxLayout()
        row.addWidget(QLabel("Reference grid:", self))
        found = find_reference(self.path)
        self.reference = QLineEdit("" if found is None else str(found), self)
        self.reference.setPlaceholderText("The fit (.mat) or a NIfTI of the image")
        row.addWidget(self.reference, 1)
        choose = QPushButton("Choose...", self)
        choose.clicked.connect(self._choose_reference)
        row.addWidget(choose)
        layout.addLayout(row)

    def _build_table(self, layout: QVBoxLayout) -> None:
        """Adds the table of the file's variables, each with its Load as menu."""
        self.table = QTableWidget(len(self.entries), len(COLUMNS), self)
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.load_as: dict[str, QComboBox] = {}
        self._notes: dict[str, QTableWidgetItem] = {}
        for r, entry in enumerate(self.entries.values()):
            texts = (entry.path, _shape(entry), _class(entry), _bytes(entry))
            for c, text in enumerate(texts):
                self.table.setItem(r, c, QTableWidgetItem(text))
            combo = QComboBox(self.table)
            for load_as, label in LOAD_AS_LABELS.items():
                combo.addItem(label, load_as)
            combo.currentIndexChanged.connect(self._update)
            self.table.setCellWidget(r, 4, combo)
            note = QTableWidgetItem("")
            note.setForeground(Qt.GlobalColor.red)
            self.table.setItem(r, 5, note)
            self._notes[entry.path] = note
            self.load_as[entry.path] = combo
        layout.addWidget(self.table, 1)

    def _build_bottom(self, layout: QVBoxLayout) -> None:
        """Adds the tractogram to colour, the problems line and the buttons."""
        values_row = QHBoxLayout()
        values_row.addWidget(QLabel("Values per point colour:", self))
        self.tractogram = QComboBox(self)
        for item in self.viewer.scene.in_category(Category.TRACTOGRAMS):
            self.tractogram.addItem(item.name, item.id)
        values_row.addWidget(self.tractogram, 1)
        layout.addLayout(values_row)

        self.problem = QLabel("", self)
        self.problem.setWordWrap(True)
        layout.addWidget(self.problem)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Load")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    @property
    def preset(self) -> MatlabPreset:
        """Returns the preset chosen."""
        return next(p for p, b in self.presets.items() if b.isChecked())

    def set_load_as(self, variable: str, load_as: LoadAs) -> None:
        """Chooses what a variable is loaded as, as its row's menu does."""
        combo = self.load_as[variable]
        combo.setCurrentIndex(combo.findData(load_as))

    def choice(self) -> MatlabChoice:
        """Returns what the dialog says to load."""
        text = self.reference.text().strip()
        variables = {
            v: LoadAs(c.currentData())
            for v, c in self.load_as.items()
            if c.currentData() != LoadAs.NOTHING
        }
        tractogram = self.tractogram.currentData()
        return MatlabChoice(
            self.path,
            self.preset,
            Path(text) if text else None,
            variables if self.preset == MatlabPreset.MANUAL else {},
            tractogram if isinstance(tractogram, int) else None,
        )

    def row_problem(self, variable: str) -> str:
        """Returns the reason shown on a variable's row; "" if its choice fits."""
        return self._notes[variable].text()

    def _choose_reference(self) -> None:
        """Asks for the reference grid's file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Reference grid", str(self.path.parent), _REFERENCE_FILTER
        )
        if path:
            self.reference.setText(path)

    def _update(self) -> None:
        """Shows each row's problem and enables Load when the choice can be loaded."""
        choice = self.choice()
        manual = choice.preset == MatlabPreset.MANUAL
        self.table.setEnabled(manual)
        self.tractogram.setEnabled(manual)
        grid = reference_grid(self.path, choice.reference)
        problems = []
        for variable, entry in self.entries.items():
            load_as = LoadAs(self.load_as[variable].currentData())
            reason = entry_problem(entry, load_as, grid, self.entries) or ""
            no_tractogram = self.tractogram.count() == 0
            if not reason and load_as == LoadAs.POINT_VALUES and no_tractogram:
                reason = "No tractogram is loaded to colour."
            self._notes[variable].setText(reason)
            if manual and reason:
                problems.append(f"{variable}: {reason}")
        if manual and not choice.variables:
            problems.append("Choose what to load for at least one variable.")
        if not manual and needs_reference(self.file, choice.preset) and grid is None:
            problems.append("This file stores no grid: choose a reference grid.")
        self.problem.setText("\n".join(problems))
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setEnabled(not problems)


def _shape(entry: MatlabEntry) -> str:
    """Returns an entry's shape as text, such as "96 x 96 x 60"."""
    return " x ".join(map(str, entry.shape)) if entry.shape else "?"


def _class(entry: MatlabEntry) -> str:
    """Returns an entry's MATLAB class, marked if sparse."""
    return f"{entry.matlab_class} (sparse)" if entry.sparse else entry.matlab_class


def _bytes(entry: MatlabEntry) -> str:
    """Returns the size of an entry's data as text; "" if unknown."""
    size = _BYTES.get(entry.matlab_class)
    if size is None or entry.sparse or not entry.shape:
        return ""
    return _human(size * int(np.prod(entry.shape, dtype=np.int64)))


def _human(n: int) -> str:
    """Returns a number of bytes as text in B, KB, MB or GB."""
    if n < 1024:
        return f"{n} B"
    value = n / 1024
    for unit in ("KB", "MB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"
