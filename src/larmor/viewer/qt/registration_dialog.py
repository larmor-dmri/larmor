"""The dialog that opens files through a registration computed by another tool."""

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from larmor.viewer.model.item import Category
from larmor.viewer.model.registration import INTERPOLATIONS, Registration
from larmor.viewer.model.registry import REGISTRY

# FODs are left out: they would need reorienting, which is not supported.
CATEGORIES = tuple(c for c in REGISTRY if c != Category.FODS)
_BY_CATEGORY = "By category"
_FORMATS = {
    "Guess from the files": None,
    "ANTs": "ants",
    "FSL": "fsl",
    "MRtrix3": "mrtrix",
}


@dataclass(frozen=True)
class RegistrationChoice:
    """Holds the files, the category and the registration chosen in the dialog."""

    paths: tuple[Path, ...]
    category: Category
    registration: Registration


class RegistrationDialog(QDialog):
    """Asks for files, a category, transform files and a reference image."""

    def __init__(self, category: Category, parent: QWidget | None = None) -> None:
        """Creates the dialog, with category chosen."""
        super().__init__(parent)
        self.setWindowTitle("Open Through a Registration")
        self._create_fields(category)
        form = self._build_form()
        note = QLabel(
            "Transforms are listed as antsApplyTransforms lists them: those that "
            "resample an image into the reference space. Tractograms and surfaces "
            "are brought into the same space.",
            self,
        )
        note.setWordWrap(True)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(self.problem)
        layout.addWidget(self.buttons)

    def _create_fields(self, category: Category) -> None:
        """Creates the dialog's fields, with category chosen."""
        self.files = QListWidget(self)
        self.category = QComboBox(self)
        for c in CATEGORIES:
            self.category.addItem(REGISTRY[c].label, c)
        if category in CATEGORIES:
            self.category.setCurrentIndex(CATEGORIES.index(category))
        self.transforms = QListWidget(self)
        self.format = QComboBox(self)
        for label, value in _FORMATS.items():
            self.format.addItem(label, value)
        self.format.setToolTip(
            "The tool that wrote the transforms. A 4D NIfTI warp can be FSL's or "
            "MRtrix3's, so it needs one chosen."
        )
        self.reference = QLineEdit(self)
        self.reference.setPlaceholderText("An image of the space to bring them into")
        self.fsl_input = QLineEdit(self)
        self.fsl_input.setPlaceholderText("The FLIRT or FNIRT input, if not the file")
        self.fsl_input.setToolTip(
            "For FSL transforms: the input image of the FLIRT or FNIRT run. "
            "Images default to themselves; tractograms and surfaces need it."
        )
        self.inverse = QCheckBox("Go the other way", self)
        self.inverse.setToolTip(
            "Bring the files from the reference space into the space the "
            "transforms bring images from."
        )
        self.interpolation = QComboBox(self)
        self.interpolation.addItems([_BY_CATEGORY, *INTERPOLATIONS])
        self.interpolation.setToolTip(
            "For images. By category: linear for volumes, nearest for overlays "
            "(labels stay whole) and peaks."
        )
        self.problem = QLabel(self)
        self.problem.setWordWrap(True)

    def _build_form(self) -> QFormLayout:
        """Lays the fields out in a form, with their buttons.

        Returns:
            QFormLayout: The form.
        """
        add_files = QPushButton("Add...", self)
        add_files.clicked.connect(self._ask_files)
        add_transform = QPushButton("Add...", self)
        add_transform.clicked.connect(self._ask_transform)
        add_pair = QPushButton("Add warp and inverse...", self)
        add_pair.setToolTip("A warp with the inverse warp its tool wrote.")
        add_pair.clicked.connect(self._ask_pair)
        remove = QPushButton("Remove", self)
        remove.clicked.connect(self._remove_transform)
        browse = QPushButton("Browse...", self)
        browse.clicked.connect(lambda: self._ask_image(self.reference))
        browse_fsl = QPushButton("Browse...", self)
        browse_fsl.clicked.connect(lambda: self._ask_image(self.fsl_input))

        transform_buttons = QHBoxLayout()
        for button in (add_transform, add_pair, remove):
            transform_buttons.addWidget(button)
        reference_row = QHBoxLayout()
        reference_row.addWidget(self.reference)
        reference_row.addWidget(browse)
        fsl_row = QHBoxLayout()
        fsl_row.addWidget(self.fsl_input)
        fsl_row.addWidget(browse_fsl)

        form = QFormLayout()
        form.addRow("Files", self.files)
        form.addRow("", add_files)
        form.addRow("Open as", self.category)
        form.addRow("Transforms", self.transforms)
        form.addRow("", transform_buttons)
        form.addRow("Format", self.format)
        form.addRow("Reference image", reference_row)
        form.addRow("FSL input image", fsl_row)
        form.addRow("", self.inverse)
        form.addRow("Interpolation", self.interpolation)
        return form

    def add_files(self, paths: list[str | Path]) -> None:
        """Adds files to open."""
        self.files.addItems([str(p) for p in paths])

    def add_transform(
        self, path: str | Path, inverse: str | Path | None = None
    ) -> None:
        """Adds a transform file, with the inverse warp its tool wrote if given."""
        self.transforms.addItem(str(path) if inverse is None else f"{path},{inverse}")

    def choice(self) -> RegistrationChoice:
        """Returns the files, category and registration chosen.

        Returns:
            RegistrationChoice: The choice.

        Raises:
            ValueError: If no file or no transform is chosen, or a choice is wrong.
        """
        paths = tuple(
            Path(self.files.item(i).text()) for i in range(self.files.count())
        )
        if not paths:
            raise ValueError("Add the files to open.")
        transforms: list[str | Path] = [
            self.transforms.item(i).text() for i in range(self.transforms.count())
        ]
        interpolation = self.interpolation.currentText()
        registration = Registration.from_files(
            transforms,
            self.reference.text() or None,
            self.inverse.isChecked(),
            self.format.currentData(),
            None if interpolation == _BY_CATEGORY else interpolation,
            fsl_input=self.fsl_input.text() or None,
        )
        category = Category(self.category.currentData())
        return RegistrationChoice(paths, category, registration)

    def _accept(self) -> None:
        """Closes the dialog if the choice is valid; else shows why not."""
        try:
            self.choice()
        except ValueError as e:
            self.problem.setText(str(e))
            return
        self.accept()

    def _ask_files(self) -> None:
        """Asks for files to open and adds them."""
        paths, _ = QFileDialog.getOpenFileNames(self, "Files to open")
        self.add_files(list(paths))

    def _ask_transform(self) -> None:
        """Asks for a transform file and adds it."""
        path, _ = QFileDialog.getOpenFileName(self, "Transform file")
        if path:
            self.add_transform(path)

    def _ask_pair(self) -> None:
        """Asks for a warp and its inverse warp and adds them as one transform."""
        warp, _ = QFileDialog.getOpenFileName(self, "Warp")
        if not warp:
            return
        inverse, _ = QFileDialog.getOpenFileName(self, "Its inverse warp")
        if inverse:
            self.add_transform(warp, inverse)

    def _remove_transform(self) -> None:
        """Removes the transforms selected in the list."""
        for item in self.transforms.selectedItems():
            self.transforms.takeItem(self.transforms.row(item))

    def _ask_image(self, field: QLineEdit) -> None:
        """Asks for an image and puts its path in a field."""
        path, _ = QFileDialog.getOpenFileName(self, "Image")
        if path:
            field.setText(path)
