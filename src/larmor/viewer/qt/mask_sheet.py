"""The New mask sheet, which makes a mask image from a DWI, a threshold or labels."""

import threading
from functools import partial
from pathlib import Path
from typing import Final

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from larmor.types import Volume
from larmor.viewer.model.item import ItemId
from larmor.viewer.model.labels import label_name, labels_in, lookup_table
from larmor.viewer.model.mask_maker import (
    Combination,
    MaskMaker,
    MaskRecipe,
    MaskSource,
    b0_of,
    voxel_volume_ml,
)
from larmor.viewer.model.options import OverlayOptions
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.qt.loading import MainThread

# The sources listed on the left, with Paint shown but not yet offered.
SOURCE_NAMES: Final = {
    MaskSource.DWI: "From the DWI",
    MaskSource.THRESHOLD: "Threshold an image",
    MaskSource.LABELS: "Parcellation labels",
}
PAINT_NAME: Final = "Paint (later)"
# The name a new mask starts with, for each source.
DEFAULT_NAMES: Final = {
    MaskSource.DWI: "Brain mask",
    MaskSource.THRESHOLD: "Threshold mask",
    MaskSource.LABELS: "Labels mask",
}
COMBINATION_NAMES: Final = {
    Combination.UNION: "Union",
    Combination.INTERSECT: "Intersect",
    Combination.SUBTRACT: "Subtract",
}
SOURCE = Qt.ItemDataRole.UserRole  # where a source row keeps its MaskSource
LABEL = Qt.ItemDataRole.UserRole  # where a label row keeps its label
PREVIEW_SIZE: Final = (360, 360)  # px, the preview's smallest size
PREVIEW_DELAY: Final = 120  # ms after the last change before the preview is made
OVERLAY_ALPHA: Final = 0.55  # how strongly the mask's red covers the image
SAVE_FILTER: Final = "Images (*.nii *.nii.gz *.mif *.mif.gz);;All files (*)"
MAX_STEPS: Final = 20  # voxels a mask can grow or shrink by
COMPUTING: Final = "Computing..."  # the preview's text while a brain mask is made


def open_mask_sheet(
    parent: QWidget | None, viewer: Viewer, bvals: Path | None = None
) -> ItemId | None:
    """Opens the New mask sheet and waits until it closes.

    Args:
        parent: The window it belongs to.
        viewer: The viewer whose images it reads and to which the mask is added.
        bvals: The b-values file of the DWI, filled in for the brain mask.

    Returns:
        The new mask's item id, an overlay; None if it was cancelled.
    """
    sheet = MaskSheet(viewer, parent)
    if bvals is not None:
        sheet.bvals.setText(str(bvals))
        sheet.changed()
    sheet.exec()
    return sheet.created


class MaskSheet(QDialog):
    """Makes a mask from a source, with a live preview on one slice.

    On the left the source, in the middle the preview with its voxel count and
    volume, on the right the source's settings, the clean-up, the combination
    and the name. Create mask adds the mask as an overlay; Save... adds it and
    asks for a file to save it to.
    """

    def __init__(self, viewer: Viewer, parent: QWidget | None = None) -> None:
        """Builds the sheet, starting from the first source that has an image."""
        super().__init__(parent)
        self.setWindowTitle("New mask")
        self.setModal(True)
        self.viewer = viewer
        self.maker = MaskMaker(viewer)
        self.created: ItemId | None = None
        self.preview_mask: np.ndarray | None = None  # the mask previewed, None if none
        self._named = False  # the user typed the name
        self._background: tuple[tuple[object, ...], np.ndarray] | None = None
        # Whether a brain mask is being made on another thread.
        self._computing = False
        # The source key of the last brain mask that failed, and why.
        self._failed: tuple[tuple[object, ...], str] | None = None
        self._main = MainThread(self)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(PREVIEW_DELAY)
        self.timer.timeout.connect(self.update_preview)

        self._build_sources()
        self._build_preview()
        self._build_settings()
        footer = self._build_footer()
        body = QHBoxLayout()
        body.addWidget(self.sources)
        body.addLayout(self.preview_column, 1)
        body.addWidget(self.settings)
        layout = QVBoxLayout(self)
        layout.addLayout(body, 1)
        layout.addLayout(footer)
        first = next(
            (s for s in SOURCE_NAMES if self.maker.sources(s)), MaskSource.THRESHOLD
        )
        self.choose_source(first)

    # Building

    def _build_sources(self) -> None:
        """Builds the list of sources, Paint disabled."""
        self.sources = QListWidget(self)
        self.sources.setMaximumWidth(180)
        for source, name in SOURCE_NAMES.items():
            row = QListWidgetItem(name)
            row.setData(SOURCE, source)
            self.sources.addItem(row)
        paint = QListWidgetItem(PAINT_NAME)
        paint.setFlags(Qt.ItemFlag.NoItemFlags)
        self.sources.addItem(paint)
        self.sources.currentItemChanged.connect(self._source_clicked)

    def _source_clicked(self, row: QListWidgetItem | None, _: object) -> None:
        """Shows the settings of the source chosen in the list."""
        if row is not None and row.data(SOURCE) is not None:
            if row.data(SOURCE) != self.source:
                self.choose_source(row.data(SOURCE))

    def _build_preview(self) -> None:
        """Builds the preview, its slice slider and its count."""
        self.preview = QLabel(self)
        self.preview.setObjectName("screenshotPreview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(*PREVIEW_SIZE)
        self.preview.setWordWrap(True)
        self.slice = QSlider(Qt.Orientation.Horizontal, self)
        self.slice.setToolTip("The axial slice shown; the wheel steps it too")
        self.slice.valueChanged.connect(lambda _: self._show_slice())
        self.count = QLabel(self)
        self.preview_column = QVBoxLayout()
        self.preview_column.addWidget(self.preview, 1)
        self.preview_column.addWidget(self.slice)
        self.preview_column.addWidget(self.count)

    def _build_settings(self) -> None:
        """Builds the source pages, the clean-up, the combination and the name."""
        self.settings = QWidget(self)
        self.settings.setMinimumWidth(280)
        self.pages = QStackedWidget(self.settings)
        self.pages.addWidget(self._dwi_page())
        self.pages.addWidget(self._threshold_page())
        self.pages.addWidget(self._labels_page())
        form = QFormLayout()
        self._clean_up_rows(form)
        self._combine_rows(form)
        self.name = QLineEdit(self.settings)
        self.name.textEdited.connect(self._name_typed)
        form.addRow("Name", self.name)
        self.error = QLabel(self.settings)
        self.error.setObjectName("fieldError")
        self.error.setWordWrap(True)
        self.error.hide()
        layout = QVBoxLayout(self.settings)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.pages)
        layout.addLayout(form)
        layout.addWidget(self.error)
        layout.addStretch(1)

    def _clean_up_rows(self, form: QFormLayout) -> None:
        """Adds the clean-up: dilate, erode, fill holes and keep the largest piece."""
        self.dilate = QSpinBox(self.settings)
        self.erode = QSpinBox(self.settings)
        for box, tip in ((self.dilate, "Grow"), (self.erode, "Then shrink")):
            box.setRange(0, MAX_STEPS)
            box.setSuffix(" voxels")
            box.setToolTip(f"{tip} the mask by this many voxels")
            box.valueChanged.connect(self.changed)
        self.fill_holes = QCheckBox("Fill holes", self.settings)
        self.largest = QCheckBox("Keep the largest piece", self.settings)
        for check in (self.fill_holes, self.largest):
            check.toggled.connect(self.changed)
        clean = QLabel("Clean up", self.settings)
        clean.setObjectName("formTitle")
        form.addRow(clean)
        form.addRow("Dilate", self.dilate)
        form.addRow("Erode", self.erode)
        form.addRow(self.fill_holes)
        form.addRow(self.largest)

    def _combine_rows(self, form: QFormLayout) -> None:
        """Adds the mask to combine with and how."""
        self.combine_with = QComboBox(self.settings)
        self.combination = QComboBox(self.settings)
        for combination, name in COMBINATION_NAMES.items():
            self.combination.addItem(name, combination)
        self.combine_with.activated.connect(self.changed)
        self.combination.activated.connect(self.changed)
        combine = QLabel("Combine with", self.settings)
        combine.setObjectName("formTitle")
        form.addRow(combine)
        form.addRow("Mask", self.combine_with)
        form.addRow("How", self.combination)

    def _image_box(self, page: QWidget) -> QComboBox:
        """Returns a chooser of a source's image that updates the sheet."""
        box = QComboBox(page)
        box.activated.connect(lambda _: self._image_changed())
        return box

    def _dwi_page(self) -> QWidget:
        """Builds the settings of the brain mask from a DWI."""
        page = QWidget(self)
        form = QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)
        self.dwi_box = self._image_box(page)
        self.bvals = QLineEdit(page)
        self.bvals.setPlaceholderText("None: the first volume is the b0")
        self.bvals.editingFinished.connect(self.changed)
        browse = QPushButton("Browse...", page)
        browse.clicked.connect(self._browse_bvals)
        row = QHBoxLayout()
        row.addWidget(self.bvals, 1)
        row.addWidget(browse)
        note = QLabel("The brain mask of the mean b0, as lar mask computes it.", page)
        note.setWordWrap(True)
        form.addRow(note)
        form.addRow("DWI", self.dwi_box)
        form.addRow("b-values", row)
        return page

    def _threshold_page(self) -> QWidget:
        """Builds the settings of a threshold of an image."""
        page = QWidget(self)
        form = QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)
        self.threshold_box = self._image_box(page)
        self.lower_bound = QDoubleSpinBox(page)
        self.upper_bound = QDoubleSpinBox(page)
        for box in (self.lower_bound, self.upper_bound):
            box.setDecimals(3)
            box.setRange(-1e9, 1e9)
            box.valueChanged.connect(self.changed)
        self.use_upper = QCheckBox("Upper bound", page)
        self.use_upper.toggled.connect(self.upper_bound.setEnabled)
        self.use_upper.toggled.connect(self.changed)
        self.upper_bound.setEnabled(False)
        form.addRow("Image", self.threshold_box)
        form.addRow("At least", self.lower_bound)
        form.addRow(self.use_upper)
        form.addRow("At most", self.upper_bound)
        return page

    def _labels_page(self) -> QWidget:
        """Builds the settings of a mask of parcellation labels."""
        page = QWidget(self)
        form = QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)
        self.labels_box = self._image_box(page)
        self.label_search = QLineEdit(page)
        self.label_search.setPlaceholderText("Search by number or name")
        self.label_search.setClearButtonEnabled(True)
        self.label_search.textChanged.connect(self._filter_labels)
        self.label_list = QListWidget(page)
        self.label_list.setMinimumHeight(160)
        self.label_list.itemChanged.connect(lambda _: self.changed())
        form.addRow("Parcellation", self.labels_box)
        form.addRow(self.label_search)
        form.addRow(self.label_list)
        return page

    def _build_footer(self) -> QHBoxLayout:
        """Builds Cancel, Save... and Create mask."""
        self.cancel_button = QPushButton("Cancel", self)
        self.save_button = QPushButton("Save...", self)
        self.save_button.setToolTip("Create the mask, then save it to a file")
        self.create_button = QPushButton("Create mask", self)
        self.create_button.setDefault(True)
        self.cancel_button.clicked.connect(self.reject)
        self.save_button.clicked.connect(self.create_and_save)
        self.create_button.clicked.connect(self.create_mask)
        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(self.cancel_button)
        footer.addWidget(self.save_button)
        footer.addWidget(self.create_button)
        return footer

    # Settings

    @property
    def source(self) -> MaskSource:
        """Returns the source chosen."""
        return list(SOURCE_NAMES)[self.pages.currentIndex()]

    def choose_source(self, source: MaskSource) -> None:
        """Shows a source's settings and lists the images it can be made from."""
        index = list(SOURCE_NAMES).index(source)
        self.pages.setCurrentIndex(index)
        row = self.sources.item(index)
        if self.sources.currentItem() is not row:
            self.sources.setCurrentItem(row)
        box = self._source_box()
        box.clear()
        for item in self.maker.sources(source):
            box.addItem(item.name, item.id)
        if not self._named:
            self.name.setText(DEFAULT_NAMES[source])
        self._image_changed()

    def _source_box(self) -> QComboBox:
        """Returns the chooser of the current source's image."""
        return {
            MaskSource.DWI: self.dwi_box,
            MaskSource.THRESHOLD: self.threshold_box,
            MaskSource.LABELS: self.labels_box,
        }[self.source]

    def recipe(self) -> MaskRecipe:
        """Returns the recipe the settings describe."""
        image = self._source_box().currentData()
        combine = self.combine_with.currentData()
        return MaskRecipe(
            source=self.source,
            image=image,
            bvals=self.bvals.text().strip(),
            lower=self.lower_bound.value(),
            upper=self.upper_bound.value() if self.use_upper.isChecked() else None,
            labels=tuple(
                int(row.data(LABEL))
                for row in self._label_rows()
                if row.checkState() == Qt.CheckState.Checked
            ),
            dilate=self.dilate.value(),
            erode=self.erode.value(),
            fill_holes=self.fill_holes.isChecked(),
            largest=self.largest.isChecked(),
            combine_with=combine,
            combination=self.combination.currentData(),
        )

    def _image_changed(self) -> None:
        """Lists what depends on the image: masks to combine, labels, range, slices."""
        image = self._source_box().currentData()
        self.combine_with.clear()
        self.combine_with.addItem("None", None)
        if image is not None:
            for item in self.maker.masks_like(image):
                self.combine_with.addItem(item.name, item.id)
            volume = self.viewer.scene.item(image).data
            assert isinstance(volume, Volume)
            depth = volume.data.shape[2]
            self.slice.setRange(0, depth - 1)
            self.slice.setValue(depth // 2)
        if self.source == MaskSource.LABELS:
            self._list_labels(image)
        self.changed()

    def _list_labels(self, image: ItemId | None) -> None:
        """Lists a parcellation's labels with boxes, all unticked."""
        self.label_list.blockSignals(True)
        self.label_list.clear()
        if image is not None:
            item = self.viewer.scene.item(image)
            o = item.options
            table = (
                lookup_table(o.lookup_table) if isinstance(o, OverlayOptions) else {}
            )
            assert isinstance(item.data, Volume)
            for label in labels_in(item.data, 0):
                row = QListWidgetItem(f"{label}  {label_name(label, table)}")
                row.setData(LABEL, label)
                row.setFlags(row.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                row.setCheckState(Qt.CheckState.Unchecked)
                self.label_list.addItem(row)
        self.label_list.blockSignals(False)
        self._filter_labels(self.label_search.text())

    def _label_rows(self) -> list[QListWidgetItem]:
        """Returns the rows of the labels list."""
        return [self.label_list.item(i) for i in range(self.label_list.count())]

    def _filter_labels(self, text: str) -> None:
        """Hides the label rows whose number and name do not hold the text."""
        words = text.strip().lower()
        for row in self._label_rows():
            row.setHidden(words not in row.text().lower())

    def _browse_bvals(self) -> None:
        """Asks for the b-values file."""
        path, _ = QFileDialog.getOpenFileName(self, "b-values file")
        if path:
            self.bvals.setText(path)
            self.changed()

    def _name_typed(self, _: str) -> None:
        """Keeps the name the user typed when the source changes."""
        self._named = True

    # Preview

    def changed(self) -> None:
        """Makes the preview again shortly after the last change."""
        self.timer.start()

    def update_preview(self) -> None:
        """Makes the mask for the settings and shows the slice.

        A brain mask not made yet is made on another thread, the preview saying
        so meanwhile; the preview is made again once it is ready.
        """
        self.timer.stop()
        recipe = self.recipe()
        if (
            recipe.source == MaskSource.DWI
            and recipe.image is not None
            and not self.maker.kept(recipe)
        ):
            self._make_brain_mask(recipe)
            return
        try:
            self.preview_mask = self.maker.make(recipe)
        except (ValueError, OSError) as e:
            self.preview_mask = None
            self._show_error(str(e))
        else:
            self.error.hide()
        self._show_slice()

    def _make_brain_mask(self, recipe: MaskRecipe) -> None:
        """Makes the brain mask of a recipe on another thread, unless it failed."""
        key = self.maker.source_key(recipe)
        self.preview_mask = None
        if self._failed is not None and self._failed[0] == key:
            self._show_error(self._failed[1])
            self._show_slice()
            return
        self.error.hide()
        self.preview.setPixmap(QPixmap())
        self.preview.setText(COMPUTING)
        self.count.setText("")
        if self._computing:
            return  # the running one makes the preview again when it ends
        self._computing = True

        def work() -> None:
            """Makes the mask, which the maker keeps, and posts how it went."""
            failure = None
            try:
                self.maker.source_mask(recipe)
            except (ValueError, OSError) as e:
                failure = str(e)
            self._main.post(partial(self._brain_mask_made, key, failure))

        threading.Thread(target=work, name="mask-preview", daemon=True).start()

    def _brain_mask_made(self, key: tuple[object, ...], failure: str | None) -> None:
        """Makes the preview again once a brain mask is made, or failed."""
        self._computing = False
        if failure is not None:
            self._failed = (key, failure)
        self.update_preview()

    def _show_error(self, text: str) -> None:
        """Shows why the mask cannot be made."""
        self.error.setText(text)
        self.error.show()

    def _base(self) -> np.ndarray | None:
        """Returns the image under the preview, (X, Y, Z), kept while it stays."""
        recipe = self.recipe()
        if recipe.image is None:
            return None
        volume = self.viewer.scene.item(recipe.image).data
        assert isinstance(volume, Volume)
        key = (volume, recipe.source, recipe.bvals)
        if self._background is not None and self._background[0] == key:
            return self._background[1]
        try:
            base = b0_of(
                volume, recipe.bvals if recipe.source == MaskSource.DWI else ""
            )
        except ValueError, OSError:
            base = b0_of(volume)
        self._background = (key, base)
        return base

    def _show_slice(self) -> None:
        """Draws the slice of the image with the mask in red, and the count."""
        base = self._base()
        if base is None:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("No image to make the mask from.")
            self.count.setText("")
            return
        z = min(self.slice.value(), base.shape[2] - 1)
        grey = _grey(base[:, :, z])
        rgb = np.repeat(grey[..., None], 3, axis=-1).astype(np.float32)
        if self.preview_mask is not None and self.preview_mask.shape == base.shape:
            inside = self.preview_mask[:, :, z]
            red = np.array([230.0, 40.0, 40.0], np.float32)
            rgb[inside] = (1 - OVERLAY_ALPHA) * rgb[inside] + OVERLAY_ALPHA * red
        self.preview.setPixmap(self._pixmap(rgb))
        if self.preview_mask is None:
            self.count.setText("")
            return
        volume = self.viewer.scene.item(self._source_box().currentData()).data
        assert isinstance(volume, Volume)
        voxels = int(np.count_nonzero(self.preview_mask))
        ml = voxels * voxel_volume_ml(volume)
        self.count.setText(f"{voxels:,} voxels, {ml:.1f} mL (slice {z})")

    def _pixmap(self, rgb: np.ndarray) -> QPixmap:
        """Returns an (X, Y, 3) slice as a picture, x across and y up, fitted."""
        picture = np.ascontiguousarray(np.flipud(np.transpose(rgb, (1, 0, 2))))
        image = QImage(
            picture.astype(np.uint8).tobytes(),
            picture.shape[1],
            picture.shape[0],
            3 * picture.shape[1],
            QImage.Format.Format_RGB888,
        )
        size = self.preview.size()
        return QPixmap.fromImage(image).scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        """Steps the preview's slice with the wheel."""
        step = 1 if event.angleDelta().y() > 0 else -1
        self.slice.setValue(self.slice.value() + step)

    # Creating

    def create_mask(self) -> ItemId | None:
        """Adds the mask as an overlay and closes; shows why if it cannot."""
        try:
            self.created = self.maker.create(self.recipe(), self.name.text())
        except (ValueError, OSError) as e:
            self._show_error(str(e))
            return None
        self.accept()
        return self.created

    def create_and_save(self) -> None:
        """Adds the mask, then asks for a file and saves it there."""
        recipe = self.recipe()
        try:
            created = self.maker.create(recipe, self.name.text())
        except (ValueError, OSError) as e:
            self._show_error(str(e))
            return
        self.created = created
        source = self.viewer.scene.item(recipe.image).source if recipe.image else None
        folder = source.parent if source is not None else Path()
        suggested = str(folder / f"{self.name.text().strip()}.nii.gz")
        path, _ = QFileDialog.getSaveFileName(self, "Save mask", suggested, SAVE_FILTER)
        if path:
            try:
                self.viewer.save_item(created, path)
            except (ValueError, OSError) as e:
                self._show_error(f"The mask is created but not saved: {e}")
                return
        self.accept()


def _grey(values: np.ndarray) -> np.ndarray:
    """Returns a slice scaled to 0 to 255 between its 1st and 99th percentiles."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.zeros(values.shape, np.float32)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        high = low + 1
    scaled = (np.nan_to_num(values, nan=low) - low) / (high - low)
    grey: np.ndarray = np.clip(scaled, 0, 1).astype(np.float32) * 255
    return grey
