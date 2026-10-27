"""The Export sheet, which saves the views as images or a video with a preset."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QHideEvent,
    QImage,
    QPainter,
    QPixmap,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from rendercanvas.qt import loop

from larmor.viewer.model.camera import PRESETS, Camera3D, CameraPreset
from larmor.viewer.model.events import Event
from larmor.viewer.model.export import (
    ExportKind,
    ExportPreset,
    load_presets,
    presets_folder,
    save_preset,
)
from larmor.viewer.model.item import VIEW_GAP, VIEW_KINDS, VIEW_NAMES, ViewKind
from larmor.viewer.model.options import Colour, OptionError, check_colour
from larmor.viewer.model.session import SUFFIX
from larmor.viewer.model.settings import SettingChanged, SettingName, Settings
from larmor.viewer.model.viewer import (
    ROTATION_DURATION,
    ROTATION_FPS,
    ROTATION_SPEED,
    Viewer,
)
from larmor.viewer.qt.style import form_layout
from larmor.viewer.qt.view_grid import ViewGrid
from larmor.viewer.render.offscreen import (
    GIF_MAX_FPS,
    VIEWS_2D,
    OffscreenRenderer,
    layout,
    view_path,
)

ALL_VIEWS = "All views"
USE_CURRENT = "Use current"
PRESET_CAMERA = "The preset's"
PREVIEW_SIZE = (360, 270)  # the logical pixels the preview fits in
REFRESH_DELAY = 150  # ms after the last change before the preview is drawn again
CHECKER = 8  # the side of a square of the board behind a transparent preview
SHEET_SIZE = (860, 520)  # px
# Each format the dialog offers: its filter, its default suffix, and whether it has
# an alpha channel.
FORMATS: dict[str, tuple[str, bool]] = {
    "PNG (*.png)": (".png", True),
    "TIFF (*.tif *.tiff)": (".tif", True),
    "JPEG (*.jpg *.jpeg)": (".jpg", False),
}
VIDEO_FORMATS: dict[str, tuple[str, bool]] = {
    "WebP (*.webp)": (".webp", True),
    "GIF (*.gif)": (".gif", False),
}
SPEED_RANGE = (-360.0, 360.0)  # degrees per second
DURATION_RANGE = (0.1, 600.0)  # seconds
FPS_RANGE = (1, 60)
# The settings whose colours the images take.
COLOURS = (SettingName.BACKGROUND, SettingName.CROSSHAIR)


def preset_rows(preset: ExportPreset) -> list[tuple[str, str]]:
    """Returns a preset's values as the sheet's small table shows them."""
    size = "the views' size" if preset.size is None else "{} x {}".format(*preset.size)
    overlay = {None: "as the views", True: "on", False: "off"}[preset.overlay]
    rows = [
        ("Size of each view", size),
        ("Scale", f"{preset.scale:g} x"),
        ("Background", preset.background or "the views'"),
        ("Information overlay", overlay),
        ("Transparent", "yes" if preset.transparent else "no"),
    ]
    for category, values in preset.overrides.items():
        text = ", ".join(
            f"{name.value.replace('_', ' ')} {getattr(v, 'value', v)}"
            for name, v in values.items()
        )
        rows.append((category.value.capitalize(), text))
    rows.append(("3D camera", "saved" if preset.camera is not None else "the view's"))
    return rows


class ExportSheet(QDialog):
    """Saves one image, the 2D views as images, or a rotation video, with a preset.

    The preview shows the framing, the background, the overlay and the camera;
    the preset's option overrides apply only while the files are drawn.
    """

    def __init__(
        self,
        viewer: Viewer,
        grid: ViewGrid,
        settings: Settings,
        parent: QWidget | None = None,
    ) -> None:
        """Builds the preview and the fields, starting from the Screen preset."""
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.setModal(False)
        self.viewer = viewer
        self.grid = grid
        self.settings = settings
        # Drawn while the sheet is shown; it follows the scene like the views.
        self.renderer: OffscreenRenderer | None = None
        self._drawing = False  # while the sheet draws, its own changes are ignored
        self.presets = load_presets()
        self.kind = ExportKind.IMAGE
        self.view: ViewKind | None = viewer.scene.view.selected_view
        self.preset = next(iter(self.presets.values()))
        self.camera: str = USE_CURRENT
        self.transparent = self.preset.transparent
        self.save_session = False

        self._build_layout()
        self.resize(*SHEET_SIZE)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.setInterval(REFRESH_DELAY)
        self.refresh_timer.timeout.connect(self.refresh)
        self._show_choices()
        unsubscribe_scene = viewer.scene.subscribe(self._on_event)
        unsubscribe_settings = settings.subscribe(self._on_setting)
        self.destroyed.connect(lambda *_: unsubscribe_scene())
        self.destroyed.connect(lambda *_: unsubscribe_settings())
        # The views' size sets the image's: follow their resizes.
        grid.installEventFilter(self)

    # Building

    def _build_layout(self) -> None:
        """Lays out the preview, the fields, and Cancel and Save."""
        self.preview = QLabel(self)
        self.preview.setObjectName("screenshotPreview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(*PREVIEW_SIZE)
        form = QWidget(self)
        self.rows = form_layout(form)
        self._build_fields(form)
        self._build_video_fields(form)
        buttons = QDialogButtonBox(self)
        self.cancel_button = buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.save_button = QPushButton("Save...", self)
        buttons.addButton(self.save_button, QDialogButtonBox.ButtonRole.AcceptRole)
        self.cancel_button.clicked.connect(self.close)
        self.save_button.clicked.connect(self._ask_and_save)
        right = QVBoxLayout()
        right.addWidget(form, 1)
        right.addWidget(buttons)
        outer = QHBoxLayout(self)
        outer.addWidget(self.preview, 1)
        outer.addLayout(right)

    def _build_fields(self, form: QWidget) -> None:
        """Adds What, View, Preset and its table, Camera and the two switches."""
        rows = self.rows
        self.kind_box = QComboBox(form)
        for kind in ExportKind:
            self.kind_box.addItem(kind.value, kind)
        self.kind_box.activated.connect(self._choose_kind)
        rows.addRow("What", self.kind_box)
        self.view_box = QComboBox(form)
        for view in VIEW_KINDS:
            self.view_box.addItem(VIEW_NAMES[view], view)
        self.view_box.addItem(ALL_VIEWS, None)
        self.view_box.activated.connect(self._choose_view)
        rows.addRow("View", self.view_box)
        self.preset_box = QComboBox(form)
        self.preset_box.activated.connect(self._choose_preset)
        rows.addRow("Preset", self.preset_box)
        self.preset_table = QTableWidget(0, 2, form)
        self.preset_table.horizontalHeader().setVisible(False)
        self.preset_table.verticalHeader().setVisible(False)
        self.preset_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.preset_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        header = self.preset_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.preset_table.setMaximumHeight(170)
        rows.addRow(self.preset_table)
        self.edit_preset = QPushButton("Edit preset...", form)
        self.edit_preset.clicked.connect(self.ask_edit_preset)
        rows.addRow("", self.edit_preset)
        self.camera_box = QComboBox(form)
        self.camera_box.activated.connect(self._choose_camera)
        rows.addRow("Camera", self.camera_box)
        self.transparent_box = QCheckBox("Transparent background", form)
        self.transparent_box.clicked.connect(self._choose_transparent)
        rows.addRow(self.transparent_box)
        self.session_box = QCheckBox("Also save the session beside the image", form)
        self.session_box.clicked.connect(lambda on: setattr(self, "save_session", on))
        rows.addRow(self.session_box)
        self.size_label = QLabel(form)
        rows.addRow("Image size", self.size_label)

    def _build_video_fields(self, form: QWidget) -> None:
        """Adds the rotation video's speed, duration and frames per second."""
        self.speed_field = QDoubleSpinBox(form)
        self.speed_field.setRange(*SPEED_RANGE)
        self.speed_field.setDecimals(1)
        self.speed_field.setSuffix(" deg/s")
        self.speed_field.setValue(ROTATION_SPEED)
        self.rows.addRow("Speed", self.speed_field)
        self.duration_field = QDoubleSpinBox(form)
        self.duration_field.setRange(*DURATION_RANGE)
        self.duration_field.setDecimals(1)
        self.duration_field.setSuffix(" s")
        self.duration_field.setValue(ROTATION_DURATION)
        self.rows.addRow("Duration", self.duration_field)
        self.fps_field = QSpinBox(form)
        self.fps_field.setRange(*FPS_RANGE)
        self.fps_field.setValue(ROTATION_FPS)
        self.rows.addRow("Frames per second", self.fps_field)
        for field in (self.speed_field, self.duration_field, self.fps_field):
            field.valueChanged.connect(lambda _: self._show_size())

    # What is drawn

    def effective(self) -> ExportPreset:
        """Returns the preset with the sheet's camera and transparency."""
        camera: Camera3D | None
        if self.camera == PRESET_CAMERA:
            camera = self.preset.camera
        elif self.camera == USE_CURRENT:
            camera = None
        else:
            forward, up = PRESETS[CameraPreset(self.camera)]
            zoom = self.viewer.scene.view.cameras.three_d.zoom
            camera = Camera3D(forward, up, zoom)
        return replace(self.preset, camera=camera, transparent=self.transparent)

    def shown_views(self) -> tuple[tuple[ViewKind, ...], ...]:
        """Returns the views the export shows, row by row."""
        if self.kind is ExportKind.VIEWS:
            return (VIEWS_2D,)
        if self.kind is ExportKind.ROTATION:
            return ((ViewKind.THREE_D,),)
        return layout(self.viewer, self.view)

    def base(self) -> tuple[tuple[int, int], float]:
        """Returns each view's size in logical pixels, and the pixels per one.

        A preset without a size takes the views' size on screen and the screen's
        pixel ratio times its scale.
        """
        if self.preset.size is not None:
            return self.preset.size, self.preset.scale
        maximised = self.viewer.scene.view.maximised
        kind = self.shown_views()[0][0]
        if not self.grid.panels[kind].isVisible() and maximised is not None:
            kind = maximised  # a hidden view would fill the window if shown
        canvas = self.grid.panels[kind].canvas
        size = (max(1, canvas.width()), max(1, canvas.height()))
        return size, canvas.devicePixelRatioF() * self.preset.scale

    def image_size(self) -> tuple[int, int]:
        """Returns the width and height in pixels of each image or frame Save writes."""
        (width, height), ratio = self.base()
        # As the off-screen canvas rounds its size.
        side = (round(width * ratio + 0.01), round(height * ratio + 0.01))
        if self.kind is not ExportKind.IMAGE:
            return side  # one view per file or frame
        rows = self.shown_views()
        columns = max(len(row) for row in rows)
        gap = round(VIEW_GAP * ratio)
        return (
            columns * side[0] + (columns - 1) * gap,
            len(rows) * side[1] + (len(rows) - 1) * gap,
        )

    # The preview

    def refresh(self) -> None:
        """Draws the preview again, small, with the current choices."""
        self.refresh_timer.stop()
        self._show_size()
        if self.renderer is None or self._drawing:
            return  # while saving, the preview waits: Save draws it again after
        (width, height), _ = self.base()
        rows = self.shown_views()
        columns = max(len(row) for row in rows)
        total = (
            columns * width + (columns - 1) * VIEW_GAP,
            len(rows) * height + (len(rows) - 1) * VIEW_GAP,
        )
        fit = min(PREVIEW_SIZE[0] / total[0], PREVIEW_SIZE[1] / total[1])
        ratio = self.preview.devicePixelRatioF()
        preset = self.effective()
        self._drawing = True
        try:
            self.renderer.resize((width, height), fit * ratio)
            self._set_colours(preset)
            # Only this renderer's camera moves.
            self.renderer.scene_renderer.navigation.place_3d(preset.camera)
            image = self.renderer.render_rows(
                rows, overlay=preset.overlay, transparent=preset.transparent
            )
        finally:
            self._drawing = False
        pixmap = QPixmap.fromImage(_qimage(image))
        pixmap.setDevicePixelRatio(ratio)
        self.preview.setPixmap(pixmap)

    def _set_colours(self, preset: ExportPreset) -> None:
        """Gives the renderer the preset's background, else the views' colours."""
        assert self.renderer is not None
        values = self.settings.values
        self.renderer.set_colours(
            preset.background or values.background, values.crosshair
        )

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        """Starts following the scene, and draws the preview."""
        super().showEvent(event)
        if self.renderer is None:
            size, ratio = self.base()
            # Its slow work runs in the background, as the views' does; Save
            # waits for it.
            self.renderer = OffscreenRenderer(
                self.viewer, size, ratio, loop=loop, changed=self._on_changed
            )
        self.refresh()

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        """Stops following the scene: a hidden sheet draws nothing."""
        super().hideEvent(event)
        self.refresh_timer.stop()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Draws the preview again once the views have been resized."""
        if event.type() == QEvent.Type.Resize:
            self._request_refresh()
        return False

    def _request_refresh(self) -> None:
        """Draws the preview again after a pause, while the sheet is shown."""
        if self.renderer is not None:
            self.refresh_timer.start()

    def _on_changed(self) -> None:
        """Refreshes the preview when the scene changes, unless the sheet is drawing."""
        if not self._drawing:
            self._request_refresh()

    def _on_event(self, event: Event) -> None:
        """Refreshes the preview after a scene event."""
        if not self._drawing:
            self._request_refresh()

    def _on_setting(self, event: SettingChanged) -> None:
        """Draws the preview again with the views' new colours."""
        if event.name in COLOURS:
            self._request_refresh()

    # Saving

    def save(self, path: str | Path) -> list[Path]:
        """Writes what the sheet chose to path, and the session beside it if asked.

        Returns:
            The files written; none if a video was cancelled.
        """
        path = Path(path)
        preset = self.effective()
        written: list[Path] = []
        with self._full_size(preset) as renderer:
            if self.kind is ExportKind.IMAGE:
                written.append(
                    renderer.save(
                        self.view,
                        path,
                        overlay=preset.overlay,
                        transparent=preset.transparent,
                    )
                )
            elif self.kind is ExportKind.VIEWS:
                written += renderer.save_views(
                    path, overlay=preset.overlay, transparent=preset.transparent
                )
            else:
                video = self._record(renderer, preset, path)
                written += [video] if video is not None else []
        if written and self.save_session:
            session = path.with_name(path.name.split(".")[0] + SUFFIX)
            self.viewer.save_session(session)
            written.append(session)
        return written

    @contextmanager
    def _full_size(self, preset: ExportPreset) -> Iterator[OffscreenRenderer]:
        """Yields the renderer at the export's size, inside the preset's options.

        Raises:
            RuntimeError: If the sheet is hidden.
        """
        if self.renderer is None:
            raise RuntimeError("The Export sheet draws only while it is shown.")
        size, ratio = self.base()
        self._drawing = True
        try:
            self.renderer.resize(size, ratio)
            self._set_colours(preset)
            with self.viewer.preset_options(preset):
                yield self.renderer
        finally:
            self._drawing = False
            self.refresh()  # back to the preview's size

    def _record(
        self, renderer: OffscreenRenderer, preset: ExportPreset, path: Path
    ) -> Path | None:
        """Records the rotation video behind a modal progress dialog with Cancel."""
        speed, duration = self.speed_field.value(), self.duration_field.value()
        fps = self.fps_field.value()
        from larmor.viewer.render.offscreen import rotation_frames

        dialog = QProgressDialog(
            "Recording the rotation...",
            "Cancel",
            0,
            rotation_frames(speed, duration, fps),
            self,
        )
        dialog.setWindowTitle("Export")
        # Nothing else can change the scene while the frames are drawn.
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        dialog.setMinimumDuration(0)

        def progress(done: int, total: int) -> bool:
            dialog.setValue(done)  # a modal dialog handles events here
            return not dialog.wasCanceled()

        try:
            return renderer.save_rotation(
                path,
                speed,
                duration,
                fps,
                overlay=preset.overlay,
                transparent=preset.transparent,
                on_progress=progress,
            )
        finally:
            dialog.close()
            dialog.deleteLater()

    def _ask_and_save(self) -> None:
        """Asks for a file, then saves; formats with no alpha only when opaque."""
        video = self.kind is ExportKind.ROTATION
        formats = VIDEO_FORMATS if video else FORMATS
        fps = self.fps_field.value()
        filters = [
            f
            for f, (suffix, alpha) in formats.items()
            if (alpha or not self.transparent)
            and not (video and suffix == ".gif" and fps > GIF_MAX_FPS)
        ]
        if video:
            name = "rotation.webp"
        elif self.kind is ExportKind.VIEWS:
            name = "views.png"
        else:
            name = f"export_{self.view or 'all'}.png"
        views = self.kind is ExportKind.VIEWS
        # The views go to three files named after the one chosen, which is never
        # written: the three, not it, are asked about.
        options = (
            QFileDialog.Option.DontConfirmOverwrite if views else QFileDialog.Option(0)
        )
        path, chosen = QFileDialog.getSaveFileName(
            self, "Export", name, ";;".join(filters), options=options
        )
        if not path:
            return  # cancelled
        file = Path(path)
        if not file.suffix and chosen in formats:
            file = file.with_suffix(formats[chosen][0])
        existing = [view_path(file, k) for k in VIEWS_2D if view_path(file, k).exists()]
        if views and existing and not self._confirm_replace(existing):
            return
        try:
            self.save(file)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Export", f"Cannot save {file}: {error}")

    def _confirm_replace(self, files: list[Path]) -> bool:
        """Returns whether to replace the files, asking once."""
        names = "\n".join(f.name for f in files)
        answer = QMessageBox.question(
            self,
            "Export",
            f"These files already exist:\n{names}\n\nReplace them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    # Presets

    def ask_edit_preset(self) -> PresetDialog:
        """Opens the dialog that edits the chosen preset and saves it as a file."""
        dialog = PresetDialog(self.preset, self.viewer, self)
        dialog.accepted.connect(lambda: self.store_preset(dialog.preset()))
        dialog.open()
        return dialog

    def store_preset(self, preset: ExportPreset) -> None:
        """Saves a preset to its file in the settings folder and chooses it."""
        try:
            save_preset(preset, presets_folder())
        except OSError as error:
            QMessageBox.warning(self, "Export", f"Cannot save the preset: {error}")
            return
        self.presets = load_presets()
        self.preset = self.presets.get(preset.name, preset)
        self.transparent = self.preset.transparent
        self._changed()

    # Showing and choosing

    def _show_choices(self) -> None:
        """Shows the choices in the fields, and only the rows of the chosen kind."""
        self.kind_box.setCurrentIndex(self.kind_box.findData(self.kind))
        self.view_box.setCurrentIndex(self.view_box.findData(self.view))
        self.preset_box.clear()
        self.preset_box.addItems(list(self.presets))
        self.preset_box.setCurrentText(self.preset.name)
        rows = preset_rows(self.preset)
        self.preset_table.setRowCount(len(rows))
        for row, (key, value) in enumerate(rows):
            self.preset_table.setItem(row, 0, QTableWidgetItem(key))
            self.preset_table.setItem(row, 1, QTableWidgetItem(value))
        self.camera_box.clear()
        if self.preset.camera is not None:
            self.camera_box.addItem(PRESET_CAMERA)
        self.camera_box.addItem(USE_CURRENT)
        for preset in CameraPreset:
            self.camera_box.addItem(preset.value.capitalize(), preset.value)
        if self.camera == PRESET_CAMERA and self.preset.camera is None:
            self.camera = USE_CURRENT
        index = self.camera_box.findData(self.camera)
        self.camera_box.setCurrentIndex(
            index if index >= 0 else self.camera_box.findText(self.camera)
        )
        self.transparent_box.setChecked(self.transparent)
        self.session_box.setChecked(self.save_session)
        video = self.kind is ExportKind.ROTATION
        self.rows.setRowVisible(self.view_box, self.kind is ExportKind.IMAGE)
        for field in (self.speed_field, self.duration_field, self.fps_field):
            self.rows.setRowVisible(field, video)
        self._show_size()

    def _show_size(self) -> None:
        """Shows the size of what Save writes."""
        size = "{} x {} pixels".format(*self.image_size())
        if self.kind is ExportKind.VIEWS:
            size = f"3 images of {size}"
        self.size_label.setText(size)

    def _changed(self) -> None:
        """Shows the choices and draws the preview again."""
        self._show_choices()
        self._request_refresh()

    def _choose_kind(self, index: int) -> None:
        """Takes what to save."""
        self.kind = ExportKind(self.kind_box.itemData(index))
        self._changed()

    def _choose_view(self, index: int) -> None:
        """Takes the view chosen in the drop-down."""
        data = self.view_box.itemData(index)  # Qt hands a StrEnum back as text
        self.view = None if data is None else ViewKind(data)
        self._changed()

    def choose_preset(self, name: str) -> None:
        """Chooses a preset by name, with its transparency and camera."""
        self.preset = self.presets[name]
        self.transparent = self.preset.transparent
        self.camera = PRESET_CAMERA if self.preset.camera is not None else USE_CURRENT
        self._changed()

    def _choose_preset(self, index: int) -> None:
        """Takes the preset chosen in the drop-down."""
        self.choose_preset(self.preset_box.itemText(index))

    def _choose_camera(self, index: int) -> None:
        """Takes the camera chosen: the preset's, the current one or a position."""
        data = self.camera_box.itemData(index)
        self.camera = str(data) if data is not None else self.camera_box.itemText(index)
        self._changed()

    def _choose_transparent(self, on: bool) -> None:
        """Takes whether the background is transparent."""
        self.transparent = on
        self._changed()


class PresetDialog(QDialog):
    """Edits a preset's name, size, scale, colours and switches, and its camera."""

    def __init__(
        self, preset: ExportPreset, viewer: Viewer, parent: QWidget | None = None
    ) -> None:
        """Shows the preset's values in the fields."""
        super().__init__(parent)
        self.setWindowTitle("Edit preset")
        self.original = preset
        self.viewer = viewer
        form = QFormLayout(self)
        self.name = QLineEdit(preset.name, self)
        form.addRow("Name", self.name)
        self._build_size(form, preset)
        self.scale = QDoubleSpinBox(self)
        self.scale.setRange(0.25, 8.0)
        self.scale.setSingleStep(0.5)
        self.scale.setValue(preset.scale)
        form.addRow("Scale", self.scale)
        self.background = QLineEdit(preset.background or "", self)
        self.background.setPlaceholderText("#rrggbb, or empty for the views' colour")
        form.addRow("Background", self.background)
        self.overlay = QComboBox(self)
        choices: tuple[tuple[str, bool | None], ...] = (
            ("As the views", None),
            ("On", True),
            ("Off", False),
        )
        for text, value in choices:
            self.overlay.addItem(text, value)
        self.overlay.setCurrentIndex(self.overlay.findData(preset.overlay))
        form.addRow("Information overlay", self.overlay)
        self.transparent = QCheckBox("Transparent background", self)
        self.transparent.setChecked(preset.transparent)
        form.addRow(self.transparent)
        self.keep_camera = QCheckBox("Save the 3D view's camera as it is now", self)
        form.addRow(self.keep_camera)
        self.error = QLabel(self)
        self.error.setObjectName("fieldError")
        self.error.hide()
        form.addRow(self.error)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _build_size(self, form: QFormLayout, preset: ExportPreset) -> None:
        """Adds the size's switch and its width and height."""
        self.screen_size = QCheckBox("The views' size on screen", self)
        self.screen_size.setChecked(preset.size is None)
        form.addRow("Size", self.screen_size)
        self.width_box, self.height_box = QSpinBox(self), QSpinBox(self)
        for box, value in zip(
            (self.width_box, self.height_box), preset.size or (800, 800), strict=True
        ):
            box.setRange(16, 16384)
            box.setValue(value)
        sizes = QHBoxLayout()
        sizes.addWidget(self.width_box)
        sizes.addWidget(QLabel("x", self))
        sizes.addWidget(self.height_box)
        form.addRow("Width and height", sizes)

    def preset(self) -> ExportPreset:
        """Returns the preset the fields describe; its option overrides are kept.

        Raises:
            ValueError: If the name is empty or the background is not a colour.
        """
        name = self.name.text().strip()
        if not name:
            raise ValueError("A preset needs a name.")
        text = self.background.text().strip()
        background = None
        if text:
            try:
                check_colour(text)
            except OptionError as error:
                raise ValueError(str(error)) from error
            background = Colour(text.lower())
        camera = (
            self.viewer.scene.view.cameras.three_d
            if self.keep_camera.isChecked()
            else self.original.camera
        )
        return replace(
            self.original,
            name=name,
            size=None
            if self.screen_size.isChecked()
            else (self.width_box.value(), self.height_box.value()),
            scale=self.scale.value(),
            background=background,
            overlay=self.overlay.currentData(),
            transparent=self.transparent.isChecked(),
            camera=camera,
        )

    def _accept(self) -> None:
        """Closes the dialog if the fields make a preset; else says why not."""
        try:
            self.preset()
        except ValueError as error:
            self.error.setText(str(error))
            self.error.show()
            return
        self.accept()


def _qimage(image: np.ndarray) -> QImage:
    """Converts an RGB or RGBA image to a QImage, over a checkerboard if RGBA."""
    height, width, channels = image.shape
    pixels = np.ascontiguousarray(image)
    kind = (
        QImage.Format.Format_RGBA8888 if channels == 4 else QImage.Format.Format_RGB888
    )
    picture = QImage(pixels.data, width, height, channels * width, kind).copy()
    if channels == 3:
        return picture
    board = QImage(width, height, QImage.Format.Format_RGB32)
    painter = QPainter(board)
    painter.fillRect(board.rect(), QBrush(_checker()))
    painter.drawImage(0, 0, picture)
    painter.end()
    return board


def _checker() -> QPixmap:
    """Returns a tile of the checkerboard, two light and mid grey squares a side."""
    tile = QPixmap(2 * CHECKER, 2 * CHECKER)
    tile.fill(QColor("#cccccc"))
    painter = QPainter(tile)
    painter.fillRect(0, 0, CHECKER, CHECKER, QColor("#999999"))
    painter.fillRect(CHECKER, CHECKER, CHECKER, CHECKER, QColor("#999999"))
    painter.end()
    return tile
