"""The Screenshots panel, which saves views as images or a rotation video."""

from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, QSignalBlocker, Qt, QTimer
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
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from rendercanvas.qt import loop

from larmor.viewer.model.events import Event
from larmor.viewer.model.item import VIEW_GAP, VIEW_KINDS, VIEW_NAMES, ViewKind
from larmor.viewer.model.options import OptionError
from larmor.viewer.model.settings import (
    SettingChanged,
    SettingName,
    Settings,
    setting_fields,
    validate_setting,
)
from larmor.viewer.model.viewer import (
    ROTATION_DURATION,
    ROTATION_FPS,
    ROTATION_SPEED,
    Viewer,
)
from larmor.viewer.qt.field import CheckField, SliderField, field_for
from larmor.viewer.qt.style import form_layout
from larmor.viewer.qt.views import ViewGrid
from larmor.viewer.render.offscreen import (
    GIF_MAX_FPS,
    VIEWS_2D,
    OffscreenRenderer,
    layout,
    rotation_angle,
    rotation_frames,
    view_path,
)


class ScreenshotMode(StrEnum):
    """What the panel saves; each value is its name in the drop-down."""

    IMAGE = "One image"
    VIEWS = "2D views as separate images"
    ROTATION = "3D rotation video"


ALL_VIEWS = "All views"
PREVIEW_SIZE = (240, 180)  # the logical pixels the preview fits in
REFRESH_DELAY = 150  # ms after the last change before the preview is drawn again
CHECKER = 8  # the side of a square of the board behind a transparent preview
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
PREVIEW_FPS = 30  # the most frames per second the playing preview draws
# The settings whose colours the images take.
COLOURS = (SettingName.BACKGROUND, SettingName.CROSSHAIR)


class ScreenshotPanel(QWidget):
    """Saves views to image files, or the 3D view turning to a video."""

    def __init__(
        self,
        viewer: Viewer,
        grid: ViewGrid,
        settings: Settings,
        parent: QWidget | None = None,
    ) -> None:
        """Builds the choices, starting from the settings, and the preview."""
        super().__init__(parent)
        self.viewer = viewer
        self.grid = grid
        self.settings = settings
        # Drawn while the panel is shown; it follows the scene like the views.
        self.renderer: OffscreenRenderer | None = None
        self._drawing = False  # while the panel draws, its own changes are ignored
        values = settings.values
        self.mode = ScreenshotMode.IMAGE
        self.view: ViewKind | None = viewer.scene.view.selected_view
        self.scale = values.screenshot_scale
        self.overlay = values.screenshot_overlay
        self.transparent = values.screenshot_transparent
        self.speed = ROTATION_SPEED
        self.duration = ROTATION_DURATION
        self.fps = ROTATION_FPS
        # The playing preview: whether Play is chosen, and the frame it shows.
        self.playing = True
        self.preview_frame = 0
        self._recording = False

        form = QWidget(self)
        self.form_rows = form_layout(form)
        self._build_image_fields(form)
        self._build_video_fields(form)
        self._build_layout(form)
        self._create_timers()

        self._show_choices()
        self._update_playback()
        # The viewer and the settings outlive the panel: stop listening when it goes.
        unsubscribe_scene = viewer.scene.subscribe(self._on_event)
        unsubscribe_settings = settings.subscribe(self._on_setting)
        self.destroyed.connect(lambda *_: unsubscribe_scene())
        self.destroyed.connect(lambda *_: unsubscribe_settings())
        # The views' size sets the image's: follow their resizes.
        grid.installEventFilter(self)

    def _build_image_fields(self, form: QWidget) -> None:
        """Adds the fields of what to save and of the image to the form."""
        rows = self.form_rows
        self.mode_box = QComboBox(form)
        for mode in ScreenshotMode:
            self.mode_box.addItem(mode.value, mode)
        self.mode_box.activated.connect(self._choose_mode)
        rows.addRow("Save", self.mode_box)
        self.view_box = QComboBox(form)
        for kind in VIEW_KINDS:
            self.view_box.addItem(VIEW_NAMES[kind], kind)
        self.view_box.addItem(ALL_VIEWS, None)
        self.view_box.setCurrentIndex(self.view_box.findData(self.view))
        self.view_box.activated.connect(self._choose_view)
        rows.addRow("View", self.view_box)

        info = setting_fields()
        f, scale_info = info[SettingName.SCREENSHOT_SCALE]
        scale_field = field_for(f, scale_info, form)
        assert isinstance(scale_field, SliderField)
        self.scale_field = scale_field
        self.scale_field.edited.connect(self._choose_scale)
        rows.addRow("Scale (x screen)", self.scale_field)
        self.overlay_field = CheckField(form)
        self.overlay_field.edited.connect(self._choose_overlay)
        rows.addRow("Information overlay", self.overlay_field)
        self.transparent_field = CheckField(form)
        self.transparent_field.edited.connect(self._choose_transparent)
        rows.addRow("Transparent background", self.transparent_field)

    def _build_video_fields(self, form: QWidget) -> None:
        """Adds the rotation video's fields and the image size to the form."""
        rows = self.form_rows
        self.speed_field = QDoubleSpinBox(form)
        self.speed_field.setRange(*SPEED_RANGE)
        self.speed_field.setDecimals(1)
        self.speed_field.setSuffix(" °/s")
        self.speed_field.valueChanged.connect(self._choose_speed)
        rows.addRow("Speed", self.speed_field)
        self.duration_field = QDoubleSpinBox(form)
        self.duration_field.setRange(*DURATION_RANGE)
        self.duration_field.setDecimals(1)
        self.duration_field.setSuffix(" s")
        self.duration_field.valueChanged.connect(self._choose_duration)
        rows.addRow("Duration", self.duration_field)
        self.fps_field = QSpinBox(form)
        self.fps_field.setRange(*FPS_RANGE)
        self.fps_field.valueChanged.connect(self._choose_fps)
        rows.addRow("Frames per second", self.fps_field)
        self.size_label = QLabel(form)
        rows.addRow("Image size", self.size_label)

    def _build_layout(self, form: QWidget) -> None:
        """Lays out the form, the preview, the Play button and the Save button."""
        self.preview = QLabel(self)
        self.preview.setObjectName("screenshotPreview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(*PREVIEW_SIZE)
        self.play_button = QPushButton(self)
        self.play_button.clicked.connect(self._toggle_play)
        play_row = QHBoxLayout()
        play_row.addWidget(self.play_button)
        play_row.addStretch(1)

        self.save_button = QPushButton("Save...", self)
        self.save_button.clicked.connect(self._ask_and_save)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.save_button)

        outer = QVBoxLayout(self)
        outer.addWidget(form)
        outer.addWidget(self.preview, stretch=1)
        outer.addLayout(play_row)
        outer.addLayout(buttons)

    def _create_timers(self) -> None:
        """Creates the timers that refresh the preview and play it."""
        # Changes come in bursts (a drag in a view): the preview waits for a pause.
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.setInterval(REFRESH_DELAY)
        self.refresh_timer.timeout.connect(self.refresh)
        # Each tick draws the next frame of the playing preview.
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._next_frame)

    def shown_views(self) -> tuple[tuple[ViewKind, ...], ...]:
        """Returns the views the preview shows, row by row."""
        if self.mode is ScreenshotMode.VIEWS:
            return (VIEWS_2D,)
        if self.mode is ScreenshotMode.ROTATION:
            return ((ViewKind.THREE_D,),)
        return layout(self.viewer, self.view)

    def base(self) -> tuple[tuple[int, int], float]:
        """Returns each view's size on screen in logical pixels, and the pixel ratio."""
        maximised = self.viewer.scene.view.maximised
        kind = self.shown_views()[0][0]
        if not self.grid.panels[kind].isVisible() and maximised is not None:
            kind = maximised  # a hidden view would fill the window if shown
        canvas = self.grid.panels[kind].canvas
        size = (max(1, canvas.width()), max(1, canvas.height()))
        return size, canvas.devicePixelRatioF()

    def image_size(self) -> tuple[int, int]:
        """Returns the width and height in pixels of each image or frame Save writes."""
        (width, height), ratio = self.base()
        ratio *= self.scale
        # As the off-screen canvas rounds its size.
        side = (round(width * ratio + 0.01), round(height * ratio + 0.01))
        if self.mode is not ScreenshotMode.IMAGE:
            return side  # one view per file or frame
        rows = self.shown_views()
        columns = max(len(row) for row in rows)
        gap = round(VIEW_GAP * ratio)
        return (
            columns * side[0] + (columns - 1) * gap,
            len(rows) * side[1] + (len(rows) - 1) * gap,
        )

    def frames(self) -> int:
        """Returns the number of frames of the rotation video."""
        return rotation_frames(self.speed, self.duration, self.fps)

    def preview_fps(self) -> int:
        """Returns the playing preview's frames per second, the video's up to 30."""
        return min(self.fps, PREVIEW_FPS)

    def preview_frames(self) -> int:
        """Returns the number of frames the playing preview loops through."""
        return rotation_frames(self.speed, self.duration, self.preview_fps())

    def preview_angle(self) -> float:
        """Returns the degrees the preview's scene has turned at its current frame."""
        return rotation_angle(self.speed, self.preview_fps(), self.preview_frame)

    def is_playing(self) -> bool:
        """Returns whether the preview is playing the rotation."""
        return self.play_timer.isActive()

    def existing_view_files(self, path: str | Path) -> tuple[Path, ...]:
        """Returns the files of path's views that exist, which `save_views` replaces."""
        p = Path(path)
        return tuple(
            view_path(p, kind) for kind in VIEWS_2D if view_path(p, kind).exists()
        )

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
        # A miniature: the views' layout at screen size, drawn with fewer pixels.
        fit = min(PREVIEW_SIZE[0] / total[0], PREVIEW_SIZE[1] / total[1])
        ratio = self.preview.devicePixelRatioF()
        self._drawing = True
        try:
            self.renderer.resize((width, height), fit * ratio)
            if self.mode is ScreenshotMode.ROTATION:
                # Only this renderer's camera turns, as when recording.
                start = self.viewer.scene.view.cameras.three_d
                self.renderer.scene_renderer.navigation.place_3d(
                    start.rotated(self.preview_angle(), 0.0)
                )
            image = self.renderer.render_rows(
                rows, overlay=self.overlay, transparent=self.transparent
            )
        finally:
            self._drawing = False
        pixmap = QPixmap.fromImage(_qimage(image))
        pixmap.setDevicePixelRatio(ratio)
        self.preview.setPixmap(pixmap)

    def save(self, path: str | Path) -> Path:
        """Writes the chosen view or views, at the chosen scale of the screen."""
        with self._full_size() as renderer:
            return renderer.save(
                self.view, path, overlay=self.overlay, transparent=self.transparent
            )

    def save_views(self, path: str | Path) -> tuple[Path, ...]:
        """Writes each 2D view to its own file, named after path and the view."""
        with self._full_size() as renderer:
            return renderer.save_views(
                path, overlay=self.overlay, transparent=self.transparent
            )

    def record(self, path: str | Path) -> Path | None:
        """Writes the 3D view turning, showing progress; None if cancelled."""
        self._recording = True
        self._update_playback()  # the preview stops while the video is recorded
        try:
            return self._record(path)
        finally:
            self._recording = False
            self._update_playback()

    def _record(self, path: str | Path) -> Path | None:
        """Records the rotation video behind a modal progress dialog with Cancel."""
        with self._full_size() as renderer:
            dialog = QProgressDialog(
                "Recording the rotation...", "Cancel", 0, self.frames(), self
            )
            dialog.setWindowTitle("Save Screenshot")
            # Nothing else can change the scene while the frames are drawn.
            dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            dialog.setMinimumDuration(0)

            def progress(done: int, total: int) -> bool:
                dialog.setValue(done)  # a modal dialog handles events here
                return not dialog.wasCanceled()

            try:
                return renderer.save_rotation(
                    path,
                    self.speed,
                    self.duration,
                    self.fps,
                    overlay=self.overlay,
                    transparent=self.transparent,
                    on_progress=progress,
                )
            finally:
                dialog.close()
                dialog.deleteLater()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Draws the preview again once the views have been resized."""
        if event.type() == QEvent.Type.Resize:
            self._request_refresh()
        return False

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        """Starts following the scene, and draws the preview."""
        super().showEvent(event)
        if self.renderer is None:
            size, ratio = self.base()
            # Its slow work runs in the background, as the views' does, so the
            # panel does not slow them down; Save waits for it.
            self.renderer = OffscreenRenderer(
                self.viewer, size, ratio, loop=loop, changed=self._on_changed
            )
            values = self.settings.values
            self.renderer.set_colours(values.background, values.crosshair)
        self._update_playback()
        self.refresh()

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        """Stops following the scene: a hidden panel draws nothing."""
        super().hideEvent(event)
        self.refresh_timer.stop()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self._update_playback()

    @contextmanager
    def _full_size(self) -> Iterator[OffscreenRenderer]:
        """Yields the renderer at each view's size on screen times the scale.

        Raises:
            RuntimeError: If the panel is hidden.
        """
        if self.renderer is None:
            raise RuntimeError("The Screenshots panel draws only while it is shown.")
        size, ratio = self.base()
        self._drawing = True
        try:
            self.renderer.resize(size, ratio * self.scale)
            yield self.renderer
        finally:
            self._drawing = False
            self.refresh()  # back to the preview's size

    def _update_playback(self) -> None:
        """Plays the preview if chosen while the rotation is shown; else stops it."""
        self.play_button.setText("Pause" if self.playing else "Play")
        shown = (
            self.mode is ScreenshotMode.ROTATION
            and self.renderer is not None
            and not self._recording
        )
        if shown and self.playing:
            self.play_timer.setInterval(round(1000 / self.preview_fps()))
            if not self.play_timer.isActive():
                self.play_timer.start()
            return
        self.play_timer.stop()
        if not shown:
            self._rewind()  # paused, the preview keeps its frame; stopped, it goes back

    def _rewind(self) -> None:
        """Goes back to the first frame, with the 3D camera of the view state."""
        self.preview_frame = 0
        if self.renderer is not None:
            self.renderer.scene_renderer.navigation.place_3d()

    def _next_frame(self) -> None:
        """Shows the playing preview's next frame."""
        if self.renderer is None or self._drawing:
            return
        self.preview_frame = (self.preview_frame + 1) % self.preview_frames()
        self.refresh()

    def _toggle_play(self) -> None:
        """Plays or pauses the preview."""
        self.playing = not self.playing
        self._update_playback()

    def _keep_angle(self, angle: float) -> None:
        """Moves the preview to the frame nearest angle after a new timing."""
        count = self.preview_frames()
        step = rotation_angle(self.speed, self.preview_fps(), 1)
        frame = round(angle / step) if step else self.preview_frame
        self.preview_frame = frame if 0 <= frame < count else 0
        self._update_playback()
        if self.mode is ScreenshotMode.ROTATION:
            self._request_refresh()

    def _request_refresh(self) -> None:
        """Draws the preview again after a pause, while the panel is shown."""
        if self.renderer is not None:
            self.refresh_timer.start()

    def _on_changed(self) -> None:
        """Refreshes the preview when the scene changes, unless the panel is drawing."""
        if not self._drawing:
            self._request_refresh()

    def _on_event(self, event: Event) -> None:
        """Refreshes the preview after a scene event."""
        self._request_refresh()

    def _on_setting(self, event: SettingChanged) -> None:
        """Takes the new colours when the background or crosshair setting changes."""
        if event.name in COLOURS and self.renderer is not None:
            values = self.settings.values
            self.renderer.set_colours(values.background, values.crosshair)
            self._request_refresh()

    def _show_choices(self) -> None:
        """Shows the choices in the fields, and only the rows of the chosen mode."""
        self.mode_box.setCurrentIndex(self.mode_box.findData(self.mode))
        self.scale_field.show_value(self.scale)
        self.overlay_field.show_value(self.overlay)
        self.transparent_field.show_value(self.transparent)
        # Only the user's edits reach the choices.
        with QSignalBlocker(self.speed_field):
            self.speed_field.setValue(self.speed)
        with QSignalBlocker(self.duration_field):
            self.duration_field.setValue(self.duration)
        with QSignalBlocker(self.fps_field):
            self.fps_field.setValue(self.fps)
        video = self.mode is ScreenshotMode.ROTATION
        self.form_rows.setRowVisible(self.view_box, self.mode is ScreenshotMode.IMAGE)
        for field in (self.speed_field, self.duration_field, self.fps_field):
            self.form_rows.setRowVisible(field, video)
        self.play_button.setVisible(video)
        self._show_size()

    def _show_size(self) -> None:
        """Shows the size of what Save writes."""
        size = "{} x {} pixels".format(*self.image_size())
        if self.mode is ScreenshotMode.VIEWS:
            size = f"3 images of {size}"
        elif self.mode is ScreenshotMode.ROTATION:
            turn = self.speed * self.frames() / self.fps
            size = f"{size}, {self.frames()} frames, {turn:g}\N{DEGREE SIGN}"
        self.size_label.setText(size)

    def _choose_mode(self, index: int) -> None:
        """Takes the mode chosen in the drop-down."""
        mode = ScreenshotMode(self.mode_box.itemData(index))
        if mode is not self.mode:
            self.mode = mode
            self.playing = True  # the rotation plays from the start when chosen
            self._rewind()
        self._update_playback()
        self._changed()

    def _choose_view(self, index: int) -> None:
        """Takes the view chosen in the drop-down."""
        data = self.view_box.itemData(index)  # Qt hands a StrEnum back as text
        self.view = None if data is None else ViewKind(data)
        self._changed()

    def _choose_scale(self, value: object) -> None:
        """Takes the scale typed or dragged; invalid text keeps the last scale."""
        try:
            checked = validate_setting(SettingName.SCREENSHOT_SCALE, value)
        except OptionError:
            checked = self.scale  # text that is not a scale: show the last one
        assert isinstance(checked, int)
        self.scale = checked
        self._changed()

    def _choose_overlay(self, value: object) -> None:
        """Takes whether to draw the information overlay."""
        self.overlay = bool(value)
        self._changed()

    def _choose_transparent(self, value: object) -> None:
        """Takes whether the background is transparent."""
        self.transparent = bool(value)
        self._changed()

    def _choose_speed(self, value: float) -> None:
        """Takes the rotation speed, keeping the preview's angle."""
        angle = self.preview_angle()
        self.speed = value
        self._show_size()
        self._keep_angle(angle)

    def _choose_duration(self, value: float) -> None:
        """Takes the video's duration, keeping the preview's angle."""
        angle = self.preview_angle()
        self.duration = value
        self._show_size()
        self._keep_angle(angle)

    def _choose_fps(self, value: int) -> None:
        """Takes the video's frames per second, keeping the preview's angle."""
        angle = self.preview_angle()
        self.fps = value
        self._show_size()
        self._keep_angle(angle)

    def _changed(self) -> None:
        """Shows the choices and draws the preview again."""
        self._show_choices()
        self._request_refresh()

    def _ask_and_save(self) -> None:
        """Asks for a file, then saves; formats with no alpha only when opaque."""
        video = self.mode is ScreenshotMode.ROTATION
        formats = VIDEO_FORMATS if video else FORMATS
        filters = [
            f
            for f, (suffix, alpha) in formats.items()
            if (alpha or not self.transparent)
            and not (video and suffix == ".gif" and self.fps > GIF_MAX_FPS)
        ]
        if video:
            name = "rotation.webp"
        elif self.mode is ScreenshotMode.VIEWS:
            name = "screenshot.png"
        else:
            name = f"screenshot_{self.view or 'all'}.png"
        views = self.mode is ScreenshotMode.VIEWS
        # The views go to three files named after the one chosen, which is never
        # written: the three, not it, are asked about.
        options = (
            QFileDialog.Option.DontConfirmOverwrite if views else QFileDialog.Option(0)
        )
        path, chosen = QFileDialog.getSaveFileName(
            self, "Save Screenshot", name, ";;".join(filters), options=options
        )
        if not path:
            return  # cancelled
        file = Path(path)
        if not file.suffix and chosen in formats:
            file = file.with_suffix(formats[chosen][0])
        if views and not self._confirm_replace(self.existing_view_files(file)):
            return
        try:
            if video:
                self.record(file)
            elif self.mode is ScreenshotMode.VIEWS:
                self.save_views(file)
            else:
                self.save(file)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Save Screenshot", f"Cannot save {file}: {error}")

    def _confirm_replace(self, files: tuple[Path, ...]) -> bool:
        """Returns whether to replace the files, asking once if there are any."""
        if not files:
            return True
        names = "\n".join(f.name for f in files)
        answer = QMessageBox.question(
            self,
            "Save Screenshot",
            f"These files already exist:\n{names}\n\nReplace them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes


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
