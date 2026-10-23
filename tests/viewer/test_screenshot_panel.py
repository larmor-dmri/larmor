from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageSequence
from PySide6.QtWidgets import QFileDialog, QMenu, QMessageBox
from pytestqt.qtbot import QtBot

from larmor.types import Volume
from larmor.viewer import Category, Viewer, ViewKind
from larmor.viewer.model.settings import AppSettings, Settings
from larmor.viewer.qt.docks import Panel
from larmor.viewer.qt.main_window import MainWindow
from larmor.viewer.qt.screenshot_panel import ScreenshotMode, ScreenshotPanel
from larmor.viewer.render.offscreen import OffscreenRenderer
from larmor.viewer.render.scene_renderer import SceneRenderer


@pytest.fixture
def window(qtbot: QtBot) -> MainWindow:
    viewer = Viewer()
    viewer.add(Volume(np.ones((8, 8, 8), np.float32), np.eye(4)), Category.VOLUMES)
    viewer.select_view(ViewKind.CORONAL)
    settings = Settings(
        AppSettings(
            screenshot_scale=3, screenshot_overlay=False, screenshot_transparent=True
        )
    )
    window = MainWindow(viewer, settings=settings)
    qtbot.addWidget(window)
    window.resize(800, 600)
    window.show()
    qtbot.waitExposed(window)
    return window


@pytest.fixture
def dialog(window: MainWindow, qtbot: QtBot) -> ScreenshotPanel:
    dialog = window.show_screenshot()
    qtbot.waitUntil(dialog.isVisible)
    return dialog


def _file_menu(window: MainWindow) -> QMenu:
    file = next(a for a in window.menuBar().actions() if a.text() == "&File").menu()
    assert isinstance(file, QMenu)
    return file


def test_the_window_starts_from_the_settings_and_the_selected_view(
    dialog: ScreenshotPanel,
) -> None:
    assert dialog.view_box.currentData() == ViewKind.CORONAL
    assert (dialog.scale, dialog.overlay, dialog.transparent) == (3, False, True)
    assert dialog.scale_field.text.text() == "3"
    assert not dialog.overlay_field.isChecked()
    assert dialog.transparent_field.isChecked()


def test_a_new_scale_changes_the_image_size(dialog: ScreenshotPanel) -> None:
    (width, height), ratio = dialog.base()
    assert dialog.image_size() == (round(width * ratio * 3), round(height * ratio * 3))
    dialog.scale_field.slider.setValue(0)  # a scale of 1
    assert dialog.scale == 1
    assert dialog.image_size() == (round(width * ratio), round(height * ratio))


@pytest.mark.parametrize("view", [ViewKind.CORONAL, None], ids=["one view", "all"])
def test_save_writes_an_image_of_the_size_shown(
    view: ViewKind | None, dialog: ScreenshotPanel, tmp_path: Path
) -> None:
    dialog.view_box.setCurrentIndex(dialog.view_box.findData(view))
    dialog.view_box.activated.emit(dialog.view_box.currentIndex())
    dialog.scale_field.slider.setValue(1)  # a scale of 2
    path = dialog.save(tmp_path / "a.png")
    with Image.open(path) as image:
        assert image.size == dialog.image_size()
        assert image.mode == "RGBA"


def test_file_screenshots_toggles_one_panel(window: MainWindow, qtbot: QtBot) -> None:
    file = _file_menu(window)
    action = next(a for a in file.actions() if a.text() == "Screenshots")
    action.trigger()
    panel = window.docks.widget(Panel.SCREENSHOTS)
    qtbot.waitUntil(panel.isVisible)
    action.trigger()
    assert not panel.isVisible()
    action.trigger()
    assert window.findChildren(ScreenshotPanel) == [panel]
    qtbot.waitUntil(panel.isVisible)


def test_at_scale_1_one_view_has_the_pixels_of_its_canvas(
    window: MainWindow, dialog: ScreenshotPanel, tmp_path: Path
) -> None:
    dialog.view_box.setCurrentIndex(dialog.view_box.findData(ViewKind.AXIAL))
    dialog.view_box.activated.emit(dialog.view_box.currentIndex())
    dialog.scale_field.slider.setValue(0)  # a scale of 1
    path = dialog.save(tmp_path / "a.png")
    canvas = window.views.panels[ViewKind.AXIAL].canvas
    with Image.open(path) as image:
        assert image.size == canvas.get_physical_size()


def _choose_mode(dialog: ScreenshotPanel, mode: ScreenshotMode) -> None:
    dialog.mode_box.setCurrentIndex(dialog.mode_box.findData(mode))
    dialog.mode_box.activated.emit(dialog.mode_box.currentIndex())


@pytest.mark.parametrize("mode", list(ScreenshotMode))
def test_each_mode_shows_only_its_fields(
    mode: ScreenshotMode, dialog: ScreenshotPanel
) -> None:
    _choose_mode(dialog, mode)
    rows = dialog.form_rows
    assert rows.isRowVisible(dialog.view_box) == (mode is ScreenshotMode.IMAGE)
    for field in (dialog.speed_field, dialog.duration_field, dialog.fps_field):
        assert rows.isRowVisible(field) == (mode is ScreenshotMode.ROTATION)


def test_save_views_writes_three_images_of_the_size_shown(
    dialog: ScreenshotPanel, tmp_path: Path
) -> None:
    _choose_mode(dialog, ScreenshotMode.VIEWS)
    dialog.scale_field.slider.setValue(0)  # a scale of 1
    paths = dialog.save_views(tmp_path / "a.png")
    assert len(paths) == 3
    for path in paths:
        with Image.open(path) as image:
            assert image.size == dialog.image_size()


def test_record_writes_a_video_of_the_frames_shown(
    dialog: ScreenshotPanel, tmp_path: Path
) -> None:
    _choose_mode(dialog, ScreenshotMode.ROTATION)
    dialog.scale_field.slider.setValue(0)  # a scale of 1
    dialog.duration_field.setValue(1.0)
    dialog.fps_field.setValue(2)
    assert dialog.frames() == 2
    path = dialog.record(tmp_path / "a.webp")
    assert path == tmp_path / "a.webp"
    with Image.open(path) as image:
        assert sum(1 for _ in ImageSequence.Iterator(image)) == 2
        assert image.size == dialog.image_size()


@pytest.mark.parametrize("replace", [False, True], ids=["no", "yes"])
def test_saving_views_asks_once_about_the_existing_files(
    replace: bool,
    dialog: ScreenshotPanel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _choose_mode(dialog, ScreenshotMode.VIEWS)
    dialog.scale_field.slider.setValue(0)  # a scale of 1
    chosen = tmp_path / "a.png"
    coronal = tmp_path / "a_coronal.png"
    coronal.write_bytes(b"old")
    (tmp_path / "a_other.png").write_bytes(b"other")
    assert dialog.existing_view_files(chosen) == (coronal,)
    options: list[object] = []

    def get_save_file_name(*args: object, **kwargs: object) -> tuple[str, str]:
        options.append(kwargs.get("options"))
        return str(chosen), ""

    questions: list[object] = []

    def question(*args: object) -> QMessageBox.StandardButton:
        questions.append(args[2])
        yes, no = QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No
        return yes if replace else no

    monkeypatch.setattr(QFileDialog, "getSaveFileName", get_save_file_name)
    monkeypatch.setattr(QMessageBox, "question", question)
    dialog.save_button.click()
    assert options == [QFileDialog.Option.DontConfirmOverwrite]
    assert len(questions) == 1
    assert "a_coronal.png" in str(questions[0])
    written = sorted(p.name for p in tmp_path.glob("a_*.png"))
    if replace:
        assert written == [
            f"a_{v}.png" for v in ("axial", "coronal", "other", "sagittal")
        ]
        assert coronal.read_bytes() != b"old"
    else:
        assert written == ["a_coronal.png", "a_other.png"]
        assert coronal.read_bytes() == b"old"
    assert not chosen.exists()


def _position(renderer: SceneRenderer) -> np.ndarray:
    view = renderer.navigation.view_3d
    assert view is not None
    return np.array(view.camera.world.position)


def test_the_rotation_preview_turns_only_its_own_camera_and_loops(
    window: MainWindow, dialog: ScreenshotPanel
) -> None:
    _choose_mode(dialog, ScreenshotMode.ROTATION)
    dialog.duration_field.setValue(1.0)
    dialog.fps_field.setValue(2)
    assert dialog.is_playing()
    assert dialog.preview_frames() == 2
    assert dialog.renderer is not None
    own = dialog.renderer.scene_renderer
    cameras = window.viewer.scene.view.cameras
    start, main = _position(own), _position(window.renderer)
    dialog._next_frame()
    assert dialog.preview_frame == 1
    assert not np.allclose(_position(own), start)
    assert window.viewer.scene.view.cameras == cameras
    np.testing.assert_array_equal(_position(window.renderer), main)
    dialog._next_frame()
    assert dialog.preview_frame == 0
    np.testing.assert_allclose(_position(own), start, atol=1e-9)


def test_pause_keeps_the_frame_and_leaving_rewinds(
    qtbot: QtBot, dialog: ScreenshotPanel
) -> None:
    assert dialog.renderer is not None
    own = dialog.renderer.scene_renderer
    start = _position(own)
    _choose_mode(dialog, ScreenshotMode.ROTATION)
    dialog._next_frame()
    dialog.play_button.click()
    assert not dialog.is_playing()
    assert dialog.preview_frame == 1
    assert not np.allclose(_position(own), start)
    dialog.play_button.click()
    assert dialog.is_playing()
    dialog.play_button.click()  # paused on a turned frame
    _choose_mode(dialog, ScreenshotMode.IMAGE)
    assert not dialog.is_playing()
    assert dialog.preview_frame == 0
    np.testing.assert_allclose(_position(own), start, atol=1e-9)
    _choose_mode(dialog, ScreenshotMode.ROTATION)
    dialog._next_frame()
    dialog.hide()
    assert not dialog.is_playing()
    assert dialog.preview_frame == 0
    dialog.show()
    qtbot.waitExposed(dialog)
    assert dialog.is_playing()


def test_recording_stops_the_preview_until_it_is_done(
    dialog: ScreenshotPanel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _choose_mode(dialog, ScreenshotMode.ROTATION)
    playing: list[bool] = []

    def save_rotation(
        self: OffscreenRenderer, path: Path, *args: object, **kwargs: object
    ) -> Path:
        playing.append(dialog.is_playing())
        return path

    monkeypatch.setattr(OffscreenRenderer, "save_rotation", save_rotation)
    dialog.record(tmp_path / "a.webp")
    assert playing == [False]
    assert dialog.is_playing()
