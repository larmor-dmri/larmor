"""The Export sheet: what it saves, the presets, and the session beside the image."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageSequence
from PySide6.QtWidgets import QFileDialog, QMenu, QMessageBox
from pytestqt.qtbot import QtBot

from larmor.types import Tractogram, Volume
from larmor.viewer import Category, OptionName, TractStyle, Viewer, ViewKind
from larmor.viewer.model.export import ExportKind, ExportPreset, load_presets
from larmor.viewer.model.options import Colour
from larmor.viewer.qt.export_sheet import ExportSheet, PresetDialog
from larmor.viewer.qt.main_window import MainWindow


@pytest.fixture(autouse=True)
def _own_presets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The presets edited here go to a folder of this test's own.
    from larmor.viewer.model import settings

    monkeypatch.setattr(settings, "settings_path", lambda: tmp_path / "viewer.toml")


@pytest.fixture
def window(qtbot: QtBot) -> MainWindow:
    viewer = Viewer()
    data = np.arange(512, dtype=np.float32).reshape(8, 8, 8)
    viewer.add(Volume(data, np.eye(4)), Category.VOLUMES)
    viewer.select_view(ViewKind.CORONAL)
    window = MainWindow(viewer)
    qtbot.addWidget(window)
    window.resize(800, 600)
    window.show()
    qtbot.waitExposed(window)
    return window


@pytest.fixture
def sheet(window: MainWindow, qtbot: QtBot) -> ExportSheet:
    sheet = window.show_export()
    qtbot.waitUntil(sheet.isVisible)
    # The sheet's renderer uploads the scene through the event loop.
    qtbot.wait(50)
    return sheet


def _choose(sheet: ExportSheet, kind: ExportKind) -> None:
    sheet.kind_box.setCurrentIndex(sheet.kind_box.findData(kind))
    sheet.kind_box.activated.emit(sheet.kind_box.currentIndex())


def test_file_export_opens_one_sheet_on_the_selected_view(
    window: MainWindow, qtbot: QtBot
) -> None:
    file = next(a for a in window.menuBar().actions() if a.text() == "&File").menu()
    assert isinstance(file, QMenu)
    action = next(a for a in file.actions() if a.text() == "&Export...")
    action.trigger()
    [sheet] = window.findChildren(ExportSheet)
    qtbot.waitUntil(sheet.isVisible)
    assert sheet.view_box.currentData() == ViewKind.CORONAL
    assert sheet.preset.name == "Screen"
    sheet.close()
    window.header.export_button.click()
    assert window.findChildren(ExportSheet) == [sheet] and sheet.isVisible()


@pytest.mark.parametrize("view", [ViewKind.CORONAL, None], ids=["one view", "all"])
def test_save_writes_an_image_of_the_size_shown(
    view: ViewKind | None, sheet: ExportSheet, tmp_path: Path
) -> None:
    sheet.view_box.setCurrentIndex(sheet.view_box.findData(view))
    sheet.view_box.activated.emit(sheet.view_box.currentIndex())
    [path] = sheet.save(tmp_path / "a.png")
    with Image.open(path) as image:
        assert image.size == sheet.image_size()
        assert image.mode == "RGB"


@pytest.mark.parametrize("view", [ViewKind.AXIAL, None], ids=["one view", "all"])
def test_a_preset_with_a_size_draws_that_size_times_its_scale(
    view: ViewKind | None, sheet: ExportSheet, tmp_path: Path
) -> None:
    sheet.choose_preset("Website")
    sheet.view_box.setCurrentIndex(sheet.view_box.findData(view))
    sheet.view_box.activated.emit(sheet.view_box.currentIndex())
    assert sheet.image_size() == (1800, 1200)
    assert sheet.camera_box.currentText() == "The preset's"
    [path] = sheet.save(tmp_path / "w.png")
    with Image.open(path) as image:
        assert image.size == (1800, 1200)


def test_the_figure_preset_is_transparent_until_the_box_says_otherwise(
    sheet: ExportSheet, tmp_path: Path
) -> None:
    sheet.choose_preset("Figure")
    assert sheet.transparent_box.isChecked()
    [path] = sheet.save(tmp_path / "f.png")
    with Image.open(path) as image:
        assert image.mode == "RGBA"
    sheet.transparent_box.click()
    [path] = sheet.save(tmp_path / "g.png")
    with Image.open(path) as image:
        assert image.mode == "RGB"


def test_a_presets_options_apply_only_while_it_draws(
    window: MainWindow, sheet: ExportSheet, tmp_path: Path
) -> None:
    viewer = window.viewer
    points = np.array([[-3.0, 0, 0], [3.0, 0, 0]])
    tracts = viewer.add(Tractogram(points, np.array([0, 2])), Category.TRACTOGRAMS)
    seen: list[object] = []
    viewer.scene.subscribe(
        lambda event: (
            seen.append(getattr(event, "new", None))
            if getattr(event, "name", None) == OptionName.STYLE
            else None
        )
    )
    sheet.choose_preset("Talk")
    sheet.save(tmp_path / "t.png")
    assert seen == [TractStyle.TUBES, TractStyle.LINES]
    assert getattr(viewer.scene.item(tracts.id).options, OptionName.STYLE) == (
        TractStyle.LINES
    )


def test_the_session_can_be_saved_beside_the_image(
    window: MainWindow, sheet: ExportSheet, tmp_path: Path
) -> None:
    sheet.session_box.click()
    written = sheet.save(tmp_path / "fig.png")
    assert [p.name for p in written] == ["fig.png", "fig.lar.json"]


@pytest.mark.parametrize("kind", list(ExportKind))
def test_each_kind_shows_only_its_fields(kind: ExportKind, sheet: ExportSheet) -> None:
    _choose(sheet, kind)
    rows = sheet.rows
    assert rows.isRowVisible(sheet.view_box) == (kind is ExportKind.IMAGE)
    for field in (sheet.speed_field, sheet.duration_field, sheet.fps_field):
        assert rows.isRowVisible(field) == (kind is ExportKind.ROTATION)


def test_save_views_writes_three_images(sheet: ExportSheet, tmp_path: Path) -> None:
    _choose(sheet, ExportKind.VIEWS)
    paths = sheet.save(tmp_path / "a.png")
    assert [p.name for p in paths] == ["a_axial.png", "a_coronal.png", "a_sagittal.png"]
    for path in paths:
        with Image.open(path) as image:
            assert image.size == sheet.image_size()


def test_a_rotation_video_has_the_frames_asked_for(
    sheet: ExportSheet, tmp_path: Path
) -> None:
    _choose(sheet, ExportKind.ROTATION)
    sheet.duration_field.setValue(1.0)
    sheet.fps_field.setValue(2)
    [path] = sheet.save(tmp_path / "a.webp")
    with Image.open(path) as image:
        assert sum(1 for _ in ImageSequence.Iterator(image)) == 2
        assert image.size == sheet.image_size()


@pytest.mark.parametrize("replace", [False, True], ids=["no", "yes"])
def test_saving_views_asks_once_about_the_existing_files(
    replace: bool, sheet: ExportSheet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _choose(sheet, ExportKind.VIEWS)
    chosen = tmp_path / "a.png"
    coronal = tmp_path / "a_coronal.png"
    coronal.write_bytes(b"old")
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(chosen), "")
    )
    questions: list[object] = []

    def question(*args: object) -> QMessageBox.StandardButton:
        questions.append(args[2])
        yes, no = QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No
        return yes if replace else no

    monkeypatch.setattr(QMessageBox, "question", question)
    sheet.save_button.click()
    assert len(questions) == 1 and "a_coronal.png" in str(questions[0])
    assert (coronal.read_bytes() != b"old") == replace
    assert not chosen.exists()


def test_an_edited_preset_is_saved_as_a_file_and_chosen(
    sheet: ExportSheet, qtbot: QtBot
) -> None:
    dialog = sheet.ask_edit_preset()
    assert isinstance(dialog, PresetDialog)
    dialog.name.setText("Poster")
    dialog.screen_size.setChecked(False)
    dialog.width_box.setValue(640)
    dialog.height_box.setValue(480)
    dialog.background.setText("#102030")
    dialog._accept()
    assert sheet.preset.name == "Poster"
    saved = load_presets()["Poster"]
    assert saved == ExportPreset(
        "Poster", size=(640, 480), background=Colour("#102030")
    )
    assert sheet.preset_box.currentText() == "Poster"


def test_a_preset_without_a_name_or_with_a_bad_colour_is_refused(
    sheet: ExportSheet,
) -> None:
    dialog = sheet.ask_edit_preset()
    dialog.background.setText("red")
    dialog._accept()
    assert not dialog.error.isHidden() and "#rrggbb" in dialog.error.text()
    assert dialog.isVisible()
