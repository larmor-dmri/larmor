"""The Dissect workspace in the window: the steps, the jobs and the saves."""

import json
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox
from pytestqt.qtbot import QtBot
from viewer_helpers import tract_options

from larmor.io import load_volume, save_tractogram
from larmor.types import Tractogram, Volume
from larmor.viewer import Category, LabelMode, LabelSet, OptionName
from larmor.viewer.model.dissect import (
    DissectStep,
    DwiFiles,
    FodMethod,
    FodSettings,
    StartFrom,
    TrackSettings,
)
from larmor.viewer.model.shortcuts import Command
from larmor.viewer.model.view_state import Workspace
from larmor.viewer.qt.dissect_steps import MaskTarget
from larmor.viewer.qt.main_window import MainWindow

DATA = Path(__file__).parents[1] / "data"
DKI, TRACKING = DATA / "dki", DATA / "tracking"


@pytest.fixture
def window(qtbot: QtBot) -> MainWindow:
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def _messages(window: MainWindow) -> list[str]:
    return [box.text() for box in window.findChildren(QMessageBox)]


def test_the_switch_the_key_and_the_menu_show_dissect(window: MainWindow) -> None:
    window.header.workspaces[Workspace.DISSECT].click()
    assert window.viewer.scene.view.workspace == Workspace.DISSECT
    assert window.left.currentWidget() is window.dissect_panel
    window.shortcuts[Command.WORKSPACE_VIEW].trigger()
    assert window.left.currentWidget() is window.scene_panel
    window.shortcuts[Command.WORKSPACE_DISSECT].trigger()
    assert window.right.currentWidget() is window.dissect_page


def test_a_tractogram_start_goes_to_regions_past_the_skipped_steps(
    window: MainWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    path = tmp_path / "t.tck"
    points = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], np.float32)
    save_tractogram(Tractogram(points, np.array([0, 2, 4])), path)
    window.start_dissection(StartFrom.TRACTOGRAM, path)
    qtbot.waitUntil(lambda: window.dissection.tractogram is not None)
    assert window.dissect_steps.page == DissectStep.REGIONS
    rows = window.dissect_panel.rows
    assert not rows[DissectStep.FODS].isEnabled()
    assert rows[DissectStep.TRACK].status.text().startswith("Skipped")
    page = window.dissect_page
    assert page.back_button.text() == "Back: Start"
    assert "2 of 2 streamlines kept" in page.regions.kept.text()


def test_a_fod_start_loads_the_file_in_its_basis(
    window: MainWindow, qtbot: QtBot
) -> None:
    window.start_dissection(StartFrom.FOD, TRACKING / "fod.nii.gz")
    fods = window.dissect_page.fods
    assert window.dissect_steps.page == DissectStep.FODS
    assert fods.load_choice.isChecked() and not fods.estimate_choice.isEnabled()
    fods.load_button.click()
    qtbot.waitUntil(lambda: window.dissection.fod is not None)
    fod = window.dissection.fod
    assert fod is not None and window.viewer.scene.item(fod).source is not None


def test_estimating_then_tracking_adds_saveable_items_and_keeps_the_run(
    window: MainWindow, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    viewer, state = window.viewer, window.dissection
    window.start_dissection(StartFrom.DWI, DKI / "dwi.nii.gz")
    mask = viewer.load(DKI / "mask.nii.gz", Category.OVERLAYS).id
    dwi = DwiFiles(DKI / "dwi.nii.gz", DKI / "bvals", DKI / "bvecs")
    window.estimate_fods(dwi, mask, FodSettings())
    qtbot.waitUntil(lambda: state.fod is not None, timeout=60_000)
    assert window.dissect_page.fods.save_button.isVisibleTo(window.dissect_page)
    names = [viewer.scene.item(i).name for i in state.made]
    assert names == ["dwi_fod", "dwi_tissue_fraction"]

    assert state.fod is not None
    window.track_streamlines(state.fod, mask, mask, TrackSettings(streamlines=50))
    qtbot.waitUntil(lambda: state.tractogram is not None, timeout=60_000)
    assert state.tractogram is not None
    run = state.runs[state.tractogram].run
    assert {"seed", "step_size", "max_angle", "cutoff"} <= set(run.parameters)

    page = window.dissect_page.save
    page.refresh()
    assert len(page.checks) == 3  # the FOD, the fractions and the tractogram
    window.save_dissection(tmp_path, "b", page.chosen())
    assert (tmp_path / "b.tck").is_file() and (tmp_path / "b_run.json").is_file()
    assert window.dissection.unsaved(viewer.scene) == []
    record = json.loads((tmp_path / "b_run.json").read_text(encoding="utf-8"))
    assert Path(record["inputs"]["fod"]).is_file()

    asked: list[str] = []

    def no(*args: object) -> QMessageBox.StandardButton:
        asked.append(str(args[2]))
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", no)
    before = (tmp_path / "b.tck").stat().st_mtime_ns
    window.save_dissection(tmp_path, "b", [])
    assert asked and "b.tck" in asked[0]
    assert (tmp_path / "b.tck").stat().st_mtime_ns == before


def test_cancel_stops_the_fit_and_a_failed_fit_says_why(
    window: MainWindow, qtbot: QtBot
) -> None:
    viewer = window.viewer
    mask = viewer.load(DKI / "mask.nii.gz", Category.OVERLAYS).id
    dwi = DwiFiles(DKI / "dwi.nii.gz", DKI / "bvals", DKI / "bvecs")
    window.estimate_fods(dwi, mask, FodSettings())
    window.dissect_page.fods.job.cancel.click()
    qtbot.waitUntil(lambda: not window.jobs.busy, timeout=60_000)
    assert viewer.scene.in_category(Category.FODS) == []

    window.estimate_fods(DwiFiles(DKI / "missing.nii.gz"), mask, FodSettings())
    qtbot.waitUntil(
        lambda: any("FOD" in t or "missing" in t for t in _messages(window)),
        timeout=60_000,
    )


def test_the_fods_page_shows_only_the_choices_of_the_method(
    window: MainWindow,
) -> None:
    fods = window.dissect_page.fods
    fods.csf_diffusivity.setText("2e-3")
    fods.response.set_path(Path("wm.txt"))
    for method, csf, response in (
        (FodMethod.CSD, False, True),
        (FodMethod.GRL, True, False),
    ):
        fods.method.show_value(method)
        fods.method.activated.emit(fods.method.currentIndex())
        assert fods.more_form.isRowVisible(fods.csf_diffusivity) == csf
        assert fods.more_form.isRowVisible(fods.response) == response
        settings = fods.settings()
        assert (settings.csf_diffusivity == 2e-3) == csf
        assert (settings.response is not None) == response


def test_saving_a_session_offers_to_save_the_unsaved_items_first(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    viewer = window.viewer
    fod = viewer.add(load_volume(TRACKING / "fod.nii.gz"), Category.FODS, name="fod")
    monkeypatch.setattr(
        QMessageBox, "exec", lambda self: QMessageBox.StandardButton.Save
    )
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(tmp_path / "s.lar.json"), ""),
    )
    window._ask_save_session()
    assert viewer.scene.item(fod.id).source == tmp_path / "s_fod.nii.gz"
    session = json.loads((tmp_path / "s.lar.json").read_text(encoding="utf-8"))
    assert [item["name"] for item in session["items"]] == ["fod"]


def _start_from_a_tractogram(window: MainWindow, qtbot: QtBot, tmp_path: Path) -> None:
    path = tmp_path / "t.tck"
    points = np.array([[0.0, 0, 0], [1, 0, 0], [2, 2, 2], [2, 2, 1]], np.float32)
    save_tractogram(Tractogram(points, np.array([0, 2, 4])), path)
    window.start_dissection(StartFrom.TRACTOGRAM, path)
    qtbot.waitUntil(lambda: window.dissection.tractogram is not None)


def test_a_labels_overlay_shows_its_parcellation_section(
    window: MainWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    _start_from_a_tractogram(window, qtbot, tmp_path)
    regions = window.dissect_page.regions
    assert regions.parcellation is None
    data = np.zeros((3, 3, 3), np.int16)
    data[2, 2, 2] = 7
    viewer = window.viewer
    item = viewer.add(Volume(data, np.eye(4)), Category.OVERLAYS, name="parc")
    lut = tmp_path / "lut.txt"
    lut.write_text("7 Brain-Stem 7 8 9 0\n")
    viewer.set(item.id, OptionName.LOOKUP_TABLE, str(lut))
    viewer.set(item.id, OptionName.LABELS, True)
    window.dissect_page.refresh_pages()
    section = regions.parcellation
    assert section is not None and section.parcellation_id == item.id
    section.add_label(LabelSet.A, 7)
    section.set_mode(LabelMode.STAY_WITHIN)
    assert "0 of 2 streamlines kept" in regions.kept.text()


@pytest.mark.parametrize("target", list(MaskTarget))
def test_a_new_mask_fills_the_field_that_asked_for_it(
    window: MainWindow,
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: MaskTarget,
) -> None:
    _start_from_a_tractogram(window, qtbot, tmp_path)
    viewer = window.viewer
    mask = viewer.add(
        Volume(np.ones((3, 3, 3), np.uint8), np.eye(4)), Category.OVERLAYS, name="m"
    )
    monkeypatch.setattr(
        "larmor.viewer.qt.mask_sheet.open_mask_sheet", lambda *_: mask.id
    )
    window.request_new_mask(target)
    page = window.dissect_page
    assert mask.id in window.dissection.made
    if target == MaskTarget.FIT:
        assert page.fods.mask_box.value() == mask.id
    elif target == MaskTarget.TRACK:
        assert page.track.mask_box.value() == mask.id
    else:
        tract = window.dissection.tractogram
        assert tract is not None
        (region,) = viewer.scene.in_category(Category.REGIONS)
        assert region.id in tract_options(viewer.scene.item(tract)).regions


@pytest.mark.parametrize("replace", [True, False])
def test_saving_items_beside_a_session_asks_before_replacing_files(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replace: bool
) -> None:
    viewer = window.viewer
    fod = viewer.add(load_volume(TRACKING / "fod.nii.gz"), Category.FODS, name="fod")
    existing = tmp_path / "s_fod.nii.gz"
    existing.write_bytes(b"older")
    asked: list[str] = []

    def question(*args: object, **kwargs: object) -> QMessageBox.StandardButton:
        asked.append(str(args[2]))
        return (
            QMessageBox.StandardButton.Yes if replace else QMessageBox.StandardButton.No
        )

    monkeypatch.setattr(QMessageBox, "question", question)
    saved = window.save_items_beside([fod.id], tmp_path / "s.lar.json")
    assert saved == replace
    assert len(asked) == 1 and "s_fod.nii.gz" in asked[0]
    assert (existing.read_bytes() != b"older") == replace


def test_items_of_the_same_name_are_saved_beside_a_session_to_their_own_files(
    window: MainWindow, tmp_path: Path
) -> None:
    viewer = window.viewer
    data = load_volume(TRACKING / "fod.nii.gz")
    first = viewer.add(data, Category.FODS, name="fod")
    second = viewer.add(data, Category.FODS, name="fod")
    assert window.save_items_beside([first.id, second.id], tmp_path / "s.lar.json")
    sources = {viewer.scene.item(i).source for i in (first.id, second.id)}
    assert sources == {tmp_path / "s_fod.nii.gz", tmp_path / "s_fod_2.nii.gz"}
