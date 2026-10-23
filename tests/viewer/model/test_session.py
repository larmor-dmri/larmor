import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from larmor.io import save_tractogram, save_volume
from larmor.types import Tractogram, Volume
from larmor.viewer import (
    Category,
    Colormap,
    Convention,
    Item,
    OptionName,
    SessionError,
    Viewer,
    ViewKind,
)


def _volume() -> Volume:
    return Volume(np.arange(8, dtype=np.float32).reshape(2, 2, 2), np.eye(4))


def _image(data: np.ndarray) -> Volume:
    return Volume(data.astype(np.float32), np.eye(4))


def _tractogram() -> Tractogram:
    points = np.arange(12, dtype=np.float32).reshape(-1, 3)
    return Tractogram(points=points, offsets=np.array([0, 2, 4]))


def _files(folder: Path) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = {n: folder / f"{n}.nii.gz" for n in ("t1", "b0", "fa")}
    for path in paths.values():
        save_volume(_volume(), path)
    rng = np.random.default_rng(0)
    paths["fod"] = folder / "fod.nii.gz"
    save_volume(_image(rng.random((4, 4, 4, 15))), paths["fod"])  # lmax 4
    paths["peaks"] = folder / "peaks.nii.gz"
    save_volume(_image(rng.random((4, 4, 4, 6))), paths["peaks"])
    paths["cst"] = folder / "cst.tck"
    save_tractogram(_tractogram(), paths["cst"])
    return paths


def _arranged(folder: Path) -> Viewer:
    files = _files(folder)
    viewer = Viewer()
    # A removed item, so saved ids differ from those a fresh viewer gives.
    viewer.remove(viewer.load(files["b0"], Category.VOLUMES).id)
    t1 = viewer.load(files["t1"], Category.VOLUMES)
    viewer.load(files["b0"], Category.VOLUMES)
    fa = viewer.load(files["fa"], Category.OVERLAYS)
    cst = viewer.load(files["cst"], Category.TRACTOGRAMS)
    fod = viewer.load(files["fod"], Category.FODS)
    peaks = viewer.load(files["peaks"], Category.PEAKS)
    viewer.set(t1.id, OptionName.VISIBLE, True)  # t1 main, not the last volume
    viewer.set(t1.id, OptionName.COLORMAP, Colormap.HOT)
    viewer.set(fa.id, OptionName.OPACITY_2D, 0.4)
    viewer.set(fa.id, OptionName.THRESHOLD, 3.0)
    viewer.set(fa.id, OptionName.MODULATE, t1.id)
    viewer.set(cst.id, OptionName.SUBSET, 40.0)
    viewer.set(fod.id, OptionName.SCALE, 2.0)
    viewer.set(peaks.id, OptionName.LINE_WIDTH, 4.0)
    viewer.move_cursor((1.0, 0.5, 0.25))
    viewer.zoom(ViewKind.CORONAL, 2.0)
    viewer.pan(ViewKind.CORONAL, 0.3, -0.2)
    viewer.rotate_camera(45.0, 0.0)
    viewer.toggle_maximised(ViewKind.SAGITTAL)
    viewer.set_convention(Convention.LAS)
    viewer.select(fa.id)
    return viewer


def _by_name(viewer: Viewer) -> dict[str, Item]:
    return {i.name: i for i in viewer.scene.items}


def test_reopening_a_session_restores_files_options_and_views(tmp_path: Path) -> None:
    saved = _arranged(tmp_path)
    assert saved.save_session(tmp_path / "s.lar.json") == []

    viewer = Viewer()
    report = viewer.load_session(tmp_path / "s.lar.json")

    assert report.problems == ()
    old, new = _by_name(saved), _by_name(viewer)
    assert [i.name for i in viewer.scene.items] == [i.name for i in saved.scene.items]
    for name, item in old.items():
        assert new[name].source == item.source
        expected = item.options
        if name == "fa.nii.gz":
            expected = replace(expected, modulate=new["t1.nii.gz"].id)  # type: ignore[call-arg]
        assert new[name].options == expected, name
    assert viewer.scene.view == replace(
        saved.scene.view,
        main_volume=new["t1.nii.gz"].id,
        selected=new["fa.nii.gz"].id,
    )


def test_asymmetric_peaks_reopen_from_a_session(tmp_path: Path) -> None:
    path = tmp_path / "peaks.nii.gz"
    save_volume(_image(np.ones((2, 2, 2, 3))), path)
    saved = Viewer()
    peaks = saved.load(path, Category.PEAKS)
    saved.set(peaks.id, OptionName.ASYMMETRIC_PEAKS, True)
    assert saved.save_session(tmp_path / "s.lar.json") == []

    viewer = Viewer()
    report = viewer.load_session(tmp_path / "s.lar.json")

    assert report.problems == ()
    (item,) = viewer.scene.items
    assert getattr(item.options, OptionName.ASYMMETRIC_PEAKS) is True


def test_a_moved_project_folder_opens_through_the_relative_paths(
    tmp_path: Path,
) -> None:
    _arranged(tmp_path / "a").save_session(tmp_path / "a" / "s.lar.json")
    shutil.move(tmp_path / "a", tmp_path / "b")

    viewer = Viewer()
    report = viewer.load_session(tmp_path / "b" / "s.lar.json")

    assert report.problems == ()
    assert {i.source for i in viewer.scene.items} == set(
        _files(tmp_path / "b").values()
    )


def test_a_file_loaded_by_a_relative_path_reopens_from_another_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _files(tmp_path / "data")
    monkeypatch.chdir(tmp_path)
    viewer = Viewer()
    viewer.load(Path("data/t1.nii.gz"), Category.VOLUMES)
    monkeypatch.chdir(tmp_path / "data")
    viewer.save_session(tmp_path / "s.lar.json")

    report = Viewer().load_session(tmp_path / "s.lar.json")

    assert report.problems == ()
    assert [i.source for i in report.items] == [tmp_path / "data" / "t1.nii.gz"]


def test_an_item_reference_saved_as_true_is_not_read_as_an_id(
    tmp_path: Path,
) -> None:
    files = _files(tmp_path)
    viewer = Viewer()
    t1 = viewer.load(files["t1"], Category.VOLUMES)
    viewer.load(files["fa"], Category.OVERLAYS)
    viewer.save_session(tmp_path / "s.lar.json")
    data = json.loads((tmp_path / "s.lar.json").read_text())
    assert t1.id == 1  # so that true, equal to 1, would name t1
    data["items"][1]["options"]["modulate"] = True
    (tmp_path / "s.lar.json").write_text(json.dumps(data))

    opened = Viewer()
    opened.load_session(tmp_path / "s.lar.json")

    assert _by_name(opened)["fa.nii.gz"].options.modulate is None  # type: ignore[attr-defined]


def test_a_missing_file_is_reported_and_the_rest_opens(tmp_path: Path) -> None:
    saved = _arranged(tmp_path)
    saved.save_session(tmp_path / "s.lar.json")
    (tmp_path / "cst.tck").unlink()

    report = Viewer().load_session(tmp_path / "s.lar.json")

    [problem] = report.problems
    assert "cst.tck" in problem
    assert [i.name for i in report.items] == [
        "t1.nii.gz",
        "b0.nii.gz",
        "fa.nii.gz",
        "fod.nii.gz",
        "peaks.nii.gz",
    ]


def test_an_item_without_a_file_is_left_out_and_named(tmp_path: Path) -> None:
    viewer = Viewer()
    viewer.add(_volume(), Category.VOLUMES, name="from python")
    viewer.load(_files(tmp_path)["cst"], Category.TRACTOGRAMS)

    assert viewer.save_session(tmp_path / "s.lar.json") == ["from python"]
    report = Viewer().load_session(tmp_path / "s.lar.json")
    assert [i.name for i in report.items] == ["cst.tck"]


def test_a_refused_option_is_reported_and_the_others_applied(tmp_path: Path) -> None:
    _arranged(tmp_path).save_session(tmp_path / "s.lar.json")
    data = json.loads((tmp_path / "s.lar.json").read_text())
    [fa] = [i for i in data["items"] if i["name"] == "fa.nii.gz"]
    fa["options"]["opacity_2d"] = 5.0
    (tmp_path / "s.lar.json").write_text(json.dumps(data))

    viewer = Viewer()
    report = viewer.load_session(tmp_path / "s.lar.json")

    [problem] = report.problems
    assert "fa.nii.gz" in problem
    assert "Opacity (2D)" in problem
    options = _by_name(viewer)["fa.nii.gz"].options
    assert options.opacity_2d == 1.0
    assert options.threshold == 3.0  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "content",
    [
        {"items": []},
        {"format": "larmor-viewer-session", "version": 2, "items": []},
    ],
    ids=["not-a-session", "newer-version"],
)
def test_an_unreadable_session_is_refused_and_the_scene_kept(
    content: dict[str, object], tmp_path: Path
) -> None:
    path = tmp_path / "s.lar.json"
    path.write_text(json.dumps(content))
    viewer = Viewer()
    viewer.add(_volume(), Category.VOLUMES)
    before = viewer.scene.items

    with pytest.raises(SessionError):
        viewer.load_session(path)
    assert viewer.scene.items == before


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_a_volume_with_infinite_or_no_finite_values_saves_and_reopens(
    tmp_path: Path,
) -> None:
    adc = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    adc[0, 0, 0] = np.inf  # a division by zero outside the brain
    save_volume(Volume(adc, np.eye(4)), tmp_path / "adc.nii.gz")
    empty = np.full((2, 2, 2), np.nan, np.float32)
    save_volume(Volume(empty, np.eye(4)), tmp_path / "empty.nii.gz")
    viewer = Viewer()
    viewer.load(tmp_path / "adc.nii.gz", Category.VOLUMES)
    viewer.load(tmp_path / "empty.nii.gz", Category.OVERLAYS)
    viewer.save_session(tmp_path / "s.lar.json")
    opened = Viewer()
    report = opened.load_session(tmp_path / "s.lar.json")
    assert not report.problems
    adc_item, empty_item = opened.scene.items
    assert adc_item.value_range == (1.0, 7.0)  # the finite voxels
    assert empty_item.value_range == (0.0, 1.0)


def test_a_view_state_naming_an_item_not_in_the_scene_is_refused(
    tmp_path: Path,
) -> None:
    viewer = _arranged(tmp_path)
    before = viewer.scene.view
    with pytest.raises(ValueError, match="no item 999"):
        viewer.set_view_state(replace(before, selected=999))
    assert viewer.scene.view == before
