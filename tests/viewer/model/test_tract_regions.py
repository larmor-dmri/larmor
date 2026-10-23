import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from viewer_helpers import tract_options

from larmor.io import load_tractogram, load_volume
from larmor.tractography import Sphere, passes_through
from larmor.types import Tractogram, Volume
from larmor.viewer import Category, Item, OptionError, OptionName, RegionRole, Viewer
from larmor.viewer.model import viewer as viewer_module
from larmor.viewer.model.regions import RegionFilter, passes_regions

EDITING = Path(__file__).parents[2] / "data" / "editing"
TRACKS = EDITING / "tracks.tck"
# On an oblique grid of its own, unlike the tractogram's space.
SHELL = EDITING / "shell_mask.nii.gz"
# Overlaps the shell partly, so that include and exclude both change the selection,
# with and without ends_only.
BIG = (-112.0, -55.0, 74.0)
RADIUS = 9.0


def _regions(item: Item) -> tuple[object, ...]:
    o = tract_options(item)
    return (o.include_masks, o.exclude_masks, o.include_spheres, o.exclude_spheres)


def _filtered(viewer: Viewer) -> tuple[Item, Item]:
    tracts = viewer.load(TRACKS, Category.TRACTOGRAMS)
    shell = viewer.load(SHELL, Category.OVERLAYS)
    viewer.add_sphere(tracts.id, RADIUS, centre=BIG)
    viewer.add_mask(tracts.id, shell.id, role=RegionRole.EXCLUDE)
    return tracts, shell


def _library(exclude_mask: bool, ends_only: bool) -> np.ndarray:
    t = load_tractogram(TRACKS)
    mask = passes_through(t, load_volume(SHELL), ends_only=ends_only)
    sphere = passes_through(t, Sphere(np.array(BIG), RADIUS), ends_only=ends_only)
    kept: np.ndarray = sphere & ~mask if exclude_mask else sphere & mask
    return kept


@pytest.mark.parametrize("ends_only", [False, True], ids=["all-points", "ends-only"])
@pytest.mark.parametrize("exclude_mask", [False, True], ids=["include", "exclude"])
def test_the_selection_is_the_library_filter_of_the_regions(
    exclude_mask: bool, ends_only: bool
) -> None:
    viewer = Viewer()
    tracts = viewer.load(TRACKS, Category.TRACTOGRAMS)
    shell = viewer.load(SHELL, Category.OVERLAYS)
    viewer.add_sphere(tracts.id, RADIUS, centre=BIG)
    viewer.add_mask(tracts.id, shell.id, role=_role(exclude_mask))
    viewer.set(tracts.id, OptionName.ENDS_ONLY, ends_only)
    expected = _library(exclude_mask, ends_only)
    assert 0 < expected.sum() < len(expected)
    selection = viewer.region_selection(tracts.id)
    assert selection is not None
    np.testing.assert_array_equal(selection, expected)
    assert viewer.kept_streamlines(tracts.id) == expected.sum()


def _role(exclude: bool) -> RegionRole:
    return RegionRole.EXCLUDE if exclude else RegionRole.INCLUDE


def test_a_sphere_without_a_centre_goes_at_the_cursor() -> None:
    viewer = Viewer()
    tracts = viewer.load(TRACKS, Category.TRACTOGRAMS)
    viewer.move_cursor(BIG)
    viewer.add_sphere(tracts.id, RADIUS, role=RegionRole.EXCLUDE)
    assert tract_options(tracts).exclude_spheres == ((*BIG, RADIUS),)


def test_turning_the_filter_off_keeps_every_streamline_and_the_regions() -> None:
    viewer = Viewer()
    tracts, _ = _filtered(viewer)
    regions = _regions(tracts)
    viewer.set(tracts.id, OptionName.REGION_FILTER, False)
    assert viewer.region_selection(tracts.id) is None
    assert viewer.kept_streamlines(tracts.id) == 500
    assert _regions(tracts) == regions
    viewer.set(tracts.id, OptionName.REGION_FILTER, True)
    assert viewer.kept_streamlines(tracts.id) == _library(True, False).sum()


def test_clearing_the_regions_keeps_every_streamline() -> None:
    viewer = Viewer()
    tracts, _ = _filtered(viewer)
    viewer.clear_regions(tracts.id)
    assert _regions(tracts) == ((), (), (), ())
    assert viewer.region_selection(tracts.id) is None
    assert viewer.kept_streamlines(tracts.id) == 500


def test_a_removed_mask_leaves_the_regions_and_the_others_stay() -> None:
    viewer = Viewer()
    tracts, shell = _filtered(viewer)
    other = viewer.load(SHELL, Category.VOLUMES)
    viewer.add_mask(tracts.id, other.id)
    viewer.remove(shell.id)
    o = tract_options(tracts)
    assert (o.include_masks, o.exclude_masks) == ((other.id,), ())
    assert o.include_spheres == ((*BIG, RADIUS),)


@pytest.mark.parametrize("radius", [0.0, -1.0])
def test_a_sphere_whose_radius_is_not_positive_is_refused(radius: float) -> None:
    viewer = Viewer()
    tracts = viewer.load(TRACKS, Category.TRACTOGRAMS)
    with pytest.raises(OptionError, match="radius"):
        viewer.add_sphere(tracts.id, radius, centre=BIG)
    assert _regions(tracts) == ((), (), (), ())


@pytest.mark.parametrize("mask", ["4d", "tractogram"])
def test_a_mask_that_is_not_an_image_of_one_volume_is_refused(mask: str) -> None:
    viewer = Viewer()
    tracts = viewer.load(TRACKS, Category.TRACTOGRAMS)
    if mask == "4d":
        data = np.ones((2, 2, 2, 3), np.float32)
        other = viewer.add(Volume(data, np.eye(4)), Category.OVERLAYS)
    else:
        other = viewer.load(TRACKS, Category.TRACTOGRAMS)
    with pytest.raises(OptionError):
        viewer.add_mask(tracts.id, other.id)
    assert _regions(tracts) == ((), (), (), ())


def test_region_actions_refuse_an_item_that_is_not_a_tractogram() -> None:
    viewer = Viewer()
    shell = viewer.load(SHELL, Category.OVERLAYS)
    with pytest.raises(OptionError, match="not a tractogram"):
        viewer.add_sphere(shell.id, RADIUS, centre=BIG)


def test_a_session_keeps_the_regions_and_what_they_select(tmp_path: Path) -> None:
    saved = Viewer()
    # A removed item, so saved ids differ from those a fresh viewer gives.
    saved.remove(saved.add(Volume(np.zeros((2, 2, 2)), np.eye(4)), Category.VOLUMES).id)
    tracts, _ = _filtered(saved)
    saved.set(tracts.id, OptionName.ENDS_ONLY, True)
    assert saved.save_session(tmp_path / "s.lar.json") == []

    viewer = Viewer()
    report = viewer.load_session(tmp_path / "s.lar.json")

    assert report.problems == ()
    new = {i.name: i for i in viewer.scene.items}
    reopened = tract_options(new["tracks.tck"])
    assert reopened.exclude_masks == (new["shell_mask.nii.gz"].id,)
    assert reopened.exclude_masks != tract_options(tracts).exclude_masks
    assert reopened.include_spheres == ((*BIG, RADIUS),)
    assert reopened.ends_only
    kept = viewer.kept_streamlines(new["tracks.tck"].id)
    assert kept == saved.kept_streamlines(tracts.id) == _library(True, True).sum()


def test_the_selection_is_filtered_once_per_change_of_the_regions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    library = passes_regions

    def counted(tractogram: Tractogram, regions: RegionFilter) -> np.ndarray:
        calls.append(regions)
        return library(tractogram, regions)

    monkeypatch.setattr(viewer_module, "passes_regions", counted)
    viewer = Viewer()
    tracts, _ = _filtered(viewer)
    first = viewer.region_selection(tracts.id)
    assert viewer.kept_streamlines(tracts.id) == _library(True, False).sum()
    assert viewer.region_selection(tracts.id) is first
    assert len(calls) == 1
    viewer.set(tracts.id, OptionName.ENDS_ONLY, True)
    assert viewer.kept_streamlines(tracts.id) == _library(True, True).sum()
    assert len(calls) == 2


@pytest.mark.parametrize("first_exclude", [False, True], ids=["include", "exclude"])
def test_a_mask_in_one_list_is_refused_in_the_other(first_exclude: bool) -> None:
    viewer = Viewer()
    tracts = viewer.load(TRACKS, Category.TRACTOGRAMS)
    shell = viewer.load(SHELL, Category.OVERLAYS)
    viewer.add_mask(tracts.id, shell.id, role=_role(first_exclude))
    before = _regions(tracts)
    with pytest.raises(OptionError, match="shell_mask.nii.gz is one of the"):
        viewer.add_mask(tracts.id, shell.id, role=_role(not first_exclude))
    assert _regions(tracts) == before


def test_a_session_reports_a_mask_whose_file_did_not_open(tmp_path: Path) -> None:
    mask = tmp_path / "shell_mask.nii.gz"
    shutil.copy(SHELL, mask)
    saved = Viewer()
    tracts = saved.load(TRACKS, Category.TRACTOGRAMS)
    saved.add_mask(tracts.id, saved.load(mask, Category.OVERLAYS).id)
    saved.save_session(tmp_path / "s.lar.json")
    mask.unlink()

    viewer = Viewer()
    report = viewer.load_session(tmp_path / "s.lar.json")

    (reopened,) = viewer.scene.items
    assert tract_options(reopened).include_masks == ()
    assert any(
        p.startswith("tracks.tck: Include masks: shell_mask.nii.gz is dropped")
        for p in report.problems
    )


def test_a_session_reports_a_mask_entry_that_is_not_an_id(tmp_path: Path) -> None:
    saved = Viewer()
    tracts, _ = _filtered(saved)
    path = tmp_path / "s.lar.json"
    saved.save_session(path)
    data = json.loads(path.read_text())
    (entry,) = [i for i in data["items"] if i["name"] == "tracks.tck"]
    entry["options"]["exclude_masks"] = [[1, 2], {"id": 3}]
    path.write_text(json.dumps(data))

    viewer = Viewer()
    report = viewer.load_session(path)

    reopened = next(i for i in viewer.scene.items if i.name == "tracks.tck")
    assert tract_options(reopened).exclude_masks == ()
    assert "tracks.tck: Exclude masks: item [1, 2] is dropped" in " ".join(
        report.problems
    )


def test_the_selection_work_keeps_nothing_until_it_is_remembered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[RegionFilter] = []

    def counted(tractogram: Tractogram, regions: RegionFilter) -> np.ndarray:
        calls.append(regions)
        return passes_regions(tractogram, regions)

    monkeypatch.setattr(viewer_module, "passes_regions", counted)
    viewer = Viewer()
    tracts, _ = _filtered(viewer)
    key, work = viewer._selection_work(tracts.id)
    selection = work()
    viewer.region_selection(tracts.id)  # the work kept nothing: filters again
    assert len(calls) == 2
    viewer._remember_selection(tracts.id, key, selection)
    assert viewer.region_selection(tracts.id) is selection
    assert len(calls) == 2
