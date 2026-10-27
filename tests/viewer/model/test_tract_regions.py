import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from viewer_helpers import tract_options

from larmor.io import load_tractogram, load_volume
from larmor.tractography import Box, Sphere, passes_through, points_inside
from larmor.types import Tractogram, Volume
from larmor.viewer import Category, Item, OptionError, OptionName, RegionRole, Viewer
from larmor.viewer.model import viewer as viewer_module
from larmor.viewer.model.events import FilterChanged
from larmor.viewer.model.options import ROLE_COLOURS, Colour, RegionOptions
from larmor.viewer.model.regions import RegionFilter, passes_regions

EDITING = Path(__file__).parents[2] / "data" / "editing"
TRACKS = EDITING / "tracks.tck"
# On an oblique grid of its own, unlike the tractogram's space.
SHELL = EDITING / "shell_mask.nii.gz"
# Overlaps the shell partly, so that include and exclude both change the selection,
# with and without ends_only.
BIG = (-112.0, -55.0, 74.0)
RADIUS = 9.0


def _region(viewer: Viewer, region_id: int) -> RegionOptions:
    o = viewer.scene.item(region_id).options
    assert isinstance(o, RegionOptions)
    return o


def _filtered(viewer: Viewer) -> tuple[Item, Item, int, int]:
    tracts = viewer.load(TRACKS, Category.TRACTOGRAMS)
    shell = viewer.load(SHELL, Category.OVERLAYS)
    sphere = viewer.add_sphere(BIG, RADIUS, apply=True)
    mask = viewer.add_mask(shell.id, RegionRole.EXCLUDE, apply=True)
    return tracts, shell, sphere, mask


def _library(exclude_mask: bool, ends_only: bool) -> np.ndarray:
    t = load_tractogram(TRACKS)
    mask = passes_through(t, load_volume(SHELL), ends_only=ends_only)
    sphere = passes_through(t, Sphere(np.array(BIG), RADIUS), ends_only=ends_only)
    kept: np.ndarray = sphere & ~mask if exclude_mask else sphere & mask
    return kept


def _role(exclude: bool) -> RegionRole:
    return RegionRole.EXCLUDE if exclude else RegionRole.INCLUDE


@pytest.mark.parametrize("ends_only", [False, True], ids=["all-points", "ends-only"])
@pytest.mark.parametrize("exclude_mask", [False, True], ids=["include", "exclude"])
def test_the_selection_is_the_library_filter_of_the_regions(
    exclude_mask: bool, ends_only: bool
) -> None:
    viewer = Viewer()
    tracts = viewer.load(TRACKS, Category.TRACTOGRAMS)
    shell = viewer.load(SHELL, Category.OVERLAYS)
    viewer.apply_region(tracts.id, viewer.add_sphere(BIG, RADIUS))
    viewer.apply_region(tracts.id, viewer.add_mask(shell.id, _role(exclude_mask)))
    viewer.set(tracts.id, OptionName.ENDS_ONLY, ends_only)
    expected = _library(exclude_mask, ends_only)
    assert 0 < expected.sum() < len(expected)
    selection = viewer.region_selection(tracts.id)
    assert selection is not None
    np.testing.assert_array_equal(selection, expected)
    assert viewer.kept_streamlines(tracts.id) == expected.sum()


def test_a_box_region_keeps_what_the_library_box_keeps() -> None:
    viewer = Viewer()
    tracts = viewer.load(TRACKS, Category.TRACTOGRAMS)
    size = (20.0, 10.0, 15.0)
    viewer.add_box(BIG, size, apply=True)
    expected = passes_through(load_tractogram(TRACKS), Box(np.array(BIG), size))
    assert 0 < expected.sum() < len(expected)
    np.testing.assert_array_equal(viewer.region_selection(tracts.id), expected)


def test_a_point_is_in_a_box_up_to_half_its_size_along_each_axis() -> None:
    box = Box(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
    points = np.array(
        [[2.0, 4.0, 6.0], [0.0, 0.0, 0.0], [2.01, 2.0, 3.0], [1.0, -0.01, 3.0]]
    )
    np.testing.assert_array_equal(
        points_inside(points, box), [True, True, False, False]
    )


def test_a_sphere_without_a_centre_or_radius_goes_at_the_cursor_with_the_last_one() -> (
    None
):
    viewer = Viewer()
    viewer.add_sphere((0.0, 0.0, 0.0), 7.5)
    viewer.move_cursor(BIG)
    region = viewer.add_sphere(role=RegionRole.EXCLUDE)
    o = _region(viewer, region)
    assert (o.centre, o.radius, o.role) == (BIG, 7.5, RegionRole.EXCLUDE)
    assert viewer.scene.item(region).name == "Sphere 2"
    assert isinstance(viewer.scene.item(region).data, Sphere)


def test_turning_the_filter_off_keeps_every_streamline_and_the_regions() -> None:
    viewer = Viewer()
    tracts, _, sphere, mask = _filtered(viewer)
    viewer.set(tracts.id, OptionName.REGION_FILTER, False)
    assert viewer.region_selection(tracts.id) is None
    assert viewer.kept_streamlines(tracts.id) == 500
    assert tract_options(tracts).regions == (sphere, mask)
    viewer.set(tracts.id, OptionName.REGION_FILTER, True)
    assert viewer.kept_streamlines(tracts.id) == _library(True, False).sum()


def test_clearing_unapplies_the_regions_and_keeps_their_items() -> None:
    viewer = Viewer()
    tracts, _, sphere, mask = _filtered(viewer)
    viewer.clear_regions(tracts.id)
    assert tract_options(tracts).regions == ()
    assert viewer.kept_streamlines(tracts.id) == 500
    assert [r.id for r in viewer.scene.in_category(Category.REGIONS)] == [sphere, mask]


def test_one_region_filters_two_tractograms_and_moving_it_filters_both_again() -> None:
    viewer = Viewer()
    first = viewer.load(TRACKS, Category.TRACTOGRAMS)
    second = viewer.load(TRACKS, Category.TRACTOGRAMS)
    sphere = viewer.add_sphere(BIG, RADIUS, apply=True)
    kept = viewer.kept_streamlines(first.id)
    assert 0 < kept == viewer.kept_streamlines(second.id) < 500
    viewer.set(sphere, OptionName.CENTRE, (500.0, 500.0, 500.0))
    assert viewer.kept_streamlines(first.id) == 0
    assert viewer.kept_streamlines(second.id) == 0


def test_removing_a_region_unapplies_it_everywhere() -> None:
    viewer = Viewer()
    first = viewer.load(TRACKS, Category.TRACTOGRAMS)
    second = viewer.load(TRACKS, Category.TRACTOGRAMS)
    sphere = viewer.add_sphere(BIG, RADIUS, apply=True)
    other = viewer.add_box(BIG, (5.0, 5.0, 5.0), apply=True)
    viewer.remove(sphere)
    assert tract_options(first).regions == tract_options(second).regions == (other,)


def test_a_mask_region_goes_with_its_image() -> None:
    viewer = Viewer()
    tracts, shell, sphere, mask = _filtered(viewer)
    viewer.remove(shell.id)
    assert [r.id for r in viewer.scene.in_category(Category.REGIONS)] == [sphere]
    assert tract_options(tracts).regions == (sphere,)


@pytest.mark.parametrize("radius", [0.0, -1.0])
def test_a_sphere_whose_radius_is_not_positive_is_refused(radius: float) -> None:
    viewer = Viewer()
    with pytest.raises(OptionError, match="Radius"):
        viewer.add_sphere(BIG, radius)
    assert viewer.scene.in_category(Category.REGIONS) == []


@pytest.mark.parametrize("mask", ["4d", "tractogram"])
def test_a_mask_that_is_not_an_image_of_one_volume_is_refused(mask: str) -> None:
    viewer = Viewer()
    if mask == "4d":
        data = np.ones((2, 2, 2, 3), np.float32)
        other = viewer.add(Volume(data, np.eye(4)), Category.OVERLAYS)
    else:
        other = viewer.load(TRACKS, Category.TRACTOGRAMS)
    with pytest.raises(OptionError):
        viewer.add_mask(other.id)
    assert viewer.scene.in_category(Category.REGIONS) == []


def test_applying_refuses_an_item_that_is_not_a_tractogram_or_not_a_region() -> None:
    viewer = Viewer()
    tracts = viewer.load(TRACKS, Category.TRACTOGRAMS)
    shell = viewer.load(SHELL, Category.OVERLAYS)
    sphere = viewer.add_sphere(BIG, RADIUS)
    with pytest.raises(OptionError, match="not a tractogram"):
        viewer.apply_region(shell.id, sphere)
    with pytest.raises(OptionError, match="not a region"):
        viewer.apply_region(tracts.id, shell.id)


def test_a_region_keeps_its_shape() -> None:
    viewer = Viewer()
    sphere = viewer.add_sphere(BIG, RADIUS)
    with pytest.raises(OptionError, match="shape"):
        viewer.set(sphere, OptionName.SHAPE, "box")


def test_a_colour_follows_the_role_until_it_is_changed() -> None:
    viewer = Viewer()
    sphere = viewer.add_sphere(BIG, RADIUS)
    viewer.set(sphere, OptionName.ROLE, RegionRole.WITHIN)
    assert _region(viewer, sphere).colour == ROLE_COLOURS[RegionRole.WITHIN]
    viewer.set(sphere, OptionName.COLOUR, "#123456")
    viewer.set(sphere, OptionName.ROLE, RegionRole.EXCLUDE)
    assert _region(viewer, sphere).colour == Colour("#123456")


def test_moving_a_sphere_moves_its_data() -> None:
    viewer = Viewer()
    sphere = viewer.add_sphere(BIG, RADIUS)
    viewer.set(sphere, OptionName.CENTRE, (1.0, 2.0, 3.0))
    viewer.set(sphere, OptionName.RADIUS, 4.0)
    data = viewer.scene.item(sphere).data
    assert isinstance(data, Sphere)
    assert (tuple(data.centre), data.radius) == ((1.0, 2.0, 3.0), 4.0)
    assert viewer.region_radius == 4.0


def test_a_finished_filter_says_how_many_streamlines_it_keeps() -> None:
    viewer = Viewer()
    tracts, *_ = _filtered(viewer)
    events: list[FilterChanged] = []
    viewer.scene.subscribe(
        lambda e: events.append(e) if isinstance(e, FilterChanged) else None
    )
    viewer.kept_streamlines(tracts.id)
    viewer.kept_streamlines(tracts.id)  # the same regions: nothing new
    viewer.clear_regions(tracts.id)
    viewer.kept_streamlines(tracts.id)
    kept = int(_library(True, False).sum())
    assert [(e.item.id, e.kept, e.total) for e in events] == [
        (tracts.id, kept, 500),
        (tracts.id, 500, 500),
    ]


def test_saving_the_kept_streamlines_writes_what_the_view_keeps(
    tmp_path: Path,
) -> None:
    viewer = Viewer()
    tracts, *_ = _filtered(viewer)
    path = viewer.save_kept(tracts.id, tmp_path / "kept.tck")
    saved = load_tractogram(path)
    selection = viewer.region_selection(tracts.id)
    assert selection is not None
    data = tracts.data
    assert isinstance(data, Tractogram)
    assert saved.num_streamlines == viewer.kept_streamlines(tracts.id)
    np.testing.assert_allclose(saved.points, data[selection].points, atol=1e-5)


def test_a_session_keeps_the_regions_and_what_they_select(tmp_path: Path) -> None:
    saved = Viewer()
    # A removed item, so saved ids differ from those a fresh viewer gives.
    saved.remove(saved.add(Volume(np.zeros((2, 2, 2)), np.eye(4)), Category.VOLUMES).id)
    tracts, _, sphere, _ = _filtered(saved)
    saved.set(sphere, OptionName.COLOUR, "#abcdef")
    saved.set(tracts.id, OptionName.ENDS_ONLY, True)
    assert saved.save_session(tmp_path / "s.lar.json") == []

    viewer = Viewer()
    report = viewer.load_session(tmp_path / "s.lar.json")

    assert report.problems == ()
    new = {i.name: i for i in viewer.scene.items}
    regions = viewer.scene.in_category(Category.REGIONS)
    assert [r.name for r in regions] == ["Sphere 1", "shell_mask.nii.gz (mask)"]
    reopened_sphere, reopened_mask = (r.options for r in regions)
    assert isinstance(reopened_sphere, RegionOptions)
    assert isinstance(reopened_mask, RegionOptions)
    assert (reopened_sphere.centre, reopened_sphere.radius) == (BIG, RADIUS)
    assert reopened_sphere.colour == "#abcdef"
    assert reopened_mask.role == RegionRole.EXCLUDE
    assert reopened_mask.mask == new["shell_mask.nii.gz"].id
    assert tract_options(new["tracks.tck"]).regions == tuple(r.id for r in regions)
    kept = viewer.kept_streamlines(new["tracks.tck"].id)
    assert kept == saved.kept_streamlines(tracts.id) == _library(True, True).sum()


def test_a_session_drops_a_mask_region_whose_image_did_not_open(
    tmp_path: Path,
) -> None:
    mask = tmp_path / "shell_mask.nii.gz"
    shutil.copy(SHELL, mask)
    saved = Viewer()
    saved.load(TRACKS, Category.TRACTOGRAMS)
    saved.add_mask(saved.load(mask, Category.OVERLAYS).id, apply=True)
    saved.save_session(tmp_path / "s.lar.json")
    mask.unlink()

    viewer = Viewer()
    report = viewer.load_session(tmp_path / "s.lar.json")

    (reopened,) = viewer.scene.items
    assert tract_options(reopened).regions == ()
    assert any(
        p.startswith("shell_mask.nii.gz (mask) is dropped") for p in report.problems
    )


def test_a_session_reports_a_region_entry_it_cannot_read(tmp_path: Path) -> None:
    saved = Viewer()
    _filtered(saved)
    path = tmp_path / "s.lar.json"
    saved.save_session(path)
    data = json.loads(path.read_text())
    data["regions"][0]["options"]["radius"] = "big"
    path.write_text(json.dumps(data))

    report = Viewer().load_session(path)

    assert any(p.startswith("Sphere 1: radius is ignored") for p in report.problems)


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
    tracts, *_ = _filtered(viewer)
    first = viewer.region_selection(tracts.id)
    assert viewer.kept_streamlines(tracts.id) == _library(True, False).sum()
    assert viewer.region_selection(tracts.id) is first
    assert len(calls) == 1
    viewer.set(tracts.id, OptionName.ENDS_ONLY, True)
    assert viewer.kept_streamlines(tracts.id) == _library(True, True).sum()
    assert len(calls) == 2


def test_the_selection_work_keeps_nothing_until_it_is_remembered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[RegionFilter] = []

    def counted(tractogram: Tractogram, regions: RegionFilter) -> np.ndarray:
        calls.append(regions)
        return passes_regions(tractogram, regions)

    monkeypatch.setattr(viewer_module, "passes_regions", counted)
    viewer = Viewer()
    tracts, *_ = _filtered(viewer)
    key, work = viewer._selection_work(tracts.id)
    selection = work()
    viewer.region_selection(tracts.id)  # the work kept nothing: filters again
    assert len(calls) == 2
    viewer._remember_selection(tracts.id, key, selection)
    assert viewer.region_selection(tracts.id) is selection
    assert len(calls) == 2
