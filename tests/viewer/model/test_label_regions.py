from pathlib import Path

import numpy as np
import pytest
from viewer_helpers import tract_options

from larmor.io import load_tractogram, load_volume
from larmor.tractography import endpoint_nodes, parcellation_labels, stays_within
from larmor.types import Volume
from larmor.viewer import (
    Category,
    LabelMode,
    LabelSet,
    OptionError,
    OptionName,
    RegionRole,
    Viewer,
)
from larmor.viewer.model.label_regions import slice_cells
from larmor.viewer.model.options import RegionOptions

CONNECTOME = Path(__file__).parents[2] / "data" / "connectome"
TRACKS = CONNECTOME / "tracks.tck"
PARCELLATION = CONNECTOME / "parcellation.nii.gz"
LUT = CONNECTOME / "lut.txt"


def _scene() -> tuple[Viewer, int, int]:
    viewer = Viewer()
    tracts = viewer.load(TRACKS, Category.TRACTOGRAMS)
    parcellation = viewer.load(PARCELLATION, Category.OVERLAYS)
    return viewer, tracts.id, parcellation.id


def _ends(radius: float) -> np.ndarray:
    return endpoint_nodes(
        load_tractogram(TRACKS), load_volume(PARCELLATION), radius=radius
    )


def _common_pair() -> tuple[int, int]:
    """The two labels joined by the most streamlines, so the tests keep some."""
    ends = _ends(4.0)
    ends = ends[(ends > 0).all(axis=1) & (ends[:, 0] != ends[:, 1])]
    pairs, counts = np.unique(np.sort(ends, axis=1), axis=0, return_counts=True)
    a, b = pairs[np.argmax(counts)]
    return int(a), int(b)


@pytest.mark.parametrize("radius", [0.0, 4.0])
def test_from_a_to_b_keeps_the_streamlines_endpoint_nodes_joins(radius: float) -> None:
    viewer, tracts, parcellation = _scene()
    a, b = _common_pair()
    viewer.add_label_region(
        parcellation, labels_a=[a], labels_b=[b], end_radius=radius, apply=True
    )
    ends = _ends(radius)
    expected = ((ends[:, 0] == a) & (ends[:, 1] == b)) | (
        (ends[:, 0] == b) & (ends[:, 1] == a)
    )
    assert 0 < expected.sum() < len(expected)
    np.testing.assert_array_equal(viewer.region_selection(tracts), expected)


def test_an_empty_set_takes_any_end_and_exclude_drops_the_streamlines() -> None:
    viewer, tracts, parcellation = _scene()
    a, _ = _common_pair()
    region = viewer.add_label_region(
        parcellation, labels_a=[a], role=RegionRole.EXCLUDE, apply=True
    )
    ends = _ends(4.0)
    touches = (ends == a).any(axis=1)
    np.testing.assert_array_equal(viewer.region_selection(tracts), ~touches)
    viewer.set(region, OptionName.ROLE, RegionRole.INCLUDE)
    np.testing.assert_array_equal(viewer.region_selection(tracts), touches)


@pytest.mark.parametrize("ends_only", [False, True], ids=["all-points", "ends-only"])
def test_stay_within_keeps_what_stays_within_gives_for_the_labels_mask(
    ends_only: bool,
) -> None:
    viewer, tracts, parcellation = _scene()
    image = load_volume(PARCELLATION)
    labels = parcellation_labels(image)
    # Half of the labels, so that some streamlines stay within and some do not.
    chosen = [int(v) for v in np.unique(labels) if v > 0]
    viewer.add_label_region(
        parcellation, LabelMode.STAY_WITHIN, labels_a=chosen[: len(chosen) // 2]
    )
    viewer.apply_region(tracts, viewer.scene.in_category(Category.REGIONS)[0].id)
    viewer.set(tracts, OptionName.ENDS_ONLY, ends_only)
    mask = Volume(
        np.isin(labels, chosen[: len(chosen) // 2]).astype(np.uint8), image.affine
    )
    expected = stays_within(load_tractogram(TRACKS), mask, ends_only=ends_only)
    assert 0 < expected.sum() < len(expected)
    np.testing.assert_array_equal(viewer.region_selection(tracts), expected)


def test_a_session_keeps_a_label_region_and_what_it_keeps(tmp_path: Path) -> None:
    saved, tracts, parcellation = _scene()
    a, b = _common_pair()
    saved.add_label_region(
        parcellation, labels_a=[a], labels_b=[b], end_radius=2.5, apply=True
    )
    kept = saved.kept_streamlines(tracts)
    assert saved.save_session(tmp_path / "s.lar.json") == []

    viewer = Viewer()
    report = viewer.load_session(tmp_path / "s.lar.json")

    assert report.problems == ()
    (region,) = viewer.scene.in_category(Category.REGIONS)
    o = region.options
    assert isinstance(o, RegionOptions)
    assert (o.label_mode, o.labels_a, o.labels_b, o.end_radius) == (
        LabelMode.FROM_A_TO_B,
        (a,),
        (b,),
        2.5,
    )
    new = {i.name: i for i in viewer.scene.items}
    assert o.parcellation == new["parcellation.nii.gz"].id
    assert tract_options(new["tracks.tck"]).regions == (region.id,)
    assert viewer.kept_streamlines(new["tracks.tck"].id) == kept


def test_a_click_adds_the_label_under_it_and_shift_click_removes_it() -> None:
    viewer, _, parcellation = _scene()
    image = load_volume(PARCELLATION)
    labels = parcellation_labels(image)
    voxel = np.argwhere(labels > 0)[0]
    x, y, z = (float(c) for c in image.affine[:3, :3] @ voxel + image.affine[:3, 3])
    point = (x, y, z)
    label = int(labels[tuple(voxel)])
    viewer.set(parcellation, OptionName.LABELS, True)

    # With no label region yet, the first click makes one on the parcellation.
    assert viewer.click_label(point) == label
    (region,) = viewer.scene.in_category(Category.REGIONS)
    viewer.edit_labels(region.id, LabelSet.B)
    viewer.click_label(point)
    o = viewer.scene.item(region.id).options
    assert isinstance(o, RegionOptions)
    assert (o.labels_a, o.labels_b) == ((label,), (label,))
    viewer.click_label(point, remove=True)
    o = viewer.scene.item(region.id).options
    assert isinstance(o, RegionOptions)
    assert o.labels_b == ()


def test_label_regions_check_their_labels_and_go_with_their_parcellation() -> None:
    viewer, _, parcellation = _scene()
    region = viewer.add_label_region(parcellation)
    with pytest.raises(OptionError, match="from 1"):
        viewer.set(region, OptionName.LABELS_A, (0, 3))
    viewer.remove(parcellation)
    assert viewer.scene.in_category(Category.REGIONS) == []


def test_slice_cells_are_the_chosen_voxels_of_the_slice() -> None:
    labels = np.zeros((4, 5, 3), dtype=np.int32)
    labels[1:3, 1:4, 1] = 7
    labels[0, 0, 1] = 2
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    cut = slice_cells(labels, affine, (7,), axis=2, coordinate=2.0)
    assert len(cut.cells) == 6
    # The outline of a 2 by 3 block of cells has 10 sides.
    assert len(cut.edges) == 10
    centres = cut.cells.mean(axis=1)
    np.testing.assert_allclose(sorted(centres[:, 0]), [2, 2, 2, 4, 4, 4])
    assert len(slice_cells(labels, affine, (7,), axis=2, coordinate=0.0).cells) == 0
