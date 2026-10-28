"""Saving an item to a file, and replacing an item's data in place."""

from pathlib import Path

import numpy as np
import pytest
from viewer_helpers import record_events

from larmor.io import (
    load_tractogram,
    load_volume,
    read_tsf,
    save_tractogram,
    write_tsf,
)
from larmor.types import Grid, Surface, Tractogram, Volume
from larmor.viewer import Category, Colouring, OptionName, Viewer
from larmor.viewer.model.events import DataChanged, Event, OptionChanged, SourceChanged
from larmor.viewer.model.options import (
    OverlayOptions,
    RegionOptions,
    SurfaceOptions,
    TractogramOptions,
    VolumeOptions,
)


def _tractogram(shift: float = 0.0) -> Tractogram:
    points = np.arange(18, dtype=np.float32).reshape(-1, 3) + shift
    return Tractogram(points, np.array([0, 2, 6]))


def _volume(scale: float = 1.0) -> Volume:
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4) * scale
    return Volume(data, np.diag([2.0, 2.0, 2.0, 1.0]))


@pytest.mark.parametrize("name", ["bundle.tck", "bundle.trk"])
def test_a_tractogram_saved_reads_back_the_same(tmp_path: Path, name: str) -> None:
    viewer = Viewer()
    tract = _tractogram()
    # A TRK needs a grid: the main volume's, as the tractogram has none.
    viewer.add(Volume(np.zeros((20, 20, 20), np.float32), np.eye(4)), Category.VOLUMES)
    item = viewer.add(tract, Category.TRACTOGRAMS)
    viewer.save_item(item.id, tmp_path / name)
    back = load_tractogram(tmp_path / name)
    np.testing.assert_allclose(back.points, tract.points, atol=1e-4)
    np.testing.assert_array_equal(back.offsets, tract.offsets)


def test_a_volume_saved_reads_back_the_same(tmp_path: Path) -> None:
    viewer = Viewer()
    volume = _volume()
    item = viewer.add(volume, Category.OVERLAYS)
    viewer.save_item(item.id, tmp_path / "map.nii.gz")
    back = load_volume(tmp_path / "map.nii.gz")
    np.testing.assert_array_equal(back.data, volume.data)
    np.testing.assert_array_equal(back.affine, volume.affine)


def test_a_saved_item_refers_to_its_file_and_a_session_reopens_it(
    tmp_path: Path,
) -> None:
    viewer = Viewer()
    item = viewer.add(_tractogram(), Category.TRACTOGRAMS, name="Bundle")
    viewer.set(item.id, OptionName.COLOUR, "#ff0000")
    events = record_events(viewer)
    viewer.save_item(item.id, tmp_path / "bundle.tck")
    assert item.source == tmp_path / "bundle.tck"
    assert events == [SourceChanged(item)]
    assert viewer.save_session(tmp_path / "s.lar.json") == []
    report = Viewer().load_session(tmp_path / "s.lar.json")
    [back] = report.items
    assert back.name == "Bundle"
    assert isinstance(back.options, TractogramOptions)
    assert back.options.colour == "#ff0000"


def test_a_trk_takes_the_tractograms_own_grid_first(tmp_path: Path) -> None:
    viewer = Viewer()
    viewer.add(Volume(np.zeros((4, 4, 4), np.float32), np.eye(4)), Category.VOLUMES)
    own = Grid((30, 30, 30), np.diag([2.0, 2.0, 2.0, 1.0]))
    tract = Tractogram(_tractogram().points, _tractogram().offsets, reference=own)
    item = viewer.add(tract, Category.TRACTOGRAMS)
    viewer.save_item(item.id, tmp_path / "t.trk")
    back = load_tractogram(tmp_path / "t.trk")
    assert back.reference is not None
    assert back.reference.shape == (30, 30, 30)


def test_a_surface_or_a_wrong_extension_is_refused(tmp_path: Path) -> None:
    viewer = Viewer()
    surface = Surface(np.eye(3), np.array([[0, 1, 2]]))
    s = viewer.add(surface, Category.SURFACES)
    t = viewer.add(_tractogram(), Category.TRACTOGRAMS)
    with pytest.raises(ValueError, match="cannot be saved"):
        viewer.save_item(s.id, tmp_path / "s.gii")
    with pytest.raises(ValueError):
        viewer.save_item(t.id, tmp_path / "t.nii.gz")
    assert t.source is None
    assert list(tmp_path.iterdir()) == []


def test_replacing_data_keeps_id_options_and_layers_order() -> None:
    viewer = Viewer()
    first = viewer.add(_tractogram(), Category.TRACTOGRAMS, name="A")
    item = viewer.add(_tractogram(), Category.TRACTOGRAMS, name="B")
    last = viewer.add(_tractogram(), Category.TRACTOGRAMS, name="C")
    viewer.set(item.id, OptionName.COLOUR, "#00ff00")
    options = item.options
    events = record_events(viewer)
    new = _tractogram(shift=5.0)
    viewer.replace_data(item.id, new)
    assert [i.id for i in viewer.scene.items] == [first.id, item.id, last.id]
    assert viewer.scene.item(item.id).data is new
    assert item.options == options
    assert events == [DataChanged(item)]


def test_replaced_data_no_longer_refers_to_the_file(tmp_path: Path) -> None:
    viewer = Viewer()
    item = viewer.add(_tractogram(), Category.TRACTOGRAMS)
    viewer.save_item(item.id, tmp_path / "t.tck")
    viewer.replace_data(item.id, _tractogram(shift=1.0))
    assert item.source is None
    assert viewer.save_session(tmp_path / "s.lar.json") == [item.name]


def _overlay(lo: float, hi: float) -> Volume:
    return Volume(np.linspace(lo, hi, 8, dtype=np.float32).reshape(2, 2, 2), np.eye(4))


@pytest.mark.parametrize(
    ("new", "expected"),
    [
        ((0.0, 4.0), (0.8, 4.0)),  # wider: the maximum stays at the end
        ((10.0, 20.0), (12.0, 20.0)),  # shifted past the old range
        ((-20.0, -10.0), (-18.0, -10.0)),  # shifted below it
    ],
)
def test_range_options_keep_their_place_in_the_new_range(
    new: tuple[float, float], expected: tuple[float, float]
) -> None:
    viewer = Viewer()
    item = viewer.add(_overlay(0.0, 1.0), Category.OVERLAYS)
    viewer.set(item.id, OptionName.MINIMUM, 0.2)  # the maximum stays at the end
    windows: list[tuple[float, float]] = []

    def window(event: Event) -> None:
        o = item.options
        assert isinstance(o, OverlayOptions)
        windows.append((o.minimum, o.maximum))

    viewer.scene.subscribe(window)
    viewer.replace_data(item.id, _overlay(*new))
    assert item.value_range == pytest.approx(new)
    assert windows[-1] == pytest.approx(expected)
    assert all(lo <= hi for lo, hi in windows)  # also between the events


def test_the_replace_event_comes_before_the_range_options() -> None:
    viewer = Viewer()
    item = viewer.add(_overlay(0.0, 1.0), Category.OVERLAYS)
    events = record_events(viewer)
    viewer.replace_data(item.id, _overlay(0.0, 2.0))
    assert isinstance(events[0], DataChanged)
    changed = [e.name for e in events if isinstance(e, OptionChanged)]
    assert set(changed) == {OptionName.MAXIMUM}  # the minimum and threshold stay 0


def test_a_tractograms_tsf_values_are_cleared_by_a_replace(tmp_path: Path) -> None:
    viewer = Viewer()
    tract = _tractogram()
    item = viewer.add(tract, Category.TRACTOGRAMS)
    write_tsf(np.arange(6.0), tract.offsets, tmp_path / "v.tsf")
    viewer.set(item.id, OptionName.VALUES_FILE, str(tmp_path / "v.tsf"))
    viewer.replace_data(item.id, _tractogram(shift=1.0))
    o = item.options
    assert isinstance(o, TractogramOptions)
    assert o.values_file == ""
    assert o.colouring == TractogramOptions().colouring != Colouring.VALUES
    assert viewer._tsf_point_values(item.id) is None


def test_overlay_labels_gone_from_the_new_data_are_dropped() -> None:
    viewer = Viewer()
    labels = np.array([1, 2, 3, 0], np.float32).reshape(1, 2, 2)
    item = viewer.add(Volume(labels, np.eye(4)), Category.OVERLAYS)
    viewer.set(item.id, OptionName.SURFACE_LABELS, (1, 3))
    viewer.replace_data(item.id, Volume(np.where(labels == 3, 0, labels), np.eye(4)))
    assert isinstance(item.options, OverlayOptions)
    assert item.options.surface_labels == (1,)
    assert viewer.labels(item.id) == (1, 2)


def test_a_surfaces_values_gone_from_the_new_data_are_cleared() -> None:
    viewer = Viewer()
    faces = np.array([[0, 1, 2]])
    values = {"thickness": np.array([1.0, 2.0, 3.0])}
    item = viewer.add(Surface(np.eye(3), faces, vertex_data=values), Category.SURFACES)
    viewer.set(item.id, OptionName.COLOUR_BY, "thickness")
    viewer.replace_data(item.id, Surface(np.eye(3), faces))
    assert isinstance(item.options, SurfaceOptions)
    assert item.options.colour_by is None


def _five(path: Path) -> Tractogram:
    points = np.arange(30, dtype=np.float32).reshape(-1, 3)
    tract = Tractogram(points, np.arange(0, 12, 2))
    save_tractogram(tract, path)
    return tract


def test_a_subset_is_never_saved_over_its_own_file(tmp_path: Path) -> None:
    whole = tmp_path / "whole.tck"
    _five(whole)
    viewer = Viewer()
    item = viewer.load(whole, Category.TRACTOGRAMS, max_streamlines=2)
    with pytest.raises(ValueError, match="2 of the 5 streamlines"):
        viewer.save_item(item.id, tmp_path / "." / "whole.tck")
    assert load_tractogram(whole).num_streamlines == 5
    viewer.save_item(item.id, tmp_path / "part.tck")
    assert load_tractogram(tmp_path / "part.tck").num_streamlines == 2


def test_a_saved_subset_takes_its_cut_tsf_values_along(tmp_path: Path) -> None:
    tract = _five(tmp_path / "whole.tck")
    write_tsf(np.arange(10.0), tract.offsets, tmp_path / "whole.tsf")
    viewer = Viewer()
    item = viewer.load(tmp_path / "whole.tck", Category.TRACTOGRAMS, max_streamlines=2)
    viewer.set(item.id, OptionName.VALUES_FILE, str(tmp_path / "whole.tsf"))
    shown = viewer._tsf_point_values(item.id)
    viewer.save_item(item.id, tmp_path / "part.tck")
    assert isinstance(item.options, TractogramOptions)
    assert item.options.values_file == str(tmp_path / "part.tsf")
    assert isinstance(item.data, Tractogram)
    values, _ = read_tsf(tmp_path / "part.tsf", tractogram=item.data)  # matches
    np.testing.assert_array_equal(values, shown)
    viewer.save_session(tmp_path / "s.lar.json")
    assert Viewer().load_session(tmp_path / "s.lar.json").problems == ()


def _masked() -> tuple[Viewer, int, int]:
    """A tractogram of two streamlines, one at x = 0 and one at x = 3, and a mask."""
    viewer = Viewer()
    points = np.array([[0, 0, 0], [0, 1, 0], [3, 0, 0], [3, 1, 0]], np.float32)
    tract = viewer.add(Tractogram(points, np.array([0, 2, 4])), Category.TRACTOGRAMS)
    mask = np.zeros((4, 2, 1), np.float32)
    mask[0] = 1
    overlay = viewer.add(Volume(mask, np.eye(4)), Category.OVERLAYS)
    viewer.add_mask(overlay.id, apply=True)
    return viewer, tract.id, overlay.id


def _other_mask() -> Volume:
    mask = np.zeros((4, 2, 1), np.float32)
    mask[3] = 1
    return Volume(mask, np.eye(4))


def _kept(viewer: Viewer, tract: int) -> list[bool]:
    selection = viewer.region_selection(tract)
    assert selection is not None
    return [bool(k) for k in selection]


def test_a_replaced_mask_filters_the_tractograms_using_it_again() -> None:
    viewer, tract, overlay = _masked()
    assert _kept(viewer, tract) == [True, False]
    viewer.replace_data(overlay, _other_mask())
    assert _kept(viewer, tract) == [False, True]


def test_a_selection_made_before_a_replace_is_not_kept() -> None:
    viewer, tract, overlay = _masked()
    key, work = viewer._selection_work(tract)  # as on the render thread
    viewer.replace_data(overlay, _other_mask())
    viewer._remember_selection(tract, key, work())  # arrives late
    assert tract not in viewer._selections
    assert _kept(viewer, tract) == [False, True]


def test_a_volume_index_past_the_new_volumes_becomes_the_last() -> None:
    viewer = Viewer()
    data = np.zeros((2, 2, 2, 5), np.float32)
    item = viewer.add(Volume(data, np.eye(4)), Category.VOLUMES)
    viewer.set(item.id, OptionName.VOLUME_INDEX, 4)
    viewer.replace_data(item.id, Volume(data[..., :2], np.eye(4)))
    assert isinstance(item.options, VolumeOptions)
    assert item.options.volume_index == 1


def test_data_of_another_type_is_refused_and_changes_nothing() -> None:
    viewer = Viewer()
    item = viewer.add(_tractogram(), Category.TRACTOGRAMS)
    data = item.data
    with pytest.raises(ValueError):
        viewer.replace_data(item.id, _volume())
    assert item.data is data


def _dti_modulated() -> tuple[Viewer, int, int]:
    """A DTI image modulated by a 3D image on its grid."""
    viewer = Viewer()
    vectors = viewer.add(
        Volume(np.zeros((2, 2, 2, 3), np.float32), np.eye(4)), Category.VOLUMES
    )
    weights = viewer.add(
        Volume(np.zeros((2, 2, 2), np.float32), np.eye(4)), Category.OVERLAYS
    )
    viewer.set(vectors.id, OptionName.DTI, True)
    viewer.set(vectors.id, OptionName.MODULATE, weights.id)
    return viewer, vectors.id, weights.id


def test_a_modulating_image_replaced_off_the_grid_modulates_no_more() -> None:
    viewer, vectors, weights = _dti_modulated()
    viewer.replace_data(weights, Volume(np.zeros((3, 3, 3), np.float32), np.eye(4)))
    o = viewer.scene.item(vectors).options
    assert isinstance(o, VolumeOptions)
    assert o.modulate is None


def test_a_dti_image_replaced_by_a_3d_one_turns_dti_and_modulate_off() -> None:
    viewer, vectors, _ = _dti_modulated()
    viewer.replace_data(vectors, Volume(np.zeros((3, 3, 3), np.float32), np.eye(4)))
    o = viewer.scene.item(vectors).options
    assert isinstance(o, VolumeOptions)
    assert (o.dti, o.modulate) == (False, None)


def test_a_mask_replaced_by_several_volumes_is_no_longer_the_regions_mask() -> None:
    viewer, tract, overlay = _masked()
    viewer.replace_data(overlay, Volume(np.zeros((4, 2, 1, 3), np.float32), np.eye(4)))
    region = viewer.scene.in_category(Category.REGIONS)[0]
    assert isinstance(region.options, RegionOptions)
    assert region.options.mask is None
    assert viewer.kept_streamlines(tract) == 2
