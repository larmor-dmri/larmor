"""Tractograms coloured by value."""

from pathlib import Path

import numpy as np
import pytest
from render_helpers import (
    _streamlines,
    _tracts,
)

from larmor.io import load_tractogram, read_tsf, write_tsf
from larmor.tractography import sample_volume
from larmor.types import Tractogram, Volume
from larmor.viewer import (
    Category,
    Colormap,
    Colouring,
    OptionName,
    Viewer,
)
from larmor.viewer.model import viewer as viewer_module
from larmor.viewer.model.registry import SUBSET_SEED
from larmor.viewer.render import (
    tractogram_adapter,
)
from larmor.viewer.render.geometry import (
    colormap_colours,
)
from larmor.viewer.render.scene_renderer import SceneRenderer
from larmor.viewer.render.tractogram_adapter import (
    TractogramAdapter,
)
from larmor.viewer.render.volume_adapter import (
    COLORMAP_MAPS,
)

pytestmark = pytest.mark.usefixtures("fresh_slab_budget")


@pytest.fixture
def view_size() -> int:
    return 96


def _line(points: int = 5) -> Tractogram:
    # One streamline along x through voxel centres: the drawn order is known.
    x = np.arange(points, dtype=np.float32)
    return Tractogram(np.stack([x, x * 0 + 1, x * 0 + 1], 1), np.array([0, points]))


def _rgb_table(name: Colormap) -> np.ndarray:
    return np.asarray(COLORMAP_MAPS[name].texture.data[:, :3], np.float32)


def test_tsf_values_colour_the_points_through_the_colormap_clamped_to_its_range(
    scene_renderer: tuple[Viewer, SceneRenderer], tmp_path: Path
) -> None:
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _line())
    values = np.array([-1.0, 0.0, 0.5, 1.0, 2.0], np.float32)
    write_tsf(values, np.array([0, 5]), tmp_path / "v.tsf")
    item = adapter.item.id
    viewer.set(item, OptionName.VALUES_FILE, str(tmp_path / "v.tsf"))
    viewer.set(item, OptionName.COLORMAP, Colormap.HOT)
    viewer.set(item, OptionName.MINIMUM, 0.0)
    viewer.set(item, OptionName.MAXIMUM, 1.0)
    table = _rgb_table(Colormap.HOT)
    middle = colormap_colours(
        np.array([0.5]), COLORMAP_MAPS[Colormap.HOT].texture.data, 0, 1
    )
    expected = np.array([table[0], table[0], middle[0], table[-1], table[-1]])
    assert adapter.drawn.values is not None
    assert np.array_equal(adapter.drawn.values, values)
    np.testing.assert_allclose(adapter.drawn.colours, expected, atol=1e-6)
    assert {o.material.color_mode for o in adapter.lines.values()} == {"vertex"}


def test_a_sampled_volume_gives_the_drawn_points_its_values_and_nan_is_grey(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    data = np.zeros((5, 3, 3), np.float32)
    data[:, 1, 1] = [10.0, 11.0, np.nan, 13.0, 14.0]
    volume = viewer.add(Volume(data, np.eye(4)), Category.VOLUMES)
    adapter = _tracts(viewer, renderer, _line())
    item = adapter.item.id
    viewer.set(item, OptionName.POINTS, 3)  # kept: points 0, 2 and 4
    viewer.set(item, OptionName.VALUES_VOLUME, volume.id)
    viewer.set(item, OptionName.COLORMAP, Colormap.GRAY)
    viewer.set(item, OptionName.MINIMUM, 10.0)
    viewer.set(item, OptionName.MAXIMUM, 14.0)
    assert adapter.drawn.values is not None
    np.testing.assert_array_equal(adapter.drawn.values, [10.0, np.nan, 14.0])
    table = _rgb_table(Colormap.GRAY)
    expected = np.array([table[0], [0.5, 0.5, 0.5], table[-1]], np.float32)
    np.testing.assert_allclose(adapter.drawn.colours, expected, atol=1e-6)


def test_colouring_by_value_without_values_is_grey(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _line())
    viewer.set(adapter.item.id, OptionName.COLOURING, Colouring.VALUES)
    assert np.all(adapter.drawn.colours == 0.5)


def test_a_new_colormap_or_range_recolours_without_sampling_again(
    scene_renderer: tuple[Viewer, SceneRenderer], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def counted(tractogram: Tractogram, volume: Volume) -> np.ndarray:
        calls.append(tractogram.num_points)
        return sample_volume(tractogram, volume)

    monkeypatch.setattr(tractogram_adapter, "sample_volume", counted)
    viewer, renderer = scene_renderer
    data = np.arange(45, dtype=np.float32).reshape(5, 3, 3)
    volume = viewer.add(Volume(data, np.eye(4)), Category.VOLUMES)
    adapter = _tracts(viewer, renderer, _line())
    item = adapter.item.id
    viewer.set(item, OptionName.VALUES_VOLUME, volume.id)
    assert calls == [5]
    before = adapter.drawn.colours.copy()
    viewer.set(item, OptionName.COLORMAP, Colormap.HOT)
    viewer.set(item, OptionName.MINIMUM, 20.0)
    viewer.set(item, OptionName.MAXIMUM, 30.0)
    assert calls == [5]
    assert not np.array_equal(adapter.drawn.colours, before)
    assert adapter.drawn.values is not None
    expected = colormap_colours(
        adapter.drawn.values, COLORMAP_MAPS[Colormap.HOT].texture.data, 20.0, 30.0
    )
    np.testing.assert_allclose(adapter.drawn.colours, expected)
    # New drawn points are sampled again.
    viewer.set(item, OptionName.POINTS, 3)
    assert calls == [5, 3]


def test_tsf_values_are_those_of_the_kept_points_of_each_streamline(
    scene_renderer: tuple[Viewer, SceneRenderer], tmp_path: Path
) -> None:
    viewer, renderer = scene_renderer
    line = _line()
    two = Tractogram(
        np.concatenate([line.points, line.points + 2]), np.array([0, 5, 10])
    )
    adapter = _tracts(viewer, renderer, two)
    values = np.arange(10, dtype=np.float32)
    write_tsf(values, np.array([0, 5, 10]), tmp_path / "v.tsf")
    item = adapter.item.id
    viewer.set(item, OptionName.POINTS, 3)  # kept: points 0, 2 and 4 of each
    viewer.set(item, OptionName.VALUES_FILE, str(tmp_path / "v.tsf"))
    assert adapter.drawn.values is not None
    drawn = adapter.drawn.values.reshape(2, 3)
    # The streamlines are drawn in a random order.
    assert sorted(map(tuple, drawn.tolist())) == [(0, 2, 4), (5, 7, 9)]


def test_a_tsf_file_colours_a_tractogram_read_as_a_subset_of_its_file(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    data = Path(__file__).parents[1] / "data"
    tracks = data / "editing" / "tracks.tck"
    item = viewer.load(tracks, Category.TRACTOGRAMS, max_streamlines=100)
    adapter = renderer.adapters[item.id]
    assert isinstance(adapter, TractogramAdapter)
    viewer.set(item.id, OptionName.POINTS, 1000)  # every point drawn
    viewer.set(item.id, OptionName.VALUES_FILE, str(data / "tsf" / "bzero.tsf"))

    whole = load_tractogram(tracks)
    kept = whole.subsample_indices(100, seed=SUBSET_SEED)
    values, offsets = read_tsf(data / "tsf" / "bzero.tsf", tractogram=whole)
    by_streamline = [values[offsets[i] : offsets[i + 1]] for i in kept]
    drawn = adapter.drawn
    assert drawn.values is not None
    for k, j in enumerate(adapter._order):
        start, stop = drawn.offsets[k], drawn.offsets[k + 1]
        assert np.array_equal(drawn.values[start:stop], by_streamline[j])


def test_the_renderer_tells_how_many_streamlines_the_regions_keep(
    scene_renderer: tuple[Viewer, SceneRenderer], monkeypatch: pytest.MonkeyPatch
) -> None:
    viewer, renderer = scene_renderer
    told: list[str] = []
    renderer.subscribe_notices(told.append)
    data = _streamlines()
    adapter = _tracts(viewer, renderer, data)
    viewer.add_sphere(adapter.item.id, 15.0, centre=(0.0, 0.0, 0.0))
    assert told == ["Tractogram 1: 69 of 200 streamlines pass the regions."]
    # The viewer keeps the adapter's selection: counting filters no more.
    monkeypatch.setattr(viewer_module, "passes_regions", None)
    assert viewer.kept_streamlines(adapter.item.id) == 69
