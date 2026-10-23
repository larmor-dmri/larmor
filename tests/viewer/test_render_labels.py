"""Overlays shown as labels."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pygfx as gfx
import pytest
from render_helpers import (
    _adapter,
    _image,
    _one_grid,
    _palette,
    _rgb,
    _same_map_on_every_slice,
    _table,
    _thresholded,
)

from larmor.io import read_lookup_table
from larmor.types import Label, Volume
from larmor.viewer import (
    Category,
    Colormap,
    Item,
    OptionName,
    Viewer,
    ViewKind,
)
from larmor.viewer.model import labels as model_labels
from larmor.viewer.render import (
    volume_adapter,
)
from larmor.viewer.render.scene_renderer import SceneRenderer
from larmor.viewer.render.view import ViewRenderer
from larmor.viewer.render.volume_adapter import (
    VolumeAdapter,
    _label_colours,
)

pytestmark = pytest.mark.usefixtures("fresh_slab_budget")


@pytest.fixture
def view_size() -> int:
    return 96


def test_label_0_is_transparent_and_the_others_opaque() -> None:
    colours = _table(_label_colours(np.array([0.0, 1.0, 7.0, -4.0]), {}))
    assert list(colours[:, 3]) == [0.0, 1.0, 1.0, 1.0]


def test_label_0_is_transparent_even_with_a_colour_in_the_table() -> None:
    table = {0: Label("Unknown", (255, 255, 255)), 5: Label("Five", (255, 0, 51))}
    colours = _table(_label_colours(np.array([0.0, 5.0]), table))
    assert colours[0, 3] == 0.0
    assert np.array_equal(colours[1], np.array([1.0, 0.0, 0.2, 1.0], np.float32))


def test_table_colours_are_scaled_to_0_1_and_others_use_the_palette() -> None:
    table = {3: Label("Three", (10, 128, 255)), 2000: Label("Big", (0, 0, 0))}
    colours = _table(_label_colours(np.array([0.0, 3.0, 4.0, 2000.0]), table))
    assert colours.shape == (4, 4)
    assert colours[1] == pytest.approx([10 / 255, 128 / 255, 1.0, 1.0])
    assert np.array_equal(colours[2], _palette(4))
    assert np.array_equal(colours[3], [0.0, 0.0, 0.0, 1.0])


def test_the_built_in_colours_depend_only_on_the_label() -> None:
    values = np.arange(1.0, 40.0)
    first = _table(_label_colours(values, {}))
    again = _table(_label_colours(values[::-1], {}))[::-1]
    assert np.array_equal(first, again)
    assert all(np.array_equal(first[k], _palette(v)) for k, v in enumerate(values))


def test_consecutive_labels_have_distinct_colours() -> None:
    colours = _table(_label_colours(np.arange(1.0, 200.0), {}))[:, :3]
    # Hues 0.618 of a turn apart differ by at least a tenth of a turn.
    assert np.abs(np.diff(colours, axis=0)).max(axis=1).min() > 0.1
    assert len({tuple(c) for c in colours}) == len(colours)


@pytest.mark.parametrize(
    ("value", "label"), [(2.9999, 3), (3.4, 3), (0.3, 0), (-0.2, 0), (7.6, 8)]
)
def test_label_values_are_rounded(value: float, label: int) -> None:
    table = {3: Label("Three", (255, 0, 0))}
    colour = _table(_label_colours(np.array([value]), table))[0]
    assert np.array_equal(colour, _table(_label_colours(np.array([label]), table))[0])


# The parcellation below: labels 3 and 2000 in two blocks, 0 elsewhere.
LABEL_VALUES = np.array([0, 3, 2000])


# Points a quarter voxel inside each block from its edge with label 0: nearest
# sampling shows the label, linear would blend it with its neighbour.
LOW_BLOCK, HIGH_BLOCK = np.array((0.75, 0.75, 4.0)), np.array((6.25, 6.25, 4.0))


LUT = "3 Three 200 30 60 0\n2000 Big 10 220 90 0\n"


def _parcellation(dtype: type = np.int16) -> Volume:
    data: np.ndarray = np.zeros((8, 8, 8), dtype)
    data[1:3, 1:3, :] = 3
    data[5:7, 5:7, :] = 2000
    return Volume(data, np.eye(4))


def _indices(volume: Volume) -> np.ndarray:
    return np.searchsorted(LABEL_VALUES, volume.data)


def _lut(tmp_path: Path, text: str = LUT, name: str = "lut.txt") -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _labelled(
    views: dict[ViewKind, ViewRenderer], volume: Volume | None = None
) -> tuple[Viewer, Item, VolumeAdapter]:
    viewer = Viewer()
    item = viewer.add(_parcellation() if volume is None else volume, Category.OVERLAYS)
    adapter = _adapter(viewer, views, item)
    return viewer, item, adapter


def _grids(adapter: VolumeAdapter) -> list[gfx.Texture]:
    return [obj.geometry.grid for obj in adapter.slices.values()]


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.float32, np.uint16])
def test_labels_slices_share_a_texture_of_the_label_indices(
    dtype: type, views: dict[ViewKind, ViewRenderer]
) -> None:
    volume = _parcellation(dtype)
    viewer, item, adapter = _labelled(views, volume)
    (values,) = set(map(id, _grids(adapter)))
    viewer.set(item.id, OptionName.LABELS, True)
    grid, *others = _grids(adapter)
    assert all(other is grid for other in others)
    assert id(grid) != values
    assert grid.data.dtype == np.uint16
    assert np.array_equal(grid.data.T, _indices(volume))


def test_labels_put_each_index_on_the_centre_of_its_colour(
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, item, adapter = _labelled(views)
    viewer.set(item.id, OptionName.LABELS, True)
    expected = _table(_label_colours(LABEL_VALUES, {}))
    for obj in adapter.slices.values():
        assert obj.material.clim == (-0.5, 2.5)
        assert obj.material.interpolation == "nearest"
        assert np.array_equal(_table(obj.material.map), expected)


def test_labels_ignore_the_threshold_range_colormap_and_interpolation(
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, item, adapter = _labelled(views)
    viewer.set(item.id, OptionName.LABELS, True)
    before = _same_map_on_every_slice(adapter)
    for name, value in [
        (OptionName.THRESHOLD, 1000.0),
        (OptionName.MINIMUM, 5.0),
        (OptionName.MAXIMUM, 50.0),
        (OptionName.COLORMAP, Colormap.HOT),
        (OptionName.INTERPOLATE, True),
    ]:
        viewer.set(item.id, name, value)
        for obj in adapter.slices.values():
            assert obj.material.clim == (-0.5, 2.5)
            assert obj.material.interpolation == "nearest"
        assert np.array_equal(_same_map_on_every_slice(adapter), before)


def test_labels_keep_the_other_options(views: dict[ViewKind, ViewRenderer]) -> None:
    viewer, item, adapter = _labelled(views)
    viewer.set(item.id, OptionName.LABELS, True)
    viewer.set(item.id, OptionName.OPACITY_2D, 0.5)
    viewer.set(item.id, OptionName.VISIBLE, False)
    for obj in adapter.slices.values():
        assert obj.material.opacity == 0.5
        assert not obj.visible


def test_labels_off_shows_the_values_with_their_options_again(
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, item, adapter = _labelled(views)
    viewer.set(item.id, OptionName.THRESHOLD, 1000.0)
    viewer.set(item.id, OptionName.COLORMAP, Colormap.HOT)
    viewer.set(item.id, OptionName.MAXIMUM, 1500.0)
    values = _grids(adapter)[0]
    image = [
        (o.material.clim, o.material.interpolation, _table(o.material.map))
        for o in adapter.slices.values()
    ]
    viewer.set(item.id, OptionName.LABELS, True)
    viewer.set(item.id, OptionName.LABELS, False)
    assert all(grid is values for grid in _grids(adapter))
    for obj, (clim, interpolation, cmap) in zip(
        adapter.slices.values(), image, strict=True
    ):
        assert obj.material.clim == clim == (0.0, 1500.0)
        assert obj.material.interpolation == interpolation == "linear"
        assert np.array_equal(_table(obj.material.map), cmap)
    # 1000 of 0 to 1500 hides ceil(170.7) entries of hot.
    assert np.array_equal(cmap, _thresholded(Colormap.HOT, 171))


def test_the_index_texture_is_made_once(
    views: dict[ViewKind, ViewRenderer], monkeypatch: pytest.MonkeyPatch
) -> None:
    counted: list[tuple[int, ...]] = []
    unique: Any = np.unique

    def spy(data: np.ndarray, **kwargs: bool) -> object:
        if kwargs.get("return_inverse"):
            counted.append(data.shape)
        return unique(data, **kwargs)

    monkeypatch.setattr(np, "unique", spy)
    viewer, item, adapter = _labelled(views)
    assert counted == []  # only when first shown as labels
    viewer.set(item.id, OptionName.LABELS, True)
    first = _grids(adapter)[0]
    for labels in (False, True, False, True):
        viewer.set(item.id, OptionName.LABELS, labels)
    viewer.set(item.id, OptionName.OPACITY_2D, 0.5)
    assert counted == [(8, 8, 8)]
    assert all(grid is first for grid in _grids(adapter))


def test_a_4d_parcellation_shows_its_first_volume_as_labels(
    views: dict[ViewKind, ViewRenderer],
) -> None:
    volume = _parcellation()
    data = np.stack([volume.data, np.full_like(volume.data, 9)], axis=-1)
    viewer, item, adapter = _labelled(views, Volume(data, np.eye(4)))
    viewer.set(item.id, OptionName.LABELS, True)
    assert np.array_equal(_grids(adapter)[0].data.T, _indices(volume))
    assert len(_same_map_on_every_slice(adapter)) == 3


# Volume 1 of this series: label 7 in a block, 0 elsewhere.
SECOND_VALUES = np.array([0, 7])


def _label_series() -> Volume:
    first = _parcellation().data
    second = np.zeros_like(first)
    second[3:5, 3:5, :] = 7
    return Volume(np.stack([first, second], axis=-1), np.eye(4))


@pytest.mark.parametrize("labels_first", [True, False], ids=["then-index", "index"])
def test_a_4d_parcellation_shows_the_labels_of_the_chosen_volume(
    labels_first: bool, views: dict[ViewKind, ViewRenderer]
) -> None:
    volume = _label_series()
    viewer, item, adapter = _labelled(views, volume)
    steps = [(OptionName.LABELS, True), (OptionName.VOLUME_INDEX, 1)]
    for name, value in steps if labels_first else steps[::-1]:
        viewer.set(item.id, name, value)
    grid = _one_grid(adapter)
    assert grid.data.dtype == np.uint16
    assert np.array_equal(
        grid.data.T, np.searchsorted(SECOND_VALUES, volume.data[..., 1])
    )
    expected = _table(_label_colours(SECOND_VALUES, {}))
    assert np.array_equal(_same_map_on_every_slice(adapter), expected)
    assert all(o.material.clim == (-0.5, 1.5) for o in adapter.slices.values())


def test_a_4d_parcellation_back_at_volume_0_shows_its_labels_again(
    views: dict[ViewKind, ViewRenderer],
) -> None:
    volume = _label_series()
    viewer, item, adapter = _labelled(views, volume)
    viewer.set(item.id, OptionName.LABELS, True)
    viewer.set(item.id, OptionName.VOLUME_INDEX, 1)
    viewer.set(item.id, OptionName.VOLUME_INDEX, 0)
    assert np.array_equal(_one_grid(adapter).data.T, _indices(_parcellation()))
    expected = _table(_label_colours(LABEL_VALUES, {}))
    assert np.array_equal(_same_map_on_every_slice(adapter), expected)


def test_a_4d_parcellation_shown_as_an_image_again_shows_the_chosen_volume(
    views: dict[ViewKind, ViewRenderer],
) -> None:
    volume = _label_series()
    viewer, item, adapter = _labelled(views, volume)
    viewer.set(item.id, OptionName.LABELS, True)
    viewer.set(item.id, OptionName.VOLUME_INDEX, 1)
    viewer.set(item.id, OptionName.LABELS, False)
    assert np.array_equal(_one_grid(adapter).data.T, volume.data[..., 1])


def test_scaled_label_values_take_their_table_colours(
    views: dict[ViewKind, ViewRenderer], tmp_path: Path
) -> None:
    # Stored 1 and 2 are 3 and 2000 scaled.
    data = np.zeros((4, 4, 4), np.int16)
    data[0, 0, 0], data[1, 1, 1] = 1, 2
    viewer, item, adapter = _labelled(
        views, Volume(data, np.eye(4), slope=1997.0, intercept=-1994.0)
    )
    viewer.set(item.id, OptionName.LOOKUP_TABLE, _lut(tmp_path))
    viewer.set(item.id, OptionName.LABELS, True)
    colours = _same_map_on_every_slice(adapter)
    assert colours[1] == pytest.approx([200 / 255, 30 / 255, 60 / 255, 1.0])
    assert colours[2] == pytest.approx([10 / 255, 220 / 255, 90 / 255, 1.0])
    # Scaled, stored 0 is -1994: not label 0, so opaque.
    assert colours[0, 3] == 1.0


def test_a_lookup_table_colours_its_labels(
    views: dict[ViewKind, ViewRenderer], tmp_path: Path
) -> None:
    viewer, item, adapter = _labelled(views)
    viewer.set(item.id, OptionName.LABELS, True)
    viewer.set(item.id, OptionName.LOOKUP_TABLE, _lut(tmp_path))
    table = {3: Label("Three", (200, 30, 60)), 2000: Label("Big", (10, 220, 90))}
    expected = _table(_label_colours(LABEL_VALUES, table))
    assert np.array_equal(_same_map_on_every_slice(adapter), expected)
    viewer.set(item.id, OptionName.LOOKUP_TABLE, "")
    expected = _table(_label_colours(LABEL_VALUES, {}))
    assert np.array_equal(_same_map_on_every_slice(adapter), expected)


def test_the_lookup_table_is_read_again_only_when_its_path_changes(
    views: dict[ViewKind, ViewRenderer],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read: list[str] = []

    def spy(path: str) -> dict[int, Label]:
        read.append(path)
        return read_lookup_table(path)

    monkeypatch.setattr(model_labels, "read_lookup_table", spy)
    first, second = _lut(tmp_path), _lut(tmp_path, "3 Three 1 2 3 0\n", "two.txt")
    viewer, item, adapter = _labelled(views)
    viewer.set(item.id, OptionName.LOOKUP_TABLE, first)
    assert read == []  # not shown as labels yet
    viewer.set(item.id, OptionName.LABELS, True)
    viewer.set(item.id, OptionName.OPACITY_2D, 0.5)
    viewer.set(item.id, OptionName.LABELS, False)
    viewer.set(item.id, OptionName.LABELS, True)
    assert read == [first]
    viewer.set(item.id, OptionName.LOOKUP_TABLE, second)
    assert read == [first, second]
    assert _same_map_on_every_slice(adapter)[1] == pytest.approx(
        [1 / 255, 2 / 255, 3 / 255, 1.0]
    )


def test_a_lookup_table_gone_since_it_was_set_gives_the_built_in_colours(
    views: dict[ViewKind, ViewRenderer],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = _lut(tmp_path)
    viewer, item, adapter = _labelled(views)
    viewer.set(item.id, OptionName.LOOKUP_TABLE, path)
    Path(path).unlink()
    with caplog.at_level(logging.WARNING, logger="larmor"):
        viewer.set(item.id, OptionName.LABELS, True)
    expected = _table(_label_colours(LABEL_VALUES, {}))
    assert np.array_equal(_same_map_on_every_slice(adapter), expected)
    assert path in caplog.text
    assert "built-in colours" in caplog.text


def test_too_many_labels_show_as_an_image_with_a_warning(
    views: dict[ViewKind, ViewRenderer],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(volume_adapter, "MAX_LABELS", 2)
    viewer, item, adapter = _labelled(views)
    values = _grids(adapter)[0]
    with caplog.at_level(logging.WARNING, logger="larmor"):
        viewer.set(item.id, OptionName.LABELS, True)
    assert all(grid is values for grid in _grids(adapter))
    for obj in adapter.slices.values():
        assert obj.material.clim == (0.0, 2000.0)
        assert obj.material.interpolation == "linear"
    assert np.array_equal(
        _same_map_on_every_slice(adapter), _thresholded(Colormap.GRAY, 1)
    )
    assert f"{item.name} has 3 distinct values" in caplog.text


def test_as_many_labels_as_allowed_show_as_labels(
    views: dict[ViewKind, ViewRenderer], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(volume_adapter, "MAX_LABELS", 3)
    viewer, item, adapter = _labelled(views)
    viewer.set(item.id, OptionName.LABELS, True)
    assert all(o.material.clim == (-0.5, 2.5) for o in adapter.slices.values())


@pytest.mark.parametrize("with_table", [True, False], ids=["table", "built-in"])
def test_a_rendered_parcellation_shows_its_colours_over_the_volume(
    with_table: bool,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
    tmp_path: Path,
) -> None:
    viewer, _ = scene_renderer
    # Opaque mid grey; the corners, off the axial slice at z = 4, give it 0 to 1.
    grey = np.full((8, 8, 8), 0.5, np.float32)
    grey[0, 0, 0], grey[7, 7, 7] = 0.0, 1.0
    volume = viewer.add(Volume(grey, np.eye(4)), Category.VOLUMES)
    viewer.set(volume.id, OptionName.INTERPOLATE, False)
    overlay = viewer.add(_parcellation(), Category.OVERLAYS)
    if with_table:
        viewer.set(overlay.id, OptionName.LOOKUP_TABLE, _lut(tmp_path))
    viewer.set(overlay.id, OptionName.LABELS, True)
    if with_table:
        low, high = [200, 30, 60], [10, 220, 90]
    else:
        palette = np.round(_table(_label_colours(LABEL_VALUES, {}))[:, :3] * 255)
        low, high = palette[1].astype(int).tolist(), palette[2].astype(int).tolist()
    view = views[ViewKind.AXIAL]
    image = _image(view)
    assert _rgb(view, image, LOW_BLOCK) == low
    assert _rgb(view, image, HIGH_BLOCK) == high
    # +-2 for the sampling of the volume's 256-entry map.
    for channel in _rgb(view, image, np.array((4.0, 4.0, 4.0))):
        assert channel == pytest.approx(0.5 * 255, abs=2)
