"""The texture data of the volume adapter, which needs no GPU."""

from typing import Literal

import numpy as np
import pygfx as gfx
import pytest
from render_helpers import (
    AXIS,
    VIEWS_2D,
    _table,
    _thresholded,
)

from larmor.types import Volume
from larmor.viewer import (
    Category,
    Colormap,
    ViewKind,
)
from larmor.viewer.model.options import (
    COLORMAPS,
)
from larmor.viewer.model.registry import REGISTRY
from larmor.viewer.render.adapter import Adapter
from larmor.viewer.render.geometry import (
    colormap_colours,
)
from larmor.viewer.render.scene_renderer import ADAPTERS
from larmor.viewer.render.volume_adapter import (
    COLORMAP_MAPS,
    _plane,
    _texture_data,
    _threshold_cmap,
    _volume_at,
)

pytestmark = pytest.mark.usefixtures("fresh_slab_budget")


@pytest.fixture
def view_size() -> int:
    return 96


@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.float64,
        np.int64,
        np.uint64,
        np.float32,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
    ],
)
def test_images_go_up_as_float32_so_the_gpu_interpolates_them(dtype: type) -> None:
    data: np.ndarray = (np.arange(24) % 3).reshape(2, 3, 4).astype(dtype)
    texture = _texture_data(data)
    assert texture.dtype == np.float32
    assert texture.shape == (4, 3, 2)
    assert texture.flags.c_contiguous
    assert np.array_equal(texture.T, data.astype(np.float32))


def test_a_fortran_ordered_volume_is_uploaded_without_a_copy() -> None:
    data = np.asfortranarray(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    texture = _texture_data(Volume(data, np.eye(4)).data)
    assert np.shares_memory(texture, data)
    assert texture.flags.c_contiguous
    assert np.array_equal(texture.T, data)


def test_a_c_ordered_volume_is_copied_into_a_contiguous_texture() -> None:
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    texture = _texture_data(data)
    assert not np.shares_memory(texture, data)
    assert texture.flags.c_contiguous
    assert np.array_equal(texture.T, data)


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("order", ["C", "F"])
def test_a_4d_volume_uploads_the_volume_at_its_index(
    order: Literal["C", "F"], index: int
) -> None:
    data = np.arange(72, dtype=np.float32).reshape((2, 3, 4, 3), order=order)
    texture = _texture_data(_volume_at(data, index))
    assert np.array_equal(texture.T, data[..., index])
    # Each volume of a Fortran-ordered series is itself contiguous.
    assert np.shares_memory(texture, data) == (order == "F")


def test_a_3d_volume_is_its_own_volume_at_index_0() -> None:
    data = np.zeros((2, 3, 4), np.float32)
    assert _volume_at(data, 0) is data


@pytest.mark.parametrize("kind", VIEWS_2D)
def test_a_slice_plane_passes_through_the_cursor_across_its_axis(
    kind: ViewKind,
) -> None:
    cursor = (1.5, -2.0, 7.25)
    a, b, c, d = _plane(kind, cursor)
    normal = np.array([a, b, c])
    expected = np.zeros(3)
    expected[AXIS[kind]] = 1.0
    assert np.array_equal(np.abs(normal), expected)
    # pygfx keeps the points where a x + b y + c z + d = 0.
    on = np.array(cursor)
    off = on.copy()
    off[AXIS[kind]] += 1.0
    assert normal @ on + d == 0.0
    assert normal @ off + d != 0.0
    for other in range(3):
        if other != AXIS[kind]:
            moved = on.copy()
            moved[other] += 5.0
            assert normal @ moved + d == 0.0


def test_every_category_of_images_has_an_adapter() -> None:
    images = {c for c, t in REGISTRY.items() if t.data_type is Volume}
    assert images == {
        Category.VOLUMES,
        Category.OVERLAYS,
        Category.FODS,
        Category.PEAKS,
    }
    assert images <= set(ADAPTERS)
    assert set(ADAPTERS) <= set(REGISTRY)
    assert all(issubclass(a, Adapter) for a in ADAPTERS.values())


# gray_r and blue-dark-red are the viewer's own.
OWN_MAPS = (Colormap.GRAY_R, Colormap.BLUE_RED)


@pytest.mark.parametrize("name", [c for c in COLORMAPS if c not in OWN_MAPS])
def test_every_colormap_exists_in_pygfx(name: Colormap) -> None:
    assert isinstance(getattr(gfx.cm, name), gfx.TextureMap)


def test_the_colormaps_cover_the_options() -> None:
    assert set(COLORMAP_MAPS) == set(COLORMAPS)


@pytest.mark.parametrize("name", COLORMAPS)
def test_every_colormap_is_opaque(name: Colormap) -> None:
    # An image slice must hide what lies behind it, even at its dark end.
    alpha = np.asarray(COLORMAP_MAPS[name].texture.data)[..., 3]
    assert np.all(alpha == 1)


def test_blue_red_is_dark_at_zero_of_a_symmetric_range_blue_below_red_above() -> None:
    table = np.asarray(COLORMAP_MAPS[Colormap.BLUE_RED].texture.data)
    below, zero, above = colormap_colours(np.array([-0.5, 0.0, 0.5]), table, -1, 1)
    assert zero.max() < 0.2
    assert below[2] > 2 * max(below[:2]) and above[0] > 2 * max(above[1:])


# Thresholds below, inside and above the colour range 0 to 1.
_THRESHOLDS = [-0.5, 0.0, 0.5, 1.0, 1.5]


@pytest.mark.parametrize("threshold", _THRESHOLDS)
@pytest.mark.parametrize("name", COLORMAPS)
def test_a_thresholded_colormap_leaves_its_source_alone(
    name: Colormap, threshold: float
) -> None:
    before = _table(COLORMAP_MAPS[name]).copy()
    _threshold_cmap(COLORMAP_MAPS[name], 0.0, 1.0, threshold, (-1.0, 2.0))
    assert np.array_equal(_table(COLORMAP_MAPS[name]), before)


# A mask's zeros, at its default threshold of 0, are hidden: values at the threshold
# fall in the first entry.
@pytest.mark.parametrize("name", COLORMAPS)
def test_a_threshold_at_the_data_minimum_hides_only_the_first_entry(
    name: Colormap,
) -> None:
    cmap, limits = _threshold_cmap(COLORMAP_MAPS[name], 0.2, 0.9, 0.2, (0.2, 1.0))
    assert np.array_equal(_table(cmap), _thresholded(name, 1))
    assert limits == (0.2, 0.9)


# Data below the minimum is clamped onto the first entry, so that one is hidden.
@pytest.mark.parametrize("name", [Colormap.GRAY, Colormap.HOT])
def test_a_threshold_at_the_minimum_hides_the_data_below_it(name: Colormap) -> None:
    cmap, _ = _threshold_cmap(COLORMAP_MAPS[name], 0.2, 0.9, 0.2, (0.0, 1.0))
    assert np.array_equal(_table(cmap), _thresholded(name, 1))


# Inside the range 0 to 1 the table is the colormap's, with the entries whose centre
# (k + 0.5) / 256 is below the threshold hidden. The top entry holds the maximum, so
# it stays shown at a threshold of 1 if the data reaches it.
@pytest.mark.parametrize(
    ("threshold", "data_max", "hidden"),
    [
        (0.0, 1.0, 1),
        (0.25, 1.0, 64),
        (0.3, 1.0, 77),
        (0.5, 1.0, 128),
        (1.0, 1.0, 255),
        (1.0, 0.9, 256),
    ],
)
@pytest.mark.parametrize("name", [Colormap.GRAY, Colormap.HOT])
def test_a_threshold_in_the_range_hides_the_entries_below_it(
    name: Colormap, threshold: float, data_max: float, hidden: int
) -> None:
    cmap, limits = _threshold_cmap(
        COLORMAP_MAPS[name], 0.0, 1.0, threshold, (0, data_max)
    )
    assert np.array_equal(_table(cmap), _thresholded(name, hidden))
    assert limits == (0.0, 1.0)


def _shown_as(
    value: float,
    minimum: float,
    maximum: float,
    threshold: float,
    data_range: tuple[float, float],
    name: Colormap = Colormap.GRAY,
) -> np.ndarray:
    # As the shader samples it: the value's place in the limits, clamped to the ends.
    cmap, (lo, hi) = _threshold_cmap(
        COLORMAP_MAPS[name], minimum, maximum, threshold, data_range
    )
    table = _table(cmap)
    n = len(table)
    entry: np.ndarray = table[int(np.clip((value - lo) / (hi - lo) * n, 0, n - 1))]
    return entry


def _source(name: Colormap, k: int) -> np.ndarray:
    entry: np.ndarray = _table(COLORMAP_MAPS[name])[k]
    return entry


# FA-like: coloured from 0.3 to 1, with the threshold at 0.2 below that range.
@pytest.mark.parametrize("name", [Colormap.GRAY, Colormap.HOT])
def test_a_threshold_below_the_range_hides_the_data_below_it(name: Colormap) -> None:
    for value in (0.0, 0.1, 0.19):
        assert _shown_as(value, 0.3, 1.0, 0.2, (0.0, 1.0), name)[3] == 0.0, value
    for value in (0.21, 0.25, 0.3):
        colour = _shown_as(value, 0.3, 1.0, 0.2, (0.0, 1.0), name)
        assert np.array_equal(colour, _source(name, 0)), value
    assert np.array_equal(
        _shown_as(1.0, 0.3, 1.0, 0.2, (0.0, 1.0), name), _source(name, -1)
    )


def test_a_threshold_below_the_range_keeps_the_values_between_in_the_colormap() -> None:
    # Grey is linear: 0.5 sits 0.2 / 0.7 of the way up, give or take an entry.
    colour = _shown_as(0.5, 0.3, 1.0, 0.2, (0.0, 1.0))
    assert colour[:3] == pytest.approx([0.2 / 0.7] * 3, abs=2 / 255)
    assert colour[3] == 1.0


def test_a_threshold_below_the_range_and_the_data_shows_the_lowest_value() -> None:
    assert _shown_as(0.25, 0.3, 1.0, 0.2, (0.25, 1.0))[3] == 1.0


@pytest.mark.parametrize("name", [Colormap.GRAY, Colormap.HOT])
def test_a_threshold_above_the_range_shows_the_values_over_it_in_the_top_colour(
    name: Colormap,
) -> None:
    for value in (0.0, 0.5, 1.0, 1.5, 1.99):
        assert _shown_as(value, 0.0, 1.0, 2.0, (0.0, 3.0), name)[3] == 0.0, value
    for value in (2.0, 2.5, 3.0):
        colour = _shown_as(value, 0.0, 1.0, 2.0, (0.0, 3.0), name)
        assert np.array_equal(colour, _source(name, -1)), value


def test_a_threshold_above_the_data_hides_everything() -> None:
    cmap, _ = _threshold_cmap(COLORMAP_MAPS[Colormap.GRAY], 0.0, 1.0, 2.0, (0.0, 1.5))
    assert np.all(_table(cmap)[:, 3] == 0.0)


@pytest.mark.parametrize(
    ("minimum", "maximum", "threshold", "limits"),
    [
        (0.3, 1.0, 0.2, (0.2, 1.0)),
        (0.3, 1.0, 0.5, (0.3, 1.0)),
        (0.0, 1.0, 2.0, (0.0, 2.0)),
    ],
)
def test_the_limits_are_the_range_widened_to_the_threshold(
    minimum: float, maximum: float, threshold: float, limits: tuple[float, float]
) -> None:
    _, shown = _threshold_cmap(
        COLORMAP_MAPS[Colormap.GRAY], minimum, maximum, threshold, (0.0, 3.0)
    )
    assert shown == limits


# A constant overlay of 3s: its minimum and maximum are both 3.
@pytest.mark.parametrize(("threshold", "shown"), [(2.0, True), (4.0, False)])
@pytest.mark.parametrize("name", [Colormap.GRAY, Colormap.HOT])
def test_a_constant_overlay_shows_only_if_it_reaches_the_threshold(
    name: Colormap, threshold: float, shown: bool
) -> None:
    colour = _shown_as(3.0, 3.0, 3.0, threshold, (3.0, 3.0), name)
    if shown:
        assert np.array_equal(colour, _source(name, 0))
    else:
        assert colour[3] == 0.0


def test_a_constant_overlay_at_its_threshold_is_hidden() -> None:
    cmap, limits = _threshold_cmap(
        COLORMAP_MAPS[Colormap.HOT], 3.0, 3.0, 3.0, (3.0, 3.0)
    )
    assert limits == (3.0, 3.0)
    # The shader puts every value of a zero-width range on the first entry.
    assert _table(cmap)[0, 3] == 0.0
