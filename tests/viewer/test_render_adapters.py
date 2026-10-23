"""Adapters and views, off screen."""

import numpy as np
import pygfx as gfx
import pytest
from render_helpers import (
    VIEWS_2D,
    _adapter,
    _objects,
    _one_grid,
    _same_map_on_every_slice,
    _series,
    _table,
    _thresholded,
    _volume,
)

from larmor.types import Volume
from larmor.viewer import (
    Category,
    Colormap,
    OptionName,
    Viewer,
    ViewKind,
)
from larmor.viewer.model.item import VIEW_KINDS
from larmor.viewer.model.options import (
    OverlayOptions,
    Scope,
    option_fields,
)
from larmor.viewer.render.scene_renderer import SceneRenderer
from larmor.viewer.render.view import ViewRenderer
from larmor.viewer.render.volume_adapter import (
    COLORMAP_MAPS,
    VolumeAdapter,
    _plane,
)

pytestmark = pytest.mark.usefixtures("fresh_slab_budget")


@pytest.fixture
def view_size() -> int:
    return 96


def test_a_volume_is_one_texture_in_a_slice_per_2d_view(
    views: dict[ViewKind, ViewRenderer],
) -> None:
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = [10.0, -20.0, 5.0]
    volume = _volume(affine=affine)
    viewer = Viewer()
    adapter = _adapter(viewer, views, viewer.add(volume, Category.VOLUMES))
    assert list(adapter.slices) == VIEWS_2D
    textures = {id(obj.geometry.grid) for obj in adapter.slices.values()}
    assert len(textures) == 1
    texture = adapter.slices[ViewKind.AXIAL].geometry.grid
    assert np.shares_memory(texture.data, volume.data)
    for obj in adapter.slices.values():
        assert isinstance(obj, gfx.Volume)
        assert isinstance(obj.material, gfx.VolumeSliceMaterial)
        assert np.allclose(obj.local.matrix, affine)
    three_d = adapter.three_d
    assert three_d is not None and three_d.rendering is not None
    assert all(o.geometry.grid is texture for o in three_d.slices.values())
    assert three_d.rendering.geometry.grid is texture
    assert _objects(views) == {
        **{k: [adapter.slices[k]] for k in VIEWS_2D},
        ViewKind.THREE_D: [*three_d.slices.values(), three_d.rendering],
    }


def test_the_slices_start_at_the_cursor(views: dict[ViewKind, ViewRenderer]) -> None:
    viewer = Viewer()
    item = viewer.add(_volume(), Category.OVERLAYS)
    viewer.move_cursor((1.0, 2.0, 3.0))
    adapter = _adapter(viewer, views, item)
    for kind, obj in adapter.slices.items():
        assert np.array_equal(obj.material.plane, _plane(kind, (1.0, 2.0, 3.0)))


def test_slices_made_on_the_first_image_follow_the_centred_cursor(
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer = Viewer()
    renderer = SceneRenderer(viewer, views)
    item = viewer.add(_volume(), Category.VOLUMES)
    cursor = viewer.scene.view.cursor
    assert cursor == (4.0, 4.0, 4.0)
    adapter = renderer.adapters[item.id]
    assert isinstance(adapter, VolumeAdapter)
    for kind, obj in adapter.slices.items():
        assert np.array_equal(obj.material.plane, _plane(kind, cursor))


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
def test_the_slices_start_with_the_item_options(
    category: Category, views: dict[ViewKind, ViewRenderer]
) -> None:
    viewer = Viewer()
    adapter = _adapter(viewer, views, viewer.add(_volume(2.0, 10.0), category))
    for obj in adapter.slices.values():
        assert obj.visible
        assert obj.material.opacity == 1.0
        # The stored range 0 to 5, whatever the scaling.
        assert obj.material.clim == pytest.approx((0.0, 5.0))
        if category == Category.VOLUMES:
            assert obj.material.map is COLORMAP_MAPS[Colormap.GRAY]
        else:
            # The threshold starts at the minimum: only the minimum is hidden.
            assert np.array_equal(
                _table(obj.material.map), _thresholded(Colormap.GRAY, 1)
            )
        assert obj.material.interpolation == "linear"


@pytest.mark.parametrize(
    ("slope", "intercept", OptionName.MINIMUM, OptionName.MAXIMUM, "clim"),
    [
        (1.0, 0.0, 1.0, 4.0, (1.0, 4.0)),
        (2.0, 10.0, 14.0, 30.0, (2.0, 10.0)),
        # Reversed in stored units, so the scaled minimum is still the dark end.
        (-0.5, 1.0, -1.5, 2.0, (5.0, -2.0)),
    ],
)
def test_minimum_and_maximum_become_the_stored_colour_limits(
    slope: float,
    intercept: float,
    minimum: float,
    maximum: float,
    clim: tuple[float, float],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer = Viewer()
    item = viewer.add(_volume(slope, intercept), Category.VOLUMES)
    adapter = _adapter(viewer, views, item)
    viewer.set(item.id, OptionName.MINIMUM, minimum)
    viewer.set(item.id, OptionName.MAXIMUM, maximum)
    for obj in adapter.slices.values():
        assert obj.material.clim == pytest.approx(clim)


# The scaled data spans 10 to 20 (stored 0 to 5), or 1 to -1.5 with a negative slope.
@pytest.mark.parametrize(
    ("slope", "intercept", "minimum", "maximum", "threshold", "clim"),
    [
        (2.0, 10.0, 12.0, 18.0, 16.0, (1.0, 4.0)),
        (2.0, 10.0, 14.0, 18.0, 10.0, (0.0, 4.0)),
        (2.0, 10.0, 14.0, 18.0, 22.0, (2.0, 6.0)),
        (-0.5, 1.0, -1.0, 0.5, -1.5, (5.0, 1.0)),
    ],
)
def test_an_overlay_colour_limits_take_in_its_threshold_in_stored_units(
    slope: float,
    intercept: float,
    minimum: float,
    maximum: float,
    threshold: float,
    clim: tuple[float, float],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer = Viewer()
    item = viewer.add(_volume(slope, intercept), Category.OVERLAYS)
    adapter = _adapter(viewer, views, item)
    viewer.set(item.id, OptionName.MINIMUM, minimum)
    viewer.set(item.id, OptionName.MAXIMUM, maximum)
    viewer.set(item.id, OptionName.THRESHOLD, threshold)
    for obj in adapter.slices.values():
        assert obj.material.clim == pytest.approx(clim)


_OPTIONS: list[tuple[OptionName, object, str, object]] = [
    (OptionName.VISIBLE, False, "visible", False),
    (OptionName.OPACITY_2D, 0.4, "opacity", 0.4),
    (OptionName.COLORMAP, Colormap.HOT, "map", COLORMAP_MAPS[Colormap.HOT]),
    (OptionName.COLORMAP, Colormap.VIRIDIS, "map", COLORMAP_MAPS[Colormap.VIRIDIS]),
    (OptionName.INTERPOLATE, False, "interpolation", "nearest"),
]


def _shown(obj: gfx.Volume, attribute: str) -> object:
    return obj.visible if attribute == "visible" else getattr(obj.material, attribute)


@pytest.mark.parametrize(("name", "value", "attribute", "expected"), _OPTIONS)
def test_an_option_change_shows_on_every_slice_and_redraws_its_views(
    name: OptionName,
    value: object,
    attribute: str,
    expected: object,
    views: dict[ViewKind, ViewRenderer],
    draws: dict[ViewKind, int],
) -> None:
    viewer = Viewer()
    item = viewer.add(_volume(), Category.OVERLAYS)
    adapter = _adapter(viewer, views, item)
    draws.update(dict.fromkeys(draws, 0))
    viewer.set(item.id, name, value)
    for obj in adapter.slices.values():
        shown = _shown(obj, attribute)
        if attribute == "map":
            # An overlay's map is a copy of the colormap; at the starting threshold
            # every entry but the first, the minimum's, is as in the colormap.
            assert isinstance(shown, gfx.TextureMap)
            assert isinstance(expected, gfx.TextureMap)
            assert np.array_equal(_table(shown)[1:], _table(expected)[1:])
            assert _table(shown)[0, 3] == 0.0
        elif isinstance(expected, float):
            # The material keeps its opacity as a float32.
            assert shown == pytest.approx(expected, rel=1e-6)
        else:
            assert shown == expected
    _, info = option_fields(OverlayOptions)[name]
    assert draws == {
        k: int(k != ViewKind.THREE_D or info.scope != Scope.TWO_D) for k in VIEW_KINDS
    }


@pytest.mark.parametrize(("name", "value", "attribute", "expected"), _OPTIONS)
def test_another_item_option_change_leaves_the_slices_alone(
    name: OptionName,
    value: object,
    attribute: str,
    expected: object,
    views: dict[ViewKind, ViewRenderer],
    draws: dict[ViewKind, int],
) -> None:
    viewer = Viewer()
    adapter = _adapter(viewer, views, viewer.add(_volume(), Category.OVERLAYS))
    other = viewer.add(_volume(), Category.OVERLAYS)
    before = {k: _shown(o, attribute) for k, o in adapter.slices.items()}
    draws.update(dict.fromkeys(draws, 0))
    viewer.set(other.id, name, value)
    assert {k: _shown(o, attribute) for k, o in adapter.slices.items()} == before
    assert sum(draws.values()) == 0


def test_a_volume_slice_map_is_the_colormap_itself(
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer = Viewer()
    item = viewer.add(_volume(), Category.VOLUMES)
    adapter = _adapter(viewer, views, item)
    viewer.set(item.id, OptionName.COLORMAP, Colormap.HOT)
    assert all(
        o.material.map is COLORMAP_MAPS[Colormap.HOT] for o in adapter.slices.values()
    )


# _volume() spans 0 to 5; each case starts from a threshold of 2.5 (half the range).
_THRESHOLD_CHANGES: list[tuple[OptionName, object, Colormap, int]] = [
    (OptionName.THRESHOLD, 1.25, Colormap.GRAY, 64),
    # The top entry holds the maximum, which the data reaches.
    (OptionName.THRESHOLD, 5.0, Colormap.GRAY, 255),
    # The entries whose centre is below 1.25 / 3.75 of the range: ceil(85.3 - 0.5).
    (OptionName.MINIMUM, 1.25, Colormap.GRAY, 85),
    (OptionName.MAXIMUM, 10.0, Colormap.GRAY, 64),
    (OptionName.COLORMAP, Colormap.HOT, Colormap.HOT, 128),
]


@pytest.mark.parametrize(("name", "value", "colormap", "hidden"), _THRESHOLD_CHANGES)
def test_an_overlay_map_follows_its_threshold_range_and_colormap(
    name: OptionName,
    value: object,
    colormap: Colormap,
    hidden: int,
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer = Viewer()
    item = viewer.add(_volume(), Category.OVERLAYS)
    adapter = _adapter(viewer, views, item)
    viewer.set(item.id, OptionName.THRESHOLD, 2.5)
    assert np.array_equal(
        _same_map_on_every_slice(adapter), _thresholded(Colormap.GRAY, 128)
    )
    viewer.set(item.id, name, value)
    assert np.array_equal(
        _same_map_on_every_slice(adapter), _thresholded(colormap, hidden)
    )


def _entry_for(obj: gfx.Volume, stored: float) -> np.ndarray:
    # The entry the shader takes for a stored value: its place in the colour limits.
    lo, hi = obj.material.clim
    table = _table(obj.material.map)
    n = len(table)
    entry: np.ndarray = table[int(np.clip((stored - lo) / (hi - lo) * n, 0, n - 1))]
    return entry


@pytest.mark.parametrize(("threshold", "shown"), [(2.0, True), (4.0, False)])
def test_a_constant_overlay_shows_only_if_it_reaches_its_threshold(
    threshold: float, shown: bool, views: dict[ViewKind, ViewRenderer]
) -> None:
    viewer = Viewer()
    item = viewer.add(Volume(np.full((4, 4, 4), 3.0), np.eye(4)), Category.OVERLAYS)
    assert isinstance(item.options, OverlayOptions)
    assert item.options.minimum == item.options.maximum == 3.0
    adapter = _adapter(viewer, views, item)
    viewer.set(item.id, OptionName.THRESHOLD, threshold)
    for obj in adapter.slices.values():
        colour = _entry_for(obj, 3.0)
        if shown:
            assert np.array_equal(colour, _table(COLORMAP_MAPS[Colormap.GRAY])[0])
        else:
            assert colour[3] == 0.0


def _planes(adapter: VolumeAdapter) -> dict[ViewKind, list[float]]:
    return {k: list(o.material.plane) for k, o in adapter.slices.items()}


@pytest.mark.parametrize(
    ("move", "changed"),
    [
        ((1.0, 0.0, 0.0), [ViewKind.SAGITTAL]),
        ((0.0, 1.0, 0.0), [ViewKind.CORONAL]),
        ((0.0, 0.0, 1.0), [ViewKind.AXIAL]),
        ((1.0, 0.0, 1.0), [ViewKind.AXIAL, ViewKind.SAGITTAL]),
        ((1.0, 1.0, 1.0), [ViewKind.AXIAL, ViewKind.CORONAL, ViewKind.SAGITTAL]),
    ],
)
def test_a_cursor_move_updates_only_the_slices_whose_coordinate_changed(
    move: tuple[float, float, float],
    changed: list[ViewKind],
    views: dict[ViewKind, ViewRenderer],
    draws: dict[ViewKind, int],
) -> None:
    viewer = Viewer()
    adapter = _adapter(viewer, views, viewer.add(_volume(), Category.OVERLAYS))
    start = viewer.scene.view.cursor
    before = _planes(adapter)
    draws.update(dict.fromkeys(draws, 0))
    new = (start[0] + move[0], start[1] + move[1], start[2] + move[2])
    viewer.move_cursor(new)
    after = _planes(adapter)
    for kind in VIEWS_2D:
        if kind in changed:
            assert after[kind] == list(_plane(kind, new))
        else:
            assert after[kind] == before[kind]
    assert draws == {k: int(k in changed or k == ViewKind.THREE_D) for k in VIEW_KINDS}
    three_d = adapter.three_d
    assert three_d is not None
    assert {k: list(o.material.plane) for k, o in three_d.slices.items()} == after


def test_other_view_changes_leave_the_slices_alone(
    views: dict[ViewKind, ViewRenderer], draws: dict[ViewKind, int]
) -> None:
    viewer = Viewer()
    adapter = _adapter(viewer, views, viewer.add(_volume(), Category.OVERLAYS))
    before = _planes(adapter)
    draws.update(dict.fromkeys(draws, 0))
    viewer.select_view(ViewKind.CORONAL)
    viewer.toggle_maximised(ViewKind.AXIAL)
    viewer.select(Category.OVERLAYS)
    assert _planes(adapter) == before
    assert sum(draws.values()) == 0


def test_closing_an_adapter_takes_its_slices_out_of_the_views(
    views: dict[ViewKind, ViewRenderer], draws: dict[ViewKind, int]
) -> None:
    viewer = Viewer()
    adapter = _adapter(viewer, views, viewer.add(_volume(), Category.OVERLAYS))
    draws.update(dict.fromkeys(draws, 0))
    assert _objects(views)[ViewKind.THREE_D] != []
    adapter.close()
    assert _objects(views) == dict.fromkeys(VIEW_KINDS, [])
    # The 3D view may redraw once per object it removes or replaces.
    assert all(n >= 1 for n in draws.values())


@pytest.mark.parametrize("index", [0, 2])
def test_an_image_added_with_a_volume_index_uploads_that_volume(
    index: int, views: dict[ViewKind, ViewRenderer]
) -> None:
    viewer = Viewer()
    volume = _series()
    item = viewer.add(volume, Category.OVERLAYS)
    viewer.set(item.id, OptionName.VOLUME_INDEX, index)
    grid = _one_grid(_adapter(viewer, views, item))
    assert np.array_equal(grid.data.T, volume.data[..., index])
    assert np.shares_memory(grid.data, volume.data)


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
def test_a_volume_index_change_swaps_every_slice_to_that_volume(
    category: Category,
    views: dict[ViewKind, ViewRenderer],
    draws: dict[ViewKind, int],
) -> None:
    viewer = Viewer()
    volume = _series()
    item = viewer.add(volume, category)
    adapter = _adapter(viewer, views, item)
    before = _one_grid(adapter)
    draws.update(dict.fromkeys(draws, 0))
    viewer.set(item.id, OptionName.VOLUME_INDEX, 2)
    grid = _one_grid(adapter)
    assert grid is not before
    assert np.array_equal(grid.data.T, volume.data[..., 2])
    assert np.shares_memory(grid.data, volume.data)
    # The 3D view may redraw once per object it removes or replaces.
    assert all(n >= 1 for n in draws.values())


def test_a_volume_index_change_leaves_the_other_items_texture(
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer = Viewer()
    changed = viewer.add(_series(), Category.OVERLAYS)
    other = _adapter(viewer, views, viewer.add(_series(), Category.OVERLAYS))
    _adapter(viewer, views, changed)
    before = _one_grid(other)
    viewer.set(changed.id, OptionName.VOLUME_INDEX, 1)
    assert _one_grid(other) is before


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (OptionName.OPACITY_2D, 0.5),
        (OptionName.COLORMAP, Colormap.HOT),
        (OptionName.MINIMUM, 1.0),
        (OptionName.INTERPOLATE, False),
        (OptionName.THRESHOLD, 1.5),
    ],
)
def test_another_option_change_keeps_the_texture_of_the_chosen_volume(
    name: OptionName, value: object, views: dict[ViewKind, ViewRenderer]
) -> None:
    viewer = Viewer()
    item = viewer.add(_series(), Category.OVERLAYS)
    viewer.set(item.id, OptionName.VOLUME_INDEX, 1)
    adapter = _adapter(viewer, views, item)
    before = _one_grid(adapter)
    viewer.set(item.id, name, value)
    assert _one_grid(adapter) is before
