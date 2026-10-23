from collections.abc import Callable

import numpy as np
import pygfx as gfx
import pytest

from larmor.types import Volume
from larmor.viewer import (
    Basis,
    Category,
    Colouring,
    GlyphScaling,
    OptionName,
    Viewer,
    ViewKind,
)
from larmor.viewer.render import fod_adapter
from larmor.viewer.render.colour_space import physical_colours
from larmor.viewer.render.fod_adapter import REBUILDING, FodAdapter
from larmor.viewer.render.glyphs import Glyphs, slice_glyphs, sphere
from larmor.viewer.render.view import ViewRenderer
from larmor.viewer.render.volume_3d import PLANES

START = (3.0, 3.0, 3.0)


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    # The (voxel axis, index) of every slice whose glyphs are made.
    made: list[tuple[int, int]] = []
    real: Callable[..., Glyphs] = slice_glyphs

    def record(
        volume: Volume, axis: int, index: int, *args: object, **kw: object
    ) -> Glyphs:
        made.append((axis, index))
        return real(volume, axis, index, *args, **kw)

    monkeypatch.setattr(fod_adapter, "slice_glyphs", record)
    return made


@pytest.fixture
def viewer() -> Viewer:
    viewer = Viewer()
    data = np.zeros((8, 8, 8, 6), np.float32)
    data[..., 0] = 1.0
    item = viewer.add(Volume(data, np.eye(4)), Category.FODS)
    viewer.set(item.id, OptionName.DETAIL, 2)
    viewer.move_cursor(START)
    return viewer


@pytest.fixture
def adapter(
    viewer: Viewer,
    views: dict[ViewKind, ViewRenderer],
    builds: list[tuple[int, int]],
) -> FodAdapter:
    (item,) = viewer.scene.items
    adapter = FodAdapter(item, views, viewer, 0.0)
    viewer.scene.subscribe(adapter.on_event)
    assert sorted(builds) == [(0, 3), (1, 3), (2, 3)]
    builds.clear()
    return adapter


@pytest.mark.parametrize("world_axis", [0, 1, 2])
def test_a_cursor_move_rebuilds_only_the_view_whose_slice_changed(
    world_axis: int,
    viewer: Viewer,
    adapter: FodAdapter,
    builds: list[tuple[int, int]],
) -> None:
    moved = list(START)
    moved[world_axis] += 1.0
    viewer.move_cursor((moved[0], moved[1], moved[2]))
    assert builds == [(world_axis, 4)]
    # Within the same voxels: no slice changed.
    moved[world_axis] += 0.2
    viewer.move_cursor((moved[0], moved[1], moved[2]))
    assert builds == [(world_axis, 4)]


_MATERIAL: dict[OptionName, tuple[object, str, object]] = {
    OptionName.COLOURING: (Colouring.SINGLE, "color_mode", "uniform"),
    OptionName.COLOUR: ("#ff0000", "color", gfx.Color("#ff0000")),
    OptionName.OPACITY_2D: (0.5, "opacity", 0.5),
}


@pytest.mark.parametrize("name", list(_MATERIAL))
def test_a_material_option_rebuilds_no_glyphs(
    name: OptionName,
    viewer: Viewer,
    adapter: FodAdapter,
    builds: list[tuple[int, int]],
) -> None:
    value, attribute, shown = _MATERIAL[name]
    (item,) = viewer.scene.items
    if name == OptionName.COLOUR:  # used only in single colouring
        viewer.set(item.id, OptionName.COLOURING, Colouring.SINGLE)
    viewer.set(item.id, name, value)
    assert builds == []
    for mesh in adapter.meshes.values():
        assert getattr(mesh.material, attribute) == shown


def test_a_hidden_fod_builds_nothing_until_shown(
    viewer: Viewer, adapter: FodAdapter, builds: list[tuple[int, int]]
) -> None:
    (item,) = viewer.scene.items
    viewer.set(item.id, OptionName.VISIBLE, False)
    viewer.move_cursor((4.0, 5.0, 6.0))
    viewer.set(item.id, OptionName.SCALE, 2.0)
    assert builds == []
    assert not any(m.visible for m in adapter.meshes.values())
    viewer.set(item.id, OptionName.VISIBLE, True)
    assert sorted(builds) == [(0, 4), (1, 5), (2, 6)]
    assert all(m.visible for m in adapter.meshes.values())
    assert set(adapter.meshes) == set(PLANES)


_REBUILT: dict[OptionName, object] = {
    OptionName.BASIS: Basis.DESCOTEAUX,
    OptionName.SCALE: 2.0,
    OptionName.SCALING: GlyphScaling.PER_VOXEL,
    OptionName.THRESHOLD: 0.5,
    OptionName.ANISOTROPY: 0.5,
    OptionName.DETAIL: 3,
}


def test_every_option_that_shapes_the_glyphs_is_covered() -> None:
    assert set(_REBUILT) == set(REBUILDING)


@pytest.mark.parametrize("name", list(_REBUILT))
def test_an_option_that_shapes_the_glyphs_rebuilds_every_view(
    name: OptionName,
    viewer: Viewer,
    adapter: FodAdapter,
    builds: list[tuple[int, int]],
) -> None:
    (item,) = viewer.scene.items
    viewer.set(item.id, name, _REBUILT[name])
    assert sorted(builds) == [(0, 3), (1, 3), (2, 3)]


def test_an_oblique_fods_glyphs_lie_on_the_cursors_plane(
    views: dict[ViewKind, ViewRenderer],
) -> None:
    # 2 mm voxels tilted 10 degrees about x: the axial slice of voxels is not flat.
    t = np.radians(10)
    rotation = np.array(
        [[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]]
    )
    affine = np.eye(4)
    affine[:3, :3] = rotation * 2.0
    data = np.zeros((16, 16, 16, 6), np.float32)
    data[..., 0] = 1.0  # round glyphs, centred on their vertices' mean
    viewer = Viewer()
    item = viewer.add(Volume(data, affine), Category.FODS)
    viewer.set(item.id, OptionName.DETAIL, 2)
    cursor = affine[:3, :3] @ [7.5, 7.5, 7.5]
    viewer.move_cursor((float(cursor[0]), float(cursor[1]), float(cursor[2])))
    adapter = FodAdapter(item, views, viewer, 0.0)
    for kind, axis in PLANES.items():
        mesh = adapter.meshes[kind]
        n = adapter._counts[kind]
        centres = mesh.geometry.positions.data.reshape(n, -1, 3).mean(axis=1)
        np.testing.assert_allclose(centres[:, axis], cursor[axis], atol=1e-4)


def test_the_glyphs_colours_are_given_to_their_meshes_as_linear_light(
    adapter: FodAdapter,
) -> None:
    # A mesh's vertex colours are linear: the glyphs' sRGB colours, converted.
    # One glyph of the detail the fixture sets: each direction at full brightness,
    # mixed with 30% white.
    rgb = np.abs(sphere(2).vertices)
    srgb = 0.7 * rgb / rgb.max(axis=1, keepdims=True) + 0.3
    for mesh in adapter.meshes.values():
        colours = np.asarray(mesh.geometry.colors.data)[: len(srgb)]
        np.testing.assert_allclose(colours, physical_colours(srgb), atol=1e-6)
