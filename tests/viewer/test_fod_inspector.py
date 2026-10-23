from collections.abc import Callable, Iterator

import numpy as np
import pytest

from larmor.types import Volume
from larmor.viewer import (
    Basis,
    Category,
    Colouring,
    FodInspector,
    ItemId,
    OptionName,
    Viewer,
)
from larmor.viewer.model.options import Colour
from larmor.viewer.model.viewer import ZOOM_RANGE
from larmor.viewer.render import fod_inspector
from larmor.viewer.render.colour_space import physical_colours
from larmor.viewer.render.fod_inspector import NO_FOD
from larmor.viewer.render.glyphs import Glyphs, voxel_glyph

# 2 mm voxels: voxel (i, j, k) is centred at 2 (i, j, k) mm.
AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
EMPTY = (3, 3, 3)


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple[int, int, int], str]]:
    made: list[tuple[tuple[int, int, int], str]] = []
    real: Callable[..., Glyphs] = voxel_glyph

    def record(
        volume: Volume, voxel: tuple[int, int, int], basis: str, *args: object
    ) -> Glyphs:
        made.append((voxel, basis))
        return real(volume, voxel, basis, *args)

    monkeypatch.setattr(fod_inspector, "voxel_glyph", record)
    return made


@pytest.fixture
def viewer() -> Viewer:
    viewer = Viewer()
    data = np.zeros((4, 4, 4, 6), np.float32)
    data[..., 0] = 1.0
    data[..., 3] = 0.4  # not round, so the basis changes the shape
    data[EMPTY] = 0.0
    viewer.add(Volume(data, AFFINE), Category.FODS, name="F")
    viewer.move_cursor((2.0, 2.0, 2.0))
    return viewer


@pytest.fixture
def inspector(
    gpu: None, viewer: Viewer, builds: list[tuple[tuple[int, int, int], str]]
) -> Iterator[FodInspector]:
    with FodInspector(viewer, viewer.scene.items[0].id) as inspector:
        assert builds == [((1, 1, 1), "tournier")]
        builds.clear()
        yield inspector


@pytest.mark.parametrize(
    ("position", "rebuilt"),
    [((2.9, 1.2, 2.0), []), ((4.0, 2.0, 2.0), [((2, 1, 1), "tournier")])],
)
def test_the_glyph_is_made_again_only_when_the_cursor_changes_voxel(
    position: tuple[float, float, float],
    rebuilt: list[tuple[tuple[int, int, int], str]],
    viewer: Viewer,
    inspector: FodInspector,
    builds: list[tuple[tuple[int, int, int], str]],
) -> None:
    viewer.move_cursor(position)
    assert builds == rebuilt
    assert inspector.voxel == (rebuilt[0][0] if rebuilt else (1, 1, 1))
    assert "voxel " + ", ".join(map(str, inspector.voxel)) in inspector.label_text


@pytest.mark.parametrize(
    ("name", "value", "rebuilt"),
    [
        (OptionName.BASIS, Basis.DESCOTEAUX, True),
        (OptionName.COLOURING, Colouring.SINGLE, False),
        (OptionName.COLOUR, Colour("#ff0000"), False),
    ],
)
def test_the_basis_makes_the_glyph_again_and_the_colours_do_not(
    name: OptionName,
    value: object,
    rebuilt: bool,
    viewer: Viewer,
    inspector: FodInspector,
    builds: list[tuple[tuple[int, int, int], str]],
) -> None:
    viewer.set(inspector.item.id, name, value)
    assert builds == ([((1, 1, 1), "descoteaux")] if rebuilt else [])


def test_single_colouring_draws_the_glyph_in_the_items_colour(
    viewer: Viewer, inspector: FodInspector
) -> None:
    viewer.set(inspector.item.id, OptionName.COLOUR, Colour("#ff0000"))
    viewer.set(inspector.item.id, OptionName.COLOURING, Colouring.SINGLE)
    material = inspector.mesh.material
    assert material.color_mode == "uniform"
    assert material.color.hex == "#ff0000"


@pytest.mark.parametrize("position", [(-2.0, 2.0, 2.0), (6.0, 6.0, 6.0)])
def test_there_is_no_fod_outside_the_grid_or_in_an_all_zero_voxel(
    position: tuple[float, float, float], viewer: Viewer, inspector: FodInspector
) -> None:
    assert inspector.mesh.visible
    viewer.move_cursor(position)
    assert not inspector.mesh.visible
    assert NO_FOD in inspector.label_text


def test_the_inspector_draws_the_glyph_off_screen(inspector: FodInspector) -> None:
    image = inspector.render()
    assert image.shape == (*fod_inspector.DEFAULT_SIZE[::-1], 3)
    background = image[0, 0]
    assert np.any(np.abs(image.astype(int) - background).sum(axis=2) > 30)


def test_removing_the_item_closes_the_inspector_and_stops_it_following(
    gpu: None, viewer: Viewer, builds: list[tuple[tuple[int, int, int], str]]
) -> None:
    before = len(viewer.scene._subscribers)
    item_id: ItemId = viewer.scene.items[0].id
    removed: list[bool] = []
    inspector = FodInspector(viewer, item_id, removed=lambda: removed.append(True))
    assert len(viewer.scene._subscribers) == before + 1
    viewer.remove(item_id)
    assert removed == [True]
    assert len(viewer.scene._subscribers) == before
    assert inspector.canvas.get_closed()


@pytest.mark.parametrize(("factor", "limit"), [(1e6, 1), (1e-6, 0)])
def test_zoom_stays_within_the_views_zoom_range(
    factor: float, limit: int, inspector: FodInspector
) -> None:
    inspector.zoom(factor)
    assert inspector.camera.zoom == ZOOM_RANGE[limit]
    inspector.reset_camera()
    assert inspector.camera.zoom == 1.0


def test_the_glyphs_colours_are_given_to_its_mesh_as_linear_light(
    viewer: Viewer, inspector: FodInspector
) -> None:
    # A mesh's vertex colours are linear: the glyph's sRGB colours, converted.
    volume = viewer.scene.items[0].data
    assert isinstance(volume, Volume)
    glyph = voxel_glyph(volume, (1, 1, 1), "tournier", fod_inspector.DETAIL)
    # Each direction at full brightness, mixed with 30% white.
    rgb = np.abs(glyph.positions)
    srgb = 0.7 * rgb / rgb.max(axis=1, keepdims=True) + 0.3
    colours = np.asarray(inspector.mesh.geometry.colors.data)
    np.testing.assert_allclose(colours, physical_colours(srgb), atol=1e-5)
    assert not np.allclose(colours, srgb, atol=1e-3)
