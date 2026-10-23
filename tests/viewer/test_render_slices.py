"""Rendered slices and their alignment in world coordinates."""

import math

import numpy as np
import pytest
from render_helpers import (
    AXIS,
    BLOCK,
    VIEWS_2D,
    WHITE,
    _image,
    _rotated,
    _series,
    _volume,
)

from larmor.types import Volume
from larmor.viewer import (
    Category,
    Item,
    OptionName,
    Viewer,
    ViewKind,
)
from larmor.viewer.model.options import (
    VolumeOptions,
)
from larmor.viewer.render.scene_renderer import SceneRenderer
from larmor.viewer.render.view import ViewRenderer

pytestmark = pytest.mark.usefixtures("fresh_slab_budget")


@pytest.fixture
def view_size() -> int:
    return 96


def _grey(view: ViewRenderer, image: np.ndarray, world: np.ndarray) -> int:
    ndc = view.camera.camera_matrix @ np.array([*world, 1.0])
    x, y = ndc[:2] / ndc[3]
    height, width = image.shape[:2]
    row, col = int((1 - y) / 2 * height), int((x + 1) / 2 * width)
    assert 0 <= row < height and 0 <= col < width, "outside the view"
    return int(image[row, col, 0])


def _world(affine: np.ndarray, voxel: tuple[float, float, float]) -> np.ndarray:
    point: np.ndarray = affine[:3, :3] @ np.array(voxel) + affine[:3, 3]
    return point


OUTSIDE = (6.0, 1.0, 6.0)


@pytest.mark.parametrize("affine", [np.eye(4), _rotated()], ids=["identity", "oblique"])
@pytest.mark.parametrize("kind", VIEWS_2D)
def test_a_slice_shows_the_block_at_its_world_place(
    kind: ViewKind,
    affine: np.ndarray,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    item = viewer.add(_volume(affine=affine), Category.VOLUMES)
    viewer.set(item.id, OptionName.INTERPOLATE, False)
    block = _world(affine, BLOCK)
    viewer.move_cursor((block[0], block[1], block[2]))
    # A point of the same slice, outside the block.
    outside = _world(affine, OUTSIDE)
    outside[AXIS[kind]] = block[AXIS[kind]]
    view = views[kind]
    image = _image(view)
    assert _grey(view, image, block) == WHITE
    assert _grey(view, image, outside) == 0


def test_the_slice_follows_the_cursor_off_the_block(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    viewer.add(_volume(), Category.VOLUMES)
    viewer.move_cursor(BLOCK)
    view = views[ViewKind.AXIAL]
    assert _grey(view, _image(view), np.array(BLOCK)) == WHITE
    viewer.move_cursor((BLOCK[0], BLOCK[1], 6.0))
    assert _grey(view, _image(view), np.array((*BLOCK[:2], 6.0))) == 0


def test_a_4d_volume_shows_the_chosen_volume(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    item = viewer.add(_series(), Category.VOLUMES)
    viewer.set(item.id, OptionName.INTERPOLATE, False)
    # In the block of volume 2 (value 3, the maximum), outside that of volume 0.
    point = np.array((3.0, 5.0, 3.0))
    viewer.move_cursor((3.0, 5.0, 3.0))
    view = views[ViewKind.AXIAL]
    assert _grey(view, _image(view), point) == 0
    viewer.set(item.id, OptionName.VOLUME_INDEX, 2)
    assert _grey(view, _image(view), point) == WHITE


# Alignment in world coordinates: a bright voxel on another grid, or in an oblique
# volume, lands on the screen where the main volume's voxel at that point does.

MARK = np.array([5.0, 9.0, 7.0])


def _marked(
    shape: tuple[int, int, int], voxel: np.ndarray, affine: np.ndarray
) -> Volume:
    data = np.zeros(shape, np.float32)
    data[voxel[0], voxel[1], voxel[2]] = 1.0
    return Volume(data, affine)


def _bright_centre(view: ViewRenderer) -> np.ndarray:
    image = _image(view)[..., 0]
    rows, cols = np.nonzero(image > WHITE // 2)
    assert len(rows) > 0, "nothing bright in the view"
    centre: np.ndarray = np.array([rows.mean(), cols.mean()])
    return centre


def _predicted(view: ViewRenderer, world: np.ndarray) -> np.ndarray:
    # The pixel whose centre is at the point's place on the screen.
    ndc = view.camera.camera_matrix @ np.array([*world, 1.0])
    x, y = ndc[:2] / ndc[3]
    height, width = _image(view).shape[:2]
    pixel: np.ndarray = np.array(
        [(1 - y) / 2 * height - 0.5, (x + 1) / 2 * width - 0.5]
    )
    return pixel


def _show_only(viewer: Viewer, shown: Item, hidden: Item) -> None:
    viewer.set(hidden.id, OptionName.VISIBLE, False)
    viewer.set(shown.id, OptionName.VISIBLE, True)


def _rotation_about_z(degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _aligned(
    viewer: Viewer, main: Item, other: Item, view: ViewRenderer
) -> tuple[np.ndarray, np.ndarray]:
    for item in (main, other):
        viewer.set(item.id, OptionName.INTERPOLATE, False)
    viewer.move_cursor((MARK[0], MARK[1], MARK[2]))
    _show_only(viewer, other, main)
    seen = _bright_centre(view)
    _show_only(viewer, main, other)
    return seen, _bright_centre(view)


# At 96 pixels across 16 mm a voxel is 6 pixels; the centres agree to a pixel.
PIXEL = 1.0


@pytest.mark.parametrize("kind", VIEWS_2D)
def test_an_overlay_on_another_grid_lines_up_with_the_volume(
    kind: ViewKind,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    main = viewer.add(
        _marked((16, 16, 16), MARK.astype(int), np.eye(4)), Category.VOLUMES
    )
    # 2 mm voxels from a shifted origin; voxel (3, 4, 2) sits at the mark.
    voxel = np.array([3, 4, 2])
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = MARK - 2.0 * voxel
    overlay = viewer.add(_marked((8, 8, 8), voxel, affine), Category.OVERLAYS)
    seen, expected = _aligned(viewer, main, overlay, views[kind])
    assert seen == pytest.approx(expected, abs=PIXEL)
    assert expected == pytest.approx(_predicted(views[kind], MARK), abs=PIXEL)


@pytest.mark.parametrize("kind", VIEWS_2D)
def test_an_oblique_volume_lines_up_with_the_volume(
    kind: ViewKind,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    main = viewer.add(
        _marked((16, 16, 16), MARK.astype(int), np.eye(4)), Category.VOLUMES
    )
    # Rotated 30 degrees about z; voxel (4, 4, 4) sits at the mark.
    voxel = np.array([4, 4, 4])
    affine = np.eye(4)
    affine[:3, :3] = _rotation_about_z(30.0)
    affine[:3, 3] = MARK - affine[:3, :3] @ voxel
    oblique = viewer.add(_marked((10, 10, 10), voxel, affine), Category.VOLUMES)
    seen, expected = _aligned(viewer, main, oblique, views[kind])
    assert seen == pytest.approx(_predicted(views[kind], MARK), abs=PIXEL)
    assert seen == pytest.approx(expected, abs=PIXEL)


def test_the_limits_in_scaled_units_set_the_rendered_grey(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    # Stored 0 and 5 are 10 and 20 scaled; 5 is above the maximum.
    item = viewer.add(_volume(2.0, 10.0), Category.VOLUMES)
    viewer.set(item.id, OptionName.MAXIMUM, 15.0)
    viewer.move_cursor(BLOCK)
    view = views[ViewKind.AXIAL]
    image = _image(view)
    assert _grey(view, image, np.array(BLOCK)) == WHITE
    hidden = (OUTSIDE[0], OUTSIDE[1], BLOCK[2])
    assert _grey(view, image, np.array(hidden)) == 0


def test_hiding_the_volume_blanks_its_slices(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    item = viewer.add(_volume(), Category.VOLUMES)
    viewer.move_cursor(BLOCK)
    viewer.set(item.id, OptionName.VISIBLE, False)
    view = views[ViewKind.AXIAL]
    assert _grey(view, _image(view), np.array(BLOCK)) == 0


def test_an_overlay_added_before_a_volume_shows_on_top_of_it(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    viewer.add(_volume(), Category.OVERLAYS)
    # Opaque black over the whole overlay plane: it hides the block if on top. Its
    # one bright voxel, off that plane, gives it a range.
    data = np.zeros((8, 8, 8), np.float32)
    data[7, 7, 7] = 1.0
    viewer.add(Volume(data, np.eye(4)), Category.VOLUMES)
    viewer.move_cursor(BLOCK)
    view = views[ViewKind.AXIAL]
    assert _grey(view, _image(view), np.array(BLOCK)) == WHITE


@pytest.mark.parametrize("value", [0.25, 0.5, 0.75])
def test_a_voxel_value_shows_as_its_place_on_the_gray_colormap(
    value: float,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    data = np.zeros((4, 4, 4), np.float32)
    data[1:3, 1:3, :] = value
    data[0, 0, 0] = 1.0
    item = viewer.add(Volume(data, np.eye(4)), Category.VOLUMES)
    viewer.set(item.id, OptionName.INTERPOLATE, False)
    view = views[ViewKind.AXIAL]
    image = _image(view)
    # An opaque linear grey: 0 to 1 over the 256 entries of the map, +-2 for its
    # sampling.
    assert _grey(view, image, np.array((1.5, 1.5, 1.5))) == pytest.approx(
        value * 255, abs=2
    )


@pytest.mark.parametrize("kind", VIEWS_2D)
def test_the_centred_slice_of_an_even_volume_is_one_voxel_plane(
    kind: ViewKind,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    # Only the centre plane across the view is 1: a cursor between two planes would
    # blend it with its zero neighbour to mid grey.
    data = np.zeros((8, 8, 8), np.float32)
    data[(slice(None),) * AXIS[kind] + (4,)] = 1.0
    viewer, _ = scene_renderer
    item = viewer.add(Volume(data, np.eye(4)), Category.VOLUMES)
    assert isinstance(item.options, VolumeOptions) and item.options.interpolate
    view = views[kind]
    cursor = np.array(viewer.scene.view.cursor)
    # +-2 for the sampling of the 256-entry map, as above.
    assert _grey(view, _image(view), cursor) == pytest.approx(WHITE, abs=2)


def test_an_overlay_hides_its_values_below_the_threshold(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    # A mid-grey volume; its two corner voxels, off the axial slice at z = 4, give it
    # the range 0 to 1.
    grey = np.full((8, 8, 8), 0.5, np.float32)
    grey[0, 0, 0], grey[7, 7, 7] = 0.0, 1.0
    volume = viewer.add(Volume(grey, np.eye(4)), Category.VOLUMES)
    data = np.zeros((8, 8, 8), np.float32)
    data[1:3, 1:3, :] = 1.0
    data[5:7, 5:7, :] = 3.0
    overlay = viewer.add(Volume(data, np.eye(4)), Category.OVERLAYS)
    for item in (volume, overlay):
        viewer.set(item.id, OptionName.INTERPOLATE, False)
    viewer.set(overlay.id, OptionName.THRESHOLD, 2.0)
    low, high = np.array((1.5, 1.5, 4.0)), np.array((5.5, 5.5, 4.0))
    view = views[ViewKind.AXIAL]
    image = _image(view)
    # +-2 for the sampling of the 256-entry map.
    assert _grey(view, image, low) == pytest.approx(0.5 * 255, abs=2)
    assert _grey(view, image, high) == WHITE
    viewer.set(overlay.id, OptionName.THRESHOLD, 3.5)
    image = _image(view)
    assert _grey(view, image, low) == pytest.approx(0.5 * 255, abs=2)
    assert _grey(view, image, high) == pytest.approx(0.5 * 255, abs=2)


def test_an_overlay_threshold_below_its_colour_range_shows_the_values_between(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    grey = np.full((8, 8, 8), 0.5, np.float32)
    grey[0, 0, 0], grey[7, 7, 7] = 0.0, 1.0
    volume = viewer.add(Volume(grey, np.eye(4)), Category.VOLUMES)
    # FA-like, coloured from 0.3 to 1 with the threshold at 0.2; 1 is off the slice.
    data = np.zeros((8, 8, 8), np.float32)
    data[0:2, 0:2, :] = 0.1
    data[3:5, 3:5, :] = 0.25
    data[6:8, 6:8, :] = 0.5
    data[0, 7, 0] = 1.0
    overlay = viewer.add(Volume(data, np.eye(4)), Category.OVERLAYS)
    for item in (volume, overlay):
        viewer.set(item.id, OptionName.INTERPOLATE, False)
    viewer.set(overlay.id, OptionName.MINIMUM, 0.3)
    viewer.set(overlay.id, OptionName.THRESHOLD, 0.2)
    view = views[ViewKind.AXIAL]
    image = _image(view)
    # +-2 for the sampling of the 256-entry map.
    assert _grey(view, image, np.array((0.5, 0.5, 4.0))) == pytest.approx(
        0.5 * 255, abs=2
    )
    assert _grey(view, image, np.array((3.5, 3.5, 4.0))) == 0
    assert _grey(view, image, np.array((6.5, 6.5, 4.0))) == pytest.approx(
        0.2 / 0.7 * 255, abs=2
    )
