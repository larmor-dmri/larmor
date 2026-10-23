from collections.abc import Iterator

import numpy as np
import pytest

from larmor.types import Grid, Volume
from larmor.viewer import Category, OptionName, Viewer, ViewKind
from larmor.viewer.model.readout import Vec3, closest_axis, slice_index
from larmor.viewer.render.offscreen import OffscreenRenderer
from larmor.viewer.render.view_labels import label_text

# Voxel axis 0 runs mostly along world z, axis 1 along x and axis 2 along y, each
# with a small tilt towards another world axis.
OBLIQUE = np.array(
    [
        [0.0, 2.0, 0.3, 5.0],
        [0.2, 0.0, 1.5, -7.0],
        [3.0, 0.1, 0.0, 11.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
GRID = Grid((10, 20, 30), OBLIQUE)
# Rounded as the readout's voxel: 4.4 -> 4, 6.6 -> 7.
VOXEL = (4.4, 6.6, 25.0)
# A wide view, so that the label in the top left corner is clear of the
# orientation letter at the top centre.
SIZE = (240, 96)


def _world(voxel: tuple[float, float, float]) -> Vec3:
    x, y, z = (OBLIQUE @ np.array([*voxel, 1.0]))[:3]
    return float(x), float(y), float(z)


# The slice a 2D view shows


def test_the_closest_voxel_axis_is_the_largest_component_of_the_world_axis() -> None:
    assert [closest_axis(GRID, axis) for axis in range(3)] == [1, 2, 0]


@pytest.mark.parametrize(
    ("axis", "expected"), [(0, (7, 20)), (1, (25, 30)), (2, (4, 10))]
)
def test_the_slice_index_is_the_rounded_voxel_along_the_closest_axis(
    axis: int, expected: tuple[int, int]
) -> None:
    assert slice_index(GRID, _world(VOXEL), axis) == expected


@pytest.mark.parametrize(("voxel", "index"), [(-1.0, -1), (10.0, 10)])
def test_the_slice_index_can_lie_outside_the_grid(voxel: float, index: int) -> None:
    assert slice_index(GRID, _world((voxel, 0.0, 0.0)), 2) == (index, 10)


# The label's text


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (ViewKind.AXIAL, "**Axial**  z 4 / 10"),
        (ViewKind.CORONAL, "**Coronal**  y 25 / 30"),
        (ViewKind.SAGITTAL, "**Sagittal**  x 7 / 20"),
        (ViewKind.THREE_D, "**3D**"),
    ],
)
def test_the_label_names_the_view_and_its_slice(kind: ViewKind, expected: str) -> None:
    assert label_text(kind, GRID, _world(VOXEL)) == expected


@pytest.mark.parametrize("voxel", [-1.0, 10.0])
def test_the_label_shows_a_dash_outside_the_grid(voxel: float) -> None:
    text = label_text(ViewKind.AXIAL, GRID, _world((voxel, 0.0, 0.0)))
    assert text == "**Axial**  z - / 10"


@pytest.mark.parametrize(
    ("kind", "name"),
    [
        (ViewKind.AXIAL, "Axial"),
        (ViewKind.CORONAL, "Coronal"),
        (ViewKind.SAGITTAL, "Sagittal"),
        (ViewKind.THREE_D, "3D"),
    ],
)
def test_without_a_grid_the_label_is_only_the_name(kind: ViewKind, name: str) -> None:
    assert label_text(kind, None, (0.0, 0.0, 0.0)) == f"**{name}**"


# The labels in the views


def _volume(shape: int, voxel_size: float) -> Volume:
    data = np.zeros((shape, shape, shape), np.float32)
    data[0, 0, 0] = 1.0  # a range, away from the slices the tests show
    return Volume(data, np.diag([voxel_size, voxel_size, voxel_size, 1.0]))


@pytest.fixture
def viewer(gpu: None) -> Viewer:
    viewer = Viewer()
    viewer.add(_volume(8, 1.0), Category.VOLUMES)
    viewer.toggle_crosshair()
    return viewer


@pytest.fixture
def renderer(viewer: Viewer) -> Iterator[OffscreenRenderer]:
    with OffscreenRenderer(viewer, SIZE) as renderer:
        yield renderer


def test_the_labels_follow_the_cursor(
    viewer: Viewer, renderer: OffscreenRenderer
) -> None:
    labels = renderer.scene_renderer.view_labels
    viewer.move_cursor((1.0, 2.0, 3.0))
    assert {k: labels.text(k) for k in ViewKind} == {
        ViewKind.AXIAL: "Axial  z 3 / 8",
        ViewKind.CORONAL: "Coronal  y 2 / 8",
        ViewKind.SAGITTAL: "Sagittal  x 1 / 8",
        ViewKind.THREE_D: "3D",
    }
    viewer.move_cursor((1.0, 2.0, 5.0))
    assert labels.text(ViewKind.AXIAL) == "Axial  z 5 / 8"


def test_the_labels_follow_a_new_main_volume_on_another_grid(
    viewer: Viewer, renderer: OffscreenRenderer
) -> None:
    labels = renderer.scene_renderer.view_labels
    first = viewer.scene.items[0]
    viewer.add(_volume(4, 2.0), Category.VOLUMES)
    viewer.move_cursor((4.0, 4.0, 4.0))
    assert labels.text(ViewKind.AXIAL) == "Axial  z 2 / 4"
    viewer.set(first.id, OptionName.VISIBLE, True)
    assert viewer.scene.view.main_volume == first.id
    assert labels.text(ViewKind.AXIAL) == "Axial  z 4 / 8"


# Screenshots


def _top_left(image: np.ndarray) -> np.ndarray:
    corner: np.ndarray = image[:16, :32].astype(int)
    return corner


def test_the_label_is_drawn_with_the_overlay_only(
    renderer: OffscreenRenderer,
) -> None:
    shown = _top_left(renderer.render(ViewKind.AXIAL, overlay=True))
    hidden = _top_left(renderer.render(ViewKind.AXIAL, overlay=False))
    # Without the overlay the corner is the plain background; with it, the text.
    assert np.ptp(hidden) == 0
    assert np.abs(shown - hidden).max() > 50


@pytest.mark.parametrize("kind", list(ViewKind))
def test_screenshots_never_draw_the_enlarge_button(
    kind: ViewKind, renderer: OffscreenRenderer
) -> None:
    image = renderer.render(kind, overlay=False)
    # The window draws the button in the top right corner, whatever the overlay.
    assert np.ptp(image[:30, -30:].astype(int)) == 0
