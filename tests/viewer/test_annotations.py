from collections.abc import Iterator
from typing import Any

import numpy as np
import pygfx as gfx
import pytest

from larmor.types import Volume
from larmor.viewer import Category, Convention, OptionName, Readout, Viewer, ViewKind
from larmor.viewer.model.camera import Camera2D, screen_axes
from larmor.viewer.render.annotations import readout_text
from larmor.viewer.render.backing import Backing
from larmor.viewer.render.scene_renderer import SceneRenderer
from larmor.viewer.render.view import ViewRenderer

VIEWS_2D: list[ViewKind] = [ViewKind.AXIAL, ViewKind.CORONAL, ViewKind.SAGITTAL]
SIZE = 96


@pytest.fixture
def view_size() -> int:
    return SIZE


@pytest.fixture
def texts(monkeypatch: pytest.MonkeyPatch) -> dict[int, str]:
    # pygfx has no getter for a Text's string, so record what each one is given.
    shown: dict[int, str] = {}
    original: Any = gfx.Text.set_text

    def spy(self: object, text: str) -> None:
        shown[id(self)] = text
        original(self, text)

    monkeypatch.setattr(gfx.Text, "set_text", spy)
    return shown


@pytest.fixture
def scene_renderer(
    views: dict[ViewKind, ViewRenderer], texts: dict[int, str]
) -> Iterator[tuple[Viewer, SceneRenderer]]:
    viewer = Viewer()
    renderer = SceneRenderer(viewer, views)
    yield viewer, renderer
    renderer.close()


def _image(view: ViewRenderer) -> np.ndarray:
    return np.asarray(view.canvas.draw()).astype(int)


def _green(image: np.ndarray) -> np.ndarray:
    # The crosshair is green on black; an anti-aliased line split over two pixel rows
    # is dimmer, but still green.
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    green: np.ndarray = (g > 2 * r + 20) & (g > 2 * b + 20)
    return green


def _white(image: np.ndarray) -> np.ndarray:
    white: np.ndarray = image[..., :3].min(axis=2) > 200
    return white


def _pixel(view: ViewRenderer, world: np.ndarray) -> tuple[float, float]:
    ndc = view.camera.camera_matrix @ np.array([*world, 1.0])
    x, y = ndc[:2] / ndc[3]
    return (1 - y) / 2 * SIZE, (x + 1) / 2 * SIZE


# The crosshair

CURSOR = np.array([20.0, -30.0, 40.0])


@pytest.mark.parametrize("convention", list(Convention))
@pytest.mark.parametrize("kind", VIEWS_2D)
def test_the_crosshair_crosses_at_the_cursor(
    kind: ViewKind,
    convention: Convention,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    viewer.toggle_overlay()
    viewer.set_convention(convention)
    viewer.move_cursor((CURSOR[0], CURSOR[1], CURSOR[2]))
    view = views[kind]
    green = _green(_image(view))
    # A line across the whole view is a row (and a column) of green pixels; the
    # anti-aliased line of 1 pixel spreads over up to three, so +-1.5 pixels.
    rows = np.nonzero(green.sum(axis=1) > SIZE // 2)[0]
    cols = np.nonzero(green.sum(axis=0) > SIZE // 2)[0]
    row, col = _pixel(view, CURSOR)
    assert len(rows) > 0 and len(cols) > 0
    assert np.all(np.abs(rows + 0.5 - row) <= 1.5)
    assert np.all(np.abs(cols + 0.5 - col) <= 1.5)


def test_toggle_crosshair_hides_and_shows_it_in_every_2d_view(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    viewer.toggle_overlay()
    viewer.toggle_crosshair()
    assert [_green(_image(views[k])).sum() for k in VIEWS_2D] == [0, 0, 0]
    viewer.toggle_crosshair()
    assert all(_green(_image(views[k])).sum() > SIZE for k in VIEWS_2D)


def test_the_3d_view_has_no_crosshair_or_text(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    image = _image(views[ViewKind.THREE_D])
    assert _green(image).sum() == 0
    assert _white(image).sum() == 0


def test_the_3d_view_holds_no_crosshair_object(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, renderer = scene_renderer
    viewer.add(Volume(np.ones((8, 8, 8), np.float32), np.eye(4)), Category.VOLUMES)
    assert ViewKind.THREE_D not in renderer.annotations.crosshairs
    view = views[ViewKind.THREE_D]
    lines: list[gfx.WorldObject] = []
    for root in (view.scene, view.guides):
        root.traverse(
            lambda obj: (
                lines.append(obj) if isinstance(obj, gfx.Line) and obj.visible else None
            )
        )
    assert lines == []


def test_hiding_the_grid_box_removes_it_from_every_view(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, renderer = scene_renderer
    viewer.new_phantom((20.0, 20.0, 20.0), 2.0)
    boxes = [group.children[0] for group in renderer.phantom_guides.groups.values()]
    assert len(boxes) == len(views) and all(box.visible for box in boxes)
    viewer.show_grid_box(False)
    assert not any(box.visible for box in boxes)
    viewer.start_bundle()  # always outlined while drawing
    assert all(box.visible for box in boxes)


# The information overlay switch


def test_the_overlay_switch_hides_the_text_but_not_the_crosshair(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    for kind in VIEWS_2D:
        image = _image(views[kind])
        assert _white(image).sum() > 0, kind
        assert _green(image).sum() > SIZE, kind
    viewer.toggle_overlay()
    for kind in VIEWS_2D:
        image = _image(views[kind])
        assert _white(image).sum() == 0, kind
        assert _green(image).sum() > SIZE, kind


# The orientation labels


def _place(label: gfx.Text) -> str:
    x, y, _ = label.local.position
    if abs(x) > abs(y):
        return "right" if x > 0 else "left"
    return "top" if y > 0 else "bottom"


def _labels(
    renderer: SceneRenderer, kind: ViewKind, texts: dict[int, str]
) -> dict[str, str]:
    labels = renderer.annotations.labels[kind].values()
    return {_place(label): texts[id(label)] for label in labels}


LETTERS: dict[tuple[ViewKind, Convention], dict[str, str]] = {
    (ViewKind.AXIAL, Convention.RAS): {
        "right": "R",
        "left": "L",
        "top": "A",
        "bottom": "P",
    },
    (ViewKind.AXIAL, Convention.LAS): {
        "right": "L",
        "left": "R",
        "top": "A",
        "bottom": "P",
    },
    (ViewKind.CORONAL, Convention.RAS): {
        "right": "R",
        "left": "L",
        "top": "S",
        "bottom": "I",
    },
    (ViewKind.CORONAL, Convention.LAS): {
        "right": "L",
        "left": "R",
        "top": "S",
        "bottom": "I",
    },
    (ViewKind.SAGITTAL, Convention.RAS): {
        "right": "A",
        "left": "P",
        "top": "S",
        "bottom": "I",
    },
    (ViewKind.SAGITTAL, Convention.LAS): {
        "right": "A",
        "left": "P",
        "top": "S",
        "bottom": "I",
    },
}


@pytest.mark.parametrize("kind", VIEWS_2D)
def test_each_2d_view_has_its_orientation_letters_at_its_edges(
    kind: ViewKind,
    scene_renderer: tuple[Viewer, SceneRenderer],
    texts: dict[int, str],
) -> None:
    viewer, renderer = scene_renderer
    assert _labels(renderer, kind, texts) == LETTERS[(kind, Convention.RAS)]
    viewer.set_convention(Convention.LAS)
    assert _labels(renderer, kind, texts) == LETTERS[(kind, Convention.LAS)]
    viewer.set_convention(Convention.RAS)
    assert _labels(renderer, kind, texts) == LETTERS[(kind, Convention.RAS)]


@pytest.mark.parametrize("kind", VIEWS_2D)
def test_the_letters_name_the_world_direction_of_each_edge(kind: ViewKind) -> None:
    # Independent of the table above: the screen axes, read as letters.
    names = {0: "LR", 1: "PA", 2: "IS"}
    for convention in Convention:
        right, up = (np.array(a) for a in screen_axes(kind, convention))
        letters = LETTERS[(kind, convention)]
        for place, direction in (("right", right), ("top", up)):
            axis = int(np.argmax(np.abs(direction)))
            assert letters[place] == names[axis][int(direction[axis] > 0)]


# Radiological display


def _random_volume() -> Volume:
    data = np.random.default_rng(7).uniform(0, 1, (8, 8, 8)).astype(np.float32)
    return Volume(data, np.eye(4))


@pytest.mark.parametrize(
    ("kind", "mirrored"),
    [(ViewKind.AXIAL, True), (ViewKind.CORONAL, True), (ViewKind.SAGITTAL, False)],
)
def test_las_shows_the_mirror_image_of_ras(
    kind: ViewKind,
    mirrored: bool,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    viewer.toggle_overlay()
    viewer.toggle_crosshair()
    item = viewer.add(_random_volume(), Category.VOLUMES)
    viewer.set(item.id, OptionName.INTERPOLATE, False)
    ras = _image(views[kind])
    assert np.any(ras[..., 0] != ras[:, ::-1, 0])
    viewer.set_convention(Convention.LAS)
    las = _image(views[kind])
    expected = ras[:, ::-1] if mirrored else ras
    # A pixel whose centre falls on the edge between two voxels may show either:
    # software renderers (Mesa's lavapipe) round such nearest-voxel samples the
    # other way in the mirrored view. Every other pixel must match.
    edges = np.zeros(expected.shape[:2], dtype=bool)
    across = np.any(expected[:, 1:] != expected[:, :-1], axis=-1)
    down = np.any(expected[1:] != expected[:-1], axis=-1)
    edges[:, 1:] |= across
    edges[:, :-1] |= across
    edges[1:] |= down
    edges[:-1] |= down
    differs = np.any(las != expected, axis=-1)
    assert not np.any(differs & ~edges)


# The readout


def _readout(renderer: SceneRenderer, texts: dict[int, str]) -> dict[ViewKind, str]:
    return {k: texts[id(t)] for k, t in renderer.annotations.readouts.items()}


def _block() -> Volume:
    data = np.zeros((8, 8, 8, 2), np.float32)
    data[1:3, 4:7, 2:4, 0] = 5.0
    data[1:3, 4:7, 2:4, 1] = 7.0
    return Volume(data, np.eye(4), slope=2.0, intercept=1.0)


def test_the_readout_shows_name_value_voxel_and_position_in_every_2d_view(
    scene_renderer: tuple[Viewer, SceneRenderer],
    texts: dict[int, str],
) -> None:
    viewer, renderer = scene_renderer
    item = viewer.add(_block(), Category.VOLUMES, name="t1")
    viewer.move_cursor((1.0, 5.0, 2.0))
    expected = "t1\nvalue 11\nvoxel 1, 5, 2, 0\n1.0, 5.0, 2.0 mm"
    assert _readout(renderer, texts) == dict.fromkeys(VIEWS_2D, expected)
    viewer.move_cursor((6.0, 1.0, 6.0))
    expected = "t1\nvalue 1\nvoxel 6, 1, 6, 0\n6.0, 1.0, 6.0 mm"
    assert _readout(renderer, texts) == dict.fromkeys(VIEWS_2D, expected)
    viewer.move_cursor((1.0, 5.0, 2.0))
    viewer.set(item.id, OptionName.VOLUME_INDEX, 1)
    expected = "t1\nvalue 15\nvoxel 1, 5, 2, 1\n1.0, 5.0, 2.0 mm"
    assert _readout(renderer, texts) == dict.fromkeys(VIEWS_2D, expected)


def test_the_readout_follows_the_interpolation_switch(
    scene_renderer: tuple[Viewer, SceneRenderer],
    texts: dict[int, str],
) -> None:
    viewer, renderer = scene_renderer
    item = viewer.add(_block(), Category.VOLUMES, name="t1")
    # Halfway between voxel 2 (in the block) and voxel 3 along x.
    viewer.move_cursor((2.5, 5.0, 2.0))
    assert "value 6\n" in _readout(renderer, texts)[ViewKind.AXIAL]
    viewer.set(item.id, OptionName.INTERPOLATE, False)
    # Halves round up, as nearest sampling draws it: voxel 3, outside the block.
    assert "value 1\nvoxel 3, 5, 2, 0\n" in _readout(renderer, texts)[ViewKind.AXIAL]


def test_outside_the_image_the_readout_says_so(
    scene_renderer: tuple[Viewer, SceneRenderer],
    texts: dict[int, str],
) -> None:
    viewer, renderer = scene_renderer
    viewer.add(_block(), Category.VOLUMES, name="t1")
    viewer.move_cursor((-3.0, 5.0, 2.0))
    expected = "t1\noutside the image\n-3.0, 5.0, 2.0 mm"
    assert _readout(renderer, texts) == dict.fromkeys(VIEWS_2D, expected)


def test_without_a_volume_the_readout_is_the_position(
    scene_renderer: tuple[Viewer, SceneRenderer],
    texts: dict[int, str],
) -> None:
    viewer, renderer = scene_renderer
    assert _readout(renderer, texts) == dict.fromkeys(VIEWS_2D, "0.0, 0.0, 0.0 mm")
    viewer.add(_block(), Category.OVERLAYS)
    viewer.move_cursor((1.0, 5.0, 2.0))
    assert _readout(renderer, texts) == dict.fromkeys(VIEWS_2D, "1.0, 5.0, 2.0 mm")


def test_removing_the_main_volume_leaves_the_position_only(
    scene_renderer: tuple[Viewer, SceneRenderer],
    texts: dict[int, str],
) -> None:
    viewer, renderer = scene_renderer
    item = viewer.add(_block(), Category.VOLUMES, name="t1")
    viewer.move_cursor((1.0, 5.0, 2.0))
    viewer.remove(item.id)
    assert _readout(renderer, texts) == dict.fromkeys(VIEWS_2D, "1.0, 5.0, 2.0 mm")


def test_readout_text_lists_the_lines_in_order() -> None:
    readout = Readout((1.25, -2.0, 3.0), "fa", (1, 2, 3), 0.123456789)
    assert (
        readout_text(readout) == "fa\nvalue 0.123457\nvoxel 1, 2, 3\n1.2, -2.0, 3.0 mm"
    )


# The backing behind the text


def _readout_box(renderer: SceneRenderer, kind: ViewKind) -> tuple[float, float]:
    """The width and height, in NDC, of the backing behind the readout."""
    readout = renderer.annotations.readouts[kind]
    children = renderer.annotations.backings[kind].children
    (backing,) = [b for b in children if isinstance(b, Backing) and b.text is readout]
    positions = np.asarray(backing.geometry.positions.data)
    width, height = np.ptp(positions[:, :2], axis=0)
    return float(width), float(height)


@pytest.mark.parametrize("kind", VIEWS_2D)
def test_the_readout_backing_is_sized_to_its_text_before_each_draw(
    kind: ViewKind,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, renderer = scene_renderer
    _image(views[kind])
    one_line = _readout_box(renderer, kind)
    assert min(one_line) > 0
    viewer.add(_block(), Category.VOLUMES, name="t1")  # four lines
    _image(views[kind])
    four_lines = _readout_box(renderer, kind)
    assert four_lines[1] > 2 * one_line[1]


def test_the_backing_darkens_a_bright_image_and_hides_with_the_overlay(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    data = np.ones((8, 8, 8), np.float32)
    data[0, 0, 0] = 0.0  # gives the white a range
    viewer.add(Volume(data, np.eye(4)), Category.VOLUMES)
    viewer.move_cursor((4.0, 4.0, 4.0))
    view = views[ViewKind.AXIAL]
    # The bottom-left corner: inside the readout's padding, clear of its glyphs.
    corner = (SIZE - 1, 0)
    assert _image(view)[corner][:3].max() < 230
    viewer.toggle_overlay()
    assert _image(view)[corner][:3].tolist() == [255] * 3


def test_closing_removes_the_crosshair_and_the_text(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    _, renderer = scene_renderer
    renderer.close()
    for kind in VIEWS_2D:
        assert list(views[kind].guides.children) == []
        assert list(views[kind].overlay.children) == []


# From the screen to the world


@pytest.mark.parametrize("convention", list(Convention))
@pytest.mark.parametrize("kind", VIEWS_2D)
def test_to_world_finds_the_world_point_under_a_pixel(
    kind: ViewKind, convention: Convention, views: dict[ViewKind, ViewRenderer]
) -> None:
    view = views[kind]
    field = ((-10.0, -20.0, -30.0), (30.0, 40.0, 10.0))
    view.set_camera(field, Camera2D(2.0, (5.0, -3.0)), convention)
    right, up = (np.array(a) for a in screen_axes(kind, convention))
    centre = np.array([10.0, 10.0, -10.0]) + 5.0 * right - 3.0 * up
    in_plane = np.array([right, up])
    # The middle of the canvas is the field's centre moved by the pan.
    assert np.allclose(in_plane @ view.to_world(SIZE / 2, SIZE / 2), in_plane @ centre)
    point = centre + 7.0 * right - 4.0 * up
    row, col = _pixel(view, point)
    assert np.allclose(in_plane @ view.to_world(col, row), in_plane @ point)
    # Logical pixels: right and down on the canvas are the screen's right and down.
    a, b = (np.array(view.to_world(x, y)) for x, y in ((40.0, 40.0), (50.0, 60.0)))
    # The field's longer side across the screen fills the 96 pixels at zoom 1.
    size = np.array(field[1]) - np.array(field[0])
    scale = max(abs(size @ right), abs(size @ up)) / 2.0 / SIZE
    assert np.allclose(in_plane @ (b - a), [10.0 * scale, -20.0 * scale])


def test_a_backing_uploads_its_box_only_when_it_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backing = Backing(gfx.Text(text="fa", screen_space=True))
    uploads: list[None] = []
    monkeypatch.setattr(
        backing.geometry.positions, "update_full", lambda: uploads.append(None)
    )
    backing.place(200.0, 100.0)
    backing.place(200.0, 100.0)
    assert len(uploads) == 1
    backing.place(400.0, 100.0)
    assert len(uploads) == 2
