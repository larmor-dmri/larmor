from collections.abc import Iterator
from typing import Any

import numpy as np
import pygfx as gfx
import pytest
from rendercanvas.offscreen import RenderCanvas

from larmor.types import Tractogram, Volume
from larmor.viewer import (
    Category,
    Colormap,
    Colouring,
    Item,
    OptionName,
    Viewer,
    ViewKind,
)
from larmor.viewer.model.item import VIEW_KINDS
from larmor.viewer.render.colour_bars import BOTTOM, RIGHT, TOP, WIDTH
from larmor.viewer.render.scene_renderer import SceneRenderer
from larmor.viewer.render.view import ViewRenderer
from larmor.viewer.render.volume_adapter import COLORMAP_MAPS

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


def _volume() -> Volume:
    data = np.zeros((8, 8, 8), np.float32)
    data[1:3, 4:7, 2:4] = 5.0
    return Volume(data, np.eye(4))


def _with_bar(viewer: Viewer, category: Category, data: Volume | None = None) -> Item:
    item = viewer.add(_volume() if data is None else data, category)
    viewer.set(item.id, OptionName.COLOUR_BAR, True)
    return item


# A bar as seen: its colormap, the text above it and the text below it.
type Bar = tuple[gfx.TextureMap, str, str]


def _bars(renderer: SceneRenderer, kind: ViewKind, texts: dict[int, str]) -> list[Bar]:
    """The bars of a view, from the right edge leftwards."""
    children = list(renderer.colour_bars.groups[kind].children)
    meshes = [c for c in children if isinstance(c, gfx.Mesh)]
    labels = [c for c in children if isinstance(c, gfx.Text)]
    assert len(labels) == 2 * len(meshes)
    found: list[tuple[float, Bar]] = []
    for mesh in meshes:
        positions = np.asarray(mesh.geometry.positions.data)
        right = float(positions[:, 0].max())
        centre = right - WIDTH / 2
        mine = [t for t in labels if t.local.position[0] == pytest.approx(centre)]
        heights = {round(float(t.local.position[1]), 6): t for t in mine}
        assert sorted(heights) == [BOTTOM, TOP]
        bar = (mesh.material.map, texts[id(heights[TOP])], texts[id(heights[BOTTOM])])
        found.append((right, bar))
    found.sort(key=lambda f: -f[0])
    return [bar for _, bar in found]


def _bars_everywhere(renderer: SceneRenderer, texts: dict[int, str]) -> list[Bar]:
    """The bars, the same in every 2D view."""
    first, *others = [_bars(renderer, k, texts) for k in VIEWS_2D]
    assert all(other == first for other in others)
    return first


def _rights(renderer: SceneRenderer, kind: ViewKind) -> list[float]:
    meshes = [
        c for c in renderer.colour_bars.groups[kind].children if isinstance(c, gfx.Mesh)
    ]
    return [float(np.asarray(m.geometry.positions.data)[:, 0].max()) for m in meshes]


GRAY, HOT = COLORMAP_MAPS[Colormap.GRAY], COLORMAP_MAPS[Colormap.HOT]


# Which items have a bar


def test_each_visible_colormapped_item_with_its_bar_on_has_one_in_layers_order(
    scene_renderer: tuple[Viewer, SceneRenderer], texts: dict[int, str]
) -> None:
    viewer, renderer = scene_renderer
    first = _with_bar(viewer, Category.OVERLAYS)
    viewer.add(Tractogram(np.zeros((2, 3)), np.array([0, 2])), Category.TRACTOGRAMS)
    second = _with_bar(viewer, Category.VOLUMES)
    viewer.add(_volume(), Category.OVERLAYS)  # its bar left off
    third = _with_bar(viewer, Category.OVERLAYS)
    for item, name in [
        (first, Colormap.HOT),
        (second, Colormap.VIRIDIS),
        (third, Colormap.GRAY),
    ]:
        viewer.set(item.id, OptionName.COLORMAP, name)
    maps = [bar[0] for bar in _bars_everywhere(renderer, texts)]
    assert maps == [
        HOT,
        COLORMAP_MAPS[Colormap.VIRIDIS],
        COLORMAP_MAPS[Colormap.GRAY],
    ]
    # The first stands at the right edge.
    for kind in VIEWS_2D:
        assert max(_rights(renderer, kind)) == pytest.approx(RIGHT)


def _hide(viewer: Viewer, item: Item, how: str) -> None:
    if how == "hidden":
        viewer.set(item.id, OptionName.VISIBLE, False)
    else:
        viewer.set(item.id, OptionName.LABELS, True)


@pytest.mark.parametrize(
    ("category", "how"),
    [
        (Category.VOLUMES, "hidden"),
        (Category.OVERLAYS, "hidden"),
        (Category.OVERLAYS, "labels"),
    ],
)
def test_hidden_items_and_overlays_shown_as_labels_have_no_bar_even_when_on(
    category: Category,
    how: str,
    scene_renderer: tuple[Viewer, SceneRenderer],
    texts: dict[int, str],
) -> None:
    viewer, renderer = scene_renderer
    items = {c: _with_bar(viewer, c) for c in (Category.VOLUMES, Category.OVERLAYS)}
    kept = items[
        Category.OVERLAYS if category == Category.VOLUMES else Category.VOLUMES
    ]
    viewer.set(kept.id, OptionName.COLORMAP, Colormap.HOT)
    _hide(viewer, items[category], how)
    assert [bar[0] for bar in _bars_everywhere(renderer, texts)] == [HOT]


def test_items_already_in_the_scene_get_bars(
    views: dict[ViewKind, ViewRenderer], texts: dict[int, str]
) -> None:
    viewer = Viewer()
    _with_bar(viewer, Category.VOLUMES)
    renderer = SceneRenderer(viewer, views)
    assert _bars_everywhere(renderer, texts) == [(GRAY, "5", "0")]
    renderer.close()


def test_the_3d_view_has_no_bars(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, renderer = scene_renderer
    _with_bar(viewer, Category.VOLUMES)
    assert ViewKind.THREE_D not in renderer.colour_bars.groups
    # Only the view's name, and the box behind it.
    labels = renderer.view_labels
    named = [labels.labels[ViewKind.THREE_D], labels.backings[ViewKind.THREE_D]]
    assert list(views[ViewKind.THREE_D].overlay.children) == named


def test_a_tractogram_with_its_bar_on_has_one_only_while_coloured_by_value(
    scene_renderer: tuple[Viewer, SceneRenderer], texts: dict[int, str]
) -> None:
    viewer, renderer = scene_renderer
    fa = viewer.add(_volume(), Category.OVERLAYS)
    tracts = viewer.add(
        Tractogram(np.zeros((2, 3)), np.array([0, 2])), Category.TRACTOGRAMS
    )
    viewer.set(tracts.id, OptionName.COLOUR_BAR, True)
    assert _bars_everywhere(renderer, texts) == []
    viewer.set(tracts.id, OptionName.VALUES_VOLUME, fa.id)
    viewer.set(tracts.id, OptionName.COLORMAP, Colormap.HOT)
    assert _bars_everywhere(renderer, texts) == [(HOT, "5", "0")]
    viewer.set(tracts.id, OptionName.COLOURING, Colouring.DIRECTION)
    assert _bars_everywhere(renderer, texts) == []


# What a bar shows


def test_a_bar_shows_the_colormap_maximum_above_and_minimum_below(
    scene_renderer: tuple[Viewer, SceneRenderer], texts: dict[int, str]
) -> None:
    viewer, renderer = scene_renderer
    item = _with_bar(viewer, Category.VOLUMES)
    viewer.set(item.id, OptionName.COLORMAP, Colormap.VIRIDIS)
    viewer.set(item.id, OptionName.MINIMUM, 0.0123456)
    viewer.set(item.id, OptionName.MAXIMUM, 1234.5678)
    # Four significant figures.
    assert _bars_everywhere(renderer, texts) == [
        (COLORMAP_MAPS[Colormap.VIRIDIS], "1235", "0.01235")
    ]


# Starting from a volume and an overlay of _volume(), both gray from 0 to 5.
_CHANGES: list[tuple[OptionName, object, list[Bar]]] = [
    (OptionName.COLORMAP, Colormap.HOT, [(GRAY, "5", "0"), (HOT, "5", "0")]),
    (OptionName.MINIMUM, 1.5, [(GRAY, "5", "0"), (GRAY, "5", "1.5")]),
    (OptionName.MAXIMUM, 4.25, [(GRAY, "5", "0"), (GRAY, "4.25", "0")]),
    (OptionName.VISIBLE, False, [(GRAY, "5", "0")]),
    (OptionName.LABELS, True, [(GRAY, "5", "0")]),
    (OptionName.COLOUR_BAR, False, [(GRAY, "5", "0")]),
]


@pytest.mark.parametrize(("name", "value", "expected"), _CHANGES)
def test_a_change_to_what_a_bar_shows_updates_the_bars(
    name: OptionName,
    value: object,
    expected: list[Bar],
    scene_renderer: tuple[Viewer, SceneRenderer],
    texts: dict[int, str],
) -> None:
    viewer, renderer = scene_renderer
    _with_bar(viewer, Category.VOLUMES)
    overlay = _with_bar(viewer, Category.OVERLAYS)
    assert _bars_everywhere(renderer, texts) == [(GRAY, "5", "0")] * 2
    viewer.set(overlay.id, name, value)
    assert _bars_everywhere(renderer, texts) == expected


def test_adding_and_removing_items_updates_the_bars(
    scene_renderer: tuple[Viewer, SceneRenderer], texts: dict[int, str]
) -> None:
    viewer, renderer = scene_renderer
    volume = _with_bar(viewer, Category.VOLUMES)
    overlay = _with_bar(viewer, Category.OVERLAYS)
    viewer.set(overlay.id, OptionName.COLORMAP, Colormap.HOT)
    _with_bar(viewer, Category.OVERLAYS)
    assert len(_bars_everywhere(renderer, texts)) == 3
    viewer.remove(volume.id)
    assert [bar[0] for bar in _bars_everywhere(renderer, texts)] == [
        HOT,
        GRAY,
    ]
    viewer.remove(Category.OVERLAYS)
    assert _bars_everywhere(renderer, texts) == []


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
def test_a_new_item_has_a_bar_only_while_its_colour_bar_is_on(
    category: Category,
    scene_renderer: tuple[Viewer, SceneRenderer],
    texts: dict[int, str],
) -> None:
    viewer, renderer = scene_renderer
    item = viewer.add(_volume(), category)
    assert getattr(item.options, OptionName.COLOUR_BAR) is False
    assert _bars_everywhere(renderer, texts) == []
    viewer.set(item.id, OptionName.COLOUR_BAR, True)
    assert _bars_everywhere(renderer, texts) == [(GRAY, "5", "0")]
    viewer.set(item.id, OptionName.COLOUR_BAR, False)
    assert _bars_everywhere(renderer, texts) == []


def test_the_colour_bar_on_a_category_gives_its_visible_items_bars(
    scene_renderer: tuple[Viewer, SceneRenderer], texts: dict[int, str]
) -> None:
    viewer, renderer = scene_renderer
    viewer.add(_volume(), Category.VOLUMES)
    shown = viewer.add(_volume(), Category.OVERLAYS)
    hidden = viewer.add(_volume(), Category.OVERLAYS)
    viewer.set(shown.id, OptionName.COLORMAP, Colormap.HOT)
    viewer.set(hidden.id, OptionName.VISIBLE, False)
    viewer.add(_volume(), Category.OVERLAYS)
    viewer.set(Category.OVERLAYS, OptionName.COLOUR_BAR, True)
    assert [bar[0] for bar in _bars_everywhere(renderer, texts)] == [HOT, GRAY]


def _children(renderer: SceneRenderer) -> dict[ViewKind, list[object]]:
    return {k: list(g.children) for k, g in renderer.colour_bars.groups.items()}


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (OptionName.OPACITY_2D, 0.5),
        (OptionName.INTERPOLATE, False),
        (OptionName.THRESHOLD, 2.0),
    ],
)
def test_another_option_change_keeps_the_bars(
    name: OptionName, value: object, scene_renderer: tuple[Viewer, SceneRenderer]
) -> None:
    viewer, renderer = scene_renderer
    _with_bar(viewer, Category.VOLUMES)
    overlay = _with_bar(viewer, Category.OVERLAYS)
    before = _children(renderer)
    assert all(before.values())
    viewer.set(overlay.id, name, value)
    viewer.move_cursor((1.0, 2.0, 3.0))
    after = _children(renderer)
    assert all(a is b for k in before for a, b in zip(after[k], before[k], strict=True))


_IMAGE_SHOWN: list[tuple[OptionName, object]] = [
    (OptionName.COLORMAP, Colormap.HOT),
    (OptionName.MINIMUM, 1.5),
    (OptionName.MAXIMUM, 4.25),
    (OptionName.VISIBLE, False),
]


@pytest.mark.parametrize(
    ("category", "name", "value"),
    [
        *[(Category.VOLUMES, n, v) for n, v in _IMAGE_SHOWN],
        *[(Category.OVERLAYS, n, v) for n, v in _IMAGE_SHOWN],
        (Category.OVERLAYS, OptionName.LABELS, True),
    ],
)
def test_a_shown_option_change_on_an_item_without_a_bar_keeps_the_bars(
    category: Category,
    name: OptionName,
    value: object,
    scene_renderer: tuple[Viewer, SceneRenderer],
    texts: dict[int, str],
) -> None:
    viewer, renderer = scene_renderer
    # On an overlay: a second volume would hide the first.
    _with_bar(viewer, Category.OVERLAYS)
    without = viewer.add(_volume(), category)
    before = _children(renderer)
    assert all(before.values())
    viewer.set(without.id, name, value)
    after = _children(renderer)
    assert all(a is b for k in before for a, b in zip(after[k], before[k], strict=True))
    assert _bars_everywhere(renderer, texts) == [(GRAY, "5", "0")]


def test_closing_the_scene_renderer_removes_the_bars(
    views: dict[ViewKind, ViewRenderer], texts: dict[int, str]
) -> None:
    viewer = Viewer()
    renderer = SceneRenderer(viewer, views)
    _with_bar(viewer, Category.VOLUMES)
    renderer.close()
    assert all(list(v.overlay.children) == [] for v in views.values())
    viewer.add(_volume(), Category.OVERLAYS)
    assert all(list(v.overlay.children) == [] for v in views.values())


# Drawn bars

# A bar is 2% of the view wide; the flush filter blurs about a pixel into it from
# each side, so the drawn bars are checked in views wide enough for a clean middle.
WIDE = 384
# The bar's middle column, and rows a sixteenth of its height in from each end.
_COLUMN = int((RIGHT - WIDTH / 2 + 1) / 2 * WIDE)
_INSET = (TOP - BOTTOM) / 16
_TOP = int((1 - (TOP - _INSET)) / 2 * WIDE)
_BOTTOM = int((1 - (BOTTOM + _INSET)) / 2 * WIDE)


@pytest.fixture
def wide(gpu: None) -> Iterator[tuple[Viewer, dict[ViewKind, ViewRenderer]]]:
    views = {k: ViewRenderer(RenderCanvas(size=(WIDE, WIDE)), k) for k in VIEW_KINDS}
    viewer = Viewer()
    renderer = SceneRenderer(viewer, views)
    yield viewer, views
    renderer.close()


def _image(view: ViewRenderer) -> np.ndarray:
    return np.asarray(view.canvas.draw())


def _top_and_bottom(view: ViewRenderer) -> tuple[list[int], list[int]]:
    image = _image(view)
    top: list[int] = image[_TOP, _COLUMN, :3].tolist()
    bottom: list[int] = image[_BOTTOM, _COLUMN, :3].tolist()
    return top, bottom


# 1/16 and 15/16 of the way up the 256-entry map, give or take its sampling.
_DARK = pytest.approx([255 / 16] * 3, abs=2)
_LIGHT = pytest.approx([255 * 15 / 16] * 3, abs=2)


@pytest.mark.parametrize("kind", VIEWS_2D)
def test_a_drawn_gray_bar_runs_from_black_at_the_bottom_to_white_at_the_top(
    kind: ViewKind, wide: tuple[Viewer, dict[ViewKind, ViewRenderer]]
) -> None:
    viewer, views = wide
    _with_bar(viewer, Category.VOLUMES)
    top, bottom = _top_and_bottom(views[kind])
    assert top == _LIGHT
    assert bottom == _DARK


def test_the_bars_draw_over_an_overlay_that_fills_the_view(
    wide: tuple[Viewer, dict[ViewKind, ViewRenderer]],
) -> None:
    viewer, views = wide
    _with_bar(viewer, Category.VOLUMES)  # sets the field of view
    # White across +-28 mm; its 0 corner, off every slice, gives it a range. An
    # overlay's slice is not depth tested, so only the drawing order keeps the bars
    # over it.
    data = np.ones((64, 64, 64), np.float32)
    data[0, 0, 0] = 0.0
    affine = np.eye(4)
    affine[:3, 3] = -28.0
    _with_bar(viewer, Category.OVERLAYS, Volume(data, affine))
    view = views[ViewKind.AXIAL]
    # Top left: clear of the readout's backing in the bottom left.
    assert _image(view)[_TOP, 2, :3].tolist() == [255] * 3
    top, bottom = _top_and_bottom(view)
    assert top == _LIGHT
    assert bottom == _DARK


# The information overlay switch


def _shown(views: dict[ViewKind, ViewRenderer]) -> dict[ViewKind, bool]:
    return {k: v.overlay_visible for k, v in views.items()}


def test_toggling_the_overlay_hides_and_shows_it_in_every_view(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    assert _shown(views) == dict.fromkeys(VIEW_KINDS, True)
    viewer.toggle_overlay()
    assert _shown(views) == dict.fromkeys(VIEW_KINDS, False)
    viewer.toggle_overlay()
    assert _shown(views) == dict.fromkeys(VIEW_KINDS, True)


@pytest.mark.parametrize("visible", [True, False])
def test_the_views_start_with_the_viewer_overlay_state(
    visible: bool, views: dict[ViewKind, ViewRenderer]
) -> None:
    viewer = Viewer()
    if not visible:
        viewer.toggle_overlay()
    for view in views.values():
        view.show_overlay(not visible)
    renderer = SceneRenderer(viewer, views)
    assert _shown(views) == dict.fromkeys(VIEW_KINDS, visible)
    renderer.close()


def _quad(left: float, right: float, colour: str) -> gfx.Mesh:
    positions = np.array(
        [[left, -1, 0], [right, -1, 0], [left, 1, 0], [right, 1, 0]], np.float32
    )
    geometry = gfx.Geometry(
        positions=positions, indices=np.array([[0, 1, 2], [2, 1, 3]], np.int32)
    )
    return gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=colour))


RED, GREEN, BLACK = [255, 0, 0], [0, 255, 0], [0, 0, 0]


@pytest.mark.parametrize("visible", [True, False])
def test_a_view_draws_its_overlay_over_the_scene_only_when_shown(
    visible: bool, views: dict[ViewKind, ViewRenderer]
) -> None:
    view = views[ViewKind.AXIAL]
    # A green box in the middle of a 40 mm field; the overlay covers the left quarter.
    view.set_camera(field=((-20.0, -20.0, -20.0), (20.0, 20.0, 20.0)))
    box = gfx.Mesh(gfx.box_geometry(10, 10, 10), gfx.MeshBasicMaterial(color="#0f0"))
    view.add(box)
    view.overlay.add(_quad(-1.0, -0.5, "#f00"))
    view.show_overlay(visible)
    image = _image(view)
    middle = image[SIZE // 2, SIZE // 2, :3].tolist()
    left = image[SIZE // 2, 2, :3].tolist()
    assert middle == GREEN
    assert left == (RED if visible else BLACK)
