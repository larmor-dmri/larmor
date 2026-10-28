"""The scene renderer and the views' cameras."""

from functools import partial

import numpy as np
import pytest
from render_helpers import (
    VIEWS_2D,
    _in_3d,
    _mesh,
    _ndc,
    _objects,
    _rotated,
    _volume,
)

from larmor.types import Tractogram
from larmor.viewer import (
    Category,
    Convention,
    Item,
    OptionName,
    Viewer,
    ViewKind,
)
from larmor.viewer.model.camera import Camera2D, screen_axes, view_directions
from larmor.viewer.model.events import Event, ViewStateChanged
from larmor.viewer.model.item import VIEW_KINDS
from larmor.viewer.render.scene_renderer import ADAPTERS, SceneRenderer
from larmor.viewer.render.tractogram_adapter import (
    TractogramAdapter,
)
from larmor.viewer.render.view import Vec3, ViewRenderer
from larmor.viewer.render.volume_adapter import (
    VolumeAdapter,
)

pytestmark = pytest.mark.usefixtures("fresh_slab_budget")


@pytest.fixture
def view_size() -> int:
    return 96


def _tractogram() -> Tractogram:
    return Tractogram(np.array([[0.0, 0, 0], [1, 1, 1]]), np.array([0, 2]))


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
def test_an_added_image_gets_an_adapter_of_its_category(
    category: Category,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, renderer = scene_renderer
    item = viewer.add(_volume(), category)
    adapter = renderer.adapters[item.id]
    assert type(adapter) is ADAPTERS[category]
    assert adapter.item is item
    assert isinstance(adapter, VolumeAdapter)
    for kind in VIEWS_2D:
        assert adapter.slices[kind] in views[kind].scene.children


def test_a_category_without_an_adapter_is_not_drawn(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every category has an adapter now; take one away.
    monkeypatch.delitem(ADAPTERS, Category.TRACTOGRAMS)
    viewer, renderer = scene_renderer
    item = viewer.add(_tractogram(), Category.TRACTOGRAMS)
    assert renderer.adapters == {}
    assert _objects(views) == dict.fromkeys(VIEW_KINDS, [])
    viewer.remove(item.id)


@pytest.mark.parametrize("how", ["item", "category"])
def test_a_removed_image_leaves_the_views(
    how: str,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, renderer = scene_renderer
    gone = viewer.add(_volume(), Category.OVERLAYS)
    kept = viewer.add(_volume(), Category.VOLUMES)
    viewer.remove(gone.id if how == "item" else Category.OVERLAYS)
    assert set(renderer.adapters) == {kept.id}
    kept_adapter = renderer.adapters[kept.id]
    assert isinstance(kept_adapter, VolumeAdapter)
    assert _objects(views) == {
        **{k: [kept_adapter.slices[k]] for k in VIEWS_2D},
        ViewKind.THREE_D: _in_3d(kept_adapter),
    }


def test_items_already_in_the_scene_get_adapters(
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer = Viewer()
    volume = viewer.add(_volume(), Category.VOLUMES)
    tracts = viewer.add(_tractogram(), Category.TRACTOGRAMS)
    overlay = viewer.add(_volume(), Category.OVERLAYS)
    renderer = SceneRenderer(viewer, views)
    assert list(renderer.adapters) == [volume.id, tracts.id, overlay.id]
    assert [type(a) for a in renderer.adapters.values()] == [
        VolumeAdapter,
        TractogramAdapter,
        VolumeAdapter,
    ]
    renderer.close()


def test_events_reach_the_adapters(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    item = viewer.add(_volume(), Category.OVERLAYS)
    viewer.set(item.id, OptionName.VISIBLE, False)
    adapter = renderer.adapters[item.id]
    assert isinstance(adapter, VolumeAdapter)
    assert not any(o.visible for o in adapter.slices.values())


def _render_order(renderer: SceneRenderer, item: Item) -> float:
    adapter = renderer.adapters[item.id]
    assert isinstance(adapter, VolumeAdapter)
    (order,) = {obj.render_order for obj in adapter.slices.values()}
    return float(order)


def test_overlays_draw_above_volumes_in_the_order_they_were_added(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    first = viewer.add(_volume(), Category.OVERLAYS)
    volume = viewer.add(_volume(), Category.VOLUMES)
    second = viewer.add(_volume(), Category.OVERLAYS)
    later = viewer.add(_volume(), Category.VOLUMES)
    orders = [_render_order(renderer, i) for i in (first, volume, second, later)]
    assert max(orders[1], orders[3]) < orders[0] < orders[2]


def test_a_closed_scene_renderer_stops_listening_and_clears_the_views(
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer = Viewer()
    baseline = len(viewer.scene._subscribers)
    renderer = SceneRenderer(viewer, views)
    viewer.add(_volume(), Category.VOLUMES)
    renderer.close()
    assert len(viewer.scene._subscribers) == baseline
    assert _objects(views) == dict.fromkeys(VIEW_KINDS, [])
    viewer.add(_volume(), Category.OVERLAYS)
    assert _objects(views) == dict.fromkeys(VIEW_KINDS, [])


def test_closing_a_scene_renderer_twice_is_harmless(
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer = Viewer()
    renderer = SceneRenderer(viewer, views)
    viewer.add(_volume(), Category.VOLUMES)
    renderer.close()
    renderer.close()
    assert renderer.adapters == {}
    assert _objects(views) == dict.fromkeys(VIEW_KINDS, [])


def test_an_event_already_on_its_way_reaches_no_adapter_after_close(
    views: dict[ViewKind, ViewRenderer], monkeypatch: pytest.MonkeyPatch
) -> None:
    viewer = Viewer()
    item = viewer.add(_volume(), Category.VOLUMES)
    renderer: SceneRenderer | None = None

    def close_on_cursor(event: Event) -> None:
        if isinstance(event, ViewStateChanged) and renderer is not None:
            renderer.close()

    # Subscribed first, so it closes the renderer while the scene is still handing
    # the same event to the renderer.
    viewer.scene.subscribe(close_on_cursor)
    renderer = SceneRenderer(viewer, views)
    reached: list[Event] = []
    monkeypatch.setattr(renderer.adapters[item.id], "on_event", reached.append)
    viewer.move_cursor((1.0, 1.0, 1.0))
    assert reached == []


def _field_extent(field: tuple[Vec3, Vec3], axis: Vec3) -> float:
    return float(abs((np.array(field[1]) - np.array(field[0])) @ np.array(axis)))


@pytest.mark.parametrize("convention", list(Convention))
@pytest.mark.parametrize("kind", VIEWS_2D)
def test_a_2d_view_frames_the_field_of_view_of_the_first_grid(
    kind: ViewKind,
    convention: Convention,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    viewer.set_convention(convention)
    viewer.add(_volume(affine=_rotated()), Category.VOLUMES)
    field = viewer.scene.view.field_of_view
    view = views[kind]
    forward, up = view_directions(kind, convention)
    assert np.allclose(view.camera.world.forward, forward)
    assert np.allclose(view.camera.world.up, up)
    centre = (np.array(field[0]) + np.array(field[1])) / 2
    assert np.allclose(_ndc(view, centre), 0.0, atol=1e-6)
    # The square view fits the longer side of the box across the screen.
    right, up = screen_axes(kind, convention)
    half = max(_field_extent(field, right), _field_extent(field, up)) / 2
    assert np.allclose(_ndc(view, centre + half * np.array(right)), [1.0, 0.0])
    assert np.allclose(_ndc(view, centre + half * np.array(up)), [0.0, 1.0])


@pytest.mark.parametrize("kind", VIEWS_2D)
def test_an_empty_viewer_frames_the_200_mm_cube_around_the_origin(
    kind: ViewKind,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    right, up = screen_axes(kind, Convention.RAS)
    view = views[kind]
    assert np.allclose(_ndc(view, np.zeros(3)), 0.0, atol=1e-6)
    assert np.allclose(_ndc(view, 100.0 * np.array(right)), [1.0, 0.0])
    assert np.allclose(_ndc(view, 100.0 * np.array(up)), [0.0, 1.0])


@pytest.mark.parametrize("kind", VIEWS_2D)
def test_zoom_and_pan_move_only_that_view_camera(
    kind: ViewKind,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    right, up = (np.array(a) for a in screen_axes(kind, Convention.RAS))
    others = {k: views[k].camera.get_state() for k in VIEWS_2D if k != kind}
    viewer.pan(kind, 10.0, -20.0)
    view = views[kind]
    centre = 10.0 * right - 20.0 * up
    assert np.allclose(_ndc(view, centre), 0.0, atol=1e-6)
    viewer.zoom(kind, 2.0)
    assert np.allclose(_ndc(view, centre + 25.0 * right), [0.5, 0.0])
    viewer.reset_camera(kind)
    assert np.allclose(_ndc(view, 50.0 * up), [0.0, 0.5])
    for other, state in others.items():
        after = views[other].camera.get_state()
        for key, value in state.items():
            assert np.array_equal(np.asarray(after[key]), np.asarray(value)), key


def test_setting_the_camera_redraws_only_when_it_moves(
    views: dict[ViewKind, ViewRenderer], draws: dict[ViewKind, int]
) -> None:
    view = views[ViewKind.AXIAL]
    field = ((-20.0, -20.0, -20.0), (20.0, 20.0, 20.0))
    view.set_camera(field, Camera2D(2.0, (5.0, -3.0)), Convention.LAS)
    view.set_camera(field, Camera2D(2.0, (5.0, -3.0)), Convention.LAS)
    view.set_camera()
    assert draws[ViewKind.AXIAL] == 1


@pytest.mark.parametrize("kind", VIEW_KINDS)
def test_a_later_object_keeps_the_camera(
    kind: ViewKind, views: dict[ViewKind, ViewRenderer]
) -> None:
    view = views[kind]
    view.add(_mesh((0.0, 0.0, 0.0)))
    before = view.camera.get_state()
    view.add(_mesh((300.0, 300.0, 300.0)))
    after = view.camera.get_state()
    assert after.keys() == before.keys()
    for key, value in before.items():
        assert np.array_equal(np.asarray(after[key]), np.asarray(value)), key


def test_replacing_data_rebuilds_that_items_adapter_only(
    scene_renderer: tuple[Viewer, SceneRenderer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    viewer, renderer = scene_renderer
    viewer.add(_volume(), Category.VOLUMES)
    item = viewer.add(_tractogram(), Category.TRACTOGRAMS)
    viewer.add(_tractogram(), Category.TRACTOGRAMS)
    before = dict(renderer.adapters)
    built: list[Item] = []
    add = renderer._add

    def counted(i: Item) -> None:
        built.append(i)
        add(i)

    monkeypatch.setattr(renderer, "_add", counted)
    viewer.replace_data(item.id, Tractogram(np.zeros((3, 3)), np.array([0, 3])))
    assert built == [item]
    assert renderer.adapters[item.id] is not before[item.id]
    others = {k: v for k, v in renderer.adapters.items() if k != item.id}
    assert others == {k: v for k, v in before.items() if k != item.id}


def test_replacing_a_mask_refilters_the_tractograms_using_it(
    scene_renderer: tuple[Viewer, SceneRenderer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    viewer, renderer = scene_renderer
    mask = viewer.add(_volume(), Category.OVERLAYS)
    used = viewer.add(_tractogram(), Category.TRACTOGRAMS)
    other = viewer.add(_tractogram(), Category.TRACTOGRAMS)
    viewer.apply_region(used.id, viewer.add_mask(mask.id))
    refiltered: list[int] = []
    for item in (used, other):
        adapter = renderer.adapters[item.id]
        monkeypatch.setattr(adapter, "refilter", partial(refiltered.append, item.id))
    viewer.replace_data(mask.id, _volume())
    assert refiltered == [used.id]


def test_replacing_a_values_volume_redraws_the_tractogram_coloured_by_it(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    values = viewer.add(_volume(), Category.VOLUMES)
    tract = viewer.add(_tractogram(), Category.TRACTOGRAMS)
    viewer.set(tract.id, OptionName.VALUES_VOLUME, values.id)
    before = renderer.adapters[tract.id]
    viewer.replace_data(values.id, _volume())
    assert renderer.adapters[tract.id] is not before
