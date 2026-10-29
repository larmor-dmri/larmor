"""Tractogram adapters."""

import logging
from collections.abc import Callable

import numpy as np
import pygfx as gfx
import pytest
from render_helpers import (
    AXIS,
    BLOCK,
    VIEWS_2D,
    WHITE,
    _image,
    _mesh,
    _rgb,
    _streamlines,
    _tracts,
    _volume,
)

from larmor.tractography import Sphere, passes_through
from larmor.types import Tractogram
from larmor.viewer import (
    CameraPreset,
    Category,
    Colouring,
    OptionName,
    TractStyle,
    Viewer,
    ViewKind,
)
from larmor.viewer.model import viewer as viewer_module
from larmor.viewer.model.registry import SUBSET_SEED
from larmor.viewer.render import (
    tractogram_adapter,
)
from larmor.viewer.render.geometry import (
    SEGMENT_BYTES,
    random_order,
)
from larmor.viewer.render.scene_renderer import SceneRenderer
from larmor.viewer.render.tractogram_adapter import (
    SLAB_BUDGET,
    TractogramAdapter,
)
from larmor.viewer.render.view import ViewRenderer

pytestmark = pytest.mark.usefixtures("fresh_slab_budget")


@pytest.fixture
def view_size() -> int:
    return 96


def _starts(adapter: TractogramAdapter) -> set[bytes]:
    # Each streamline of _streamlines() starts at its own point.
    return {bytes(adapter.drawn.points[i]) for i in adapter.drawn.offsets[:-1]}


def _copy_bytes(adapter: TractogramAdapter) -> int:
    return adapter.drawn.num_segments * SEGMENT_BYTES


def test_colour_and_opacity_change_the_materials_and_keep_the_geometry(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _streamlines())
    (geometry,) = {id(obj.geometry) for obj in adapter.lines.values()}
    slabs = [id(s.geometry) for band in adapter.slabs.values() for s in band]
    item = adapter.item.id
    viewer.set(item, OptionName.COLOURING, Colouring.SINGLE)
    viewer.set(item, OptionName.COLOUR, "#ff0000")
    viewer.set(item, OptionName.OPACITY_2D, 0.5)
    viewer.set(item, OptionName.OPACITY_3D, 0.25)
    assert {id(obj.geometry) for obj in adapter.lines.values()} == {geometry}
    assert [id(s.geometry) for band in adapter.slabs.values() for s in band] == slabs
    for kind, obj in adapter.lines.items():
        materials = [obj.material, *(s.material for s in adapter.slabs.get(kind, []))]
        for material in materials:
            assert material.color_mode == "uniform"
            assert tuple(material.color)[:3] == (1.0, 0.0, 0.0)
            expected = 0.25 if kind == ViewKind.THREE_D else 0.5
            assert material.opacity == expected


def test_per_tractogram_colouring_gives_each_tractogram_its_own_colour(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    colours = []
    for _ in range(3):
        adapter = _tracts(viewer, renderer, _streamlines(5))
        viewer.set(adapter.item.id, OptionName.COLOURING, Colouring.PER_TRACTOGRAM)
        material = adapter.lines[ViewKind.THREE_D].material
        assert material.color_mode == "uniform"
        colours.append(tuple(material.color))
    assert len(set(colours)) == 3


def test_subset_and_points_rebuild_the_lines_once_ready(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _streamlines())
    all_starts = _starts(adapter)
    assert len(all_starts) == 200
    viewer.set(adapter.item.id, OptionName.SUBSET, 50.0)
    viewer.set(adapter.item.id, OptionName.POINTS, 5)
    half = _starts(adapter)
    assert len(half) == 100 and half < all_starts
    assert set(np.diff(adapter.drawn.offsets).tolist()) == {5}
    (geometry,) = {obj.geometry for obj in adapter.lines.values()}
    # 5 points and a NaN separator per streamline.
    assert len(geometry.positions.data) == 100 * 6
    # The sorted copies follow the new lines.
    assert SLAB_BUDGET.used == 3 * _copy_bytes(adapter)
    # A smaller subset is a prefix of the same order.
    viewer.set(adapter.item.id, OptionName.SLAB, False)
    viewer.set(adapter.item.id, OptionName.SUBSET, 20.0)
    fifth = _starts(adapter)
    assert len(fifth) == 40 and fifth < half
    again = _tracts(viewer, renderer, _streamlines())
    viewer.set(again.item.id, OptionName.SLAB, False)
    viewer.set(again.item.id, OptionName.SUBSET, 20.0)
    assert _starts(again) == fifth


def _kept_starts(data: Tractogram, radius: float) -> set[bytes]:
    kept = passes_through(data, Sphere(np.zeros(3), radius))
    return {bytes(data.points[o]) for o in data.offsets[:-1][kept]}


def test_the_drawn_streamlines_are_a_subset_of_those_the_regions_keep(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    data = _streamlines()
    adapter = _tracts(viewer, renderer, data)
    kept = _kept_starts(data, 15.0)
    viewer.add_sphere((0.0, 0.0, 0.0), 15.0, apply=True)
    assert _starts(adapter) == kept
    # Filtering first: 40% of the 69 kept, not those kept among 40% of all.
    viewer.set(adapter.item.id, OptionName.SUBSET, 40.0)
    drawn = _starts(adapter)
    assert len(kept) == 69
    assert len(drawn) == 28 and drawn < kept


def test_the_kept_streamlines_are_drawn_in_the_fixed_random_order(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    data = _streamlines()
    adapter = _tracts(viewer, renderer, data)
    viewer.add_sphere((0.0, 0.0, 0.0), 15.0, apply=True)
    selection = passes_through(data, Sphere(np.zeros(3), 15.0))
    order = random_order(data.num_streamlines, SUBSET_SEED)
    expected = order[selection[order]]  # the random order, of the kept ones
    starts = data.points[data.offsets[:-1]]

    def drawn_order() -> list[bytes]:
        return [bytes(p) for p in adapter.drawn.points[adapter.drawn.offsets[:-1]]]

    assert drawn_order() == [bytes(p) for p in starts[expected]]
    viewer.set(adapter.item.id, OptionName.SUBSET, 40.0)
    forty = drawn_order()
    viewer.set(adapter.item.id, OptionName.SUBSET, 20.0)
    twenty = drawn_order()
    assert forty == [bytes(p) for p in starts[expected[:28]]]
    assert twenty == forty[: len(twenty)]  # 20% is a prefix of 40%


class _LaterWorker:
    """Holds the newest request until `run`, so lines can be made while it waits."""

    def __init__(self) -> None:
        self.queue: list[Callable[[], None]] = []

    @property
    def pending(self) -> bool:
        return bool(self.queue)

    def submit(
        self,
        work: Callable[[], object],
        done: Callable[[object], None],
        failed: object = None,
        ended: object = None,
    ) -> None:
        self.queue[:] = [lambda: done(work())]

    def run(self) -> None:
        queue, self.queue = self.queue, []
        for request in queue:
            request()

    def cancel(self) -> None:
        self.queue.clear()

    def close(self) -> None:
        self.queue.clear()


def test_the_first_lines_are_made_in_the_background(
    scene_renderer: tuple[Viewer, SceneRenderer], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Gathering the points of a large tractogram takes seconds: adding one must
    # leave it to the worker, then draw the lines and sort them once they come.
    workers: list[_LaterWorker] = []

    def later(*_: object) -> _LaterWorker:
        workers.append(_LaterWorker())
        return workers[-1]

    monkeypatch.setattr(tractogram_adapter, "Worker", later)
    viewer, renderer = scene_renderer
    data = _streamlines()
    adapter = _tracts(viewer, renderer, data)
    assert len(adapter.drawn.offsets) == 1 and not adapter.sorted_copies
    while any(w.pending for w in workers):
        for worker in workers:
            worker.run()
    assert len(adapter.drawn.offsets) - 1 == data.num_streamlines
    assert set(adapter.sorted_copies) == {0, 1, 2}


def test_other_regions_keeping_as_many_streamlines_still_change_the_lines(
    scene_renderer: tuple[Viewer, SceneRenderer], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(TractogramAdapter, "draw_cap", 10)
    viewer, renderer = scene_renderer
    data = _streamlines()
    adapter = _tracts(viewer, renderer, data)
    order = random_order(data.num_streamlines, SUBSET_SEED)
    starts = data.points[data.offsets[:-1]]

    def first_kept(centre: tuple[float, float, float]) -> list[bytes]:
        kept = passes_through(data, Sphere(np.array(centre), 15.0))
        assert kept.sum() > 10  # both cut to the cap: the same count drawn
        return [bytes(p) for p in starts[order[kept[order]]][:10]]

    def drawn() -> list[bytes]:
        return [bytes(p) for p in adapter.drawn.points[adapter.drawn.offsets[:-1]]]

    viewer.add_sphere((0.0, 0.0, 0.0), 15.0, apply=True)
    assert drawn() == first_kept((0.0, 0.0, 0.0))
    # The new lines wait while the colours change: the change must not recolour
    # the lines of the old regions in their place.
    later = _LaterWorker()
    monkeypatch.setattr(adapter, "_lines_worker", later)
    viewer.clear_regions(adapter.item.id)
    later.run()
    viewer.add_sphere((8.0, 8.0, 8.0), 15.0, apply=True)
    viewer.set(adapter.item.id, OptionName.MINIMUM, 0.5)
    later.run()
    assert drawn() == first_kept((8.0, 8.0, 8.0)) != first_kept((0.0, 0.0, 0.0))


def test_regions_that_fail_tell_the_user(
    scene_renderer: tuple[Viewer, SceneRenderer],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    viewer, renderer = scene_renderer
    told: list[str] = []
    renderer.subscribe_notices(told.append)
    adapter = _tracts(viewer, renderer, _streamlines())
    every = _starts(adapter)

    def broken(*_: object) -> np.ndarray:
        raise RuntimeError("broken filter")

    monkeypatch.setattr(viewer_module, "passes_regions", broken)
    # Off screen the work runs at once; the scene logs its error.
    viewer.add_sphere((0.0, 0.0, 0.0), 15.0, apply=True)
    assert "broken filter" in caplog.text
    assert told == ["Tractogram 1: the regions could not be applied; see the log."]
    assert _starts(adapter) == every


@pytest.mark.parametrize("undo", ["filter-off", "clear"])
def test_undoing_the_filter_draws_every_streamline_again(
    undo: str, scene_renderer: tuple[Viewer, SceneRenderer]
) -> None:
    viewer, renderer = scene_renderer
    data = _streamlines()
    adapter = _tracts(viewer, renderer, data)
    every = _starts(adapter)
    viewer.add_sphere((0.0, 0.0, 0.0), 15.0, apply=True)
    assert _starts(adapter) < every
    if undo == "clear":
        viewer.clear_regions(adapter.item.id)
    else:
        viewer.set(adapter.item.id, OptionName.REGION_FILTER, False)
    assert _starts(adapter) == every


def test_the_drawn_streamlines_stop_at_the_cap(
    scene_renderer: tuple[Viewer, SceneRenderer], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert TractogramAdapter.draw_cap == 100_000
    monkeypatch.setattr(TractogramAdapter, "draw_cap", 30)
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _streamlines())
    assert len(adapter.drawn.offsets) - 1 == 30


def test_the_lines_are_in_random_order_so_any_prefix_is_a_random_subset(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    data = _streamlines()
    adapter = _tracts(viewer, renderer, data)
    index = {bytes(data.points[o]): s for s, o in enumerate(data.offsets[:-1])}
    order = [index[bytes(adapter.drawn.points[o])] for o in adapter.drawn.offsets[:-1]]
    assert sorted(order) == list(range(200))
    assert order != sorted(order)


# With 30 points a streamline takes 31 vertices with its NaN, so 3 fit in 100; with
# 5 points, 16 fit.
@pytest.mark.parametrize(("points", "prefix"), [(30, 93), (5, 96)])
def test_the_3d_view_draws_a_prefix_of_the_lines_only_while_interacting(
    points: int,
    prefix: int,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tractogram_adapter, "INTERACTIVE_VERTICES", 100)
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _streamlines())
    viewer.set(adapter.item.id, OptionName.POINTS, points)
    full = (0, 200 * (points + 1))

    def ranges() -> dict[ViewKind, tuple[int, ...]]:
        # The views share one buffer: read each range right after its own draw.
        drawn: dict[ViewKind, tuple[int, ...]] = {}
        for kind, view in views.items():
            view.canvas.draw()
            drawn[kind] = tuple(adapter.lines[kind].geometry.positions.draw_range)
        return drawn

    for view in views.values():
        view.start_interaction()
    expected = {k: (0, prefix) if k == ViewKind.THREE_D else full for k in views}
    assert ranges() == expected
    for view in views.values():
        view.end_interaction()
    assert ranges() == dict.fromkeys(views, full)


def _tubes(adapter: TractogramAdapter) -> gfx.Mesh:
    assert adapter.tubes is not None
    return adapter.tubes


def test_tubes_replace_the_lines_in_the_3d_view_only_until_set_back_to_lines(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _streamlines(20, 5))
    viewer.set(adapter.item.id, OptionName.SLAB, False)  # whole lines in 2D
    viewer.set(adapter.item.id, OptionName.STYLE, TractStyle.TUBES)
    mesh = _tubes(adapter)
    assert mesh.parent is views[ViewKind.THREE_D].scene
    assert mesh.visible and not adapter.lines[ViewKind.THREE_D].visible
    for kind in VIEWS_2D:
        assert mesh not in views[kind].scene.children
        assert adapter.lines[kind].visible
    viewer.set(adapter.item.id, OptionName.STYLE, TractStyle.LINES)
    assert adapter.tubes is None and mesh.parent is None
    assert adapter.lines[ViewKind.THREE_D].visible


def test_tubes_get_the_lines_colours_as_linear_light(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    # A mesh's vertex colours are linear, a line's sRGB: the same colour on both.
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _streamlines(20, 5))
    viewer.set(adapter.item.id, OptionName.STYLE, TractStyle.TUBES)
    srgb = adapter.drawn.colours[0]
    tube = _tubes(adapter).geometry.colors.data[0]
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    assert tube == pytest.approx(linear, abs=1e-6)
    assert (srgb > 0.05).any() and (tube < srgb + 1e-6).all()


def test_a_new_radius_rebuilds_the_tubes_and_a_colour_only_recolours_them(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _streamlines(20, 5))
    item = adapter.item.id
    viewer.set(item, OptionName.STYLE, TractStyle.TUBES)
    thin = _tubes(adapter).geometry
    viewer.set(item, OptionName.TUBE_RADIUS, 1.0)
    thick = _tubes(adapter).geometry
    assert thick is not thin
    # A ring four times as wide; float32 positions ~20 mm out: ~1e-5 relative.
    width = np.ptp(thin.positions.data[:8], axis=0).max()
    assert np.ptp(thick.positions.data[:8], axis=0).max() == pytest.approx(
        4 * width, rel=1e-5
    )
    viewer.set(item, OptionName.COLOURING, Colouring.SINGLE)
    viewer.set(item, OptionName.COLOUR, "#ff0000")
    assert _tubes(adapter).geometry is thick
    material = _tubes(adapter).material
    assert material.color_mode == "uniform"
    assert tuple(material.color)[:3] == (1.0, 0.0, 0.0)


def test_the_3d_view_draws_a_prefix_of_the_tubes_only_while_interacting(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Tubes of 5 points take 4 * 2 * 8 = 64 triangles each: 3 fit in 200.
    monkeypatch.setattr(tractogram_adapter, "INTERACTIVE_TRIANGLES", 200)
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _streamlines(20, 5))
    viewer.set(adapter.item.id, OptionName.STYLE, TractStyle.TUBES)
    view = views[ViewKind.THREE_D]
    indices = _tubes(adapter).geometry.indices
    view.start_interaction()
    view.canvas.draw()
    assert tuple(indices.draw_range) == (0, 3 * 64)
    view.end_interaction()
    view.canvas.draw()
    assert tuple(indices.draw_range) == (0, 20 * 64)


def test_ambient_occlusion_dims_the_3d_view_only_and_comes_off_again(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _streamlines(200, 10))
    item = adapter.item.id
    shared = adapter.lines[ViewKind.THREE_D].geometry
    viewer.set(item, OptionName.STYLE, TractStyle.TUBES)
    undimmed = _tubes(adapter).geometry.colors.data.copy()
    viewer.set(item, OptionName.AMBIENT_OCCLUSION, True)
    light = adapter.light
    assert light is not None and light.shape == (len(adapter.drawn.points),)
    assert light.min() < 0.9  # some points are hidden
    three_d = adapter.lines[ViewKind.THREE_D].geometry
    assert three_d is not shared and three_d.positions is shared.positions
    for kind in VIEWS_2D:
        assert adapter.lines[kind].geometry is shared
    # A line's sRGB colour is dimmed by the light's power 1 / 2.2; NaN rows
    # separate the streamlines, so point p of streamline s is row p + s.
    colours = three_d.colors.data
    for p in (0, 15, 37, len(light) - 1):
        row = p + int(np.searchsorted(adapter.drawn.offsets, p, side="right")) - 1
        expected = adapter.drawn.colours[p] * light[p] ** (1 / 2.2)
        assert colours[row] == pytest.approx(expected, abs=1e-6)
    tube = _tubes(adapter).geometry.colors.data
    assert tube == pytest.approx(undimmed * np.repeat(light, 8)[:, None], abs=1e-6)
    # In one colour the 3D view takes the dimmed colours from its vertices.
    viewer.set(item, OptionName.COLOURING, Colouring.SINGLE)
    assert _tubes(adapter).material.color_mode == "vertex"
    assert adapter.lines[VIEWS_2D[0]].material.color_mode == "uniform"
    viewer.set(item, OptionName.AMBIENT_OCCLUSION, False)
    assert adapter.light is None
    assert adapter.lines[ViewKind.THREE_D].geometry is shared
    assert _tubes(adapter).material.color_mode == "uniform"
    assert _tubes(adapter).geometry.colors.data == pytest.approx(undimmed)


def test_option_changes_upload_no_colours_that_did_not_change(
    scene_renderer: tuple[Viewer, SceneRenderer], monkeypatch: pytest.MonkeyPatch
) -> None:
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _streamlines(50, 10))
    item = adapter.item.id
    viewer.set(item, OptionName.STYLE, TractStyle.TUBES)
    uploads: list[int] = []
    monkeypatch.setattr(
        adapter, "_set_tube_colours", lambda colours: uploads.append(len(colours))
    )
    for opacity in (0.9, 0.8, 1.0):  # without light: nothing to restore
        viewer.set(item, OptionName.OPACITY_3D, opacity)
    assert uploads == []
    viewer.set(item, OptionName.AMBIENT_OCCLUSION, True)
    assert len(uploads) == 1
    dimmed = adapter.lines[ViewKind.THREE_D].geometry
    for opacity in (0.9, 0.8, 1.0):  # the same light and colours
        viewer.set(item, OptionName.OPACITY_3D, opacity)
    assert len(uploads) == 1 and adapter.lines[ViewKind.THREE_D].geometry is dimmed
    viewer.set(item, OptionName.AMBIENT_OCCLUSION, False)  # restored once
    viewer.set(item, OptionName.OPACITY_3D, 0.5)
    assert len(uploads) == 2


def test_above_the_tube_limit_the_3d_view_keeps_lines_and_says_so_once(
    scene_renderer: tuple[Viewer, SceneRenderer], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(TractogramAdapter, "tube_limit", 100)
    viewer, renderer = scene_renderer
    notices: list[str] = []
    renderer.subscribe_notices(notices.append)
    adapter = _tracts(viewer, renderer, _streamlines(200, 5))
    item = adapter.item.id
    viewer.set(item, OptionName.STYLE, TractStyle.TUBES)
    assert adapter.tubes is None and adapter.lines[ViewKind.THREE_D].visible
    assert len(notices) == 1 and "200 streamlines" in notices[0]
    viewer.set(item, OptionName.OPACITY_3D, 0.5)
    assert len(notices) == 1
    viewer.set(item, OptionName.SUBSET, 50.0)
    assert adapter.tubes is not None and adapter.tube_notice() is None
    viewer.set(item, OptionName.SUBSET, 100.0)
    assert adapter.tubes is None and len(notices) == 2


def test_more_points_per_streamline_lower_the_streamlines_drawn_as_tubes(
    scene_renderer: tuple[Viewer, SceneRenderer], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A budget of 10 * 19 segments: 10 streamlines fit at 20 points, not at 200.
    monkeypatch.setattr(TractogramAdapter, "tube_limit", 10)
    viewer, renderer = scene_renderer
    notices: list[str] = []
    renderer.subscribe_notices(notices.append)
    adapter = _tracts(viewer, renderer, _streamlines(10, 5))
    item = adapter.item.id
    viewer.set(item, OptionName.STYLE, TractStyle.TUBES)
    assert adapter.tubes is not None and notices == []
    viewer.set(item, OptionName.POINTS, 200)
    assert adapter.tubes is None and adapter.lines[ViewKind.THREE_D].visible
    assert len(notices) == 1 and "of 200 points" in notices[0]
    viewer.set(item, OptionName.POINTS, 20)
    assert adapter.tubes is not None and adapter.tube_notice() is None


def test_a_new_subscriber_is_given_the_notices_that_already_apply_once(
    scene_renderer: tuple[Viewer, SceneRenderer], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(TractogramAdapter, "tube_limit", 100)
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _streamlines(200, 5))
    viewer.set(adapter.item.id, OptionName.STYLE, TractStyle.TUBES)
    notices: list[str] = []
    renderer.subscribe_notices(notices.append)
    assert notices == [adapter.tube_notice()]
    viewer.set(adapter.item.id, OptionName.OPACITY_3D, 0.5)
    assert len(notices) == 1


def test_a_removed_tractogram_leaves_no_notice_for_a_new_subscriber(
    scene_renderer: tuple[Viewer, SceneRenderer], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(TractogramAdapter, "tube_limit", 100)
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _streamlines(200, 5))
    viewer.set(adapter.item.id, OptionName.STYLE, TractStyle.TUBES)
    viewer.remove(adapter.item.id)
    notices: list[str] = []
    renderer.subscribe_notices(notices.append)
    assert notices == []


def _segment_set(points: np.ndarray) -> set[bytes]:
    return set(map(bytes, points.astype(np.float32).reshape(-1, 2, 3)))


def _drawn_segments(adapter: TractogramAdapter, kind: ViewKind) -> set[bytes]:
    drawn: set[bytes] = set()
    for slab in adapter.slabs[kind]:
        if slab.visible:
            offset, count = slab.geometry.positions.draw_range
            drawn |= _segment_set(slab.geometry.positions.data[offset : offset + count])
    return drawn


def _meeting(
    adapter: TractogramAdapter, axis: int, low: float, high: float
) -> set[bytes]:
    lines = adapter.drawn
    pairs = [
        lines.points[i : i + 2]
        for s in range(len(lines.offsets) - 1)
        for i in range(lines.offsets[s], lines.offsets[s + 1] - 1)
    ]
    return {
        bytes(p)
        for p in np.array(pairs, np.float32)
        if p[:, axis].max() >= low and p[:, axis].min() <= high
    }


def test_the_slab_clips_first_then_draws_ranges_of_the_sorted_segments(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _streamlines())
    # Off screen the sorted copies are made at once.
    assert set(adapter.sorted_copies) == {0, 1, 2}
    for cursor in [(1.0, 2.0, 3.0), (-8.0, 6.0, 10.0)]:
        viewer.move_cursor(cursor)
        for kind in VIEWS_2D:
            axis = AXIS[kind]
            assert not adapter.lines[kind].visible
            drawn = _drawn_segments(adapter, kind)
            meeting = _meeting(adapter, axis, cursor[axis] - 2.5, cursor[axis] + 2.5)
            assert meeting and meeting <= drawn
            assert len(drawn) < adapter.drawn.num_segments


def test_above_the_memory_budget_the_slab_stays_with_clipping_planes(
    scene_renderer: tuple[Viewer, SceneRenderer],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(SLAB_BUDGET, "limit", 1)
    viewer, renderer = scene_renderer
    with caplog.at_level(logging.INFO, logger="larmor"):
        adapter = _tracts(viewer, renderer, _streamlines())
    assert adapter.sorted_copies == {}
    assert SLAB_BUDGET.used == 0
    assert "memory budget" in caplog.text
    for kind in VIEWS_2D:
        assert adapter.lines[kind].visible
        assert len(adapter.lines[kind].material.clipping_planes) == 2


def test_a_slab_past_every_segment_draws_nothing(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    # Every point within 4 mm of the origin, every segment at most 8 mm long.
    adapter = _tracts(viewer, renderer, _sheet((0.0, 0.0, 0.0), 0.5))
    viewer.move_cursor((90.0, 90.0, 90.0))
    assert viewer.scene.view.cursor == (90.0, 90.0, 90.0)
    for kind in VIEWS_2D:
        assert _drawn_segments(adapter, kind) == set()


@pytest.mark.parametrize("how", ["slab off", "removed"])
def test_the_sorted_copies_give_their_memory_back(
    how: str, scene_renderer: tuple[Viewer, SceneRenderer]
) -> None:
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _streamlines())
    assert SLAB_BUDGET.used == 3 * _copy_bytes(adapter)
    if how == "slab off":
        viewer.set(adapter.item.id, OptionName.SLAB, False)
        assert adapter.sorted_copies == {}
        for kind in VIEWS_2D:
            assert adapter.lines[kind].visible
            assert len(adapter.lines[kind].material.clipping_planes) == 0
    else:
        viewer.remove(adapter.item.id)
    assert SLAB_BUDGET.used == 0


def test_a_maximised_view_gets_the_only_sorted_copy(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    viewer.toggle_maximised(ViewKind.CORONAL)
    adapter = _tracts(viewer, renderer, _streamlines())
    assert set(adapter.sorted_copies) == {AXIS[ViewKind.CORONAL]}
    assert SLAB_BUDGET.used == _copy_bytes(adapter)
    viewer.toggle_maximised(ViewKind.CORONAL)
    assert set(adapter.sorted_copies) == {0, 1, 2}


def _sheet(centre: tuple[float, float, float], step: float) -> Tractogram:
    # Dense parallel lines along x in the plane z = centre[2], so every pixel
    # around the centre shows one.
    y = centre[1] + np.arange(-4.0, 4.0, step)
    a = np.stack([np.full_like(y, centre[0] - 4), y, np.full_like(y, centre[2])], 1)
    b = a + [8.0, 0.0, 0.0]
    points = np.stack([a, b], axis=1).reshape(-1, 3)
    return Tractogram(points, np.arange(len(y) + 1) * 2)


def test_2d_tracts_draw_over_an_overlay(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, renderer = scene_renderer
    viewer.add(_volume(), Category.OVERLAYS)
    viewer.move_cursor(BLOCK)
    view = views[ViewKind.AXIAL]
    assert _rgb(view, _image(view), np.array(BLOCK)) == [WHITE] * 3
    # Along x: direction colouring makes them pure red.
    _tracts(viewer, renderer, _sheet(BLOCK, 0.04))
    assert _rgb(view, _image(view), np.array(BLOCK)) == [WHITE, 0, 0]


@pytest.mark.parametrize("box_order", [-1e9, 1e9], ids=["box first", "box last"])
@pytest.mark.parametrize(("box_z", "seen"), [(7.0, False), (-7.0, True)])
def test_an_opaque_3d_tract_is_hidden_only_behind_an_opaque_object(
    box_z: float,
    seen: bool,
    box_order: float,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, renderer = scene_renderer
    view = views[ViewKind.THREE_D]
    # Opaque like the tract, so in its queue, and drawn before or after it: depth
    # alone decides what shows.
    box = _mesh((0.0, 0.0, box_z))
    box.material.alpha_mode = "solid"
    box.render_order = box_order
    view.add(box)
    _tracts(viewer, renderer, _sheet((0.0, 0.0, 0.0), 0.1))
    # Looking down from above (+z, the camera about 15 mm up): a 10 mm box at
    # z = 7 lies between it and the tracts, one at z = -7 behind them. Zoomed out,
    # the lines are dense enough on screen to cover the centre pixel.
    viewer.set_camera(ViewKind.THREE_D, CameraPreset.AXIAL)
    viewer.zoom(ViewKind.THREE_D, 0.25)
    rgb = _rgb(view, _image(view), np.zeros(3))
    assert rgb == ([WHITE, 0, 0] if seen else [WHITE] * 3)
