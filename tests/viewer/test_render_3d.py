"""The 3D view: surfaces, background work, objects and rendering."""

import importlib.resources
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import cast

import numpy as np
import pygfx as gfx
import pytest
from render_helpers import (
    WHITE,
    _adapter,
    _image,
    _ndc,
    _palette,
    _rotated,
    _series,
    _volume,
)
from rendercanvas import BaseRenderCanvas

from larmor.types import Volume
from larmor.viewer import (
    CameraPreset,
    Category,
    Colormap,
    Item,
    OptionName,
    Viewer,
    ViewKind,
)
from larmor.viewer.model.item import VIEW_KINDS
from larmor.viewer.render import (
    overlay_slice,
    tissue_slice,
    volume_3d,
)
from larmor.viewer.render.scene_renderer import SceneRenderer
from larmor.viewer.render.surfaces import LabelMesh, mask_surface
from larmor.viewer.render.view import Vec3, ViewRenderer
from larmor.viewer.render.volume_3d import Volume3D
from larmor.viewer.render.volume_adapter import (
    COLORMAP_MAPS,
    VolumeAdapter,
)
from larmor.viewer.render.volume_blend import (
    Gradient,
    VolumeBlendMaterial,
    volume_gradient,
)
from larmor.viewer.render.worker import Worker

pytestmark = pytest.mark.usefixtures("fresh_slab_budget")


@pytest.fixture
def view_size() -> int:
    return 96


def _ball(
    centre: tuple[float, float, float], radius: float, shape: int = 16
) -> np.ndarray:
    grid = np.indices((shape, shape, shape)).transpose(1, 2, 3, 0)
    inside: np.ndarray = np.linalg.norm(grid - np.array(centre), axis=-1) <= radius
    return inside.astype(np.uint8)


def test_a_mask_surface_is_closed_and_faces_outwards_where_the_grid_cuts_it() -> None:
    # The ball runs off the grid at x = 0: the border closes it half a voxel out.
    centre = np.array([2.0, 8.0, 8.0])
    surface = mask_surface(_ball((2.0, 8.0, 8.0), 4.0), 0.5)
    assert surface is not None
    v, f, n = surface.vertices, surface.faces, surface.normals
    assert v.dtype == np.float32 and n.dtype == np.float32 and f.dtype == np.int32
    assert np.allclose(v.min(axis=0), [-0.5, 3.5, 3.5])
    assert np.allclose(v.max(axis=0), [6.5, 12.5, 12.5])
    # Closed and consistently wound: each directed edge once, and its reverse too.
    edges = {
        tuple(e) for e in np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    }
    assert len(edges) == 3 * len(f)
    assert all((b, a) in edges for a, b in edges)
    # Counter-clockwise seen from outside, which pygfx takes as the front.
    wound = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    assert np.all(np.einsum("ij,ij->i", wound, v[f].mean(axis=1) - centre) > 0)
    assert np.all(np.einsum("ij,ij->i", n, v - centre) > 0)


@pytest.mark.parametrize("level", [0.0, 1.0, -1.0])
def test_a_level_the_data_never_crosses_has_no_surface(level: float) -> None:
    assert mask_surface(_ball((8.0, 8.0, 8.0), 4.0), level) is None


class _Loop:
    def __init__(self) -> None:
        self.calls: list[tuple[Callable[..., None], tuple[object, ...]]] = []

    def call_soon_threadsafe(self, f: Callable[..., None], *args: object) -> None:
        self.calls.append((f, args))

    def run(self) -> None:
        for f, args in self.calls:
            f(*args)


def _canvas(loop: _Loop) -> BaseRenderCanvas:
    group = SimpleNamespace(get_loop=lambda: loop)
    return cast(BaseRenderCanvas, SimpleNamespace(_rc_canvas_group=group))


def test_the_worker_hands_back_only_the_newest_request() -> None:
    loop = _Loop()
    worker = Worker(_canvas(loop))
    started, release = threading.Event(), threading.Event()
    ran: list[str] = []
    done: list[str] = []

    def work(name: str) -> Callable[[], str]:
        def run() -> str:
            if name == "first":
                started.set()
                release.wait(5)
            ran.append(name)
            return name

        return run

    # The first runs and finishes after the others were asked for; the second
    # is replaced before it starts.
    worker.submit(work("first"), done.append)
    # Other tests' work may still hold the shared thread.
    assert started.wait(5)
    for name in ("second", "third"):
        worker.submit(work(name), done.append)
    release.set()
    # One thread runs every worker's requests in turn: one more marks the end.
    end = threading.Event()
    Worker(_canvas(_Loop())).submit(end.set, lambda _: None)
    assert end.wait(5)
    loop.run()
    assert ran == ["first", "third"]
    assert done == ["third"]


def test_a_cancelled_worker_hands_back_nothing_but_ends_every_request() -> None:
    loop = _Loop()
    pool = ThreadPoolExecutor(1)
    worker = Worker(_canvas(loop), pool)
    started, release = threading.Event(), threading.Event()
    done: list[str] = []
    ended: list[str] = []

    def first() -> str:
        started.set()
        release.wait(5)
        return "first"

    worker.submit(first, done.append, ended=lambda: ended.append("first"))
    assert started.wait(5)
    worker.submit(lambda: "second", done.append, ended=lambda: ended.append("second"))
    worker.cancel()
    release.set()
    pool.shutdown(wait=True)
    loop.run()
    # Nothing is handed back, but both requests end: a budget held for them is freed.
    assert done == []
    assert sorted(ended) == ["first", "second"]


def _3d(adapter: VolumeAdapter) -> Volume3D:
    assert adapter.three_d is not None
    return adapter.three_d


def test_opacity_3d_changes_only_the_rendering_material_and_0_hides_it(
    views: dict[ViewKind, ViewRenderer], draws: dict[ViewKind, int]
) -> None:
    viewer = Viewer()
    item = viewer.add(_volume(), Category.VOLUMES)
    three_d = _3d(_adapter(viewer, views, item))
    rendering = three_d.rendering
    assert rendering is not None
    assert isinstance(rendering.material, VolumeBlendMaterial)
    grid = rendering.geometry.grid
    draws.update(dict.fromkeys(draws, 0))
    viewer.set(item.id, OptionName.OPACITY_3D, 0.3)
    assert rendering.material.opacity == pytest.approx(0.3, rel=1e-6)
    assert rendering.visible
    assert rendering.geometry.grid is grid
    assert draws == {k: int(k == ViewKind.THREE_D) for k in VIEW_KINDS}
    viewer.set(item.id, OptionName.OPACITY_3D, 0.0)
    assert not rendering.visible


def test_the_rendering_takes_larger_steps_only_while_the_view_is_dragged(
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer = Viewer()
    item = viewer.add(_volume(), Category.VOLUMES)
    rendering = _3d(_adapter(viewer, views, item)).rendering
    assert rendering is not None
    view = views[ViewKind.THREE_D]
    view.interacting = True
    view.canvas.draw()
    assert rendering.material.step_size == volume_3d.INTERACTIVE_STEP
    view.end_interaction()
    view.canvas.draw()
    assert rendering.material.step_size == volume_3d.STEP


def test_slices_3d_off_hides_only_the_3d_slices(
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer = Viewer()
    item = viewer.add(_volume(), Category.VOLUMES)
    adapter = _adapter(viewer, views, item)
    three_d = _3d(adapter)
    assert all(o.visible for o in three_d.slices.values())
    viewer.set(item.id, OptionName.SLICES_3D, False)
    assert not any(o.visible for o in three_d.slices.values())
    assert all(o.visible for o in adapter.slices.values())
    assert three_d.rendering is not None and three_d.rendering.visible


@pytest.fixture
def levels(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    computed: list[float] = []

    def recorded(data: np.ndarray, level: float) -> LabelMesh | None:
        computed.append(level)
        return mask_surface(data, level)

    monkeypatch.setattr(volume_3d, "mask_surface", recorded)
    return computed


@pytest.mark.parametrize(("threshold", "stored"), [(None, 2.5), (9.0, 4.0)])
def test_a_surface_is_at_the_threshold_or_else_mid_range_in_stored_units(
    threshold: float | None,
    stored: float,
    views: dict[ViewKind, ViewRenderer],
    levels: list[float],
) -> None:
    viewer = Viewer()
    # Stored 0 and 5 are 1 and 11.
    item = viewer.add(_volume(slope=2.0, intercept=1.0), Category.OVERLAYS)
    if threshold is not None:
        viewer.set(item.id, OptionName.THRESHOLD, threshold)
    three_d = _3d(_adapter(viewer, views, item))
    assert levels == [stored]
    assert three_d.surface is not None and three_d.surface.parent is not None
    assert three_d.rendering is None


@pytest.mark.parametrize(
    ("name", "value", "recomputed"),
    [
        (OptionName.THRESHOLD, 0.8, True),
        (OptionName.VOLUME_INDEX, 1, True),
        (OptionName.OPACITY_3D, 0.4, False),
        (OptionName.COLORMAP, Colormap.HOT, False),
    ],
)
def test_a_surface_is_computed_again_only_for_a_new_volume_or_level(
    name: OptionName,
    value: object,
    recomputed: bool,
    views: dict[ViewKind, ViewRenderer],
    levels: list[float],
) -> None:
    viewer = Viewer()
    item = viewer.add(_series(), Category.OVERLAYS)
    # Volume 0 holds 0 and 1, below the middle of the item's range.
    viewer.set(item.id, OptionName.THRESHOLD, 0.5)
    three_d = _3d(_adapter(viewer, views, item))
    surface = three_d.surface
    assert surface is not None and len(levels) == 1
    viewer.set(item.id, name, value)
    assert len(levels) == 1 + recomputed
    assert (three_d.surface is not surface) == recomputed
    if name == OptionName.OPACITY_3D:
        assert surface.material.opacity == pytest.approx(0.4, rel=1e-6)


@pytest.mark.parametrize("before", [True, False], ids=["from-the-start", "later"])
def test_a_labels_overlay_has_no_surface_until_labels_are_off(
    before: bool, views: dict[ViewKind, ViewRenderer], levels: list[float]
) -> None:
    viewer = Viewer()
    item = viewer.add(_volume(), Category.OVERLAYS)
    if before:
        viewer.set(item.id, OptionName.LABELS, True)
    three_d = _3d(_adapter(viewer, views, item))
    viewer.set(item.id, OptionName.LABELS, True)
    assert three_d.surface is None or not three_d.surface.visible
    assert len(levels) == int(not before)
    viewer.set(item.id, OptionName.LABELS, False)
    assert three_d.surface is not None and three_d.surface.visible
    assert three_d.surface.parent is not None
    assert len(levels) == 1


def test_surface_3d_off_hides_only_the_surface_and_on_shows_it_without_recomputing(
    views: dict[ViewKind, ViewRenderer], levels: list[float]
) -> None:
    viewer = Viewer()
    item = viewer.add(_volume(), Category.OVERLAYS)
    adapter = _adapter(viewer, views, item)
    three_d = _3d(adapter)
    surface = three_d.surface
    assert surface is not None and surface.visible
    viewer.set(item.id, OptionName.SURFACE_3D, False)
    assert not surface.visible
    assert all(o.visible for o in three_d.slices.values())
    assert all(o.visible for o in adapter.slices.values())
    viewer.set(item.id, OptionName.SURFACE_3D, True)
    assert three_d.surface is surface
    assert three_d.surface.visible
    assert len(levels) == 1


def test_a_surface_of_an_old_level_stays_hidden_until_its_replacement_is_ready(
    views: dict[ViewKind, ViewRenderer], monkeypatch: pytest.MonkeyPatch
) -> None:
    viewer = Viewer()
    item = viewer.add(_volume(), Category.OVERLAYS)
    three_d = _3d(_adapter(viewer, views, item))
    old = three_d.surface
    assert old is not None and old.visible
    pending: list[Callable[[], None]] = []

    def later(
        work: Callable[[], LabelMesh | None],
        done: Callable[[LabelMesh | None], None],
        failed: Callable[[], None],
    ) -> None:
        pending.append(lambda: done(work()))

    monkeypatch.setattr(three_d._worker, "submit", later)
    viewer.set(item.id, OptionName.SURFACE_3D, False)
    viewer.set(item.id, OptionName.THRESHOLD, 3.0)
    viewer.set(item.id, OptionName.SURFACE_3D, True)
    assert not old.visible
    assert len(pending) == 1
    pending[0]()
    assert three_d.surface is not old
    assert three_d.surface is not None and three_d.surface.visible


def test_a_failed_surface_is_computed_again_when_next_asked_for(
    views: dict[ViewKind, ViewRenderer],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    computed: list[float] = []

    def fails_once(data: np.ndarray, level: float) -> LabelMesh | None:
        computed.append(level)
        if len(computed) == 1:
            raise RuntimeError("no surface")
        return mask_surface(data, level)

    monkeypatch.setattr(volume_3d, "mask_surface", fails_once)
    viewer = Viewer()
    item = viewer.add(_volume(), Category.OVERLAYS)
    viewer.set(item.id, OptionName.SURFACE_3D, False)
    three_d = _3d(_adapter(viewer, views, item))
    # Off screen the work runs at once; the scene logs its error.
    viewer.set(item.id, OptionName.SURFACE_3D, True)
    assert "no surface" in caplog.text
    assert three_d.surface is None
    viewer.set(item.id, OptionName.OPACITY_3D, 0.5)
    assert computed == [2.5, 2.5]
    assert three_d.surface is not None and three_d.surface.visible


def test_the_worker_hands_a_failure_to_failed_on_the_loop() -> None:
    loop = _Loop()
    worker = Worker(_canvas(loop))
    done: list[object] = []
    failed: list[bool] = []

    def work() -> object:
        raise RuntimeError("no surface")

    worker.submit(work, done.append, lambda: failed.append(True))
    end = threading.Event()
    Worker(_canvas(_Loop())).submit(end.set, lambda _: None)
    assert end.wait(5)
    assert failed == []  # not on the worker's thread
    loop.run()
    assert failed == [True]
    assert done == []


def _mirrored() -> Volume:
    # The block of _volume() at the same world place, its x axis flipped.
    plain = _volume()
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    affine[0, 3] = plain.data.shape[0] - 1
    return Volume(np.asfortranarray(plain.data[::-1]), affine)


def test_a_mirrored_volume_renders_as_the_same_volume_unmirrored(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    view = views[ViewKind.THREE_D]
    images = []
    for volume in (_volume(), _mirrored()):
        item = _rendered_only(viewer, volume)
        images.append(_image(view)[..., :3].astype(int))
        viewer.remove(item.id)
    plain, mirrored = images
    assert mirrored.max() > 30
    # The rays sample the same texels, only in another order along the grid.
    assert np.abs(mirrored - plain).max() <= 2


def _flipped_labels(size: int) -> Volume:
    # Label 3 on every voxel of an identity grid of this size, stored with x and y
    # reversed: pygfx splits its slices into other triangles than the volume's.
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine[:2, 3] = size - 1
    return Volume(np.full((size, size, size), 3, np.int16), affine)


@pytest.mark.parametrize(("right", "up"), [(0.0, 0.0), (35.0, 20.0), (-70.0, -40.0)])
def test_an_overlay_on_a_flipped_grid_covers_the_volume_slices_in_3d(
    right: float,
    up: float,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    size = 16
    # White slices only: no volume rendering behind them.
    volume = _rendered_only(viewer, _dense(size), opacity=0.0)
    viewer.set(volume.id, OptionName.SLICES_3D, True)
    overlay = viewer.add(_flipped_labels(size), Category.OVERLAYS)
    viewer.set(overlay.id, OptionName.LABELS, True)
    viewer.move_cursor((7.3, 8.4, 7.7))
    viewer.rotate_camera(right, up)
    rgb = _image(views[ViewKind.THREE_D])[..., :3].astype(float).reshape(-1, 3)
    colour = _palette(3)[:3] * 255
    assert (np.abs(rgb - colour).max(axis=1) <= 2).mean() > 0.1
    # The slices' edges blend the overlay with the black background, which only
    # scales its colour; any of the volume's white would tint it towards grey.
    shade = np.clip(rgb @ colour / (colour @ colour), 0.0, 1.0)
    off = np.abs(rgb - shade[:, None] * colour).max(axis=1)
    assert off.max() <= 12, rgb[off > 12][:5]


def _overlay_slice_shader() -> overlay_slice.OverlaySliceShader:
    grid = gfx.Texture(np.zeros((2, 2, 2), np.float32), dim=3)
    volume = gfx.Volume(gfx.Geometry(grid=grid), overlay_slice.OverlaySliceMaterial())
    return overlay_slice.OverlaySliceShader(volume)


def test_the_overlay_slice_shader_moves_each_vertex_towards_the_camera() -> None:
    code = _overlay_slice_shader().get_code()
    # Times w, so the offset survives the perspective divide.
    offset = "ndc_pos.z - 0.0001 * ndc_pos.w"
    assert f"vec4<f32>(ndc_pos.xy, {offset}, ndc_pos.w);" in code
    assert "varyings.position = vec4<f32>(ndc_pos);" not in code


def test_the_overlay_slice_shader_fails_if_pygfx_changed_the_line_it_replaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shader = _overlay_slice_shader()
    monkeypatch.setattr(
        overlay_slice, "load_wgsl", lambda name: "varyings.position = ndc_pos;"
    )
    with pytest.raises(RuntimeError, match="volume_slice.wgsl"):
        shader.get_code()


def test_the_3d_view_frames_the_field_of_view_and_keeps_it_as_items_come(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    viewer.add(_volume(affine=_rotated()), Category.VOLUMES)
    lo, hi = (np.array(p) for p in viewer.scene.view.field_of_view)
    centre = (lo + hi) / 2
    # Framed tighter than the sphere round the field, which overflows the view;
    # its corners may leave it too, but the middle of each face stays on it.
    faces = [
        np.where(np.arange(3) == axis, end, centre)
        for axis in range(3)
        for end in (lo, hi)
    ]
    view = views[ViewKind.THREE_D]
    assert np.allclose(_ndc(view, centre), 0.0, atol=1e-6)
    assert all(np.all(np.abs(_ndc(view, f)) < 1.0) for f in faces)
    camera = viewer.scene.view.cameras.three_d
    right = np.cross(camera.forward, camera.up)
    radius = np.linalg.norm(hi - lo) / 2
    assert abs(_ndc(view, centre + radius * right)[0]) > 1.0
    before = view.camera.get_state()
    viewer.add(_volume(affine=np.diag([9.0, 9.0, 9.0, 1.0])), Category.OVERLAYS)
    after = view.camera.get_state()
    for key, value in before.items():
        assert np.array_equal(np.asarray(after[key]), np.asarray(value)), key


@pytest.mark.parametrize(
    ("preset", "right", "up"),
    [
        (CameraPreset.AXIAL, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        (CameraPreset.CORONAL, (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        (CameraPreset.SAGITTAL, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ],
)
def test_the_3d_camera_shows_a_preset_and_its_zoom(
    preset: CameraPreset,
    right: Vec3,
    up: Vec3,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    viewer.add(_volume(), Category.VOLUMES)
    lo, hi = (np.array(p) for p in viewer.scene.view.field_of_view)
    centre = (lo + hi) / 2
    view = views[ViewKind.THREE_D]
    viewer.set_camera(ViewKind.THREE_D, preset)
    on_right = _ndc(view, centre + 2.0 * np.array(right))
    on_top = _ndc(view, centre + 2.0 * np.array(up))
    assert on_right[0] > 0 and on_right[1] == pytest.approx(0.0, abs=1e-6)
    assert on_top[1] > 0 and on_top[0] == pytest.approx(0.0, abs=1e-6)
    viewer.zoom(ViewKind.THREE_D, 2.0)
    assert np.allclose(_ndc(view, centre + 2.0 * np.array(right)), 2 * on_right)


def test_a_rendered_3d_view_shows_the_volume_until_its_opacity_is_0(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    item = viewer.add(_volume(), Category.VOLUMES)
    # Only the volume rendering: no slices.
    viewer.set(item.id, OptionName.SLICES_3D, False)
    view = views[ViewKind.THREE_D]
    # Blended, a small block is dimmer than its brightest voxel, but not black.
    assert _image(view)[..., :3].max() > 30
    viewer.set(item.id, OptionName.OPACITY_3D, 0.0)
    assert _image(view)[..., :3].max() == 0


def test_an_opaque_object_behind_a_see_through_surface_stays_visible(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    mask = Volume(_ball((8.0, 8.0, 8.0), 4.0).astype(np.float32), np.eye(4))
    item = viewer.add(mask, Category.OVERLAYS)
    viewer.set(item.id, OptionName.SLICES_3D, False)
    viewer.set(item.id, OptionName.OPACITY_3D, 0.3)
    view = views[ViewKind.THREE_D]
    # A red block, standing for a tract, beyond the ball as the camera sees it.
    forward = np.array(viewer.scene.view.cameras.three_d.forward)
    behind = np.array([8.0, 8.0, 8.0]) + 6.0 * forward
    tract = gfx.Mesh(
        gfx.box_geometry(3, 3, 3), gfx.MeshBasicMaterial(color=(1.0, 0.0, 0.0))
    )
    tract.local.position = behind
    view.add(tract)
    image = _image(view)
    height, width = image.shape[:2]
    red, green, _ = (int(c) for c in image[height // 2, width // 2, :3])
    # Through the ball's surface (grey), the block still tints the pixel red.
    assert red > green + 60


def _rendering(adapter: VolumeAdapter) -> gfx.Volume:
    rendering = _3d(adapter).rendering
    assert rendering is not None
    return rendering


@pytest.mark.parametrize("colormap", [Colormap.GRAY, Colormap.HOT, Colormap.VIRIDIS])
def test_the_volume_rendering_takes_the_item_colormap(
    colormap: Colormap, views: dict[ViewKind, ViewRenderer]
) -> None:
    viewer = Viewer()
    item = viewer.add(_volume(), Category.VOLUMES)
    rendering = _rendering(_adapter(viewer, views, item))
    viewer.set(item.id, OptionName.COLORMAP, colormap)
    assert rendering.material.map is COLORMAP_MAPS[colormap]


@pytest.mark.parametrize(
    ("slope", "intercept", "minimum", "maximum", "clim"),
    [
        (1.0, 0.0, 1.0, 4.0, (1.0, 4.0)),
        (2.0, 10.0, 14.0, 30.0, (2.0, 10.0)),
        (-0.5, 1.0, -1.5, 2.0, (5.0, -2.0)),
    ],
)
def test_the_volume_rendering_limits_are_minimum_and_maximum_in_stored_units(
    slope: float,
    intercept: float,
    minimum: float,
    maximum: float,
    clim: tuple[float, float],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer = Viewer()
    item = viewer.add(_volume(slope, intercept), Category.VOLUMES)
    rendering = _rendering(_adapter(viewer, views, item))
    viewer.set(item.id, OptionName.MINIMUM, minimum)
    viewer.set(item.id, OptionName.MAXIMUM, maximum)
    assert rendering.material.clim == pytest.approx(clim)


def test_the_volume_rendering_shader_ships_next_to_its_module() -> None:
    package = importlib.resources.files("larmor.viewer.render")
    assert package.joinpath("volume_blend.wgsl").is_file()


def _true_gradient(values: np.ndarray) -> np.ndarray:
    # Central differences in the texture's x, y, z (axes 2, 1, 0); np.gradient
    # takes the full one-sided difference at an edge, the rule there is half of it.
    out = []
    for axis in (2, 1, 0):
        g = np.gradient(values.astype(np.float64), axis=axis)
        edge: list[int | slice] = [slice(None)] * 3
        for end in (0, -1):
            edge[axis] = end
            g[tuple(edge)] /= 2
        out.append(g)
    return np.stack(out, axis=-1)


def test_the_volume_gradient_is_central_differences_scaled_into_float16() -> None:
    values = np.random.default_rng(0).uniform(-300, 900, (4, 5, 6)).astype(np.float32)
    gradient = volume_gradient(values)
    stored = np.asarray(gradient.texture.data)
    assert stored.dtype == np.float16 and stored.shape == (4, 5, 6, 4)
    expected = _true_gradient(values)
    sizes = np.abs(expected)
    assert gradient.scale == pytest.approx(np.median(sizes[sizes > 0]), rel=1e-6)
    assert not stored[..., 3].any()
    # float16 keeps about 3 decimal digits of the scaled values, which are about 1.
    real = stored[..., :3].astype(np.float64) * gradient.scale
    assert np.allclose(real, expected, rtol=2e-3, atol=1e-3 * gradient.scale)


def test_one_outlier_voxel_does_not_flatten_the_gradient_everywhere_else() -> None:
    # An MD-like map, 0.8e-3 in a sphere and 3e-3 around it, with one failed fit.
    z, y, x = np.indices((20, 20, 20))
    inside = (x - 10) ** 2 + (y - 10) ** 2 + (z - 10) ** 2 < 36
    values = np.where(inside, 0.8e-3, 3e-3).astype(np.float32)
    values[1, 1, 1] = 1e5
    stored = np.asarray(volume_gradient(values).texture.data)[..., :3]
    assert np.isfinite(stored).all()
    rim = _true_gradient(values) != 0
    rim[:4, :4, :4] = False  # the outlier's own neighbourhood saturates
    assert (stored[rim] != 0).all()


def test_a_zero_volume_has_a_zero_gradient_at_scale_1() -> None:
    gradient = volume_gradient(np.zeros((3, 3, 3), np.float32))
    assert gradient.scale == 1.0
    assert not np.asarray(gradient.texture.data).any()


def _gradient_scale(material: VolumeBlendMaterial) -> float:
    return float(material.uniform_buffer.data["gradient_scale"])


def test_the_rendering_is_unlit_until_it_has_a_gradient() -> None:
    material = VolumeBlendMaterial()
    assert material.gradient is None
    assert _gradient_scale(material) == 1.0


def test_a_gradient_sets_its_scale_in_the_material_uniforms() -> None:
    material = VolumeBlendMaterial()
    gradient = volume_gradient(np.arange(27, dtype=np.float32).reshape(3, 3, 3))
    material.gradient = gradient
    assert material.gradient is gradient
    assert _gradient_scale(material) == pytest.approx(gradient.scale)
    assert gradient.scale != 1.0
    material.gradient = None
    assert _gradient_scale(material) == 1.0


@pytest.fixture
def gradients(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    asked: list[np.ndarray] = []

    def counted(values: np.ndarray) -> Gradient:
        asked.append(values)
        return volume_gradient(values)

    monkeypatch.setattr(volume_3d, "volume_gradient", counted)
    return asked


def _material(adapter: VolumeAdapter) -> VolumeBlendMaterial:
    material = _rendering(adapter).material
    assert isinstance(material, VolumeBlendMaterial)
    return material


def test_the_gradient_is_computed_once_when_the_rendering_is_first_shown(
    views: dict[ViewKind, ViewRenderer], gradients: list[np.ndarray]
) -> None:
    viewer = Viewer()
    item = viewer.add(_volume(), Category.VOLUMES)
    viewer.set(item.id, OptionName.OPACITY_3D, 0.0)
    adapter = _adapter(viewer, views, item)
    assert gradients == []
    assert _material(adapter).gradient is None
    viewer.set(item.id, OptionName.OPACITY_3D, 0.5)
    viewer.set(item.id, OptionName.OPACITY_3D, 0.7)
    assert len(gradients) == 1
    assert _material(adapter).gradient is not None


class _Held:
    def __init__(self, *args: object) -> None:
        self.queue: list[tuple[Callable[[], object], Callable[[object], None]]] = []

    def submit(
        self, work: Callable[[], object], done: Callable[[object], None], *_: object
    ) -> None:
        self.queue.append((work, done))

    def close(self) -> None:
        pass


def _held_gradients(
    views: dict[ViewKind, ViewRenderer], monkeypatch: pytest.MonkeyPatch
) -> tuple[Viewer, Item, VolumeAdapter, _Held]:
    monkeypatch.setattr(volume_3d, "Worker", _Held)
    viewer = Viewer()
    item = viewer.add(_series(), Category.VOLUMES)
    adapter = _adapter(viewer, views, item)
    worker = _3d(adapter)._gradient_worker
    assert isinstance(worker, _Held)
    return viewer, item, adapter, worker


def test_a_new_volume_index_keeps_the_old_lighting_until_its_own_is_ready(
    views: dict[ViewKind, ViewRenderer], monkeypatch: pytest.MonkeyPatch
) -> None:
    viewer, item, adapter, worker = _held_gradients(views, monkeypatch)
    work, done = worker.queue[0]
    done(work())
    old = _material(adapter).gradient
    assert old is not None
    viewer.set(item.id, OptionName.VOLUME_INDEX, 2)
    assert _material(adapter).gradient is old  # no flash unlit while stepping
    assert len(worker.queue) == 2
    work, done = worker.queue[1]
    done(work())
    gradient = _material(adapter).gradient
    assert gradient is not None
    expected = volume_gradient(_series().data[..., 2].T)
    assert gradient.scale == expected.scale
    assert np.array_equal(gradient.texture.data, expected.texture.data)


def test_a_gradient_of_a_volume_no_longer_shown_is_ignored(
    views: dict[ViewKind, ViewRenderer], monkeypatch: pytest.MonkeyPatch
) -> None:
    viewer, item, adapter, worker = _held_gradients(views, monkeypatch)
    viewer.set(item.id, OptionName.VOLUME_INDEX, 1)
    (old_work, old_done), (work, done) = worker.queue
    old_done(old_work())
    assert _material(adapter).gradient is None
    done(work())
    assert _material(adapter).gradient is not None


def test_a_failed_gradient_is_asked_for_again_when_next_shown(
    views: dict[ViewKind, ViewRenderer],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    asked: list[int] = []

    def fails_once(values: np.ndarray) -> Gradient:
        asked.append(1)
        if len(asked) == 1:
            raise RuntimeError("no gradient")
        return volume_gradient(values)

    monkeypatch.setattr(volume_3d, "volume_gradient", fails_once)
    viewer = Viewer()
    item = viewer.add(_volume(), Category.VOLUMES)
    viewer.set(item.id, OptionName.OPACITY_3D, 0.0)
    adapter = _adapter(viewer, views, item)
    # Off screen the work runs at once; the scene logs its error.
    viewer.set(item.id, OptionName.OPACITY_3D, 0.5)
    assert "no gradient" in caplog.text
    assert _material(adapter).gradient is None
    viewer.set(item.id, OptionName.OPACITY_3D, 0.7)
    assert len(asked) == 2
    assert _material(adapter).gradient is not None


def _rendered_only(viewer: Viewer, volume: Volume, opacity: float = 1.0) -> Item:
    item = viewer.add(volume, Category.VOLUMES)
    viewer.set(item.id, OptionName.SLICES_3D, False)
    viewer.set(item.id, OptionName.OPACITY_3D, opacity)
    return item


def test_values_at_or_below_the_minimum_are_transparent(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    item = _rendered_only(viewer, _volume())
    view = views[ViewKind.THREE_D]
    assert _image(view)[..., :3].max() > 0
    # The block's 5 becomes the minimum, so every voxel is at or below it.
    viewer.set(item.id, OptionName.MAXIMUM, 10.0)
    viewer.set(item.id, OptionName.MINIMUM, 5.0)
    assert _image(view)[..., :3].max() == 0


def _dense(size: int = 32) -> Volume:
    # All at the maximum but one corner voxel, which gives the item a range.
    data = np.ones((size, size, size), np.float32, order="F")
    data[0, 0, 0] = 0.0
    return Volume(data, np.eye(4))


def test_a_higher_opacity_renders_brighter(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    item = _rendered_only(viewer, _dense(16), opacity=0.2)
    view = views[ViewKind.THREE_D]
    dim = _image(view)[..., :3].mean()
    viewer.set(item.id, OptionName.OPACITY_3D, 1.0)
    bright = _image(view)[..., :3].mean()
    assert dim > 0
    assert bright > 1.5 * dim


def _redness_of_a_block_at(
    viewer: Viewer, view: ViewRenderer, depth: float, with_volume: bool
) -> int:
    # A red opaque block on the camera's axis, depth voxels along it past the point
    # where the axis enters the 32-voxel volume (negative: in front of it).
    _rendered_only(viewer, _dense(), opacity=1.0 if with_volume else 0.0)
    forward = np.array(viewer.scene.view.cameras.three_d.forward)
    entry = 16.0 / np.abs(forward).max()
    block = gfx.Mesh(
        gfx.box_geometry(3, 3, 3),
        gfx.MeshBasicMaterial(color=(1.0, 0.0, 0.0), alpha_mode="solid"),
    )
    block.local.position = np.full(3, 15.5) + (depth - entry) * forward
    view.add(block)
    image = _image(view)
    height, width = image.shape[:2]
    red, green, _ = (int(c) for c in image[height // 2, width // 2, :3])
    return red - green


@pytest.mark.parametrize("with_volume", [True, False], ids=["volume", "none"])
@pytest.mark.parametrize("depth", [-6.0, 24.0], ids=["in-front", "inside"])
def test_an_opaque_block_is_hidden_only_by_the_volume_in_front_of_it(
    depth: float,
    with_volume: bool,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    redness = _redness_of_a_block_at(
        viewer, views[ViewKind.THREE_D], depth, with_volume
    )
    # More than one slab of dense tissue in front lets little of the block through.
    hidden = with_volume and depth > 16.0
    assert (redness < WHITE // 4) == hidden, redness


def _red_behind_the_slices(
    viewer: Viewer,
    view: ViewRenderer,
    data: np.ndarray,
    minimum: float | None = None,
    options: tuple[tuple[OptionName, object], ...] = (),
) -> int:
    # The slices only, and a red block behind them where they cross on the camera's
    # axis: how red the centre is, high where the slices leave their voxel out.
    item = _rendered_only(viewer, Volume(data, np.eye(4)), opacity=0.0)
    viewer.set(item.id, OptionName.SLICES_3D, True)
    if minimum is not None:
        viewer.set(item.id, OptionName.MAXIMUM, minimum + 1.0)
        viewer.set(item.id, OptionName.MINIMUM, minimum)
    for name, value in options:
        viewer.set(item.id, name, value)
    forward = np.array(viewer.scene.view.cameras.three_d.forward)
    block = gfx.Mesh(
        gfx.box_geometry(3, 3, 3),
        gfx.MeshBasicMaterial(color=(1.0, 0.0, 0.0), alpha_mode="solid"),
    )
    block.local.position = np.array(viewer.scene.view.cursor) + 8.0 * forward
    view.add(block)
    image = _image(view)
    height, width = image.shape[:2]
    red, green, _ = (int(c) for c in image[height // 2, width // 2, :3])
    return red - green


# Zero, faint noise above it, faint ringing below it, and zero in an image almost
# all zero (its 1st and 99th percentiles both 0).
@pytest.mark.parametrize(
    ("background", "tissue"), [(0.0, 8), (0.02, 8), (-0.001, 8), (0.0, 4)]
)
def test_the_3d_slices_leave_out_the_background_and_show_what_is_behind_it(
    background: float,
    tissue: int,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    data = np.full((32, 32, 32), background, np.float32, order="F")
    data[:tissue, :tissue, :tissue] = 1.0
    redness = _red_behind_the_slices(viewer, views[ViewKind.THREE_D], data)
    assert redness > WHITE // 2, redness


def _cornered(centre: float, rest: float) -> np.ndarray:
    # The centre's value, where the slices cross, inside a corner of tissue at 1.
    data = np.full((32, 32, 32), rest, np.float32, order="F")
    data[:8, :8, :8] = 1.0
    data[12:20, 12:20, 12:20] = centre
    return data


@pytest.mark.parametrize(
    ("data", "minimum"),
    [
        (np.ones((32, 32, 32), np.float32), None),  # an image of one value
        (_cornered(0.0, -1.0), None),  # a signed image's zero
        (_cornered(-0.5, -1.0), None),  # and its negative values
        (_cornered(1.0, 0.0), 5.0),  # tissue below a raised minimum
    ],
    ids=["constant", "signed-zero", "signed-negative", "below-minimum"],
)
def test_the_3d_slices_keep_what_is_not_background(
    data: np.ndarray,
    minimum: float | None,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    redness = _red_behind_the_slices(viewer, views[ViewKind.THREE_D], data, minimum)
    assert redness < WHITE // 4, redness


def test_the_background_cutoff_ignores_a_bright_outlier() -> None:
    values = np.zeros((10, 10, 10), np.float32)
    values[5:] = 100.0
    plain = tissue_slice.background_cutoff(values)
    values[0, 0, 0] = 1e6
    assert tissue_slice.background_cutoff(values) == pytest.approx(plain)
    assert plain == pytest.approx(tissue_slice.CUTOFF * 100.0)


@pytest.mark.parametrize(("centre", "seen"), [(0.0, True), (0.1, False)])
def test_dti_colour_slices_in_3d_leave_out_only_black(
    centre: float,
    seen: bool,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    # Black around a dim blue centre, or black there too: blue alone, so that the
    # test reads the largest of the three channels, not red.
    data = np.zeros((32, 32, 32, 3), np.float32, order="F")
    data[:8, :8, :8, 0] = 1.0
    data[12:20, 12:20, 12:20, 2] = centre
    redness = _red_behind_the_slices(
        viewer, views[ViewKind.THREE_D], data, options=((OptionName.DTI, True),)
    )
    assert (redness > WHITE // 2) == seen, redness


def test_each_volume_of_a_4d_image_has_its_own_background_cutoff(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, _ = scene_renderer
    # Volume 1's tissue is far below the cut-off of volume 0, which is shown first.
    data = np.zeros((32, 32, 32, 2), np.float32, order="F")
    data[4:28, 4:28, 4:28, 0] = 100.0
    data[4:28, 4:28, 4:28, 1] = 1.0
    redness = _red_behind_the_slices(
        viewer,
        views[ViewKind.THREE_D],
        data,
        options=((OptionName.VOLUME_INDEX, 1),),
    )
    assert redness < WHITE // 4, redness


def test_the_background_cutoff_is_found_once_per_volume(
    views: dict[ViewKind, ViewRenderer], monkeypatch: pytest.MonkeyPatch
) -> None:
    found: list[int] = []
    real = tissue_slice.background_cutoff

    def counted(values: np.ndarray) -> float:
        found.append(1)
        return real(values)

    monkeypatch.setattr(volume_3d, "background_cutoff", counted)
    viewer = Viewer()
    item = viewer.add(_series(), Category.VOLUMES)
    _adapter(viewer, views, item)
    for index in (1, 2, 0, 1):
        viewer.set(item.id, OptionName.VOLUME_INDEX, index)
    assert len(found) == 3
