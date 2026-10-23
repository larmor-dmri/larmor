"""Streamlines ray traced through their voxel grid in the 3D view."""

from collections.abc import Callable

import numpy as np
import pytest
from render_helpers import VIEWS_2D, _image, _mesh, _tracts

from larmor.types import Tractogram
from larmor.viewer import CameraPreset, OptionName, TractStyle, Viewer, ViewKind
from larmor.viewer.render.line_voxels import LineVoxels
from larmor.viewer.render.ray_traced_lines import RayTracedLines
from larmor.viewer.render.scene_renderer import SceneRenderer
from larmor.viewer.render.view import ViewRenderer


@pytest.fixture
def view_size() -> int:
    return 128


def _sheet() -> Tractogram:
    # Straight streamlines side by side along x, a few mm apart, in the plane y = 0.
    t = np.linspace(-20, 20, 9)
    lines = [np.stack([t, 0 * t, 0 * t + z], -1) for z in np.arange(-12, 13, 3.0)]
    points = np.concatenate(lines).astype(np.float32)
    return Tractogram(points, np.arange(0, len(points) + 1, 9))


def _cross(front: float) -> Tractogram:
    # One streamline along x at y = front, one along z at y = -front.
    t = np.linspace(-20, 20, 9)
    along_x = np.stack([t, 0 * t + front, 0 * t], -1)
    along_z = np.stack([0 * t, 0 * t - front, t], -1)
    points = np.concatenate([along_x, along_z]).astype(np.float32)
    return Tractogram(points, np.array([0, 9, 18]))


def _lit(image: np.ndarray, floor: int = 20) -> np.ndarray:
    return image[..., :3].max(axis=-1) > floor


def test_ray_tracing_replaces_the_3d_lines_and_tubes_only(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _sheet())
    item = adapter.item.id
    viewer.set(item, OptionName.STYLE, TractStyle.TUBES)
    viewer.set(item, OptionName.RAY_TRACED, True)
    traced = adapter.traced
    assert traced is not None and traced.visible
    assert traced.parent is views[ViewKind.THREE_D].scene
    assert adapter.tubes is None  # not made while the lines are ray traced
    assert not adapter.lines[ViewKind.THREE_D].visible
    for kind in VIEWS_2D:
        assert traced not in views[kind].scene.children
    viewer.set(item, OptionName.RAY_TRACED, False)
    assert adapter.traced is None and traced.parent is None
    assert adapter.tubes is not None and adapter.tubes.visible


def test_the_ray_traced_capsules_cover_the_pixels_of_the_tubes(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _sheet())
    item = adapter.item.id
    viewer.set_camera(ViewKind.THREE_D, CameraPreset.CORONAL)
    viewer.rotate_camera(25.0, 15.0)
    viewer.set(item, OptionName.TUBE_RADIUS, 0.8)
    viewer.set(item, OptionName.STYLE, TractStyle.TUBES)
    # A box in front of part of the sheet: both must hide behind it alike.
    views[ViewKind.THREE_D].add(_mesh((-8.0, -15.0, 0.0)))
    tubes = _lit(_image(views[ViewKind.THREE_D]))
    viewer.set(item, OptionName.RAY_TRACED, True)
    traced = _lit(_image(views[ViewKind.THREE_D]))
    union = (tubes | traced).sum()
    assert union > 500
    assert (tubes & traced).sum() / union > 0.9


@pytest.mark.parametrize("front", [-6.0, 6.0])
def test_opaque_lines_hide_the_ones_behind_and_translucent_ones_blend_them(
    front: float,
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _cross(front))
    item = adapter.item.id
    viewer.set_camera(ViewKind.THREE_D, CameraPreset.CORONAL)
    viewer.set(item, OptionName.TUBE_RADIUS, 1.5)
    viewer.set(item, OptionName.RAY_TRACED, True)
    view = views[ViewKind.THREE_D]
    camera = np.asarray(view.camera.world.position)
    # By direction: red along x, blue along z; the nearer one is in front.
    x_in_front = abs(camera[1] - front) < abs(camera[1] + front)
    seen, hidden = (0, 2) if x_in_front else (2, 0)
    centre = view.canvas.get_physical_size()[1] // 2
    opaque = _image(view)[centre - 1 : centre + 2, centre - 1 : centre + 2, :3]
    # The lamp's highlight is white: the hidden colour's channel is low, not 0.
    assert opaque[..., hidden].mean() < 0.25 * opaque[..., seen].mean()
    viewer.set(item, OptionName.OPACITY_3D, 0.5)
    blended = _image(view)[centre - 1 : centre + 2, centre - 1 : centre + 2, :3]
    assert blended[..., hidden].mean() > 2 * opaque[..., hidden].mean() + 20
    assert blended[..., seen].mean() > blended[..., hidden].mean()


def _counted(method: Callable[..., object], calls: list[str]) -> Callable[..., object]:
    def spy(*args: object, **kwargs: object) -> object:
        calls.append(method.__name__)
        return method(*args, **kwargs)

    return spy


def test_the_grid_is_built_once_and_options_reupload_nothing(
    scene_renderer: tuple[Viewer, SceneRenderer], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    for owner, name in (
        (LineVoxels, "build"),
        (LineVoxels, "light_at_points"),
        (RayTracedLines, "set_colours"),
    ):
        monkeypatch.setattr(owner, name, _counted(getattr(owner, name), calls))
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _sheet())
    item = adapter.item.id
    viewer.set(item, OptionName.RAY_TRACED, True)
    viewer.set(item, OptionName.AMBIENT_OCCLUSION, True)
    assert calls.count("build") == 1 and calls.count("light_at_points") == 1
    assert calls.count("set_colours") == 1
    calls.clear()
    for opacity in (0.9, 0.5, 1.0):
        viewer.set(item, OptionName.OPACITY_3D, opacity)
    assert calls == []


def test_the_lamp_light_is_0_in_empty_voxels_after_a_rebuild(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _sheet())
    item = adapter.item.id
    viewer.set(item, OptionName.RAY_TRACED, True)
    viewer.set(item, OptionName.TUBE_RADIUS, 2.0)
    _image(views[ViewKind.THREE_D])  # the shadow pass runs before each frame
    viewer.set(item, OptionName.TUBE_RADIUS, 0.3)
    _image(views[ViewKind.THREE_D])
    voxels, traced = adapter._voxels, adapter.traced
    assert voxels is not None and traced is not None
    occupancy = voxels.read(voxels.occupancy)[: voxels.size**3]
    shadow = voxels.read(traced.material.buffers["shadow"])[: voxels.size**3]
    assert (shadow[occupancy > 0] > 0).any()
    assert np.all(shadow[occupancy <= 0] == 0)


def test_lists_too_large_for_the_gpu_leave_lines_and_say_so_once(
    scene_renderer: tuple[Viewer, SceneRenderer], monkeypatch: pytest.MonkeyPatch
) -> None:
    init = LineVoxels.__init__

    def small(self: LineVoxels, *args: object, **kw: object) -> None:
        init(self, *args, **kw)  # type: ignore[arg-type]
        self.max_list_bytes = 64

    monkeypatch.setattr(LineVoxels, "__init__", small)
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _sheet())
    notices: list[str] = []
    adapter.tell = notices.append
    item = adapter.item.id
    viewer.set(item, OptionName.RAY_TRACED, True)
    viewer.set(item, OptionName.OPACITY_3D, 0.5)
    assert adapter.traced is None or not adapter.traced.visible
    assert adapter.lines[ViewKind.THREE_D].visible
    assert len(notices) == 1 and "ray tracing" in notices[0]


def test_both_options_off_let_the_gpu_grids_and_the_ray_tracer_go(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _sheet())
    item = adapter.item.id
    viewer.set(item, OptionName.AMBIENT_OCCLUSION, True)
    viewer.set(item, OptionName.RAY_TRACED, True)
    traced = adapter.traced
    assert adapter._voxels is not None and traced is not None
    viewer.set(item, OptionName.RAY_TRACED, False)
    assert adapter._voxels is not None  # still lighting the lines
    viewer.set(item, OptionName.AMBIENT_OCCLUSION, False)
    assert adapter._voxels is None and adapter.traced is None
    assert traced.parent is None and adapter.light is None
    assert adapter.lines[ViewKind.THREE_D].visible


def test_ray_tracing_after_ambient_occlusion_builds_the_segment_lists(
    scene_renderer: tuple[Viewer, SceneRenderer],
    views: dict[ViewKind, ViewRenderer],
) -> None:
    # Ambient occlusion alone builds the grid without lists; ray tracing then
    # needs them.
    viewer, renderer = scene_renderer
    adapter = _tracts(viewer, renderer, _sheet())
    item = adapter.item.id
    viewer.set_camera(ViewKind.THREE_D, CameraPreset.CORONAL)
    viewer.rotate_camera(25.0, 15.0)
    viewer.set(item, OptionName.TUBE_RADIUS, 0.8)
    viewer.set(item, OptionName.STYLE, TractStyle.TUBES)
    viewer.set(item, OptionName.AMBIENT_OCCLUSION, True)
    # Shaded darker in hidden places: any colour counts.
    tubes = _lit(_image(views[ViewKind.THREE_D]), floor=3)
    viewer.set(item, OptionName.RAY_TRACED, True)
    traced = _lit(_image(views[ViewKind.THREE_D]), floor=3)
    union = (tubes | traced).sum()
    assert union > 500
    assert (tubes & traced).sum() / union > 0.9


def test_streamlines_of_one_point_dim_without_tubes_to_colour(
    scene_renderer: tuple[Viewer, SceneRenderer],
) -> None:
    viewer, renderer = scene_renderer
    points = np.array([[0, 0, 0], [5, 5, 5], [9, 1, 2]], np.float32)
    adapter = _tracts(viewer, renderer, Tractogram(points, np.arange(4)))
    item = adapter.item.id
    viewer.set(item, OptionName.STYLE, TractStyle.TUBES)
    viewer.set(item, OptionName.AMBIENT_OCCLUSION, True)
    viewer.set(item, OptionName.OPACITY_3D, 0.5)
    assert adapter.light is not None
