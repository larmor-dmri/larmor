import math

import numpy as np
import pytest
from scipy.ndimage import map_coordinates
from viewer_helpers import record_events

from larmor.types import Tractogram, Volume
from larmor.viewer import (
    CameraPreset,
    Category,
    Convention,
    ItemId,
    OptionName,
    Readout,
    Viewer,
    ViewKind,
)
from larmor.viewer.model.camera import (
    EMPTY_FIELD,
    PRESETS,
    Camera2D,
    Camera3D,
    Cameras,
    Vec3,
    screen_axes,
)
from larmor.viewer.model.events import Event, ViewStateChanged
from larmor.viewer.model.view_state import ViewField
from larmor.viewer.model.viewer import ZOOM_RANGE

VIEWS_2D: list[ViewKind] = [ViewKind.AXIAL, ViewKind.CORONAL, ViewKind.SAGITTAL]
AXIS: dict[ViewKind, int] = {
    ViewKind.SAGITTAL: 0,
    ViewKind.CORONAL: 1,
    ViewKind.AXIAL: 2,
}


def _view_events(events: list[Event], name: ViewField) -> list[ViewStateChanged]:
    return [e for e in events if isinstance(e, ViewStateChanged) and e.name == name]


def _volume(
    shape: tuple[int, ...] = (2, 3, 4), affine: np.ndarray | None = None
) -> Volume:
    data = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    return Volume(data, np.eye(4) if affine is None else affine)


def _scaled_affine() -> np.ndarray:
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = [10.0, -20.0, 5.0]
    return affine


def _oblique() -> np.ndarray:
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    affine = np.eye(4)
    affine[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) * [
        2.0,
        1.5,
        2.5,
    ]
    affine[:3, 3] = [10.0, -20.0, 5.0]
    return affine


def _camera(viewer: Viewer, view: ViewKind) -> Camera2D:
    camera = viewer.scene.view.cameras.of(view)
    assert isinstance(camera, Camera2D)
    return camera


def _on_screen(viewer: Viewer, view: ViewKind, point: np.ndarray) -> np.ndarray:
    # Where a world point sits on the screen, in mm at zoom 1 from the centre.
    right, up = (np.array(a) for a in screen_axes(view, viewer.scene.view.convention))
    offset = point - np.array(viewer.view_centre(view))
    zoom = _camera(viewer, view).zoom
    return np.array([offset @ right, offset @ up]) * zoom


# The field of view


def test_an_empty_viewer_shows_the_200_mm_cube_around_the_origin() -> None:
    assert Viewer().scene.view.field_of_view == EMPTY_FIELD
    assert EMPTY_FIELD == ((-100.0, -100.0, -100.0), (100.0, 100.0, 100.0))


def test_the_field_of_view_reaches_the_outer_edges_of_the_voxels() -> None:
    viewer = Viewer()
    viewer.add(_volume((2, 3, 4), _scaled_affine()), Category.VOLUMES)
    # Half a voxel beyond the end centres: x from 10 - 1 to 10 + 2 * 1.5, and so on.
    assert viewer.scene.view.field_of_view == (
        (9.0, -21.5, 3.0),
        (13.0, -12.5, 19.0),
    )


def test_the_field_of_view_of_an_oblique_grid_holds_all_its_corners() -> None:
    viewer = Viewer()
    affine = _oblique()
    shape = np.array([2, 3, 4])
    viewer.add(_volume((2, 3, 4), affine), Category.VOLUMES)
    corners = np.array(
        [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float
    )
    world = (corners * shape - 0.5) @ affine[:3, :3].T + affine[:3, 3]
    lo, hi = viewer.scene.view.field_of_view
    assert np.allclose(lo, world.min(axis=0))
    assert np.allclose(hi, world.max(axis=0))


@pytest.mark.parametrize(
    "category", [Category.VOLUMES, Category.OVERLAYS, Category.TRACTOGRAMS]
)
def test_the_first_item_with_a_grid_sets_the_field_of_view(
    category: Category,
) -> None:
    viewer = Viewer()
    events = record_events(viewer)
    data: Volume | Tractogram = _volume((2, 3, 4), _scaled_affine())
    if category == Category.TRACTOGRAMS:
        data = Tractogram(
            np.array([[0.0, 0.0, 0.0], [4.0, 2.0, 1.0]]), np.array([0, 2])
        )
    viewer.add(data, category)
    assert viewer.scene.view.field_of_view != EMPTY_FIELD
    assert len(_view_events(events, ViewField.FIELD_OF_VIEW)) == 1


@pytest.mark.parametrize("second", [Category.VOLUMES, Category.OVERLAYS])
@pytest.mark.parametrize("first", [Category.VOLUMES, Category.OVERLAYS])
def test_later_grids_keep_the_field_of_view(first: Category, second: Category) -> None:
    viewer = Viewer()
    viewer.add(_volume((2, 3, 4), _scaled_affine()), first)
    field = viewer.scene.view.field_of_view
    events = record_events(viewer)
    # A volume after an overlay becomes the main volume and the reference grid.
    viewer.add(_volume((9, 9, 9), _oblique()), second)
    assert viewer.scene.view.field_of_view == field
    assert _view_events(events, ViewField.FIELD_OF_VIEW) == []


def test_a_new_scene_after_removing_everything_starts_with_fresh_cameras() -> None:
    viewer = Viewer()
    viewer.add(_volume((2, 3, 4), _scaled_affine()), Category.VOLUMES)
    for view in VIEWS_2D:
        viewer.zoom(view, 3.0)
        viewer.pan(view, 5.0, -4.0)
    viewer.zoom(ViewKind.THREE_D, 3.0)
    viewer.rotate_camera(40.0, 10.0)
    viewer.remove(Category.VOLUMES)
    events = record_events(viewer)
    viewer.add(_volume((5, 6, 7)), Category.VOLUMES)
    assert all(_camera(viewer, view) == Camera2D() for view in VIEWS_2D)
    assert viewer.scene.view.cameras.three_d == Camera3D()
    assert viewer.scene.view.field_of_view == (
        (-0.5, -0.5, -0.5),
        (4.5, 5.5, 6.5),
    )
    names = [e.name for e in events if isinstance(e, ViewStateChanged)]
    assert names.index(ViewField.CAMERAS) < names.index(ViewField.FIELD_OF_VIEW)


# Stepping through the volumes of a 4D image


def _index(viewer: Viewer, item_id: int) -> object:
    return getattr(viewer.scene.item(item_id).options, OptionName.VOLUME_INDEX)


@pytest.mark.parametrize(("n", "expected"), [(1, 1), (2, 2), (5, 3), (-1, 0)])
def test_step_volume_moves_the_main_volume_and_stops_at_the_ends(
    n: int, expected: int
) -> None:
    viewer = Viewer()
    main = viewer.add(_volume((2, 3, 4, 4)), Category.VOLUMES)
    viewer.step_volume(n)
    assert _index(viewer, main.id) == expected


def test_step_volume_stops_at_the_first_volume() -> None:
    viewer = Viewer()
    main = viewer.add(_volume((2, 3, 4, 4)), Category.VOLUMES)
    viewer.set(main.id, OptionName.VOLUME_INDEX, 2)
    viewer.step_volume(-5)
    assert _index(viewer, main.id) == 0


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
def test_step_volume_moves_the_selected_4d_image_not_the_main_volume(
    category: Category,
) -> None:
    viewer = Viewer()
    main = viewer.add(_volume((2, 3, 4, 3)), Category.VOLUMES)
    other = viewer.add(_volume((2, 3, 4, 3)), category)
    if category == Category.VOLUMES:
        # Showing the second volume makes it the main one; show the first again.
        viewer.set(main.id, OptionName.VISIBLE, True)
    assert viewer.scene.view.main_volume == main.id
    viewer.select(other.id)
    viewer.step_volume(1)
    assert _index(viewer, other.id) == 1
    assert _index(viewer, main.id) == 0


@pytest.mark.parametrize("selected", ["3d overlay", "tractogram", "category", "none"])
def test_step_volume_moves_the_main_volume_when_the_selection_has_no_4th_axis(
    selected: str,
) -> None:
    viewer = Viewer()
    main = viewer.add(_volume((2, 3, 4, 3)), Category.VOLUMES)
    overlay = viewer.add(_volume((2, 3, 4)), Category.OVERLAYS)
    tractogram = viewer.add(
        Tractogram(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), np.array([0, 2])),
        Category.TRACTOGRAMS,
    )
    targets: dict[str, ItemId | Category | None] = {
        "3d overlay": overlay.id,
        "tractogram": tractogram.id,
        "category": Category.OVERLAYS,
        "none": None,
    }
    viewer.select(targets[selected])
    viewer.step_volume(1)
    assert _index(viewer, main.id) == 1


@pytest.mark.parametrize("content", ["empty", "tractogram"])
def test_step_volume_without_an_image_does_nothing(content: str) -> None:
    viewer = Viewer()
    if content == "tractogram":
        item = viewer.add(
            Tractogram(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), np.array([0, 2])),
            Category.TRACTOGRAMS,
        )
        viewer.select(item.id)
    events = record_events(viewer)
    viewer.step_volume(1)
    assert events == []


@pytest.mark.parametrize("n", [True, 1.0, "1", None])
def test_step_volume_refuses_what_is_not_a_whole_number(n: object) -> None:
    viewer = Viewer()
    main = viewer.add(_volume((2, 3, 4, 3)), Category.VOLUMES)
    with pytest.raises(TypeError, match="whole number"):
        viewer.step_volume(n)  # type: ignore[arg-type]  # wrong on purpose
    assert _index(viewer, main.id) == 0


# Zoom, pan and reset


def test_zoom_changes_only_the_zoom_of_that_view() -> None:
    viewer = Viewer()
    viewer.zoom(ViewKind.CORONAL, 2.0)
    expected = Cameras(coronal=Camera2D(zoom=2.0))
    assert viewer.scene.view.cameras == expected
    viewer.zoom(ViewKind.CORONAL, 1.5)
    assert _camera(viewer, ViewKind.CORONAL) == Camera2D(zoom=3.0)


@pytest.mark.parametrize(
    ("factor", "expected"),
    [(100.0, ZOOM_RANGE[1]), (1e-3, ZOOM_RANGE[0]), (64.0, 64.0), (0.25, 0.25)],
)
@pytest.mark.parametrize("view", [ViewKind.AXIAL, ViewKind.THREE_D])
def test_the_zoom_stays_within_its_range(
    view: ViewKind, factor: float, expected: float
) -> None:
    assert ZOOM_RANGE == (0.25, 64.0)
    viewer = Viewer()
    viewer.zoom(view, factor)
    assert viewer.scene.view.cameras.of(view).zoom == expected


@pytest.mark.parametrize("factor", [2.0, 0.5, 1000.0])
@pytest.mark.parametrize("convention", list(Convention))
@pytest.mark.parametrize("view", VIEWS_2D)
def test_zoom_about_a_point_keeps_it_in_place_on_the_screen(
    view: ViewKind, convention: Convention, factor: float
) -> None:
    viewer = Viewer()
    viewer.add(_volume((10, 12, 14), _scaled_affine()), Category.VOLUMES)
    viewer.set_convention(convention)
    viewer.pan(view, 3.0, -2.0)
    about = np.array(viewer.scene.view.cursor) + [7.0, -5.0, 4.0]
    about[AXIS[view]] = viewer.scene.view.cursor[AXIS[view]]
    before = _on_screen(viewer, view, about)
    assert np.all(before != 0)
    viewer.zoom(view, factor, about=(about[0], about[1], about[2]))
    assert _camera(viewer, view).zoom != 1.0
    assert np.allclose(_on_screen(viewer, view, about), before)


def test_zoom_without_a_point_keeps_the_pan() -> None:
    viewer = Viewer()
    viewer.pan(ViewKind.AXIAL, 3.0, -2.0)
    viewer.zoom(ViewKind.AXIAL, 2.0)
    assert _camera(viewer, ViewKind.AXIAL) == Camera2D(2.0, (3.0, -2.0))


@pytest.mark.parametrize("convention", list(Convention))
@pytest.mark.parametrize("view", VIEWS_2D)
def test_pan_moves_the_view_centre_along_the_screen(
    view: ViewKind, convention: Convention
) -> None:
    viewer = Viewer()
    viewer.set_convention(convention)
    before = np.array(viewer.view_centre(view))
    viewer.pan(view, 3.0, -2.0)
    viewer.pan(view, 1.0, 1.0)
    assert _camera(viewer, view).pan == (4.0, -1.0)
    right, up = (np.array(a) for a in screen_axes(view, convention))
    assert np.allclose(viewer.view_centre(view), before + 4.0 * right - 1.0 * up)
    others = [v for v in VIEWS_2D if v != view]
    assert all(_camera(viewer, v) == Camera2D() for v in others)


def test_reset_camera_brings_back_zoom_1_and_no_pan_in_that_view_only() -> None:
    viewer = Viewer()
    for view in (ViewKind.AXIAL, ViewKind.SAGITTAL):
        viewer.zoom(view, 3.0)
        viewer.pan(view, 5.0, 6.0)
    viewer.reset_camera(ViewKind.AXIAL)
    assert _camera(viewer, ViewKind.AXIAL) == Camera2D()
    assert _camera(viewer, ViewKind.SAGITTAL) == Camera2D(3.0, (5.0, 6.0))


def _camera_3d(viewer: Viewer) -> Camera3D:
    return viewer.scene.view.cameras.three_d


def _preset(preset: CameraPreset, zoom: float = 1.0) -> Camera3D:
    forward, up = PRESETS[preset]
    return Camera3D(forward, up, zoom)


def _same_pose(camera: Camera3D, expected: Camera3D) -> bool:
    return bool(
        np.allclose(camera.forward, expected.forward, atol=1e-12)
        and np.allclose(camera.up, expected.up, atol=1e-12)
    )


@pytest.mark.parametrize(
    ("preset", "right", "up"),
    [
        # From above: the subject's right on the right, anterior up.
        (CameraPreset.AXIAL, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        # From the front: the subject's left on the right, superior up.
        (CameraPreset.CORONAL, (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        # From the subject's right: anterior on the right, superior up.
        (CameraPreset.SAGITTAL, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ],
)
def test_a_preset_puts_the_subject_on_the_screen_as_named(
    preset: CameraPreset, right: Vec3, up: Vec3
) -> None:
    viewer = Viewer()
    viewer.set_camera(ViewKind.THREE_D, preset)
    camera = _camera_3d(viewer)
    # A camera looking along forward has the screen's right along forward x up.
    assert np.allclose(np.cross(camera.forward, camera.up), right)
    assert np.allclose(camera.up, up)


def test_a_preset_keeps_the_zoom() -> None:
    viewer = Viewer()
    viewer.zoom(ViewKind.THREE_D, 2.5)
    viewer.rotate_camera(30.0, -20.0)
    viewer.set_camera(ViewKind.THREE_D, CameraPreset.SAGITTAL)
    assert _camera_3d(viewer) == _preset(CameraPreset.SAGITTAL, 2.5)


def test_turning_right_brings_the_side_on_the_left_to_the_front() -> None:
    viewer = Viewer()
    viewer.set_camera(ViewKind.THREE_D, CameraPreset.CORONAL)
    # From the front the subject's right is on the screen's left.
    viewer.rotate_camera(90.0, 0.0)
    assert _same_pose(_camera_3d(viewer), _preset(CameraPreset.SAGITTAL))


def test_turning_up_goes_over_the_poles_without_sticking() -> None:
    viewer = Viewer()
    viewer.set_camera(ViewKind.THREE_D, CameraPreset.CORONAL)
    poses = []
    for _ in range(4):
        viewer.rotate_camera(0.0, 90.0)
        poses.append(_camera_3d(viewer))
    # The front turns up: from below, from behind upside down, from above, back.
    assert _same_pose(poses[0], Camera3D((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
    assert _same_pose(poses[1], Camera3D((0.0, 1.0, 0.0), (0.0, 0.0, -1.0)))
    assert _same_pose(poses[2], Camera3D((0.0, 0.0, -1.0), (0.0, -1.0, 0.0)))
    assert _same_pose(poses[3], _preset(CameraPreset.CORONAL))


def test_many_turns_keep_forward_and_up_unit_and_at_right_angles() -> None:
    viewer = Viewer()
    rng = np.random.default_rng(0)
    for right, up in rng.uniform(-180.0, 180.0, (2000, 2)):
        viewer.rotate_camera(float(right), float(up))
    camera = _camera_3d(viewer)
    # Unrenormalised, the lengths drift by about 1e-13 over these turns.
    assert np.linalg.norm(camera.forward) == pytest.approx(1.0, abs=1e-14)
    assert np.linalg.norm(camera.up) == pytest.approx(1.0, abs=1e-14)
    assert np.dot(camera.forward, camera.up) == pytest.approx(0.0, abs=1e-14)


def test_reset_brings_back_the_starting_3d_camera_only() -> None:
    viewer = Viewer()
    viewer.zoom(ViewKind.AXIAL, 3.0)
    viewer.zoom(ViewKind.THREE_D, 3.0)
    viewer.rotate_camera(50.0, 20.0)
    viewer.reset_camera(ViewKind.THREE_D)
    assert _camera_3d(viewer) == Camera3D()
    assert _camera(viewer, ViewKind.AXIAL) == Camera2D(3.0)


def test_the_view_centre_is_the_field_centre_in_the_cursor_slice() -> None:
    viewer = Viewer()
    viewer.add(_volume((2, 3, 4), _scaled_affine()), Category.VOLUMES)
    viewer.move_cursor((1.0, 2.0, 3.0))
    # The middle of (9, -21.5, 3) to (13, -12.5, 19).
    assert viewer.view_centre(ViewKind.AXIAL) == (11.0, -17.0, 3.0)
    assert viewer.view_centre(ViewKind.CORONAL) == (11.0, 2.0, 11.0)
    assert viewer.view_centre(ViewKind.SAGITTAL) == (1.0, -17.0, 11.0)


_BAD_CAMERA_CALLS = {
    "zoom 0": lambda v: v.zoom(ViewKind.AXIAL, 0.0),
    "zoom -1": lambda v: v.zoom(ViewKind.AXIAL, -1.0),
    "zoom nan": lambda v: v.zoom(ViewKind.AXIAL, math.nan),
    "zoom inf": lambda v: v.zoom(ViewKind.AXIAL, math.inf),
    "zoom bool": lambda v: v.zoom(ViewKind.AXIAL, True),
    "zoom text": lambda v: v.zoom(ViewKind.AXIAL, "2"),
    "zoom about": lambda v: v.zoom(ViewKind.AXIAL, 2.0, about=(0.0, math.nan, 0.0)),
    "zoom 3d about": lambda v: v.zoom(ViewKind.THREE_D, 2.0, about=(0.0, 0.0, 0.0)),
    "pan nan": lambda v: v.pan(ViewKind.AXIAL, math.nan, 0.0),
    "pan text": lambda v: v.pan(ViewKind.AXIAL, 0.0, "1"),
    "pan bool": lambda v: v.pan(ViewKind.AXIAL, True, 0.0),
    "pan 3d": lambda v: v.pan(ViewKind.THREE_D, 1.0, 0.0),
    "centre 3d": lambda v: v.view_centre(ViewKind.THREE_D),
    "rotate nan": lambda v: v.rotate_camera(math.nan, 0.0),
    "rotate text": lambda v: v.rotate_camera(0.0, "10"),
    "rotate bool": lambda v: v.rotate_camera(True, 0.0),
    "preset 2d": lambda v: v.set_camera(ViewKind.AXIAL, CameraPreset.AXIAL),
}
_NOT_ENUM_CAMERA_CALLS = {
    "preset text": lambda v: v.set_camera(ViewKind.THREE_D, "axial"),
    "preset view text": lambda v: v.set_camera("3d", CameraPreset.AXIAL),
    "zoom view text": lambda v: v.zoom("3d", 2.0),
    "reset view text": lambda v: v.reset_camera("3d"),
}


@pytest.mark.parametrize(
    ("call", "error"),
    [(c, ValueError) for c in _BAD_CAMERA_CALLS]
    + [(c, TypeError) for c in _NOT_ENUM_CAMERA_CALLS],
)
def test_bad_camera_values_are_refused_and_change_nothing(
    call: str, error: type[Exception]
) -> None:
    viewer = Viewer()
    viewer.zoom(ViewKind.AXIAL, 2.0)
    viewer.rotate_camera(10.0, 0.0)
    before = viewer.scene.view
    events = record_events(viewer)
    calls = _BAD_CAMERA_CALLS | _NOT_ENUM_CAMERA_CALLS
    with pytest.raises(error):
        calls[call](viewer)
    assert events == []
    assert viewer.scene.view == before


# Convention and crosshair


def test_a_convention_switch_keeps_every_view_centred_on_the_same_place() -> None:
    viewer = Viewer()
    viewer.add(_volume((2, 3, 4), _scaled_affine()), Category.VOLUMES)
    for n, view in enumerate(VIEWS_2D, start=1):
        viewer.pan(view, 3.0 * n, -2.0 * n)
    before = {view: viewer.view_centre(view) for view in VIEWS_2D}
    for convention in (Convention.LAS, Convention.RAS):
        viewer.set_convention(convention)
        for view in VIEWS_2D:
            assert np.allclose(viewer.view_centre(view), before[view])


def test_set_convention_stores_it_with_one_event() -> None:
    viewer = Viewer()
    events = record_events(viewer)
    viewer.set_convention(Convention.LAS)
    viewer.set_convention(Convention.LAS)
    assert viewer.scene.view.convention == Convention.LAS
    assert events == [
        ViewStateChanged(ViewField.CONVENTION, Convention.RAS, Convention.LAS)
    ]


def test_las_mirrors_the_screen_right_of_the_axial_and_coronal_views() -> None:
    for view in (ViewKind.AXIAL, ViewKind.CORONAL):
        ras = screen_axes(view, Convention.RAS)
        las = screen_axes(view, Convention.LAS)
        assert ras[0] == (1.0, 0.0, 0.0)
        assert las[0] == (-1.0, 0.0, 0.0)
        assert ras[1] == las[1]
    assert screen_axes(ViewKind.SAGITTAL, Convention.LAS) == screen_axes(
        ViewKind.SAGITTAL, Convention.RAS
    )


def test_toggle_crosshair_flips_it_with_one_event_each_time() -> None:
    viewer = Viewer()
    assert viewer.scene.view.crosshair_visible
    events = record_events(viewer)
    seen = []
    for _ in range(2):
        viewer.toggle_crosshair()
        seen.append(viewer.scene.view.crosshair_visible)
    assert seen == [False, True]
    assert events == [
        ViewStateChanged(ViewField.CROSSHAIR_VISIBLE, True, False),
        ViewStateChanged(ViewField.CROSSHAIR_VISIBLE, False, True),
    ]


# The crosshair's readout

SLOPE, INTERCEPT = 0.5, -3.0


def _random_series() -> Volume:
    data = np.random.default_rng(42).uniform(0, 100, (5, 6, 7, 3)).astype(np.float32)
    return Volume(data, _oblique(), slope=SLOPE, intercept=INTERCEPT)


def _world(affine: np.ndarray, voxel: np.ndarray) -> tuple[float, float, float]:
    x, y, z = affine[:3, :3] @ voxel + affine[:3, 3]
    return (float(x), float(y), float(z))


# Voxel positions inside the grid, between voxel centres.
_INSIDE = [
    np.array([1.3, 2.7, 4.1]),
    np.array([0.2, 0.9, 5.6]),
    np.array([3.8, 4.4, 0.5]),
    np.array([4.0, 5.0, 6.0]),
]


def _nearest(voxel: np.ndarray) -> tuple[int, ...]:
    # Halves round up, as nearest sampling draws the slices.
    return tuple(int(v) for v in np.floor(voxel + 0.5))


@pytest.mark.parametrize("index", [0, 2])
@pytest.mark.parametrize("voxel", _INSIDE, ids=str)
def test_the_readout_is_the_trilinear_value_in_scaled_units(
    voxel: np.ndarray, index: int
) -> None:
    viewer = Viewer()
    volume = _random_series()
    item = viewer.add(volume, Category.VOLUMES, name="dwi")
    viewer.set(item.id, OptionName.VOLUME_INDEX, index)
    viewer.move_cursor(_world(volume.affine, voxel))
    stored = volume.data[..., index].astype(np.float64)
    expected = map_coordinates(stored, voxel.reshape(3, 1), order=1)[0]
    readout = viewer.readout()
    assert readout.name == "dwi"
    assert readout.position == viewer.scene.view.cursor
    # float64 throughout; only the affine's round trip differs.
    assert readout.value == pytest.approx(SLOPE * expected + INTERCEPT, abs=1e-9)
    assert readout.voxel == (*_nearest(voxel), index)


@pytest.mark.parametrize("voxel", _INSIDE, ids=str)
def test_the_readout_is_the_nearest_voxel_value_without_interpolation(
    voxel: np.ndarray,
) -> None:
    viewer = Viewer()
    volume = _random_series()
    item = viewer.add(volume, Category.VOLUMES)
    viewer.set(item.id, OptionName.INTERPOLATE, False)
    viewer.move_cursor(_world(volume.affine, voxel))
    i, j, k = _nearest(voxel)
    expected = SLOPE * float(volume.data[i, j, k, 0]) + INTERCEPT
    assert viewer.readout().value == pytest.approx(expected, abs=1e-9)
    assert viewer.readout().voxel == (i, j, k, 0)


@pytest.mark.parametrize("interpolate", [True, False])
def test_the_readout_picks_the_upper_voxel_halfway_between_two(
    interpolate: bool,
) -> None:
    viewer = Viewer()
    # Scaled, not oblique: the halves survive the affine's round trip exactly.
    item = viewer.add(_volume((3, 4, 5), _scaled_affine()), Category.VOLUMES)
    viewer.set(item.id, OptionName.INTERPOLATE, interpolate)
    viewer.move_cursor(_world(_scaled_affine(), np.array([0.5, 1.5, 2.5])))
    assert viewer.readout().voxel == (1, 2, 3)


def test_the_readout_of_a_3d_volume_has_a_3d_voxel() -> None:
    viewer = Viewer()
    viewer.add(_volume((2, 3, 4)), Category.VOLUMES)
    viewer.move_cursor((1.0, 2.0, 3.0))
    assert viewer.readout() == Readout((1.0, 2.0, 3.0), "Volume 1", (1, 2, 3), 23.0)


@pytest.mark.parametrize(
    "voxel",
    [
        np.array([-0.6, 2.0, 3.0]),
        np.array([2.0, 6.0, 3.0]),
        np.array([2.0, 3.0, 6.6]),
    ],
    ids=str,
)
def test_the_readout_outside_the_grid_has_no_voxel_or_value(
    voxel: np.ndarray,
) -> None:
    viewer = Viewer()
    volume = _random_series()
    viewer.add(volume, Category.VOLUMES, name="dwi")
    position = _world(volume.affine, voxel)
    viewer.move_cursor(position)
    assert viewer.readout() == Readout(viewer.scene.view.cursor, "dwi", None, None)


@pytest.mark.parametrize("content", ["empty", "overlay", "tractogram"])
def test_without_a_volume_the_readout_is_the_position_only(content: str) -> None:
    viewer = Viewer()
    if content == "overlay":
        viewer.add(_volume(), Category.OVERLAYS)
    elif content == "tractogram":
        viewer.add(
            Tractogram(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), np.array([0, 2])),
            Category.TRACTOGRAMS,
        )
    viewer.move_cursor((1.0, 2.0, 3.0))
    assert viewer.readout() == Readout((1.0, 2.0, 3.0))
