"""The undo and redo of a phantom's bundles, through the Viewer and its tools."""

from pathlib import Path

import numpy as np
import pytest

from larmor.types import Tractogram
from larmor.viewer import (
    BundleEnd,
    BundleField,
    Category,
    OptionName,
    ProxyMap,
    Tool,
    Viewer,
)
from larmor.viewer.model.bundle_history import HISTORY_LIMIT
from larmor.viewer.model.bundle_tools import Plane, Pointer
from larmor.viewer.model.options import TractogramOptions
from larmor.viewer.model.phantom import (
    BundleSettings,
    Draft,
    GeometrySettings,
    PhantomState,
    fullness_of,
)
from larmor.viewer.model.shortcuts import Command, run
from larmor.viewer.model.view_state import Workspace

Vec3 = tuple[float, float, float]
SCALE = 4.0  # pixels per mm


class FlatPlane:
    """An axial view whose pixels are x and y times SCALE."""

    axis = 2
    mm_per_pixel = 1 / SCALE

    def project(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points)[:, :2] * SCALE

    def world(self, x: float, y: float, depth: float) -> np.ndarray:
        return np.array([x / SCALE, y / SCALE, depth])


PLANE: Plane = FlatPlane()


def _pointer(world: Vec3) -> Pointer:
    return Pointer(PLANE, world[0] * SCALE, world[1] * SCALE)


def _drag(viewer: Viewer, start: Vec3, end: Vec3, steps: int = 6) -> None:
    tools = viewer.bundle_tools
    assert tools.press(_pointer(start))
    for k in range(1, steps + 1):
        t = k / steps
        x, y, z = (a + (b - a) * t for a, b in zip(start, end, strict=True))
        tools.move(_pointer((x, y, z)))
    tools.release(_pointer(end))


def _click(viewer: Viewer, world: Vec3) -> None:
    viewer.bundle_tools.press(_pointer(world))
    viewer.bundle_tools.release(_pointer(world))


@pytest.fixture
def viewer() -> Viewer:
    made = Viewer()
    made.new_phantom((200.0, 200.0, 200.0), 2.0)
    made.set_workspace(Workspace.SIMULATE)
    made.move_cursor((0.0, 0.0, 0.0))
    return made


def _state(viewer: Viewer) -> PhantomState:
    state = viewer.phantom
    assert state is not None
    return state


def _draft(viewer: Viewer) -> Draft:
    draft = _state(viewer).draft
    assert draft is not None
    return draft


def _only(viewer: Viewer, index: int = 0) -> BundleSettings:
    return _state(viewer).bundles[index]


def _geometry(viewer: Viewer, index: int = 0) -> GeometrySettings:
    geometry = _only(viewer, index).geometry
    assert geometry is not None
    return geometry


def _straight(viewer: Viewer, x: float = 40.0) -> BundleSettings:
    """Adds a finished straight bundle along y through the axial plane."""
    viewer.start_bundle()
    viewer.set_bundle_points([(x, -70.0, 0.0), (x, 70.0, 0.0)])
    viewer.finish_bundle()
    return _state(viewer).bundles[-1]


def _points(viewer: Viewer, item_id: int) -> np.ndarray:
    data = viewer.scene.item(item_id).data
    assert isinstance(data, Tractogram)
    return np.asarray(data.points)


def _has_item(viewer: Viewer, item_id: int) -> bool:
    return any(i.id == item_id for i in viewer.scene.items)


# The Pen


def test_a_stroke_and_enter_go_back_together_and_come_again(viewer: Viewer) -> None:
    viewer.set_tool(Tool.DRAW_BUNDLE)
    _drag(viewer, (0.0, -40.0, 0.0), (0.0, 40.0, 0.0))
    assert run(viewer, Command.FINISH_BUNDLE)
    bundle = _only(viewer)
    points = _points(viewer, bundle.item_id)
    assert viewer.undo_bundles()
    state = _state(viewer)
    assert state.bundles == () and not _has_item(viewer, bundle.item_id)
    assert viewer.scene.view.tool == Tool.DRAW_BUNDLE  # the Pen waits for a stroke
    assert state.draft is not None and state.draft.is_empty
    assert viewer.check_phantom() != []  # nothing is left to run
    assert viewer.redo_bundles()
    assert _only(viewer) == bundle
    np.testing.assert_array_equal(_points(viewer, bundle.item_id), points)
    assert viewer.check_phantom() == []


def test_a_removed_bundle_comes_back_with_its_id_and_options(viewer: Viewer) -> None:
    bundle = _straight(viewer)
    viewer.set(bundle.item_id, OptionName.OPACITY_2D, 0.5)
    viewer.remove(bundle.item_id)
    assert not _has_item(viewer, bundle.item_id)
    assert viewer.undo_bundles()
    assert _only(viewer) == bundle
    options = viewer.scene.item(bundle.item_id).options
    assert isinstance(options, TractogramOptions)
    assert options.opacity_2d == 0.5


def test_each_stroke_and_click_of_a_bundle_being_drawn_is_a_step(
    viewer: Viewer,
) -> None:
    viewer.set_tool(Tool.DRAW_BUNDLE)
    _drag(viewer, (0.0, -40.0, 0.0), (0.0, 0.0, 0.0))
    one = _draft(viewer).points
    _click(viewer, (0.0, 0.0, 0.0))  # a click on its tip goes on from there: no step
    _click(viewer, (10.0, 20.0, 0.0))
    two = _draft(viewer).points
    assert len(two) == len(one) + 1
    assert viewer.undo_bundles()
    assert _draft(viewer).points == one
    assert viewer.redo_bundles()
    assert _draft(viewer).points == two
    assert viewer.undo_bundles() and viewer.undo_bundles()
    assert _draft(viewer).is_empty
    assert not viewer.undo_bundles()


def test_a_dragged_point_and_backspace_are_steps(viewer: Viewer) -> None:
    viewer.set_tool(Tool.DRAW_BUNDLE)
    for point in ((0.0, -20.0, 0.0), (10.0, 10.0, 0.0), (20.0, 40.0, 0.0)):
        _click(viewer, point)
    placed = _draft(viewer).points
    _drag(viewer, (10.0, 10.0, 0.0), (17.0, 10.0, 0.0))
    moved = _draft(viewer).points
    assert moved != placed
    assert run(viewer, Command.REMOVE_POINT)
    assert len(_draft(viewer).points) == 2
    assert viewer.undo_bundles()
    assert _draft(viewer).points == moved
    assert viewer.undo_bundles()
    assert _draft(viewer).points == placed


def test_undo_after_finishing_reopens_the_bundle_at_its_step_before(
    viewer: Viewer,
) -> None:
    viewer.set_tool(Tool.DRAW_BUNDLE)
    _drag(viewer, (0.0, -40.0, 0.0), (0.0, 0.0, 0.0))
    one = _draft(viewer).points
    _drag(viewer, (0.0, 0.0, 0.0), (20.0, 30.0, 0.0))
    run(viewer, Command.FINISH_BUNDLE)
    assert viewer.undo_bundles()
    assert _state(viewer).bundles == ()
    assert _draft(viewer).points == one
    assert viewer.redo_bundles()
    assert len(_state(viewer).bundles) == 1


def test_an_edit_that_changes_nothing_is_no_step(viewer: Viewer) -> None:
    bundle = _straight(viewer)
    viewer.set_bundle(bundle.item_id, BundleField.SEED, 7)
    viewer.edit_bundle(bundle.item_id)
    viewer.cancel_bundle()
    viewer.edit_bundle(bundle.item_id)
    viewer.finish_bundle()
    viewer.start_bundle()  # the Pen waiting is no step either
    assert viewer.undo_bundles()
    assert _only(viewer) == bundle


def test_an_edit_with_the_pen_goes_back_with_its_streamlines(viewer: Viewer) -> None:
    bundle = _straight(viewer)
    points = _points(viewer, bundle.item_id)
    viewer.set_tool(Tool.DRAW_BUNDLE)
    _drag(viewer, (40.0, 70.0, 0.0), (60.0, 90.0, 0.0))
    assert _draft(viewer).editing is not None
    assert viewer.undo_bundles()
    assert _draft(viewer).editing is None
    assert _only(viewer) == bundle
    np.testing.assert_array_equal(_points(viewer, bundle.item_id), points)


def test_cancelling_a_bundle_can_be_taken_back(viewer: Viewer) -> None:
    viewer.set_tool(Tool.DRAW_BUNDLE)
    _drag(viewer, (0.0, -40.0, 0.0), (0.0, 40.0, 0.0))
    points = _draft(viewer).points
    viewer.cancel_bundle()
    assert not viewer.drawing
    assert viewer.undo_bundles()
    assert _draft(viewer).points == points
    assert viewer.scene.view.tool == Tool.DRAW_BUNDLE
    assert _draft(viewer).item_id is not None


# The other tools


def test_a_width_drag_is_one_step_and_escape_leaves_none(viewer: Viewer) -> None:
    bundle = _straight(viewer)
    viewer.set_tool(Tool.WIDTH)
    tools = viewer.bundle_tools
    tools.press(_pointer((38.0, 0.0, 0.0)))
    tools.move(_pointer((30.0, 0.0, 0.0)))
    assert _only(viewer) != bundle
    assert not viewer.undo_bundles()  # during a drag there is nothing to take back
    assert run(viewer, Command.LEAVE_DRAW_MODE)
    assert _only(viewer) == bundle
    _drag(viewer, (38.0, 0.0, 0.0), (30.0, 0.0, 0.0))
    made = _only(viewer)
    assert made != bundle
    assert viewer.undo_bundles()
    assert _only(viewer) == bundle
    assert viewer.scene.view.tool == Tool.WIDTH
    assert viewer.redo_bundles()
    assert _only(viewer) == made
    assert not viewer.redo_bundles()


@pytest.mark.parametrize(
    ("tool", "start", "end"),
    [
        (Tool.BEND, (40.0, 0.0, 0.0), (55.0, 0.0, 0.0)),
        (Tool.TWIST, (40.0, 0.0, 0.0), (55.0, 0.0, 0.0)),
        (Tool.FAN, (30.0, 72.0, 0.0), (50.0, 72.0, 0.0)),
    ],
)
def test_each_tools_drag_goes_back_with_its_streamlines(
    viewer: Viewer, tool: Tool, start: Vec3, end: Vec3
) -> None:
    bundle = _straight(viewer)
    points = _points(viewer, bundle.item_id)
    viewer.set_tool(tool)
    _drag(viewer, start, end)
    made = _only(viewer)
    assert made != bundle
    shaped = _points(viewer, bundle.item_id)
    assert viewer.undo_bundles()
    assert _only(viewer) == bundle
    np.testing.assert_array_equal(_points(viewer, bundle.item_id), points)
    assert viewer.redo_bundles()
    assert _only(viewer) == made
    np.testing.assert_array_equal(_points(viewer, bundle.item_id), shaped)


def test_a_bend_goes_back_with_the_branch_that_followed(viewer: Viewer) -> None:
    _straight(viewer, 0.0)
    viewer.set_tool(Tool.DRAW_BUNDLE)
    _drag(viewer, (0.0, 0.0, 0.0), (30.0, 40.0, 0.0))
    run(viewer, Command.FINISH_BUNDLE)
    bundles = _state(viewer).bundles
    assert bundles[1].branch is not None
    viewer.set_tool(Tool.BEND)
    viewer.bundle_tools.set_option("reach", 150)
    _drag(viewer, (0.0, -40.0, 0.0), (12.0, -40.0, 0.0))
    assert _state(viewer).bundles != bundles
    assert viewer.undo_bundles()
    assert _state(viewer).bundles == bundles


def test_a_branch_goes_back_with_the_streamlines_it_took(viewer: Viewer) -> None:
    parent = _straight(viewer, 0.0)
    viewer.set_tool(Tool.DRAW_BUNDLE)
    _drag(viewer, (0.0, 0.0, 0.0), (30.0, 40.0, 0.0))
    run(viewer, Command.FINISH_BUNDLE)
    assert _geometry(viewer).num_streamlines < 1000
    assert viewer.undo_bundles()
    assert _state(viewer).bundles == (parent,)


def test_a_cut_goes_back_and_a_removal_with_its_branches(viewer: Viewer) -> None:
    _straight(viewer, 0.0)
    viewer.set_tool(Tool.DRAW_BUNDLE)
    _drag(viewer, (0.0, 0.0, 0.0), (30.0, 40.0, 0.0))
    run(viewer, Command.FINISH_BUNDLE)
    bundles = _state(viewer).bundles
    viewer.set_tool(Tool.ERASE)
    _drag(viewer, (-10.0, -40.0, 0.0), (10.0, -40.0, 0.0))
    assert _state(viewer).bundles != bundles
    assert viewer.undo_bundles()
    assert _state(viewer).bundles == bundles
    _click(viewer, (0.0, -40.0, 0.0))  # removes the bundle and its branch
    assert _state(viewer).bundles == ()
    assert viewer.undo_bundles()
    assert _state(viewer).bundles == bundles
    assert all(_has_item(viewer, b.item_id) for b in bundles)


# The settings of a bundle


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (BundleField.TISSUE, "GM"),
        (BundleField.NUM_STREAMLINES, 300),
        (BundleField.RADIUS, 4.0),
        (BundleField.TANGLE_MM, 50.0),
        (BundleField.SEED, 7),
        (BundleField.NUM_POINTS, 50),
        (BundleField.STREAMLINE_AREA, 0.1),
    ],
)
def test_a_setting_of_a_bundle_goes_back(
    viewer: Viewer, name: BundleField, value: object
) -> None:
    bundle = _straight(viewer)
    points = _points(viewer, bundle.item_id)
    viewer.set_bundle(bundle.item_id, name, value)
    made = _only(viewer)
    assert made != bundle
    assert viewer.undo_bundles()
    assert _only(viewer) == bundle
    np.testing.assert_array_equal(_points(viewer, bundle.item_id), points)
    assert viewer.redo_bundles()
    assert _only(viewer) == made


def test_the_fullness_goes_back(viewer: Viewer) -> None:
    bundle = _straight(viewer)
    viewer.set_bundle_fullness(bundle.item_id, 2.0)
    assert fullness_of(_only(viewer)) == pytest.approx(2.0)
    assert viewer.undo_bundles()
    assert fullness_of(_only(viewer)) is None


def test_a_slider_dragged_is_one_step(viewer: Viewer) -> None:
    bundle = _straight(viewer)
    viewer.begin_bundle_step()
    for fullness in (1.0, 1.5, 2.0, 2.5):
        viewer.set_bundle_fullness(bundle.item_id, fullness)
    viewer.end_bundle_step()
    assert viewer.undo_bundles()
    assert _only(viewer) == bundle


def test_a_change_of_a_parent_goes_back_with_its_branch(viewer: Viewer) -> None:
    _straight(viewer, 0.0)
    viewer.set_tool(Tool.DRAW_BUNDLE)
    _drag(viewer, (0.0, 0.0, 0.0), (30.0, 40.0, 0.0))
    run(viewer, Command.FINISH_BUNDLE)
    viewer.set_tool(Tool.CURSOR)
    bundles = _state(viewer).bundles
    viewer.set_bundle(bundles[0].item_id, BundleField.RADIUS, 4.0)
    assert _geometry(viewer, 1).radius == 4.0
    assert viewer.undo_bundles()
    assert _state(viewer).bundles == bundles


def test_landings_and_rings_set_directly_go_back(viewer: Viewer) -> None:
    bundle = _straight(viewer)
    viewer.set_bundle_rings(
        bundle.item_id, [[0.0, 2.0, 2.0, 0.0], [1.0, 3.0, 3.0, 0.0]]
    )
    viewer.set_bundle_landing(
        bundle.item_id, BundleEnd.END, [(20.0, 70.0, 0.0), (60.0, 70.0, 0.0)]
    )
    assert viewer.undo_bundles() and viewer.undo_bundles()
    assert _only(viewer) == bundle


# Tractograms used as bundles


def _tracts() -> Tractogram:
    points = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    return Tractogram(points, np.array([0, 2]))


def test_using_a_tractogram_goes_back_and_leaves_its_item(viewer: Viewer) -> None:
    item = viewer.add(_tracts(), Category.TRACTOGRAMS, name="loaded")
    viewer.use_as_bundle(item.id)
    assert len(_state(viewer).bundles) == 1
    assert viewer.undo_bundles()
    assert _state(viewer).bundles == ()
    assert _has_item(viewer, item.id)
    assert viewer.redo_bundles()
    assert _only(viewer).item_id == item.id


def test_removing_a_tractogram_used_as_a_bundle_goes_back(viewer: Viewer) -> None:
    item = viewer.add(_tracts(), Category.TRACTOGRAMS, name="loaded")
    viewer.use_as_bundle(item.id)
    viewer.remove(item.id)
    assert not _has_item(viewer, item.id)
    assert viewer.undo_bundles()
    assert _only(viewer).item_id == item.id
    assert viewer.scene.item(item.id).name == "loaded"


# The history itself


def test_a_new_change_drops_what_could_be_done_again(viewer: Viewer) -> None:
    bundle = _straight(viewer)
    viewer.set_bundle(bundle.item_id, BundleField.RADIUS, 3.0)
    assert viewer.undo_bundles()
    viewer.set_bundle(bundle.item_id, BundleField.RADIUS, 4.0)
    assert not viewer.redo_bundles()
    assert _geometry(viewer).radius == 4.0


def test_the_history_is_bounded_and_keeps_the_latest(viewer: Viewer) -> None:
    bundle = _straight(viewer)
    for k in range(HISTORY_LIMIT + 20):
        viewer.set_bundle(bundle.item_id, BundleField.SEED, k + 10)
    steps = 0
    while viewer.undo_bundles():
        steps += 1
    assert steps == HISTORY_LIMIT
    assert _geometry(viewer).seed == 29


def test_a_new_or_opened_phantom_starts_the_history_again(
    viewer: Viewer, tmp_path: Path
) -> None:
    bundle = _straight(viewer)
    viewer.set_bundle(bundle.item_id, BundleField.RADIUS, 3.0)
    viewer.save_phantom(tmp_path / "p.phantom.json")
    viewer.load_phantom(tmp_path / "p.phantom.json")
    assert not viewer.undo_bundles()
    assert len(_state(viewer).bundles) == 1
    viewer.new_phantom((60.0, 60.0, 60.0), 2.0)
    assert not viewer.undo_bundles()


def test_a_session_starts_the_history_again(viewer: Viewer, tmp_path: Path) -> None:
    _straight(viewer)
    viewer.save_session(tmp_path / "s.lar.json")
    other = Viewer()
    assert other.load_session(tmp_path / "s.lar.json").problems == ()
    assert not other.undo_bundles()


def test_undo_leaves_the_tissues_and_the_scan_alone(viewer: Viewer) -> None:
    bundle = _straight(viewer)
    viewer.set_bundle(bundle.item_id, BundleField.RADIUS, 3.0)
    tissues, scan = _state(viewer).tissues, _state(viewer).scan
    viewer.remove_tissue("CSF")
    assert viewer.undo_bundles()
    assert _state(viewer).tissues != tissues  # the tissue stays gone
    assert _state(viewer).scan == scan


def test_the_keys_work_only_in_the_workspace(viewer: Viewer) -> None:
    bundle = _straight(viewer)
    viewer.set_bundle(bundle.item_id, BundleField.RADIUS, 3.0)
    assert run(viewer, Command.UNDO_BUNDLES)
    assert _geometry(viewer).radius == GeometrySettings.radius
    assert run(viewer, Command.REDO_BUNDLES)
    assert _geometry(viewer).radius == 3.0
    viewer.set_workspace(Workspace.VIEW)
    assert not run(viewer, Command.UNDO_BUNDLES)


def test_undo_follows_the_proxy_maps(viewer: Viewer) -> None:
    bundle = _straight(viewer)
    proxy = viewer.show_proxy(ProxyMap.FILL)
    assert proxy is not None
    before = np.array(viewer.scene.item(proxy.id).data.data)  # type: ignore[attr-defined]
    viewer.set_bundle(bundle.item_id, BundleField.RADIUS, 6.0)
    assert viewer.undo_bundles()
    now = viewer.scene.item(proxy.id).data.data  # type: ignore[attr-defined]
    np.testing.assert_array_equal(now, before)
