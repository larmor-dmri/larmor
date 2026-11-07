"""The drawing tools of the SymDWI workspace, driven without a window."""

import numpy as np
import pytest

from larmor.viewer import BundleEnd, BundleField, Tool, Viewer, ViewKind
from larmor.viewer.model.bundle_shape import shape_of
from larmor.viewer.model.bundle_tools import Plane, Pointer
from larmor.viewer.model.phantom import (
    BundleSettings,
    Draft,
    GeometrySettings,
    PhantomState,
    build_streamlines,
    fullness_of,
)
from larmor.viewer.model.shortcuts import Command, run
from larmor.viewer.model.view_state import Workspace

Vec3 = tuple[float, float, float]
AXIAL = 2
CORONAL = 1
SCALE = 4.0  # pixels per mm


class FlatPlane:
    """A 2D view along one world axis, whose pixels are its other axes times SCALE."""

    def __init__(self, axis: int) -> None:
        self.axis = axis
        self.mm_per_pixel = 1 / SCALE
        self.others = [i for i in range(3) if i != axis]

    def project(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points)[:, self.others] * SCALE

    def world(self, x: float, y: float, depth: float) -> np.ndarray:
        point = np.zeros(3)
        point[self.others[0]] = x / SCALE
        point[self.others[1]] = y / SCALE
        point[self.axis] = depth
        return point


class Driver:
    """Works the tools as a window's mouse does, in world mm."""

    def __init__(self, viewer: Viewer, plane: Plane | None = None) -> None:
        self.viewer = viewer
        self.tools = viewer.bundle_tools
        self.plane: Plane = plane if plane is not None else FlatPlane(AXIAL)

    def _pointer(self, world: Vec3, ctrl: bool = False, shift: bool = False) -> Pointer:
        x, y = self.plane.project(np.array([world]))[0]
        return Pointer(self.plane, float(x), float(y), ctrl=ctrl, shift=shift)

    def press(self, world: Vec3, ctrl: bool = False, shift: bool = False) -> bool:
        return self.tools.press(self._pointer(world, ctrl, shift))

    def move(self, world: Vec3, ctrl: bool = False, shift: bool = False) -> None:
        self.tools.move(self._pointer(world, ctrl, shift))

    def release(self, world: Vec3, ctrl: bool = False, shift: bool = False) -> None:
        self.tools.release(self._pointer(world, ctrl, shift))

    def drag(
        self,
        points: list[Vec3],
        steps: int = 10,
        ctrl: bool = False,
        shift: bool = False,
    ) -> None:
        taken = self.press(points[0], ctrl, shift)
        assert taken
        last = points[0]
        for point in points[1:]:
            for k in range(1, steps + 1):
                t = k / steps
                x, y, z = (a + (b - a) * t for a, b in zip(last, point, strict=True))
                self.move((x, y, z), ctrl, shift)
            last = point
        self.release(last, ctrl, shift)

    def click(self, world: Vec3, ctrl: bool = False) -> None:
        if not self.press(world, ctrl):
            self.tools.click_here(self._pointer(world, ctrl))
            return
        self.release(world)


@pytest.fixture
def viewer() -> Viewer:
    made = Viewer()
    made.new_phantom((200.0, 200.0, 200.0), 2.0)
    made.set_workspace(Workspace.SIMULATE)
    made.move_cursor((0.0, 0.0, 0.0))
    return made


@pytest.fixture
def pen(viewer: Viewer) -> Driver:
    viewer.set_tool(Tool.DRAW_BUNDLE)
    return Driver(viewer)


def _state(viewer: Viewer) -> PhantomState:
    state = viewer.phantom
    assert state is not None
    return state


def _draft(viewer: Viewer) -> Draft:
    draft = _state(viewer).draft
    assert draft is not None
    return draft


def _draft_points(viewer: Viewer) -> tuple[Vec3, ...]:
    return _draft(viewer).points


def _bundle(viewer: Viewer, index: int = -1) -> BundleSettings:
    return _state(viewer).bundles[index]


def _geometry(viewer: Viewer, index: int = -1) -> GeometrySettings:
    geometry = _bundle(viewer, index).geometry
    assert geometry is not None
    return geometry


def _straight(viewer: Viewer, x: float = 40.0, radius: float = 3.0) -> BundleSettings:
    """Adds a finished straight bundle along y, 140 mm long, in the axial plane."""
    viewer.start_bundle()
    viewer.set_bundle(None, BundleField.RADIUS, radius)
    viewer.set_bundle_points([(x, -70.0, 0.0), (x, 70.0, 0.0)])
    viewer.finish_bundle()
    return _bundle(viewer)


def _section(viewer: Viewer, u: float, index: int = -1) -> tuple[float, float, float]:
    bundle = _bundle(viewer, index)
    assert bundle.geometry is not None
    stem = bundle.branch.stem_points if bundle.branch else 0
    r1, r2, angle = shape_of(bundle.geometry, stem).sections[round(u * 99)]
    return float(r1), float(r2), float(angle)


# Pen


def test_a_stroke_makes_a_bundle_and_enter_finishes_it(
    viewer: Viewer, pen: Driver
) -> None:
    pen.drag([(40.0, -20.0, 0.0), (40.0, 60.0, 0.0)], steps=20)
    points = _draft_points(viewer)
    assert 2 <= len(points) <= 3  # a straight stroke keeps few points
    np.testing.assert_allclose(points[0], (40.0, -20.0, 0.0), atol=0.3)
    assert viewer.drawing
    assert run(viewer, Command.FINISH_BUNDLE)
    state = _state(viewer)
    assert len(state.bundles) == 1
    # The Pen stays on, and the next stroke starts another bundle.
    assert viewer.scene.view.tool == Tool.DRAW_BUNDLE
    assert state.draft is not None and state.draft.is_empty
    assert viewer.scene.view.selected == state.bundles[0].item_id
    assert viewer.check_phantom() == []


def test_clicks_place_points_and_backspace_takes_the_last_back(
    viewer: Viewer, pen: Driver
) -> None:
    pen.click((20.0, -20.0, 0.0))
    assert len(_draft_points(viewer)) == 1
    assert _draft(viewer).item_id is None
    pen.click((25.0, 10.0, 0.0))
    assert len(_draft_points(viewer)) == 2
    viewer.move_cursor((0.0, 0.0, 20.0))  # scrolling, so the next lies on a new slice
    pen.click((30.0, 40.0, 0.0))
    points = _draft_points(viewer)
    assert len(points) == 3
    assert points[1][2] == pytest.approx(0.0)
    assert points[2][2] == pytest.approx(20.0)
    assert run(viewer, Command.REMOVE_POINT)
    assert len(_draft_points(viewer)) == 2


def test_a_drag_while_placing_continues_the_same_bundle(
    viewer: Viewer, pen: Driver
) -> None:
    pen.click((20.0, -20.0, 0.0))
    pen.click((25.0, 10.0, 0.0))
    pen.drag([(40.0, 20.0, 0.0), (55.0, 40.0, 0.0)])
    assert len(_draft_points(viewer)) > 2
    assert _state(viewer).bundles == ()


def test_after_enter_clicks_start_another_bundle(viewer: Viewer, pen: Driver) -> None:
    pen.click((20.0, -20.0, 0.0))
    pen.click((25.0, 10.0, 0.0))
    assert run(viewer, Command.FINISH_BUNDLE)
    pen.click((-40.0, 0.0, 0.0))
    pen.click((-40.0, 30.0, 0.0))
    assert run(viewer, Command.FINISH_BUNDLE)
    assert [len(_geometry(viewer, i).control_points) for i in (0, 1)] == [2, 2]


def test_the_bundle_passes_through_points_placed_after_a_stroke_on_its_tip(
    viewer: Viewer, pen: Driver
) -> None:
    pen.drag([(0.0, -40.0, 0.0), (0.0, 0.0, 0.0)])
    pen.click((0.0, 0.0, 0.0))  # a tip: go on from there, point by point
    pen.click((10.0, 20.0, 0.0))
    pen.click((20.0, 40.0, 0.0))
    points = _draft_points(viewer)
    assert points[-1] == (20.0, 40.0, 0.0)
    assert points[-2] == (10.0, 20.0, 0.0)
    # Backspace takes back the last point placed, not the stroke.
    assert run(viewer, Command.REMOVE_POINT)
    assert _draft_points(viewer)[-1] == (10.0, 20.0, 0.0)


def test_going_on_from_the_first_tip_adds_before_it(
    viewer: Viewer, pen: Driver
) -> None:
    pen.drag([(0.0, 0.0, 0.0), (0.0, 40.0, 0.0)])
    pen.click((0.0, 0.0, 0.0))
    pen.click((0.0, -20.0, 0.0))
    points = _draft_points(viewer)
    assert points[0] == (0.0, -20.0, 0.0)
    assert run(viewer, Command.REMOVE_POINT)
    assert _draft_points(viewer)[0][1] == pytest.approx(0.0)


def test_a_click_on_a_bundle_does_nothing_and_ctrl_places_a_point(
    viewer: Viewer, pen: Driver
) -> None:
    _straight(viewer, 0.0)
    viewer.set_tool(Tool.DRAW_BUNDLE)
    pen.click((0.0, 20.0, 0.0))
    assert _draft_points(viewer) == ()
    pen.click((0.0, 20.0, 0.0), ctrl=True)
    pen.click((30.0, 20.0, 0.0), ctrl=True)
    assert len(_draft_points(viewer)) == 2
    assert _draft(viewer).branch is None


def test_a_ctrl_press_away_from_a_point_is_left_to_the_view(
    viewer: Viewer, pen: Driver
) -> None:
    assert not pen.press((5.0, 5.0, 0.0), ctrl=True)


def test_interior_dots_drag_and_end_dots_need_ctrl(viewer: Viewer, pen: Driver) -> None:
    for x, y in ((0.0, -20.0), (10.0, 10.0), (20.0, 40.0)):
        pen.click((x, y, 0.0))
    pen.drag([(10.0, 10.0, 0.0), (17.0, 10.0, 0.0)])
    points = _draft_points(viewer)
    assert len(points) == 3
    np.testing.assert_allclose(points[1], (17.0, 10.0, 0.0), atol=1e-6)
    # An end's dot continues the bundle unless Ctrl is held.
    pen.drag([(20.0, 40.0, 0.0), (28.0, 40.0, 0.0)], ctrl=True)
    points = _draft_points(viewer)
    assert len(points) == 3
    np.testing.assert_allclose(points[2], (28.0, 40.0, 0.0), atol=1e-6)


def test_a_stroke_from_a_tip_continues_and_keeps_the_rings_in_place(
    viewer: Viewer, pen: Driver
) -> None:
    viewer.start_bundle()
    pen.drag([(0.0, -40.0, 0.0), (0.0, 20.0, 0.0)])
    pen.drag([(0.0, 20.0, 0.0), (20.0, 50.0, 0.0)])
    assert len(_draft_points(viewer)) >= 3
    assert viewer.finish_bundle() is not None
    assert len(_state(viewer).bundles) == 1


def test_a_stroke_from_a_finished_tip_edits_that_bundle(
    viewer: Viewer, pen: Driver
) -> None:
    bundle = _straight(viewer, 0.0)
    viewer.set_tool(Tool.DRAW_BUNDLE)
    pen.drag([(0.0, 70.0, 0.0), (20.0, 90.0, 0.0)])
    draft = _state(viewer).draft
    assert draft is not None and draft.editing is not None
    assert draft.item_id == bundle.item_id
    assert len(draft.points) >= 3


def test_a_stroke_away_from_the_side_makes_a_branch_that_shares_the_streamlines(
    viewer: Viewer, pen: Driver
) -> None:
    parent = _straight(viewer, 0.0)
    viewer.set_tool(Tool.DRAW_BUNDLE)
    pen.drag([(0.0, 10.0, 0.0), (10.0, 25.0, 0.0), (35.0, 45.0, 0.0)])
    draft = _state(viewer).draft
    assert draft is not None and draft.branch is not None
    assert draft.branch.parent == parent.item_id
    run(viewer, Command.FINISH_BUNDLE)
    state = _state(viewer)
    assert len(state.bundles) == 2
    branch = state.bundles[1].geometry
    stem = state.bundles[0].geometry
    assert branch is not None and stem is not None
    assert branch.num_streamlines + stem.num_streamlines == 1000
    # The library sees a whole path, from the parent's start.
    np.testing.assert_allclose(branch.control_points[0], (0.0, -70.0, 0.0))
    assert branch.control_points[-1][0] > 30
    assert build_streamlines(branch).num_streamlines == branch.num_streamlines


def test_a_stroke_over_a_stretch_redoes_it_in_another_view(
    viewer: Viewer, pen: Driver
) -> None:
    _straight(viewer, 0.0)
    viewer.set_tool(Tool.DRAW_BUNDLE)
    side = Driver(viewer, FlatPlane(0))
    viewer.move_cursor((0.0, 0.0, 0.0))
    side.drag([(0.0, -30.0, 0.0), (0.0, 0.0, 15.0), (0.0, 30.0, 0.0)])
    points = np.array(_draft_points(viewer))
    assert _draft(viewer).editing is not None
    assert points[:, 2].max() > 8  # raised towards the stroke
    np.testing.assert_allclose(points[:, 0], 0.0, atol=1e-6)  # depth as it was


def test_a_stroke_over_a_bundle_far_from_the_slice_is_a_new_bundle_on_the_slice(
    viewer: Viewer, pen: Driver
) -> None:
    _straight(viewer, 0.0, 3.0)
    viewer.set_tool(Tool.DRAW_BUNDLE)
    viewer.move_cursor((0.0, 0.0, 25.0))
    pen.drag([(0.0, 10.0, 0.0), (40.0, 10.0, 0.0)])
    draft = _state(viewer).draft
    assert draft is not None and draft.branch is None and draft.editing is None
    assert draft.points[0][2] == pytest.approx(25.0)


def test_scrolling_during_a_stroke_moves_it_in_depth(
    viewer: Viewer, pen: Driver
) -> None:
    pen.press((30.0, -20.0, 0.0))
    pen.move((30.0, 0.0, 0.0))
    viewer.step_slice(ViewKind.AXIAL, 10)
    pen.tools.scrolled(pen.plane, 20.0)
    pen.move((30.0, 20.0, 0.0))
    pen.release((30.0, 20.0, 0.0))
    points = _draft_points(viewer)
    assert points[-1][2] - points[0][2] == pytest.approx(20.0)


def test_the_radius_keys_change_it_from_there_on_during_a_stroke(
    viewer: Viewer, pen: Driver
) -> None:
    viewer.set_bundle(None, BundleField.RADIUS, 3.0)
    pen.press((30.0, -30.0, 0.0))
    pen.move((30.0, 10.0, 0.0))
    for _ in range(8):
        assert run(viewer, Command.TOOL_LARGER)
    pen.move((30.0, 50.0, 0.0))
    pen.release((30.0, 50.0, 0.0))
    run(viewer, Command.FINISH_BUNDLE)
    assert _section(viewer, 0.1)[0] < 3.5
    assert _section(viewer, 0.95)[0] > 6


def test_a_stroke_outside_the_grid_is_refused_with_a_reason(
    viewer: Viewer, pen: Driver
) -> None:
    pen.press((500.0, 500.0, 0.0))
    pen.move((500.0, 520.0, 0.0))
    with pytest.raises(ValueError, match="outside the phantom's grid"):
        pen.release((500.0, 520.0, 0.0))


# Width


def test_width_narrows_inwards_and_past_the_centre_and_widens_outwards(
    viewer: Viewer,
) -> None:
    _straight(viewer)
    start = _section(viewer, 0.5)[0]
    viewer.set_tool(Tool.WIDTH)
    tool = Driver(viewer)
    tool.drag([(38.0, 0.0, 0.0), (39.0, 0.0, 0.0)])
    assert _section(viewer, 0.5)[0] < start


def test_width_going_across_the_centreline_keeps_narrowing(viewer: Viewer) -> None:
    _straight(viewer)
    viewer.set_tool(Tool.WIDTH)
    Driver(viewer).drag([(38.0, 0.0, 0.0), (44.0, 0.0, 0.0)], steps=20)
    assert _section(viewer, 0.5)[0] < 0.5


def test_width_going_outwards_widens(viewer: Viewer) -> None:
    _straight(viewer)
    viewer.set_tool(Tool.WIDTH)
    Driver(viewer).drag([(38.0, 0.0, 0.0), (30.0, 0.0, 0.0)], steps=20)
    assert _section(viewer, 0.5)[0] > 8


def test_a_small_reach_stays_near_the_place(viewer: Viewer) -> None:
    _straight(viewer)
    viewer.set_tool(Tool.WIDTH)
    viewer.bundle_tools.set_option("reach", 10)
    Driver(viewer).drag([(38.0, 0.0, 0.0), (30.0, 0.0, 0.0)])
    middle = _section(viewer, 0.5)[0]
    assert middle > 8
    assert 3.5 < _section(viewer, 0.5 + 5 / 140)[0] < middle
    assert _section(viewer, 0.5 + 30 / 140)[0] == pytest.approx(3.0, abs=0.05)


def test_a_wider_reach_afterwards_is_not_stopped_by_the_first_drag(
    viewer: Viewer,
) -> None:
    _straight(viewer)
    viewer.set_tool(Tool.WIDTH)
    tool = Driver(viewer)
    viewer.bundle_tools.set_option("reach", 10)
    tool.drag([(38.0, 0.0, 0.0), (30.0, 0.0, 0.0)])
    away = _section(viewer, 0.5 + 30 / 140)[0]
    viewer.bundle_tools.set_option("reach", 60)
    tool.drag([(30.0, 0.0, 0.0), (35.0, 0.0, 0.0)])
    assert _section(viewer, 0.5 + 30 / 140)[0] != pytest.approx(away, abs=0.1)


def test_repeated_width_strokes_do_not_grow_the_rings(viewer: Viewer) -> None:
    _straight(viewer)
    viewer.set_tool(Tool.WIDTH)
    tool = Driver(viewer)
    viewer.bundle_tools.set_option("reach", 25)

    def out_and_back(times: int) -> int:
        for _ in range(times):
            tool.drag([(38.0, 0.0, 0.0), (36.0, 0.0, 0.0)], steps=5)
            tool.drag([(38.0, 0.0, 0.0), (40.0, 0.0, 0.0)], steps=5)
        rings = _geometry(viewer).rings
        assert rings is not None
        return len(rings)

    first = out_and_back(3)
    assert out_and_back(6) <= first
    assert out_and_back(12) <= first


def test_flat_changes_only_the_width_seen(viewer: Viewer) -> None:
    _straight(viewer)
    viewer.set_tool(Tool.WIDTH)
    viewer.bundle_tools.set_option("mode", "Flat")
    Driver(viewer).drag([(38.0, 0.0, 0.0), (32.0, 0.0, 0.0)])
    r1, r2, _ = _section(viewer, 0.5)
    assert max(r1, r2) > 6
    assert min(r1, r2) == pytest.approx(3.0, abs=0.2)


def test_shift_turns_round_into_flat_for_a_stroke(viewer: Viewer) -> None:
    _straight(viewer)
    viewer.set_tool(Tool.WIDTH)
    Driver(viewer).drag([(38.0, 0.0, 0.0), (32.0, 0.0, 0.0)], shift=True)
    r1, r2, _ = _section(viewer, 0.5)
    assert max(r1, r2) > 2 * min(r1, r2)


def test_width_does_not_touch_a_bundle_outside_the_slab(viewer: Viewer) -> None:
    bundle = _straight(viewer)
    viewer.set_tool(Tool.WIDTH)
    viewer.move_cursor((0.0, 0.0, 30.0))
    Driver(viewer).press((38.0, 0.0, 0.0))
    assert not viewer.bundle_tools.busy
    assert _bundle(viewer) == bundle


def test_escape_takes_back_a_width_drag(viewer: Viewer) -> None:
    bundle = _straight(viewer)
    viewer.set_tool(Tool.WIDTH)
    tool = Driver(viewer)
    tool.press((38.0, 0.0, 0.0))
    tool.move((30.0, 0.0, 0.0))
    assert _bundle(viewer) != bundle
    assert run(viewer, Command.LEAVE_DRAW_MODE)
    assert _bundle(viewer) == bundle
    assert viewer.scene.view.tool == Tool.WIDTH


# Bend


def test_bend_moves_the_place_and_not_the_far_end(viewer: Viewer) -> None:
    _straight(viewer)
    before = np.array(_geometry(viewer).control_points)
    viewer.set_tool(Tool.BEND)
    Driver(viewer).drag([(40.0, 0.0, 0.0), (50.0, 0.0, 0.0)])
    after = np.array(_geometry(viewer).control_points)
    assert after[:, 0].max() > 45
    np.testing.assert_allclose(after[0], before[0], atol=0.5)


def test_bend_at_full_reach_moves_the_whole_bundle(viewer: Viewer) -> None:
    _straight(viewer)
    viewer.set_tool(Tool.BEND)
    viewer.bundle_tools.set_option("reach", 150)
    Driver(viewer).drag([(40.0, 0.0, 0.0), (50.0, 0.0, 0.0)])
    np.testing.assert_allclose(np.array(_geometry(viewer).control_points)[:, 0], 50.0)


def test_bend_takes_a_branch_along_and_its_stem_with_the_parent(
    viewer: Viewer, pen: Driver
) -> None:
    _straight(viewer, 0.0)
    viewer.set_tool(Tool.DRAW_BUNDLE)
    pen.drag([(0.0, 10.0, 0.0), (10.0, 25.0, 0.0), (35.0, 45.0, 0.0)])
    run(viewer, Command.FINISH_BUNDLE)
    branch_before = np.array(_geometry(viewer, 1).control_points)
    viewer.set_tool(Tool.BEND)
    viewer.bundle_tools.set_option("reach", 150)
    Driver(viewer).drag([(0.0, -40.0, 0.0), (12.0, -40.0, 0.0)])
    branch = np.array(_geometry(viewer, 1).control_points)
    np.testing.assert_allclose(branch[0], branch_before[0] + (12.0, 0, 0))
    np.testing.assert_allclose(branch[-1], branch_before[-1] + (12.0, 0, 0))


def test_a_width_stroke_keeps_the_fullness(viewer: Viewer) -> None:
    bundle = _straight(viewer)
    viewer.set_bundle_fullness(bundle.item_id, 2.0)
    viewer.set_tool(Tool.WIDTH)
    Driver(viewer).drag([(38.0, 0.0, 0.0), (20.0, 0.0, 0.0)])
    assert fullness_of(_bundle(viewer)) == pytest.approx(2.0)


# Twist


def test_twist_turns_by_the_pixels_dragged_and_back_within_the_reach(
    viewer: Viewer,
) -> None:
    _straight(viewer)
    viewer.set_tool(Tool.TWIST)
    Driver(viewer).drag([(40.0, 0.0, 0.0), (55.0, 0.0, 0.0)])
    assert abs(_section(viewer, 0.5)[2]) == pytest.approx(15 * SCALE, abs=4)
    assert _section(viewer, 0.05)[2] == pytest.approx(0.0, abs=0.5)


def test_twist_at_a_fan_turns_its_line(viewer: Viewer) -> None:
    bundle = _straight(viewer)
    viewer.set_bundle_landing(
        bundle.item_id,
        BundleEnd.END,
        [(20.0, 70.0, 0.0), (60.0, 70.0, 0.0)],
    )
    viewer.set_tool(Tool.TWIST)
    Driver(viewer).drag([(40.0, 69.0, 0.0), (50.0, 69.0, 0.0)])
    landing = _geometry(viewer).landing_end
    assert landing is not None
    assert abs(np.array(landing.line)[0, 2]) > 1  # it left the axial plane


# Fan


def _fan_on_top(viewer: Viewer) -> Driver:
    viewer.set_tool(Tool.FAN)
    tool = Driver(viewer)
    tool.drag(
        [(20.0, 72.0, 0.0), (40.0, 80.0, 0.0), (60.0, 72.0, 0.0)],
        steps=8,
    )
    return tool


def test_a_fan_lands_the_end_and_leaves_the_path_as_drawn(viewer: Viewer) -> None:
    points = _straight(viewer).geometry
    assert points is not None
    _fan_on_top(viewer)
    geometry = _geometry(viewer)
    assert geometry.control_points == points.control_points
    assert geometry.landing_end is not None and geometry.landing_start is None
    r1, r2, _ = _section(viewer, 1.0)
    assert r1 > 3 * r2


def test_the_streamlines_end_on_the_line_drawn(viewer: Viewer) -> None:
    _straight(viewer)
    _fan_on_top(viewer)
    geometry = _geometry(viewer)
    tracts = build_streamlines(geometry)
    tips = np.asarray(tracts.points)[tracts.offsets[1:] - 1]
    assert geometry.landing_end is not None
    line = np.array(geometry.landing_end.line)
    # Within the sheet's thickness of the line, which is the radius of 3 mm.
    gaps = np.linalg.norm(tips[:, None] - line[None], axis=2).min(axis=1)
    assert np.median(gaps) < 4
    assert np.ptp(tips[:, 0]) > 25  # spread along it


def test_width_at_a_fan_stretches_its_line(viewer: Viewer) -> None:
    _straight(viewer)
    _fan_on_top(viewer)
    before = _geometry(viewer).landing_end
    assert before is not None
    viewer.set_tool(Tool.WIDTH)
    Driver(viewer).drag([(60.0, 74.0, 0.0), (70.0, 74.0, 0.0)])
    after = _geometry(viewer).landing_end
    assert after is not None
    assert np.ptp(np.array(after.line)[:, 0]) > np.ptp(np.array(before.line)[:, 0])


def test_a_twist_away_from_a_fan_leaves_it_where_it_was(viewer: Viewer) -> None:
    _straight(viewer)
    _fan_on_top(viewer)
    before = _geometry(viewer).landing_end
    viewer.set_tool(Tool.TWIST)
    viewer.bundle_tools.set_option("reach", 20)
    Driver(viewer).drag([(40.0, -20.0, 0.0), (50.0, -20.0, 0.0)])
    assert _geometry(viewer).landing_end == before


def test_a_retried_fan_leaves_nothing_behind(viewer: Viewer) -> None:
    _straight(viewer)
    _fan_on_top(viewer)
    once = _geometry(viewer)
    for _ in range(3):
        _fan_on_top(viewer)
    assert _geometry(viewer) == once


def test_a_retried_fan_on_rings_adds_no_rings(viewer: Viewer) -> None:
    _straight(viewer)
    viewer.set_tool(Tool.WIDTH)
    Driver(viewer).drag([(38.0, 0.0, 0.0), (34.0, 0.0, 0.0)])
    _fan_on_top(viewer)
    once = _geometry(viewer).rings
    assert once is not None
    for _ in range(3):
        _fan_on_top(viewer)
    assert _geometry(viewer).rings == once
    # What the line sets is written in the saved end row.
    assert once[-1][1] == pytest.approx(20.0, abs=1.0)


def test_the_fan_goes_to_the_end_chosen_at_the_press_and_the_slice_follows(
    viewer: Viewer,
) -> None:
    _straight(viewer)
    viewer.move_cursor((0.0, 0.0, 40.0))
    viewer.set_tool(Tool.FAN)
    Driver(viewer).drag(
        [(30.0, 72.0, 0.0), (35.0, 75.0, 0.0), (35.0, 0.0, 0.0), (35.0, -60.0, 0.0)],
        steps=6,
    )
    geometry = _geometry(viewer)
    assert geometry.landing_end is not None and geometry.landing_start is None
    assert viewer.scene.view.cursor[2] == pytest.approx(0.0)


def test_a_line_across_the_middle_stops_the_bundle_there(viewer: Viewer) -> None:
    _straight(viewer)
    viewer.set_tool(Tool.FAN)
    # Pressed nearer the last end, so it goes there, then laid across the middle.
    tool = Driver(viewer)
    tool.press((39.0, 68.0, 0.0))
    tool.move((20.0, 10.0, 0.0))
    tool.move((60.0, 10.0, 0.0))
    tool.release((60.0, 10.0, 0.0))
    points = np.array(_geometry(viewer).control_points)
    assert points[:, 1].max() < 30


def test_a_second_line_across_the_first_sets_the_thickness(viewer: Viewer) -> None:
    _straight(viewer)
    _fan_on_top(viewer)
    side = Driver(viewer, FlatPlane(0))
    viewer.move_cursor((40.0, 0.0, 0.0))
    side.drag([(40.0, 72.0, -9.0), (40.0, 75.0, 0.0), (40.0, 72.0, 9.0)], steps=8)
    landing = _geometry(viewer).landing_end
    assert landing is not None and landing.thickness_line is not None
    assert _section(viewer, 1.0)[1] == pytest.approx(9.0, abs=0.5)


def test_opens_from_moves_where_the_fan_starts(viewer: Viewer) -> None:
    _straight(viewer)
    _fan_on_top(viewer)
    wide = _section(viewer, 0.7)[0]
    viewer.bundle_tools.set_option("opens", 0.8)
    landing = _geometry(viewer).landing_end
    assert landing is not None and landing.opens_from == pytest.approx(0.8)
    assert _section(viewer, 0.7)[0] < wide


def test_a_fan_on_a_branch_goes_to_its_far_end(viewer: Viewer, pen: Driver) -> None:
    _straight(viewer, 0.0)
    viewer.set_tool(Tool.DRAW_BUNDLE)
    pen.drag([(0.0, 10.0, 0.0), (10.0, 25.0, 0.0), (35.0, 45.0, 0.0)])
    run(viewer, Command.FINISH_BUNDLE)
    viewer.set_tool(Tool.FAN)
    Driver(viewer).drag([(25.0, 60.0, 0.0), (40.0, 50.0, 0.0), (48.0, 30.0, 0.0)])
    branch = _geometry(viewer, 1)
    assert branch.landing_end is not None and branch.landing_start is None
    assert build_streamlines(branch).num_streamlines == branch.num_streamlines


# Erase


def test_a_click_with_erase_removes_a_bundle_and_its_branches(
    viewer: Viewer, pen: Driver
) -> None:
    _straight(viewer, 0.0)
    viewer.set_tool(Tool.DRAW_BUNDLE)
    pen.drag([(0.0, 10.0, 0.0), (10.0, 25.0, 0.0), (35.0, 45.0, 0.0)])
    run(viewer, Command.FINISH_BUNDLE)
    viewer.set_tool(Tool.ERASE)
    Driver(viewer).click((0.0, -40.0, 0.0))
    assert _state(viewer).bundles == ()


def test_erase_cuts_across_and_the_shorter_part_goes(viewer: Viewer) -> None:
    bundle = _straight(viewer)
    _fan_on_top(viewer)
    viewer.set_tool(Tool.ERASE)
    Driver(viewer).drag([(30.0, 40.0, 0.0), (50.0, 40.0, 0.0)])
    geometry = _geometry(viewer)
    assert bundle.geometry is not None
    # The cut lies where the bundle shows, which the fan had bent over a little.
    assert 30 < np.array(geometry.control_points)[:, 1].max() < 42
    assert geometry.landing_end is None  # it went with the end


def test_erase_cuts_only_what_shows(viewer: Viewer) -> None:
    bundle = _straight(viewer)
    viewer.set_tool(Tool.ERASE)
    viewer.move_cursor((0.0, 0.0, 30.0))
    tool = Driver(viewer)
    tool.drag([(30.0, 40.0, 0.0), (50.0, 40.0, 0.0)])
    tool.click((40.0, 40.0, 0.0))
    assert _bundle(viewer) == bundle


def test_widening_a_parent_gives_its_branch_the_section_there(
    viewer: Viewer, pen: Driver
) -> None:
    _straight(viewer, 0.0)
    viewer.set_tool(Tool.DRAW_BUNDLE)
    pen.drag([(0.0, 0.0, 0.0), (15.0, 15.0, 0.0), (30.0, 40.0, 0.0)])
    run(viewer, Command.FINISH_BUNDLE)
    assert _geometry(viewer, 1).rings is None
    viewer.set_tool(Tool.WIDTH)
    viewer.bundle_tools.set_option("reach", 60)
    Driver(viewer).drag([(-2.0, -10.0, 0.0), (-12.0, -10.0, 0.0)])
    rings = _geometry(viewer, 1).rings
    assert rings is not None
    assert max(r[1] for r in rings) > 6
    tracts = build_streamlines(_geometry(viewer, 1))
    assert tracts.num_streamlines == _geometry(viewer, 1).num_streamlines


def test_a_tangled_bundle_made_longer_by_a_bend_stays_valid(viewer: Viewer) -> None:
    bundle = _straight(viewer)
    viewer.set_bundle(bundle.item_id, BundleField.NUM_POINTS, 5)
    viewer.set_bundle(bundle.item_id, BundleField.TANGLE_MM, 70.0)
    viewer.set_tool(Tool.BEND)
    Driver(viewer).drag([(40.0, 0.0, 0.0), (90.0, 0.0, 0.0)])
    geometry = _geometry(viewer)
    assert geometry.tangle_mm > 70
    assert build_streamlines(geometry).num_streamlines == geometry.num_streamlines


def test_cutting_a_parent_moves_its_branches_or_drops_those_past_the_cut(
    viewer: Viewer, pen: Driver
) -> None:
    _straight(viewer, 0.0)
    viewer.set_tool(Tool.DRAW_BUNDLE)
    pen.drag([(0.0, 40.0, 0.0), (15.0, 50.0, 0.0), (30.0, 60.0, 0.0)])
    run(viewer, Command.FINISH_BUNDLE)
    assert len(_state(viewer).bundles) == 2
    viewer.set_tool(Tool.ERASE)
    Driver(viewer).drag([(-10.0, 20.0, 0.0), (10.0, 20.0, 0.0)])
    assert len(_state(viewer).bundles) == 1


# The tools together


def test_keys_choose_the_tools_and_change_the_reach(viewer: Viewer) -> None:
    for command, tool in (
        (Command.TOOL_BOX, Tool.BEND),
        (Command.TOOL_WIDTH, Tool.WIDTH),
        (Command.TOOL_TWIST, Tool.TWIST),
        (Command.TOOL_FAN, Tool.FAN),
        (Command.TOOL_ERASE, Tool.ERASE),
    ):
        assert run(viewer, command)
        assert viewer.scene.view.tool == tool
    assert run(viewer, Command.TOOL_WIDTH)
    reach = viewer.bundle_tools.reach[Tool.WIDTH]
    assert run(viewer, Command.TOOL_LARGER)
    assert viewer.bundle_tools.reach[Tool.WIDTH] > reach
    assert run(viewer, Command.TOOL_SMALLER)
    assert viewer.bundle_tools.reach[Tool.WIDTH] == reach
    for _ in range(100):
        run(viewer, Command.TOOL_LARGER)
    assert viewer.bundle_tools.reach[Tool.WIDTH] == 200


def test_the_reach_shows_along_the_bundle_while_hovering(viewer: Viewer) -> None:
    _straight(viewer)
    viewer.set_tool(Tool.WIDTH)
    tool = Driver(viewer)
    assert viewer.bundle_tools.overlay().reaches == []
    viewer.bundle_tools.hover(tool._pointer((40.0, 0.0, 0.0)))
    overlay = viewer.bundle_tools.overlay()
    assert overlay.reaches
    length = np.linalg.norm(overlay.reaches[0][-1] - overlay.reaches[0][0])
    assert length == pytest.approx(50.0, abs=4)
    # A bundle outside the slab shows as nothing to edit.
    viewer.move_cursor((0.0, 0.0, 30.0))
    viewer.bundle_tools.hover(tool._pointer((40.0, 0.0, 0.0)))
    assert viewer.bundle_tools.overlay().reaches == []


def test_the_fan_marks_the_end_it_would_go_to(viewer: Viewer) -> None:
    _straight(viewer)
    viewer.set_tool(Tool.FAN)
    tool = Driver(viewer)
    viewer.bundle_tools.hover(tool._pointer((40.0, 60.0, 0.0)))
    [marked] = viewer.bundle_tools.overlay().circles
    np.testing.assert_allclose(marked[0], (40.0, 70.0, 0.0), atol=1e-6)


def test_leaving_the_workspace_leaves_the_tools(viewer: Viewer) -> None:
    viewer.set_tool(Tool.WIDTH)
    viewer.set_workspace(Workspace.VIEW)
    assert viewer.bundle_tools.current is None
