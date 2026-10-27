import math

import numpy as np
import pytest

from larmor.viewer import Tool, Viewer
from larmor.viewer.model.phantom_editor import simplify_stroke
from larmor.viewer.model.shortcuts import Command, run
from larmor.viewer.model.view_state import ViewState


def _tool(viewer: Viewer) -> Tool:
    return viewer.scene.view.tool


def test_a_straight_stroke_keeps_its_two_ends() -> None:
    stroke = [(float(x), 2.0 * x, 0.0) for x in range(30)]
    assert simplify_stroke(stroke, 1.0) == [(0.0, 0.0, 0.0), (29.0, 58.0, 0.0)]


def test_a_corner_is_kept_and_wiggles_within_the_tolerance_are_not() -> None:
    wiggle = [(float(x), 0.3 * (-1) ** x, 0.0) for x in range(11)]
    up = [(10.0, float(y), 0.0) for y in range(1, 11)]
    kept = simplify_stroke(wiggle + up, 1.0)
    assert [tuple(np.round(p)) for p in kept] == [(0, 0, 0), (10, 0, 0), (10, 10, 0)]


def test_a_long_curve_gives_at_most_twelve_points_with_both_ends() -> None:
    circle = [(20 * math.cos(t), 20 * math.sin(t), 0.0) for t in np.linspace(0, 6, 400)]
    kept = simplify_stroke(circle, 0.01)
    assert 2 < len(kept) <= 12
    assert kept[0] == circle[0] and kept[-1] == circle[-1]


def test_a_stroke_that_never_moves_gives_one_point() -> None:
    assert simplify_stroke([(1.0, 2.0, 3.0)] * 5, 1.0) == [(1.0, 2.0, 3.0)]


def test_the_bundle_tool_is_chosen_exactly_while_a_bundle_is_drawn() -> None:
    viewer = Viewer()
    viewer.start_bundle()
    assert _tool(viewer) == Tool.DRAW_BUNDLE
    viewer.cancel_bundle()
    assert _tool(viewer) == Tool.CURSOR
    viewer.set_tool(Tool.DRAW_BUNDLE)
    assert viewer.drawing
    viewer.add_bundle_point((0.0, 0.0, 0.0))
    viewer.add_bundle_point((10.0, 0.0, 0.0))
    viewer.set_tool(Tool.SPHERE)  # finishes the bundle, which has two points
    assert not viewer.drawing and _tool(viewer) == Tool.SPHERE
    state = viewer.phantom
    assert state is not None and len(state.bundles) == 1


def test_a_stroke_outside_the_grid_is_refused() -> None:
    viewer = Viewer()
    viewer.start_bundle()
    with pytest.raises(ValueError, match="outside the phantom's grid"):
        viewer.add_bundle_stroke([(1e4, 0.0, 0.0), (2e4, 0.0, 0.0)])
    assert viewer.add_bundle_stroke([(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]) == 2


def test_a_view_state_keeps_its_tool_but_not_a_bundle_tool_without_drawing() -> None:
    viewer = Viewer()
    viewer.set_view_state(ViewState(tool=Tool.BOX))
    assert _tool(viewer) == Tool.BOX
    viewer.set_view_state(ViewState(tool=Tool.DRAW_BUNDLE))
    assert _tool(viewer) == Tool.CURSOR


def test_escape_stops_drawing_first_and_chooses_the_cursor_after() -> None:
    viewer = Viewer()
    viewer.set_tool(Tool.ZOOM_BOX)
    viewer.start_bundle()
    assert run(viewer, Command.LEAVE_DRAW_MODE)
    assert not viewer.drawing and _tool(viewer) == Tool.CURSOR
    viewer.set_tool(Tool.ZOOM_BOX)
    assert run(viewer, Command.LEAVE_DRAW_MODE)
    assert _tool(viewer) == Tool.CURSOR
    assert not run(viewer, Command.LEAVE_DRAW_MODE)
