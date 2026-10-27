import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

from larmor.simulation.bundles import centreline_spline
from larmor.viewer import Tool, Viewer
from larmor.viewer.model.phantom_editor import (
    BUNDLE_POINTS_DRAWN,
    STROKE_POINTS,
    simplify_stroke,
)
from larmor.viewer.model.shortcuts import Command, run
from larmor.viewer.model.view_state import ViewState


def _tool(viewer: Viewer) -> Tool:
    return viewer.scene.view.tool


def test_a_straight_stroke_keeps_its_two_ends() -> None:
    stroke = [(float(x), 2.0 * x, 0.0) for x in range(30)]
    assert simplify_stroke(stroke, 1.0) == [(0.0, 0.0, 0.0), (29.0, 58.0, 0.0)]


def test_an_arc_gives_evenly_spaced_points_whose_centreline_follows_it() -> None:
    # A parabola, sampled densely near one end, so the samples are uneven.
    x = 30.0 * np.linspace(0.0, 1.0, 300) ** 2
    stroke = [(float(a), a * a / 20.0, 0.0) for a in x]
    kept = np.array(simplify_stroke(stroke, 0.5))
    assert 2 < len(kept) <= STROKE_POINTS
    fine = np.linspace(0.0, 30.0, 100001)
    arc = np.concatenate(
        [[0.0], np.cumsum(np.hypot(np.diff(fine), np.diff(fine**2 / 20.0)))]
    )
    gaps = np.diff(np.interp(kept[:, 0], fine, arc))
    np.testing.assert_allclose(gaps, gaps.mean(), rtol=0.01)
    centreline, _ = centreline_spline(kept)
    curve = centreline(np.linspace(0.0, 1.0, 2000))
    nearest = cKDTree(np.column_stack([fine, fine**2 / 20.0, 0 * fine]))
    assert nearest.query(curve)[0].max() <= 0.5


def test_a_long_curve_gives_at_most_the_cap_of_points_with_both_ends() -> None:
    circle = [(20 * math.cos(t), 20 * math.sin(t), 0.0) for t in np.linspace(0, 6, 400)]
    kept = simplify_stroke(circle, 0.01)
    assert 2 < len(kept) <= STROKE_POINTS
    assert kept[0] == circle[0] and kept[-1] == circle[-1]


def test_a_drawn_bundle_is_shown_with_smooth_bends() -> None:
    viewer = Viewer()
    viewer.start_bundle()
    viewer.add_bundle_stroke([(-20.0, 0.0, 0.0), (0.0, 15.0, 0.0), (20.0, 0.0, 0.0)])
    item = viewer.finish_bundle()
    assert item.options.points == BUNDLE_POINTS_DRAWN  # type: ignore[attr-defined]


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


def test_backspace_takes_back_a_whole_stroke_and_a_click_alone() -> None:
    viewer = Viewer()
    viewer.start_bundle()
    viewer.add_bundle_point((-20.0, 0.0, 0.0))
    added = viewer.add_bundle_stroke(
        [(0.0, 0.0, 0.0), (10.0, 10.0, 0.0), (20.0, 0.0, 0.0)]
    )
    assert added == 3
    viewer.remove_bundle_point()
    assert _draft_points(viewer) == ((-20.0, 0.0, 0.0),)
    viewer.remove_bundle_point()
    assert _draft_points(viewer) == ()


def _draft_points(viewer: Viewer) -> tuple[tuple[float, float, float], ...]:
    state = viewer.scene.phantom
    assert state is not None and state.draft is not None
    return state.draft.points


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
