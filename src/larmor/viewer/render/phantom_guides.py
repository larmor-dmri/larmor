"""Draws the phantom's box, the bundle being drawn, its pinned voxel and the tools."""

from collections.abc import Sequence

import numpy as np
import pygfx as gfx

from larmor.types import Grid
from larmor.viewer.model.bundle_tools import OUTLINE_COLOUR, POINT_COLOUR, Overlay
from larmor.viewer.model.events import (
    BreakdownChanged,
    Event,
    PhantomChanged,
    ViewStateChanged,
)
from larmor.viewer.model.item import ViewKind
from larmor.viewer.model.view_state import ViewField, Workspace
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.render.view import ViewRenderer
from larmor.viewer.render.volume_adapter import SLICE_AXIS

BOX_COLOUR = "#d9a441"
MARKER_COLOUR = "#4cc9f0"  # the voxel pinned in the Signal breakdown
POINT_SIZE = 9.0  # px
LINE_WIDTH = 1.5  # px
REACH_WIDTH = 3.0  # px, the stretch a tool would change
REACH_OPACITY = 0.55
TICK_SIZE = 5.0  # px, the rows of a bundle along it
CIRCLE_SIZE = 16.0  # px, a marked tip
# The box's 12 edges, as pairs of corner numbers whose bits are x, y and z.
_EDGES = [
    (a, b)
    for a in range(8)
    for b in range(a + 1, 8)
    if bin(a ^ b).count("1") == 1  # corners that differ along one axis
]


class _ToolLines:
    """Holds in one view what the drawing tools draw over it."""

    def __init__(self, group: gfx.Group) -> None:
        """Adds the empty lines and points to a view's group, hidden."""
        self.outlines = _line(OUTLINE_COLOUR)
        self.under_pointer = _line(OUTLINE_COLOUR)
        self.reaches = _line(OUTLINE_COLOUR, REACH_WIDTH, REACH_OPACITY)
        self.dashed = _line(OUTLINE_COLOUR, dashed=True)
        self.strokes: dict[str, gfx.Line] = {}
        self.ticks = gfx.Points(_empty(), _dots(OUTLINE_COLOUR, TICK_SIZE))
        self.circles = gfx.Points(
            _empty(),
            gfx.PointsMarkerMaterial(
                marker="ring",
                size=CIRCLE_SIZE,
                color=OUTLINE_COLOUR,
                edge_width=0.0,
                depth_test=False,
            ),
        )
        self.ticks.visible = self.circles.visible = False
        self.group = group
        group.add(
            self.outlines,
            self.under_pointer,
            self.reaches,
            self.dashed,
            self.ticks,
            self.circles,
        )
        self._shown: bytes = b""

    def show(self, overlay: Overlay, under_pointer: bool) -> bool:
        """Shows what the tools draw now.

        Args:
            overlay: What they draw.
            under_pointer: Whether this view is the one under the pointer.

        Returns:
            Whether anything changed since the last time.
        """
        local = overlay.under_pointer if under_pointer else []
        strokes = [line for line, _ in overlay.strokes]
        parts = [
            overlay.outlines,
            local,
            overlay.reaches,
            overlay.dashed,
            strokes,
            overlay.ticks,
            overlay.circles,
        ]
        shown = b"|".join(p.tobytes() for part in parts for p in part)
        if shown == self._shown:
            return False
        self._shown = shown
        _set_lines(self.outlines, overlay.outlines)
        _set_lines(self.under_pointer, local)
        _set_lines(self.reaches, overlay.reaches)
        _set_lines(self.dashed, overlay.dashed)
        _set_points(self.ticks, overlay.ticks)
        _set_points(self.circles, overlay.circles)
        by_colour: dict[str, list[np.ndarray]] = {}
        for line, colour in overlay.strokes:
            by_colour.setdefault(colour, []).append(line)
        for colour in by_colour:
            if colour not in self.strokes:
                self.strokes[colour] = _line(colour)
                self.group.add(self.strokes[colour])
        for colour, stroke in self.strokes.items():
            _set_lines(stroke, by_colour.get(colour, []))
        return True


class PhantomGuides:
    """Draws the grid box, the bundle being drawn and the pinned voxel in every view.

    It also draws what the phantom's drawing tools show, such as how far a tool reaches.
    """

    def __init__(self, viewer: Viewer, views: dict[ViewKind, ViewRenderer]) -> None:
        """Adds a group to each view's guides, drawn over the items."""
        self.viewer = viewer
        self.views = views
        self.groups: dict[ViewKind, gfx.Group] = {}
        self.tool_lines: dict[ViewKind, _ToolLines] = {}
        self._marked: tuple[int, int, int] | None = None  # the voxel drawn
        for kind, view in views.items():
            group = gfx.Group()
            group.add(_box(), _polyline(), _points(), _box(MARKER_COLOUR, 2.0))
            view.guides.add(group)
            self.groups[kind] = group
            self.tool_lines[kind] = _ToolLines(group)
        self._stop_tools = viewer.bundle_tools.subscribe(self._draw_tools)
        self._draw()

    def on_event(self, event: Event) -> None:
        """Redraws when the phantom, the workspace or the pinned voxel changes."""
        if (
            isinstance(event, PhantomChanged)
            or (
                isinstance(event, ViewStateChanged)
                and event.name in (ViewField.WORKSPACE, ViewField.TOOL)
            )
            # Not for every voxel the pointer passes over.
            or (isinstance(event, BreakdownChanged) and self._pinned() != self._marked)
        ):
            self._draw()

    def _pinned(self) -> tuple[int, int, int] | None:
        """Returns the voxel pinned in the Signal breakdown, if any."""
        breakdown = self.viewer.scene.breakdown
        if self.viewer.scene.phantom is None or breakdown is None:
            return None
        return breakdown.voxel if breakdown.pinned else None

    def close(self) -> None:
        """Removes the guides from the views."""
        self._stop_tools()
        for kind, group in self.groups.items():
            view = self.views[kind]
            view.guides.remove(group)
            view.request_draw()
        self.groups.clear()

    def _draw(self) -> None:
        """Draws the box, the bundle's points and the pinned voxel again."""
        state = self.viewer.scene.phantom
        drawing = state is not None and state.draft is not None
        # Outside the SymDWI workspace the box is outlined only while drawing.
        simulating = self.viewer.scene.view.workspace == Workspace.SIMULATE
        boxed = state is not None and (drawing or (simulating and state.show_box))
        corners = _corners(state.grid) if state is not None and boxed else None
        points = (
            np.zeros((0, 3), np.float32)
            if state is None or state.draft is None
            else np.array(state.draft.own_points, np.float32).reshape(-1, 3)
        )
        self._marked = self._pinned()
        cube = None
        if state is not None and self._marked is not None:
            cube = _voxel_corners(state.grid, self._marked)
        for kind, group in self.groups.items():
            box, line, dots, marker = group.children[:4]
            box.visible = corners is not None
            if corners is not None:
                box.geometry = _edges(corners)
            marker.visible = cube is not None
            if cube is not None:
                marker.geometry = _edges(cube)
            line.visible = len(points) >= 2
            dots.visible = len(points) >= 1
            if len(points):
                line.geometry = gfx.Geometry(positions=points)
                dots.geometry = gfx.Geometry(positions=points)
            self.views[kind].request_draw()
        self._draw_tools()

    def _draw_tools(self) -> None:
        """Draws what the drawing tools show, in every view."""
        overlay = self.viewer.bundle_tools.overlay()
        for kind, lines in self.tool_lines.items():
            here = SLICE_AXIS.get(kind) == overlay.under_pointer_axis
            if lines.show(overlay, here):
                self.views[kind].request_draw()


def _corners(grid: Grid) -> np.ndarray:
    """Returns the 8 corners of the grid's box, at the voxels' outer edges, in mm."""
    shape = np.array(grid.shape, dtype=float)
    bits = np.array([[(c >> k) & 1 for k in range(3)] for c in range(8)], dtype=float)
    voxels = bits * shape - 0.5
    corners: np.ndarray = voxels @ grid.affine[:3, :3].T + grid.affine[:3, 3]
    return corners


def _voxel_corners(grid: Grid, voxel: tuple[int, int, int]) -> np.ndarray:
    """Returns the 8 corners of one voxel, in world mm."""
    bits = np.array([[(c >> k) & 1 for k in range(3)] for c in range(8)], dtype=float)
    voxels = np.asarray(voxel, dtype=float) + bits - 0.5
    corners: np.ndarray = voxels @ grid.affine[:3, :3].T + grid.affine[:3, 3]
    return corners


def _edges(corners: np.ndarray) -> gfx.Geometry:
    """Returns the 12 edges of a box with these corners, as line segments."""
    ends = np.array([corners[i] for edge in _EDGES for i in edge])
    return gfx.Geometry(positions=ends.astype(np.float32))


def _box(colour: str = BOX_COLOUR, thickness: float = 1.0) -> gfx.Line:
    """Returns an empty box outline, drawn over the items."""
    material = gfx.LineSegmentMaterial(
        thickness=thickness, color=colour, depth_test=False
    )
    return gfx.Line(gfx.Geometry(positions=np.zeros((2, 3), np.float32)), material)


def _polyline() -> gfx.Line:
    """Returns an empty line for the bundle being drawn."""
    material = gfx.LineMaterial(
        thickness=LINE_WIDTH, color=POINT_COLOUR, depth_test=False
    )
    return gfx.Line(gfx.Geometry(positions=np.zeros((2, 3), np.float32)), material)


def _points() -> gfx.Points:
    """Returns empty points for the bundle being drawn."""
    return gfx.Points(_empty(), _dots(POINT_COLOUR, POINT_SIZE))


def _dots(colour: str, size: float) -> gfx.PointsMaterial:
    """Returns the material of round dots a fixed size on the screen."""
    return gfx.PointsMaterial(size=size, color=colour, depth_test=False)


def _empty() -> gfx.Geometry:
    """Returns the geometry of an object with nothing to show yet."""
    return gfx.Geometry(positions=np.zeros((2, 3), np.float32))


def _line(
    colour: str,
    thickness: float = LINE_WIDTH,
    opacity: float = 1.0,
    *,
    dashed: bool = False,
) -> gfx.Line:
    """Returns an empty, hidden line of the tools, drawn over the items."""
    material = gfx.LineMaterial(
        thickness=thickness,
        color=colour,
        opacity=opacity,
        alpha_mode="auto" if opacity >= 1 else "blend",
        dash_pattern=(4, 3) if dashed else (),
        depth_test=False,
    )
    line = gfx.Line(_empty(), material)
    line.visible = False
    return line


def _set_lines(line: gfx.Line, polylines: Sequence[np.ndarray]) -> None:
    """Shows polylines in one line object, broken by NaN points; hides it for none."""
    line.visible = bool(polylines)
    if polylines:
        breaks = np.full((1, 3), np.nan, np.float32)
        parts = [part for p in polylines for part in (p.astype(np.float32), breaks)]
        line.geometry = gfx.Geometry(positions=np.vstack(parts[:-1]))


def _set_points(points: gfx.Points, groups: Sequence[np.ndarray]) -> None:
    """Shows groups of points (N, 3) in one object; hides it for none."""
    points.visible = bool(groups)
    if groups:
        points.geometry = gfx.Geometry(positions=np.vstack(groups).astype(np.float32))
