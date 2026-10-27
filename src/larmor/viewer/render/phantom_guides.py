"""Draws the phantom's grid box, the bundle being drawn and the pinned voxel."""

import numpy as np
import pygfx as gfx

from larmor.types import Grid
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

BOX_COLOUR = "#d9a441"
POINT_COLOUR = "#ff4fa3"
MARKER_COLOUR = "#4cc9f0"  # the voxel pinned in the Signal breakdown
POINT_SIZE = 9.0  # px
LINE_WIDTH = 1.5  # px
# The box's 12 edges, as pairs of corner numbers whose bits are x, y and z.
_EDGES = [
    (a, b)
    for a in range(8)
    for b in range(a + 1, 8)
    if bin(a ^ b).count("1") == 1  # corners that differ along one axis
]


class PhantomGuides:
    """Draws the grid box, the bundle being drawn and the pinned voxel in every view."""

    def __init__(self, viewer: Viewer, views: dict[ViewKind, ViewRenderer]) -> None:
        """Adds a group to each view's guides, drawn over the items."""
        self.viewer = viewer
        self.views = views
        self.groups: dict[ViewKind, gfx.Group] = {}
        self._marked: tuple[int, int, int] | None = None  # the voxel drawn
        for kind, view in views.items():
            group = gfx.Group()
            group.add(_box(), _polyline(), _points(), _box(MARKER_COLOUR, 2.0))
            view.guides.add(group)
            self.groups[kind] = group
        self._draw()

    def on_event(self, event: Event) -> None:
        """Redraws when the phantom, the workspace or the pinned voxel changes."""
        if (
            isinstance(event, PhantomChanged)
            or (
                isinstance(event, ViewStateChanged)
                and event.name == ViewField.WORKSPACE
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
        for kind, group in self.groups.items():
            view = self.views[kind]
            view.guides.remove(group)
            view.request_draw()
        self.groups.clear()

    def _draw(self) -> None:
        """Draws the box, the bundle's points and the pinned voxel again."""
        state = self.viewer.scene.phantom
        drawing = state is not None and state.draft is not None
        # Outside the Simulate workspace the box is outlined only while drawing.
        simulating = self.viewer.scene.view.workspace == Workspace.SIMULATE
        boxed = state is not None and (drawing or (simulating and state.show_box))
        corners = _corners(state.grid) if state is not None and boxed else None
        points = (
            np.zeros((0, 3), np.float32)
            if state is None or state.draft is None
            else np.array(state.draft.points, np.float32).reshape(-1, 3)
        )
        self._marked = self._pinned()
        cube = None
        if state is not None and self._marked is not None:
            cube = _voxel_corners(state.grid, self._marked)
        for kind, group in self.groups.items():
            box, line, dots, marker = group.children
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
    material = gfx.PointsMaterial(size=POINT_SIZE, color=POINT_COLOUR, depth_test=False)
    return gfx.Points(gfx.Geometry(positions=np.zeros((1, 3), np.float32)), material)
