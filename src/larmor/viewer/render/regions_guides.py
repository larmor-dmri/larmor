"""Draws the regions, the detail windows' boxes and the shapes being drawn."""

import math
from collections.abc import Sequence

import numpy as np
import pygfx as gfx

from larmor.viewer.model.camera import screen_axes
from larmor.viewer.model.detail import VIEW_AXIS, detail_field
from larmor.viewer.model.events import (
    DetailChanged,
    DetailClosed,
    DetailOpened,
    Event,
    ItemAdded,
    ItemRemoved,
    OptionChanged,
    ViewStateChanged,
)
from larmor.viewer.model.item import Category, Item, ViewKind
from larmor.viewer.model.options import RegionOptions, RegionShape
from larmor.viewer.model.regions import Circle, Rectangle, cross_section
from larmor.viewer.model.view_state import ViewField
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.render.view import Vec3, ViewRenderer

LINE_WIDTH = 1.5  # px, as the phantom's guides
SELECTED_WIDTH = 2.5  # px, the selected region's outline
HANDLE_SIZE = 8.0  # px, the centre dot and the resize handles
DASH = (4.0, 3.0)  # px on and off, a sphere cut off its centre
CIRCLE_POINTS = 97  # the points of a circle, the first repeated at the end
DETAIL_COLOUR = "#ffd166"  # the detail windows' boxes
SKETCH_COLOUR = "#ffffff"  # a shape being drawn, before it is made
# The view-state fields whose change moves the regions' drawing.
FOLLOWED = (ViewField.CURSOR, ViewField.SELECTED, ViewField.CONVENTION)


class RegionGuides:
    """Draws each visible region in every view, the selected one with its handles.

    A 2D view draws a sphere's or a box's cross-section with its slice over the
    items; the 3D view draws the regions as translucent shapes among them. Masks
    are not drawn, as their overlay shows them, and label regions have their own
    drawing.
    """

    def __init__(self, viewer: Viewer, views: dict[ViewKind, ViewRenderer]) -> None:
        """Adds a group to each 2D view's guides; the 3D view's waits for a region."""
        self.viewer = viewer
        self.views = views
        self.groups: dict[ViewKind, gfx.Group] = {}
        for kind, view in views.items():
            group = gfx.Group()
            if kind in VIEW_AXIS:
                view.guides.add(group)
            self.groups[kind] = group
        self._draw()

    def on_event(self, event: Event) -> None:
        """Draws the regions again when one changes, or the slices or selection do."""
        if isinstance(event, (ItemAdded, ItemRemoved)):
            redraw = event.item.category == Category.REGIONS
        elif isinstance(event, OptionChanged):
            redraw = event.item.category == Category.REGIONS
        elif isinstance(event, ViewStateChanged):
            redraw = event.name in FOLLOWED and bool(self._regions())
        else:
            redraw = False
        if redraw:
            self._draw()

    def close(self) -> None:
        """Removes the drawings from the views."""
        for kind, group in self.groups.items():
            view = self.views[kind]
            if kind in VIEW_AXIS:
                view.guides.remove(group)
            elif group.parent is not None:
                view.remove(group)
            view.request_draw()
        self.groups.clear()

    def _regions(self) -> list[Item]:
        """Returns the visible regions with a shape, in scene tree order."""
        return [
            r
            for r in self.viewer.scene.in_category(Category.REGIONS)
            if isinstance(r.options, RegionOptions)
            and r.options.visible
            and r.options.shape in (RegionShape.SPHERE, RegionShape.BOX)
        ]

    def _draw(self) -> None:
        """Draws every region again in every view."""
        selected = self.viewer.scene.view.selected
        cursor = self.viewer.scene.view.cursor
        convention = self.viewer.scene.view.convention
        regions = self._regions()
        for kind, group in self.groups.items():
            group.clear()
            for region in regions:
                o = region.options
                assert isinstance(o, RegionOptions)
                chosen = region.id == selected
                if kind in VIEW_AXIS:
                    axis = VIEW_AXIS[kind]
                    shape = cross_section(o, axis, cursor[axis])
                    if shape is not None:
                        right, _ = screen_axes(kind, convention)
                        group.add(*_outline(shape, axis, o, chosen, right))
                else:
                    group.add(_solid(o, chosen))
            if kind not in VIEW_AXIS:
                # In the scene only while there are regions, among the items.
                if group.children and group.parent is None:
                    self.views[kind].add(group)
                elif not group.children and group.parent is not None:
                    self.views[kind].remove(group)
            self.views[kind].request_draw()


def _outline(
    shape: Circle | Rectangle,
    axis: int,
    options: RegionOptions,
    chosen: bool,
    right: Vec3,
) -> list[gfx.WorldObject]:
    """Returns a cross-section's outline and, when chosen, its handles."""
    width = SELECTED_WIDTH if chosen else LINE_WIDTH
    dashed = isinstance(shape, Circle) and not shape.on_slice
    material = gfx.LineMaterial(
        thickness=width,
        color=options.colour,
        opacity=options.opacity_2d,
        dash_pattern=DASH if dashed else (),
        depth_test=False,
    )
    material.alpha_mode = "solid" if options.opacity_2d >= 1 else "blend"
    points = (
        _circle(shape, axis) if isinstance(shape, Circle) else _rectangle(shape, axis)
    )
    objects: list[gfx.WorldObject] = [
        gfx.Line(gfx.Geometry(positions=points.astype(np.float32)), material)
    ]
    if chosen:
        if isinstance(shape, Circle):
            centre = np.asarray(shape.centre, dtype=float)
            handles = [centre, centre + shape.radius * np.asarray(right)]
        else:
            handles = [
                (np.asarray(shape.low) + np.asarray(shape.high)) / 2,
                *points[:4],
            ]
        objects.append(_handles(handles, options.colour))
    return objects


def _circle(shape: Circle, axis: int) -> np.ndarray:
    """Returns the points of a circle in the plane across `axis`, closed."""
    first, second = (a for a in range(3) if a != axis)
    angles = np.linspace(0.0, 2 * math.pi, CIRCLE_POINTS)
    points = np.tile(np.asarray(shape.centre, dtype=float), (CIRCLE_POINTS, 1))
    points[:, first] += shape.radius * np.cos(angles)
    points[:, second] += shape.radius * np.sin(angles)
    return points


def _rectangle(shape: Rectangle | tuple[Vec3, Vec3], axis: int) -> np.ndarray:
    """Returns a rectangle's four corners in the plane across `axis`, closed."""
    low, high = (shape.low, shape.high) if isinstance(shape, Rectangle) else shape
    first, second = (a for a in range(3) if a != axis)
    corners = []
    for u, v in ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0)):
        corner = np.array(low, dtype=float)
        corner[first] = (low, high)[u][first]
        corner[second] = (low, high)[v][second]
        corners.append(corner)
    return np.array(corners)


def _handles(points: Sequence[np.ndarray], colour: str) -> gfx.Points:
    """Returns square dots at points, drawn over everything."""
    material = gfx.PointsMarkerMaterial(
        size=HANDLE_SIZE,
        color=colour,
        marker="square",
        edge_width=0.0,
        depth_test=False,
    )
    positions = np.array(points, dtype=np.float32).reshape(-1, 3)
    return gfx.Points(gfx.Geometry(positions=positions), material)


def _solid(options: RegionOptions, chosen: bool) -> gfx.Mesh:
    """Returns a region as a translucent sphere or box at its place in 3D."""
    if options.shape == RegionShape.SPHERE:
        geometry = gfx.sphere_geometry(options.radius, 48, 24)
    else:
        x, y, z = options.size
        geometry = gfx.box_geometry(x, y, z)
    opacity = min(options.opacity_3d * (1.6 if chosen else 1.0), 1.0)
    material = gfx.MeshPhongMaterial(
        color=options.colour, opacity=opacity, depth_write=opacity >= 1
    )
    material.alpha_mode = "solid" if opacity >= 1 else "blend"
    mesh = gfx.Mesh(geometry, material)
    mesh.local.position = tuple(options.centre)
    return mesh


class DetailBoxes:
    """Outlines each detail window's rectangle in its 2D view, over the items."""

    def __init__(self, viewer: Viewer, views: dict[ViewKind, ViewRenderer]) -> None:
        """Adds a group to each 2D view's guides."""
        self.viewer = viewer
        self.views = {k: v for k, v in views.items() if k in VIEW_AXIS}
        self.visible = True
        self.groups: dict[ViewKind, gfx.Group] = {}
        for kind, view in self.views.items():
            group = gfx.Group()
            view.guides.add(group)
            self.groups[kind] = group
        self._draw()

    def on_event(self, event: Event) -> None:
        """Draws the boxes again when a detail or the cursor changes."""
        if isinstance(event, (DetailOpened, DetailChanged, DetailClosed)) or (
            isinstance(event, ViewStateChanged)
            and event.name in (ViewField.CURSOR, ViewField.MAIN_VOLUME)
            and self.viewer.scene.details
        ):
            self._draw()

    def show(self, visible: bool) -> None:
        """Shows or hides the boxes, as a figure that draws its own does."""
        self.visible = visible
        for kind, group in self.groups.items():
            group.visible = visible
            self.views[kind].request_draw()

    def close(self) -> None:
        """Removes the boxes from the views."""
        for kind, group in self.groups.items():
            self.views[kind].guides.remove(group)
            self.views[kind].request_draw()
        self.groups.clear()

    def _draw(self) -> None:
        """Outlines every detail's rectangle in its view."""
        grid = self.viewer._detail_grid()
        cursor = self.viewer.scene.view.cursor
        for kind, group in self.groups.items():
            group.clear()
            if grid is None:
                continue
            for detail in self.viewer.scene.details.values():
                if detail.view != kind:
                    continue
                low, high = detail_field(grid, kind, detail.box, cursor)
                axis = VIEW_AXIS[kind]
                low = _on_axis(low, axis, cursor[axis])
                high = _on_axis(high, axis, cursor[axis])
                material = gfx.LineMaterial(
                    thickness=LINE_WIDTH, color=DETAIL_COLOUR, depth_test=False
                )
                points = _rectangle((low, high), axis).astype(np.float32)
                group.add(gfx.Line(gfx.Geometry(positions=points), material))
            self.views[kind].request_draw()


def _on_axis(point: Vec3, axis: int, value: float) -> Vec3:
    """Returns a point with its coordinate on `axis` replaced."""
    moved = list(point)
    moved[axis] = value
    return (moved[0], moved[1], moved[2])


class Sketch:
    """Draws a shape being drawn with the mouse in one 2D view, until it is made.

    It holds a line (a freehand stroke or a rectangle) in the view's guides; the
    views' input shows and clears it.
    """

    def __init__(self, view: ViewRenderer) -> None:
        """Adds an empty, hidden line to the view's guides."""
        self.view = view
        material = gfx.LineMaterial(
            thickness=LINE_WIDTH, color=SKETCH_COLOUR, depth_test=False
        )
        self.line = gfx.Line(
            gfx.Geometry(positions=np.zeros((2, 3), np.float32)), material
        )
        self.line.visible = False
        view.guides.add(self.line)

    def show_line(
        self, points: Sequence[Vec3] | np.ndarray, colour: str = SKETCH_COLOUR
    ) -> None:
        """Draws a polyline through points, world mm."""
        positions = np.array(points, dtype=np.float32).reshape(-1, 3)
        if len(positions) < 2:
            positions = np.repeat(positions[:1], 2, axis=0)
        self.line.geometry = gfx.Geometry(positions=positions)
        self.line.material.color = colour
        self.line.visible = len(points) > 0
        self.view.request_draw()

    def show_box(
        self, low: Vec3, high: Vec3, axis: int, colour: str = SKETCH_COLOUR
    ) -> None:
        """Draws a rectangle between two corners in the plane across `axis`."""
        corners = _rectangle((low, high), axis)
        self.show_line(corners, colour)

    def clear(self) -> None:
        """Hides the shape."""
        if self.line.visible:
            self.line.visible = False
            self.view.request_draw()
