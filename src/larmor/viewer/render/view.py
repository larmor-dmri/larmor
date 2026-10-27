"""Holds the pygfx renderer, scene and camera of one view."""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pygfx as gfx
from rendercanvas import BaseRenderCanvas

from larmor.viewer.model.camera import (
    Camera2D,
    Convention,
    field_of_view,
    screen_axes,
    view_directions,
)
from larmor.viewer.model.item import ViewKind

BACKGROUND = "#000000"
# Share of the canvas's resolution a view renders at while the user interacts.
INTERACTIVE_SCALE = 0.5
Vec3 = tuple[float, float, float]

# Finds a point under the pointer that the GPU's picking misses, such as on a thin
# line: given the view and the canvas point, it returns a world point or None.
type Picker = Callable[["ViewRenderer", float, float], np.ndarray | None]

# Where each 2D view looks from in the default convention.
CAMERA_DIRECTIONS_UP: dict[ViewKind, tuple[Vec3, Vec3]] = {
    kind: view_directions(kind, Convention.RAS)
    for kind in (ViewKind.AXIAL, ViewKind.CORONAL, ViewKind.SAGITTAL)
}


class ViewRenderer:
    """Draws one view's scene into a canvas, updating on demand."""

    def __init__(self, canvas: BaseRenderCanvas, kind: ViewKind) -> None:
        """Creates the renderer for a view."""
        self.kind = kind
        self.renderer = gfx.WgpuRenderer(canvas)
        self.scene = gfx.Scene()
        self.background = gfx.Background.from_color(BACKGROUND)
        self.scene.add(self.background)
        self.camera: gfx.PerspectiveCamera = (
            gfx.PerspectiveCamera(50)
            if kind == ViewKind.THREE_D
            else gfx.OrthographicCamera()
        )
        # The crosshair, drawn over the items with the view's camera.
        self.guides = gfx.Scene()
        # Text and colour bars, drawn over the scene in screen space (-1 to 1).
        self.overlay = gfx.Scene()
        self.overlay_camera = gfx.NDCCamera()
        self.overlay_visible = True
        # The window's controls, such as the enlarge button, drawn last with the
        # overlay's camera, whether the overlay is shown or not; never off screen.
        self.controls = gfx.Scene()
        self.canvas = canvas
        # The pickers `pick` asks besides the GPU, such as the 3D view's lines'.
        self.pickers: list[Picker] = []
        # A 2D view's camera, as the world box it shows at zoom 1, its zoom and pan,
        # and the side of the slice it looks from.
        self.field = field_of_view(None)
        self.camera_2d = Camera2D()
        self.convention = Convention.RAS
        # While the user interacts, the view draws at INTERACTIVE_SCALE (`scale`)
        # if lower_resolution is on, and only once it has something new to draw.
        self.interacting = False
        self.lower_resolution = False
        self.scale = 1.0
        self._drew_interacting = False  # asked to draw since the interaction began
        self._full_ratio = 1.0  # the canvas's own pixel ratio, at scale 1
        self._full_size = (1.0, 1.0)  # its logical size then, which rounding can change
        self._placed: tuple[object, ...] | None = None  # the camera as last placed
        if kind in CAMERA_DIRECTIONS_UP:
            self._place_camera()
        canvas.request_draw(self.draw)

    def add(self, obj: gfx.WorldObject) -> None:
        """Adds an object to the view's scene."""
        self.scene.add(obj)
        self.request_draw()

    def remove(self, obj: gfx.WorldObject) -> None:
        """Removes an object from the view's scene."""
        self.scene.remove(obj)
        self.request_draw()

    def request_draw(self) -> None:
        """Draws the view soon; at reduced resolution while the user interacts."""
        self._drew_interacting |= self.interacting
        if self.interacting and self.lower_resolution and self.scale == 1.0:
            self._set_scale(INTERACTIVE_SCALE)  # a new size draws the view
        else:
            self.canvas.request_draw()

    def set_lower_resolution(self, lower: bool) -> None:
        """Draws at reduced resolution while the user interacts, or never."""
        self.lower_resolution = lower
        if not lower and self.scale != 1.0:
            self._set_scale(1.0)

    def start_interaction(self) -> None:
        """Marks the user as interacting until end_interaction."""
        self.interacting = True

    def end_interaction(self) -> None:
        """Draws once in full detail, if the view drew while the user interacted."""
        self.interacting = False
        drew, self._drew_interacting = self._drew_interacting, False
        if self.scale != 1.0:
            self._set_scale(1.0)  # a new size draws the view
        elif drew:
            # The 3D view drew with fewer samples and streamlines meanwhile.
            self.canvas.request_draw()

    def set_colours(self, background: str, guides: str) -> None:
        """Sets the background colour and the colour of the lines over the items."""
        self.background.material.set_colors(background)
        for obj in self.guides.children:
            if isinstance(obj, gfx.Line):
                obj.material.color = guides
        self.request_draw()

    def show_overlay(self, visible: bool) -> None:
        """Shows or hides the text and colour bars."""
        self.overlay_visible = visible
        self.request_draw()

    def set_camera(
        self,
        field: tuple[Vec3, Vec3] | None = None,
        camera: Camera2D | None = None,
        convention: Convention | None = None,
    ) -> None:
        """Points a 2D view's camera at a box, with a zoom, pan and side."""
        if field is not None:
            self.field = field
        if camera is not None:
            self.camera_2d = camera
        if convention is not None:
            self.convention = convention
        if self._place_camera():
            self.request_draw()

    def to_world(self, x: float, y: float) -> Vec3:
        """Returns the world point (mm) under a canvas point given in logical pixels."""
        width, height = self.logical_size()
        self.camera.set_view_size(width, height)
        ndc = np.array([2 * x / width - 1, 1 - 2 * y / height, 0.5, 1.0])
        point = np.linalg.inv(self.camera.camera_matrix) @ ndc
        world = point[:3] / point[3]
        return (float(world[0]), float(world[1]), float(world[2]))

    def to_screen(self, point: Vec3 | np.ndarray) -> tuple[float, float]:
        """Returns the canvas point, in logical pixels, where a world point shows."""
        width, height = self.logical_size()
        self.camera.set_view_size(width, height)
        world = np.array([*np.asarray(point, dtype=float)[:3], 1.0])
        ndc = self.camera.camera_matrix @ world
        ndc = ndc[:3] / ndc[3]
        return (float((ndc[0] + 1) * width / 2), float((1 - ndc[1]) * height / 2))

    def pixel_size(self) -> float:
        """Returns how many mm one logical pixel spans in a 2D view's plane."""
        a = np.array(self.to_world(0.0, 0.0))
        b = np.array(self.to_world(1.0, 0.0))
        return float(np.linalg.norm(b - a))

    def ray(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        """Returns the ray through a canvas point: its start on the near plane, unit.

        Returns:
            The start and the direction, world mm.
        """
        width, height = self.logical_size()
        self.camera.set_view_size(width, height)
        inverse = np.linalg.inv(self.camera.camera_matrix)
        ends = []
        for depth in (0.0, 1.0):
            point = inverse @ np.array(
                [2 * x / width - 1, 1 - 2 * y / height, depth, 1]
            )
            ends.append(point[:3] / point[3])
        direction = ends[1] - ends[0]
        return ends[0], direction / np.linalg.norm(direction)

    def pick(self, x: float, y: float) -> Vec3 | None:
        """Returns the nearest world point drawn under a canvas point, if any.

        The GPU's picking finds the objects that write picks (slices, surfaces,
        tubes); the view's pickers find the others. It needs the view drawn
        once.

        Args:
            x: The canvas point's x, logical pixels.
            y: Its y, logical pixels.

        Returns:
            The point, world mm; None if nothing is under it.
        """
        found: list[np.ndarray] = []
        try:
            info = self.renderer.get_pick_info((x, y))
        except Exception:  # noqa: BLE001 (no picking before a first draw)
            info = {}
        point = picked_point(info)
        if point is not None:
            found.append(point)
        for picker in self.pickers:
            other = picker(self, x, y)
            if other is not None:
                found.append(other)
        if not found:
            return None
        origin, direction = self.ray(x, y)
        nearest = min(found, key=lambda p: float((p - origin) @ direction))
        return (float(nearest[0]), float(nearest[1]), float(nearest[2]))

    def draw(self) -> None:
        """Renders a view, its crosshair, its overlay unless hidden, then controls."""
        # Rendered at the canvas's own resolution while interacting, where pygfx
        # would otherwise render at twice it and smooth the result down.
        self.renderer.pixel_scale = 1.0 if self.scale != 1.0 else None
        self.renderer.render(self.scene, self.camera, flush=False)
        self.renderer.render(self.guides, self.camera, flush=False)
        if self.overlay_visible:
            self.renderer.render(self.overlay, self.overlay_camera, flush=False)
        if self.controls.children:
            self.renderer.render(self.controls, self.overlay_camera, flush=False)
        self.renderer.flush()

    def _set_scale(self, scale: float) -> None:
        """Renders at scale times the canvas's resolution, from the next draw on."""
        # The canvas's texture shrinks, and Qt stretches the image over the widget.
        info = self.canvas._size_info
        ratio = info["native_pixel_ratio"]
        if not self._scaled():  # the canvas's own: at scale 1, or since a resize
            self._full_ratio = ratio
            self._full_size = self.canvas.get_logical_size()
        width, height = self._full_size
        new_ratio = self._full_ratio * scale
        info.set_physical_size(
            max(1, round(width * new_ratio)),
            max(1, round(height * new_ratio)),
            new_ratio,
        )
        self.scale = scale
        self.canvas.request_draw()

    def _scaled(self) -> bool:
        """Returns whether the canvas has the reduced size set here, not its own."""
        ratio = self.canvas._size_info["native_pixel_ratio"]
        return self.scale != 1.0 and math.isclose(ratio, self._full_ratio * self.scale)

    def logical_size(self) -> tuple[float, float]:
        """Returns the canvas's size in logical pixels, as at full resolution."""
        if self._scaled():
            return self._full_size
        width, height = self.canvas.get_logical_size()
        return (width, height)

    def _place_camera(self) -> bool:
        """Points the camera as the view state says; False if it does not move."""
        forward, up = view_directions(self.kind, self.convention)
        right, _ = screen_axes(self.kind, self.convention)
        lo, hi = (np.asarray(p, dtype=float) for p in self.field)
        size = hi - lo
        pan = self.camera_2d.pan
        centre = (lo + hi) / 2 + pan[0] * np.asarray(right) + pan[1] * np.asarray(up)
        width = float(abs(size @ right)) or 1.0
        height = float(abs(size @ up)) or 1.0
        # The orthographic depth range reaches a hundred times this either side.
        depth = float(max(size.max(), 1.0))
        # A change, say of convention, can leave a view as it was; it is not redrawn.
        placed = (width, height, self.camera_2d.zoom, depth, up, tuple(centre), forward)
        if placed == self._placed:
            return False
        self._placed = placed
        camera = self.camera
        camera.width = width
        camera.height = height
        camera.zoom = self.camera_2d.zoom
        camera.depth = depth
        camera.world.reference_up = up
        camera.world.position = centre
        camera.world.forward = forward
        return True


def picked_point(info: dict[str, Any]) -> np.ndarray | None:
    """Returns the world point of the GPU's pick info, for a mesh, line or volume.

    Returns:
        (3,) the point, world mm; None if nothing was picked or the object is of
        another kind.
    """
    obj = info.get("world_object")
    if not isinstance(obj, gfx.WorldObject):
        return None
    geometry = obj.geometry
    local: np.ndarray | None = None
    if "face_index" in info and isinstance(obj, gfx.Mesh):
        positions = np.asarray(geometry.positions.data, dtype=float)
        faces = np.asarray(geometry.indices.data).reshape(-1, 3)
        face = faces[int(info["face_index"])]
        weights = np.asarray(info["face_coord"], dtype=float)[:3]
        local = weights @ positions[face]
    elif "vertex_index" in info and isinstance(obj, gfx.Line):
        positions = np.asarray(geometry.positions.data, dtype=float)
        index = int(info["vertex_index"])
        along = float(info["segment_coord"])
        other = index + 1 if along >= 0 else index - 1
        other = min(max(other, 0), len(positions) - 1)
        local = positions[index] + abs(along) * (positions[other] - positions[index])
    elif "voxel_coord" in info and isinstance(obj, gfx.Volume):
        local = np.add(info["index"], info["voxel_coord"], dtype=float)
    if local is None or not np.isfinite(local).all():
        return None
    world = obj.world.matrix @ np.array([*local, 1.0])
    point: np.ndarray = world[:3] / world[3]
    return point
