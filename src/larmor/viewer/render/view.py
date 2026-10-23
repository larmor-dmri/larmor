"""Holds the pygfx renderer, scene and camera of one view."""

import math

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
