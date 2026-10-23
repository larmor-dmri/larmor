"""Draws the FOD glyph of the voxel at the cursor large, in a canvas of its own."""

import math
import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
import pygfx as gfx
from PIL import Image
from rendercanvas import BaseRenderCanvas
from rendercanvas.offscreen import RenderCanvas

from larmor.types import Volume
from larmor.viewer.model.camera import Camera3D
from larmor.viewer.model.events import (
    DataChanged,
    Event,
    ItemRemoved,
    OptionChanged,
    ViewStateChanged,
)
from larmor.viewer.model.item import ItemId
from larmor.viewer.model.options import Colouring, FodOptions, OptionName
from larmor.viewer.model.view_state import ViewField
from larmor.viewer.model.viewer import ZOOM_RANGE, Viewer
from larmor.viewer.render.annotations import EDGE, FONT_SIZE
from larmor.viewer.render.fod_adapter import empty_geometry, glyph_material
from larmor.viewer.render.geometry import nearest_voxel
from larmor.viewer.render.glyphs import Glyphs, voxel_glyph
from larmor.viewer.render.view import BACKGROUND
from larmor.viewer.render.view_3d import add_glyph_lights

# Subdivision of the glyph's sphere, which gives 4002 vertices (252 on the slices).
DETAIL = 20
# Lengths of the axes and of their letters' distance; the largest lobe reaches 1.
AXIS_LENGTH = 1.25
LETTER_DISTANCE = 1.4
FRAME = 1.7  # the radius the view shows at zoom 1
FIELD_OF_VIEW = 40.0  # degrees
DEFAULT_SIZE = (500, 500)
# The letter at each end of the three world axes, and the axis's colour.
AXES = (
    ("R", "L", (1.0, 0.3, 0.3)),
    ("A", "P", (0.3, 1.0, 0.3)),
    ("S", "I", (0.4, 0.6, 1.0)),
)
# Options that make the glyph again; colouring and colour change only the material.
REBUILDING = (OptionName.BASIS,)
NO_FOD = "No FOD here"


class FodInspector:
    """Draws one FOD item's glyph at the cursor, centred, large and in high detail.

    It follows the cursor and the item's basis and colouring. The other glyph
    options of the slices, such as threshold and scale, do not apply. The camera
    is the inspector's own, not the model's. Paused, it only notes that the scene
    changed, and catches up when it resumes.
    """

    def __init__(
        self,
        viewer: Viewer,
        item_id: ItemId,
        canvas: BaseRenderCanvas | None = None,
        removed: Callable[[], object] | None = None,
    ) -> None:
        """Builds the scene and draws the item's glyph at the cursor.

        Args:
            viewer: The viewer whose scene holds the item.
            item_id: The FOD item to draw.
            canvas: The canvas to draw on. Defaults to None: draws off screen.
            removed: Called once the item is removed from the scene, after the inspector
                has closed. Defaults to None.

        Raises:
            ValueError: If the item is not an FOD.
        """
        self.viewer = viewer
        self.item = viewer.scene.item(item_id)
        if not isinstance(self.item.options, FodOptions):
            raise ValueError(f"Item {item_id} is not an FOD.")
        self._removed = removed
        self._own_canvas = canvas is None
        self.canvas = canvas if canvas is not None else RenderCanvas(size=DEFAULT_SIZE)
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self.background = gfx.Background.from_color(BACKGROUND)
        self.scene.add(self.background)
        self.camera_3d = gfx.PerspectiveCamera(FIELD_OF_VIEW)
        add_glyph_lights(self.scene, self.camera_3d)
        self.camera = Camera3D()

        material = glyph_material()
        self.mesh = gfx.Mesh(empty_geometry(), material)
        self.mesh.visible = False
        self.scene.add(self.mesh)
        self.axes = _axes()
        self.scene.add(self.axes)

        # The voxel and position, in screen space over the scene.
        self.overlay = gfx.Scene()
        self.overlay_camera = gfx.NDCCamera()
        self.label = gfx.Text(
            text="",
            font_size=FONT_SIZE,
            screen_space=True,
            anchor="bottom-left",
            material=gfx.TextMaterial(
                color="#fff", outline_color="#000", outline_thickness=0.25
            ),
        )
        self.label.local.position = (-EDGE, -EDGE, 0.0)
        self.overlay.add(self.label)
        self.label_text = ""

        self.voxel: tuple[int, int, int] | None = None
        self.build_time = 0.0  # seconds the last glyph took
        self._built: tuple[tuple[int, int, int] | None, str] | None = None
        self._closed = False
        self._paused = False
        self._stale = False  # the scene changed while paused
        self._place_camera()
        self._show()
        self._rebuild()
        self._unsubscribe = viewer.scene.subscribe(self.on_event)
        self.canvas.request_draw(self.draw)

    # Events

    def on_event(self, event: Event) -> None:
        """Follows the cursor, the item's data, basis and colouring, and its removal.

        Args:
            event: The scene event.
        """
        if self._closed:
            return
        if isinstance(event, ItemRemoved):
            if event.item.id == self.item.id:
                self.close()
                if self._removed is not None:
                    self._removed()
            return
        if isinstance(event, DataChanged) and event.item.id == self.item.id:
            self._built = None  # new coefficients at the same voxel
        if self._paused:
            self._stale = True
        else:
            self._follow(event)

    def _follow(self, event: Event) -> None:
        """Makes or colours the glyph again for an event that concerns it."""
        if isinstance(event, DataChanged):
            if event.item.id == self.item.id:
                self._rebuild()
        elif isinstance(event, OptionChanged):
            if event.item.id != self.item.id:
                return
            if event.name in REBUILDING:
                self._rebuild()
            elif event.name in (OptionName.COLOURING, OptionName.COLOUR):
                self._show()
        elif isinstance(event, ViewStateChanged) and event.name == ViewField.CURSOR:
            self._rebuild()

    def pause(self) -> None:
        """Stops making the glyph and drawing until it resumes."""
        self._paused = True

    def resume(self) -> None:
        """Colours the glyph and makes it again if the scene changed meanwhile."""
        self._paused = False
        if self._stale:
            self._stale = False
            self._show()
            self._rebuild()  # the same voxel and basis keep the glyph

    def close(self) -> None:
        """Stops following the scene; frees the canvas if it made it."""
        if self._closed:
            return
        self._closed = True
        self._unsubscribe()
        if self._own_canvas:
            self.canvas.close()

    # Camera

    def rotate(self, right: float, up: float) -> None:
        """Turns the glyph by degrees on the screen, as the 3D view turns its scene.

        Args:
            right: Degrees to turn the glyph to the right.
            up: Degrees to turn it upwards.
        """
        self.camera = self.camera.rotated(right, up)
        self._place_camera()

    def zoom(self, factor: float) -> None:
        """Zooms by a factor, within the views' zoom range.

        Args:
            factor: The factor to multiply the zoom by.
        """
        zoom = min(max(self.camera.zoom * factor, ZOOM_RANGE[0]), ZOOM_RANGE[1])
        self.camera = Camera3D(self.camera.forward, self.camera.up, zoom)
        self._place_camera()

    def reset_camera(self) -> None:
        """Looks at the glyph as at first: as the 3D view's reset, at zoom 1."""
        self.camera = Camera3D()
        self._place_camera()

    def _place_camera(self) -> None:
        """Points the camera at the origin, framing the axes' letters at zoom 1."""
        cam = self.camera_3d
        distance = FRAME / math.sin(math.radians(FIELD_OF_VIEW) / 2)
        cam.zoom = self.camera.zoom
        cam.depth_range = (0.1 * distance, distance + 2 * FRAME)
        cam.world.reference_up = self.camera.up
        cam.world.position = -distance * np.asarray(self.camera.forward)
        cam.world.forward = self.camera.forward
        self.canvas.request_draw()

    # Drawing

    def draw(self) -> None:
        """Renders the glyph and the axes, then the text."""
        self.renderer.render(self.scene, self.camera_3d, flush=False)
        self.renderer.render(self.overlay, self.overlay_camera, flush=False)
        self.renderer.flush()

    def render(self) -> np.ndarray:
        """Draws off screen and returns the image.

        Returns:
            The image, (height, width, 3) uint8 RGB.
        """
        image: np.ndarray = np.asarray(self.canvas.draw())[..., :3].copy()
        return image

    def save(self, path: str | Path) -> Path:
        """Draws off screen and writes the image to a file.

        Args:
            path: The image file to write.

        Returns:
            The path written.
        """
        p = Path(path)
        Image.fromarray(self.render()).save(p)
        return p

    def _options(self) -> FodOptions:
        """Returns the item's FOD options."""
        o = self.item.options
        assert isinstance(o, FodOptions)
        return o

    def _show(self) -> None:
        """Colours the glyph as the item's options say."""
        o = self._options()
        material = self.mesh.material
        if o.colouring == Colouring.SINGLE:
            material.color_mode = "uniform"
            material.color = o.colour
        else:
            material.color_mode = "vertex"
        self.canvas.request_draw()

    def _rebuild(self) -> None:
        """Makes the glyph again if the voxel or the basis changed; writes the label."""
        volume = self.item.data
        assert isinstance(volume, Volume)
        cursor = self.viewer.scene.view.cursor
        self.voxel = nearest_voxel(volume.affine, volume.grid.shape, cursor)
        key = (self.voxel, self._options().basis)
        if key != self._built:
            self._built = key
            glyph = None
            if self.voxel is not None:
                start = time.perf_counter()
                glyph = voxel_glyph(volume, self.voxel, key[1], DETAIL)
                self.build_time = time.perf_counter() - start
            self._take(glyph)
        self._write_label(glyph_drawn=self.mesh.visible)
        self.canvas.request_draw()

    def _take(self, glyph: Glyphs | None) -> None:
        """Puts a glyph in the mesh, or hides the mesh if there is none."""
        if glyph is None or glyph.count == 0:
            self.mesh.visible = False
            return
        self.mesh.geometry = gfx.Geometry(
            positions=glyph.positions,
            normals=glyph.normals,
            colors=glyph.colours,
            indices=glyph.faces,
        )
        self.mesh.visible = True

    def _write_label(self, glyph_drawn: bool) -> None:
        """Writes the item's name, the voxel and its position in mm in the corner."""
        volume = self.item.data
        assert isinstance(volume, Volume)
        lines = [self.item.name]
        if self.voxel is None:
            position = self.viewer.scene.view.cursor
            lines.append(NO_FOD)
        else:
            centre = volume.affine[:3, :3] @ self.voxel + volume.affine[:3, 3]
            position = (float(centre[0]), float(centre[1]), float(centre[2]))
            lines.append("voxel " + ", ".join(str(v) for v in self.voxel))
            if not glyph_drawn:
                lines.append(NO_FOD)
        lines.append(", ".join(f"{v:.1f}" for v in position) + " mm")
        text = "\n".join(lines)
        if text != self.label_text:
            self.label_text = text
            self.label.set_text(text)

    def __enter__(self) -> Self:
        """Returns the inspector, which closes at the end of the block."""
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Closes the inspector."""
        self.close()


def _axes() -> gfx.Group:
    """Returns the world axes through the glyph's centre, each end with its letter."""
    group = gfx.Group()
    positions, colours = [], []
    for axis, (plus, minus, colour) in enumerate(AXES):
        direction = np.eye(3)[axis]
        dim = tuple(0.45 * c for c in colour)
        positions += [np.zeros(3), AXIS_LENGTH * direction]
        positions += [np.zeros(3), -AXIS_LENGTH * direction]
        colours += [colour, colour, dim, dim]
        for letter, sign, tint in ((plus, 1, colour), (minus, -1, dim)):
            text = gfx.Text(
                text=letter,
                font_size=FONT_SIZE + 2,
                screen_space=True,
                anchor="middle-center",
                material=gfx.TextMaterial(color=(*tint, 1.0)),
            )
            text.local.position = tuple(sign * LETTER_DISTANCE * direction)
            group.add(text)
    line = gfx.Line(
        gfx.Geometry(
            positions=np.array(positions, np.float32),
            colors=np.array(colours, np.float32),
        ),
        gfx.LineSegmentMaterial(thickness=2.0, color_mode="vertex"),
    )
    group.add(line)
    return group
