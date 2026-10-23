"""Draws the enlarge and restore button in a window view's top right corner."""

import numpy as np
import pygfx as gfx

from larmor.viewer.render.backing import BACKING_COLOUR
from larmor.viewer.render.view import ViewRenderer

GLYPH_COLOUR = "#c4c4c4"  # as the view's label
SIZE = 14.0  # the glyph's side, in logical pixels
MARGIN = 8.0  # from the view's top and right edges to the glyph
PADDING = 4.0  # how far the box behind the glyph, and a click on it, reach past it
GRID = 24.0  # the side of the grid the glyphs are drawn on, as Lucide's icons
Segments = tuple[tuple[tuple[float, float], tuple[float, float]], ...]
# Lucide's maximize-2 and minimize-2, as line segments on their grid (y down).
ENLARGE: Segments = (
    ((15, 3), (21, 3)),
    ((21, 3), (21, 9)),
    ((9, 21), (3, 21)),
    ((3, 21), (3, 15)),
    ((21, 3), (14, 10)),
    ((3, 21), (10, 14)),
)
RESTORE: Segments = (
    ((4, 14), (10, 14)),
    ((10, 14), (10, 20)),
    ((20, 10), (14, 10)),
    ((14, 10), (14, 4)),
    ((14, 10), (21, 3)),
    ((3, 21), (10, 14)),
)
# Render queue of the box, after everything else; the glyph comes next, over it.
QUEUE = 4000


class ViewButton:
    """Draws the button in a view's controls; the window hit-tests it."""

    def __init__(self, view: ViewRenderer) -> None:
        """Adds the glyph and its box to the view's controls."""
        self.view = view
        self.maximised = False
        self.box = gfx.Mesh(
            gfx.Geometry(
                positions=np.zeros((4, 3), np.float32),
                indices=np.array([[0, 1, 2], [2, 1, 3]], np.int32),
            ),
            gfx.MeshBasicMaterial(
                color=BACKING_COLOUR,
                alpha_mode="blend",
                depth_test=False,
                render_queue=QUEUE,
            ),
        )
        self.glyph = gfx.Line(
            gfx.Geometry(positions=np.zeros((2 * len(ENLARGE), 3), np.float32)),
            gfx.LineSegmentMaterial(
                thickness=1.5,
                color=GLYPH_COLOUR,
                alpha_mode="blend",
                depth_test=False,
                render_queue=QUEUE + 1,
            ),
        )
        view.controls.add(self.box, self.glyph)
        view.renderer.add_event_handler(self._place, "before_render")

    def set_maximised(self, maximised: bool) -> None:
        """Draws the restore glyph while the view fills the window, else enlarge."""
        self.maximised = maximised
        self.view.request_draw()

    def rect(self) -> tuple[float, float, float, float]:
        """Returns the area a click hits in logical pixels: left, top, right, bottom."""
        width, _ = self.view.logical_size()
        right = width - MARGIN + PADDING
        return (
            right - SIZE - 2 * PADDING,
            MARGIN - PADDING,
            right,
            MARGIN + SIZE + PADDING,
        )

    def hit(self, x: float, y: float) -> bool:
        """Returns whether a canvas point, in logical pixels, is on the button."""
        left, top, right, bottom = self.rect()
        return left <= x <= right and top <= y <= bottom

    def close(self) -> None:
        """Removes the button from the view."""
        self.view.controls.remove(self.box, self.glyph)
        self.view.renderer.remove_event_handler(self._place, "before_render")
        self.view.request_draw()

    def _place(self, event: gfx.WindowEvent) -> None:
        """Lays the box and glyph out in the view's screen space before each draw."""
        width, height = float(event.width), float(event.height)

        def ndc(x: float, y: float) -> list[float]:
            """Returns a point in logical pixels in screen space (-1 to 1)."""
            return [2 * x / width - 1, 1 - 2 * y / height, 0.0]

        left, top, right, bottom = self.rect()
        _set(
            self.box,
            [ndc(left, bottom), ndc(right, bottom), ndc(left, top), ndc(right, top)],
        )
        x0, y0 = right - PADDING - SIZE, top + PADDING
        scale = SIZE / GRID
        segments = RESTORE if self.maximised else ENLARGE
        points = [
            ndc(x0 + u * scale, y0 + v * scale)
            for segment in segments
            for u, v in segment
        ]
        _set(self.glyph, points)


def _set(obj: gfx.WorldObject, points: list[list[float]]) -> None:
    """Sets an object's positions, uploading them only if they changed."""
    positions = np.array(points, np.float32)
    buffer = obj.geometry.positions
    if np.array_equal(buffer.data, positions):
        return
    buffer.data[:] = positions
    buffer.update_full()
