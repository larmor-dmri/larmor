"""Draws a faint dark box behind the text over a view, so it reads on bright images."""

import numpy as np
import pygfx as gfx

BACKING_COLOUR = (0.0, 0.0, 0.0, 0.55)
PADDING = (5.0, 2.0)  # logical pixels either side of the text, across and up


class Backing(gfx.Mesh):  # type: ignore[misc]
    """Draws a translucent box behind a text in a view's overlay."""

    def __init__(self, text: gfx.Text) -> None:
        """Creates the box for text; `place` sizes it before each draw."""
        geometry = gfx.Geometry(
            positions=np.zeros((4, 3), np.float32),
            indices=np.array([[0, 1, 2], [2, 1, 3]], np.int32),
        )
        # Blended boxes are drawn after text by default; this queue puts them first.
        material = gfx.MeshBasicMaterial(
            color=BACKING_COLOUR,
            alpha_mode="blend",
            depth_test=False,
            render_queue=text.material.render_queue - 1,
        )
        super().__init__(geometry, material)
        self.text = text

    def place(self, width: float, height: float) -> None:
        """Fits the box around the text, in a view of this many logical pixels."""
        # The text's layout in logical pixels from its anchor; the private _aabb, as
        # pygfx's public bounds of screen-space text are only its anchor.
        (left, bottom, _), (right, top, _) = self.text._aabb
        self.visible = self.text.visible and right > left
        if not self.visible:
            return
        x, y = self.text.local.position[:2]
        pad_x, pad_y = PADDING
        x0 = x + 2 * (left - pad_x) / width
        x1 = x + 2 * (right + pad_x) / width
        y0 = y + 2 * (bottom - pad_y) / height
        y1 = y + 2 * (top + pad_y) / height
        positions = np.array([[x0, y0, 0], [x1, y0, 0], [x0, y1, 0], [x1, y1, 0]])
        buffer = self.geometry.positions
        if np.array_equal(buffer.data, positions):
            return
        buffer.data[:] = positions
        buffer.update_full()


def backings(texts: list[gfx.Text]) -> gfx.Group:
    """Returns a group with a box behind each text, to add beside the texts."""
    group = gfx.Group()
    for text in texts:
        group.add(Backing(text))
    return group


def place_backings(root: gfx.WorldObject, event: gfx.WindowEvent) -> None:
    """Fits every backing under root to the view; runs before each render."""
    for obj in root.iter(lambda o: isinstance(o, Backing)):
        obj.place(float(event.width), float(event.height))
