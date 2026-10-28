"""Draws the crosshair, the orientation labels and the readout in the 2D views."""

from functools import partial

import numpy as np
import pygfx as gfx

from larmor.viewer.model.camera import Convention, screen_axes
from larmor.viewer.model.events import (
    DataChanged,
    Event,
    ItemAdded,
    ItemRemoved,
    OptionChanged,
    ViewStateChanged,
)
from larmor.viewer.model.item import VIEW_NAMES, ViewKind
from larmor.viewer.model.options import OptionName
from larmor.viewer.model.readout import Readout
from larmor.viewer.model.view_state import ViewField
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.render.backing import backings, place_backings
from larmor.viewer.render.view import CAMERA_DIRECTIONS_UP, Vec3, ViewRenderer

CROSSHAIR_COLOUR = "#3fbf3f"
CROSSHAIR_REACH = 1e4  # mm either side of the cursor, past any view's edges
FONT_SIZE = 13
EDGE = 0.98  # how far towards the edges, in screen space (-1 to 1), text sits
# The options of the main volume the readout shows.
READ = {OptionName.VOLUME_INDEX, OptionName.INTERPOLATE}
# About the mean width of one character of the text, logical pixels.
CHAR_WIDTH = 0.52 * FONT_SIZE
# Logical pixels kept free beside the readout, and between a text and a letter.
MARGIN = 12.0
# The most characters of a view's name and slice in its top left corner, as in
# "Sagittal  x 100 / 256", besides the name.
CORNER_CHARS = 13
# The letter for each world direction, as (axis, sign).
LETTERS = {
    (0, 1): "R",
    (0, -1): "L",
    (1, 1): "A",
    (1, -1): "P",
    (2, 1): "S",
    (2, -1): "I",
}
# Each label's place and anchor, by its screen direction (right, up).
PLACES = {
    (1, 0): ((EDGE, 0.0, 0.0), "middle-right"),
    (-1, 0): ((-EDGE, 0.0, 0.0), "middle-left"),
    (0, 1): ((0.0, EDGE, 0.0), "top-center"),
    (0, -1): ((0.0, -EDGE, 0.0), "bottom-center"),
}


class Annotations:
    """Draws the crosshair, the orientation labels and the readout in each 2D view."""

    def __init__(self, viewer: Viewer, views: dict[ViewKind, ViewRenderer]) -> None:
        """Adds the crosshair to each 2D view's scene and the text to its overlay."""
        self.viewer = viewer
        self.views = {k: v for k, v in views.items() if k in CAMERA_DIRECTIONS_UP}
        self.crosshairs: dict[ViewKind, gfx.Line] = {}
        self.labels: dict[ViewKind, dict[tuple[int, int], gfx.Text]] = {}
        self.readouts: dict[ViewKind, gfx.Text] = {}
        # A faint box behind each text, sized to it before each draw.
        self.backings: dict[ViewKind, gfx.Group] = {}
        self._placers: dict[ViewKind, partial[None]] = {}
        self._fitters: dict[ViewKind, partial[None]] = {}
        self.readout = ""  # the text the readouts show, before fitting a view
        self._widths: dict[ViewKind, float] = {}  # each view's, at its last draw
        self._shown: dict[ViewKind, str] = {}  # each view's readout, as fitted
        for kind, view in self.views.items():
            self.crosshairs[kind] = _crosshair()
            view.guides.add(self.crosshairs[kind])
            self.labels[kind] = {
                direction: _text("", position, anchor)
                for direction, (position, anchor) in PLACES.items()
            }
            for label in self.labels[kind].values():
                view.overlay.add(label)
            self.readouts[kind] = _text("", (-EDGE, -EDGE, 0.0), "bottom-left")
            view.overlay.add(self.readouts[kind])
            texts = [*self.labels[kind].values(), self.readouts[kind]]
            self.backings[kind] = backings(texts)
            view.overlay.add(self.backings[kind])
            # First, so that the backings follow the texts as fitted.
            self._fitters[kind] = partial(self._fit, kind)
            view.renderer.add_event_handler(self._fitters[kind], "before_render")
            self._placers[kind] = partial(place_backings, self.backings[kind])
            view.renderer.add_event_handler(self._placers[kind], "before_render")
        self._move_crosshairs()
        self._write_labels()
        self._write_readout()

    def on_event(self, event: Event) -> None:
        """Follows the cursor, the convention and what the readout shows."""
        if isinstance(event, ViewStateChanged):
            if event.name == ViewField.CURSOR:
                self._move_crosshairs()
                self._write_readout()
            elif event.name == ViewField.MAIN_VOLUME:
                self._write_readout()
            elif event.name == ViewField.CONVENTION:
                self._write_labels()
            elif event.name == ViewField.CROSSHAIR_VISIBLE:
                self._move_crosshairs()
        elif isinstance(event, (ItemAdded, ItemRemoved, DataChanged)):
            self._write_readout()
        elif (
            isinstance(event, OptionChanged)
            and event.name in READ
            and event.item.id == self.viewer.scene.view.main_volume
        ):
            self._write_readout()

    def close(self) -> None:
        """Removes the crosshair and the text from the views."""
        for kind, view in self.views.items():
            view.guides.remove(self.crosshairs[kind])
            for label in self.labels[kind].values():
                view.overlay.remove(label)
            view.overlay.remove(self.readouts[kind])
            view.overlay.remove(self.backings[kind])
            view.renderer.remove_event_handler(self._placers[kind], "before_render")
            view.renderer.remove_event_handler(self._fitters[kind], "before_render")
            view.request_draw()
        # An event already on its way finds nothing left to update.
        self.views = {}
        self.crosshairs.clear()
        self.labels.clear()
        self.readouts.clear()
        self.backings.clear()
        self._placers.clear()
        self._fitters.clear()

    def _move_crosshairs(self) -> None:
        """Puts each view's crosshair through the cursor, or hides it."""
        cursor = np.asarray(self.viewer.scene.view.cursor, dtype=float)
        convention = self.viewer.scene.view.convention
        visible = self.viewer.scene.view.crosshair_visible
        for kind, line in self.crosshairs.items():
            line.visible = visible
            right, up = (np.asarray(a) for a in screen_axes(kind, convention))
            ends = [
                cursor - CROSSHAIR_REACH * right,
                cursor + CROSSHAIR_REACH * right,
                cursor - CROSSHAIR_REACH * up,
                cursor + CROSSHAIR_REACH * up,
            ]
            line.geometry = gfx.Geometry(positions=np.array(ends, np.float32))
            self.views[kind].request_draw()

    def _write_labels(self) -> None:
        """Writes the orientation letters for the convention."""
        convention = self.viewer.scene.view.convention
        for kind, labels in self.labels.items():
            for direction, label in labels.items():
                label.set_text(_letter(kind, convention, direction))
            self.views[kind].request_draw()

    def _write_readout(self) -> None:
        """Writes the readout at the cursor, if it changed."""
        text = readout_text(self.viewer.readout())
        if text == self.readout:
            return
        self.readout = text
        for kind in self.readouts:
            self._fit_texts(kind)
            self.views[kind].request_draw()

    def _fit(self, kind: ViewKind, event: gfx.WindowEvent) -> None:
        """Fits a view's text to its width before it draws, again if it changed."""
        self._widths[kind] = float(event.width)
        if self._fit_texts(kind):
            self.views[kind].request_draw()  # the backing follows the new text

    def _fit_texts(self, kind: ViewKind) -> bool:
        """Cuts the readout's name short to fit the view, and hides letters in the way.

        A letter at the top or bottom edge shows only while the text in the corner
        beside it leaves it room. Before the view's first draw, its width is
        unknown and the readout is shown whole.

        Returns:
            Whether the readout's text changed.
        """
        width = self._widths.get(kind)
        text = self.readout if width is None else fitted_readout(self.readout, width)
        changed = text != self._shown.get(kind)
        if changed:
            self.readouts[kind].set_text(text)
            self._shown[kind] = text
        if width is not None:
            room = width / 2 - MARGIN
            widest = max(len(line) for line in text.split("\n"))
            corner = len(VIEW_NAMES[kind]) + CORNER_CHARS
            self.labels[kind][(0, -1)].visible = widest * CHAR_WIDTH <= room
            self.labels[kind][(0, 1)].visible = corner * CHAR_WIDTH <= room
        return changed


def readout_text(readout: Readout) -> str:
    """Returns the readout as the views show it: name, value, voxel and position."""
    lines = []
    if readout.name is not None:
        lines.append(readout.name)
        if readout.voxel is None or readout.value is None:
            lines.append("outside the image")
        else:
            lines.append(f"value {readout.value:.6g}")
            lines.append("voxel " + ", ".join(str(v) for v in readout.voxel))
    lines.append(", ".join(f"{v:.1f}" for v in readout.position) + " mm")
    return "\n".join(lines)


def fitted_readout(text: str, width: float) -> str:
    """Returns a readout with its first line, the image's name, cut to a view's width.

    A name cut short ends in "...".
    """
    name, newline, rest = text.partition("\n")
    if not newline:
        return text  # the position only
    room = max(int((width - 2 * MARGIN) / CHAR_WIDTH), 4)
    if len(name) > room:
        name = name[: room - 3] + "..."
    return f"{name}\n{rest}"


def _letter(kind: ViewKind, convention: Convention, direction: tuple[int, int]) -> str:
    """Returns the orientation letter at one edge of a 2D view."""
    right, up = (np.asarray(a) for a in screen_axes(kind, convention))
    world = direction[0] * right + direction[1] * up
    axis = int(np.argmax(np.abs(world)))
    return LETTERS[(axis, 1 if world[axis] > 0 else -1)]


def _crosshair() -> gfx.Line:
    """Returns a crosshair of two segments, placed later."""
    material = gfx.LineSegmentMaterial(
        thickness=1.0, color=CROSSHAIR_COLOUR, depth_test=False
    )
    line = gfx.Line(gfx.Geometry(positions=np.zeros((4, 3), np.float32)), material)
    return line


def _text(text: str, position: Vec3, anchor: str) -> gfx.Text:
    """Returns an outlined screen-space text at a position in the overlay."""
    label = gfx.Text(
        text=text,
        font_size=FONT_SIZE,
        screen_space=True,
        anchor=anchor,
        material=gfx.TextMaterial(
            color="#fff", outline_color="#000", outline_thickness=0.25
        ),
    )
    label.local.position = position
    return label
