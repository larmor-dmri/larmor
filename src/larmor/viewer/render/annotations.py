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
from larmor.viewer.model.item import ViewKind
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
        self.readout = ""  # the text the readouts show
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
            view.request_draw()
        # An event already on its way finds nothing left to update.
        self.views = {}
        self.crosshairs.clear()
        self.labels.clear()
        self.readouts.clear()
        self.backings.clear()
        self._placers.clear()

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
        for kind, label in self.readouts.items():
            label.set_text(text)
            self.views[kind].request_draw()


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
