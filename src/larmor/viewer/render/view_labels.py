"""Draws each view's name and slice, such as "Axial  z 80 / 181", in its corner."""

from functools import partial

import pygfx as gfx

from larmor.types import Grid
from larmor.viewer.model.events import (
    DataChanged,
    Event,
    ItemAdded,
    ItemRemoved,
    ViewStateChanged,
)
from larmor.viewer.model.item import VIEW_NAMES, ViewKind
from larmor.viewer.model.readout import slice_index
from larmor.viewer.model.view_state import ViewField
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.render.annotations import EDGE, FONT_SIZE
from larmor.viewer.render.backing import backings, place_backings
from larmor.viewer.render.view import Vec3, ViewRenderer
from larmor.viewer.render.volume_3d import PLANES

LABEL_COLOUR = "#c4c4c4"  # muted beside the readout's white
AXIS_LETTERS = ("x", "y", "z")
# View state fields, besides the cursor, that can change the reference grid, so
# the labels are written again with the grid found again.
GRID_CHANGES = {ViewField.MAIN_VOLUME}


class ViewLabels:
    """Draws each view's name and, in a 2D view, its slice of the reference grid."""

    def __init__(self, viewer: Viewer, views: dict[ViewKind, ViewRenderer]) -> None:
        """Adds a label to each view's overlay."""
        self.viewer = viewer
        self.views = dict(views)
        self.labels: dict[ViewKind, gfx.Text] = {}
        self.backings: dict[ViewKind, gfx.Group] = {}
        self._placers: dict[ViewKind, partial[None]] = {}
        self._texts: dict[ViewKind, str] = {}  # each label's Markdown, as drawn
        self.grid = viewer._reference_grid()  # found again only when it can change
        for kind, view in self.views.items():
            label = gfx.Text(
                text="",
                font_size=FONT_SIZE,
                screen_space=True,
                anchor="top-left",
                material=gfx.TextMaterial(
                    color=LABEL_COLOUR, outline_color="#000", outline_thickness=0.25
                ),
            )
            label.local.position = (-EDGE, EDGE, 0.0)
            self.labels[kind] = label
            view.overlay.add(label)
            self.backings[kind] = backings([label])
            view.overlay.add(self.backings[kind])
            self._placers[kind] = partial(place_backings, self.backings[kind])
            view.renderer.add_event_handler(self._placers[kind], "before_render")
        self._write()

    def on_event(self, event: Event) -> None:
        """Follows the cursor and the reference grid."""
        if isinstance(event, (ItemAdded, ItemRemoved, DataChanged)) or (
            isinstance(event, ViewStateChanged) and event.name in GRID_CHANGES
        ):
            self.grid = self.viewer._reference_grid()
            self._write()
        elif isinstance(event, ViewStateChanged) and event.name == ViewField.CURSOR:
            self._write()

    def text(self, kind: ViewKind) -> str:
        """Returns the label a view shows, without its markup."""
        return self._texts.get(kind, "").replace("**", "")

    def close(self) -> None:
        """Removes the labels from the views."""
        for kind, view in self.views.items():
            view.overlay.remove(self.labels[kind])
            view.overlay.remove(self.backings[kind])
            view.renderer.remove_event_handler(self._placers[kind], "before_render")
            view.request_draw()
        self.views = {}
        self.labels.clear()
        self._texts.clear()
        self.backings.clear()
        self._placers.clear()

    def _write(self) -> None:
        """Writes each view's label, if it changed."""
        for kind, label in self.labels.items():
            text = label_text(kind, self.grid, self.viewer.scene.view.cursor)
            if text != self._texts.get(kind):
                label.set_markdown(text)
                self._texts[kind] = text
                self.views[kind].request_draw()


def label_text(kind: ViewKind, grid: Grid | None, cursor: Vec3) -> str:
    """Returns a view's label in Markdown, its name in bold, then a 2D view's slice."""
    name = f"**{VIEW_NAMES[kind]}**"
    if kind not in PLANES or grid is None:
        return name
    axis = PLANES[kind]
    index, count = slice_index(grid, cursor, axis)
    shown = str(index) if 0 <= index < count else "-"
    return f"{name}  {AXIS_LETTERS[axis]} {shown} / {count}"
