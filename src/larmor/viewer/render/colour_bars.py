"""Draws a colour bar in the 2D views for each visible item that asks for one."""

from functools import partial

import numpy as np
import pygfx as gfx

from larmor.viewer.model.events import (
    DataChanged,
    Event,
    ItemAdded,
    ItemRemoved,
    OptionChanged,
)
from larmor.viewer.model.item import Item, ViewKind
from larmor.viewer.model.options import (
    Colouring,
    OptionName,
    OverlayOptions,
    TractogramOptions,
    VolumeOptions,
)
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.render.backing import backings, place_backings
from larmor.viewer.render.view import ViewRenderer
from larmor.viewer.render.volume_adapter import COLORMAP_MAPS, VIEWS_2D

# The options a bar shows; a change to any other leaves the bars as they are.
SHOWN = {
    OptionName.VISIBLE,
    OptionName.COLORMAP,
    OptionName.MINIMUM,
    OptionName.MAXIMUM,
    OptionName.LABELS,
    OptionName.COLOUR_BAR,
    OptionName.DTI,
    OptionName.COLOURING,
}
# Places in screen space (-1 to 1); bars stand at the right edge, the first rightmost.
WIDTH = 0.04
GAP = 0.2  # from one bar to the next, room for its numbers
TOP, BOTTOM = 0.7, -0.8  # the top clear of the enlarge button in the corner
RIGHT = 0.9
FONT_SIZE = 12


class ColourBars:
    """Draws the colour bars of the visible items that have them on."""

    def __init__(self, viewer: Viewer, views: dict[ViewKind, ViewRenderer]) -> None:
        """Adds a group for the bars to each 2D view's overlay."""
        self.viewer = viewer
        self.views = {kind: views[kind] for kind in VIEWS_2D if kind in views}
        self.groups: dict[ViewKind, gfx.Group] = {}
        self._placers: dict[ViewKind, partial[None]] = {}
        for kind, view in self.views.items():
            self.groups[kind] = gfx.Group()
            view.overlay.add(self.groups[kind])
            # The numbers' backings, sized to them before each draw.
            self._placers[kind] = partial(place_backings, self.groups[kind])
            view.renderer.add_event_handler(self._placers[kind], "before_render")
        self._build()

    def items(self) -> list[Item]:
        """Returns the items with a bar, in scene tree order."""
        shown = []
        for item in self.viewer.scene.items:
            o = item.options
            if isinstance(o, TractogramOptions):
                if o.visible and o.colour_bar and o.colouring == Colouring.VALUES:
                    shown.append(item)
                continue
            if not isinstance(o, VolumeOptions) or not (o.visible and o.colour_bar):
                continue
            if isinstance(o, OverlayOptions) and o.labels:
                continue  # one colour per label, not a range
            if o.dti:
                continue  # colours by direction, not a range
            shown.append(item)
        return shown

    def on_event(self, event: Event) -> None:
        """Rebuilds the bars when what they show changes."""
        if isinstance(event, (ItemAdded, ItemRemoved, DataChanged)):
            self._build()
        elif isinstance(event, OptionChanged) and event.name in SHOWN:
            # Only an item with a bar, or one turning it on or off, changes the bars.
            o = event.item.options
            if event.name == OptionName.COLOUR_BAR or (
                isinstance(o, (VolumeOptions, TractogramOptions)) and o.colour_bar
            ):
                self._build()

    def close(self) -> None:
        """Removes the bars from the views."""
        for kind, group in self.groups.items():
            self.views[kind].overlay.remove(group)
            self.views[kind].renderer.remove_event_handler(
                self._placers[kind], "before_render"
            )
            self.views[kind].request_draw()
        self.groups.clear()
        self._placers.clear()

    def _build(self) -> None:
        """Draws the bars again in every view."""
        items = self.items()
        for kind, group in self.groups.items():
            group.clear()
            for k, item in enumerate(items):
                objects = _bar(item, RIGHT - k * GAP)
                group.add(*objects)
                group.add(backings([o for o in objects if isinstance(o, gfx.Text)]))
            self.views[kind].request_draw()


def _bar(item: Item, right: float) -> list[gfx.WorldObject]:
    """Returns a bar of the item's colormap, its maximum above and minimum below."""
    o = item.options
    assert isinstance(o, (VolumeOptions, TractogramOptions))
    left = right - WIDTH
    positions = np.array(
        [[left, BOTTOM, 0], [right, BOTTOM, 0], [left, TOP, 0], [right, TOP, 0]],
        np.float32,
    )
    geometry = gfx.Geometry(
        positions=positions,
        indices=np.array([[0, 1, 2], [2, 1, 3]], np.int32),
        texcoords=np.array([0, 0, 1, 1], np.float32),  # bottom to top
    )
    bar = gfx.Mesh(geometry, gfx.MeshBasicMaterial(map=COLORMAP_MAPS[o.colormap]))
    centre = right - WIDTH / 2
    return [
        bar,
        _label(f"{o.maximum:.4g}", (centre, TOP, 0), "bottom-center"),
        _label(f"{o.minimum:.4g}", (centre, BOTTOM, 0), "top-center"),
    ]


def _label(text: str, position: tuple[float, float, float], anchor: str) -> gfx.Text:
    """Returns an outlined screen-space number at a position in the overlay."""
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
