"""Draws peaks as short lines through the voxels of each 2D view's slice."""

from functools import partial

import numpy as np
import pygfx as gfx

from larmor.types import Volume
from larmor.viewer.model.events import Event, OptionChanged, ViewStateChanged
from larmor.viewer.model.item import Item, ViewKind
from larmor.viewer.model.options import Colouring, PeaksOptions
from larmor.viewer.model.view_state import ViewField
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.render.adapter import Adapter
from larmor.viewer.render.geometry import (
    largest_amplitude,
    outward_halves,
    peak_segments,
    slice_index,
)
from larmor.viewer.render.tractogram_adapter import QUEUE_2D, SLAB_AXIS
from larmor.viewer.render.view import ViewRenderer
from larmor.viewer.render.worker import Worker

# What a view's segments are built from, the voxel slice (None outside the grid),
# the scale, the amplitude switch, the threshold and whether peaks are one-sided.
type Key = tuple[tuple[int, int] | None, float, bool, float, bool]


class PeaksAdapter(Adapter):
    """Draws the peaks image's slice in each 2D view, as one line object of segments.

    A view's segments are built again in the background only when its slice or an
    option that shapes them changes. A hidden item builds nothing.
    """

    def __init__(
        self,
        item: Item,
        views: dict[ViewKind, ViewRenderer],
        viewer: Viewer,
        order: float,
    ) -> None:
        """Adds an empty line object to each 2D view and builds what is shown."""
        super().__init__(item, views, viewer, order)
        assert isinstance(item.data, Volume)
        self._largest: float | None = None  # the image's largest amplitude, once
        self._asked: dict[ViewKind, Key | None] = {}  # the key last built or asked
        self._workers: dict[ViewKind, Worker] = {}
        self.lines: dict[ViewKind, gfx.Line] = {}
        for kind, view in views.items():
            if kind not in SLAB_AXIS:
                continue
            material = gfx.LineSegmentMaterial(
                color_mode="vertex", thickness_space="screen", depth_test=False
            )
            material.render_queue = QUEUE_2D
            obj = gfx.Line(_geometry(None), material)
            obj.render_order = order
            self.lines[kind] = obj
            self._asked[kind] = None
            self._workers[kind] = Worker(view.canvas)
            view.add(obj)
        self._show()
        self._build()

    def on_event(self, event: Event) -> None:
        """Follows this item's options and the cursor."""
        if isinstance(event, OptionChanged) and event.item.id == self.item.id:
            self._show()
            self._build()
        elif isinstance(event, ViewStateChanged) and event.name == ViewField.CURSOR:
            self._build()

    def close(self) -> None:
        """Removes the lines and drops the work in progress."""
        for kind, obj in self.lines.items():
            self._workers[kind].close()
            self.views[kind].remove(obj)

    def _options(self) -> PeaksOptions:
        """Returns the item's peaks options."""
        o = self.item.options
        assert isinstance(o, PeaksOptions)
        return o

    def _show(self) -> None:
        """Applies the options that change only the material."""
        o = self._options()
        single = o.colouring == Colouring.SINGLE
        for kind, obj in self.lines.items():
            material = obj.material
            material.color_mode = "uniform" if single else "vertex"
            material.color = gfx.Color(o.colour).rgb if single else (1.0, 1.0, 1.0)
            material.thickness = o.line_width
            material.opacity = o.opacity_2d
            material.alpha_mode = "solid" if o.opacity_2d >= 1 else "blend"
            obj.visible = o.visible
            self.views[kind].request_draw()

    def _build(self) -> None:
        """Asks for the segments of each view whose key changed, if shown."""
        o = self._options()
        if not o.visible:
            return
        volume = self.item.data
        assert isinstance(volume, Volume)
        cursor = np.array(self.viewer.scene.view.cursor)
        for kind in self.lines:
            where = slice_index(
                volume.affine, volume.data.shape, SLAB_AXIS[kind], cursor
            )
            key: Key = (where, o.scale, o.amplitude, o.threshold, o.asymmetric_peaks)
            if key == self._asked[kind]:
                continue
            self._asked[kind] = key
            self._workers[kind].submit(
                partial(self._segments, key),
                partial(self._take, kind),
                partial(self._failed, kind, key),
            )

    def _segments(self, key: Key) -> tuple[np.ndarray, np.ndarray] | None:
        """Builds one view's segments; runs on the worker."""
        where, scale, amplitude, threshold, one_sided = key
        if where is None:
            return None
        volume = self.item.data
        assert isinstance(volume, Volume)
        if self._largest is None and (amplitude or threshold > 0):
            self._largest = largest_amplitude(
                volume.data, volume.slope, volume.intercept
            )
        largest = self._largest or 0.0
        reference = largest if amplitude else None
        axis, index = where
        # At scale 1 the largest peak spans the smallest voxel side.
        length = scale * float(volume.voxel_sizes.min())
        segments = peak_segments(
            volume.data,
            volume.affine,
            axis,
            index,
            length,
            reference,
            threshold * largest,
            volume.slope,
            volume.intercept,
        )
        return outward_halves(segments) if one_sided else segments

    def _take(
        self, kind: ViewKind, segments: tuple[np.ndarray, np.ndarray] | None
    ) -> None:
        """Puts a view's new segments in its line object."""
        self.lines[kind].geometry = _geometry(segments)
        self.views[kind].request_draw()

    def _failed(self, kind: ViewKind, key: Key) -> None:
        """Forgets a failed request, so the segments are asked for again."""
        if self._asked[kind] == key:
            self._asked[kind] = None  # asked again next time


def _geometry(segments: tuple[np.ndarray, np.ndarray] | None) -> gfx.Geometry:
    """Returns the segments' geometry, or one empty pair as pygfx needs a vertex."""
    if segments is None or len(segments[0]) == 0:
        empty = np.full((2, 3), np.nan, np.float32)
        return gfx.Geometry(positions=empty, colors=np.zeros((2, 3), np.float32))
    positions, colours = segments
    return gfx.Geometry(positions=positions, colors=colours)
