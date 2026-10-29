"""Draws tractograms as lines in the 2D views, and as lines or tubes in 3D."""

import colorsys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from typing import ClassVar

import numpy as np
import pygfx as gfx

from larmor._log import LarmorLogger
from larmor.tractography import sample_volume
from larmor.types import Tractogram, Volume
from larmor.viewer.model.events import Event, OptionChanged, ViewStateChanged
from larmor.viewer.model.item import Item, ViewKind
from larmor.viewer.model.options import (
    POINTS_PER_STREAMLINE,
    Colormap,
    Colouring,
    OptionName,
    TractogramOptions,
    TractStyle,
)
from larmor.viewer.model.regions import RegionKey, region_filter
from larmor.viewer.model.registry import SUBSET_SEED
from larmor.viewer.model.settings import AppSettings
from larmor.viewer.model.view_state import ViewField
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.render.adapter import Adapter
from larmor.viewer.render.colour_space import physical_colours
from larmor.viewer.render.geometry import (
    BAND_QUANTILES,
    SEGMENT_BYTES,
    Lines,
    SortedSegments,
    Tubes,
    drawn_count,
    nan_separated,
    prefix_triangles,
    prefix_vertices,
    random_order,
    recoloured,
    reduce,
    slab_ranges,
    sorted_segments,
    tubes,
)
from larmor.viewer.render.line_voxels import LIT_PERCENTILE, LineVoxels, lit_share
from larmor.viewer.render.ray_traced_lines import (
    RayTracedLines,
    RayTracedLinesMaterial,
)
from larmor.viewer.render.view import ViewRenderer
from larmor.viewer.render.volume_adapter import COLORMAP_MAPS
from larmor.viewer.render.worker import Worker

SLAB_AXIS: dict[ViewKind, int] = {
    ViewKind.SAGITTAL: 0,
    ViewKind.CORONAL: 1,
    ViewKind.AXIAL: 2,
}
# Render queue of every item in the 2D views, pygfx's queue for the volumes'
# slices, so they draw in scene tree order whatever their opacity.
QUEUE_2D = 2600
# Render queue of translucent lines in the 3D view, before the transparent volume
# rendering and mask surfaces (pygfx's queue 3000), so those draw over them.
QUEUE_3D_TRANSLUCENT = 2900
# Options whose change rebuilds the drawn lines.
RESHAPING = (OptionName.SUBSET, OptionName.POINTS)
# Options whose change filters the streamlines again.
FILTERING = (OptionName.REGION_FILTER, OptionName.ENDS_ONLY, OptionName.REGIONS)
# Options of an applied region whose change filters the streamlines again.
REGION_FILTERING = (
    OptionName.ROLE,
    OptionName.CENTRE,
    OptionName.RADIUS,
    OptionName.SIZE,
    OptionName.MASK,
)
# Options whose change can call for other colours or values on the drawn lines.
RECOLOURING = (
    OptionName.COLOURING,
    OptionName.VALUES_FILE,
    OptionName.VALUES_VOLUME,
    OptionName.COLORMAP,
    OptionName.MINIMUM,
    OptionName.MAXIMUM,
)
# Where the values to colour by come from, as ("file", path), ("volume", item id)
# or ("none", None); and the lines' colours as (colormap, min, max), with no
# colormap for colours by direction.
type ValuesKey = tuple[str, str | int | None]
type ColoursKey = tuple[Colormap | None, float, float]
# What the drawn lines are: (selection version, count), points per streamline,
# values and colours. The version counts up with each new region selection, and
# count is the number of streamlines drawn.
type LinesKey = tuple[tuple[int, int], int, ValuesKey | None, ColoursKey | None]
BANDS = len(BAND_QUANTILES) + 1
# How near the pointer, in logical pixels, a drawn point of a 3D line is picked.
PICK_PIXELS = 5.0
# Vertices the 3D view draws while the user interacts, a random part of the lines
# as they are laid out in random order.
INTERACTIVE_VERTICES = 4_000_000
# Triangles of tubes drawn while the user interacts, likewise.
INTERACTIVE_TRIANGLES = 4_000_000


class SlabMemory:
    """Counts the memory the slabs' sorted copies take, against a budget.

    One budget is shared by every tractogram.
    """

    def __init__(self, limit: int) -> None:
        """Creates the budget."""
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def reserve(self, nbytes: int) -> bool:
        """Takes nbytes from the budget if they fit, and returns whether they did."""
        with self._lock:
            if self.used + nbytes > self.limit:
                return False
            self.used += nbytes
            return True

    def release(self, nbytes: int) -> None:
        """Gives nbytes back."""
        with self._lock:
            self.used -= nbytes


SLAB_BUDGET = SlabMemory(AppSettings().slab_bytes)  # a setting

# A thread for the lines, one for the tubes and one per slab axis.
_POOL = ThreadPoolExecutor(
    max_workers=2 + len(SLAB_AXIS), thread_name_prefix="larmor-tractogram"
)
# sRGB as a power of 2.2, enough to dim a line's colour by a share of light.
_GAMMA = 1 / 2.2


class _Share:
    """Holds a sorted copy's SLAB_BUDGET bytes until it is dropped and its work ends."""

    def __init__(self, nbytes: int) -> None:
        """Holds nbytes, already reserved."""
        self.nbytes = nbytes
        self._lock = threading.Lock()  # end comes from a worker thread
        self._ended = False
        self._dropped = False
        self._released = False

    def end(self) -> None:
        """Notes that the copy's work is over (run, failed or skipped)."""
        self._settle(ended=True)

    def drop(self) -> None:
        """Notes that the adapter no longer wants the copy."""
        self._settle(ended=False)

    def _settle(self, ended: bool) -> None:
        """Notes an end or a drop, and gives the bytes back once both happened."""
        with self._lock:
            if ended:
                self._ended = True
            else:
                self._dropped = True
            release = self._ended and self._dropped and not self._released
            self._released = self._released or release
        if release:
            SLAB_BUDGET.release(self.nbytes)


class TractogramAdapter(Adapter):
    """Draws a tractogram with one line object per view; 2D views show a slab.

    The lines drawn are a subset of the streamlines with fewer points each, in
    random order, so the 3D view can draw a prefix while the user interacts. Each
    2D view gets a copy of the segments sorted along its axis, made in the
    background within SLAB_BUDGET; until it is ready, or if it does not fit,
    clipping planes cut the slab. The lines, the region filter, the tubes and the
    sorted copies are made in the background, and their results come back on the
    main thread; nothing is drawn until the first lines come back. The region
    filter runs over all the streamlines loaded, and the subset and the points per
    streamline apply to those kept. Tubes are drawn only within
    tube_limit streamlines of POINTS_PER_STREAMLINE points; until the mesh is
    ready the lines stay. With ambient occlusion, the 3D view dims the lines or
    tubes by the light at each point; ray traced, it draws them as capsules of the
    tube radius instead.
    """

    draw_cap: ClassVar[int] = AppSettings().draw_cap  # a setting
    # A setting, the streamlines of POINTS_PER_STREAMLINE points drawn as tubes.
    tube_limit: ClassVar[int] = AppSettings().tube_limit

    def __init__(
        self,
        item: Item,
        views: dict[ViewKind, ViewRenderer],
        viewer: Viewer,
        order: float,
        tell: Callable[[str], None] = LarmorLogger.info,
    ) -> None:
        """Adds the line objects to the views and asks for their lines.

        `tell` is given a notice when the regions cannot be applied.
        """
        super().__init__(item, views, viewer, order)
        assert isinstance(item.data, Tractogram)
        # The streamlines in one random order, fixed; the order drawn keeps those
        # the regions keep, in the same order.
        self._full_order = random_order(item.data.num_streamlines, SUBSET_SEED)
        self._order = self._full_order
        # Which streamlines the regions keep, (S,) bool; None for all.
        self.selection: np.ndarray | None = None
        self._selection_version = 0
        self.tell = tell
        canvas = next(iter(views.values())).canvas
        self._lines_worker = Worker(canvas, _POOL)
        self._tubes_worker = Worker(canvas, _POOL)
        self._selection_worker = Worker(canvas, _POOL)
        self._sort_workers = {
            axis: Worker(canvas, _POOL) for axis in SLAB_AXIS.values()
        }
        self.sorted_copies: dict[int, SortedSegments] = {}  # by axis, once sorted
        self._shares: dict[int, _Share] = {}  # by axis, from request until dropped
        self._closed = False

        self._count = self._drawn_count()  # of the lines last asked for
        key = self._wanted()
        # The lines: subset, points, values and colours; and what they are. No
        # lines until the first ones come back from the background, under a key
        # that no request has, so that a change of colours meanwhile asks for lines.
        empty = np.empty((0, 3), np.float32)
        self.drawn = Lines(empty, np.zeros(1, np.int64), empty)
        self._drawn_key: LinesKey = ((-1, 0), key[1], None, None)
        self._asked_key: LinesKey = key  # of the lines last asked for
        self._prefix = prefix_vertices(self.drawn.offsets, INTERACTIVE_VERTICES)
        self._init_3d_state()
        self._geometry = geometry = _line_geometry(self.drawn)  # all views' lines
        self._before_draw: dict[ViewKind, Callable[[object], None]] = {}
        self.lines: dict[ViewKind, gfx.Line] = {}
        # A 2D view's slab, a line object per band of sorted segments.
        self.slabs: dict[ViewKind, list[gfx.Line]] = {}
        for kind, view in views.items():
            self._add_lines(kind, view, geometry)
        self._show()
        # Made in the background, as gathering the points of a large tractogram is
        # slow; the sorted copies, the tubes and the light follow the lines.
        self._lines_worker.submit(
            self._make_lines(key), partial(self._take_lines, key=key)
        )
        if region_filter(self.viewer.scene, item) is not None:
            self._request_selection()

    def _init_3d_state(self) -> None:
        """Sets the 3D view's tubes, light and ray tracer to nothing made yet."""
        self.tubes: gfx.Mesh | None = None  # the 3D view's, once built
        self._tubes_prefix = 0  # the triangles drawn while interacting
        self._tubes_asked: tuple[Lines, float] | None = None  # the last asked for
        self._tubes_of: Lines | None = None  # the lines the tubes were made of
        # What the 3D lines' and the tubes' colours were last dimmed for; None if
        # they are their own.
        self._lines_dimmed: tuple[object, ...] | None = None
        self._tubes_dimmed: tuple[object, ...] | None = None
        self._traced_colours: tuple[object, ...] | None = None  # given the ray tracer
        self._told_too_large: int | None = None  # the lists' entries last told of
        # The ambient light at the drawn points, scaled by light_scale so the most
        # exposed get 1, and the points and radius it is for.
        self.light: np.ndarray | None = None
        self.light_scale = 1.0
        self._light_of: tuple[np.ndarray, float] | None = None
        self._voxels: LineVoxels | None = None  # the GPU's grids, once needed
        # What the grids were built for: points, radius, and with segment lists.
        self._voxels_of: tuple[np.ndarray, float, bool] | None = None
        self.traced: RayTracedLines | None = None  # the 3D view's, once asked for

    def _add_lines(
        self, kind: ViewKind, view: ViewRenderer, geometry: gfx.Geometry
    ) -> None:
        """Adds a view's line object, and a 2D view's slab, to the view."""
        two_d = kind in SLAB_AXIS
        material = gfx.LineThinMaterial(color_mode="vertex", depth_test=not two_d)
        if two_d:
            material.render_queue = QUEUE_2D
        obj = gfx.Line(geometry, material)
        obj.render_order = self.order
        self.lines[kind] = obj
        view.add(obj)
        if two_d:
            shared = gfx.LineThinSegmentMaterial(color_mode="vertex", depth_test=False)
            shared.render_queue = QUEUE_2D
            self.slabs[kind] = []
            for _ in range(BANDS):
                slab = gfx.Line(_segment_geometry(None), shared)
                slab.render_order = self.order
                slab.visible = False
                self.slabs[kind].append(slab)
                view.add(slab)
        self._before_draw[kind] = partial(self._prepare, kind)
        view.canvas.add_event_handler(self._before_draw[kind], "before_draw")
        if not two_d:
            # Thin lines write no picks on the GPU: a click finds them on screen.
            view.pickers.append(self._pick_line)

    # Events

    def on_event(self, event: Event) -> None:
        """Follows this item's options, the cursor and the views on screen."""
        if isinstance(event, OptionChanged) and event.item.id == self.item.id:
            self._on_option(event)
        elif (
            isinstance(event, OptionChanged)
            and event.name in REGION_FILTERING
            and event.item.id in self._options().regions
        ):
            self._request_selection()
        elif isinstance(event, ViewStateChanged):
            if event.name == ViewField.CURSOR:
                assert isinstance(event.old, tuple) and isinstance(event.new, tuple)
                for kind, axis in SLAB_AXIS.items():
                    if kind in self.slabs and event.old[axis] != event.new[axis]:
                        self._show_slab(kind)
            elif event.name == ViewField.MAXIMISED:
                self.request_sorts()

    def _on_option(self, event: OptionChanged) -> None:
        """Makes again what an option of this item changes, then shows the lines."""
        if event.name in RESHAPING:
            self._request_lines()
            if not self._tubes_fit():
                self._drop_tubes()  # the lines show while the new ones are made
        elif event.name in RECOLOURING:
            self._request_colours()
        elif event.name in FILTERING:
            self._request_selection()
        elif event.name == OptionName.SLAB:
            if event.new:
                self.request_sorts()
            else:
                self._drop_sorts()
        elif event.name in (
            OptionName.STYLE,
            OptionName.TUBE_RADIUS,
            OptionName.VISIBLE,
            OptionName.AMBIENT_OCCLUSION,
            OptionName.RAY_TRACED,
        ):
            o = self._options()
            if not (o.ambient_occlusion or o.ray_traced):
                self._drop_voxels()
            self.update_tubes()
            self.update_light()
        self._show()

    def close(self) -> None:
        """Removes the lines and drops the work in progress."""
        self._closed = True
        self._drop_sorts()
        self._lines_worker.close()
        self._tubes_worker.close()
        self._selection_worker.close()
        for worker in self._sort_workers.values():
            worker.close()
        for kind, view in self.views.items():
            view.canvas.remove_event_handler(self._before_draw[kind], "before_draw")
            if self._pick_line in view.pickers:
                view.pickers.remove(self._pick_line)
            view.remove(self.lines[kind])
            for slab in self.slabs.get(kind, []):
                view.remove(slab)
        self._drop_tubes()
        self._drop_voxels()

    def _drop_voxels(self) -> None:
        """Lets the GPU grids and the ray tracer go, and their memory with them."""
        if self.traced is not None:
            self.views[ViewKind.THREE_D].remove(self.traced)
            self.traced = None
        self._voxels = self._voxels_of = self._light_of = self.light = None
        self._traced_colours = None

    def reshape(self) -> None:
        """Makes the lines again after a change of the drawing cap, if it counts."""
        if self._drawn_count() != self._count:
            self._request_lines()
        else:  # a larger budget can fit copies that did not
            self.request_sorts()
        self.update_tubes()

    def tube_notice(self) -> str | None:
        """Returns why the 3D view draws lines for tubes; None if it draws as asked."""
        o = self._options()
        if o.style != TractStyle.TUBES or not o.visible:
            return None
        if self._tubes_fit():
            return None
        return (
            f"{self.item.name}: {self._count:,} streamlines of {o.points:,} points "
            f"drawn, more than the tubes' budget of {self.tube_limit:,} streamlines "
            f"of {POINTS_PER_STREAMLINE} points, so the 3D view shows lines. Lower "
            "the subset or the points per streamline, or raise the tube limit in "
            "Settings."
        )

    def _tubes_fit(self) -> bool:
        """Returns whether the segments of the lines asked for fit the tubes' budget."""
        # From the count and the option, so it holds before the lines are made. It
        # can only overestimate, as a streamline shorter than the points is counted
        # at full length.
        work = self._count * (self._options().points - 1)
        return work <= self.tube_limit * (POINTS_PER_STREAMLINE - 1)

    # Drawing

    def _colour_mode(self) -> tuple[str, tuple[float, ...]]:
        """Returns the materials' colour mode and colour for the item's colouring."""
        o = self._options()
        if o.colouring in (Colouring.DIRECTION, Colouring.VALUES):
            return "vertex", (1.0, 1.0, 1.0)
        if o.colouring == Colouring.SINGLE:
            return "uniform", gfx.Color(o.colour).rgb
        # Hues a golden-ratio turn apart, so neighbouring items differ.
        hue = (self.item.id * 0.618033988749895) % 1.0
        return "uniform", colorsys.hsv_to_rgb(hue, 0.8, 1.0)

    def _show(self) -> None:
        """Colours and shows the lines, slabs, tubes and ray tracer in every view."""
        o = self._options()
        mode, colour = self._colour_mode()
        light = self._drawn_light()
        for kind, obj in self.lines.items():
            two_d = kind in SLAB_AXIS
            opacity = o.opacity_2d if two_d else o.opacity_3d
            view_mode, view_colour = mode, colour
            if not two_d:
                self._dim(light, mode, colour)
                if light is not None:  # the colours, dimmed, are on the vertices
                    view_mode, view_colour = "vertex", (1.0, 1.0, 1.0)
            materials = [obj.material]
            if two_d:
                materials.append(self.slabs[kind][0].material)  # shared
            if not two_d and self.tubes is not None:
                materials.append(self.tubes.material)
            for material in materials:
                material.color_mode = view_mode
                material.color = view_colour
                material.opacity = opacity
                material.alpha_mode = "solid" if opacity >= 1 else "blend"
                if not two_d:  # 0: pygfx's own queue, for opaque lines
                    material.render_queue = 0 if opacity >= 1 else QUEUE_3D_TRANSLUCENT
            if two_d:
                self._show_slab(kind)
            else:
                traced = self._trace(mode, colour, opacity)
                obj.visible = o.visible and self.tubes is None and not traced
                if self.tubes is not None:
                    self.tubes.visible = o.visible and not traced
                self.views[kind].request_draw()

    def _show_slab(self, kind: ViewKind) -> None:
        """Shows a 2D view's lines: all, clipped to the slab, or the slab's range."""
        o = self._options()
        axis = SLAB_AXIS[kind]
        lines, slabs = self.lines[kind], self.slabs[kind]
        centre = self.viewer.scene.view.cursor[axis]
        low, high = centre - o.slab_thickness / 2, centre + o.slab_thickness / 2
        sorted_copy = self.sorted_copies.get(axis) if o.slab else None
        planes: list[tuple[float, float, float, float]] = []
        if o.slab:
            a, b, c = (1.0 if i == axis else 0.0 for i in range(3))
            # pygfx keeps the points p of a plane (n, w) where n . p >= w.
            planes = [(a, b, c, low), (-a, -b, -c, -high)]
        if sorted_copy is not None:
            slabs[0].material.clipping_planes = planes
            ranges = slab_ranges(sorted_copy, low, high)
            for slab, (start, stop), band in zip(
                slabs, ranges, sorted_copy.bands, strict=True
            ):
                # Each band's buffer holds that band alone, from its first segment.
                if stop > start:  # an empty range may start past the buffer's end
                    offset = 2 * (start - band[0])
                    slab.geometry.positions.draw_range = (offset, 2 * (stop - start))
                slab.visible = o.visible and stop > start
            lines.visible = False
        else:
            lines.material.clipping_planes = planes
            lines.visible = o.visible
            for slab in slabs:
                slab.visible = False
        self.views[kind].request_draw()

    def _options(self) -> TractogramOptions:
        """Returns the item's tractogram options."""
        o = self.item.options
        assert isinstance(o, TractogramOptions)
        return o

    # Derived arrays

    def _drawn_count(self) -> int:
        """Returns the streamlines drawn for the current subset and drawing cap."""
        t = self.item.data
        assert isinstance(t, Tractogram)
        return drawn_count(len(self._order), self._options().subset, self.draw_cap)

    def _values(self) -> tuple[ValuesKey, np.ndarray | Volume | None]:
        """Returns where the values to colour by come from, and the values or volume."""
        o = self._options()
        if o.values_volume is not None:
            volume = self.viewer.scene.item(o.values_volume).data
            assert isinstance(volume, Volume)
            return ("volume", o.values_volume), volume
        values = self.viewer._tsf_point_values(self.item.id) if o.values_file else None
        if values is not None:
            return ("file", o.values_file), values
        return ("none", None), None

    def _wanted(self) -> LinesKey:
        """Returns the key of the lines the options call for; None where any will do."""
        o = self._options()
        values: ValuesKey | None = None
        colours: ColoursKey | None = None
        if o.colouring == Colouring.VALUES:
            values = self._values()[0]
            colours = (o.colormap, o.minimum, o.maximum)
        elif o.colouring == Colouring.DIRECTION:
            colours = (None, 0.0, 1.0)
        return ((self._selection_version, self._count), o.points, values, colours)

    def _make_lines(self, key: LinesKey) -> Callable[[], Lines]:
        """Returns the work that makes the lines `key` describes."""
        t = self.item.data
        assert isinstance(t, Tractogram)
        # In random order: any prefix of the lines is a random subset of them.
        order = self._order[: key[0][1]]
        source = self._values()[1] if key[2] is not None else None
        return partial(
            _lines, t.points, t.offsets, order, key[1], source, self._table(key[3])
        )

    def _table(
        self, colours: ColoursKey | None
    ) -> tuple[np.ndarray | None, float, float]:
        """Returns a colormap's table and range for a key; no table for direction."""
        if colours is None:
            return (None, 0.0, 1.0)
        colormap, low, high = colours
        if colormap is None:
            return (None, 0.0, 1.0)
        return (COLORMAP_MAPS[colormap].texture.data, low, high)

    def _pick_line(self, view: ViewRenderer, x: float, y: float) -> np.ndarray | None:
        """Returns the point of a drawn 3D line nearest the camera near a canvas point.

        A segment counts when it passes within PICK_PIXELS of the point on screen.

        Returns:
            (3,) the point, world mm; None if the lines are hidden or none is near.
        """
        o = self._options()
        if not o.visible or o.opacity_3d <= 0:
            return None
        points = self.drawn.points.astype(np.float64)
        starts = np.ones(len(points), dtype=bool)
        starts[-1:] = False
        starts[self.drawn.offsets[1:-1] - 1] = False  # no segment across streamlines
        first = np.flatnonzero(starts)
        if len(first) == 0:
            return None
        width, height = view.logical_size()
        view.camera.set_view_size(width, height)
        matrix = view.camera.camera_matrix
        clip = points @ matrix[:, :3].T + matrix[:, 3]
        w = clip[:, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            ndc = clip[:, :3] / w[:, None]
        screen = np.stack(
            [(ndc[:, 0] + 1) * width / 2, (1 - ndc[:, 1]) * height / 2], 1
        )
        a, b = screen[first], screen[first + 1]
        step = b - a
        length = np.einsum("ij,ij->i", step, step)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.einsum("ij,ij->i", np.array([x, y]) - a, step) / length
        t = np.clip(np.nan_to_num(t), 0.0, 1.0)
        distance = np.linalg.norm(a + t[:, None] * step - (x, y), axis=1)
        depth = ndc[first, 2] + t * (ndc[first + 1, 2] - ndc[first, 2])
        visible = (w[first] > 0) & (w[first + 1] > 0) & (depth >= 0) & (depth <= 1)
        near = visible & (distance <= PICK_PIXELS)
        if not near.any():
            return None
        chosen = np.flatnonzero(near)[np.argmin(depth[near])]
        start, end = points[first[chosen]], points[first[chosen] + 1]
        nearest: np.ndarray = start + t[chosen] * (end - start)
        return nearest

    def _request_lines(self) -> None:
        """Asks for the lines the options call for, in the background."""
        self._count = self._drawn_count()
        self._drop_sorts()
        key = self._asked_key = self._wanted()
        self._lines_worker.submit(
            self._make_lines(key), partial(self._take_lines, key=key)
        )

    def refilter(self) -> None:
        """Filters the streamlines again, as when a mask's data changed."""
        self._request_selection()

    def _request_selection(self) -> None:
        """Filters the streamlines by the regions in the background, if any."""
        regions = region_filter(self.viewer.scene, self.item)
        key, work = self.viewer._selection_work(self.item.id)
        if regions is None:
            self._selection_worker.cancel()
            if self.selection is not None:
                self._take_selection(None, key)
            return
        self._selection_worker.submit(
            work,
            partial(self._take_selection, key=key),
            self._selection_failed,
        )

    def _selection_failed(self) -> None:
        """Tells the user the regions could not be applied."""
        self.tell(f"{self.item.name}: the regions could not be applied; see the log.")

    def _take_selection(
        self, selection: np.ndarray | None, key: RegionKey | None = None
    ) -> None:
        """Draws the streamlines kept, keeping their random order.

        With the region options it was made for, the viewer keeps the selection.
        """
        if key is not None:
            self.viewer._remember_selection(self.item.id, key, selection)
        self.selection = selection
        self._selection_version += 1
        order = self._full_order
        self._order = order if selection is None else order[selection[order]]
        self._request_lines()

        if not self._tubes_fit():
            self._drop_tubes()  # the lines show while the new ones are made
        self._show()

    def _request_colours(self) -> None:
        """Gives the lines the values and colours the options call for, if missing."""
        wanted = self._wanted()
        drawn = self._drawn_key
        if self._lines_worker.pending and self._asked_key[:3] != drawn[:3]:
            # New lines are being made: new colours of the drawn ones would not do.
            self._request_lines()
        elif wanted[:2] != drawn[:2] or wanted[2] not in (None, drawn[2]):
            self._request_lines()
        elif wanted[3] not in (None, drawn[3]) or self._lines_worker.pending:
            # Same points and values: only their colours change.
            key = self._asked_key = (*drawn[:3], wanted[3])
            work = partial(recoloured, self.drawn, *self._table(wanted[3]))
            self._lines_worker.submit(work, partial(self._take_lines, key=key))

    def _take_lines(self, lines: Lines, key: LinesKey) -> None:
        """Draws new lines and asks again for what is made from them."""
        self.drawn = lines
        self._drawn_key = key
        self._prefix = prefix_vertices(lines.offsets, INTERACTIVE_VERTICES)
        self._geometry = geometry = _line_geometry(lines)
        for kind, obj in self.lines.items():
            obj.geometry = geometry
            self.views[kind].request_draw()
        self._drop_sorts()  # any made meanwhile are of the old lines
        self.request_sorts()
        self.update_tubes()
        self.update_light()
        if self._options().ambient_occlusion or self._options().ray_traced:
            self._show()  # dimmed or traced again

    def update_tubes(self) -> None:
        """Asks for the tubes the options call for, or drops them if none are."""
        o = self._options()
        if o.style != TractStyle.TUBES or o.ray_traced or not self._tubes_fit():
            self._drop_tubes()
            return
        if len(self.drawn.offsets) - 1 != self._count:
            return  # the lines are being made again: their tubes come after them
        if not o.visible or self._closed or ViewKind.THREE_D not in self.views:
            return  # made when shown
        asked = self._tubes_asked
        if asked is not None and asked[0] is self.drawn and asked[1] == o.tube_radius:
            return
        key = self._tubes_asked = (self.drawn, o.tube_radius)
        self._tubes_worker.submit(
            partial(tubes, self.drawn, o.tube_radius),
            partial(self._take_tubes, of=self.drawn),
            partial(self._tubes_failed, key),
        )

    def _take_tubes(self, made: Tubes, of: Lines | None = None) -> None:
        """Puts the tubes made in the 3D view."""
        view = self.views[ViewKind.THREE_D]
        geometry = _tube_geometry(made)
        if self.tubes is None:
            material = gfx.MeshPhongMaterial(
                color_mode="vertex", side="front", pick_write=True
            )
            self.tubes = gfx.Mesh(geometry, material)
            self.tubes.render_order = self.order
            view.add(self.tubes)
        else:
            self.tubes.geometry = geometry
        self._tubes_of = of
        self._tubes_dimmed = None
        self._tubes_prefix = prefix_triangles(made.ends, INTERACTIVE_TRIANGLES)
        self._show()

    def _tubes_failed(self, key: tuple[Lines, float]) -> None:
        """Forgets the failed request, so the tubes are asked for again."""
        if self._tubes_asked is key:
            self._tubes_asked = None  # asked again next time

    def _drop_tubes(self) -> None:
        """Drops the tubes and their work, and shows the 3D view's lines again."""
        self._tubes_worker.cancel()
        self._tubes_asked = None
        if self.tubes is not None:
            view = self.views[ViewKind.THREE_D]
            view.remove(self.tubes)
            self.tubes = self._tubes_of = self._tubes_dimmed = None
            self.lines[ViewKind.THREE_D].visible = self._options().visible
            view.request_draw()

    def update_light(self) -> None:
        """Works out the ambient light at the drawn points if the options ask for it."""
        o = self._options()
        if not o.ambient_occlusion or ViewKind.THREE_D not in self.views:
            self.light = self._light_of = None
            return
        if len(self.drawn.offsets) - 1 != self._count:
            return  # the lines are being made again: their light comes after them
        if not o.visible or self._closed:
            return  # made when shown
        key = (self.drawn.points, o.tube_radius)
        known = self._light_of
        if known is not None and known[0] is key[0] and known[1] == key[1]:
            return
        # With segment lists if the ray tracer will need them: built once.
        light = self._built_voxels(lists=o.ray_traced).light_at_points()
        top = float(np.percentile(light, LIT_PERCENTILE)) if len(light) else 1.0
        self.light_scale = 1.0 / max(top, 1e-6)
        self.light = lit_share(light)
        self._light_of = key

    def _built_voxels(self, lists: bool) -> LineVoxels:
        """Returns the GPU's grids of the drawn lines, built again if of others."""
        o = self._options()
        if self._voxels is None:
            self._voxels = LineVoxels()
        built = self._voxels_of
        if (
            built is None
            or built[0] is not self.drawn.points
            or built[1] != o.tube_radius
            or (lists and not built[2])
        ):
            self._voxels.build(self.drawn, o.tube_radius, lists=lists)
            self._voxels_of = (self.drawn.points, o.tube_radius, lists)
        return self._voxels

    def _trace(self, mode: str, colour: tuple[float, ...], opacity: float) -> bool:
        """Ray traces the drawn lines in the 3D view if the options ask for it."""
        o = self._options()
        if not (o.ray_traced and o.visible and not self._closed):
            if self.traced is not None:
                self.traced.visible = False
            return False
        voxels = self._built_voxels(lists=True)
        if not voxels.lists_fit:
            if self.traced is not None:
                self.traced.visible = False
            if self._told_too_large != voxels.num_entries:
                self._told_too_large = voxels.num_entries
                self.tell(
                    f"{self.item.name}: ray tracing needs {voxels.num_entries:,} "
                    f"entries in its voxel lists, more than the GPU holds in one "
                    "buffer, so the 3D view shows lines. Lower the subset, the "
                    "points per streamline or the tube radius."
                )
            return False
        if self.traced is None:
            self.traced = RayTracedLines(RayTracedLinesMaterial())
            self.traced.render_order = self.order
            self.views[ViewKind.THREE_D].add(self.traced)
        scale = self.light_scale if self._drawn_light() is not None else None
        self.traced.set_grid(voxels, o.tube_radius, scale)
        key = (self.drawn.points, self.drawn.colours, mode, colour)
        if not _same(key, self._traced_colours):
            self.traced.set_colours(self._base_colours(mode, colour))
            self._traced_colours = key
        material = self.traced.material
        assert isinstance(material, RayTracedLinesMaterial)
        material.opacity = opacity
        material.transparent = opacity < 1
        material.alpha_mode = "solid" if opacity >= 1 else "blend"
        material.render_queue = 0 if opacity >= 1 else QUEUE_3D_TRANSLUCENT
        return True

    def _drawn_light(self) -> np.ndarray | None:
        """Returns the light at the drawn points if wanted and made for them."""
        of = self._light_of
        if not self._options().ambient_occlusion or of is None:
            return None
        return self.light if of[0] is self.drawn.points else None

    def _base_colours(self, mode: str, colour: tuple[float, ...]) -> np.ndarray:
        """Returns the drawn points' sRGB colours, their own or the uniform colour."""
        if mode == "vertex":
            return self.drawn.colours
        n = len(self.drawn.points)
        return np.broadcast_to(np.asarray(colour[:3], np.float32), (n, 3))

    def _dim(
        self, light: np.ndarray | None, mode: str, colour: tuple[float, ...]
    ) -> None:
        """Gives the 3D view's lines and tubes their colours times the light, or not.

        Only what changed is made and uploaded again.
        """
        view = self.lines[ViewKind.THREE_D]
        key = (self.drawn.points, self.drawn.colours, light, mode, colour)
        if light is None:
            if view.geometry is not self._geometry:
                view.geometry = self._geometry
            self._lines_dimmed = None
            if self._tubes_dimmed is not None and self._tubes_of is not None:
                self._set_tube_colours(physical_colours(self._tubes_of.colours))
            self._tubes_dimmed = None
            return
        if not _same(key, self._lines_dimmed):
            # Light scales linear values, so a line's sRGB colour by its power 1 / 2.2.
            base = self._base_colours(mode, colour)
            dimmed = replace(self.drawn, colours=base * light[:, None] ** _GAMMA)
            view.geometry = gfx.Geometry(
                positions=self._geometry.positions, colors=nan_separated(dimmed)[1]
            )
            self._lines_dimmed = key
        tubes_key = (self.tubes, *key)
        if (
            self.tubes is not None
            and self._tubes_of is not None
            and self._tubes_of.points is self.drawn.points
            and not _same(tubes_key, self._tubes_dimmed)
        ):
            physical = physical_colours(self._base_colours(mode, colour))
            self._set_tube_colours(physical * light[:, None])
            self._tubes_dimmed = tubes_key

    def _set_tube_colours(self, colours: np.ndarray) -> None:
        """Gives the tubes a colour per point of their lines, on each ring vertex."""
        assert self.tubes is not None
        buffer = self.tubes.geometry.colors
        sides = buffer.nitems // max(len(colours), 1)
        if sides == 0 or sides * len(colours) != buffer.nitems:
            return  # the placeholder of no tubes, or tubes of other lines
        buffer.data.reshape(len(colours), sides, 3)[:] = colours[:, None, :]
        buffer.update_full()

    def request_sorts(self) -> None:
        """Starts the sorted copies the 2D views on screen lack, if they fit."""
        if not self._options().slab or self._closed:
            return
        maximised = self.viewer.scene.view.maximised
        kinds = [k for k in self.slabs if maximised in (None, k)]
        nbytes = self.drawn.num_segments * SEGMENT_BYTES
        for kind in kinds:
            axis = SLAB_AXIS[kind]
            if axis in self._shares:
                continue
            if not SLAB_BUDGET.reserve(nbytes):
                LarmorLogger.info(
                    "The slab of %s along the %s axis would exceed its memory "
                    "budget; it is cut with clipping planes, which is slower.",
                    self.item.name,
                    "xyz"[axis],
                )
                continue
            share = self._shares[axis] = _Share(nbytes)
            self._sort_workers[axis].submit(
                partial(sorted_segments, self.drawn, axis),
                partial(self._take_sort, axis, share),
                partial(self._sort_failed, axis, share),
                share.end,
            )

    def _take_sort(self, axis: int, share: _Share, segments: SortedSegments) -> None:
        """Gives the slabs along axis their sorted copy, unless it was dropped."""
        if self._shares.get(axis) is not share:
            return  # dropped meanwhile
        self.sorted_copies[axis] = segments
        for kind, slabs in self.slabs.items():
            if SLAB_AXIS[kind] == axis:
                for slab, band in zip(slabs, segments.bands, strict=True):
                    slab.geometry = _segment_geometry(segments, band)
                self._show_slab(kind)

    def _sort_failed(self, axis: int, share: _Share) -> None:
        """Drops a failed sorted copy's share of the budget."""
        if self._shares.get(axis) is share:
            del self._shares[axis]
            share.drop()

    def _drop_sorts(self) -> None:
        """Drops the sorted copies and their work, and shows the lines clipped."""
        for worker in self._sort_workers.values():
            worker.cancel()
        for share in self._shares.values():
            share.drop()  # its memory comes back once its work has ended too
        self._shares.clear()
        self.sorted_copies.clear()
        for kind, slabs in self.slabs.items():
            for slab in slabs:
                slab.geometry = _segment_geometry(None)
            self._show_slab(kind)

    def _prepare(self, kind: ViewKind, event: object = None) -> None:
        """Sets the range of the lines a view draws, before it draws."""
        # The views share the lines' buffer: each sets the range it draws.
        positions = self.lines[kind].geometry.positions
        size = positions.nitems
        if kind not in SLAB_AXIS and self.views[kind].interacting:
            size = min(self._prefix, size)
        if tuple(positions.draw_range) != (0, size):
            positions.draw_range = (0, size)
        if kind == ViewKind.THREE_D and self.tubes is not None:
            indices = self.tubes.geometry.indices
            count = indices.nitems
            if self.views[kind].interacting:
                count = min(self._tubes_prefix, count)
            if tuple(indices.draw_range) != (0, count):
                indices.draw_range = (0, count)


def _lines(
    points: np.ndarray,
    offsets: np.ndarray,
    order: np.ndarray,
    n_points: int,
    source: np.ndarray | Volume | None,
    colours: tuple[np.ndarray | None, float, float],
) -> Lines:
    """Returns the streamlines `order` with n_points each, their values and colours.

    The values are a TSF file's, kept at the drawn points, or a volume sampled at
    them; the colours come from the values through the table, or from the
    directions without one.

    Returns:
        The lines.
    """
    if isinstance(source, Volume):
        lines = reduce(points, offsets, order, n_points)
        values = sample_volume(Tractogram(lines.points, lines.offsets), source)
        lines = replace(lines, values=values)
    else:
        lines = reduce(points, offsets, order, n_points, source)
    return recoloured(lines, *colours)


def _line_geometry(lines: Lines) -> gfx.Geometry:
    """Returns one polyline for all streamlines, broken by NaN points."""
    positions, colours = nan_separated(lines)
    if len(positions) == 0:  # pygfx needs a vertex
        positions = np.full((1, 3), np.nan, np.float32)
        colours = np.zeros((1, 3), np.float32)
    return gfx.Geometry(positions=positions, colors=colours)


def _tube_geometry(made: Tubes) -> gfx.Geometry:
    """Returns the tubes' mesh, or one degenerate triangle if there are none."""
    if len(made.indices) == 0:  # pygfx needs a triangle
        point = np.zeros((1, 3), np.float32)
        return gfx.Geometry(
            positions=point,
            normals=point,
            colors=point,
            indices=np.zeros((1, 3), np.uint32),
        )
    return gfx.Geometry(
        positions=made.positions,
        normals=made.normals,
        colors=made.colours,
        indices=made.indices,
    )


def _segment_geometry(
    segments: SortedSegments | None, band: tuple[int, int, float] = (0, 0, 0.0)
) -> gfx.Geometry:
    """Returns one band of sorted segments as pairs of vertices, or an empty pair."""
    start, stop, _ = band
    if segments is None or stop == start:
        empty = np.full((2, 3), np.nan, np.float32)
        return gfx.Geometry(positions=empty, colors=np.zeros((2, 3), np.float32))
    # Views of the copy, so each band's buffer uploads its own part of it.
    return gfx.Geometry(
        positions=segments.positions[2 * start : 2 * stop],
        colors=segments.colours[2 * start : 2 * stop],
    )


def _same(key: tuple[object, ...], known: tuple[object, ...] | None) -> bool:
    """Returns whether two keys hold the same arrays (by identity) and values."""
    if known is None or len(key) != len(known):
        return False
    return all(
        a is b if isinstance(a, np.ndarray | gfx.Mesh) or a is None else a == b
        for a, b in zip(key, known, strict=True)
    )
