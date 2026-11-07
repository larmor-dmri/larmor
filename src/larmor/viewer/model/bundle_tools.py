"""The drawing tools of the SymDWI workspace, as logic without widgets."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import TYPE_CHECKING, Final, Protocol

import numpy as np

from larmor.viewer.model.bundle_shape import (
    MIN_WIDTH,
    Cut,
    Frame,
    Rows,
    Shape,
    bent,
    bent_branch,
    brush_rows,
    crossing_of,
    cut,
    ensure_row,
    extended,
    followed_radius,
    landed,
    prune_rows,
    read_line,
    redrawn,
    rotated_line,
    rows_from_track,
    scaled_line,
    settled,
    shape_of,
    simplify_stroke,
    stroke_length,
    vec3,
    visible_width,
    with_family,
    with_landing,
    with_opening,
    with_rows,
)
from larmor.viewer.model.item import ItemId
from larmor.viewer.model.options import SLAB_THICKNESS, TractogramOptions
from larmor.viewer.model.phantom import (
    BundleField,
    BundleSettings,
    Draft,
    GeometrySettings,
    LandingSettings,
    PhantomState,
    Vec3,
    drawable,
    with_geometry,
)
from larmor.viewer.model.view_state import Tool

if TYPE_CHECKING:
    from larmor.viewer.model.phantom_editor import PhantomEditor
    from larmor.viewer.model.viewer import Viewer

# The tools that work on a phantom's bundles, in the order the toolbar shows them.
DRAWING_TOOLS: Final = (
    Tool.DRAW_BUNDLE,
    Tool.BEND,
    Tool.WIDTH,
    Tool.TWIST,
    Tool.FAN,
    Tool.ERASE,
)
# How near, in logical pixels, a press must be to a control point to take it.
HANDLE_PIXELS: Final = 6.0
# How near, in logical pixels, a press must be to a bundle's tip to take it.
TIP_PIXELS: Final = 16.0
# A press that moves less than this, in logical pixels, is a click.
CLICK_PIXELS: Final = 4.0
# A stroke shorter than this draws nothing, mm.
SHORTEST_STROKE: Final = 2.0
# A Fan's line shorter than this does nothing, mm.
SHORTEST_LINE: Final = 4.0
# An Erase stroke shorter than this is a click that removes a bundle, mm.
SHORTEST_CUT: Final = 3.0
# The Pen's radius from its keys: its range and step, mm.
PEN_RADII: Final = (0.5, 20.0)
PEN_STEP: Final = 0.5
# The largest Bend reach, at which the whole bundle moves, and the others', mm.
WHOLE_BUNDLE: Final = 150.0
LARGEST_REACH: Final = 200.0
SMALLEST_REACH: Final = 5.0
# Reach steps by this much below `REACH_BIG`, and by `STEP_BIG` from there on, mm.
REACH_STEP: Final = 5.0
REACH_BIG: Final = 50.0
STEP_BIG: Final = 10.0
# The range of a fan's Opens from.
OPENS_RANGE: Final = (0.1, 0.9)
# How far from a slice a bundle counts as far when the views show all of it, mm.
FAR_WITHOUT_SLAB: Final = 15.0

# Where the Pen's strokes and points show, as the bundle being drawn does.
POINT_COLOUR: Final = "#ff4fa3"
OUTLINE_COLOUR: Final = "#ffffff"
ERASE_COLOUR: Final = "#ff8888"


class Plane(Protocol):
    """Tells the tools how a 2D view shows the world."""

    @property
    def axis(self) -> int:
        """The world axis the view looks along."""
        ...

    @property
    def mm_per_pixel(self) -> float:
        """How many mm one logical pixel spans."""
        ...

    def project(self, points: np.ndarray) -> np.ndarray:
        """Returns where world points (N, 3) show, in logical pixels, (N, 2)."""
        ...

    def world(self, x: float, y: float, depth: float) -> np.ndarray:
        """Returns the world point under a pixel, at a depth along the view's axis."""
        ...


@dataclass(frozen=True)
class Pointer:
    """Where the pointer is in a 2D view, and the keys held."""

    plane: Plane
    x: float
    y: float
    ctrl: bool = False  # Ctrl, or Cmd on macOS
    shift: bool = False


@dataclass(frozen=True, eq=False)
class Placed:
    """A bundle that the tools can reach, measured.

    Attributes:
        item_id: Its tractogram item.
        shape: Its measures.
        is_draft: Whether it is the bundle being drawn.
        slab: How thick its slab is in the 2D views, mm; None if they show all of it.
    """

    item_id: ItemId
    shape: Shape
    is_draft: bool
    slab: float | None


@dataclass(frozen=True, eq=False)
class Hit:
    """A place along a bundle under the pointer."""

    placed: Placed
    u: float
    k: int
    far: bool  # too far from the slice for the bundle to show there


@dataclass(frozen=True, eq=False)
class Tip:
    """An end of a bundle under the pointer."""

    placed: Placed
    at_end: bool  # its last point rather than its first
    k: int
    dist: float


class _Start(Enum):
    """What lies under a stroke's press."""

    EMPTY = auto()
    TIP = auto()
    BODY = auto()


@dataclass(frozen=True, eq=False)
class _StrokeStart:
    """What a stroke starts on, and the depth it starts at."""

    kind: _Start
    depth: float
    placed: Placed | None = None
    at_end: bool = True
    u: float = 0.0
    far: bool = False


class _Reading(Enum):
    """What a stroke means, from where it started and where it stopped."""

    NEW = auto()
    EXTEND = auto()
    BRANCH = auto()
    REDRAW = auto()


@dataclass(frozen=True, eq=False)
class _Read:
    """What a stroke means, and on what."""

    kind: _Reading
    placed: Placed | None = None
    at_end: bool = True
    u: float = 0.0  # where it left, or where a redone stretch starts
    stop_u: float = 0.0  # where a redone stretch stops


@dataclass(eq=False)
class Overlay:
    """What the tools draw over the views, in world mm.

    Attributes:
        outlines: Thin white lines, such as a section's ring.
        reaches: Thick faint lines that show how far along a tool acts.
        dashed: Dashed lines, such as a fan's saved lines or the next point's line.
        strokes: Lines being drawn, with their colours.
        ticks: The places of a bundle's rows, (N, 3).
        circles: Rings a fixed size on the screen, such as a marked tip, (N, 3).
        under_pointer: Thin white lines for the 2D view under the pointer alone.
        under_pointer_axis: The world axis that view looks along; None for none.
    """

    outlines: list[np.ndarray] = field(default_factory=list)
    reaches: list[np.ndarray] = field(default_factory=list)
    dashed: list[np.ndarray] = field(default_factory=list)
    strokes: list[tuple[np.ndarray, str]] = field(default_factory=list)
    ticks: list[np.ndarray] = field(default_factory=list)
    circles: list[np.ndarray] = field(default_factory=list)
    under_pointer: list[np.ndarray] = field(default_factory=list)
    under_pointer_axis: int | None = None


@dataclass(frozen=True)
class ToolSlider:
    """A slider of a tool's options bar.

    Attributes:
        key: The option's name, for `BundleTools.set_option`.
        label: The name it shows.
        low: The lowest value.
        high: The highest value.
        step: A step of the slider.
        value: The value now.
        text: The value in words.
    """

    key: str
    label: str
    low: float
    high: float
    step: float
    value: float
    text: str


@dataclass(frozen=True)
class ToolChoice:
    """A choice between words on a tool's options bar."""

    key: str
    choices: tuple[str, ...]
    value: str


class BundleTools:
    """Holds the drawing tools' options and their drag, and carries out what they do.

    The window sends the pointer here while a tool of `DRAWING_TOOLS` is chosen, in a
    2D view only, and draws `overlay()` over the views. The tools change the phantom
    through its editor, and a drag is one step of its undo history, which Escape takes
    back before it ends.

    Attributes:
        reach: How far along a bundle Bend, Width and Twist act, mm, by tool.
        flat: Whether Width changes only the width seen in the view.
        opens: Where a new fan begins to open, as a share of the way along.
        active: The bundle last worked on, which the Fan's options follow.
        label: A few words on what the drag makes; None for nothing to say.
    """

    def __init__(self, viewer: Viewer) -> None:
        """Creates the tools of a viewer, with nothing being dragged."""
        self.viewer = viewer
        self.reach: dict[Tool, float] = {
            Tool.BEND: 40.0,
            Tool.WIDTH: 25.0,
            Tool.TWIST: 40.0,
        }
        self.flat = False
        self.opens = 0.5
        self.active: ItemId | None = None
        self.label: str | None = None
        self.pointer: Pointer | None = None
        self.hover_hit: Hit | None = None
        self._cache: dict[ItemId, Shape] = {}
        self._listeners: list[Callable[[], None]] = []
        self._step_open = False  # a drag holds a step of the undo history open
        self._tools: dict[Tool, _Tool] = {
            Tool.DRAW_BUNDLE: _Pen(self),
            Tool.BEND: _Bend(self),
            Tool.WIDTH: _Width(self),
            Tool.TWIST: _Twist(self),
            Tool.FAN: _Fan(self),
            Tool.ERASE: _Erase(self),
        }

    # Following

    def subscribe(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Calls listener when the overlay or options change; returns its undoing."""
        self._listeners.append(listener)

        def undo() -> None:
            """Stops calling the listener, once."""
            if listener in self._listeners:
                self._listeners.remove(listener)

        return undo

    def changed(self) -> None:
        """Tells the listeners that the overlay or the options changed."""
        for listener in self._listeners:
            listener()

    # The pointer

    @property
    def current(self) -> _Tool | None:
        """The tool chosen, if it is a drawing tool."""
        return self._tools.get(self.viewer.scene.view.tool)

    @property
    def busy(self) -> bool:
        """Whether a tool is being dragged."""
        tool = self.current
        return tool is not None and tool.dragging

    def press(self, pointer: Pointer) -> bool:
        """Sends a press to the tool.

        Returns:
            Whether the tool takes it. It leaves a press with Ctrl or Cmd to the view,
            which pans, unless it lands on a control point.
        """
        tool = self.current
        if tool is None:
            return False
        self.pointer = pointer
        self._end_step()
        taken = tool.press(pointer)
        if tool.dragging:
            self.viewer._phantom.begin_step()
            self._step_open = True
        self.changed()
        return taken

    def _end_step(self) -> None:
        """Lets go of the step of the undo history that a drag holds open."""
        if self._step_open:
            self._step_open = False
            self.viewer._phantom.end_step()

    def move(self, pointer: Pointer) -> None:
        """Sends a drag to the tool."""
        tool = self.current
        if tool is not None:
            self.pointer = pointer
            tool.move(pointer)
            self.changed()

    def release(self, pointer: Pointer) -> None:
        """Ends the drag.

        Raises:
            ValueError: If what it would make is refused, such as a point outside the
                phantom's grid.
        """
        tool = self.current
        if tool is None:
            return
        self.pointer = pointer
        try:
            tool.release(pointer)
        finally:
            self._end_step()
            self.label = None
            self._hover(pointer)
            self.changed()

    def click_here(self, pointer: Pointer) -> None:
        """Places a point with the Pen right here, even over a bundle.

        Raises:
            ValueError: If the point is outside the phantom's grid.
        """
        tool = self.current
        if isinstance(tool, _Pen):
            with self.viewer._phantom.one_step():
                tool.click_here(pointer)
            self.changed()

    def scrolled(self, plane: Plane, by: float) -> None:
        """Tells the tool that the slice moved by this many mm during a drag."""
        tool = self.current
        if tool is not None and by:
            tool.scrolled(plane, by)
            self.changed()

    def hover(self, pointer: Pointer) -> None:
        """Follows the pointer while no button is down."""
        if self.current is None:
            return
        self._hover(pointer)
        self.changed()

    def _hover(self, pointer: Pointer) -> None:
        """Finds what lies under the pointer, for the overlay."""
        self.pointer = pointer
        self.hover_hit = self.pick(pointer.plane, pointer.x, pointer.y)
        tool = self.current
        if tool is not None:
            tool.hover(pointer)

    def leave(self) -> None:
        """Forgets the pointer, which left the views."""
        if self.pointer is None and self.hover_hit is None:
            return
        self.pointer = self.hover_hit = None
        self.label = None
        tool = self.current
        if tool is not None:
            tool.leave()
        self.changed()

    def cancel_drag(self) -> bool:
        """Takes back the drag going on; returns whether there was one."""
        tool = self.current
        if tool is None or not tool.dragging:
            return False
        tool.cancel()
        self._end_step()
        self.label = None
        self.changed()
        return True

    def overlay(self) -> Overlay:
        """Returns what the tool chosen draws over the views."""
        out = Overlay()
        tool = self.current
        if tool is not None:
            tool.overlay(out)
        return out

    @property
    def hint(self) -> str:
        """A sentence on how to use the tool chosen; empty for no drawing tool."""
        tool = self.current
        return tool.hint() if tool is not None else ""

    # Options

    def options(self) -> tuple[ToolSlider | ToolChoice, ...]:
        """Returns the options of the tool chosen."""
        tool = self.current
        return tool.options() if tool is not None else ()

    def set_option(self, key: str, value: float | str) -> None:
        """Changes an option of the tool chosen, by its key in `options`."""
        tool = self.current
        if tool is not None:
            tool.set_option(key, value)
            self.changed()

    def step(self, direction: int) -> None:
        """Changes the Pen's radius or the reach by a step, as [ and ] do.

        Args:
            direction: 1 for larger, -1 for smaller.
        """
        tool = self.current
        if tool is not None:
            tool.step(direction)
            self.changed()

    # Finding bundles

    @property
    def state(self) -> PhantomState | None:
        """The phantom; None if none is open."""
        return self.viewer.scene.phantom

    def placed(self) -> list[Placed]:
        """Returns the drawn bundles that show, the one being drawn among them."""
        state = self.state
        if state is None:
            return []
        draft = state.draft
        found = []
        seen = set()
        for bundle in state.bundles:
            if draft is not None and draft.item_id == bundle.item_id:
                continue
            if bundle.geometry is not None:
                stem = bundle.branch.stem_points if bundle.branch else 0
                found.append(self._measured(bundle.item_id, bundle.geometry, stem))
                seen.add(bundle.item_id)
        if draft is not None and draft.item_id is not None and drawable(draft.points):
            stem = draft.branch.stem_points if draft.branch else 0
            geometry = replace(draft.geometry, control_points=draft.points)
            found.append(self._measured(draft.item_id, geometry, stem, True))
            seen.add(draft.item_id)
        self._cache = {k: v for k, v in self._cache.items() if k in seen}
        return [p for p in found if p is not None]

    def _measured(
        self,
        item_id: ItemId,
        geometry: GeometrySettings,
        stem_points: int,
        is_draft: bool = False,
    ) -> Placed | None:
        """Returns a bundle measured, again only if it changed; None if it is hidden."""
        found = [i for i in self.viewer.scene.items if i.id == item_id]
        if not found or not found[0].options.visible:
            return None  # its item is going, or hidden
        o = found[0].options
        known = self._cache.get(item_id)
        if (
            known is None
            or known.stem_points != stem_points
            or known.geometry != geometry
        ):
            try:
                known = shape_of(geometry, stem_points)
            except ValueError:
                return None
            self._cache[item_id] = known
        slab: float | None = SLAB_THICKNESS
        if isinstance(o, TractogramOptions):
            slab = o.slab_thickness if o.slab else None
        return Placed(item_id, known, is_draft, slab)

    def slice_at(self, plane: Plane) -> float:
        """Returns the depth of a view's slice, mm."""
        return self.viewer.scene.view.cursor[plane.axis]

    def far_from_slice(self, plane: Plane, placed: Placed) -> np.ndarray:
        """Returns whether each place along a bundle lies too far to show in a view."""
        radii = placed.shape.sections[:, :2].max(axis=1)
        if placed.slab is None:
            reach = np.maximum(FAR_WITHOUT_SLAB, radii)
        else:
            reach = placed.slab / 2 + radii
        depth = placed.shape.centre[:, plane.axis] - self.slice_at(plane)
        far: np.ndarray = np.abs(depth) > reach
        return far

    def pick(
        self, plane: Plane, x: float, y: float, *, any_depth: bool = False
    ) -> Hit | None:
        """Returns the place along a bundle under the pointer.

        What does not show cannot be picked, unless any_depth asks for it. A view
        looking along a bundle sees many places at one spot, so the slice decides
        among them.
        """
        best: tuple[float, Hit] | None = None
        at = self.slice_at(plane)
        for placed in self.placed():
            shape = placed.shape
            screen = plane.project(shape.centre)
            dist = np.hypot(screen[:, 0] - x, screen[:, 1] - y)
            width = shape.widths(plane.axis) / plane.mm_per_pixel
            reach = np.maximum(width, 8.0) + 4.0
            depth = np.abs(shape.centre[:, plane.axis] - at)
            along = np.abs(shape.tangents[:, plane.axis]) > 0.8
            score = dist / reach + depth * np.where(along, 0.05, 1e-4)
            far = self.far_from_slice(plane, placed)
            usable = dist <= reach
            usable[: shape.first_own] = False
            if not any_depth and placed.slab is not None:
                usable &= ~far
            if not usable.any():
                continue
            k = int(np.argmin(np.where(usable, score, np.inf)))
            if best is None or score[k] < best[0]:
                hit = Hit(placed, float(shape.u[k]), k, bool(far[k]))
                best = (float(score[k]), hit)
        return best[1] if best is not None else None

    def tip_at(self, plane: Plane, x: float, y: float) -> Tip | None:
        """Returns a bundle's tip near the pointer; a branch has only its far end."""
        best = None
        for tip in self._tips(plane, x, y):
            if tip.dist <= TIP_PIXELS and (best is None or tip.dist < best.dist):
                best = tip
        return best

    def nearest_tip(self, plane: Plane, x: float, y: float) -> Tip | None:
        """Returns the tip nearest the pointer, however far it is."""
        tips = list(self._tips(plane, x, y))
        return min(tips, key=lambda t: t.dist) if tips else None

    def _tips(self, plane: Plane, x: float, y: float) -> list[Tip]:
        """Returns the bundles' tips, with their distances from the pointer."""
        tips = []
        for placed in self.placed():
            shape = placed.shape
            ends = [(True, len(shape.u) - 1)]
            if shape.stem_points <= 1:
                ends.append((False, 0))
            for at_end, k in ends:
                px, py = plane.project(shape.centre[k][None])[0]
                dist = math.hypot(float(px) - x, float(py) - y)
                tips.append(Tip(placed, at_end, k, dist))
        return tips

    def worked_on(self, item_id: ItemId) -> None:
        """Marks a bundle as the one last worked on, and selects it for the panel."""
        self.active = item_id
        if self.viewer.scene.view.selected != item_id:
            self.viewer.select(item_id)


class _Tool:
    """One drawing tool. A tool that takes a press holds its drag until the release."""

    def __init__(self, tools: BundleTools) -> None:
        """Creates the tool, with nothing being dragged."""
        self.tools = tools

    @property
    def viewer(self) -> Viewer:
        """The viewer."""
        return self.tools.viewer

    @property
    def editor(self) -> PhantomEditor:
        """The editor of the viewer's phantom."""
        return self.tools.viewer._phantom

    @property
    def dragging(self) -> bool:
        """Whether a drag is going on."""
        return False

    def hint(self) -> str:
        """Returns a sentence on how to use the tool."""
        return ""

    def press(self, pointer: Pointer) -> bool:
        """Starts a drag; returns whether the tool takes the press."""
        return False

    def move(self, pointer: Pointer) -> None:
        """Carries on the drag."""

    def release(self, pointer: Pointer) -> None:
        """Ends the drag."""

    def cancel(self) -> None:
        """Takes back the drag."""

    def scrolled(self, plane: Plane, by: float) -> None:
        """Follows the slice, which moved by this many mm during a drag."""

    def hover(self, pointer: Pointer) -> None:
        """Follows the pointer while no button is down."""

    def leave(self) -> None:
        """Forgets the pointer, which left the views."""

    def overlay(self, out: Overlay) -> None:
        """Adds what the tool draws over the views."""

    def options(self) -> tuple[ToolSlider | ToolChoice, ...]:
        """Returns the tool's options."""
        return ()

    def set_option(self, key: str, value: float | str) -> None:
        """Changes an option."""

    def step(self, direction: int) -> None:
        """Changes the tool's size by a step."""


@dataclass(eq=False)
class _Grab:
    """A drag on a place along a bundle.

    Attributes:
        item_id: The bundle.
        shape: The bundle as it was at the press.
        before: The phantom's bundles at the press, by item, for taking it back.
        plane: The view it is dragged in.
        u: The position grabbed.
        depth: The depth the drag works at, along the view's axis.
        start: The point pressed at that depth.
        made: The geometry the drag last made.
        last: The pointer's last place.
    """

    item_id: ItemId
    shape: Shape
    before: dict[ItemId, BundleSettings]
    plane: Plane
    u: float
    depth: float
    start: np.ndarray
    made: GeometrySettings | None = None
    last: Pointer | None = None


class _PlaceTool(_Tool):
    """A tool that grabs a place along a bundle and shows what it would change."""

    tool: Tool

    def __init__(self, tools: BundleTools) -> None:
        """Creates the tool, with nothing being dragged."""
        super().__init__(tools)
        self.grab: _Grab | None = None

    @property
    def dragging(self) -> bool:
        """Whether a place is being dragged."""
        return self.grab is not None

    def _reach(self) -> float:
        """Returns how far along the tool acts, mm."""
        return self.tools.reach[self.tool]

    def _grab_at(self, pointer: Pointer, hit: Hit) -> _Grab:
        """Returns the drag of a place, with the bundle marked as worked on."""
        state = self.tools.state
        assert state is not None
        shape = hit.placed.shape
        depth = float(shape.frame(hit.u).centre[pointer.plane.axis])
        self.tools.worked_on(hit.placed.item_id)
        return _Grab(
            hit.placed.item_id,
            shape,
            {b.item_id: b for b in state.bundles},
            pointer.plane,
            hit.u,
            depth,
            pointer.plane.world(pointer.x, pointer.y, depth),
        )

    def _apply(self, geometry: GeometrySettings, *, final: bool = False) -> None:
        """Gives the bundle dragged a new shape, its branches following."""
        grab = self.grab
        state = self.tools.state
        if grab is None or state is None:
            return
        grab.made = geometry
        changed = {grab.item_id: with_geometry(grab.before[grab.item_id], geometry)}
        changed = with_family(list(grab.before.values()), changed)
        self.editor.set_shapes(changed, final=final)

    def cancel(self) -> None:
        """Puts every bundle back as it was at the press."""
        grab, self.grab = self.grab, None
        if grab is not None:
            self.editor.set_shapes(grab.before, final=True)

    def _finish(self, geometry: GeometrySettings | None) -> None:
        """Ends the drag with this shape; None if nothing changed."""
        if geometry is not None and self.grab is not None:
            self._apply(settled(geometry, self.grab.shape.stem_points), final=True)
        self.grab = None

    def _place(self) -> tuple[Shape, float] | None:
        """Returns the bundle and position the tool would change now."""
        if self.grab is not None:
            return self.grab.shape, self.grab.u
        hit = self.tools.hover_hit
        # What was under the pointer may have changed since it last moved.
        if hit is None or not any(
            p.shape is hit.placed.shape for p in self.tools.placed()
        ):
            return None
        return hit.placed.shape, hit.u

    def _add_reach(self, out: Overlay) -> None:
        """Marks the stretch of the bundle the tool would change."""
        place = self._place()
        if place is None:
            return
        shape, u = place
        span = self._reach() / shape.length
        keep = np.abs(shape.u - u) <= span
        keep[: shape.first_own] = False
        if keep.sum() >= 2:
            out.reaches.append(shape.centre[keep])

    def _add_place(self, out: Overlay) -> None:
        """Marks the section the tool would change, and the rows of the bundle."""
        place = self._place()
        if place is None:
            return
        shape, u = place
        out.outlines.extend(section_outline(shape.frame(u)))
        ticks = np.array(
            [shape.centre[shape.k_at(r[0])] for r in shape.rows if r[0] >= shape.stem_u]
        )
        if len(ticks):
            out.ticks.append(ticks)

    def options(self) -> tuple[ToolSlider | ToolChoice, ...]:
        """Returns the reach."""
        return (self._reach_slider(),)

    def _reach_slider(self) -> ToolSlider:
        """Returns the slider of the reach."""
        reach = self._reach()
        return ToolSlider(
            "reach",
            "Reach",
            SMALLEST_REACH,
            LARGEST_REACH,
            SMALLEST_REACH,
            reach,
            f"{reach:g} mm",
        )

    def set_option(self, key: str, value: float | str) -> None:
        """Changes the reach, also during a drag."""
        if key == "reach":
            assert isinstance(value, (int, float))
            top = WHOLE_BUNDLE if self.tool == Tool.BEND else LARGEST_REACH
            self.tools.reach[self.tool] = min(top, max(SMALLEST_REACH, float(value)))
            self._redo()

    def step(self, direction: int) -> None:
        """Changes the reach by a step, as a brush's size."""
        reach = self._reach()
        size = STEP_BIG if reach >= REACH_BIG else REACH_STEP
        self.set_option("reach", reach + direction * size)

    def _redo(self) -> None:
        """Works the drag out again, as an option changed during it."""
        if self.grab is not None and self.grab.last is not None:
            self.move(self.grab.last)


def section_outline(frame: Frame) -> list[np.ndarray]:
    """Returns a section's ring and its first radius, as polylines in world mm."""
    t = np.linspace(0.0, 2 * np.pi, 41)[:, None]
    ring = (
        frame.centre
        + np.cos(t) * frame.r1 * frame.first
        + np.sin(t) * frame.r2 * frame.second
    )
    across = frame.centre + np.array([[-1.0], [1.0]]) * frame.r1 * frame.first
    return [ring, across]


class _Bend(_PlaceTool):
    """Moves a place along a bundle; nearby parts and branches follow."""

    tool = Tool.BEND

    def hint(self) -> str:
        """Returns a sentence on how to use the tool."""
        return (
            "Bend: drag any part of a bundle, and nearby parts follow. Reach, or [ and "
            "], says how far along. At full reach the whole bundle moves."
        )

    def press(self, pointer: Pointer) -> bool:
        """Grabs the place under the pointer."""
        if pointer.ctrl:
            return False
        hit = self.tools.pick(pointer.plane, pointer.x, pointer.y)
        if hit is not None:
            self.grab = self._grab_at(pointer, hit)
        return True

    def move(self, pointer: Pointer) -> None:
        """Moves the place grabbed in the view's plane."""
        grab = self.grab
        if grab is None:
            return
        grab.last = pointer
        delta = pointer.plane.world(pointer.x, pointer.y, grab.depth) - grab.start
        self._bend(grab, delta, final=False)

    def _bend(self, grab: _Grab, delta: np.ndarray, *, final: bool) -> None:
        """Moves the bundle and its own branches, then their stems follow."""
        reach = self._reach()
        whole = reach >= WHOLE_BUNDLE
        geometry = bent(grab.shape, grab.u, delta, reach, whole)
        if final:
            geometry = settled(geometry, grab.shape.stem_points)
        grab.made = geometry
        changed = {grab.item_id: with_geometry(grab.before[grab.item_id], geometry)}
        for bundle in grab.before.values():
            link = bundle.branch
            if link is None or link.parent != grab.item_id or bundle.geometry is None:
                continue
            moved = bent_branch(
                shape_of(bundle.geometry, link.stem_points),
                link.at,
                grab.shape.length,
                grab.u,
                delta,
                reach,
                whole,
            )
            changed[bundle.item_id] = with_geometry(bundle, moved)
        self.editor.set_shapes(
            with_family(list(grab.before.values()), changed), final=final
        )

    def release(self, pointer: Pointer) -> None:
        """Ends the drag."""
        grab = self.grab
        if grab is not None and grab.made is not None:
            delta = pointer.plane.world(pointer.x, pointer.y, grab.depth) - grab.start
            self._bend(grab, delta, final=True)
        self.grab = None

    def _place(self) -> tuple[Shape, float] | None:
        """Returns the bundle as dragged so far, so that the marks move with it."""
        grab = self.grab
        if grab is None or grab.made is None:
            return super()._place()
        return shape_of(grab.made, grab.shape.stem_points), grab.u

    def _reach_slider(self) -> ToolSlider:
        """Returns the slider of the reach, whose top moves the whole bundle."""
        reach = self._reach()
        text = "whole bundle" if reach >= WHOLE_BUNDLE else f"{reach:g} mm"
        return ToolSlider(
            "reach", "Reach", SMALLEST_REACH, WHOLE_BUNDLE, SMALLEST_REACH, reach, text
        )

    def overlay(self, out: Overlay) -> None:
        """Marks the stretch that moves."""
        self._add_reach(out)


@dataclass(eq=False)
class _WidthGrab(_Grab):
    """A Width drag, with the section at the press and which way is outwards.

    Attributes:
        frame: The section at the press.
        width: How wide the bundle looks there, mm.
        across: The way across it in the view.
        outwards: 1 or -1, settled at the press, so that going past the centre keeps
            narrowing.
        offset: How far out the press was, mm.
    """

    frame: Frame | None = None
    width: float = 0.0
    across: np.ndarray = field(default_factory=lambda: np.zeros(3))
    outwards: float = 1.0
    offset: float = 0.0


class _Width(_PlaceTool):
    """Widens and narrows a bundle at a place, as a soft brush."""

    tool = Tool.WIDTH

    def hint(self) -> str:
        """Returns a sentence on how to use the tool."""
        return (
            "Width: drag on a bundle, out to widen and in to narrow. Reach, or [ and "
            "], sets how far along it acts. Flat, or Shift, changes only the width you "
            "see in this view."
        )

    def options(self) -> tuple[ToolSlider | ToolChoice, ...]:
        """Returns Round or Flat, and the reach."""
        mode = "Flat" if self.tools.flat else "Round"
        return (ToolChoice("mode", ("Round", "Flat"), mode), self._reach_slider())

    def set_option(self, key: str, value: float | str) -> None:
        """Changes Round or Flat, or the reach."""
        if key == "mode":
            self.tools.flat = value == "Flat"
        else:
            super().set_option(key, value)

    def press(self, pointer: Pointer) -> bool:
        """Grabs the place under the pointer, and settles which way is outwards."""
        if pointer.ctrl:
            return False
        hit = self.tools.pick(pointer.plane, pointer.x, pointer.y)
        if hit is None:
            return True
        base = self._grab_at(pointer, hit)
        frame = base.shape.frame(hit.u)
        width, across = visible_width(
            pointer.plane.axis, frame, base.start - frame.centre
        )
        side = float((base.start - frame.centre) @ across)
        outwards = 1.0 if side >= 0 else -1.0
        self.grab = _WidthGrab(
            base.item_id,
            base.shape,
            base.before,
            base.plane,
            base.u,
            base.depth,
            base.start,
            frame=frame,
            width=width,
            across=across,
            outwards=outwards,
            offset=side * outwards,
        )
        return True

    def move(self, pointer: Pointer) -> None:
        """Changes the section at the place, and those nearby by their shares."""
        grab = self.grab
        if not isinstance(grab, _WidthGrab):
            return
        grab.last = pointer
        self._apply(self._made(grab, pointer))

    def _made(self, grab: _WidthGrab, pointer: Pointer) -> GeometrySettings:
        """Returns the bundle with the width the pointer asks for."""
        shape, frame = grab.shape, grab.frame
        assert frame is not None
        world = pointer.plane.world(pointer.x, pointer.y, grab.depth)
        reached = float((world - frame.centre) @ grab.across) * grab.outwards
        width = max(MIN_WIDTH, grab.width + reached - grab.offset)
        low = shape.stem_u
        rows: Rows = [list(r) for r in shape.rows]
        u = _snapped(shape, grab.u)
        reach_u = min(0.9 * (1 - low), self._reach() / shape.length)
        ring, others = brush_rows(rows, u, reach_u, low)
        was = list(ring)
        geometry = shape.geometry
        landing, at_end = _landing_at(geometry, ring[0])
        tangent = frame.tangent
        seen = max(grab.width, 0.01)
        if self.tools.flat == pointer.shift:  # Round, unless Shift turns Flat round
            ring[1] = followed_radius(ring[1], seen, width)
            ring[2] = followed_radius(ring[2], seen, width)
            if landing is not None:
                factors = (ring[1] / was[1], ring[2] / was[2])
                geometry = with_landing(
                    geometry, at_end, _scaled(landing, tangent, *factors)
                )
        else:
            geometry = self._flat(ring, grab, width, geometry, landing, at_end)
        for other, share in others:
            other[1] = followed_radius(other[1], was[1], ring[1], share)
            other[2] = followed_radius(other[2], was[2], ring[2], share)
            # A round row can share a new direction without any twist showing.
            round_row = abs(other[1] - other[2]) < 0.05 * max(other[1], other[2])
            if ring[3] != was[3] and round_row:
                other[3] = ring[3]
        self.tools.label = f"{width:.1f} mm"
        return with_rows(geometry, rows)

    def _flat(
        self,
        ring: list[float],
        grab: _WidthGrab,
        width: float,
        geometry: GeometrySettings,
        landing: LandingSettings | None,
        at_end: bool,
    ) -> GeometrySettings:
        """Changes only the width seen in the view, and returns the bundle."""
        frame = grab.frame
        assert frame is not None
        across = grab.across
        c1, c2 = float(across @ frame.first), float(across @ frame.second)
        if abs(ring[1] - ring[2]) < 0.05 * max(ring[1], ring[2]) and landing is None:
            # Flat on a round section picks a direction.
            ring[3] = math.degrees(
                math.atan2(float(across @ frame.binormal), float(across @ frame.normal))
            )
            ring[1] = width
            return geometry
        if abs(c1) >= abs(c2):
            value = math.sqrt(max(width**2 - (ring[2] * c2) ** 2, 0.04))
            value /= max(abs(c1), 0.05)
            factors = (value / ring[1], 1.0)
            ring[1] = value
        else:
            value = math.sqrt(max(width**2 - (ring[1] * c1) ** 2, 0.04))
            value /= max(abs(c2), 0.05)
            factors = (1.0, value / ring[2])
            ring[2] = value
        if landing is None:
            return geometry
        return with_landing(
            geometry, at_end, _scaled(landing, frame.tangent, factors[0], factors[1])
        )

    def release(self, pointer: Pointer) -> None:
        """Ends the drag, and takes away the rows the stroke left with nothing to do."""
        grab = self.grab
        made = None
        if grab is not None and grab.made is not None:
            assert grab.made.rings is not None
            rows = prune_rows([list(r) for r in grab.made.rings], grab.shape.stem_u)
            made = with_rows(grab.made, rows)
        self._finish(made)

    def overlay(self, out: Overlay) -> None:
        """Marks the stretch that changes and the section at the place."""
        self._add_reach(out)
        self._add_place(out)


def _snapped(shape: Shape, u: float) -> float:
    """Returns a position, at the end of the bundle's own part if near it."""
    low = shape.stem_u
    snap = min(0.04, 6 / (shape.length * max(1 - low, 1e-6)))
    if u > 1 - snap:
        return 1.0
    if u < low + snap:
        # A branch's first section is its parent's, so it cannot change.
        return 0.0 if low == 0 else low + snap
    return u


def _landing_at(
    geometry: GeometrySettings, u: float
) -> tuple[LandingSettings | None, bool]:
    """Returns the landing at a row's position, if an end, and whether the last."""
    if u == 1:
        return geometry.landing_end, True
    if u == 0:
        return geometry.landing_start, False
    return None, True


def _scaled(
    landing: LandingSettings, tangent: np.ndarray, first: float, second: float
) -> LandingSettings:
    """Returns a landing with its lines stretched across the path by these factors."""
    read = read_line(landing.line, tangent)
    line = scaled_line(landing.line, read.direction, read.middle, first)
    thickness = landing.thickness_line
    if thickness is not None:
        other = read_line(thickness, tangent)
        thickness = scaled_line(thickness, other.direction, other.middle, second)
    return replace(landing, line=line, thickness_line=thickness)


@dataclass(eq=False)
class _TwistGrab(_Grab):
    """A Twist drag, with the section at the press and the circling so far.

    Attributes:
        frame: The section at the press.
        seen: The angle of the pointer around the bundle when last seen, looking along
            it; None until then.
        total: How far the pointer has circled, in degrees.
    """

    frame: Frame | None = None
    seen: float | None = None
    total: float = 0.0


class _Twist(_PlaceTool):
    """Rolls a bundle at a place; it turns back within the reach."""

    tool = Tool.TWIST

    def hint(self) -> str:
        """Returns a sentence on how to use the tool."""
        return (
            "Twist: drag across a bundle to roll it at that place, and it turns back "
            "within Reach, or [ and ]. Looking along a bundle, circle it. Streamlines "
            "are coloured by their place while this tool is on, so a twist shows as "
            "colours changing sides."
        )

    def press(self, pointer: Pointer) -> bool:
        """Grabs the place under the pointer."""
        if pointer.ctrl:
            return False
        hit = self.tools.pick(pointer.plane, pointer.x, pointer.y)
        if hit is None:
            return True
        base = self._grab_at(pointer, hit)
        self.grab = _TwistGrab(
            base.item_id,
            base.shape,
            base.before,
            base.plane,
            base.u,
            base.depth,
            base.start,
            frame=base.shape.frame(hit.u),
        )
        return True

    def move(self, pointer: Pointer) -> None:
        """Turns the section at the place, by pixels dragged or the angle circled."""
        grab = self.grab
        if not isinstance(grab, _TwistGrab):
            return
        grab.last = pointer
        self._apply(self._made(grab, pointer, self._degrees(grab, pointer)))

    def _degrees(self, grab: _TwistGrab, pointer: Pointer) -> float:
        """Returns how far the pointer asks the section to turn, in degrees."""
        frame = grab.frame
        assert frame is not None
        world = pointer.plane.world(pointer.x, pointer.y, grab.depth)
        view = np.eye(3)[pointer.plane.axis]
        axis = np.cross(frame.tangent, view)
        size = float(np.linalg.norm(axis))
        if size > 0.6:
            pixels = float((world - grab.start) @ (axis / size))
            return pixels / pointer.plane.mm_per_pixel
        # Looking along the bundle, circling it turns it.
        q = world - frame.centre
        angle = math.degrees(
            math.atan2(float(q @ frame.binormal), float(q @ frame.normal))
        )
        if grab.seen is not None:
            step = (angle - grab.seen + 180) % 360 - 180
            grab.total += step
        grab.seen = angle
        return grab.total

    def _made(
        self, grab: _TwistGrab, pointer: Pointer, degrees: float
    ) -> GeometrySettings:
        """Returns the bundle turned at the place."""
        shape = grab.shape
        low = shape.stem_u
        rows: Rows = [list(r) for r in shape.rows]
        u = _snapped(shape, grab.u)
        reach_u = min(0.9 * (1 - low), self._reach() / shape.length)
        ring = ensure_row(rows, u, reach_u, low)
        ring[3] += degrees
        geometry = shape.geometry
        # At a landed end the drawn line holds the direction, so the line turns.
        landing, at_end = _landing_at(geometry, ring[0])
        if landing is not None:
            tangent = shape.tangents[-1 if at_end else 0]
            middle = read_line(landing.line, tangent).middle
            thickness = landing.thickness_line
            turned = replace(
                landing,
                line=rotated_line(landing.line, middle, tangent, degrees),
                thickness_line=None
                if thickness is None
                else rotated_line(thickness, middle, tangent, degrees),
            )
            geometry = with_landing(geometry, at_end, turned)
        self.tools.label = f"{degrees:+.0f} deg"
        return with_rows(geometry, rows)

    def release(self, pointer: Pointer) -> None:
        """Ends the drag, and takes away the rows the stroke left with nothing to do."""
        grab = self.grab
        made = None
        if grab is not None and grab.made is not None:
            assert grab.made.rings is not None
            rows = prune_rows([list(r) for r in grab.made.rings], grab.shape.stem_u)
            made = with_rows(grab.made, rows)
        self._finish(made)

    def overlay(self, out: Overlay) -> None:
        """Marks the section at the place."""
        self._add_place(out)


@dataclass(eq=False)
class _Line:
    """A line or stroke drawn on the screen, kept in pixels and at a depth."""

    plane: Plane
    screen: list[tuple[float, float]]
    world: list[Vec3]
    depth: float

    def add(self, pointer: Pointer) -> bool:
        """Adds a point a pixel or more from the last; returns whether it did."""
        last = self.screen[-1]
        if math.hypot(pointer.x - last[0], pointer.y - last[1]) < 1:
            return False
        self.screen.append((pointer.x, pointer.y))
        self.world.append(vec3(self.plane.world(pointer.x, pointer.y, self.depth)))
        return True

    def move_depth(self, by: float) -> None:
        """Moves the stroke in depth at the pointer, and carries on from there."""
        self.depth += by
        x, y = self.screen[-1]
        self.screen.append((x, y))
        self.world.append(vec3(self.plane.world(x, y, self.depth)))

    @property
    def length(self) -> float:
        """The stroke's length, mm."""
        return stroke_length(self.world)

    @property
    def travel(self) -> float:
        """How far from its start the stroke has gone on the screen, in pixels."""
        x0, y0 = self.screen[0]
        return max(math.hypot(x - x0, y - y0) for x, y in self.screen)


def _start_line(plane: Plane, pointer: Pointer, depth: float) -> _Line:
    """Returns a line that starts at the pointer, at a depth."""
    point = vec3(plane.world(pointer.x, pointer.y, depth))
    return _Line(plane, [(pointer.x, pointer.y)], [point], depth)


@dataclass(eq=False)
class _FanDrag:
    """A line being drawn across a bundle's end.

    Attributes:
        tip: The end, settled at the press.
        line: The line so far.
        before: The phantom's bundles at the press, by item.
        made: What the line last made.
    """

    tip: Tip
    line: _Line
    before: dict[ItemId, BundleSettings]
    made: GeometrySettings | None = None
    cut: Cut | None = None


class _Fan(_Tool):
    """Draws the line a bundle's end lands on."""

    def __init__(self, tools: BundleTools) -> None:
        """Creates the tool, with nothing being drawn."""
        super().__init__(tools)
        self.drag: _FanDrag | None = None
        self.nearest: Tip | None = None

    @property
    def dragging(self) -> bool:
        """Whether a line is being drawn."""
        return self.drag is not None

    def hint(self) -> str:
        """Returns a sentence on how to use the tool."""
        return (
            "Fan: press near the circled end and draw a line across it, and the "
            "streamlines end on that line. A new line replaces the old one. A line at "
            "right angles, from another view, sets the thickness."
        )

    def _target(self) -> ItemId | None:
        """Returns the bundle the options follow, the last worked on or else drawn."""
        placed = self.tools.placed()
        ids = [p.item_id for p in placed]
        if self.tools.active in ids:
            return self.tools.active
        return ids[-1] if ids else None

    def _target_end(self) -> tuple[Placed, bool] | None:
        """Returns that bundle and the end whose fan the options show, if it has one."""
        target = self._target()
        for placed in self.tools.placed():
            if placed.item_id == target:
                geometry = placed.shape.geometry
                if geometry.landing_end is not None:
                    return placed, True
                if geometry.landing_start is not None:
                    return placed, False
        return None

    def options(self) -> tuple[ToolSlider | ToolChoice, ...]:
        """Returns where the fan begins to open."""
        value = self.tools.opens
        found = self._target_end()
        if found is not None:
            placed, at_end = found
            shape = placed.shape
            landing = (
                shape.geometry.landing_end if at_end else shape.geometry.landing_start
            )
            assert landing is not None
            own = (landing.opens_from - shape.stem_u) / max(1 - shape.stem_u, 1e-6)
            value = own if at_end else 1 - own
        low, high = OPENS_RANGE
        return (
            ToolSlider(
                "opens",
                "Opens from",
                low,
                high,
                0.05,
                value,
                f"{round(value * 100)}% along",
            ),
        )

    def set_option(self, key: str, value: float | str) -> None:
        """Moves where the fan of the bundle worked on begins to open."""
        if key != "opens":
            return
        assert isinstance(value, (int, float))
        self.tools.opens = float(value)
        found = self._target_end()
        state = self.tools.state
        if found is None or state is None:
            return
        placed, at_end = found
        made = with_opening(placed.shape, at_end, float(value))
        if made is not None:
            bundle = state.bundle(placed.item_id)
            self.editor.set_shapes(
                {
                    placed.item_id: with_geometry(
                        bundle, settled(made, placed.shape.stem_points)
                    )
                }
            )

    def press(self, pointer: Pointer) -> bool:
        """Settles the end the line goes to, and takes the slice to it."""
        if pointer.ctrl:
            return False
        state = self.tools.state
        tip = self.tools.nearest_tip(pointer.plane, pointer.x, pointer.y)
        if tip is None or state is None:
            self.tools.label = "Draw a bundle first."
            return True
        plane = pointer.plane
        depth = float(tip.placed.shape.centre[tip.k, plane.axis])
        cursor = list(self.viewer.scene.view.cursor)
        cursor[plane.axis] = depth
        self.viewer.move_cursor((cursor[0], cursor[1], cursor[2]))
        self.tools.worked_on(tip.placed.item_id)
        self.drag = _FanDrag(
            tip,
            _start_line(plane, pointer, depth),
            {b.item_id: b for b in state.bundles},
        )
        return True

    def move(self, pointer: Pointer) -> None:
        """Lands the end on the line so far, always from the bundle as it was."""
        drag = self.drag
        if drag is None or not drag.line.add(pointer):
            return
        self._preview(drag, final=False)

    def _preview(self, drag: _FanDrag, *, final: bool) -> None:
        """Works out what the line makes, and gives the bundle that shape."""
        if drag.line.length < SHORTEST_LINE:
            return
        plane = drag.line.plane
        tolerance = max(0.15, 1.5 * plane.mm_per_pixel)
        result = landed(
            drag.tip.placed.shape,
            drag.tip.at_end,
            drag.line.world,
            tolerance,
            self.tools.opens,
        )
        if result is None:
            return
        drag.made, drag.cut = result.geometry, result.cut
        self.tools.label = (
            "Thickness of the fan" if result.second else "Streamlines land here"
        )
        item_id = drag.tip.placed.item_id
        bundle = drag.before[item_id]
        changed = {item_id: with_geometry(bundle, result.geometry)}
        self.editor.set_shapes(changed, final=final)

    def release(self, pointer: Pointer) -> None:
        """Ends the line; one too short leaves all as it was."""
        drag, self.drag = self.drag, None
        if drag is None:
            return
        if drag.line.length < SHORTEST_LINE or drag.made is None:
            self.editor.set_shapes(drag.before, final=True)
            return
        item_id = drag.tip.placed.item_id
        made = settled(drag.made, drag.tip.placed.shape.stem_points)
        changed = {item_id: with_geometry(drag.before[item_id], made)}
        doomed = _after_cut(drag.before, item_id, drag.cut, changed)
        self.editor.set_shapes(
            with_family(list(drag.before.values()), changed), final=True
        )
        for gone in doomed:
            self.viewer.remove(gone)

    def cancel(self) -> None:
        """Puts every bundle back as it was at the press."""
        drag, self.drag = self.drag, None
        if drag is not None:
            self.editor.set_shapes(drag.before, final=True)

    def hover(self, pointer: Pointer) -> None:
        """Marks the end nearest the pointer."""
        self.nearest = self.tools.nearest_tip(pointer.plane, pointer.x, pointer.y)

    def leave(self) -> None:
        """Forgets the end marked."""
        self.nearest = None

    def overlay(self, out: Overlay) -> None:
        """Draws the line, the circle at its end, and the fans' saved lines."""
        drag = self.drag
        marked = drag.tip if drag is not None else self.nearest
        if drag is not None and len(drag.line.world) >= 2:
            out.strokes.append((np.array(drag.line.world), OUTLINE_COLOUR))
        if marked is not None:
            out.circles.append(marked.placed.shape.centre[marked.k][None])
        for placed in self.tools.placed():
            _add_landing_lines(out, placed.shape)


def _add_landing_lines(out: Overlay, shape: Shape) -> None:
    """Adds a bundle's fans; a thickness line counts from the first line's middle."""
    geometry = shape.geometry
    for landing, k in ((geometry.landing_end, -1), (geometry.landing_start, 0)):
        if landing is None:
            continue
        line = np.array(landing.line, dtype=np.float64)
        out.dashed.append(line)
        if landing.thickness_line is not None:
            tangent = shape.tangents[k]
            middle = read_line(landing.line, tangent).middle
            by = middle - read_line(landing.thickness_line, tangent).middle
            out.dashed.append(np.array(landing.thickness_line, dtype=np.float64) + by)


def _after_cut(
    bundles: Mapping[ItemId, BundleSettings],
    item_id: ItemId,
    the_cut: Cut | None,
    changed: dict[ItemId, BundleSettings],
) -> list[ItemId]:
    """Moves a cut bundle's branches along, and returns those that went with the cut.

    Args:
        bundles: The phantom's bundles.
        item_id: The bundle cut.
        the_cut: The cut; None for no cut.
        changed: The bundles to change, where the moved branches are put.

    Returns:
        The items of the branches lost, and of what hangs on them.
    """
    if the_cut is None:
        return []
    doomed: list[ItemId] = []
    for bundle in bundles.values():
        link = bundle.branch
        if link is None or link.parent != item_id:
            continue
        at = the_cut.branch_at(link.at)
        if at is None:
            doomed.append(bundle.item_id)
        else:
            changed[bundle.item_id] = replace(bundle, branch=replace(link, at=at))
    return _with_descendants(bundles, doomed)


def _with_descendants(
    bundles: Mapping[ItemId, BundleSettings], items: Sequence[ItemId]
) -> list[ItemId]:
    """Returns the items, and every branch that hangs on them, parents first."""
    found = list(items)
    grew = True
    while grew:
        grew = False
        for bundle in bundles.values():
            link = bundle.branch
            if (
                link is not None
                and link.parent in found
                and bundle.item_id not in found
            ):
                found.append(bundle.item_id)
                grew = True
    return found


class _Erase(_Tool):
    """Removes a bundle by a click, and cuts one by a stroke across it."""

    def __init__(self, tools: BundleTools) -> None:
        """Creates the tool, with nothing being drawn."""
        super().__init__(tools)
        self.line: _Line | None = None
        self.pressed: Hit | None = None

    @property
    def dragging(self) -> bool:
        """Whether a stroke is being drawn."""
        return self.line is not None

    def hint(self) -> str:
        """Returns a sentence on how to use the tool."""
        return (
            "Erase: click a bundle to remove it, with its branches. Drag across a "
            "bundle to cut it there, and the shorter part goes. Only what shows in the "
            "view can be cut."
        )

    def press(self, pointer: Pointer) -> bool:
        """Notes the bundle under the pointer, and starts a stroke."""
        if pointer.ctrl:
            return False
        plane = pointer.plane
        self.pressed = self.tools.pick(plane, pointer.x, pointer.y)
        self.line = _start_line(plane, pointer, self.tools.slice_at(plane))
        return True

    def move(self, pointer: Pointer) -> None:
        """Carries on the stroke."""
        if self.line is not None:
            self.line.add(pointer)

    def release(self, pointer: Pointer) -> None:
        """Removes the bundle clicked, or cuts the first one the stroke crosses."""
        line, self.line = self.line, None
        state = self.tools.state
        if line is None or state is None:
            return
        bundles = {b.item_id: b for b in state.bundles}
        if line.length < SHORTEST_CUT:
            if self.pressed is not None:
                self._remove(bundles, self.pressed.placed.item_id)
            return
        screen = np.array(line.screen)
        for placed in self.tools.placed():
            shape = placed.shape
            first = shape.first_own
            crossing = crossing_of(screen, line.plane.project(shape.centre[first:]))
            if crossing is None:
                continue
            u = (first + crossing[0]) / (len(shape.u) - 1)
            k = shape.k_at(u)
            if (
                placed.slab is not None
                and self.tools.far_from_slice(line.plane, placed)[k]
            ):
                continue
            self._cut(bundles, placed, u)
            return

    def _remove(
        self, bundles: Mapping[ItemId, BundleSettings], item_id: ItemId
    ) -> None:
        """Removes a bundle and what hangs on it."""
        for gone in reversed(_with_descendants(bundles, [item_id])):
            self.viewer.remove(gone)

    def _cut(
        self, bundles: Mapping[ItemId, BundleSettings], placed: Placed, u: float
    ) -> None:
        """Cuts a bundle and keeps its longer part; a branch keeps its stem."""
        shape = placed.shape
        keep_start = shape.stem_points > 1 or u > 0.5
        the_cut = cut(shape, u, keep_start)
        if the_cut is None:
            self._remove(bundles, placed.item_id)
            return
        changed = {
            placed.item_id: with_geometry(bundles[placed.item_id], the_cut.geometry)
        }
        doomed = _after_cut(bundles, placed.item_id, the_cut, changed)
        self.editor.set_shapes(with_family(list(bundles.values()), changed), final=True)
        for gone in reversed(doomed):
            self.viewer.remove(gone)
        self.tools.worked_on(placed.item_id)

    def cancel(self) -> None:
        """Forgets the stroke."""
        self.line = None

    def overlay(self, out: Overlay) -> None:
        """Draws the stroke."""
        if self.line is not None and len(self.line.world) >= 2:
            out.strokes.append((np.array(self.line.world), ERASE_COLOUR))


@dataclass(eq=False)
class _StrokeDrag:
    """A stroke of the Pen.

    Attributes:
        line: The stroke so far.
        start: What it started on.
        start_depth: The depth it started at.
        track: The radius from each distance along the stroke on, mm.
    """

    line: _Line
    start: _StrokeStart
    start_depth: float
    track: list[tuple[float, float]]


@dataclass(eq=False)
class _PointDrag:
    """A control point of the bundle being drawn, being dragged, and all as before."""

    index: int
    depth: float
    before: tuple[Vec3, ...]


class _Pen(_Tool):
    """Draws a bundle by strokes and by single points."""

    def __init__(self, tools: BundleTools) -> None:
        """Creates the tool, with nothing being drawn."""
        super().__init__(tools)
        self.stroke: _StrokeDrag | None = None
        self.point: _PointDrag | None = None
        self.tip: Tip | None = None

    @property
    def dragging(self) -> bool:
        """Whether a stroke or a point is being dragged."""
        return self.stroke is not None or self.point is not None

    @property
    def _draft(self) -> Draft | None:
        """The bundle being drawn."""
        state = self.tools.state
        return state.draft if state is not None else None

    @property
    def _placing(self) -> bool:
        """Whether points are being placed one by one on the bundle being drawn."""
        draft = self._draft
        return draft is not None and draft.placing and bool(draft.points)

    @property
    def radius(self) -> float:
        """The Pen's radius, the stroke's during a stroke and else the bundle's."""
        if self.stroke is not None:
            return self.stroke.track[-1][1]
        draft = self._draft
        return draft.geometry.radius if draft is not None else GeometrySettings.radius

    def hint(self) -> str:
        """Returns a sentence on how to use the tool."""
        if self._placing:
            return (
                "Placing points: click to add one, scroll to change depth, Backspace "
                "takes the last one back, Enter finishes."
            )
        return (
            "Pen: drag to draw, or click to place points, and Enter finishes. Scroll "
            "changes depth, also while drawing. Start on a tip to continue, on the "
            "side to branch, or draw over a stretch to redo it."
        )

    def options(self) -> tuple[ToolSlider | ToolChoice, ...]:
        """Returns the radius."""
        low, high = PEN_RADII
        radius = self.radius
        return (
            ToolSlider(
                "radius", "Radius", low, high, PEN_STEP, radius, f"{radius:.1f} mm"
            ),
        )

    def set_option(self, key: str, value: float | str) -> None:
        """Changes the radius of the bundle being drawn."""
        if key == "radius":
            assert isinstance(value, (int, float))
            self._set_radius(float(value))

    def step(self, direction: int) -> None:
        """Changes the radius by a step, from here on while a stroke is being drawn."""
        low, high = PEN_RADII
        self._set_radius(min(high, max(low, self.radius + direction * PEN_STEP)))

    def _set_radius(self, radius: float) -> None:
        """Sets the radius of the bundle, or from here on along the stroke going on."""
        if self.stroke is not None:
            self.stroke.track.append((self.stroke.line.length, radius))
        elif self._draft is not None:
            self.viewer.set_bundle(None, BundleField.RADIUS, radius)

    # What lies under the pointer

    def _dot_at(self, pointer: Pointer) -> int | None:
        """Returns the index of the draft's own control point under the pointer.

        Its two ends are for continuing, unless Ctrl or Cmd is held.
        """
        draft = self._draft
        if draft is None or not draft.points:
            return None
        points = np.array(draft.points, dtype=np.float64)
        first = draft.branch.stem_points if draft.branch else 0
        last = len(points) - 1
        screen = pointer.plane.project(points)
        for index in range(first, last + 1):
            if not pointer.ctrl and index in (first, last):
                continue
            x, y = screen[index]
            if math.hypot(float(x) - pointer.x, float(y) - pointer.y) <= HANDLE_PIXELS:
                return index
        return None

    def _start_info(self, pointer: Pointer) -> _StrokeStart:
        """Returns what a stroke from the pointer starts on.

        A bundle that does not show here is out of reach, unless it is the one being
        drawn, whose path always shows. A bundle far from the slice that lies under
        the pointer gives a new stroke on the slice, which can still redo it.
        """
        plane = pointer.plane
        at = self.tools.slice_at(plane)
        tip = self.tools.tip_at(plane, pointer.x, pointer.y)
        if tip is not None:
            placed = tip.placed
            far = bool(self.tools.far_from_slice(plane, placed)[tip.k])
            if not far or placed.is_draft or placed.slab is None:
                depth = float(placed.shape.centre[tip.k, plane.axis])
                u = 1.0 if tip.at_end else 0.0
                return _StrokeStart(_Start.TIP, depth, placed, tip.at_end, u)
        hit = self.tools.pick(plane, pointer.x, pointer.y, any_depth=True)
        if hit is not None:
            placed = hit.placed
            if not hit.far or placed.is_draft or placed.slab is None:
                depth = at if hit.far else float(placed.shape.centre[hit.k, plane.axis])
                return _StrokeStart(_Start.BODY, depth, placed, True, hit.u, hit.far)
        return _StrokeStart(_Start.EMPTY, at)

    def _read(self, drag: _StrokeDrag) -> _Read:
        """Returns which of the four things a stroke means."""
        start = drag.start
        placed = start.placed
        if start.kind == _Start.EMPTY or placed is None:
            return _Read(_Reading.NEW)
        shape = placed.shape
        plane = drag.line.plane
        x, y = drag.line.screen[-1]
        screen = plane.project(shape.centre)
        dist = np.hypot(screen[:, 0] - x, screen[:, 1] - y)
        dist[: shape.first_own] = np.inf
        k = int(np.argmin(dist))
        stop_u = float(shape.u[k])
        width = float(shape.widths(plane.axis)[k]) / plane.mm_per_pixel
        reach = max(width, 10.0) + 3.0
        # It stops on the same bundle somewhere else: it redoes the stretch between.
        if (
            float(dist[k]) <= reach
            and abs(stop_u - start.u) > 0.06
            and drag.line.length > 6
        ):
            return _Read(_Reading.REDRAW, placed, True, start.u, stop_u)
        if start.far:
            return _Read(_Reading.NEW)
        if start.kind == _Start.TIP:
            return _Read(_Reading.EXTEND, placed, start.at_end)
        return _Read(_Reading.BRANCH, placed, True, start.u)

    def _fit(self, world: Sequence[Vec3], plane: Plane) -> list[Vec3]:
        """Returns few control points for a stroke, inside the phantom's grid."""
        state = self.editor.require()
        grid = self.viewer._reference_grid() or state.grid
        voxel = float(np.linalg.norm(grid.affine[:3, :3], axis=0).min())
        tolerance = min(voxel, max(0.2, 2.5 * plane.mm_per_pixel))
        kept = [p for p in simplify_stroke(world, tolerance) if state.contains(p)]
        if not kept:
            raise ValueError(
                "The stroke is outside the phantom's grid; draw inside the box."
            )
        return kept

    # Pressing and dragging

    def press(self, pointer: Pointer) -> bool:
        """Takes a control point under the pointer, or starts a stroke."""
        dot = self._dot_at(pointer)
        draft = self._draft
        if dot is not None and draft is not None:
            depth = draft.points[dot][pointer.plane.axis]
            self.point = _PointDrag(dot, depth, draft.points)
            return True
        if pointer.ctrl:
            return False
        self._start_stroke(pointer, self._start_info(pointer))
        return True

    def _start_stroke(self, pointer: Pointer, start: _StrokeStart) -> None:
        """Starts a stroke; from a tip on another slice, the slice goes there."""
        plane = pointer.plane
        placing = self._placing
        if start.kind == _Start.TIP and not placing:
            self._go_to_depth(plane, start.depth)
        depth = self.tools.slice_at(plane) if placing else start.depth
        self.stroke = _StrokeDrag(
            _start_line(plane, pointer, depth), start, depth, [(0.0, self.radius)]
        )

    def _go_to_depth(self, plane: Plane, depth: float) -> None:
        """Moves the view's slice to a depth."""
        cursor = list(self.viewer.scene.view.cursor)
        cursor[plane.axis] = depth
        self.viewer.move_cursor((cursor[0], cursor[1], cursor[2]))

    def move(self, pointer: Pointer) -> None:
        """Moves the point taken, or carries on the stroke."""
        if self.point is not None:
            draft = self._draft
            if draft is not None:
                world = pointer.plane.world(pointer.x, pointer.y, self.point.depth)
                points = list(draft.points)
                points[self.point.index] = vec3(world)
                self.editor.set_points(points)
        elif self.stroke is not None:
            self.stroke.line.add(pointer)
            self.tools.label = self._words()

    def _words(self) -> str | None:
        """Returns what the stroke would make, for the status bar."""
        drag = self.stroke
        if drag is None or drag.line.travel < CLICK_PIXELS or self._placing:
            return None
        return {
            _Reading.NEW: "New bundle",
            _Reading.EXTEND: "Continue",
            _Reading.BRANCH: "Branch",
            _Reading.REDRAW: "Redo this stretch",
        }[self._read(drag).kind]

    def scrolled(self, plane: Plane, by: float) -> None:
        """Moves the stroke in depth with the slice."""
        if self.stroke is not None:
            self.stroke.line.move_depth(by)

    def release(self, pointer: Pointer) -> None:
        """Ends the point's drag, or makes the click or the stroke."""
        self.point = None
        drag, self.stroke = self.stroke, None
        if drag is None:
            return
        scrolled = drag.line.depth != drag.start_depth
        if drag.line.travel < CLICK_PIXELS and not scrolled:
            self._click(drag.line.plane, drag.line.screen[0], drag.start)
            return
        if drag.line.length < SHORTEST_STROKE:
            return
        self._draw_stroke(drag)
        # The radius the stroke ended with stays the Pen's.
        if drag.track[-1][1] != drag.track[0][1] and self._draft is not None:
            self.viewer.set_bundle(None, BundleField.RADIUS, drag.track[-1][1])

    def cancel(self) -> None:
        """Forgets the stroke, or puts the point back."""
        if self.point is not None:
            self.editor.set_points(self.point.before)
        self.stroke = self.point = None

    # Clicks

    def click_here(self, pointer: Pointer) -> None:
        """Places a point right here, whatever lies under it."""
        start = _StrokeStart(_Start.EMPTY, self.tools.slice_at(pointer.plane))
        self._click(pointer.plane, (pointer.x, pointer.y), start)

    def _click(
        self, plane: Plane, at: tuple[float, float], start: _StrokeStart
    ) -> None:
        """Places a control point on the slice, or takes a tip to continue from.

        The first two points make a bundle, and the next ones lengthen it. A click on
        a bundle does nothing.

        Raises:
            ValueError: If the point is outside the phantom's grid.
        """
        placing = self._placing
        if not placing and start.kind == _Start.TIP and start.placed is not None:
            self._draft_of(start.placed.item_id)
            self.editor.set_placing(True, not start.at_end)
            return
        if not placing and start.kind == _Start.BODY and not start.far:
            return
        point = vec3(plane.world(at[0], at[1], self.tools.slice_at(plane)))
        state = self.editor.require()
        if not state.contains(point):
            raise ValueError(
                "The point ({:.1f}, {:.1f}, {:.1f}) mm is outside the phantom's "
                "grid; draw inside the box.".format(*point)
            )
        if not placing:
            self._fresh_draft()
            self.viewer.add_bundle_point(point)
            self.editor.set_placing(True, False)
            self.viewer.move_cursor(point)
            return
        draft = self._draft
        assert draft is not None
        end = draft.points[0] if draft.at_start else draft.points[-1]
        if math.dist(point, end) < 1:
            return
        self._lengthen(draft, [end, point], not draft.at_start, None)
        self.viewer.move_cursor(point)

    # Strokes

    def _draw_stroke(self, drag: _StrokeDrag) -> None:
        """Does what the stroke means."""
        draft = self._draft
        plane = drag.line.plane
        world = list(drag.line.world)
        track = drag.track if len({r for _, r in drag.track}) > 1 else None
        if self._placing and draft is not None:
            if len(draft.points) == 1:
                self._new(drag, [draft.points[0], *world], track, keep_placing=True)
            else:
                end = draft.points[0] if draft.at_start else draft.points[-1]
                added = self._fit([end, *world], plane)
                length = stroke_length([end, *world])
                self._lengthen(draft, added, not draft.at_start, track, length)
            self.viewer.move_cursor(world[-1])
            return
        reading = self._read(drag)
        if reading.kind == _Reading.NEW:
            self._new(drag, world, track)
            self.viewer.move_cursor(world[-1])
            return
        assert reading.placed is not None
        item_id = reading.placed.item_id
        if reading.kind == _Reading.EXTEND:
            current = self._draft_of(item_id)
            added = self._fit(world, plane)
            self._lengthen(current, added, reading.at_end, track, stroke_length(world))
            self.viewer.move_cursor(world[-1])
        elif reading.kind == _Reading.BRANCH:
            added = self._fit(world, plane)
            self._close_draft()
            self.editor.start_branch(item_id, reading.u, added)
        else:
            current = self._draft_of(item_id)
            geometry = replace(current.geometry, control_points=current.points)
            shape = shape_of(
                geometry, current.branch.stem_points if current.branch else 0
            )
            made = redrawn(
                shape,
                world,
                plane.axis,
                reading.u,
                reading.stop_u,
                lambda polyline: self._fit(polyline, plane),
            )
            self.editor.set_draft_shape(made)

    def _new(
        self,
        drag: _StrokeDrag,
        world: list[Vec3],
        track: list[tuple[float, float]] | None,
        *,
        keep_placing: bool = False,
    ) -> None:
        """Makes a stroke the bundle being drawn, finishing one that has its points."""
        plane = drag.line.plane
        points = self._fit(world, plane)
        if len(points) < 2:
            return
        if not keep_placing:
            self._fresh_draft()
        else:
            self.editor.set_points(())
        self.viewer.note_points(points)
        draft = self.editor.draft()[1]
        geometry = replace(draft.geometry, control_points=tuple(points))
        if track is not None:
            rows = rows_from_track(track, stroke_length(world))
            geometry = with_rows(geometry, rows)
        self.editor.set_draft_shape(geometry)
        if keep_placing:
            self.editor.set_placing(True, False)

    def _lengthen(
        self,
        draft: Draft,
        added: Sequence[Vec3],
        at_end: bool,
        track: list[tuple[float, float]] | None,
        length: float | None = None,
    ) -> None:
        """Lengthens the bundle being drawn from one end."""
        if len(added) < 2:
            return
        if len(draft.points) < 2:
            self.viewer.note_points(added[1:])
            self.editor.set_points((draft.points[0], *added[1:]))
            return
        total = stroke_length(added) if length is None else length
        stem = draft.branch.stem_points if draft.branch else 0
        shape = shape_of(replace(draft.geometry, control_points=draft.points), stem)
        made, _ = extended(shape, added, at_end, total, track)
        self.viewer.note_points(added[1:], at_start=not at_end)
        self.editor.set_draft_shape(made)

    def _fresh_draft(self) -> None:
        """Makes the bundle being drawn an empty one, finishing one with its points."""
        draft = self._draft
        if draft is None or not draft.is_empty:
            self._close_draft()
            self.viewer.start_bundle()

    def _close_draft(self) -> None:
        """Finishes the bundle being drawn if it has its points, or drops it."""
        draft = self._draft
        if draft is None:
            return
        if drawable(draft.points):
            self.editor.finish()
        else:
            self.editor.cancel()

    def _draft_of(self, item_id: ItemId) -> Draft:
        """Returns the bundle being drawn after making it this one."""
        draft = self._draft
        if draft is not None and draft.item_id == item_id:
            return draft
        self._close_draft()
        self.editor.edit(item_id)
        self.viewer.forget_point_groups()
        return self.editor.draft()[1]

    # Showing

    def hover(self, pointer: Pointer) -> None:
        """Marks a tip under the pointer, and says what a stroke from here would do."""
        self.tip = None
        self.tools.label = None
        if self._placing:
            return
        start = self._start_info(pointer)
        if start.kind == _Start.TIP:
            self.tip = self.tools.tip_at(pointer.plane, pointer.x, pointer.y)
            self.tools.label = "Continue"
        elif start.kind == _Start.BODY and not start.far:
            self.tools.label = "Branch"

    def leave(self) -> None:
        """Forgets the tip marked."""
        self.tip = None

    def overlay(self, out: Overlay) -> None:
        """Draws the stroke, the next point's line, the tip marked and the brush."""
        if self.stroke is not None and len(self.stroke.line.world) >= 2:
            out.strokes.append((np.array(self.stroke.line.world), POINT_COLOUR))
        pointer = self.tools.pointer
        draft = self._draft
        if pointer is None or draft is None:
            return
        plane = pointer.plane
        here = plane.world(pointer.x, pointer.y, self.tools.slice_at(plane))
        out.under_pointer_axis = plane.axis
        if self._placing and self.stroke is None:
            end = draft.points[0] if draft.at_start else draft.points[-1]
            out.under_pointer.append(np.array([end, here], dtype=np.float64))
        if self.tip is not None and self.stroke is None:
            out.circles.append(self.tip.placed.shape.centre[self.tip.k][None])
        out.under_pointer.append(_brush_circle(here, plane, self.radius))


def _brush_circle(centre: np.ndarray, plane: Plane, radius: float) -> np.ndarray:
    """Returns a circle around a point in a view's plane, 2 pixels wide at least."""
    size = max(radius, 2 * plane.mm_per_pixel)
    axes = [i for i in range(3) if i != plane.axis]
    t = np.linspace(0.0, 2 * np.pi, 41)
    circle = np.tile(centre, (41, 1))
    circle[:, axes[0]] += size * np.cos(t)
    circle[:, axes[1]] += size * np.sin(t)
    return circle
