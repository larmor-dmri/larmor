"""The four views, each a render canvas in a thin frame, and their mouse input."""

import math
from collections.abc import Callable, Sequence
from enum import Enum, auto

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QMouseEvent, QNativeGestureEvent, QWheelEvent
from PySide6.QtWidgets import QFrame, QToolTip, QVBoxLayout, QWidget
from rendercanvas.qt import QRenderWidget

from larmor.viewer.model.camera import Vec3, screen_axes
from larmor.viewer.model.detail import (
    VoxelBox,
    detail_field,
    plane_axes,
    size_text,
    voxel_box,
)
from larmor.viewer.model.item import VIEW_GRID, Category, Item, ViewKind
from larmor.viewer.model.options import (
    SLAB_THICKNESS,
    OptionError,
    OptionName,
    RegionOptions,
    RegionShape,
    TractogramOptions,
)
from larmor.viewer.model.regions import (
    Handle,
    box_between,
    hit_region,
    ray_hit,
    sphere_radius_at,
)
from larmor.viewer.model.view_state import Tool
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.render.gpu import finish_gpu_work
from larmor.viewer.render.phantom_guides import POINT_COLOUR
from larmor.viewer.render.regions_guides import DETAIL_COLOUR, Sketch
from larmor.viewer.render.view import ViewRenderer
from larmor.viewer.render.view_button import ViewButton
from larmor.viewer.render.volume_adapter import SLICE_AXIS

POSITIONS: dict[ViewKind, tuple[int, int]] = {
    kind: (row, col)
    for row, kinds in enumerate(VIEW_GRID)
    for col, kind in enumerate(kinds)
}
WHEEL_STEP = 120  # Qt's angle delta for one notch of a mouse wheel
WHEEL_ZOOM = 1.1  # the zoom of one notch with Ctrl or Cmd
# A drag across the 3D view's height (or width, if smaller) turns it half a turn.
DRAG_DEGREES = 180.0
# Ctrl is Qt's ControlModifier off macOS; on macOS that is Cmd, and Ctrl is Meta.
ZOOM_MODIFIERS = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
FULL_RESOLUTION_DELAY = 150  # ms after the last drag or scroll step
MOUSE_EVENTS = {
    QEvent.Type.MouseButtonPress,
    QEvent.Type.MouseButtonDblClick,
    QEvent.Type.MouseButtonRelease,
}
# The most frames a second the views and the Inspect FOD canvas draw;
# rendercanvas's default of 30 is below the 60 the 3D view aims at.
MAX_FPS = 120.0
# How near, in logical pixels, a press must be to a handle or an outline to take it.
HANDLE_PIXELS = 6.0
# How far, in logical pixels, the pointer moves before a press becomes a drag.
CLICK_DISTANCE = 4
# The smallest radius or side a drag gives a region, mm.
MIN_SIZE = 0.1


class Drag(Enum):
    """What a left drag on a view does."""

    CURSOR = auto()
    PAN = auto()
    ROTATE = auto()  # the 3D view; a press without moving clicks with the tool
    MOVE_REGION = auto()
    RESIZE_REGION = auto()
    PLACE_SPHERE = auto()
    PLACE_BOX = auto()
    ZOOM_BOX = auto()
    MOVE_DETAIL = auto()
    STROKE = auto()  # a freehand stroke of the bundle being drawn
    MOVE_POINT = auto()  # a control point of the bundle being drawn
    PICK_LABEL = auto()  # a label of a label region, added or removed


class ViewPanel(QFrame):
    """Shows one view's canvas in a thin frame, in the accent while it is selected.

    The render layer draws the view's name and an enlarge button in its corners.
    """

    def __init__(self, kind: ViewKind, parent: QWidget | None = None) -> None:
        """Creates the view's canvas and renderer, not selected."""
        super().__init__(parent)
        self.kind = kind
        self.setObjectName("viewPanel")
        self.canvas = QRenderWidget(self, update_mode="ondemand", max_fps=MAX_FPS)
        self.canvas.setMouseTracking(True)  # for the button's tooltip
        self.view = ViewRenderer(self.canvas, kind)
        self.button = ViewButton(self.view)
        self.maximised = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, stretch=1)

        self.set_maximised(False)
        self.set_selected(False)

    def set_maximised(self, maximised: bool) -> None:
        """Shows the button's restore or enlarge glyph."""
        self.maximised = maximised
        self.button.set_maximised(maximised)

    def tooltip(self) -> str:
        """Returns the enlarge button's tooltip, which says what it does."""
        return "Show all views" if self.maximised else "Fill the window with this view"

    def set_selected(self, selected: bool) -> None:
        """Draws the frame in the accent if selected, as the stylesheet says."""
        self.selected = selected
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)


class ViewInput(QObject):
    """Turns the mouse on a view's canvas into actions on the viewer.

    What a click and a drag do follows the view state's pointer tool; Ctrl+drag
    pans and the wheel steps the slice under every tool, and a press on a
    region's handles or a detail's box moves or resizes it first.
    """

    interacted = Signal()  # a drag, scroll or pinch step, sent before its action
    refused = Signal(str)  # why a click did nothing, for the status bar
    notice = Signal(str)  # what a drag makes, such as a region's size
    # A right click: the view, the world position under the pointer (None in 3D
    # when nothing is there) and the global point for a context menu.
    menu_requested = Signal(object, object, QPoint)

    def __init__(self, viewer: Viewer, panel: ViewPanel) -> None:
        """Listens to the mouse on the panel's canvas."""
        super().__init__(panel)
        self.viewer = viewer
        self.panel = panel
        self.kind = panel.kind
        self._dragging: Drag | None = None
        self._last: QPointF | None = None
        self._pressed_at: QPointF | None = None  # where the left button went down
        self._moved = False  # the pointer left the click distance since then
        self._wheel = 0  # angle delta not yet turned into slices
        self._last_type: QEvent.Type | None = None  # the last mouse button event
        self._on_button = False  # the pointer is over the enlarge button
        # What the drag works on: a region or detail id, a point's index, the
        # world point pressed, the offset of a region's centre from it, a box's
        # fixed corner, the stroke drawn, and a detail's box when pressed.
        self._target: int | None = None
        self._anchor: Vec3 = (0.0, 0.0, 0.0)
        self._offset = np.zeros(3)
        self._stroke: list[Vec3] = []
        self._box: VoxelBox | None = None
        self._shift = False  # Shift was held when the button went down
        self.sketch = Sketch(panel.view) if self.kind in SLICE_AXIS else None
        panel.canvas.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Handles the canvas's mouse, wheel and pinch events."""
        kind = event.type()
        # Whether Qt sent the second press of a double click before it, as it can.
        pressed = self._last_type == QEvent.Type.MouseButtonPress
        if kind in MOUSE_EVENTS:
            self._last_type = kind
        if kind == QEvent.Type.MouseButtonPress:
            assert isinstance(event, QMouseEvent)
            self._press(event)
        elif kind == QEvent.Type.MouseButtonDblClick:
            assert isinstance(event, QMouseEvent)
            self._double_click(event, pressed)
        elif kind == QEvent.Type.MouseMove:
            assert isinstance(event, QMouseEvent)
            self._move(event)
        elif kind == QEvent.Type.MouseButtonRelease:
            assert isinstance(event, QMouseEvent)
            self._release(event)
        elif kind == QEvent.Type.Wheel:
            assert isinstance(event, QWheelEvent)
            self.interacted.emit()
            self._wheel_event(event)
            return True
        elif kind == QEvent.Type.NativeGesture:
            assert isinstance(event, QNativeGestureEvent)
            if event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                self.interacted.emit()
                self._zoom(1.0 + event.value(), event.position())
                return True
        return False

    # Pressing

    def _press(self, event: QMouseEvent) -> None:
        """Selects the view, then opens a menu or starts what the tool drags."""
        self.viewer.select_view(self.kind)
        if event.button() == Qt.MouseButton.RightButton:
            self._menu(event)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = event.position()
        if self.panel.button.hit(point.x(), point.y()):
            self.viewer.toggle_maximised(self.kind)
            return
        self._pressed_at, self._moved, self._last = point, False, point
        self._shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if self.kind not in SLICE_AXIS:
            self._dragging = Drag.ROTATE
        elif event.modifiers() & ZOOM_MODIFIERS:
            self._dragging = Drag.PAN
        else:
            self._press_2d(point)

    def _press_2d(self, point: QPointF) -> None:
        """Starts a drag in a 2D view: a handle's, else the tool's."""
        world = self._in_slice(point)
        self._anchor = world
        tool = self.viewer.scene.view.tool
        if tool == Tool.DRAW_BUNDLE and self.viewer.drawing:
            index = self._draft_point_at(point)
            if index is not None:
                self._target, self._dragging = index, Drag.MOVE_POINT
            else:
                self._stroke, self._dragging = [world], Drag.STROKE
            return
        if self._press_region(point, world) or self._press_detail(point, tool):
            return
        if tool == Tool.SPHERE:
            self._target, self._dragging = None, Drag.PLACE_SPHERE
        elif tool == Tool.BOX:
            self._target, self._dragging = None, Drag.PLACE_BOX
        elif tool == Tool.ZOOM_BOX:
            self._box, self._dragging = None, Drag.ZOOM_BOX
        elif tool == Tool.LABEL_REGION:
            self._dragging = Drag.PICK_LABEL
        else:
            self._dragging = Drag.CURSOR
            self._move_cursor(point)

    def _press_region(self, point: QPointF, world: Vec3) -> bool:
        """Selects the region under a press and starts moving or resizing it."""
        found = self._region_at(world)
        if found is None:
            return False
        region, handle = found
        o = region.options
        assert isinstance(o, RegionOptions)
        self.viewer.select(region.id)
        self._target = region.id
        axis = SLICE_AXIS[self.kind]
        resize = (o.shape == RegionShape.SPHERE and handle == Handle.EDGE) or (
            handle == Handle.CORNER
        )
        if not resize:
            self._dragging = Drag.MOVE_REGION
            self._offset = np.asarray(o.centre) - np.asarray(world)
            self._offset[axis] = 0.0
        elif o.shape == RegionShape.SPHERE:
            self._dragging = Drag.RESIZE_REGION
        else:
            # The corner opposite the one pressed stays where it is.
            centre = np.asarray(o.centre)
            fixed = 2 * centre - np.asarray(world)
            self._anchor = (float(fixed[0]), float(fixed[1]), float(fixed[2]))
            self._dragging = Drag.RESIZE_REGION
        return True

    def _press_detail(self, point: QPointF, tool: Tool) -> bool:
        """Starts moving a detail whose box's edge is pressed, or inside it with Z."""
        grid = self.viewer._detail_grid()
        if grid is None:
            return False
        for detail in self.viewer.scene.details.values():
            if detail.view != self.kind:
                continue
            low, high = detail_field(grid, self.kind, detail.box, self._anchor)
            xs, ys = zip(
                *(self.panel.view.to_screen(p) for p in (low, high)), strict=True
            )
            left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
            x, y = point.x(), point.y()
            inside = left <= x <= right and top <= y <= bottom
            near = (
                left - HANDLE_PIXELS <= x <= right + HANDLE_PIXELS
                and top - HANDLE_PIXELS <= y <= bottom + HANDLE_PIXELS
            )
            edge = near and not (
                left + HANDLE_PIXELS < x < right - HANDLE_PIXELS
                and top + HANDLE_PIXELS < y < bottom - HANDLE_PIXELS
            )
            if edge or (inside and tool == Tool.ZOOM_BOX):
                self._target, self._box = detail.id, detail.box
                self._dragging = Drag.MOVE_DETAIL
                return True
        return False

    def _double_click(self, event: QMouseEvent, pressed: bool) -> None:
        """Enlarges or restores the view; on the button, a second click of it."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = event.position()
        if self.panel.button.hit(point.x(), point.y()):
            if not pressed:  # else the second press has clicked the button already
                self.viewer.toggle_maximised(self.kind)
            return
        self._dragging = None
        self.viewer.toggle_maximised(self.kind)

    # Dragging

    def _move(self, event: QMouseEvent) -> None:
        """Carries on the drag, or follows the pointer when no button is down."""
        if not event.buttons() & Qt.MouseButton.LeftButton:
            self._dragging = None
            self._hover(event)
        point = event.position()
        if self._pressed_at is not None and not self._moved:
            moved = point - self._pressed_at
            self._moved = moved.manhattanLength() >= CLICK_DISTANCE
            if not self._moved and self._dragging in (Drag.PAN, Drag.ROTATE):
                return  # still a click: no pan or turn yet
        if self._dragging is not None:
            self.interacted.emit()
        if self._dragging == Drag.CURSOR:
            self._move_cursor(point)
        elif self._dragging == Drag.PAN and self._last is not None:
            self._pan(self._last, point)
        elif self._dragging == Drag.ROTATE and self._last is not None:
            self._rotate(self._last, point)
        elif self._dragging is not None and self._moved:
            self._drag_shape(self._dragging, self._in_slice(point), event)
        self._last = point

    def _drag_shape(self, drag: Drag, world: Vec3, event: QMouseEvent) -> None:
        """Carries on a drag that places, moves or resizes a shape, or draws one."""
        if drag == Drag.MOVE_REGION:
            self._move_region(world)
        elif drag == Drag.RESIZE_REGION:
            self._resize_region(world)
        elif drag == Drag.PLACE_SPHERE:
            self._place_sphere(world)
        elif drag == Drag.PLACE_BOX:
            self._place_box(world)
        elif drag == Drag.ZOOM_BOX:
            self._zoom_box(world, event.modifiers())
        elif drag == Drag.MOVE_DETAIL:
            self._move_detail(world)
        elif drag == Drag.STROKE:
            self._stroke.append(world)
            assert self.sketch is not None
            self.sketch.show_line(self._stroke, POINT_COLOUR)
        elif drag == Drag.MOVE_POINT:
            self._move_point(world)

    def _move_region(self, world: Vec3) -> None:
        """Moves the region dragged by its centre, in the view's plane."""
        target = self._target
        assert target is not None
        o = self._region_options(target)
        centre = np.asarray(world) + self._offset
        centre[SLICE_AXIS[self.kind]] = o.centre[SLICE_AXIS[self.kind]]
        self._act(lambda: self.viewer.set(target, OptionName.CENTRE, centre))

    def _resize_region(self, world: Vec3) -> None:
        """Resizes the region dragged by its ring or corner."""
        target = self._target
        assert target is not None
        o = self._region_options(target)
        axis = SLICE_AXIS[self.kind]
        if o.shape == RegionShape.SPHERE:
            radius = max(sphere_radius_at(o, axis, world[axis], world), MIN_SIZE)
            self._act(lambda: self.viewer.set(target, OptionName.RADIUS, radius))
            self.notice.emit(f"Radius {radius:.1f} mm")
            return
        centre, size = box_between(o, axis, self._anchor, world)
        size = _at_least(size, MIN_SIZE)
        self._act(lambda: self._set_box(target, centre, size))

    def _set_box(self, target: int, centre: Vec3, size: Vec3) -> None:
        """Gives a box region a centre and a size, and says its size."""
        self.viewer.set(target, OptionName.CENTRE, centre)
        self.viewer.set(target, OptionName.SIZE, size)
        self.notice.emit("Box {:.1f} x {:.1f} x {:.1f} mm".format(*size))

    def _place_sphere(self, world: Vec3) -> None:
        """Places a sphere at the press, its radius reaching the pointer."""
        centre = self._anchor
        axis = SLICE_AXIS[self.kind]
        flat = np.asarray(world) - np.asarray(centre)
        flat[axis] = 0.0
        radius = max(float(np.linalg.norm(flat)), MIN_SIZE)
        if self._target is None:
            self._target = self.viewer.add_sphere(centre, radius, apply=True)
            self.viewer.select(self._target)
        else:
            self.viewer.set(self._target, OptionName.RADIUS, radius)
        self.notice.emit(f"Radius {radius:.1f} mm")

    def _place_box(self, world: Vec3) -> None:
        """Places a box between the press and the pointer, a slab's depth across."""
        axis = SLICE_AXIS[self.kind]
        a, b = np.asarray(self._anchor), np.asarray(world)
        centre = (a + b) / 2
        size = np.maximum(np.abs(a - b), MIN_SIZE)
        centre[axis] = self._anchor[axis]
        size[axis] = self._slab_depth()
        lengths = (float(size[0]), float(size[1]), float(size[2]))
        where = (float(centre[0]), float(centre[1]), float(centre[2]))
        if self._target is None:
            self._target = self.viewer.add_box(where, lengths, apply=True)
            self.viewer.select(self._target)
        else:
            self._set_box(self._target, where, lengths)
            return
        self.notice.emit("Box {:.1f} x {:.1f} x {:.1f} mm".format(*lengths))

    def _zoom_box(self, world: Vec3, modifiers: Qt.KeyboardModifier) -> None:
        """Draws the rectangle a zoom box covers, snapped to the voxels."""
        grid = self.viewer._detail_grid()
        if grid is None or self.sketch is None:
            return
        self._box = voxel_box(
            grid,
            self.kind,
            self._anchor,
            world,
            square=bool(modifiers & Qt.KeyboardModifier.ShiftModifier),
            from_centre=bool(modifiers & Qt.KeyboardModifier.AltModifier),
        )
        low, high = detail_field(grid, self.kind, self._box, world)
        self.sketch.show_box(low, high, SLICE_AXIS[self.kind], DETAIL_COLOUR)
        self.notice.emit(size_text(grid, self.kind, self._box))

    def _move_detail(self, world: Vec3) -> None:
        """Moves the dragged detail's box by whole voxels with the pointer."""
        grid = self.viewer._detail_grid()
        if grid is None or self._target is None or self._box is None:
            return
        _, first, second = plane_axes(grid, self.kind)
        inverse = np.linalg.inv(grid.affine)
        start = inverse[:3, :3] @ np.asarray(self._anchor) + inverse[:3, 3]
        now = inverse[:3, :3] @ np.asarray(world) + inverse[:3, 3]
        step = np.rint(now - start).astype(int)
        i, j, size_i, size_j = self._box
        moved = (i + int(step[first]), j + int(step[second]), size_i, size_j)
        target = self._target
        self._act(lambda: self.viewer.move_detail(target, moved))

    def _move_point(self, world: Vec3) -> None:
        """Moves the dragged control point in the view's plane."""
        state = self.viewer.phantom
        if state is None or state.draft is None or self._target is None:
            return
        points = list(state.draft.points)
        axis = SLICE_AXIS[self.kind]
        moved = list(world)
        moved[axis] = points[self._target][axis]
        points[self._target] = (moved[0], moved[1], moved[2])
        self._act(lambda: self.viewer.set_bundle_points(points))

    # Releasing

    def _release(self, event: QMouseEvent) -> None:
        """Ends the drag; a press without moving clicks with the tool."""
        drag, self._dragging = self._dragging, None
        if self.sketch is not None:
            self.sketch.clear()
        if event.button() != Qt.MouseButton.LeftButton or drag is None:
            return
        if drag == Drag.ROTATE and not self._moved:
            self._click_3d(event.position())
        elif drag == Drag.STROKE:
            self._finish_stroke()
        elif drag == Drag.ZOOM_BOX and self._moved and self._box is not None:
            box = self._box
            self._act(lambda: self.viewer.open_detail(self.kind, box))
        elif not self._moved:
            self._click_2d(drag, self._in_slice(event.position()))

    def _click_2d(self, drag: Drag, world: Vec3) -> None:
        """Clicks with the tool: places a region, picks a label, or adds a point."""
        region: int | None = None
        if drag == Drag.PAN:
            if self.viewer.scene.view.tool == Tool.DRAW_BUNDLE and self.viewer.drawing:
                self._add_point(world)
        elif drag == Drag.PLACE_SPHERE:
            region = self._act(lambda: self.viewer.add_sphere(self._anchor, apply=True))
        elif drag == Drag.PLACE_BOX:
            region = self._act(lambda: self.viewer.add_box(self._anchor, apply=True))
        elif drag == Drag.PICK_LABEL:
            remove = self._shift
            self._act(lambda: self.viewer.click_label(world, remove=remove))
        if region is not None:
            self.viewer.select(region)

    def _finish_stroke(self) -> None:
        """Adds the control points of the stroke drawn, and moves the cursor there."""
        stroke = self._stroke
        self._stroke = []
        added = self._act(lambda: self.viewer.add_bundle_stroke(stroke))
        if added:
            state = self.viewer.phantom
            assert state is not None and state.draft is not None
            self.viewer.move_cursor(state.draft.points[-1])

    def _add_point(self, world: Vec3) -> None:
        """Adds one control point where clicked, and moves the cursor there."""
        try:
            self.viewer.add_bundle_point(world)
        except ValueError as e:
            self.refused.emit(str(e))
            return
        self.viewer.move_cursor(world)

    def _click_3d(self, point: QPointF) -> None:
        """Places a region at the point picked in 3D, or selects a region clicked."""
        tool = self.viewer.scene.view.tool
        if tool in (Tool.SPHERE, Tool.BOX):
            picked = self.panel.view.pick(point.x(), point.y())
            if picked is None:
                self.refused.emit(
                    "Nothing is under the pointer: click a streamline, a slice or a "
                    "surface to place the region on it."
                )
                return
            if tool == Tool.SPHERE:
                placed = self.viewer.add_sphere(picked, apply=True)
            else:
                placed = self.viewer.add_box(picked, apply=True)
            self.viewer.select(placed)
        elif tool == Tool.CURSOR:
            hit = self._region_on_ray(point)
            if hit is not None:
                self.viewer.select(hit.id)

    # Finding what is under the pointer

    def _regions(self) -> list[Item]:
        """Returns the visible sphere and box regions, the selected one first."""
        selected = self.viewer.scene.view.selected
        regions = [
            r
            for r in reversed(self.viewer.scene.in_category(Category.REGIONS))
            if isinstance(r.options, RegionOptions)
            and r.options.visible
            and r.options.shape in (RegionShape.SPHERE, RegionShape.BOX)
        ]
        return sorted(regions, key=lambda r: r.id != selected)

    def _region_at(self, world: Vec3) -> tuple[Item, Handle] | None:
        """Returns the region whose outline or centre is under a point, and which."""
        axis = SLICE_AXIS[self.kind]
        tolerance = HANDLE_PIXELS * self.panel.view.pixel_size()
        for region in self._regions():
            o = region.options
            assert isinstance(o, RegionOptions)
            handle = hit_region(o, axis, world[axis], world, tolerance)
            if handle is not None:
                return region, handle
        return None

    def _region_on_ray(self, point: QPointF) -> Item | None:
        """Returns the nearest region the ray through a 3D point meets."""
        origin, direction = self.panel.view.ray(point.x(), point.y())
        hits = []
        for region in self._regions():
            o = region.options
            assert isinstance(o, RegionOptions)
            distance = ray_hit(o, origin, direction)
            if distance is not None:
                hits.append((distance, region.id, region))
        return min(hits, key=lambda h: (h[0], h[1]))[2] if hits else None

    def _draft_point_at(self, point: QPointF) -> int | None:
        """Returns the index of the bundle's control point under a canvas point."""
        state = self.viewer.phantom
        if state is None or state.draft is None:
            return None
        for index, p in enumerate(state.draft.points):
            x, y = self.panel.view.to_screen(p)
            if math.hypot(x - point.x(), y - point.y()) <= HANDLE_PIXELS:
                return index
        return None

    def _region_options(self, region_id: int) -> RegionOptions:
        """Returns a region's options."""
        o = self.viewer.scene.item(region_id).options
        assert isinstance(o, RegionOptions)
        return o

    def _slab_depth(self) -> float:
        """Returns a new box's depth: the slab of the selected or first tractogram."""
        tractograms = self.viewer.scene.in_category(Category.TRACTOGRAMS)
        selected = self.viewer.scene.view.selected
        chosen = [t for t in tractograms if t.id == selected] or tractograms
        if chosen and isinstance(chosen[0].options, TractogramOptions):
            return chosen[0].options.slab_thickness
        return SLAB_THICKNESS

    def _act[T](self, action: Callable[[], T]) -> T | None:
        """Runs an action of the viewer; says why in the status bar if refused."""
        try:
            return action()
        except (OptionError, ValueError) as e:
            self.refused.emit(str(e))
            return None

    # The menu, the pointer and the camera

    def _menu(self, event: QMouseEvent) -> None:
        """Asks for the context menu of the point under the pointer."""
        point = event.position()
        if self.kind in SLICE_AXIS:
            position: Vec3 | None = self._in_slice(point)
        else:
            position = self.panel.view.pick(point.x(), point.y())
        self.menu_requested.emit(self.kind, position, event.globalPosition().toPoint())

    def _hover(self, event: QMouseEvent) -> None:
        """Shows the button's tooltip; the Signal breakdown follows the pointer."""
        point = event.position()
        if self.viewer.breakdown is not None and self.kind in SLICE_AXIS:
            self.viewer.hover_breakdown(self._in_slice(point))
        on_button = self.panel.button.hit(point.x(), point.y())
        if on_button:
            self.panel.canvas.setToolTip(self.panel.tooltip())
        elif self._on_button:
            self.panel.canvas.setToolTip("")
            QToolTip.hideText()
        self._on_button = on_button

    def _move_cursor(self, point: QPointF) -> None:
        """Moves the cursor to the point clicked, in this view's slice.

        While the Signal breakdown shows, a click on the phantom pins its voxel.
        """
        position = self._in_slice(point)
        self.viewer.move_cursor(position)
        phantom = self.viewer.phantom
        if (
            self.viewer.breakdown is not None
            and phantom is not None
            and phantom.contains(position)
        ):
            try:
                self.viewer.pin_breakdown(position)
            except ValueError:
                pass  # on the grid's outer edge, which no voxel holds

    def _in_slice(self, point: QPointF) -> tuple[float, float, float]:
        """Returns the world position under a canvas point, in this view's slice."""
        world = list(self.panel.view.to_world(point.x(), point.y()))
        axis = SLICE_AXIS[self.kind]
        world[axis] = self.viewer.scene.view.cursor[axis]
        return (world[0], world[1], world[2])

    def _pan(self, start: QPointF, end: QPointF) -> None:
        """Pans so that the point under the mouse follows it."""
        a = self.panel.view.to_world(start.x(), start.y())
        b = self.panel.view.to_world(end.x(), end.y())
        right, up = screen_axes(self.kind, self.viewer.scene.view.convention)
        moved = [p - q for p, q in zip(a, b, strict=True)]
        self.viewer.pan(
            self.kind,
            sum(m * r for m, r in zip(moved, right, strict=True)),
            sum(m * u for m, u in zip(moved, up, strict=True)),
        )

    def _rotate(self, start: QPointF, end: QPointF) -> None:
        """Turns the 3D scene with the mouse: right and up on the screen."""
        size = min(self.panel.canvas.width(), self.panel.canvas.height())
        degrees = DRAG_DEGREES / max(size, 1)
        moved = end - start
        # Qt's y grows downwards.
        self.viewer.rotate_camera(moved.x() * degrees, -moved.y() * degrees)

    def _wheel_event(self, event: QWheelEvent) -> None:
        """Steps the slice a notch at a time; zooms in 3D or with Ctrl or Cmd.

        In 3D, the wheel over the selected region resizes it.
        """
        delta = event.angleDelta().y()
        if self.kind not in SLICE_AXIS:
            if not self._resize_selected(event.position(), delta):
                self._zoom(WHEEL_ZOOM ** (delta / WHEEL_STEP), event.position())
            return
        if event.modifiers() & ZOOM_MODIFIERS:
            self._zoom(WHEEL_ZOOM ** (delta / WHEEL_STEP), event.position())
            return
        # A trackpad sends small deltas: a slice per full notch's worth.
        self._wheel += delta
        steps = int(self._wheel / WHEEL_STEP)
        if steps:
            self._wheel -= steps * WHEEL_STEP
            self.viewer.step_slice(self.kind, steps)

    def _resize_selected(self, point: QPointF, delta: int) -> bool:
        """Grows or shrinks the selected region under a 3D point; False if none."""
        region = self._region_on_ray(point)
        if region is None or region.id != self.viewer.scene.view.selected:
            return False
        o = region.options
        assert isinstance(o, RegionOptions)
        factor = WHEEL_ZOOM ** (delta / WHEEL_STEP)
        if o.shape == RegionShape.SPHERE:
            radius = max(o.radius * factor, MIN_SIZE)
            self._act(lambda: self.viewer.set(region.id, OptionName.RADIUS, radius))
            self.notice.emit(f"Radius {radius:.1f} mm")
        else:
            size = _at_least(tuple(s * factor for s in o.size), MIN_SIZE)
            self._act(lambda: self.viewer.set(region.id, OptionName.SIZE, size))
            self.notice.emit("Box {:.1f} x {:.1f} x {:.1f} mm".format(*size))
        return True

    def _zoom(self, factor: float, point: QPointF) -> None:
        """Zooms about the point under the mouse (the 3D view: about its centre)."""
        if not math.isfinite(factor) or factor <= 0:
            return
        if self.kind not in SLICE_AXIS:
            self.viewer.zoom(self.kind, factor)
            return
        about = list(self.panel.view.to_world(point.x(), point.y()))
        axis = SLICE_AXIS[self.kind]
        about[axis] = self.viewer.scene.view.cursor[axis]
        self.viewer.zoom(self.kind, factor, about=(about[0], about[1], about[2]))


def _at_least(size: Sequence[float], low: float) -> Vec3:
    """Returns three lengths, none below low."""
    a, b, c = (max(float(s), low) for s in size)
    return (a, b, c)


def stop_canvases(root: QWidget) -> None:
    """Stops the canvases in root drawing, and waits for their last frames.

    Call it before root's widgets are deleted, as when its window closes.
    """
    for canvas in root.findChildren(QRenderWidget):
        _drop_frame_copy(canvas)
        # Closed but kept, so that it goes with root, not before.
        canvas.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        canvas.close()
    finish_gpu_work()


def _drop_frame_copy(canvas: QRenderWidget) -> None:
    """Waits for the frame a canvas is copying back from the GPU, and drops it.

    Otherwise the copy shows the frame once it arrives, after the canvas may have
    been deleted. rendercanvas offers no public way to do this; without its
    downloader nothing is dropped.
    """
    context = getattr(canvas, "_canvas_context", None)
    downloader = getattr(context, "_downloader", None)
    clear = getattr(downloader, "_clear_pending_download", None)
    if callable(clear):
        clear()
