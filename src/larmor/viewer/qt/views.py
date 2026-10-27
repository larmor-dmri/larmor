"""The four views, each a render canvas in a thin frame, and their mouse input."""

import math
from enum import Enum, auto

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QMouseEvent, QNativeGestureEvent, QWheelEvent
from PySide6.QtWidgets import QFrame, QToolTip, QVBoxLayout, QWidget
from rendercanvas.qt import QRenderWidget

from larmor.viewer.model.camera import screen_axes
from larmor.viewer.model.item import VIEW_GRID, ViewKind
from larmor.viewer.model.viewer import Viewer
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


class Drag(Enum):
    """What a left drag on a view does."""

    CURSOR = auto()
    PAN = auto()
    ROTATE = auto()  # the 3D view


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
    """Turns the mouse on a view's canvas into actions on the viewer."""

    interacted = Signal()  # a drag, scroll or pinch step, sent before its action
    refused = Signal(str)  # why a click did nothing, for the status bar
    # A right click on the phantom in a 2D view: the world position and the
    # global point for a context menu.
    menu_requested = Signal(object, QPoint)

    def __init__(self, viewer: Viewer, panel: ViewPanel) -> None:
        """Listens to the mouse on the panel's canvas."""
        super().__init__(panel)
        self.viewer = viewer
        self.panel = panel
        self.kind = panel.kind
        self._dragging: Drag | None = None
        self._last: QPointF | None = None
        self._wheel = 0  # angle delta not yet turned into slices
        self._last_type: QEvent.Type | None = None  # the last mouse button event
        self._on_button = False  # the pointer is over the enlarge button
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
            self._dragging = None
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

    def _press(self, event: QMouseEvent) -> None:
        """Selects the view, then opens a menu, starts a drag or adds a bundle point."""
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
        if self.kind not in SLICE_AXIS:
            self._dragging = Drag.ROTATE
        elif event.modifiers() & ZOOM_MODIFIERS:
            self._dragging = Drag.PAN
        elif self.viewer.drawing:
            # A point of the bundle being drawn, in this view's slice; no drag.
            position = self._in_slice(event.position())
            try:
                self.viewer.add_bundle_point(position)
            except ValueError as e:
                self.refused.emit(str(e))
                return
            self.viewer.move_cursor(position)
        else:
            self._dragging = Drag.CURSOR
            self._move_cursor(event.position())
        self._last = event.position()

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

    def _move(self, event: QMouseEvent) -> None:
        """Carries on the drag, or follows the pointer when no button is down."""
        if not event.buttons() & Qt.MouseButton.LeftButton:
            self._dragging = None
            self._hover(event)
        if self._dragging is not None:
            self.interacted.emit()
        if self._dragging == Drag.CURSOR:
            self._move_cursor(event.position())
        elif self._dragging == Drag.PAN and self._last is not None:
            self._pan(self._last, event.position())
        elif self._dragging == Drag.ROTATE and self._last is not None:
            self._rotate(self._last, event.position())
        self._last = event.position()

    def _menu(self, event: QMouseEvent) -> None:
        """Asks for the context menu of a voxel of the phantom, in a 2D view."""
        phantom = self.viewer.phantom
        if self.kind not in SLICE_AXIS or phantom is None:
            return
        position = self._in_slice(event.position())
        if phantom.contains(position):
            self.menu_requested.emit(position, event.globalPosition().toPoint())

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
        """Zooms in 3D or with Ctrl or Cmd; else steps the slice a notch at a time."""
        delta = event.angleDelta().y()
        if self.kind not in SLICE_AXIS or event.modifiers() & ZOOM_MODIFIERS:
            self._zoom(WHEEL_ZOOM ** (delta / WHEEL_STEP), event.position())
            return
        # A trackpad sends small deltas: a slice per full notch's worth.
        self._wheel += delta
        steps = int(self._wheel / WHEEL_STEP)
        if steps:
            self._wheel -= steps * WHEEL_STEP
            self.viewer.step_slice(self.kind, steps)

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
