"""The Inspect FOD panel, which shows one FOD's glyph at the crosshair, large."""

import math
from typing import override

from PySide6.QtCore import QEvent, QObject, QPointF, Qt, Signal
from PySide6.QtGui import QCloseEvent, QMouseEvent, QNativeGestureEvent, QWheelEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from rendercanvas.qt import QRenderWidget

from larmor.viewer.model.item import Category, ItemId
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.qt.docks import ItemPanel, PanelContent
from larmor.viewer.qt.views import DRAG_DEGREES, MAX_FPS, WHEEL_STEP, WHEEL_ZOOM
from larmor.viewer.render.fod_inspector import FodInspector


class FodInspectorView(PanelContent):
    """Draws an FOD item's glyph at the cursor; a drag turns it, the wheel zooms.

    It closes when its item is removed, and stops following the scene once closed.
    """

    closed = Signal(int)  # the item's id

    def __init__(
        self, viewer: Viewer, item_id: ItemId, parent: QWidget | None = None
    ) -> None:
        """Creates the canvas and the inspector drawing into it."""
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.item_id = item_id
        self.title = QLabel(viewer.scene.item(item_id).name, self)
        self.title.setObjectName("formTitle")

        self.canvas = QRenderWidget(self, update_mode="ondemand", max_fps=MAX_FPS)
        self.inspector = FodInspector(viewer, item_id, self.canvas, removed=self.close)
        self.reset = QPushButton("Reset view", self)
        self.reset.clicked.connect(self.inspector.reset_camera)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(6, 4, 6, 6)
        buttons.addWidget(self.title, stretch=1)
        buttons.addWidget(self.reset)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, stretch=1)
        layout.addLayout(buttons)

        self._input = _InspectorInput(self.inspector, self.canvas)
        # Deleted with its parent, it gets no close event: stop listening then too.
        # Not `self` in the lambda: it would keep the widget alive.
        inspector = self.inspector
        self.destroyed.connect(lambda *_: inspector.close())

    @override
    def pause(self) -> None:
        """Stops the inspector drawing until it resumes."""
        self.inspector.pause()

    @override
    def resume(self) -> None:
        """Draws the glyph at the cursor as it is now."""
        self.inspector.resume()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Stops following the scene, then closes."""
        self.inspector.close()
        # rendercanvas closes its canvases on quit; a deleted one would raise there.
        self.canvas.close()
        self.closed.emit(self.item_id)
        super().closeEvent(event)


class FodInspectorPanel(ItemPanel):
    """Shows the selected FOD, or the last one selected, at the crosshair."""

    def __init__(self, viewer: Viewer, parent: QWidget | None = None) -> None:
        """Starts on the selected FOD, else the first."""
        super().__init__(
            viewer,
            Category.FODS,
            lambda item_id, panel: FodInspectorView(viewer, item_id, panel),
            "Open an FOD image to inspect its glyph at the crosshair here.",
            parent,
        )


class _InspectorInput(QObject):
    """Turns the mouse on the inspector's canvas into turns and zooms of its camera."""

    def __init__(self, inspector: FodInspector, canvas: QRenderWidget) -> None:
        """Listens to the mouse on the canvas."""
        super().__init__(canvas)
        self.inspector = inspector
        self.canvas = canvas
        self._last: QPointF | None = None
        canvas.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Turns the glyph on a left drag; zooms on the wheel or a pinch."""
        kind = event.type()
        if kind == QEvent.Type.MouseButtonPress:
            assert isinstance(event, QMouseEvent)
            if event.button() == Qt.MouseButton.LeftButton:
                self._last = event.position()
        elif kind == QEvent.Type.MouseMove:
            assert isinstance(event, QMouseEvent)
            if not event.buttons() & Qt.MouseButton.LeftButton:
                self._last = None
            elif self._last is not None:
                size = max(min(self.canvas.width(), self.canvas.height()), 1)
                moved = event.position() - self._last
                degrees = DRAG_DEGREES / size
                # Qt's y grows downwards.
                self.inspector.rotate(moved.x() * degrees, -moved.y() * degrees)
                self._last = event.position()
        elif kind == QEvent.Type.MouseButtonRelease:
            self._last = None
        elif kind == QEvent.Type.Wheel:
            assert isinstance(event, QWheelEvent)
            self._zoom(WHEEL_ZOOM ** (event.angleDelta().y() / WHEEL_STEP))
            return True
        elif kind == QEvent.Type.NativeGesture:
            assert isinstance(event, QNativeGestureEvent)
            if event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                self._zoom(1.0 + event.value())
                return True
        return False

    def _zoom(self, factor: float) -> None:
        """Zooms the inspector's camera by a finite, positive factor."""
        if math.isfinite(factor) and factor > 0:
            self.inspector.zoom(factor)
