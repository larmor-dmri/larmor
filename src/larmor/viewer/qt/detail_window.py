"""The floating window of a detail: a rectangle of a 2D view, drawn on its own."""

from collections.abc import Callable
from typing import ClassVar
from weakref import ref

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QCloseEvent, QResizeEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMenu,
    QSpinBox,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from rendercanvas.qt import QRenderWidget

from larmor.viewer.model.detail import (
    DETAIL_SIZE,
    DetailId,
    DetailOption,
    FigureBackground,
    screen_millimetres,
)
from larmor.viewer.model.events import (
    DetailChanged,
    DetailClosed,
    DetailOpened,
    Event,
    ViewStateChanged,
)
from larmor.viewer.model.view_state import ViewField
from larmor.viewer.model.viewer import GLYPH_DETAILS, Viewer
from larmor.viewer.render.detail_renderer import DetailRenderer

# The file types a detail's picture and figure are saved as.
IMAGE_FILTER = "PNG image (*.png);;TIFF image (*.tif *.tiff)"
MAX_FPS = 60.0
MAX_SCALE = 8


class _AspectBox(QWidget):
    """Holds the canvas at the shape of the detail's rectangle, centred in it."""

    def __init__(self, canvas: QWidget, parent: QWidget | None = None) -> None:
        """Holds canvas, square at first."""
        super().__init__(parent)
        self.canvas = canvas
        canvas.setParent(self)
        self.aspect = 1.0  # width over height
        self.setMinimumSize(80, 80)

    def set_aspect(self, aspect: float) -> None:
        """Gives the canvas another width over height."""
        self.aspect = aspect
        self._fit()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        """Fits the canvas into the new size."""
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        """Makes the canvas as large as fits at the aspect, centred."""
        width, height = self.width(), self.height()
        if width / max(height, 1) > self.aspect:
            w, h = round(height * self.aspect), height
        else:
            w, h = width, round(width / self.aspect)
        self.canvas.setGeometry((width - w) // 2, (height - h) // 2, w, h)


class DetailWindow(QWidget):
    """Shows one detail window's picture, with a toolbar for its settings and exports.

    Closing the window closes the detail. The window remembers its height for
    the next detail opened.
    """

    # The canvas height the last detail window had, for the next one.
    last_height: ClassVar[int] = DETAIL_SIZE

    def __init__(
        self, viewer: Viewer, detail_id: DetailId, parent: QWidget | None = None
    ) -> None:
        """Creates the canvas, its drawing and the toolbar, at the box's shape."""
        super().__init__(parent, Qt.WindowType.Window)
        self.viewer = viewer
        self.detail_id = detail_id
        self._closing = False
        self.canvas = QRenderWidget(update_mode="ondemand", max_fps=MAX_FPS)
        self.box = _AspectBox(self.canvas, self)
        self.renderer = DetailRenderer(viewer, detail_id, self.canvas)
        self._create_toolbar()
        self.message = QLabel(self)
        self.message.setWordWrap(True)
        self.message.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.box, stretch=1)
        layout.addWidget(self.message)
        self._unsubscribe = viewer.scene.subscribe(self._on_event)
        self._show_detail()
        aspect = self.box.aspect
        height = DetailWindow.last_height
        self.resize(
            QSize(round(height * aspect), height + self.toolbar.sizeHint().height())
        )

    def _create_toolbar(self) -> None:
        """Creates the toolbar's switches, the glyph detail and the exports."""
        self.toolbar = QToolBar(self)
        self.linked = self._switch(
            "Linked", DetailOption.LINKED, "Follow the crosshair's slice"
        )
        self.voxel_grid = self._switch(
            "Voxel grid", DetailOption.VOXEL_GRID, "Draw the voxels' edges"
        )
        self.scale_bar = self._switch(
            "Scale bar", DetailOption.SCALE_BAR, "Draw a scale bar"
        )
        self.toolbar.addSeparator()
        self.toolbar.addWidget(QLabel(" Glyph detail ", self))
        self.glyph_detail = QSpinBox(self)
        self.glyph_detail.setRange(*GLYPH_DETAILS)
        self.glyph_detail.setToolTip("How smooth the FOD glyphs are")
        self.glyph_detail.valueChanged.connect(
            lambda value: self._set(DetailOption.GLYPH_DETAIL, int(value))
        )
        self.toolbar.addWidget(self.glyph_detail)
        self.toolbar.addSeparator()
        self.scale = QSpinBox(self)
        self.scale.setRange(1, MAX_SCALE)
        self.scale.setValue(2)
        self.scale.setSuffix(" x")
        self.scale.setToolTip("The saved picture's size, times the window's")
        self.toolbar.addWidget(self.scale)
        self.save_action = self.toolbar.addAction("Save 2 x...")
        self.save_action.triggered.connect(self._ask_save)
        self.scale.valueChanged.connect(
            lambda value: self.save_action.setText(f"Save {value} x...")
        )
        figure = QToolButton(self)
        figure.setText("Figure with inset...")
        figure.setToolTip("The view with the box, and the detail beside it")
        figure.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(figure)
        for background in FigureBackground:
            action = menu.addAction(f"On {background}...")
            action.triggered.connect(lambda _=False, b=background: self._ask_figure(b))
        figure.setMenu(menu)
        self.toolbar.addWidget(figure)

    def _switch(self, text: str, name: DetailOption, tip: str) -> QAction:
        """Returns a checkable toolbar action that sets a detail's switch."""
        action = self.toolbar.addAction(text)
        action.setCheckable(True)
        action.setToolTip(tip)
        action.toggled.connect(lambda on: self._set(name, bool(on)))
        return action

    def _set(self, name: DetailOption, value: object) -> None:
        """Changes one setting of the detail."""
        detail = self.viewer.scene.details.get(self.detail_id)
        if detail is not None and getattr(detail, name) != value:
            self.viewer.set_detail(self.detail_id, name, value)

    def _on_event(self, event: Event) -> None:
        """Follows the detail and the slice; closes once the detail is closed."""
        if isinstance(event, DetailClosed) and event.detail.id == self.detail_id:
            self._closing = True
            self.close()
        elif isinstance(event, DetailChanged) and event.detail.id == self.detail_id:
            self._show_detail()
        elif isinstance(event, ViewStateChanged) and event.name == ViewField.CURSOR:
            self.setWindowTitle(self.viewer.detail_title(self.detail_id))

    def _show_detail(self) -> None:
        """Shows the detail's title, settings and shape."""
        detail = self.viewer.scene.detail(self.detail_id)
        self.setWindowTitle(self.viewer.detail_title(self.detail_id))
        for action, on in (
            (self.linked, detail.linked),
            (self.voxel_grid, detail.voxel_grid),
            (self.scale_bar, detail.scale_bar),
        ):
            action.blockSignals(True)
            action.setChecked(on)
            action.blockSignals(False)
        self.glyph_detail.blockSignals(True)
        self.glyph_detail.setValue(detail.glyph_detail)
        self.glyph_detail.blockSignals(False)
        grid = self.viewer._detail_grid()
        if grid is not None:
            width, height = screen_millimetres(grid, detail.view, detail.box)
            self.box.set_aspect(width / height)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        """Remembers the picture's size, for the next window and the exports."""
        super().resizeEvent(event)
        size = self.canvas.size()
        if size.width() > 0 and size.height() > 0 and self.isVisible():
            DetailWindow.last_height = size.height()
            if self.detail_id in self.viewer.scene.details:
                self.viewer.resize_detail(self.detail_id, (size.width(), size.height()))

    def _ask_save(self) -> None:
        """Asks for a file and saves the picture at the chosen scale."""
        path, _ = QFileDialog.getSaveFileName(self, "Save Detail", "", IMAGE_FILTER)
        if path:
            self.save(path)

    def save(self, path: str) -> None:
        """Saves the picture at the chosen scale; says so, or why not, below."""
        self._report(
            lambda: self.viewer.save_detail(self.detail_id, path, self.scale.value()),
            path,
        )

    def _ask_figure(self, background: FigureBackground) -> None:
        """Asks for a file and saves the figure on a background."""
        path, _ = QFileDialog.getSaveFileName(self, "Save Figure", "", IMAGE_FILTER)
        if path:
            self.save_figure(path, background)

    def save_figure(self, path: str, background: FigureBackground) -> None:
        """Saves the figure on a background; says so, or why not, below."""
        self._report(
            lambda: self.viewer.save_detail_figure(
                self.detail_id, path, self.scale.value(), background
            ),
            path,
        )

    def _report(self, save: Callable[[], object], path: str) -> None:
        """Runs a save and shows where it went, or the error."""
        try:
            save()
        except (OSError, ValueError) as error:
            self.message.setText(str(error))
        else:
            self.message.setText(f"Saved to {path}")
        self.message.show()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Closes the detail too, and frees the drawing."""
        self._unsubscribe()
        self._unsubscribe = lambda: None
        self.renderer.close()
        if not self._closing and self.detail_id in self.viewer.scene.details:
            self._closing = True
            self.viewer.close_detail(self.detail_id)
        super().closeEvent(event)


def follow_details(parent: QWidget, viewer: Viewer) -> None:
    """Opens a DetailWindow for each detail the viewer opens from now on.

    Args:
        parent: The windows' parent, the main window.
        viewer: The viewer whose details are shown.
    """
    windows: dict[DetailId, DetailWindow] = {}
    # Weakly, so that the viewer, which can outlive the window, does not keep it.
    owner = ref(parent)

    def follow(event: Event) -> None:
        """Opens a window for a detail opened; forgets one closed."""
        window_parent = owner()
        if isinstance(event, DetailOpened) and window_parent is not None:
            window = DetailWindow(viewer, event.detail.id, window_parent)
            windows[event.detail.id] = window
            window.show()
        elif isinstance(event, DetailClosed):
            windows.pop(event.detail.id, None)

    unsubscribe = viewer.scene.subscribe(follow)
    parent.destroyed.connect(lambda *_: unsubscribe())
