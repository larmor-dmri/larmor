"""The right-click menu of a view, and the wiring of the views' input to a window."""

from collections.abc import Callable, Iterable
from weakref import WeakMethod, ref

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QMainWindow, QMenu, QWidget

from larmor.viewer.model.camera import Vec3
from larmor.viewer.model.detail import MENU_BOX, VIEW_AXIS, centred_box
from larmor.viewer.model.item import ViewKind
from larmor.viewer.model.viewer import Viewer
from larmor.viewer.qt.detail_window import follow_details
from larmor.viewer.qt.views import ViewInput

NOTICE_MS = 5000  # how long the status bar shows a view's notice
# Called with a position to show the Signal breakdown pinned to its voxel.
type ShowBreakdown = Callable[[Vec3], None]


def build_view_menu(
    parent: QWidget,
    viewer: Viewer,
    kind: ViewKind,
    position: Vec3 | None,
    show_breakdown: ShowBreakdown,
) -> QMenu:
    """Returns the right-click menu of a view at a world position.

    It adds a sphere or box region there, opens a detail window around it (in a
    2D view), and, on the phantom, shows its voxel's Signal breakdown.

    Args:
        parent: The menu's parent widget.
        viewer: The viewer the menu acts on.
        kind: The view clicked.
        position: The world point clicked, mm; None in 3D when nothing is there.
        show_breakdown: Shows the breakdown pinned to a position's voxel.

    Returns:
        The menu, deleted once closed.
    """
    menu = QMenu(parent)
    menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    if position is None:
        nothing = menu.addAction("Nothing under the pointer")
        nothing.setEnabled(False)
        return menu
    sphere = menu.addAction("Add sphere region here")
    sphere.triggered.connect(
        lambda: viewer.select(viewer.add_sphere(position, apply=True))
    )
    box = menu.addAction("Add box region here")
    box.triggered.connect(lambda: viewer.select(viewer.add_box(position, apply=True)))
    grid = viewer._reference_grid()
    if kind in VIEW_AXIS and grid is not None:
        zoom = menu.addAction("Zoom box from here")
        square = centred_box(grid, kind, position, MENU_BOX)
        zoom.triggered.connect(lambda: viewer.open_detail(kind, square))
    phantom = viewer.phantom
    if kind in VIEW_AXIS and phantom is not None and phantom.contains(position):
        menu.addSeparator()
        here = menu.addAction("Signal breakdown here")
        here.triggered.connect(lambda: show_breakdown(position))
    return menu


def connect_views(
    window: QMainWindow,
    viewer: Viewer,
    inputs: Iterable[ViewInput],
    show_breakdown: ShowBreakdown,
) -> None:
    """Shows the views' notices in the status bar, their menus, and detail windows.

    Args:
        window: The main window, whose status bar shows the notices.
        viewer: The window's viewer.
        inputs: The views' mouse input.
        show_breakdown: Shows the breakdown pinned to a position's voxel.
    """
    status_bar = window.statusBar()
    # Weakly, so that the views' connections do not keep the window alive.
    owner = ref(window)
    shower = WeakMethod(show_breakdown)

    def popup(kind: ViewKind, position: Vec3 | None, point: QPoint) -> None:
        """Shows the menu of a view at a point on the screen."""
        parent, show = owner(), shower()
        if parent is not None and show is not None:
            build_view_menu(parent, viewer, kind, position, show).popup(point)

    for view_input in inputs:
        view_input.notice.connect(lambda text: status_bar.showMessage(text, NOTICE_MS))
        view_input.menu_requested.connect(popup)
    follow_details(window, viewer)
