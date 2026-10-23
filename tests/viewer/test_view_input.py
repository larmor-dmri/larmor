import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QNativeGestureEvent, QPointingDevice, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from larmor.types import Volume
from larmor.viewer import Category, Convention, ViewKind
from larmor.viewer.model.camera import Camera2D, screen_axes
from larmor.viewer.qt.main_window import MainWindow
from larmor.viewer.qt.views import ViewPanel

VIEWS_2D: list[ViewKind] = [ViewKind.AXIAL, ViewKind.CORONAL, ViewKind.SAGITTAL]
AXIS: dict[ViewKind, int] = {
    ViewKind.SAGITTAL: 0,
    ViewKind.CORONAL: 1,
    ViewKind.AXIAL: 2,
}
LEFT = Qt.MouseButton.LeftButton
NONE = Qt.KeyboardModifier.NoModifier
# Ctrl off macOS, Cmd on it (Qt's ControlModifier there); Meta is Ctrl on macOS.
ZOOM_KEYS = [Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.MetaModifier]


@pytest.fixture
def window(qtbot: QtBot) -> MainWindow:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    # 1 mm voxels, so a slice step is 1 mm.
    data = np.zeros((16, 16, 16), np.float32)
    data[0, 0, 0] = 1.0
    window.viewer.add(Volume(data, np.eye(4)), Category.VOLUMES)
    return window


def _panel(window: MainWindow, view: ViewKind) -> ViewPanel:
    return window.views.panels[view]


def _middle(panel: ViewPanel) -> QPoint:
    return QPoint(panel.canvas.width() // 2, panel.canvas.height() // 2)


def _in_plane(view: ViewKind, convention: Convention, point: object) -> np.ndarray:
    right, up = screen_axes(view, convention)
    projected: np.ndarray = np.array([right, up]) @ np.asarray(point)
    return projected


def _under(panel: ViewPanel, point: QPoint | QPointF) -> np.ndarray:
    return np.array(panel.view.to_world(point.x(), point.y()))


def _camera(window: MainWindow, view: ViewKind) -> Camera2D:
    camera = window.viewer.scene.view.cameras.of(view)
    assert isinstance(camera, Camera2D)
    return camera


def _off_centre(panel: ViewPanel) -> QPointF:
    return QPointF(panel.canvas.width() * 0.3, panel.canvas.height() * 0.7)


def _wheel(panel: ViewPanel, delta: int, modifiers: Qt.KeyboardModifier = NONE) -> None:
    position = _off_centre(panel)
    event = QWheelEvent(
        position,
        panel.canvas.mapToGlobal(position),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(panel.canvas, event)


# Selecting a view


def _selected(window: MainWindow) -> dict[ViewKind, object]:
    # The stylesheet draws the frame in the accent through this property.
    return {k: p.property("selected") for k, p in window.views.panels.items()}


@pytest.mark.parametrize("view", list(ViewKind))
def test_a_press_selects_the_view_and_highlights_only_its_frame(
    view: ViewKind, window: MainWindow
) -> None:
    panel = _panel(window, view)
    QTest.mouseClick(panel.canvas, Qt.MouseButton.RightButton, NONE, _middle(panel))
    assert window.viewer.scene.view.selected_view == view
    assert _selected(window) == {k: k == view for k in ViewKind}


def test_a_view_selected_from_python_is_highlighted(window: MainWindow) -> None:
    window.viewer.select_view(ViewKind.SAGITTAL)
    assert _selected(window) == {k: k == ViewKind.SAGITTAL for k in ViewKind}


@pytest.mark.parametrize("view", list(ViewKind))
def test_the_enlarge_button_marks_only_the_maximised_view(
    view: ViewKind, window: MainWindow
) -> None:
    def maximised() -> dict[ViewKind, object]:
        panels = window.views.panels.items()
        return {k: p.button.maximised for k, p in panels}

    window.viewer.toggle_maximised(view)
    assert maximised() == {k: k == view for k in ViewKind}
    window.viewer.toggle_maximised(view)
    assert maximised() == dict.fromkeys(ViewKind, False)


def _send(panel: ViewPanel, kind: QEvent.Type, point: QPointF, held: bool) -> None:
    buttons = LEFT if held else Qt.MouseButton.NoButton
    button = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseMove else LEFT
    global_point = panel.canvas.mapToGlobal(point)
    event = QMouseEvent(kind, point, global_point, button, buttons, NONE)
    QApplication.sendEvent(panel.canvas, event)


def _double_click_and_drag(panel: ViewPanel, start: QPointF, end: QPointF) -> None:
    # Qt's order for a double click; the hand moves before letting go.
    press, release = QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease
    _send(panel, press, start, True)
    _send(panel, release, start, False)
    _send(panel, press, start, True)
    _send(panel, QEvent.Type.MouseButtonDblClick, start, True)
    _send(panel, QEvent.Type.MouseMove, end, True)
    _send(panel, release, end, False)


def test_a_double_click_toggles_maximised_and_the_cursor_stays_where_pressed(
    window: MainWindow,
) -> None:
    panel = _panel(window, ViewKind.AXIAL)
    start = QPointF(_middle(panel))
    end = start + QPointF(40, 30)
    pressed = _under(panel, start)
    _double_click_and_drag(panel, start, end)
    assert window.viewer.scene.view.maximised is ViewKind.AXIAL
    assert panel.button.maximised
    assert np.allclose(window.viewer.scene.view.cursor[:2], pressed[:2])
    # The view fills the window now, so another point is under the press.
    pressed = _under(panel, start)
    _double_click_and_drag(panel, start, end)
    assert window.viewer.scene.view.maximised is None
    assert not panel.button.maximised
    assert np.allclose(window.viewer.scene.view.cursor[:2], pressed[:2])


# Moving the cursor


@pytest.mark.parametrize("convention", list(Convention))
@pytest.mark.parametrize("view", VIEWS_2D)
def test_a_left_click_moves_the_cursor_in_that_slice(
    view: ViewKind, convention: Convention, window: MainWindow
) -> None:
    window.viewer.set_convention(convention)
    panel = _panel(window, view)
    before = np.array(window.viewer.scene.view.cursor)
    click = _middle(panel) + QPoint(40, -30)
    QTest.mouseClick(panel.canvas, LEFT, NONE, click)
    cursor = np.array(window.viewer.scene.view.cursor)
    assert cursor[AXIS[view]] == before[AXIS[view]]
    # To the right of and above the middle of the view, on the screen.
    centre = window.viewer.view_centre(view)
    offset = _in_plane(view, convention, cursor - np.array(centre))
    assert offset[0] > 0 and offset[1] > 0
    assert np.allclose(
        _in_plane(view, convention, cursor),
        _in_plane(view, convention, _under(panel, click)),
    )


def test_a_click_in_the_3d_view_leaves_the_cursor(window: MainWindow) -> None:
    before = window.viewer.scene.view.cursor
    panel = _panel(window, ViewKind.THREE_D)
    QTest.mouseClick(panel.canvas, LEFT, NONE, _middle(panel) + QPoint(40, 30))
    assert window.viewer.scene.view.cursor == before


def test_a_left_drag_moves_the_cursor_with_the_mouse(window: MainWindow) -> None:
    panel = _panel(window, ViewKind.AXIAL)
    start, end = _middle(panel), _middle(panel) + QPoint(-50, 20)
    QTest.mousePress(panel.canvas, LEFT, NONE, start)
    QTest.mouseMove(panel.canvas, end)
    QTest.mouseRelease(panel.canvas, LEFT, NONE, end)
    cursor = window.viewer.scene.view.cursor
    assert np.allclose(cursor[:2], _under(panel, end)[:2])
    # After the release, moving the mouse moves nothing.
    QTest.mouseMove(panel.canvas, start)
    assert window.viewer.scene.view.cursor == cursor


# Scrolling and zooming


@pytest.mark.parametrize("view", VIEWS_2D)
def test_a_wheel_notch_steps_the_slice_of_that_view(
    view: ViewKind, window: MainWindow
) -> None:
    before = np.array(window.viewer.scene.view.cursor)
    _wheel(_panel(window, view), 120)
    _wheel(_panel(window, view), 120)
    _wheel(_panel(window, view), -120)
    expected = before.copy()
    expected[AXIS[view]] += 1.0
    assert np.array_equal(window.viewer.scene.view.cursor, expected)


def test_small_trackpad_deltas_add_up_to_one_slice(window: MainWindow) -> None:
    panel = _panel(window, ViewKind.AXIAL)
    z = window.viewer.scene.view.cursor[2]
    seen = []
    for _ in range(5):
        _wheel(panel, 30)
        seen.append(window.viewer.scene.view.cursor[2] - z)
    assert seen == [0.0, 0.0, 0.0, 1.0, 1.0]


@pytest.mark.parametrize("modifier", ZOOM_KEYS)
@pytest.mark.parametrize("view", VIEWS_2D)
def test_ctrl_wheel_zooms_about_the_mouse(
    view: ViewKind, modifier: Qt.KeyboardModifier, window: MainWindow
) -> None:
    panel = _panel(window, view)
    cursor = window.viewer.scene.view.cursor
    convention = window.viewer.scene.view.convention
    position = _off_centre(panel)
    before = _in_plane(view, convention, _under(panel, position))
    _wheel(panel, 120, modifier)
    assert _camera(window, view).zoom == pytest.approx(1.1)
    assert window.viewer.scene.view.cursor == cursor
    after = _in_plane(view, convention, _under(panel, position))
    assert np.allclose(after, before, atol=1e-9)


def test_a_pinch_zooms_about_the_fingers(window: MainWindow) -> None:
    panel = _panel(window, ViewKind.CORONAL)
    position = QPointF(panel.canvas.width() * 0.25, panel.canvas.height() * 0.25)
    before = _in_plane(ViewKind.CORONAL, Convention.RAS, _under(panel, position))
    event = QNativeGestureEvent(
        Qt.NativeGestureType.ZoomNativeGesture,
        QPointingDevice.primaryPointingDevice(),
        2,
        position,
        position,
        panel.canvas.mapToGlobal(position),
        0.5,
        QPointF(0, 0),
        0,
    )
    QApplication.sendEvent(panel.canvas, event)
    assert _camera(window, ViewKind.CORONAL).zoom == pytest.approx(1.5)
    after = _in_plane(ViewKind.CORONAL, Convention.RAS, _under(panel, position))
    assert np.allclose(after, before, atol=1e-9)


# Panning


@pytest.mark.parametrize("modifier", ZOOM_KEYS)
@pytest.mark.parametrize("view", VIEWS_2D)
def test_ctrl_left_drag_pans_with_the_mouse_and_leaves_the_cursor(
    view: ViewKind, modifier: Qt.KeyboardModifier, window: MainWindow
) -> None:
    panel = _panel(window, view)
    cursor = window.viewer.scene.view.cursor
    convention = window.viewer.scene.view.convention
    start, end = _middle(panel), _middle(panel) + QPoint(60, 25)
    grabbed = _in_plane(view, convention, _under(panel, start))
    QTest.mousePress(panel.canvas, LEFT, modifier, start)
    QTest.mouseMove(panel.canvas, end)
    QTest.mouseRelease(panel.canvas, LEFT, modifier, end)
    assert _camera(window, view).pan != (0.0, 0.0)
    assert window.viewer.scene.view.cursor == cursor
    # The point grabbed is under the mouse where it was let go.
    assert np.allclose(_in_plane(view, convention, _under(panel, end)), grabbed)
