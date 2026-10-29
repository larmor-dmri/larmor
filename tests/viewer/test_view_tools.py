"""The views' pointer tools: regions, zoom boxes, bundles, keys and the menu."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QShortcut
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from larmor.types import Tractogram, Volume
from larmor.viewer import (
    Category,
    DetailOption,
    FigureBackground,
    RegionRole,
    Tool,
    Viewer,
    ViewKind,
)
from larmor.viewer.model.detail import DetailId
from larmor.viewer.model.options import RegionOptions, TractogramOptions
from larmor.viewer.qt.detail_window import DetailWindow
from larmor.viewer.qt.main_window import MainWindow
from larmor.viewer.qt.views import ViewInput, ViewPanel

LEFT = Qt.MouseButton.LeftButton
NONE = Qt.KeyboardModifier.NoModifier
CTRL = Qt.KeyboardModifier.ControlModifier


@pytest.fixture
def window(shared_window: MainWindow) -> Iterator[MainWindow]:
    viewer = shared_window.viewer
    viewer.region_radius = 5.0
    # 1 mm voxels; ones, so the 3D view's slices are not background.
    viewer.add(Volume(np.ones((32, 32, 32), np.float32), np.eye(4)), Category.VOLUMES)
    points = np.array([[0.0, 0, 0], [31, 31, 31]], np.float32)
    viewer.add(Tractogram(points, np.array([0, 2])), Category.TRACTOGRAMS)
    # The keys reach the window only while it is the active one.
    shared_window.activateWindow()
    assert QTest.qWaitForWindowActive(shared_window)
    yield shared_window
    for detail_window in shared_window.findChildren(DetailWindow):
        detail_window.close()


def _detail_windows(window: MainWindow, detail_id: DetailId) -> list[DetailWindow]:
    return [w for w in window.findChildren(DetailWindow) if w.detail_id == detail_id]


def _panel(window: MainWindow, view: ViewKind) -> ViewPanel:
    return window.views.panels[view]


def _input(window: MainWindow, view: ViewKind) -> ViewInput:
    return window.views.inputs[view]


def _point(panel: ViewPanel, world: tuple[float, float, float]) -> QPoint:
    x, y = panel.view.to_screen(world)
    return QPoint(round(x), round(y))


def _world(panel: ViewPanel, point: QPoint) -> np.ndarray:
    return np.array(panel.view.to_world(point.x(), point.y()))


def _drag(
    panel: ViewPanel,
    start: QPoint,
    end: QPoint,
    modifiers: Qt.KeyboardModifier = NONE,
    steps: int = 4,
) -> None:
    QTest.mousePress(panel.canvas, LEFT, modifiers, start)
    for k in range(1, steps + 1):
        QTest.mouseMove(panel.canvas, start + (end - start) * k / steps)
    QTest.mouseRelease(panel.canvas, LEFT, modifiers, end)


def _regions(viewer: Viewer) -> list[RegionOptions]:
    regions = []
    for region in viewer.scene.in_category(Category.REGIONS):
        assert isinstance(region.options, RegionOptions)
        regions.append(region.options)
    return regions


def _applied(viewer: Viewer) -> tuple[int, ...]:
    (tracts,) = viewer.scene.in_category(Category.TRACTOGRAMS)
    assert isinstance(tracts.options, TractogramOptions)
    return tracts.options.regions


# Spheres and boxes


def test_a_sphere_click_places_the_last_radius_at_the_click_and_applies_it(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    viewer.set_tool(Tool.SPHERE)
    panel = _panel(window, ViewKind.AXIAL)
    click = _point(panel, (20.0, 10.0, 0.0))
    QTest.mouseClick(panel.canvas, LEFT, NONE, click)
    (sphere,) = _regions(viewer)
    expected = _world(panel, click)
    expected[2] = viewer.scene.view.cursor[2]
    np.testing.assert_allclose(sphere.centre, expected)
    assert sphere.radius == 5.0
    (region,) = viewer.scene.in_category(Category.REGIONS)
    assert _applied(viewer) == (region.id,)
    assert viewer.scene.view.selected == region.id


def test_a_sphere_drag_from_the_centre_sets_the_radius_and_says_it(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    viewer.set_tool(Tool.SPHERE)
    notices: list[str] = []
    _input(window, ViewKind.CORONAL).notice.connect(notices.append)
    panel = _panel(window, ViewKind.CORONAL)
    cursor = viewer.scene.view.cursor
    start = _point(panel, cursor)
    end = _point(panel, (cursor[0] + 8.0, cursor[1], cursor[2]))
    _drag(panel, start, end)
    (sphere,) = _regions(viewer)
    assert sphere.radius == pytest.approx(8.0, abs=0.3)
    assert notices[-1].startswith("Radius 8")


def test_a_box_drag_draws_a_rectangle_a_slab_deep(window: MainWindow) -> None:
    viewer = window.viewer
    viewer.set_tool(Tool.BOX)
    panel = _panel(window, ViewKind.AXIAL)
    z = viewer.scene.view.cursor[2]
    _drag(panel, _point(panel, (4.0, 6.0, z)), _point(panel, (14.0, 10.0, z)))
    (box,) = _regions(viewer)
    np.testing.assert_allclose(box.centre, (9.0, 8.0, z), atol=0.3)
    np.testing.assert_allclose(box.size, (10.0, 4.0, 5.0), atol=0.3)


def test_dragging_a_regions_centre_moves_it_and_its_ring_resizes_it(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    cursor = viewer.scene.view.cursor
    sphere = viewer.add_sphere(cursor, 6.0)
    panel = _panel(window, ViewKind.AXIAL)
    target = (cursor[0] + 5.0, cursor[1] + 3.0, cursor[2])
    _drag(panel, _point(panel, cursor), _point(panel, target))
    o = _regions(viewer)[0]
    np.testing.assert_allclose(o.centre, target, atol=0.3)
    assert viewer.scene.view.selected == sphere
    ring = (target[0] + 6.0, target[1], target[2])
    _drag(panel, _point(panel, ring), _point(panel, (target[0] + 9.0, *target[1:])))
    assert _regions(viewer)[0].radius == pytest.approx(9.0, abs=0.3)
    # The cursor did not move: the presses took the region's handles.
    assert viewer.scene.view.cursor == cursor


def test_a_3d_click_places_a_sphere_on_what_is_under_the_pointer(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    viewer.set_tool(Tool.SPHERE)
    panel = _panel(window, ViewKind.THREE_D)
    panel.canvas.force_draw()
    refused: list[str] = []
    _input(window, ViewKind.THREE_D).refused.connect(refused.append)
    middle = QPoint(panel.canvas.width() // 2, panel.canvas.height() // 2)
    QTest.mouseClick(panel.canvas, LEFT, NONE, middle)
    (sphere,) = _regions(viewer)
    picked = panel.view.pick(middle.x(), middle.y())
    assert picked is not None
    np.testing.assert_allclose(sphere.centre, picked)
    QTest.mouseClick(panel.canvas, LEFT, NONE, QPoint(1, 1))
    assert len(_regions(viewer)) == 1
    assert refused and refused[-1].startswith("Nothing is under the pointer")


def test_the_label_tool_adds_the_label_clicked_and_shift_click_removes_it(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    labels = np.zeros((32, 32, 32), np.int32)
    labels[16:, :, :] = 5
    parcellation = viewer.add(Volume(labels, np.eye(4)), Category.OVERLAYS)
    region = viewer.add_label_region(parcellation.id, apply=True)
    viewer.set_tool(Tool.LABEL_REGION)
    panel = _panel(window, ViewKind.AXIAL)
    click = _point(panel, (24.0, 10.0, 0.0))
    QTest.mouseClick(panel.canvas, LEFT, NONE, click)
    (o,) = _regions(viewer)
    assert o.labels_a == (5,)
    QTest.mouseClick(panel.canvas, LEFT, Qt.KeyboardModifier.ShiftModifier, click)
    o = viewer.scene.item(region).options
    assert isinstance(o, RegionOptions)
    assert o.labels_a == ()


# Zoom boxes


def test_a_zoom_box_drag_opens_a_detail_window_snapped_to_the_voxels(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    viewer.set_tool(Tool.ZOOM_BOX)
    panel = _panel(window, ViewKind.CORONAL)
    y = viewer.scene.view.cursor[1]
    _drag(panel, _point(panel, (4.2, y, 6.1)), _point(panel, (9.6, y, 12.3)))
    (detail,) = viewer.scene.details.values()
    assert detail.box == (4, 6, 7, 7)
    windows = _detail_windows(window, detail.id)
    assert len(windows) == 1
    assert windows[0].windowTitle().endswith("7 x 7 voxels")
    # Dragging the box's edge moves the detail by whole voxels.
    edge = _point(panel, (3.5, y, 9.0))
    _drag(panel, edge, _point(panel, (6.5, y, 9.0)))
    assert viewer.scene.details[detail.id].box == (7, 6, 7, 7)
    windows[0].close()
    assert viewer.scene.details == {}


# Drawing bundles


def test_a_freehand_stroke_adds_its_simplified_points(window: MainWindow) -> None:
    viewer = window.viewer
    viewer.start_bundle()
    assert viewer.scene.view.tool == Tool.DRAW_BUNDLE
    panel = _panel(window, ViewKind.AXIAL)
    z = viewer.scene.view.cursor[2]
    _drag(panel, _point(panel, (4.0, 4.0, z)), _point(panel, (24.0, 4.0, z)), steps=8)
    state = viewer.phantom
    assert state is not None and state.draft is not None
    # A straight stroke keeps its two ends.
    np.testing.assert_allclose(
        state.draft.points, [(4.0, 4.0, z), (24.0, 4.0, z)], atol=0.3
    )
    np.testing.assert_allclose(viewer.scene.view.cursor, state.draft.points[-1])


def test_ctrl_click_adds_one_point_and_ctrl_drag_pans_instead(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    viewer.start_bundle()
    panel = _panel(window, ViewKind.AXIAL)
    z = viewer.scene.view.cursor[2]
    QTest.mouseClick(panel.canvas, LEFT, CTRL, _point(panel, (10.0, 12.0, z)))
    _drag(panel, _point(panel, (20.0, 12.0, z)), _point(panel, (25.0, 20.0, z)), CTRL)
    state = viewer.phantom
    assert state is not None and state.draft is not None
    np.testing.assert_allclose(state.draft.points, [(10.0, 12.0, z)], atol=0.3)
    camera = viewer.scene.view.cameras.axial
    assert camera.pan != (0.0, 0.0)


def test_dragging_a_control_point_moves_it_in_the_views_plane(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    viewer.start_bundle()
    viewer.set_bundle_points([(5.0, 5.0, 3.0), (20.0, 20.0, 9.0)])
    panel = _panel(window, ViewKind.AXIAL)
    z = viewer.scene.view.cursor[2]
    _drag(panel, _point(panel, (5.0, 5.0, z)), _point(panel, (8.0, 12.0, z)))
    state = viewer.phantom
    assert state is not None and state.draft is not None
    # The point keeps its own depth.
    np.testing.assert_allclose(state.draft.points[0], (8.0, 12.0, 3.0), atol=0.3)


# Keys and the menu


def test_keys_choose_the_tools_and_escape_returns_to_the_cursor(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    tools = []
    for key in (Qt.Key.Key_O, Qt.Key.Key_B, Qt.Key.Key_Z, Qt.Key.Key_V):
        QTest.keyClick(window, key)
        tools.append(viewer.scene.view.tool)
    assert tools == [Tool.SPHERE, Tool.BOX, Tool.ZOOM_BOX, Tool.CURSOR]
    QTest.keyClick(window, Qt.Key.Key_O)
    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert viewer.scene.view.tool == Tool.CURSOR


def test_tab_cycles_the_role_and_delete_removes_the_selected_region(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    sphere = viewer.add_sphere()
    viewer.select(sphere)
    # As after a click in the view; a panel's widget would keep Tab.
    canvas = _panel(window, ViewKind.AXIAL).canvas
    canvas.setFocus()
    QTest.keyClick(canvas, Qt.Key.Key_Tab)
    assert _regions(viewer)[0].role == RegionRole.EXCLUDE
    QTest.keyClick(window, Qt.Key.Key_Delete)
    assert viewer.scene.in_category(Category.REGIONS) == []


def test_a_right_click_offers_regions_and_a_zoom_box_there(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    panel = _panel(window, ViewKind.SAGITTAL)
    asked: list[tuple[object, object]] = []
    _input(window, ViewKind.SAGITTAL).menu_requested.connect(
        lambda kind, position, _: asked.append((kind, position))
    )
    point = QPointF(_point(panel, (0.0, 10.0, 20.0)))
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        point,
        panel.canvas.mapToGlobal(point),
        Qt.MouseButton.RightButton,
        Qt.MouseButton.RightButton,
        NONE,
    )
    QApplication.sendEvent(panel.canvas, press)
    ((kind, position),) = asked
    assert kind == ViewKind.SAGITTAL
    assert isinstance(position, tuple)
    menu = window.view_menu(position, ViewKind.SAGITTAL)
    texts = [a.text() for a in menu.actions()]
    assert texts == [
        "Add sphere region here",
        "Add box region here",
        "Zoom box from here",
    ]
    menu.actions()[2].trigger()
    (detail,) = viewer.scene.details.values()
    assert detail.box[2:] == (12, 12)


def test_a_closed_detail_window_is_deleted(window: MainWindow, qtbot: QtBot) -> None:
    detail_id = window.viewer.open_detail(ViewKind.AXIAL, (2, 2, 8, 4))
    (detail_window,) = _detail_windows(window, detail_id)
    detail_window.close()
    qtbot.waitUntil(lambda: not _detail_windows(window, detail_id))
    assert detail_id not in window.viewer.scene.details


def test_a_new_detail_window_leaves_the_keys_with_the_main_window(
    window: MainWindow, qtbot: QtBot
) -> None:
    window.viewer.set_tool(Tool.ZOOM_BOX)
    detail_id = window.viewer.open_detail(ViewKind.AXIAL, (2, 2, 8, 4))
    qtbot.wait(20)
    assert QApplication.activeWindow() is window
    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert window.viewer.scene.view.tool == Tool.CURSOR
    (detail_window,) = _detail_windows(window, detail_id)
    keys = {s.key().toString(): s for s in detail_window.findChildren(QShortcut)}
    assert "Ctrl+W" in keys
    keys["Esc"].activated.emit()
    assert detail_id not in window.viewer.scene.details


def test_a_wide_detail_window_fits_the_screen_and_records_its_size(
    window: MainWindow, qtbot: QtBot
) -> None:
    detail_id = window.viewer.open_detail(ViewKind.AXIAL, (0, 10, 32, 4))
    (detail_window,) = _detail_windows(window, detail_id)
    qtbot.waitExposed(detail_window)
    room = detail_window.screen().availableGeometry()
    assert detail_window.width() <= room.width()
    canvas = detail_window.canvas.size()
    assert window.viewer.scene.details[detail_id].size == (
        canvas.width(),
        canvas.height(),
    )
    for bar in (detail_window.toolbar, detail_window.exports):
        assert bar.sizeHint().width() <= detail_window.width()


def test_the_detail_window_s_toolbar_sets_the_detail_and_saves_it(
    window: MainWindow, tmp_path: Path
) -> None:
    viewer = window.viewer
    detail_id = viewer.open_detail(ViewKind.AXIAL, (2, 2, 8, 4))
    (detail_window,) = _detail_windows(window, detail_id)
    detail_window.voxel_grid.trigger()
    detail_window.glyph_detail.setValue(6)
    detail = viewer.scene.details[detail_id]
    assert (detail.voxel_grid, detail.glyph_detail) == (False, 6)
    viewer.set_detail(detail_id, DetailOption.LINKED, False)
    assert not detail_window.linked.isChecked()
    assert detail_window.box.aspect == pytest.approx(2.0)
    detail_window.scale.setValue(1)
    detail_window.save(str(tmp_path / "detail.png"))
    assert (tmp_path / "detail.png").is_file()
    detail_window.save_figure(str(tmp_path / "figure.png"), FigureBackground.BLACK)
    assert (tmp_path / "figure.png").is_file()
    assert detail_window.message.text() == f"Saved to {tmp_path / 'figure.png'}"
