"""The drawing tools of the SymDWI workspace, through the window's mouse and keys."""

from collections.abc import Iterator

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QLineEdit, QWidget

from larmor.viewer import (
    BundleField,
    OptionName,
    Tool,
    TractStyle,
    Viewer,
    ViewKind,
)
from larmor.viewer.model.options import TractogramOptions
from larmor.viewer.model.phantom import GeometrySettings, PhantomState, fullness_of
from larmor.viewer.model.view_state import Workspace
from larmor.viewer.qt.header import WORKSPACE_NAMES
from larmor.viewer.qt.main_window import MainWindow
from larmor.viewer.qt.phantom_panel import PhantomPanel
from larmor.viewer.qt.shortcuts import _widget_keeps
from larmor.viewer.qt.views import ViewPanel
from larmor.viewer.render.tractogram_adapter import TractogramAdapter
from larmor.viewer.render.worker import flush_all

LEFT = Qt.MouseButton.LeftButton
NONE = Qt.KeyboardModifier.NoModifier
CTRL = Qt.KeyboardModifier.ControlModifier


@pytest.fixture
def window(shared_window: MainWindow) -> Iterator[MainWindow]:
    viewer = shared_window.viewer
    viewer.set_workspace(Workspace.SIMULATE)
    viewer.new_phantom((100.0, 100.0, 100.0), 2.0)
    viewer.move_cursor((0.0, 0.0, 0.0))
    shared_window.activateWindow()
    assert QTest.qWaitForWindowActive(shared_window)
    yield shared_window


def _panel(window: MainWindow, view: ViewKind = ViewKind.AXIAL) -> ViewPanel:
    return window.views.panels[view]


def _point(panel: ViewPanel, world: tuple[float, float, float]) -> QPoint:
    x, y = panel.view.to_screen(world)
    return QPoint(round(x), round(y))


def _drag(
    panel: ViewPanel,
    points: list[tuple[float, float, float]],
    modifiers: Qt.KeyboardModifier = NONE,
    steps: int = 5,
) -> None:
    QTest.mousePress(panel.canvas, LEFT, modifiers, _point(panel, points[0]))
    last = np.array(points[0])
    for point in points[1:]:
        for k in range(1, steps + 1):
            at = last + (np.array(point) - last) * k / steps
            QTest.mouseMove(panel.canvas, _point(panel, (at[0], at[1], at[2])))
        last = np.array(point)
    QTest.mouseRelease(panel.canvas, LEFT, modifiers, _point(panel, points[-1]))


def _hover(panel: ViewPanel, world: tuple[float, float, float]) -> None:
    """Moves the pointer over a view with no button down, at once."""
    at = QPointF(_point(panel, world))
    QApplication.sendEvent(
        panel.canvas,
        QMouseEvent(
            QEvent.Type.MouseMove,
            at,
            panel.canvas.mapToGlobal(at),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            NONE,
        ),
    )


def _tool(viewer: Viewer) -> Tool:
    return viewer.scene.view.tool


def _state(viewer: Viewer) -> PhantomState:
    state = viewer.phantom
    assert state is not None
    return state


def _geometry(viewer: Viewer, index: int = -1) -> GeometrySettings:
    geometry = _state(viewer).bundles[index].geometry
    assert geometry is not None
    return geometry


def _bundle(viewer: Viewer, x: float = 10.0) -> int:
    viewer.start_bundle()
    viewer.set_bundle_points([(x, -30.0, 0.0), (x, 30.0, 0.0)])
    return viewer.finish_bundle().id


def _form(window: MainWindow) -> PhantomPanel:
    return window.step_page.panel


def _adapter(window: MainWindow, item_id: int) -> TractogramAdapter:
    adapter = window.renderer.adapters[item_id]
    assert isinstance(adapter, TractogramAdapter)
    return adapter


def test_the_workspace_is_called_symdwi(window: MainWindow) -> None:
    assert WORKSPACE_NAMES[Workspace.SIMULATE] == "SymDWI"
    assert window.header.workspaces[Workspace.SIMULATE].text() == "SymDWI"


def test_the_toolbar_and_keys_choose_the_tools(window: MainWindow) -> None:
    viewer = window.viewer
    window.toolbar.tools[Tool.WIDTH].click()
    assert _tool(viewer) == Tool.WIDTH
    for key, tool in (
        (Qt.Key.Key_B, Tool.BEND),
        (Qt.Key.Key_T, Tool.TWIST),
        (Qt.Key.Key_F, Tool.FAN),
        (Qt.Key.Key_E, Tool.ERASE),
        (Qt.Key.Key_W, Tool.WIDTH),
        (Qt.Key.Key_P, Tool.DRAW_BUNDLE),
    ):
        QTest.keyClick(window, key)
        assert _tool(viewer) == tool
    assert viewer.drawing


def test_b_chooses_the_box_in_dissect(window: MainWindow) -> None:
    viewer = window.viewer
    viewer.set_workspace(Workspace.DISSECT)
    QTest.keyClick(window, Qt.Key.Key_B)
    assert viewer.scene.view.tool == Tool.BOX


def test_the_options_bar_follows_the_tool_and_its_keys(window: MainWindow) -> None:
    viewer = window.viewer
    bar = window.bundle_tool_bar
    holder = bar.parentWidget()
    assert holder is not None
    assert not bar.wanted() and not holder.isVisible()
    viewer.set_tool(Tool.WIDTH)
    assert holder.isVisible()
    assert set(bar.sliders) == {"reach"} and set(bar.choices) == {"mode"}
    slider = bar.sliders["reach"][0]
    before = slider.value()
    QTest.keyClick(window, Qt.Key.Key_BracketRight)
    assert viewer.bundle_tools.reach[Tool.WIDTH] == 30
    assert slider.value() > before
    bar.choices["mode"]["Flat"].click()
    assert viewer.bundle_tools.flat
    bar.choices["mode"]["Round"].click()
    slider.setValue(1000)
    assert viewer.bundle_tools.reach[Tool.WIDTH] == 200
    viewer.bundle_tools.reach[Tool.WIDTH] = 25.0
    viewer.set_tool(Tool.DRAW_BUNDLE)
    assert set(bar.sliders) == {"radius"}
    QTest.keyClick(window, Qt.Key.Key_BracketRight)
    state = viewer.phantom
    assert state is not None and state.draft is not None
    assert state.draft.geometry.radius > GeometrySettings().radius - 1e-9
    viewer.set_workspace(Workspace.VIEW)
    assert not holder.isVisible()


def test_a_stroke_then_enter_makes_a_bundle_and_the_pen_stays(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    QTest.keyClick(window, Qt.Key.Key_P)
    panel = _panel(window)
    _drag(panel, [(-20.0, -20.0, 0.0), (-10.0, 0.0, 0.0), (0.0, 20.0, 0.0)])
    QTest.keyClick(window, Qt.Key.Key_Return)
    assert len(_state(viewer).bundles) == 1
    assert viewer.scene.view.tool == Tool.DRAW_BUNDLE
    assert not _form(window).draw_card.isVisibleTo(_form(window))
    assert viewer.check_phantom() == []


def test_clicks_and_backspace_place_and_take_back_points(window: MainWindow) -> None:
    viewer = window.viewer
    QTest.keyClick(window, Qt.Key.Key_P)
    panel = _panel(window)
    for point in ((-20.0, -20.0, 0.0), (-10.0, 0.0, 0.0), (0.0, 20.0, 0.0)):
        QTest.mouseClick(panel.canvas, LEFT, NONE, _point(panel, point))
        QTest.qWait(1)
    draft = _state(viewer).draft
    assert draft is not None and len(draft.points) == 3
    QTest.keyClick(window, Qt.Key.Key_Backspace)
    draft = _state(viewer).draft
    assert draft is not None and len(draft.points) == 2


def test_a_double_click_with_the_pen_places_points_and_does_not_enlarge(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    QTest.keyClick(window, Qt.Key.Key_P)
    panel = _panel(window)
    point = _point(panel, (10.0, 10.0, 0.0))
    QTest.mouseDClick(panel.canvas, LEFT, NONE, point)
    assert viewer.scene.view.maximised is None


def test_ctrl_click_over_a_bundle_places_a_point(window: MainWindow) -> None:
    viewer = window.viewer
    _bundle(viewer)
    QTest.keyClick(window, Qt.Key.Key_P)
    panel = _panel(window)
    QTest.mouseClick(panel.canvas, LEFT, NONE, _point(panel, (10.0, 5.0, 0.0)))
    draft = _state(viewer).draft
    assert draft is not None and draft.points == ()
    QTest.mouseClick(panel.canvas, LEFT, CTRL, _point(panel, (10.0, 5.0, 0.0)))
    draft = _state(viewer).draft
    assert draft is not None and len(draft.points) == 1


def test_scrolling_during_a_stroke_carries_it_to_another_depth(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    QTest.keyClick(window, Qt.Key.Key_P)
    panel = _panel(window)
    a, b = (-20.0, -20.0, 0.0), (0.0, 0.0, 0.0)
    QTest.mousePress(panel.canvas, LEFT, NONE, _point(panel, a))
    QTest.mouseMove(panel.canvas, _point(panel, b))
    at = QPointF(_point(panel, b))
    QApplication.sendEvent(
        panel.canvas,
        QWheelEvent(
            at,
            panel.canvas.mapToGlobal(at),
            QPoint(0, 0),
            QPoint(0, 5 * 120),
            LEFT,
            NONE,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        ),
    )
    end = _point(panel, (20.0, 20.0, 0.0))
    QTest.mouseMove(panel.canvas, end)
    QTest.mouseRelease(panel.canvas, LEFT, NONE, end)
    draft = _state(viewer).draft
    assert draft is not None
    depths = [p[2] for p in draft.points]
    assert depths[0] == pytest.approx(0.0)
    assert depths[-1] > 5


def test_a_width_stroke_shapes_a_bundle_and_shows_its_reach(window: MainWindow) -> None:
    viewer = window.viewer
    item = _bundle(viewer)
    viewer.set_tool(Tool.WIDTH)
    panel = _panel(window)
    _hover(panel, (11.0, 0.0, 0.0))
    lines = window.renderer.phantom_guides.tool_lines[ViewKind.AXIAL]
    assert lines.reaches.visible
    _drag(panel, [(11.0, 0.0, 0.0), (20.0, 0.0, 0.0)])
    geometry = _geometry(viewer)
    assert geometry.rings is not None and len(geometry.rings) > 2
    # The panel follows the bundle worked on.
    assert viewer.scene.view.selected == item


def test_every_tool_only_turns_the_3d_view(window: MainWindow) -> None:
    viewer = window.viewer
    _bundle(viewer)
    before = _geometry(viewer)
    panel = _panel(window, ViewKind.THREE_D)
    camera = viewer.scene.view.cameras.three_d
    for tool in (Tool.BEND, Tool.WIDTH, Tool.TWIST, Tool.FAN, Tool.ERASE):
        viewer.set_tool(tool)
        centre = QPoint(panel.canvas.width() // 2, panel.canvas.height() // 2)
        QTest.mousePress(panel.canvas, LEFT, NONE, centre)
        QTest.mouseMove(panel.canvas, centre + QPoint(30, 5))
        QTest.mouseRelease(panel.canvas, LEFT, NONE, centre + QPoint(30, 5))
    assert _geometry(viewer) == before
    assert viewer.scene.view.cameras.three_d != camera


def test_the_fan_marks_the_end_and_a_line_lands_it(window: MainWindow) -> None:
    viewer = window.viewer
    _bundle(viewer)
    viewer.move_cursor((0.0, 0.0, 20.0))
    viewer.set_tool(Tool.FAN)
    panel = _panel(window)
    _hover(panel, (10.0, 25.0, 20.0))
    lines = window.renderer.phantom_guides.tool_lines[ViewKind.AXIAL]
    assert lines.circles.visible
    _drag(
        panel,
        [(-5.0, 32.0, 20.0), (10.0, 36.0, 20.0), (25.0, 32.0, 20.0)],
    )
    assert viewer.scene.view.cursor[2] == pytest.approx(0.0)  # the slice went there
    assert _geometry(viewer).landing_end is not None
    assert lines.dashed.visible  # the fan's line


def test_escape_takes_back_a_drag_and_keeps_the_tool(window: MainWindow) -> None:
    viewer = window.viewer
    _bundle(viewer)
    before = _state(viewer).bundles[0]
    viewer.set_tool(Tool.WIDTH)
    panel = _panel(window)
    QTest.mousePress(panel.canvas, LEFT, NONE, _point(panel, (11.0, 0.0, 0.0)))
    QTest.mouseMove(panel.canvas, _point(panel, (20.0, 0.0, 0.0)))
    QTest.keyClick(window, Qt.Key.Key_Escape)
    QTest.mouseMove(panel.canvas, _point(panel, (25.0, 0.0, 0.0)))
    QTest.mouseRelease(panel.canvas, LEFT, NONE, _point(panel, (25.0, 0.0, 0.0)))
    assert _state(viewer).bundles[0] == before
    assert viewer.scene.view.tool == Tool.WIDTH


def test_twist_colours_the_streamlines_by_their_place(window: MainWindow) -> None:
    viewer = window.viewer
    item = _bundle(viewer)
    flush_all()
    adapter = _adapter(window, item)
    direction = adapter.drawn.colours.copy()
    viewer.set_tool(Tool.TWIST)
    flush_all()
    adapter = _adapter(window, item)
    assert not np.array_equal(adapter.drawn.colours, direction)
    # A streamline keeps one colour along it, whatever the direction it runs in.
    first = adapter.drawn.colours[adapter.drawn.offsets[0] : adapter.drawn.offsets[1]]
    assert np.ptp(first, axis=0).max() < 1e-5
    viewer.set_tool(Tool.CURSOR)
    flush_all()
    np.testing.assert_allclose(_adapter(window, item).drawn.colours, direction)


# The fields of a bundle


def test_the_fullness_field_says_a_plain_tube_fills_its_tube(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    item = _bundle(viewer)
    viewer.select(item)
    field = _form(window).fullness
    assert field.value.text() == "Tube"
    assert "fills the voxels inside it" in field.note.text()
    assert not field.reset.isEnabled()
    field.slider.setValue(20)
    bundle = _state(viewer).bundle(item)
    assert bundle.streamline_area is not None
    assert fullness_of(bundle) == pytest.approx(2.0)
    assert field.value.text() == "2.0"
    assert "not as a tube" in field.note.text()
    field.reset.click()
    assert _state(viewer).bundle(item).streamline_area is None
    assert field.value.text() == "Tube"


def test_the_fullness_field_stores_no_area_at_the_default_of_a_shape(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    item = _bundle(viewer)
    viewer.set_bundle_rings(item, [[0.0, 2.0, 2.0, 0.0], [1.0, 3.0, 3.0, 0.0]])
    viewer.select(item)
    field = _form(window).fullness
    assert field.value.text() == "3.0"
    field.slider.setValue(50)
    assert _state(viewer).bundle(item).streamline_area is not None
    field.slider.setValue(30)
    assert _state(viewer).bundle(item).streamline_area is None


def test_the_tangle_field_sets_it_and_says_when_it_is_the_shortest(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    item = _bundle(viewer)
    viewer.set_bundle(item, BundleField.NUM_POINTS, 5)
    viewer.select(item)
    field = _form(window).tangle
    field.slider.setValue(100)  # 5 mm, under what 5 points over 60 mm can show
    geometry = _geometry(viewer)
    assert geometry.tangle_mm == pytest.approx(30.0, abs=0.1)
    assert "min" in field.value.text()
    field.slider.setValue(0)
    assert _geometry(viewer).tangle_mm == 0
    assert field.value.text() == "Off"


def test_the_radius_field_is_for_a_round_bundle(window: MainWindow) -> None:
    viewer = window.viewer
    item = _bundle(viewer)
    viewer.select(item)
    form = _form(window)
    box = form.geometry_boxes[BundleField.RADIUS]
    assert box.isEnabled()
    viewer.set_bundle_rings(item, [[0.0, 2.0, 2.0, 0.0], [1.0, 3.0, 3.0, 0.0]])
    assert not box.isEnabled()


def test_true_size_tubes_draw_at_their_area_outside_the_options_range(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    item = _bundle(viewer)
    viewer.select(item)
    form = _form(window)
    form.true_size.click()
    options = viewer.scene.item(item).options
    assert isinstance(options, TractogramOptions)
    assert options.style == TractStyle.TUBES and options.tube_true_size
    radius = options.drawn_tube_radius
    assert radius == pytest.approx(2.0 / np.sqrt(1000))
    assert radius < 0.1  # within the option's range only by chance of the numbers
    flush_all()
    adapter = _adapter(window, item)
    assert adapter.tubes is not None
    # The first ring of the mesh lies that far from its line's first point.
    positions = adapter.tubes.geometry.positions.data
    first_point = adapter.drawn.points[0]
    assert np.linalg.norm(positions[0] - first_point) == pytest.approx(radius, rel=1e-3)
    # It follows Fullness.
    form.fullness.slider.setValue(100)
    options = viewer.scene.item(item).options
    assert isinstance(options, TractogramOptions)
    assert options.drawn_tube_radius == pytest.approx(2.0 * np.sqrt(10 / 1000))
    form.true_size.click()
    options = viewer.scene.item(item).options
    assert isinstance(options, TractogramOptions)
    assert not options.tube_true_size and options.style == TractStyle.LINES


def test_the_inspector_offers_true_size_tubes(window: MainWindow) -> None:
    viewer = window.viewer
    item = _bundle(viewer)
    viewer.set(item, OptionName.TUBE_TRUE_SIZE, True)
    options = viewer.scene.item(item).options
    assert isinstance(options, TractogramOptions)
    assert options.tube_true_radius > 0


# Undo and redo with the keys


def _spin_line(box: QWidget) -> QLineEdit:
    line = box.findChild(QLineEdit)
    assert isinstance(line, QLineEdit)
    return line


def test_ctrl_z_undoes_a_bundle_setting_with_its_field_focused(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    item = _bundle(viewer)
    viewer.select(item)
    radius = _form(window).geometry_boxes[BundleField.RADIUS]
    assert isinstance(radius, QDoubleSpinBox)
    radius.setFocus()
    radius.setValue(5.0)
    assert _geometry(viewer).radius == 5.0
    QTest.keyClick(window, Qt.Key.Key_Z, CTRL)
    assert _geometry(viewer).radius == GeometrySettings().radius
    assert radius.value() == pytest.approx(GeometrySettings().radius)
    QTest.keyClick(window, Qt.Key.Key_Z, CTRL | Qt.KeyboardModifier.ShiftModifier)
    assert _geometry(viewer).radius == 5.0


def test_a_text_field_being_typed_in_keeps_its_own_undo(window: MainWindow) -> None:
    viewer = window.viewer
    item = _bundle(viewer)
    viewer.set_bundle(item, BundleField.RADIUS, 3.0)
    viewer.select(item)
    box = _form(window).geometry_boxes[BundleField.RADIUS]
    line = _spin_line(box)
    box.setFocus()
    line.selectAll()
    QTest.keyClicks(line, "7")
    assert line.text() == "7" and _keeps(box, CTRL)
    QTest.keyClick(line, Qt.Key.Key_Z, CTRL)
    assert line.text() != "7"  # the field took its own typing back
    assert _geometry(viewer).radius == 3.0
    assert viewer.undo_bundles()  # and the bundle's own history is still there
    assert viewer.redo_bundles()
    # Once typing is taken, the field has none left and the key acts on the bundle.
    box.setFocus()
    line.selectAll()
    QTest.keyClicks(line, "7")
    QTest.keyClick(line, Qt.Key.Key_Return)
    assert _geometry(viewer).radius == 7.0
    assert not _keeps(box, CTRL)
    QTest.keyClick(window, Qt.Key.Key_Z, CTRL)
    assert _geometry(viewer).radius == 3.0


def _keeps(widget: QWidget, modifiers: Qt.KeyboardModifier) -> bool:
    event = QKeyEvent(QEvent.Type.ShortcutOverride, Qt.Key.Key_Z, modifiers)
    return _widget_keeps(widget, event)


def test_a_slider_of_the_form_dragged_with_the_mouse_is_one_step(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    item = _bundle(viewer)
    viewer.select(item)
    slider = _form(window).tangle.slider
    bundle = _state(viewer).bundle(item)
    slider.sliderPressed.emit()
    for at in (10, 20, 30):
        slider.setValue(at)
    slider.sliderReleased.emit()
    assert _geometry(viewer).tangle_mm > 0
    QTest.keyClick(window, Qt.Key.Key_Z, CTRL)
    assert _state(viewer).bundle(item) == bundle


def test_undo_draws_the_views_again_and_keeps_the_pen(window: MainWindow) -> None:
    viewer = window.viewer
    QTest.keyClick(window, Qt.Key.Key_P)
    panel = _panel(window)
    _drag(panel, [(-20.0, -20.0, 0.0), (0.0, 20.0, 0.0)])
    QTest.keyClick(window, Qt.Key.Key_Return)
    item = _state(viewer).bundles[0].item_id
    assert item in window.renderer.adapters
    QTest.keyClick(window, Qt.Key.Key_Z, CTRL)
    assert _state(viewer).bundles == ()
    assert item not in window.renderer.adapters
    assert _tool(viewer) == Tool.DRAW_BUNDLE
    QTest.keyClick(window, Qt.Key.Key_Z, CTRL | Qt.KeyboardModifier.ShiftModifier)
    assert item in window.renderer.adapters
    assert len(_state(viewer).bundles) == 1
