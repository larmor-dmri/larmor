import gc
import weakref
from collections.abc import Callable
from importlib.resources import files

import pygfx as gfx
import pytest
import shiboken6
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot
from rendercanvas.qt import QRenderWidget
from viewer_helpers import double_click, zeros_volume

from larmor.viewer import APP_NAME, Category, OptionName, Viewer, ViewKind
from larmor.viewer.app import ICON, application
from larmor.viewer.model.events import Event
from larmor.viewer.qt import views
from larmor.viewer.qt.docks import PANELS, Panel, Side
from larmor.viewer.qt.field import SliderField
from larmor.viewer.qt.layers import DELETE_COLUMN, VISIBILITY_COLUMN, LayersPanel
from larmor.viewer.qt.main_window import MainWindow
from larmor.viewer.qt.options_panel import OptionsPanel
from larmor.viewer.qt.views import MAX_FPS, ViewGrid, ViewPanel
from larmor.viewer.render.scene_renderer import SceneRenderer
from larmor.viewer.render.volume_adapter import VolumeAdapter

KINDS: list[ViewKind] = list(ViewKind)


@pytest.fixture
def window(qtbot: QtBot) -> MainWindow:
    window = MainWindow()
    qtbot.addWidget(window)
    return window


@pytest.fixture
def shown(window: MainWindow, qtbot: QtBot) -> MainWindow:
    window.show()
    qtbot.waitExposed(window)
    return window


def _visible(window: MainWindow) -> list[ViewKind]:
    return [k for k, panel in window.views.panels.items() if panel.isVisible()]


def _click_button(panel: ViewPanel) -> None:
    """Clicks the enlarge button the render layer draws in the view's corner."""
    left, top, right, bottom = panel.button.rect()
    centre = QPoint(round((left + right) / 2), round((top + bottom) / 2))
    QTest.mouseClick(
        panel.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, centre
    )


def test_every_view_canvas_draws_up_to_120_frames_a_second(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[object] = []

    class Recorded(QRenderWidget):  # type: ignore[misc]  # untyped base
        def __init__(self, *args: object, **kwargs: object) -> None:
            asked.append(kwargs.get("max_fps"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(views, "QRenderWidget", Recorded)
    window = MainWindow()
    qtbot.addWidget(window)
    assert MAX_FPS == 120.0
    assert asked == [MAX_FPS] * len(KINDS)


def test_layers_and_options_start_docked_either_side_of_the_views(
    shown: MainWindow,
) -> None:
    docks = shown.docks
    assert shown.centralWidget() is docks.manager
    assert [p for p in Panel if docks.is_open(p)] == [Panel.LAYERS, Panel.OPTIONS]
    assert isinstance(docks.widget(Panel.LAYERS), LayersPanel)
    assert isinstance(docks.widget(Panel.OPTIONS), OptionsPanel)
    assert isinstance(shown.views, ViewGrid)
    left = [w.mapTo(shown, QPoint()).x() for w in (shown.layers, shown.views)]
    right = shown.options.mapTo(shown, QPoint()).x()
    assert left[0] < left[1] < right


def test_view_grid_has_the_four_views(window: MainWindow) -> None:
    assert list(window.views.panels) == KINDS
    assert [p.kind for p in window.views.panels.values()] == KINDS


def test_menu_bar_has_file_view_settings_help(window: MainWindow) -> None:
    titles = [a.text().replace("&", "") for a in window.menuBar().actions()]
    assert titles == ["File", "View", "Settings", "Help"]


def test_all_views_are_shown_at_start(shown: MainWindow) -> None:
    assert _visible(shown) == KINDS


@pytest.mark.parametrize("kind", KINDS)
def test_view_button_fills_the_window_and_restores(
    kind: ViewKind, shown: MainWindow
) -> None:
    panel = shown.views.panels[kind]
    _click_button(panel)
    assert _visible(shown) == [kind]
    _click_button(panel)
    assert _visible(shown) == KINDS


@pytest.mark.parametrize("kind", [ViewKind.AXIAL, ViewKind.CORONAL, ViewKind.SAGITTAL])
def test_view_button_leaves_the_cursor_and_shows_the_state_in_its_glyph(
    kind: ViewKind, shown: MainWindow
) -> None:
    # A left press elsewhere in a 2D view would move the cursor there.
    panel = shown.views.panels[kind]
    cursor = shown.viewer.scene.view.cursor
    _click_button(panel)
    assert panel.button.maximised
    _click_button(panel)
    assert not panel.button.maximised
    assert shown.viewer.scene.view.cursor == cursor


def test_another_view_button_while_maximised_switches_to_that_view(
    shown: MainWindow,
) -> None:
    _click_button(shown.views.panels[ViewKind.AXIAL])
    # Hidden panels' buttons cannot be clicked by the user; the action can be called.
    shown.viewer.toggle_maximised(ViewKind.THREE_D)
    assert _visible(shown) == [ViewKind.THREE_D]


def test_button_tooltip_follows_the_state(shown: MainWindow) -> None:
    panel = shown.views.panels[ViewKind.CORONAL]
    before = panel.tooltip()
    _click_button(panel)
    assert panel.tooltip() != before
    _click_button(panel)
    assert panel.tooltip() == before


def test_maximising_from_python_shows_only_that_view_in_restore_state(
    shown: MainWindow,
) -> None:
    shown.viewer.toggle_maximised(ViewKind.CORONAL)
    assert _visible(shown) == [ViewKind.CORONAL]
    assert shown.views.panels[ViewKind.CORONAL].tooltip() == "Show all views"


def test_view_button_updates_the_model(shown: MainWindow) -> None:
    panel = shown.views.panels[ViewKind.SAGITTAL]
    _click_button(panel)
    assert shown.viewer.scene.view.maximised is ViewKind.SAGITTAL
    _click_button(panel)
    assert shown.viewer.scene.view.maximised is None
    assert _visible(shown) == KINDS


_OTHER_CHANGES: dict[str, Callable[[Viewer], None]] = {
    "cursor": lambda v: v.move_cursor((10, 0, 0)),
    "selected": lambda v: v.select(Category.VOLUMES),
    "selected_view": lambda v: v.select_view(ViewKind.SAGITTAL),
}


@pytest.mark.parametrize("change", list(_OTHER_CHANGES))
@pytest.mark.parametrize("maximised", [None, ViewKind.AXIAL])
def test_other_view_state_changes_keep_the_visible_views(
    change: str, maximised: ViewKind | None, shown: MainWindow
) -> None:
    if maximised is not None:
        shown.viewer.toggle_maximised(maximised)
    before = _visible(shown)
    _OTHER_CHANGES[change](shown.viewer)
    assert _visible(shown) == before


def test_window_on_a_maximised_viewer_shows_only_that_view(qtbot: QtBot) -> None:
    viewer = Viewer()
    viewer.toggle_maximised(ViewKind.AXIAL)
    window = MainWindow(viewer)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    assert _visible(window) == [ViewKind.AXIAL]
    assert window.views.panels[ViewKind.AXIAL].tooltip() == "Show all views"


def test_window_uses_the_viewer_it_is_given(qtbot: QtBot) -> None:
    viewer = Viewer()
    window = MainWindow(viewer)
    qtbot.addWidget(window)
    assert window.viewer is viewer


def test_windows_without_a_viewer_get_their_own(qtbot: QtBot) -> None:
    first, second = MainWindow(), MainWindow()
    qtbot.addWidget(first)
    qtbot.addWidget(second)
    assert isinstance(first.viewer, Viewer)
    assert first.viewer is not second.viewer


def test_application_has_name_and_icon(qtbot: QtBot) -> None:
    app = application()
    assert app.applicationName() == APP_NAME
    assert not app.windowIcon().isNull()


def test_icon_file_is_in_the_package() -> None:
    assert (files("larmor.viewer") / "resources" / "icon.svg").is_file()
    assert str(ICON) == str(files("larmor.viewer") / "resources" / "icon.svg")


@pytest.mark.parametrize("kind", KINDS)
def test_each_view_has_a_scene_and_a_camera(kind: ViewKind, window: MainWindow) -> None:
    view = window.views.panels[kind].view
    assert isinstance(view.scene, gfx.Scene)
    camera = (
        gfx.PerspectiveCamera if kind == ViewKind.THREE_D else gfx.OrthographicCamera
    )
    assert type(view.camera) is camera


def test_the_window_renderer_draws_into_the_panels_views(window: MainWindow) -> None:
    assert isinstance(window.renderer, SceneRenderer)
    assert window.renderer.viewer is window.viewer
    panels = window.views.panels
    assert window.renderer.views == {k: p.view for k, p in panels.items()}


def test_a_volume_added_to_the_window_viewer_is_drawn_in_the_2d_views(
    window: MainWindow,
) -> None:
    item = window.viewer.add(zeros_volume(), Category.VOLUMES)
    adapter = window.renderer.adapters[item.id]
    assert isinstance(adapter, VolumeAdapter)
    for kind, obj in adapter.slices.items():
        assert obj in window.views.panels[kind].view.scene.children
    window.viewer.remove(item.id)
    assert window.renderer.adapters == {}


# A viewer outlives its window


def _window_on(viewer: Viewer) -> list[MainWindow]:
    # Not given to qtbot.addWidget, which would keep it alive; held in a list so
    # that deleting it drops the only reference.
    windows = [MainWindow(viewer)]
    viewer.add(zeros_volume(), Category.VOLUMES, name="V")
    viewer.select(Category.VOLUMES)  # so the Options panel has fields
    return windows


def _delete(windows: list[MainWindow], how: str) -> None:
    window = windows.pop()
    if how == "deleteLater":
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        assert not shiboken6.isValid(window)
    del window
    gc.collect()


_CHANGES: dict[str, Callable[[Viewer], object]] = {
    "option": lambda v: v.set(v.scene.items[0].id, OptionName.OPACITY_2D, 0.5),
    "category option": lambda v: v.set(Category.VOLUMES, OptionName.VISIBLE, False),
    "add": lambda v: v.add(zeros_volume(), Category.VOLUMES, name="W"),
    "remove": lambda v: v.remove(v.scene.items[0].id),
    "select": lambda v: v.select(v.scene.items[0].id),
    "toggle_maximised": lambda v: v.toggle_maximised(ViewKind.AXIAL),
}


@pytest.mark.parametrize("change", list(_CHANGES))
@pytest.mark.parametrize("how", ["del", "deleteLater"])
def test_a_deleted_window_stops_listening_to_its_viewer(
    how: str, change: str, qapp: QApplication
) -> None:
    viewer = Viewer()
    before = len(viewer.scene._subscribers)
    first: list[Event] = []
    viewer.scene.subscribe(first.append)
    windows = _window_on(viewer)
    last: list[Event] = []
    viewer.scene.subscribe(last.append)
    _delete(windows, how)
    assert len(viewer.scene._subscribers) == before + 2
    first.clear()
    last.clear()
    _CHANGES[change](viewer)
    assert first
    assert last == first


@pytest.mark.parametrize("how", ["del", "deleteLater"])
def test_a_second_window_on_the_viewer_works_after_the_first_is_deleted(
    how: str, qtbot: QtBot
) -> None:
    viewer = Viewer()
    before = len(viewer.scene._subscribers)
    windows = _window_on(viewer)
    per_window = len(viewer.scene._subscribers) - before
    _delete(windows, how)
    second = MainWindow(viewer)
    qtbot.addWidget(second)
    assert len(viewer.scene._subscribers) == before + per_window
    item = viewer.add(zeros_volume(), Category.VOLUMES, name="W")
    viewer.select(item.id)
    viewer.set(item.id, OptionName.OPACITY_2D, 0.5)
    viewer.toggle_maximised(ViewKind.AXIAL)
    assert second.layers.currentItem() is second.layers._rows[item.id]
    field = second.options._fields[OptionName.OPACITY_2D]
    assert isinstance(field, SliderField)
    assert field.text.text() == "0.5"
    assert [k for k, p in second.views.panels.items() if not p.isHidden()] == [
        ViewKind.AXIAL
    ]


@pytest.mark.parametrize("how", ["del", "deleteLater"])
def test_a_deleted_window_closes_its_renderer_and_is_freed(
    how: str, qapp: QApplication
) -> None:
    viewer = Viewer()
    before = len(viewer.scene._subscribers)
    windows = _window_on(viewer)
    renderer = windows[0].renderer
    item = viewer.scene.items[0]
    assert set(renderer.adapters) == {item.id}
    freed = weakref.ref(windows[0])
    _delete(windows, how)
    assert freed() is None
    assert len(viewer.scene._subscribers) == before
    assert renderer.adapters == {}
    # The closed renderer follows the scene no more.
    viewer.add(zeros_volume(), Category.OVERLAYS)
    assert renderer.adapters == {}


def _click_row(
    window: MainWindow,
    qtbot: QtBot,
    name: str,
    column: int = 0,
    double: bool = False,
) -> None:
    layers = window.layers
    # A panel opened or closed lays Layers out again only once events run.
    qtbot.waitUntil(layers.isVisible)
    flags = Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchRecursive
    (row,) = layers.findItems(name, flags, 0)
    x = (
        layers.header().sectionViewportPosition(column)
        + layers.columnWidth(column) // 2
    )
    at = QPoint(x, layers.visualItemRect(row).center().y())
    if double:
        double_click(layers.viewport(), at)
    else:
        QTest.mouseClick(
            layers.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            at,
        )


RIGHT = [p for p, info in PANELS.items() if info.side is Side.RIGHT]


def _right_open(window: MainWindow) -> set[Panel]:
    return {p for p in RIGHT if window.docks.is_open(p)}


@pytest.mark.parametrize("name", ["t1", "Volumes"])
def test_double_clicking_a_layer_name_shows_options_beside_another_panel(
    name: str, shown: MainWindow, qtbot: QtBot
) -> None:
    shown.viewer.add(zeros_volume(), Category.VOLUMES, name="t1")
    shown.docks.close(Panel.OPTIONS)
    shown.docks.open(Panel.INSPECT_FOD)
    _click_row(shown, qtbot, name, double=True)
    assert _right_open(shown) == {Panel.OPTIONS, Panel.INSPECT_FOD}


def test_clicking_a_layer_name_with_the_right_side_empty_shows_options(
    shown: MainWindow, qtbot: QtBot
) -> None:
    shown.viewer.add(zeros_volume(), Category.VOLUMES, name="t1")
    shown.docks.close(Panel.OPTIONS)
    _click_row(shown, qtbot, "t1")
    assert _right_open(shown) == {Panel.OPTIONS}


def test_clicking_a_layer_name_beside_another_right_panel_keeps_the_panels(
    shown: MainWindow, qtbot: QtBot
) -> None:
    shown.viewer.add(zeros_volume(), Category.VOLUMES, name="t1")
    shown.docks.close(Panel.OPTIONS)
    shown.docks.open(Panel.INSPECT_FOD)
    _click_row(shown, qtbot, "t1")
    assert _right_open(shown) == {Panel.INSPECT_FOD}


@pytest.mark.parametrize("column", [VISIBILITY_COLUMN, DELETE_COLUMN])
@pytest.mark.parametrize("double", [False, True])
def test_clicking_a_layer_eye_or_cross_never_shows_options(
    column: int, double: bool, shown: MainWindow, qtbot: QtBot
) -> None:
    shown.viewer.add(zeros_volume(), Category.VOLUMES, name="t1")
    shown.viewer.add(zeros_volume(), Category.VOLUMES, name="t2")
    shown.docks.close(Panel.OPTIONS)
    _click_row(shown, qtbot, "t1", column, double)
    assert _right_open(shown) == set()
