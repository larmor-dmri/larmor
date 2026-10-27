import sys

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QMenu, QWidget
from pytestqt.qtbot import QtBot

from larmor.types import Volume
from larmor.viewer import CameraPreset, Category, Item, OptionName, ViewKind
from larmor.viewer.model.camera import PRESETS, Camera2D, Camera3D
from larmor.viewer.model.shortcuts import (
    MOUSE_CONTROLS,
    SHORTCUTS,
    ZOOM_STEP,
    Command,
    views_text,
)
from larmor.viewer.qt.field import ChoiceField, SliderField
from larmor.viewer.qt.main_window import MainWindow
from larmor.viewer.qt.shortcuts import ShortcutsDialog

Key = Qt.Key
CTRL = Qt.KeyboardModifier.ControlModifier
SHIFT = Qt.KeyboardModifier.ShiftModifier


@pytest.fixture
def window(shared_window: MainWindow) -> MainWindow:
    return shared_window


def _series(window: MainWindow) -> Item:
    # 2 mm voxels, so a slice step is 2 mm.
    data = np.zeros((8, 8, 8, 3), np.float32)
    data[0, 0, 0, :] = 1.0
    return window.viewer.add(
        Volume(data, np.diag([2.0, 2.0, 2.0, 1.0])), Category.VOLUMES, name="dwi"
    )


def _index(window: MainWindow, item: Item) -> object:
    options = window.viewer.scene.item(item.id).options
    return getattr(options, OptionName.VOLUME_INDEX)


def _zoom(window: MainWindow, view: ViewKind) -> float:
    camera = window.viewer.scene.view.cameras.of(view)
    assert isinstance(camera, Camera2D)
    return camera.zoom


def _visible(window: MainWindow) -> list[ViewKind]:
    return [k for k, panel in window.views.panels.items() if panel.isVisible()]


# Keys in the main window


@pytest.mark.parametrize(
    ("key", "view"),
    [
        (Key.Key_A, ViewKind.AXIAL),
        (Key.Key_C, ViewKind.CORONAL),
        (Key.Key_S, ViewKind.SAGITTAL),
        (Key.Key_3, ViewKind.THREE_D),
    ],
)
def test_a_view_key_enlarges_that_view_and_brings_back_all_four(
    key: Qt.Key, view: ViewKind, window: MainWindow
) -> None:
    QTest.keyClick(window, key)
    assert window.viewer.scene.view.maximised == view
    assert _visible(window) == [view]
    QTest.keyClick(window, key)
    assert window.viewer.scene.view.maximised is None
    assert len(_visible(window)) == 4


@pytest.mark.parametrize(("key", "step"), [(Key.Key_Up, 1), (Key.Key_Down, -1)])
@pytest.mark.parametrize(
    ("view", "axis"),
    [(ViewKind.AXIAL, 2), (ViewKind.CORONAL, 1), (ViewKind.SAGITTAL, 0)],
)
def test_up_and_down_step_the_slice_of_the_selected_view(
    view: ViewKind, axis: int, key: Qt.Key, step: int, window: MainWindow
) -> None:
    _series(window)
    window.viewer.select_view(view)
    before = list(window.viewer.scene.view.cursor)
    QTest.keyClick(window, key)
    before[axis] += 2.0 * step
    assert list(window.viewer.scene.view.cursor) == before


def test_left_and_right_step_the_volume(window: MainWindow) -> None:
    item = _series(window)
    QTest.keyClick(window, Key.Key_Right)
    QTest.keyClick(window, Key.Key_Right)
    assert _index(window, item) == 2
    QTest.keyClick(window, Key.Key_Left)
    assert _index(window, item) == 1


@pytest.mark.parametrize(
    ("key", "zoom"),
    [(Key.Key_Plus, ZOOM_STEP), (Key.Key_Equal, ZOOM_STEP), (Key.Key_Minus, 0.8)],
)
def test_ctrl_plus_and_minus_zoom_the_selected_view(
    key: Qt.Key, zoom: float, window: MainWindow
) -> None:
    window.viewer.select_view(ViewKind.CORONAL)
    QTest.keyClick(window, key, CTRL)
    assert _zoom(window, ViewKind.CORONAL) == pytest.approx(zoom)
    assert _zoom(window, ViewKind.AXIAL) == 1.0


def test_the_i_key_toggles_the_information_overlay(window: MainWindow) -> None:
    QTest.keyClick(window, Key.Key_I)
    assert not window.viewer.scene.view.overlay_visible
    QTest.keyClick(window, Key.Key_I)
    assert window.viewer.scene.view.overlay_visible


def test_the_x_key_toggles_the_crosshair(window: MainWindow) -> None:
    QTest.keyClick(window, Key.Key_X)
    assert not window.viewer.scene.view.crosshair_visible
    QTest.keyClick(window, Key.Key_X)
    assert window.viewer.scene.view.crosshair_visible


@pytest.mark.parametrize(
    ("key", "preset"),
    [
        (Key.Key_A, CameraPreset.AXIAL),
        (Key.Key_C, CameraPreset.CORONAL),
        (Key.Key_S, CameraPreset.SAGITTAL),
        (Key.Key_R, None),
    ],
)
def test_the_3d_camera_keys_move_the_3d_camera_from_a_2d_view(
    key: Qt.Key, preset: CameraPreset | None, window: MainWindow
) -> None:
    viewer = window.viewer
    viewer.select_view(ViewKind.CORONAL)
    viewer.rotate_camera(30.0, 20.0)
    QTest.keyClick(window, key, SHIFT)
    expected = Camera3D() if preset is None else Camera3D(*PRESETS[preset])
    assert viewer.scene.view.cameras.three_d == expected


@pytest.mark.parametrize("view", list(ViewKind))
def test_the_r_key_resets_the_selected_view(view: ViewKind, window: MainWindow) -> None:
    viewer = window.viewer
    viewer.select_view(view)
    viewer.zoom(view, 2.0)
    if view == ViewKind.THREE_D:
        viewer.rotate_camera(30.0, 20.0)
    else:
        viewer.pan(view, 5.0, -3.0)
    QTest.keyClick(window, Key.Key_R)
    expected = Camera3D() if view == ViewKind.THREE_D else Camera2D()
    assert viewer.scene.view.cameras.of(view) == expected


# Keys while another widget has the focus


def _focus(widget: QWidget, qtbot: QtBot) -> None:
    widget.setFocus()
    qtbot.waitUntil(lambda: QApplication.focusWidget() is widget)


def test_letters_typed_in_an_options_text_field_stay_text(
    window: MainWindow, qtbot: QtBot
) -> None:
    item = _series(window)
    window.viewer.select(item.id)
    field = window.options._fields[OptionName.OPACITY_2D]
    assert isinstance(field, SliderField)
    _focus(field.text, qtbot)
    field.text.clear()
    QTest.keyClicks(field.text, "acs3i")
    assert field.text.text() == "acs3i"
    assert window.viewer.scene.view.maximised is None
    assert window.viewer.scene.view.overlay_visible


def test_the_arrows_in_an_options_text_field_stay_with_it(
    window: MainWindow, qtbot: QtBot
) -> None:
    item = _series(window)
    window.viewer.select(item.id)
    field = window.options._fields[OptionName.OPACITY_2D]
    assert isinstance(field, SliderField)
    _focus(field.text, qtbot)
    field.text.setText("0.5")
    before = window.viewer.scene.view.cursor
    QTest.keyClick(field.text, Key.Key_Left)
    assert field.text.cursorPosition() == 2
    assert _index(window, item) == 0
    for key in (Key.Key_Up, Key.Key_Down):
        QTest.keyClick(field.text, key)
        assert window.viewer.scene.view.cursor == before


def test_ctrl_keys_still_zoom_from_a_text_field(
    window: MainWindow, qtbot: QtBot
) -> None:
    item = _series(window)
    window.viewer.select(item.id)
    field = window.options._fields[OptionName.OPACITY_2D]
    assert isinstance(field, SliderField)
    _focus(field.text, qtbot)
    QTest.keyClick(field.text, Key.Key_Plus, CTRL)
    assert _zoom(window, ViewKind.AXIAL) == pytest.approx(ZOOM_STEP)


def _widget(window: MainWindow, name: str) -> QWidget:
    if name == "layers":
        return window.layers
    fields = window.options._fields
    if name == "slider":
        field = fields[OptionName.OPACITY_2D]
        assert isinstance(field, SliderField)
        return field.slider
    combo = fields[OptionName.COLORMAP]
    assert isinstance(combo, ChoiceField)
    return combo


@pytest.mark.parametrize("name", ["layers", "slider", "combo"])
def test_the_arrows_still_step_after_the_focus_moves_to_a_panel(
    name: str, window: MainWindow, qtbot: QtBot
) -> None:
    item = _series(window)
    window.viewer.select(item.id)
    widget = _widget(window, name)
    _focus(widget, qtbot)
    z = window.viewer.scene.view.cursor[2]
    QTest.keyClick(widget, Key.Key_Up)
    assert window.viewer.scene.view.cursor[2] == z + 2.0
    QTest.keyClick(widget, Key.Key_Right)
    assert _index(window, item) == 1


# The actions behind the keys


def test_every_registry_entry_has_a_window_action_with_its_keys(
    window: MainWindow,
) -> None:
    assert list(window.shortcuts) == list(Command)
    for command, action in window.shortcuts.items():
        keys = [QKeySequence(k) for k in SHORTCUTS[command].keys]
        assert action.shortcuts() == keys
        assert action in window.actions()
        # A key that works only at times, such as Enter while drawing, starts off.
        enabled = SHORTCUTS[command].active and SHORTCUTS[command].enabled_when is None
        assert action.isEnabled() == enabled


def test_every_registry_key_is_a_distinct_qt_key_sequence() -> None:
    sequences = [QKeySequence(k).toString() for s in SHORTCUTS.values() for k in s.keys]
    assert all(sequences)
    assert len(sequences) == len(set(sequences))


def _menu(window: MainWindow, title: str) -> QMenu:
    for action in window.menuBar().actions():
        if action.text().replace("&", "") == title:
            menu = action.menu()
            assert isinstance(menu, QMenu)
            return menu
    raise AssertionError(f"no {title} menu")


def _entry(menu: QMenu, text: str) -> QAction:
    for action in menu.actions():
        if action.text().replace("&", "") == text:
            return action
    raise AssertionError(f"no {text} entry")


def test_the_view_menu_overlay_item_is_the_shortcut_action(
    window: MainWindow,
) -> None:
    item = _entry(_menu(window, "View"), "Information overlay")
    assert item is window.shortcuts[Command.INFORMATION_OVERLAY]
    assert item.isCheckable()
    item.trigger()
    assert not window.viewer.scene.view.overlay_visible


@pytest.mark.parametrize("visible", [True, False])
def test_the_view_menu_overlay_tick_follows_the_view_state_when_opened(
    visible: bool, window: MainWindow
) -> None:
    menu = _menu(window, "View")
    item = _entry(menu, "Information overlay")
    # The tick is set wrong on purpose: opening the menu reads the view state.
    window.viewer.toggle_overlay()
    if visible:
        window.viewer.toggle_overlay()
    item.setChecked(not visible)
    menu.aboutToShow.emit()
    assert item.isChecked() == visible


def test_the_view_menu_crosshair_item_is_the_shortcut_action(
    window: MainWindow,
) -> None:
    item = _entry(_menu(window, "View"), "Crosshair")
    assert item is window.shortcuts[Command.CROSSHAIR]
    assert item.isCheckable()
    item.trigger()
    assert not window.viewer.scene.view.crosshair_visible


def test_the_view_menu_reset_view_item_is_the_shortcut_action(
    window: MainWindow,
) -> None:
    item = _entry(_menu(window, "View"), "Reset view")
    assert item is window.shortcuts[Command.RESET_VIEW]
    window.viewer.select_view(ViewKind.SAGITTAL)
    window.viewer.zoom(ViewKind.SAGITTAL, 2.0)
    item.trigger()
    assert _zoom(window, ViewKind.SAGITTAL) == 1.0


@pytest.mark.parametrize("visible", [True, False])
def test_the_view_menu_crosshair_tick_follows_the_view_state_when_opened(
    visible: bool, window: MainWindow
) -> None:
    menu = _menu(window, "View")
    item = _entry(menu, "Crosshair")
    # The tick is set wrong on purpose: opening the menu reads the view state.
    window.viewer.toggle_crosshair()
    if visible:
        window.viewer.toggle_crosshair()
    item.setChecked(not visible)
    menu.aboutToShow.emit()
    assert item.isChecked() == visible


@pytest.mark.parametrize(
    ("text", "command"),
    [
        ("Axial", Command.CAMERA_AXIAL),
        ("Coronal", Command.CAMERA_CORONAL),
        ("Sagittal", Command.CAMERA_SAGITTAL),
        ("Reset", Command.CAMERA_RESET),
    ],
)
def test_the_3d_camera_menu_holds_the_shortcut_actions(
    text: str, command: Command, window: MainWindow
) -> None:
    camera = _entry(_menu(window, "View"), "3D camera").menu()
    assert isinstance(camera, QMenu)
    item = _entry(camera, text)
    assert item is window.shortcuts[command]
    assert item.isEnabled()


@pytest.mark.parametrize(
    ("title", "text", "command"),
    [
        ("File", "Open...", Command.OPEN),
        ("File", "Quit", Command.QUIT),
        ("Settings", "Settings", Command.SETTINGS),
    ],
)
def test_the_file_and_settings_menus_hold_the_shortcut_actions(
    title: str, text: str, command: Command, window: MainWindow
) -> None:
    item = _entry(_menu(window, title), text)
    assert item is window.shortcuts[command]
    assert item.isEnabled()


def test_settings_stays_in_its_menu_on_macos(window: MainWindow) -> None:
    item = window.shortcuts[Command.SETTINGS]
    assert item.menuRole() == QAction.MenuRole.NoRole


def test_ctrl_o_opens_the_file_dialog(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def dialog(*args: object) -> tuple[list[str], str]:
        calls.append("open")
        return [], ""

    monkeypatch.setattr(QFileDialog, "getOpenFileNames", dialog)
    QTest.keyClick(window, Key.Key_O, CTRL)
    assert calls == ["open"]


def test_ctrl_q_closes_the_window(window: MainWindow, qtbot: QtBot) -> None:
    try:
        QTest.keyClick(window, Key.Key_Q, CTRL)
        assert not window.isVisible()
    finally:
        # The window is shared with the tests after this one.
        window.show()
        qtbot.waitExposed(window)
        window.activateWindow()
        qtbot.waitActive(window)


# Help, Shortcuts


def _rows(dialog: ShortcutsDialog, mouse: bool = False) -> list[list[str]]:
    table = dialog.mouse_table if mouse else dialog.table
    rows = []
    for row in range(table.rowCount()):
        cells = [table.item(row, column) for column in range(table.columnCount())]
        rows.append([cell.text() if cell is not None else "" for cell in cells])
    return rows


def test_the_shortcuts_list_has_a_row_per_registry_entry(qtbot: QtBot) -> None:
    dialog = ShortcutsDialog()
    qtbot.addWidget(dialog)
    rows = _rows(dialog)
    assert len(rows) == len(SHORTCUTS)
    native = QKeySequence.SequenceFormat.NativeText
    for (keys, description, _), shortcut in zip(rows, SHORTCUTS.values(), strict=True):
        assert keys == ", ".join(
            QKeySequence(k).toString(native) for k in shortcut.keys
        )
        assert description.startswith(shortcut.description)
        assert description.endswith("(not available yet)") == (not shortcut.active)


def test_the_shortcuts_list_says_where_each_shortcut_works(qtbot: QtBot) -> None:
    dialog = ShortcutsDialog()
    qtbot.addWidget(dialog)
    where = dict(zip(SHORTCUTS, (row[2] for row in _rows(dialog)), strict=True))
    assert where[Command.MAXIMISE_AXIAL] == "any"
    assert where[Command.NEXT_SLICE] == "2D views"
    assert where[Command.CAMERA_RESET] == "any"
    for command in (Command.OPEN, Command.SETTINGS, Command.QUIT):
        assert where[command] == "any"


def test_the_shortcuts_list_has_a_row_per_mouse_control(qtbot: QtBot) -> None:
    dialog = ShortcutsDialog()
    qtbot.addWidget(dialog)
    rows = _rows(dialog, mouse=True)
    assert len(rows) == len(MOUSE_CONTROLS)
    for row, control in zip(rows, MOUSE_CONTROLS, strict=True):
        gesture = control.gesture
        if sys.platform == "darwin":
            gesture = gesture.replace("Ctrl", "Cmd")
        assert row == [gesture, control.description, views_text(control.views)]


def test_help_shortcuts_opens_the_list(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[int] = []

    def exec_(dialog: ShortcutsDialog) -> int:
        opened.append(dialog.table.rowCount())
        return 0

    monkeypatch.setattr(ShortcutsDialog, "exec", exec_)
    _entry(_menu(window, "Help"), "Shortcuts").trigger()
    assert opened == [len(SHORTCUTS)]
