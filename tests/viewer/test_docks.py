import re
from pathlib import Path

import numpy as np
import PySide6QtAds as QtAds
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QSplitter
from pytestqt.qtbot import QtBot

from larmor.types import Volume
from larmor.viewer import Category
from larmor.viewer.model.settings import AppSettings, Settings
from larmor.viewer.model.shortcuts import SHORTCUTS
from larmor.viewer.qt import docks
from larmor.viewer.qt.docks import (
    LAYOUT_FILE,
    MIN_PANEL_WIDTH,
    PANELS,
    Panel,
    Side,
)
from larmor.viewer.qt.labels_panel import LabelsList
from larmor.viewer.qt.main_window import MainWindow

DEFAULT_OPEN = {Panel.LAYERS, Panel.OPTIONS}
CENTRE = QtAds.DockWidgetArea.CenterDockWidgetArea


def _window(qtbot: QtBot, folder: Path | None = None) -> MainWindow:
    settings = Settings(AppSettings(), folder / "viewer.toml") if folder else None
    window = MainWindow(settings=settings)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    return window


@pytest.fixture
def window(qtbot: QtBot) -> MainWindow:
    return _window(qtbot)


def _open(window: MainWindow) -> set[Panel]:
    return {p for p in Panel if window.docks.is_open(p)}


def _assert_default(window: MainWindow) -> None:
    assert _open(window) == DEFAULT_OPEN
    assert all(window.docks.is_pinned(p) for p in Panel)


def test_layers_and_options_start_open_and_pinned_and_their_buttons_checked(
    window: MainWindow,
) -> None:
    _assert_default(window)
    checked = {p for p, a in window.docks.actions.items() if a.isChecked()}
    assert checked == DEFAULT_OPEN


@pytest.mark.parametrize("panel", list(Panel))
def test_a_panel_button_toggles_its_panel_and_names_it_and_its_key(
    panel: Panel, window: MainWindow
) -> None:
    action = window.docks.actions[panel]
    info = PANELS[panel]
    keys = ", ".join(
        QKeySequence(k).toString(QKeySequence.SequenceFormat.NativeText)
        for k in SHORTCUTS[info.command].keys
    )
    assert action.toolTip() == f"{info.title} ({keys})"
    was_open = window.docks.is_open(panel)
    action.trigger()
    assert window.docks.is_open(panel) is not was_open
    assert action.isChecked() is not was_open
    action.trigger()
    assert window.docks.is_open(panel) is was_open
    assert action.isChecked() is was_open


def test_ctrl_0_hides_the_open_panels_and_brings_back_exactly_those(
    window: MainWindow, qtbot: QtBot
) -> None:
    # Not the default set, so bringing back the start panels would fail.
    window.docks.open(Panel.HELP)
    window.docks.close(Panel.LAYERS)
    window.activateWindow()
    qtbot.waitActive(window)
    QTest.keyClick(window, Qt.Key.Key_0, Qt.KeyboardModifier.ControlModifier)
    assert _open(window) == set()
    QTest.keyClick(window, Qt.Key.Key_0, Qt.KeyboardModifier.ControlModifier)
    assert _open(window) == {Panel.OPTIONS, Panel.HELP}


def test_the_layout_is_saved_on_close_and_restored_by_the_next_window(
    qtbot: QtBot, tmp_path: Path
) -> None:
    first = _window(qtbot, tmp_path)
    first.docks.open(Panel.LABELS)
    first.docks.close(Panel.OPTIONS)
    first.docks.set_pinned(Panel.LAYERS, False)
    opened = _open(first)
    assert Panel.LABELS in opened and Panel.OPTIONS not in opened
    first.close()
    assert (tmp_path / LAYOUT_FILE).is_file()
    second = _window(qtbot, tmp_path)
    assert _open(second) == opened
    assert {p for p in Panel if not second.docks.is_pinned(p)} == {Panel.LAYERS}


def test_a_restored_layout_shows_the_panels_it_opens(
    qtbot: QtBot, tmp_path: Path
) -> None:
    # Labels was closed in the new window before the restore opens it.
    first = _window(qtbot, tmp_path)
    first.docks.open(Panel.LABELS)
    first.close()
    second = _window(qtbot, tmp_path)
    shown = {p for p in Panel if second.docks.docks[p].isVisible()}
    assert shown == DEFAULT_OPEN | {Panel.LABELS}


def test_a_window_never_shown_saves_no_layout(qtbot: QtBot, tmp_path: Path) -> None:
    window = MainWindow(settings=Settings(AppSettings(), tmp_path / "viewer.toml"))
    qtbot.addWidget(window)
    window.close()
    assert not (tmp_path / LAYOUT_FILE).exists()


def test_a_window_without_a_settings_file_saves_no_layout(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    window.close()
    assert window.layout_path is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("garbled", [True, False])
def test_an_unreadable_or_other_version_layout_gives_the_default(
    garbled: bool, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / LAYOUT_FILE
    if garbled:
        path.write_bytes(b"not a layout")
    else:
        other = _window(qtbot, tmp_path)
        other.docks.close(Panel.OPTIONS)
        other.docks.open(Panel.HELP)
        monkeypatch.setattr(docks, "LAYOUT_VERSION", docks.LAYOUT_VERSION + 1)
        other.close()
        monkeypatch.undo()
        assert path.is_file()
    _assert_default(_window(qtbot, tmp_path))


def test_a_layout_from_before_the_floating_editor_is_dropped(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Version 2 kept the Phantom editor docked with the other panels.
    old = _window(qtbot, tmp_path)
    old.docks.open(Panel.PHANTOM)
    old.docks.docks[Panel.PHANTOM].setFloating()
    old.docks.manager.addDockWidget(
        QtAds.BottomDockWidgetArea,
        old.docks.docks[Panel.PHANTOM],
        old.docks.docks[Panel.LAYERS].dockAreaWidget(),
    )
    monkeypatch.setattr(docks, "LAYOUT_VERSION", 2)
    old.close()
    monkeypatch.undo()
    window = _window(qtbot, tmp_path)
    _assert_default(window)
    window.docks.open(Panel.PHANTOM)
    assert window.docks.docks[Panel.PHANTOM].isInFloatingContainer()


def test_a_panel_missing_from_a_restored_layout_opens_docked_on_its_side(
    window: MainWindow,
) -> None:
    # A layout saved before the Signal breakdown panel existed does not name it.
    saved = window.docks.save_state()
    old = re.sub(rb'<Widget Name="signal_breakdown"[^>]*/>', b"", saved)
    assert old != saved
    assert window.docks.restore_state(old)
    window.docks.open(Panel.SIGNAL_BREAKDOWN)
    dock = window.docks.docks[Panel.SIGNAL_BREAKDOWN]
    assert not dock.isInFloatingContainer()
    options = window.docks.docks[Panel.OPTIONS]
    assert dock.dockContainer() is options.dockContainer()
    assert (
        dock.mapTo(window, dock.rect().topLeft()).x()
        >= options.mapTo(window, options.rect().topLeft()).x() - 1
    )


def test_reset_layout_restores_the_default_and_forgets_the_saved_one(
    qtbot: QtBot, tmp_path: Path
) -> None:
    window = _window(qtbot, tmp_path)
    phantom = window.docks.docks[Panel.PHANTOM]
    window.docks.open(Panel.PHANTOM)
    window.docks.manager.addDockWidget(
        QtAds.BottomDockWidgetArea,
        phantom,
        window.docks.docks[Panel.LAYERS].dockAreaWidget(),
    )
    window.docks.open(Panel.HELP)
    assert window.layout_path is not None
    window.docks.save(window.layout_path)
    window.reset_layout()
    _assert_default(window)
    assert not window.layout_path.exists()
    window.docks.open(Panel.PHANTOM)
    assert phantom.isInFloatingContainer()


def test_choose_labels_opens_the_panel_on_that_overlay(window: MainWindow) -> None:
    def overlay(name: str) -> int:
        volume = Volume(np.ones((3, 3, 3), np.int16), np.eye(4))
        return window.viewer.add(volume, Category.OVERLAYS, name=name).id

    overlay("a")
    second = overlay("b")
    panel = window.choose_labels(second)
    assert window.docks.is_open(Panel.LABELS)
    assert isinstance(panel.content, LabelsList)
    assert panel.content.item_id == second


def _areas(window: MainWindow) -> list[QtAds.CDockAreaWidget]:
    m = window.docks.manager
    return [m.dockArea(i) for i in range(m.dockAreaCount())]


def test_no_dock_area_takes_a_panel_as_a_tab(window: MainWindow) -> None:
    assert all(CENTRE not in a.allowedAreas() for a in _areas(window))
    d = window.docks.docks
    window.docks.manager.addDockWidget(
        QtAds.BottomDockWidgetArea, d[Panel.HELP], d[Panel.LAYERS].dockAreaWidget()
    )
    assert all(CENTRE not in a.allowedAreas() for a in _areas(window))


def test_a_layout_with_two_panels_as_tabs_is_refused(qtbot: QtBot) -> None:
    first = _window(qtbot)
    d = first.docks.docks
    first.docks.manager.addDockWidgetTabToArea(
        d[Panel.LABELS], d[Panel.LAYERS].dockAreaWidget()
    )
    assert any(a.dockWidgetsCount() == 2 for a in _areas(first))
    state = first.docks.save_state()
    second = _window(qtbot)
    assert not second.docks.restore_state(state)
    assert all(a.dockWidgetsCount() == 1 for a in _areas(second))
    _assert_default(second)


@pytest.mark.parametrize("pinned", [True, False])
def test_a_panel_closed_by_its_dock_unchecks_its_button(
    pinned: bool, window: MainWindow
) -> None:
    panel = Panel.LABELS
    docks = window.docks
    docks.set_pinned(panel, pinned)
    docks.open(panel)
    assert docks.actions[panel].isChecked() and docks.is_open(panel)
    container = docks.docks[panel].autoHideDockContainer()
    if pinned:
        docks.docks[panel].toggleView(False)
    else:
        container.collapseView(True)
    assert not docks.is_open(panel)
    assert not docks.actions[panel].isChecked()


def test_a_side_dragged_narrow_keeps_its_panel_usable_and_saved(
    qtbot: QtBot, tmp_path: Path
) -> None:
    window = _window(qtbot, tmp_path)
    layers = window.docks.docks[Panel.LAYERS]
    # The window's column splitter: the first horizontal one above Layers.
    split = layers.dockAreaWidget().parentWidget()
    while not (
        isinstance(split, QSplitter)
        and split.orientation() == Qt.Orientation.Horizontal
    ):
        split = split.parentWidget()
    split.moveSplitter(1, 1)
    QApplication.processEvents()
    assert layers.width() >= MIN_PANEL_WIDTH
    window.close()
    assert (tmp_path / LAYOUT_FILE).is_file()


@pytest.mark.parametrize(
    ("method", "panel"),
    [("inspect_fod", Panel.INSPECT_FOD), ("choose_labels", Panel.LABELS)],
)
def test_an_item_panel_refuses_an_item_of_another_category(
    method: str, panel: Panel, window: MainWindow
) -> None:
    volume = Volume(np.zeros((3, 3, 3), np.float32), np.eye(4))
    item = window.viewer.add(volume, Category.VOLUMES, name="t1")
    with pytest.raises(ValueError, match="t1"):
        getattr(window, method)(item.id)
    assert not window.docks.is_open(panel)


def _assert_stripes_shown(window: MainWindow) -> None:
    for stripe in window.docks.stripes.values():
        assert stripe.isVisible()
    # The window fits its screen, so the right stripe is on it.
    screen = window.screen().availableGeometry()
    assert window.minimumSizeHint().width() <= screen.width()


def test_a_layout_too_wide_for_the_screen_gives_the_default_and_both_stripes(
    qtbot: QtBot, tmp_path: Path
) -> None:
    # Every panel side by side needs more width than the 800 px offscreen screen.
    first = _window(qtbot, tmp_path)
    for panel in Panel:
        first.docks.open(panel)
        first.docks.manager.addDockWidget(
            QtAds.RightDockWidgetArea, first.docks.docks[panel]
        )
    first.close()
    second = _window(qtbot, tmp_path)
    _assert_default(second)
    _assert_stripes_shown(second)


def test_a_restore_shows_a_hidden_stripe_again(window: MainWindow) -> None:
    state = window.docks.save_state()
    window.docks.stripes[Side.RIGHT].hide()
    window.docks.restore_state(state)
    _assert_stripes_shown(window)


def test_side_widths_set_the_columns_beside_the_views(
    window: MainWindow, qtbot: QtBot
) -> None:
    window.resize(1200, 800)
    window.docks.set_side_widths(200, 450)
    qtbot.wait(50)
    left = window.docks.docks[Panel.LAYERS].dockAreaWidget().width()
    right = window.docks.docks[Panel.OPTIONS].dockAreaWidget().width()
    assert abs(left - 200) <= 10
    assert abs(right - 450) <= 10
