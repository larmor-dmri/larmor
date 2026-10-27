"""The panels docked beside the views, and the layout saved between windows."""

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from larmor.viewer.model.settings import AppSettings, Settings
from larmor.viewer.model.view_state import Workspace
from larmor.viewer.qt import docks
from larmor.viewer.qt.docks import LAYOUT_FILE, SIDE_WIDTHS, SIDES, Panel
from larmor.viewer.qt.main_window import MainWindow


def _window(qtbot: QtBot, folder: Path | None = None) -> MainWindow:
    settings = Settings(AppSettings(), folder / "viewer.toml") if folder else None
    window = MainWindow(settings=settings)
    qtbot.addWidget(window)
    window.resize(1400, 800)
    window.show()
    qtbot.waitExposed(window)
    return window


def _widths(window: MainWindow) -> tuple[int, int]:
    left = window.docks.docks[Panel.LEFT].dockAreaWidget().width()
    right = window.docks.docks[Panel.RIGHT].dockAreaWidget().width()
    return left, right


def _near(widths: tuple[int, int], expected: tuple[int, int]) -> bool:
    return all(abs(a - b) <= 12 for a, b in zip(widths, expected, strict=True))


def test_the_side_panels_start_open_at_their_widths(qtbot: QtBot) -> None:
    window = _window(qtbot)
    assert [p for p in Panel if window.docks.is_open(p)] == [Panel.LEFT, Panel.RIGHT]
    assert _near(_widths(window), SIDE_WIDTHS)


def test_ctrl_0_hides_the_side_panels_and_brings_them_back(qtbot: QtBot) -> None:
    window = _window(qtbot)
    window.activateWindow()
    # waitActive can return before the first window of a run is active.
    qtbot.waitUntil(lambda: QApplication.activeWindow() is window)
    QTest.keyClick(window, Qt.Key.Key_0, Qt.KeyboardModifier.ControlModifier)
    assert not window.docks.is_open(Panel.LEFT)
    assert not window.docks.is_open(Panel.RIGHT)
    QTest.keyClick(window, Qt.Key.Key_0, Qt.KeyboardModifier.ControlModifier)
    assert window.docks.is_open(Panel.LEFT) and window.docks.is_open(Panel.RIGHT)


def test_the_layout_is_saved_on_close_and_restored_by_the_next_window(
    qtbot: QtBot, tmp_path: Path
) -> None:
    first = _window(qtbot, tmp_path)
    first.docks.set_side_widths(200, 420)
    qtbot.wait(20)
    first.close()
    assert (tmp_path / LAYOUT_FILE).is_file()
    second = _window(qtbot, tmp_path)
    assert _near(_widths(second), (200, 420))


@pytest.mark.parametrize("garbled", [True, False])
def test_an_unreadable_or_older_layout_gives_the_default(
    garbled: bool, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / LAYOUT_FILE
    if garbled:
        path.write_bytes(b"not a layout")
    else:
        older = _window(qtbot, tmp_path)
        older.docks.set_side_widths(200, 420)
        monkeypatch.setattr(docks, "LAYOUT_VERSION", docks.LAYOUT_VERSION - 1)
        older.close()
        monkeypatch.undo()
    assert _near(_widths(_window(qtbot, tmp_path)), SIDE_WIDTHS)


def test_a_window_never_shown_saves_no_layout(qtbot: QtBot, tmp_path: Path) -> None:
    window = MainWindow(settings=Settings(AppSettings(), tmp_path / "viewer.toml"))
    qtbot.addWidget(window)
    window.close()
    assert not (tmp_path / LAYOUT_FILE).exists()


def test_each_workspace_shows_its_own_panels_either_side(qtbot: QtBot) -> None:
    window = _window(qtbot)
    assert window.scene_panel.isVisible() and window.inspector.isVisible()
    assert window.docks.docks[Panel.LEFT].windowTitle() == "Scene"
    window.viewer.set_workspace(Workspace.SIMULATE)
    assert window.steps_panel.isVisible() and window.step_page.isVisible()
    assert not window.scene_panel.isVisible()
    assert window.docks.docks[Panel.RIGHT].windowTitle() == "Step"


def test_the_side_panels_have_no_title_bar_and_the_breakdown_keeps_its_own(
    qtbot: QtBot,
) -> None:
    window = _window(qtbot)
    for panel in SIDES:
        area = window.docks.docks[panel].dockAreaWidget()
        assert not area.titleBar().isVisible()
    window.show_breakdown()
    area = window.docks.docks[Panel.SIGNAL_BREAKDOWN].dockAreaWidget()
    qtbot.waitUntil(area.titleBar().isVisible)


def test_the_signal_breakdown_opens_only_when_asked_for(
    qtbot: QtBot, tmp_path: Path
) -> None:
    first = _window(qtbot, tmp_path)
    first.viewer.set_workspace(Workspace.SIMULATE)
    assert not first.docks.is_open(Panel.SIGNAL_BREAKDOWN)
    first.show_breakdown()
    first.close()  # saves the layout with the breakdown open
    second = _window(qtbot, tmp_path)
    second.viewer.set_workspace(Workspace.SIMULATE)
    assert not second.docks.is_open(Panel.SIGNAL_BREAKDOWN)
