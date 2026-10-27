"""The window's header, the views' toolbar and the layouts of the views."""

from pathlib import Path

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QMenu
from pytestqt.qtbot import QtBot
from viewer_helpers import zeros_volume

from larmor.viewer import Category, Tool, ViewKind
from larmor.viewer.model.view_state import Layout, Workspace
from larmor.viewer.qt.header import NO_DATA
from larmor.viewer.qt.main_window import MainWindow


@pytest.fixture
def window(qtbot: QtBot) -> MainWindow:
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()
    qtbot.waitExposed(window)
    return window


def _shown(window: MainWindow) -> list[ViewKind]:
    return [k for k, p in window.views.panels.items() if p.isVisible()]


def _left(window: MainWindow, kind: ViewKind) -> int:
    return window.views.panels[kind].mapTo(window, QPoint()).x()


def test_the_workspace_switch_shows_and_follows_the_workspace(
    window: MainWindow,
) -> None:
    header = window.header
    header.workspaces[Workspace.SIMULATE].click()
    assert window.viewer.scene.view.workspace == Workspace.SIMULATE
    assert header.run_button.isVisible()
    window.viewer.set_workspace(Workspace.VIEW)
    assert header.workspaces[Workspace.VIEW].isChecked()
    assert not header.run_button.isVisible()


def test_the_header_names_the_session_or_else_the_main_volume(
    window: MainWindow, tmp_path: Path
) -> None:
    assert window.header.title.text() == NO_DATA
    window.viewer.add(zeros_volume(), Category.VOLUMES, name="t1.nii.gz")
    assert window.header.title.text() == "t1.nii.gz"
    window.save_session(tmp_path / "study")
    assert window.header.title.text() == "study"


def test_header_run_waits_for_a_phantom_that_can_run(window: MainWindow) -> None:
    viewer = window.viewer
    viewer.set_workspace(Workspace.SIMULATE)
    run = window.header.run_button
    assert not run.isEnabled()
    viewer.new_phantom()
    assert not run.isEnabled() and "bundle" in run.toolTip()
    viewer.start_bundle()
    for point in [(-10.0, 0.0, 0.0), (10.0, 0.0, 0.0)]:
        viewer.add_bundle_point(point)
    viewer.finish_bundle()
    assert run.isEnabled()
    window.job_bar.show_job(None)  # no job: Run stays enabled
    window.header.set_running(True)
    assert not run.isEnabled()


def test_the_toolbar_offers_each_workspaces_tools_and_sets_them(
    window: MainWindow,
) -> None:
    tools = window.toolbar.tools
    shown = {t for t, b in tools.items() if b.isVisible()}
    assert shown == {Tool.CURSOR, Tool.SPHERE, Tool.BOX, Tool.ZOOM_BOX}
    tools[Tool.BOX].click()
    assert window.viewer.scene.view.tool == Tool.BOX
    assert tools[Tool.BOX].isChecked() and not tools[Tool.CURSOR].isChecked()
    window.viewer.set_workspace(Workspace.SIMULATE)
    shown = {t for t, b in tools.items() if b.isVisible()}
    assert shown == {Tool.CURSOR, Tool.DRAW_BUNDLE}
    assert tools[Tool.CURSOR].isChecked()


def test_draw_bundle_chooses_the_tool_and_starts_a_bundle(window: MainWindow) -> None:
    viewer = window.viewer
    viewer.set_workspace(Workspace.SIMULATE)
    window.toolbar.tools[Tool.DRAW_BUNDLE].click()
    assert viewer.drawing and viewer.scene.view.tool == Tool.DRAW_BUNDLE
    assert window.toolbar.tools[Tool.DRAW_BUNDLE].isChecked()


def test_the_crosshair_and_overlay_toggles_follow_the_view_state(
    window: MainWindow,
) -> None:
    toolbar = window.toolbar
    toolbar.crosshair.click()
    assert not window.viewer.scene.view.crosshair_visible
    window.viewer.toggle_overlay()
    assert not toolbar.overlay.isChecked()


def test_one_large_and_three_small_puts_the_selected_view_on_the_left(
    window: MainWindow, qtbot: QtBot
) -> None:
    viewer = window.viewer
    viewer.select_view(ViewKind.SAGITTAL)
    window.toolbar.layouts[Layout.ONE_PLUS_THREE].click()
    qtbot.wait(20)
    assert _shown(window) == list(ViewKind)
    large = window.views.panels[ViewKind.SAGITTAL]
    others = [k for k in ViewKind if k != ViewKind.SAGITTAL]
    assert all(_left(window, k) > _left(window, ViewKind.SAGITTAL) for k in others)
    assert all(large.width() > 2 * window.views.panels[k].width() for k in others)
    tops = sorted(window.views.panels[k].mapTo(window, QPoint()).y() for k in others)
    assert tops[0] < tops[1] < tops[2]


def test_one_view_shows_the_selected_view_and_a_maximised_view_wins(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    viewer.select_view(ViewKind.CORONAL)
    viewer.set_layout(Layout.ONE)
    assert _shown(window) == [ViewKind.CORONAL]
    viewer.set_layout(Layout.ONE_PLUS_THREE)
    viewer.toggle_maximised(ViewKind.THREE_D)
    assert _shown(window) == [ViewKind.THREE_D]
    viewer.toggle_maximised(ViewKind.THREE_D)
    assert _shown(window) == list(ViewKind)


def _submenu(window: MainWindow, title: str) -> QMenu:
    view = next(a.menu() for a in window.menuBar().actions() if a.text() == "&View")
    assert isinstance(view, QMenu)
    menu = next(a.menu() for a in view.actions() if a.text().replace("&", "") == title)
    assert isinstance(menu, QMenu)
    return menu


def test_the_view_menu_sets_the_layout_and_the_workspace(window: MainWindow) -> None:
    layouts = _submenu(window, "Layout")
    next(a for a in layouts.actions() if a.text() == "One view").trigger()
    assert window.viewer.scene.view.layout == Layout.ONE
    workspaces = _submenu(window, "Workspace")
    next(a for a in workspaces.actions() if a.text() == "&Simulate").trigger()
    assert window.viewer.scene.view.workspace == Workspace.SIMULATE
