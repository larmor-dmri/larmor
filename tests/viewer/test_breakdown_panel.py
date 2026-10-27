"""The Signal breakdown panel, and the pointer, clicks and keys that drive it."""

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QWidget
from pytestqt.qtbot import QtBot

from larmor.plot import plot_signal_breakdown
from larmor.viewer import ScanField, Viewer, ViewKind
from larmor.viewer.model.breakdown import voxel_at
from larmor.viewer.model.shortcuts import Command
from larmor.viewer.model.view_state import Workspace
from larmor.viewer.qt.breakdown_panel import BreakdownPanel
from larmor.viewer.qt.docks import Panel
from larmor.viewer.qt.main_window import MainWindow

LINE = [(-12.0, 0.0, 0.0), (0.0, 2.0, 0.0), (12.0, 0.0, 0.0)]


def phantom(viewer: Viewer) -> None:
    """A bundle over a GM background, with a short two-shell scan."""
    viewer.new_phantom((30.0, 30.0, 30.0), 2.0)
    viewer.set_scan(ScanField.SHELLS, [[1000, 6], [3000, 6]])
    viewer.set_background_tissue("GM")
    viewer.start_bundle()
    for point in LINE:
        viewer.add_bundle_point(point)
    viewer.finish_bundle()


@pytest.fixture
def panel(qtbot: QtBot) -> BreakdownPanel:
    viewer = Viewer()
    phantom(viewer)
    panel = BreakdownPanel(viewer)
    qtbot.addWidget(panel)
    panel.show()
    viewer.pin_breakdown((0.0, 1.0, 0.0))
    return panel


@pytest.fixture
def window(qtbot: QtBot) -> MainWindow:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    phantom(window.viewer)
    return window


def laid_out(qtbot: QtBot, window: MainWindow) -> QWidget:
    """The axial canvas, once it is shown with its size."""
    canvas: QWidget = window.views.panels[ViewKind.AXIAL].canvas
    qtbot.waitUntil(
        lambda: canvas.isVisible() and canvas.width() > 20 and canvas.height() > 20
    )
    return canvas


def hover(qtbot: QtBot, window: MainWindow) -> tuple[QWidget, QPoint]:
    """Moves the pointer near the canvas middle and waits for the voxel under it."""
    canvas = laid_out(qtbot, window)
    # A move to the pointer's own position sends no event, so move away first.
    QTest.mouseMove(canvas, QPoint(5, 5))
    target = QPoint(canvas.width() // 2 + 3, canvas.height() // 2 + 2)
    QTest.mouseMove(canvas, target)
    viewer = window.viewer
    qtbot.waitUntil(
        lambda: viewer.breakdown is not None and viewer.breakdown.voxel is not None,
        timeout=5000,
    )
    return canvas, target


def bars(ax: Axes) -> list[tuple[float, float, float, float, tuple[float, ...]]]:
    return [
        (p.get_x(), p.get_y(), p.get_height(), p.get_fill(), tuple(p.get_facecolor()))
        for container in ax.containers
        for p in container
    ]


def test_a_narrow_panel_keeps_the_legend_below_the_chart(
    panel: BreakdownPanel, qtbot: QtBot
) -> None:
    assert panel.minimumWidth() >= 340
    panel.resize(panel.minimumWidth(), 720)
    qtbot.waitUntil(lambda: panel.canvas.width() <= panel.minimumWidth())
    panel.canvas.draw()  # type: ignore[no-untyped-call]
    (ax,) = panel.figure.axes
    renderer = panel.canvas.get_renderer()  # type: ignore[no-untyped-call]
    legend = ax.get_legend()
    assert legend is not None
    box = legend.get_window_extent(renderer)
    assert not box.overlaps(ax.get_window_extent(renderer))
    assert box.x1 <= panel.figure.bbox.width + 1


def test_the_panel_draws_the_bars_the_plain_function_draws(
    panel: BreakdownPanel,
) -> None:
    viewer = panel.viewer
    viewer.set_scan(ScanField.TE_MS, [110.0])
    state = viewer.breakdown
    assert state is not None and state.breakdown is not None
    (ax,) = panel.figure.axes
    plain = plot_signal_breakdown(state.breakdown, Figure().subplots())
    assert bars(ax) == bars(plain)
    assert all(fill for *_, fill, _ in bars(ax))  # no outlines of earlier bars
    assert len(bars(ax)) > 3
    assert panel.heading.text().startswith("Voxel (")
    assert panel.rows.rowCount() > 1  # the column names, then the tissues


def test_one_settings_change_gives_one_update(
    panel: BreakdownPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[None] = []
    redraw = panel.redraw

    def counted() -> None:
        calls.append(None)
        redraw()

    monkeypatch.setattr(panel, "redraw", counted)
    panel.viewer.set_scan(ScanField.TR_MS, 4000.0)
    assert len(calls) == 1


def test_the_b_value_list_chooses_the_shell(panel: BreakdownPanel) -> None:
    index = panel.b_value.findData(3000.0)
    panel.b_value.setCurrentIndex(index)
    panel.b_value.activated.emit(index)
    state = panel.viewer.breakdown
    assert state is not None and state.breakdown is not None
    assert state.breakdown.b_value == pytest.approx(3000.0)
    assert panel.direction.count() == 6
    assert panel.direction.currentText().startswith(
        f"Volume {state.breakdown.volume}: ("
    )


def test_the_pointer_and_clicks_drive_the_dock_and_escape_unpins(
    window: MainWindow, qtbot: QtBot
) -> None:
    window.show_breakdown()
    viewer = window.viewer
    canvas, middle = hover(qtbot, window)
    view = window.views.panels[ViewKind.AXIAL].view
    grid = viewer.phantom.grid  # type: ignore[union-attr]
    at = view.to_world(middle.x(), middle.y())
    expected = voxel_at(grid, (at[0], at[1], viewer.scene.view.cursor[2]))
    qtbot.waitUntil(lambda: viewer.breakdown.voxel == expected)  # type: ignore[union-attr]
    state = viewer.breakdown
    assert state is not None and not state.pinned
    QTest.mouseClick(
        canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, middle
    )
    assert viewer.breakdown.pinned  # type: ignore[union-attr]
    guides = window.renderer.phantom_guides.groups[ViewKind.AXIAL]
    assert guides.children[3].visible  # the pinned voxel's outline
    dock = window.docks.docks[Panel.SIGNAL_BREAKDOWN]
    assert "voxel (" in dock.windowTitle()
    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert not viewer.breakdown.pinned  # type: ignore[union-attr]
    assert not guides.children[3].visible


def test_a_right_click_offers_the_breakdown_of_that_voxel(window: MainWindow) -> None:
    position = (6.0, 0.0, 0.0)
    menu = window.view_menu(position)
    (action,) = [a for a in menu.actions() if a.text() == "Signal breakdown here"]
    action.trigger()
    assert window.docks.is_open(Panel.SIGNAL_BREAKDOWN)
    state = window.viewer.breakdown
    assert state is not None and state.pinned
    grid = window.viewer.phantom.grid  # type: ignore[union-attr]
    assert state.voxel == voxel_at(grid, position)
    assert np.isfinite(state.breakdown.signal)  # type: ignore[union-attr]


def test_toggling_the_panel_again_and_again_ends_in_a_sane_state(
    window: MainWindow, qtbot: QtBot
) -> None:
    viewer = window.viewer
    canvas = laid_out(qtbot, window)
    middle = QPoint(canvas.width() // 2, canvas.height() // 2)
    dock = window.docks.docks[Panel.SIGNAL_BREAKDOWN]
    subscribers = None
    for round_ in range(6):
        if round_ % 3 == 0:
            window.shortcuts[Command.SIGNAL_BREAKDOWN].trigger()  # the stripe
        elif round_ % 3 == 1:
            QTest.keyClick(window, Qt.Key.Key_6, Qt.KeyboardModifier.ControlModifier)
        else:
            window.show_breakdown()
        assert window.docks.is_open(Panel.SIGNAL_BREAKDOWN)
        QTest.mouseMove(canvas, middle + QPoint(round_, 0))
        QTest.mouseClick(
            canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, middle
        )
        assert viewer.breakdown is not None and viewer.breakdown.pinned
        QTest.keyClick(window, Qt.Key.Key_Escape)
        assert not viewer.breakdown.pinned
        if subscribers is None:
            subscribers = len(viewer.scene._subscribers)
        if round_ % 2:
            dock.closeDockWidget()  # its close button
        else:
            window.shortcuts[Command.SIGNAL_BREAKDOWN].trigger()
        assert not window.docks.is_open(Panel.SIGNAL_BREAKDOWN)
        assert viewer.breakdown is None
    assert len(window.findChildren(BreakdownPanel)) == 1
    assert len(viewer.scene._subscribers) == subscribers


def test_the_panel_closes_in_the_view_workspace_and_comes_back_in_simulate(
    window: MainWindow,
) -> None:
    window.show_breakdown()
    assert window.viewer.scene.view.workspace == Workspace.SIMULATE
    window.viewer.set_workspace(Workspace.VIEW)
    assert not window.docks.is_open(Panel.SIGNAL_BREAKDOWN)
    assert window.viewer.breakdown is None
    window.viewer.set_workspace(Workspace.SIMULATE)
    assert window.docks.is_open(Panel.SIGNAL_BREAKDOWN)


def test_without_a_voxel_only_the_hint_shows_and_the_chart_follows_the_theme(
    qtbot: QtBot,
) -> None:
    viewer = Viewer()
    phantom(viewer)
    panel = BreakdownPanel(viewer)
    qtbot.addWidget(panel)
    panel.show()
    assert panel.canvas.isHidden() and not panel.hint.isHidden()
    palette = panel.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#202124"))
    panel.setPalette(palette)
    viewer.pin_breakdown((0.0, 1.0, 0.0))
    assert not panel.canvas.isHidden()
    assert to_hex(panel.figure.get_facecolor()) == "#202124"
