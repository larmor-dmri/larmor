from collections.abc import Callable

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QNativeGestureEvent, QPointingDevice, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from larmor.types import Volume
from larmor.viewer import Category, ViewKind
from larmor.viewer.model.options import VolumeOptions
from larmor.viewer.model.settings import SettingName
from larmor.viewer.qt.main_window import MainWindow
from larmor.viewer.qt.views import FULL_RESOLUTION_DELAY, ViewPanel
from larmor.viewer.render import volume_3d
from larmor.viewer.render.volume_adapter import VolumeAdapter

CTRL = Qt.KeyboardModifier.ControlModifier
LEFT = Qt.MouseButton.LeftButton
type Draw = tuple[ViewKind, tuple[int, int]]


def _pan(panel: ViewPanel) -> None:
    start = QPoint(panel.canvas.width() // 2, panel.canvas.height() // 2)
    QTest.mousePress(panel.canvas, LEFT, CTRL, start)
    QTest.mouseMove(panel.canvas, start + QPoint(30, 10))
    QTest.mouseRelease(panel.canvas, LEFT, CTRL, start + QPoint(30, 10))


def _zoom(panel: ViewPanel) -> None:
    position = QPointF(panel.canvas.width() / 2, panel.canvas.height() / 2)
    event = QWheelEvent(
        position,
        panel.canvas.mapToGlobal(position),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        CTRL,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(panel.canvas, event)


def _pinch(panel: ViewPanel) -> None:
    position = QPointF(panel.canvas.width() / 2, panel.canvas.height() / 2)
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


@pytest.fixture
def window(qtbot: QtBot) -> MainWindow:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    window.viewer.add(
        Volume(np.zeros((16, 16, 16), np.float32), np.eye(4)), Category.VOLUMES
    )
    return window


@pytest.fixture
def draws(window: MainWindow, qtbot: QtBot) -> list[Draw]:
    # Each draw ends in one flush; its texture size is the resolution drawn at.
    log: list[Draw] = []
    for kind, panel in window.views.panels.items():
        view = panel.view
        flush = view.renderer.flush

        def logged(
            kind: ViewKind = kind,
            panel: ViewPanel = panel,
            flush: Callable[[], None] = flush,
        ) -> None:
            width, height = panel.canvas.get_physical_size()
            log.append((kind, (width, height)))
            flush()

        view.renderer.flush = logged
    # The first draws of the new volume, before each test starts counting.
    qtbot.waitUntil(lambda: {k for k, _ in log} >= set(ViewKind))
    qtbot.wait(2 * FULL_RESOLUTION_DELAY)
    log.clear()
    return log


@pytest.mark.parametrize("lower", [True, False], ids=["lower", "full"])
@pytest.mark.parametrize(
    "interact", [_pan, _zoom, _pinch], ids=["pan", "zoom", "pinch"]
)
def test_an_interaction_draws_only_its_view_at_half_size_only_if_the_setting_asks(
    interact: Callable[[ViewPanel], None],
    lower: bool,
    window: MainWindow,
    draws: list[Draw],
    qtbot: QtBot,
) -> None:
    window.settings.set(SettingName.LOWER_RESOLUTION, lower)
    panel = window.views.panels[ViewKind.AXIAL]
    full = panel.canvas.get_physical_size()
    half = (round(full[0] / 2), round(full[1] / 2))
    interact(panel)
    qtbot.waitUntil(lambda: (ViewKind.AXIAL, full) in draws)
    # Nothing more once the input has stopped: no idle redraws.
    qtbot.wait(2 * FULL_RESOLUTION_DELAY)
    # At full size the view still draws again when the input stops, so the 3D
    # view's interactive detail (ray steps, streamlines) is replaced.
    during = (ViewKind.AXIAL, half) if lower else (ViewKind.AXIAL, full)
    assert draws == [during, (ViewKind.AXIAL, full)]


def test_the_3d_rendering_draws_again_in_full_detail_after_a_drag_at_full_size(
    window: MainWindow, draws: list[Draw], qtbot: QtBot
) -> None:
    window.settings.set(SettingName.LOWER_RESOLUTION, False)
    (adapter,) = window.renderer.adapters.values()
    assert isinstance(adapter, VolumeAdapter) and adapter.three_d is not None
    # A volume's 3D opacity is 1 by default: its rendering is drawn.
    assert isinstance(adapter.item.options, VolumeOptions)
    assert adapter.item.options.opacity_3d == 1.0
    rendering = adapter.three_d.rendering
    assert rendering is not None and rendering.visible
    panel = window.views.panels[ViewKind.THREE_D]
    # The step each 3D draw used: before_draw sets it, flush ends the draw.
    steps: list[float] = []
    flush = panel.view.renderer.flush

    def logged() -> None:
        steps.append(rendering.material.step_size)
        flush()

    panel.view.renderer.flush = logged
    start = QPoint(panel.canvas.width() // 2, panel.canvas.height() // 2)
    QTest.mousePress(panel.canvas, LEFT, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(panel.canvas, start + QPoint(30, 10))
    QTest.mouseRelease(
        panel.canvas, LEFT, Qt.KeyboardModifier.NoModifier, start + QPoint(30, 10)
    )
    qtbot.waitUntil(lambda: volume_3d.INTERACTIVE_STEP in steps)
    qtbot.wait(2 * FULL_RESOLUTION_DELAY)
    assert steps == [volume_3d.INTERACTIVE_STEP, volume_3d.STEP]
