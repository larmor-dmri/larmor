"""Plain helpers shared by the viewer tests."""

import socket
from typing import TYPE_CHECKING

import numpy as np

from larmor.types import Volume
from larmor.viewer import Item, Viewer
from larmor.viewer.model.events import Event
from larmor.viewer.model.options import TractogramOptions

if TYPE_CHECKING:
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QWidget


def zeros_volume(shape: tuple[int, ...] = (2, 2, 2)) -> Volume:
    return Volume(np.zeros(shape, np.float32), np.eye(4))


def record_events(viewer: Viewer) -> list[Event]:
    events: list[Event] = []
    viewer.scene.subscribe(events.append)
    return events


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


def double_click(widget: QWidget, at: QPoint) -> None:
    # A real double click; QTest.mouseDClick alone sends no first click and no
    # final release, so the view never reports it.
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    left, none = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    QTest.mouseClick(widget, left, none, at)
    QTest.mouseDClick(widget, left, none, at)
    QTest.mouseRelease(widget, left, none, at)


def tract_options(item: Item) -> TractogramOptions:
    o = item.options
    assert isinstance(o, TractogramOptions)
    return o
