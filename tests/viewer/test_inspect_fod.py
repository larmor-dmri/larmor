"""The FOD glyph at the crosshair, in the inspector's Cursor tab."""

from collections.abc import Callable

import numpy as np
import pytest
from PySide6.QtWidgets import QPushButton
from pytestqt.qtbot import QtBot

from larmor.types import Volume
from larmor.viewer import Category, Colouring, Item, ItemId, OptionName
from larmor.viewer.qt.docks import ItemPanel
from larmor.viewer.qt.fod_inspector_panel import FodInspectorView
from larmor.viewer.qt.inspector import InspectorTab
from larmor.viewer.qt.main_window import MainWindow
from larmor.viewer.render import fod_inspector
from larmor.viewer.render.glyphs import Glyphs, voxel_glyph


@pytest.fixture
def window(qtbot: QtBot) -> MainWindow:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    return window


def _fod(window: MainWindow, name: str = "F") -> Item:
    data = np.zeros((2, 2, 2, 6), np.float32)
    data[..., 0] = 1.0
    return window.viewer.add(Volume(data, np.eye(4)), Category.FODS, name=name)


def _buttons(window: MainWindow) -> list[QPushButton]:
    form = window.options.widget()
    assert form is not None
    return [
        b for b in form.findChildren(QPushButton) if b.text() == "Show at the cursor"
    ]


@pytest.mark.parametrize("selected", ["fod", "fods", "volume"])
def test_the_inspect_button_is_shown_for_a_selected_fod_only(
    selected: str, window: MainWindow
) -> None:
    fod = _fod(window)
    volume = window.viewer.add(
        Volume(np.zeros((2, 2, 2), np.float32), np.eye(4)), Category.VOLUMES
    )
    targets: dict[str, ItemId | Category] = {
        "fod": fod.id,
        "fods": Category.FODS,
        "volume": volume.id,
    }
    window.viewer.select(targets[selected])
    assert len(_buttons(window)) == (selected == "fod")


def _panel(window: MainWindow) -> ItemPanel:
    return window.inspector.cursor_tab.fod


def _hide(window: MainWindow) -> None:
    window.inspector.show_tab(InspectorTab.ITEM)


def _show(window: MainWindow) -> None:
    window.inspector.show_tab(InspectorTab.CURSOR)


def _shown(window: MainWindow) -> FodInspectorView:
    content = _panel(window).content
    assert isinstance(content, FodInspectorView)
    return content


def test_the_button_shows_the_cursor_tab_on_its_fod(window: MainWindow) -> None:
    _fod(window, "A")
    second = _fod(window, "B")
    window.viewer.select(second.id)
    [button] = _buttons(window)
    button.click()
    assert window.inspector.currentWidget() is window.inspector.cursor_tab
    assert _shown(window).item_id == second.id


def test_the_panel_follows_the_selected_fod_and_keeps_it_for_other_items(
    window: MainWindow,
) -> None:
    first, second = _fod(window, "A"), _fod(window, "B")
    volume = window.viewer.add(
        Volume(np.zeros((2, 2, 2), np.float32), np.eye(4)), Category.VOLUMES
    )
    window.inspect_fod()
    assert _shown(window).item_id == first.id
    window.viewer.select(second.id)
    assert _shown(window).item_id == second.id
    window.viewer.select(volume.id)
    assert _shown(window).item_id == second.id


def test_the_glyph_shows_only_while_there_is_an_fod(window: MainWindow) -> None:
    panel = window.inspect_fod()
    assert panel.item_id is None and panel.isHidden()
    fod = _fod(window)
    assert _shown(window).item_id == fod.id
    assert not panel.isHidden()


def test_removing_the_fod_shown_moves_to_another_and_stops_the_old_one(
    window: MainWindow,
) -> None:
    kept, removed = _fod(window, "A"), _fod(window, "B")
    window.inspect_fod(removed.id)
    old = _shown(window)
    window.viewer.remove(removed.id)
    assert old.inspector._closed
    assert _shown(window).item_id == kept.id
    window.viewer.remove(kept.id)
    assert _panel(window).content is None


def test_closing_the_main_window_stops_the_inspector(window: MainWindow) -> None:
    window.inspect_fod(_fod(window).id)
    inspector = _shown(window).inspector
    window.close()
    assert inspector._closed


def test_closing_the_main_window_closes_the_inspector_canvas(
    window: MainWindow,
) -> None:
    # A canvas left open is closed again by rendercanvas at quit, after Qt
    # deleted it, which raises.
    window.inspect_fod(_fod(window).id)
    canvas = _shown(window).canvas
    window.close()
    assert canvas.get_closed()


def test_a_hidden_panel_makes_the_glyph_once_when_shown_at_the_new_cursor(
    monkeypatch: pytest.MonkeyPatch, window: MainWindow
) -> None:
    made: list[tuple[int, int, int]] = []
    real: Callable[..., Glyphs] = voxel_glyph

    def record(volume: Volume, voxel: tuple[int, int, int], *args: object) -> Glyphs:
        made.append(voxel)
        return real(volume, voxel, *args)

    monkeypatch.setattr(fod_inspector, "voxel_glyph", record)
    fod = _fod(window)
    window.viewer.move_cursor((0.0, 0.0, 0.0))
    window.inspect_fod(fod.id)
    shown = _shown(window)
    assert made == [(0, 0, 0)]
    made.clear()
    _hide(window)
    window.viewer.move_cursor((1.0, 1.0, 1.0))
    assert made == []
    _show(window)
    assert _shown(window) is shown
    assert made == [(1, 1, 1)]
    assert shown.inspector.voxel == (1, 1, 1)


def test_a_hidden_panel_applies_a_colouring_change_once_shown(
    window: MainWindow,
) -> None:
    fod = _fod(window)
    window.inspect_fod(fod.id)
    material = _shown(window).inspector.mesh.material
    _hide(window)
    window.viewer.set(fod.id, OptionName.COLOURING, Colouring.SINGLE)
    assert material.color_mode == "vertex"
    _show(window)
    assert material.color_mode == "uniform"
