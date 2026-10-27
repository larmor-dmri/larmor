"""The inspector's Cursor and Scene tabs; the Item tab is the options form."""

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from larmor.types import Volume
from larmor.viewer import CameraPreset, Category, Convention, OptionName, Viewer
from larmor.viewer.model.camera import PRESETS
from larmor.viewer.model.settings import SettingName, Settings
from larmor.viewer.model.view_state import Layout
from larmor.viewer.qt.inspector import Inspector, InspectorTab, layer_readout


@pytest.fixture
def viewer() -> Viewer:
    viewer = Viewer()
    data = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    viewer.add(Volume(data, np.eye(4)), Category.VOLUMES, name="t1")
    labels = np.zeros((3, 3, 3), np.int16)
    labels[1, 1, 1] = 4
    viewer.add(Volume(labels, np.eye(4)), Category.OVERLAYS, name="parc")
    return viewer


@pytest.fixture
def inspector(qtbot: QtBot, viewer: Viewer) -> Inspector:
    inspector = Inspector(viewer, Settings())
    qtbot.addWidget(inspector)
    inspector.show()
    return inspector


def _overlay(viewer: Viewer) -> int:
    return viewer.scene.in_category(Category.OVERLAYS)[0].id


def test_the_readout_gives_each_visible_items_value_or_label(viewer: Viewer) -> None:
    viewer.move_cursor((1.0, 1.0, 1.0))
    overlay = _overlay(viewer)
    assert layer_readout(viewer) == [("t1", "13"), ("parc", "4")]
    viewer.set(overlay, OptionName.LABELS, True)
    assert layer_readout(viewer)[1] == ("parc", "4 Label 4")
    viewer.set(overlay, OptionName.VISIBLE, False)
    viewer.move_cursor((9.0, 9.0, 9.0))
    assert layer_readout(viewer) == [("t1", "outside")]


def test_the_cursor_tab_follows_the_cursor(
    viewer: Viewer, inspector: Inspector
) -> None:
    inspector.show_tab(InspectorTab.CURSOR)
    rows = inspector.cursor_tab.readout
    viewer.move_cursor((2.0, 2.0, 2.0))
    first = rows.topLevelItem(0)
    assert first is not None and first.text(1) == "26"
    assert "(2.0, 2.0, 2.0)" in inspector.cursor_tab.position.text()


def test_the_labels_list_shows_while_the_overlay_shows_labels(
    viewer: Viewer, inspector: Inspector
) -> None:
    labels = inspector.cursor_tab.labels
    assert labels.isHidden()
    viewer.set(_overlay(viewer), OptionName.LABELS, True)
    inspector.cursor_tab.refresh()
    assert not labels.isHidden()
    assert labels.item_id == _overlay(viewer)


def test_the_scene_tab_sets_the_layout_the_crosshair_and_the_camera(
    viewer: Viewer, inspector: Inspector
) -> None:
    scene = inspector.scene_tab
    box = scene.layout_box
    box.setCurrentIndex(box.findData(Layout.ONE))
    box.activated.emit(box.currentIndex())
    assert viewer.scene.view.layout == Layout.ONE
    scene.crosshair.click()
    assert not viewer.scene.view.crosshair_visible
    scene.camera_buttons[CameraPreset.SAGITTAL].click()
    assert (
        viewer.scene.view.cameras.three_d.forward == (PRESETS[CameraPreset.SAGITTAL][0])
    )
    viewer.set_layout(Layout.GRID)
    assert box.currentData() == Layout.GRID


def test_the_scene_tab_edits_the_scene_settings(qtbot: QtBot, viewer: Viewer) -> None:
    settings = Settings()
    inspector = Inspector(viewer, settings)
    qtbot.addWidget(inspector)
    field = inspector.scene_tab.settings_form._fields[SettingName.CONVENTION]
    field.edited.emit(Convention.LAS)
    assert settings.values.convention == Convention.LAS
