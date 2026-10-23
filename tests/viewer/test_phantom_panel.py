"""The Phantom panel, draw mode in the 2D views, and opening phantoms in the window."""

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QCheckBox, QFormLayout, QLabel
from pytestqt.qtbot import QtBot

from larmor.viewer import TissuePreset, Viewer, ViewKind
from larmor.viewer.model.settings import PhantomLayout, SettingName
from larmor.viewer.model.shortcuts import Command
from larmor.viewer.qt.docks import Panel
from larmor.viewer.qt.field import FieldLabel
from larmor.viewer.qt.main_window import MainWindow
from larmor.viewer.qt.phantom_panel import ChoiceBox, PhantomPanel


@pytest.fixture
def window(qtbot: QtBot) -> MainWindow:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    return window


def test_the_panel_shows_a_hint_until_a_phantom_opens(qtbot: QtBot) -> None:
    viewer = Viewer()
    panel = PhantomPanel(viewer)
    qtbot.addWidget(panel)
    panel.show()
    assert panel.hint.isVisible()
    assert not panel.sections["Tissues"].isVisible()
    viewer.new_phantom()
    assert not panel.hint.isVisible()
    assert panel.sections["Tissues"].isVisible()


def test_a_tissue_field_edits_the_tissue(qtbot: QtBot) -> None:
    viewer = Viewer()
    viewer.new_phantom()
    panel = PhantomPanel(viewer)
    qtbot.addWidget(panel)
    panel.parameter_boxes["proton_density"].setValue(0.5)
    state = viewer.phantom
    assert state is not None
    wm = state.tissue("WM")
    assert wm.preset == TissuePreset.STANDARD_MODEL
    assert wm.value("proton_density") == pytest.approx(0.5)


def test_the_bundle_tissue_and_preset_combos_edit_the_phantom(
    qtbot: QtBot,
) -> None:
    viewer = Viewer()
    viewer.new_phantom()
    viewer.start_bundle()
    for point in [(-10.0, 0.0, 0.0), (10.0, 0.0, 0.0)]:
        viewer.add_bundle_point(point)
    bundle = viewer.finish_bundle()
    panel = PhantomPanel(viewer)
    qtbot.addWidget(panel)
    combo = panel.bundle_table.cellWidget(0, 1)
    assert isinstance(combo, ChoiceBox)
    combo.setCurrentIndex(combo.findText("GM"))
    combo.activated.emit(combo.currentIndex())
    state = viewer.phantom
    assert state is not None and state.bundle(bundle.id).tissue == "GM"
    assert not panel.error.isVisibleTo(panel)
    panel.tissue_list.setCurrentRow(1)  # GM
    panel.preset.setCurrentIndex(panel.preset.findText("Free water (CSF)"))
    panel.preset.activated.emit(panel.preset.currentIndex())
    state = viewer.phantom
    assert state is not None
    assert state.tissue("GM").preset == TissuePreset.FREE_WATER
    assert not panel.error.isVisibleTo(panel)


def test_every_field_and_section_has_help(qtbot: QtBot) -> None:
    viewer = Viewer()
    viewer.new_phantom()
    viewer.start_bundle()  # so the bundle's fields show too
    panel = PhantomPanel(viewer)
    qtbot.addWidget(panel)
    assert all(section.help for section in panel.sections.values())
    missing = []
    for form in panel.findChildren(QFormLayout):
        for row in range(form.rowCount()):
            label = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
            field = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
            if label is None or field is None:
                continue
            widget = label.widget()
            if not isinstance(widget, FieldLabel) or not widget.toolTip():
                missing.append(widget.text() if isinstance(widget, QLabel) else row)
    for box in panel.findChildren(QCheckBox):
        if not box.toolTip():
            missing.append(box.text())
    if not panel.adopt.toolTip():
        missing.append("Use a tractogram")
    assert missing == []


def test_an_empty_phantom_shows_how_to_start(qtbot: QtBot) -> None:
    viewer = Viewer()
    viewer.new_phantom()
    panel = PhantomPanel(viewer)
    qtbot.addWidget(panel)
    assert panel.start_hint.isVisibleTo(panel)
    viewer.start_bundle()
    for point in [(-10.0, 0.0, 0.0), (10.0, 0.0, 0.0)]:
        viewer.add_bundle_point(point)
    viewer.finish_bundle()
    assert not panel.start_hint.isVisibleTo(panel)


def test_the_editor_opens_as_a_window_of_its_own(window: MainWindow) -> None:
    window.new_phantom()
    dock = window.docks.docks[Panel.PHANTOM]
    assert dock.isInFloatingContainer()
    panel = window.show_phantom()
    assert panel.window() is not window
    window.settings.set(SettingName.PHANTOM_LAYOUT, PhantomLayout.ALL)
    assert panel._column_count > 1  # wide enough for sections side by side


def test_a_click_outside_the_grid_is_refused_in_the_status_bar(
    window: MainWindow,
) -> None:
    viewer = window.viewer
    viewer.new_phantom((10.0, 10.0, 10.0), 2.0)
    viewer.set_view_state(viewer.scene.view)
    viewer.zoom(ViewKind.AXIAL, 0.25)
    viewer.start_bundle()
    panel = window.views.panels[ViewKind.AXIAL]
    QTest.mouseClick(
        panel.canvas,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        QPoint(2, 2),
    )
    state = viewer.phantom
    assert state is not None and state.draft is not None
    assert state.draft.points == ()
    assert "outside the phantom's grid" in window.statusBar().currentMessage()


def test_a_click_while_drawing_adds_a_point_in_the_slice(window: MainWindow) -> None:
    viewer = window.viewer
    viewer.new_phantom()
    viewer.start_bundle()
    panel = window.views.panels[ViewKind.AXIAL]
    click = QPoint(panel.canvas.width() // 3, panel.canvas.height() // 3)
    QTest.mouseClick(
        panel.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, click
    )
    state = viewer.phantom
    assert state is not None and state.draft is not None
    [point] = state.draft.points
    under = panel.view.to_world(click.x(), click.y())
    assert np.allclose(point[:2], under[:2])
    assert point[2] == viewer.scene.view.cursor[2]
    assert np.allclose(viewer.scene.view.cursor, point)


def test_enter_is_free_for_other_widgets_outside_draw_mode(window: MainWindow) -> None:
    finish = window.shortcuts[Command.FINISH_BUNDLE]
    assert not finish.isEnabled()
    window.viewer.start_bundle()
    assert finish.isEnabled()
    window.viewer.cancel_bundle()
    assert not finish.isEnabled()


def test_the_window_opens_a_phantom_file_in_the_background(
    window: MainWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    source = Viewer()
    source.new_phantom((30.0, 30.0, 30.0), 2.0)
    source.start_bundle()
    for point in [(-10.0, 0.0, 0.0), (10.0, 0.0, 0.0)]:
        source.add_bundle_point(point)
    source.finish_bundle()
    source.save_phantom(tmp_path / "scene.phantom.json")
    window.open_phantom(tmp_path / "scene.phantom.json")
    qtbot.waitUntil(lambda: window.viewer.phantom is not None, timeout=10_000)
    state = window.viewer.phantom
    assert state is not None and len(state.bundles) == 1
    assert window.docks.is_open(Panel.PHANTOM)
