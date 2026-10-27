"""The scene tree: its rows, eyes, badges, removal, renaming and footer."""

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLineEdit, QMessageBox
from pytestqt.qtbot import QtBot
from viewer_helpers import zeros_volume

from larmor.types import Tractogram, Volume
from larmor.viewer import Category, Colormap, ItemId, OptionName, Viewer
from larmor.viewer.qt.scene_tree import (
    BADGE_COLUMN,
    EYE_COLUMN,
    NAME_COLUMN,
    ScenePanel,
    SceneTree,
)


@pytest.fixture
def viewer() -> Viewer:
    return Viewer()


@pytest.fixture
def panel(qtbot: QtBot, viewer: Viewer) -> ScenePanel:
    panel = ScenePanel(viewer)
    qtbot.addWidget(panel)
    panel.show()
    return panel


@pytest.fixture
def tree(panel: ScenePanel) -> SceneTree:
    return panel.tree


def _tracts() -> Tractogram:
    # Two streamlines: one along x near the origin, one far away.
    points = np.array(
        [[-5.0, 0, 0], [0.0, 0, 0], [5.0, 0, 0], [40.0, 40, 40], [50.0, 40, 40]]
    )
    return Tractogram(points, np.array([0, 3, 5]))


def _click(tree: SceneTree, target: ItemId | Category, column: int) -> None:
    row = tree.row(target)
    rect = tree.visualItemRect(row)
    x = tree.header().sectionViewportPosition(column) + 4
    QTest.mouseClick(
        tree.viewport(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        rect.center().__class__(x, rect.center().y()),
    )


def test_only_categories_with_items_show_with_their_items(
    viewer: Viewer, tree: SceneTree
) -> None:
    t1 = viewer.add(zeros_volume(), Category.VOLUMES, name="t1")
    shown = [c for c in Category if not tree.row(c).isHidden()]
    assert shown == [Category.VOLUMES]
    assert tree.row(t1.id).text(NAME_COLUMN) == "t1"
    assert tree.row(t1.id).toolTip(NAME_COLUMN) == "From Python"


def test_the_eye_shows_and_hides_its_item(viewer: Viewer, tree: SceneTree) -> None:
    mask = viewer.add(zeros_volume(), Category.OVERLAYS)
    _click(tree, mask.id, EYE_COLUMN)
    assert not viewer.scene.item(mask.id).options.visible
    assert tree.row(mask.id).toolTip(EYE_COLUMN) == "Hidden"
    _click(tree, mask.id, EYE_COLUMN)
    assert viewer.scene.item(mask.id).options.visible


def test_an_overlays_badge_shows_its_colormap_threshold_or_labels(
    viewer: Viewer, tree: SceneTree
) -> None:
    data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    mask = viewer.add(Volume(data, np.eye(4)), Category.OVERLAYS)
    badge = tree.row(mask.id)
    viewer.set(mask.id, OptionName.COLORMAP, Colormap.HOT)
    assert badge.text(BADGE_COLUMN) == "hot"
    viewer.set(mask.id, OptionName.THRESHOLD, 2.5)
    assert badge.text(BADGE_COLUMN) == "hot > 2.5"
    viewer.set(mask.id, OptionName.LABELS, True)
    assert badge.text(BADGE_COLUMN) == "labels"


def test_a_tractograms_badge_counts_the_streamlines_its_regions_keep(
    viewer: Viewer, tree: SceneTree
) -> None:
    tracts = viewer.add(_tracts(), Category.TRACTOGRAMS)
    assert tree.row(tracts.id).text(BADGE_COLUMN) == ""
    viewer.add_sphere((0.0, 0.0, 0.0), 3.0, apply=True)
    assert tree.row(tracts.id).text(BADGE_COLUMN) == "1 / 2"


def test_an_fods_badge_names_its_basis(viewer: Viewer, tree: SceneTree) -> None:
    data = np.zeros((2, 2, 2, 6), np.float32)
    fod = viewer.add(Volume(data, np.eye(4)), Category.FODS)
    basis = viewer.scene.item(fod.id).options.basis  # type: ignore[attr-defined]
    assert tree.row(fod.id).text(BADGE_COLUMN) == basis.value


def test_delete_removes_the_selected_item(viewer: Viewer, tree: SceneTree) -> None:
    kept = viewer.add(zeros_volume(), Category.VOLUMES, name="kept")
    gone = viewer.add(zeros_volume(), Category.OVERLAYS, name="gone")
    viewer.select(gone.id)
    tree.setFocus()
    QTest.keyClick(tree, Qt.Key.Key_Delete)
    assert [i.id for i in viewer.scene.items] == [kept.id]


@pytest.mark.parametrize("answer", ["Yes", "No"])
def test_removing_a_category_of_several_items_asks_first(
    answer: str, viewer: Viewer, tree: SceneTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    for _ in range(2):
        viewer.add(zeros_volume(), Category.OVERLAYS)
    button = getattr(QMessageBox.StandardButton, answer)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: button)
    viewer.select(Category.OVERLAYS)
    QTest.keyClick(tree, Qt.Key.Key_Delete)
    assert len(viewer.scene.items) == (0 if answer == "Yes" else 2)


def test_the_context_menu_removes_saves_and_renames(
    viewer: Viewer, tree: SceneTree, qtbot: QtBot
) -> None:
    item = viewer.add(zeros_volume(), Category.VOLUMES, name="t1")
    menu = tree.context_menu(item.id)
    actions = {a.text(): a for a in menu.actions()}
    assert list(actions) == ["Remove", "Save As...", "Rename"]
    with qtbot.waitSignal(tree.save_requested) as saved:
        actions["Save As..."].trigger()
    assert saved.args == [item.id]
    actions["Rename"].trigger()
    editor = tree.findChild(QLineEdit)
    assert editor is not None
    editor.setText("brain")
    tree.commitData(editor)  # as Return or leaving the field does
    assert viewer.scene.item(item.id).name == "brain"
    assert tree.row(item.id).text(NAME_COLUMN) == "brain"
    actions["Remove"].trigger()
    assert viewer.scene.items == []


def test_an_empty_name_is_put_back(viewer: Viewer, tree: SceneTree) -> None:
    item = viewer.add(zeros_volume(), Category.VOLUMES, name="t1")
    tree.row(item.id).setText(NAME_COLUMN, "  ")
    assert viewer.scene.item(item.id).name == "t1"
    assert tree.row(item.id).text(NAME_COLUMN) == "t1"


def test_a_row_selected_in_the_tree_is_selected_in_the_viewer(
    viewer: Viewer, tree: SceneTree
) -> None:
    item = viewer.add(zeros_volume(), Category.VOLUMES)
    tree.setCurrentItem(tree.row(Category.VOLUMES))
    assert viewer.scene.view.selected == Category.VOLUMES
    viewer.select(item.id)
    assert tree.currentItem() is tree.row(item.id)


def test_the_footer_asks_to_open_files_and_sessions(
    panel: ScenePanel, qtbot: QtBot, tmp_path: Path
) -> None:
    with qtbot.waitSignal(panel.open_requested):
        panel.open_button.click()
    menu = panel.session_button.menu()
    assert menu is not None
    texts = [a.text() for a in menu.actions()]
    assert texts == ["Open Session...", "Save Session..."]
    with qtbot.waitSignal(panel.save_session_requested):
        menu.actions()[1].trigger()
