import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QIcon, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QTreeWidgetItem
from pytestqt.qtbot import QtBot
from viewer_helpers import double_click, record_events, zeros_volume

from larmor.types import Surface, Tractogram, Volume
from larmor.viewer import Category, ItemId, OptionName, Viewer, ViewKind
from larmor.viewer.model.events import Event, ItemRemoved, ViewStateChanged
from larmor.viewer.model.registry import REGISTRY
from larmor.viewer.model.settings import Theme
from larmor.viewer.model.view_state import ViewField
from larmor.viewer.model.viewer import MIXED
from larmor.viewer.qt.layers import (
    DELETE_COLUMN,
    VISIBILITY_COLUMN,
    LayersPanel,
    _target,
)
from larmor.viewer.qt.main_window import MainWindow
from larmor.viewer.qt.style import apply_style, icon_path

type Row = tuple[str, object, str]
type Tree = list[tuple[str, object, list[Row]]]

USER = Qt.ItemDataRole.UserRole
EMPTY: Tree = [(t.label, t.category, []) for t in REGISTRY.values()]


def _peaks() -> Volume:
    return Volume(np.zeros((4, 4, 4, 3), np.float32), np.eye(4))


def _tractogram() -> Tractogram:
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 2, 0]], np.float32)
    return Tractogram(points=points, offsets=np.array([0, 2, 4]))


def _row(row: QTreeWidgetItem) -> Row:
    # Item rows have nothing under them; a nested row would show up as a mismatch.
    assert row.childCount() == 0
    return (row.text(0), row.data(0, USER), row.toolTip(0))


def _children(parent: QTreeWidgetItem) -> list[QTreeWidgetItem]:
    children = [parent.child(i) for i in range(parent.childCount())]
    assert all(c is not None for c in children)
    return [c for c in children if c is not None]


def _tops(panel: LayersPanel) -> list[QTreeWidgetItem]:
    return _children(panel.invisibleRootItem())


def _tree(panel: LayersPanel) -> Tree:
    return [
        (top.text(0), top.data(0, USER), [_row(c) for c in _children(top)])
        for top in _tops(panel)
    ]


def _with(tree: Tree, category: Category, rows: list[Row]) -> Tree:
    return [
        (label, key, rows if key == category else kids) for label, key, kids in tree
    ]


@pytest.fixture(autouse=True)
def _no_message_box(monkeypatch: pytest.MonkeyPatch) -> None:
    # A real dialog would block the run; fail instead.
    def refuse(*args: object) -> None:
        raise AssertionError("a QMessageBox would open")

    monkeypatch.setattr(QMessageBox, "question", refuse)


@pytest.fixture
def panel(viewer: Viewer, qtbot: QtBot) -> LayersPanel:
    panel = LayersPanel(viewer)
    qtbot.addWidget(panel)
    return panel


def test_empty_scene_shows_every_category(panel: LayersPanel) -> None:
    assert _tree(panel) == EMPTY


def test_header_is_hidden(panel: LayersPanel) -> None:
    assert panel.isHeaderHidden()


def test_items_appear_under_their_category_in_order(
    viewer: Viewer, panel: LayersPanel
) -> None:
    t1 = viewer.add(
        zeros_volume((4, 4, 4)),
        Category.VOLUMES,
        name="T1",
        source=Path("/data/t1.nii.gz"),
    )
    fa = viewer.add(zeros_volume((4, 4, 4)), Category.OVERLAYS, name="FA")
    cst = viewer.add(
        _tractogram(), Category.TRACTOGRAMS, name="CST", source=Path("cst.tck")
    )
    t2 = viewer.add(zeros_volume((4, 4, 4)), Category.VOLUMES, name="T2")
    volumes: list[Row] = [
        (t1.name, t1.id, str(Path("/data/t1.nii.gz"))),
        ("T2", t2.id, "From Python"),
    ]
    expected = _with(EMPTY, Category.VOLUMES, volumes)
    expected = _with(expected, Category.OVERLAYS, [("FA", fa.id, "From Python")])
    tracts: list[Row] = [("CST", cst.id, str(Path("cst.tck")))]
    assert _tree(panel) == _with(expected, Category.TRACTOGRAMS, tracts)


def test_panel_on_a_viewer_with_items_shows_them(viewer: Viewer, qtbot: QtBot) -> None:
    t1 = viewer.add(zeros_volume((4, 4, 4)), Category.VOLUMES, name="T1")
    cst = viewer.add(_tractogram(), Category.TRACTOGRAMS, name="CST")
    panel = LayersPanel(viewer)
    qtbot.addWidget(panel)
    expected = _with(EMPTY, Category.VOLUMES, [("T1", t1.id, "From Python")])
    assert _tree(panel) == _with(
        expected, Category.TRACTOGRAMS, [("CST", cst.id, "From Python")]
    )


@pytest.mark.parametrize("removed", [0, 1, 2])
def test_removing_an_item_removes_only_its_row(
    removed: int, viewer: Viewer, panel: LayersPanel
) -> None:
    ids: list[ItemId] = [
        viewer.add(zeros_volume((4, 4, 4)), Category.OVERLAYS, name=name).id
        for name in ("A", "B", "C")
    ]
    viewer.remove(ids[removed])
    rows: list[Row] = [(n, i, "From Python") for n, i in zip("ABC", ids, strict=True)]
    del rows[removed]
    assert _tree(panel) == _with(EMPTY, Category.OVERLAYS, rows)


def test_removing_every_item_keeps_the_categories(
    viewer: Viewer, panel: LayersPanel
) -> None:
    items = [
        viewer.add(zeros_volume((4, 4, 4)), Category.VOLUMES, name="T1"),
        viewer.add(_tractogram(), Category.TRACTOGRAMS, name="CST"),
    ]
    for item in items:
        viewer.remove(item.id)
    assert _tree(panel) == EMPTY


_OTHER_EVENTS: dict[str, Callable[[Viewer, ItemId], None]] = {
    "option": lambda v, i: v.set(i, OptionName.OPACITY_2D, 0.5),
    "category option": lambda v, i: v.set(
        Category.OVERLAYS, OptionName.OPACITY_2D, 0.5
    ),
    "select": lambda v, i: v.select(i),
    "cursor": lambda v, i: v.move_cursor((1.0, 2.0, 3.0)),
    "maximised": lambda v, i: v.toggle_maximised(ViewKind.AXIAL),
}


@pytest.mark.parametrize("change", list(_OTHER_EVENTS))
def test_other_events_keep_the_rows(
    change: str, viewer: Viewer, panel: LayersPanel
) -> None:
    item_id = viewer.add(zeros_volume((4, 4, 4)), Category.OVERLAYS, name="FA").id
    before = _tops(panel)
    expected = _tree(panel)
    _OTHER_EVENTS[change](viewer, item_id)
    assert _tree(panel) == expected
    # Same row objects: the tree was not rebuilt, so nothing the user sees resets.
    assert _tops(panel) == before


@pytest.mark.parametrize("event", ["start", "added", "removed"])
def test_categories_are_expanded(event: str, viewer: Viewer, qtbot: QtBot) -> None:
    item = viewer.add(zeros_volume((4, 4, 4)), Category.VOLUMES, name="T1")
    panel = LayersPanel(viewer)
    qtbot.addWidget(panel)
    for top in _tops(panel):
        top.setExpanded(False)
    if event == "start":
        panel = LayersPanel(viewer)
        qtbot.addWidget(panel)
    elif event == "added":
        viewer.add(_tractogram(), Category.TRACTOGRAMS, name="CST")
    else:
        viewer.remove(item.id)
    assert [top.isExpanded() for top in _tops(panel)] == [True] * len(EMPTY)


def test_window_layers_show_the_window_viewer_items(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    assert isinstance(window.layers, LayersPanel)
    assert window.layers.viewer is window.viewer
    item = window.viewer.add(zeros_volume((4, 4, 4)), Category.VOLUMES, name="T1")
    assert _tree(window.layers) == _with(
        EMPTY, Category.VOLUMES, [("T1", item.id, "From Python")]
    )


# Visibility and selection

EYE = VISIBILITY_COLUMN
TIPS = {True: "Visible", False: "Hidden", MIXED: "Some visible"}


def _all_rows(panel: LayersPanel) -> list[QTreeWidgetItem]:
    return [r for top in _tops(panel) for r in [top, *_children(top)]]


def _find(panel: LayersPanel, target: ItemId | Category) -> QTreeWidgetItem:
    (row,) = [r for r in _all_rows(panel) if r.data(0, USER) == target]
    return row


def _eye(panel: LayersPanel, target: ItemId | Category) -> str:
    return _find(panel, target).toolTip(EYE)


def _icon_for(panel: LayersPanel, state: object) -> QIcon:
    if state is MIXED:
        return panel.icon_mixed
    return panel.icon_visible if state else panel.icon_invisible


def _current(panel: LayersPanel) -> object:
    row = panel.currentItem()
    return None if row is None else row.data(0, USER)


def _visible(viewer: Viewer, ids: list[ItemId]) -> list[bool]:
    return [viewer.scene.item(i).options.visible for i in ids]


def _selected_events(events: list[Event]) -> list[Event]:
    return [
        e
        for e in events
        if isinstance(e, ViewStateChanged) and e.name == ViewField.SELECTED
    ]


def _overlays(viewer: Viewer, states: tuple[bool, ...]) -> list[ItemId]:
    ids = [
        viewer.add(zeros_volume((4, 4, 4)), Category.OVERLAYS, name=f"O{i}").id
        for i in range(len(states))
    ]
    for i, state in zip(ids, states, strict=True):
        viewer.set(i, OptionName.VISIBLE, state)
    return ids


def _volumes(viewer: Viewer, n: int) -> list[ItemId]:
    return [
        viewer.add(zeros_volume((4, 4, 4)), Category.VOLUMES, name=f"V{i}").id
        for i in range(n)
    ]


def test_every_row_has_an_eye(viewer: Viewer, panel: LayersPanel) -> None:
    _volumes(viewer, 2)
    _overlays(viewer, (True, False))
    viewer.add(_tractogram(), Category.TRACTOGRAMS, name="CST")
    for row in _all_rows(panel):
        assert row.toolTip(EYE) in TIPS.values()
        assert not row.icon(EYE).isNull()


@pytest.mark.parametrize(OptionName.VISIBLE, [True, False])
def test_item_eye_shows_the_item_visibility(
    visible: bool, viewer: Viewer, panel: LayersPanel
) -> None:
    (item_id,) = _overlays(viewer, (visible,))
    row = _find(panel, item_id)
    assert row.toolTip(EYE) == TIPS[visible]
    assert row.icon(EYE).cacheKey() == _icon_for(panel, visible).cacheKey()


@pytest.mark.parametrize(
    ("states", "expected"),
    [((True, True), True), ((False, False), False), ((True, False), MIXED)],
)
def test_category_eye_shows_the_category_visibility(
    states: tuple[bool, ...], expected: object, viewer: Viewer, panel: LayersPanel
) -> None:
    _overlays(viewer, states)
    row = _find(panel, Category.OVERLAYS)
    assert row.toolTip(EYE) == TIPS[expected]
    assert row.icon(EYE).cacheKey() == _icon_for(panel, expected).cacheKey()


def test_volumes_eye_follows_the_main_volume(
    viewer: Viewer, panel: LayersPanel
) -> None:
    ids = _volumes(viewer, 3)
    assert [_eye(panel, i) for i in ids] == ["Hidden", "Hidden", "Visible"]
    # Two of three volumes are hidden, yet the category is not "Some visible".
    assert _eye(panel, Category.VOLUMES) == "Visible"
    viewer.set(ids[2], OptionName.VISIBLE, False)
    assert _eye(panel, Category.VOLUMES) == "Hidden"


_PYTHON_CHANGES: dict[str, tuple[Callable[[Viewer, list[ItemId]], None], list[str]]] = {
    "hide item": (lambda v, o: v.set(o[0], OptionName.VISIBLE, False), ["Hidden"]),
    "show item": (lambda v, o: v.set(o[1], OptionName.VISIBLE, True), ["Visible"]),
    "hide category": (
        lambda v, o: v.set(Category.OVERLAYS, OptionName.VISIBLE, False),
        ["Hidden"],
    ),
    "show category": (
        lambda v, o: v.set(Category.OVERLAYS, OptionName.VISIBLE, True),
        ["Visible"],
    ),
}


@pytest.mark.parametrize("change", list(_PYTHON_CHANGES))
def test_overlay_eyes_follow_changes_from_python(
    change: str, viewer: Viewer, panel: LayersPanel
) -> None:
    ids = _overlays(viewer, (True, False))
    before = _tops(panel)
    action, (category,) = _PYTHON_CHANGES[change]
    action(viewer, ids)
    assert _eye(panel, Category.OVERLAYS) == category
    expected = [TIPS[v] for v in _visible(viewer, ids)]
    assert [_eye(panel, i) for i in ids] == expected
    assert _tops(panel) == before


def test_showing_a_volume_from_python_hides_the_others(
    viewer: Viewer, panel: LayersPanel
) -> None:
    ids = _volumes(viewer, 2)
    before = _tops(panel)
    viewer.set(ids[0], OptionName.VISIBLE, True)
    assert [_eye(panel, i) for i in ids] == ["Visible", "Hidden"]
    assert _eye(panel, Category.VOLUMES) == "Visible"
    assert _tops(panel) == before


def test_adding_a_volume_shows_only_the_new_one(
    viewer: Viewer, panel: LayersPanel
) -> None:
    ids = _volumes(viewer, 2)
    assert [_eye(panel, i) for i in ids] == ["Hidden", "Visible"]


@pytest.mark.parametrize("main_visible", [True, False])
def test_removing_the_main_volume_hands_its_eye_over(
    main_visible: bool, viewer: Viewer, panel: LayersPanel
) -> None:
    ids = _volumes(viewer, 3)
    viewer.set(ids[2], OptionName.VISIBLE, main_visible)
    viewer.remove(ids[2])
    assert [_eye(panel, i) for i in ids[:2]] == [TIPS[main_visible], "Hidden"]
    assert _eye(panel, Category.VOLUMES) == TIPS[main_visible]


@pytest.mark.parametrize(OptionName.VISIBLE, [True, False])
def test_clicking_an_item_eye_toggles_the_item(
    visible: bool, viewer: Viewer, panel: LayersPanel
) -> None:
    ids = _overlays(viewer, (visible, True))
    panel.itemClicked.emit(_find(panel, ids[0]), EYE)
    assert _visible(viewer, ids) == [not visible, True]
    assert _eye(panel, ids[0]) == TIPS[not visible]


def test_clicking_a_hidden_volume_eye_makes_it_the_only_visible(
    viewer: Viewer, panel: LayersPanel
) -> None:
    ids = _volumes(viewer, 3)
    panel.itemClicked.emit(_find(panel, ids[0]), EYE)
    assert _visible(viewer, ids) == [True, False, False]
    assert viewer.scene.view.main_volume == ids[0]


@pytest.mark.parametrize(
    ("states", "after"),
    [((True, True), False), ((False, False), True), ((True, False), True)],
)
def test_clicking_a_category_eye_hides_if_open_else_shows_all(
    states: tuple[bool, ...], after: bool, viewer: Viewer, panel: LayersPanel
) -> None:
    ids = _overlays(viewer, states)
    panel.itemClicked.emit(_find(panel, Category.OVERLAYS), EYE)
    assert _visible(viewer, ids) == [after] * len(ids)
    assert _eye(panel, Category.OVERLAYS) == TIPS[after]


@pytest.mark.parametrize("main_visible", [True, False])
def test_clicking_the_volumes_eye_toggles_only_the_main_volume(
    main_visible: bool, viewer: Viewer, panel: LayersPanel
) -> None:
    ids = _volumes(viewer, 3)
    viewer.set(ids[2], OptionName.VISIBLE, main_visible)
    panel.itemClicked.emit(_find(panel, Category.VOLUMES), EYE)
    assert _visible(viewer, ids) == [False, False, not main_visible]
    assert viewer.scene.view.main_volume == ids[2]
    assert _eye(panel, Category.VOLUMES) == TIPS[not main_visible]


@pytest.mark.parametrize("target", ["item", "category"])
@pytest.mark.parametrize(OptionName.VISIBLE, [True, False])
def test_clicking_the_name_column_keeps_visibility(
    target: str, visible: bool, viewer: Viewer, panel: LayersPanel
) -> None:
    ids = _overlays(viewer, (visible,))
    events = record_events(viewer)
    panel.itemClicked.emit(
        _find(panel, ids[0] if target == "item" else Category.OVERLAYS), 0
    )
    assert _visible(viewer, ids) == [visible]
    assert events == []


@pytest.mark.parametrize("category", list(Category))
def test_clicking_an_empty_category_eye_changes_nothing(
    category: Category, viewer: Viewer, panel: LayersPanel
) -> None:
    events = record_events(viewer)
    panel.itemClicked.emit(_find(panel, category), EYE)
    assert events == []
    assert _eye(panel, category) == "Visible"


def _mouse_click(
    panel: LayersPanel, qtbot: QtBot, target: ItemId | Category, column: int
) -> None:
    panel.show()
    qtbot.waitExposed(panel)
    rect = panel.visualItemRect(_find(panel, target))
    x = panel.header().sectionViewportPosition(column) + panel.columnWidth(column) // 2
    QTest.mouseClick(
        panel.viewport(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        QPoint(x, rect.center().y()),
    )


def test_mouse_click_on_an_eye_toggles_and_selects(
    viewer: Viewer, panel: LayersPanel, qtbot: QtBot
) -> None:
    ids = _overlays(viewer, (True, True))
    _mouse_click(panel, qtbot, ids[1], EYE)
    assert _visible(viewer, ids) == [True, False]
    assert viewer.scene.view.selected == ids[1]


def test_mouse_click_on_a_name_selects_without_toggling(
    viewer: Viewer, panel: LayersPanel, qtbot: QtBot
) -> None:
    ids = _overlays(viewer, (True, True))
    _mouse_click(panel, qtbot, ids[1], 0)
    assert _visible(viewer, ids) == [True, True]
    assert viewer.scene.view.selected == ids[1]


@pytest.mark.parametrize("target", ["item", "category"])
def test_current_row_selects_its_target(
    target: str, viewer: Viewer, panel: LayersPanel
) -> None:
    (item_id,) = _overlays(viewer, (True,))
    key: ItemId | Category = item_id if target == "item" else Category.OVERLAYS
    events = record_events(viewer)
    panel.setCurrentItem(_find(panel, key))
    assert viewer.scene.view.selected == key
    assert _selected_events(events) == [ViewStateChanged(ViewField.SELECTED, None, key)]


def test_a_row_gives_back_its_category_member_or_its_item_id(
    viewer: Viewer, panel: LayersPanel
) -> None:
    (item_id,) = _overlays(viewer, (True,))
    for category in Category:
        assert _target(_find(panel, category)) is category
    assert _target(_find(panel, item_id)) == item_id


def test_a_category_row_selects_the_category_member(
    viewer: Viewer, panel: LayersPanel
) -> None:
    panel.setCurrentItem(_find(panel, Category.OVERLAYS))
    assert viewer.scene.view.selected is Category.OVERLAYS


@pytest.mark.parametrize("target", ["item", "category"])
def test_select_from_python_makes_the_row_current(
    target: str, viewer: Viewer, panel: LayersPanel
) -> None:
    (item_id,) = _overlays(viewer, (True,))
    key: ItemId | Category = item_id if target == "item" else Category.OVERLAYS
    events = record_events(viewer)
    viewer.select(key)
    assert _current(panel) == key
    assert len(_selected_events(events)) == 1


def test_select_none_clears_the_current_row(viewer: Viewer, panel: LayersPanel) -> None:
    (item_id,) = _overlays(viewer, (True,))
    viewer.select(item_id)
    viewer.select(None)
    assert panel.currentItem() is None
    assert panel.selectedItems() == []


def test_removing_the_selected_item_selects_nothing(
    viewer: Viewer, panel: LayersPanel
) -> None:
    ids = _overlays(viewer, (True, True))
    panel.setCurrentItem(_find(panel, ids[0]))
    viewer.remove(ids[0])
    assert viewer.scene.view.selected is None
    assert panel.currentItem() is None
    assert panel.selectedItems() == []


_KEEPING_CHANGES: dict[str, Callable[[Viewer, ItemId], object]] = {
    "add": lambda v, i: v.add(zeros_volume((4, 4, 4)), Category.VOLUMES, name="T1"),
    "remove another": lambda v, i: v.remove(
        v.add(zeros_volume((4, 4, 4)), Category.OVERLAYS, name="X").id
    ),
    "option": lambda v, i: v.set(i, OptionName.OPACITY_2D, 0.5),
    "visibility": lambda v, i: v.set(i, OptionName.VISIBLE, False),
    "category visibility": lambda v, i: v.set(
        Category.OVERLAYS, OptionName.VISIBLE, False
    ),
}


@pytest.mark.parametrize("target", ["item", "category"])
@pytest.mark.parametrize("change", list(_KEEPING_CHANGES))
def test_changes_keep_the_selection(
    change: str, target: str, viewer: Viewer, panel: LayersPanel
) -> None:
    (item_id,) = _overlays(viewer, (True,))
    key: ItemId | Category = item_id if target == "item" else Category.OVERLAYS
    panel.setCurrentItem(_find(panel, key))
    events = record_events(viewer)
    _KEEPING_CHANGES[change](viewer, item_id)
    assert viewer.scene.view.selected == key
    assert _current(panel) == key
    assert _selected_events(events) == []


def test_panel_on_a_viewer_with_a_selection_shows_eyes_and_current_row(
    viewer: Viewer, qtbot: QtBot
) -> None:
    volumes = _volumes(viewer, 2)
    overlays = _overlays(viewer, (True, False))
    viewer.select(overlays[1])
    panel = LayersPanel(viewer)
    qtbot.addWidget(panel)
    assert [_eye(panel, i) for i in volumes] == ["Hidden", "Visible"]
    assert [_eye(panel, i) for i in overlays] == ["Visible", "Hidden"]
    assert _eye(panel, Category.VOLUMES) == "Visible"
    assert _eye(panel, Category.OVERLAYS) == "Some visible"
    assert _current(panel) == overlays[1]
    assert viewer.scene.view.selected == overlays[1]


# Removing from Layers

BIN = DELETE_COLUMN
CATEGORIES = list(Category)


def _asks(
    panel: LayersPanel, monkeypatch: pytest.MonkeyPatch, answer: bool
) -> list[str]:
    asked: list[str] = []

    def confirm(text: str) -> bool:
        asked.append(text)
        return answer

    monkeypatch.setattr(panel, "_confirm_remove", confirm)
    return asked


def _ids(viewer: Viewer) -> list[ItemId]:
    return [i.id for i in viewer.scene.items]


def test_every_row_has_the_delete_icon(viewer: Viewer, panel: LayersPanel) -> None:
    _volumes(viewer, 2)
    _overlays(viewer, (True, False))
    viewer.add(_tractogram(), Category.TRACTOGRAMS, name="CST")
    rows = _all_rows(panel)
    assert len(rows) == 5 + len(EMPTY)
    for row in rows:
        assert row.icon(BIN).cacheKey() == panel.icon_delete.cacheKey()


def test_only_the_hovered_row_shows_the_strong_delete_icon(
    viewer: Viewer, panel: LayersPanel
) -> None:
    def strong() -> list[bool]:
        hover = panel.icon_delete_hover.cacheKey()
        return [r.icon(BIN).cacheKey() == hover for r in rows]

    ids = _volumes(viewer, 2)
    panel.show()
    rows = [_find(panel, i) for i in ids]
    for hovered in (0, 1):
        centre = panel.visualItemRect(rows[hovered]).center()
        QTest.mouseMove(panel.viewport(), centre)
        assert strong() == [i == hovered for i in range(2)]
    QApplication.sendEvent(panel, QEvent(QEvent.Type.Leave))
    assert strong() == [False, False]
    assert rows[0].icon(BIN).cacheKey() == panel.icon_delete.cacheKey()


def _delete_tips(panel: LayersPanel) -> list[tuple[str, list[str]]]:
    return [
        (top.toolTip(BIN), [c.toolTip(BIN) for c in _children(top)])
        for top in _tops(panel)
    ]


def test_every_row_has_the_delete_tooltip(viewer: Viewer, panel: LayersPanel) -> None:
    categories = [f"Remove all {label} from the viewer" for label, _, _ in EMPTY]
    assert _delete_tips(panel) == [(tip, []) for tip in categories]
    # Items added after the panel exists go through a rebuild of the tree.
    _volumes(viewer, 2)
    _overlays(viewer, (True,))
    viewer.add(_tractogram(), Category.TRACTOGRAMS, name="CST")
    item = "Remove from the viewer"
    counts: dict[object, int] = {
        Category.VOLUMES: 2,
        Category.OVERLAYS: 1,
        Category.TRACTOGRAMS: 1,
    }
    assert _delete_tips(panel) == [
        (tip, [item] * counts.get(key, 0))
        for tip, (_, key, _) in zip(categories, EMPTY, strict=True)
    ]


@pytest.mark.parametrize("removed", [0, 1, 2])
def test_clicking_an_item_delete_removes_only_it(
    removed: int, viewer: Viewer, panel: LayersPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = _asks(panel, monkeypatch, True)
    ids = _overlays(viewer, (True, True, True))
    panel.itemClicked.emit(_find(panel, ids[removed]), BIN)
    rows: list[Row] = [(f"O{n}", i, "From Python") for n, i in enumerate(ids)]
    del rows[removed]
    assert _tree(panel) == _with(EMPTY, Category.OVERLAYS, rows)
    assert asked == []


def test_clicking_the_visible_main_volume_delete_opens_the_next_eye(
    viewer: Viewer, panel: LayersPanel
) -> None:
    ids = _volumes(viewer, 3)
    panel.itemClicked.emit(_find(panel, ids[2]), BIN)
    assert _ids(viewer) == ids[:2]
    assert viewer.scene.view.main_volume == ids[0]
    assert [_eye(panel, i) for i in ids[:2]] == ["Visible", "Hidden"]


def test_clicking_the_selected_item_delete_selects_nothing(
    viewer: Viewer, panel: LayersPanel
) -> None:
    ids = _overlays(viewer, (True, True))
    panel.setCurrentItem(_find(panel, ids[1]))
    panel.itemClicked.emit(_find(panel, ids[1]), BIN)
    assert _ids(viewer) == ids[:1]
    assert viewer.scene.view.selected is None
    assert panel.currentItem() is None


def _fill(viewer: Viewer, category: Category, n: int) -> list[ItemId]:
    data = {
        Category.TRACTOGRAMS: _tractogram,
        Category.PEAKS: _peaks,
        Category.SURFACES: _surface,
    }.get(category, lambda: zeros_volume((4, 4, 4)))
    return [viewer.add(data(), category, name=f"{category}{i}").id for i in range(n)]


def _surface() -> Surface:
    return Surface(np.eye(3), np.array([[0, 1, 2]]))


def _others(viewer: Viewer) -> list[ItemId]:
    # One item in every category, so a category delete that spills over shows.
    return [_fill(viewer, key, 1)[0] for key in CATEGORIES]


@pytest.mark.parametrize(
    ("category", "label"),
    [
        (Category.VOLUMES, "Volumes"),
        (Category.OVERLAYS, "Overlays"),
        (Category.TRACTOGRAMS, "Tractograms"),
        (Category.PEAKS, "Peaks"),
    ],
)
@pytest.mark.parametrize("n", [2, 3])
def test_a_confirmed_category_delete_empties_the_category(
    category: Category,
    label: str,
    n: int,
    viewer: Viewer,
    panel: LayersPanel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    others = [i for i in _others(viewer) if viewer.scene.item(i).category != category]
    _fill(viewer, category, n - 1)
    asked = _asks(panel, monkeypatch, True)
    panel.itemClicked.emit(_find(panel, category), BIN)
    assert len(asked) == 1
    assert re.search(rf"\b{n}\b", asked[0])
    assert label in asked[0]
    assert viewer.scene.in_category(category) == []
    assert _ids(viewer) == others
    assert [t.data(0, USER) for t in _tops(panel)] == CATEGORIES


@pytest.mark.parametrize("category", CATEGORIES)
def test_a_refused_category_delete_changes_nothing(
    category: Category,
    viewer: Viewer,
    panel: LayersPanel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _others(viewer)
    _fill(viewer, category, 2)
    expected = _tree(panel)
    asked = _asks(panel, monkeypatch, False)
    events = record_events(viewer)
    panel.itemClicked.emit(_find(panel, category), BIN)
    assert len(asked) == 1
    assert events == []
    assert _tree(panel) == expected


@pytest.mark.parametrize("category", CATEGORIES)
@pytest.mark.parametrize("n", [0, 1])
def test_a_category_delete_with_at_most_one_item_does_not_ask(
    category: Category,
    n: int,
    viewer: Viewer,
    panel: LayersPanel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    others = _fill(
        viewer,
        Category.OVERLAYS if category != Category.OVERLAYS else Category.VOLUMES,
        2,
    )
    _fill(viewer, category, n)
    asked = _asks(panel, monkeypatch, False)
    panel.itemClicked.emit(_find(panel, category), BIN)
    assert asked == []
    assert viewer.scene.in_category(category) == []
    assert _ids(viewer) == others


@pytest.mark.parametrize("target", ["item", "category"])
@pytest.mark.parametrize("column", [0, EYE])
def test_clicking_other_columns_removes_nothing(
    target: str,
    column: int,
    viewer: Viewer,
    panel: LayersPanel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = _overlays(viewer, (True, True))
    asked = _asks(panel, monkeypatch, True)
    events = record_events(viewer)
    panel.itemClicked.emit(
        _find(panel, ids[0] if target == "item" else Category.OVERLAYS), column
    )
    assert _ids(viewer) == ids
    assert [e for e in events if isinstance(e, ItemRemoved)] == []
    assert asked == []


def _pixels(icon: QIcon) -> QImage:
    return icon.pixmap(16, 16).toImage()


def test_a_theme_change_tints_the_eyes_again(viewer: Viewer, qtbot: QtBot) -> None:
    apply_style(Theme.DARK, "#448aff")
    dark = _pixels(QIcon(icon_path("eye-open")))
    viewer.add(zeros_volume((4, 4, 4)), Category.VOLUMES, name="T1")
    panel = LayersPanel(viewer)
    qtbot.addWidget(panel)
    apply_style(Theme.LIGHT, "#448aff")
    (row,) = _children(_tops(panel)[0])
    eye = _pixels(row.icon(VISIBILITY_COLUMN))
    assert eye == _pixels(QIcon(icon_path("eye-open")))
    assert eye != dark


def test_double_clicking_a_category_name_leaves_it_expanded(
    viewer: Viewer, panel: LayersPanel, qtbot: QtBot
) -> None:
    _overlays(viewer, (True,))
    panel.show()
    qtbot.waitExposed(panel)
    row = _find(panel, Category.OVERLAYS)
    double_click(panel.viewport(), panel.visualItemRect(row).center())
    assert row.isExpanded()
