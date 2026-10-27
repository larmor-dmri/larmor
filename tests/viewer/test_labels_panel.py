from pathlib import Path

import numpy as np
import pytest
import shiboken6
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget
from pytestqt.qtbot import QtBot

from larmor.types import Volume
from larmor.viewer import Category, ItemId, OptionName, Viewer
from larmor.viewer.model.options import OverlayOptions
from larmor.viewer.qt.docks import ItemPanel
from larmor.viewer.qt.labels_panel import LABEL, LabelsButton, LabelsList


# The labels as the inspector's Cursor tab lists them: the selected overlay's.
def _labels_panel(viewer: Viewer) -> ItemPanel:
    return ItemPanel(
        viewer,
        Category.OVERLAYS,
        lambda item_id, panel: LabelsList(viewer, item_id, panel),
        "No overlay.",
    )


LUT = "2 Left-Thalamus 1 2 3 0\n5 Right-Thalamus 4 5 6 0\n7 Brain-Stem 7 8 9 0\n"


@pytest.fixture
def viewer(tmp_path: Path) -> Viewer:
    viewer = Viewer()
    data = np.zeros((3, 3, 3), np.int16)
    data[0, 0, 0], data[1, 1, 1], data[2, 2, 2] = 2, 5, 7
    item = viewer.add(Volume(data, np.eye(4)), Category.OVERLAYS, name="parc")
    lut = tmp_path / "lut.txt"
    lut.write_text(LUT)
    viewer.set(item.id, OptionName.LOOKUP_TABLE, str(lut))
    viewer.set(item.id, OptionName.LABELS, True)
    return viewer


def _id(viewer: Viewer) -> ItemId:
    return viewer.scene.items[0].id


@pytest.fixture
def window(qtbot: QtBot, viewer: Viewer) -> LabelsList:
    window = LabelsList(viewer, _id(viewer))
    qtbot.addWidget(window)
    return window


def _chosen(viewer: Viewer) -> tuple[int, ...]:
    o = viewer.scene.item(_id(viewer)).options
    assert isinstance(o, OverlayOptions)
    return o.surface_labels


def _visible(window: LabelsList) -> list[int]:
    rows = [window.list.item(i) for i in range(window.list.count())]
    return [row.data(LABEL) for row in rows if not row.isHidden()]


def _checked(window: LabelsList) -> list[int]:
    rows = [window.list.item(i) for i in range(window.list.count())]
    return [
        row.data(LABEL) for row in rows if row.checkState() == Qt.CheckState.Checked
    ]


def test_the_button_is_enabled_only_while_the_overlay_is_shown_as_labels(
    qtbot: QtBot, viewer: Viewer
) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    viewer.set(_id(viewer), OptionName.LABELS, False)
    button = LabelsButton(viewer, _id(viewer), parent)
    assert not button.isEnabled()
    viewer.set(_id(viewer), OptionName.LABELS, True)
    assert button.isEnabled()
    viewer.set(_id(viewer), OptionName.LABELS, False)
    assert not button.isEnabled()


def test_the_rows_are_the_labels_with_their_names(window: LabelsList) -> None:
    texts = [window.list.item(i).text() for i in range(window.list.count())]
    assert texts == ["2  Left-Thalamus", "5  Right-Thalamus", "7  Brain-Stem"]
    assert window.count.text() == "0 of 3 shown"


@pytest.mark.parametrize(
    ("text", "kept"), [("thalamus", [2, 5]), ("7", [7]), ("", [2, 5, 7])]
)
def test_the_search_keeps_the_rows_whose_number_or_name_holds_it(
    text: str, kept: list[int], window: LabelsList
) -> None:
    window.search.setText(text)
    assert _visible(window) == kept


def test_all_shows_only_the_rows_the_search_keeps(
    viewer: Viewer, window: LabelsList
) -> None:
    window.search.setText("Right")
    window.all.click()
    assert _chosen(viewer) == (5,)
    window.search.setText("Stem")
    window.all.click()
    assert _chosen(viewer) == (5, 7)


def test_none_hides_only_the_rows_the_search_keeps(
    viewer: Viewer, window: LabelsList
) -> None:
    viewer.set(_id(viewer), OptionName.SURFACE_LABELS, [2, 5, 7])
    window.search.setText("thalamus")
    window.none.click()
    assert _chosen(viewer) == (7,)


def test_checking_a_row_shows_its_label_and_unchecking_hides_it(
    viewer: Viewer, window: LabelsList
) -> None:
    row = window.list.item(1)
    row.setCheckState(Qt.CheckState.Checked)
    assert _chosen(viewer) == (5,)
    row.setCheckState(Qt.CheckState.Unchecked)
    assert _chosen(viewer) == ()


def test_a_change_from_python_updates_the_boxes_and_the_count(
    viewer: Viewer, window: LabelsList
) -> None:
    viewer.set(_id(viewer), OptionName.SURFACE_LABELS, [2, 7])
    assert _checked(window) == [2, 7]
    assert window.count.text() == "2 of 3 shown"


def test_removing_the_item_closes_its_list(qtbot: QtBot, viewer: Viewer) -> None:
    # Not given to qtbot: the list deletes itself when it closes.
    item_id = _id(viewer)
    window = LabelsList(viewer, item_id)
    window.show()
    with qtbot.waitSignal(window.closed) as closed:
        viewer.remove(item_id)
    assert closed.args == [item_id]
    qtbot.waitUntil(lambda: not shiboken6.isValid(window))


def _listed(panel: ItemPanel) -> ItemId:
    assert isinstance(panel.content, LabelsList)
    return panel.content.item_id


def test_the_panel_follows_the_selected_overlay_and_keeps_it_for_other_items(
    qtbot: QtBot, viewer: Viewer
) -> None:
    first = _id(viewer)
    second = viewer.add(
        Volume(np.ones((3, 3, 3), np.int16), np.eye(4)), Category.OVERLAYS, name="b"
    ).id
    volume = viewer.add(Volume(np.zeros((3, 3, 3)), np.eye(4)), Category.VOLUMES).id
    panel = _labels_panel(viewer)
    qtbot.addWidget(panel)
    panel.show()
    assert _listed(panel) == first
    viewer.select(second)
    assert _listed(panel) == second
    viewer.select(volume)
    assert panel.item_id == second
    viewer.remove(second)
    assert panel.item_id == first
    viewer.remove(first)
    assert panel.item_id is None and panel.content is None
    assert not panel.hint.isHidden()


@pytest.fixture
def panel(qtbot: QtBot, viewer: Viewer) -> ItemPanel:
    panel = _labels_panel(viewer)
    qtbot.addWidget(panel)
    panel.show()
    return panel


def _list(panel: ItemPanel) -> LabelsList:
    assert isinstance(panel.content, LabelsList)
    return panel.content


def test_hiding_and_showing_the_panel_keeps_its_list_and_search(
    panel: ItemPanel,
) -> None:
    shown = _list(panel)
    shown.search.setText("Stem")
    panel.hide()
    panel.show()
    assert _list(panel) is shown
    assert shown.search.text() == "Stem"
    assert _visible(shown) == [7]


def _texts(window: LabelsList) -> list[str]:
    return [window.list.item(i).text() for i in range(window.list.count())]


@pytest.mark.parametrize("name", [OptionName.SURFACE_LABELS, OptionName.LOOKUP_TABLE])
def test_a_hidden_panel_catches_up_with_its_overlay_only_when_shown(
    name: OptionName, tmp_path: Path, viewer: Viewer, panel: ItemPanel
) -> None:
    shown = _list(panel)
    before = (_texts(shown), _checked(shown))
    other = tmp_path / "other.txt"
    other.write_text(LUT.replace("Brain-Stem", "Pons"))
    value = [2, 7] if name == OptionName.SURFACE_LABELS else str(other)
    panel.hide()
    viewer.set(_id(viewer), name, value)
    assert (_texts(shown), _checked(shown)) == before
    panel.show()
    if name == OptionName.SURFACE_LABELS:
        assert _checked(shown) == [2, 7]
        assert shown.count.text() == "2 of 3 shown"
    else:
        assert _texts(shown)[-1] == "7  Pons"


def test_an_overlay_selected_while_minimised_is_listed_only_when_shown(
    qtbot: QtBot, viewer: Viewer, panel: ItemPanel
) -> None:
    # Minimising hides the panel but leaves it visible to Qt.
    second = viewer.add(
        Volume(np.ones((3, 3, 3), np.int16), np.eye(4)), Category.OVERLAYS, name="b"
    ).id
    panel.showMinimized()
    QApplication.processEvents()
    viewer.select(second)
    assert panel.item_id == second and panel.content is None
    panel.showNormal()
    qtbot.waitUntil(lambda: panel.content is not None)
    assert _list(panel).item_id == second
