from pathlib import Path

import numpy as np
import pytest
from pytestqt.qtbot import QtBot
from viewer_helpers import tract_options

from larmor.types import Tractogram, Volume
from larmor.viewer import Category, LabelMode, LabelSet, OptionName, Viewer
from larmor.viewer.model.options import RegionOptions
from larmor.viewer.qt.parcellation_section import LabelPicker, ParcellationSection

LUT = "2 Left-Thalamus 1 2 3 0\n5 Right-Thalamus 4 5 6 0\n7 Brain-Stem 7 8 9 0\n"


@pytest.fixture
def section(tmp_path: Path, qtbot: QtBot) -> ParcellationSection:
    viewer = Viewer()
    data = np.zeros((3, 3, 3), np.int16)
    data[0, 0, 0], data[1, 1, 1], data[2, 2, 2] = 2, 5, 7
    item = viewer.add(Volume(data, np.eye(4)), Category.OVERLAYS, name="parc")
    lut = tmp_path / "lut.txt"
    lut.write_text(LUT)
    viewer.set(item.id, OptionName.LOOKUP_TABLE, str(lut))
    points = np.array([[0.0, 0, 0], [2, 2, 2]], np.float32)
    viewer.add(Tractogram(points, np.array([0, 2])), Category.TRACTOGRAMS)
    widget = ParcellationSection(viewer, item.id)
    qtbot.addWidget(widget)
    return widget


def _options(section: ParcellationSection) -> RegionOptions:
    o = section.options()
    assert o is not None
    return o


def test_the_first_label_adds_an_applied_region_and_shows_a_chip(
    section: ParcellationSection, qtbot: QtBot
) -> None:
    viewer = section.viewer
    assert section.title.text() == "parc"
    assert section.region_id is None
    with qtbot.waitSignal(section.changed):
        section.add_label(LabelSet.A, 2)
    assert _options(section).labels_a == (2,)
    (tracts,) = viewer.scene.in_category(Category.TRACTOGRAMS)
    assert tract_options(tracts).regions == (section.ensure_region(),)
    chip = section.sets[LabelSet.A].chips[2]
    assert chip.text().startswith("Left-Thalamus")
    chip.click()
    assert _options(section).labels_a == ()


def test_stay_within_hides_set_b_and_the_radius(section: ParcellationSection) -> None:
    section.add_label(LabelSet.B, 5)
    section.modes[LabelMode.STAY_WITHIN].click()
    assert _options(section).label_mode == LabelMode.STAY_WITHIN
    assert section.sets[LabelSet.B].isHidden()
    assert section.radius_row.isHidden()
    section.modes[LabelMode.FROM_A_TO_B].click()
    assert not section.sets[LabelSet.B].isHidden()


def test_the_edited_set_is_the_one_the_label_tool_changes(
    section: ParcellationSection,
) -> None:
    section.sets[LabelSet.B].edit_button.click()
    assert section.viewer.label_target == (section.region_id, LabelSet.B)
    section.viewer.click_label((1.0, 1.0, 1.0))
    assert _options(section).labels_b == (5,)
    assert 5 in section.sets[LabelSet.B].chips


def test_the_end_radius_field_sets_the_region_s_radius(
    section: ParcellationSection,
) -> None:
    section.radius.setValue(2.5)
    section.radius.editingFinished.emit()
    assert _options(section).end_radius == 2.5


def test_the_picker_searches_by_name_and_chooses_with_enter(qtbot: QtBot) -> None:
    picker = LabelPicker({2: "Left-Thalamus", 5: "Right-Thalamus", 7: "Brain-Stem"})
    qtbot.addWidget(picker)
    picker.search.setText("right")
    picker.search.returnPressed.emit()
    assert picker.label == 5
