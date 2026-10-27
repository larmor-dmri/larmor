import re
from collections.abc import Callable
from dataclasses import Field, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import shiboken6
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QColorDialog,
    QFileDialog,
    QFormLayout,
    QLabel,
    QToolButton,
    QWidget,
)
from pytestqt.qtbot import QtBot
from viewer_helpers import record_events

from larmor.io import read_lookup_table
from larmor.types import Surface, Tractogram, Volume
from larmor.viewer import (
    Basis,
    Category,
    Colormap,
    Colouring,
    GlyphScaling,
    ItemId,
    OptionError,
    OptionName,
    TractStyle,
    Viewer,
    ViewKind,
)
from larmor.viewer.model.events import Event, OptionChanged, ViewStateChanged
from larmor.viewer.model.options import (
    COLORMAPS,
    DTI_UNUSED,
    POINTS_PER_STREAMLINE,
    SLAB_THICKNESS,
    FodOptions,
    ItemOptions,
    OptionGroup,
    OptionInfo,
    OverlayOptions,
    VolumeOptions,
    option,
    option_fields,
    validate,
)
from larmor.viewer.model.registry import FILE_CATEGORIES
from larmor.viewer.model.surface import SurfaceData
from larmor.viewer.model.view_state import ViewField
from larmor.viewer.model.viewer import MIXED
from larmor.viewer.qt.field import (
    CheckField,
    ChoiceField,
    ColourField,
    ItemField,
    LengthsField,
    OptionField,
    PathField,
    SliderField,
    add_field_row,
    field_for,
)
from larmor.viewer.qt.options_panel import OptionsPanel

OPACITY = OptionInfo("Opacity", minimum=0.0, maximum=1.0, step=0.01)
NAMES = [OptionName.VISIBLE, OptionName.OPACITY_2D, OptionName.OPACITY_3D]
# In the panel's order: by group, then as declared.
IMAGE_NAMES = [
    *NAMES,
    OptionName.INTERPOLATE,
    OptionName.COLORMAP,
    OptionName.MINIMUM,
    OptionName.MAXIMUM,
    OptionName.COLOUR_BAR,
    OptionName.SLICES_3D,
]
LABELS = {
    OptionName.VISIBLE: "Visible",
    OptionName.OPACITY_2D: "Opacity (2D)",
    OptionName.OPACITY_3D: "Opacity (3D)",
}
IMAGE_LABELS = {
    **LABELS,
    OptionName.COLORMAP: "Colormap",
    OptionName.INTERPOLATE: "Interpolate",
    OptionName.MINIMUM: "Minimum",
    OptionName.MAXIMUM: "Maximum",
    OptionName.COLOUR_BAR: "Colour bar",
    OptionName.SLICES_3D: "Slices",
}
DEFAULTS: dict[OptionName, object] = {
    OptionName.VISIBLE: True,
    OptionName.OPACITY_2D: 1.0,
    OptionName.OPACITY_3D: 1.0,
}
# _volume() spans 0 to 1, as the defaults of an empty category do.
IMAGE_DEFAULTS: dict[OptionName, object] = {
    **DEFAULTS,
    OptionName.COLORMAP: Colormap.GRAY,
    OptionName.INTERPOLATE: True,
    OptionName.MINIMUM: 0.0,
    OptionName.MAXIMUM: 1.0,
    OptionName.COLOUR_BAR: False,
    OptionName.SLICES_3D: True,
}
OVERLAY_NAMES = [
    *NAMES,
    OptionName.INTERPOLATE,
    OptionName.COLORMAP,
    OptionName.MINIMUM,
    OptionName.MAXIMUM,
    OptionName.THRESHOLD,
    OptionName.LABELS,
    OptionName.LOOKUP_TABLE,
    OptionName.COLOUR_BAR,
    OptionName.SLICES_3D,
    OptionName.SURFACE_3D,
]
OVERLAY_LABELS = {
    **IMAGE_LABELS,
    OptionName.THRESHOLD: "Threshold",
    OptionName.SURFACE_3D: "Surface",
    OptionName.LABELS: "Labels",
    OptionName.LOOKUP_TABLE: "Lookup table",
}
OVERLAY_DEFAULTS: dict[OptionName, object] = {
    **IMAGE_DEFAULTS,
    OptionName.THRESHOLD: 0.0,
    OptionName.SURFACE_3D: True,
    OptionName.LABELS: False,
    OptionName.LOOKUP_TABLE: "",
}
TRACT_NAMES = [
    *NAMES,
    OptionName.COLOURING,
    OptionName.COLOUR,
    OptionName.COLORMAP,
    OptionName.MINIMUM,
    OptionName.MAXIMUM,
    OptionName.SUBSET,
    OptionName.POINTS,
    OptionName.REGION_FILTER,
    OptionName.ENDS_ONLY,
    OptionName.SLAB,
    OptionName.SLAB_THICKNESS,
    OptionName.COLOUR_BAR,
    OptionName.STYLE,
    OptionName.TUBE_RADIUS,
    OptionName.AMBIENT_OCCLUSION,
    OptionName.RAY_TRACED,
]
TRACT_LABELS = {
    **LABELS,
    OptionName.COLOURING: "Colouring",
    OptionName.COLOUR: "Colour",
    OptionName.COLORMAP: "Colormap",
    OptionName.MINIMUM: "Minimum",
    OptionName.MAXIMUM: "Maximum",
    OptionName.COLOUR_BAR: "Colour bar",
    OptionName.SUBSET: "Subset (%)",
    OptionName.POINTS: "Points per streamline",
    OptionName.REGION_FILTER: "Filter by regions",
    OptionName.ENDS_ONLY: "Ends only",
    OptionName.SLAB: "Slab",
    OptionName.SLAB_THICKNESS: "Slab thickness (mm)",
    OptionName.STYLE: "Style",
    OptionName.TUBE_RADIUS: "Tube radius (mm)",
    OptionName.AMBIENT_OCCLUSION: "Ambient occlusion",
    OptionName.RAY_TRACED: "Ray traced",
}
TRACT_DEFAULTS: dict[OptionName, object] = {
    **DEFAULTS,
    OptionName.COLOURING: Colouring.DIRECTION,
    OptionName.COLOUR: "#ffffff",
    OptionName.COLORMAP: Colormap.VIRIDIS,
    OptionName.MINIMUM: 0.0,
    OptionName.MAXIMUM: 1.0,
    OptionName.COLOUR_BAR: False,
    OptionName.SUBSET: 100.0,
    OptionName.POINTS: POINTS_PER_STREAMLINE,
    OptionName.REGION_FILTER: True,
    OptionName.ENDS_ONLY: False,
    OptionName.SLAB: True,
    OptionName.SLAB_THICKNESS: SLAB_THICKNESS,
    OptionName.STYLE: TractStyle.LINES,
    OptionName.TUBE_RADIUS: 0.25,
    OptionName.AMBIENT_OCCLUSION: False,
    OptionName.RAY_TRACED: False,
}
FOD_NAMES = [
    *NAMES,
    OptionName.BASIS,
    OptionName.SCALE,
    OptionName.SCALING,
    OptionName.THRESHOLD,
    OptionName.ANISOTROPY,
    OptionName.DETAIL,
    OptionName.COLOURING,
    OptionName.COLOUR,
]
FOD_LABELS = {
    **LABELS,
    OptionName.BASIS: "Basis",
    OptionName.SCALE: "Scale",
    OptionName.SCALING: "Scaling",
    OptionName.THRESHOLD: "Threshold",
    OptionName.ANISOTROPY: "Minimum anisotropy",
    OptionName.DETAIL: "Sphere detail",
    OptionName.COLOURING: "Colouring",
    OptionName.COLOUR: "Colour",
}
FOD_DEFAULTS: dict[OptionName, object] = {
    **DEFAULTS,
    OptionName.BASIS: Basis.TOURNIER,
    OptionName.SCALE: 1.0,
    OptionName.SCALING: GlyphScaling.IMAGE,
    OptionName.THRESHOLD: 0.05,
    OptionName.ANISOTROPY: 0.0,
    OptionName.DETAIL: 8.0,
    OptionName.COLOURING: Colouring.DIRECTION,
    OptionName.COLOUR: "#ffffff",
}
PEAKS_NAMES = [
    *NAMES,
    OptionName.SCALE,
    OptionName.AMPLITUDE,
    OptionName.THRESHOLD,
    OptionName.ASYMMETRIC_PEAKS,
    OptionName.LINE_WIDTH,
    OptionName.COLOURING,
    OptionName.COLOUR,
]
PEAKS_LABELS = {
    **LABELS,
    OptionName.SCALE: "Scale",
    OptionName.AMPLITUDE: "Length by amplitude",
    OptionName.THRESHOLD: "Threshold",
    OptionName.ASYMMETRIC_PEAKS: "Asymmetric peaks",
    OptionName.LINE_WIDTH: "Line width (px)",
    OptionName.COLOURING: "Colouring",
    OptionName.COLOUR: "Colour",
}
PEAKS_DEFAULTS: dict[OptionName, object] = {
    **DEFAULTS,
    OptionName.SCALE: 1.0,
    OptionName.AMPLITUDE: True,
    OptionName.THRESHOLD: 0.0,
    OptionName.ASYMMETRIC_PEAKS: False,
    OptionName.LINE_WIDTH: 2.0,
    OptionName.COLOURING: Colouring.DIRECTION,
    OptionName.COLOUR: "#ffffff",
}
# A category has no per-item options, nor has a 3D image item: its only volume is 0.
NAMES_OF = {
    Category.VOLUMES: IMAGE_NAMES,
    Category.OVERLAYS: OVERLAY_NAMES,
    Category.TRACTOGRAMS: TRACT_NAMES,
    Category.FODS: FOD_NAMES,
    Category.PEAKS: PEAKS_NAMES,
}
LABELS_OF = {
    Category.VOLUMES: IMAGE_LABELS,
    Category.OVERLAYS: OVERLAY_LABELS,
    Category.TRACTOGRAMS: TRACT_LABELS,
    Category.FODS: FOD_LABELS,
    Category.PEAKS: PEAKS_LABELS,
}
DEFAULTS_OF = {
    Category.VOLUMES: IMAGE_DEFAULTS,
    Category.OVERLAYS: OVERLAY_DEFAULTS,
    Category.TRACTOGRAMS: TRACT_DEFAULTS,
    Category.FODS: FOD_DEFAULTS,
    Category.PEAKS: PEAKS_DEFAULTS,
}
# An item's page shows what its category's does: the DTI fields appear only for an
# image of 3 volumes (tested on their own below), and the volume index only for 4D.
# A surface item without values or cortex label: no medial wall to hide.
SURFACE_NAMES = [
    *NAMES,
    OptionName.CUT_LINES,
    OptionName.LINE_WIDTH,
    OptionName.COLOUR,
    OptionName.COLOUR_BY,
    OptionName.COLORMAP,
    OptionName.MINIMUM,
    OptionName.MAXIMUM,
    OptionName.SHOW_NORMALS,
    OptionName.NORMALS_COUNT,
    OptionName.NORMALS_LENGTH,
    OptionName.NORMALS_SPACING,
    OptionName.NORMALS_COLOURING,
    OptionName.NORMALS_COLOUR,
]
SURFACE_LABELS = {
    **LABELS,
    OptionName.CUT_LINES: "Cut lines",
    OptionName.LINE_WIDTH: "Line width (px)",
    OptionName.COLOUR: "Colour",
    OptionName.COLOUR_BY: "Colour by",
    OptionName.COLORMAP: "Colormap",
    OptionName.MINIMUM: "Minimum",
    OptionName.MAXIMUM: "Maximum",
    OptionName.SHOW_NORMALS: "Show normals",
    OptionName.NORMALS_COUNT: "Normals in 3D",
    OptionName.NORMALS_LENGTH: "Normal length (mm)",
    OptionName.NORMALS_SPACING: "Spacing in 2D (mm)",
    OptionName.NORMALS_COLOURING: "Normals colouring",
    OptionName.NORMALS_COLOUR: "Normals colour",
}
SURFACE_DEFAULTS: dict[OptionName, object] = {
    **DEFAULTS,
    OptionName.CUT_LINES: True,
    OptionName.LINE_WIDTH: 2.0,
    OptionName.COLOUR: "#d9d9d9",
    OptionName.COLOUR_BY: None,
    OptionName.COLORMAP: Colormap.GRAY_R,
    OptionName.MINIMUM: -0.3,
    OptionName.MAXIMUM: 0.3,
    OptionName.SHOW_NORMALS: False,
    OptionName.NORMALS_COUNT: 200_000,
    OptionName.NORMALS_LENGTH: 2.0,
    OptionName.NORMALS_SPACING: 3.0,
    OptionName.NORMALS_COLOURING: Colouring.DIRECTION,
    OptionName.NORMALS_COLOUR: "#ffffff",
}
# A tractogram's values to colour by are its own, after its Colour.
TRACT_ITEM_NAMES = [
    *TRACT_NAMES[:5],
    OptionName.VALUES_FILE,
    OptionName.VALUES_VOLUME,
    *TRACT_NAMES[5:],
]
TRACT_ITEM_LABELS = {
    **TRACT_LABELS,
    OptionName.VALUES_FILE: "TSF file",
    OptionName.VALUES_VOLUME: "Sampled volume",
}
TRACT_ITEM_DEFAULTS: dict[OptionName, object] = {
    **TRACT_DEFAULTS,
    OptionName.VALUES_FILE: "",
    OptionName.VALUES_VOLUME: None,
}
ITEM_NAMES_OF = {
    **NAMES_OF,
    Category.TRACTOGRAMS: TRACT_ITEM_NAMES,
    Category.SURFACES: SURFACE_NAMES,
}
ITEM_LABELS_OF = {
    **LABELS_OF,
    Category.TRACTOGRAMS: TRACT_ITEM_LABELS,
    Category.SURFACES: SURFACE_LABELS,
}
ITEM_DEFAULTS_OF = {
    **DEFAULTS_OF,
    Category.TRACTOGRAMS: TRACT_ITEM_DEFAULTS,
    Category.SURFACES: SURFACE_DEFAULTS,
}
ITEM_OVERLAY_DEFAULTS = ITEM_DEFAULTS_OF[Category.OVERLAYS]
CHECKED = Qt.CheckState.Checked
UNCHECKED = Qt.CheckState.Unchecked
PARTIAL = Qt.CheckState.PartiallyChecked


def _write_lut(path: Path) -> Path:
    path.write_text("1 One 10 20 30 0\n")
    return path


@pytest.fixture
def lut(tmp_path: Path) -> Path:
    return _write_lut(tmp_path / "lut.txt")


def _volume(lo: float = 0.0, hi: float = 1.0) -> Volume:
    data = np.linspace(lo, hi, 64).reshape(4, 4, 4)
    return Volume(data, np.eye(4))


def _fod() -> Volume:
    return Volume(np.ones((4, 4, 4, 6), np.float32), np.eye(4))


def _peaks() -> Volume:
    return Volume(np.zeros((4, 4, 4, 3), np.float32), np.eye(4))


def _surface() -> Surface:
    return Surface(np.eye(3), np.array([[0, 1, 2]]))


def _tractogram() -> Tractogram:
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 2, 0]], np.float32)
    return Tractogram(points=points, offsets=np.array([0, 2, 4]))


def _sent(field: OptionField) -> list[object]:
    sent: list[object] = []
    field.edited.connect(sent.append)
    return sent


def _type(field: SliderField, text: str) -> None:
    field.text.selectAll()
    QTest.keyClicks(field.text, text)
    QTest.keyClick(field.text, Qt.Key.Key_Return)


def _pick(field: ChoiceField, text: str) -> None:
    # Typing on a closed drop-down picks the matching entry, as a user does.
    QTest.keyClicks(field, text)


# Fields alone


@pytest.fixture
def slider(qtbot: QtBot) -> SliderField:
    field = SliderField(OPACITY)
    qtbot.addWidget(field)
    return field


@pytest.fixture
def check(qtbot: QtBot) -> CheckField:
    field = CheckField()
    qtbot.addWidget(field)
    return field


@pytest.mark.parametrize("value", [True, False, MIXED])
def test_check_field_show_value_sends_nothing(value: object, check: CheckField) -> None:
    for start in (True, False, MIXED):
        check.show_value(start)
        sent = _sent(check)
        check.show_value(value)
        assert sent == []


@pytest.mark.parametrize("value", [0.0, 0.42, 1.0, MIXED])
def test_slider_field_show_value_sends_nothing(
    value: object, slider: SliderField
) -> None:
    slider.show_value(0.5)
    sent = _sent(slider)
    slider.show_value(value)
    assert sent == []


@pytest.mark.parametrize(("value", "state"), [(True, CHECKED), (False, UNCHECKED)])
def test_check_field_shows_a_value_two_state(
    value: bool, state: Qt.CheckState, check: CheckField
) -> None:
    check.show_value(MIXED)
    check.show_value(value)
    assert check.checkState() == state
    assert not check.isTristate()


def test_check_field_shows_mixed_as_partially_checked(check: CheckField) -> None:
    check.show_value(True)
    check.show_value(MIXED)
    assert check.checkState() == PARTIAL
    assert check.isTristate()


@pytest.mark.parametrize("start", [True, False])
def test_check_field_click_sends_the_new_state_once(
    start: bool, check: CheckField
) -> None:
    check.show_value(start)
    sent = _sent(check)
    check.click()
    assert sent == [not start]
    assert all(type(v) is bool for v in sent)


def test_clicking_a_mixed_check_field_sends_true(check: CheckField) -> None:
    check.show_value(MIXED)
    sent = _sent(check)
    check.click()
    assert sent == [True]
    check.show_value(True)
    assert not check.isTristate()


def test_check_field_after_mixed_never_clicks_into_partial(check: CheckField) -> None:
    check.show_value(MIXED)
    check.show_value(False)
    sent = _sent(check)
    check.click()
    assert sent == [True]
    assert check.checkState() == CHECKED


@pytest.mark.parametrize(
    ("value", "index"), [(0.0, 0), (0.07, 7), (0.29, 29), (0.5, 50), (1.0, 100)]
)
def test_slider_field_shows_a_value_on_slider_and_text(
    value: float, index: int, slider: SliderField
) -> None:
    slider.show_value(value)
    assert slider.slider.value() == index
    assert slider.text.text() == f"{value:g}"


@pytest.mark.parametrize("value", [0.0, 0.07, 0.35, 0.5, 0.7, 1.0])
def test_slider_field_round_trips_value_and_index(
    value: float, slider: SliderField
) -> None:
    slider.show_value(value)
    index = slider.slider.value()
    slider.show_value(0.3)
    sent = _sent(slider)
    slider.slider.setValue(index)
    # Exact: 0.07 / 0.01 is 7.000000000000001 and 35 * 0.01 is 0.35000000000000003.
    assert sent == [value]
    assert repr(sent[0]) == repr(value)


def test_slider_range_is_the_number_of_steps(slider: SliderField) -> None:
    assert (slider.slider.minimum(), slider.slider.maximum()) == (0, 100)


def test_slider_field_with_an_offset_minimum(qtbot: QtBot) -> None:
    field = SliderField(OptionInfo("Gain", minimum=2.0, maximum=4.0, step=0.5))
    qtbot.addWidget(field)
    assert field.slider.maximum() == 4
    field.show_value(3.5)
    assert field.slider.value() == 3
    sent = _sent(field)
    field.slider.setValue(1)
    assert sent == [2.5]


@pytest.mark.parametrize(("text", "value"), [("0.25", 0.25), ("1", 1.0), ("2", 2.0)])
def test_typed_number_is_sent_once_as_float(
    text: str, value: float, slider: SliderField
) -> None:
    slider.show_value(0.5)
    sent = _sent(slider)
    _type(slider, text)
    assert sent == [value]
    assert type(sent[0]) is float


@pytest.mark.parametrize("text", ["abc", "0,5", ""])
def test_typed_non_number_is_sent_as_the_raw_text(
    text: str, slider: SliderField
) -> None:
    slider.show_value(0.5)
    sent = _sent(slider)
    slider.text.setText(text)
    slider.text.editingFinished.emit()
    assert sent == [text]


def test_slider_field_shows_mixed_as_empty_text(slider: SliderField) -> None:
    slider.show_value(0.5)
    slider.show_value(MIXED)
    assert slider.text.text() == ""
    assert slider.text.placeholderText() == "mixed"


@pytest.mark.parametrize(
    "info",
    [
        OptionInfo("Opacity"),
        OptionInfo("Opacity", minimum=0.0),
        OptionInfo("Opacity", maximum=1.0),
    ],
)
def test_slider_field_without_bounds_is_refused(info: OptionInfo) -> None:
    with pytest.raises(TypeError, match="Opacity"):
        SliderField(info)


def test_slider_field_accepts_a_minimum_of_zero(qtbot: QtBot) -> None:
    field = SliderField(OptionInfo("Opacity", minimum=0.0, maximum=0.0, step=0.1))
    qtbot.addWidget(field)
    assert field.slider.maximum() == 0


# Bounds from the data, for options with no range of their own


def test_slider_field_without_its_own_range_spans_the_bounds(qtbot: QtBot) -> None:
    field = SliderField(OptionInfo("Minimum"), bounds=(-2.0, 8.0))
    qtbot.addWidget(field)
    assert field.minimum == -2.0
    assert field.step == pytest.approx(0.01)
    assert (field.slider.minimum(), field.slider.maximum()) == (0, 1000)


@pytest.mark.parametrize(
    ("value", "index"), [(-2.0, 0), (0.5, 250), (3.0, 500), (7.99, 999), (8.0, 1000)]
)
def test_slider_field_on_bounds_round_trips_value_and_index(
    value: float, index: int, qtbot: QtBot
) -> None:
    field = SliderField(OptionInfo("Minimum"), bounds=(-2.0, 8.0))
    qtbot.addWidget(field)
    field.show_value(value)
    assert field.slider.value() == index
    field.show_value(1.0)
    sent = _sent(field)
    field.slider.setValue(index)
    assert sent == [value]


def test_slider_field_on_a_constant_image_has_one_position(qtbot: QtBot) -> None:
    field = SliderField(OptionInfo("Minimum"), bounds=(5.0, 5.0))
    qtbot.addWidget(field)
    assert field.step == 1.0
    assert field.slider.maximum() == 0
    field.show_value(5.0)
    assert field.text.text() == "5"


@pytest.mark.parametrize(
    ("info", "start", "positions"),
    [
        (OPACITY, 0.0, 100),
        # Only the missing end comes from the bounds.
        (OptionInfo("Gain", minimum=1.0), 1.0, 1000),
        (OptionInfo("Gain", maximum=1.0), -5.0, 1000),
    ],
)
def test_an_option_own_range_wins_over_the_bounds(
    info: OptionInfo, start: float, positions: int, qtbot: QtBot
) -> None:
    field = SliderField(info, bounds=(-5.0, 11.0))
    qtbot.addWidget(field)
    assert field.minimum == start
    assert field.slider.maximum() == positions


# Whole-number sliders

INDEX = OptionInfo("Index", minimum=0, maximum=4, step=1)


@pytest.fixture
def int_slider(qtbot: QtBot) -> SliderField:
    field = SliderField(INDEX, field_type=int)
    qtbot.addWidget(field)
    return field


@pytest.mark.parametrize(
    ("field_type", "values"), [(int, [0, 3, 4]), (float, [0.0, 3.0, 4.0])]
)
def test_slider_moves_are_sent_as_the_field_type(
    field_type: type[int] | type[float], values: list[object], qtbot: QtBot
) -> None:
    field = SliderField(INDEX, field_type=field_type)
    qtbot.addWidget(field)
    field.show_value(field_type(1))
    sent = _sent(field)
    for index in (0, 3, 4):
        field.slider.setValue(index)
    assert sent == values
    assert all(type(v) is field_type for v in sent)


def test_a_slider_is_a_float_slider_by_default(qtbot: QtBot) -> None:
    field = SliderField(INDEX)
    qtbot.addWidget(field)
    sent = _sent(field)
    field.slider.setValue(2)
    _type(field, "3")
    assert sent == [2.0, 3.0]
    assert all(type(v) is float for v in sent)


def test_a_typed_whole_number_is_sent_as_an_int(int_slider: SliderField) -> None:
    int_slider.show_value(0)
    sent = _sent(int_slider)
    _type(int_slider, "3")
    assert sent == [3]
    assert type(sent[0]) is int


@pytest.mark.parametrize("text", ["2.5", "abc", ""])
def test_typed_text_that_is_no_whole_number_is_sent_as_the_raw_text(
    text: str, int_slider: SliderField
) -> None:
    int_slider.show_value(0)
    sent = _sent(int_slider)
    int_slider.text.setText(text)
    int_slider.text.editingFinished.emit()
    assert sent == [text]


@pytest.mark.parametrize("value", [0, 2, 4])
def test_an_int_slider_shows_an_int_on_slider_and_text(
    value: int, int_slider: SliderField
) -> None:
    int_slider.show_value(MIXED)
    sent = _sent(int_slider)
    int_slider.show_value(value)
    assert int_slider.slider.value() == value
    assert int_slider.text.text() == str(value)
    assert sent == []


def test_field_for_gives_an_int_option_an_int_slider(qtbot: QtBot) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    field = field_for(
        *option_fields(VolumeOptions)[OptionName.VOLUME_INDEX], parent, (0.0, 6.0)
    )
    assert isinstance(field, SliderField)
    assert field.slider.maximum() == 6
    sent = _sent(field)
    field.slider.setValue(5)
    _type(field, "2")
    assert sent == [5, 2]
    assert all(type(v) is int for v in sent)


# Choices


CHOICES = (Colormap.GRAY, Colormap.HOT, Colormap.VIRIDIS)


@pytest.fixture
def choice(qtbot: QtBot) -> ChoiceField:
    field = ChoiceField(CHOICES)
    qtbot.addWidget(field)
    return field


def test_choice_field_lists_the_choices_in_order(choice: ChoiceField) -> None:
    assert [choice.itemText(i) for i in range(choice.count())] == list(CHOICES)


@pytest.mark.parametrize("value", [*CHOICES, MIXED])
def test_choice_field_show_value_sends_nothing(
    value: object, choice: ChoiceField
) -> None:
    for start in (*CHOICES, MIXED):
        choice.show_value(start)
        sent = _sent(choice)
        choice.show_value(value)
        assert sent == []


@pytest.mark.parametrize("value", CHOICES)
def test_choice_field_shows_a_value_as_the_current_choice(
    value: str, choice: ChoiceField
) -> None:
    choice.show_value(MIXED)
    choice.show_value(value)
    assert choice.currentText() == value


def test_choice_field_shows_mixed_as_blank(choice: ChoiceField) -> None:
    choice.show_value(Colormap.HOT)
    choice.show_value(MIXED)
    assert choice.currentIndex() == -1
    assert choice.currentText() == ""


@pytest.mark.parametrize("start", [Colormap.GRAY, Colormap.VIRIDIS, MIXED])
def test_a_picked_choice_is_sent_once_as_its_text(
    start: object, choice: ChoiceField
) -> None:
    choice.show_value(start)
    sent = _sent(choice)
    _pick(choice, "hot")
    assert sent == ["hot"]


def test_each_choice_stepped_to_with_the_keyboard_is_sent(choice: ChoiceField) -> None:
    choice.show_value(Colormap.GRAY)
    sent = _sent(choice)
    QTest.keyClick(choice, Qt.Key.Key_Down)
    QTest.keyClick(choice, Qt.Key.Key_Down)
    assert sent == ["hot", "viridis"]


@dataclass(frozen=True)
class _Others:
    name: str = option("a", "Name")
    gain: complex = option(1j, "Gain", minimum=0, maximum=9)
    mode: str = option("a", "Mode", choices=("a", "b"))


def _field(cls: type[Any], name: str) -> tuple[Field[Any], OptionInfo]:
    (f,) = [f for f in fields(cls) if f.name == name]
    return f, f.metadata["option"]


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        (OptionName.VISIBLE, CheckField),
        (OptionName.OPACITY_2D, SliderField),
        (OptionName.OPACITY_3D, SliderField),
        (OptionName.COLORMAP, ChoiceField),
        (OptionName.INTERPOLATE, CheckField),
        (OptionName.MINIMUM, SliderField),
        (OptionName.MAXIMUM, SliderField),
        (OptionName.THRESHOLD, SliderField),
        (OptionName.LABELS, CheckField),
        (OptionName.COLOUR_BAR, CheckField),
        (OptionName.LOOKUP_TABLE, PathField),
        (OptionName.VOLUME_INDEX, SliderField),
    ],
)
def test_field_for_picks_the_widget_by_choices_then_type(
    name: OptionName, kind: type[QWidget], qtbot: QtBot
) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    f, info = option_fields(OverlayOptions)[name]
    field = field_for(f, info, parent, (0.0, 1.0))
    assert type(field) is kind
    assert field.parent() is parent


def test_field_for_gives_a_text_option_with_choices_a_drop_down(qtbot: QtBot) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    field = field_for(*_field(_Others, "mode"), parent)
    assert isinstance(field, ChoiceField)
    assert [field.itemText(i) for i in range(field.count())] == ["a", "b"]


def test_the_colormap_drop_down_offers_every_colormap(qtbot: QtBot) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    field = field_for(*option_fields(VolumeOptions)[OptionName.COLORMAP], parent)
    assert isinstance(field, ChoiceField)
    assert tuple(field.itemText(i) for i in range(field.count())) == COLORMAPS


def test_field_for_passes_the_bounds_to_a_slider(qtbot: QtBot) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    field = field_for(
        *option_fields(VolumeOptions)[OptionName.MINIMUM], parent, (-3.0, 7.0)
    )
    assert isinstance(field, SliderField)
    assert field.minimum == -3.0
    assert field.slider.maximum() == 1000


@pytest.mark.parametrize("name", ["name", "gain"])
def test_field_for_refuses_other_types(name: str, qtbot: QtBot) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    with pytest.raises(TypeError, match=name):
        field_for(*_field(_Others, name), parent)


# The panel


@pytest.fixture
def panel(viewer: Viewer, qtbot: QtBot) -> OptionsPanel:
    panel = OptionsPanel(viewer)
    qtbot.addWidget(panel)
    return panel


def _changes(events: list[Event]) -> list[tuple[ItemId, str, object, object]]:
    return [
        (e.item.id, e.name, e.old, e.new)
        for e in events
        if isinstance(e, OptionChanged)
    ]


def _options(viewer: Viewer, ids: list[ItemId], name: OptionName) -> list[object]:
    return [getattr(viewer.scene.item(i).options, name) for i in ids]


def _overlays(viewer: Viewer, n: int) -> list[ItemId]:
    return [viewer.add(_volume(), Category.OVERLAYS, name=f"O{i}").id for i in range(n)]


def _volumes(viewer: Viewer, n: int) -> list[ItemId]:
    return [viewer.add(_volume(), Category.VOLUMES, name=f"V{i}").id for i in range(n)]


def _check(panel: OptionsPanel) -> CheckField:
    field = panel._fields[OptionName.VISIBLE]
    assert isinstance(field, CheckField)
    return field


def _slider(panel: OptionsPanel, name: OptionName) -> SliderField:
    field = panel._fields[name]
    assert isinstance(field, SliderField)
    return field


def _form(panel: OptionsPanel) -> QWidget:
    form = panel.widget()
    assert form is not None
    return form


def _labels(panel: OptionsPanel) -> list[str]:
    return [
        re.sub("<[^>]+>", "", label.text())
        for label in _form(panel).findChildren(QLabel)
    ]


def _row_labels(panel: OptionsPanel) -> dict[OptionName, str]:
    layout = _form(panel).layout()
    assert isinstance(layout, QFormLayout)
    labels: dict[OptionName, str] = {}
    for name, field in panel._fields.items():
        label = layout.labelForField(field)
        assert isinstance(label, QLabel)
        labels[name] = label.text()
    return labels


def _widgets(panel: OptionsPanel) -> list[QWidget]:
    form = _form(panel)
    return [
        *form.findChildren(CheckField),
        *form.findChildren(ChoiceField),
        *form.findChildren(ColourField),
        *form.findChildren(ItemField),
        *form.findChildren(PathField),
        *form.findChildren(SliderField),
    ]


def _error(panel: OptionsPanel, name: OptionName) -> str | None:
    label = panel._errors[name]
    return None if label.isHidden() else label.text()


def _shown(panel: OptionsPanel) -> dict[OptionName, object]:
    shown: dict[OptionName, object] = {}
    for name, field in panel._fields.items():
        if isinstance(field, CheckField):
            state = field.checkState()
            shown[name] = MIXED if state == PARTIAL else state == CHECKED
        elif isinstance(field, ChoiceField):
            shown[name] = MIXED if field.currentIndex() == -1 else field.currentText()
        elif isinstance(field, PathField):
            mixed = field.text.placeholderText() == "mixed"
            shown[name] = MIXED if mixed else field.text.text()
        elif isinstance(field, ColourField):
            shown[name] = MIXED if field.text() == "mixed" else field.colour
        elif isinstance(field, ItemField):
            # An item by its name, as the drop-down shows it.
            chosen = None if field.currentText() == "None" else field.currentText()
            shown[name] = MIXED if field.currentIndex() == -1 else chosen
        elif isinstance(field, LengthsField):
            shown[name] = tuple(box.value() for box in field.boxes)
        else:
            text = field.text.text()
            shown[name] = MIXED if text == "" else float(text)
    return shown


def test_nothing_selected_shows_a_hint_and_no_fields(panel: OptionsPanel) -> None:
    assert panel._fields == {}
    assert _widgets(panel) == []
    (hint,) = _labels(panel)
    assert hint


@pytest.mark.parametrize("category", FILE_CATEGORIES)
def test_selected_item_shows_its_name_and_one_field_per_option(
    category: Category, viewer: Viewer, panel: OptionsPanel
) -> None:
    data = {
        Category.TRACTOGRAMS: _tractogram,
        Category.FODS: _fod,
        Category.PEAKS: _peaks,
        Category.SURFACES: _surface,
    }.get(category, _volume)()
    item = viewer.add(data, category, name="brain.nii.gz")
    viewer.set(item.id, OptionName.OPACITY_3D, 0.3)
    viewer.select(item.id)
    names = ITEM_NAMES_OF[category]
    assert list(panel._fields) == names
    assert list(panel._errors) == names
    assert _row_labels(panel) == ITEM_LABELS_OF[category]
    assert "brain.nii.gz" in _labels(panel)
    assert _shown(panel) == {**ITEM_DEFAULTS_OF[category], OptionName.OPACITY_3D: 0.3}
    assert [_error(panel, n) for n in names] == [None] * len(names)


@pytest.mark.parametrize(
    ("category", "label"),
    [
        (Category.VOLUMES, "Volumes"),
        (Category.OVERLAYS, "Overlays"),
        (Category.TRACTOGRAMS, "Tractograms"),
        (Category.FODS, "FODs"),
        (Category.PEAKS, "Peaks"),
    ],
)
def test_selected_category_shows_its_label_and_fields(
    category: Category, label: str, viewer: Viewer, panel: OptionsPanel
) -> None:
    viewer.select(category)
    assert label in _labels(panel)
    assert list(panel._fields) == NAMES_OF[category]
    assert _row_labels(panel) == LABELS_OF[category]
    assert _shown(panel) == DEFAULTS_OF[category]


_SELECTIONS: dict[
    str, tuple[Callable[[list[ItemId]], ItemId | Category | None], str]
] = {
    "other item": (lambda ids: ids[1], "O1"),
    "category": (lambda ids: Category.TRACTOGRAMS, "Tractograms"),
    "nothing": (lambda ids: None, ""),
}


@pytest.mark.parametrize("change", list(_SELECTIONS))
def test_selection_change_rebuilds_the_fields(
    change: str, viewer: Viewer, panel: OptionsPanel
) -> None:
    ids = _overlays(viewer, 2)
    viewer.set(ids[1], OptionName.OPACITY_2D, 0.25)
    viewer.select(ids[0])
    old = _widgets(panel)
    assert len(old) == len(ITEM_NAMES_OF[Category.OVERLAYS])
    target, title = _SELECTIONS[change]
    viewer.select(target(ids))
    assert not any(shiboken6.isValid(w) for w in old)
    assert sorted(map(id, _widgets(panel))) == sorted(map(id, panel._fields.values()))
    if title:
        assert title in _labels(panel)
        assert "O0" not in _labels(panel)
        expected = 0.25 if change == "other item" else 1.0
        assert _shown(panel)[OptionName.OPACITY_2D] == expected
    else:
        assert panel._fields == {}
        assert panel._errors == {}


_OTHER_EVENTS: dict[str, Callable[[Viewer, list[ItemId]], object]] = {
    "option": lambda v, ids: v.set(ids[0], OptionName.OPACITY_2D, 0.5),
    "other item option": lambda v, ids: v.set(ids[1], OptionName.VISIBLE, False),
    "category option": lambda v, ids: v.set(
        Category.OVERLAYS, OptionName.OPACITY_3D, 0.5
    ),
    "add": lambda v, ids: v.add(_volume(), Category.OVERLAYS, name="New"),
    "remove other": lambda v, ids: v.remove(ids[1]),
    "cursor": lambda v, ids: v.move_cursor((1.0, 2.0, 3.0)),
    "maximised": lambda v, ids: v.toggle_maximised(ViewKind.AXIAL),
    "selected view": lambda v, ids: v.select_view(ViewKind.SAGITTAL),
}


@pytest.mark.parametrize("target", ["item", "category"])
@pytest.mark.parametrize("change", list(_OTHER_EVENTS))
def test_other_events_keep_the_fields(
    change: str, target: str, viewer: Viewer, panel: OptionsPanel
) -> None:
    ids = _overlays(viewer, 2)
    viewer.select(ids[0] if target == "item" else Category.OVERLAYS)
    form = _form(panel)
    before = dict(panel._fields)
    _OTHER_EVENTS[change](viewer, ids)
    assert panel.widget() is form
    assert panel._fields == before


_PYTHON_SETS: list[tuple[OptionName, object]] = [
    (OptionName.VISIBLE, False),
    (OptionName.OPACITY_2D, 0.3),
    (OptionName.OPACITY_3D, 0.07),
    (OptionName.COLORMAP, Colormap.HOT),
    (OptionName.INTERPOLATE, False),
    (OptionName.MINIMUM, 0.3),
    (OptionName.MAXIMUM, 0.75),
    (OptionName.THRESHOLD, 0.6),
    (OptionName.LABELS, True),
    (OptionName.COLOUR_BAR, True),
]
# Slider positions per unit: 0.01 steps for opacities, a thousand across 0 to 1.
_PER_UNIT = {
    OptionName.OPACITY_2D: 100,
    OptionName.OPACITY_3D: 100,
    OptionName.MINIMUM: 1000,
    OptionName.MAXIMUM: 1000,
    OptionName.THRESHOLD: 1000,
}


@pytest.mark.parametrize(("name", "value"), _PYTHON_SETS)
def test_item_changes_from_python_show_in_the_fields(
    name: OptionName, value: object, viewer: Viewer, panel: OptionsPanel
) -> None:
    (item_id,) = _overlays(viewer, 1)
    viewer.select(item_id)
    viewer.set(item_id, name, value)
    assert _shown(panel) == {**ITEM_OVERLAY_DEFAULTS, name: value}
    if isinstance(value, float):
        assert _slider(panel, name).slider.value() == round(value * _PER_UNIT[name])


@pytest.mark.parametrize(("name", "value"), _PYTHON_SETS)
def test_category_changes_from_python_show_in_the_fields(
    name: OptionName, value: object, viewer: Viewer, panel: OptionsPanel
) -> None:
    ids = _overlays(viewer, 2)
    viewer.select(Category.OVERLAYS)
    viewer.set(ids[0], name, value)
    assert _shown(panel)[name] is MIXED
    viewer.set(ids[1], name, value)
    assert _shown(panel)[name] == value


def _edit_visible(panel: OptionsPanel) -> None:
    _check(panel).click()


def _edit_check(name: OptionName) -> Callable[[OptionsPanel], None]:
    def click(panel: OptionsPanel) -> None:
        field = panel._fields[name]
        assert isinstance(field, CheckField)
        field.click()

    return click


def _edit_choice(name: OptionName, text: str) -> Callable[[OptionsPanel], None]:
    def pick(panel: OptionsPanel) -> None:
        field = panel._fields[name]
        assert isinstance(field, ChoiceField)
        _pick(field, text)

    return pick


def _edit_slider(name: OptionName) -> Callable[[OptionsPanel], None]:
    return lambda panel: _slider(panel, name).slider.setValue(
        round(0.25 * _PER_UNIT[name])
    )


def _edit_typed(name: OptionName) -> Callable[[OptionsPanel], None]:
    return lambda panel: _type(_slider(panel, name), "0.25")


_EDITS: dict[str, tuple[OptionName, Callable[[OptionsPanel], None], object]] = {
    "visible click": (OptionName.VISIBLE, _edit_visible, False),
    "opacity_2d slider": (
        OptionName.OPACITY_2D,
        _edit_slider(OptionName.OPACITY_2D),
        0.25,
    ),
    "opacity_2d typed": (
        OptionName.OPACITY_2D,
        _edit_typed(OptionName.OPACITY_2D),
        0.25,
    ),
    "opacity_3d slider": (
        OptionName.OPACITY_3D,
        _edit_slider(OptionName.OPACITY_3D),
        0.25,
    ),
    "opacity_3d typed": (
        OptionName.OPACITY_3D,
        _edit_typed(OptionName.OPACITY_3D),
        0.25,
    ),
    "colormap pick": (
        OptionName.COLORMAP,
        _edit_choice(OptionName.COLORMAP, "hot"),
        Colormap.HOT,
    ),
    "interpolate click": (
        OptionName.INTERPOLATE,
        _edit_check(OptionName.INTERPOLATE),
        False,
    ),
    "minimum slider": (OptionName.MINIMUM, _edit_slider(OptionName.MINIMUM), 0.25),
    "minimum typed": (OptionName.MINIMUM, _edit_typed(OptionName.MINIMUM), 0.25),
    "maximum slider": (OptionName.MAXIMUM, _edit_slider(OptionName.MAXIMUM), 0.25),
    "maximum typed": (OptionName.MAXIMUM, _edit_typed(OptionName.MAXIMUM), 0.25),
    "threshold slider": (
        OptionName.THRESHOLD,
        _edit_slider(OptionName.THRESHOLD),
        0.25,
    ),
    "threshold typed": (
        OptionName.THRESHOLD,
        _edit_typed(OptionName.THRESHOLD),
        0.25,
    ),
    "labels click": (OptionName.LABELS, _edit_check(OptionName.LABELS), True),
    "colour bar click": (
        OptionName.COLOUR_BAR,
        _edit_check(OptionName.COLOUR_BAR),
        True,
    ),
}


@pytest.mark.parametrize("edit", list(_EDITS))
def test_item_edit_changes_only_that_option_of_that_item(
    edit: str, viewer: Viewer, panel: OptionsPanel
) -> None:
    ids = _overlays(viewer, 2)
    viewer.select(ids[0])
    events = record_events(viewer)
    name, action, value = _EDITS[edit]
    old = getattr(viewer.scene.item(ids[0]).options, name)
    action(panel)
    assert _changes(events) == [(ids[0], name, old, value)]
    assert _shown(panel)[name] == value
    assert _error(panel, name) is None


@pytest.mark.parametrize("edit", list(_EDITS))
def test_category_edit_changes_only_that_option_on_every_item(
    edit: str, viewer: Viewer, panel: OptionsPanel, lut: Path
) -> None:
    ids = _overlays(viewer, 3)
    other = viewer.add(_volume(), Category.VOLUMES, name="T1").id
    # Every option differs between the items, so a spill-over shows.
    viewer.set(ids[0], OptionName.VISIBLE, False)
    viewer.set(ids[1], OptionName.OPACITY_2D, 0.5)
    viewer.set(ids[2], OptionName.OPACITY_3D, 0.5)
    viewer.set(ids[0], OptionName.COLORMAP, Colormap.JET)
    viewer.set(ids[1], OptionName.INTERPOLATE, False)
    viewer.set(ids[2], OptionName.MINIMUM, 0.5)
    viewer.set(ids[0], OptionName.MAXIMUM, 0.5)
    viewer.set(ids[1], OptionName.THRESHOLD, 0.5)
    viewer.set(ids[2], OptionName.LABELS, True)
    viewer.set(ids[0], OptionName.LOOKUP_TABLE, str(lut))
    viewer.set(ids[1], OptionName.COLOUR_BAR, True)
    viewer.set(ids[2], OptionName.SLICES_3D, False)
    viewer.set(ids[0], OptionName.SURFACE_3D, False)
    other_options = viewer.scene.item(other).options
    before = {n: _options(viewer, ids, n) for n in OVERLAY_NAMES}
    viewer.select(Category.OVERLAYS)
    assert _shown(panel) == {n: MIXED for n in OVERLAY_NAMES}
    name, action, value = _EDITS[edit]
    checks = (
        OptionName.VISIBLE,
        OptionName.INTERPOLATE,
        OptionName.LABELS,
        OptionName.COLOUR_BAR,
        OptionName.SLICES_3D,
        OptionName.SURFACE_3D,
    )
    if name in checks:
        value = True  # a click on a mixed checkbox ticks it
    action(panel)
    after = {n: _options(viewer, ids, n) for n in OVERLAY_NAMES}
    assert after == {**before, name: [value] * 3}
    assert viewer.scene.item(other).options == other_options
    assert _shown(panel) == {n: value if n == name else MIXED for n in OVERLAY_NAMES}


def test_a_colour_chosen_for_mixed_tractograms_colours_them_all(
    viewer: Viewer, panel: OptionsPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = [viewer.add(_tractogram(), Category.TRACTOGRAMS).id for _ in range(2)]
    viewer.set(ids[0], OptionName.COLOUR, "#ff0000")
    viewer.select(Category.TRACTOGRAMS)
    field = panel._fields[OptionName.COLOUR]
    assert isinstance(field, ColourField)
    assert _shown(panel)[OptionName.COLOUR] is MIXED
    monkeypatch.setattr(QColorDialog, "getColor", lambda *_: QColor("#FF8000"))
    field.click()
    assert _options(viewer, ids, OptionName.COLOUR) == ["#ff8000"] * 2
    assert _shown(panel)[OptionName.COLOUR] == "#ff8000"


def test_mixed_category_visibility_is_fixed_by_a_click(
    viewer: Viewer, panel: OptionsPanel
) -> None:
    ids = _overlays(viewer, 2)
    viewer.set(ids[0], OptionName.VISIBLE, False)
    viewer.select(Category.OVERLAYS)
    field = _check(panel)
    assert field.checkState() == PARTIAL
    field.click()
    assert _options(viewer, ids, OptionName.VISIBLE) == [True, True]
    assert field.checkState() == CHECKED
    assert not field.isTristate()
    field.click()
    assert _options(viewer, ids, OptionName.VISIBLE) == [False, False]
    assert field.checkState() == UNCHECKED


def test_mixed_opacity_text_shows_the_value_after_an_edit(
    viewer: Viewer, panel: OptionsPanel
) -> None:
    ids = _overlays(viewer, 2)
    viewer.set(ids[0], OptionName.OPACITY_2D, 0.5)
    viewer.select(Category.OVERLAYS)
    field = _slider(panel, OptionName.OPACITY_2D)
    assert field.text.text() == ""
    assert field.text.placeholderText() == "mixed"
    field.slider.setValue(30)
    assert field.text.text() == "0.3"
    assert _options(viewer, ids, OptionName.OPACITY_2D) == [0.3, 0.3]


def test_volumes_visible_follows_the_main_volume(
    viewer: Viewer, panel: OptionsPanel
) -> None:
    ids = _volumes(viewer, 3)
    viewer.select(Category.VOLUMES)
    # Two of three volumes are hidden, yet the field is not mixed.
    assert _check(panel).checkState() == CHECKED
    viewer.set(ids[2], OptionName.VISIBLE, False)
    assert _check(panel).checkState() == UNCHECKED
    viewer.set(ids[0], OptionName.VISIBLE, True)
    assert _check(panel).checkState() == CHECKED


def test_volumes_checkbox_hides_and_shows_only_the_main_volume(
    viewer: Viewer, panel: OptionsPanel
) -> None:
    ids = _volumes(viewer, 3)
    viewer.select(Category.VOLUMES)
    field = _check(panel)
    field.click()
    assert _options(viewer, ids, OptionName.VISIBLE) == [False, False, False]
    assert field.checkState() == UNCHECKED
    field.click()
    assert _options(viewer, ids, OptionName.VISIBLE) == [False, False, True]
    assert viewer.scene.view.main_volume == ids[2]
    assert field.checkState() == CHECKED


def test_volumes_checkbox_without_volumes_stays_checked(
    viewer: Viewer, panel: OptionsPanel
) -> None:
    viewer.select(Category.VOLUMES)
    events = record_events(viewer)
    _check(panel).click()
    assert events == []
    assert _check(panel).checkState() == CHECKED
    assert _error(panel, OptionName.VISIBLE) is None


def test_ticking_a_hidden_volume_hides_the_others(
    viewer: Viewer, panel: OptionsPanel
) -> None:
    ids = _volumes(viewer, 3)
    viewer.select(ids[0])
    field = _check(panel)
    assert field.checkState() == UNCHECKED
    field.click()
    assert _options(viewer, ids, OptionName.VISIBLE) == [True, False, False]
    assert viewer.scene.view.main_volume == ids[0]
    assert field.checkState() == CHECKED


@pytest.mark.parametrize("target", ["item", "category"])
@pytest.mark.parametrize("name", [OptionName.OPACITY_2D, OptionName.OPACITY_3D])
@pytest.mark.parametrize("text", ["2", "-0.5", "abc", "nan", "inf", ""])
def test_rejected_typed_value_shows_the_message_and_changes_nothing(
    text: str, name: OptionName, target: str, viewer: Viewer, panel: OptionsPanel
) -> None:
    ids = _overlays(viewer, 2)
    viewer.select(ids[0] if target == "item" else Category.OVERLAYS)
    events = record_events(viewer)
    field = _slider(panel, name)
    field.text.setText(text)
    field.text.editingFinished.emit()
    value: object = text
    try:
        value = float(text)
    except ValueError:
        pass
    with pytest.raises(ValueError) as expected:
        validate(ItemOptions, name, value)
    assert _error(panel, name) == str(expected.value)
    others = [n for n in panel._fields if n != name]
    assert [_error(panel, n) for n in others] == [None] * len(others)
    assert events == []
    assert _options(viewer, ids, name) == [1.0, 1.0]


def test_rejection_message_is_visible_on_screen(
    viewer: Viewer, panel: OptionsPanel, qtbot: QtBot
) -> None:
    (item_id,) = _overlays(viewer, 1)
    viewer.select(item_id)
    panel.show()
    qtbot.waitExposed(panel)
    _type(_slider(panel, OptionName.OPACITY_2D), "2")
    error = panel._errors[OptionName.OPACITY_2D]
    assert error.isVisible()
    assert "Opacity (2D)" in error.text()


@pytest.mark.parametrize("target", ["item", "category"])
def test_a_valid_entry_after_a_rejection_hides_the_message(
    target: str, viewer: Viewer, panel: OptionsPanel
) -> None:
    ids = _overlays(viewer, 2)
    viewer.select(ids[0] if target == "item" else Category.OVERLAYS)
    field = _slider(panel, OptionName.OPACITY_3D)
    _type(field, "2")
    assert _error(panel, OptionName.OPACITY_3D) is not None
    field.slider.setValue(40)
    assert _error(panel, OptionName.OPACITY_3D) is None
    assert field.text.text() == "0.4"
    expected = [0.4, 1.0] if target == "item" else [0.4, 0.4]
    assert _options(viewer, ids, OptionName.OPACITY_3D) == expected


def test_typing_the_current_value_clears_the_error_and_tidies_the_text(
    viewer: Viewer, panel: OptionsPanel
) -> None:
    (item_id,) = _overlays(viewer, 1)
    viewer.select(item_id)
    field = _slider(panel, OptionName.OPACITY_2D)
    _type(field, "abc")
    assert _error(panel, OptionName.OPACITY_2D) is not None
    events = record_events(viewer)
    _type(field, "1.0")
    assert events == []
    assert _error(panel, OptionName.OPACITY_2D) is None
    assert field.text.text() == "1"
    assert field.slider.value() == 100


def test_rejected_entry_keeps_the_other_fields_editable(
    viewer: Viewer, panel: OptionsPanel
) -> None:
    (item_id,) = _overlays(viewer, 1)
    viewer.select(item_id)
    _type(_slider(panel, OptionName.OPACITY_2D), "2")
    _check(panel).click()
    assert viewer.scene.item(item_id).options.visible is False
    assert _error(panel, OptionName.OPACITY_2D) is not None
    assert _error(panel, OptionName.VISIBLE) is None


def test_removing_the_selected_item_empties_the_panel(
    viewer: Viewer, panel: OptionsPanel
) -> None:
    ids = _overlays(viewer, 2)
    viewer.select(ids[0])
    viewer.remove(ids[0])
    assert panel._fields == {}
    assert _widgets(panel) == []
    assert len(_labels(panel)) == 1


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
def test_removing_items_of_the_selected_category_updates_it(
    category: Category, viewer: Viewer, panel: OptionsPanel
) -> None:
    ids = [viewer.add(_volume(), category, name=f"A{i}").id for i in range(2)]
    viewer.set(ids[1], OptionName.OPACITY_2D, 0.5)
    viewer.select(category)
    assert _shown(panel)[OptionName.OPACITY_2D] is MIXED
    viewer.remove(ids[0])
    assert _shown(panel)[OptionName.OPACITY_2D] == 0.5
    viewer.remove(category)
    assert _shown(panel)[OptionName.OPACITY_2D] == 1.0


@pytest.mark.parametrize("target", ["item", "category"])
def test_panel_on_a_viewer_with_a_selection_shows_it(
    target: str, viewer: Viewer, qtbot: QtBot
) -> None:
    ids = _overlays(viewer, 2)
    viewer.set(ids[0], OptionName.OPACITY_2D, 0.3)
    viewer.set(ids[1], OptionName.OPACITY_2D, 0.3)
    viewer.set(ids[0], OptionName.VISIBLE, False)
    viewer.select(ids[0] if target == "item" else Category.OVERLAYS)
    panel = OptionsPanel(viewer)
    qtbot.addWidget(panel)
    assert ("O0" if target == "item" else "Overlays") in _labels(panel)
    visible: object = False if target == "item" else MIXED
    expected = {
        **(ITEM_OVERLAY_DEFAULTS if target == "item" else OVERLAY_DEFAULTS),
        OptionName.VISIBLE: visible,
        OptionName.OPACITY_2D: 0.3,
    }
    assert _shown(panel) == expected


# Events refresh only the fields they change

_UNRELATED: dict[str, Callable[[Viewer, list[ItemId]], object]] = {
    "cursor": lambda v, ids: v.move_cursor((1.0, 2.0, 3.0)),
    "maximised": lambda v, ids: v.toggle_maximised(ViewKind.AXIAL),
    "selected view": lambda v, ids: v.select_view(ViewKind.SAGITTAL),
    "other field": lambda v, ids: v.set(ids[0], OptionName.VISIBLE, False),
    "other category option": lambda v, ids: v.set(ids[2], OptionName.OPACITY_2D, 0.5),
    "other category add": lambda v, ids: v.add(_volume(), Category.VOLUMES, name="New"),
    "other category remove": lambda v, ids: v.remove(ids[2]),
    "other item same field": lambda v, ids: v.set(ids[1], OptionName.OPACITY_2D, 0.5),
}


def _reject(field: SliderField) -> None:
    _type(field, "2")


def _unconfirmed(field: SliderField) -> None:
    field.text.setText("0.3")  # no Return, no focus out


# An item of the selected category is not unrelated to the category.
_UNRELATED_TO = [
    (change, target)
    for change in _UNRELATED
    for target in ("item", "category")
    if (change, target) != ("other item same field", "category")
]


@pytest.mark.parametrize("entry", [_reject, _unconfirmed])
@pytest.mark.parametrize(("change", "target"), _UNRELATED_TO)
def test_unrelated_events_keep_the_typed_text_and_its_error(
    change: str,
    target: str,
    entry: Callable[[SliderField], None],
    viewer: Viewer,
    panel: OptionsPanel,
) -> None:
    ids = [*_overlays(viewer, 2), *_volumes(viewer, 1)]
    viewer.select(ids[0] if target == "item" else Category.OVERLAYS)
    field = _slider(panel, OptionName.OPACITY_2D)
    entry(field)
    text, error = field.text.text(), _error(panel, OptionName.OPACITY_2D)
    _UNRELATED[change](viewer, ids)
    assert field.text.text() == text
    assert _error(panel, OptionName.OPACITY_2D) == error


@pytest.mark.parametrize("target", ["item", "category"])
@pytest.mark.parametrize(
    ("name", "value"), [(OptionName.VISIBLE, False), (OptionName.OPACITY_3D, 0.3)]
)
def test_a_python_change_refreshes_only_its_field(
    name: OptionName, value: object, target: str, viewer: Viewer, panel: OptionsPanel
) -> None:
    ids = _overlays(viewer, 1)
    viewer.select(ids[0] if target == "item" else Category.OVERLAYS)
    field = _slider(panel, OptionName.OPACITY_2D)
    _type(field, "2")
    error = _error(panel, OptionName.OPACITY_2D)
    viewer.set(ids[0], name, value)
    assert _shown(panel)[name] == value
    assert field.text.text() == "2"
    assert error is not None
    assert _error(panel, OptionName.OPACITY_2D) == error


@pytest.mark.parametrize("target", ["item", "category"])
def test_a_python_change_to_the_rejected_field_shows_it_and_hides_the_error(
    target: str, viewer: Viewer, panel: OptionsPanel
) -> None:
    ids = _overlays(viewer, 1)
    viewer.select(ids[0] if target == "item" else Category.OVERLAYS)
    _type(_slider(panel, OptionName.OPACITY_2D), "2")
    _type(_slider(panel, OptionName.OPACITY_3D), "abc")
    viewer.set(ids[0], OptionName.OPACITY_2D, 0.3)
    assert _slider(panel, OptionName.OPACITY_2D).text.text() == "0.3"
    assert _error(panel, OptionName.OPACITY_2D) is None
    assert _slider(panel, OptionName.OPACITY_3D).text.text() == "abc"
    assert _error(panel, OptionName.OPACITY_3D) is not None


def test_adding_and_removing_an_item_of_the_category_refreshes_every_field(
    viewer: Viewer, panel: OptionsPanel, lut: Path
) -> None:
    ids = _overlays(viewer, 2)
    changed: dict[OptionName, object] = {
        OptionName.VISIBLE: False,
        OptionName.OPACITY_2D: 0.5,
        OptionName.OPACITY_3D: 0.5,
        OptionName.COLORMAP: Colormap.HOT,
        OptionName.INTERPOLATE: False,
        OptionName.MINIMUM: 0.5,
        OptionName.MAXIMUM: 0.5,
        OptionName.THRESHOLD: 0.5,
        OptionName.LABELS: True,
        OptionName.LOOKUP_TABLE: str(lut),
        OptionName.COLOUR_BAR: True,
        OptionName.SLICES_3D: False,
        OptionName.SURFACE_3D: False,
    }
    for item_id in ids:
        for name, value in changed.items():
            viewer.set(item_id, name, value)
    viewer.select(Category.OVERLAYS)
    _type(_slider(panel, OptionName.OPACITY_2D), "2")
    new = viewer.add(_volume(), Category.OVERLAYS, name="New").id
    assert _shown(panel) == {n: MIXED for n in OVERLAY_NAMES}
    assert [_error(panel, n) for n in OVERLAY_NAMES] == [None] * len(OVERLAY_NAMES)
    _type(_slider(panel, OptionName.OPACITY_3D), "abc")
    viewer.remove(new)
    assert _shown(panel) == changed
    assert [_error(panel, n) for n in OVERLAY_NAMES] == [None] * len(OVERLAY_NAMES)


def test_a_new_main_volume_refreshes_only_the_volumes_visible_field(
    viewer: Viewer, panel: OptionsPanel
) -> None:
    hidden, main = _volumes(viewer, 2)
    viewer.select(Category.VOLUMES)
    _type(_slider(panel, OptionName.OPACITY_2D), "2")
    assert _check(panel).checkState() == CHECKED
    events = record_events(viewer)
    # Only the main volume changes, to a hidden one; the Viewer's actions always
    # change a visibility with it.
    viewer.scene.set_view(ViewField.MAIN_VOLUME, hidden)
    assert [type(e) for e in events] == [ViewStateChanged]
    assert _check(panel).checkState() == UNCHECKED
    assert _slider(panel, OptionName.OPACITY_2D).text.text() == "2"
    assert _error(panel, OptionName.OPACITY_2D) is not None
    viewer.scene.set_view(ViewField.MAIN_VOLUME, main)
    assert _check(panel).checkState() == CHECKED


# The data's range for minimum and maximum


def _span(panel: OptionsPanel, name: OptionName) -> tuple[float, float]:
    field = _slider(panel, name)
    return (field.minimum, field.minimum + field.slider.maximum() * field.step)


def test_an_item_minimum_and_maximum_sliders_span_its_value_range(
    viewer: Viewer, panel: OptionsPanel
) -> None:
    item = viewer.add(_volume(-2.0, 8.0), Category.OVERLAYS, name="O")
    viewer.add(_volume(-100.0, 100.0), Category.OVERLAYS, name="Other")
    viewer.select(item.id)
    for name in (OptionName.MINIMUM, OptionName.MAXIMUM):
        assert _span(panel, name) == pytest.approx((-2.0, 8.0))
    assert _slider(panel, OptionName.MINIMUM).slider.value() == 0
    assert _slider(panel, OptionName.MAXIMUM).slider.value() == 1000


def test_a_category_minimum_and_maximum_sliders_span_all_its_items(
    viewer: Viewer, panel: OptionsPanel
) -> None:
    viewer.add(_volume(-2.0, 3.0), Category.OVERLAYS, name="A")
    viewer.add(_volume(1.0, 8.0), Category.OVERLAYS, name="B")
    viewer.add(_volume(-100.0, 100.0), Category.VOLUMES, name="Other category")
    viewer.select(Category.OVERLAYS)
    for name in (OptionName.MINIMUM, OptionName.MAXIMUM):
        assert _span(panel, name) == pytest.approx((-2.0, 8.0))
    assert _shown(panel)[OptionName.MINIMUM] is MIXED


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
def test_an_empty_category_sliders_span_0_to_1(
    category: Category, viewer: Viewer, panel: OptionsPanel
) -> None:
    viewer.select(category)
    for name in (OptionName.MINIMUM, OptionName.MAXIMUM):
        assert _span(panel, name) == (0.0, 1.0)
    assert _slider(panel, OptionName.MAXIMUM).slider.value() == 1000


@pytest.mark.parametrize("target", ["item", "category"])
@pytest.mark.parametrize("text", ["-5", "1e6"])
def test_a_typed_minimum_outside_the_data_range_is_accepted(
    target: str, text: str, viewer: Viewer, panel: OptionsPanel
) -> None:
    ids = _overlays(viewer, 1)
    viewer.select(ids[0] if target == "item" else Category.OVERLAYS)
    _type(_slider(panel, OptionName.MINIMUM), text)
    assert _options(viewer, ids, OptionName.MINIMUM) == [float(text)]
    assert _error(panel, OptionName.MINIMUM) is None


# An overlay's threshold


def test_an_overlay_threshold_slider_spans_its_value_range(
    viewer: Viewer, panel: OptionsPanel
) -> None:
    item = viewer.add(_volume(-2.0, 8.0), Category.OVERLAYS, name="O")
    viewer.add(_volume(-100.0, 100.0), Category.OVERLAYS, name="Other")
    viewer.select(item.id)
    assert _row_labels(panel)[OptionName.THRESHOLD] == "Threshold"
    assert _span(panel, OptionName.THRESHOLD) == pytest.approx((-2.0, 8.0))
    # It starts at the minimum: the left end.
    assert _slider(panel, OptionName.THRESHOLD).slider.value() == 0
    assert _shown(panel)[OptionName.THRESHOLD] == -2.0


@pytest.mark.parametrize("text", ["3", "-5", "1e6"])
def test_a_typed_threshold_is_set(
    text: str, viewer: Viewer, panel: OptionsPanel
) -> None:
    item = viewer.add(_volume(-2.0, 8.0), Category.OVERLAYS, name="O")
    viewer.select(item.id)
    _type(_slider(panel, OptionName.THRESHOLD), text)
    assert _options(viewer, [item.id], OptionName.THRESHOLD) == [float(text)]
    assert _error(panel, OptionName.THRESHOLD) is None


@pytest.mark.parametrize("target", ["item", "category"])
def test_volumes_have_no_threshold_field(
    target: str, viewer: Viewer, panel: OptionsPanel
) -> None:
    (item_id,) = _volumes(viewer, 1)
    viewer.select(item_id if target == "item" else Category.VOLUMES)
    assert OptionName.THRESHOLD not in panel._fields
    assert "Threshold" not in _row_labels(panel).values()


# Labels and the lookup table


@pytest.fixture
def path_field(qtbot: QtBot) -> PathField:
    field = PathField(OptionInfo("Lookup table", path=True))
    qtbot.addWidget(field)
    return field


def _browse_returns(monkeypatch: pytest.MonkeyPatch, chosen: str) -> list[str]:
    titles: list[str] = []

    def choose(parent: QWidget, title: str) -> tuple[str, str]:
        titles.append(title)
        return chosen, ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", choose)
    return titles


def _path_field(panel: OptionsPanel) -> PathField:
    field = panel._fields[OptionName.LOOKUP_TABLE]
    assert isinstance(field, PathField)
    return field


def test_path_field_shows_built_in_when_empty(path_field: PathField) -> None:
    path_field.show_value("/data/lut.txt")
    path_field.show_value("")
    assert path_field.text.text() == ""
    assert path_field.text.placeholderText() == "built-in"
    assert path_field.button.text() == "Browse..."


def test_path_field_shows_a_path_as_its_text(path_field: PathField) -> None:
    path_field.show_value(MIXED)
    path_field.show_value("/data/lut.txt")
    assert path_field.text.text() == "/data/lut.txt"
    assert path_field.text.placeholderText() == "built-in"


def test_path_field_shows_mixed_as_empty_text(path_field: PathField) -> None:
    path_field.show_value("/data/lut.txt")
    path_field.show_value(MIXED)
    assert path_field.text.text() == ""
    assert path_field.text.placeholderText() == "mixed"


@pytest.mark.parametrize("value", ["", "/data/lut.txt", MIXED])
def test_path_field_show_value_sends_nothing(
    value: object, path_field: PathField
) -> None:
    for start in ("", "/data/lut.txt", MIXED):
        path_field.show_value(start)
        sent = _sent(path_field)
        path_field.show_value(value)
        assert sent == []


def test_a_typed_path_is_sent_once_as_text(path_field: PathField) -> None:
    sent = _sent(path_field)
    path_field.text.setText("/data/lut.txt")
    path_field.text.editingFinished.emit()
    assert sent == ["/data/lut.txt"]


def test_a_browsed_file_is_shown_and_sent_once(
    path_field: PathField, monkeypatch: pytest.MonkeyPatch
) -> None:
    titles = _browse_returns(monkeypatch, "/data/lut.txt")
    sent = _sent(path_field)
    path_field.button.click()
    assert titles == ["Lookup table"]
    assert sent == ["/data/lut.txt"]
    assert path_field.text.text() == "/data/lut.txt"


def test_a_cancelled_browse_sends_nothing_and_keeps_the_text(
    path_field: PathField, monkeypatch: pytest.MonkeyPatch
) -> None:
    _browse_returns(monkeypatch, "")
    path_field.show_value("/data/old.txt")
    sent = _sent(path_field)
    path_field.button.click()
    assert sent == []
    assert path_field.text.text() == "/data/old.txt"


def test_an_overlay_has_a_labels_checkbox_and_a_lookup_table_field(
    viewer: Viewer, panel: OptionsPanel
) -> None:
    (item_id,) = _overlays(viewer, 1)
    viewer.select(item_id)
    assert isinstance(panel._fields[OptionName.LABELS], CheckField)
    field = _path_field(panel)
    assert field.text.text() == ""
    assert field.text.placeholderText() == "built-in"


@pytest.mark.parametrize("target", ["item", "category"])
def test_volumes_have_no_label_fields(
    target: str, viewer: Viewer, panel: OptionsPanel
) -> None:
    (item_id,) = _volumes(viewer, 1)
    viewer.select(item_id if target == "item" else Category.VOLUMES)
    assert OptionName.LABELS not in panel._fields
    assert OptionName.LOOKUP_TABLE not in panel._fields
    assert not _form(panel).findChildren(PathField)


def test_a_mixed_lookup_table_shows_mixed(
    viewer: Viewer, panel: OptionsPanel, lut: Path
) -> None:
    ids = _overlays(viewer, 2)
    viewer.set(ids[0], OptionName.LOOKUP_TABLE, str(lut))
    viewer.select(Category.OVERLAYS)
    field = _path_field(panel)
    assert field.text.text() == ""
    assert field.text.placeholderText() == "mixed"
    viewer.set(ids[1], OptionName.LOOKUP_TABLE, str(lut))
    assert field.text.text() == str(lut)
    assert field.text.placeholderText() == "built-in"


def _type_path(panel: OptionsPanel, text: str) -> None:
    field = _path_field(panel)
    field.text.setText(text)
    field.text.editingFinished.emit()


def _browse(panel: OptionsPanel, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    _browse_returns(monkeypatch, text)
    _path_field(panel).button.click()


@pytest.mark.parametrize("how", ["typed", "browsed"])
@pytest.mark.parametrize("target", ["item", "category"])
def test_a_chosen_lookup_table_is_set(
    target: str,
    how: str,
    viewer: Viewer,
    panel: OptionsPanel,
    lut: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = _overlays(viewer, 2)
    viewer.select(ids[0] if target == "item" else Category.OVERLAYS)
    events = record_events(viewer)
    if how == "typed":
        _type_path(panel, str(lut))
    else:
        _browse(panel, monkeypatch, str(lut))
    changed = ids[:1] if target == "item" else ids
    assert _changes(events) == [
        (i, OptionName.LOOKUP_TABLE, "", str(lut)) for i in changed
    ]
    assert _shown(panel)[OptionName.LOOKUP_TABLE] == str(lut)
    assert _error(panel, OptionName.LOOKUP_TABLE) is None


def test_clearing_the_lookup_table_text_returns_to_the_built_in_colours(
    viewer: Viewer, panel: OptionsPanel, lut: Path
) -> None:
    (item_id,) = _overlays(viewer, 1)
    viewer.set(item_id, OptionName.LOOKUP_TABLE, str(lut))
    viewer.select(item_id)
    _type_path(panel, "")
    assert _options(viewer, [item_id], OptionName.LOOKUP_TABLE) == [""]
    assert _path_field(panel).text.placeholderText() == "built-in"


def test_a_cancelled_browse_changes_nothing(
    viewer: Viewer, panel: OptionsPanel, lut: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (item_id,) = _overlays(viewer, 1)
    viewer.set(item_id, OptionName.LOOKUP_TABLE, str(lut))
    viewer.select(item_id)
    events = record_events(viewer)
    _browse(panel, monkeypatch, "")
    assert events == []
    assert _shown(panel)[OptionName.LOOKUP_TABLE] == str(lut)


@pytest.mark.parametrize("how", ["typed", "browsed"])
@pytest.mark.parametrize("target", ["item", "category"])
@pytest.mark.parametrize("text", [None, "1 One 10 20\n"], ids=["missing", "garbled"])
def test_a_bad_lookup_table_shows_the_reader_message_and_changes_nothing(
    text: str | None,
    target: str,
    how: str,
    viewer: Viewer,
    panel: OptionsPanel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = tmp_path / "bad.txt"
    if text is not None:
        bad.write_text(text)
    ids = _overlays(viewer, 2)
    viewer.select(ids[0] if target == "item" else Category.OVERLAYS)
    events = record_events(viewer)
    if how == "typed":
        _type_path(panel, str(bad))
    else:
        _browse(panel, monkeypatch, str(bad))
    with pytest.raises(ValueError) as expected:
        read_lookup_table(bad)
    assert _error(panel, OptionName.LOOKUP_TABLE) == str(expected.value)
    others = [n for n in panel._fields if n != OptionName.LOOKUP_TABLE]
    assert [_error(panel, n) for n in others] == [None] * len(others)
    assert events == []
    assert _options(viewer, ids, OptionName.LOOKUP_TABLE) == ["", ""]
    assert _path_field(panel).text.text() == str(bad)


# The volume shown from 4D data


def _series(count: int | None) -> Volume:
    shape = (2, 2, 2) if count is None else (2, 2, 2, count)
    return Volume(np.zeros(shape, np.float32), np.eye(4))


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
@pytest.mark.parametrize(("count", "last"), [(5, 4), (2, 1)])
def test_an_image_volume_index_slider_spans_its_volumes_in_steps_of_1(
    category: Category,
    count: int | None,
    last: int,
    viewer: Viewer,
    panel: OptionsPanel,
) -> None:
    item = viewer.add(_series(count), category, name="dwi")
    viewer.add(_series(9), category, name="other")
    viewer.select(item.id)
    field = _slider(panel, OptionName.VOLUME_INDEX)
    assert field.step == 1
    assert _span(panel, OptionName.VOLUME_INDEX) == (0, last)
    assert _row_labels(panel)[OptionName.VOLUME_INDEX] == "Volume index"
    assert field.text.text() == "0"


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
def test_a_4d_item_shows_the_volume_index_after_the_common_fields(
    category: Category, viewer: Viewer, panel: OptionsPanel
) -> None:
    item = viewer.add(_series(5), category)  # not 3: that also shows the DTI fields
    viewer.select(item.id)
    names = ITEM_NAMES_OF[category]
    assert list(panel._fields) == [
        *NAMES,
        OptionName.VOLUME_INDEX,
        *names[len(NAMES) :],
    ]
    assert list(panel._errors) == list(panel._fields)


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
@pytest.mark.parametrize("count", [1, None])
def test_a_3d_item_has_no_volume_index_field(
    category: Category, count: int | None, viewer: Viewer, panel: OptionsPanel
) -> None:
    item = viewer.add(_series(count), category)
    viewer.add(_series(9), category, name="other")
    viewer.select(item.id)
    assert list(panel._fields) == ITEM_NAMES_OF[category]
    assert OptionName.VOLUME_INDEX not in panel._errors
    assert "Volume index" not in _labels(panel)


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
def test_an_item_volume_index_set_while_its_category_is_selected_keeps_the_fields(
    category: Category, viewer: Viewer, panel: OptionsPanel
) -> None:
    item = viewer.add(_series(5), category)
    viewer.select(category)
    form = _form(panel)
    before = dict(panel._fields)
    shown = _shown(panel)
    viewer.set(item.id, OptionName.VOLUME_INDEX, 2)
    assert _options(viewer, [item.id], OptionName.VOLUME_INDEX) == [2]
    assert panel.widget() is form
    assert panel._fields == before
    assert _shown(panel) == shown


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
def test_a_category_has_no_volume_index_field(
    category: Category, viewer: Viewer, panel: OptionsPanel
) -> None:
    viewer.add(_series(5), category)
    viewer.select(category)
    assert OptionName.VOLUME_INDEX not in panel._fields
    assert "Volume index" not in _row_labels(panel).values()
    assert not [label for label in _labels(panel) if "Volume index" in label]


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
def test_moving_the_volume_index_slider_sets_the_item_index(
    category: Category, viewer: Viewer, panel: OptionsPanel
) -> None:
    ids = [viewer.add(_series(5), category, name=f"A{i}").id for i in range(2)]
    viewer.select(ids[0])
    events = record_events(viewer)
    _slider(panel, OptionName.VOLUME_INDEX).slider.setValue(3)
    assert _changes(events) == [(ids[0], OptionName.VOLUME_INDEX, 0, 3)]
    assert _options(viewer, ids, OptionName.VOLUME_INDEX) == [3, 0]
    assert type(_options(viewer, ids, OptionName.VOLUME_INDEX)[0]) is int
    assert _slider(panel, OptionName.VOLUME_INDEX).text.text() == "3"
    assert _error(panel, OptionName.VOLUME_INDEX) is None


def test_a_typed_volume_index_is_set(viewer: Viewer, panel: OptionsPanel) -> None:
    item = viewer.add(_series(5), Category.OVERLAYS)
    viewer.select(item.id)
    _type(_slider(panel, OptionName.VOLUME_INDEX), "4")
    assert _options(viewer, [item.id], OptionName.VOLUME_INDEX) == [4]
    assert _slider(panel, OptionName.VOLUME_INDEX).slider.value() == 4
    assert _error(panel, OptionName.VOLUME_INDEX) is None


@pytest.mark.parametrize("text", ["2.5", "-1", "5", "abc"])
def test_a_rejected_volume_index_shows_the_message_and_changes_nothing(
    text: str, viewer: Viewer, panel: OptionsPanel
) -> None:
    item = viewer.add(_series(5), Category.OVERLAYS)
    viewer.select(item.id)
    events = record_events(viewer)
    _type(_slider(panel, OptionName.VOLUME_INDEX), text)
    value: object = text
    try:
        value = int(text)
    except ValueError:
        pass
    # The message the Viewer gives for the same value on a like item.
    reference = Viewer()
    like = reference.add(_series(5), Category.OVERLAYS).id
    with pytest.raises(OptionError) as expected:
        reference.set(like, OptionName.VOLUME_INDEX, value)
    assert _error(panel, OptionName.VOLUME_INDEX) == str(expected.value)
    assert events == []
    assert _options(viewer, [item.id], OptionName.VOLUME_INDEX) == [0]


@pytest.mark.parametrize("category", [Category.VOLUMES, Category.OVERLAYS])
@pytest.mark.parametrize("target", ["item", "category"])
def test_an_image_item_or_category_has_an_unticked_colour_bar_checkbox(
    category: Category, target: str, viewer: Viewer, panel: OptionsPanel
) -> None:
    item = viewer.add(_volume(), category)
    viewer.select(item.id if target == "item" else category)
    field = panel._fields[OptionName.COLOUR_BAR]
    assert isinstance(field, CheckField)
    assert _row_labels(panel)[OptionName.COLOUR_BAR] == "Colour bar"
    assert field.checkState() == UNCHECKED


# DTI colours and the image that modulates them


def _directions() -> Volume:
    return Volume(np.zeros((4, 4, 4, 3), np.float32), np.eye(4))


def _item_field(panel: OptionsPanel) -> ItemField:
    field = panel._fields[OptionName.MODULATE]
    assert isinstance(field, ItemField)
    return field


def _choices(field: ItemField) -> list[str]:
    return [field.itemText(i) for i in range(field.count())]


def test_modulate_offers_the_images_on_the_grid_as_they_come_and_go(
    viewer: Viewer, panel: OptionsPanel
) -> None:
    item = viewer.add(_directions(), Category.OVERLAYS, name="v1")
    viewer.select(item.id)
    assert _choices(_item_field(panel)) == ["None"]
    fa = viewer.add(_volume(), Category.OVERLAYS, name="fa").id
    viewer.add(_series(None), Category.OVERLAYS, name="other grid")
    assert _choices(_item_field(panel)) == ["None", "fa"]
    viewer.remove(fa)
    assert _choices(_item_field(panel)) == ["None"]


def test_a_chosen_modulating_image_is_set(viewer: Viewer, panel: OptionsPanel) -> None:
    item = viewer.add(_directions(), Category.OVERLAYS, name="v1")
    fa = viewer.add(_volume(), Category.OVERLAYS, name="fa").id
    viewer.set(item.id, OptionName.DTI, True)
    viewer.select(item.id)
    events = record_events(viewer)
    QTest.keyClicks(_item_field(panel), "fa")
    assert _changes(events) == [(item.id, OptionName.MODULATE, None, fa)]
    assert _shown(panel)[OptionName.MODULATE] == "fa"


def test_dti_colours_grey_out_the_fields_they_do_not_use(
    viewer: Viewer, panel: OptionsPanel
) -> None:
    item = viewer.add(_directions(), Category.OVERLAYS, name="v1")
    viewer.select(item.id)
    unused = [n for n in panel._fields if n in DTI_UNUSED]
    assert sorted(unused) == sorted(DTI_UNUSED)

    def enabled() -> dict[OptionName, bool]:
        return {n: panel._fields[n].isEnabled() for n in [*unused, OptionName.MODULATE]}

    off = {**{n: True for n in unused}, OptionName.MODULATE: False}
    assert enabled() == off
    viewer.set(item.id, OptionName.DTI, True)
    assert enabled() == {n: not on for n, on in off.items()}
    viewer.set(item.id, OptionName.DTI, False)
    assert enabled() == off
    others = [n for n in panel._fields if n not in off]
    assert all(panel._fields[n].isEnabled() for n in others)


@pytest.mark.parametrize(("count", "shown"), [(None, False), (3, True), (9, False)])
def test_the_dti_fields_are_shown_only_for_an_image_of_3_volumes(
    count: int | None, shown: bool, viewer: Viewer, panel: OptionsPanel
) -> None:
    item = viewer.add(_series(count), Category.VOLUMES)
    viewer.select(item.id)
    assert (OptionName.DTI in panel._fields) is shown
    assert (OptionName.MODULATE in panel._fields) is shown


# Groups

G = OptionGroup


def _header_texts(panel: OptionsPanel) -> list[str]:
    layout = _form(panel).layout()
    assert isinstance(layout, QFormLayout)
    texts = []
    for row in range(layout.rowCount()):
        item = layout.itemAt(row, QFormLayout.ItemRole.SpanningRole)
        widget = item.widget() if item is not None else None
        if isinstance(widget, QToolButton) and widget.objectName() == "optionGroup":
            texts.append(widget.text())
    return texts


@pytest.mark.parametrize(
    ("category", "groups"),
    [
        (Category.VOLUMES, [G.DISPLAY, G.COLOUR, G.TWO_D, G.THREE_D]),
        (Category.OVERLAYS, [G.DISPLAY, G.COLOUR, G.LABELS, G.TWO_D, G.THREE_D]),
        (
            Category.TRACTOGRAMS,
            [
                G.DISPLAY,
                G.COLOUR,
                G.STREAMLINES,
                G.REGIONS,
                G.SLAB,
                G.TWO_D,
                G.THREE_D,
            ],
        ),
        (Category.FODS, [G.DISPLAY, G.GLYPHS, G.COLOUR]),
        (Category.PEAKS, [G.DISPLAY, G.LINES, G.COLOUR]),
    ],
)
def test_each_item_type_shows_its_groups_in_order(
    category: Category, groups: list[OptionGroup], viewer: Viewer, panel: OptionsPanel
) -> None:
    data = {
        Category.TRACTOGRAMS: _tractogram,
        Category.FODS: _fod,
        Category.PEAKS: _peaks,
    }.get(category, _volume)()
    viewer.select(viewer.add(data, category).id)
    assert _header_texts(panel) == [g.value for g in groups]


def test_a_folded_group_hides_only_its_fields_and_stays_folded_in_its_category(
    viewer: Viewer, panel: OptionsPanel
) -> None:
    ids = _overlays(viewer, 2)
    viewer.add(_tractogram(), Category.TRACTOGRAMS)
    viewer.select(ids[0])
    panel._headers[G.COLOUR].click()

    def colour_folded() -> bool:
        # Fields whose options do not apply now, such as a tractogram's
        # colormap while it is coloured by direction, stay hidden either way.
        hidden = [
            panel._fields[n].isHidden()
            for n in panel._members[G.COLOUR]
            if panel._applies(n)
        ]
        others = [
            panel._fields[n].isHidden()
            for g, names in panel._members.items()
            if g != G.COLOUR
            for n in names
            if panel._applies(n)
        ]
        assert not any(others)
        assert all(hidden) or not any(hidden)
        assert panel._headers[G.COLOUR].isChecked() is not hidden[0]
        return hidden[0]

    assert colour_folded()
    viewer.select(ids[1])
    assert colour_folded()
    viewer.select(Category.TRACTOGRAMS)
    assert not colour_folded()
    viewer.select(ids[0])
    panel._headers[G.COLOUR].click()
    assert not colour_folded()
    viewer.select(ids[1])
    assert not colour_folded()


def test_an_extra_sits_after_its_group_fields_and_folds_with_them(
    viewer: Viewer, qtbot: QtBot
) -> None:
    def extra(item: ItemId, parent: QWidget) -> QWidget:
        return QLabel("extra", parent)

    panel = OptionsPanel(viewer, extras={Category.OVERLAYS: (G.LABELS, extra)})
    qtbot.addWidget(panel)
    (item,) = _overlays(viewer, 1)
    viewer.select(item)
    layout = _form(panel).layout()
    assert isinstance(layout, QFormLayout)
    (widget,) = [w for w in _form(panel).findChildren(QLabel) if w.text() == "extra"]

    def row(w: QWidget) -> int:
        position = layout.getWidgetPosition(w)
        assert isinstance(position, tuple)
        return int(position[0])

    lookup_row = row(panel._fields[OptionName.LOOKUP_TABLE])
    extra_row = row(widget)
    assert extra_row == lookup_row + 2  # after the field's error row
    after = layout.itemAt(extra_row + 1, QFormLayout.ItemRole.SpanningRole)
    assert after is not None and after.widget() is panel._headers[G.TWO_D]
    panel._headers[G.LABELS].click()
    assert widget.isHidden()
    panel._headers[G.LABELS].click()
    assert not widget.isHidden()


# A surface's values and medial wall


def _colour_by(panel: OptionsPanel) -> ItemField:
    field = panel._fields[OptionName.COLOUR_BY]
    assert isinstance(field, ItemField)
    return field


def test_colour_by_offers_none_then_the_surfaces_values_and_sets_one(
    viewer: Viewer, panel: OptionsPanel
) -> None:
    values = {"curv": np.zeros(3), "thickness": np.arange(3.0)}
    surface = Surface(np.eye(3), np.array([[0, 1, 2]]), vertex_data=values)
    item = viewer.add(surface, Category.SURFACES, name="lh.white").id
    viewer.select(item)
    assert _choices(_colour_by(panel)) == ["None", "curv", "thickness"]
    assert _shown(panel)[OptionName.COLOUR_BY] == "curv"
    QTest.keyClicks(_colour_by(panel), "thickness")
    assert _options(viewer, [item], OptionName.COLOUR_BY) == ["thickness"]


@pytest.mark.parametrize("cortex", [False, True])
def test_hide_medial_wall_is_offered_only_for_a_surface_with_a_cortex_label(
    cortex: bool, viewer: Viewer, panel: OptionsPanel
) -> None:
    mask = np.array([True, True, False]) if cortex else None
    item = viewer.add(SurfaceData(_surface(), mask), Category.SURFACES, name="lh.white")
    viewer.select(item.id)
    assert (OptionName.HIDE_MEDIAL_WALL in panel._fields) is cortex


# Help


def _label(panel: OptionsPanel, name: OptionName) -> QLabel:
    layout = _form(panel).layout()
    assert isinstance(layout, QFormLayout)
    label = layout.labelForField(panel._fields[name])
    assert isinstance(label, QLabel)
    return label


def test_a_field_with_help_shows_it_on_its_label_icon_and_field(
    viewer: Viewer, panel: OptionsPanel
) -> None:
    viewer.select(viewer.add(_fod(), Category.FODS, name="fod").id)
    help = option_fields(FodOptions)[OptionName.THRESHOLD][1].help
    label = _label(panel, OptionName.THRESHOLD)
    icon = label.findChild(QLabel, "helpIcon")
    assert icon is not None
    assert not icon.pixmap().isNull()
    assert label.text() == "Threshold"
    assert [label.toolTip(), icon.toolTip()] == [help, help]
    assert panel._fields[OptionName.THRESHOLD].toolTip() == help


@pytest.mark.parametrize("name", [OptionName.VISIBLE, OptionName.OPACITY_2D])
def test_a_field_without_help_has_a_plain_label_and_no_tooltip(
    name: OptionName, viewer: Viewer, panel: OptionsPanel
) -> None:
    viewer.select(viewer.add(_fod(), Category.FODS, name="fod").id)
    label = _label(panel, name)
    assert label.findChild(QLabel, "helpIcon") is None
    assert label.toolTip() == ""
    assert panel._fields[name].toolTip() == ""


def test_the_help_icon_widens_a_form_but_not_its_minimum(qtbot: QtBot) -> None:
    def form(help: str) -> QWidget:
        widget = QWidget()
        qtbot.addWidget(widget)
        add_field_row(
            QFormLayout(widget), "Threshold", help, CheckField(widget), widget
        )
        return widget

    plain, helped = form(""), form("Values below it are hidden.")
    assert helped.minimumSizeHint().width() == plain.minimumSizeHint().width()
    assert helped.sizeHint().width() > plain.sizeHint().width()


def _shown_fields(panel: OptionsPanel) -> set[OptionName]:
    return {n for n, f in panel._fields.items() if not f.isHidden()}


def test_a_tractogram_shows_the_colour_fields_of_its_colouring(
    viewer: Viewer, panel: OptionsPanel
) -> None:
    item = viewer.add(_tractogram(), Category.TRACTOGRAMS)
    viewer.select(item.id)
    by_value = {OptionName.COLORMAP, OptionName.MINIMUM, OptionName.MAXIMUM}
    assert not (by_value | {OptionName.COLOUR}) & _shown_fields(panel)
    viewer.set(item.id, OptionName.COLOURING, Colouring.SINGLE)
    assert OptionName.COLOUR in _shown_fields(panel)
    assert not by_value & _shown_fields(panel)
    viewer.set(item.id, OptionName.COLOURING, Colouring.VALUES)
    assert by_value <= _shown_fields(panel)
    assert OptionName.COLOUR not in _shown_fields(panel)


@pytest.mark.parametrize("category", [Category.FODS, Category.PEAKS])
def test_glyphs_have_no_3d_opacity_as_the_3d_view_does_not_draw_them(
    category: Category, viewer: Viewer, panel: OptionsPanel
) -> None:
    data = _fod() if category == Category.FODS else _peaks()
    viewer.select(viewer.add(data, category).id)
    assert OptionName.OPACITY_2D in _shown_fields(panel)
    assert OptionName.OPACITY_3D not in _shown_fields(panel)


def test_a_region_shows_the_radius_of_a_sphere_and_the_size_of_a_box(
    viewer: Viewer, panel: OptionsPanel
) -> None:
    viewer.select(viewer.add_sphere((0.0, 0.0, 0.0), 4.0))
    assert OptionName.RADIUS in _shown_fields(panel)
    assert OptionName.SIZE not in _shown_fields(panel)
    box = viewer.add_box((0.0, 0.0, 0.0), (6.0, 8.0, 10.0))
    viewer.select(box)
    assert OptionName.RADIUS not in _shown_fields(panel)
    assert OptionName.SIZE in _shown_fields(panel)
    field = panel._fields[OptionName.SIZE]
    assert isinstance(field, LengthsField)
    assert [b.value() for b in field.boxes] == [6.0, 8.0, 10.0]
    field.boxes[2].setValue(12.0)
    assert getattr(viewer.scene.item(box).options, OptionName.SIZE) == (6.0, 8.0, 12.0)
