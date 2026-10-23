import math
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pytest

from larmor.io import read_lookup_table
from larmor.viewer import Colormap, GlyphScaling, OptionError, OptionName
from larmor.viewer.model.options import (
    COLORMAPS,
    Colour,
    FodOptions,
    ItemOptions,
    OptionGroup,
    OverlayOptions,
    PeaksOptions,
    Scope,
    TractogramOptions,
    VolumeOptions,
    option,
    option_fields,
    validate,
)
from larmor.viewer.model.registry import REGISTRY


# Every field must be an OptionName member, so these reuse names with other types to
# reach each branch of validate.
@dataclass(frozen=True)
class _Typed(ItemOptions):
    minimum: int = option(3, "Count", minimum=0)
    colormap: str = option("a", "Mode", choices=("a", "b"))
    interpolate: str = option("x", "Name")
    maximum: float = option(1.0, "Width", maximum=5.0)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (OptionName.VISIBLE, False),
        (OptionName.OPACITY_2D, 0.5),
        (OptionName.MINIMUM, 7),
        (OptionName.INTERPOLATE, "y"),
    ],
)
def test_a_value_of_the_right_type_is_returned_unchanged(
    name: OptionName, value: object
) -> None:
    result = validate(_Typed, name, value)
    assert result == value
    assert type(result) is type(value)


def test_an_int_for_a_float_is_returned_as_a_float() -> None:
    result = validate(_Typed, OptionName.OPACITY_2D, 1)
    assert result == 1.0
    assert type(result) is float


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        (OptionName.VISIBLE, 1, "Visible must be on or off"),
        (OptionName.OPACITY_2D, "0.5", "Opacity \\(2D\\) must be a number"),
        (OptionName.OPACITY_2D, True, "Opacity \\(2D\\) must be a number"),
        (OptionName.MINIMUM, 1.0, "Count must be a whole number"),
        (OptionName.MINIMUM, True, "Count must be a whole number"),
        (OptionName.INTERPOLATE, 3, "Name must be text"),
    ],
)
def test_a_value_of_the_wrong_type_is_refused(
    name: OptionName, value: object, message: str
) -> None:
    with pytest.raises(OptionError, match=message):
        validate(_Typed, name, value)


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_a_value_outside_a_two_sided_range_names_the_range(value: float) -> None:
    with pytest.raises(OptionError, match=r"Opacity \(3D\) must be between 0 and 1"):
        validate(_Typed, OptionName.OPACITY_3D, value)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_the_ends_of_a_range_are_accepted(value: float) -> None:
    assert validate(_Typed, OptionName.OPACITY_3D, value) == value


def test_a_value_below_a_minimum_says_at_least() -> None:
    with pytest.raises(OptionError, match="Count must be at least 0; got -1"):
        validate(_Typed, OptionName.MINIMUM, -1)


def test_a_value_above_a_maximum_says_at_most() -> None:
    with pytest.raises(OptionError, match="Width must be at most 5"):
        validate(_Typed, OptionName.MAXIMUM, 6)


def test_a_value_not_among_the_choices_lists_them() -> None:
    with pytest.raises(OptionError, match="Mode must be one of a, b; got 'c'"):
        validate(_Typed, OptionName.COLORMAP, "c")
    assert validate(_Typed, OptionName.COLORMAP, "b") == "b"


def test_an_option_the_class_lacks_is_a_value_error_naming_it() -> None:
    with pytest.raises(ValueError, match="ItemOptions has no option 'colormap'") as e:
        validate(ItemOptions, OptionName.COLORMAP, Colormap.HOT)
    assert isinstance(e.value, OptionError)


def test_option_fields_are_in_declaration_order_with_their_info() -> None:
    known = option_fields(_Typed)
    assert list(known) == [
        OptionName.VISIBLE,
        OptionName.OPACITY_2D,
        OptionName.OPACITY_3D,
        OptionName.MINIMUM,
        OptionName.COLORMAP,
        OptionName.INTERPOLATE,
        OptionName.MAXIMUM,
    ]
    labels = [info.label for _, info in known.values()]
    assert labels[:3] == ["Visible", "Opacity (2D)", "Opacity (3D)"]
    assert known[OptionName.OPACITY_2D][1].scope == Scope.TWO_D
    assert known[OptionName.OPACITY_3D][1].scope == Scope.THREE_D
    assert known[OptionName.VISIBLE][1].scope == Scope.BOTH
    assert known[OptionName.COLORMAP][1].choices == ("a", "b")
    assert known[OptionName.MINIMUM][0].default == 3


def test_option_fields_of_an_instance_match_its_class() -> None:
    assert list(option_fields(_Typed())) == list(option_fields(_Typed))


def test_option_names_are_exactly_the_fields_of_the_registry_options() -> None:
    names = {f.name for kind in REGISTRY.values() for f in fields(kind.options)}
    assert names == set(OptionName)


def test_option_fields_refuse_a_field_that_is_no_option_name() -> None:
    @dataclass(frozen=True)
    class _Unnamed(ItemOptions):
        tint: str = option("red", "Tint")

    with pytest.raises(ValueError, match="'tint' is not a valid OptionName"):
        option_fields(_Unnamed)


# A volume's own options


@pytest.mark.parametrize("name", [OptionName.MINIMUM, OptionName.MAXIMUM])
@pytest.mark.parametrize("value", [-1e9, -3.5, 0.0, 1e12])
def test_minimum_and_maximum_accept_any_finite_number(
    name: OptionName, value: float
) -> None:
    assert validate(VolumeOptions, name, value) == value


@pytest.mark.parametrize("name", [OptionName.MINIMUM, OptionName.MAXIMUM])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_minimum_and_maximum_refuse_non_finite_numbers(
    name: OptionName, value: float
) -> None:
    with pytest.raises(OptionError, match="finite"):
        validate(VolumeOptions, name, value)


@pytest.mark.parametrize("colormap", COLORMAPS)
def test_every_listed_colormap_is_accepted(colormap: Colormap) -> None:
    result = validate(VolumeOptions, OptionName.COLORMAP, colormap)
    assert result is colormap


@pytest.mark.parametrize("colormap", COLORMAPS)
def test_a_colormap_given_as_text_becomes_its_member(colormap: Colormap) -> None:
    result = validate(VolumeOptions, OptionName.COLORMAP, colormap.value)
    assert result is colormap


@pytest.mark.parametrize("value", ["grey", "Gray", ""])
def test_an_unknown_colormap_is_refused(value: str) -> None:
    with pytest.raises(OptionError, match="Colormap must be one of gray, hot"):
        validate(VolumeOptions, OptionName.COLORMAP, value)


def test_overlays_have_the_volume_options() -> None:
    volume = list(option_fields(VolumeOptions))
    assert list(option_fields(OverlayOptions))[: len(volume)] == volume
    assert OverlayOptions().colormap is Colormap.GRAY
    assert OverlayOptions().interpolate is True


@pytest.mark.parametrize("value", [-1e9, -3.5, 0.0, 1e12])
def test_threshold_accepts_any_finite_number(value: float) -> None:
    assert validate(OverlayOptions, OptionName.THRESHOLD, value) == value


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_threshold_refuses_non_finite_numbers(value: float) -> None:
    with pytest.raises(OptionError, match="Threshold must be a finite number"):
        validate(OverlayOptions, OptionName.THRESHOLD, value)


# An overlay shown as labels


def _lut(tmp_path: Path, text: str = "3 Three 255 0 0 0\n") -> Path:
    path = tmp_path / "lut.txt"
    path.write_text(text)
    return path


def test_an_overlay_starts_as_an_image_with_the_built_in_colours() -> None:
    known = option_fields(OverlayOptions)
    assert OverlayOptions().labels is False
    assert OverlayOptions().lookup_table == ""
    assert known[OptionName.LABELS][1].label == "Labels"
    assert known[OptionName.LOOKUP_TABLE][1].label == "Lookup table"
    assert known[OptionName.LOOKUP_TABLE][1].path
    assert not known[OptionName.LABELS][1].path


@pytest.mark.parametrize("name", [OptionName.LABELS, OptionName.LOOKUP_TABLE])
def test_volumes_have_no_label_options(name: OptionName) -> None:
    assert name not in option_fields(VolumeOptions)
    with pytest.raises(OptionError, match=f"VolumeOptions has no option '{name}'"):
        validate(VolumeOptions, name, False if name == OptionName.LABELS else "")


@pytest.mark.parametrize("value", [True, False])
def test_labels_is_on_or_off(value: bool) -> None:
    assert validate(OverlayOptions, OptionName.LABELS, value) is value
    with pytest.raises(OptionError, match="Labels must be on or off"):
        validate(OverlayOptions, OptionName.LABELS, 1)


def test_no_lookup_table_is_accepted() -> None:
    assert validate(OverlayOptions, OptionName.LOOKUP_TABLE, "") == ""


def test_a_readable_lookup_table_is_accepted(tmp_path: Path) -> None:
    path = str(_lut(tmp_path))
    assert validate(OverlayOptions, OptionName.LOOKUP_TABLE, path) == path


def test_a_lookup_table_that_is_not_text_is_refused(tmp_path: Path) -> None:
    with pytest.raises(OptionError, match="Lookup table must be text"):
        validate(OverlayOptions, OptionName.LOOKUP_TABLE, _lut(tmp_path))


@pytest.mark.parametrize("text", [None, "3 Three 255 0\n", "# nothing\n"])
def test_a_missing_or_bad_lookup_table_is_refused_with_the_reader_message(
    text: str | None, tmp_path: Path
) -> None:
    path = tmp_path / "lut.txt" if text is None else _lut(tmp_path, text)
    with pytest.raises(ValueError) as expected:
        read_lookup_table(path)
    with pytest.raises(OptionError) as error:
        validate(OverlayOptions, OptionName.LOOKUP_TABLE, str(path))
    assert str(error.value) == str(expected.value)


def test_validate_runs_an_option_check_on_the_converted_value() -> None:
    seen: list[object] = []

    def check(value: object) -> None:
        seen.append(value)
        if value == 0.5:
            raise OptionError("Half is refused.")

    @dataclass(frozen=True)
    class _Checked(ItemOptions):
        threshold: float = option(0.0, "Threshold", check=check)

    assert validate(_Checked, OptionName.THRESHOLD, 2) == 2.0
    with pytest.raises(OptionError, match="Half is refused"):
        validate(_Checked, OptionName.THRESHOLD, 0.5)
    assert seen == [2.0, 0.5]
    assert type(seen[0]) is float
    # A value of the wrong type is refused before the check.
    with pytest.raises(OptionError, match="must be a number"):
        validate(_Checked, OptionName.THRESHOLD, "x")
    assert len(seen) == 2


def test_a_tractogram_starts_white_and_takes_a_colour_as_lowercase_rrggbb() -> None:
    assert TractogramOptions().colour == "#ffffff"
    colour = validate(TractogramOptions, OptionName.COLOUR, "#FF80aB")
    assert colour == "#ff80ab"
    assert type(colour) is Colour


@pytest.mark.parametrize("value", ["red", "#fff", "#ff80ag", "ff80ab", "#ff80ab0", 5])
def test_a_colour_not_written_rrggbb_is_refused(value: object) -> None:
    with pytest.raises(OptionError):
        validate(TractogramOptions, OptionName.COLOUR, value)


# Options set per item only


@pytest.mark.parametrize("per_item", [True, False])
def test_option_passes_per_item_into_its_info(per_item: bool) -> None:
    @dataclass(frozen=True)
    class _PerItem(ItemOptions):
        volume_index: int = option(0, "Index", per_item=per_item)

    (_, info) = option_fields(_PerItem)[OptionName.VOLUME_INDEX]
    assert info.per_item is per_item


def test_only_the_volume_index_and_its_per_item_options_are_set_per_item() -> None:
    per_item = {
        name
        for kind in REGISTRY.values()
        for name, (_, info) in option_fields(kind.options).items()
        if info.per_item
    }
    assert per_item == {
        OptionName.VOLUME_INDEX,
        OptionName.DTI,
        OptionName.MODULATE,
        OptionName.SURFACE_LABELS,
        OptionName.COLOUR_BY,
        OptionName.VALUES_FILE,
        OptionName.VALUES_VOLUME,
        OptionName.INCLUDE_MASKS,
        OptionName.EXCLUDE_MASKS,
        OptionName.INCLUDE_SPHERES,
        OptionName.EXCLUDE_SPHERES,
        OptionName.WITHIN_MASKS,
        OptionName.WITHIN_SPHERES,
    }


def test_the_volume_index_is_a_whole_number_from_0_in_steps_of_1() -> None:
    f, info = option_fields(VolumeOptions)[OptionName.VOLUME_INDEX]
    assert f.default == 0
    assert type(f.default) is int
    assert (info.minimum, info.maximum, info.step) == (0, None, 1)


@pytest.mark.parametrize("options", [VolumeOptions, OverlayOptions])
def test_the_volume_index_comes_first_after_the_options_of_every_item(
    options: type[ItemOptions],
) -> None:
    names = list(option_fields(options))
    assert names[: len(fields(ItemOptions)) + 1] == [
        *option_fields(ItemOptions),
        OptionName.VOLUME_INDEX,
    ]


@pytest.mark.parametrize(("value", "message"), [(-1, "at least 0"), (1.5, "whole")])
def test_a_negative_or_fractional_volume_index_is_refused(
    value: object, message: str
) -> None:
    with pytest.raises(OptionError, match=message):
        validate(VolumeOptions, OptionName.VOLUME_INDEX, value)


def test_the_colour_bar_is_off_by_default_and_affects_only_the_2d_views() -> None:
    f, info = option_fields(VolumeOptions)[OptionName.COLOUR_BAR]
    assert f.default is False
    assert info.scope == Scope.TWO_D
    assert info.per_item is False


# Options naming another item


@pytest.mark.parametrize(
    ("value", "expected"), [(None, None), (3, 3), (np.int64(2), 2)]
)
def test_an_item_option_takes_an_id_or_none(value: object, expected: object) -> None:
    result = validate(VolumeOptions, OptionName.MODULATE, value)
    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize("value", [True, False, "1", 1.0])
def test_an_item_option_refuses_anything_but_a_whole_number(value: object) -> None:
    with pytest.raises(OptionError, match="Modulate by"):
        validate(VolumeOptions, OptionName.MODULATE, value)


def test_every_option_of_every_item_type_has_a_group() -> None:
    for kind in REGISTRY.values():
        for name, (_, info) in option_fields(kind.options).items():
            assert isinstance(info.group, OptionGroup), (kind.label, name)


def test_an_fod_starts_whole_image_scaled_at_threshold_0_05_and_detail_8() -> None:
    options = FodOptions()
    assert (options.scaling, options.threshold, options.detail) == (
        GlyphScaling.IMAGE,
        0.05,
        8,
    )


# The same option on several item types explains what it does for each.
@pytest.mark.parametrize(
    ("options", "name", "phrase"),
    [
        (VolumeOptions, OptionName.INTERPOLATE, "trilinear"),
        (OverlayOptions, OptionName.THRESHOLD, "transparent"),
        (FodOptions, OptionName.THRESHOLD, "no glyph"),
        (PeaksOptions, OptionName.THRESHOLD, "not drawn"),
        (FodOptions, OptionName.DETAIL, "smoother"),
        (TractogramOptions, OptionName.COLOURING, "direction"),
        (FodOptions, OptionName.COLOURING, None),
        (PeaksOptions, OptionName.COLOURING, None),
        (VolumeOptions, OptionName.VISIBLE, None),
    ],
)
def test_an_option_has_the_help_of_its_item_type(
    options: type[ItemOptions], name: OptionName, phrase: str | None
) -> None:
    help = option_fields(options)[name][1].help
    if phrase is None:
        assert help == ""
    else:
        assert phrase in help
