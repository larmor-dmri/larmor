"""Declares the options of the viewer's items and checks their values."""

import math
import re
from collections.abc import Callable, Iterable
from dataclasses import Field, dataclass, field, fields
from enum import StrEnum
from numbers import Integral, Real
from typing import Any, Final

from larmor.io import read_lookup_table


class Scope(StrEnum):
    """The views an option affects."""

    TWO_D = "2d"
    THREE_D = "3d"
    BOTH = "both"


class Colormap(StrEnum):
    """The colormaps an image can be shown with."""

    GRAY = "gray"
    HOT = "hot"
    VIRIDIS = "viridis"
    INFERNO = "inferno"
    MAGMA = "magma"
    PLASMA = "plasma"
    CIVIDIS = "cividis"
    JET = "jet"
    BONE = "bone"
    COOL = "cool"
    GRAY_R = "gray_r"  # white to black
    BLUE_RED = (
        "blue_dark_red"  # diverging: blue below the middle, dark at it, red above
    )


# Every colormap, in the order the choosers list them.
COLORMAPS: tuple[Colormap, ...] = tuple(Colormap)


class OptionGroup(StrEnum):
    """The sections of the inspector's Item tab, in the order they are shown."""

    DISPLAY = "Display"
    GLYPHS = "Glyphs"
    LINES = "Lines"
    COLOUR = "Colour"
    LABELS = "Labels"
    STREAMLINES = "Streamlines"
    REGIONS = "Regions"
    NORMALS = "Normals"
    SLAB = "Slab (2D)"
    TWO_D = "2D"
    THREE_D = "3D"


class OptionName(StrEnum):
    """The names of the options an item can have, such as its colormap or opacity.

    `Viewer.set` takes these names. Each category of item has its own set of
    options, and an option that a category does not have is refused.
    """

    VISIBLE = "visible"
    OPACITY_2D = "opacity_2d"
    OPACITY_3D = "opacity_3d"
    COLORMAP = "colormap"
    INTERPOLATE = "interpolate"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    THRESHOLD = "threshold"
    LABELS = "labels"
    LOOKUP_TABLE = "lookup_table"
    VOLUME_INDEX = "volume_index"
    COLOUR_BAR = "colour_bar"
    SLICES_3D = "slices_3d"
    SURFACE_3D = "surface_3d"
    COLOURING = "colouring"
    COLOUR = "colour"
    SUBSET = "subset"
    POINTS = "points"
    SLAB = "slab"
    SLAB_THICKNESS = "slab_thickness"
    BASIS = "basis"
    SCALE = "scale"
    SCALING = "scaling"
    DETAIL = "detail"
    ANISOTROPY = "anisotropy"
    AMPLITUDE = "amplitude"
    ASYMMETRIC_PEAKS = "asymmetric_peaks"
    LINE_WIDTH = "line_width"
    SURFACE_LABELS = "surface_labels"
    DTI = "dti"
    MODULATE = "modulate"
    STYLE = "style"
    TUBE_RADIUS = "tube_radius"
    AMBIENT_OCCLUSION = "ambient_occlusion"
    RAY_TRACED = "ray_traced"
    COLOUR_BY = "colour_by"
    CUT_LINES = "cut_lines"
    HIDE_MEDIAL_WALL = "hide_medial_wall"
    SHOW_NORMALS = "show_normals"
    NORMALS_COUNT = "normals_count"
    NORMALS_LENGTH = "normals_length"
    NORMALS_SPACING = "normals_spacing"
    NORMALS_COLOURING = "normals_colouring"
    NORMALS_COLOUR = "normals_colour"
    VALUES_FILE = "values_file"
    VALUES_VOLUME = "values_volume"
    REGION_FILTER = "region_filter"
    INCLUDE_MASKS = "include_masks"
    EXCLUDE_MASKS = "exclude_masks"
    INCLUDE_SPHERES = "include_spheres"
    EXCLUDE_SPHERES = "exclude_spheres"
    WITHIN_MASKS = "within_masks"
    WITHIN_SPHERES = "within_spheres"
    ENDS_ONLY = "ends_only"


@dataclass(frozen=True)
class OptionInfo:
    """What the interface needs to know about an option."""

    label: str
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] | None = None
    scope: Scope = Scope.BOTH
    path: bool = False  # a file name, edited with a file chooser
    check: Callable[[Any], None] | None = None  # raises OptionError for a bad value
    per_item: bool = False  # set per item only, not for a whole category
    panel: bool = True  # False for one edited in its own window, not the panel
    item_ref: bool = False  # another item's id, or None for no item
    data_key: bool = False  # the name of some of the item's data, or None
    group: OptionGroup | None = None  # its section of the Item tab; None for settings
    help: str = ""  # one sentence explaining it, shown as a tooltip; "" for none
    # Shown in the panel only while this holds of the item's options; None for always.
    shown_when: Callable[[Any], bool] | None = None
    # Editable only while this holds of the item's options; None for always.
    enabled_when: Callable[[Any], bool] | None = None


class OptionError(ValueError):
    """Raised for a value an option does not accept; the message says why."""


class Colour(str):
    """A colour as "#rrggbb", edited with a colour chooser."""


class Spheres(tuple[tuple[float, float, float, float], ...]):
    """Spheres as (x, y, z, radius) in world coordinates, mm."""


def check_colour(value: object) -> None:
    """Raises OptionError unless the value is a colour written "#rrggbb"."""
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", str(value)):
        raise OptionError(f"A colour is written #rrggbb; got {value!r}.")


def option[T](
    default: T,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    step: float | None = None,
    choices: tuple[str, ...] | None = None,
    scope: Scope = Scope.BOTH,
    path: bool = False,
    check: Callable[[Any], None] | None = None,
    per_item: bool = False,
    panel: bool = True,
    item_ref: bool = False,
    data_key: bool = False,
    group: OptionGroup | None = None,
    help: str = "",
    shown_when: Callable[[Any], bool] | None = None,
    enabled_when: Callable[[Any], bool] | None = None,
) -> T:
    """Returns a dataclass field with its OptionInfo in its metadata (key "option")."""
    info = OptionInfo(
        label,
        minimum,
        maximum,
        step,
        choices,
        scope,
        path,
        check,
        per_item,
        panel,
        item_ref,
        data_key,
        group,
        help,
        shown_when,
        enabled_when,
    )
    return field(default=default, metadata={"option": info})


def option_fields(
    options: type[ItemOptions] | ItemOptions,
) -> dict[OptionName, tuple[Field[Any], OptionInfo]]:
    """Returns the options' fields and infos of a class or instance, in order."""
    # OptionName(...) fails for a field with no member, so the two cannot drift apart.
    return {OptionName(f.name): (f, f.metadata["option"]) for f in fields(options)}


def validate(options: type[ItemOptions], name: OptionName, value: object) -> object:
    """Returns value as the option's type if the option accepts it."""
    known = option_fields(options)
    if name not in known:
        raise OptionError(f"{options.__name__} has no option '{name}'.")
    f, info = known[name]
    if info.item_ref:
        return _item_ref(f, info, value)
    if info.data_key:
        # Which names the item has depends on its data: the Viewer checks that.
        if value is not None and not isinstance(value, str):
            raise OptionError(f"{info.label} must be a name or None; got {value!r}.")
        return value
    kind: type = type(f.default)
    value = _as_kind(info, kind, value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        _check_range(info, value)
    if info.choices is not None:
        if value not in info.choices:
            raise OptionError(
                f"{info.label} must be one of {', '.join(info.choices)}; got {value!r}."
            )
        value = info.choices[info.choices.index(value)]  # the member, not the text
    if info.check is not None:
        info.check(value)
    if kind is Colour:
        value = Colour(str(value).lower())
    return value


def _item_ref(f: Field[Any], info: OptionInfo, value: object) -> object:
    """Returns value as an item's id or None, or as a tuple of ids."""
    # Which items it may name depends on the scene: the Viewer checks that.
    if isinstance(f.default, tuple):
        return _whole_numbers(info, value)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise OptionError(f"{info.label} must be an item's id or None; got {value!r}.")
    return int(value)


def _as_kind(info: OptionInfo, kind: type, value: object) -> object:
    """Returns value converted to the type of the option's default, or raises."""
    if kind is bool:
        if not isinstance(value, bool):
            raise OptionError(f"{info.label} must be on or off; got {value!r}.")
    elif kind is float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise OptionError(f"{info.label} must be a number; got {value!r}.")
        value = float(value)
    elif kind is int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise OptionError(f"{info.label} must be a whole number; got {value!r}.")
        value = int(value)
    elif kind in (str, Colour) and not isinstance(value, str):
        raise OptionError(f"{info.label} must be text; got {value!r}.")
    elif kind is tuple:
        value = _whole_numbers(info, value)
    elif kind is Spheres:
        value = _spheres(info, value)
    return value


def _check_range(info: OptionInfo, value: float) -> None:
    """Raises OptionError if a number is not finite or outside the option's range."""
    if not math.isfinite(value):
        raise OptionError(f"{info.label} must be a finite number; got {value!r}.")
    too_low = info.minimum is not None and value < info.minimum
    too_high = info.maximum is not None and value > info.maximum
    if too_low or too_high:
        raise OptionError(f"{info.label} must be {range_text(info)}; got {value:g}.")


def _whole_numbers(info: OptionInfo, value: object) -> tuple[int, ...]:
    """Returns a collection of whole numbers as a sorted tuple without repeats."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise OptionError(f"{info.label} must be whole numbers; got {value!r}.")
    values = list(value)
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in values):
        raise OptionError(f"{info.label} must be whole numbers; got {value!r}.")
    return tuple(sorted({int(v) for v in values}))


def _spheres(info: OptionInfo, value: object) -> Spheres:
    """Returns a collection of (x, y, z, radius) as Spheres, in the order given."""
    error = OptionError(
        f"{info.label} must be spheres as (x, y, z, radius) in mm, each a finite "
        f"number and the radius positive; got {value!r}."
    )
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise error
    spheres: list[tuple[float, float, float, float]] = []
    for sphere in value:
        if isinstance(sphere, (str, bytes)) or not isinstance(sphere, Iterable):
            raise error
        numbers = list(sphere)
        if len(numbers) != 4 or any(
            isinstance(n, bool) or not isinstance(n, Real) or not math.isfinite(n)
            for n in numbers
        ):
            raise error
        x, y, z, radius = (float(n) for n in numbers)
        if radius <= 0:
            raise error
        spheres.append((x, y, z, radius))
    return Spheres(spheres)


def _check_lookup_table(value: object) -> None:
    """Raises OptionError unless the value is "" or a readable lookup table."""
    if value == "":
        return
    try:
        read_lookup_table(str(value))
    except ValueError as exc:
        raise OptionError(str(exc)) from exc


def range_text(info: OptionInfo) -> str:
    """Returns the legal range of an option, in words."""
    if info.minimum is not None and info.maximum is not None:
        return f"between {info.minimum:g} and {info.maximum:g}"
    if info.minimum is not None:
        return f"at least {info.minimum:g}"
    return f"at most {info.maximum:g}"


def _drawn_in_3d(options: object) -> bool:
    """Tells whether the 3D view draws items with these options; it skips glyphs."""
    return not isinstance(options, (FodOptions, PeaksOptions))


def _without_dti(options: object) -> bool:
    """Tells whether an image is shown without DTI colours."""
    return not getattr(options, "dti", False)


def _with_dti(options: object) -> bool:
    """Tells whether an image is shown with DTI colours."""
    return bool(getattr(options, "dti", False))


def _single_colour(options: object) -> bool:
    """Tells whether the item is drawn in its one colour."""
    return getattr(options, "colouring", None) == Colouring.SINGLE


def _by_value(options: object) -> bool:
    """Tells whether a tractogram is coloured by a value per point."""
    return getattr(options, "colouring", None) == Colouring.VALUES


def _normals_single(options: object) -> bool:
    """Tells whether a surface's normals are drawn in one colour."""
    return getattr(options, "normals_colouring", None) == Colouring.SINGLE


@dataclass(frozen=True)
class ItemOptions:
    """The options every item has."""

    visible: bool = option(True, "Visible", group=OptionGroup.DISPLAY)
    opacity_2d: float = option(
        1.0,
        "Opacity (2D)",
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        scope=Scope.TWO_D,
        group=OptionGroup.DISPLAY,
    )
    opacity_3d: float = option(
        1.0,
        "Opacity (3D)",
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        scope=Scope.THREE_D,
        group=OptionGroup.DISPLAY,
        shown_when=_drawn_in_3d,
    )


@dataclass(frozen=True)
class VolumeOptions(ItemOptions):
    """A volume's options (its colormap, range, interpolation and frame)."""

    volume_index: int = option(
        0,
        "Volume index",
        minimum=0,
        step=1,
        per_item=True,
        group=OptionGroup.DISPLAY,
        help="Which volume of a 4D image is shown.",
    )
    colormap: Colormap = option(
        Colormap.GRAY,
        "Colormap",
        choices=COLORMAPS,
        group=OptionGroup.COLOUR,
        enabled_when=_without_dti,
    )
    interpolate: bool = option(
        True,
        "Interpolate",
        group=OptionGroup.DISPLAY,
        help="Smooth (trilinear) or blocky (nearest voxel) slices.",
    )
    # No fixed bounds: the slider spans the data's range, and a volume starts at it.
    minimum: float = option(
        0.0,
        "Minimum",
        group=OptionGroup.COLOUR,
        enabled_when=_without_dti,
    )
    maximum: float = option(
        1.0,
        "Maximum",
        group=OptionGroup.COLOUR,
        enabled_when=_without_dti,
    )
    colour_bar: bool = option(
        False,
        "Colour bar",
        scope=Scope.TWO_D,
        group=OptionGroup.TWO_D,
        enabled_when=_without_dti,
    )
    slices_3d: bool = option(
        True,
        "Slices",
        scope=Scope.THREE_D,
        group=OptionGroup.THREE_D,
        help="Draws the image's three slices where they are in the 3D view.",
    )
    dti: bool = option(
        False,
        "DTI colours",
        per_item=True,
        group=OptionGroup.COLOUR,
        help="Shows a 3-volume direction image as red, green and blue for x, y and z.",
    )
    modulate: int | None = option(
        None,
        "Modulate by",
        per_item=True,
        item_ref=True,
        group=OptionGroup.COLOUR,
        help="An image on the same grid whose values, clipped to 0 to 1, scale the "
        "colours (such as FA).",
        enabled_when=_with_dti,
    )


# The options a DTI colour map does not use, as its colours come from the data.
DTI_UNUSED: Final = frozenset(
    {
        OptionName.COLORMAP,
        OptionName.MINIMUM,
        OptionName.MAXIMUM,
        OptionName.THRESHOLD,
        OptionName.LABELS,
        OptionName.LOOKUP_TABLE,
        OptionName.COLOUR_BAR,
    }
)


@dataclass(frozen=True)
class OverlayOptions(VolumeOptions):
    """An overlay's options: a volume's, a threshold, and showing it as labels."""

    threshold: float = option(
        0.0,
        "Threshold",
        group=OptionGroup.COLOUR,
        help="Values at or below it are transparent, so only the part of the map "
        "above it shows.",
        enabled_when=_without_dti,
    )
    # The mesh where the image crosses the threshold, drawn in the 3D view; for an
    # overlay shown as labels, a mesh per label of surface_labels instead.
    surface_3d: bool = option(
        True,
        "Surface",
        scope=Scope.THREE_D,
        group=OptionGroup.THREE_D,
        help="Draws the surface where the overlay crosses its threshold, such as a "
        "mask's outline, in the 3D view.",
    )
    # Shows the image as a parcellation, one colour per label.
    labels: bool = option(
        False,
        "Labels",
        group=OptionGroup.LABELS,
        help="Shows the image as a parcellation: one colour per label.",
        enabled_when=_without_dti,
    )
    lookup_table: str = option(
        "",
        "Lookup table",
        path=True,
        check=_check_lookup_table,
        group=OptionGroup.LABELS,
        help="The file naming and colouring the labels (FreeSurfer or MRtrix "
        "format); empty for built-in colours.",
        enabled_when=_without_dti,
    )  # "" for the built-in colours
    # The labels drawn as surfaces in the 3D view, chosen in the Cursor tab; none
    # at first, as an aparc+aseg's hundred surfaces take a while to compute.
    surface_labels: tuple[int, ...] = option(
        (),
        "Labels in 3D",
        scope=Scope.THREE_D,
        per_item=True,
        panel=False,
        group=OptionGroup.THREE_D,
    )


class Colouring(StrEnum):
    """How a tractogram's streamlines are coloured."""

    DIRECTION = "direction"  # red, green, blue along x, y, z
    SINGLE = "single_colour"  # the tractogram's Colour option
    PER_TRACTOGRAM = "per_tractogram"  # a different colour for each tractogram
    VALUES = "by_value"  # a value per point, through a colormap


COLOURINGS: tuple[Colouring, ...] = tuple(Colouring)


class TractStyle(StrEnum):
    """How the 3D view draws a tractogram's streamlines."""

    LINES = "lines"
    TUBES = "tubes"  # lit tubes, up to the tube limit (a setting)


TRACT_STYLES: tuple[TractStyle, ...] = tuple(TractStyle)


class RegionRole(StrEnum):
    """What a region does to the streamlines of a tractogram."""

    INCLUDE = "include"  # streamlines pass through it
    EXCLUDE = "exclude"  # streamlines that pass through it are dropped
    WITHIN = "within"  # streamlines stay inside it


# The defaults new tractograms start with; the Settings sheet changes them.
POINTS_PER_STREAMLINE: Final = 20
SLAB_THICKNESS: Final = 5.0  # mm, centred on the slice


@dataclass(frozen=True)
class TractogramOptions(ItemOptions):
    """A tractogram's options (colour, values, subset, points, regions, slab, 3D)."""

    colouring: Colouring = option(
        Colouring.DIRECTION,
        "Colouring",
        choices=COLOURINGS,
        group=OptionGroup.COLOUR,
        help="By direction (red left-right, green front-back, blue up-down), in one "
        "colour, a colour per tractogram, or by value: a TSF file's values or a "
        "volume sampled along the streamlines, through the colormap.",
    )
    colour: Colour = option(
        Colour("#ffffff"),
        "Colour",
        check=check_colour,
        group=OptionGroup.COLOUR,
        shown_when=_single_colour,
    )
    # The values to colour by: a TSF file, or a volume sampled at the drawn points.
    # Setting one clears the other and resets the colormap's range, so both come
    # before the colormap and the range, which a session then restores.
    values_file: str = option(
        "",
        "TSF file",
        path=True,
        per_item=True,
        group=OptionGroup.COLOUR,
        help="A TSF file with a value per point of this tractogram (from MRtrix's "
        "tcksample or Larmor), for colouring by value; empty for none.",
    )
    values_volume: int | None = option(
        None,
        "Sampled volume",
        per_item=True,
        item_ref=True,
        group=OptionGroup.COLOUR,
        help="A loaded 3D image, such as FA, sampled (trilinear) at the drawn points "
        "for colouring by value; 0 outside the image.",
    )
    colormap: Colormap = option(
        Colormap.VIRIDIS,
        "Colormap",
        choices=COLORMAPS,
        group=OptionGroup.COLOUR,
        shown_when=_by_value,
    )
    minimum: float = option(
        0.0,
        "Minimum",
        group=OptionGroup.COLOUR,
        shown_when=_by_value,
    )
    maximum: float = option(
        1.0,
        "Maximum",
        group=OptionGroup.COLOUR,
        shown_when=_by_value,
    )
    colour_bar: bool = option(
        False,
        "Colour bar",
        scope=Scope.TWO_D,
        group=OptionGroup.TWO_D,
        help="Shows the colormap's range in the 2D views when colouring by value.",
        shown_when=_by_value,
    )
    # A percentage of the streamlines, drawn up to the drawing cap (a setting).
    subset: float = option(
        100.0,
        "Subset (%)",
        minimum=0.0,
        maximum=100.0,
        step=0.1,
        group=OptionGroup.STREAMLINES,
        help="The share of the streamlines drawn, taken at random, the same ones "
        "each time.",
    )
    points: int = option(
        POINTS_PER_STREAMLINE,
        "Points per streamline",
        minimum=2,
        maximum=1000,
        step=1,
        group=OptionGroup.STREAMLINES,
        help="How many points each drawn streamline keeps; more follow the curves "
        "better but draw slower.",
    )
    # The regions the drawn streamlines must pass through, must not, and must stay
    # within; the subset and the points per streamline then apply to the streamlines
    # kept.
    region_filter: bool = option(
        True,
        "Filter by regions",
        group=OptionGroup.REGIONS,
        help="Draws only the streamlines that pass through every include region, "
        "through no exclude region and stay within every within region; off, all "
        "are drawn and the regions are kept.",
    )
    ends_only: bool = option(
        False,
        "Ends only",
        group=OptionGroup.REGIONS,
        help="Tests only the first and last point of each streamline against the "
        "regions, to keep the streamlines that start or end in them.",
    )
    include_masks: tuple[int, ...] = option(
        (),
        "Include masks",
        per_item=True,
        panel=False,
        item_ref=True,
        group=OptionGroup.REGIONS,
    )
    exclude_masks: tuple[int, ...] = option(
        (),
        "Exclude masks",
        per_item=True,
        panel=False,
        item_ref=True,
        group=OptionGroup.REGIONS,
    )
    include_spheres: Spheres = option(
        Spheres(),
        "Include spheres",
        per_item=True,
        panel=False,
        group=OptionGroup.REGIONS,
    )
    exclude_spheres: Spheres = option(
        Spheres(),
        "Exclude spheres",
        per_item=True,
        panel=False,
        group=OptionGroup.REGIONS,
    )
    within_masks: tuple[int, ...] = option(
        (),
        "Within masks",
        per_item=True,
        panel=False,
        item_ref=True,
        group=OptionGroup.REGIONS,
    )
    within_spheres: Spheres = option(
        Spheres(),
        "Within spheres",
        per_item=True,
        panel=False,
        group=OptionGroup.REGIONS,
    )
    slab: bool = option(
        True,
        "Slab",
        scope=Scope.TWO_D,
        group=OptionGroup.SLAB,
        help="In the 2D views, draws only the parts of streamlines within the slab "
        "around the slice.",
    )
    slab_thickness: float = option(
        SLAB_THICKNESS,
        "Slab thickness (mm)",
        minimum=0.1,
        maximum=100.0,
        step=0.1,
        scope=Scope.TWO_D,
        group=OptionGroup.SLAB,
        help="How thick the slab around the slice is.",
    )
    # The 2D views always draw lines.
    style: TractStyle = option(
        TractStyle.LINES,
        "Style",
        choices=TRACT_STYLES,
        scope=Scope.THREE_D,
        group=OptionGroup.THREE_D,
        help="Lines, or lit tubes up to the tube limit in Settings.",
    )
    tube_radius: float = option(
        0.25,
        "Tube radius (mm)",
        minimum=0.05,
        maximum=2.0,
        step=0.05,
        scope=Scope.THREE_D,
        group=OptionGroup.THREE_D,
    )
    ambient_occlusion: bool = option(
        False,
        "Ambient occlusion",
        scope=Scope.THREE_D,
        group=OptionGroup.THREE_D,
        help="Darkens the lines or tubes that others hide from the light around "
        "them, so dense bundles show their depth; worked out on the GPU.",
    )
    ray_traced: bool = option(
        False,
        "Ray traced",
        scope=Scope.THREE_D,
        group=OptionGroup.THREE_D,
        help="Draws the streamlines as capsules of the tube radius, ray traced "
        "through a voxel grid on the GPU, with shadows; below full opacity, every "
        "layer shows in order.",
    )


class Basis(StrEnum):
    """The SH bases an FOD's coefficients can be in; the file does not say which."""

    TOURNIER = "tournier"  # MRtrix
    DESCOTEAUX = "descoteaux"
    LEEMANS = "leemans"  # ExploreDTI, MRIToolkit


BASES: tuple[Basis, ...] = tuple(Basis)


class GlyphScaling(StrEnum):
    """What an FOD glyph's size is relative to."""

    PER_VOXEL = "per_voxel"  # each glyph's largest lobe reaches the same size
    IMAGE = "whole_image"  # the image's high amplitude does


GLYPH_SCALINGS: tuple[GlyphScaling, ...] = tuple(GlyphScaling)


@dataclass(frozen=True)
class FodOptions(ItemOptions):
    """An FOD's options: its basis and how its glyphs are drawn in the 2D views."""

    basis: Basis = option(
        Basis.TOURNIER,
        "Basis",
        choices=BASES,
        group=OptionGroup.GLYPHS,
        help="The SH basis of the coefficients, which the file does not record: "
        "tournier for MRtrix.",
    )
    scale: float = option(
        1.0,
        "Scale",
        minimum=0.1,
        maximum=5.0,
        step=0.05,
        scope=Scope.TWO_D,
        group=OptionGroup.GLYPHS,
        help="The glyphs' size; at 1, the largest lobe reaches the edge of its voxel "
        "(with whole-image scaling, a lobe of the image's high amplitude does).",
    )
    scaling: GlyphScaling = option(
        GlyphScaling.IMAGE,
        "Scaling",
        choices=GLYPH_SCALINGS,
        scope=Scope.TWO_D,
        group=OptionGroup.GLYPHS,
        help="Per voxel scales each glyph on its own; whole image scales all by the "
        "image's high amplitude (the 90th percentile of the voxels' largest).",
    )
    threshold: float = option(
        0.05,
        "Threshold",
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        scope=Scope.TWO_D,
        group=OptionGroup.GLYPHS,
        help="A share of the image's high amplitude (the 90th percentile of the "
        "voxels' largest); a voxel whose largest is below it gets no glyph.",
    )
    # The anisotropy from the SH coefficients c, sqrt(1 - c0^2 / sum(c^2)).
    anisotropy: float = option(
        0.0,
        "Minimum anisotropy",
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        scope=Scope.TWO_D,
        group=OptionGroup.GLYPHS,
        help="0 for a round FOD, near 1 for a sharp one; raising it hides the round "
        "FODs of grey matter and CSF.",
    )
    # Each glyph's sphere has 10 * detail ** 2 + 2 vertices.
    detail: int = option(
        8,
        "Sphere detail",
        minimum=2,
        maximum=12,
        step=1,
        scope=Scope.TWO_D,
        group=OptionGroup.GLYPHS,
        help="How smooth each glyph is; higher is smoother and slower.",
    )
    colouring: Colouring = option(
        Colouring.DIRECTION,
        "Colouring",
        choices=(Colouring.DIRECTION, Colouring.SINGLE),
        scope=Scope.TWO_D,
        group=OptionGroup.COLOUR,
    )
    colour: Colour = option(
        Colour("#ffffff"),
        "Colour",
        check=check_colour,
        scope=Scope.TWO_D,
        group=OptionGroup.COLOUR,
        shown_when=_single_colour,
    )


# A surface's colour by its file's name, white matter yellow and pial red.
SURFACE_COLOURS: Final = {"white": Colour("#ffff00"), "pial": Colour("#ff0000")}
SURFACE_COLOUR: Final = Colour("#d9d9d9")  # any other surface
# The number of normals drawn in 3D at first.
NORMALS_COUNT: Final = 200_000


@dataclass(frozen=True)
class SurfaceOptions(ItemOptions):
    """A surface's options: its colour or values, and its lines in the 2D views."""

    colour: Colour = option(
        SURFACE_COLOUR, "Colour", check=check_colour, group=OptionGroup.COLOUR
    )
    # A name of the surface's values per vertex, or None for its colour. Before the
    # colormap and range, which a new name resets, so a session restores them.
    colour_by: str | None = option(
        None,
        "Colour by",
        per_item=True,
        data_key=True,
        group=OptionGroup.COLOUR,
        help="Colours the surface by its values per vertex, such as curv "
        "(curvature: sulci dark, gyri light) or a GIfTI file's thickness; None "
        "gives the surface's own colour.",
    )
    colormap: Colormap = option(
        Colormap.GRAY_R, "Colormap", choices=COLORMAPS, group=OptionGroup.COLOUR
    )
    minimum: float = option(-0.3, "Minimum", group=OptionGroup.COLOUR)
    maximum: float = option(0.3, "Maximum", group=OptionGroup.COLOUR)
    cut_lines: bool = option(
        True,
        "Cut lines",
        scope=Scope.TWO_D,
        group=OptionGroup.LINES,
        help="Draws the line where each slice cuts the surface on the 2D views, to "
        "check the surface against the T1.",
    )
    line_width: float = option(
        2.0,
        "Line width (px)",
        minimum=0.5,
        maximum=10.0,
        step=0.5,
        scope=Scope.TWO_D,
        group=OptionGroup.LINES,
    )
    # Needs a cortex label; masked on display only.
    hide_medial_wall: bool = option(
        False,
        "Hide medial wall",
        group=OptionGroup.DISPLAY,
        help="Hides the flat patch FreeSurfer closes each hemisphere with across "
        "the midline, which is not cortex; needs the subject's "
        "label/lh.cortex.label.",
    )
    show_normals: bool = option(
        False,
        "Show normals",
        group=OptionGroup.NORMALS,
        help="Draws the surface's normals: in 3D from a subset of its vertices, in "
        "2D along the cut lines; on a good surface they all point outward.",
    )
    # Capped at the surface's vertex count when drawn.
    normals_count: int = option(
        NORMALS_COUNT,
        "Normals in 3D",
        minimum=1,
        maximum=1_000_000,
        step=1,
        scope=Scope.THREE_D,
        group=OptionGroup.NORMALS,
        help="How many vertices show their normal in 3D, taken at random, the same "
        "ones each time.",
    )
    normals_length: float = option(
        2.0,
        "Normal length (mm)",
        minimum=0.1,
        maximum=20.0,
        step=0.1,
        group=OptionGroup.NORMALS,
        help="How long each normal's line is.",
    )
    normals_spacing: float = option(
        3.0,
        "Spacing in 2D (mm)",
        minimum=0.5,
        maximum=50.0,
        step=0.5,
        scope=Scope.TWO_D,
        group=OptionGroup.NORMALS,
        help="How far apart the normals are along the cut lines in the 2D views.",
    )
    normals_colouring: Colouring = option(
        Colouring.DIRECTION,
        "Normals colouring",
        choices=(Colouring.DIRECTION, Colouring.SINGLE),
        group=OptionGroup.NORMALS,
        help="By direction (red left-right, green front-back, blue up-down), or in "
        "the normals' colour.",
    )
    normals_colour: Colour = option(
        Colour("#ffffff"),
        "Normals colour",
        check=check_colour,
        group=OptionGroup.NORMALS,
        help="The normals' colour when they are drawn in one colour.",
        shown_when=_normals_single,
    )


@dataclass(frozen=True)
class PeaksOptions(ItemOptions):
    """A peaks image's options: the lines' length, threshold, width and colour."""

    scale: float = option(
        1.0,
        "Scale",
        minimum=0.05,
        maximum=10.0,
        step=0.05,
        scope=Scope.TWO_D,
        group=OptionGroup.LINES,
        help="The lines' length; at 1, the image's largest peak spans one voxel.",
    )
    amplitude: bool = option(
        True,
        "Length by amplitude",
        scope=Scope.TWO_D,
        group=OptionGroup.LINES,
        help="On, a line's length follows its peak's amplitude; off, every line has "
        "the same length.",
    )
    threshold: float = option(
        0.0,
        "Threshold",
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        scope=Scope.TWO_D,
        group=OptionGroup.LINES,
        help="A share of the image's largest amplitude; a peak below it is not drawn.",
    )
    # Draws a peak and its opposite as two lobes, each on its own side.
    asymmetric_peaks: bool = option(
        False,
        "Asymmetric peaks",
        scope=Scope.TWO_D,
        group=OptionGroup.LINES,
        help="Draw each peak from the voxel centre outwards only, for peaks of an "
        "asymmetric FOD whose directions' signs matter.",
    )
    line_width: float = option(
        2.0,
        "Line width (px)",
        minimum=0.5,
        maximum=10.0,
        step=0.5,
        scope=Scope.TWO_D,
        group=OptionGroup.LINES,
    )
    colouring: Colouring = option(
        Colouring.DIRECTION,
        "Colouring",
        choices=(Colouring.DIRECTION, Colouring.SINGLE),
        scope=Scope.TWO_D,
        group=OptionGroup.COLOUR,
    )
    colour: Colour = option(
        Colour("#ffffff"),
        "Colour",
        check=check_colour,
        scope=Scope.TWO_D,
        group=OptionGroup.COLOUR,
        shown_when=_single_colour,
    )
