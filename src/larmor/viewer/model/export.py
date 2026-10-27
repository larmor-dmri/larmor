"""Holds the export presets, which set how images and videos of the views are drawn."""

import json
import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

from larmor._log import LarmorLogger
from larmor.viewer.model import settings
from larmor.viewer.model.camera import Camera3D
from larmor.viewer.model.item import VIEW_GAP, Category, ViewKind
from larmor.viewer.model.options import (
    Colour,
    OptionError,
    OptionName,
    TractStyle,
    check_colour,
    validate,
)
from larmor.viewer.model.registry import REGISTRY

# The folder of the presets' files, beside the settings file.
PRESETS_FOLDER = "export_presets"
# The size of each view, in logical pixels, when a preset has none and no window
# gives one.
DEFAULT_SIZE = (800, 800)


class ExportKind(StrEnum):
    """What an export writes; each value is its name in the Export sheet."""

    IMAGE = "One image"
    VIEWS = "2D views as separate images"
    ROTATION = "3D rotation video"


@dataclass(frozen=True)
class ExportPreset:
    """How an export is drawn: its size, colours, overlay and option overrides.

    Attributes:
        name: The name the Export sheet and `Viewer.export` know it by.
        size: The width and height of each image or video frame in logical
            pixels; an image of several views shares it among them. None gives
            each view the size it has on screen, or `DEFAULT_SIZE` without a
            window.
        scale: Pixels per logical pixel in the image.
        background: The colour behind the images, "#rrggbb"; None for the views'.
        overlay: Whether to draw the information overlay; None follows the views.
        transparent: Whether the background is transparent.
        overrides: Option values given to every item of a category while the
            export is drawn, and taken back after it.
        camera: The 3D camera the export is drawn from; None for the 3D view's.
    """

    name: str
    size: tuple[int, int] | None = None
    scale: float = 1.0
    background: Colour | None = None
    overlay: bool | None = None
    transparent: bool = False
    overrides: Mapping[Category, Mapping[OptionName, object]] = field(
        default_factory=dict
    )
    camera: Camera3D | None = None


def _three_quarter(elevation: float) -> Camera3D:
    """Returns a camera from the subject's front right, raised by elevation degrees."""
    a, e = math.radians(45.0), math.radians(elevation)
    forward = (-math.cos(e) * math.sin(a), -math.cos(e) * math.cos(a), -math.sin(e))
    up = (-math.sin(e) * math.sin(a), -math.sin(e) * math.cos(a), math.cos(e))
    return Camera3D(forward, up)


_TUBES: dict[OptionName, object] = {
    OptionName.STYLE: TractStyle.TUBES,
    OptionName.AMBIENT_OCCLUSION: True,
}
# The presets shipped with the viewer; a file of the same name replaces one.
DEFAULT_PRESETS: tuple[ExportPreset, ...] = (
    ExportPreset("Screen"),
    ExportPreset(
        "Figure",
        scale=2.0,
        background=Colour("#ffffff"),
        overlay=False,
        transparent=True,
    ),
    ExportPreset(
        "Talk",
        scale=2.0,
        background=Colour("#000000"),
        overlay=False,
        overrides={Category.TRACTOGRAMS: _TUBES},
    ),
    ExportPreset(
        "Website",
        size=(1800, 1200),
        background=Colour("#000000"),
        overlay=False,
        overrides={
            Category.TRACTOGRAMS: {**_TUBES, OptionName.TUBE_RADIUS: 0.25},
            Category.FODS: {OptionName.DETAIL: 12},
            Category.VOLUMES: {OptionName.INTERPOLATE: False},
            Category.OVERLAYS: {OptionName.INTERPOLATE: False},
        },
        camera=_three_quarter(12.0),
    ),
)


def view_size(
    size: tuple[int, int], rows: Sequence[Sequence[ViewKind]]
) -> tuple[int, int]:
    """Returns each view's size in an image of the views laid out in rows.

    The views and the lines between them fill the image's size; the image is a
    pixel smaller when the views do not share it evenly.

    Args:
        size: The image's width and height in logical pixels.
        rows: The views, row by row.
    """
    columns = max(len(row) for row in rows)
    width = (size[0] - (columns - 1) * VIEW_GAP) // columns
    height = (size[1] - (len(rows) - 1) * VIEW_GAP) // len(rows)
    return max(width, 1), max(height, 1)


def presets_folder() -> Path:
    """Returns the folder of the user's preset files, beside the settings file."""
    return settings.settings_path().parent / PRESETS_FOLDER


def load_presets(folder: Path | None = None) -> dict[str, ExportPreset]:
    """Returns the presets by name: the shipped ones, then those of the folder's files.

    A file whose preset has a shipped preset's name replaces it. A file that
    cannot be read is skipped with a warning in the log.

    Args:
        folder: The folder of the preset files; `presets_folder()` if None.

    Returns:
        The presets, the shipped ones first.
    """
    folder = folder if folder is not None else presets_folder()
    presets = {p.name: p for p in DEFAULT_PRESETS}
    for path in sorted(folder.glob("*.toml")) if folder.is_dir() else []:
        try:
            preset = read_preset(path)
        except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
            LarmorLogger.warning("Ignoring the export preset %s: %s", path, error)
            continue
        presets[preset.name] = preset
    return presets


def find_preset(name: str, folder: Path | None = None) -> ExportPreset:
    """Returns the preset of that name.

    Raises:
        ValueError: If there is no preset of that name; the message lists them.
    """
    presets = load_presets(folder)
    if name not in presets:
        raise ValueError(
            f"There is no export preset {name!r}; the presets are {', '.join(presets)}."
        )
    return presets[name]


def preset_path(name: str, folder: Path | None = None) -> Path:
    """Returns the file a preset of that name is saved to."""
    folder = folder if folder is not None else presets_folder()
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.lower())
    return folder / f"{stem or 'preset'}.toml"


def save_preset(preset: ExportPreset, folder: Path | None = None) -> Path:
    """Writes a preset to its file in the folder, replacing it; returns the path."""
    path = preset_path(preset.name, folder)
    write_preset(preset, path)
    return path


def write_preset(preset: ExportPreset, path: Path) -> None:
    """Writes a preset to a TOML file, replacing it whole."""
    lines = ["# An export preset of the viewer, written by its Export sheet."]
    lines.append(f"name = {json.dumps(preset.name)}")
    if preset.size is not None:
        lines.append(f"size = [{preset.size[0]}, {preset.size[1]}]")
    lines.append(f"scale = {preset.scale!r}")
    if preset.background is not None:
        lines.append(f"background = {json.dumps(str(preset.background))}")
    if preset.overlay is not None:
        lines.append(f"overlay = {_toml(preset.overlay)}")
    lines.append(f"transparent = {_toml(preset.transparent)}")
    if preset.camera is not None:
        camera = preset.camera
        lines += [
            "",
            "[camera]",
            f"forward = [{', '.join(repr(v) for v in camera.forward)}]",
            f"up = [{', '.join(repr(v) for v in camera.up)}]",
            f"zoom = {camera.zoom!r}",
        ]
    for category, values in preset.overrides.items():
        lines += ["", f"[overrides.{category.value}]"]
        lines += [f"{name.value} = {_toml(value)}" for name, value in values.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_preset(path: Path) -> ExportPreset:
    """Reads a preset from a TOML file.

    Returns:
        The preset.

    Raises:
        OSError: If the file cannot be read.
        tomllib.TOMLDecodeError: If it is not TOML.
        ValueError: If a value is missing or of the wrong kind.
    """
    with path.open("rb") as file:
        raw = tomllib.load(file)
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("a preset needs a name")
    size = raw.get("size")
    if size is not None:
        if (
            not isinstance(size, list)
            or len(size) != 2
            or not all(isinstance(v, int) and v > 0 for v in size)
        ):
            raise ValueError(f"the size {size!r} is not two positive whole numbers")
        size = (size[0], size[1])
    scale = raw.get("scale", 1.0)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError(f"the scale {scale!r} is not a positive number")
    background = raw.get("background")
    if background is not None:
        try:
            check_colour(background)
        except OptionError as error:
            raise ValueError(str(error)) from error
        background = Colour(str(background).lower())
    overlay = raw.get("overlay")
    transparent = raw.get("transparent", False)
    for flag in (transparent,) if overlay is None else (overlay, transparent):
        if not isinstance(flag, bool):
            raise ValueError(f"{flag!r} is not true or false")
    return ExportPreset(
        name,
        size,
        float(scale),
        background,
        overlay,
        transparent,
        _read_overrides(raw.get("overrides", {})),
        _read_camera(raw.get("camera")),
    )


def _read_overrides(raw: object) -> dict[Category, dict[OptionName, object]]:
    """Reads a preset's option overrides, checked against the options' types.

    Returns:
        The overrides by category.

    Raises:
        ValueError: If a category or an option is unknown.
    """
    if not isinstance(raw, dict):
        raise ValueError("the overrides are not a table")
    overrides: dict[Category, dict[OptionName, object]] = {}
    for key, values in raw.items():
        category = Category(key)
        if category not in REGISTRY or not isinstance(values, dict):
            raise ValueError(f"the overrides of {key!r} are not a table")
        options = REGISTRY[category].options
        known = {f.name for f in fields(options)}
        overrides[category] = {}
        for name, value in values.items():
            if name not in known:
                raise ValueError(f"{key} have no option {name!r}")
            try:
                checked = validate(options, OptionName(name), value)
            except OptionError as error:
                raise ValueError(str(error)) from error
            overrides[category][OptionName(name)] = checked
    return overrides


def _read_camera(raw: object) -> Camera3D | None:
    """Reads a preset's 3D camera.

    Returns:
        The camera, or None if the preset has none.

    Raises:
        ValueError: If the camera's directions are not three numbers each.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("the camera is not a table")
    vectors: list[tuple[float, float, float]] = []
    for key in ("forward", "up"):
        value: Any = raw.get(key)
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(f"the camera's {key} is not three numbers")
        vectors.append((float(value[0]), float(value[1]), float(value[2])))
    return Camera3D(vectors[0], vectors[1], float(raw.get("zoom", 1.0)))


def _toml(value: object) -> str:
    """Returns a preset's value as TOML."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, Enum):
        value = value.value
    return json.dumps(str(value))
