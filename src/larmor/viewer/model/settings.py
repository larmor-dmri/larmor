"""Holds the viewer's application-wide settings and saves them as TOML."""

import json
import math
import os
import sys
import tomllib
from collections.abc import Callable
from dataclasses import Field, dataclass, fields, replace
from enum import Enum, StrEnum
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from larmor._log import LarmorLogger
from larmor.viewer.model.camera import Convention
from larmor.viewer.model.options import (
    POINTS_PER_STREAMLINE,
    SLAB_THICKNESS,
    Colour,
    OptionError,
    OptionInfo,
    check_colour,
    option,
    range_text,
)
from larmor.viewer.model.registry import MAX_STREAMLINES

# The name of the settings folder in the system's configuration folder.
CONFIG_FOLDER = "larmor"
# The name of the settings file.
SETTINGS_FILE = "viewer.toml"
# The default port of the MCP server.
MCP_PORT = 8765
# Settings that earlier versions saved and this one no longer has; read quietly.
RETIRED = frozenset(
    {
        "screenshot_scale",
        "screenshot_overlay",
        "screenshot_transparent",
        "phantom_layout",
    }
)


class Theme(StrEnum):
    """The window's colours, as the system's or always light or dark."""

    SYSTEM = "system"
    LIGHT = "light"
    DARK = "dark"


class SettingName(StrEnum):
    """The settings; each is a field of AppSettings."""

    THEME = "theme"
    ACCENT = "accent"
    CONVENTION = "convention"
    BACKGROUND = "background"
    CROSSHAIR = "crosshair"
    INTERPOLATE = "interpolate"
    READ_LIMIT = "read_limit"
    DRAW_CAP = "draw_cap"
    TUBE_LIMIT = "tube_limit"
    POINTS_PER_STREAMLINE = "points_per_streamline"
    SLAB_THICKNESS = "slab_thickness"
    SLAB_MEMORY = "slab_memory"
    LOWER_RESOLUTION = "lower_resolution"
    GPU = "gpu"
    MCP_PORT = "mcp_port"


@dataclass(frozen=True)
class AppSettings:
    """The preferences of the whole viewer, as the Settings panel shows them."""

    theme: Theme = option(
        Theme.SYSTEM,
        "Theme",
        choices=tuple(Theme),
        help="System follows the computer's light or dark mode; light and dark "
        "keep one.",
    )
    accent: Colour = option(
        Colour("#448aff"),
        "Accent colour",
        check=check_colour,
        help="The one colour in the panels, marking the selected item, the focused "
        "field and checked boxes.",
    )
    convention: Convention = option(
        Convention.RAS,
        "Display convention",
        choices=tuple(Convention),
        help="RAS shows the subject's right on the right of the screen, LAS on the "
        "left.",
    )
    background: Colour = option(
        Colour("#000000"),
        "Background",
        check=check_colour,
        help="The colour behind the images in the views.",
    )
    crosshair: Colour = option(
        Colour("#3fbf3f"),
        "Crosshair",
        check=check_colour,
        help="The colour of the crosshair in the views.",
    )
    interpolate: bool = option(
        True,
        "Interpolate new images",
        help="Whether images opened from now on start smooth (trilinear) or blocky "
        "(nearest voxel).",
    )
    # 0 reads every streamline.
    read_limit: int = option(
        MAX_STREAMLINES,
        "Streamlines read (0: all)",
        minimum=0,
        maximum=10_000_000,
        step=1000,
        help="A tractogram with more streamlines is reduced to a random subset of "
        "this size when read.",
    )
    draw_cap: int = option(
        100_000,
        "Streamlines drawn",
        minimum=1000,
        maximum=10_000_000,
        step=1000,
        help="The most streamlines drawn of each tractogram, whatever its subset.",
    )
    # The tubes' budget, counted in streamlines of POINTS_PER_STREAMLINE points.
    tube_limit: int = option(
        40_000,
        f"Streamlines drawn as tubes (at {POINTS_PER_STREAMLINE} points)",
        minimum=1000,
        maximum=200_000,  # a mesh of about 1.9 GB
        step=1000,
        help=f"How many streamlines of {POINTS_PER_STREAMLINE} points may be drawn "
        "as tubes; more points allow fewer, and past it the 3D view draws lines.",
    )
    points_per_streamline: int = option(
        POINTS_PER_STREAMLINE,
        "Points per streamline",
        minimum=2,
        maximum=200,
        step=1,
        help="How many points each streamline of a new tractogram keeps; more "
        "follow the curves better but draw slower.",
    )
    slab_thickness: float = option(
        SLAB_THICKNESS,
        "Slab thickness (mm)",
        minimum=0.5,
        maximum=50.0,
        step=0.5,
        help="The slab new tractograms start with: the 2D views show only the "
        "streamlines within it around the slice.",
    )
    slab_memory: float = option(
        3.0,
        "Slab memory (GB)",
        minimum=0.5,
        maximum=64.0,
        step=0.5,
        help="The memory the 2D views' slabs may take; past it, they are cut in a "
        "slower way.",
    )
    lower_resolution: bool = option(
        False,
        "Lower resolution while moving",
        help="Draws the views at half resolution while you drag or scroll; turn it "
        "on when the viewer runs on a remote display, such as over ssh -X.",
    )
    # The summary of a wgpu adapter; "" for the default one.
    gpu: str = option(
        "",
        "GPU",
        help="The graphics card the viewer draws with, on a computer with several; "
        "applies when the viewer starts again.",
    )
    # The port of the MCP server, which listens on 127.0.0.1.
    mcp_port: int = option(
        MCP_PORT,
        "Port",
        minimum=1024,
        maximum=65535,
        step=1,
        help="Where the viewer serves agents, on this computer only.",
    )

    @property
    def max_streamlines(self) -> int | None:
        """The read limit as the loader takes it; None reads every streamline."""
        return self.read_limit or None

    @property
    def slab_bytes(self) -> int:
        """The slab memory in bytes."""
        return round(self.slab_memory * 1024**3)


def setting_fields() -> dict[SettingName, tuple[Field[Any], OptionInfo]]:
    """Returns the settings' fields and their infos, in declaration order."""
    # SettingName(...) fails for a field with no member, so the two cannot drift apart.
    return {SettingName(f.name): (f, f.metadata["option"]) for f in fields(AppSettings)}


def validate_setting(name: SettingName, value: object) -> object:
    """Returns value as the setting's type if the setting accepts it."""
    if not isinstance(name, SettingName):
        raise TypeError(
            f"Expected a SettingName, such as SettingName.THEME; got {name!r}."
        )
    f, info = setting_fields()[name]
    kind: type = type(f.default)
    if kind is bool:
        if not isinstance(value, bool):
            raise OptionError(f"{info.label} must be on or off; got {value!r}.")
    elif kind in (int, float):
        value = _setting_number(info, kind, value)
    elif not isinstance(value, str):
        raise OptionError(f"{info.label} must be text; got {value!r}.")
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


def _setting_number(info: OptionInfo, kind: type, value: object) -> int | float:
    """Returns value as an int or float setting, checked against its range."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise OptionError(f"{info.label} must be a number; got {value!r}.")
    number: int | float
    if kind is int:
        if not isinstance(value, Integral):
            raise OptionError(f"{info.label} must be a whole number; got {value!r}.")
        number = int(value)
    else:
        number = float(value)
    if not math.isfinite(number):
        raise OptionError(f"{info.label} must be a finite number; got {number!r}.")
    too_low = info.minimum is not None and number < info.minimum
    too_high = info.maximum is not None and number > info.maximum
    if too_low or too_high:
        raise OptionError(f"{info.label} must be {range_text(info)}; got {number:g}.")
    return number


@dataclass(frozen=True)
class SettingChanged:
    """One setting changed from `old` to `new`."""

    name: SettingName
    old: object
    new: object


class Settings:
    """The current settings, changed one at a time and saved after each change."""

    def __init__(
        self, values: AppSettings | None = None, path: Path | None = None
    ) -> None:
        """Holds values; saves them to path on every change, unless path is None."""
        self.values = values if values is not None else AppSettings()
        self.path = path
        self._subscribers: list[Callable[[SettingChanged], None]] = []

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        """Reads the settings saved in path (the user's settings file by default)."""
        path = path if path is not None else settings_path()
        return cls(read_settings(path), path)

    def subscribe(
        self, callback: Callable[[SettingChanged], None]
    ) -> Callable[[], None]:
        """Calls callback with every change from now on; returns a function to stop."""
        self._subscribers.append(callback)
        return lambda: self._subscribers.remove(callback)

    def set(self, name: SettingName, value: object) -> None:
        """Changes a setting, saves the settings and tells the subscribers."""
        value = validate_setting(name, value)
        old = getattr(self.values, name)
        if value == old:
            return
        changes: dict[str, Any] = {name: value}
        self.values = replace(self.values, **changes)
        if self.path is not None:
            try:
                write_settings(self.values, self.path)
            except OSError as error:  # the change still applies in this session
                LarmorLogger.warning(
                    "Cannot save the settings to %s: %s", self.path, error
                )
        event = SettingChanged(name, old, value)
        # A copy, as a callback may unsubscribe while the loop runs.
        for callback in list(self._subscribers):
            callback(event)


def config_folder() -> Path:
    """Returns the folder for the user's settings, as each system expects it."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / CONFIG_FOLDER


def settings_path() -> Path:
    """Returns the path of the user's settings file."""
    return config_folder() / SETTINGS_FILE


def read_settings(path: Path) -> AppSettings:
    """Reads the settings in a TOML file; defaults for a missing file or bad value."""
    try:
        with path.open("rb") as file:
            saved = tomllib.load(file)
    except FileNotFoundError:
        return AppSettings()
    except (OSError, tomllib.TOMLDecodeError) as error:
        LarmorLogger.warning("Cannot read the settings in %s: %s", path, error)
        return AppSettings()
    values: dict[str, Any] = {}
    for key, value in saved.items():
        if key in RETIRED:
            continue
        try:
            name = SettingName(key)
            values[str(name)] = validate_setting(name, value)
        except (ValueError, OptionError) as error:  # unknown key or bad value
            LarmorLogger.warning("Ignoring the setting %s in %s: %s", key, path, error)
    return replace(AppSettings(), **values)


def write_settings(settings: AppSettings, path: Path) -> None:
    """Writes the settings to a TOML file, replacing it whole."""
    lines = ["# The viewer's settings, written by its Settings panel."]
    for name in setting_fields():
        lines.append(f"{name} = {_toml_value(getattr(settings, name))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written beside it, then moved over it, so a crash never leaves half a file.
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def _toml_value(value: object) -> str:
    """Returns a setting's value as TOML."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, Enum):
        value = value.value
    # JSON's string escapes are valid in a TOML basic string.
    return json.dumps(str(value), ensure_ascii=False)
