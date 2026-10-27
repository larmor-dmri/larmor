"""Writes the scene's items and view state to a JSON session file and reads them."""

import json
import math
import os
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

from larmor.viewer.model.camera import Camera2D, Camera3D, Cameras, Convention
from larmor.viewer.model.item import Category, Item, ItemId, ViewKind
from larmor.viewer.model.matlab_import import MatlabPart
from larmor.viewer.model.registration import Registration
from larmor.viewer.model.scene import Scene
from larmor.viewer.model.view_state import Tool, ViewField, ViewState

# The format name a session file holds.
FORMAT: Final = "larmor-viewer-session"
VERSION: Final = 1  # the newest version this code reads and the one it writes
SUFFIX: Final = ".lar.json"  # the usual extension, not required


class SessionError(ValueError):
    """Raised for a file that is not a session this version can read."""


@dataclass(frozen=True)
class SessionItem:
    """One item of a session file, with where its file is and how it was shown."""

    id: ItemId  # its id when saved, which the view state and options name
    category: Category
    name: str
    path: Path | None  # absolute
    relative_path: Path | None  # from the session file's folder
    options: Mapping[str, object]  # option name to value, as JSON holds them
    # What the item was read from inside a MATLAB file, its reference located.
    part: MatlabPart | None = None
    # The registration it was loaded through, its files located.
    registration: Registration | None = None

    def locate(self) -> Path | None:
        """Returns the relative path if its file exists, else the absolute path."""
        for candidate in (self.relative_path, self.path):
            if candidate is not None and candidate.is_file():
                return candidate
        return None


@dataclass(frozen=True)
class Session:
    """A session file as read, with its items in Layers order and its view state."""

    items: tuple[SessionItem, ...]
    view: Mapping[str, object]  # view-state field to value, as JSON holds them
    problems: tuple[str, ...] = ()  # entries that could not be read, for the user
    # The phantom editor's state, items named by their saved ids; None if none.
    phantom: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SessionReport:
    """What opening a session did.

    Attributes:
        items: The items opened, in Layers order.
        problems: The files that could not be opened and the options or view-state
            fields that were refused, one sentence each.
    """

    items: tuple[Item, ...] = ()
    problems: tuple[str, ...] = field(default=())


def missing_file(entry: SessionItem) -> str:
    """Returns the sentence saying that a session item's file cannot be opened."""
    where = [str(p) for p in (entry.relative_path, entry.path) if p is not None]
    return f"Cannot open {entry.name}: no file at {' or '.join(where) or 'no path'}."


def session_data(
    scene: Scene,
    folder: Path,
    phantom: Mapping[str, object] | None = None,
    owned: Collection[ItemId] = (),
) -> tuple[dict[str, object], list[str]]:
    """Returns the scene as a session file's JSON data, and the items left out.

    Items without a file (objects given from Python) are left out; those in
    `owned`, which the phantom block makes again, are not reported.

    Returns:
        The JSON data, and the names of the items left out.
    """
    plain = scene.to_dict()
    items: list[dict[str, object]] = []
    skipped: list[str] = []
    raw_items = plain["items"]
    assert isinstance(raw_items, list)
    for raw in raw_items:
        if raw["source"] is None:
            if raw["id"] not in owned:
                skipped.append(raw["name"])
            continue
        source = Path(raw["source"]).absolute()
        entry: dict[str, object] = {
            "id": raw["id"],
            "category": raw["category"],
            "name": raw["name"],
            "path": str(source),
            "relative_path": _relative(source, folder),
            "options": raw["options"],
        }
        part = raw.get("part")
        if isinstance(part, dict):
            reference = part.get("reference")
            if isinstance(reference, str):
                part = part | {"relative_reference": _relative(Path(reference), folder)}
            entry["matlab"] = part
        registration = raw.get("registration")
        if registration is not None:
            located = Registration.from_json(registration)
            entry["registration"] = located.to_json(folder)
        items.append(entry)
    data = {"format": FORMAT, "version": VERSION, "items": items, "view": plain["view"]}
    if phantom is not None:
        data["phantom"] = dict(phantom)
    return data, skipped


def write_session(
    scene: Scene,
    path: Path,
    phantom: Mapping[str, object] | None = None,
    owned: Collection[ItemId] = (),
) -> list[str]:
    """Writes the scene to a session file; returns the names of the items left out."""
    path = Path(path).absolute()
    data, skipped = session_data(scene, path.parent, phantom, owned)
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return skipped


def read_session(path: Path) -> Session:
    """Reads a session file; raises SessionError if it is not one this code reads."""
    path = Path(path).absolute()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise SessionError(f"{path} is not a session file: {e}.") from e
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise SessionError(f"{path} is not a Larmor viewer session.")
    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise SessionError(f"{path} has no valid session version: {version!r}.")
    if version > VERSION:
        raise SessionError(
            f"{path} was saved by a newer Larmor (session version {version}); "
            f"this one reads up to version {VERSION}. Update Larmor to open it."
        )
    items: list[SessionItem] = []
    problems: list[str] = []
    raw_items = data.get("items", [])
    for number, raw in enumerate(raw_items if isinstance(raw_items, list) else [], 1):
        try:
            items.append(_item(raw, path.parent))
        except (KeyError, TypeError, ValueError) as e:
            problems.append(f"Item {number} of the session cannot be read: {e}.")
    view = data.get("view", {})
    phantom = data.get("phantom")
    return Session(
        tuple(items),
        view if isinstance(view, dict) else {},
        tuple(problems),
        phantom if isinstance(phantom, dict) else None,
    )


def _item(raw: object, folder: Path) -> SessionItem:
    """Reads one item's entry; raises KeyError, TypeError or ValueError if malformed."""
    if not isinstance(raw, dict):
        raise TypeError("it is not a JSON object")
    item_id = raw["id"]
    if isinstance(item_id, bool) or not isinstance(item_id, int):
        raise TypeError(f"its id is not a whole number: {item_id!r}")
    options = raw.get("options", {})
    if not isinstance(options, dict):
        raise TypeError("its options are not a JSON object")
    path, relative = raw.get("path"), raw.get("relative_path")
    part = None
    if raw.get("matlab") is not None:
        part = MatlabPart.from_json(raw["matlab"])
        moved = raw["matlab"].get("relative_reference")
        if isinstance(moved, str):
            candidate = Path(os.path.normpath(folder / moved))
            if candidate.is_file() or part.reference is None:
                part = replace(part, reference=candidate)
    registration = None
    if raw.get("registration") is not None:
        registration = Registration.from_json(raw["registration"], folder)
    return SessionItem(
        id=item_id,
        category=Category(raw["category"]),
        name=str(raw["name"]),
        path=Path(path) if isinstance(path, str) else None,
        relative_path=(
            Path(os.path.normpath(folder / relative))
            if isinstance(relative, str)
            else None
        ),
        options=options,
        part=part,
        registration=registration,
    )


def _relative(source: Path, folder: Path) -> str | None:
    """Returns source's path from folder with forward slashes; None across drives."""
    try:
        return Path(os.path.relpath(source, folder)).as_posix()
    except ValueError:  # Windows: another drive
        return None


def view_state(
    raw: Mapping[str, object], ids: Mapping[ItemId, ItemId], current: ViewState
) -> tuple[ViewState, list[str]]:
    """Returns the view state a session holds, with its items' ids mapped by `ids`.

    A field that is missing, or names an item not opened, keeps its current value;
    one that cannot be read keeps it too and is reported.

    Returns:
        The view state, and the problems found.
    """
    changes: dict[str, Any] = {}
    problems: list[str] = []
    for name in ViewField:
        if name not in raw:
            continue
        try:
            value = _VIEW_FIELDS[name](raw[name], ids)
        except _UnmappedError:
            continue
        except (KeyError, TypeError, ValueError) as e:
            problems.append(f"The view's {name.replace('_', ' ')} is ignored: {e}.")
            continue
        changes[name] = value
    return replace(current, **changes), problems


class _UnmappedError(Exception):
    """Raised when the view state names an item that was not opened."""


def _text(value: object) -> str:
    """Returns value if it is text, else raises TypeError."""
    if not isinstance(value, str):
        raise TypeError(f"{value!r} is not text")
    return value


def _number(value: object) -> float:
    """Returns value as a finite float, else raises TypeError or ValueError."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{value!r} is not a number")
    if not math.isfinite(value):
        raise ValueError(f"{value!r} is not finite")
    return float(value)


def _vec3(value: object) -> tuple[float, float, float]:
    """Returns a list of three finite numbers as a tuple."""
    if not isinstance(value, list) or len(value) != 3:
        raise TypeError(f"{value!r} is not three numbers")
    x, y, z = (_number(v) for v in value)
    return (x, y, z)


def _flag(value: object, ids: Mapping[ItemId, ItemId]) -> bool:
    """Returns value if it is a bool, else raises TypeError."""
    if not isinstance(value, bool):
        raise TypeError(f"{value!r} is not true or false")
    return value


def _item_id(value: object, ids: Mapping[ItemId, ItemId]) -> ItemId:
    """Maps a saved item id to the opened item's id; raises _UnmappedError if none."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{value!r} is not an item's id")
    if value not in ids:
        raise _UnmappedError
    return ids[value]


def _main_volume(value: object, ids: Mapping[ItemId, ItemId]) -> ItemId | None:
    """Maps the saved main volume's id, or None."""
    return None if value is None else _item_id(value, ids)


def _selected(value: object, ids: Mapping[ItemId, ItemId]) -> ItemId | Category | None:
    """Maps the saved selection, an item's id or a category; None if not opened."""
    if value is None:
        return None
    if isinstance(value, str):
        return Category(value)
    try:
        return _item_id(value, ids)
    except _UnmappedError:
        return None  # its item was not opened: nothing selected


def _cameras(value: object, ids: Mapping[ItemId, ItemId]) -> Cameras:
    """Reads the saved cameras; a view missing from them keeps the default camera."""
    if not isinstance(value, dict):
        raise TypeError(f"{value!r} is not a JSON object")
    cameras = Cameras()
    for view in (ViewKind.AXIAL, ViewKind.CORONAL, ViewKind.SAGITTAL):
        raw = value.get(str(view))
        if isinstance(raw, dict):
            pan = raw.get("pan", [0.0, 0.0])
            if not isinstance(pan, list) or len(pan) != 2:
                raise TypeError(f"the {view} pan {pan!r} is not two numbers")
            camera = Camera2D(
                _number(raw.get("zoom", 1.0)), (_number(pan[0]), _number(pan[1]))
            )
            cameras = cameras.with_camera(view, camera)
    raw = value.get("three_d")
    if isinstance(raw, dict):
        default = Camera3D()
        camera_3d = Camera3D(
            _vec3(raw.get("forward", list(default.forward))),
            _vec3(raw.get("up", list(default.up))),
            _number(raw.get("zoom", 1.0)),
        )
        cameras = cameras.with_camera(ViewKind.THREE_D, camera_3d)
    return cameras


def _field_of_view(
    value: object, ids: Mapping[ItemId, ItemId]
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Reads the field of view as its lowest and highest corner."""
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError(f"{value!r} is not two corners")
    return (_vec3(value[0]), _vec3(value[1]))


def _maximised(value: object, ids: Mapping[ItemId, ItemId]) -> ViewKind | None:
    """Reads the maximised view, or None."""
    return None if value is None else ViewKind(_text(value))


# How each view-state field is read back from JSON, given the map of item ids.
_VIEW_FIELDS: Final[
    dict[ViewField, Callable[[object, Mapping[ItemId, ItemId]], object]]
] = {
    ViewField.CURSOR: lambda value, ids: _vec3(value),
    ViewField.MAIN_VOLUME: _main_volume,
    ViewField.SELECTED: _selected,
    ViewField.SELECTED_VIEW: lambda value, ids: ViewKind(_text(value)),
    ViewField.MAXIMISED: _maximised,
    ViewField.CONVENTION: lambda value, ids: Convention(_text(value)),
    ViewField.OVERLAY_VISIBLE: _flag,
    ViewField.CROSSHAIR_VISIBLE: _flag,
    ViewField.CAMERAS: _cameras,
    ViewField.FIELD_OF_VIEW: _field_of_view,
    ViewField.TOOL: lambda value, ids: Tool(_text(value)),
}
