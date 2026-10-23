import logging
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from larmor.viewer import Convention, OptionError
from larmor.viewer.model import settings as settings_module
from larmor.viewer.model.settings import (
    AppSettings,
    SettingChanged,
    SettingName,
    Settings,
    Theme,
    config_folder,
    setting_fields,
    validate_setting,
)


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "config" / "viewer.toml"


def test_changed_settings_are_read_back_from_their_file(path: Path) -> None:
    settings = Settings(path=path)
    changes: list[tuple[SettingName, object]] = [
        (SettingName.THEME, Theme.DARK),
        (SettingName.CONVENTION, Convention.LAS),
        (SettingName.ACCENT, "#12Ab9F"),
        (SettingName.BACKGROUND, "#AbCdEf"),
        (SettingName.INTERPOLATE, False),
        (SettingName.READ_LIMIT, 0),
        (SettingName.SLAB_THICKNESS, 7.5),
        (SettingName.SCREENSHOT_TRANSPARENT, True),
        (SettingName.LOWER_RESOLUTION, True),
        # Quotes and a backslash must survive TOML's string escapes.
        (SettingName.GPU, 'GPU "A" \\ (Vulkan)'),
    ]
    for name, value in changes:
        settings.set(name, value)
    loaded = Settings.load(path).values
    assert loaded == settings.values
    assert loaded.background == "#abcdef"
    assert loaded.accent == "#12ab9f"
    assert loaded.max_streamlines is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (SettingName.INTERPOLATE, 1),
        (SettingName.READ_LIMIT, "100"),
        (SettingName.READ_LIMIT, True),
        (SettingName.READ_LIMIT, 1.5),
        (SettingName.READ_LIMIT, -1),
        (SettingName.SCREENSHOT_SCALE, 5),
        (SettingName.SLAB_THICKNESS, math.inf),
        (SettingName.SLAB_THICKNESS, 0.1),
        (SettingName.THEME, "blue"),
        (SettingName.BACKGROUND, "red"),
        (SettingName.ACCENT, "#12345"),
        (SettingName.GPU, 3),
    ],
)
def test_a_bad_value_is_refused_and_nothing_is_saved_or_sent(
    name: SettingName, value: object, path: Path
) -> None:
    settings = Settings(path=path)
    events: list[SettingChanged] = []
    settings.subscribe(events.append)
    with pytest.raises(OptionError):
        settings.set(name, value)
    assert settings.values == AppSettings()
    assert not path.exists()
    assert events == []


@pytest.mark.parametrize(
    ("minimum", "maximum", "value", "message"),
    [(5.0, None, 1, "at least 5"), (None, 5.0, 9, "at most 5")],
    ids=["minimum only", "maximum only"],
)
def test_a_value_outside_a_one_sided_range_names_its_bound(
    minimum: float | None,
    maximum: float | None,
    value: int,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every real setting has both bounds, so give one a single bound.
    known = setting_fields()
    f, info = known[SettingName.READ_LIMIT]
    known[SettingName.READ_LIMIT] = (f, replace(info, minimum=minimum, maximum=maximum))
    monkeypatch.setattr(settings_module, "setting_fields", lambda: known)
    with pytest.raises(OptionError, match=f"must be {message}; got {value}"):
        validate_setting(SettingName.READ_LIMIT, value)


def test_a_setting_is_named_by_its_enum_member(path: Path) -> None:
    with pytest.raises(TypeError, match="SettingName"):
        Settings(path=path).set("theme", Theme.DARK)  # type: ignore[arg-type]


def test_each_change_is_sent_once_and_an_unchanged_value_not_at_all(
    path: Path,
) -> None:
    settings = Settings(path=path)
    events: list[SettingChanged] = []
    settings.subscribe(events.append)
    settings.set(SettingName.THEME, Theme.DARK)
    settings.set(SettingName.THEME, "dark")
    settings.set(SettingName.THEME, Theme.LIGHT)
    assert events == [
        SettingChanged(SettingName.THEME, Theme.SYSTEM, Theme.DARK),
        SettingChanged(SettingName.THEME, Theme.DARK, Theme.LIGHT),
    ]


@pytest.mark.parametrize(
    ("text", "expected", "warned"),
    [
        (None, AppSettings(), False),
        ("theme = \n", AppSettings(), True),
        ('theme = "dark"\ncolour = "red"\n', AppSettings(theme=Theme.DARK), True),
        ('theme = "dark"\nread_limit = -5\n', AppSettings(theme=Theme.DARK), True),
    ],
    ids=["missing", "broken", "unknown key", "bad value"],
)
def test_a_missing_or_bad_file_loads_the_defaults_for_what_it_lacks(
    text: str | None,
    expected: AppSettings,
    warned: bool,
    path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    if text is not None:
        path.parent.mkdir()
        path.write_text(text)
    with caplog.at_level(logging.WARNING, logger="larmor"):
        loaded = Settings.load(path).values
    assert loaded == expected
    assert bool(caplog.records) == warned


def test_a_failed_save_keeps_the_old_file_and_the_change(
    path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = Settings(path=path)
    settings.set(SettingName.THEME, Theme.DARK)
    saved = path.read_text()

    def fail(self: Path, target: Path) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", fail)
    with caplog.at_level(logging.WARNING, logger="larmor"):
        settings.set(SettingName.THEME, Theme.LIGHT)
    assert path.read_text() == saved
    assert settings.values.theme == Theme.LIGHT
    assert "Cannot save" in caplog.text


@pytest.mark.parametrize(
    ("platform", "variable", "base"),
    [
        ("win32", "APPDATA", "roaming"),
        ("darwin", None, "home/Library/Application Support"),
        ("linux", "XDG_CONFIG_HOME", "xdg"),
        ("linux", None, "home/.config"),
    ],
)
def test_the_settings_folder_is_where_each_system_keeps_them(
    platform: str,
    variable: str | None,
    base: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    for name in ("HOME", "USERPROFILE"):  # the home folder on POSIX and on Windows
        monkeypatch.setenv(name, str(tmp_path / "home"))
    for name in ("APPDATA", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(name, raising=False)
    if variable is not None:
        monkeypatch.setenv(variable, str(tmp_path / base))
    assert config_folder() == tmp_path / base / "larmor"


def test_every_setting_has_help() -> None:
    assert all(info.help for _, info in setting_fields().values())
