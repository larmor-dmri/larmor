import logging
import re
from pathlib import Path

import pytest
from PySide6.QtCore import QLocale, QObject, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication, QDoubleSpinBox

from larmor.viewer import app as app_module
from larmor.viewer.model import settings as settings_module
from larmor.viewer.model.settings import SettingName, Theme
from larmor.viewer.qt import style
from larmor.viewer.qt.style import apply_style

ACCENT = "#ff00aa"
SELECTED_RULE = 'QFrame#viewPanel[selected="true"]'


def _rule(sheet: str, selector: str) -> str:
    """The body of the last rule with exactly this selector."""
    return sheet.rsplit(selector + " {", 1)[1].split("}", 1)[0]


def _window_lightness(qapp: QApplication) -> float:
    return qapp.palette().color(QPalette.ColorRole.Window).lightnessF()


def test_the_stylesheet_has_our_rules_with_the_accent_filled_in(
    qapp: QApplication,
) -> None:
    apply_style(Theme.DARK, ACCENT)
    sheet = qapp.styleSheet()
    rules = re.sub(r"/\*.*?\*/", "", sheet, flags=re.DOTALL)
    assert "${" not in rules
    assert ACCENT in _rule(sheet, SELECTED_RULE)


def test_dark_and_light_give_different_styles(qapp: QApplication) -> None:
    apply_style(Theme.DARK, ACCENT)
    dark, dark_window = qapp.styleSheet(), _window_lightness(qapp)
    apply_style(Theme.LIGHT, ACCENT)
    assert qapp.styleSheet() != dark
    assert dark_window < 0.5 < _window_lightness(qapp)


def test_an_unchanged_style_is_not_applied_again(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    apply_style(Theme.DARK, ACCENT)
    applied: list[str] = []
    monkeypatch.setattr(qapp, "setStyleSheet", applied.append)
    apply_style(Theme.DARK, ACCENT)
    assert applied == []


@pytest.mark.parametrize(
    ("name", "value"),
    [(SettingName.THEME, Theme.LIGHT), (SettingName.ACCENT, ACCENT)],
)
def test_a_theme_or_accent_setting_change_restyles(
    name: SettingName,
    value: object,
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Fresh settings in a file of their own, so the change outlives no test.
    path = tmp_path / settings_module.SETTINGS_FILE
    monkeypatch.setattr(settings_module, "settings_path", lambda: path)
    monkeypatch.setattr(app_module, "_SETTINGS", None)
    app_module.application()
    before = qapp.styleSheet()
    assert before != ""
    app_module.settings().set(name, value)
    assert qapp.styleSheet() != before


def test_number_boxes_use_a_decimal_point_whatever_the_system_language(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / settings_module.SETTINGS_FILE
    monkeypatch.setattr(settings_module, "settings_path", lambda: path)
    monkeypatch.setattr(app_module, "_SETTINGS", None)
    QLocale.setDefault(QLocale(QLocale.Language.Italian, QLocale.Country.Italy))
    app_module.application()
    box = QDoubleSpinBox()
    box.setDecimals(1)
    box.setValue(19.8)
    assert box.text() == "19.8"


class _Hints(QObject):
    colorSchemeChanged = Signal(Qt.ColorScheme)  # noqa: N815

    def __init__(self) -> None:
        super().__init__()
        self.scheme = Qt.ColorScheme.Dark

    def colorScheme(self) -> Qt.ColorScheme:  # noqa: N802
        return self.scheme


def test_following_the_system_restyles_when_its_scheme_changes(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The offscreen platform cannot switch its scheme, so the hints are faked.
    hints = _Hints()

    class _Gui:
        @staticmethod
        def styleHints() -> _Hints:  # noqa: N802
            return hints

    monkeypatch.setattr(style, "QGuiApplication", _Gui)
    monkeypatch.setattr(style, "_listening", False)
    apply_style(Theme.SYSTEM, ACCENT)
    assert _window_lightness(qapp) < 0.5
    hints.scheme = Qt.ColorScheme.Light
    hints.colorSchemeChanged.emit(Qt.ColorScheme.Light)
    assert _window_lightness(qapp) > 0.5


def _urls(sheet: str) -> list[str]:
    return re.findall(r"url\(([^)]*)\)", sheet)


def test_an_unwritable_cache_falls_back_to_plain_icons_with_one_warning(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def refuse(*args: object) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(style, "_tint_icons", refuse)
    with caplog.at_level(logging.WARNING, logger="larmor"):
        apply_style(Theme.DARK, ACCENT)
        apply_style(Theme.LIGHT, ACCENT)
    assert len(caplog.records) == 1
    sources = Path(str(style.ICON_SOURCES))
    assert Path(style.icon_path("check")) == sources / "check.svg"
    assert f'url("{sources.as_posix()}/check.svg")' in qapp.styleSheet()
    for url in _urls(qapp.styleSheet()):  # each plain icon by its own file name
        assert Path(url.strip('"')).is_file(), url


def test_icon_urls_are_quoted_and_found_with_a_space_in_the_cache_path(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(style, "_cache", lambda: tmp_path / "my cache")
    apply_style(Theme.DARK, ACCENT)
    ours = [u for u in _urls(qapp.styleSheet()) if "my cache" in u]
    assert ours
    for url in ours:
        assert url.startswith('"') and url.endswith('"')
        assert Path(url.strip('"')).is_file()


@pytest.mark.parametrize("dark", [True, False])
def test_the_chrome_colours_are_neutral_grey(dark: bool) -> None:
    for name, value in style.COLOURS[dark].items():
        colour = QColor(value)
        if name != "danger":
            assert colour.red() == colour.green() == colour.blue(), name


@pytest.mark.parametrize("theme", [Theme.DARK, Theme.LIGHT])
def test_styling_uses_fusion_tabular_digits_and_the_theme_palette(
    theme: Theme, qapp: QApplication
) -> None:
    # Offscreen Qt starts in Fusion, so start from another style.
    qapp.setStyle("Windows")
    apply_style(theme, ACCENT)
    assert qapp.font().featureValue(QFont.Tag("tnum")) == 1
    colours = style.COLOURS[theme == Theme.DARK]
    palette = qapp.palette()
    role = QPalette.ColorRole
    for r, name in ((role.Window, "window"), (role.Base, "field"), (role.Text, "text")):
        assert palette.color(r) == QColor(colours[name]), name
    # The stylesheet wraps the style and hides its name until it is cleared.
    qapp.setStyleSheet("")
    assert qapp.style().name() == "fusion"


def test_every_icon_the_stylesheet_names_has_a_lucide_source_with_its_licence() -> None:
    named = set(re.findall(r"\$\{icons\}/([\w-]+)\.svg", style.STYLESHEET.read_text()))
    assert named
    assert named <= set(style.ICONS)
    for source, _ in style.ICONS.values():
        assert (style.ICON_SOURCES / f"{source}.svg").is_file(), source
    assert (style.ICON_SOURCES / "LICENSE-lucide.txt").is_file()


def test_tinted_icons_carry_the_theme_colour_instead_of_current_color(
    qapp: QApplication,
) -> None:
    apply_style(Theme.LIGHT, ACCENT)
    for name, (_, colour) in style.ICONS.items():
        svg = Path(style.icon_path(name)).read_text(encoding="utf-8")
        assert "currentColor" not in svg, name
        if colour in style.COLOURS[False]:
            assert style.COLOURS[False][colour] in svg, name
