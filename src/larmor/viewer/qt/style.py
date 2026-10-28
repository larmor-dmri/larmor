"""Styles the viewer with Qt's Fusion style, a neutral palette and viewer.qss."""

import re
from importlib.resources import files
from pathlib import Path
from string import Template

from PySide6.QtCore import QSize, QStandardPaths, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication, QFormLayout, QWidget

from larmor._log import LarmorLogger
from larmor.viewer.model.settings import Theme

RESOURCES = files("larmor.viewer") / "resources"
STYLESHEET = RESOURCES / "viewer.qss"
ICON_SOURCES = RESOURCES / "icons"  # Lucide icons

# The colours viewer.qss and the icons use, by theme; the accent is a setting.
# The chrome is neutral grey (equal red, green and blue) so it does not change how
# an image's contrast is seen; only the accent and danger are coloured.
COLOURS: dict[bool, dict[str, str]] = {
    True: {  # dark
        "window": "#2d2d2d",
        "field": "#1f1f1f",
        "bar": "#262626",
        "border": "#3b3b3b",
        "text": "#e1e1e1",
        "muted": "#a2a2a2",
        "faint": "#747474",
        "hover": "#3b3b3b",
        "pressed": "#4d4d4d",
        "danger": "#ff6b6b",
    },
    False: {  # light
        "window": "#f8f8f8",
        "field": "#ffffff",
        "bar": "#f0f0f0",
        "border": "#e1e1e1",
        "text": "#000000",
        "muted": "#737373",
        "faint": "#b1b1b1",
        "hover": "#ededed",
        "pressed": "#d4d4d4",
        "danger": "#d93025",
    },
}
# Each tinted icon: its Lucide source and the colour it is drawn in.
ICONS = {
    "eye-open": ("eye", "text"),
    "eye-closed": ("eye-off", "faint"),
    "eye-mixed": ("eye-dashed", "muted"),
    "enlarge": ("maximize-2", "muted"),
    "restore": ("minimize-2", "muted"),
    "check": ("check", "on_accent"),
    "dash": ("minus", "on_accent"),
    "chevron-right": ("chevron-right", "muted"),
    "chevron-down": ("chevron-down", "muted"),
    "chevron-up": ("chevron-up", "muted"),
    "close": ("x", "muted"),
    "info": ("info", "muted"),
    "step-ready": ("check", "accent"),
    "step-waiting": ("circle", "faint"),
    "settings": ("settings", "muted"),
    "help": ("circle-help", "muted"),
}

_requested: tuple[Theme, str] | None = None  # the theme and accent last asked for
_applied: tuple[bool, str] | None = None  # dark or not, and the accent, as styled
_icons: Path | None = None  # the folder of the icons tinted for the applied theme
_listening = False
_warned_cache = False  # the cache could not be written


def apply_style(theme: Theme, accent: str) -> None:
    """Styles the application; SYSTEM follows Qt's colour scheme as it changes."""
    global _requested, _listening
    _requested = (theme, accent)
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        return
    if not _listening:
        QGuiApplication.styleHints().colorSchemeChanged.connect(_on_scheme)
        _listening = True
    _restyle(app)


def icon_path(name: str) -> str:
    """Returns an icon tinted for the theme; the plain one before styling."""
    if _icons is None:
        return str(ICON_SOURCES / f"{ICONS[name][0]}.svg")
    return str(_icons / f"{name}.svg")


def screen_room(widget: QWidget, margin: int = 40) -> QSize:
    """Returns the room a window has on the screen widget is on, less a margin.

    Args:
        widget: A widget on the screen, or one about to be shown.
        margin: The room left free on each side, px.
    """
    room = widget.screen().availableGeometry().size()
    return QSize(max(room.width() - 2 * margin, 1), max(room.height() - 2 * margin, 1))


def form_layout(form: QWidget) -> QFormLayout:
    """Creates the layout of a form of fields, spaced alike in each panel using it."""
    layout = QFormLayout(form)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setHorizontalSpacing(12)
    layout.setVerticalSpacing(8)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    # In a narrow panel, a field that does not fit goes below its label.
    layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    return layout


def is_dark(theme: Theme) -> bool:
    """Returns whether a theme is dark; an unknown system scheme counts as dark."""
    if theme == Theme.SYSTEM:
        return QGuiApplication.styleHints().colorScheme() != Qt.ColorScheme.Light
    return theme == Theme.DARK


def _on_scheme(*_: object) -> None:
    """Styles the application again when the system's colour scheme changes."""
    app = QApplication.instance()
    if isinstance(app, QApplication) and _requested is not None:
        _restyle(app)


def _restyle(app: QApplication) -> None:
    """Styles the application for the theme and accent asked for, if they changed."""
    global _applied, _icons
    assert _requested is not None
    theme, accent = _requested
    dark = is_dark(theme)
    if _applied == (dark, accent):
        return
    colours = {**COLOURS[dark], **_accents(accent)}
    folder = _cache() / f"{'dark' if dark else 'light'}-{accent.lstrip('#')}"
    icons: Path | None = folder
    try:
        _tint_icons(folder, colours)
    except OSError as error:
        _warn_cache(error)
        icons = None
    _applied = (dark, accent)
    _icons = icons
    # Each icon by its path, so a plain one keeps its Lucide file name.
    text = re.sub(
        r"\$\{icons\}/([\w-]+)\.svg",
        lambda m: Path(icon_path(m[1])).as_posix(),
        STYLESHEET.read_text(encoding="utf-8"),
    )
    sheet = Template(text).safe_substitute(colours)
    app.setStyle("Fusion")
    app.setPalette(_palette(colours))
    app.setFont(_tabular(app.font()))
    app.setStyleSheet(sheet)


def _accents(accent: str) -> dict[str, str]:
    """Returns the accent and its lighter, softer and contrasting variants."""
    colour = QColor(accent)
    r, g, b = colour.red(), colour.green(), colour.blue()
    return {
        "accent": colour.name(),
        "accent_light": colour.lighter(130).name(),
        "accent_soft": f"rgba({r}, {g}, {b}, 0.22)",
        "accent_faint": f"rgba({r}, {g}, {b}, 0.12)",
        "on_accent": "#000000" if colour.lightnessF() > 0.75 else "#ffffff",
    }


def _cache() -> Path:
    """Returns the folder for the generated theme files and icons."""
    location = QStandardPaths.StandardLocation.CacheLocation
    return Path(QStandardPaths.writableLocation(location)) / "style"


def _warn_cache(error: OSError) -> None:
    """Warns once that the theme files could not be written."""
    global _warned_cache
    if not _warned_cache:
        LarmorLogger.warning(
            f"Could not write the viewer's theme files ({error}); the viewer uses "
            "plain icons."
        )
        _warned_cache = True


def _tint_icons(folder: Path, colours: dict[str, str]) -> None:
    """Writes each icon into folder, drawn in its colour of the theme."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, (source, colour) in ICONS.items():
        svg = (ICON_SOURCES / f"{source}.svg").read_text(encoding="utf-8")
        tinted = svg.replace("currentColor", colours[colour])
        (folder / f"{name}.svg").write_text(tinted, encoding="utf-8")


def _tabular(font: QFont) -> QFont:
    """Returns the font with tabular digits, so numbers do not jump as they change."""
    font.setFeature(QFont.Tag("tnum"), 1)
    return font


def _palette(colours: dict[str, str]) -> QPalette:
    """Returns the palette for what the stylesheet leaves to Fusion, such as dialogs."""
    palette = QPalette()
    role = QPalette.ColorRole
    for r, name in (
        (role.Window, "window"),
        (role.Base, "field"),
        (role.AlternateBase, "bar"),
        (role.Button, "window"),
        (role.ToolTipBase, "field"),
        (role.WindowText, "text"),
        (role.Text, "text"),
        (role.ButtonText, "text"),
        (role.ToolTipText, "text"),
        (role.PlaceholderText, "muted"),
        (role.Highlight, "accent"),
        (role.HighlightedText, "on_accent"),
        (role.Link, "accent"),
    ):
        palette.setColor(r, QColor(colours[name]))
    # Greyed-out widgets draw their text faint.
    disabled = QPalette.ColorGroup.Disabled
    for r in (role.WindowText, role.Text, role.ButtonText):
        palette.setColor(disabled, r, QColor(colours["faint"]))
    return palette
