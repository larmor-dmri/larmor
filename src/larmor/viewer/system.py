"""Checks, before Qt is imported, that the viewer can start."""

import ctypes
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

INSTALL_HINT = (
    "Install it with: pip install 'larmor-dmri[viewer]' "
    "(or, in a uv project: uv add 'larmor-dmri[viewer]')."
)

# The Debian/Ubuntu and Fedora packages of each library Qt needs on X11.
X11_LIBRARIES: dict[str, tuple[str, str]] = {
    "libxcb.so.1": ("libxcb1", "libxcb"),
    "libX11.so.6": ("libx11-6", "libX11"),
    "libX11-xcb.so.1": ("libx11-xcb1", "libX11-xcb"),
    "libSM.so.6": ("libsm6", "libSM"),
    "libICE.so.6": ("libice6", "libICE"),
    "libxkbcommon.so.0": ("libxkbcommon0", "libxkbcommon"),
    "libxkbcommon-x11.so.0": ("libxkbcommon-x11-0", "libxkbcommon-x11"),
    "libxcb-cursor.so.0": ("libxcb-cursor0", "xcb-util-cursor"),
    "libxcb-icccm.so.4": ("libxcb-icccm4", "xcb-util-wm"),
    "libxcb-image.so.0": ("libxcb-image0", "xcb-util-image"),
    "libxcb-keysyms.so.1": ("libxcb-keysyms1", "xcb-util-keysyms"),
    "libxcb-render-util.so.0": ("libxcb-render-util0", "xcb-util-renderutil"),
    "libxcb-util.so.1": ("libxcb-util1", "xcb-util"),
    "libxcb-xinerama.so.0": ("libxcb-xinerama0", "libxcb"),
    "libxcb-xkb.so.1": ("libxcb-xkb1", "libxcb"),
    "libxcb-randr.so.0": ("libxcb-randr0", "libxcb"),
    "libxcb-shape.so.0": ("libxcb-shape0", "libxcb"),
    "libxcb-xfixes.so.0": ("libxcb-xfixes0", "libxcb"),
    "libxcb-sync.so.1": ("libxcb-sync1", "libxcb"),
    "libxcb-shm.so.0": ("libxcb-shm0", "libxcb"),
    "libxcb-glx.so.0": ("libxcb-glx0", "libxcb"),
    "libxcb-render.so.0": ("libxcb-render0", "libxcb"),
    "libfontconfig.so.1": ("libfontconfig1", "fontconfig"),
    "libfreetype.so.6": ("libfreetype6", "freetype"),
    "libEGL.so.1": ("libegl1", "libglvnd-egl"),
    "libGL.so.1": ("libgl1", "libglvnd-glx"),
    "libOpenGL.so.0": ("libopengl0", "libglvnd-opengl"),
    "libdbus-1.so.3": ("libdbus-1-3", "dbus-libs"),
    "libglib-2.0.so.0": ("libglib2.0-0", "glib2"),
}


class ViewerUnavailableError(RuntimeError):
    """Raised when the viewer cannot start on this machine."""


# The package that installs each module the viewer imports.
MODULES = {"PySide6": "PySide6", "PySide6QtAds": "PySide6-QtAds", "pygfx": "pygfx"}


def missing_modules() -> list[str]:
    """Returns the packages of the viewer's dependencies that are not installed."""
    return [
        package
        for module, package in MODULES.items()
        if importlib.util.find_spec(module) is None
    ]


def _qt_xcb_plugin() -> Path | None:
    """Returns Qt's X11 platform plugin in the installed PySide6, or None."""
    spec = importlib.util.find_spec("PySide6")
    if spec is None or not spec.submodule_search_locations:
        return None
    root = Path(next(iter(spec.submodule_search_locations)))
    plugin = root / "Qt" / "plugins" / "platforms" / "libqxcb.so"
    return plugin if plugin.is_file() else None


def _ldd_not_found(path: Path) -> list[str] | None:
    """Returns the libraries path needs that cannot be found; None if ldd fails."""
    ldd = shutil.which("ldd")
    if ldd is None:
        return None
    try:
        result = subprocess.run(
            [ldd, str(path)], capture_output=True, text=True, check=False, timeout=10
        )
    except OSError, subprocess.SubprocessError:
        return None
    if result.returncode != 0:
        return None
    # A missing library is listed as "libX11-xcb.so.1 => not found".
    return [
        line.split("=>")[0].strip()
        for line in result.stdout.splitlines()
        if "not found" in line
    ]


def missing_x11_libraries() -> list[str]:
    """Returns the libraries Qt's X11 plugin needs that cannot be loaded."""
    plugin = _qt_xcb_plugin()
    if plugin is not None and (not_found := _ldd_not_found(plugin)) is not None:
        return not_found
    missing: list[str] = []
    for name in X11_LIBRARIES:
        try:
            ctypes.CDLL(name)
        except OSError:
            missing.append(name)
    return missing


def check_viewer_can_start() -> None:
    """Raises ViewerUnavailableError if the viewer cannot start."""
    if missing := missing_modules():
        raise ViewerUnavailableError(
            f"The viewer needs {' and '.join(missing)}, which are not installed. "
            f"{INSTALL_HINT}"
        )
    if (
        not sys.platform.startswith("linux")
        or os.environ.get("QT_QPA_PLATFORM", "xcb") != "xcb"
    ):
        return  # macOS, Windows, or a platform chosen by hand (offscreen in tests)
    if os.environ.get("WAYLAND_DISPLAY"):
        return  # a Wayland desktop; Qt does not need the X libraries
    if not os.environ.get("DISPLAY"):
        raise ViewerUnavailableError(
            "There is no display to open the viewer on. On a remote server, connect "
            "with ssh -X (or -Y)."
        )
    if missing := missing_x11_libraries():
        known = [m for m in missing if m in X11_LIBRARIES]
        unknown = [m for m in missing if m not in X11_LIBRARIES]
        message = "Qt cannot open a window: these libraries are missing: "
        message += f"{', '.join(missing)}."
        if known:
            debian = sorted({X11_LIBRARIES[m][0] for m in known})
            fedora = sorted({X11_LIBRARIES[m][1] for m in known})
            message += (
                f"\nInstall them with: sudo apt install {' '.join(debian)}\n"
                f"(Fedora: sudo dnf install {' '.join(fedora)})."
            )
        if unknown:
            message += (
                f"\nInstall the packages of your distribution that provide "
                f"{', '.join(unknown)}."
            )
        message += (
            "\nWithout administrator rights, see 'Using the viewer on a Linux server' "
            "in docs/getting-started.md."
        )
        raise ViewerUnavailableError(message)
