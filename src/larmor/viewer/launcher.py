"""Installs and removes the desktop launchers that start this install's viewer.

Linux gets a menu entry, macOS an app in ~/Applications and Windows a Start-menu
shortcut. The paths written are kept in a manifest, so that uninstalling removes
exactly them.
"""

import base64
import configparser
import json
import os
import plistlib
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from importlib.resources import files
from pathlib import Path

from larmor import __version__
from larmor._log import LarmorLogger
from larmor.viewer import APP_ID, APP_NAME
from larmor.viewer.model.settings import config_folder

ICON_SVG = files("larmor.viewer") / "resources" / "icon.svg"
MANIFEST_FILE = "launcher.json"
DESCRIPTION = "View volumes, tractograms, FODs and peaks"
# Marks the files an install writes, so that it knows them without its manifest.
MARKER_KEY = "X-Larmor-Launcher"
MARKER_COMMENT = f"<!-- {MARKER_KEY} -->"

# Registers apps with macOS, so that Finder offers them for the files they open.
LSREGISTER = Path(
    "/System/Library/Frameworks/CoreServices.framework/Frameworks/"
    "LaunchServices.framework/Support/lsregister"
)
MACOS_MINIMUM = "12.0"  # the oldest macOS that Qt 6 supports
# The image sizes of a macOS iconset, as (size in points, scale).
ICONSET = [(s, k) for s in (16, 32, 128, 256, 512) for k in (1, 2)]
ICO_SIZE = 256


class Platform(Enum):
    """The desktop systems a launcher is made for."""

    LINUX = "linux"
    MACOS = "macos"
    WINDOWS = "windows"


@dataclass(frozen=True)
class FileType:
    """A kind of file the launcher opens."""

    name: str  # what file managers show
    mime: str  # the MIME type, for Linux
    extensions: tuple[str, ...]  # without the leading dot


FILE_TYPES = (
    FileType("NIfTI image", "application/x-nifti", ("nii", "nii.gz")),
    FileType("MRtrix image", "application/x-mrtrix-image", ("mif", "mif.gz", "mih")),
    FileType("MRtrix tractogram", "application/x-mrtrix-tracks", ("tck",)),
    FileType("TrackVis tractogram", "application/x-trackvis-tracks", ("trk",)),
)


def this_platform() -> Platform:
    """Returns the system this process runs on; Linux for any other Unix."""
    if sys.platform == "win32":
        return Platform.WINDOWS
    if sys.platform == "darwin":
        return Platform.MACOS
    return Platform.LINUX


def viewer_command(platform: Platform) -> list[str]:
    """Returns the command, as a list of arguments, that starts this install's viewer.

    The paths are absolute. On Windows, pythonw.exe runs it without a console
    window when it exists.
    """
    python = Path(sys.executable)
    if platform == Platform.WINDOWS:
        windowed = python.with_name("pythonw.exe")
        python = windowed if windowed.exists() else python
    # -P keeps the folder it starts in (home, for a launcher) from shadowing packages.
    return [str(python), "-P", "-m", "larmor.main", "bv"]


def _desktop_quote(arg: str) -> str:
    """Quotes and escapes one argument of a .desktop file's Exec key."""
    arg = arg.replace("%", "%%")
    if any(c in arg for c in " \t\n\"'\\><~|&;$*?#()`"):
        for c in '\\"`$':
            arg = arg.replace(c, "\\" + c)
        arg = f'"{arg}"'
    return arg.replace("\\", "\\\\")  # the key's value is itself unescaped first


def desktop_entry(command: Sequence[str]) -> str:
    """Returns the .desktop file of the Linux menu entry."""
    mimes = "".join(f"{t.mime};" for t in FILE_TYPES)
    exec_line = " ".join(_desktop_quote(a) for a in command)
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={APP_NAME}\n"
        "GenericName=Neuroimaging viewer\n"
        f"Comment={DESCRIPTION}\n"
        f"Exec={exec_line} %F\n"
        f"Icon={APP_ID}\n"
        "Terminal=false\n"
        "Categories=Science;MedicalSoftware;Graphics;Viewer;\n"
        f"MimeType={mimes}\n"
        # X11 matches windows by their WM_CLASS, which Qt sets to the app's name;
        # Wayland matches them by the file's name, which is the app id.
        f"StartupWMClass={APP_NAME}\n"
        "StartupNotify=true\n"
        f"{MARKER_KEY}=true\n"
    )


def mime_package() -> str:
    """Returns the shared-mime-info package that names the file types Linux lacks."""
    types = []
    for t in FILE_TYPES:
        # A longer pattern than *.gz wins, so x.nii.gz is NIfTI, not gzip.
        globs = "".join(f'    <glob pattern="*.{e}"/>\n' for e in t.extensions)
        types.append(
            f'  <mime-type type="{t.mime}">\n'
            f"    <comment>{t.name}</comment>\n"
            f"{globs}"
            "  </mime-type>\n"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">\n'
        f"{''.join(types)}"
        "</mime-info>\n"
        f"{MARKER_COMMENT}\n"
    )


def linux_icon() -> bytes:
    """Returns the viewer's SVG icon, marked as the launcher's."""
    return ICON_SVG.read_bytes().rstrip() + f"\n{MARKER_COMMENT}\n".encode()


def info_plist() -> bytes:
    """Returns the Info.plist of the macOS app."""
    documents = [
        {
            "CFBundleTypeName": t.name,
            # macOS matches only the last part of a name, so .nii.gz is any .gz file;
            # the app is therefore offered for them, never their default.
            "CFBundleTypeExtensions": sorted(
                {e.rsplit(".", 1)[-1] for e in t.extensions}
            ),
            "CFBundleTypeRole": "Viewer",
            "LSHandlerRank": "Alternate",
        }
        for t in FILE_TYPES
    ]
    plist = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": APP_ID,
        "CFBundleExecutable": APP_NAME,
        "CFBundleIconFile": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": __version__,
        "CFBundleVersion": __version__,
        "CFBundleDocumentTypes": documents,
        "LSMinimumSystemVersion": MACOS_MINIMUM,
        "NSHighResolutionCapable": True,
    }
    return plistlib.dumps(plist)


def macos_script(command: Sequence[str]) -> str:
    """Returns the macOS app's executable, a script that runs the viewer."""
    return f'#!/bin/sh\nexec {shlex.join(command)} "$@"\n'


def _ps_quote(text: str) -> str:
    """Returns text as a PowerShell string literal, in single quotes."""
    return "'" + text.replace("'", "''") + "'"


# Sets the shortcut's AppUserModelID, so that Windows groups the viewer's window
# (which sets the same ID) with it on the taskbar.
_SET_APP_ID = """Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class LauncherAppId {
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct PropertyKey { public Guid FormatId; public uint PropertyId; }
    [StructLayout(LayoutKind.Explicit, Size = 24)]
    public struct PropVariant {
        [FieldOffset(0)] public ushort Type;
        [FieldOffset(8)] public IntPtr Value;
    }
    [ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"),
     InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IPropertyStore {
        void GetCount(out uint count);
        void GetAt(uint index, out PropertyKey key);
        void GetValue(ref PropertyKey key, out PropVariant value);
        void SetValue(ref PropertyKey key, ref PropVariant value);
        void Commit();
    }
    [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
    static extern void SHGetPropertyStoreFromParsingName(
        string path, IntPtr context, int flags, ref Guid iid, out IPropertyStore store);
    public static void Set(string path, string id) {
        Guid iid = typeof(IPropertyStore).GUID;
        IPropertyStore store;
        SHGetPropertyStoreFromParsingName(path, IntPtr.Zero, 2, ref iid, out store);
        PropertyKey key = new PropertyKey();
        key.FormatId = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3");
        key.PropertyId = 5;
        PropVariant value = new PropVariant();
        value.Type = 31;
        value.Value = Marshal.StringToCoTaskMemUni(id);
        try { store.SetValue(ref key, ref value); store.Commit(); }
        finally { Marshal.FreeCoTaskMem(value.Value); Marshal.ReleaseComObject(store); }
    }
}
'@
"""


def windows_shortcut_script(shortcut: Path, command: Sequence[str], icon: Path) -> str:
    """Returns the PowerShell script that writes the Start-menu shortcut."""
    target, *args = command
    arguments = subprocess.list2cmdline(args)
    return (
        "$ErrorActionPreference = 'Stop'\n"
        "$shell = New-Object -ComObject WScript.Shell\n"
        f"$link = $shell.CreateShortcut({_ps_quote(str(shortcut))})\n"
        f"$link.TargetPath = {_ps_quote(target)}\n"
        f"$link.Arguments = {_ps_quote(arguments)}\n"
        f"$link.IconLocation = {_ps_quote(str(icon))}\n"
        f"$link.WorkingDirectory = {_ps_quote(str(Path.home()))}\n"
        f"$link.Description = {_ps_quote(DESCRIPTION)}\n"
        "$link.Save()\n"
        "try {\n"
        f"{_SET_APP_ID}"
        f"    [LauncherAppId]::Set({_ps_quote(str(shortcut))}, {_ps_quote(APP_ID)})\n"
        "} catch {\n"
        '    Write-Warning "The shortcut has no app ID: $_"\n'
        "}\n"
    )


def _xdg_data_home() -> Path:
    """Returns the user's data folder on Linux."""
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


def _start_menu() -> Path:
    """Returns the user's Start-menu programs folder on Windows."""
    base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    return base / "Microsoft" / "Windows" / "Start Menu" / "Programs"


def manifest_path() -> Path:
    """Returns the file that lists what the launcher's install wrote."""
    return config_folder() / MANIFEST_FILE


def _write_file(path: Path, content: str | bytes, mode: int = 0o644) -> None:
    """Writes a file with the given permissions, making its folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode() if isinstance(content, str) else content)
    path.chmod(mode)


def _run(*command: str) -> None:
    """Runs a system tool; its error output becomes an OSError's message."""
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        name = Path(command[0]).name
        raise OSError(f"{name} failed: {result.stderr.strip() or result.returncode}")


@dataclass
class _Written:
    """Records the files an install writes and the folders it makes."""

    paths: list[Path] = field(default_factory=list)
    folders: list[Path] = field(default_factory=list)

    def folder(self, path: Path) -> None:
        """Makes a folder, noting each one that did not exist."""
        missing = [p for p in (path, *path.parents) if not p.exists()]
        path.mkdir(parents=True, exist_ok=True)
        self.folders.extend(reversed(missing))

    def write(self, path: Path, content: str | bytes, mode: int = 0o644) -> None:
        """Writes a file, noting it and the folders made for it."""
        self.folder(path.parent)
        _write_file(path, content, mode)
        self.paths.append(path)


def _write_icns(path: Path) -> None:
    """Writes the viewer's icon as a macOS .icns, through iconutil."""
    from larmor.viewer.qt.app_icon import render_icon

    with tempfile.TemporaryDirectory() as folder:
        iconset = Path(folder) / f"{APP_NAME}.iconset"
        iconset.mkdir()
        for size, scale in ICONSET:
            name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
            (iconset / name).write_bytes(render_icon(size * scale))
        _run("iconutil", "-c", "icns", "-o", str(path), str(iconset))


def _run_if_present(*command: str) -> None:
    """Runs a system tool that refreshes a cache, if installed, ignoring errors."""
    if shutil.which(command[0]) is None and not Path(command[0]).is_file():
        return
    subprocess.run(command, check=False, capture_output=True)


def _linux_paths() -> tuple[Path, Path, Path]:
    """Returns the paths of the menu entry, its icon and the file types' package."""
    data = _xdg_data_home()
    return (
        data / "applications" / f"{APP_ID}.desktop",
        data / "icons" / "hicolor" / "scalable" / "apps" / f"{APP_ID}.svg",
        data / "mime" / "packages" / f"{APP_ID}.xml",
    )


def _macos_app() -> Path:
    """Returns the path of the macOS app."""
    return Path.home() / "Applications" / f"{APP_NAME}.app"


def _windows_paths() -> tuple[Path, Path]:
    """Returns the paths of the icon and the Start-menu shortcut."""
    return config_folder() / f"{APP_NAME}.ico", _start_menu() / f"{APP_NAME}.lnk"


def _targets(platform: Platform) -> list[Path]:
    """Returns the paths an install writes on platform."""
    if platform == Platform.LINUX:
        return list(_linux_paths())
    if platform == Platform.MACOS:
        return [_macos_app()]
    return list(_windows_paths())


def _is_our_app(app: Path) -> bool:
    """Returns whether a macOS app has the viewer's bundle id."""
    try:
        with (app / "Contents" / "Info.plist").open("rb") as file:
            return bool(plistlib.load(file).get("CFBundleIdentifier") == APP_ID)
    except OSError, plistlib.InvalidFileException, AttributeError:
        return False


def _is_our_desktop_entry(path: Path) -> bool:
    """Returns whether a .desktop file carries the launcher's marker key."""
    entry = configparser.ConfigParser(interpolation=None, strict=False)
    entry.optionxform = str  # type: ignore[assignment, method-assign]  # keys keep their case
    try:
        entry.read_string(path.read_text(encoding="utf-8"))
    except OSError, UnicodeDecodeError, configparser.Error:
        return False
    return entry.get("Desktop Entry", MARKER_KEY, fallback="") == "true"


def _is_our_shortcut(path: Path) -> bool:
    """Returns whether a .lnk holds the launcher's description and command."""
    try:
        data = path.read_bytes()
    except OSError:
        return False
    return all(
        text.encode("utf-16-le") in data for text in (DESCRIPTION, "larmor.main")
    )


def _is_ours(path: Path) -> bool:
    """Returns whether an install wrote a target, judged by content or folder."""
    name = path.name
    try:
        if name.endswith(".app"):
            return _is_our_app(path)
        if name.endswith(".desktop"):
            return _is_our_desktop_entry(path)
        if name.endswith(".lnk"):
            return _is_our_shortcut(path)
        if name.endswith((".svg", ".xml")):
            return path.read_bytes().rstrip().endswith(MARKER_COMMENT.encode())
        if name.endswith(".ico"):
            # Known by its folder, the viewer's own settings folder, as its bytes
            # change with the icon or Qt.
            return path.parent == config_folder()
    except OSError:
        return False
    return False


def _exists(path: Path) -> bool:
    """Returns whether a path exists, counting a broken symbolic link."""
    return path.exists() or path.is_symlink()


def _check_free(platform: Platform) -> None:
    """Refuses to write over a path the launcher did not write itself.

    A path in the manifest is checked too: another program's file may have
    taken its place since.

    Raises:
        OSError: If a target exists and is not known as the launcher's by its
            content.
    """
    for target in _targets(platform):
        if not _exists(target) or _is_ours(target):
            continue
        raise OSError(
            f"{target} exists and was not written by --install-launcher (nor by an "
            "older one, if the settings folder holding its manifest was deleted); "
            "move or remove it, then install the launcher again."
        )


def _install_linux(written: _Written, command: list[str]) -> None:
    """Writes the Linux menu entry, its icon and file types, and refreshes caches."""
    desktop, icon, package = _linux_paths()
    applications, mime = desktop.parent, package.parent.parent
    written.write(desktop, desktop_entry(command))
    written.write(icon, linux_icon())
    written.write(package, mime_package())
    _run_if_present("update-mime-database", str(mime))
    _run_if_present("update-desktop-database", str(applications))


def _install_macos(written: _Written, command: list[str]) -> None:
    """Writes the macOS app and registers it with the system."""
    app = _macos_app()
    contents = app / "Contents"
    written.folder(app.parent)
    written.paths.append(app)  # one entry: removing it removes all it holds
    _write_file(contents / "Info.plist", info_plist())
    _write_file(contents / "MacOS" / APP_NAME, macos_script(command), mode=0o755)
    (contents / "Resources").mkdir(exist_ok=True)
    _write_icns(contents / "Resources" / f"{APP_NAME}.icns")
    _run_if_present(str(LSREGISTER), "-f", str(app))


def _install_windows(written: _Written, command: list[str]) -> None:
    """Writes the Windows icon and the Start-menu shortcut."""
    from larmor.viewer.qt.app_icon import render_icon

    icon, shortcut = _windows_paths()
    written.folder(icon.parent)
    written.paths.append(icon)
    _write_file(icon, render_icon(ICO_SIZE, "ICO"))
    written.folder(shortcut.parent)
    script = windows_shortcut_script(shortcut, command, icon)
    written.paths.append(shortcut)
    # Encoded, so that no quote in the script meets the command line's quoting.
    encoded = base64.b64encode(script.encode("utf-16-le")).decode()
    _run("powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded)


_INSTALLERS = {
    Platform.LINUX: _install_linux,
    Platform.MACOS: _install_macos,
    Platform.WINDOWS: _install_windows,
}


def install_launcher() -> list[Path]:
    """Adds a launcher that starts this install's viewer, replacing an earlier one.

    Paths that exist but are not the launcher's own are never written over.

    Returns:
        The files and folders written.

    Raises:
        OSError: If a target exists that the launcher did not write (nothing is
            written then), the manifest is broken, or a file, the icon or the
            shortcut cannot be written.
    """
    platform = this_platform()
    _read_manifest()  # a broken one fails here, before anything is written
    _check_free(platform)
    uninstall_launcher()
    written = _Written()
    try:
        _INSTALLERS[platform](written, viewer_command(platform))
    finally:  # what was written is removed by an uninstall, even after an error
        _save_manifest(platform, written)
    return written.paths


def _save_manifest(platform: Platform, written: _Written) -> None:
    """Writes the manifest of what an install wrote, with the folders it made."""
    path = manifest_path()
    made = [p for p in (path.parent, *path.parent.parents) if not p.exists()]
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "platform": platform.value,
        "paths": [str(p) for p in written.paths],
        "folders": [str(p) for p in [*written.folders, *reversed(made)]],
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class _Manifest:
    """Holds what an install wrote, as read back from its manifest."""

    platform: Platform
    paths: list[Path]
    folders: list[Path]


def _read_manifest() -> _Manifest | None:
    """Returns the manifest of the installed launcher, or None if there is none.

    Raises:
        OSError: If the manifest cannot be read or is not one this module wrote.
    """
    path = manifest_path()
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        return _Manifest(
            Platform(saved["platform"]),
            [Path(p) for p in saved["paths"]],
            [Path(p) for p in saved["folders"]],
        )
    except FileNotFoundError:
        return None
    except (ValueError, KeyError, TypeError) as error:  # JSONDecodeError is one
        raise OSError(
            f"The launcher's manifest {path} is broken ({error!r}); delete it, and "
            "any launcher files left, by hand."
        ) from error


def uninstall_launcher() -> list[Path]:
    """Removes what the launcher's install wrote, and nothing else.

    Folders the install made are removed too, once empty. A listed path that no
    longer carries the launcher's mark is left, with a warning.

    Returns:
        The files and folders removed; none if no launcher is installed.

    Raises:
        OSError: If the manifest is broken.
    """
    manifest = _read_manifest() or _found_launcher()
    if manifest is None:
        return []
    platform, folders = manifest.platform, manifest.folders
    removed = [p for p in manifest.paths if _remove_written(platform, p)]
    if platform == Platform.LINUX:
        removed += _refresh_linux(folders)
    manifest_path().unlink(missing_ok=True)
    for folder in reversed(folders):  # the deepest first
        try:
            folder.rmdir()
        except OSError:  # not empty, or gone already
            continue
        removed.append(folder)
    return removed


def _found_launcher() -> _Manifest | None:
    """Returns the launcher files found by their content, or None if there are none.

    Used when the manifest is missing; logs a warning when files are found.
    """
    platform = this_platform()
    paths = [t for t in _targets(platform) if _exists(t) and _is_ours(t)]
    if not paths:
        return None
    LarmorLogger.warning(
        "No launcher manifest at %s (was the settings folder deleted?); "
        "removing the launcher files known by their content.",
        manifest_path(),
    )
    return _Manifest(platform, paths, [])


def _remove_written(platform: Platform, written: Path) -> bool:
    """Removes a path an install wrote, if it is still the launcher's.

    Returns:
        Whether the path was removed.
    """
    if not _exists(written):
        return False
    if not _is_ours(written):  # replaced by another program's since
        LarmorLogger.warning("Leaving %s: it is no longer the launcher's.", written)
        return False
    if platform == Platform.MACOS and written.suffix == ".app":
        _run_if_present(str(LSREGISTER), "-u", str(written))
    if written.is_dir() and not written.is_symlink():
        shutil.rmtree(written)
    elif written.exists() or written.is_symlink():
        written.unlink()
    else:
        return False
    return True


def _refresh_linux(folders: list[Path]) -> list[Path]:
    """Rebuilds the menu's and the file types' caches without the launcher's files.

    A cache the install made is removed instead, as it then holds only what the
    cache tools wrote.

    Returns:
        The paths removed.
    """
    data = _xdg_data_home()
    removed = []
    mime = data / "mime"
    if mime in folders and not any((mime / "packages").glob("*")):
        shutil.rmtree(mime, ignore_errors=True)
        removed.append(mime)
    else:
        _run_if_present("update-mime-database", str(mime))
    applications = data / "applications"
    if applications in folders and not any(applications.glob("*.desktop")):
        cache = applications / "mimeinfo.cache"
        if cache.exists():
            cache.unlink()
            removed.append(cache)
    else:
        _run_if_present("update-desktop-database", str(applications))
    return removed
