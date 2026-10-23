import base64
import configparser
import importlib.util
import json
import logging
import plistlib
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from larmor.viewer import APP_ID, APP_NAME, launcher
from larmor.viewer.launcher import (
    FILE_TYPES,
    MANIFEST_FILE,
    Platform,
    desktop_entry,
    info_plist,
    install_launcher,
    macos_script,
    uninstall_launcher,
    viewer_command,
    windows_shortcut_script,
)

COMMAND = ["/opt/my env/bin/python", "-P", "-m", "larmor.main", "bv"]


def test_desktop_entry_runs_the_command_on_the_files_with_its_icon_and_types() -> None:
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str  # type: ignore[assignment,method-assign]
    parser.read_string(desktop_entry(COMMAND))
    entry = parser["Desktop Entry"]
    assert shlex.split(entry["Exec"]) == [*COMMAND, "%F"]
    assert entry["Icon"] == APP_ID
    assert entry["StartupWMClass"] == APP_NAME
    assert set(filter(None, entry["MimeType"].split(";"))) == {
        t.mime for t in FILE_TYPES
    }


def test_info_plist_names_the_bundle_and_the_files_it_opens() -> None:
    plist = plistlib.loads(info_plist())
    assert plist["CFBundleName"] == APP_NAME
    assert plist["CFBundleIdentifier"] == APP_ID
    assert plist["CFBundleExecutable"] == APP_NAME
    extensions = {
        e for d in plist["CFBundleDocumentTypes"] for e in d["CFBundleTypeExtensions"]
    }
    # macOS matches the last suffix only, so .nii.gz and .mif.gz are offered as .gz.
    assert extensions == {"nii", "gz", "mif", "mih", "tck", "trk"}


@pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX shell")
def test_macos_script_passes_its_arguments_to_the_command(tmp_path: Path) -> None:
    script = tmp_path / "launch"
    script.write_text(macos_script(["printf", "<%s>"]))
    result = subprocess.run(
        ["sh", str(script), "a.nii", "my file.tck"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "<a.nii><my file.tck>"


def test_windows_command_runs_pythonw_when_it_is_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pythonw.exe").touch()
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python.exe"))
    assert viewer_command(Platform.WINDOWS)[0] == str(tmp_path / "pythonw.exe")


def test_windows_script_writes_the_shortcut_to_the_python_it_is_given() -> None:
    shortcut = Path("D:/Profiles/O'Brien/Start Menu/BrainViewer.lnk")
    target = "C:/Python/pythonw.exe"
    script = windows_shortcut_script(shortcut, [target, "-P", "bv"], Path("i.ico"))
    quoted = str(shortcut).replace("'", "''")
    assert f"$shell.CreateShortcut('{quoted}')" in script
    assert f"$link.TargetPath = '{target}'" in script
    assert "$link.Arguments = '-P bv'" in script


@pytest.fixture
def places(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Every folder the launcher writes to is under tmp_path.
    for name in ("HOME", "USERPROFILE"):  # the home folder on POSIX and on Windows
        monkeypatch.setenv(name, str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setattr(launcher, "config_folder", lambda: tmp_path / "config")
    return tmp_path


def _uninstall_is_clean(places: Path, written: Sequence[Path]) -> None:
    assert all(p.exists() for p in written)
    assert launcher.manifest_path().exists()
    removed = uninstall_launcher()
    assert set(written) <= set(removed)
    assert not any(p.exists() for p in written)
    assert not launcher.manifest_path().exists()
    assert not (places / "config").exists()
    assert uninstall_launcher() == []


@pytest.fixture
def linux(places: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(launcher, "this_platform", lambda: Platform.LINUX)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    return places


@pytest.fixture
def macos(places: Path, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    if importlib.util.find_spec("PySide6") is None:
        pytest.skip("the icon is drawn by Qt")
    calls: list[list[str]] = []

    def run(
        command: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append([Path(command[0]).name, *command[1:2]])
        if command[0] == "iconutil":  # writes the .icns it is asked for
            Path(command[command.index("-o") + 1]).write_bytes(b"icns")
        return subprocess.CompletedProcess(list(command), 0, "", "")

    monkeypatch.setattr(launcher, "this_platform", lambda: Platform.MACOS)
    monkeypatch.setattr("larmor.viewer.launcher.subprocess.run", run)
    monkeypatch.setattr(shutil, "which", lambda name: name)
    return calls


def test_linux_install_is_undone_by_uninstall_keeping_folders_it_found(
    linux: Path,
) -> None:
    applications = linux / "data" / "applications"
    applications.mkdir(parents=True)
    (applications / "other.desktop").write_text("")
    (linux / "data" / "icons").mkdir()
    written = install_launcher()
    assert applications / f"{APP_ID}.desktop" in written
    _uninstall_is_clean(linux, written)
    assert (applications / "other.desktop").exists()
    assert sorted(p.name for p in (linux / "data").iterdir()) == [
        "applications",
        "icons",
    ]


def test_linux_install_refuses_a_file_it_did_not_write_and_writes_nothing(
    linux: Path,
) -> None:
    icon = linux / "data" / "icons" / "hicolor" / "scalable" / "apps" / f"{APP_ID}.svg"
    icon.parent.mkdir(parents=True)
    icon.write_text("mine")
    with pytest.raises(OSError, match=re.escape(str(icon))):
        install_launcher()
    assert icon.read_text() == "mine"
    assert sorted(p.name for p in (linux / "data").iterdir()) == ["icons"]
    assert not launcher.manifest_path().exists()


def test_macos_install_is_undone_by_uninstall(
    places: Path, macos: list[list[str]]
) -> None:
    written = install_launcher()
    app = places / "home" / "Applications" / f"{APP_NAME}.app"
    assert written == [app]
    assert (app / "Contents" / "Resources" / f"{APP_NAME}.icns").exists()
    script = app / "Contents" / "MacOS" / APP_NAME
    if sys.platform != "win32":  # Windows keeps no executable bit
        assert script.stat().st_mode & 0o111
    _uninstall_is_clean(places, written)
    assert not (places / "home" / "Applications").exists()
    assert macos == [["iconutil", "-c"], ["lsregister", "-f"], ["lsregister", "-u"]]


@pytest.mark.parametrize(
    ("bundle_id", "accepted"), [("org.other.app", False), (APP_ID, True)]
)
def test_macos_install_replaces_an_app_only_if_it_has_the_viewers_id(
    places: Path, macos: list[list[str]], bundle_id: str, accepted: bool
) -> None:
    app = places / "home" / "Applications" / f"{APP_NAME}.app"
    plist = app / "Contents" / "Info.plist"
    plist.parent.mkdir(parents=True)
    old = plistlib.dumps({"CFBundleIdentifier": bundle_id})
    plist.write_bytes(old)
    if accepted:
        assert install_launcher() == [app]
        assert plistlib.loads(plist.read_bytes())["CFBundleName"] == APP_NAME
    else:
        with pytest.raises(OSError, match=re.escape(str(app))):
            install_launcher()
        assert plist.read_bytes() == old
        assert sorted(p.name for p in app.rglob("*")) == ["Contents", "Info.plist"]
        assert not launcher.manifest_path().exists()


@pytest.mark.parametrize(
    "manifest", ["{not json", '{"platform": "linux", "paths": []}'], ids=["json", "key"]
)
@pytest.mark.parametrize("action", [uninstall_launcher, install_launcher])
def test_a_broken_manifest_is_reported_by_name(
    linux: Path, manifest: str, action: Callable[[], list[Path]]
) -> None:
    path = launcher.manifest_path()
    path.parent.mkdir(parents=True)
    path.write_text(manifest)
    with pytest.raises(OSError, match=MANIFEST_FILE):
        action()
    assert not (linux / "data").exists()


def test_linux_files_known_by_their_mark_are_replaced_and_removed_without_a_manifest(
    linux: Path,
) -> None:
    first = install_launcher()
    launcher.manifest_path().unlink()
    assert install_launcher() == first
    launcher.manifest_path().unlink()
    assert set(first) <= set(uninstall_launcher())
    assert not any(p.exists() for p in first)


def test_macos_app_is_removed_by_its_id_once_the_settings_folder_is_gone(
    places: Path, macos: list[list[str]], caplog: pytest.LogCaptureFixture
) -> None:
    [app] = install_launcher()
    shutil.rmtree(places / "config")
    with caplog.at_level(logging.WARNING, logger="larmor"):
        assert uninstall_launcher() == [app]
    assert not app.exists()
    assert "No launcher manifest" in caplog.text


def test_linux_reinstalls_then_uninstall_leave_nothing_behind(linux: Path) -> None:
    for _ in range(3):
        install_launcher()
    uninstall_launcher()
    assert not (linux / "data").exists()
    assert not (linux / "config").exists()


def test_macos_app_taken_over_since_install_is_refused_and_left(
    places: Path, macos: list[list[str]], caplog: pytest.LogCaptureFixture
) -> None:
    [app] = install_launcher()
    plist = app / "Contents" / "Info.plist"
    theirs = plistlib.dumps({"CFBundleIdentifier": "org.other.app"})
    plist.write_bytes(theirs)
    with pytest.raises(OSError, match=re.escape(str(app))):
        install_launcher()
    with caplog.at_level(logging.WARNING, logger="larmor"):
        assert app not in uninstall_launcher()
    assert str(app) in caplog.text
    assert plist.read_bytes() == theirs
    assert macos == [["iconutil", "-c"], ["lsregister", "-f"]]  # the first install


def test_linux_entry_taken_over_since_install_is_refused_and_left(
    linux: Path, caplog: pytest.LogCaptureFixture
) -> None:
    desktop, icon, _ = install_launcher()
    theirs = "[Desktop Entry]\nName=Mine\n"
    desktop.write_text(theirs)
    with pytest.raises(OSError, match=re.escape(str(desktop))):
        install_launcher()
    with caplog.at_level(logging.WARNING, logger="larmor"):
        removed = uninstall_launcher()
    assert icon in removed
    assert desktop not in removed
    assert str(desktop) in caplog.text
    assert desktop.read_text() == theirs


def test_windows_reinstall_keeps_its_own_icon_after_the_icon_changes(
    places: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_icon = pytest.importorskip("larmor.viewer.qt.app_icon")

    def powershell(
        command: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        # Writes the shortcut the script names, holding what Windows stores in one.
        script = base64.b64decode(command[-1]).decode("utf-16-le")
        match = re.search(r"CreateShortcut\('([^']*)'\)", script)
        assert match is not None
        link = Path(match[1])
        link.write_bytes(
            "".join((launcher.DESCRIPTION, "larmor.main")).encode("utf-16-le")
        )
        return subprocess.CompletedProcess(list(command), 0, "", "")

    monkeypatch.setattr(launcher, "this_platform", lambda: Platform.WINDOWS)
    monkeypatch.setattr("larmor.viewer.launcher.subprocess.run", powershell)
    monkeypatch.setattr(app_icon, "render_icon", lambda size, image_format: b"old")
    icon, link = install_launcher()
    manifest = json.loads(launcher.manifest_path().read_text())
    assert manifest["paths"] == [str(icon), str(link)]
    monkeypatch.setattr(app_icon, "render_icon", lambda size, image_format: b"new")
    assert install_launcher() == [icon, link]
    assert icon.read_bytes() == b"new"
    assert set(uninstall_launcher()) >= {icon, link}
    assert not icon.exists()
    assert not link.exists()
