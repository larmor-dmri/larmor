import importlib.machinery
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

from larmor.viewer import ViewerUnavailableError, system
from larmor.viewer.system import (
    INSTALL_HINT,
    X11_LIBRARIES,
    check_viewer_can_start,
    missing_modules,
    missing_x11_libraries,
)

REAL_FIND_SPEC = importlib.util.find_spec


def _without(module: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def find_spec(name: str) -> importlib.machinery.ModuleSpec | None:
        return None if name == module else REAL_FIND_SPEC(name)

    monkeypatch.setattr("larmor.viewer.system.importlib.util.find_spec", find_spec)


def _modules_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    # The X checks come after the module check; they must run without the extra.
    monkeypatch.setattr(system, "missing_modules", lambda: [])


def _linux_x11(monkeypatch: pytest.MonkeyPatch) -> None:
    _modules_installed(monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    # Without the plugin, the libraries of the table are loaded one by one (the case
    # these tests cover); with it, ldd decides (tested below).
    monkeypatch.setattr(system, "_qt_xcb_plugin", lambda: None)


def _cdll_failing_for(names: set[str], monkeypatch: pytest.MonkeyPatch) -> None:
    def cdll(name: str) -> object:
        if name in names:
            raise OSError(f"{name}: cannot open shared object file")
        return object()

    monkeypatch.setattr("larmor.viewer.system.ctypes.CDLL", cdll)


@pytest.mark.parametrize(
    ("module", "package"),
    [("PySide6", "PySide6"), ("PySide6QtAds", "PySide6-QtAds"), ("pygfx", "pygfx")],
)
def test_missing_module_is_named_with_the_install_hint(
    module: str, package: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _without(module, monkeypatch)
    assert missing_modules() == [package]
    with pytest.raises(ViewerUnavailableError) as error:
        check_viewer_can_start()
    assert package in str(error.value)
    assert INSTALL_HINT in str(error.value)


def test_no_display_on_linux_suggests_ssh_x(monkeypatch: pytest.MonkeyPatch) -> None:
    _linux_x11(monkeypatch)
    monkeypatch.delenv("DISPLAY")
    with pytest.raises(ViewerUnavailableError, match="ssh -X"):
        check_viewer_can_start()


@pytest.mark.parametrize(
    "missing",
    [{"libxcb-cursor.so.0"}, {"libxkbcommon-x11.so.0", "libxcb-icccm.so.4"}],
)
def test_missing_x_libraries_are_named_with_their_packages(
    missing: set[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _linux_x11(monkeypatch)
    _cdll_failing_for(missing, monkeypatch)
    with pytest.raises(ViewerUnavailableError) as error:
        check_viewer_can_start()
    message = str(error.value)
    named = message.splitlines()[0]
    apt = re.search(r"apt install ([^\n]*)", message)
    dnf = re.search(r"dnf install ([^)]*)\)", message)
    assert apt is not None and dnf is not None
    for library in missing:
        debian, fedora = X11_LIBRARIES[library]
        assert library in named
        assert debian in apt[1].split()
        assert fedora in dnf[1].split()
    for library in X11_LIBRARIES.keys() - missing:
        assert library not in named


def test_x_check_passes_when_all_libraries_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _linux_x11(monkeypatch)
    _cdll_failing_for(set(), monkeypatch)
    check_viewer_can_start()


@pytest.mark.parametrize(
    ("platform", "env"),
    [
        ("darwin", {}),
        ("win32", {}),
        ("linux", {"QT_QPA_PLATFORM": "offscreen"}),
        ("linux", {"WAYLAND_DISPLAY": "wayland-0"}),
    ],
)
def test_x_libraries_are_not_checked_without_x11(
    platform: str, env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _linux_x11(monkeypatch)
    monkeypatch.setattr(sys, "platform", platform)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    _cdll_failing_for(set(X11_LIBRARIES), monkeypatch)
    check_viewer_can_start()


# ldd


def test_ldd_decides_what_is_missing_when_the_plugin_is_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _linux_x11(monkeypatch)
    plugin = tmp_path / "libqxcb.so"
    monkeypatch.setattr(system, "_qt_xcb_plugin", lambda: plugin)
    monkeypatch.setattr(
        system, "_ldd_not_found", lambda path: ["libX11-xcb.so.1", "libfoo.so.2"]
    )
    _cdll_failing_for(set(X11_LIBRARIES), monkeypatch)  # ignored when ldd answers
    assert missing_x11_libraries() == ["libX11-xcb.so.1", "libfoo.so.2"]
    with pytest.raises(ViewerUnavailableError) as error:
        check_viewer_can_start()
    message = str(error.value)
    assert "sudo apt install libx11-xcb1\n" in message
    assert "sudo dnf install libX11-xcb)" in message
    assert "provide libfoo.so.2" in message


def test_the_table_is_used_when_ldd_cannot_tell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _linux_x11(monkeypatch)
    monkeypatch.setattr(system, "_qt_xcb_plugin", lambda: tmp_path / "libqxcb.so")
    monkeypatch.setattr(system, "_ldd_not_found", lambda path: None)
    _cdll_failing_for({"libxcb-cursor.so.0"}, monkeypatch)
    assert missing_x11_libraries() == ["libxcb-cursor.so.0"]


def test_ldd_output_is_parsed_for_libraries_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = (
        "\tlinux-vdso.so.1 (0x00007ffd)\n"
        "\tlibQt6XcbQpa.so.6 => /x/Qt/lib/libQt6XcbQpa.so.6 (0x00007f)\n"
        "\tlibxcb-cursor.so.0 => not found\n"
        "\tlibX11-xcb.so.1 => not found\n"
    )

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["ldd"], 0, output, "")

    monkeypatch.setattr(
        "larmor.viewer.system.shutil.which", lambda name: "/usr/bin/ldd"
    )
    monkeypatch.setattr("larmor.viewer.system.subprocess.run", run)
    assert system._ldd_not_found(tmp_path / "libqxcb.so") == [
        "libxcb-cursor.so.0",
        "libX11-xcb.so.1",
    ]


def test_ldd_cannot_tell_without_ldd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("larmor.viewer.system.shutil.which", lambda name: None)
    assert system._ldd_not_found(tmp_path / "libqxcb.so") is None


def test_ldd_cannot_tell_when_it_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "not a dynamic executable", or a different ldd: no "not found" lines, but that
    # says nothing about the libraries, so the table must be used instead.
    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["ldd"], 1, "", "not a dynamic executable")

    monkeypatch.setattr(
        "larmor.viewer.system.shutil.which", lambda name: "/usr/bin/ldd"
    )
    monkeypatch.setattr("larmor.viewer.system.subprocess.run", run)
    assert system._ldd_not_found(tmp_path / "libqxcb.so") is None
