import importlib.util
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.io import loadmat

from larmor.sd import AnisotropicRF, IsotropicRF, ResponseFunction
from larmor.sd.kernel import Kernel
from larmor.sd.sphere import DIRS_300, get_dirs

DATA_DIR = Path(__file__).parent / "data"
VIEWER_TESTS = Path(__file__).parent / "viewer"

# Set before Qt is first imported, so the viewer tests never need a display. Only
# the Qt tests are skipped without the viewer extra; its checks need no Qt.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
collect_ignore: list[str] = []
if any(
    importlib.util.find_spec(m) is None for m in ("PySide6", "PySide6QtAds", "pygfx")
):
    collect_ignore = [
        "viewer/test_window.py",
        "viewer/test_layers.py",
        "viewer/test_options_panel.py",
        "viewer/test_render.py",
        "viewer/test_colour_bars.py",
        "viewer/test_opening.py",
        "viewer/test_annotations.py",
        "viewer/test_keys.py",
        "viewer/test_view_input.py",
        "viewer/test_settings_panel.py",
        "viewer/test_redraws.py",
        "viewer/test_style.py",
        "viewer/test_fod_adapter.py",
        "viewer/test_peaks_adapter.py",
        "viewer/test_surface_adapter.py",
        "viewer/test_screenshot_panel.py",
        "viewer/test_dti_render.py",
        "viewer/test_help_panel.py",
        "viewer/test_fod_inspector.py",
        "viewer/test_inspect_fod.py",
        "viewer/test_labels_3d.py",
        "viewer/test_labels_panel.py",
        "viewer/test_mcp_server.py",
        "viewer/test_session_window.py",
        "viewer/test_docks.py",
        "viewer/test_matlab_dialog.py",
        "viewer/test_registration_dialog.py",
        "viewer/test_save_and_jobs.py",
        "viewer/test_phantom_panel.py",
        "viewer/test_phantom_steps.py",
        "viewer/test_breakdown_panel.py",
    ]
# Screenshots need pygfx but no Qt.
if importlib.util.find_spec("pygfx") is None:
    collect_ignore += [
        "viewer/test_screenshot.py",
        "viewer/test_worker.py",
        "viewer/test_view_labels.py",
    ]
if importlib.util.find_spec("mcp") is None:
    collect_ignore += ["viewer/test_mcp_server.py"]


# The viewer's tests that build Qt widgets: the main window, its panels and dialogs.
# Each such test takes seconds with a software renderer, so CI on Linux and Windows,
# which have no GPU, leaves them out with -m "not qt_widgets"; macOS runs them all.
QT_WIDGET_TESTS = {
    "test_breakdown_panel.py",
    "test_docks.py",
    "test_help_panel.py",
    "test_inspect_fod.py",
    "test_keys.py",
    "test_labels_panel.py",
    "test_layers.py",
    "test_matlab_dialog.py",
    "test_opening.py",
    "test_options_panel.py",
    "test_phantom_panel.py",
    "test_phantom_steps.py",
    "test_redraws.py",
    "test_regions_editor.py",
    "test_registration_dialog.py",
    "test_save_and_jobs.py",
    "test_screenshot_panel.py",
    "test_session_window.py",
    "test_settings_panel.py",
    "test_view_input.py",
    "test_window.py",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.path.parent == VIEWER_TESTS and item.path.name in QT_WIDGET_TESTS:
            item.add_marker(pytest.mark.qt_widgets)


@pytest.fixture(autouse=True, scope="session")
def _scratch_viewer_settings(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    # Never the user's own settings or MCP token: tests get scratch ones.
    from larmor.viewer import mcp_server
    from larmor.viewer.model import settings

    folder = tmp_path_factory.mktemp("config")
    path = folder / settings.SETTINGS_FILE
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(settings, "settings_path", lambda: path)
        patch.setattr(mcp_server, "token_path", lambda: folder / mcp_server.TOKEN_FILE)
        yield


@pytest.fixture(autouse=True)
def _scratch_home(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The launcher writes and deletes in the home, data and config folders; no test
    # may reach the user's own. A folder apart from tmp_path, which tests list.
    home = tmp_path_factory.mktemp("home")
    for name in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("APPDATA", str(home / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    monkeypatch.delenv("LARMOR_DATA_DIR", raising=False)


@pytest.fixture(autouse=True)
def _no_launcher_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    # iconutil, lsregister, PowerShell and update-*-database change the real system
    # (lsregister writes the LaunchServices database even for an app in a temp home).
    # A test that wants them sets launcher.subprocess.run to a stub of its own.
    from larmor.viewer import launcher

    def refuse(command: object, *args: object, **kwargs: object) -> None:
        raise RuntimeError(f"A test ran the launcher's system tool {command!r}.")

    tools = SimpleNamespace(run=refuse, list2cmdline=subprocess.list2cmdline)
    monkeypatch.setattr(launcher, "subprocess", tools)


@pytest.fixture(scope="session")
def freesurfer_subjects() -> Path:
    """FreeSurfer's sample subjects, from FREESURFER_HOME; skips the test without them.

    The committed surfaces in tests/data/surfaces cover the formats; these tests
    also check the full-size fsaverage files, on a machine with FreeSurfer.
    """
    home = os.environ.get("FREESURFER_HOME")
    subjects = Path(home) / "subjects" if home else None
    if subjects is None or not (subjects / "fsaverage5").is_dir():
        pytest.skip("needs FreeSurfer's fsaverage: set FREESURFER_HOME to run it")
    return subjects


@pytest.fixture(scope="session")
def external_data() -> Path:
    """The folder of large datasets in LARMOR_TEST_DATA; skips the test without it."""
    folder = os.environ.get("LARMOR_TEST_DATA")
    if not folder or not Path(folder).is_dir():
        pytest.skip("needs a dataset that is not committed: set LARMOR_TEST_DATA")
    return Path(folder)


@pytest.fixture(autouse=True)
def _restored_viewer_style(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[None]:
    # The style is application-wide: each viewer test starts unstyled, leaves the
    # app as it found it and writes its theme files to a scratch folder, never the
    # user's cache. Other tests skip this, so they never import Qt.
    style = sys.modules.get("larmor.viewer.qt.style")
    if style is None or VIEWER_TESTS not in request.node.path.parents:
        yield
        return
    # Qt's classes come through the style module: Qt is only imported in viewer/qt.
    q_palette = style.QPalette
    qapp = request.getfixturevalue("qapp")
    cache = tmp_path_factory.mktemp("style")
    sheet, palette, name = (
        qapp.styleSheet(),
        q_palette(qapp.palette()),
        qapp.style().name(),
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(style, "_cache", lambda: cache)
        for state, value in (
            ("_requested", None),
            ("_applied", None),
            ("_icons", None),
            ("_warned_cache", False),
        ):
            patch.setattr(style, state, value)
        yield
    qapp.setStyleSheet(sheet)
    qapp.setPalette(palette)
    qapp.setStyle(name)


# Plain numpy references of compiled code sit next to the tests that use them
# (tests/sd: reference_solvers.py, reference_peaks.py; tests/tractography:
# reference_random.py, reference_tracking.py; tests/dwi: reference_fitting.py;
# tests/viewer: viewer_helpers.py).
# importlib import mode puts no test folder on sys.path, so they are added here;
# appended, so they shadow nothing.
sys.path.append(str(Path(__file__).parent / "sd"))
sys.path.append(str(Path(__file__).parent / "tractography"))
sys.path.append(str(Path(__file__).parent / "dwi"))
sys.path.append(str(Path(__file__).parent / "viewer"))
sys.path.append(str(Path(__file__).parent / "cli"))

# Fixtures shared by the viewer and the command tests. Plugins, not conftest.py
# files: mypy takes two of them in one tests folder for modules of the same name.
pytest_plugins = ["viewer_fixtures", "cli_fixtures"]

# MATLAB results of tests/matlab_scripts/run_grl.m on a 6x6x6 block, see
# tests/data/README.md. loadmat output has no fixed structure, so the fixtures
# return dict[str, Any].


@pytest.fixture(scope="session")
def grl() -> dict[str, Any]:
    """MATLAB's inputs and kernels: gradients, image block, mask, kernels and nu."""
    grl_obj: dict[str, Any] = loadmat(DATA_DIR / "grl_obj.mat", simplify_cells=True)[
        "grl_obj"
    ]
    return grl_obj


@pytest.fixture(scope="session")
def deconv() -> dict[str, Any]:
    """MATLAB's deconvolution results on the block: FODs and fractions."""
    deconv_obj: dict[str, Any] = loadmat(
        DATA_DIR / "deconv_obj.mat", simplify_cells=True
    )["deconv_obj"]
    return deconv_obj


@pytest.fixture(scope="session")
def leemans_tournier_ref() -> dict[str, Any]:
    """MRIToolkit's FOD of the block in its own basis and in its "mrtrix" basis.

    Made by tests/matlab_scripts/make_leemans_tournier_ref.m, see tests/data/README.md.
    """
    ref: dict[str, Any] = loadmat(
        DATA_DIR / "leemans_tournier_ref.mat", simplify_cells=True
    )["ref"]
    return ref


def _rfs() -> list[ResponseFunction]:
    """The response functions run_grl.m uses."""
    return [
        AnisotropicRF(eigen_vals=[1.7e-3, 0.2e-3, 0.2e-3], mean_kurtosis=0.0),
        IsotropicRF(diffusivity=0.7e-3),
        IsotropicRF(diffusivity=3e-3),
    ]


@pytest.fixture(scope="session")
def kernel_hr(grl: dict[str, Any]) -> Kernel:
    return Kernel(
        _rfs(),
        bvals=grl["bvals"].astype(np.float64),
        bvecs=grl["bvecs"].astype(np.float64),
        dirs=DIRS_300,
        weights=grl["shell_weight"],
    )


@pytest.fixture(scope="session")
def kernel_lr(grl: dict[str, Any]) -> Kernel:
    bvals = grl["bvals"].astype(np.float64)
    return Kernel(
        _rfs(),
        bvals=bvals,
        bvecs=grl["bvecs"].astype(np.float64),
        dirs=get_dirs(min(len(bvals), 45)),
        weights=grl["shell_weight"],
    )


@pytest.fixture(scope="session")
def wm_rf_ref() -> dict[str, Any]:
    """MRIToolkit's white matter response estimate on the block in tests/data/dki.

    Made by tests/matlab_scripts/make_wm_rf_ref.m, see tests/data/README.md.
    """
    ref: dict[str, Any] = loadmat(
        DATA_DIR / "wm_rf" / "wm_rf_ref.mat", simplify_cells=True
    )["ref"]
    return ref
