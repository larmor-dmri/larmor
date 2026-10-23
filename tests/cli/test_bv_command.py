import importlib.util
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

import larmor.viewer
from larmor.main import app
from larmor.viewer import ViewerUnavailableError, system
from larmor.viewer.loading import DEFAULT

MESSAGE = "The viewer cannot start here; install the missing packages."


def _run(*args: str) -> Result:
    return CliRunner().invoke(app, list(args), env={"TERM": "dumb"})


def test_help_lists_bv() -> None:
    result = _run("--help")
    assert result.exit_code == 0
    assert "bv" in result.output.split()


def test_bv_reports_why_the_viewer_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse() -> None:
        raise ViewerUnavailableError(MESSAGE)

    monkeypatch.setattr(system, "check_viewer_can_start", refuse)
    result = _run("bv")
    assert result.exit_code == 1
    assert f"ERROR: {MESSAGE}" in result.output
    assert "Traceback" not in result.output


NEEDS_QT = pytest.mark.skipif(
    any(
        importlib.util.find_spec(m) is None
        for m in ("PySide6", "PySide6QtAds", "pygfx")
    ),
    reason="needs the viewer extra",
)


@NEEDS_QT
def test_bv_runs_the_viewer_once_the_checks_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from larmor.viewer import app as viewer_app

    calls: list[str] = []

    def show(*args: object, **kwargs: object) -> None:
        calls.append("show")

    monkeypatch.setattr(system, "check_viewer_can_start", lambda: calls.append("check"))
    monkeypatch.setattr(viewer_app, "show", show)
    result = _run("bv")
    assert result.exit_code == 0, result.output
    assert calls == ["check", "show"]


@pytest.fixture
def view_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def view(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(larmor.viewer, "view", view)
    return calls


def test_bv_without_options_opens_an_empty_viewer(
    view_calls: list[dict[str, object]],
) -> None:
    result = _run("bv")
    assert result.exit_code == 0, result.output
    assert view_calls == [
        {
            "volumes": None,
            "overlays": None,
            "tracts": None,
            "fods": None,
            "peaks": None,
            "surfaces": None,
            "max_streamlines": DEFAULT,
            "session": None,
            "phantom": None,
            "simulation": None,
        }
    ]


@pytest.mark.parametrize(
    ("flag", "keyword"),
    [
        ("--volumes", "volumes"),
        ("-v", "volumes"),
        ("--overlays", "overlays"),
        ("-o", "overlays"),
        ("--tracts", "tracts"),
        ("--tracts", "tracts"),
        ("--fods", "fods"),
        ("-f", "fods"),
        ("--peaks", "peaks"),
        ("-p", "peaks"),
        ("--surfaces", "surfaces"),
    ],
)
def test_each_category_option_gives_its_files_to_view(
    flag: str, keyword: str, view_calls: list[dict[str, object]]
) -> None:
    result = _run("bv", flag, "a.nii.gz", flag, "b.nii.gz")
    assert result.exit_code == 0, result.output
    [call] = view_calls
    assert call[keyword] == [Path("a.nii.gz"), Path("b.nii.gz")]
    others = {"volumes", "overlays", "tracts", "fods", "peaks", "surfaces"} - {keyword}
    assert all(call[k] is None for k in others)


def test_mixed_options_keep_their_categories(
    view_calls: list[dict[str, object]],
) -> None:
    result = _run(
        "bv", "--tracts", "cst.tck", "-v", "t1.nii", "-o", "fa.nii", "-v", "b0.nii"
    )
    assert result.exit_code == 0, result.output
    [call] = view_calls
    assert call["volumes"] == [Path("t1.nii"), Path("b0.nii")]
    assert call["overlays"] == [Path("fa.nii")]
    assert call["tracts"] == [Path("cst.tck")]


@pytest.mark.parametrize("limit", ["0", "5"])
def test_max_streamlines_is_passed_to_view(
    limit: str, view_calls: list[dict[str, object]]
) -> None:
    result = _run("bv", "--max-streamlines", limit)
    assert result.exit_code == 0, result.output
    assert view_calls[0]["max_streamlines"] == int(limit)


@pytest.mark.parametrize(
    ("flag", "key", "value"),
    [
        ("--phantom", "phantom", "scene.phantom.json"),
        ("--simulation", "simulation", "sim"),
    ],
)
def test_phantom_and_simulation_are_passed_to_view(
    flag: str,
    key: str,
    value: str,
    view_calls: list[dict[str, object]],
) -> None:
    result = _run("bv", flag, value)
    assert result.exit_code == 0, result.output
    assert view_calls[0][key] == Path(value)


@pytest.fixture
def shown(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    # The real view() runs its checks; only the window is left out.
    calls: list[object] = []
    monkeypatch.setattr(system, "check_viewer_can_start", lambda: None)
    monkeypatch.setattr(
        "larmor.viewer.app.show", lambda *args, **kwargs: calls.append(kwargs)
    )
    return calls


def _error_line(result: Result) -> str:
    assert result.exit_code == 1
    lines: list[str] = [x for x in result.output.splitlines() if x.strip()]
    [line] = lines
    assert line.startswith("ERROR:")
    return line


def test_a_missing_phantom_file_is_one_error_line(
    shown: list[object], tmp_path: Path
) -> None:
    result = _run("bv", "--phantom", str(tmp_path / "missing.phantom.json"))
    assert "missing.phantom.json" in _error_line(result)
    assert shown == []


def test_a_folder_that_is_not_a_simulation_is_one_error_line(
    shown: list[object], tmp_path: Path
) -> None:
    (tmp_path / "simulation.json").write_text('{"format": "something else"}')
    result = _run("bv", "--simulation", str(tmp_path))
    assert "simulation.json" in _error_line(result)
    assert shown == []


def test_a_phantom_and_a_simulation_together_are_refused(
    shown: list[object], tmp_path: Path
) -> None:
    (tmp_path / "scene.phantom.json").write_text("{}")
    result = _run(
        "bv",
        "--phantom",
        str(tmp_path / "scene.phantom.json"),
        "--simulation",
        str(tmp_path),
    )
    assert "not both" in _error_line(result)
    assert shown == []


@pytest.mark.parametrize("flag", ["--session", "-s"])
def test_session_is_passed_to_view(
    flag: str, view_calls: list[dict[str, object]]
) -> None:
    result = _run("bv", flag, "s.lar.json", "--tracts", "cst.tck")
    assert result.exit_code == 0, result.output
    [call] = view_calls
    assert call["session"] == Path("s.lar.json")
    assert call["tracts"] == [Path("cst.tck")]


def test_bv_without_max_streamlines_leaves_the_limit_to_the_settings(
    view_calls: list[dict[str, object]],
) -> None:
    result = _run("bv", "--tracts", "cst.tck")
    assert result.exit_code == 0, result.output
    assert view_calls[0]["max_streamlines"] is DEFAULT


def test_negative_max_streamlines_is_refused_before_view(
    view_calls: list[dict[str, object]],
) -> None:
    result = _run("bv", "--max-streamlines", "-1")
    assert result.exit_code != 0
    assert "--max-streamlines" in result.output
    assert view_calls == []


def test_checks_run_before_files_are_looked_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse() -> None:
        raise ViewerUnavailableError(MESSAGE)

    monkeypatch.setattr(system, "check_viewer_can_start", refuse)
    result = _run("bv", "-v", "missing.nii.gz", "-o", "x.png")
    assert result.exit_code == 1
    assert f"ERROR: {MESSAGE}" in result.output


@pytest.mark.parametrize(
    ("args", "port"),
    [(["--mcp"], None), (["--mcp-port", "9000"], 9000)],
    ids=["mcp", "mcp-port"],
)
def test_mcp_options_ask_view_to_serve_agents(
    args: list[str], port: int | None, view_calls: list[dict[str, object]]
) -> None:
    result = _run("bv", *args)
    assert result.exit_code == 0, result.output
    assert view_calls[0]["mcp"] is True
    assert view_calls[0]["mcp_port"] == port


def test_files_given_by_name_open_in_their_categories(
    view_calls: list[dict[str, object]],
) -> None:
    result = _run("bv", "cst.tck", "t1.nii", "s.lar.json", "af.trk", "-v", "b0.nii")
    assert result.exit_code == 0, result.output
    [call] = view_calls
    assert call["volumes"] == [Path("t1.nii"), Path("b0.nii")]
    assert call["tracts"] == [Path("cst.tck"), Path("af.trk")]
    assert call["session"] == Path("s.lar.json")


@pytest.mark.parametrize(
    "args",
    [["a.lar.json", "--session", "b.lar.json"], ["a.lar.json", "b.lar.json"]],
    ids=["file-and-option", "two-files"],
)
def test_two_sessions_are_refused(
    args: list[str], view_calls: list[dict[str, object]]
) -> None:
    result = _run("bv", *args)
    assert result.exit_code != 0
    assert "one session" in result.output
    assert view_calls == []


def test_uninstall_launcher_without_one_says_so_and_opens_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    view_calls: list[dict[str, object]],
) -> None:
    from larmor.viewer import launcher

    monkeypatch.setattr(launcher, "config_folder", lambda: tmp_path)
    for name in ("HOME", "XDG_DATA_HOME", "APPDATA"):  # never the user's own folders
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    result = _run("bv", "--uninstall-launcher")
    assert result.exit_code == 0, result.output
    assert "No launcher was installed." in result.output
    assert view_calls == []


def test_a_broken_launcher_manifest_is_one_error_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from larmor.viewer import launcher

    monkeypatch.setattr(launcher, "config_folder", lambda: tmp_path)
    for name in ("HOME", "XDG_DATA_HOME", "APPDATA"):  # never the user's own folders
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    (tmp_path / launcher.MANIFEST_FILE).write_text("{not json")
    result = _run("bv", "--uninstall-launcher")
    assert result.exit_code == 1
    [line] = [x for x in result.output.splitlines() if x.strip()]
    assert line.startswith("ERROR:")
    assert launcher.MANIFEST_FILE in result.output


@pytest.mark.parametrize(
    "args",
    [["--install-launcher", "t1.nii"], ["--uninstall-launcher", "--tracts", "cst.tck"]],
)
def test_the_launcher_options_take_no_files(
    args: list[str], view_calls: list[dict[str, object]]
) -> None:
    result = _run("bv", *args)
    assert result.exit_code == 2
    assert view_calls == []
