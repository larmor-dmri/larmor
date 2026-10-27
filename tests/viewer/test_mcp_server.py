import asyncio
import dataclasses
import http.client
import inspect
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx2
import nibabel as nib
import numpy as np
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, ImageContent, TextContent
from viewer_helpers import free_port

from larmor.io import Progress, save_surface
from larmor.io.formats import write_freesurfer_values
from larmor.types import Surface, Tractogram
from larmor.viewer import (
    CameraPreset,
    Category,
    Colormap,
    PortInUseError,
    Viewer,
    ViewKind,
)
from larmor.viewer.loading import Loader
from larmor.viewer.mcp_server import McpServer, actions, new_token, to_json
from larmor.viewer.model import registry
from larmor.viewer.model.options import SurfaceOptions, TractogramOptions, VolumeOptions
from larmor.viewer.model.viewer import AGENT_ACTION
from larmor.viewer.qt.help_panel import GUIDES

TIMEOUT = 10.0  # seconds; a local call answers in milliseconds


class MainThread:
    """Runs posted calls on the test's event loop, as the window's main thread."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.posted = 0

    def post(self, function: Callable[[], None]) -> None:
        self.posted += 1
        assert self.loop is not None
        self.loop.call_soon_threadsafe(function)


@pytest.fixture(scope="module")
def main() -> MainThread:
    return MainThread()


@pytest.fixture(scope="module")
def server(
    main: MainThread, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[McpServer]:
    token_file = tmp_path_factory.mktemp("mcp") / "token"
    server = McpServer(Viewer(), main.post, token_file=token_file)
    server.start(free_port())
    yield server
    server.stop()


@asynccontextmanager
async def connect(server: McpServer, main: MainThread) -> AsyncIterator[Client]:
    main.loop = asyncio.get_running_loop()
    headers = {"Authorization": f"Bearer {server.token}"}
    async with (
        httpx2.AsyncClient(headers=headers, timeout=TIMEOUT) as http,
        Client(streamable_http_client(server.url(), http_client=http)) as client,
    ):
        yield client


def status(server: McpServer, headers: dict[str, str]) -> int:
    # An initialize request: the endpoint answers 200 once the guard lets it through.
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        }
    )
    assert server.port is not None
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=TIMEOUT)
    try:
        connection.request(
            "POST",
            "/mcp",
            body,
            {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                **headers,
            },
        )
        return connection.getresponse().status
    finally:
        connection.close()


# Security


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("no token", 401),
        ("wrong token", 401),
        ("foreign host", 403),
        ("foreign origin", 403),
        ("localhost and own origin", 200),
    ],
)
def test_requests_need_the_token_and_this_machine_as_host_and_origin(
    case: str, expected: int, server: McpServer
) -> None:
    own = f"127.0.0.1:{server.port}"
    token = {"Authorization": f"Bearer {server.token}"}
    headers = {
        "no token": {"Host": own},
        "wrong token": {"Host": own, "Authorization": "Bearer wrong"},
        "foreign host": {"Host": f"evil.example:{server.port}", **token},
        "foreign origin": {"Host": own, "Origin": "http://evil.example", **token},
        "localhost and own origin": {
            "Host": f"localhost:{server.port}",
            "Origin": f"http://{own}",
            **token,
        },
    }[case]
    assert status(server, headers) == expected


def test_a_new_token_refuses_the_old_one_at_once(server: McpServer) -> None:
    own = {"Host": f"127.0.0.1:{server.port}"}
    old = server.token
    new = server.renew_token()
    assert new != old
    assert status(server, {**own, "Authorization": f"Bearer {old}"}) == 401
    assert status(server, {**own, "Authorization": f"Bearer {new}"}) == 200


def test_a_token_renewed_by_another_viewer_applies_at_once(server: McpServer) -> None:
    own = {"Host": f"127.0.0.1:{server.port}"}
    old = server.token
    new = new_token(server.token_file)  # as another viewer's New token writes it
    assert status(server, {**own, "Authorization": f"Bearer {old}"}) == 401
    assert status(server, {**own, "Authorization": f"Bearer {new}"}) == 200


def test_a_port_another_program_holds_on_every_address_is_refused(
    main: MainThread, tmp_path: Path
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as other:
        other.bind(("0.0.0.0", 0))
        other.listen(1)
        port = other.getsockname()[1]
        server = McpServer(Viewer(), main.post, token_file=tmp_path / "token")
        with pytest.raises(PortInUseError, match=str(port)):
            server.start(port)
        assert not server.running


def test_the_server_starts_again_on_its_port_after_serving_a_request(
    main: MainThread, tmp_path: Path
) -> None:
    server = McpServer(Viewer(), main.post, token_file=tmp_path / "token")
    port = free_port()
    server.start(port)
    # A client keeping its connection open: stop closes it, and the port's side
    # of it waits in TIME_WAIT.
    client = http.client.HTTPConnection("127.0.0.1", port, timeout=TIMEOUT)
    try:
        headers = {"Authorization": f"Bearer {server.token}"}
        client.request("GET", "/mcp", headers=headers)
        client.getresponse().read()
        server.stop()
        server.start(port)
        assert server.port == port
        assert status(server, headers) == 200
    finally:
        client.close()
        server.stop()


@pytest.mark.parametrize("before", [None, ""], ids=["missing", "empty"])
def test_without_a_saved_token_every_request_is_refused_and_none_is_written(
    before: str | None, main: MainThread, tmp_path: Path
) -> None:
    path = tmp_path / "token"
    server = McpServer(Viewer(), main.post, token_file=path)
    server.start(free_port())
    try:
        path.unlink()
        if before is not None:
            path.write_text(before)
        own = {"Host": f"127.0.0.1:{server.port}"}
        assert status(server, {**own, "Authorization": "Bearer "}) == 401
        assert status(server, {**own, "Authorization": "Bearer x"}) == 401
        assert (path.read_text() if path.exists() else None) == before
    finally:
        server.stop()


def test_a_new_token_replaces_the_file_in_one_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "token"
    path.write_text("old\n")
    replaced: list[tuple[Path, Path, str]] = []
    real = os.replace

    def replace(source: str, target: Path) -> None:
        replaced.append((Path(source), Path(target), Path(source).read_text()))
        real(source, target)

    monkeypatch.setattr(os, "replace", replace)
    token = new_token(path)
    [(source, target, text)] = replaced
    assert (source.parent, target, text) == (tmp_path, path, token + "\n")
    assert list(tmp_path.iterdir()) == [path]


def test_starting_the_server_leaves_the_root_logger_alone(
    main: MainThread, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = logging.getLogger()
    # As in the viewer, which adds no handler of its own; pytest's would make the
    # mcp package's set-up a no-op.
    monkeypatch.setattr(root, "handlers", [])
    handlers, level = root.handlers[:], root.level
    server = McpServer(Viewer(), main.post, token_file=tmp_path / "token")
    server.start(free_port())
    try:
        assert root.handlers == handlers
        assert root.level == level
    finally:
        server.stop()


def access_entries(path: Path) -> list[str]:
    """The entries of path's Windows access list, as icacls prints them."""
    output = subprocess.run(
        ["icacls", str(path)], capture_output=True, text=True, check=True
    ).stdout
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    lines[0] = lines[0].removeprefix(str(path)).strip()
    return [line for line in lines if not line.startswith("Successfully processed")]


@pytest.mark.parametrize("before", [None, 0o644], ids=["new file", "readable file"])
def test_the_token_file_is_readable_by_the_user_only(
    before: int | None, tmp_path: Path
) -> None:
    path = tmp_path / "token"
    if before is not None:
        path.write_text("old\n")
        path.chmod(before)
    token = McpServer(Viewer(), lambda f: None, token_file=path).renew_token()
    assert path.read_text().strip() == token
    if sys.platform == "win32":
        user = os.environ["USERNAME"]
        domain = os.environ.get("USERDOMAIN")
        account = f"{domain}\\{user}" if domain else user
        # Windows also grants the system and the administrators, who can read any
        # file, and the file's owner, the user; no other user or group.
        allowed = {
            f"{account}:(F)".lower(),
            "nt authority\\system:(f)",
            "builtin\\administrators:(f)",
            "owner rights:(f)",
        }
        entries = {e.lower() for e in access_entries(path)}
        assert f"{account}:(F)".lower() in entries
        assert entries <= allowed
    else:
        assert path.stat().st_mode & 0o777 == 0o600


def test_on_windows_the_token_file_is_restricted_before_the_token_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "token"
    calls: list[tuple[list[str], str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, Path(command[1]).read_text()))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("SystemRoot", "C:\\Windows")
    monkeypatch.setenv("USERNAME", "ada")
    monkeypatch.setenv("USERDOMAIN", "LAB")
    token = new_token(path)
    [(command, text)] = calls
    icacls = str(Path("C:\\Windows", "System32", "icacls.exe"))
    temporary = Path(command[1])
    assert command == [
        icacls,
        str(temporary),
        "/inheritance:r",
        "/grant:r",
        "LAB\\ada:F",
    ]
    assert (temporary.parent, text) == (tmp_path, "")
    assert path.read_text() == token + "\n"


@pytest.mark.parametrize("failure", ["icacls fails", "no user"])
def test_on_windows_a_token_file_that_cannot_be_restricted_is_not_kept(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "token"

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 5, "", "Access is denied.")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(subprocess, "run", run)
    if failure == "no user":
        monkeypatch.delenv("USERNAME", raising=False)
    else:
        monkeypatch.setenv("USERNAME", "ada")
    with pytest.raises(PermissionError, match="Cannot restrict the token file"):
        new_token(path)
    assert list(tmp_path.iterdir()) == []


# Tools


def test_each_agent_action_is_a_tool_described_by_its_docstring(
    server: McpServer, main: MainThread
) -> None:
    async def listed() -> dict[str, Any]:
        async with connect(server, main) as client:
            return {t.name: t for t in (await client.list_tools()).tools}

    tools = asyncio.run(listed())
    marked = {
        name: method
        for name, method in inspect.getmembers(Viewer, inspect.isfunction)
        if getattr(method, AGENT_ACTION, False)
    }
    assert {"load", "set", "move_cursor", "set_camera", "screenshot"} <= set(marked)
    assert set(tools) == {*marked, "scene"}
    assert "add" not in tools
    for name, method in marked.items():
        doc = inspect.getdoc(method)
        assert doc is not None
        assert tools[name].description.startswith(doc.splitlines()[0])
        assert ">>>" not in tools[name].description
    schema = tools["set_camera"].input_schema
    preset = schema["properties"]["preset"]
    if "$ref" in preset:
        preset = schema["$defs"][preset["$ref"].rsplit("/", 1)[-1]]
    assert preset["enum"] == [p.value for p in CameraPreset]
    # A callback such as a video's progress cannot come from an agent: left out.
    rotation = tools["record_rotation"].input_schema["properties"]
    assert "progress" not in rotation
    assert {"path", "speed", "duration", "fps"} <= set(rotation)


def test_the_agents_guide_lists_every_tool_and_only_tools() -> None:
    guide = (Path(str(GUIDES)) / "12-agents.md").read_text(encoding="utf-8")
    section = guide.split("## What an agent can do")[1].split("\n## ")[0]
    listed = set(re.findall(r"`(\w+)`", section))
    assert listed == {*actions(Viewer()), "scene"}


def _volume_file(path: Path) -> Path:
    data = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    return path


def _call(
    server: McpServer, main: MainThread, name: str, arguments: dict[str, Any]
) -> CallToolResult:
    async def call() -> CallToolResult:
        async with connect(server, main) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(call())


def test_an_agent_loads_sets_moves_the_cursor_and_the_camera(
    server: McpServer, main: MainThread, tmp_path: Path
) -> None:
    path = _volume_file(tmp_path / "v.nii.gz")
    calls: list[tuple[str, dict[str, Any]]] = [
        ("load", {"path": str(path), "category": "volumes"}),
        ("set", {"target": "volumes", "name": "colormap", "value": "hot"}),
        ("move_cursor", {"position": [1.0, 2.0, 3.0]}),
        ("set_camera", {"view": "3d", "preset": "coronal"}),
    ]

    async def drive() -> list[CallToolResult]:
        async with connect(server, main) as client:
            return [await client.call_tool(n, a) for n, a in calls]

    viewer = server.viewer
    viewer.remove(Category.VOLUMES)
    posted = main.posted
    results = asyncio.run(drive())
    assert [r.is_error for r in results] == [False] * len(calls)
    assert main.posted - posted == len(calls)
    [item] = viewer.scene.in_category(Category.VOLUMES)
    assert item.source == path
    assert isinstance(item.options, VolumeOptions)
    assert item.options.colormap == Colormap.HOT
    assert viewer.scene.view.cursor == (1.0, 2.0, 3.0)
    assert viewer.scene.view.cameras.three_d.forward == (0.0, -1.0, 0.0)
    loaded = results[0].content[0]
    assert isinstance(loaded, TextContent)
    assert json.loads(loaded.text)["id"] == item.id


def test_an_agent_adds_a_region_by_its_role(
    server: McpServer, main: MainThread
) -> None:
    viewer = server.viewer
    tracts = viewer.add(
        Tractogram(np.zeros((2, 3)), np.array([0, 2])), Category.TRACTOGRAMS
    )
    arguments = {"radius": 4.0, "role": "exclude", "apply": True}
    result = _call(server, main, "add_sphere", arguments)
    assert not result.is_error
    options = viewer.scene.item(tracts.id).options
    assert isinstance(options, TractogramOptions)
    (region,) = viewer.scene.in_category(Category.REGIONS)
    assert options.regions == (region.id,)
    assert region.options.role == "exclude"


def test_an_agent_imports_a_matlab_file_by_preset_and_by_variable(
    server: McpServer, main: MainThread
) -> None:
    fit = Path(__file__).parents[1] / "data" / "mritoolkit" / "dti_fit.mat"
    viewer = server.viewer
    for item in viewer.scene.items:
        viewer.remove(item.id)
    preset = _call(
        server, main, "load_matlab", {"path": str(fit), "preset": "tensor_fit"}
    )
    manual = _call(
        server,
        main,
        "load_matlab",
        {"path": str(fit), "preset": "manual", "variables": {"DT{1}": "overlay"}},
    )
    assert not preset.is_error and not manual.is_error
    names = [i.name for i in viewer.scene.items]
    assert "dti_fit FA" in names and "dti_fit DT{1}" in names
    assert len(json.loads(_text(preset))) == 5


def test_an_action_error_comes_back_as_a_tool_error_with_its_message(
    server: McpServer, main: MainThread
) -> None:
    result = _call(server, main, "set_camera", {"view": "axial", "preset": "axial"})
    assert result.is_error
    [content] = result.content
    assert isinstance(content, TextContent)
    assert "The axial view has no camera positions." in content.text


@pytest.fixture
def loading_server(main: MainThread, tmp_path: Path) -> Iterator[McpServer]:
    viewer = Viewer()
    loader = Loader(viewer, main.post)
    server = McpServer(
        viewer, main.post, tmp_path / "token", background={"load": loader.load}
    )
    server.start(free_port())
    yield server
    server.stop()
    loader.stop()


def _text(result: CallToolResult) -> str:
    [content] = result.content
    assert isinstance(content, TextContent)
    return content.text


def test_an_agents_loads_give_the_items_and_scene_the_loader_gives(
    loading_server: McpServer, main: MainThread, tmp_path: Path
) -> None:
    a = _volume_file(tmp_path / "a.nii.gz")
    bad = tmp_path / "bad.nii"
    bad.write_bytes(b"not a nifti file")
    b = tmp_path / "b.nii.gz"
    nib.save(nib.Nifti1Image(np.full((3, 3, 3), 7, np.float32), np.eye(4)), b)
    files = [(a, Category.VOLUMES), (bad, Category.VOLUMES), (b, Category.VOLUMES)]

    async def drive() -> list[CallToolResult]:
        async with connect(loading_server, main) as client:
            return [
                await client.call_tool("load", {"path": str(p), "category": c.value})
                for p, c in files
            ]

    results = asyncio.run(drive())
    # The same files through a loader of its own, as the window opens them.
    window = Viewer()
    told: list[tuple[object, str | None]] = []

    async def through_loader() -> None:
        loader = Loader(window, main.post)
        main.loop = asyncio.get_running_loop()
        for p, c in files:
            # As JSON when told: a later volume hides this one.
            loader.load(p, c, None, then=lambda i, e: told.append((to_json(i), e)))
        while len(told) < len(files):
            await asyncio.sleep(0.01)

    asyncio.run(through_loader())
    assert [r.is_error for r in results] == [False, True, False]
    assert json.loads(_text(results[0])) == told[0][0]
    assert json.loads(_text(results[2])) == told[2][0]
    error = told[1][1]
    assert error is not None and error.startswith(f"Cannot open {bad} as volume")
    assert error in _text(results[1])
    scene = loading_server.viewer.scene
    assert [to_json(i) for i in scene.items] == [to_json(i) for i in window.scene.items]
    assert [i.source for i in scene.items] == [a, b]
    assert scene.view.main_volume == window.scene.view.main_volume


def test_an_agent_loads_a_freesurfer_surface_with_its_curvature(
    loading_server: McpServer, main: MainThread, tmp_path: Path
) -> None:
    path = tmp_path / "lh.white"
    save_surface(Surface(np.eye(3), np.array([[0, 1, 2]])), path)
    write_freesurfer_values(np.array([0.1, -0.1, 0.2]), tmp_path / "lh.curv")
    arguments = {"path": str(path), "category": "surfaces"}
    result = _call(loading_server, main, "load", arguments)
    assert not result.is_error, _text(result)
    [item] = loading_server.viewer.scene.items
    assert json.loads(_text(result))["id"] == item.id
    assert item.category == Category.SURFACES
    assert isinstance(item.options, SurfaceOptions)
    assert item.options.colour_by == "curv"


def test_the_main_thread_runs_other_work_while_an_agents_load_is_read(
    loading_server: McpServer,
    main: MainThread,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _volume_file(tmp_path / "v.nii.gz")
    volumes = registry.item_type(Category.VOLUMES)
    read: list[float] = []

    def slow(path: Path, progress: Progress | None, limit: int | None) -> object:
        read.append(time.monotonic())
        time.sleep(0.3)
        data = volumes.load(path, progress, limit)
        read.append(time.monotonic())
        return data

    slowed = dataclasses.replace(volumes, load=slow)
    monkeypatch.setitem(registry.REGISTRY, Category.VOLUMES, slowed)
    ticks: list[float] = []

    async def drive() -> CallToolResult:
        async def tick() -> None:
            while True:
                ticks.append(time.monotonic())
                await asyncio.sleep(0.01)

        async with connect(loading_server, main) as client:
            ticking = asyncio.create_task(tick())
            try:
                return await client.call_tool(
                    "load", {"path": str(path), "category": "volumes"}
                )
            finally:
                ticking.cancel()

    result = asyncio.run(drive())
    assert not result.is_error
    start, end = read
    assert any(start < t < end for t in ticks)
    [item] = loading_server.viewer.scene.items
    assert json.loads(_text(result))["id"] == item.id


def test_an_agents_load_without_a_limit_reads_with_the_loaders_limit(
    main: MainThread, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _volume_file(tmp_path / "v.nii.gz")
    volumes = registry.item_type(Category.VOLUMES)
    limits: list[int | None] = []

    def read(path: Path, progress: Progress | None, limit: int | None) -> object:
        limits.append(limit)
        return volumes.load(path, progress, limit)

    recorded = dataclasses.replace(volumes, load=read)
    monkeypatch.setitem(registry.REGISTRY, Category.VOLUMES, recorded)
    viewer = Viewer()
    loader = Loader(viewer, main.post, max_streamlines=7)
    server = McpServer(
        viewer, main.post, tmp_path / "token", background={"load": loader.load}
    )
    server.start(free_port())
    try:
        result = _call(server, main, "load", {"path": str(path), "category": "volumes"})
    finally:
        server.stop()
        loader.stop()
    assert not result.is_error
    assert limits == [7]


def test_stopping_the_server_ends_an_agents_load_still_being_read_at_once(
    main: MainThread, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _volume_file(tmp_path / "v.nii.gz")
    volumes = registry.item_type(Category.VOLUMES)
    started, go = threading.Event(), threading.Event()

    def gated(path: Path, progress: Progress | None, limit: int | None) -> object:
        started.set()
        assert go.wait(TIMEOUT)
        return volumes.load(path, progress, limit)

    gate = dataclasses.replace(volumes, load=gated)
    monkeypatch.setitem(registry.REGISTRY, Category.VOLUMES, gate)
    viewer = Viewer()
    loader = Loader(viewer, main.post)
    server = McpServer(
        viewer, main.post, tmp_path / "token", background={"load": loader.load}
    )
    server.start(free_port())
    escaped: list[object] = []

    async def drive() -> tuple[float, CallToolResult | BaseException]:
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, context: escaped.append(context)
        )
        async with connect(server, main) as client:
            arguments = {"path": str(path), "category": "volumes"}
            call = asyncio.create_task(client.call_tool("load", arguments))
            assert await asyncio.to_thread(started.wait, TIMEOUT)
            begun = time.monotonic()
            await asyncio.to_thread(server.stop)
            try:
                outcome: CallToolResult | BaseException = await call
            except Exception as error:
                outcome = error
            took = time.monotonic() - begun
            # The read ends after the stop: telling its item must raise nothing.
            go.set()
            while loader.busy:
                await asyncio.sleep(0.01)
            return took, outcome

    try:
        took, outcome = asyncio.run(drive())
    finally:
        go.set()
        server.stop()
        loader.stop()
    # Well under uvicorn's 2 s wait for open requests on shutdown.
    assert took < 1.0
    assert isinstance(outcome, CallToolResult)
    assert outcome.is_error
    assert "closed before the file was read" in _text(outcome)
    assert escaped == []


def test_an_agent_sees_the_screenshot_it_takes(
    gpu: None, server: McpServer, main: MainThread, tmp_path: Path
) -> None:
    path = tmp_path / "axial.png"
    arguments = {"view": ViewKind.AXIAL.value, "path": str(path), "size": [16, 16]}
    result = _call(server, main, "screenshot", arguments)
    assert not result.is_error
    image, text = result.content
    assert isinstance(image, ImageContent)
    assert image.mime_type == "image/png"
    assert isinstance(text, TextContent)
    assert json.loads(text.text) == str(path)
    assert path.is_file()


def test_the_window_loads_no_mcp_until_the_server_starts() -> None:
    # A fresh interpreter: this test process has imported mcp already.
    heavy = ("mcp", "uvicorn", "starlette")
    code = (
        "import sys\n"
        "import larmor.viewer.app, larmor.viewer.qt.main_window\n"
        f"print(','.join(m for m in {heavy!r} if m in sys.modules))"
    )
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
