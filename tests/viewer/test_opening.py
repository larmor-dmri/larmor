import asyncio
import dataclasses
import socket
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import httpx2
import numpy as np
import pytest
import shiboken6
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from PySide6.QtCore import (
    QCoreApplication,
    QEvent,
    QMimeData,
    QPoint,
    QPointF,
    Qt,
    QTimer,
    QUrl,
)
from PySide6.QtGui import (
    QDragEnterEvent,
    QDragLeaveEvent,
    QDragMoveEvent,
    QDropEvent,
    QFileOpenEvent,
)
from PySide6.QtWidgets import QApplication, QFileDialog, QMenu, QWidget
from pytestqt.qtbot import QtBot
from viewer_helpers import free_port, zeros_volume

from larmor.io import Progress, save_surface, save_tractogram, save_volume
from larmor.types import Surface, Tractogram, Volume
from larmor.viewer import (
    MAX_STREAMLINES,
    Category,
    ItemId,
    Viewer,
    ViewKind,
    loading,
    system,
    view,
    wait,
)
from larmor.viewer import app as viewer_app
from larmor.viewer.mcp_server import McpServer
from larmor.viewer.model.registry import REGISTRY, ItemType, item_type
from larmor.viewer.model.settings import AppSettings, SettingName, Settings
from larmor.viewer.model.surface import SurfaceData
from larmor.viewer.qt.layers import LayersPanel
from larmor.viewer.qt.loading import ErrorBox
from larmor.viewer.qt.main_window import MainWindow

WAIT = 10.0  # seconds; generous, a tiny file reads in milliseconds
WAIT_MS = int(WAIT * 1000)


def _tractogram(n: int = 2) -> Tractogram:
    points = np.arange(n * 2 * 3, dtype=np.float32).reshape(-1, 3)
    return Tractogram(points=points, offsets=np.arange(0, 2 * n + 1, 2))


def _volume_file(path: Path) -> Path:
    save_volume(zeros_volume(), path)
    return path


def _tractogram_file(path: Path, n: int = 2) -> Path:
    save_tractogram(_tractogram(n), path)
    return path


def _file_for(category: Category, folder: Path) -> Path:
    if category == Category.TRACTOGRAMS:
        return _tractogram_file(folder / "t.tck")
    if category == Category.PEAKS:
        path = folder / "p.nii.gz"
        save_volume(Volume(np.zeros((2, 2, 2, 3), np.float32), np.eye(4)), path)
        return path
    if category == Category.SURFACES:
        path = folder / "lh.white"  # FreeSurfer: no extension
        save_surface(Surface(np.eye(3), np.array([[0, 1, 2]])), path)
        return path
    return _volume_file(folder / "v.nii.gz")


def _items(viewer: Viewer) -> list[tuple[str, Category]]:
    return [(i.name, i.category) for i in viewer.scene.items]


@pytest.fixture(autouse=True)
def _checks_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system, "check_viewer_can_start", lambda: None)


@pytest.fixture(autouse=True)
def _close_windows(qapp: QApplication) -> Iterator[None]:
    # view() keeps its windows alive; close them so no test sees another's.
    yield
    for w in list(viewer_app._WINDOWS.values()):
        w.close()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    assert viewer_app._WINDOWS == {}


@pytest.fixture(autouse=True)
def _no_error_box(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tests read the box's text; none needs it on screen.
    monkeypatch.setattr(ErrorBox, "show", lambda self: None)


@pytest.fixture
def window(qtbot: QtBot) -> MainWindow:
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def _settle(window: MainWindow, qtbot: QtBot) -> None:
    qtbot.waitUntil(lambda: not window.loader.busy, timeout=WAIT_MS)


class Gate:
    """A read that waits for the test: `started` once it runs, then `go`."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.go = threading.Event()

    def __call__(
        self, path: Path, progress: Progress | None, limit: int | None
    ) -> object:
        self.started.set()
        assert self.go.wait(WAIT)
        if progress is not None:
            progress(1, 1)  # raises ReadCancelledError once cancelled
        return zeros_volume()


def _gated(monkeypatch: pytest.MonkeyPatch) -> Gate:
    gate = Gate()

    def fake(category: Category) -> ItemType:
        return dataclasses.replace(item_type(category), load=gate)

    monkeypatch.setattr(loading, "item_type", fake)
    return gate


# view()


def test_view_adds_paths_and_objects_to_their_categories(tmp_path: Path) -> None:
    t1 = _volume_file(tmp_path / "t1.nii.gz")
    fa = _volume_file(tmp_path / "fa.nii")
    cst = _tractogram_file(tmp_path / "cst.tck")
    volume, tractogram = zeros_volume(), _tractogram()
    viewer = view(
        volumes=[t1, volume],
        overlays=str(fa),
        tracts=[tractogram, cst],
        block=False,
    )
    wait(viewer, timeout=WAIT)
    assert _items(viewer) == [
        ("Tractogram 1", Category.TRACTOGRAMS),
        ("t1.nii.gz", Category.VOLUMES),
        ("Volume 2", Category.VOLUMES),
        ("fa.nii", Category.OVERLAYS),
        ("cst.tck", Category.TRACTOGRAMS),
    ]


def _object_and_file(keyword: str, folder: Path) -> tuple[object, Path]:
    if keyword == "tracts":
        return _tractogram(), _tractogram_file(folder / "t.tck")
    return zeros_volume(), _volume_file(folder / "v.nii.gz")


@pytest.mark.parametrize("keyword", ["volumes", "overlays", "tracts"])
def test_view_adds_an_object_given_after_a_file_after_it_uncopied(
    keyword: str, tmp_path: Path
) -> None:
    obj, path = _object_and_file(keyword, tmp_path)
    viewer = view(**{keyword: [path, obj]}, block=False)  # type: ignore[arg-type]
    assert viewer.scene.items == []
    wait(viewer, timeout=WAIT)
    first, second = viewer.scene.items
    assert first.source == path
    assert second.data is obj
    assert second.source is None


@pytest.mark.parametrize("keyword", ["volumes", "overlays", "tracts"])
def test_view_adds_an_object_given_before_any_file_at_once(
    keyword: str, tmp_path: Path
) -> None:
    obj, path = _object_and_file(keyword, tmp_path)
    viewer = view(**{keyword: [obj, path]}, block=False)  # type: ignore[arg-type]
    assert [i.data for i in viewer.scene.items] == [obj]
    wait(viewer, timeout=WAIT)
    first, second = viewer.scene.items
    assert first.data is obj
    assert second.source == path


def test_view_adds_objects_at_once_without_copying() -> None:
    volume, overlay, tractogram = zeros_volume(), zeros_volume(), _tractogram()
    viewer = view(volumes=volume, overlays=[overlay], tracts=tractogram, block=False)
    data = {i.category: i.data for i in viewer.scene.items}
    assert data[Category.VOLUMES] is volume
    assert data[Category.OVERLAYS] is overlay
    assert data[Category.TRACTOGRAMS] is tractogram


@pytest.mark.parametrize(
    ("keyword", "good", "bad", "message"),
    [
        (
            "volumes",
            zeros_volume(),
            _tractogram(),
            "Volumes hold Volume objects, not Tr",
        ),
        (
            "overlays",
            zeros_volume(),
            np.zeros(3),
            "Overlays hold Volume objects, not nd",
        ),
        (
            "tracts",
            _tractogram(),
            zeros_volume(),
            "Tractograms hold Tractogram objects",
        ),
        (
            "surfaces",
            Surface(np.eye(3), np.array([[0, 1, 2]])),
            zeros_volume(),
            "Surfaces hold Surface objects, not Volume",
        ),
    ],
)
def test_view_refuses_an_object_of_the_wrong_type_before_adding_any(
    keyword: str, good: object, bad: object, message: str
) -> None:
    # The good object comes first: nothing is added unless all are right.
    viewer = Viewer()
    with pytest.raises(ValueError, match=message):
        view(viewer, **{keyword: [good, bad]})  # type: ignore[arg-type]
    assert viewer.scene.items == []
    assert viewer_app.window(viewer) is None


def test_view_adds_surfaces_from_a_file_and_from_python(tmp_path: Path) -> None:
    path = _file_for(Category.SURFACES, tmp_path)
    surface = Surface(np.eye(3), np.array([[0, 1, 2]]))
    viewer = view(surfaces=[surface, path], block=False)
    wait(viewer, timeout=WAIT)
    assert _items(viewer) == [
        ("Surface 1", Category.SURFACES),
        ("lh.white", Category.SURFACES),
    ]
    given, read = viewer.scene.items
    assert isinstance(given.data, SurfaceData) and given.data.surface is surface
    assert read.source == path


def test_view_refuses_a_negative_read_limit() -> None:
    with pytest.raises(ValueError, match="max_streamlines"):
        view(max_streamlines=-1, block=False)
    assert viewer_app._WINDOWS == {}


def test_view_without_blocking_returns_before_files_are_read(tmp_path: Path) -> None:
    t1 = _volume_file(tmp_path / "t1.nii.gz")
    viewer = view(volumes=t1, block=False)
    window = viewer_app.window(viewer)
    assert window is not None
    assert window.isVisible()
    assert viewer.scene.items == []
    wait(viewer, timeout=WAIT)
    assert _items(viewer) == [("t1.nii.gz", Category.VOLUMES)]


def test_wait_gives_up_after_its_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate = _gated(monkeypatch)
    viewer = view(volumes=tmp_path / "slow.nii", block=False)
    with pytest.raises(TimeoutError):
        wait(viewer, timeout=0.2)
    assert gate.started.is_set()
    gate.go.set()


def test_view_blocks_until_the_window_is_closed(tmp_path: Path) -> None:
    viewer = Viewer()
    closed: list[bool] = []

    def close() -> None:
        window = viewer_app.window(viewer)
        assert window is not None
        closed.append(window.close())

    QTimer.singleShot(50, close)
    returned = view(viewer, volumes=_volume_file(tmp_path / "t1.nii.gz"))
    assert returned is viewer
    assert closed == [True]
    assert viewer_app.window(viewer) is None


def test_view_on_an_open_viewer_reuses_its_window(tmp_path: Path) -> None:
    viewer = view(volumes=zeros_volume(), block=False)
    window = viewer_app.window(viewer)
    assert view(viewer, tracts=_tractogram_file(tmp_path / "t.tck"), block=False) is (
        viewer
    )
    assert viewer_app.window(viewer) is window
    assert list(viewer_app._WINDOWS) == [viewer]
    wait(viewer, timeout=WAIT)
    assert [i.category for i in viewer.scene.items] == [
        Category.VOLUMES,
        Category.TRACTOGRAMS,
    ]


def test_view_on_a_viewer_whose_window_was_closed_opens_a_new_one() -> None:
    viewer = view(block=False)
    first = viewer_app.window(viewer)
    assert first is not None
    first.close()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    assert not shiboken6.isValid(first)
    view(viewer, block=False)
    second = viewer_app.window(viewer)
    assert second is not None
    assert second.viewer is viewer


def test_a_screenshot_leaves_an_open_window_following_the_scene(
    tmp_path: Path,
) -> None:
    viewer = view(volumes=[zeros_volume()], block=False)
    window = viewer_app.window(viewer)
    assert window is not None
    viewer.screenshot(ViewKind.AXIAL, tmp_path / "a.png", size=(16, 16))
    viewer.add(zeros_volume(), Category.OVERLAYS)
    assert len(window.renderer.adapters) == 2


def test_view_reduces_a_tractogram_file_to_the_same_subset_each_time(
    tmp_path: Path,
) -> None:
    path = _tractogram_file(tmp_path / "big.tck", n=40)
    subsets = []
    for _ in range(2):
        viewer = view(tracts=path, max_streamlines=5, block=False)
        wait(viewer, timeout=WAIT)
        [item] = viewer.scene.items
        assert isinstance(item.data, Tractogram)
        assert item.data.num_streamlines == 5
        subsets.append(item.data.points)
    np.testing.assert_array_equal(subsets[0], subsets[1])


@pytest.mark.parametrize("limit", [None, 40])
def test_view_keeps_every_streamline_up_to_the_limit(
    limit: int | None, tmp_path: Path
) -> None:
    path = _tractogram_file(tmp_path / "t.tck", n=40)
    viewer = view(tracts=path, max_streamlines=limit, block=False)
    wait(viewer, timeout=WAIT)
    [item] = viewer.scene.items
    assert isinstance(item.data, Tractogram)
    assert item.data.num_streamlines == 40


@pytest.fixture
def settings_limit_5(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unsaved settings in place of the session's, so the limit ends with the test.
    settings = Settings()
    settings.set(SettingName.READ_LIMIT, 5)
    monkeypatch.setattr(viewer_app, "_SETTINGS", settings)


@pytest.mark.usefixtures("settings_limit_5")
def test_view_without_a_limit_reads_with_the_settings_limit(tmp_path: Path) -> None:
    path = _tractogram_file(tmp_path / "t.tck", n=40)
    viewer = view(tracts=path, block=False)
    wait(viewer, timeout=WAIT)
    [item] = viewer.scene.items
    assert isinstance(item.data, Tractogram)
    assert item.data.num_streamlines == 5


@pytest.mark.usefixtures("settings_limit_5")
@pytest.mark.parametrize(("limit", "expected"), [(None, 40), (8, 8)])
def test_a_limit_given_to_view_wins_over_the_settings(
    limit: int | None, expected: int, tmp_path: Path
) -> None:
    path = _tractogram_file(tmp_path / "t.tck", n=40)
    viewer = view(tracts=path, max_streamlines=limit, block=False)
    wait(viewer, timeout=WAIT)
    [item] = viewer.scene.items
    assert isinstance(item.data, Tractogram)
    assert item.data.num_streamlines == expected


def test_view_never_reduces_a_tractogram_object() -> None:
    tractogram = _tractogram(40)
    viewer = view(tracts=tractogram, max_streamlines=5, block=False)
    [item] = viewer.scene.items
    assert item.data is tractogram
    assert tractogram.num_streamlines == 40


# view(mcp=...)


def _move_cursor_as_an_agent(server: McpServer, position: list[float]) -> None:
    # The agent's own thread: its call reaches the window's thread through Qt.
    async def call() -> None:
        headers = {"Authorization": f"Bearer {server.token}"}
        async with (
            httpx2.AsyncClient(headers=headers, timeout=WAIT) as http,
            Client(streamable_http_client(server.url(), http_client=http)) as client,
        ):
            result = await client.call_tool("move_cursor", {"position": position})
            assert not result.is_error

    asyncio.run(call())


@pytest.mark.parametrize("given", ["mcp_port", "mcp"])
def test_view_serves_agents_whose_calls_run_in_the_window(
    given: str, monkeypatch: pytest.MonkeyPatch, qtbot: QtBot
) -> None:
    port = free_port()
    if given == "mcp":
        settings = Settings(AppSettings(mcp_port=port))
        monkeypatch.setattr(viewer_app, "_SETTINGS", settings)
        viewer = view(mcp=True, block=False)
    else:
        viewer = view(mcp_port=port, block=False)
    window = viewer_app.window(viewer)
    assert window is not None
    assert window.mcp.port == port
    agent = threading.Thread(
        target=_move_cursor_as_an_agent, args=(window.mcp, [1.0, 2.0, 3.0])
    )
    agent.start()
    qtbot.waitUntil(
        lambda: viewer.scene.view.cursor == (1.0, 2.0, 3.0), timeout=WAIT_MS
    )
    agent.join(WAIT)
    assert not agent.is_alive()


def test_closing_the_window_stops_its_mcp_server() -> None:
    port = free_port()
    viewer = view(mcp_port=port, block=False)
    window = viewer_app.window(viewer)
    assert window is not None
    window.close()
    assert not window.mcp.running
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


def test_view_on_a_taken_port_says_so_and_opens_the_window(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        viewer = view(mcp_port=port, block=False)
    window = viewer_app.window(viewer)
    assert window is not None
    assert window.isVisible()
    assert not window.mcp.running
    message = capsys.readouterr().err
    assert f"port {port}" in message
    assert "--mcp-port" in message


class Limits:
    """Reads files for real and records each file's read limit; holds `gated`."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, gated: str = "") -> None:
        self.gated = gated
        self.started = threading.Event()
        self.go = threading.Event()
        self.seen: list[tuple[str, int | None]] = []
        real = item_type

        def fake(category: Category) -> ItemType:
            kind = real(category)

            def read(
                path: Path, progress: Progress | None, limit: int | None
            ) -> object:
                self.seen.append((path.name, limit))
                if path.name == self.gated:
                    self.started.set()
                    assert self.go.wait(WAIT)
                return kind.load(path, progress, limit)

            return dataclasses.replace(kind, load=read)

        monkeypatch.setattr(loading, "item_type", fake)


def test_a_queued_tractogram_keeps_the_limit_of_its_view_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    limits = Limits(monkeypatch, gated="slow.nii.gz")
    slow = _volume_file(tmp_path / "slow.nii.gz")
    big = _tractogram_file(tmp_path / "big.tck", n=40)
    viewer = view(volumes=slow, tracts=big, max_streamlines=5, block=False)
    assert limits.started.wait(WAIT)
    view(viewer, volumes=_volume_file(tmp_path / "t1.nii.gz"), block=False)
    limits.go.set()
    wait(viewer, timeout=WAIT)
    assert ("big.tck", 5) in limits.seen
    [tracts] = [i for i in viewer.scene.items if i.category == Category.TRACTOGRAMS]
    assert isinstance(tracts.data, Tractogram)
    assert tracts.data.num_streamlines == 5


@pytest.mark.parametrize("how", ["file open", "drop"])
def test_files_opened_from_the_window_use_the_default_limit_not_views(
    how: str, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    viewer = view(
        tracts=_tractogram_file(tmp_path / "a.tck"), max_streamlines=5, block=False
    )
    wait(viewer, timeout=WAIT)
    window = viewer_app.window(viewer)
    assert window is not None
    limits = Limits(monkeypatch)
    later = _tractogram_file(tmp_path / "b.tck")
    if how == "file open":
        dialog = Dialog([later], Category.TRACTOGRAMS)
        monkeypatch.setattr(QFileDialog, "getOpenFileNames", dialog)
        _open_action(window)
    else:
        window.resize(1200, 800)
        qtbot.waitExposed(window)
        panel = window.layers
        at = _row_point(panel, Category.TRACTOGRAMS)
        try:
            assert _enter(panel, _files(later), at)
            assert _drop(panel, _files(later), at)
        finally:
            _leave(panel)
    _settle(window, qtbot)
    assert limits.seen == [("b.tck", MAX_STREAMLINES)]


def test_the_window_opens_files_with_the_default_read_limit() -> None:
    viewer = view(block=False)
    window = viewer_app.window(viewer)
    assert window is not None
    assert window.loader.max_streamlines == MAX_STREAMLINES


def test_viewer_load_reduces_with_its_read_limit(tmp_path: Path) -> None:
    path = _tractogram_file(tmp_path / "t.tck", n=40)
    item = Viewer().load(path, Category.TRACTOGRAMS, max_streamlines=5)
    assert isinstance(item.data, Tractogram)
    assert item.data.num_streamlines == 5


# The progress bar and errors


def test_the_progress_bar_shows_while_loading_and_hides_when_idle(
    window: MainWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    bar = window.loading_bar
    assert bar.isHidden()
    a = _volume_file(tmp_path / "a.nii.gz")
    b = _volume_file(tmp_path / "b.nii.gz")
    window.loader.open([a, b], Category.VOLUMES)
    assert not bar.isHidden()
    assert bar.label.text() == "Loading a.nii.gz (1 of 2)"
    _settle(window, qtbot)
    assert bar.isHidden()
    assert len(window.viewer.scene.items) == 2


def test_the_progress_bar_names_a_single_file_without_a_count(
    window: MainWindow, tmp_path: Path, qtbot: QtBot
) -> None:
    window.loader.open([_volume_file(tmp_path / "a.nii.gz")], Category.VOLUMES)
    assert window.loading_bar.label.text() == "Loading a.nii.gz"
    _settle(window, qtbot)


def test_the_cancel_button_stops_loading(
    window: MainWindow, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate = _gated(monkeypatch)
    window.loader.open([tmp_path / "a.nii", tmp_path / "b.nii"], Category.VOLUMES)
    assert gate.started.wait(WAIT)
    window.loading_bar.cancel.click()
    assert window.loading_bar.isHidden()
    assert not window.loader.busy
    gate.go.set()
    qtbot.wait(200)  # anything the worker still posts is handled
    assert window.viewer.scene.items == []
    assert window.loading_bar.isHidden()


def test_unreadable_files_are_collected_in_one_error_box(
    window: MainWindow, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shown: list[str] = []
    monkeypatch.setattr(ErrorBox, "show", lambda self: shown.append(self.text()))
    monkeypatch.setattr(ErrorBox, "isVisible", lambda self: bool(shown))
    missing = [tmp_path / "a.nii", tmp_path / "b.nii"]
    window.loader.open(missing, Category.VOLUMES)
    _settle(window, qtbot)
    [box] = [b for b in window.findChildren(ErrorBox) if b.text()]  # not the jobs'
    assert len(shown) == 1
    assert box.text() == "\n\n".join(
        f"Cannot open {p} as volumes: No such file or directory." for p in missing
    )
    assert box.windowModality() == Qt.WindowModality.NonModal


def test_closing_the_window_stops_its_loader(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate = _gated(monkeypatch)
    window = MainWindow()
    viewer, loader = window.viewer, window.loader
    loader.open([tmp_path / "a.nii"], Category.VOLUMES)
    assert gate.started.wait(WAIT)
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    assert not loader.busy
    gate.go.set()
    del window
    qapp.processEvents()
    assert viewer.scene.items == []


# File > Open


def _open_action(window: MainWindow) -> None:
    [file_menu] = [a.menu() for a in window.menuBar().actions() if a.text() == "&File"]
    assert isinstance(file_menu, QMenu)
    [action] = [a for a in file_menu.actions() if a.text() == "&Open..."]
    action.trigger()


class Dialog:
    """Stands in for QFileDialog.getOpenFileNames."""

    def __init__(self, paths: list[Path], category: Category | None) -> None:
        self.paths = paths
        self.category = category
        self.filters: list[str] = []
        self.preselected = ""

    def __call__(
        self, parent: QWidget, caption: str, folder: str, filters: str, selected: str
    ) -> tuple[list[str], str]:
        self.filters = filters.split(";;")
        self.preselected = selected
        chosen = "" if self.category is None else self.filter_of(self.category)
        return [str(p) for p in self.paths], chosen

    def filter_of(self, category: Category) -> str:
        label = item_type(category).label
        [f] = [f for f in self.filters if f.startswith(f"{label} (")]
        return f


@pytest.mark.parametrize("category", list(Category))
def test_file_open_loads_into_the_chosen_filters_category(
    category: Category,
    window: MainWindow,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dialog = Dialog([_file_for(category, tmp_path)], category)
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", dialog)
    _open_action(window)
    _settle(window, qtbot)
    assert [i.category for i in window.viewer.scene.items] == [category]


def test_file_open_offers_one_filter_per_category_with_its_extensions(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = Dialog([], None)
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", dialog)
    _open_action(window)
    assert len(dialog.filters) == len(REGISTRY) + 1
    assert dialog.filters[-1] == "MATLAB files (*.mat)"
    for category, kind in REGISTRY.items():
        patterns = dialog.filter_of(category).split("(")[1].rstrip(")").split()
        expected = [f"*{e}" for e in kind.extensions]
        assert patterns == expected + [f"{p}*" for p in kind.prefixes]
        assert {"*.nii.gz", "*.tck", "*.gii"} & set(patterns)
    assert "lh.*" in dialog.filter_of(Category.SURFACES)


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (None, Category.VOLUMES),
        (Category.OVERLAYS, Category.OVERLAYS),
        (Category.TRACTOGRAMS, Category.TRACTOGRAMS),
        ("tractogram item", Category.TRACTOGRAMS),
        ("overlay item", Category.OVERLAYS),
    ],
)
def test_file_open_preselects_the_category_selected_in_layers(
    selection: Category | str | None,
    expected: Category,
    window: MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    viewer = window.viewer
    if selection == "tractogram item":
        viewer.select(viewer.add(_tractogram(), Category.TRACTOGRAMS).id)
    elif selection == "overlay item":
        viewer.select(viewer.add(zeros_volume(), Category.OVERLAYS).id)
    elif isinstance(selection, Category):
        viewer.select(selection)
    dialog = Dialog([], None)
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", dialog)
    _open_action(window)
    assert dialog.preselected == dialog.filter_of(expected)


def test_cancelling_file_open_loads_nothing(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", Dialog([], None))
    _open_action(window)
    assert not window.loader.busy
    assert window.loading_bar.isHidden()
    assert window.viewer.scene.items == []


# Dropping files onto Layers


@pytest.fixture
def shown(window: MainWindow, qtbot: QtBot) -> Iterator[MainWindow]:
    window.resize(1200, 800)
    window.show()
    qtbot.waitExposed(window)
    yield window
    # Qt remembers the widget that accepted a drag until it ends; a later test's
    # drag would reach the deleted panel.
    _leave(window.layers)


def _mime(*urls: QUrl) -> QMimeData:
    mime = QMimeData()
    mime.setUrls(list(urls))
    return mime


def _files(*paths: Path) -> QMimeData:
    return _mime(*(QUrl.fromLocalFile(str(p)) for p in paths))


def _row_point(panel: LayersPanel, target: ItemId | Category) -> QPoint:
    return panel.visualItemRect(panel._rows[target]).center()


def _empty_point(panel: LayersPanel) -> QPoint:
    viewport = panel.viewport()
    point = QPoint(10, viewport.height() - 5)
    assert panel.itemAt(point) is None
    return point


_ACTION = Qt.DropAction.CopyAction
_BUTTONS = Qt.MouseButton.LeftButton
_MODS = Qt.KeyboardModifier.NoModifier


def _enter(panel: LayersPanel, mime: QMimeData, at: QPoint) -> bool:
    event = QDragEnterEvent(at, _ACTION, mime, _BUTTONS, _MODS)
    QApplication.sendEvent(panel.viewport(), event)
    return event.isAccepted()


def _move(panel: LayersPanel, mime: QMimeData, at: QPoint) -> bool:
    # Qt delivers moves and drops only to a widget that accepted the drag's enter.
    event = QDragMoveEvent(at, _ACTION, mime, _BUTTONS, _MODS)
    QApplication.sendEvent(panel.viewport(), event)
    return event.isAccepted()


def _drop(panel: LayersPanel, mime: QMimeData, at: QPoint) -> bool:
    event = QDropEvent(QPointF(at), _ACTION, mime, _BUTTONS, _MODS)
    QApplication.sendEvent(panel.viewport(), event)
    return event.isAccepted()


def _leave(panel: LayersPanel) -> None:
    QApplication.sendEvent(panel.viewport(), QDragLeaveEvent())


def _highlighted(panel: LayersPanel) -> set[ItemId | Category]:
    return {
        key
        for key, row in panel._rows.items()
        if row.background(0).style() != Qt.BrushStyle.NoBrush
    }


@pytest.mark.parametrize("category", list(Category))
def test_files_dropped_on_a_category_row_load_into_it(
    category: Category, shown: MainWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    panel = shown.layers
    # Real drags reach the panel only if its viewport takes drops.
    assert panel.viewport().acceptDrops()
    at = _row_point(panel, category)
    mime = _files(_file_for(category, tmp_path))
    assert _enter(panel, mime, at)
    assert _move(panel, mime, at)
    assert _drop(panel, mime, at)
    _settle(shown, qtbot)
    assert [i.category for i in shown.viewer.scene.items] == [category]


@pytest.mark.parametrize("category", [Category.OVERLAYS, Category.TRACTOGRAMS])
def test_files_dropped_on_an_item_row_load_into_its_category(
    category: Category, shown: MainWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    obj = _tractogram() if category == Category.TRACTOGRAMS else zeros_volume()
    item = shown.viewer.add(obj, category)
    panel = shown.layers
    at = _row_point(panel, item.id)
    mime = _files(_file_for(category, tmp_path))
    assert _enter(panel, mime, at)
    assert _move(panel, mime, at)
    assert _drop(panel, mime, at)
    _settle(shown, qtbot)
    assert [i.category for i in shown.viewer.scene.items] == [category, category]
    assert shown.viewer.scene.items[1].source == _file_for(category, tmp_path)


def test_several_files_dropped_at_once_all_load(
    shown: MainWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    panel = shown.layers
    paths = [_volume_file(tmp_path / n) for n in ("a.nii", "b.nii.gz")]
    assert _enter(panel, _files(*paths), _row_point(panel, Category.VOLUMES))
    assert _drop(panel, _files(*paths), _row_point(panel, Category.VOLUMES))
    _settle(shown, qtbot)
    assert _items(shown.viewer) == [
        ("a.nii", Category.VOLUMES),
        ("b.nii.gz", Category.VOLUMES),
    ]


def test_the_category_under_the_pointer_is_highlighted_while_dragging(
    shown: MainWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    panel = shown.layers
    viewer = shown.viewer
    overlay = viewer.add(zeros_volume(), Category.OVERLAYS).id
    volume = viewer.add(zeros_volume(), Category.VOLUMES).id
    mime = _files(_volume_file(tmp_path / "a.nii"))
    _enter(panel, mime, _row_point(panel, Category.OVERLAYS))
    _move(panel, mime, _row_point(panel, Category.OVERLAYS))
    assert _highlighted(panel) == {Category.OVERLAYS, overlay}
    _move(panel, mime, _row_point(panel, volume))
    assert _highlighted(panel) == {Category.VOLUMES, volume}
    _move(panel, mime, _empty_point(panel))
    assert _highlighted(panel) == set()
    _move(panel, mime, _row_point(panel, Category.VOLUMES))
    _leave(panel)
    assert _highlighted(panel) == set()
    _enter(panel, mime, _row_point(panel, Category.VOLUMES))
    _move(panel, mime, _row_point(panel, Category.VOLUMES))
    _drop(panel, mime, _row_point(panel, Category.VOLUMES))
    assert _highlighted(panel) == set()
    _settle(shown, qtbot)


def test_rows_added_while_dragging_are_highlighted_too(
    shown: MainWindow, tmp_path: Path
) -> None:
    panel = shown.layers
    mime = _files(_volume_file(tmp_path / "a.nii"))
    at = _row_point(panel, Category.VOLUMES)
    _enter(panel, mime, at)
    _move(panel, mime, at)
    item = shown.viewer.add(zeros_volume(), Category.VOLUMES).id
    _move(panel, mime, _row_point(panel, Category.VOLUMES))
    assert _highlighted(panel) == {Category.VOLUMES, item}


def test_a_drop_target_refuses_files_its_category_cannot_read(
    shown: MainWindow, tmp_path: Path
) -> None:
    panel = shown.layers
    mime = _files(_tractogram_file(tmp_path / "t.tck"))
    at = _row_point(panel, Category.VOLUMES)
    assert _enter(panel, mime, at)
    assert not _move(panel, mime, at)
    assert _highlighted(panel) == set()
    assert _move(panel, mime, _row_point(panel, Category.TRACTOGRAMS))


def _refused_drag(case: str, folder: Path) -> QMimeData:
    good = _volume_file(folder / "a.nii.gz")
    if case == "folder":
        return _files(folder)
    if case == "a folder among files":
        return _files(good, folder)
    if case == "web address":
        return _mime(QUrl("https://example.org/a.nii.gz"))
    if case == "web address among files":
        return _mime(QUrl.fromLocalFile(str(good)), QUrl("https://example.org/b.nii"))
    if case == "wrong extension":
        text = folder / "notes.txt"
        text.write_text("hello")
        return _files(text)
    if case == "missing file":
        return _files(folder / "gone.nii")
    if case == "no files":
        mime = QMimeData()
        mime.setText("a.nii.gz")
        return mime
    raise AssertionError(case)


REFUSED = [
    "folder",
    "a folder among files",
    "web address",
    "web address among files",
    "wrong extension",
    "missing file",
    "no files",
]


@pytest.mark.parametrize("case", REFUSED)
def test_a_drag_without_readable_local_files_is_refused(
    case: str, shown: MainWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    panel = shown.layers
    sent: list[object] = []
    panel.files_dropped.connect(lambda paths, category: sent.append(paths))
    mime = _refused_drag(case, tmp_path)
    at = _row_point(panel, Category.VOLUMES)
    if case != "wrong extension":
        assert not _enter(panel, mime, at)
    assert not _move(panel, mime, at)
    assert _highlighted(panel) == set()
    assert not _drop(panel, mime, at)
    assert sent == []
    assert not shown.loader.busy


def test_a_drop_on_empty_space_is_refused(shown: MainWindow, tmp_path: Path) -> None:
    panel = shown.layers
    sent: list[object] = []
    panel.files_dropped.connect(lambda paths, category: sent.append(paths))
    mime = _files(_volume_file(tmp_path / "a.nii.gz"))
    at = _empty_point(panel)
    assert _enter(panel, mime, at)
    assert not _move(panel, mime, at)
    assert not _drop(panel, mime, at)
    assert sent == []
    assert shown.viewer.scene.items == []


def test_the_dropped_signal_carries_paths_and_a_category(
    shown: MainWindow, tmp_path: Path
) -> None:
    panel = shown.layers
    sent: list[tuple[object, object]] = []
    panel.files_dropped.connect(lambda paths, category: sent.append((paths, category)))
    path = _volume_file(tmp_path / "a.nii.gz")
    assert _enter(panel, _files(path), _row_point(panel, Category.OVERLAYS))
    _drop(panel, _files(path), _row_point(panel, Category.OVERLAYS))
    assert sent == [([path], Category.OVERLAYS)]
    assert type(sent[0][1]) is Category


# Files the system asks the app to open (macOS: from Finder)


@pytest.fixture
def no_opener(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Each test starts without a filter, so none sees another's queued files.
    before = viewer_app._FILE_OPENER
    if before is not None:
        qapp.removeEventFilter(before)
    monkeypatch.setattr(viewer_app, "_FILE_OPENER", None)
    yield
    if viewer_app._FILE_OPENER is not None:
        qapp.removeEventFilter(viewer_app._FILE_OPENER)
    if before is not None:
        qapp.installEventFilter(before)


@pytest.fixture
def finder(no_opener: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    viewer_app.application()


def _finder_opens(*paths: Path) -> None:
    for path in paths:
        QApplication.sendEvent(viewer_app.application(), QFileOpenEvent(str(path)))
    QApplication.processEvents()  # the files open together, once the events are in


@pytest.mark.usefixtures("finder")
def test_a_file_asked_for_before_any_window_opens_in_the_first_one(
    tmp_path: Path,
) -> None:
    _finder_opens(_volume_file(tmp_path / "t1.nii.gz"))
    viewer = view(block=False)
    wait(viewer, timeout=WAIT)
    assert _items(viewer) == [("t1.nii.gz", Category.VOLUMES)]


@pytest.mark.usefixtures("finder")
def test_a_file_asked_for_opens_in_the_open_window(tmp_path: Path) -> None:
    viewer = view(block=False)
    _finder_opens(_tractogram_file(tmp_path / "cst.tck"))
    wait(viewer, timeout=WAIT)
    assert _items(viewer) == [("cst.tck", Category.TRACTOGRAMS)]


@pytest.mark.usefixtures("finder")
def test_a_finder_session_while_a_session_opens_is_reported_and_not_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown: list[str] = []
    monkeypatch.setattr(ErrorBox, "show", lambda self: shown.append(self.text()))
    given = tmp_path / "given.lar.json"
    Viewer().save_session(given)
    other = Viewer()
    other.load(_volume_file(tmp_path / "t1.nii.gz"), Category.VOLUMES)
    finders = tmp_path / "finders.lar.json"
    other.save_session(finders)
    _finder_opens(finders, _volume_file(tmp_path / "fa.nii.gz"))
    viewer = view(session=given, block=False)
    wait(viewer, timeout=WAIT)
    assert _items(viewer) == [("fa.nii.gz", Category.VOLUMES)]
    [message] = shown
    assert str(finders) in message


@pytest.mark.usefixtures("finder")
def test_two_finder_sessions_are_reported_and_its_volume_still_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown: list[str] = []
    monkeypatch.setattr(ErrorBox, "show", lambda self: shown.append(self.text()))
    sessions = [tmp_path / "a.lar.json", tmp_path / "b.lar.json"]
    for path in sessions:
        Viewer().save_session(path)
    _finder_opens(*sessions, _volume_file(tmp_path / "fa.nii.gz"))
    viewer = view(block=False)
    wait(viewer, timeout=WAIT)
    assert _items(viewer) == [("fa.nii.gz", Category.VOLUMES)]
    [message] = shown
    assert all(str(p) in message for p in sessions)


@pytest.mark.usefixtures("finder")
def test_two_sessions_from_finder_to_an_open_window_are_refused_as_one_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qtbot: QtBot
) -> None:
    shown: list[str] = []
    monkeypatch.setattr(ErrorBox, "show", lambda self: shown.append(self.text()))
    viewer = view(volumes=_volume_file(tmp_path / "t1.nii.gz"), block=False)
    wait(viewer, timeout=WAIT)
    sessions = [tmp_path / "a.lar.json", tmp_path / "b.lar.json"]
    for path in sessions:
        Viewer().save_session(path)
    _finder_opens(*sessions, _volume_file(tmp_path / "fa.nii.gz"))
    qtbot.waitUntil(lambda: len(viewer.scene.items) == 2, timeout=WAIT_MS)
    wait(viewer, timeout=WAIT)
    assert _items(viewer) == [
        ("t1.nii.gz", Category.VOLUMES),
        ("fa.nii.gz", Category.VOLUMES),
    ]
    [message] = shown
    assert message.startswith("Only one session")


@pytest.mark.usefixtures("no_opener")
def test_files_asked_for_are_left_to_the_arguments_off_macos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    _finder_opens(_volume_file(tmp_path / "t1.nii.gz"))
    viewer = view(block=False)
    wait(viewer, timeout=WAIT)
    assert viewer.scene.items == []


def test_application_without_streams_gives_writable_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # pythonw starts with no stdout or stderr.
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    viewer_app.application()
    for stream in (sys.stdout, sys.stderr):
        assert stream is not None
        stream.write("text")
        stream.close()
