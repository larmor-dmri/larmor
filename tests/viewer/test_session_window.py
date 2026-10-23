import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox
from pytestqt.qtbot import QtBot

from larmor.io import Progress, save_volume
from larmor.types import Volume
from larmor.viewer import Category, OptionName, SessionError, Viewer, ViewKind
from larmor.viewer.model import registry
from larmor.viewer.model.shortcuts import Command
from larmor.viewer.qt import main_window
from larmor.viewer.qt.main_window import MainWindow, _agent_session

WAIT_MS = 10_000  # generous, a tiny file reads in milliseconds


@pytest.fixture
def window(qtbot: QtBot) -> MainWindow:
    window = MainWindow()
    qtbot.addWidget(window)
    return window


@pytest.fixture
def boxes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    shown: list[str] = []

    class Box(QMessageBox):
        def show(self) -> None:
            shown.append(self.text())

    monkeypatch.setattr(main_window, "QMessageBox", Box)
    return shown


def _saved_session(folder: Path) -> tuple[Viewer, Path]:
    viewer = Viewer()
    for name in ("t1", "fa"):
        data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        save_volume(Volume(data, np.eye(4)), folder / f"{name}.nii.gz")
    t1 = viewer.load(folder / "t1.nii.gz", Category.VOLUMES)
    fa = viewer.load(folder / "fa.nii.gz", Category.OVERLAYS)
    viewer.set(fa.id, OptionName.OPACITY_2D, 0.4)
    viewer.set(fa.id, OptionName.MODULATE, t1.id)
    viewer.move_cursor((1.0, 0.5, 0.25))
    viewer.zoom(ViewKind.AXIAL, 2.0)
    viewer.rotate_camera(30.0, 15.0)
    viewer.save_session(folder / "s.lar.json")
    return viewer, folder / "s.lar.json"


def _open_session_from_menu(
    window: MainWindow, path: Path, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))
    window.shortcuts[Command.OPEN_SESSION].trigger()
    qtbot.waitUntil(lambda: not window.loader.busy, timeout=WAIT_MS)


def test_open_session_restores_options_and_views_once_files_load(
    window: MainWindow,
    boxes: list[str],
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    saved, path = _saved_session(tmp_path)
    window.viewer.add(
        Volume(np.zeros((2, 2, 2), np.float32), np.eye(4)), Category.VOLUMES
    )

    _open_session_from_menu(window, path, qtbot, monkeypatch)

    # The volume there first took id 1, so the session's items get new ids.
    t1, fa = window.viewer.scene.items
    old_t1, old_fa = saved.scene.items
    assert t1.options == old_t1.options
    assert fa.options == replace(old_fa.options, modulate=t1.id)  # type: ignore[call-arg]
    assert window.viewer.scene.view == replace(saved.scene.view, main_volume=t1.id)
    assert boxes == []


def test_open_session_reports_a_missing_file_in_a_message_box(
    window: MainWindow,
    boxes: list[str],
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, path = _saved_session(tmp_path)
    (tmp_path / "fa.nii.gz").unlink()

    _open_session_from_menu(window, path, qtbot, monkeypatch)

    [text] = boxes
    assert "fa.nii.gz" in text
    assert [i.name for i in window.viewer.scene.items] == ["t1.nii.gz"]


def test_a_session_that_cannot_be_saved_says_why(
    window: MainWindow,
    boxes: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(path: Path) -> list[str]:
        raise ValueError("Out of range float values are not JSON compliant")

    monkeypatch.setattr(window.viewer, "save_session", refuse)
    window.save_session(tmp_path / "s.lar.json")
    assert len(boxes) == 1
    assert "not JSON compliant" in boxes[0]


def test_a_name_without_the_suffix_asks_before_replacing_the_file_it_names(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "view.lar.json"
    existing.write_text("old")
    asked: list[str] = []

    def question(*args: object, **kwargs: object) -> QMessageBox.StandardButton:
        asked.append(str(args[2]))
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *args: (str(tmp_path / "view"), "")
    )
    monkeypatch.setattr(QMessageBox, "question", question)
    window.shortcuts[Command.SAVE_SESSION].trigger()
    assert asked == ["view.lar.json already exists. Replace it?"]
    assert existing.read_text() == "old"


class Gate:
    """A volume reader that blocks on `go` for the file `slow` only."""

    def __init__(self, folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.slow = folder / "slow.nii.gz"
        save_volume(Volume(np.zeros((2, 2, 2), np.float32), np.eye(4)), self.slow)
        self.started, self.go = threading.Event(), threading.Event()
        self._volumes = registry.item_type(Category.VOLUMES)
        gated = replace(self._volumes, load=self._read)
        monkeypatch.setitem(registry.REGISTRY, Category.VOLUMES, gated)

    def _read(self, file: Path, progress: Progress | None, limit: int | None) -> object:
        if file == self.slow:
            self.started.set()
            assert self.go.wait(WAIT_MS / 1000)
        return self._volumes.load(file, progress, limit)


def test_an_agents_session_cancels_a_load_still_being_read(
    window: MainWindow,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, path = _saved_session(tmp_path)
    gate = Gate(tmp_path, monkeypatch)
    loaded: list[str | None] = []
    opened: list[str | None] = []
    window.loader.load(gate.slow, Category.VOLUMES, then=lambda i, e: loaded.append(e))
    assert gate.started.wait(WAIT_MS / 1000)
    _agent_session(window.loader, path, then=lambda r, e: opened.append(e))
    gate.go.set()
    qtbot.waitUntil(lambda: not window.loader.busy, timeout=WAIT_MS)
    qtbot.waitUntil(
        lambda: not any(t.name == "viewer-loader" for t in threading.enumerate()),
        timeout=WAIT_MS,
    )
    qtbot.wait(50)  # the read's posted result, which must be dropped
    assert loaded == ["Loading slow.nii.gz was cancelled."]
    assert opened == [None]
    sources = [i.source for i in window.viewer.scene.items]
    assert sources == [tmp_path / "t1.nii.gz", tmp_path / "fa.nii.gz"]


@pytest.mark.parametrize("session", ["missing.lar.json", "slow.nii.gz"])
def test_an_agents_session_that_cannot_open_keeps_the_load_being_read(
    session: str,
    window: MainWindow,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gate = Gate(tmp_path, monkeypatch)
    loaded: list[str | None] = []
    opened: list[str | None] = []
    window.loader.load(gate.slow, Category.VOLUMES, then=lambda i, e: loaded.append(e))
    assert gate.started.wait(WAIT_MS / 1000)
    with pytest.raises((OSError, SessionError)):
        _agent_session(
            window.loader, tmp_path / session, then=lambda r, e: opened.append(e)
        )
    gate.go.set()
    qtbot.waitUntil(lambda: bool(loaded), timeout=WAIT_MS)
    assert loaded == [None]
    assert opened == []
    assert [i.source for i in window.viewer.scene.items] == [gate.slow]


def test_the_window_hands_agents_loads_and_sessions_to_its_loader(
    window: MainWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    saved, path = _saved_session(tmp_path)
    told: list[tuple[object, str | None]] = []
    window.mcp.background["load"](
        tmp_path / "t1.nii.gz", Category.VOLUMES, then=lambda i, e: told.append((i, e))
    )
    assert window.loader.busy
    qtbot.waitUntil(lambda: bool(told), timeout=WAIT_MS)
    [(item, error)] = told
    assert error is None
    assert window.viewer.scene.items == [item]
    window.mcp.background["load_session"](path, then=lambda r, e: told.append((r, e)))
    assert told[1][1] is None
    loaded = [(i.name, i.category) for i in window.viewer.scene.items]
    assert loaded == [(i.name, i.category) for i in saved.scene.items]
