"""The window's Save As entries, the Options panel after a data change, the job bar."""

import threading
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog
from pytestqt.qtbot import QtBot

from larmor.io import save_tractogram
from larmor.types import Surface, Tractogram, Volume
from larmor.viewer import Category, OptionName, Viewer
from larmor.viewer.jobs import SAVING, Report
from larmor.viewer.qt.field import SliderField
from larmor.viewer.qt.loading import ErrorBox
from larmor.viewer.qt.main_window import MainWindow
from larmor.viewer.qt.options_panel import OptionsPanel


@pytest.fixture
def window(qtbot: QtBot) -> MainWindow:
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def _tractogram() -> Tractogram:
    return Tractogram(np.arange(12.0).reshape(4, 3), np.array([0, 2, 4]))


def _answer(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))


def test_file_save_as_saves_the_selected_item(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = window.viewer.add(_tractogram(), Category.TRACTOGRAMS)
    window.viewer.select(item.id)
    _answer(monkeypatch, tmp_path / "bundle.tck")
    window._show_file_menu()
    assert window.save_item_action.isEnabled()
    window.save_item_action.trigger()
    assert (tmp_path / "bundle.tck").is_file()
    assert item.source == tmp_path / "bundle.tck"


def _action(window: MainWindow, item_id: int, text: str) -> QAction:
    menu = window.scene_tree.context_menu(item_id)
    return next(a for a in menu.actions() if a.text() == text)


def test_the_scene_trees_save_as_saves_that_item(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = window.viewer.add(_tractogram(), Category.TRACTOGRAMS)
    _answer(monkeypatch, tmp_path / "bundle.trk")
    save = _action(window, item.id, "Save As...")
    save.trigger()
    assert item.source == tmp_path / "bundle.trk"


def test_save_as_is_off_for_a_surface(window: MainWindow) -> None:
    surface = Surface(np.eye(3), np.array([[0, 1, 2]]))
    item = window.viewer.add(surface, Category.SURFACES)
    window.viewer.select(item.id)
    window._show_file_menu()
    assert not window.save_item_action.isEnabled()
    save = _action(window, item.id, "Save As...")
    assert not save.isEnabled()


def test_the_sliders_span_the_new_range_after_a_replace(qtbot: QtBot) -> None:
    viewer = Viewer()
    data = np.linspace(0.0, 1.0, 8, dtype=np.float32).reshape(2, 2, 2)
    item = viewer.add(Volume(data, np.eye(4)), Category.VOLUMES)
    viewer.select(item.id)
    panel = OptionsPanel(viewer)
    qtbot.addWidget(panel)
    viewer.replace_data(item.id, Volume(data * 50 - 10, np.eye(4)))
    field = panel._fields[OptionName.MAXIMUM]
    assert isinstance(field, SliderField)
    assert field.minimum == pytest.approx(-10.0)
    assert field.minimum + field.slider.maximum() * field.step == pytest.approx(40.0)


def test_the_job_bar_shows_the_job_and_cancel_stops_it(
    window: MainWindow, qtbot: QtBot
) -> None:
    started, go = threading.Event(), threading.Event()
    results: list[object] = []

    def work(report: Report) -> str:
        report(1, 4, "Signal")
        started.set()
        go.wait(10)
        report(2, 4, "Signal")  # cancelled by then
        return "result"

    window.jobs.start("Simulation", work, results.append)
    qtbot.waitUntil(lambda: window.job_bar.label.text() == "Simulation: Signal")
    assert not window.job_bar.isHidden()
    assert window.job_bar.progress.value() == 250
    window.job_bar.cancel.click()
    go.set()
    assert window.job_bar.isHidden()
    qtbot.waitUntil(lambda: not window.jobs.busy)  # once its thread stops
    qtbot.wait(50)
    assert results == []


def test_the_job_bar_cannot_cancel_a_job_that_is_saving(
    window: MainWindow, qtbot: QtBot
) -> None:
    go = threading.Event()

    def work(report: Report) -> str:
        report(1, 1, SAVING)
        go.wait(10)
        return "saved"

    results: list[object] = []
    window.jobs.start("Simulation", work, results.append)
    qtbot.waitUntil(lambda: "cannot be cancelled" in window.job_bar.label.text())
    assert not window.job_bar.cancel.isEnabled()
    go.set()
    qtbot.waitUntil(lambda: results == ["saved"])


def test_a_failed_job_shows_in_an_error_box(window: MainWindow, qtbot: QtBot) -> None:
    def work(report: Report) -> None:
        raise ValueError("no tissue")

    window.jobs.start("Simulation", work, lambda result: None)
    qtbot.waitUntil(lambda: not window.jobs.busy)
    boxes = [b for b in window.findChildren(ErrorBox) if b.isVisible()]
    assert [b.text() for b in boxes] == ["Simulation failed: no tissue"]


def test_save_as_never_suggests_a_subsets_own_file(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    points = np.arange(30, dtype=np.float32).reshape(-1, 3)
    save_tractogram(Tractogram(points, np.arange(0, 12, 2)), tmp_path / "whole.tck")
    item = window.viewer.load(
        tmp_path / "whole.tck", Category.TRACTOGRAMS, max_streamlines=2
    )
    asked: list[str] = []

    def dialog(
        parent: object, title: str, suggested: str, *args: object
    ) -> tuple[str, str]:
        asked.append(suggested)
        return "", ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", dialog)
    window.ask_save_item(item.id)
    assert asked == [str(tmp_path / "whole_subset.tck")]
