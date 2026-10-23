from pathlib import Path

import pytest
from PySide6.QtWidgets import QDialogButtonBox, QFileDialog, QMenu, QWidget
from pytestqt.qtbot import QtBot

from larmor.viewer import Category, LoadAs, MatlabPreset, Viewer
from larmor.viewer.qt.main_window import MATLAB_FILTER, MainWindow, _agent_matlab
from larmor.viewer.qt.matlab_dialog import MatlabImportDialog

DATA = Path(__file__).parents[1] / "data" / "mritoolkit"
FIT = DATA / "dti_fit.mat"
WAIT_MS = 10_000


def _load_button(dialog: MatlabImportDialog) -> bool:
    return dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()


def test_the_fitting_preset_is_selected_and_the_others_say_why_not(
    qtbot: QtBot,
) -> None:
    dialog = MatlabImportDialog(FIT, Viewer())
    qtbot.addWidget(dialog)
    assert dialog.preset == MatlabPreset.TENSOR_FIT
    tracts = dialog.presets[MatlabPreset.TRACTOGRAPHY]
    assert not tracts.isEnabled() and "Tracts" in tracts.toolTip()
    assert dialog.presets[MatlabPreset.MANUAL].isEnabled()
    assert _load_button(dialog)


def test_manual_rows_list_every_variable_and_refuse_a_misfit(qtbot: QtBot) -> None:
    dialog = MatlabImportDialog(FIT, Viewer())
    qtbot.addWidget(dialog)
    dialog.presets[MatlabPreset.MANUAL].setChecked(True)
    assert {"DT", "DT{1}", "hdr.hist.srow_x", "FA"} <= set(dialog.load_as)
    assert not _load_button(dialog)  # nothing chosen yet
    dialog.set_load_as("VDims", LoadAs.VOLUME)
    assert "fewer than three" in dialog.row_problem("VDims")
    assert not _load_button(dialog)
    dialog.set_load_as("VDims", LoadAs.NOTHING)
    dialog.set_load_as("FA", LoadAs.VOLUME)
    assert dialog.row_problem("FA") == "" and _load_button(dialog)
    assert dialog.choice().variables == {"FA": LoadAs.VOLUME}


def test_a_file_without_a_grid_needs_a_reference(qtbot: QtBot, tmp_path: Path) -> None:
    path = tmp_path / "grl.mat"
    path.write_bytes((DATA / "grl.mat").read_bytes())
    dialog = MatlabImportDialog(path, Viewer())
    qtbot.addWidget(dialog)
    assert dialog.preset == MatlabPreset.FOD_FIT
    assert dialog.reference.text() == "" and not _load_button(dialog)
    dialog.reference.setText(str(FIT))
    assert _load_button(dialog)


def test_an_agents_import_is_read_by_the_loader(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    told: list[tuple[object, str | None]] = []
    _agent_matlab(
        window.loader,
        FIT,
        MatlabPreset.TENSOR_FIT,
        then=lambda result, error: told.append((result, error)),
    )
    assert window.loader.busy  # queued for the worker thread, not read here
    qtbot.waitUntil(lambda: bool(told), timeout=WAIT_MS)
    [(items, error)] = told
    assert error is None and isinstance(items, list) and len(items) == 5


def test_file_open_of_a_mat_file_imports_what_the_dialog_chose(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    qtbot.addWidget(window)

    def pick(
        parent: QWidget, caption: str, folder: str, filters: str, selected: str
    ) -> tuple[list[str], str]:
        assert MATLAB_FILTER in filters.split(";;")
        return [str(FIT)], MATLAB_FILTER

    monkeypatch.setattr(QFileDialog, "getOpenFileNames", pick)
    monkeypatch.setattr(
        MatlabImportDialog, "exec", lambda self: MatlabImportDialog.DialogCode.Accepted
    )
    [file_menu] = [a.menu() for a in window.menuBar().actions() if a.text() == "&File"]
    assert isinstance(file_menu, QMenu)
    [action] = [a for a in file_menu.actions() if a.text() == "&Open..."]
    action.trigger()
    qtbot.waitUntil(lambda: len(window.viewer.scene.items) == 5, timeout=WAIT_MS)
    qtbot.waitUntil(lambda: not window.loader.busy, timeout=WAIT_MS)
    items = window.viewer.scene.items
    assert {i.category for i in items} == {Category.VOLUMES}
    assert all(i.part is not None and i.source == FIT for i in items)
