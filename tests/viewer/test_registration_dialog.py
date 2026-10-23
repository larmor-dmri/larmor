from pathlib import Path

from PySide6.QtWidgets import QDialogButtonBox
from pytestqt.qtbot import QtBot

from larmor.viewer import Category, Viewer
from larmor.viewer.qt.main_window import MainWindow
from larmor.viewer.qt.registration_dialog import RegistrationDialog

DATA = Path(__file__).parents[1] / "data" / "registration"
WAIT_MS = 10_000


def test_dialog_loads_the_files_through_the_registration(qtbot: QtBot) -> None:
    window = MainWindow(Viewer())
    qtbot.addWidget(window)
    dialog = window.open_through_registration()
    dialog.add_files([DATA / "tracks_fixed.tck"])
    dialog.category.setCurrentIndex(dialog.category.findData(Category.TRACTOGRAMS))
    dialog.add_transform(DATA / "mrtrix_warp.mif", DATA / "mrtrix_warp_inverse.mif")
    dialog.reference.setText(str(DATA / "fixed.nii.gz"))
    dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).click()
    qtbot.waitUntil(lambda: len(window.viewer.scene.items) == 1, timeout=WAIT_MS)
    item = window.viewer.scene.items[0]
    assert item.category == Category.TRACTOGRAMS
    assert item.registration is not None
    assert item.registration.transforms[0][1] == DATA / "mrtrix_warp_inverse.mif"


def test_dialog_without_transforms_says_so(qtbot: QtBot) -> None:
    dialog = RegistrationDialog(Category.VOLUMES)
    qtbot.addWidget(dialog)
    dialog.add_files([DATA / "moving.nii.gz"])
    dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).click()
    assert "transform" in dialog.problem.text()
    assert dialog.result() == 0


def test_dialog_passes_the_format_and_the_fsl_input(qtbot: QtBot) -> None:
    dialog = RegistrationDialog(Category.TRACTOGRAMS)
    qtbot.addWidget(dialog)
    dialog.add_files([DATA / "tracks_fixed.tck"])
    dialog.add_transform(DATA / "fsl_warp.nii.gz")
    dialog.format.setCurrentIndex(dialog.format.findData("fsl"))
    dialog.fsl_input.setText(str(DATA / "moving.nii.gz"))
    registration = dialog.choice().registration
    assert registration.kind == "fsl"
    assert registration.fsl_input == DATA / "moving.nii.gz"
