from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog
from pytestqt.qtbot import QtBot

from larmor.dwi import dilate_mask, label_mask, threshold_mask
from larmor.io import load_volume
from larmor.types import Volume
from larmor.viewer import Category, Viewer
from larmor.viewer.model.mask_maker import MaskSource
from larmor.viewer.qt.mask_sheet import PAINT_NAME, MaskSheet


@pytest.fixture
def sheet(qtbot: QtBot) -> MaskSheet:
    viewer = Viewer()
    x, y, z = np.indices((20, 20, 12))
    image = np.where((x - 10) ** 2 + (y - 10) ** 2 + (z - 6) ** 2 <= 30, 1.0, 0.1)
    viewer.add(Volume(image.astype(np.float32), np.eye(4)), Category.VOLUMES, name="fa")
    labels = np.zeros((20, 20, 12), np.int16)
    labels[2:6, 2:6, 4:8] = 4
    viewer.add(Volume(labels, np.eye(4)), Category.OVERLAYS, name="parc")
    widget = MaskSheet(viewer)
    qtbot.addWidget(widget)
    return widget


def _data(sheet: MaskSheet, name: str) -> np.ndarray:
    item = next(i for i in sheet.viewer.scene.items if i.name == name)
    assert isinstance(item.data, Volume)
    return np.asarray(item.data.data)


def test_the_preview_follows_the_threshold_and_the_clean_up(sheet: MaskSheet) -> None:
    sheet.choose_source(MaskSource.THRESHOLD)
    sheet.lower_bound.setValue(0.5)
    sheet.dilate.setValue(1)
    sheet.update_preview()
    expected = dilate_mask(threshold_mask(_data(sheet, "fa"), 0.5), 1)
    assert sheet.preview_mask is not None
    np.testing.assert_array_equal(sheet.preview_mask, expected)
    assert sheet.count.text().startswith(f"{expected.sum():,} voxels")
    assert not sheet.preview.pixmap().isNull()


def test_create_mask_adds_the_labels_mask_and_returns_its_id(sheet: MaskSheet) -> None:
    sheet.choose_source(MaskSource.LABELS)
    sheet.labels_box.setCurrentIndex(sheet.labels_box.findText("parc"))
    sheet._image_changed()
    (row,) = sheet._label_rows()
    row.setCheckState(Qt.CheckState.Checked)
    sheet.name.setText("Region 4")
    created = sheet.create_mask()
    assert created is not None and created == sheet.created
    item = sheet.viewer.scene.item(created)
    assert (item.name, item.category) == ("Region 4", Category.OVERLAYS)
    np.testing.assert_array_equal(
        _data(sheet, "Region 4"), label_mask(_data(sheet, "parc"), [4])
    )


def test_save_creates_the_mask_and_writes_it(
    sheet: MaskSheet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "mask.nii.gz"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *_, **__: (str(path), "")
    )
    sheet.choose_source(MaskSource.THRESHOLD)
    sheet.lower_bound.setValue(0.5)
    sheet.save_button.click()
    assert sheet.created is not None
    assert sheet.viewer.scene.item(sheet.created).source == path
    np.testing.assert_array_equal(
        load_volume(path).data, threshold_mask(_data(sheet, "fa"), 0.5)
    )


def test_paint_is_listed_but_cannot_be_chosen(sheet: MaskSheet) -> None:
    rows = [sheet.sources.item(i) for i in range(sheet.sources.count())]
    (paint,) = [r for r in rows if r.text() == PAINT_NAME]
    assert not paint.flags() & Qt.ItemFlag.ItemIsEnabled
