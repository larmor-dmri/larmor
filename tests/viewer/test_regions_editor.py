from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from larmor.io import load_tractogram
from larmor.types import Tractogram
from larmor.viewer import Category, OptionName, RegionRole, Viewer
from larmor.viewer.model.options import TractogramOptions
from larmor.viewer.qt.regions_editor import RegionsEditor


def _tractogram() -> Tractogram:
    points = np.array([[0.0, 0, 0], [1, 1, 1], [9, 9, 9], [10, 10, 10]])
    return Tractogram(points, np.array([0, 2, 4]))


def _rows(editor: RegionsEditor) -> list[tuple[str, bool]]:
    rows = [editor.list.item(i) for i in range(editor.list.count())]
    return [(r.text(), r.checkState() == Qt.CheckState.Checked) for r in rows]


def test_the_list_ticks_the_regions_applied_and_a_tick_applies_one(
    qtbot: QtBot,
) -> None:
    viewer = Viewer()
    tracts = viewer.add(_tractogram(), Category.TRACTOGRAMS)
    editor = RegionsEditor(viewer, tracts.id)
    qtbot.addWidget(editor)
    applied = viewer.add_sphere((0.0, 0.0, 0.0), 2.0, apply=True)
    viewer.add_box((9.0, 9.0, 9.0), (3.0, 3.0, 3.0), RegionRole.EXCLUDE)
    assert _rows(editor) == [("Sphere 1 (include)", True), ("Box 1 (exclude)", False)]
    editor.list.item(1).setCheckState(Qt.CheckState.Checked)
    editor.list.item(0).setCheckState(Qt.CheckState.Unchecked)
    o = tracts.options
    assert isinstance(o, TractogramOptions)
    assert applied not in o.regions and len(o.regions) == 1
    viewer.set(o.regions[0], OptionName.ROLE, RegionRole.WITHIN)
    assert _rows(editor)[1] == ("Box 1 (within)", True)


def test_save_kept_writes_the_streamlines_the_regions_keep(
    qtbot: QtBot, tmp_path: Path
) -> None:
    viewer = Viewer()
    tracts = viewer.add(_tractogram(), Category.TRACTOGRAMS)
    viewer.add_sphere((0.0, 0.0, 0.0), 2.0, apply=True)
    editor = RegionsEditor(viewer, tracts.id)
    qtbot.addWidget(editor)
    editor.save(str(tmp_path / "kept.tck"))
    assert load_tractogram(tmp_path / "kept.tck").num_streamlines == 1
    editor.save(str(tmp_path / "kept.nope"))
    assert not editor.error.isHidden()
