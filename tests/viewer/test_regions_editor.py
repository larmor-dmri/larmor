from collections.abc import Callable

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from larmor.types import Tractogram, Volume
from larmor.viewer import Category, Viewer
from larmor.viewer.model.options import TractogramOptions
from larmor.viewer.qt.regions_editor import RegionsEditor


def _tractogram() -> Tractogram:
    return Tractogram(np.array([[0.0, 0, 0], [1, 1, 1]]), np.array([0, 2]))


def test_the_buttons_add_remove_and_clear_regions_through_the_viewer(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    viewer = Viewer()
    tracts = viewer.add(_tractogram(), Category.TRACTOGRAMS)
    editor = RegionsEditor(viewer, tracts.id)
    qtbot.addWidget(editor)
    # Loaded after the editor opens: it is offered all the same.
    mask = viewer.add(
        Volume(np.ones((2, 2, 2)), np.eye(4)), Category.OVERLAYS, name="m"
    )
    # The editor's own calls only: the actions call `set` in turn.
    calls: list[str] = []
    depth = [0]
    for name in ("add_mask", "add_sphere", "clear_regions", "set"):
        action: Callable[..., None] = getattr(viewer, name)

        def spy(
            *args: object,
            _name: str = name,
            _action: Callable[..., None] = action,
            **kwargs: object,
        ) -> None:
            if depth[0] == 0:
                calls.append(_name)
            depth[0] += 1
            try:
                _action(*args, **kwargs)
            finally:
                depth[0] -= 1

        monkeypatch.setattr(viewer, name, spy)

    editor.add_mask.click()
    editor.kind_choice.setCurrentIndex(1)
    editor.radius.setValue(3.0)
    editor.add_sphere.click()
    o = tracts.options
    assert isinstance(o, TractogramOptions)
    assert o.include_masks == (mask.id,)
    x, y, z = viewer.scene.view.cursor
    assert o.exclude_spheres == ((x, y, z, 3.0),)
    assert [editor.list.item(i).text() for i in range(editor.list.count())] == [
        "Include: m",
        f"Exclude: sphere ({x:g}, {y:g}, {z:g}), 3 mm",
    ]

    editor.list.item(0).setSelected(True)
    editor.remove_button.click()
    o = tracts.options
    assert isinstance(o, TractogramOptions)
    assert (o.include_masks, len(o.exclude_spheres)) == ((), 1)
    assert editor.list.count() == 1

    editor.clear_button.click()
    o = tracts.options
    assert isinstance(o, TractogramOptions)
    assert o.exclude_spheres == ()
    assert editor.list.count() == 0
    assert calls == ["add_mask", "add_sphere", "set", "clear_regions"]
