from pathlib import Path

import numpy as np
import pytest

from larmor.io import save_volume
from larmor.types import Volume
from larmor.viewer import Category, Tool, Viewer
from larmor.viewer.model.view_state import Layout, Workspace


def test_a_tool_the_new_workspace_lacks_goes_back_to_the_cursor() -> None:
    viewer = Viewer()
    viewer.set_tool(Tool.BOX)
    viewer.set_workspace(Workspace.SIMULATE)
    assert viewer.scene.view.tool == Tool.CURSOR
    viewer.set_tool(Tool.DRAW_BUNDLE)
    viewer.set_workspace(Workspace.VIEW)
    assert viewer.scene.view.tool == Tool.CURSOR


def test_a_tool_both_workspaces_offer_is_kept() -> None:
    viewer = Viewer()
    viewer.set_tool(Tool.CURSOR)
    viewer.set_workspace(Workspace.SIMULATE)
    assert viewer.scene.view.workspace == Workspace.SIMULATE
    assert viewer.scene.view.tool == Tool.CURSOR


@pytest.mark.parametrize("action", ["set_workspace", "set_layout"])
def test_text_for_a_workspace_or_layout_is_refused(action: str) -> None:
    with pytest.raises(TypeError):
        getattr(Viewer(), action)("simulate")


def test_the_workspace_and_layout_reopen_from_a_session(tmp_path: Path) -> None:
    save_volume(Volume(np.zeros((4, 4, 4), np.float32), np.eye(4)), tmp_path / "t.nii")
    viewer = Viewer()
    viewer.load(tmp_path / "t.nii", Category.VOLUMES)
    viewer.set_workspace(Workspace.SIMULATE)
    viewer.set_layout(Layout.ONE_PLUS_THREE)
    viewer.save_session(tmp_path / "s.lar.json")
    other = Viewer()
    report = other.load_session(tmp_path / "s.lar.json")
    assert report.problems == ()
    assert other.scene.view.workspace == Workspace.SIMULATE
    assert other.scene.view.layout == Layout.ONE_PLUS_THREE
