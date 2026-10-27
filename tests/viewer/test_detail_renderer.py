"""The detail window's drawing and its exports."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from rendercanvas.offscreen import RenderCanvas

from larmor.types import Volume
from larmor.viewer import (
    Category,
    DetailOption,
    FigureBackground,
    OptionName,
    Viewer,
    ViewKind,
)
from larmor.viewer.render.detail_renderer import DetailRenderer, render_detail
from larmor.viewer.render.fod_adapter import FodAdapter
from larmor.viewer.render.glyphs import sphere
from larmor.viewer.render.volume_adapter import VolumeAdapter


@pytest.fixture
def viewer(gpu: None) -> Viewer:
    viewer = Viewer()
    data = np.zeros((10, 10, 10), np.float32)
    data[2:5, 3:6, :] = 1.0
    viewer.add(Volume(data, np.eye(4)), Category.VOLUMES, name="t1")
    fod = np.zeros((10, 10, 10, 6), np.float32)
    fod[..., 0] = 1.0
    viewer.add(Volume(fod, np.eye(4)), Category.FODS, name="wmfod.mif")
    viewer.move_cursor((4.0, 4.0, 4.0))
    return viewer


def _renderer(viewer: Viewer, detail_id: int) -> DetailRenderer:
    size = viewer.scene.detail(detail_id).size
    return DetailRenderer(viewer, detail_id, RenderCanvas(size=size))


def test_the_rectangle_fills_the_canvas_with_unsmoothed_slices(viewer: Viewer) -> None:
    detail_id = viewer.open_detail(ViewKind.AXIAL, (2, 3, 4, 2))
    renderer = _renderer(viewer, detail_id)
    width, height = renderer.view.logical_size()
    assert width / height == pytest.approx(2.0, rel=0.02)  # 4 x 2 voxels of 1 mm
    low = renderer.view.to_world(0.0, height)
    high = renderer.view.to_world(width, 0.0)
    np.testing.assert_allclose(low[:2], (1.5, 2.5), atol=0.05)
    np.testing.assert_allclose(high[:2], (5.5, 4.5), atol=0.05)
    volume = next(a for a in renderer.adapters.values() if isinstance(a, VolumeAdapter))
    (slice_,) = volume.slices.values()
    assert slice_.material.interpolation == "nearest"
    renderer.close()


def test_glyphs_are_built_for_the_box_only_at_the_detail_s_glyph_detail(
    viewer: Viewer,
) -> None:
    fod = viewer.scene.in_category(Category.FODS)[0]
    viewer.set(fod.id, OptionName.DETAIL, 3)
    detail_id = viewer.open_detail(ViewKind.AXIAL, (2, 3, 4, 2))
    viewer.set_detail(detail_id, DetailOption.GLYPH_DETAIL, 5)
    renderer = _renderer(viewer, detail_id)
    adapter = next(a for a in renderer.adapters.values() if isinstance(a, FodAdapter))
    (mesh,) = adapter.meshes.values()
    vertices = len(sphere(5).vertices)
    assert len(mesh.geometry.positions.data) == 8 * vertices  # 4 x 2 voxels
    viewer.move_detail(detail_id, (0, 0, 3, 3))
    assert len(mesh.geometry.positions.data) == 9 * vertices
    renderer.close()


def test_an_unlinked_detail_does_not_follow_the_crosshair(viewer: Viewer) -> None:
    detail_id = viewer.open_detail(ViewKind.AXIAL, (0, 0, 10, 10))
    viewer.set_detail(detail_id, DetailOption.LINKED, False)
    renderer = _renderer(viewer, detail_id)
    volume = next(a for a in renderer.adapters.values() if isinstance(a, VolumeAdapter))
    (slice_,) = volume.slices.values()
    before = tuple(slice_.material.plane)
    viewer.move_cursor((4.0, 4.0, 7.0))
    assert tuple(slice_.material.plane) == before
    viewer.set_detail(detail_id, DetailOption.LINKED, True)
    assert tuple(slice_.material.plane) != before
    renderer.close()


def test_saving_draws_the_picture_at_a_multiple_of_its_size(
    viewer: Viewer, tmp_path: Path
) -> None:
    detail_id = viewer.open_detail(ViewKind.AXIAL, (2, 3, 4, 2))
    viewer.resize_detail(detail_id, (80, 40))
    path = viewer.save_detail(detail_id, tmp_path / "detail.png", 2, transparent=True)
    with Image.open(path) as saved:
        assert saved.size == (160, 80) and saved.mode == "RGBA"
    fod = viewer.scene.in_category(Category.FODS)[0]
    viewer.set(fod.id, OptionName.VISIBLE, False)
    image = render_detail(viewer, viewer.scene.detail(detail_id), (80, 40), 1)
    # Voxels 2 to 4 along x hold 1: the left three quarters are bright.
    row = image[10, :, 0]  # between the voxel edges
    assert row[10] > 100 and row[70] < 50
    with pytest.raises(ValueError, match="between 1 and 8"):
        viewer.save_detail(detail_id, tmp_path / "big.png", 9)


def test_a_figure_shows_the_view_and_the_detail_on_the_background(
    viewer: Viewer, tmp_path: Path
) -> None:
    detail_id = viewer.open_detail(ViewKind.AXIAL, (2, 3, 4, 2))
    viewer.resize_detail(detail_id, (80, 40))
    path = viewer.save_detail_figure(
        detail_id, tmp_path / "figure.png", 1, FigureBackground.WHITE
    )
    with Image.open(path) as image:
        figure = np.asarray(image.convert("RGB"))
    # Margins of 24 px, the view 40 x 40, a gap of 48 px, the detail 80 x 40.
    assert figure.shape == (24 * 2 + 40, 24 * 2 + 40 + 48 + 80, 3)
    assert (figure[2, 2] == 255).all()
    # A connector crosses the gap between the two pictures.
    gap = figure[24:64, 24 + 40 + 10 : 24 + 40 + 38]
    assert (gap != 255).any()
