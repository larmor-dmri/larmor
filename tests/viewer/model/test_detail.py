import numpy as np
import pytest

from larmor.types import Grid, Volume
from larmor.viewer import Category, Viewer, ViewKind
from larmor.viewer.model.detail import (
    DetailOption,
    centred_box,
    scale_bar_length,
    voxel_box,
)
from larmor.viewer.model.events import DetailChanged, DetailClosed, DetailOpened

# 2 mm voxels, so mm and voxels differ.
GRID = Grid((20, 30, 40), np.diag([2.0, 2.0, 2.0, 1.0]))


def _viewer() -> Viewer:
    viewer = Viewer()
    viewer.add(Volume(np.zeros(GRID.shape), GRID.affine), Category.VOLUMES)
    return viewer


def test_a_drag_snaps_to_the_voxels_it_covers() -> None:
    # Coronal: the in-plane voxel axes are i (x) and k (z).
    box = voxel_box(GRID, ViewKind.CORONAL, (3.1, 0.0, 4.0), (9.0, 0.0, 0.9))
    # x from 1.55 to 4.5 voxels covers voxels 2 to 4; z from 0.45 to 2 covers 0 to 2.
    assert box == (2, 0, 3, 3)


def test_a_drag_within_one_voxel_gives_that_voxel() -> None:
    assert voxel_box(GRID, ViewKind.AXIAL, (4.0, 6.0, 0.0), (4.2, 6.1, 0.0)) == (
        2,
        3,
        1,
        1,
    )


def test_shift_makes_a_square_and_alt_grows_it_from_the_centre() -> None:
    start, end = (10.0, 10.0, 0.0), (14.0, 22.0, 0.0)
    square = voxel_box(GRID, ViewKind.AXIAL, start, end, square=True)
    assert square[2] == square[3] == 7
    centred = voxel_box(GRID, ViewKind.AXIAL, start, end, from_centre=True)
    # 2 voxels either side of voxel 5 along i, 6 either side along j.
    assert centred == (3, -1, 5, 13)


def test_the_menu_box_is_centred_on_the_voxel_clicked() -> None:
    assert centred_box(GRID, ViewKind.SAGITTAL, (0.0, 20.0, 30.0), 12) == (4, 9, 12, 12)


@pytest.mark.parametrize(("width", "length"), [(24.0, 2.0), (100.0, 20.0), (7.0, 1.0)])
def test_the_scale_bar_is_a_round_length_near_a_fifth_of_the_width(
    width: float, length: float
) -> None:
    assert scale_bar_length(width) == length


def test_a_detail_opens_clipped_to_the_grid_and_sends_its_events() -> None:
    viewer = _viewer()
    events: list[object] = []
    viewer.scene.subscribe(events.append)
    detail_id = viewer.open_detail(ViewKind.CORONAL, (-2, 35, 6, 10))
    detail = viewer.scene.detail(detail_id)
    assert detail.box == (0, 35, 4, 5)
    viewer.move_detail(detail_id, (1, 1, 4, 5))
    viewer.close_detail(detail_id)
    kinds = [type(e) for e in events]
    assert kinds == [DetailOpened, DetailChanged, DetailClosed]
    with pytest.raises(ValueError, match="outside the grid"):
        viewer.open_detail(ViewKind.CORONAL, (50, 50, 2, 2))
    with pytest.raises(ValueError, match="no slices"):
        viewer.open_detail(ViewKind.THREE_D, (0, 0, 2, 2))


def test_an_unlinked_detail_keeps_its_slice() -> None:
    viewer = _viewer()
    detail_id = viewer.open_detail(ViewKind.AXIAL, (0, 0, 4, 4))
    viewer.move_cursor((0.0, 0.0, 10.0))
    viewer.set_detail(detail_id, DetailOption.LINKED, False)
    viewer.move_cursor((0.0, 0.0, 30.0))
    assert viewer.detail_position(detail_id)[2] == 10.0
    assert "axial z 5" in viewer.detail_title(detail_id)
    viewer.set_detail(detail_id, DetailOption.LINKED, True)
    assert viewer.detail_position(detail_id)[2] == 30.0


def test_a_detail_refuses_a_glyph_detail_out_of_range() -> None:
    viewer = _viewer()
    detail_id = viewer.open_detail(ViewKind.AXIAL, (0, 0, 4, 4))
    viewer.set_detail(detail_id, DetailOption.GLYPH_DETAIL, 16)
    with pytest.raises(ValueError, match="between 2 and 16"):
        viewer.set_detail(detail_id, DetailOption.GLYPH_DETAIL, 17)
    assert viewer.scene.detail(detail_id).glyph_detail == 16


def test_the_title_names_the_fod_the_view_the_slice_and_the_size() -> None:
    viewer = _viewer()
    fod = Volume(np.zeros((*GRID.shape, 6), np.float32), GRID.affine)
    viewer.add(fod, Category.FODS, name="wmfod.mif")
    viewer.move_cursor((0.0, 120.0 / 2, 0.0))
    detail_id = viewer.open_detail(ViewKind.CORONAL, (0, 0, 12, 12))
    assert viewer.detail_title(detail_id) == (
        "Detail - wmfod.mif - coronal y 30 - 12 x 12 voxels"
    )
