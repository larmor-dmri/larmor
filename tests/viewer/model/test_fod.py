import numpy as np
import pytest

from larmor.types import Tractogram, Volume
from larmor.viewer import Category, Viewer


def test_an_fod_with_a_count_of_no_basis_is_refused() -> None:
    viewer = Viewer()
    volume = Volume(np.zeros((2, 2, 2, 5), np.float32), np.eye(4))
    with pytest.raises(ValueError, match="5 SH coefficients fit no layout"):
        viewer.add(volume, Category.FODS)
    assert viewer.scene.items == []


def _fod(size: float) -> Volume:
    return Volume(np.zeros((2, 3, 4, 6), np.float32), np.diag([size] * 3 + [1.0]))


def test_without_volumes_or_overlays_the_reference_grid_is_an_fod_grid() -> None:
    viewer = Viewer()
    points = np.array([[0, 0, 0], [9, 9, 9]], np.float32)
    viewer.add(
        Tractogram(points=points, offsets=np.array([0, 2])), Category.TRACTOGRAMS
    )
    viewer.add(_fod(2.0), Category.FODS)
    grid = viewer._reference_grid()
    assert grid is not None
    assert np.array_equal(grid.affine, np.diag([2.0, 2.0, 2.0, 1.0]))
    assert grid.shape == (2, 3, 4)


def test_an_overlay_grid_is_preferred_to_an_fod_added_before_it() -> None:
    viewer = Viewer()
    viewer.add(_fod(2.0), Category.FODS)
    overlay = Volume(np.zeros((2, 3, 4), np.float32), np.diag([5.0, 5.0, 5.0, 1.0]))
    viewer.add(overlay, Category.OVERLAYS)
    grid = viewer._reference_grid()
    assert grid is not None
    assert np.array_equal(grid.affine, np.diag([5.0, 5.0, 5.0, 1.0]))
