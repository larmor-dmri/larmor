"""Maps direction vectors between world coordinates and an image's voxel axes."""

import numpy as np

from larmor.types import Grid


def voxel_axes(grid: Grid) -> np.ndarray:
    """Returns the world directions of the voxel axes i, j, k as unit columns (3, 3).

    This is the affine's 3x3 part with each column divided by its length (the
    voxel size). Under shear the columns are not orthogonal.
    """
    linear = grid.affine[:3, :3]
    unit: np.ndarray = linear / np.linalg.norm(linear, axis=0)
    return unit


def map_directions(vectors: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Applies `matrix` to each vector of `vectors` (..., 3), keeping its length.

    A matrix that is not orthogonal (an image with shear) changes the lengths; each
    result is rescaled to the length of its input, so unit vectors stay unit
    vectors and zero vectors stay zero.

    Returns:
        The mapped vectors, with the shape of `vectors`.
    """
    mapped = vectors @ matrix.T
    old = np.linalg.norm(vectors, axis=-1, keepdims=True)
    new = np.linalg.norm(mapped, axis=-1, keepdims=True)
    scale = np.divide(old, new, out=np.zeros_like(new), where=new > 0)
    result: np.ndarray = mapped * scale
    return result
