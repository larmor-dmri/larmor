"""Samples images and fields at voxel coordinates."""

from typing import Literal

import numpy as np
from scipy import ndimage

# The interpolation methods for images.
Interpolation = Literal["nearest", "linear", "cubic"]


def read_only(array: np.ndarray) -> np.ndarray:
    """Returns a read-only view of `array`."""
    view = array.view()
    view.flags.writeable = False
    return view


def nearest_dtype(dtype: np.dtype, fill: float) -> np.dtype:
    """Returns the type of nearest-neighbour values, the image's if `fill` fits it."""
    with np.errstate(all="ignore"):
        fits = bool(np.array(fill).astype(dtype) == fill)
    return dtype if fits else np.result_type(dtype, np.float64)


def inside(voxels: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Marks the (N, 3) voxel coordinates within half a voxel of the outer centres.

    Returns:
        (N,) bool.
    """
    upper = np.asarray(shape[:3], dtype=np.float64) - 0.5
    with np.errstate(invalid="ignore"):
        result: np.ndarray = ((voxels >= -0.5) & (voxels <= upper)).all(axis=1)
    return result


def sample_field(data: np.ndarray, voxels: np.ndarray) -> np.ndarray:
    """Returns the trilinear values of (X, Y, Z, C) data at (N, 3) voxel coordinates.

    Within half a voxel beyond the outer centres the edge values are used; further
    out the value is 0.

    Args:
        data: The values, shape (X, Y, Z, C).
        voxels: The voxel coordinates, shape (N, 3).

    Returns:
        The values, shape (N, C).
    """
    shape = data.shape[:3]
    result = np.zeros((len(voxels), data.shape[3]), dtype=np.float64)
    keep = inside(voxels, shape)
    v = voxels[keep]
    if len(v) == 0:
        return result
    base = np.floor(v).astype(np.int64)
    frac = v - base
    total = np.zeros((len(v), data.shape[3]), dtype=np.float64)
    for corner in range(8):
        offset = np.array([(corner >> 2) & 1, (corner >> 1) & 1, corner & 1])
        index = np.clip(base + offset, 0, np.array(shape) - 1)
        weight = np.prod(np.where(offset == 1, frac, 1 - frac), axis=1)
        total += weight[:, None] * data[index[:, 0], index[:, 1], index[:, 2]]
    result[keep] = total
    return result


def cubic_coefficients(data: np.ndarray) -> np.ndarray:
    """Returns the cubic B-spline coefficients of a 3D image, mirrored at its edges."""
    coefficients: np.ndarray = ndimage.spline_filter(
        np.asarray(data, dtype=np.float64), order=3, mode="mirror"
    )
    return coefficients


def sample_image(
    data: np.ndarray, voxels: np.ndarray, interpolation: Interpolation, fill: float
) -> np.ndarray:
    """Returns the values of a 3D image at (N, 3) voxel coordinates.

    Points outside the grid get `fill`. For "cubic", `data` holds the B-spline
    coefficients (`cubic_coefficients`). "nearest" keeps the image's type; the
    others give float64.

    Args:
        data: The image, shape (X, Y, Z).
        voxels: The voxel coordinates, shape (N, 3).
        interpolation: How to interpolate.
        fill: The value outside the grid.

    Returns:
        The values, shape (N,).
    """
    keep = inside(voxels, data.shape)
    v = voxels[keep]
    if interpolation == "nearest":
        result = np.full(len(voxels), fill, dtype=nearest_dtype(data.dtype, fill))
        # Round half up, so a point halfway between two voxels takes the upper one.
        index = np.clip(np.floor(v + 0.5).astype(np.int64), 0, np.array(data.shape) - 1)
        result[keep] = data[index[:, 0], index[:, 1], index[:, 2]]
        return result
    result = np.full(len(voxels), fill, dtype=np.float64)
    if interpolation == "linear":
        result[keep] = sample_field(data[..., None], v)[:, 0]
    else:
        result[keep] = ndimage.map_coordinates(
            data, v.T, order=3, mode="mirror", prefilter=False
        )
    return result
