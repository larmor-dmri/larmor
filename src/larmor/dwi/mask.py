"""Brain masks from a b0 volume, and putting masked values back on the grid."""

from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    gaussian_filter,
    label,
)
from skimage.morphology import reconstruction


def _img_fill(data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Fills the dark regions of a 3D image that do not touch the volume's border.

    Uses greyscale reconstruction by erosion, seeded from the border voxels.

    Returns:
        The filled image, with the shape of data, (X, Y, Z).
    """
    seed = np.full_like(data, data.max())

    seed[0, :, :] = data[0, :, :]
    seed[-1, :, :] = data[-1, :, :]
    seed[:, 0, :] = data[:, 0, :]
    seed[:, -1, :] = data[:, -1, :]
    seed[:, :, 0] = data[:, :, 0]
    seed[:, :, -1] = data[:, :, -1]

    # skimage's reconstruction has no annotations; it returns a float array.
    return cast(
        npt.NDArray[np.floating],
        reconstruction(seed, data, method="erosion"),
    )


def _compute_threshold(data: npt.NDArray[np.floating]) -> np.floating:
    """Finds an intensity threshold separating background from foreground.

    Iterates the threshold to the midpoint of the medians below and above it, using
    the positive voxels up to the 99th percentile.

    Returns:
        The threshold.
    """
    v = data[data > 0].copy()
    v = v[v <= np.percentile(v, 99)]

    t: np.floating = np.median(v)
    for _ in range(100):
        t_new = (np.median(v[v <= t]) + np.median(v[v > t])) / 2
        if t_new == t:
            break
        t = t_new

    return t


def _get_largest_component(
    mask: npt.NDArray[np.bool_], step: str
) -> npt.NDArray[np.bool_]:
    """Keeps the largest connected component of a mask.

    Returns:
        The component, with the shape of `mask`.

    Raises:
        ValueError: If the mask is empty; `step` names the step that emptied it.
    """
    labels, nums = label(mask)
    if nums == 0:
        raise ValueError(
            f"The brain mask is empty after {step}. Pass a smaller morph, or give "
            "a mask explicitly."
        )
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    # numpy's stubs type this comparison as Any; it is a boolean array.
    return cast(npt.NDArray[np.bool_], labels == np.argmax(sizes))


def _make_structuring_element(size: int) -> npt.NDArray[np.bool_]:
    """Makes a roughly spherical structuring element with `size` voxels a side."""
    se = np.zeros((size, size, size))
    c = (size - 1) // 2
    se[c, c, c] = 1.0
    se = gaussian_filter(se, sigma=1.0)
    threshold: np.float64 = se[c, c, -1]
    return se >= threshold


def create_brain_mask(
    data_b0: npt.NDArray[np.floating], *, threshold: float = 0.5, morph: int = 7
) -> npt.NDArray[np.bool_]:
    """Computes a brain mask from a b0 volume.

    The volume is thresholded at an automatic level between background and
    foreground, multiplied by `threshold`; the largest connected region is kept,
    and an erosion and a dilation with a roughly spherical element of `morph`
    voxels remove thin structures outside the brain.

    Args:
        data_b0: The b0 volume, shape (X, Y, Z).
        threshold: Factor applied to the automatic intensity threshold; lower
            values give a larger mask. Defaults to 0.5.
        morph: Size in voxels of the structuring element used for erosion and
            dilation. Defaults to 7.

    Returns:
        Boolean mask, shape (X, Y, Z).

    Raises:
        ValueError: If no voxel is left after thresholding or after the
            erosion, for example when the object is smaller than the
            structuring element.
    """
    vol: npt.NDArray[np.floating] = data_b0.astype(np.float64)
    vol = _img_fill(vol)

    th = _compute_threshold(vol)
    mask = vol > th * threshold

    mask = _get_largest_component(mask, "thresholding")
    mask = binary_fill_holes(mask)

    se = _make_structuring_element(morph)

    mask = binary_erosion(mask, structure=se)
    mask = _get_largest_component(mask, "erosion")
    mask = binary_dilation(mask, structure=se)

    for i in range(mask.shape[2]):
        mask[:, :, i] = binary_fill_holes(mask[:, :, i])

    return mask.astype(bool)


def unmask(
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    fill_value: float = 0.0,
    dtype: npt.DTypeLike = np.float32,
) -> np.ndarray:
    """Puts the values of the masked voxels back on the image grid.

    Args:
        values: One value, shape (K,), or one row, shape (K, D), per voxel inside
            the mask, in the order of its True voxels.
        mask: Boolean mask, shape (X, Y, Z).
        fill_value: Value outside the mask. Defaults to 0.
        dtype: Output dtype; None keeps that of `values`. Defaults to float32.

    Returns:
        Array of shape (X, Y, Z) or (X, Y, Z, D).
    """
    if dtype is None:
        dtype = values.dtype

    extra_dims = values.shape[1:] if values.ndim > 1 else ()
    out_shape = mask.shape + extra_dims

    out = np.full(out_shape, fill_value, dtype=dtype)
    out[mask] = values.astype(dtype)
    return out
