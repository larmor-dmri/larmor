"""Brain masks from a b0 volume, mask operations, and putting masked values back."""

from collections.abc import Iterable
from typing import Literal, cast

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
    if not mask.any():
        raise ValueError(
            f"The brain mask is empty after {step}. Pass a smaller morph, or give "
            "a mask explicitly."
        )
    return largest_component(mask)


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


def threshold_mask(
    image: npt.NDArray[np.floating], lower: float, upper: float | None = None
) -> npt.NDArray[np.bool_]:
    """Makes a mask of the voxels whose value lies between two bounds.

    Args:
        image: The image, shape (X, Y, Z).
        lower: The lowest value kept, included.
        upper: The highest value kept, included; None for no upper bound.
            Defaults to None.

    Returns:
        Boolean mask, shape of `image`; NaN voxels are outside.
    """
    values = np.asarray(image)
    mask = values >= lower
    if upper is not None:
        mask &= values <= upper
    return np.asarray(mask, dtype=bool)


def label_mask(
    labels: npt.NDArray[np.number], values: Iterable[int]
) -> npt.NDArray[np.bool_]:
    """Makes a mask of the voxels of a parcellation that hold one of some labels.

    Args:
        labels: The parcellation, shape (X, Y, Z); its values are rounded to
            whole numbers.
        values: The labels kept.

    Returns:
        Boolean mask, shape of `labels`.
    """
    data = np.asarray(labels)
    if not np.issubdtype(data.dtype, np.integer):
        data = np.rint(np.nan_to_num(data, nan=0.0))
    return np.isin(data, np.fromiter(values, dtype=np.int64))


def dilate_mask(mask: npt.NDArray[np.bool_], voxels: int) -> npt.NDArray[np.bool_]:
    """Grows a mask by a number of voxels.

    Each step adds the voxels that share a face with the mask.

    Args:
        mask: Boolean mask, shape (X, Y, Z).
        voxels: The number of steps, 0 or more; 0 returns a copy.

    Returns:
        The grown mask.

    Raises:
        ValueError: If `voxels` is negative.
    """
    _check_steps(voxels)
    if voxels == 0:
        return np.array(mask, dtype=bool)
    return np.asarray(binary_dilation(mask, iterations=voxels), dtype=bool)


def erode_mask(mask: npt.NDArray[np.bool_], voxels: int) -> npt.NDArray[np.bool_]:
    """Shrinks a mask by a number of voxels.

    Each step removes the voxels that share a face with the outside; the
    outside of the image counts as outside the mask.

    Args:
        mask: Boolean mask, shape (X, Y, Z).
        voxels: The number of steps, 0 or more; 0 returns a copy.

    Returns:
        The shrunk mask.

    Raises:
        ValueError: If `voxels` is negative.
    """
    _check_steps(voxels)
    if voxels == 0:
        return np.array(mask, dtype=bool)
    return np.asarray(binary_erosion(mask, iterations=voxels), dtype=bool)


def _check_steps(voxels: int) -> None:
    """Raises ValueError if a number of dilation or erosion steps is negative."""
    if voxels < 0:
        raise ValueError(f"The number of voxels must be 0 or more, got {voxels}.")


def fill_mask_holes(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Adds to a mask the voxels it encloses in 3D.

    Args:
        mask: Boolean mask, shape (X, Y, Z).

    Returns:
        The filled mask.
    """
    return np.asarray(binary_fill_holes(mask), dtype=bool)


def largest_component(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Keeps the largest connected piece of a mask.

    Voxels are connected when they share a face.

    Args:
        mask: Boolean mask, shape (X, Y, Z).

    Returns:
        The piece, shape of `mask`; empty for an empty mask.
    """
    labels, count = label(mask)
    if count == 0:
        return np.zeros(np.shape(mask), dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    # numpy's stubs type this comparison as Any; it is a boolean array.
    return cast(npt.NDArray[np.bool_], labels == np.argmax(sizes))


def combine_masks(
    first: npt.NDArray[np.bool_],
    second: npt.NDArray[np.bool_],
    combination: Literal["union", "intersect", "subtract"],
) -> npt.NDArray[np.bool_]:
    """Combines two masks on the same grid.

    Args:
        first: Boolean mask, shape (X, Y, Z).
        second: Boolean mask of the same shape.
        combination: "union" keeps the voxels in either mask, "intersect" those
            in both, and "subtract" those in the first and not the second.

    Returns:
        The combined mask.

    Raises:
        ValueError: If the shapes differ or the combination is unknown.
    """
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(
            f"The masks must have the same shape, got {a.shape} and {b.shape}."
        )
    if combination == "union":
        combined = a | b
    elif combination == "intersect":
        combined = a & b
    elif combination == "subtract":
        combined = a & ~b
    else:
        raise ValueError(
            "The combination must be union, intersect or subtract, got "
            f"{combination!r}."
        )
    return np.asarray(combined, dtype=bool)
