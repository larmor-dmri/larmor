"""Affine and orientation helpers shared by the types."""

from collections.abc import Sequence

import numpy as np

# The world axis (0 = x, 1 = y, 2 = z) that each orientation letter refers to.
AXIS_OF_CODE = {"R": 0, "L": 0, "A": 1, "P": 1, "S": 2, "I": 2}


def check_affine(affine: np.ndarray) -> np.ndarray:
    """Returns a float64 copy of `affine` after checking that it is a usable affine.

    Args:
        affine: The (4, 4) matrix from voxel indices to world coordinates (RAS+) in mm.

    Raises:
        ValueError: If `affine` is not a finite 4x4 matrix with last row
            [0, 0, 0, 1], or its voxel axes do not span 3D space (a zero-length or
            singular axis).
    """
    checked = np.array(affine, dtype=np.float64)
    if checked.shape != (4, 4):
        raise ValueError(f"The affine must be 4x4, got shape {checked.shape}.")
    if not np.isfinite(checked).all():
        raise ValueError("The affine contains NaN or infinite values.")
    if not np.allclose(checked[3, :], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(
            f"The affine's last row must be [0, 0, 0, 1], got {checked[3]}."
        )
    lengths = np.linalg.norm(checked[:3, :3], axis=0)
    if (lengths == 0.0).any():
        raise ValueError("The affine has a voxel axis of zero length.")
    singular = np.linalg.svd(checked[:3, :3] / lengths, compute_uv=False)
    if singular.min() <= singular.max() * 1e-8:
        raise ValueError("The affine's voxel axes do not span 3D space (singular).")
    return checked


def orientation_of(affine: np.ndarray) -> tuple[str, str, str]:
    """Returns the world direction closest to each voxel axis, such as ("R", "A", "S").

    Two voxel axes never get the same world axis, however oblique the affine.

    Args:
        affine: The (4, 4) matrix from voxel indices to world coordinates (RAS+) in mm.

    Raises:
        ValueError: If the voxel axes do not span 3D space (a singular affine).

    References:
        The same rule as nibabel's `io_orientation`.
    """
    mat = affine[:3, :3]
    dirs = mat / np.linalg.norm(mat, axis=0, keepdims=True)

    # The closest orthogonal matrix to the directions, which removes any shear.
    u, singular, vt = np.linalg.svd(dirs)
    if singular.min() <= singular.max() * 1e-8:
        raise ValueError("Affine does not define a valid axis orientation.")
    rotation = u @ vt

    pos_axis = ("R", "A", "S")
    neg_axis = ("L", "P", "I")

    result = ["", "", ""]
    # The most clearly aligned voxel axis picks its world axis first.
    voxel_axes = np.argsort(-np.max(rotation**2, axis=0), kind="stable")
    for voxel_axis in voxel_axes:
        direction = rotation[:, voxel_axis]
        world_axis = int(np.argmax(np.abs(direction)))
        result[voxel_axis] = (
            pos_axis[world_axis] if direction[world_axis] >= 0 else neg_axis[world_axis]
        )
        # No later voxel axis can pick this world axis.
        rotation[world_axis, :] = 0.0

    return (result[0], result[1], result[2])


def reorientation(
    codes: tuple[str, str, str], target: str | Sequence[str]
) -> tuple[tuple[int, int, int], tuple[bool, bool, bool]]:
    """Returns the axis permutation and flips that turn orientation `codes` to `target`.

    Args:
        codes: The current orientation, such as ("L", "P", "S").
        target: The wanted orientation, such as "RAS".

    Returns:
        The permutation (new axis j is old axis order[j]) and, for each axis after the
        permutation, whether it must be flipped.

    Raises:
        ValueError: If `target` is not one of R/L, A/P and S/I each.
    """
    wanted = tuple(target)
    if (
        len(wanted) != 3
        or any(c not in AXIS_OF_CODE for c in wanted)
        or {AXIS_OF_CODE[c] for c in wanted} != {0, 1, 2}
    ):
        raise ValueError(
            f"Invalid orientation {target!r}, expected one of R/L, A/P and S/I each, "
            "such as 'RAS'."
        )

    # New axis j is the old axis that points along the same world axis as wanted[j].
    order = tuple(
        next(i for i in range(3) if AXIS_OF_CODE[codes[i]] == AXIS_OF_CODE[c])
        for c in wanted
    )
    # After the permutation, axis j has the letter codes[order[j]]; flip it if it
    # points the other way.
    flips = tuple(codes[order[j]] != wanted[j] for j in range(3))
    return (order[0], order[1], order[2]), (flips[0], flips[1], flips[2])


def permute_affine(affine: np.ndarray, order: tuple[int, int, int]) -> np.ndarray:
    """Returns the affine of a grid whose voxel axes are permuted as `order`.

    New axis j is old axis order[j].
    """
    permuted: np.ndarray = affine[:, [*order, 3]]
    return permuted


def flip_affine(affine: np.ndarray, axis: int, n: int) -> np.ndarray:
    """Returns the affine of a grid flipped along voxel `axis`, which has `n` voxels.

    Every voxel keeps its world position.
    """
    flipped = affine.copy()
    flipped[:3, 3] += flipped[:3, axis] * (n - 1)
    flipped[:, axis] *= -1
    return flipped
