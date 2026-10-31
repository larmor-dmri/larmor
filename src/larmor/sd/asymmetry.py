"""Asymmetric FOD estimation."""

import numpy as np

from larmor.sd.sh import build_basis_matrix, sh_layout
from larmor.sd.sphere import get_dirs, get_theta_phi

N_DIRS = 300
CHUNK_SIZE = 50_000


def _sphere_bases(
    lmax: int, n_half: int, basis: str = "tournier"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes direction on the full sphere and returns even and full bases."""
    dirs = get_dirs(n_half)
    dirs = np.vstack([dirs, -dirs])
    theta, phi = get_theta_phi(dirs)
    b_even = build_basis_matrix(theta, phi, lmax, basis, full=False)
    b_full = build_basis_matrix(theta, phi, lmax, basis, full=True)
    return dirs, b_even, b_full


def _offset_weights(
    dirs: np.ndarray, affine: np.ndarray, sigma: float, sharpness: float
) -> tuple[np.ndarray, np.ndarray]:
    """Returns offsets for each voxel and their relative weight for sigma, sharpness."""
    offsets = np.transpose(np.indices((3, 3, 3)) - 1).reshape(-1, 3)
    offsets = np.delete(offsets, 13, axis=0)
    weights = []
    for off in offsets:
        off_world = off @ affine[:3, :3].T
        length = np.linalg.norm(off_world)
        direction = off_world / length
        spatial_w = np.exp(-(length**2) / (2 * sigma**2))
        angular_w = np.maximum(0, dirs @ direction) ** sharpness
        weights.append(spatial_w * angular_w)

    return offsets, np.asarray(weights, dtype=np.float32)


def _neighbour_matrices(
    weights: np.ndarray, b_even: np.ndarray, b_full: np.ndarray
) -> np.ndarray:
    """Creates a matrix for each neighbor mapping a coefficient in b_even to b_full."""
    as_full = np.linalg.pinv(b_full)
    all_weights = np.vstack([np.ones(weights.shape[1]), weights])
    total = np.sum(all_weights, axis=0)
    matrices = []
    for i in range(27):
        scaled = b_even * (all_weights[i] / total)[:, None]
        matrices.append(np.array(as_full @ scaled, dtype=np.float32))

    return np.vstack([m.T for m in matrices], dtype=np.float32)


def _get_full_neighbourhoods(mask: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Computes the voxels where all neighbours are available."""
    padded_mask = np.pad(mask, 1)
    full = mask.copy()
    X, Y, Z = mask.shape

    for offset in offsets:
        a, b, c = offset
        moved = padded_mask[1 + a : X + 1 + a, 1 + b : Y + 1 + b, 1 + c : Z + 1 + c]
        full = full & moved

    return full


def _filter_full_voxels(
    fod: np.ndarray, voxels: np.ndarray, offsets: np.ndarray, matrices: np.ndarray
) -> np.ndarray:
    """Applies AFOD filtering for fully connected voxels."""
    pieces = [fod[tuple(voxels.T)]]

    for offset in offsets:
        pieces.append(fod[tuple((voxels + offset).T)])

    rows = np.hstack(pieces)
    result: np.ndarray = rows @ matrices
    return result


def _filter_border_voxels(
    fod: np.ndarray,
    mask: np.ndarray,
    voxels: np.ndarray,
    offsets: np.ndarray,
    weights: np.ndarray,
    b_even: np.ndarray,
    b_full: np.ndarray,
) -> np.ndarray:
    """Applies AFOD filtering for non-fully connected voxels."""
    num = fod[tuple(voxels.T)] @ b_even.T
    den = np.ones_like(num)

    for offset, weight in zip(offsets, weights, strict=True):
        near = voxels + offset
        on_grid = ((near >= 0) & (near < mask.shape)).all(axis=1)
        near = np.clip(near, (0, 0, 0), np.asarray(mask.shape) - 1)
        near = tuple(near.T)
        valid = on_grid & mask[near]
        amps = fod[near] @ b_even.T
        num += valid[:, None] * weight * amps
        den += valid[:, None] * weight

    result: np.ndarray = (num / den) @ np.linalg.pinv(b_full).T
    return result


def filter_afod(
    fod: np.ndarray,
    mask: np.ndarray | None,
    affine: np.ndarray,
    sigma: float = 1.25,
    sharpness: float = 2.0,
    basis: str = "tournier",
) -> np.ndarray:
    """Computes asymmetric FODs starting from symmetric ones.

    Each half of an FOD lobe gets compared with the neighbouring voxels it points to.
    If they have fibres in the same direction, that half is kept; otherwise it gets
    shrunk. Where fibres continue on both sides, the FOD stays as it is.

    Args:
        fod: Symmetric FODs.
        mask: Voxels to filter; neighbours outside it are not used. None uses
            every voxel whose FOD is not all zero.
        affine: The image's affine, which gives the neighbours' distances and
            directions in world coordinates.
        sigma: how much the neighbours count compared with the voxel itself,
            larger values give stronger asymmetry. Defaults to 1.25 (mm).
        sharpness: angular decay (the higher the value, the narrower
            the cone in which two peaks are considered similar). Defaults to 2.0.
        basis: SH basis of the input and the output, "tournier", "descoteaux"
            or "leemans". Defaults to "tournier".

    Returns:
        Asymmetric FODs, dtype float32, with 0 where outside the mask.

    Raises:
        ValueError: If mask and fod grid don't match.

    References:
        Poirier C, Descoteaux M. A unified filtering method for estimating
        asymmetric orientation distribution functions. NeuroImage 287 (2024)
        120516.

    """
    if mask is not None:
        mask = np.array(mask, dtype=np.bool)
    else:
        mask = np.any(fod != 0, axis=-1)

    if fod.shape[:3] != mask.shape:
        raise ValueError(
            "Expected fod and mask to be of same shape but "
            f"got {fod.shape} and {mask.shape}"
        )

    lmax, full = sh_layout(fod.shape[-1])
    n_coeffs = (lmax + 1) ** 2

    if full:
        return fod

    dirs, b_even, b_full = _sphere_bases(lmax, N_DIRS, basis)
    offsets, weights = _offset_weights(dirs, affine, sigma, sharpness)
    matrices = _neighbour_matrices(weights, b_even, b_full)

    full_mask = _get_full_neighbourhoods(
        mask,
        offsets,
    )
    border_mask = mask & ~full_mask
    out = np.zeros((*fod.shape[:3], n_coeffs), dtype=np.float32)

    full_voxels = np.argwhere(full_mask)
    border_voxels = np.argwhere(border_mask)
    for i in range(0, len(full_voxels), CHUNK_SIZE):
        chunk = full_voxels[i : i + CHUNK_SIZE]
        out[tuple(chunk.T)] = _filter_full_voxels(fod, chunk, offsets, matrices)

    out[border_mask] = _filter_border_voxels(
        fod, mask, border_voxels, offsets, weights, b_even, b_full
    )
    return out
