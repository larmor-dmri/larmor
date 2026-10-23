"""Checks the inputs of the DWI fits, and computes the mean b0 and S0."""

import numpy as np
from scipy.ndimage import uniform_filter

from larmor._log import LarmorLogger
from larmor.types import GradientTable, Volume

# Neighbourhood widths searched, in order, for the S0 of a voxel whose own b0s are
# implausible.
S0_NEIGHBOURHOODS = (3, 5)
# Fraction of S0 that replaces a measurement that is not finite and positive.
SIGNAL_FLOOR = 1e-3


def check_dwi(volume: Volume, gtab: GradientTable) -> None:
    """Checks that `gtab` describes the volumes of `volume`.

    Raises:
        ValueError: If the volume is not 4D or the table has another number of
            entries than the volume has volumes.
    """
    if volume.data.ndim != 4:
        raise ValueError(
            f"A DWI is 4D, (X, Y, Z, N); got a volume of shape {volume.data.shape}."
        )
    if len(gtab.bvals) != volume.data.shape[3]:
        raise ValueError(
            f"The gradient table has {len(gtab.bvals)} entries, the volume has "
            f"{volume.data.shape[3]} volumes."
        )


def as_mask(volume: Volume, mask: Volume | np.ndarray) -> np.ndarray:
    """Returns `mask` as booleans (nonzero inside), checked against `volume`'s grid.

    A mask `Volume` must be on the same grid as `volume`. A mask stored in another
    orientation of that grid (axes permuted or flipped) is reoriented to the
    volume's, with a warning, so it selects the same voxels in world space.

    Args:
        volume: The volume the mask selects voxels of.
        mask: A mask volume, or an array of the volume's spatial shape.

    Raises:
        ValueError: If the mask is on another grid than the volume, or an array
            of another shape.
    """
    shape = volume.data.shape[:3]
    if isinstance(mask, Volume):
        same = mask.data.shape[:3] == shape and np.allclose(
            mask.affine, volume.affine, rtol=0, atol=1e-3
        )
        if not same and not mask.grid.matches(volume.grid):
            raise ValueError("The mask is not on the volume's grid.")
        if not same:
            LarmorLogger.warning(
                f"The mask is stored {''.join(mask.orientation)}, the volume "
                f"{''.join(volume.orientation)}; reorienting the mask to the volume's."
            )
            mask = mask.with_data(mask.data)
            mask.to_orientation(volume.orientation)
        data = mask.data
    else:
        data = np.asarray(mask)
    if data.shape != shape:
        raise ValueError(
            f"The mask has shape {data.shape}, the volume's grid is {shape}."
        )
    inside: np.ndarray = data != 0
    return inside


def estimate_s0(volume: Volume, gtab: GradientTable, mask: np.ndarray) -> np.ndarray:
    """Returns the signal without diffusion weighting, S0, of every voxel of a mask.

    S0 is the mean of a voxel's b0 measurements. Diffusion weighting can only lower
    the signal, so a voxel whose mean b0 is at or below the mean of its
    diffusion-weighted measurements has implausible b0s (dark, or 0, as at the edge
    of the brain). Such a voxel takes the mean S0 of the plausible mask voxels in
    its 3x3x3 neighbourhood, or in its 5x5x5 neighbourhood when there is none, and
    NaN when there is none there either.

    Args:
        volume: Diffusion-weighted volume, shape (X, Y, Z, N).
        gtab: Its gradient table; its `b0_mask` selects the b0s.
        mask: Voxels to estimate S0 in, shape (X, Y, Z) bool.

    Returns:
        S0, shape (X, Y, Z), float64; the mean b0 outside the mask.

    Raises:
        ValueError: If the table does not match the volume, or has no b0.
    """
    check_dwi(volume, gtab)
    is_b0 = gtab.b0_mask
    if not np.any(is_b0):
        raise ValueError(
            f"No b0 volume: no b-value is at or below {gtab.b0_threshold} s/mm^2."
        )
    data = volume.as_float()
    # Sums over the last axis need no copy of the 4D data.
    b0_sum = data[..., is_b0].sum(axis=-1, dtype=np.float64)
    # numpy types a sum along an axis as Any; it is a float64 array.
    s0: np.ndarray = b0_sum / is_b0.sum()
    n_dw = int((~is_b0).sum())
    if n_dw == 0:
        return s0
    mean_dw = (data.sum(axis=-1, dtype=np.float64) - b0_sum) / n_dw

    finite = np.isfinite(s0) & np.isfinite(mean_dw)
    plausible = mask & finite & (s0 > mean_dw)
    to_repair = mask & finite & ~plausible
    repaired = np.full(s0.shape, np.nan)
    for width in S0_NEIGHBOURHOODS:
        open_voxels = to_repair & np.isnan(repaired)
        if not open_voxels.any():
            break
        # Window means times the window size give the count and the sum of the
        # plausible neighbours; the count is rounded to whole voxels to drop the
        # filter's rounding errors.
        size = width**3
        count = np.rint(
            uniform_filter(plausible.astype(np.float64), width, mode="constant") * size
        )
        total = uniform_filter(np.where(plausible, s0, 0.0), width, mode="constant")
        found = open_voxels & (count > 0)
        repaired[found] = total[found] * size / count[found]
    s0[to_repair] = repaired[to_repair]
    return s0


def floor_bad_samples(s: np.ndarray, s0: np.ndarray | None) -> None:
    """Sets the measurements that are not finite and positive to 1e-3 S0, in place.

    A voxel without a finite, positive S0 (or every voxel when `s0` is None, as
    for a table without b0) uses 1e-3 times the largest finite value of `s`.

    Args:
        s: The measurements, shape (n_voxels, N).
        s0: The S0 of each voxel, shape (n_voxels,), or None.
    """
    bad = ~(np.isfinite(s) & (s > 0))
    if not bad.any():
        return
    finite = s[np.isfinite(s)]
    largest = float(finite.max()) if finite.size and finite.max() > 0 else 1.0
    floor = np.full(len(s), SIGNAL_FLOOR * largest)
    if s0 is not None:
        usable = np.isfinite(s0) & (s0 > 0)
        floor[usable] = SIGNAL_FLOOR * s0[usable]
    s[bad] = np.broadcast_to(floor[:, None], s.shape)[bad]


def mean_b0(volume: Volume, gtab: GradientTable) -> np.ndarray:
    """Returns the mean of a DWI's b0 volumes.

    Args:
        volume: Diffusion-weighted volume, shape (X, Y, Z, N).
        gtab: Its gradient table; its `b0_mask` selects the b0s.

    Returns:
        The mean b0 volume, shape (X, Y, Z), float32.

    Raises:
        ValueError: If the table does not match the volume, or has no b0.
    """
    check_dwi(volume, gtab)
    is_b0 = gtab.b0_mask
    if not np.any(is_b0):
        raise ValueError(
            f"No b0 volume: no b-value is at or below {gtab.b0_threshold} s/mm^2."
        )
    # numpy types np.mean along an axis as Any; it is the image's float type.
    s0: np.ndarray = np.mean(volume.as_float()[..., is_b0], axis=-1)
    return s0
