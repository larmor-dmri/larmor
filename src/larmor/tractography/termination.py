"""Why a streamline stops, and the automatic FOD amplitude cutoff."""

from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from larmor.sd import find_peaks

# Random seed for drawing the voxels of the automatic cutoff.
_AUTO_CUTOFF_SEED = 1234
# Fraction of the mean largest peak amplitude that is the automatic cutoff.
_AUTO_CUTOFF_FRACTION = 0.2
# Largest number of voxels drawn for the automatic cutoff.
_AUTO_CUTOFF_VOXELS = 5000


class EndStatus(IntEnum):
    """Why one end of a streamline stopped.

    The tracker stores the status of both ends of each streamline in its
    streamline data, as "end_status".
    """

    # The end was not tracked, such as the seed end of a streamline tracked one
    # way only.
    NONE = 0
    # The streamline reached the maximum length.
    MAX_LENGTH = 1
    # Its next step would leave the FOD image; that point is not kept, so the
    # streamline ends at its last point on the image.
    LEFT_VOLUME = 2
    # Its next step would leave the tracking mask; that point is not kept either.
    LEFT_MASK = 3
    # The FOD amplitude fell below the cutoff.
    LOW_FOD = 4
    # The next direction turned more than the maximum angle.
    SHARP_TURN = 5
    # No next direction was found.
    NO_DIRECTION = 6


def auto_cutoff(
    coeffs: NDArray[np.floating],
    *,
    basis: str = "tournier",
    mask: NDArray[np.bool_] | None = None,
) -> float:
    """Returns the automatic FOD amplitude cutoff, a fifth of the mean largest peak.

    The mean is over up to 5000 voxels drawn at random (with a fixed seed) among
    those with a positive DC term, and inside the mask when one is given.

    Args:
        coeffs: The FOD's SH coefficients, (X, Y, Z, C).
        basis: Their SH basis. Defaults to "tournier".
        mask: The voxels to measure in, (X, Y, Z). Defaults to None: every voxel.

    Returns:
        The cutoff.

    Raises:
        ValueError: If no voxel (in the mask) has a positive DC term, or none of
            those drawn has a peak.
    """
    positive = coeffs[..., 0] > 0
    if mask is not None:
        positive &= mask
    candidates = np.flatnonzero(positive)
    if len(candidates) == 0:
        where = "inside the mask " if mask is not None else ""
        raise ValueError(f"No voxel of the FOD {where}has a positive DC term.")
    rng = np.random.default_rng(_AUTO_CUTOFF_SEED)
    n = min(_AUTO_CUTOFF_VOXELS, len(candidates))
    drawn = rng.choice(candidates, size=n, replace=False)
    indices = np.unravel_index(drawn, coeffs.shape[:-1])
    peaks = find_peaks(coeffs[indices], basis=basis, num_peaks=1)
    largest_amp = peaks.amplitudes[:, 0]

    if np.all(np.isnan(largest_amp)):
        raise ValueError("None of the FOD voxels drawn has a peak.")

    return float(_AUTO_CUTOFF_FRACTION * np.nanmean(largest_amp))
