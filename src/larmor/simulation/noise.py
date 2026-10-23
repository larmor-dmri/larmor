"""Thermal noise on a simulated magnitude image."""

import math
from collections.abc import Sequence

import numpy as np

from larmor.types import Volume


def add_noise(
    volumes: Sequence[Volume], sigma: float, *, seed: int | None = None
) -> tuple[Volume, ...]:
    """Adds complex Gaussian noise to noise-free signals and takes the magnitude.

    The magnitude is Rician, as for a single coil. The same sigma holds for every
    voxel, volume and echo, so a zero signal gives a mean of sigma sqrt(pi / 2).
    The result depends only on the seed.

    Args:
        volumes: The noise-free signal, for example one volume per echo.
        sigma: The noise standard deviation, in the signal's units, >= 0.
        seed: Seed of the noise. Defaults to None.

    Returns:
        The noisy magnitudes, float32, one per input volume, on its grid.

    Raises:
        ValueError: If sigma is negative or not finite.

    References:
        Gudbjartsson H, Patz S. The Rician distribution of noisy MRI data. Magn
        Reson Med 34 (1995) 910-914.
    """
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError(f"sigma must be finite and >= 0, got {sigma}.")
    rng = np.random.default_rng(seed)
    noisy = []
    for volume in volumes:
        signal = volume.as_float(dtype=np.float64)
        real = signal + rng.normal(0.0, sigma, signal.shape)
        imaginary = rng.normal(0.0, sigma, signal.shape)
        noisy.append(volume.with_data(np.hypot(real, imaginary).astype(np.float32)))
    return tuple(noisy)
