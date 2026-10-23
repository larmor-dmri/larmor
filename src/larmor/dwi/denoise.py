"""Denoising of diffusion-weighted images by MP-PCA."""

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from larmor import _core
from larmor._threads import n_threads
from larmor.dwi.inputs import as_mask
from larmor.types import Volume

# Voxels per thread in each call to the compiled denoising.
VOXELS_PER_THREAD = 256


# A patch width, the same along every axis, or one per axis.
PatchSize = int | np.integer | Sequence[int | np.integer] | None


@dataclass(frozen=True)
class DenoisedDWI:
    """Result of `denoise_mppca`.

    Every map is on the input's grid and zero outside the mask. The volume and
    noise are float32 and num_signal is int64; noise and num_signal have shape
    (X, Y, Z).

    Attributes:
        volume: The denoised volume.
        noise: The noise level sigma of each voxel, in the units of the signal.
        num_signal: The number of signal components P kept in each voxel's patch.
    """

    volume: Volume
    noise: np.ndarray
    num_signal: np.ndarray


def default_patch_size(n_volumes: int) -> int:
    """Returns the smallest odd patch width whose cube is at least n_volumes."""
    width = 1
    while width**3 < n_volumes:
        width += 2
    return width


def _patch_shape(
    patch_size: PatchSize, n_volumes: int, grid: tuple[int, ...]
) -> tuple[int, int, int]:
    """Returns the patch width along each axis, capped at the grid's size."""
    if patch_size is None:
        sizes = [default_patch_size(n_volumes)] * 3
    elif isinstance(patch_size, Sequence):
        sizes = [operator.index(size) for size in patch_size]
    else:
        sizes = [operator.index(patch_size)] * 3
    if len(sizes) != 3 or any(size < 1 or size % 2 == 0 for size in sizes):
        raise ValueError(
            f"patch_size must be one odd positive int or three, got {patch_size}."
        )
    x, y, z = (min(size, length) for size, length in zip(sizes, grid, strict=True))
    return x, y, z


def denoise_mppca(
    volume: Volume,
    mask: Volume | np.ndarray | None = None,
    *,
    patch_size: PatchSize = None,
    n_jobs: int = -1,
    progress: bool = False,
) -> DenoisedDWI:
    """Denoises a diffusion-weighted volume by MP-PCA and estimates its noise level.

    MP-PCA (Marchenko-Pastur principal component analysis) uses the redundancy
    between the volumes of a DWI: in a small neighbourhood, the signal is described
    by a few principal components and the rest is noise.

    Every voxel is denoised in its own patch, centred on it. The result does
    not depend on the number of threads.

    The noise is assumed Gaussian, constant within a patch and uncorrelated, and
    the Rician bias of magnitude images is not corrected.

    Args:
        volume: The diffusion-weighted volume, (X, Y, Z, M), with all
            its volumes (b0s included), as acquired.
        mask: Voxels to denoise (nonzero inside), on the volume's grid. Patches
            still use the voxels outside it. Defaults to None, every voxel.
        patch_size: The patch width in voxels, odd, the same along every axis or
            one per axis; along an axis shorter than that, the whole axis.
            Defaults to None, the smallest odd width w with w^3 >= M (5 for up
            to 125 volumes, 7 for up to 343).
        n_jobs: Number of threads, at most the cores; -1 is all cores. Defaults
            to -1.
        progress: Whether to show a progress bar. Defaults to False.

    Returns:
        DenoisedDWI: The denoised volume, the noise map sigma and the number of
        signal components per voxel.

    Raises:
        ValueError: If the volume is not 4D, the mask does not match its grid,
            patch_size is not odd and positive, n_jobs is 0 or below -1, or the
            data is complex.
        TypeError: If patch_size is not an integer or a sequence of integers.

    References:
        Veraart J, Novikov DS, Christiaens D, Ades-aron B, Sijbers J, Fieremans E.
        Denoising of diffusion MRI using random matrix theory. NeuroImage 142
        (2016) 394-406.
        Tournier JD et al. MRtrix3: A fast, flexible and open software framework
        for medical image processing and visualisation. NeuroImage 202 (2019)
        116137.
    """
    threads = n_threads(n_jobs)
    if volume.data.ndim != 4:
        raise ValueError(
            f"A DWI is 4D, (X, Y, Z, M); got a volume of shape {volume.data.shape}."
        )
    grid = volume.data.shape[:3]
    n_volumes = volume.data.shape[3]
    patch = _patch_shape(patch_size, n_volumes, grid)
    inside = np.ones(grid, dtype=bool) if mask is None else as_mask(volume, mask)

    image = np.ascontiguousarray(volume.as_float(dtype=np.float32))
    voxels = np.flatnonzero(inside).astype(np.int64)
    denoised = np.zeros((math.prod(grid), n_volumes), dtype=np.float32)
    noise = np.zeros(math.prod(grid), dtype=np.float32)
    num_signal = np.zeros(math.prod(grid), dtype=np.int64)
    chunk_voxels = VOXELS_PER_THREAD * threads
    starts = range(0, len(voxels), chunk_voxels)
    for start in tqdm(starts, desc="Denoising voxels", disable=not progress):
        chunk = voxels[start : start + chunk_voxels]
        signals, sigma, rank = _core.denoise_mppca(image, chunk, patch, threads)
        denoised[chunk] = signals
        noise[chunk] = sigma
        num_signal[chunk] = rank
    return DenoisedDWI(
        volume=volume.with_data(denoised.reshape(*grid, n_volumes)),
        noise=noise.reshape(grid),
        num_signal=num_signal.reshape(grid),
    )
