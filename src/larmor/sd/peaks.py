"""Extraction of the peaks of FODs."""

import functools
from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull

from larmor import _core
from larmor._threads import n_threads
from larmor.sd.sh import build_basis_matrix, convert_sh_basis, sh_layout
from larmor.sd.sphere import get_theta_phi

# The largest lmax the compiled search handles.
MAX_LMAX = _core.MAX_LMAX
#: The default relative peak threshold, for FODs fitted as SH coefficients (CSD,
#: MSMT-CSD) and FODs from other software.
PEAK_RELATIVE_THRESHOLD = 0.1
#: The relative peak threshold for the FODs of `SDModel` (GRL and mFOD), whose
#: truncation to SH leaves side lobes where fibres cross.
SD_MODEL_PEAK_RELATIVE_THRESHOLD = 0.2


@functools.cache
def _build_icosphere() -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Builds 642 unit vectors from an icosahedron subdivided three times."""
    t = (1 + sqrt(5)) / 2
    pts = np.asarray(
        [
            [0, 1, t],
            [0, -1, t],
            [0, 1, -t],
            [0, -1, -t],
            [1, t, 0],
            [-1, t, 0],
            [1, -t, 0],
            [-1, -t, 0],
            [t, 0, 1],
            [t, 0, -1],
            [-t, 0, 1],
            [-t, 0, -1],
        ]
    )
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)

    for _ in range(3):
        triangles = ConvexHull(pts).simplices
        edges = set()
        new_pts = []

        for a, b, c in triangles:
            edges.add((min(a, b), max(a, b)))
            edges.add((min(b, c), max(b, c)))
            edges.add((min(a, c), max(a, c)))

        for i, j in edges:
            midpoint = pts[i] + pts[j]
            midpoint = midpoint / np.linalg.norm(midpoint)
            new_pts.append(midpoint)

        pts = np.append(pts, new_pts, axis=0)

    triangles = ConvexHull(pts).simplices
    neighbors: list[set[int]] = [set() for _ in range(642)]

    for a, b, c in triangles:
        neighbors[a].add(b)
        neighbors[a].add(c)
        neighbors[b].add(a)
        neighbors[b].add(c)
        neighbors[c].add(a)
        neighbors[c].add(b)

    indices = []
    for idx, n in enumerate(neighbors):
        indices.append(sorted(n) + [idx] * (6 - len(n)))

    return pts, np.asarray(indices)


def _get_amplitudes(
    coeffs: NDArray[np.floating], lmax: int, full: bool
) -> NDArray[np.floating]:
    """Computes the amplitude of each FOD on each vertex of the icosphere."""
    vertices, _ = _build_icosphere()
    theta, phi = get_theta_phi(vertices)
    B = build_basis_matrix(theta, phi, lmax, full=full)
    result: NDArray[np.float64] = coeffs @ B.T
    return result


@dataclass(frozen=True)
class Peaks:
    """Holds the directions and amplitudes of FOD peaks that `find_peaks` returns.

    Attributes:
        directions: Unit vectors along the peaks, shape (..., P, 3) where P is the
            number of peaks asked for. Sorted by amplitude, largest first. If fewer
            than P peaks are found, the remaining ones are NaN. A symmetric FOD's
            peaks have z >= 0; an asymmetric FOD's keep their sign, since a peak and
            its opposite are two lobes.
        amplitudes: Amplitude of each peak, shape (..., P), NaN where directions is
            NaN.
    """

    directions: NDArray[np.float64]
    amplitudes: NDArray[np.float64]


def find_peaks(
    coeffs: NDArray[np.floating],
    mask: NDArray[np.bool_] | None = None,
    *,
    basis: str = "tournier",
    num_peaks: int | None = None,
    relative_threshold: float = PEAK_RELATIVE_THRESHOLD,
    threshold: float = 0.0,
    chunk_size: int = 10_000,
    n_jobs: int = -1,
) -> Peaks:
    """Extracts the peaks of FODs.

    The maxima of the FOD's amplitude are found by climbing from the vertices of
    an icosphere, and peaks closer than 0.99 in `|cos|` are merged. Maxima with
    an amplitude of 0 or below are dropped, as are those under `threshold` and
    those under `relative_threshold` times the largest peak.

    An asymmetric FOD (all SH degrees, recognised by the number of coefficients) is
    searched the same way, but a peak and its opposite are two peaks, merged only
    when closer than 0.99 in signed cos, and peaks keep their sign. A straight fibre
    then gives two peaks, one per lobe.

    Args:
        coeffs: FOD coefficients, shape (..., C), the even degrees of a symmetric
            FOD or all degrees of an asymmetric one.
        mask: Voxels to search, shape (...); the other voxels are NaN. Defaults to
            None, every voxel.
        basis: SH basis of the FODs. Defaults to "tournier".
        num_peaks: Number of peaks to extract. Defaults to None, 3 for a symmetric
            FOD and 6 for an asymmetric one, where each fibre is two peaks.
        relative_threshold: Fraction of the largest peak below which a peak is
            dropped. Defaults to 0.1. The FODs of `SDModel` (GRL and mFOD) need
            0.2, which their `SDFit.peak_relative_threshold` gives.
        threshold: Amplitude below which a peak is dropped. Defaults to 0.0.
        chunk_size: Number of voxels whose amplitudes on the icosphere are held in
            memory at once. Defaults to 10_000.
        n_jobs: Number of threads, at most the number of cores; -1 uses every
            core. Defaults to -1.

    Returns:
        Peaks: The peaks extracted.

    Raises:
        ValueError: If num_peaks or chunk_size is below 1, if n_jobs is 0 or below
            -1, if relative_threshold is not between 0 and 1, if mask does not have the
            shape of coeffs without its last axis, if the coefficient count fits no
            SH layout or convert_sh_basis refuses the basis, or if lmax is above 20.
    """
    if num_peaks is not None and num_peaks < 1:
        raise ValueError(f"num_peaks must be at least 1, got {num_peaks}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1, got {chunk_size}")
    threads = n_threads(n_jobs)
    if not 0 <= relative_threshold <= 1:
        raise ValueError(
            f"relative_threshold must be between 0 and 1, got {relative_threshold}"
        )

    num_coeffs = coeffs.shape[-1]
    leading = coeffs.shape[:-1]

    if mask is None:
        selected = np.ones(leading, dtype=bool)
    else:
        if mask.shape != leading:
            raise ValueError(
                f"mask has shape {mask.shape}, but the coefficients have "
                f"{leading} voxels"
            )
        selected = mask.astype(bool)

    lmax, full = sh_layout(num_coeffs)
    if num_peaks is None:
        num_peaks = 6 if full else 3
    if lmax > MAX_LMAX:
        raise ValueError(f"lmax must be at most {MAX_LMAX}, got {lmax}")
    voxels = np.ascontiguousarray(
        convert_sh_basis(coeffs[selected], basis, "tournier"), dtype=np.float64
    )
    n_voxels = voxels.shape[0]

    vertices, neighbours = _build_icosphere()
    found_dirs = np.empty((n_voxels, num_peaks, 3))
    found_amps = np.empty((n_voxels, num_peaks))

    neighbours = np.ascontiguousarray(neighbours, dtype=np.int64)
    for start in range(0, n_voxels, chunk_size):
        stop = start + chunk_size
        block = voxels[start:stop]
        amps_sphere = np.ascontiguousarray(
            _get_amplitudes(block, lmax, full), np.float64
        )
        found_dirs[start:stop], found_amps[start:stop] = _core.peaks_chunk(
            block,
            lmax,
            amps_sphere,
            vertices,
            neighbours,
            num_peaks,
            relative_threshold,
            threshold,
            threads,
            full=full,
        )

    directions = np.full((*leading, num_peaks, 3), np.nan)
    amplitudes = np.full((*leading, num_peaks), np.nan)
    directions[selected] = found_dirs
    amplitudes[selected] = found_amps
    return Peaks(directions, amplitudes)
