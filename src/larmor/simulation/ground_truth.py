"""The ground truth of a simulation: what each voxel holds and where the bundles run."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from larmor.simulation.phantom import Phantom, VoxelPartition
from larmor.types import Grid, Tractogram

# Angle in degrees within which a fibre joins a larger one's ground-truth peak, of
# the same bundle.
PEAK_MERGE_ANGLE: Final = 20.0
# Fraction of the voxel's largest peak below which a ground-truth peak is dropped.
PEAK_RELATIVE_THRESHOLD: Final = 0.1


@dataclass(frozen=True, eq=False)
class GroundTruth:
    """What a simulated phantom truly holds, to compare an analysis of its images with.

    `simulate` returns it next to the images, and `read_ground_truth` reads it
    back from a folder that `save_simulation` wrote. The maps are on the
    phantom's grid, (X, Y, Z) with the axes and affine of `grid`, and hold
    volume fractions from 0 to 1.

    Attributes:
        phantom: The phantom simulated, with a seed on every bundle geometry.
        grid: The phantom's grid.
        tissue_names: The phantom's tissues, in the order of tissue_fractions.
        tissue_fractions: (X, Y, Z, T) the fraction of each voxel each tissue
            fills, without the background.
        background: (X, Y, Z) the fraction no tissue fills.
        bundle_fractions: (X, Y, Z, B) the fraction of each voxel each bundle
            fills; `bundle_fractions[..., k] > 0` is bundle k's mask.
        streamlines: Each bundle's streamlines, in world coordinates (mm).
        peaks: (X, Y, Z, 3 P) the true fibre directions, in the layout
            `lar peaks` writes: each peak's direction scaled by the share of the
            voxel its fibres fill, largest first, NaN where a voxel has fewer
            peaks. A bundle's fibres within 20 degrees of each other make one
            peak; different bundles always make different peaks, however close
            their directions.
        bundle_names: A name for each bundle, or None.
        partition: Each voxel's fibres, which `signal_breakdown` takes; None when
            read from a folder.
    """

    phantom: Phantom
    grid: Grid
    tissue_names: tuple[str, ...]
    tissue_fractions: np.ndarray
    background: np.ndarray
    bundle_fractions: np.ndarray
    streamlines: tuple[Tractogram, ...]
    peaks: np.ndarray
    bundle_names: tuple[str, ...] | None = None
    partition: VoxelPartition | None = None


def build_ground_truth(
    phantom: Phantom,
    partition: VoxelPartition,
    *,
    bundle_names: Sequence[str] | None = None,
) -> GroundTruth:
    """Builds the ground truth of a phantom from its voxel partition.

    Args:
        phantom: The phantom, with a seed on every bundle geometry.
        partition: Its voxel partition, from `partition_voxels`.
        bundle_names: A name for each bundle. Defaults to None.

    Returns:
        The ground truth.

    Raises:
        ValueError: If bundle_names does not name every bundle.
    """
    oriented = np.array([t.is_oriented for t in phantom.tissues], dtype=bool)
    if bundle_names is not None and len(bundle_names) != len(phantom.bundles):
        raise ValueError(
            f"bundle_names must name {len(phantom.bundles)} bundles, got "
            f"{len(bundle_names)}."
        )
    return GroundTruth(
        phantom=phantom,
        grid=partition.grid,
        tissue_names=partition.tissue_names,
        tissue_fractions=partition.tissue_fractions,
        background=partition.background,
        bundle_fractions=partition.bundle_fractions,
        streamlines=partition.streamlines,
        peaks=_peaks(partition, oriented),
        bundle_names=None if bundle_names is None else tuple(bundle_names),
        partition=partition,
    )


def _peaks(partition: VoxelPartition, oriented: np.ndarray) -> np.ndarray:
    """Returns the peak directions scaled by amplitude, (X, Y, Z, 3 P), NaN if none.

    Args:
        partition: The voxel partition.
        oriented: Whether each tissue has a fibre direction, (T,); the fibres of
            the others give no peaks.
    """
    per_voxel: dict[int, list[tuple[np.ndarray, float]]] = {}
    offsets = partition.offsets
    kept = oriented[partition.fibre_tissue]
    for index in np.flatnonzero(np.diff(offsets)):
        span = slice(offsets[index], offsets[index + 1])
        if not kept[span].any():
            continue
        directions = partition.fibre_direction[span][kept[span]]
        weights = partition.fibre_weight[span][kept[span]]
        bundles = partition.fibre_bundle[span][kept[span]]
        found = [
            peak
            for bundle in np.unique(bundles)
            for peak in _voxel_peaks(
                directions[bundles == bundle], weights[bundles == bundle]
            )
        ]
        found.sort(key=lambda peak: -peak[1])
        largest = found[0][1]
        per_voxel[int(index)] = [
            peak for peak in found if peak[1] >= PEAK_RELATIVE_THRESHOLD * largest
        ]
    most = max((len(p) for p in per_voxel.values()), default=1)
    out = np.full((int(np.prod(partition.grid.shape)), 3 * most), np.nan, np.float32)
    for voxel, found in per_voxel.items():
        for k, (direction, amplitude) in enumerate(found):
            out[voxel, 3 * k : 3 * k + 3] = direction * amplitude
    return out.reshape(*partition.grid.shape, 3 * most)


def _voxel_peaks(
    directions: np.ndarray, weights: np.ndarray
) -> list[tuple[np.ndarray, float]]:
    """Returns the peaks of one bundle's fibres in a voxel, as (direction, amplitude).

    A peak's direction is the weighted mean of its fibres, and its amplitude
    their summed weight. The peaks are not yet sorted or thresholded.
    """
    cos_merge = math.cos(math.radians(PEAK_MERGE_ANGLE))
    sums: list[np.ndarray] = []
    amplitudes: list[float] = []
    for i in np.argsort(-weights, kind="stable"):
        d, w = directions[i], float(weights[i])
        if sums:
            axes = np.array([s / np.linalg.norm(s) for s in sums])
            dots = axes @ d
            j = int(np.argmax(np.abs(dots)))
            if abs(dots[j]) >= cos_merge:
                sums[j] = sums[j] + w * d * (1.0 if dots[j] >= 0 else -1.0)
                amplitudes[j] += w
                continue
        sums.append(w * d)
        amplitudes.append(w)
    return [
        (total / np.linalg.norm(total), amplitude)
        for total, amplitude in zip(sums, amplitudes, strict=True)
    ]
