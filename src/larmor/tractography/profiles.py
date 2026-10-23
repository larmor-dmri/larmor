"""Tract profiles, the values of maps at points along a bundle."""

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from larmor._log import LarmorLogger
from larmor.tractography.resampling import resample
from larmor.tractography.sampling import sample_volume
from larmor.types import Tractogram, Volume

# Maximum number of times the centreline is rebuilt.
MAX_CENTRELINE_PASSES = 20
# Number of points per node spacing along each streamline that are assigned to
# the nodes.
POINTS_PER_NODE = 5
# Points per block of the nearest-node search.
NEAREST_BLOCK = 65536


class Weighting(StrEnum):
    """How the streamlines' values at one profile point are averaged."""

    # Weighted by how likely each streamline is to belong to the bundle there.
    GAUSSIAN = "gaussian"
    # All streamlines weigh the same.
    MEAN = "mean"


@dataclass(frozen=True)
class TractProfile:
    """The profile of a bundle, with one value per map at each of its points.

    Args:
        values: For each map's name, the (N,) float64 profile, from the start of
            the bundle to its end; NaN where no streamline has a finite value.
        core: (N, 3) float64, the bundle's centreline, whose evenly spaced nodes
            are the profile points, in world coordinates (mm).
        reversed: (S,) bool, the streamlines of the bundle that were reversed so
            that all run from the start to the end.
    """

    values: dict[str, np.ndarray]
    core: np.ndarray
    reversed: np.ndarray

    @property
    def num_points(self) -> int:
        """Returns the number of profile points, N."""
        return len(self.core)

    def segments(self, bundle: Tractogram) -> np.ndarray:
        """Returns the profile point each point of the bundle belongs to.

        A point belongs to the node of the centreline (`core`) nearest to it, as in
        the profile.

        Args:
            bundle: The bundle the profile was computed from.

        Returns:
            np.ndarray: (P,) int64, a profile point index from 0 to N - 1 for each
                point of the bundle, in the order of `bundle.points`.

        Raises:
            ValueError: If the bundle does not have as many streamlines as the
                profile was computed from.
        """
        if bundle.num_streamlines != len(self.reversed):
            raise ValueError(
                f"The profile was computed from {len(self.reversed)} streamlines, "
                f"the bundle has {bundle.num_streamlines}."
            )
        return _nearest_node(bundle.points.astype(np.float64), self.core)

    def write_csv(self, path: str | Path) -> None:
        """Writes the profile as a CSV table, one row per profile point.

        The columns are `point` (0 at the start of the bundle), `core_x`, `core_y`,
        `core_z` (the centreline, in mm) and one column per map, named after it.

        Args:
            path: The CSV file to write.
        """
        names = list(self.values)
        with Path(path).open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["point", "core_x", "core_y", "core_z", *names])
            for i in range(self.num_points):
                writer.writerow(
                    [
                        i,
                        *(f"{c:.6g}" for c in self.core[i]),
                        *(f"{self.values[n][i]:.8g}" for n in names),
                    ]
                )


def tract_profile(
    bundle: Tractogram,
    maps: Mapping[str, Volume],
    *,
    num_points: int = 100,
    weighting: Weighting | str = Weighting.GAUSSIAN,
    start: ArrayLike | None = None,
    n_jobs: int = -1,
) -> TractProfile:
    """Computes the profile of maps (FA, MD, ...) along a bundle.

    The profile points are the `num_points` nodes of a centreline of the bundle, and
    every point of every streamline is assigned to its nearest node (Chandio et al.
    2020), so that node n holds the same place along the bundle for every
    streamline. A streamline that covers only part of the bundle contributes only
    to the nodes it reaches. Each map is sampled with trilinear interpolation at
    points a fifth of the node spacing apart along each streamline. The profile at
    a node is the weighted mean of the streamlines' mean values there; points whose
    nearest voxel is outside a map and non-finite values are left out. With a map
    that changes along the bundle, the values of the first and last nodes lie about
    a quarter of the node spacing inward.

    Without `start`, the profile starts at the end of the bundle that is left,
    posterior or inferior (in world coordinates, RAS+) along its main axis, such as
    the brainstem end of a corticospinal tract or the left end of the corpus
    callosum. When the axis's two largest components are within 20% of each other
    in size, a small change in the bundle can move the start to the other end, and a
    warning says so; pass `start` to make every subject's profile start at the same
    end. With `start`, each streamline is reversed when its last point is nearer to
    `start` than its first point.

    With "gaussian" weighting, each streamline's value at a node is weighted by the
    probability that it belongs to the bundle there, under a 3D Gaussian fitted to
    the streamlines' positions at the node (Yeatman et al. 2012). With "mean", all
    streamlines weigh the same.

    Args:
        bundle: The streamlines of one bundle, in world coordinates (mm), each of
            at least 2 points.
        maps: The maps to profile, by name; each a 3D image, or a 4D one with a
            single volume.
        num_points: The number of profile points, N, at least 2. Defaults to 100.
        weighting: "gaussian" or "mean". Defaults to "gaussian".
        start: A point (x, y, z) in mm near the end of the bundle where the
            profile should start. Defaults to None, the rule above.
        n_jobs: Number of threads for the resampling, at most the cores; -1 is all
            cores. Defaults to -1.

    Returns:
        TractProfile: The profile of each map, the centreline and the reversed
            streamlines.

    Raises:
        ValueError: If the bundle is empty or has a streamline of fewer than 2
            points, `num_points` is below 2, `weighting` is unknown, `start` is not
            three numbers, or a map has several volumes.

    References:
        Yeatman JD, Dougherty RF, Myall NJ, Wandell BA, Feldman HM. Tract profiles
        of white matter properties: automating fiber-tract quantification. PLoS
        One 7, e49790 (2012). doi:10.1371/journal.pone.0049790.
        Chandio BQ, Risacher SL, Pestilli F, et al. Bundle analytics, a
        computational framework for investigating the shapes and profiles of brain
        pathways across populations. Sci Rep 10, 17149 (2020).
        doi:10.1038/s41598-020-74054-4.
    """
    weighting = Weighting(weighting)
    if bundle.num_streamlines == 0:
        raise ValueError("The bundle has no streamlines.")
    if (np.diff(bundle.offsets) < 2).any():
        raise ValueError("Every streamline of the bundle needs at least 2 points.")
    if num_points < 2:
        raise ValueError(f"num_points must be at least 2, got {num_points}.")
    for name, volume in maps.items():
        if volume.data.ndim == 4 and volume.data.shape[3] != 1:
            raise ValueError(
                f"Map {name!r} needs to be a 3D image or a 4D one with one volume; "
                f"got shape {volume.data.shape}."
            )
    start_point = None
    if start is not None:
        start_point = np.asarray(start, dtype=np.float64)
        if start_point.shape != (3,):
            raise ValueError(f"start must be a point (x, y, z), got {start!r}.")

    plain = Tractogram(bundle.points, bundle.offsets)
    lines = resample(plain, num_points=num_points, n_jobs=n_jobs)
    positions = lines.points.astype(np.float64).reshape(-1, num_points, 3)
    reversed_ = _orient(positions, start_point)
    positions[reversed_] = positions[reversed_, ::-1]
    centreline = _centreline(positions)

    spacing = _arc_length(centreline) / (num_points - 1)
    dense = (
        resample(plain, step_size=spacing / POINTS_PER_NODE, n_jobs=n_jobs)
        if spacing > 0
        else plain
    )
    nodes = _nearest_node(dense.points.astype(np.float64), centreline)
    sizes = np.diff(dense.offsets)
    cells = np.repeat(np.arange(len(sizes)), sizes) * num_points + nodes
    shape = (len(sizes), num_points)
    if weighting == Weighting.GAUSSIAN:
        points = dense.points.astype(np.float64)
        means = np.stack([_cell_mean(cells, points[:, i], shape) for i in range(3)], 2)
        weights = _gaussian_weights(means)
    else:
        counts = np.bincount(cells, minlength=shape[0] * shape[1]).reshape(shape)
        weights = (counts > 0).astype(np.float64)
    values = {
        name: _weighted_mean(
            _cell_mean(cells, _sample_inside(dense, volume), shape), weights
        )
        for name, volume in maps.items()
    }
    return TractProfile(values, centreline, reversed_)


def _orient(positions: np.ndarray, start: np.ndarray | None) -> np.ndarray:
    """Returns which of the (S, N, 3) resampled streamlines to reverse, (S,) bool.

    Without `start`, the main axis is the leading eigenvector of the sum of the
    outer products of the end-to-end vectors, with its largest component positive,
    and a streamline is reversed when its end-to-end vector points against it.
    """
    first, last = positions[:, 0], positions[:, -1]
    if start is not None:
        to_first = np.linalg.norm(first - start, axis=1)
        to_last = np.linalg.norm(last - start, axis=1)
        reverse: np.ndarray = to_last < to_first
        return reverse
    ends = last - first
    _, vectors = np.linalg.eigh(ends.T @ ends)
    axis = vectors[:, -1]
    axis *= np.sign(axis[np.argmax(np.abs(axis))])
    second, first_size = np.sort(np.abs(axis))[-2:]
    if second >= 0.8 * first_size:
        LarmorLogger.warning(
            "The bundle's main axis (%.2f, %.2f, %.2f) has two components of similar "
            "size, so which end the profile starts at is ambiguous; pass a start "
            "point (--start) to choose it.",
            *axis,
        )
    reverse = ends @ axis < 0
    return reverse


def _arc_length(line: np.ndarray) -> float:
    """Returns the length of an (N, 3) polyline."""
    return float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())


def _even(line: np.ndarray) -> np.ndarray:
    """Returns an (N, 3) polyline resampled linearly to N evenly spaced points."""
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    )
    if arc[-1] == 0.0:
        return line
    targets = np.linspace(0.0, arc[-1], len(line))
    return np.stack([np.interp(targets, arc, line[:, i]) for i in range(3)], 1)


def _nearest_node(points: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Returns the (P,) int64 index of the node nearest to each of the (P, 3) points."""
    nearest = np.empty(len(points), dtype=np.int64)
    squared_nodes = np.einsum("nk,nk->n", nodes, nodes)
    # In blocks, so the (block, N) distances stay small.
    for begin in range(0, len(points), NEAREST_BLOCK):
        block = points[begin : begin + NEAREST_BLOCK]
        # |p - n|^2 without |p|^2, which is the same for every node.
        distances = squared_nodes - 2.0 * block @ nodes.T
        nearest[begin : begin + len(block)] = np.argmin(distances, axis=1)
    return nearest


def _centreline(positions: np.ndarray) -> np.ndarray:
    """Returns the (N, 3) centreline of the (S, N, 3) oriented, resampled streamlines.

    The first centreline is the mean of the streamlines. Each streamline is then
    spread over the nodes between those nearest to its two ends, and the centreline
    is rebuilt as the mean position at each node, until those end nodes no longer
    change. Every centreline is resampled to evenly spaced nodes.
    """
    count, num_points, _ = positions.shape
    line = _even(positions.mean(axis=0))
    nodes = np.arange(num_points)
    rows = np.arange(count)[:, None]
    previous = None
    for _ in range(MAX_CENTRELINE_PASSES):
        first = _nearest_node(positions[:, 0], line)
        last = _nearest_node(positions[:, -1], line)
        ends = np.stack([first, last], axis=1)
        if previous is not None and np.array_equal(ends, previous):
            break
        previous = ends
        # Node k gets the streamline at the fraction (k - first) / (last - first)
        # of its length, for k between its two end nodes.
        lo, hi = np.minimum(first, last)[:, None], np.maximum(first, last)[:, None]
        covered = (nodes >= lo) & (nodes <= hi)
        width = (last - first)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            fraction = np.where(width != 0, (nodes - first[:, None]) / width, 0.5)
        at = np.clip(fraction, 0.0, 1.0) * (num_points - 1)
        below = np.minimum(np.floor(at).astype(np.int64), num_points - 2)
        t = (at - below)[..., None]
        placed = (1 - t) * positions[rows, below] + t * positions[rows, below + 1]
        totals = np.einsum("sn,snk->nk", covered.astype(np.float64), placed)
        reached = covered.sum(axis=0)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            line = _even(np.where(reached > 0, totals / reached, line))
    return line


def _cell_mean(
    cells: np.ndarray, values: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Returns the (S, N) mean of the finite values in each (streamline, node) cell.

    `cells` holds the flat cell index of each value; a cell with no finite value
    gets NaN.
    """
    finite = np.isfinite(values)
    size = shape[0] * shape[1]
    total = np.bincount(cells[finite], weights=values[finite], minlength=size)
    count = np.bincount(cells[finite], minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean: np.ndarray = (total / count).reshape(shape)
    return mean


def _sample_inside(tractogram: Tractogram, volume: Volume) -> np.ndarray:
    """Returns the (P,) values of a map at the points, NaN off the map's grid."""
    values = sample_volume(tractogram, volume).astype(np.float64)
    to_voxel = np.linalg.inv(volume.affine)
    voxel = tractogram.points.astype(np.float64) @ to_voxel[:3, :3].T + to_voxel[:3, 3]
    shape = np.array(volume.data.shape[:3])
    inside = np.all((voxel > -0.5) & (voxel < shape - 0.5), axis=1)
    values[~inside] = np.nan
    return values


def _gaussian_weights(positions: np.ndarray) -> np.ndarray:
    """Returns the (S, N) Gaussian weights of the streamlines' positions (S, N, 3).

    The weight is exp(-d^2 / 2), with d the Mahalanobis distance from the mean
    position under the covariance of the positions; 0 where a position is NaN.
    """
    present = np.isfinite(positions).all(axis=2)
    reached = present.sum(axis=0)
    filled = np.where(present[..., None], positions, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = filled.sum(axis=0) / reached[:, None]
    centred = np.where(present[..., None], positions - mean, 0.0)
    # (N, 3, 3) covariance of the streamlines' positions at each node.
    dof = np.maximum(reached - 1, 1)[:, None, None]
    covariance = np.einsum("snk,snl->nkl", centred, centred) / dof
    inverse = np.linalg.pinv(covariance, hermitian=True)
    squared = np.einsum("snk,nkl,snl->sn", centred, inverse, centred)
    weights: np.ndarray = np.where(present, np.exp(-0.5 * squared), 0.0)
    return weights


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Returns the (N,) weighted mean over streamlines of the finite (S, N) values."""
    finite = np.isfinite(values)
    w = np.where(finite, weights, 0.0)
    total = w.sum(axis=0)
    weighted = np.where(finite, values, 0.0) * w
    # 0 / 0 gives NaN where no value is finite.
    with np.errstate(invalid="ignore"):
        mean: np.ndarray = weighted.sum(axis=0) / total
    return mean
