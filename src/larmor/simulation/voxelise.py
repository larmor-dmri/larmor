"""Mapping streamlines onto a voxel grid, by length and direction per bundle."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from larmor.sd.sphere import DIRS_300
from larmor.simulation.bundles import centreline_spline
from larmor.tractography import resample, streamline_lengths
from larmor.types import Grid, Tractogram

# Streamlines are resampled into pieces of this factor * min(voxel_size)
RESAMPLING_FACTOR = 0.1

# The directions that the tangents are sorted by: 300 directions spread evenly over
# the sphere, none opposite another, about 8.6 degrees from their neighbours.
BIN_DIRECTIONS = DIRS_300 / np.linalg.norm(DIRS_300, axis=1, keepdims=True)
NUM_BINS = len(BIN_DIRECTIONS)

# Resampled points handled at a time. Each batch of streamlines is reduced to
# its fibres before the next is read, which bounds the memory however large the
# bundles are.
POINTS_PER_BATCH = 1_000_000

# Directions compared with the bin directions at a time.
DIRS_PER_BLOCK = 16384


@dataclass(frozen=True, eq=False)
class Voxelisation:
    """The streamline length and fibre directions of each bundle in each voxel.

    A bundle has one entry, a fibre, per direction bin it reaches in a voxel.

    Attributes:
        grid: The voxel grid.
        num_bundles: The number of bundles.
        voxel: The voxel of each entry, a C-order index into the grid, (F,) int64.
        bundle: The bundle of each entry, (F,) int64.
        direction: The length-weighted mean of the entry's tangents, a unit
            vector, (F, 3).
        length: The entry's streamline length in mm, (F,).
    """

    grid: Grid
    num_bundles: int
    voxel: np.ndarray
    bundle: np.ndarray
    direction: np.ndarray
    length: np.ndarray

    def bundle_lengths(self) -> np.ndarray:
        """Returns the length of each bundle in each voxel, (V, B) in mm."""
        num_voxels = int(np.prod(self.grid.shape))
        lengths = np.zeros(num_voxels * self.num_bundles)
        np.add.at(lengths, self.voxel * self.num_bundles + self.bundle, self.length)
        return lengths.reshape(num_voxels, self.num_bundles)


@dataclass(frozen=True)
class _FibreSums:
    """Sums over the pieces of each fibre, which add up across batches.

    Attributes:
        key: Each fibre's voxel, bundle and bin, packed by `_group_key`, (F,).
        length: The sum of the pieces' lengths, (F,).
        direction: The sum of the pieces' flipped tangents times their lengths,
            (F, 3).
    """

    key: np.ndarray
    length: np.ndarray
    direction: np.ndarray


def voxelise(
    tractograms: Sequence[Tractogram], grid: Grid, *, n_jobs: int = -1
) -> Voxelisation:
    """Maps the streamlines of each bundle onto the voxels of a grid.

    Each streamline is resampled to have no two points at a distance greater
    than 0.1 * min(voxel_size). Each streamline segment is then mapped onto
    a voxel in the grid, and its direction binned in a histogram of 300
    directions.

    Args:
        tractograms: One tractogram per bundle, in world coordinates (mm).
        grid: The voxel grid.
        n_jobs: Number of threads for resampling, -1 for all cores. Defaults
            to -1.

    Returns:
        The length and direction per voxel, bundle and direction bin.
    """
    step = RESAMPLING_FACTOR * float(np.linalg.norm(grid.affine[:3, :3], axis=0).min())
    num_bundles = len(tractograms)
    batches = [
        _fibre_sums(
            tractogram[start:stop],
            bundle,
            num_bundles,
            grid,
            step,
            n_jobs,
        )
        for bundle, tractogram in enumerate(tractograms)
        for start, stop in _batches(tractogram, step)
    ]
    if not batches:
        return _empty_voxelisation(grid, num_bundles)

    # A fibre can span several batches: add up its sums from each.
    keys, group = np.unique(
        np.concatenate([b.key for b in batches]), return_inverse=True
    )
    length = _sum_by_group(group, np.concatenate([b.length for b in batches]))
    summed = _sum_by_group(group, np.concatenate([b.direction for b in batches]))
    direction = summed / np.linalg.norm(summed, axis=1, keepdims=True)

    # Drop the bin from each fibre's key, keeping its voxel and bundle.
    voxel_bundle = keys // NUM_BINS
    return Voxelisation(
        grid,
        num_bundles,
        voxel_bundle // num_bundles,
        voxel_bundle % num_bundles,
        direction,
        length,
    )


def _batches(tractogram: Tractogram, step: float) -> list[tuple[int, int]]:
    """Splits the streamlines into runs of about POINTS_PER_BATCH resampled points.

    Returns:
        The (start, stop) streamline indices of each batch; a streamline longer
        than a batch gets a batch of its own.
    """
    if tractogram.num_streamlines == 0:
        return []
    points = np.ceil(streamline_lengths(tractogram) / step) + 1
    batch = (np.cumsum(points) - points) // POINTS_PER_BATCH
    stops = np.flatnonzero(np.diff(batch)) + 1
    edges = [0, *stops.tolist(), tractogram.num_streamlines]
    return list(zip(edges[:-1], edges[1:], strict=True))


def _fibre_sums(
    tractogram: Tractogram,
    bundle: int,
    num_bundles: int,
    grid: Grid,
    step: float,
    n_jobs: int,
) -> _FibreSums:
    """Cuts streamlines into pieces and adds up the pieces of each fibre.

    Args:
        tractogram: Streamlines of one bundle.
        bundle: The bundle's number.
        num_bundles: The number of bundles.
        grid: The voxel grid.
        step: The spacing of the pieces, mm.
        n_jobs: Number of threads for resampling, -1 for all cores.

    Returns:
        The sums of each fibre the streamlines reach.
    """
    # Cut the streamlines into short pieces, each with a length and a tangent.
    fine = resample(tractogram, step_size=step, n_jobs=n_jobs)
    points = fine.points.astype(np.float64)
    offsets = fine.offsets

    # which points belong to the same streamline
    real = np.ones(len(points) - 1, dtype=bool)
    real[offsets[1:-1] - 1] = False
    start, end = points[:-1][real], points[1:][real]

    length = np.linalg.norm((end - start), axis=1)
    keep = length > 0
    start, end, length = start[keep], end[keep], length[keep]
    dirs = (end - start) / length[:, None]
    dirs, bins = _bin_directions(dirs)

    aff = np.linalg.inv(grid.affine)
    midpoints = (end + start) / 2
    ijk = midpoints @ aff[:3, :3].T + aff[:3, 3]

    voxel: np.ndarray = np.floor(ijk + 0.5).astype(np.int64)
    inside = np.all((voxel >= 0) & (voxel < np.array(grid.shape)), axis=1)
    voxel, length, dirs, bins = (
        voxel[inside],
        length[inside],
        dirs[inside],
        bins[inside],
    )

    idx = np.ravel_multi_index(voxel.T, grid.shape)

    key = _group_key(idx, bundle, bins, num_bundles)
    keys, group = np.unique(key, return_inverse=True)
    return _FibreSums(
        keys, _sum_by_group(group, length), _sum_by_group(group, dirs * length[:, None])
    )


def _bin_directions(dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns each direction's nearest bin direction, and the direction flipped to it.

    A direction and its opposite are the same fibre direction, so the nearest
    bin direction is the one with the largest absolute cosine, and a direction
    with a negative cosine is flipped to point the same way.

    Returns:
        The flipped (P, 3) direction and the (P,) int64 bin of each.
    """
    nearest = np.empty(len(dirs), dtype=np.int64)
    sign = np.empty(len(dirs))
    for start in range(0, len(dirs), DIRS_PER_BLOCK):
        block = slice(start, start + DIRS_PER_BLOCK)
        cosine = dirs[block] @ BIN_DIRECTIONS.T
        best = np.argmax(np.abs(cosine), axis=1)
        nearest[block] = best
        sign[block] = np.where(cosine[np.arange(len(best)), best] < 0, -1.0, 1.0)
    return dirs * sign[:, None], nearest


def _group_key(
    voxel: np.ndarray, bundle: int, direction_bin: np.ndarray, num_bundles: int
) -> np.ndarray:
    """Packs a voxel, a bundle and a direction bin into one integer per point.

    Two points get the same key exactly when they share all three, so grouping
    by the key groups by all three at once. The voxel and bundle are recovered
    with `key // NUM_BINS`, then `// num_bundles` and `% num_bundles`.

    Returns:
        The key of each point, (P,) int64.
    """
    key: np.ndarray = (voxel * num_bundles + bundle) * NUM_BINS + direction_bin
    return key


def _sum_by_group(group: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Adds up the values of each group; `group` gives each row's group number.

    Returns:
        One sum per group, (G,) for 1D values or (G, C) for (P, C) values.
    """
    if values.ndim == 1:
        return np.bincount(group, weights=values)
    return np.stack([np.bincount(group, weights=c) for c in values.T], axis=1)


def _empty_voxelisation(grid: Grid, num_bundles: int) -> Voxelisation:
    """Returns a voxelisation with no fibres."""
    empty = np.empty(0, dtype=np.int64)
    return Voxelisation(grid, num_bundles, empty, empty, np.empty((0, 3)), np.empty(0))


# The fewest and the most sample points along each axis of a voxel near a tube.
TUBE_SUBDIVISIONS = 5
TUBE_MAX_SUBDIVISIONS = 40
# The widest spacing of the sample points, as a share of the tube's radius.
TUBE_SPACING = 0.25
# The thinnest tube, as a share of the grid's largest voxel size.
MIN_TUBE_RADIUS = 0.05
# Sample points handled at a time when filling a tube.
TUBE_POINTS_PER_BATCH = 1_000_000


def tube_fill(
    control_points: np.ndarray,
    radius: float,
    fan: tuple[float, float, float],
    grid: Grid,
    *,
    subdivisions: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the fraction of each voxel inside a bundle's tube, and its direction.

    The tube follows the centreline that `create_bundle` uses, with the radius
    and fan of `BundleGeometry`, and ends at the planes across the centreline
    through its end points. A voxel's fill is the share of its subdivisions^3
    sample points inside the tube.

    Args:
        control_points: The centreline's control points, (K, 3), in world mm.
        radius: The tube's radius, mm.
        fan: (fan_start, fan_end, fan_sharpness).
        grid: The voxel grid.
        subdivisions: Number of sample points per voxel along each axis.
            Defaults to None, which spaces them at most a quarter of the radius
            apart, from TUBE_SUBDIVISIONS to TUBE_MAX_SUBDIVISIONS.

    Returns:
        The fill of each voxel in C order, in [0, 1], (V,), and the unit tangent
        of the centreline nearest each voxel's centre, zero where the fill is 0,
        (V, 3).

    Raises:
        ValueError: If the radius is not positive or is below MIN_TUBE_RADIUS
            times the largest voxel size, too thin for the samples to measure.
    """
    largest = float(np.linalg.norm(grid.affine[:3, :3], axis=0).max())
    if not radius >= MIN_TUBE_RADIUS * largest:
        raise ValueError(
            f"A tube of radius {radius:g} mm is too thin for voxels of {largest:g} "
            f"mm; its radius must be at least {MIN_TUBE_RADIUS * largest:g} mm."
        )
    if subdivisions is None:
        needed = math.ceil(largest / (TUBE_SPACING * radius))
        subdivisions = min(max(needed, TUBE_SUBDIVISIONS), TUBE_MAX_SUBDIVISIONS)
    num_voxels = int(np.prod(grid.shape))
    fill = np.zeros(num_voxels)
    direction = np.zeros((num_voxels, 3))
    tube = _sample_tube(np.asarray(control_points, dtype=np.float64), radius, fan, grid)

    # Only the voxels whose centre is within reach of the tube are sampled.
    axes = grid.affine[:3, :3]
    indices = np.indices(grid.shape).reshape(3, -1).T.astype(np.float64)
    centres = indices @ axes.T + grid.affine[:3, 3]
    half_diagonal = 0.5 * float(np.abs(axes).sum(axis=1) @ np.ones(3))
    distance, closest = tube.tree.query(
        centres, distance_upper_bound=tube.widest + half_diagonal
    )
    near = np.flatnonzero(np.isfinite(distance))
    direction[near] = tube.tangent[closest[near]]
    fill[near] = _sampled_fill(tube, centres[near], grid, subdivisions)
    direction[fill == 0] = 0.0
    return fill, direction


@dataclass(frozen=True)
class _Tube:
    """Points along a tube's centreline, with the tangent and radius at each."""

    centre: np.ndarray
    tangent: np.ndarray
    reach: np.ndarray
    widest: float
    tree: Any


def _sample_tube(
    control: np.ndarray, radius: float, fan: tuple[float, float, float], grid: Grid
) -> _Tube:
    """Samples a tube's centreline densely, with its tangent and radius there.

    The samples are close enough that the distance to the nearest one is the
    distance to the curve.

    Returns:
        The samples, with a search tree over their positions.
    """
    from scipy.spatial import cKDTree

    # The same centreline as the streamlines', so they stay inside the tube.
    spline, _ = centreline_spline(control)
    voxel_sizes = np.linalg.norm(grid.affine[:3, :3], axis=0)
    fan_start, fan_end, sharpness = fan
    step = min(float(voxel_sizes.min()), radius) / 20
    count = max(int(np.ceil(spline.length / step)) + 1, 50)
    u = np.linspace(0.0, 1.0, count)
    centre = spline(u)
    tangent = spline.derivative()(u)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    reach = radius * (1 + fan_start * (1 - u) ** sharpness + fan_end * u**sharpness)
    widest = radius * (1 + max(fan_start, fan_end))
    return _Tube(centre, tangent, reach, widest, cKDTree(centre))


def _sampled_fill(
    tube: _Tube, centres: np.ndarray, grid: Grid, subdivisions: int
) -> np.ndarray:
    """Returns the share of each voxel's sample points that lie inside the tube.

    Args:
        tube: The sampled tube.
        centres: The world centres of the voxels, (N, 3).
        grid: The voxel grid.
        subdivisions: Number of sample points per voxel along each axis.

    Returns:
        The fill of each voxel, (N,).
    """
    axes = grid.affine[:3, :3]
    # Samples this close to an end plane lie on it.
    tol = 1e-6 * float(np.linalg.norm(axes, axis=0).min())
    offsets = (np.arange(subdivisions) + 0.5) / subdivisions - 0.5
    sub = np.stack(np.meshgrid(offsets, offsets, offsets, indexing="ij"), -1)
    sub = sub.reshape(-1, 3) @ axes.T  # (S, 3) world offsets in a voxel
    last_sample = len(tube.centre) - 1
    fill = np.zeros(len(centres))
    batch = max(1, TUBE_POINTS_PER_BATCH // len(sub))
    for start in range(0, len(centres), batch):
        points = (centres[start : start + batch, None, :] + sub[None]).reshape(-1, 3)
        gap, nearest = tube.tree.query(points)
        inside = (gap <= tube.reach[nearest]).astype(np.float64)
        # Past an end, the nearest centreline point is that end point.
        first, last = nearest == 0, nearest == last_sample
        inside[first] *= _end_weight(
            (points[first] - tube.centre[0]) @ tube.tangent[0], tol
        )
        inside[last] *= _end_weight(
            (tube.centre[-1] - points[last]) @ tube.tangent[-1], tol
        )
        fill[start : start + batch] = inside.reshape(-1, len(sub)).mean(axis=1)
    return fill


def _end_weight(depth: np.ndarray, tol: float) -> np.ndarray:
    """Returns how much samples at `depth` inside an end plane count as inside the tube.

    A sample on the plane counts half, so a tube ending through a layer of samples
    fills that voxel by its true share.

    Returns:
        1 for samples inside the plane, 1/2 on it (within `tol`), else 0.
    """
    return np.where(depth > tol, 1.0, np.where(depth >= -tol, 0.5, 0.0))
