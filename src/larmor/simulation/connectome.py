"""The ground-truth connectome of a phantom, from where its bundles end."""

import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from larmor.simulation.phantom import VoxelPartition
from larmor.types import Grid, Tractogram, Volume


@dataclass(frozen=True, eq=False)
class GroundTruthConnectome:
    """The regions the bundles of a phantom connect, and how strongly.

    Each bundle has two ends, the voxels its streamlines start in and the
    voxels they stop in, inside the grid: a streamline that leaves the grid ends
    where it leaves. Ends of different bundles that share a voxel are one
    node, so bundles that start at the same place connect the same region. The
    nodes are numbered 1 to N in the order of the bundles and their ends.

    Attributes:
        parcellation: The node of each voxel, 0 where no bundle ends, on the
            phantom's grid. `lar connectome` with this image (and its default
            radial search) assigns tracked streamlines to the same nodes.
        node_names: One name per node, the ends it holds joined by "+", such as
            "CST_start+AF_end".
        connected: (N, N) symmetric, 1 where a bundle joins the two nodes, else 0.
        proportion: (N, N) symmetric, the share of the bundles' total
            cross-section that joins the two nodes; the upper triangle, diagonal
            included, sums to 1.
        bundle_nodes: (B, 2) int64, the nodes of each bundle's start and end; 0
            for both ends of a bundle with no point inside the grid.
        cross_section: (B,) each bundle's mean cross-section in mm^2, the
            volume it fills divided by the mean length of its streamlines.
    """

    parcellation: Volume
    node_names: tuple[str, ...]
    connected: np.ndarray
    proportion: np.ndarray
    bundle_nodes: np.ndarray
    cross_section: np.ndarray


def ground_truth_connectome(
    partition: VoxelPartition, *, bundle_names: Sequence[str] | None = None
) -> GroundTruthConnectome:
    """Builds the connectome the bundles of a phantom define.

    A bundle's streamlines are first turned to run the same way, so that its
    start is where most of them start. A bundle whose two ends share a node
    connects that node to itself, on the diagonal, with a warning. A bundle
    entirely outside the grid connects nothing, with a warning.

    Args:
        partition: The voxel partition of a simulation, such as
            `SimulatedDWI.partition`.
        bundle_names: A name for each bundle, used in the node names. Defaults
            to None; a missing or empty name becomes "bundle_<number>".

    Returns:
        The connectome.

    Raises:
        ValueError: If bundle_names does not name every bundle.
    """
    grid = partition.grid
    count = len(partition.streamlines)
    if bundle_names is not None and len(bundle_names) != count:
        raise ValueError(
            f"bundle_names must name {count} bundles, got {len(bundle_names)}."
        )
    names = [
        _node_name(bundle_names[k] if bundle_names is not None else "", k)
        for k in range(count)
    ]
    # Ends 2k and 2k + 1 are the start and the end of bundle k.
    regions = []
    for tractogram in partition.streamlines:
        starts, stops = _oriented_ends(*_ends_in_grid(tractogram, grid))
        regions += [_voxels(starts, grid), _voxels(stops, grid)]
    root = _merge_overlapping(regions, int(np.prod(grid.shape)))

    node_of_root: dict[int, int] = {}
    end_node = np.zeros(2 * count, dtype=np.int64)
    end_names: dict[int, list[str]] = {}
    for end, region in enumerate(regions):
        if len(region) == 0:
            continue
        node = node_of_root.setdefault(root[end], len(node_of_root) + 1)
        end_node[end] = node
        side = "start" if end % 2 == 0 else "end"
        end_names.setdefault(node, []).append(f"{names[end // 2]}_{side}")
    labels = np.zeros(int(np.prod(grid.shape)), dtype=np.int32)
    for end, region in enumerate(regions):
        labels[region] = end_node[end]

    bundle_nodes = end_node.reshape(count, 2)
    cross_section = _cross_sections(partition)
    connected, proportion = _matrices(
        bundle_nodes, cross_section, names, len(node_of_root)
    )
    return GroundTruthConnectome(
        parcellation=Volume(labels.reshape(grid.shape), grid.affine.copy()),
        node_names=tuple("+".join(end_names[n]) for n in sorted(end_names)),
        connected=connected,
        proportion=proportion,
        bundle_nodes=bundle_nodes,
        cross_section=cross_section,
    )


def _matrices(
    bundle_nodes: np.ndarray,
    cross_section: np.ndarray,
    names: Sequence[str],
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the (N, N) connected and proportion matrices of the bundles' nodes.

    Warns about a bundle outside the grid, which is left out, and about one
    whose two ends are one node.
    """
    connected = np.zeros((num_nodes, num_nodes))
    proportion = np.zeros((num_nodes, num_nodes))
    for k, (a, b) in enumerate(bundle_nodes.tolist()):
        if a == 0 or b == 0:
            warnings.warn(
                f"Bundle {names[k]} has no point inside the grid; it connects "
                "nothing in the ground-truth connectome.",
                stacklevel=3,
            )
            continue
        if a == b:
            warnings.warn(
                f"Both ends of bundle {names[k]} are in node {a}; it connects the "
                "node to itself in the ground-truth connectome.",
                stacklevel=3,
            )
        connected[a - 1, b - 1] = connected[b - 1, a - 1] = 1.0
        proportion[a - 1, b - 1] += cross_section[k]
        if a != b:
            proportion[b - 1, a - 1] += cross_section[k]
    total = np.triu(proportion).sum()
    if total > 0:
        proportion /= total
    return connected, proportion


def _node_name(name: str, index: int) -> str:
    """Returns bundle `index`'s name made safe for a connectome file.

    Characters other than letters, digits, "_" and "-" become "_"; an empty
    name becomes "bundle_<index + 1>".
    """
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", name.strip())
    return safe or f"bundle_{index + 1}"


def _ends_in_grid(tractogram: Tractogram, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Returns the (S, 3) first and last point of each streamline inside the grid.

    Streamlines with no point inside the grid are left out.
    """
    points = tractogram.points.astype(np.float64)
    _, inside = _nearest_voxels(points, grid)
    kept = np.flatnonzero(inside)
    if len(kept) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    # The streamline of each kept point; sorted, as the points are.
    owner = np.searchsorted(tractogram.offsets, kept, side="right") - 1
    _, first = np.unique(owner, return_index=True)
    last = np.append(first[1:], len(kept)) - 1
    return points[kept[first]], points[kept[last]]


def _oriented_ends(
    first: np.ndarray, last: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the (S, 3) starts and stops of the streamlines, all run one way.

    `first` and `last` are each streamline's first and last point. A streamline
    is reversed when its ends lie closer to the other end of the bundle. The
    bundle's ends are taken first from one streamline, then from the mean of the
    turned streamlines, so one odd streamline does not decide.
    """
    if len(first) == 0:
        return first, last
    head, tail = first[0], last[0]
    starts, stops = first, last
    for _ in range(2):
        kept = np.linalg.norm(first - head, axis=1) + np.linalg.norm(
            last - tail, axis=1
        )
        turned = np.linalg.norm(first - tail, axis=1) + np.linalg.norm(
            last - head, axis=1
        )
        flip = (turned < kept)[:, None]
        starts = np.where(flip, last, first)
        stops = np.where(flip, first, last)
        head, tail = starts.mean(axis=0), stops.mean(axis=0)
    return starts, stops


def _nearest_voxels(points: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Returns the (N, 3) voxels nearest to the points, and whether each is inside."""
    to_voxel = np.linalg.inv(grid.affine)
    coordinates = points @ to_voxel[:3, :3].T + to_voxel[:3, 3]
    nearest = np.floor(coordinates + 0.5).astype(np.int64)
    inside = np.all((nearest >= 0) & (nearest < np.array(grid.shape)), axis=1)
    return nearest, inside


def _voxels(points: np.ndarray, grid: Grid) -> np.ndarray:
    """Returns the sorted flat indices (C order) of the voxels holding the points.

    Points outside the grid are left out.
    """
    nearest, inside = _nearest_voxels(points, grid)
    flat = np.ravel_multi_index(tuple(nearest[inside].T), grid.shape)
    return np.unique(flat)


def _merge_overlapping(regions: list[np.ndarray], num_voxels: int) -> list[int]:
    """Returns, for each region, the first region joined to it through shared voxels."""
    parent = list(range(len(regions)))

    def find(i: int) -> int:
        """Returns the region that stands for region i's group."""
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = np.full(num_voxels, -1, dtype=np.int64)
    for index, region in enumerate(regions):
        for other in np.unique(owner[region]).tolist():
            if other >= 0:
                low, high = sorted((find(index), find(other)))
                parent[high] = low
        owner[region[owner[region] < 0]] = index
    return [find(i) for i in range(len(regions))]


def _cross_sections(partition: VoxelPartition) -> np.ndarray:
    """Returns each bundle's filled volume over its mean streamline length, mm^2."""
    voxel_volume = abs(float(np.linalg.det(partition.grid.affine[:3, :3])))
    volumes = partition.bundle_fractions.sum(axis=(0, 1, 2)) * voxel_volume
    lengths = np.array([_mean_length(t) for t in partition.streamlines])
    areas: np.ndarray = np.divide(
        volumes, lengths, out=np.zeros_like(volumes), where=lengths > 0
    )
    return areas


def _mean_length(tractogram: Tractogram) -> float:
    """Returns the mean length of the streamlines in mm, 0 if there are none."""
    if tractogram.num_streamlines == 0:
        return 0.0
    steps = np.linalg.norm(np.diff(tractogram.points, axis=0), axis=1)
    # A step from one streamline's last point to the next one's first is not a step.
    within = np.ones(len(steps), dtype=bool)
    joins = tractogram.offsets[1:-1] - 1
    within[joins[(joins >= 0) & (joins < len(steps))]] = False
    return float(steps[within].sum()) / tractogram.num_streamlines
