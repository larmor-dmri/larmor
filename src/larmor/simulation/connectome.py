"""The ground-truth connectome of a phantom, from where its bundles end."""

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from larmor._log import LarmorLogger
from larmor.simulation.phantom import VoxelPartition
from larmor.types import Grid, Tractogram, Volume

# The percentile of its endpoints' distances from its centre that is an end's
# radius.
END_RADIUS_PERCENTILE: Final = 90.0


@dataclass(frozen=True, eq=False)
class GroundTruthConnectome:
    """The regions the bundles of a phantom connect, and how strongly.

    Each bundle has two ends, where its streamlines start and where they stop.
    Only points inside the grid count, so a streamline that leaves the grid
    ends where it leaves. Streamlines whose two ends fall in the same voxel are
    left out.

    Ends that lie close together become one node, so bundles drawn from the
    same point share a node, whatever direction they leave in. The centre of an
    end is the mean of its endpoints. The radius of an end is the distance from
    its centre within which 90% of its endpoints lie, and it is at least one
    voxel. Two ends lie close together when their centres are closer than the
    larger of their two radii. If the ends of two different nodes share a
    voxel, the voxel goes to the end with more endpoints in it.

    The nodes are numbered from 1, in the order of the bundles and their ends.
    In the matrices, row and column i stand for node i + 1. A connection has no
    direction, so both matrices are symmetric.

    Attributes:
        parcellation: The node of each voxel, 0 where no bundle ends, on the
            phantom's grid. `lar connectome` with this image (and its default
            radial search) assigns tracked streamlines to the same nodes.
        node_names: The name of each node, made of the ends it holds joined by
            "+", such as "CST_start+AF_end". Bundles that share a name get their
            number added, such as "CST_1" and "CST_3".
        connected: (N, N) whether a bundle joins two nodes. The entry for two
            nodes is 1 if a bundle runs between them, and 0 if none does.
        proportion: (N, N) how large each connection is compared with all of
            them. The entry for two nodes is the cross-section of the bundles
            that join them, divided by the cross-section of all the bundles. It
            is 0 where connected is 0. Counting each connection once, in the
            upper triangle with the diagonal, the entries add up to 1.
        bundle_nodes: (B, 2) the two nodes each bundle joins, numbered 1 to N;
            both 0 for a bundle with no streamline inside the grid. Which of
            the two counts as the start is arbitrary.
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
    connects that node to itself, on the diagonal, with a logged warning. A
    bundle with no streamline inside the grid connects nothing, with a logged
    warning.

    Args:
        partition: The voxel partition of a simulation, such as
            `GroundTruth.partition`.
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
    names = _node_names(bundle_names, count)
    # Ends 2k and 2k + 1 are the start and the end of bundle k, as endpoints.
    ends: list[np.ndarray] = []
    for tractogram in partition.streamlines:
        ends += _oriented_ends(*_ends_in_grid(tractogram, grid))
    # Each end's voxels and the number of endpoints in each.
    regions = [_voxels(points, grid) for points in ends]
    voxel_size = float(np.linalg.norm(grid.affine[:3, :3], axis=0).max())
    root = _merge_near(ends, voxel_size)

    node_of_root: dict[int, int] = {}
    end_node = np.zeros(2 * count, dtype=np.int64)
    end_names: dict[int, list[str]] = {}
    for end, (region, _) in enumerate(regions):
        if len(region) == 0:
            continue
        node = node_of_root.setdefault(root[end], len(node_of_root) + 1)
        end_node[end] = node
        side = "start" if end % 2 == 0 else "end"
        end_names.setdefault(node, []).append(f"{names[end // 2]}_{side}")
    labels = np.zeros(int(np.prod(grid.shape)), dtype=np.int32)
    most = np.zeros(int(np.prod(grid.shape)), dtype=np.int64)
    for end, (region, counts) in enumerate(regions):
        more = counts > most[region]
        labels[region[more]] = end_node[end]
        most[region[more]] = counts[more]

    bundle_nodes = end_node.reshape(count, 2)
    cross_section = _cross_sections(partition)
    _warn_about(bundle_nodes, names)
    connected, proportion = connectome_matrices(
        bundle_nodes, cross_section, len(node_of_root)
    )
    return GroundTruthConnectome(
        parcellation=Volume(labels.reshape(grid.shape), grid.affine.copy()),
        node_names=tuple("+".join(end_names[n]) for n in sorted(end_names)),
        connected=connected,
        proportion=proportion,
        bundle_nodes=bundle_nodes,
        cross_section=cross_section,
    )


def connectome_matrices(
    bundle_nodes: np.ndarray, cross_section: np.ndarray, num_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the (N, N) connected and proportion matrices of the bundles' nodes.

    Args:
        bundle_nodes: (B, 2) the nodes of each bundle's ends, 0 for none; a
            bundle with an end in no node is left out.
        cross_section: (B,) each bundle's cross-section.
        num_nodes: N.

    Returns:
        The connected (0/1) and proportion matrices, both symmetric.
    """
    connected = np.zeros((num_nodes, num_nodes))
    proportion = np.zeros((num_nodes, num_nodes))
    for k, (a, b) in enumerate(bundle_nodes.tolist()):
        if a == 0 or b == 0:
            continue
        connected[a - 1, b - 1] = connected[b - 1, a - 1] = 1.0
        proportion[a - 1, b - 1] += cross_section[k]
        if a != b:
            proportion[b - 1, a - 1] += cross_section[k]
    total = np.triu(proportion).sum()
    if total > 0:
        proportion /= total
    return connected, proportion


def _warn_about(bundle_nodes: np.ndarray, names: Sequence[str]) -> None:
    """Logs the bundles that connect nothing, and those that end where they start."""
    for k, (a, b) in enumerate(bundle_nodes.tolist()):
        if a == 0 or b == 0:
            LarmorLogger.warning(
                "Bundle %s has no streamline inside the grid; it connects nothing "
                "in the ground-truth connectome.",
                names[k],
            )
        elif a == b:
            LarmorLogger.warning(
                "Both ends of bundle %s are in node %d; it connects the node to "
                "itself in the ground-truth connectome.",
                names[k],
                a,
            )


def _node_names(bundle_names: Sequence[str] | None, count: int) -> list[str]:
    """Returns a distinct name for each bundle, safe for a connectome file.

    Characters other than letters, digits, "_" and "-" become "_"; a missing or
    empty name becomes "bundle_<number>", and a name that several bundles share
    gets "_<number>" added.
    """
    given = list(bundle_names) if bundle_names is not None else [""] * count
    safe = [
        re.sub(r"[^A-Za-z0-9_-]", "_", name.strip()) or f"bundle_{k + 1}"
        for k, name in enumerate(given)
    ]
    repeated = {name for name, n in Counter(safe).items() if n > 1}
    return [f"{n}_{k + 1}" if n in repeated else n for k, n in enumerate(safe)]


def _ends_in_grid(tractogram: Tractogram, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Returns the (S, 3) first and last point of each streamline inside the grid.

    Streamlines with no point inside the grid, or whose two points are in the
    same voxel, are left out.
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
    starts, stops = points[kept[first]], points[kept[last]]
    apart = np.any(
        _nearest_voxels(starts, grid)[0] != _nearest_voxels(stops, grid)[0], axis=1
    )
    return starts[apart], stops[apart]


def _oriented_ends(
    first: np.ndarray, last: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the (S, 3) starts and stops of the streamlines, all run one way.

    `first` and `last` are each streamline's first and last point. A streamline
    is reversed when its ends lie closer to the other end of the bundle. The
    bundle's ends are taken first from one streamline, then from the mean of the
    turned streamlines, so one odd streamline does not decide. The streamlines
    are kept as they are unless turning them at least halves the radius of the
    ends: a bundle that ends where it starts has one cloud of endpoints, which
    turning would only split in two.
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
    stored = _radius(first) + _radius(last)
    if _radius(starts) + _radius(stops) > 0.5 * stored:
        return first, last
    return starts, stops


def _radius(points: np.ndarray) -> float:
    """Returns the distance from the points' mean within which 90% of them lie."""
    distances = np.linalg.norm(points - points.mean(axis=0), axis=1)
    return float(np.percentile(distances, END_RADIUS_PERCENTILE))


def _nearest_voxels(points: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Returns the (N, 3) voxels nearest to the points, and whether each is inside."""
    to_voxel = np.linalg.inv(grid.affine)
    coordinates = points @ to_voxel[:3, :3].T + to_voxel[:3, 3]
    nearest = np.floor(coordinates + 0.5).astype(np.int64)
    inside = np.all((nearest >= 0) & (nearest < np.array(grid.shape)), axis=1)
    return nearest, inside


def _voxels(points: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Returns the voxels holding the points and the number of points in each.

    The voxels are sorted flat indices (C order); points outside the grid are
    left out.
    """
    nearest, inside = _nearest_voxels(points, grid)
    flat = np.ravel_multi_index(tuple(nearest[inside].T), grid.shape)
    voxels, counts = np.unique(flat, return_counts=True)
    return voxels, counts


def _merge_near(ends: list[np.ndarray], voxel_size: float) -> list[int]:
    """Returns, for each end, the first end joined to it by nearby centres.

    Two ends are joined when their centres are closer than the larger one's
    radius (at least `voxel_size`), and so are ends joined through others. An
    end without points is joined to none.

    Returns:
        One end index per end, the same for the ends of one group.
    """
    parent = list(range(len(ends)))

    def find(i: int) -> int:
        """Returns the end that stands for end i's group."""
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    kept = [i for i, points in enumerate(ends) if len(points)]
    centres = {i: ends[i].mean(axis=0) for i in kept}
    radii = {i: max(_radius(ends[i]), voxel_size) for i in kept}
    for a, i in enumerate(kept):
        for j in kept[a + 1 :]:
            if np.linalg.norm(centres[i] - centres[j]) < max(radii[i], radii[j]):
                low, high = sorted((find(i), find(j)))
                parent[high] = low
    return [find(i) for i in range(len(ends))]


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
