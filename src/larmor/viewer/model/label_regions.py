"""Builds the label regions' tests of streamlines, and their outlines on a slice.

A label region holds sets of a parcellation's labels. In "from A to B" it keeps
the streamlines with one end in set A and the other in set B, and in "stay
within" those whose points all lie in set A's voxels.
"""

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np

from larmor.types import Tractogram, Volume
from larmor.viewer.model.item import ItemId
from larmor.viewer.model.options import LabelMode

# larmor.tractography loads scipy, so it is imported only in the functions that
# need it.

# Points whose voxels are looked up at a time.
CHUNK_POINTS: Final = 1_000_000


@dataclass(frozen=True)
class LabelRegion:
    """The data of a label region.

    Attributes:
        parcellation: The parcellation image item's id; None once it is removed.
        mode: What the region keeps.
        labels_a: Set A's labels.
        labels_b: Set B's labels, used from A to B only.
        end_radius: How far from an end a labelled voxel may be, mm.
    """

    parcellation: ItemId | None
    mode: LabelMode
    labels_a: tuple[int, ...]
    labels_b: tuple[int, ...]
    end_radius: float


class LabelCache:
    """Keeps the labels of the streamlines' ends and points, the slow part of a test.

    They are kept per tractogram and parcellation object, for the few pairs used
    last, so that changing a label set only looks the labels up. It can be used
    from any thread.
    """

    def __init__(self, size: int = 4) -> None:
        """Creates an empty cache that keeps `size` results."""
        self._size = size
        self._kept: OrderedDict[tuple[object, ...], tuple[object, ...]] = OrderedDict()
        self._lock = threading.Lock()

    def end_labels(
        self, tractogram: Tractogram, parcellation: Volume, radius: float
    ) -> np.ndarray:
        """Returns the (S, 2) labels of each streamline's ends, as `endpoint_nodes`."""
        from larmor.tractography import endpoint_nodes

        key = ("ends", id(tractogram), id(parcellation), radius)
        return self._get(
            key,
            (tractogram, parcellation),
            lambda: endpoint_nodes(tractogram, parcellation, radius=radius),
        )

    def point_labels(self, tractogram: Tractogram, parcellation: Volume) -> np.ndarray:
        """Returns the (P,) label of the voxel of each point, 0 outside the image."""
        key = ("points", id(tractogram), id(parcellation))
        return self._get(
            key,
            (tractogram, parcellation),
            lambda: point_labels(tractogram.points, parcellation),
        )

    def _get(
        self,
        key: tuple[object, ...],
        objects: tuple[object, ...],
        make: Callable[[], np.ndarray],
    ) -> np.ndarray:
        """Returns the kept value for key, made and kept if missing.

        The objects are kept with the value, so that their ids are not reused
        while the key names them.
        """
        with self._lock:
            found = self._kept.get(key)
            if found is not None:
                self._kept.move_to_end(key)
                value = found[-1]
                assert isinstance(value, np.ndarray)
                return value
        made = make()
        with self._lock:
            self._kept[key] = (*objects, made)
            while len(self._kept) > self._size:
                self._kept.popitem(last=False)
        return made


@dataclass(frozen=True, eq=False)
class LabelTest:
    """A label region prepared as a test of streamlines.

    Attributes:
        parcellation: The parcellation.
        region: The region's sets and mode.
        drop: Whether the streamlines it keeps are dropped instead (the exclude
            role).
        cache: Where the labels of the ends and points are kept.
    """

    parcellation: Volume
    region: LabelRegion
    drop: bool
    cache: LabelCache

    def kept(self, tractogram: Tractogram, ends_only: bool) -> np.ndarray:
        """Returns (S,) bool, True for the streamlines the region keeps.

        `ends_only` makes "stay within" test only the first and last points.
        """
        r = self.region
        if r.mode == LabelMode.FROM_A_TO_B:
            ends = self.cache.end_labels(tractogram, self.parcellation, r.end_radius)
            kept = joins_labels(ends, r.labels_a, r.labels_b)
        else:
            labels = self.cache.point_labels(tractogram, self.parcellation)
            kept = stays_in_labels(labels, tractogram.offsets, r.labels_a, ends_only)
        return ~kept if self.drop else kept


def tests_anything(region: LabelRegion) -> bool:
    """Tells whether a label region has the labels it needs to select streamlines.

    Returns:
        For from A to B, whether A or B holds a label; for stay within, whether
        A does.
    """
    if region.mode == LabelMode.FROM_A_TO_B:
        return bool(region.labels_a or region.labels_b)
    return bool(region.labels_a)


def joins_labels(
    ends: np.ndarray, labels_a: tuple[int, ...], labels_b: tuple[int, ...]
) -> np.ndarray:
    """Tests which streamlines have one end in set A and the other in set B.

    An empty set takes any end, labelled or not.

    Args:
        ends: (S, 2) the labels of the first and last point, 0 for none.
        labels_a: Set A.
        labels_b: Set B.

    Returns:
        (S,) bool.
    """
    top = int(max(ends.max(initial=0), *labels_a, *labels_b, 0))
    in_a = _lookup(labels_a, top)[ends]
    in_b = _lookup(labels_b, top)[ends]
    joined: np.ndarray = (in_a[:, 0] & in_b[:, 1]) | (in_b[:, 0] & in_a[:, 1])
    return joined


def stays_in_labels(
    point_labels: np.ndarray,
    offsets: np.ndarray,
    labels: tuple[int, ...],
    ends_only: bool = False,
) -> np.ndarray:
    """Tests which streamlines have every point in the voxels of some labels.

    Empty streamlines never stay within, as in `stays_within`.

    Args:
        point_labels: (P,) the label of each point, 0 for none.
        offsets: (S + 1,) the streamlines' offsets.
        labels: The labels, none of them 0.
        ends_only: Whether to test only the first and last points.

    Returns:
        (S,) bool.
    """
    top = int(max(point_labels.max(initial=0), *labels, 0))
    lookup = _lookup(labels, top)
    lengths = np.diff(offsets)
    result = np.zeros(len(lengths), dtype=bool)
    non_empty = lengths > 0
    if not non_empty.any():
        return result
    if ends_only:
        firsts = offsets[:-1][non_empty]
        lasts = offsets[1:][non_empty] - 1
        result[non_empty] = lookup[point_labels[firsts]] & lookup[point_labels[lasts]]
    else:
        inside = lookup[point_labels]
        result[non_empty] = np.logical_and.reduceat(inside, offsets[:-1][non_empty])
    return result


def _lookup(labels: tuple[int, ...], top: int) -> np.ndarray:
    """Returns (top + 1,) bool, True at the labels; all True for no labels."""
    if not labels:
        return np.ones(top + 1, dtype=bool)
    lookup = np.zeros(top + 1, dtype=bool)
    lookup[list(labels)] = True
    lookup[0] = False
    return lookup


def point_labels(points: np.ndarray, parcellation: Volume) -> np.ndarray:
    """Returns the label of the voxel nearest to each point.

    The voxel is found as a mask region finds it (halves rounded away from
    zero), so that a set's test agrees with `stays_within` on the set's mask.

    Args:
        points: (P, 3) points, world mm.
        parcellation: The parcellation, 3D or 4D with one volume.

    Returns:
        (P,) the labels, in the smallest unsigned type that holds them; 0
        outside the image.
    """
    from larmor.tractography import parcellation_labels

    labels = parcellation_labels(parcellation)
    dtype = np.min_scalar_type(int(labels.max(initial=0)))
    flat = np.ascontiguousarray(labels.astype(dtype)).ravel()
    shape = labels.shape
    to_voxel = np.linalg.inv(parcellation.affine)
    result = np.zeros(len(points), dtype=dtype)
    for start in range(0, len(points), CHUNK_POINTS):
        chunk = points[start : start + CHUNK_POINTS]
        within = np.ones(len(chunk), dtype=bool)
        index = np.zeros(len(chunk), dtype=np.intp)
        for axis in range(3):
            row = to_voxel[axis]
            coordinates = chunk[:, 0] * row[0]
            coordinates += chunk[:, 1] * row[1]
            coordinates += chunk[:, 2] * row[2]
            coordinates += row[3]
            on_axis = (coordinates > -0.5) & (coordinates < shape[axis] - 0.5)
            within &= on_axis
            np.putmask(coordinates, ~on_axis, 0.0)
            rounded = np.rint(coordinates)
            # rint sends halves to even; inside the image they are all positive,
            # so the ones it rounded down go up instead.
            rounded += coordinates - rounded == 0.5
            index *= shape[axis]
            index += rounded.astype(np.intp)
        result[start : start + len(chunk)] = np.where(within, flat[index], 0)
    return result


def label_at(parcellation: Volume, point: tuple[float, float, float]) -> int:
    """Returns the label of the voxel nearest to a point, 0 outside the image."""
    return int(point_labels(np.array([point], dtype=np.float64), parcellation)[0])


# Outlines on a slice


@dataclass(frozen=True)
class SliceCells:
    """The cells of a slice that lie in a label set, and their outline.

    Attributes:
        cells: (N, 4, 3) the corners of each cell, world mm, in order round it.
        edges: (M, 2, 3) the ends of each outline segment, world mm.
    """

    cells: np.ndarray
    edges: np.ndarray


def slice_cells(
    labels: np.ndarray,
    affine: np.ndarray,
    chosen: tuple[int, ...],
    axis: int,
    coordinate: float,
) -> SliceCells:
    """Returns the cells of a world slice whose nearest voxel holds a chosen label.

    The slice is sampled on a grid of the image's smallest voxel size, through
    the voxel centres when the image's axes are the world's.

    Args:
        labels: (X, Y, Z) the parcellation's labels.
        affine: Its voxel-to-world affine.
        chosen: The labels drawn.
        axis: The world axis across the slice.
        coordinate: The slice's place on that axis, mm.

    Returns:
        The cells and their outline; both empty if none is chosen.
    """
    empty = SliceCells(np.zeros((0, 4, 3)), np.zeros((0, 2, 3)))
    if not chosen:
        return empty
    first, second = (a for a in range(3) if a != axis)
    step = float(np.min(np.linalg.norm(affine[:3, :3], axis=0)))
    corners = np.array(
        [
            [i, j, k]
            for i in (0, labels.shape[0] - 1)
            for j in (0, labels.shape[1] - 1)
            for k in (0, labels.shape[2] - 1)
        ],
        dtype=float,
    )
    world = corners @ affine[:3, :3].T + affine[:3, 3]
    low, high = world.min(axis=0), world.max(axis=0)
    if not low[axis] - step <= coordinate <= high[axis] + step:
        return empty
    counts = [int(round((high[a] - low[a]) / step)) + 1 for a in (first, second)]
    u = low[first] + step * np.arange(counts[0])
    v = low[second] + step * np.arange(counts[1])
    uu, vv = np.meshgrid(u, v, indexing="ij")
    points = np.empty((uu.size, 3))
    points[:, first] = uu.ravel()
    points[:, second] = vv.ravel()
    points[:, axis] = coordinate
    found = point_labels(points, Volume(labels, affine))
    top = int(max(found.max(initial=0), *chosen))
    inside = _lookup(chosen, top)[found].reshape(counts)
    return SliceCells(
        _cells(inside, u, v, step, (first, second, axis), coordinate),
        _edges(inside, u, v, step, (first, second, axis), coordinate),
    )


def _place(
    a: np.ndarray, b: np.ndarray, axes: tuple[int, int, int], coordinate: float
) -> np.ndarray:
    """Returns (N, 3) world points with `a` and `b` on the slice's two axes."""
    points = np.empty((len(a), 3))
    points[:, axes[0]] = a
    points[:, axes[1]] = b
    points[:, axes[2]] = coordinate
    return points


def _cells(
    inside: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    step: float,
    axes: tuple[int, int, int],
    coordinate: float,
) -> np.ndarray:
    """Returns the (N, 4, 3) corners of the cells inside, round each cell."""
    i, j = np.nonzero(inside)
    half = step / 2
    corners = [
        _place(u[i] + du, v[j] + dv, axes, coordinate)
        for du, dv in ((-half, -half), (half, -half), (half, half), (-half, half))
    ]
    return np.stack(corners, axis=1)


def _edges(
    inside: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    step: float,
    axes: tuple[int, int, int],
    coordinate: float,
) -> np.ndarray:
    """Returns the (M, 2, 3) cell sides between a cell inside and one outside."""
    half = step / 2
    padded = np.pad(inside, 1)
    segments = []
    # Sides across u: between cells (i - 1, j) and (i, j) of the padded grid.
    i, j = np.nonzero(padded[1:, 1:-1] != padded[:-1, 1:-1])
    x = u[0] + step * i - half
    y = v[j]
    segments.append(
        np.stack(
            [
                _place(x, y - half, axes, coordinate),
                _place(x, y + half, axes, coordinate),
            ],
            axis=1,
        )
    )
    # Sides across v.
    i, j = np.nonzero(padded[1:-1, 1:] != padded[1:-1, :-1])
    x = u[i]
    y = v[0] + step * j - half
    segments.append(
        np.stack(
            [
                _place(x - half, y, axes, coordinate),
                _place(x + half, y, axes, coordinate),
            ],
            axis=1,
        )
    )
    return np.concatenate(segments)
