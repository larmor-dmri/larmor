"""Builds the arrays that feed the GPU, with numpy alone."""

from dataclasses import dataclass, replace

import numpy as np

from larmor.viewer.render.colour_space import physical_colours

# Where the slab's bands of segments split, as quantiles of their extent.
BAND_QUANTILES = (0.2, 0.4, 0.6, 0.8, 0.9, 0.97, 0.995)
# Bytes per segment of a sorted copy, two vertices of float32 position and colour
# and the segment's lowest coordinate.
SEGMENT_BYTES = 2 * 24 + 4
# Number of vertices round each point of a tube.
TUBE_SIDES = 8


@dataclass(frozen=True)
class Lines:
    """Holds streamlines as drawn, with a colour per point."""

    points: np.ndarray  # (P, 3) float32, mm
    offsets: np.ndarray  # (S + 1,) int64
    colours: np.ndarray  # (P, 3) float32, 0 to 1
    values: np.ndarray | None = None  # (P,) float32 to colour by, if any

    @property
    def num_segments(self) -> int:
        """Returns the number of segments between consecutive points of a streamline."""
        lengths = np.diff(self.offsets)
        return int(np.sum(np.maximum(lengths - 1, 0)))


@dataclass(frozen=True)
class SortedSegments:
    """Segments grouped in bands by their extent along one axis, each band sorted.

    Within a band, segments are in order of their lowest coordinate along the axis,
    so a slab's segments are one range per band.
    """

    positions: np.ndarray  # (2M, 3) float32, the two ends of each segment
    colours: np.ndarray  # (2M, 3) float32
    lowest: np.ndarray  # (M,) float32, ascending within each band
    bands: tuple[tuple[int, int, float], ...]  # (start, stop, longest extent in mm)

    @property
    def nbytes(self) -> int:
        """Returns the memory the copy takes, in bytes."""
        return self.positions.nbytes + self.colours.nbytes + self.lowest.nbytes


def random_order(n: int, seed: int) -> np.ndarray:
    """Returns a random order of n streamlines; its prefixes are the subsets."""
    return np.random.default_rng(seed).permutation(n)


def drawn_count(total: int, percent: float, cap: int) -> int:
    """Returns how many streamlines a subset of `percent` draws, at most `cap`."""
    return min(round(total * percent / 100), cap, total)


def reduce(
    points: np.ndarray,
    offsets: np.ndarray,
    chosen: np.ndarray,
    n_points: int,
    values: np.ndarray | None = None,
) -> Lines:
    """Returns the chosen streamlines with at most n_points each, evenly spaced.

    With values, one per point of the streamlines, the kept points keep theirs.

    Returns:
        Lines: The reduced lines.
    """
    starts = offsets[chosen]
    lengths = offsets[chosen + 1] - starts
    kept = np.minimum(lengths, n_points)
    new_offsets = np.zeros(len(chosen) + 1, np.int64)
    np.cumsum(kept, out=new_offsets[1:])
    total = int(new_offsets[-1])
    owner = np.repeat(np.arange(len(chosen)), kept)
    j = np.arange(total) - new_offsets[owner]
    # Index j of k kept points maps to round(j * (L - 1) / (k - 1)) of L points.
    scale = (lengths - 1) / np.maximum(kept - 1, 1)
    index = starts[owner] + np.rint(j * scale[owner]).astype(np.int64)
    kept_points = points[index]
    kept_values = None if values is None else values[index].astype(np.float32)
    colours = direction_colours(kept_points, new_offsets)
    return Lines(kept_points, new_offsets, colours, kept_values)


def recoloured(
    lines: Lines, table: np.ndarray | None, low: float = 0.0, high: float = 1.0
) -> Lines:
    """Returns the lines coloured by their values through a colormap over [low, high].

    Without a table, coloured by direction; without values, grey.

    Returns:
        Lines: The lines with their new colours.
    """
    if table is None:
        colours = direction_colours(lines.points, lines.offsets)
    elif lines.values is None:
        colours = np.full(lines.points.shape, 0.5, np.float32)
    else:
        colours = colormap_colours(lines.values, table, low, high)
    return replace(lines, colours=colours)


def direction_colours(points: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Returns each point's absolute unit tangent as RGB, grey where there is none."""
    tangent = np.zeros(points.shape, np.float32)
    if len(points) > 1:
        step = points[1:] - points[:-1]
        tangent[:-1] += step
        tangent[1:] += step
        # Steps from the end of one streamline to the start of the next do not count.
        starts = offsets[1:-1]
        starts = np.unique(starts[(starts > 0) & (starts < len(points))])
        tangent[starts - 1] -= step[starts - 1]
        tangent[starts] -= step[starts - 1]
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    colours = np.full(points.shape, 0.5, np.float32)
    np.divide(np.abs(tangent), norm, out=colours, where=norm > 0)
    return colours


def nan_separated(lines: Lines) -> tuple[np.ndarray, np.ndarray]:
    """Returns positions and colours of one polyline, a NaN after each streamline."""
    lengths = np.diff(lines.offsets)
    n = len(lines.points)
    rows = np.arange(n) + np.repeat(np.arange(len(lengths)), lengths)
    positions = np.full((n + len(lengths), 3), np.nan, np.float32)
    colours = np.zeros((n + len(lengths), 3), np.float32)
    positions[rows] = lines.points
    colours[rows] = lines.colours
    return positions, colours


def prefix_vertices(offsets: np.ndarray, budget: int) -> int:
    """Returns the vertices of nan_separated's polyline in its first streamlines.

    These are as many whole streamlines as fit in budget, at least one.
    """
    # The vertex after each streamline's NaN point.
    ends = offsets[1:] + np.arange(1, len(offsets))
    count = max(int(np.searchsorted(ends, budget, side="right")), 1)
    return int(ends[min(count, len(ends)) - 1]) if len(ends) else 0


@dataclass(frozen=True)
class Tubes:
    """Holds streamlines as tubes, a ring of vertices per point joined by triangles."""

    positions: np.ndarray  # (V, 3) float32, mm
    normals: np.ndarray  # (V, 3) float32, unit
    colours: np.ndarray  # (V, 3) float32, the lines' colours as linear light
    indices: np.ndarray  # (F, 3) uint32, in the order of the streamlines
    ends: np.ndarray  # (S,) int64, the triangles up to the end of each streamline


def tubes(lines: Lines, radius: float, sides: int = TUBE_SIDES) -> Tubes:
    """Builds tubes of `radius` mm around the lines, with rotation-minimising frames.

    The triangles of each streamline follow those of the one before.

    Returns:
        Tubes: The mesh.
    """
    lengths = np.diff(lines.offsets)
    if len(lines.points) == 0:
        empty = np.zeros((0, 3), np.float32)
        ends = np.zeros(len(lengths), np.int64)
        return Tubes(empty, empty, empty, np.zeros((0, 3), np.uint32), ends)
    normal_p, binormal_p = _tube_frames(lines, lengths)
    # (P, sides, 3), filled one side at a time so the temporaries stay one side big.
    count = len(lines.points)
    ring = np.empty((count, sides, 3), np.float32)
    positions = np.empty((count, sides, 3), np.float32)
    for s in range(sides):
        angle = 2 * np.pi * s / sides
        np.multiply(normal_p, np.float32(np.cos(angle)), out=ring[:, s])
        ring[:, s] += np.float32(np.sin(angle)) * binormal_p
        np.multiply(ring[:, s], np.float32(radius), out=positions[:, s])
        positions[:, s] += lines.points
    # Linear light, as a mesh's vertex colours are read; converted before the
    # repeat, once per point rather than per ring vertex.
    colours = np.repeat(physical_colours(lines.colours), sides, axis=0)
    indices = _tube_triangles(lines.offsets, count, sides)
    per_streamline = np.maximum(lengths - 1, 0) * 2 * sides
    return Tubes(
        positions.reshape(-1, 3),
        ring.reshape(-1, 3),
        colours,
        indices,
        np.cumsum(per_streamline),
    )


def _tube_frames(lines: Lines, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns the normal and binormal at each point, float32, in the lines' order."""
    n, longest = len(lengths), int(lengths.max())
    # Streamlines padded to the longest by repeating their last point, so each step
    # along them is one array operation over every streamline.
    owner = np.repeat(np.arange(n), lengths)
    j = np.arange(len(lines.points)) - lines.offsets[owner]
    last = lines.offsets[1:] - 1
    padded = np.empty((n, longest, 3), np.float64)
    padded[:] = lines.points[np.maximum(last, 0)][:, None, :]
    padded[owner, j] = lines.points
    step = np.diff(padded, axis=1)  # (n, longest - 1, 3); 0 past a streamline's end
    tangent = np.zeros_like(padded)
    tangent[:, :-1] += step
    tangent[:, 1:] += step
    tangent = _unit(tangent)
    normal = np.empty_like(padded)
    normal[:, 0] = _unit(np.cross(tangent[:, 0], _least_axis(tangent[:, 0])))
    for k in range(1, longest):
        normal[:, k] = _next_normal(
            normal[:, k - 1], tangent[:, k - 1], tangent[:, k], step[:, k - 1]
        )
    binormal = np.cross(tangent, normal)
    valid = np.arange(longest) < lengths[:, None]
    return normal[valid].astype(np.float32), binormal[valid].astype(np.float32)


def _tube_triangles(offsets: np.ndarray, count: int, sides: int) -> np.ndarray:
    """Returns the triangles, (F, 3) uint32, that join each streamline's rings."""
    # A segment joins point p to p + 1 of the same streamline.
    joined = np.ones(max(count - 1, 0), bool)
    inner = offsets[1:-1]
    joined[inner[(inner > 0) & (inner < count)] - 1] = False
    base = np.flatnonzero(joined).astype(np.uint32) * np.uint32(sides)
    # Per segment and side, two triangles (a, b, c) and (b, d, c), with a and b on
    # this ring, b one side on, and c and d the same on the next ring.
    indices = np.empty((len(base), sides, 6), np.uint32)
    for s in range(sides):
        a = base + np.uint32(s)
        b = base + np.uint32((s + 1) % sides)
        c, d = a + np.uint32(sides), b + np.uint32(sides)
        for k, corner in enumerate((a, b, c, b, d, c)):
            indices[:, s, k] = corner
    return indices.reshape(-1, 3)


def prefix_triangles(ends: np.ndarray, budget: int) -> int:
    """Returns the triangles of the first whole tubes within budget; at least one."""
    count = max(int(np.searchsorted(ends, budget, side="right")), 1)
    return int(ends[min(count, len(ends)) - 1]) if len(ends) else 0


def _unit(vectors: np.ndarray) -> np.ndarray:
    """Returns the vectors scaled to length 1; zero vectors stay zero."""
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    out = np.zeros_like(vectors)
    np.divide(vectors, norm, out=out, where=norm > 0)
    return out


def _least_axis(vectors: np.ndarray) -> np.ndarray:
    """Returns, for each vector, the unit axis it points least along."""
    axes = np.zeros_like(vectors)
    axes[np.arange(len(vectors)), np.argmin(np.abs(vectors), axis=-1)] = 1
    return axes


def _next_normal(
    normal: np.ndarray, tangent: np.ndarray, next_tangent: np.ndarray, step: np.ndarray
) -> np.ndarray:
    """Returns the normal carried one step along.

    It is carried by double reflection (Wang et al. 2008).
    """
    c1 = np.sum(step * step, axis=-1, keepdims=True)
    k1 = np.divide(2, c1, out=np.zeros_like(c1), where=c1 > 0)
    reflected = normal - k1 * np.sum(step * normal, axis=-1, keepdims=True) * step
    t_reflected = tangent - k1 * np.sum(step * tangent, axis=-1, keepdims=True) * step
    v2 = next_tangent - t_reflected
    c2 = np.sum(v2 * v2, axis=-1, keepdims=True)
    k2 = np.divide(2, c2, out=np.zeros_like(c2), where=c2 > 1e-12)
    carried = reflected - k2 * np.sum(v2 * reflected, axis=-1, keepdims=True) * v2
    # Kept at right angles to the tangent; past a streamline's end it is zero.
    carried -= np.sum(carried * next_tangent, axis=-1, keepdims=True) * next_tangent
    carried = _unit(carried)
    lost = np.linalg.norm(carried, axis=-1) == 0
    carried[lost] = normal[lost]
    # A streamline that starts with a repeated point has no first normal, so it is
    # started here, at right angles to the next tangent.
    unset = lost & (np.linalg.norm(normal, axis=-1) == 0)
    carried[unset] = _unit(
        np.cross(next_tangent[unset], _least_axis(next_tangent[unset]))
    )
    return carried


def sorted_segments(lines: Lines, axis: int) -> SortedSegments:
    """Returns the lines' segments in bands by extent, sorted along `axis`."""
    points, n = lines.points, len(lines.points)
    joined = np.ones(max(n - 1, 0), bool)  # point i joins point i + 1
    ends = lines.offsets[1:-1]
    joined[ends[(ends > 0) & (ends < n)] - 1] = False
    first = np.flatnonzero(joined)
    a, b = points[first, axis], points[first + 1, axis]
    lowest = np.minimum(a, b)
    extent = np.abs(b - a)
    # Band edges at quantiles of the extent; the last band holds the longest.
    edges = np.quantile(extent, BAND_QUANTILES) if len(first) else np.zeros(0)
    band = np.searchsorted(edges, extent, side="left").astype(np.int8)
    order = np.argsort(lowest)
    order = order[np.argsort(band[order], kind="stable")]
    first, lowest, extent, band = (
        first[order],
        lowest[order],
        extent[order],
        band[order],
    )
    stops = np.searchsorted(band, np.arange(len(BAND_QUANTILES) + 1), side="right")
    starts = np.concatenate([[0], stops[:-1]])
    bands = tuple(
        (int(i), int(j), float(extent[i:j].max()) if j > i else 0.0)
        for i, j in zip(starts, stops, strict=True)
    )
    positions = np.empty((len(first), 2, 3), np.float32)
    positions[:, 0] = points[first]
    positions[:, 1] = points[first + 1]
    colours = np.empty((len(first), 2, 3), np.float32)
    colours[:, 0] = lines.colours[first]
    colours[:, 1] = lines.colours[first + 1]
    return SortedSegments(
        positions.reshape(-1, 3), colours.reshape(-1, 3), lowest, bands
    )


def slab_ranges(
    segments: SortedSegments, low: float, high: float
) -> list[tuple[int, int]]:
    """Returns, for each band, the range of its segments that can reach [low, high]."""
    ranges = []
    for start, stop, longest in segments.bands:
        lowest = segments.lowest[start:stop]
        # Bounds in the array's own type, as a Python float would make numpy
        # convert the whole array to float64 first.
        bounds = np.array([low - longest, high], lowest.dtype)
        i = np.searchsorted(lowest, bounds[0], side="left")
        j = np.searchsorted(lowest, bounds[1], side="right")
        ranges.append((start + int(i), start + int(j)))
    return ranges


# Voxels along axis 0 scaled at a time when finding the largest peak, so a large
# image is never copied whole.
_AMPLITUDE_CHUNK = 16


def slice_index(
    affine: np.ndarray,
    shape: tuple[int, ...],
    world_axis: int,
    position: tuple[float, float, float] | np.ndarray,
) -> tuple[int, int] | None:
    """Returns the voxel axis closest to a world axis, and the nearest slice on it.

    For an oblique affine, this is the voxel axis that points most along the world
    axis.

    Returns:
        The voxel axis and the slice on it, or None if the slice is outside the
        grid.
    """
    m = np.asarray(affine, np.float64)[:3, :3]
    along = np.abs(m[world_axis, :]) / np.linalg.norm(m, axis=0)
    axis = int(np.argmax(along))
    voxel = np.linalg.solve(m, np.asarray(position, np.float64) - affine[:3, 3])
    index = int(np.floor(voxel[axis] + 0.5))
    return (axis, index) if 0 <= index < shape[axis] else None


def nearest_voxel(
    affine: np.ndarray,
    shape: tuple[int, ...],
    position: tuple[float, float, float] | np.ndarray,
) -> tuple[int, int, int] | None:
    """Returns the voxel whose centre is nearest a world position, None outside."""
    m = np.asarray(affine, np.float64)
    voxel = np.linalg.solve(m[:3, :3], np.asarray(position, np.float64) - m[:3, 3])
    i, j, k = (int(v) for v in np.floor(voxel + 0.5))  # halves up, as slice_index
    if all(0 <= v < n for v, n in zip((i, j, k), shape[:3], strict=True)):
        return i, j, k
    return None


def largest_amplitude(
    peaks: np.ndarray, slope: float = 1.0, intercept: float = 0.0
) -> float:
    """Returns the largest peak amplitude of a peaks image; 0 if it has none."""
    largest = 0.0
    for start in range(0, peaks.shape[0], _AMPLITUDE_CHUNK):
        chunk = _scaled(peaks[start : start + _AMPLITUDE_CHUNK], slope, intercept)
        amplitudes = np.linalg.norm(chunk.reshape(-1, 3), axis=1)
        amplitudes = amplitudes[np.isfinite(amplitudes)]
        if amplitudes.size:
            largest = max(largest, float(amplitudes.max()))
    return largest


def peak_segments(
    peaks: np.ndarray,
    affine: np.ndarray,
    axis: int,
    index: int,
    length: float,
    reference: float | None,
    threshold: float = 0.0,
    slope: float = 1.0,
    intercept: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the peaks of one voxel slice as segments through the voxels' centres.

    peaks is (X, Y, Z, 3P), each peak's world direction scaled by its amplitude;
    the slice is `index` along voxel `axis`. A peak of amplitude `reference` is
    `length` mm long, others in proportion; with no reference, every peak is.
    Peaks that are NaN, zero or below threshold are skipped.

    Returns:
        The end points (2N, 3) in world mm, a segment per pair, and their colours
        (2N, 3), the absolute direction; both float32.
    """
    plane = _scaled(np.take(peaks, index, axis=axis), slope, intercept)
    vectors = plane.reshape(plane.shape[0], plane.shape[1], -1, 3)
    amplitudes = np.linalg.norm(vectors, axis=-1)
    with np.errstate(invalid="ignore"):  # NaN peaks compare False
        keep = (amplitudes > 0) & (amplitudes >= threshold)
    a, b, p = np.nonzero(keep)
    amplitude = amplitudes[a, b, p][:, None]
    unit = vectors[a, b, p] / amplitude
    voxels = np.empty((len(a), 3))
    others = [i for i in range(3) if i != axis]
    voxels[:, axis] = index
    voxels[:, others[0]] = a
    voxels[:, others[1]] = b
    centres = voxels @ affine[:3, :3].T + affine[:3, 3]
    half = length / 2 if reference is None else length / 2 * amplitude / reference
    positions = np.empty((len(a), 2, 3), np.float32)
    positions[:, 0] = centres - half * unit
    positions[:, 1] = centres + half * unit
    colours = np.repeat(np.abs(unit).astype(np.float32)[:, None], 2, axis=1)
    return positions.reshape(-1, 3), colours.reshape(-1, 3)


def outward_halves(
    segments: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns peak segments cut to the half from the voxel's centre along the peak.

    For peaks whose signs matter, a peak and its opposite then each show their
    own side. The segments are from `peak_segments`.

    Returns:
        The halved segments and their colours.
    """
    positions, colours = segments
    halves = positions.reshape(-1, 2, 3).copy()
    halves[:, 0] = (halves[:, 0] + halves[:, 1]) / 2  # the voxel's centre
    return halves.reshape(-1, 3), colours.copy()


def _scaled(values: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Returns stored values as float32, times slope plus intercept."""
    out = np.asarray(values, np.float32)
    if slope != 1.0 or intercept != 0.0:
        out = out * np.float32(slope) + np.float32(intercept)
    return out


@dataclass(frozen=True)
class PlaneCut:
    """Holds the segments where a plane cuts a mesh, each end on a mesh edge.

    End i lies on the edge from vertex `edges[i, 0]` to `edges[i, 1]`, at
    `weights[i]` of the way along it.
    """

    positions: np.ndarray  # (2N, 3) float32, a segment per pair
    edges: np.ndarray  # (2N, 2) int64, the vertices of each end's edge
    weights: np.ndarray  # (2N,) float32, 0 at the edge's first vertex, 1 at its second

    def interpolate(self, values: np.ndarray) -> np.ndarray:
        """Returns values per vertex (V, K) at the segments' ends, (2N, K) float32."""
        w = self.weights[:, None]
        a = np.asarray(values[self.edges[:, 0]], np.float32)
        b = np.asarray(values[self.edges[:, 1]], np.float32)
        out: np.ndarray = (1 - w) * a + w * b
        return out


def plane_cut(
    vertices: np.ndarray, faces: np.ndarray, axis: int, position: float
) -> PlaneCut:
    """Returns the segments where the plane at `position` on world `axis` cuts faces.

    A vertex on the plane counts as above it, so a face crossing the plane gives
    one segment, a face touching it at a vertex or an edge none or one, and an edge
    in the plane is drawn once. Segments of no length are dropped.

    Returns:
        PlaneCut: The segments.
    """
    distance = vertices[:, axis].astype(np.float64) - position
    above = distance >= 0
    corners = above[faces]  # (F, 3)
    count = corners.sum(axis=1)
    crossing = np.flatnonzero((count == 1) | (count == 2))
    face = faces[crossing].astype(np.int64)
    # The corner alone on its side, and the two edges from it that cross the plane.
    lone = np.argmax(corners[crossing] == (count[crossing] == 1)[:, None], axis=1)
    rows = np.arange(len(crossing))
    v0 = face[rows, lone]
    v1 = face[rows, (lone + 1) % 3]
    v2 = face[rows, (lone + 2) % 3]
    starts = np.stack([v0, v0], axis=1).reshape(-1)
    ends = np.stack([v1, v2], axis=1).reshape(-1)
    d0, d1 = distance[starts], distance[ends]
    weights = d0 / (d0 - d1)  # the sides differ, so d0 - d1 is never 0
    p0 = vertices[starts].astype(np.float64)
    points = p0 + weights[:, None] * (vertices[ends] - p0)
    points[:, axis] = position  # exactly on the plane
    pairs = points.reshape(-1, 2, 3)
    kept = np.any(pairs[:, 0] != pairs[:, 1], axis=1).repeat(2)
    return PlaneCut(
        points[kept].astype(np.float32),
        np.stack([starts, ends], axis=1)[kept],
        weights[kept].astype(np.float32),
    )


def colormap_colours(
    values: np.ndarray, table: np.ndarray, low: float, high: float
) -> np.ndarray:
    """Returns values through a colormap table (N, 4) over [low, high], clipped.

    Colours are interpolated between the table's entries; NaN values are grey.

    Returns:
        The colours, (V, 3).
    """
    span = high - low
    v = np.asarray(values, np.float64)
    with np.errstate(invalid="ignore"):
        scaled = (v - low) / span if span else v * 0.0
        t = np.clip(scaled, 0.0, 1.0) * (len(table) - 1)
    finite = np.isfinite(t)
    t = np.where(finite, t, 0.0)
    i = np.minimum(t.astype(np.int64), len(table) - 2)
    f = (t - i)[:, None]
    rgb = np.asarray(table[:, :3], np.float32)
    colours: np.ndarray = ((1 - f) * rgb[i] + f * rgb[i + 1]).astype(np.float32)
    colours[~finite] = 0.5
    return colours


def normal_subset(
    num_vertices: int, count: int, mask: np.ndarray | None = None, seed: int = 0
) -> np.ndarray:
    """Returns a fixed random choice of `count` vertices (of those in mask), ascending.

    The choice is a prefix of one random order, so a larger count adds vertices
    and keeps the ones shown.

    Returns:
        The vertex indices.
    """
    order = random_order(num_vertices, seed)
    if mask is not None:
        order = order[mask[order]]
    return np.sort(order[:count])


def normal_lines(
    points: np.ndarray, normals: np.ndarray, length: float, axis: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Returns segments of `length` mm from points along their normals.

    With an axis, each normal is projected onto the plane across that axis and
    made unit again; one nearly along the axis is left out.

    Returns:
        The positions and colours, (2M, 3) float32 each; the colours are the
        normals' absolute components.
    """
    n = np.asarray(normals, np.float64)
    norm = np.linalg.norm(n, axis=1)
    unit = n / np.where(norm > 0, norm, 1.0)[:, None]
    direction = unit.copy()
    kept = norm > 0
    if axis is not None:
        direction[:, axis] = 0.0
        in_plane = np.linalg.norm(direction, axis=1)
        kept &= in_plane > 1e-3
        direction /= np.where(kept, in_plane, 1.0)[:, None]
    start = np.asarray(points, np.float64)[kept]
    ends = np.stack([start, start + length * direction[kept]], axis=1)
    colours = np.repeat(np.abs(unit[kept])[:, None], 2, axis=1)
    return (
        ends.reshape(-1, 3).astype(np.float32),
        colours.reshape(-1, 3).astype(np.float32),
    )


def sample_cut(
    cut: PlaneCut, spacing: float, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Returns points every `spacing` mm along a cut's lines, and values there.

    The points on each line are centred on it, so a line shorter than the spacing
    gets one at its middle.

    Returns:
        The positions (S, 3) and the per-vertex values interpolated there (S, K),
        float32 each.
    """
    at_ends = cut.interpolate(values)
    ends = _cut_lines(cut)
    if len(ends) == 0:
        empty = np.zeros((0, 3), np.float32)
        return empty, np.zeros((0, *at_ends.shape[1:]), np.float32)
    positions: list[np.ndarray] = []
    sampled: list[np.ndarray] = []
    for line in ends:
        p = cut.positions[line].astype(np.float64)
        arc = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        )
        total = arc[-1]
        if total <= 0:
            continue
        k = int(total // spacing)
        s = (total - k * spacing) / 2 + spacing * np.arange(k + 1)
        s = s[s <= total]
        j = np.clip(np.searchsorted(arc, s, side="right") - 1, 0, len(arc) - 2)
        step = arc[j + 1] - arc[j]
        t = np.divide(s - arc[j], step, out=np.zeros_like(s), where=step > 0)
        positions.append((1 - t)[:, None] * p[j] + t[:, None] * p[j + 1])
        v = at_ends[line].astype(np.float64)
        sampled.append((1 - t)[:, None] * v[j] + t[:, None] * v[j + 1])
    if not positions:
        empty = np.zeros((0, 3), np.float32)
        return empty, np.zeros((0, *at_ends.shape[1:]), np.float32)
    return (
        np.concatenate(positions).astype(np.float32),
        np.concatenate(sampled).astype(np.float32),
    )


def _cut_lines(cut: PlaneCut) -> list[np.ndarray]:
    """Returns the cut's segments joined into lines, as the indices of their ends.

    Segments follow each other where their ends share a mesh edge or a vertex.
    """
    num_ends = len(cut.edges)
    if num_ends == 0:
        return []
    key = np.sort(cut.edges, axis=1)
    # An end at a vertex is keyed by the vertex alone, whichever edge it was cut on.
    for at, side in ((cut.weights == 0, 0), (cut.weights == 1, 1)):
        key[at] = np.stack([cut.edges[at, side], np.full(at.sum(), -1)], axis=1)
    _, ids = np.unique(key, axis=0, return_inverse=True)
    ids = ids.reshape(-1)
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    # Each end's partner is the other end on the same edge, or -1 when it has none
    # (the line stops there) or several (a mesh that is not a manifold).
    partner = np.full(num_ends, -1, np.int64)
    first = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    sizes = np.diff(np.r_[first, num_ends])
    pairs = first[sizes == 2]
    a, b = order[pairs], order[pairs + 1]
    partner[a] = b
    partner[b] = a
    partners = partner.tolist()
    visited = [False] * (num_ends // 2)
    lines: list[np.ndarray] = []
    # Ends with no partner start open lines; every end starts what is left.
    starts = [e for e in range(num_ends) if partners[e] < 0] + list(range(num_ends))
    for start in starts:
        if visited[start // 2]:
            continue
        line = [start]
        end = start
        while True:
            visited[end // 2] = True
            exit_ = end ^ 1  # the segment's other end
            line.append(exit_)
            nxt = partners[exit_]
            if nxt < 0 or visited[nxt // 2]:
                break
            end = nxt
        lines.append(np.asarray(line, np.int64))
    return lines
