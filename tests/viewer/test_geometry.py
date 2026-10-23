import numpy as np
import pytest

from larmor.viewer.render.colour_space import physical_colours
from larmor.viewer.render.geometry import (
    Lines,
    PlaneCut,
    colormap_colours,
    direction_colours,
    drawn_count,
    largest_amplitude,
    nan_separated,
    nearest_voxel,
    normal_lines,
    normal_subset,
    outward_halves,
    peak_segments,
    plane_cut,
    prefix_triangles,
    prefix_vertices,
    random_order,
    reduce,
    sample_cut,
    slab_ranges,
    slice_index,
    sorted_segments,
    tubes,
)


def _lines(lengths: list[int], seed: int = 0) -> Lines:
    # Random walks with steps of very different sizes, so every band gets segments.
    rng = np.random.default_rng(seed)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    steps = rng.normal(size=(offsets[-1], 3)) * rng.choice(
        [0.1, 1.0, 10.0], (offsets[-1], 1)
    )
    points = np.empty((offsets[-1], 3), np.float32)
    for s in range(len(lengths)):
        a, b = offsets[s], offsets[s + 1]
        points[a:b] = rng.uniform(-20, 20, 3) + np.cumsum(steps[a:b], axis=0)
    return Lines(points, offsets, direction_colours(points, offsets))


def _segments(lines: Lines) -> np.ndarray:
    pairs = [
        lines.points[i : i + 2]
        for s in range(len(lines.offsets) - 1)
        for i in range(lines.offsets[s], lines.offsets[s + 1] - 1)
    ]
    return np.array(pairs, np.float32).reshape(-1, 2, 3)


def test_the_random_order_is_a_fixed_permutation() -> None:
    order = random_order(1000, 0)
    assert np.array_equal(np.sort(order), np.arange(1000))
    assert np.array_equal(random_order(1000, 0), order)
    assert not np.array_equal(random_order(1000, 1), order)


@pytest.mark.parametrize(
    ("total", "percent", "cap", "count"),
    [
        (1000, 100.0, 100_000, 1000),
        (1000, 25.0, 100_000, 250),
        (1000, 0.0, 100_000, 0),
        (1000, 100.0, 300, 300),
        (1000, 25.0, 300, 250),
        (1000, 50.0, 300, 300),
    ],
)
def test_the_drawn_count_is_the_subset_up_to_the_cap(
    total: int, percent: float, cap: int, count: int
) -> None:
    assert drawn_count(total, percent, cap) == count


def test_reduce_keeps_evenly_spaced_points_and_short_streamlines_whole() -> None:
    lengths = [13, 3, 10]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    # Point i is (i, 0, 0), so a kept point's x is its index.
    points = np.zeros((offsets[-1], 3), np.float32)
    points[:, 0] = np.arange(offsets[-1])
    lines = reduce(points, offsets, np.array([2, 0, 1]), 5)
    assert lines.offsets.tolist() == [0, 5, 10, 13]
    kept = lines.points[:, 0].astype(int).tolist()
    # round(j * (L - 1) / 4): 10 points give 0, 2.25, 4.5, 6.75, 9.
    assert kept[:5] == [16 + j for j in (0, 2, 4, 7, 9)]
    assert kept[5:10] == [0, 3, 6, 9, 12]
    assert kept[10:] == [13, 14, 15]


def test_direction_colours_are_the_absolute_unit_tangent() -> None:
    first = [[0, 0, 0], [-1, 0, 0], [-1, 2, 0], [-1, 2, -3]]
    # The second starts far away: no step across streamlines may count.
    second = [[50, 50, 50], [50, 50, 54]]
    points = np.array([*first, *second, [9, 9, 9]], np.float32)
    offsets = np.array([0, 4, 6, 7])
    tangents = np.array(
        [[-1, 0, 0], [-1, 2, 0], [0, 2, -3], [0, 0, -3], [0, 0, 4], [0, 0, 4]], float
    )
    expected = np.abs(tangents) / np.linalg.norm(tangents, axis=1, keepdims=True)
    colours = direction_colours(points, offsets)
    assert colours.dtype == np.float32
    assert np.allclose(colours[:6], expected, atol=1e-6)
    # A single point has no tangent.
    assert np.array_equal(colours[6], [0.5, 0.5, 0.5])


def test_one_polyline_has_a_nan_point_after_each_streamline() -> None:
    lines = _lines([3, 2, 4])
    positions, colours = nan_separated(lines)
    assert len(positions) == len(lines.points) + 3
    gaps = np.isnan(positions).any(axis=1)
    assert np.flatnonzero(gaps).tolist() == [3, 6, 11]
    assert np.array_equal(positions[~gaps], lines.points)
    assert np.array_equal(colours[~gaps], lines.colours)


# The polyline of streamlines of 3, 2 and 4 points ends them at vertices 4, 7, 12.
@pytest.mark.parametrize(
    ("budget", "vertices"),
    [(0, 4), (3, 4), (4, 4), (6, 4), (7, 7), (11, 7), (12, 12), (100, 12)],
)
def test_the_prefix_holds_the_whole_streamlines_within_the_budget_and_at_least_one(
    budget: int, vertices: int
) -> None:
    lines = _lines([3, 2, 4])
    positions, _ = nan_separated(lines)
    assert prefix_vertices(lines.offsets, budget) == vertices
    assert np.isnan(positions[vertices - 1]).all()


def test_the_prefix_of_no_streamlines_is_empty() -> None:
    assert prefix_vertices(np.array([0]), 10) == 0


def _arcs(lengths: list[int]) -> Lines:
    # Gentle 3D arcs, 1 mm apart, so no tube folds into itself at a bend.
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    t = np.concatenate([np.arange(n, dtype=np.float64) for n in lengths])
    s = np.repeat(np.arange(len(lengths)), lengths)
    points = np.stack([t, 0.1 * t**2 + 5 * s, 0.05 * t**2 - 3 * s], axis=1)
    points = points.astype(np.float32)
    return Lines(points, offsets, direction_colours(points, offsets))


def test_tubes_are_rings_at_the_radius_joined_within_each_streamline() -> None:
    lines = _arcs([5, 2, 7])
    made = tubes(lines, 0.5, sides=8)
    assert len(made.positions) == 14 * 8
    assert made.positions.dtype == made.normals.dtype == np.float32
    assert made.colours.dtype == np.float32
    assert made.indices.dtype == np.uint32
    assert made.ends.tolist() == [4 * 16, 5 * 16, 11 * 16]
    assert len(made.indices) == made.ends[-1]
    centres = np.repeat(lines.points, 8, axis=0)
    # float32 positions within ~10 mm of the origin: ~1e-6 mm.
    assert np.allclose(made.positions - centres, 0.5 * made.normals, atol=1e-5)
    assert np.allclose(np.linalg.norm(made.normals, axis=1), 1, atol=1e-6)
    # Each ring takes its point's colour as linear light, as a mesh reads it.
    linear = physical_colours(lines.colours)
    assert np.array_equal(made.colours, np.repeat(linear, 8, axis=0))
    # Each streamline's triangles use only its own rings.
    starts = np.concatenate([[0], made.ends[:-1]])
    for s, (a, b) in enumerate(zip(starts, made.ends, strict=True)):
        used = made.indices[a:b]
        assert used.min() == lines.offsets[s] * 8
        assert used.max() == lines.offsets[s + 1] * 8 - 1
    # Counter-clockwise seen from outside, so front faces face out.
    p = made.positions[made.indices].astype(np.float64)
    face = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    assert (np.sum(face * made.normals[made.indices[:, 0]], axis=1) > 0).all()


def test_a_streamline_starting_with_a_repeated_point_still_gets_its_tube() -> None:
    arc = _arcs([5])
    points = np.concatenate([arc.points[:1], arc.points])
    offsets = np.array([0, len(points)], np.int64)
    made = tubes(Lines(points, offsets, direction_colours(points, offsets)), 0.5, 8)
    rings = made.positions.reshape(-1, 8, 3)[1:] - points[1:, None, :]
    np.testing.assert_allclose(np.linalg.norm(rings, axis=-1), 0.5, rtol=1e-5)


def test_tube_frames_turn_no_more_than_the_streamline() -> None:
    # A helix: a Frenet frame turns about twice as fast as its tangent; one
    # recomputed from scratch at each point flips where the least axis changes.
    s = np.linspace(0, 4 * np.pi, 80)
    points = np.stack([np.cos(s), np.sin(s), 2 * s], axis=1).astype(np.float32)
    lines = Lines(points, np.array([0, 80]), np.ones_like(points))
    made = tubes(lines, 0.2, sides=8)
    normal = made.normals[::8].astype(np.float64)
    tangent = np.gradient(points.astype(np.float64), axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)

    def turn(v: np.ndarray) -> np.ndarray:
        return np.arccos(np.clip(np.sum(v[1:] * v[:-1], axis=1), -1, 1))

    assert (turn(normal) <= 1.2 * turn(tangent) + 1e-6).all()


# Tubes of 5, 2 and 7 points with 8 sides end at triangles 64, 80 and 176.
@pytest.mark.parametrize(
    ("budget", "triangles"), [(0, 64), (63, 64), (64, 64), (100, 80), (500, 176)]
)
def test_the_tube_prefix_holds_whole_tubes_within_the_budget_and_at_least_one(
    budget: int, triangles: int
) -> None:
    made = tubes(_arcs([5, 2, 7]), 0.5, sides=8)
    assert prefix_triangles(made.ends, budget) == triangles


def test_sorted_segments_hold_each_segment_once_sorted_in_bands_by_extent() -> None:
    lines = _lines([1, *range(2, 40)])
    expected = _segments(lines)
    for axis in range(3):
        copy = sorted_segments(lines, axis)
        pairs = copy.positions.reshape(-1, 2, 3)
        assert sorted(map(bytes, pairs)) == sorted(map(bytes, expected))
        assert np.array_equal(pairs.min(axis=1)[:, axis], copy.lowest)
        extent = np.ptp(pairs[:, :, axis], axis=1)
        starts = [b[0] for b in copy.bands]
        stops = [b[1] for b in copy.bands]
        assert starts == [0, *stops[:-1]] and stops[-1] == len(pairs)
        for start, stop, longest in copy.bands:
            assert stop > start
            assert np.all(np.diff(copy.lowest[start:stop]) >= 0)
            assert extent[start:stop].max() == np.float32(longest)
        # Bands by extent: each one's segments are no longer than the next one's.
        for (_, _, longest), (start, _, _) in zip(
            copy.bands, copy.bands[1:], strict=False
        ):
            assert longest <= extent[start:].min()


def test_slab_ranges_select_every_segment_that_reaches_into_the_slab() -> None:
    lines = _lines(list(range(2, 60)))
    rng = np.random.default_rng(1)
    for axis in range(3):
        copy = sorted_segments(lines, axis)
        pairs = copy.positions.reshape(-1, 2, 3)[:, :, axis]
        low_end, high_end = pairs.min(axis=1), pairs.max(axis=1)
        for centre in rng.uniform(-30, 30, 10):
            low, high = centre - 2.5, centre + 2.5
            ranges = slab_ranges(copy, low, high)
            chosen = np.zeros(len(pairs), bool)
            for (start, stop), (b_start, b_stop, longest) in zip(
                ranges, copy.bands, strict=True
            ):
                assert b_start <= start <= stop <= b_stop
                chosen[start:stop] = True
                # Each band's range is exactly its segments starting between
                # low - longest and high; clipping planes cut what overshoots.
                band = np.arange(b_start, b_stop)
                lowest = copy.lowest[band]
                inside = (lowest >= np.float32(low - longest)) & (
                    lowest <= np.float32(high)
                )
                assert np.array_equal(band[inside], np.arange(start, stop))
            meets = (high_end >= low) & (low_end <= high)
            assert meets.any()
            assert np.all(chosen[meets])
            assert not chosen.all()


# Voxel axis 0 runs along world y, 1 along z, 2 along x, each a different size.
CYCLED = np.array(
    [[0, 0, 4.0, -8], [2.0, 0, 0, 5], [0, -3.0, 0, 1], [0, 0, 0, 1]], np.float64
)


@pytest.mark.parametrize(("world_axis", "voxel_axis"), [(0, 2), (1, 0), (2, 1)])
def test_slice_index_is_the_nearest_slice_on_the_voxel_axis_along_the_world_axis(
    world_axis: int, voxel_axis: int
) -> None:
    voxel = np.array([1.0, 2.0, 3.0])
    off = np.full(3, 0.4)  # within half a voxel of (1, 2, 3)
    position = CYCLED[:3, :3] @ (voxel + off) + CYCLED[:3, 3]
    assert slice_index(CYCLED, (3, 4, 5), world_axis, position) == (
        voxel_axis,
        int(voxel[voxel_axis]),
    )


@pytest.mark.parametrize("voxel", [(-0.6, 0.0, 0.0), (2.6, 0.0, 0.0)])
def test_slice_index_is_none_outside_the_grid(voxel: tuple[float, ...]) -> None:
    position = CYCLED[:3, :3] @ np.array(voxel) + CYCLED[:3, 3]
    assert slice_index(CYCLED, (3, 4, 5), 1, position) is None


def test_slice_index_compares_axis_directions_not_voxel_sizes() -> None:
    # Axis 0 has the larger x component only because its voxels are bigger.
    affine = np.eye(4)
    affine[:3, 0] = 4 * np.array([0.6, 0.8, 0.0])
    affine[:3, 1] = [0.8, -0.6, 0.0]
    where = slice_index(affine, (3, 3, 3), 0, (0.0, 0.0, 0.0))
    assert where == (1, 0)


# Peaks


def _cycled() -> np.ndarray:
    # Voxel axes 0, 1, 2 run along world y, z, x, with unequal voxel sizes and a
    # small tilt: a column of the affine is not a row, and no axis is exact.
    tilt = np.deg2rad(10.0)
    c, s = np.cos(tilt), np.sin(tilt)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    cycle = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ cycle @ np.diag([1.0, 2.0, 3.0])
    affine[:3, 3] = [-10.0, 5.0, 2.0]
    return affine


def _world(affine: np.ndarray, voxel: list[float]) -> np.ndarray:
    return np.asarray(affine @ [*voxel, 1.0])[:3]


def test_the_largest_amplitude_is_the_longest_finite_peak() -> None:
    # Longer than one chunk along axis 0, with the largest peak in the last one.
    peaks = np.zeros((20, 1, 1, 6), np.float32)
    peaks[3, 0, 0, :3] = np.nan
    peaks[5, 0, 0, 3:] = [1.0, 2.0, 2.0]
    peaks[19, 0, 0, :3] = [0.0, 3.0, 4.0]
    assert largest_amplitude(peaks) == pytest.approx(5.0)
    assert largest_amplitude(peaks, slope=2.0) == pytest.approx(10.0)


def test_an_image_without_peaks_has_largest_amplitude_zero() -> None:
    assert largest_amplitude(np.full((2, 2, 2, 3), np.nan, np.float32)) == 0.0


def test_a_peak_segment_runs_through_its_voxel_centre_in_world_mm() -> None:
    affine = _cycled()
    peaks = np.zeros((3, 4, 5, 3), np.float32)
    direction = np.array([0.6, 0.0, -0.8])
    peaks[2, 1, 3] = 2.0 * direction
    positions, _ = peak_segments(peaks, affine, 1, 1, 4.0, 2.0)
    centre = _world(affine, [2.0, 1.0, 3.0])
    expected = [centre - 2.0 * direction, centre + 2.0 * direction]
    assert positions.dtype == np.float32
    # float32 positions about 10 mm from the origin: ~1e-6 mm.
    np.testing.assert_allclose(positions, expected, atol=1e-5)


@pytest.mark.parametrize(
    ("reference", "lengths"), [(2.0, [4.0, 2.0]), (None, [4.0, 4.0])]
)
def test_a_peak_length_follows_its_amplitude_unless_there_is_no_reference(
    reference: float | None, lengths: list[float]
) -> None:
    peaks = np.zeros((1, 1, 1, 6), np.float32)
    peaks[0, 0, 0] = [0.0, 0.0, 2.0, 1.0, 0.0, 0.0]
    positions, _ = peak_segments(peaks, np.eye(4), 2, 0, 4.0, reference)
    pairs = positions.reshape(-1, 2, 3)
    measured = np.linalg.norm(pairs[:, 1] - pairs[:, 0], axis=1)
    np.testing.assert_allclose(measured, lengths, rtol=1e-6)


def test_nan_zero_and_thresholded_peaks_are_skipped() -> None:
    peaks = np.zeros((5, 1, 1, 3), np.float32)
    peaks[0, 0, 0] = np.nan
    # peaks[1] stays zero.
    peaks[2, 0, 0] = [0.5, 0.0, 0.0]
    peaks[3, 0, 0] = [1.0, 0.0, 0.0]  # at the threshold: kept
    peaks[4, 0, 0] = [0.0, 2.0, 0.0]
    positions, _ = peak_segments(peaks, np.eye(4), 2, 0, 1.0, None, threshold=1.0)
    centres = positions.reshape(-1, 2, 3).mean(axis=1)
    np.testing.assert_allclose(centres, [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]], atol=1e-6)


def test_outward_halves_start_at_the_voxel_centre_and_keep_the_outer_end() -> None:
    peaks = np.zeros((2, 1, 1, 6), np.float32)
    peaks[0, 0, 0] = [0.6, 0.0, -0.8, 0.0, 2.0, 0.0]
    peaks[1, 0, 0, :3] = [-1.0, 0.0, 0.0]
    positions, colours = peak_segments(peaks, np.eye(4), 2, 0, 2.0, 2.0)
    before = positions.copy(), colours.copy()
    halves, halves_colours = outward_halves((positions, colours))
    pairs, whole = halves.reshape(-1, 2, 3), positions.reshape(-1, 2, 3)
    np.testing.assert_allclose(pairs[:, 0], [[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(pairs[:, 1], whole[:, 1])
    np.testing.assert_array_equal(halves_colours, colours)
    np.testing.assert_array_equal(positions, before[0])
    np.testing.assert_array_equal(colours, before[1])


def test_a_peak_is_coloured_by_its_absolute_direction() -> None:
    peaks = np.zeros((1, 1, 1, 3), np.float32)
    peaks[0, 0, 0] = [-1.2, 0.0, 1.6]
    _, colours = peak_segments(peaks, np.eye(4), 0, 0, 1.0, None)
    assert colours.dtype == np.float32
    np.testing.assert_allclose(colours, [[0.6, 0.0, 0.8]] * 2, rtol=1e-6)


# Voxel axes permuted and flipped: voxel i runs along -y, j along z, k along x.
PERMUTED = np.array(
    [[0, 0, 2.5, -10], [-1.5, 0, 0, 4], [0, 2, 0, 7], [0, 0, 0, 1]], float
)


@pytest.mark.parametrize(
    ("affine", "voxel"),
    [(np.diag([2.0, 2.0, 2.0, 1.0]), (1, 3, 0)), (PERMUTED, (2, 0, 4))],
)
def test_the_nearest_voxel_is_the_one_whose_centre_is_closest(
    affine: np.ndarray, voxel: tuple[int, int, int]
) -> None:
    # Off the centre by under half a voxel on each axis.
    position = affine[:3, :3] @ (np.array(voxel) + [0.4, -0.3, 0.2]) + affine[:3, 3]
    assert nearest_voxel(affine, (3, 4, 5, 6), position) == voxel


@pytest.mark.parametrize("voxel", [(-0.6, 1, 1), (2.6, 1, 1), (1, 1, 4.5)])
def test_the_nearest_voxel_is_none_outside_the_grid(
    voxel: tuple[float, float, float],
) -> None:
    position = PERMUTED[:3, :3] @ voxel + PERMUTED[:3, 3]
    assert nearest_voxel(PERMUTED, (3, 4, 5), position) is None


# A regular octahedron of radius 1: vertices +x, -x, +y, -y, +z, -z.
OCTAHEDRON = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.float32
)
OCTAHEDRON_FACES = np.array(
    [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ]
)


def _cut_segments(cut: PlaneCut) -> list[frozenset[tuple[float, ...]]]:
    pairs = cut.positions.reshape(-1, 2, 3)
    return [frozenset(tuple(float(x) for x in p) for p in pair) for pair in pairs]


@pytest.mark.parametrize(
    ("position", "corners"),
    [
        # Halfway up: the midpoints of the edges to the top vertex.
        (0.5, [(0.5, 0, 0.5), (0, 0.5, 0.5), (-0.5, 0, 0.5), (0, -0.5, 0.5)]),
        # Through the four equator vertices, whose edges lie in the plane.
        (0.0, [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]),
    ],
    ids=["between vertices", "through vertices"],
)
def test_a_plane_cuts_an_octahedron_in_its_square_outline_once(
    position: float, corners: list[tuple[float, float, float]]
) -> None:
    cut = plane_cut(OCTAHEDRON, OCTAHEDRON_FACES, 2, position)
    assert not np.isnan(cut.positions).any()
    square = [
        frozenset({tuple(map(float, corners[i])), tuple(map(float, corners[i - 1]))})
        for i in range(4)
    ]
    found = _cut_segments(cut)
    assert len(found) == 4
    assert set(found) == set(square)


@pytest.mark.parametrize("position", [1.0, 1.5, -2.0], ids=["apex", "above", "below"])
def test_a_plane_touching_or_missing_the_mesh_cuts_nothing(position: float) -> None:
    cut = plane_cut(OCTAHEDRON, OCTAHEDRON_FACES, 2, position)
    assert cut.positions.shape == (0, 3)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_values_at_a_cut_are_interpolated_linearly_along_the_edges(
    axis: int,
) -> None:
    # Values linear in position are exact at any point of an edge.
    vertices = OCTAHEDRON * np.float32(4) + np.array([1, -2, 3], np.float32)
    values = (vertices @ np.array([2.0, -3.0, 0.5]) + 7.0)[:, None]
    cut = plane_cut(vertices, OCTAHEDRON_FACES, axis, [2.0, -3.0, 4.5][axis])
    assert len(cut.positions) == 8
    expected = cut.positions.astype(np.float64) @ np.array([2.0, -3.0, 0.5]) + 7.0
    np.testing.assert_allclose(cut.interpolate(values)[:, 0], expected, rtol=1e-6)


def test_values_go_through_the_colormap_clipped_to_the_range() -> None:
    # Red, green, blue over [10, 20]; NaN is grey.
    table = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], np.float32)
    values = np.array([0.0, 10.0, 12.5, 15.0, 20.0, 99.0, np.nan])
    colours = colormap_colours(values, table, 10.0, 20.0)
    expected = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [0.5, 0.5, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0.5, 0.5, 0.5],
        ],
        np.float32,
    )
    np.testing.assert_allclose(colours, expected, atol=1e-6)


def test_the_normal_subset_is_the_same_each_time_and_grows_by_adding_vertices() -> None:
    small = normal_subset(1000, 100)
    assert np.array_equal(normal_subset(1000, 100), small)
    assert np.array_equal(small, np.unique(small)) and len(small) == 100
    assert set(small) < set(normal_subset(1000, 300))


def test_the_normal_subset_is_capped_at_the_vertex_count() -> None:
    assert np.array_equal(normal_subset(50, 1000), np.arange(50))


def test_the_normal_subset_takes_only_vertices_in_the_mask() -> None:
    mask = np.arange(1000) % 3 == 0
    chosen = normal_subset(1000, 100, mask)
    assert len(chosen) == 100 and mask[chosen].all()
    assert np.array_equal(normal_subset(1000, 1000, mask), np.flatnonzero(mask))


# Unit normals scaled by 3, so the lines must make them unit; the last has none.
NORMAL_POINTS = np.array([[1, 2, 3], [0, 0, 0], [-5, 4, 1], [7, 7, 7]], np.float32)
NORMALS = 3 * np.array([[0.6, -0.8, 0], [0, 0, -1], [-0.48, 0.6, 0.64], [0, 0, 0]])


def test_normal_lines_run_their_length_along_the_unit_normal_coloured_by_it() -> None:
    positions, colours = normal_lines(NORMAL_POINTS, NORMALS, 2.5)
    unit = NORMALS[:3] / 3
    ends = positions.reshape(-1, 2, 3)
    np.testing.assert_allclose(ends[:, 0], NORMAL_POINTS[:3])
    np.testing.assert_allclose(ends[:, 1], NORMAL_POINTS[:3] + 2.5 * unit, atol=1e-6)
    np.testing.assert_allclose(
        colours.reshape(-1, 2, 3), np.abs(unit)[:, None].repeat(2, 1), atol=1e-7
    )


def test_normal_lines_in_a_plane_are_projected_and_one_across_it_is_left_out() -> None:
    # Across z, the second normal points along z and has no direction in the plane.
    positions, colours = normal_lines(NORMAL_POINTS, NORMALS, 2.5, axis=2)
    ends = positions.reshape(-1, 2, 3)
    np.testing.assert_allclose(ends[:, 0], NORMAL_POINTS[[0, 2]])
    flat = np.array([[0.6, -0.8, 0], [-0.6, 0.75, 0]])
    flat /= np.linalg.norm(flat, axis=1, keepdims=True)
    np.testing.assert_allclose(ends[:, 1] - ends[:, 0], 2.5 * flat, atol=1e-6)
    expected = np.abs(NORMALS[[0, 2]] / 3)
    np.testing.assert_allclose(colours[::2], expected, atol=1e-7)


def _prism(sides: int, radius: float) -> tuple[np.ndarray, np.ndarray]:
    # The side walls of a regular prism from z = 0 to z = 2, open at both ends.
    angles = 2 * np.pi * np.arange(sides) / sides
    ring = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    vertices = np.concatenate(
        [np.c_[ring, np.zeros(sides)], np.c_[ring, np.full(sides, 2.0)]]
    ).astype(np.float32)
    i = np.arange(sides)
    j = (i + 1) % sides
    faces = np.concatenate([np.c_[i, j, i + sides], np.c_[j, j + sides, i + sides]])
    return vertices, faces


def _strip(x0: float, x1: float) -> tuple[np.ndarray, np.ndarray]:
    # A flat strip in y = 0 from x0 to x1, 2 mm high, in quads 1 mm wide or less;
    # its faces shuffled, so the cut's first segment is not at an end of its line.
    xs = np.linspace(x0, x1, max(2, int(np.ceil(x1 - x0)) + 1))
    n = len(xs)
    vertices = np.concatenate(
        [np.c_[xs, np.zeros(n), np.zeros(n)], np.c_[xs, np.zeros(n), np.full(n, 2.0)]]
    ).astype(np.float32)
    i = np.arange(n - 1)
    faces = np.concatenate([np.c_[i, i + 1, i + n], np.c_[i + 1, i + n + 1, i + n]])
    return vertices, np.random.default_rng(0).permutation(faces)


def test_samples_on_a_closed_cut_lie_on_it_one_spacing_apart_along_it() -> None:
    sides, radius, spacing = 12, 10.0, 3.0
    vertices, faces = _prism(sides, radius)
    cut = plane_cut(vertices, faces, 2, 1.0)
    points, values = sample_cut(cut, spacing, vertices)
    # The cut is the 12-gon at z = 1; each sample's arc length along it.
    corners = vertices[:sides, :2].astype(np.float64)
    side = np.linalg.norm(corners[1] - corners[0])
    perimeter = sides * side
    arcs = []
    for p in points[:, :2].astype(np.float64):
        a, b = corners, np.roll(corners, -1, axis=0)
        t = np.clip(np.sum((p - a) * (b - a), axis=1) / side**2, 0, 1)
        gaps = np.linalg.norm(a + t[:, None] * (b - a) - p, axis=1)
        k = int(np.argmin(gaps))
        assert gaps[k] < 1e-4
        arcs.append((k + t[k]) * side)
    np.testing.assert_allclose(points[:, 2], 1.0)
    # Around the loop: k steps of the spacing, and the rest where the line closes.
    arc = np.sort(arcs)
    steps = np.sort(np.diff(np.r_[arc, arc[0] + perimeter]))
    k = int(perimeter // spacing)
    rest = perimeter - k * spacing
    np.testing.assert_allclose(steps, [rest] + [spacing] * k, atol=1e-4)
    # Values linear in position are exact anywhere on an edge.
    np.testing.assert_allclose(values, points, atol=1e-5)


def test_samples_on_open_cuts_are_centred_on_each_and_a_short_one_gets_its_middle() -> (
    None
):
    long_v, long_f = _strip(0.0, 10.0)
    short_v, short_f = _strip(20.0, 21.0)
    vertices = np.concatenate([long_v, short_v])
    faces = np.concatenate([long_f, short_f + len(long_v)])
    points, _ = sample_cut(plane_cut(vertices, faces, 2, 1.0), 3.0, vertices)
    np.testing.assert_allclose(
        np.sort(points[:, 0]), [0.5, 3.5, 6.5, 9.5, 20.5], atol=1e-5
    )
    np.testing.assert_allclose(points[:, 1:], [[0.0, 1.0]] * 5, atol=1e-6)


def test_a_cut_through_nothing_gives_no_samples() -> None:
    vertices, faces = _prism(12, 10.0)
    points, values = sample_cut(plane_cut(vertices, faces, 2, 5.0), 3.0, vertices)
    assert points.shape == (0, 3) and values.shape == (0, 3)


def test_a_cut_through_a_vertex_on_the_plane_keeps_one_spacing_across_it() -> None:
    # A fan of six triangles in the x-z plane whose centre lies on the cut: the cut
    # runs through it from one face to the next, as one line.
    angles = np.pi / 6 + np.pi / 3 * np.arange(6)
    rim = 10 * np.c_[np.cos(angles), np.zeros(6), np.sin(angles)]
    vertices = np.concatenate([[[0.0, 0.0, 0.0]], rim]).astype(np.float32)
    k = np.arange(6)
    faces = np.c_[np.zeros(6, np.int64), k + 1, (k + 1) % 6 + 1]
    points, _ = sample_cut(plane_cut(vertices, faces, 2, 0.0), 3.0, vertices)
    np.testing.assert_allclose(
        np.sort(points[:, 0]), [-7.5, -4.5, -1.5, 1.5, 4.5, 7.5], atol=1e-5
    )
