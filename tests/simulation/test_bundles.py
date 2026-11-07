import json
from typing import Any

import numpy as np
import pytest
from scipy.spatial import cKDTree

import larmor.simulation.bundles as bundles
from larmor.simulation import BundleGeometry, create_bundle
from larmor.simulation.bundles import (
    _bend_limits,
    _squeeze_inside_bends,
    centreline_spline,
)
from larmor.simulation.voxelise import voxelise
from larmor.tractography import resample, streamline_lengths
from larmor.types import Grid, Tractogram

GRID = Grid((20, 20, 20), np.eye(4))


def straight(start: list[float], end: list[float], num: int = 50) -> Tractogram:
    points = np.linspace(start, end, num)
    return Tractogram(points, np.array([0, num]))


def test_create_bundle_is_deterministic_for_a_seed() -> None:
    geometry = BundleGeometry(np.array([[0.0, 0, 0], [10, 2, 0], [20, 0, 0]]), seed=3)
    first, second = create_bundle(geometry, index=2), create_bundle(geometry, index=2)
    np.testing.assert_array_equal(first.points, second.points)
    assert (first.streamline_data["bundle"] == 2).all()


def test_density_across_a_bundle_is_a_gaussian_cut_at_the_radius() -> None:
    radius = 5.0
    geometry = BundleGeometry(
        np.array([[0.0, 0, 0], [30, 0, 0]]),
        num_streamlines=20000,
        radius=radius,
        num_points=2,
        seed=0,
    )
    across = create_bundle(geometry).points[::2, 1:]
    distance = np.hypot(*across.T)
    assert distance.max() <= radius + 1e-5
    sd = radius / 2.5
    for inside in (1.0, 2.0, 2.4):
        expected = (1 - np.exp(-(inside**2) / 2)) / (1 - np.exp(-(2.5**2) / 2))
        assert np.mean(distance < inside * sd) == pytest.approx(expected, abs=0.01)
    # Round, not stretched along either axis.
    assert across[:, 0].std() == pytest.approx(across[:, 1].std(), rel=0.02)


def test_streamlines_stay_within_the_fanned_radius_of_a_straight_bundle() -> None:
    geometry = BundleGeometry(
        np.array([[0.0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]]),
        num_streamlines=300,
        radius=2.0,
        fan_start=1.0,
        seed=0,
    )
    bundle = create_bundle(geometry)
    distance = np.hypot(bundle.points[:, 1], bundle.points[:, 2])
    assert distance.max() <= 2.0 * (1 + 1.0) + 1e-5
    starts = bundle.points[bundle.offsets[:-1]]
    ends = bundle.points[bundle.offsets[1:] - 1]
    assert np.linalg.norm(starts - [0, 0, 0], axis=1).max() <= 4.0 + 1e-5
    assert np.linalg.norm(ends - [30, 0, 0], axis=1).max() <= 2.0 + 1e-5


# Unevenly spaced points of a freehand doodle with loops and sharp turns, mm.
DOODLE = np.array(
    [
        [-11.0, -2.6, 0],
        [-8.25, 6.5, 0],
        [-3.9, 7.1, 0],
        [1.9, -2.0, 0],
        [6.5, -3.5, 0],
        [10.0, 7.75, 0],
        [9.9, -5.9, 0],
        [-8.1, -9.5, 0],
    ]
)


def _distance_to_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    starts, ends = polygon[:-1], polygon[1:]
    line = ends - starts
    along = np.einsum("nsk,sk->ns", points[:, None] - starts, line)
    t = np.clip(along / np.einsum("sk,sk->s", line, line), 0, 1)
    nearest = starts + t[..., None] * line
    distance: np.ndarray = np.linalg.norm(points[:, None] - nearest, axis=2)
    return distance.min(axis=1)


def test_the_centreline_passes_through_every_control_point() -> None:
    centreline, u_control = centreline_spline(DOODLE)
    np.testing.assert_allclose(centreline(u_control), DOODLE, atol=1e-9)
    assert u_control[0] == 0 and u_control[-1] == 1


def test_the_centreline_stays_near_the_control_polygon_of_a_doodle() -> None:
    centreline, _ = centreline_spline(DOODLE)
    points = centreline(np.linspace(0, 1, 2000))
    assert _distance_to_polygon(points, DOODLE).max() < 1.5


def test_two_control_points_give_a_straight_centreline() -> None:
    centreline, _ = centreline_spline(np.array([[0.0, 0, 0], [10, 5, 0]]))
    points = centreline(np.linspace(0, 1, 11))
    np.testing.assert_allclose(points, np.linspace([0, 0, 0], [10, 5, 0], 11))


def test_streamlines_keep_their_distance_to_the_centreline_round_a_sharp_bend() -> None:
    control = np.array(
        [[0.0, 0, 0], [2, 0, 0], [20, 0, 0], [25, 6, 0], [24, 12, 0], [0, 12, 0]]
    )
    geometry = BundleGeometry(control, num_streamlines=200, radius=2.0, seed=0)
    streamlines = create_bundle(geometry).points.reshape(200, 100, 3)
    centreline, _ = centreline_spline(control)
    tree = cKDTree(centreline(np.linspace(0, 1, 20000)))
    distance = tree.query(streamlines.reshape(-1, 3))[0].reshape(200, 100)
    # Each streamline's distance is its offset, the same all along.
    assert np.abs(distance - distance[:, :1]).max() < 0.03 * 2.0
    assert distance.max() > 0.95 * 2.0


def _quarter_circle(bend: float) -> np.ndarray:
    angle = np.linspace(0, np.pi / 2, 40)
    return bend * np.stack([np.cos(angle), np.sin(angle), np.zeros(40)], axis=1)


def test_the_fold_guard_moves_nothing_until_a_bend_is_tight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def streamlines(bend: float) -> np.ndarray:
        arc = _quarter_circle(bend)
        geometry = BundleGeometry(arc, num_streamlines=5000, radius=5.0, seed=1)
        points: np.ndarray = create_bundle(geometry).points
        return points

    gentle, tight = streamlines(10.5), streamlines(7.0)
    monkeypatch.setattr(bundles, "FOLD_LIMIT", 1e6)
    np.testing.assert_array_equal(gentle, streamlines(10.5))
    assert not np.array_equal(tight, streamlines(7.0))


def _deepest_fold(points: np.ndarray, control: np.ndarray) -> float:
    centreline, _ = centreline_spline(control)
    u = np.linspace(0, 1, points.shape[1])
    tangents = centreline.derivative()((u[:-1] + u[1:]) / 2)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    along = np.cumsum(np.sum(np.diff(points, axis=1) * tangents, axis=2), axis=1)
    return float((np.maximum.accumulate(along, axis=1) - along).max())


def test_the_fold_guard_shrinks_folds_at_a_tight_bend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corner = np.array([[-25.0, -15, 0], [15, -15, 0], [15, 25, 0]])
    geometry = BundleGeometry(
        corner, num_streamlines=2000, radius=10.0, num_points=400, seed=1
    )
    guarded = create_bundle(geometry).points.reshape(2000, 400, 3).astype(np.float64)
    assert _deepest_fold(guarded, corner) < 0.1
    # Held back, and so still within the bundle's radius of the centreline.
    centreline, _ = centreline_spline(corner)
    tree = cKDTree(centreline(np.linspace(0, 1, 20000)))
    assert tree.query(guarded[::4].reshape(-1, 3))[0].max() <= 10.0 + 1e-3
    monkeypatch.setattr(bundles, "FOLD_LIMIT", 1e6)
    unguarded = create_bundle(geometry).points.reshape(2000, 400, 3).astype(np.float64)
    assert _deepest_fold(unguarded, corner) > 1.0


def test_squeezing_stays_inside_the_limit_and_keeps_the_streamlines_in_order() -> None:
    reach = np.linspace(-20.0, 20.0, 401)
    offsets = np.stack([np.ones(401), reach, np.zeros(401)], axis=1)[:, None]
    towards, limit = np.array([[0.0, 1.0, 0.0]]), np.array([5.0])
    squeezed = _squeeze_inside_bends(offsets, towards, limit)[:, 0]
    left_alone = reach <= bundles.FOLD_KNEE * 5.0
    np.testing.assert_array_equal(squeezed[left_alone], offsets[left_alone, 0])
    assert squeezed[:, 1].max() < 5.0
    assert np.all(np.diff(squeezed[:, 1]) > 0)
    np.testing.assert_array_equal(squeezed[:, [0, 2]], offsets[:, 0, [0, 2]])


@pytest.mark.parametrize("num_points", [100, 400])
def test_a_bends_limit_follows_its_radius_at_any_number_of_points(
    num_points: int,
) -> None:
    centreline, _ = centreline_spline(_quarter_circle(20.0))
    u = np.linspace(0, 1, num_points)
    tangents = centreline.derivative()(u)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    towards, limit = _bend_limits(tangents, centreline.length / (num_points - 1))
    middle = num_points // 2
    assert limit[middle] == pytest.approx(bundles.FOLD_LIMIT * 20.0, rel=0.07)
    to_centre = -centreline(u)[middle] / 20.0
    np.testing.assert_allclose(towards[middle], to_centre, atol=0.02)


@pytest.mark.parametrize(
    ("control", "rings"),
    [
        (
            np.array([[-25.0, -15, 0], [15, -15, 0], [15, 25, 0]]),
            np.array([[0.0, 10, 2, 90], [1.0, 10, 2, 90]]),
        ),
        (_quarter_circle(7.0), np.array([[0.0, 14, 2, 90], [1.0, 14, 2, 90]])),
    ],
    ids=["three_point_corner", "quarter_circle"],
)
def test_the_fold_guard_shrinks_the_folds_of_a_flat_section(
    control: np.ndarray, rings: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    geometry = BundleGeometry(control, num_streamlines=2000, seed=1, rings=rings)

    def fold() -> float:
        points = create_bundle(geometry).points.reshape(2000, 100, 3)
        return _deepest_fold(points.astype(np.float64), control)

    guarded = fold()
    monkeypatch.setattr(bundles, "FOLD_LIMIT", 1e6)
    assert fold() > max(1.0, 5 * guarded)


STRAIGHT = np.array([[0.0, 0, 0], [60, 0, 0]])
# The standard deviation of a Gaussian cut at the radius, as a share of the radius.
CUT_SD = 0.3702


def _along_rings(rings: np.ndarray, num: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Returns the places of a straight bundle's streamlines along r1 and r2.

    The bundle runs along x, where the frame's normal is z and its binormal -y.
    """
    geometry = BundleGeometry(STRAIGHT, num_streamlines=num, seed=0, rings=rings)
    points = create_bundle(geometry).points.reshape(num, 100, 3)
    return points[..., 2].astype(np.float64), -points[..., 1].astype(np.float64)


def test_rings_set_the_width_and_the_density_of_the_section() -> None:
    rings = np.array(
        [[0.0, 2, 2, 0], [1 / 3, 8, 2, 0], [2 / 3, 3, 6, 0], [1.0, 5, 5, 0]]
    )
    first, second = _along_rings(rings)
    for row, ring in zip((0, 33, 66, 99), rings, strict=True):
        for places, radius in ((first, ring[1]), (second, ring[2])):
            furthest = np.abs(places[:, row]).max()
            assert 0.97 * radius < furthest <= radius + 1e-4
            assert places[:, row].std() == pytest.approx(radius * CUT_SD, rel=0.01)
            inside = np.mean(np.abs(places[:, row]) < 0.4 * radius)
            assert inside == pytest.approx(0.703, abs=0.01)


def test_the_angle_of_a_ring_turns_the_section() -> None:
    first, second = _along_rings(np.array([[0.0, 6, 2, 0], [1.0, 6, 2, 90]]))
    for row in (0, 33, 66, 99):
        vectors = np.linalg.eigh(np.cov(first[:, row], second[:, row]))[1]
        long_axis = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
        assert (long_axis - 90 * row / 99 + 90) % 180 - 90 == pytest.approx(0, abs=1)


def test_a_section_stays_between_its_rings_at_a_sudden_change() -> None:
    rings = np.array(
        [
            [0.0, 2, 2, 0],
            [0.2, 2, 2, 0],
            [0.25, 10, 2, 0],
            [0.7, 10, 8, 0],
            [1.0, 3, 8, 0],
        ]
    )
    first, _ = _along_rings(rings, num=2000)
    width = np.abs(first).max(axis=0)
    u = np.linspace(0, 1, 100)
    assert width.max() <= 10.0 + 1e-4
    rising, falling = width[u <= 0.25], width[u >= 0.7]
    assert np.all(np.diff(rising) >= -1e-4) and np.all(np.diff(falling) <= 1e-4)


def test_round_rings_of_one_radius_give_the_streamlines_of_that_radius() -> None:
    control = np.array([[0.0, 0, 0], [10, 2, 0], [20, 0, 0]])
    rings = np.array([[0.0, 3, 3, 0], [1.0, 3, 3, 0]])
    ringed = BundleGeometry(control, num_streamlines=50, seed=1, rings=rings)
    plain = BundleGeometry(control, num_streamlines=50, seed=1, radius=3.0)
    np.testing.assert_allclose(
        create_bundle(ringed).points, create_bundle(plain).points, atol=1e-4
    )
    assert plain.is_tube and not ringed.is_tube


@pytest.mark.parametrize(
    ("rings", "fan_end", "message"),
    [
        ([[0.0, 1, 1, 0], [0.5, 1, 1, 0], [0.4, 1, 1, 0], [1.0, 1, 1, 0]], 0.0, "rise"),
        ([[0.0, 1, 1, 0], [0.9, 1, 1, 0]], 0.0, "rise"),
        ([[0.1, 1, 1, 0], [1.0, 1, 1, 0]], 0.0, "rise"),
        ([[0.0, 1, 1, 0]], 0.0, "rise"),
        ([[0.0, np.nan, 1, 0], [1.0, 1, 1, 0]], 0.0, "finite"),
        ([[0.0, 0, 1, 0], [1.0, 1, 1, 0]], 0.0, "positive"),
        ([[0.0, 1, 1], [1.0, 1, 1]], 0.0, "rows"),
        ([[0.0, 1, 1, 0], [1.0, 1, 1, 0]], 1.0, "not both"),
    ],
)
def test_bad_rings_are_refused(
    rings: list[list[float]], fan_end: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BundleGeometry(STRAIGHT, rings=np.array(rings), fan_end=fan_end)


def test_rings_round_trip_through_plain_values() -> None:
    rings = np.array([[0.0, 2, 2, 0], [0.4, 8, 2, 30], [1.0, 3, 3, 90]])
    values = BundleGeometry(STRAIGHT, seed=1, rings=rings).to_dict()
    read = BundleGeometry.from_dict(json.loads(json.dumps(values)))
    assert read.rings is not None
    np.testing.assert_array_equal(read.rings, rings)
    assert "rings" not in BundleGeometry(STRAIGHT).to_dict()


def _places(geometry: BundleGeometry) -> np.ndarray:
    """Returns the streamlines of a straight bundle along x, (S, N, 2) across it."""
    points = create_bundle(geometry).points.astype(np.float64)
    return points.reshape(geometry.num_streamlines, geometry.num_points, 3)[..., 1:]


def test_a_tangle_slides_a_streamline_between_its_two_places() -> None:
    # The points are 0.5 mm apart, so a slide of 10 mm takes 20 of them.
    geometry = BundleGeometry(
        np.array([[0.0, 0, 0], [40, 0, 0]]),
        num_streamlines=5000,
        radius=3.0,
        num_points=81,
        seed=1,
        tangle_mm=10.0,
    )
    places = _places(geometry)
    plain = BundleGeometry(
        geometry.control_points,
        num_streamlines=5000,
        radius=3.0,
        num_points=81,
        seed=1,
    )
    # It starts where the untangled one stays, and is opposite after two slides.
    np.testing.assert_allclose(places[:, 0], _places(plain)[:, 0], atol=1e-4)
    np.testing.assert_allclose(places[:, 40], -places[:, 0], atol=1e-4)
    # After one slide it is somewhere unrelated.
    assert abs(np.corrcoef(places[:, 0, 0], places[:, 20, 0])[0, 1]) < 0.05


@pytest.mark.parametrize(
    ("shape", "first", "second"),
    [
        ({"radius": 3.0}, 3.0, 3.0),
        ({"rings": np.array([[0.0, 8, 2, 0], [1.0, 8, 2, 0]])}, 2.0, 8.0),
    ],
    ids=["round", "flat"],
)
def test_a_tangle_keeps_the_spread_and_only_some_streamlines_pass_the_edge(
    shape: dict[str, Any], first: float, second: float
) -> None:
    geometry = BundleGeometry(
        STRAIGHT, num_streamlines=20000, seed=1, tangle_mm=20.0, **shape
    )
    places = _places(geometry)
    sizes = np.array([first, second])
    np.testing.assert_allclose(places.std(axis=0) / sizes, CUT_SD, rtol=0.02)
    shares = np.linalg.norm(places / sizes, axis=2)
    assert shares[:, 0].max() <= 1.0 + 1e-4
    assert np.mean(shares > 1 + 1e-4, axis=0).max() == pytest.approx(0.02, abs=0.005)
    assert shares.max() <= np.sqrt(2) + 1e-4


def test_a_tangle_tilts_streamlines_by_radius_over_tangle() -> None:
    geometry = BundleGeometry(
        np.array([[0.0, 0, 0], [40, 0, 0]]),
        num_streamlines=2000,
        radius=3.0,
        num_points=400,
        seed=1,
        tangle_mm=10.0,
    )
    points = create_bundle(geometry).points.astype(np.float64).reshape(2000, 400, 3)
    steps = np.diff(points, axis=1)
    tilt = np.degrees(np.arctan2(np.hypot(steps[..., 1], steps[..., 2]), steps[..., 0]))
    assert tilt.mean() == pytest.approx(40 * 3.0 / 10.0, abs=1.0)


def test_the_fold_guard_shrinks_the_folds_of_a_tangled_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corner = np.array([[-25.0, -15, 0], [15, -15, 0], [15, 25, 0]])
    geometry = BundleGeometry(
        corner, num_streamlines=2000, radius=12.0, seed=1, tangle_mm=15.0
    )

    def fold() -> float:
        points = create_bundle(geometry).points.astype(np.float64)
        return _deepest_fold(points.reshape(2000, 100, 3), corner)

    guarded = fold()
    monkeypatch.setattr(bundles, "FOLD_LIMIT", 1e6)
    assert fold() > 5 * guarded


def test_a_tangle_too_short_for_the_points_is_refused() -> None:
    ends = np.array([[0.0, 0, 0], [99.0, 0, 0]])  # points 1 mm apart
    create_bundle(BundleGeometry(ends, num_streamlines=2, seed=1, tangle_mm=2.0))
    # Exactly twice the spacing, on a length where the spacing rounds unevenly.
    other = np.array([[0.0, 0, 0], [40.0, 0, 0]])
    limit = 2 * 40.0 / 99
    create_bundle(BundleGeometry(other, num_streamlines=2, seed=1, tangle_mm=limit))
    with pytest.raises(ValueError, match="too short"):
        create_bundle(BundleGeometry(ends, num_streamlines=2, seed=1, tangle_mm=1.9))
    with pytest.raises(ValueError, match="too short"):
        create_bundle(BundleGeometry(ends, num_streamlines=2, seed=1, tangle_mm=0.25))


@pytest.mark.parametrize("tangle_mm", [-1.0, float("nan"), float("inf")])
def test_a_bad_tangle_is_refused(tangle_mm: float) -> None:
    with pytest.raises(ValueError, match="tangle_mm"):
        BundleGeometry(STRAIGHT, tangle_mm=tangle_mm)


def test_a_tangle_round_trips_through_plain_values_and_is_left_out_when_off() -> None:
    values = BundleGeometry(STRAIGHT, seed=1, tangle_mm=12.5).to_dict()
    assert BundleGeometry.from_dict(json.loads(json.dumps(values))).tangle_mm == 12.5
    assert "tangle_mm" not in BundleGeometry(STRAIGHT, tangle_mm=0.0).to_dict()


def test_dispersion_wanders_inside_and_not_at_the_ends() -> None:
    control = np.array([[0.0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]])
    geometry = BundleGeometry(
        control, num_streamlines=2000, radius=0.0, dispersion=1.5, seed=0
    )
    streamlines = create_bundle(geometry).points.reshape(2000, 100, 3)
    across = streamlines[..., 1:]
    np.testing.assert_allclose(across[:, [0, -1]], 0.0, atol=1e-12)
    at_inner_points = across[:, [33, 66]].std(axis=0)  # u = 1/3 and 2/3
    np.testing.assert_allclose(at_inner_points, 1.5, rtol=0.1)


def test_streamline_on_a_voxel_row_fills_that_row_only() -> None:
    # Voxel centres are at the affine's positions, so y = j is row j exactly.
    voxels = voxelise([straight([0.0, 7.0, 3.0], [19.0, 7.0, 3.0])], GRID)
    i, j, k = np.unravel_index(voxels.voxel, GRID.shape)
    assert set(j.tolist()) == {7} and set(k.tolist()) == {3}
    assert set(i.tolist()) == set(range(20))


def test_total_length_equals_the_length_inside_the_grid() -> None:
    bundle = create_bundle(
        BundleGeometry(np.array([[2.0, 3, 4], [10, 12, 9], [17, 15, 16]]), seed=1)
    )
    voxels = voxelise([bundle], GRID)
    total = streamline_lengths(resample(bundle, step_size=0.1)).sum()
    np.testing.assert_allclose(voxels.length.sum(), total, rtol=1e-6)
    # The grid ends at x = 19.5, the far face of its last voxel; the points are
    # 0.1 mm apart.
    half = voxelise([straight([10.0, 5, 5], [30.0, 5, 5], 201)], GRID)
    assert abs(half.length.sum() - 9.5) <= 0.1


def test_off_vertex_direction_is_kept_exactly() -> None:
    direction = np.array([1.0, 0.37, 0.21])
    direction /= np.linalg.norm(direction)
    line = straight((10 - 8 * direction).tolist(), (10 + 8 * direction).tolist())
    voxels = voxelise([line], GRID)
    angles = np.degrees(np.arccos(np.clip(np.abs(voxels.direction @ direction), 0, 1)))
    assert angles.max() < 0.01


def test_each_direction_in_a_voxel_gives_its_own_fibre() -> None:
    # A straight line gives one fibre per voxel; two lines along x and along y
    # in one voxel give fibres along each, each with its line's length.
    parallel = voxelise([straight([0.0, 7.0, 3.0], [19.0, 7.0, 3.0])], GRID)
    assert len(parallel.length) == 20
    along_x = np.linspace([9.6, 10.0, 10.0], [10.4, 10.0, 10.0], 9)
    along_y = np.linspace([10.0, 9.6, 10.0], [10.0, 10.4, 10.0], 9)
    crossing = Tractogram(np.concatenate([along_x, along_y]), np.array([0, 9, 18]))
    voxels = voxelise([crossing], GRID)
    # A line near the edge of a direction bin may give a second, tiny fibre from
    # its end pieces, within a degree of it.
    along_x = np.abs(voxels.direction[:, 0]) > np.cos(np.radians(1))
    along_y = np.abs(voxels.direction[:, 1]) > np.cos(np.radians(1))
    assert np.all(along_x | along_y)
    np.testing.assert_allclose(voxels.length[along_x].sum(), 0.8, rtol=1e-4)
    np.testing.assert_allclose(voxels.length[along_y].sum(), 0.8, rtol=1e-4)


@pytest.mark.parametrize(
    "offsets", [[0, 0, 40], [0, 40, 40], [0, 0, 40, 40]], ids=["first", "last", "both"]
)
def test_empty_streamlines_change_nothing_in_the_voxelisation(
    offsets: list[int],
) -> None:
    line = straight([0.0, 7.0, 3.0], [19.0, 7.0, 3.0], 40)
    alone = voxelise([line], GRID)
    with_empty = voxelise([Tractogram(line.points, np.array(offsets))], GRID)
    np.testing.assert_allclose(with_empty.length, alone.length)


def test_a_bundle_of_empty_streamlines_reaches_no_voxel() -> None:
    empty = Tractogram(np.zeros((0, 3), np.float32), np.array([0, 0, 0]))
    assert len(voxelise([empty], GRID).voxel) == 0


@pytest.mark.parametrize(
    ("rings", "section"),
    [
        (
            [
                [0, 3, 3, 0],
                [0.3, 3, 3, 0],
                [0.5, 6, 6, 0],
                [0.7, 3, 3, 0],
                [1, 3, 3, 0],
            ],
            9,
        ),
        (
            [
                [0, 3, 3, 0],
                [0.3, 3, 3, 0],
                [0.5, 1, 1, 0],
                [0.7, 3, 3, 0],
                [1, 3, 3, 0],
            ],
            9,
        ),
        ([[0, 8, 2, 40], [1, 8, 2, 90]], 16),
    ],
    ids=["widened locally", "narrowed locally", "flat"],
)
def test_the_median_section_is_the_size_a_bundle_mostly_has(
    rings: list[list[float]], section: float
) -> None:
    ends = np.array([[0.0, 0, 0], [40.0, 0, 0]])
    geometry = BundleGeometry(ends, rings=np.array(rings, dtype=float))
    assert geometry.median_section() == pytest.approx(np.pi * section)


def test_a_geometry_without_rings_has_no_median_section() -> None:
    with pytest.raises(ValueError, match="without rings"):
        BundleGeometry(np.array([[0.0, 0, 0], [40.0, 0, 0]])).median_section()
