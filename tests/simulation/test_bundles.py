import numpy as np
from scipy.spatial import cKDTree

from larmor.simulation import BundleGeometry, create_bundle
from larmor.simulation.bundles import centreline_spline
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
    # Each streamline's distance is its offset in the disc, the same all along.
    assert np.abs(distance - distance[:, :1]).max() < 0.03 * 2.0
    assert distance.max() > 0.95 * 2.0


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
