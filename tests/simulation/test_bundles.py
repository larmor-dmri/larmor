import numpy as np

from larmor.simulation import BundleGeometry, create_bundle
from larmor.simulation.voxelise import voxelise
from larmor.tractography import resample, streamline_lengths, track_density
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


def test_streamline_on_a_voxel_row_fills_that_row_only() -> None:
    # Voxel centres are at the affine's positions, so y = j is row j exactly.
    voxels = voxelise([straight([0.0, 7.0, 3.0], [19.0, 7.0, 3.0])], GRID)
    i, j, k = np.unravel_index(voxels.voxel, GRID.shape)
    assert set(j.tolist()) == {7} and set(k.tolist()) == {3}
    assert set(i.tolist()) == set(range(20))


def test_voxelised_mask_equals_track_density_on_the_fine_points() -> None:
    bundle = create_bundle(
        BundleGeometry(np.array([[2.0, 3, 4], [10, 12, 9], [17, 15, 16]]), seed=1)
    )
    voxels = voxelise([bundle], GRID)
    mask = np.zeros(int(np.prod(GRID.shape)), dtype=bool)
    mask[voxels.voxel] = True
    fine = resample(bundle, step_size=0.1)
    density = track_density(fine, GRID, upsample=1)
    np.testing.assert_array_equal(mask.reshape(GRID.shape), density.data > 0)


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
