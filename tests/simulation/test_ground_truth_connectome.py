import logging
import math

import numpy as np
import pytest

from larmor.simulation import (
    Bundle,
    BundleGeometry,
    GroundTruthConnectome,
    Phantom,
    VoxelPartition,
    create_bundle,
    ground_truth_connectome,
    partition_voxels,
)
from larmor.tractography import endpoint_nodes
from larmor.types import Grid, Tractogram

GRID = Grid((20, 12, 8), np.eye(4))


def _tube(
    points: list[list[float]], *, radius: float = 1.0, seed: int = 0, n: int = 30
) -> Bundle:
    geometry = BundleGeometry(
        np.array(points), num_streamlines=n, radius=radius, num_points=40, seed=seed
    )
    return Bundle(geometry)


def _partition(*bundles: Bundle) -> VoxelPartition:
    return partition_voxels(Phantom(GRID, bundles=bundles), seed=0, n_jobs=1)


def _connectome(*bundles: Bundle) -> GroundTruthConnectome:
    return ground_truth_connectome(_partition(*bundles))


def _quiet(caplog: pytest.LogCaptureFixture, *bundles: Bundle) -> GroundTruthConnectome:
    with caplog.at_level(logging.WARNING, logger="larmor"):
        found = _connectome(*bundles)
    assert caplog.records == []
    return found


def _tractogram(lines: list[np.ndarray]) -> Tractogram:
    offsets = np.cumsum([0, *(len(line) for line in lines)])
    return Tractogram(np.concatenate(lines), offsets)


def _line(start: list[float], stop: list[float]) -> np.ndarray:
    return np.linspace(start, stop, 30)


def _straight(ys: list[float], *, reverse_odd: bool = False) -> Tractogram:
    lines = []
    for i, y in enumerate(ys):
        line = _line([3.0, y, 4.0], [16.0, y, 4.0])
        lines.append(line[::-1] if reverse_odd and i % 2 else line)
    return _tractogram(lines)


def test_two_separate_equal_tubes_give_four_nodes_and_half_the_share_each() -> None:
    found = _connectome(_tube([[3, 3, 4], [16, 3, 4]]), _tube([[3, 8, 4], [16, 8, 4]]))
    assert found.node_names == (
        "bundle_1_start",
        "bundle_1_end",
        "bundle_2_start",
        "bundle_2_end",
    )
    np.testing.assert_array_equal(found.bundle_nodes, [[1, 2], [3, 4]])
    expected = np.zeros((4, 4))
    expected[0, 1] = expected[1, 0] = expected[2, 3] = expected[3, 2] = 1
    np.testing.assert_array_equal(found.connected, expected)
    # The tube fill is integrated on the voxels, so equal tubes differ by ~1%.
    np.testing.assert_allclose(found.proportion, 0.5 * expected, atol=0.02)
    assert np.triu(found.proportion).sum() == pytest.approx(1.0)
    labels = found.parcellation.data
    assert set(np.unique(labels).tolist()) == {0, 1, 2, 3, 4}
    assert labels[3, 3, 4] == 1 and labels[16, 8, 4] == 4


def test_bundles_starting_at_one_point_share_a_node() -> None:
    found = _connectome(
        _tube([[3, 6, 4], [10, 3, 4], [16, 2, 4]], seed=1),
        _tube([[3, 6, 4], [10, 9, 4], [16, 10, 4]], seed=2),
    )
    assert found.node_names == (
        "bundle_1_start+bundle_2_start",
        "bundle_1_end",
        "bundle_2_end",
    )
    np.testing.assert_array_equal(found.bundle_nodes, [[1, 2], [1, 3]])
    assert found.connected[0, 1] == found.connected[0, 2] == 1
    assert found.connected[1, 2] == 0


def test_a_bundle_looping_back_to_its_start_is_on_the_diagonal_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    straight = _tube([[3, 2, 4], [16, 2, 4]], seed=1)
    # Each streamline returns one voxel over from where it left, so the
    # bundle's start and end voxels largely coincide.
    loop = Bundle(
        _tractogram(
            [
                np.concatenate(
                    [_line([3, y, 4], [15, y, 4]), _line([15, y, 4], [3, y + 1, 4])[1:]]
                )
                for y in (6.0, 7, 8, 9)
            ]
        )
    )
    with caplog.at_level(logging.WARNING, logger="larmor"):
        found = _connectome(straight, loop)
    assert "Both ends of bundle bundle_2 are in node 3" in caplog.text
    np.testing.assert_array_equal(found.bundle_nodes, [[1, 2], [3, 3]])
    assert found.node_names[2] == "bundle_2_start+bundle_2_end"
    assert found.connected[2, 2] == 1
    assert found.proportion[2, 2] > 0
    assert np.triu(found.proportion).sum() == pytest.approx(1.0)


def test_a_bundle_leaving_the_grid_ends_where_it_leaves(
    caplog: pytest.LogCaptureFixture,
) -> None:
    found = _quiet(caplog, _tube([[3, 6, 4], [30, 6, 4]]))
    np.testing.assert_array_equal(found.bundle_nodes, [[1, 2]])
    ends = np.argwhere(found.parcellation.data == 2)
    assert (ends[:, 0] == GRID.shape[0] - 1).all()


def test_a_bundle_outside_the_grid_connects_nothing_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    inside = _tube([[3, 6, 4], [16, 6, 4]])
    outside = Bundle(
        Tractogram(np.array([[50.0, 50, 50], [60, 50, 50]]), np.array([0, 2]))
    )
    with caplog.at_level(logging.WARNING, logger="larmor"):
        found = _connectome(inside, outside)
    assert "bundle_2 has no streamline inside the grid" in caplog.text
    np.testing.assert_array_equal(found.bundle_nodes, [[1, 2], [0, 0]])
    assert found.connected.shape == (2, 2)
    assert found.proportion[0, 1] == pytest.approx(1.0)


def test_a_streamline_clipping_a_corner_of_the_grid_is_left_out(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clean = [
        _line([3.0, 5 + 0.1 * i, 4.0], [16.0, 5 + 0.1 * i, 4.0]) for i in range(30)
    ]
    corner = np.array([[-3.0, 0, 0], [0.2, 0.2, 0.2], [-3, 5, 0]])
    found = _quiet(caplog, Bundle(_tractogram([*clean, corner])))
    assert found.node_names == ("bundle_1_start", "bundle_1_end")
    np.testing.assert_array_equal(found.bundle_nodes, [[1, 2]])
    assert found.parcellation.data[0, 0, 0] == 0


def test_a_stray_end_in_another_bundles_end_keeps_the_nodes_apart(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # B's stray streamline ends inside A's end, but moves B's end centre only
    # half a voxel, so the two ends stay farther apart than either's radius.
    spread = [(y, z) for y in (1.0, 2, 3, 4, 5) for z in (2.0, 3, 4, 5, 6)]
    a = _tractogram([_line([3, y, z], [16, y, z]) for y, z in spread])
    near = [(y, z) for y in (7.0, 8, 9) for z in (3.0, 4, 5)]
    b_lines = [_line([3, y, z], [16, y, z]) for y, z in near]
    b = _tractogram([*b_lines, _line([3.0, 8, 4], [16.0, 3, 4])])
    found = _quiet(caplog, Bundle(a), Bundle(b))
    np.testing.assert_array_equal(found.bundle_nodes, [[1, 2], [3, 4]])
    assert len(found.node_names) == 4
    labels = found.parcellation.data
    assert (labels[16, 1:6, 2:7] == 2).all()
    assert (labels[16, 7:10, 3:6] == 4).all()


def test_streamlines_stored_in_both_directions_give_one_start_and_one_end() -> None:
    tractogram = _straight([5.0, 5.5, 6.0, 6.5, 7.0], reverse_odd=True)
    found = _connectome(Bundle(tractogram))
    assert found.node_names == ("bundle_1_start", "bundle_1_end")
    np.testing.assert_array_equal(found.bundle_nodes, [[1, 2]])
    assert {
        int(found.parcellation.data[3, 6, 4]),
        int(found.parcellation.data[16, 6, 4]),
    } == {1, 2}


def test_the_share_follows_the_cross_section_not_the_streamline_count() -> None:
    many_thin = Bundle(_straight([3.0 + 0.01 * i for i in range(100)]))
    few_thick = _tube([[3, 8, 4], [16, 8, 4]], radius=2.5, n=10)
    found = _connectome(many_thin, few_thick)
    thin, thick = found.cross_section
    # The tube fill is integrated on the voxels, close to pi r^2.
    assert thick == pytest.approx(math.pi * 2.5**2, rel=0.05)
    assert found.proportion[2, 3] > found.proportion[0, 1]
    assert found.proportion[0, 1] / found.proportion[2, 3] == pytest.approx(
        thin / thick
    )


def test_endpoint_nodes_on_the_parcellation_give_each_bundle_its_two_nodes() -> None:
    partition = _partition(
        _tube([[3, 6, 4], [10, 3, 4], [16, 2, 4]], seed=1),
        _tube([[3, 6, 4], [10, 9, 4], [16, 10, 4]], seed=2),
        _tube([[16, 6, 4], [3, 6.5, 4]], radius=0.5, seed=3),
    )
    found = ground_truth_connectome(partition)
    for k, streamlines in enumerate(partition.streamlines):
        assigned = endpoint_nodes(streamlines, found.parcellation, n_jobs=1)
        expected = sorted(found.bundle_nodes[k].tolist())
        assert np.sort(assigned, axis=1).tolist() == [expected] * len(assigned)


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        ([" CST left/1 ", "", "AF"], ["CST_left_1", "bundle_2", "AF"]),
        (["CST", "AF", "CST"], ["CST_1", "AF", "CST_3"]),
    ],
)
def test_bundle_names_are_made_safe_numbered_when_empty_and_told_apart(
    names: list[str], expected: list[str]
) -> None:
    partition = _partition(
        _tube([[3, 2, 2], [16, 2, 2]]),
        _tube([[3, 6, 4], [16, 6, 4]]),
        _tube([[3, 10, 6], [16, 10, 6]]),
    )
    found = ground_truth_connectome(partition, bundle_names=names)
    assert found.node_names == tuple(
        f"{name}_{side}" for name in expected for side in ("start", "end")
    )


@pytest.mark.parametrize("names", [["a"], ["a", "b", "c"]])
def test_bundle_names_of_the_wrong_length_are_refused(names: list[str]) -> None:
    partition = _partition(
        _tube([[3, 3, 4], [16, 3, 4]]), _tube([[3, 8, 4], [16, 8, 4]])
    )
    with pytest.raises(ValueError, match="must name 2 bundles"):
        ground_truth_connectome(partition, bundle_names=names)


def _partition_on(grid: Grid, *bundles: Bundle) -> VoxelPartition:
    return partition_voxels(Phantom(grid, bundles=bundles), seed=0, n_jobs=1)


def test_a_drawn_loop_ending_where_it_starts_is_one_node_on_the_diagonal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    points = [[3, 8, 4], [11, 3.5, 4], [18, 8, 4], [11, 12.5, 4], [3, 8.3, 4]]
    geometry = BundleGeometry(np.array(points), num_streamlines=200, radius=1.5, seed=1)
    grid = Grid((22, 16, 8), np.eye(4))
    with caplog.at_level(logging.WARNING, logger="larmor"):
        found = ground_truth_connectome(
            _partition_on(grid, Bundle(geometry)), bundle_names=["U"]
        )
    assert found.node_names == ("U_start+U_end",)
    np.testing.assert_array_equal(found.bundle_nodes, [[1, 1]])
    assert "Both ends of bundle U are in node 1" in caplog.text


@pytest.mark.parametrize("angle", [60.0, 90.0])
def test_thick_bundles_leaving_one_hub_at_an_angle_share_their_start(
    angle: float, caplog: pytest.LogCaptureFixture
) -> None:
    hub = np.array([4.0, 12.0, 5.0])
    bundles = []
    for k, sign in enumerate((1, -1)):
        half = math.radians(angle / 2)
        direction = np.array([math.cos(half), sign * math.sin(half), 0.0])
        line = np.array([hub, hub + 7.5 * direction, hub + 15 * direction])
        geometry = BundleGeometry(
            line, num_streamlines=60, radius=3.0, num_points=40, seed=k
        )
        bundles.append(Bundle(geometry))
    grid = Grid((24, 24, 12), np.eye(4))
    with caplog.at_level(logging.WARNING, logger="larmor"):
        found = ground_truth_connectome(_partition_on(grid, *bundles))
    assert caplog.records == []
    assert len(found.node_names) == 3
    np.testing.assert_array_equal(found.bundle_nodes, [[1, 2], [1, 3]])


def test_parallel_tubes_whose_discs_touch_stay_four_nodes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    found = _quiet(
        caplog,
        _tube([[3, 4, 4], [16, 4, 4]], radius=2.0, seed=1),
        _tube([[3, 8, 4], [16, 8, 4]], radius=2.0, seed=2),
    )
    assert len(found.node_names) == 4
    np.testing.assert_array_equal(found.bundle_nodes, [[1, 2], [3, 4]])


def test_a_fanned_bundle_stored_in_mixed_directions_has_two_ends(
    caplog: pytest.LogCaptureFixture,
) -> None:
    geometry = BundleGeometry(
        np.array([[3.0, 6, 4], [16, 6, 4]]),
        num_streamlines=60,
        radius=1.0,
        num_points=40,
        fan_start=2.0,
        fan_end=2.0,
        seed=0,
    )
    drawn = create_bundle(geometry).points.reshape(60, 40, 3)
    drawn[::3] = drawn[::3, ::-1]
    tractogram = Tractogram(drawn.reshape(-1, 3), np.arange(0, 60 * 40 + 1, 40))
    found = _quiet(caplog, Bundle(tractogram))
    assert found.node_names == ("bundle_1_start", "bundle_1_end")
    np.testing.assert_array_equal(found.bundle_nodes, [[1, 2]])
