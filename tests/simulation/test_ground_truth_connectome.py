import math
import warnings

import numpy as np
import pytest

from larmor.simulation import (
    Bundle,
    BundleGeometry,
    GroundTruthConnectome,
    Phantom,
    VoxelPartition,
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
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return ground_truth_connectome(_partition(*bundles))


def _straight(ys: list[float], *, reverse_odd: bool = False) -> Tractogram:
    x = np.linspace(3.0, 16.0, 30)
    lines = []
    for i, y in enumerate(ys):
        line = np.stack([x, np.full_like(x, y), np.full_like(x, 4.0)], axis=1)
        lines.append(line[::-1] if reverse_odd and i % 2 else line)
    return Tractogram(np.concatenate(lines), np.arange(0, 30 * len(ys) + 1, 30))


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


def test_a_bundle_looping_back_to_its_start_is_on_the_diagonal_with_a_warning() -> None:
    straight = _tube([[3, 2, 4], [16, 2, 4]], seed=1)
    loop = _tube([[3, 7, 4], [15, 7, 4], [3, 7.3, 4]], seed=2)
    with pytest.warns(UserWarning, match="Both ends of bundle bundle_2"):
        found = ground_truth_connectome(_partition(straight, loop))
    np.testing.assert_array_equal(found.bundle_nodes, [[1, 2], [3, 3]])
    assert found.node_names[2] == "bundle_2_start+bundle_2_end"
    assert found.connected[2, 2] == 1
    assert found.proportion[2, 2] > 0
    assert np.triu(found.proportion).sum() == pytest.approx(1.0)


def test_a_bundle_leaving_the_grid_ends_where_it_leaves() -> None:
    found = _connectome(_tube([[3, 6, 4], [30, 6, 4]]))
    np.testing.assert_array_equal(found.bundle_nodes, [[1, 2]])
    ends = np.argwhere(found.parcellation.data == 2)
    assert (ends[:, 0] == GRID.shape[0] - 1).all()


def test_a_bundle_outside_the_grid_connects_nothing_with_a_warning() -> None:
    inside = _tube([[3, 6, 4], [16, 6, 4]])
    outside = Bundle(
        Tractogram(np.array([[50.0, 50, 50], [60, 50, 50]]), np.array([0, 2]))
    )
    with pytest.warns(UserWarning, match="no point inside the grid"):
        found = ground_truth_connectome(_partition(inside, outside))
    np.testing.assert_array_equal(found.bundle_nodes, [[1, 2], [0, 0]])
    assert found.connected.shape == (2, 2)
    assert found.proportion[0, 1] == pytest.approx(1.0)


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


def test_bundle_names_are_made_safe_and_empty_ones_numbered() -> None:
    partition = _partition(
        _tube([[3, 3, 4], [16, 3, 4]]), _tube([[3, 8, 4], [16, 8, 4]])
    )
    found = ground_truth_connectome(partition, bundle_names=[" CST left/1 ", ""])
    assert found.node_names == (
        "CST_left_1_start",
        "CST_left_1_end",
        "bundle_2_start",
        "bundle_2_end",
    )


@pytest.mark.parametrize("names", [["a"], ["a", "b", "c"]])
def test_bundle_names_of_the_wrong_length_are_refused(names: list[str]) -> None:
    partition = _partition(
        _tube([[3, 3, 4], [16, 3, 4]]), _tube([[3, 8, 4], [16, 8, 4]])
    )
    with pytest.raises(ValueError, match="must name 2 bundles"):
        ground_truth_connectome(partition, bundle_names=names)
