from pathlib import Path

import numpy as np
import pytest

from larmor.io import load_tractogram, load_volume, read_streamline_weights
from larmor.tractography import (
    build_connectome,
    endpoint_nodes,
    node_names,
    parcellation_labels,
    streamline_lengths,
    write_connectome,
    write_node_assignments,
)
from larmor.types import Label, Tractogram, Volume

DATA = Path(__file__).parents[1] / "data" / "connectome"


@pytest.mark.parametrize(
    ("case", "radius", "weighted", "mean_length", "symmetric"),
    [
        ("default", 4.0, False, False, False),
        ("end_voxels", None, False, False, False),
        ("radius_2", 2.0, False, False, False),
        ("weighted", 4.0, True, False, False),
        ("mean_length", 4.0, True, True, False),
        ("symmetric", 4.0, False, False, True),
    ],
)
def test_connectome_matches_tck2connectome(
    connectome_ref: dict[str, np.ndarray],
    case: str,
    radius: float | None,
    weighted: bool,
    mean_length: bool,
    symmetric: bool,
) -> None:
    tracks = load_tractogram(DATA / "tracks.tck")
    parcellation = load_volume(DATA / "parcellation.nii.gz")
    weights = read_streamline_weights(DATA / "weights.txt") if weighted else None
    values = streamline_lengths(tracks) if mean_length else None

    nodes = endpoint_nodes(tracks, parcellation, radius=radius)
    matrix = build_connectome(
        nodes,
        int(parcellation_labels(parcellation).max()),
        weights=weights,
        values=values,
        symmetric=symmetric,
        zero_diagonal=symmetric,
    )

    np.testing.assert_array_equal(nodes, connectome_ref[f"nodes_{case}"])
    # Some endpoints are outside the cropped parcellation or far from any region.
    assert (nodes == 0).any()
    # tck2connectome computes lengths in float32.
    np.testing.assert_allclose(matrix, connectome_ref[f"matrix_{case}"], rtol=1e-6)


def test_the_endpoint_voxel_wins_over_equally_near_voxels() -> None:
    # The endpoint is at the corner shared by eight voxels; rounding takes voxel
    # (1, 1, 1), and the radial search keeps it although (0, 0, 0) is as near.
    labels = np.zeros((2, 2, 2), np.uint8)
    labels[0, 0, 0] = 1
    labels[1, 1, 1] = 2
    tracks = Tractogram(
        np.array([[0.5, 0.5, 0.5], [5.0, 5.0, 5.0]], np.float32), np.array([0, 2])
    )
    nodes = endpoint_nodes(tracks, Volume(labels, np.eye(4)), radius=4.0)
    np.testing.assert_array_equal(nodes, [[2, 0]])
    # Outside the image the radial search still looks for a centre within the
    # radius: (1, 1, 1) is sqrt(48) = 6.93 mm from (5, 5, 5).
    for radius, label in ((6.9, 0), (7.0, 2)):
        far = endpoint_nodes(tracks, Volume(labels, np.eye(4)), radius=radius)
        np.testing.assert_array_equal(far, [[2, label]])


def test_mean_values_are_weighted_and_empty_edges_are_zero() -> None:
    nodes = np.array([[1, 2], [2, 1], [3, 3], [0, 1]])
    matrix = build_connectome(
        nodes,
        3,
        weights=np.array([1.0, 3.0, 2.0, 5.0]),
        values=np.array([10.0, 20.0, 7.0, 100.0]),
        symmetric=True,
    )
    expected = np.array([[0, 17.5, 0], [17.5, 0, 0], [0, 0, 7.0]])
    np.testing.assert_allclose(matrix, expected)


def test_saved_matrices_keep_the_names(tmp_path: Path) -> None:
    matrix = np.array([[1.0, 2.5], [0.0, 1e-12]])
    write_connectome(matrix, tmp_path / "m.csv", names=["Left-A", "Right-B"])
    write_connectome(matrix, tmp_path / "m.txt")
    lines = (tmp_path / "m.csv").read_text().splitlines()
    assert lines[0] == ",Left-A,Right-B"
    assert lines[1] == "Left-A,1,2.5"
    np.testing.assert_array_equal(
        np.loadtxt(tmp_path / "m.csv", delimiter=",", skiprows=1, usecols=(1, 2)),
        matrix,
    )
    assert (tmp_path / "m.txt").read_text().startswith("# names: 1 2\n")
    np.testing.assert_array_equal(np.loadtxt(tmp_path / "m.txt"), matrix)
    write_node_assignments(np.array([[1, 0], [3, 2]]), tmp_path / "a.txt")
    assert (tmp_path / "a.txt").read_text() == "1 0\n3 2\n"
    with pytest.raises(ValueError, match="names"):
        write_connectome(matrix, tmp_path / "x.csv", names=["has space", "B"])


def test_parcellation_labels_round_float_values_and_refuse_negative_ones() -> None:
    data = np.array([0.0, 1.9999, 3.0, 2.2], np.float32).reshape(1, 2, 2)
    labels = parcellation_labels(Volume(data, np.eye(4)))
    np.testing.assert_array_equal(labels.ravel(), [0, 2, 3, 2])
    with pytest.raises(ValueError, match="0 or more"):
        parcellation_labels(Volume(-data, np.eye(4)))


@pytest.mark.parametrize(
    ("nodes", "arguments"),
    [
        (np.array([[1, 4]]), {}),
        (np.array([1, 2]), {}),
        (np.array([[1, 2]]), {"weights": np.ones(2)}),
        (np.array([[1, 2]]), {"values": np.array([np.inf])}),
    ],
)
def test_invalid_connectome_inputs_are_refused(
    nodes: np.ndarray, arguments: dict[str, np.ndarray]
) -> None:
    with pytest.raises(ValueError):
        build_connectome(nodes, 3, **arguments)  # type: ignore[arg-type]


def test_node_names_cover_every_label_up_to_the_largest() -> None:
    labels = np.array([0, 3, 1, 3], np.uint8).reshape(1, 2, 2)
    table = {1: Label("Left-A", (1, 2, 3)), 7: Label("Unused", (0, 0, 0))}
    names = node_names(Volume(labels, np.eye(4)), lookup_table=table)
    assert names == ["Left-A", "2", "3"]
    assert node_names(Volume(labels, np.eye(4))) == ["1", "2", "3"]


def test_the_radius_applies_to_the_endpoint_voxel_too() -> None:
    # 4 mm voxels and a 1 mm radius: voxel (0, 0, 0) is labelled 7, its neighbour
    # (1, 0, 0), centred at x = 4, is labelled 3.
    labels = np.zeros((2, 1, 1), np.uint8)
    labels[0, 0, 0] = 7
    labels[1, 0, 0] = 3
    affine = np.diag([4.0, 4.0, 4.0, 1.0])
    points = np.array(
        [[0.5, 0.0, 0.0], [1.9, 0.0, 0.0], [3.5, 0.0, 0.0], [2.5, 0.0, 0.0]],
        np.float32,
    )
    tracks = Tractogram(points, np.array([0, 2, 4]))
    nodes = endpoint_nodes(tracks, Volume(labels, affine), radius=1.0)
    # 0.5 mm from its own centre: kept. 1.9 mm from its own centre and 2.1 mm from
    # the neighbour's: no centre within 1 mm. 3.5 is in the neighbour's voxel,
    # 0.5 mm from its centre. 2.5 is in the neighbour's voxel, 1.5 mm away.
    np.testing.assert_array_equal(nodes, [[7, 0], [3, 0]])
    without = endpoint_nodes(tracks, Volume(labels, affine), radius=None)
    np.testing.assert_array_equal(without, [[7, 7], [3, 3]])
