from pathlib import Path

import numpy as np
import pytest
import reference_density

from larmor.io import (
    iter_tractogram,
    load_grid,
    load_tractogram,
    read_streamline_weights,
)
from larmor.tractography import density_grid, track_density
from larmor.types import Grid, Tractogram

DATA = Path(__file__).parents[1] / "data" / "connectome"


def _tckmap(
    connectome_ref: dict[str, np.ndarray], case: str
) -> tuple[Grid, np.ndarray]:
    """tckmap's grid (in RAS order) and its image for one case."""
    shape = tuple(int(n) for n in connectome_ref[f"tdi_{case}_shape"])
    image = np.zeros(int(np.prod(shape)))
    image[connectome_ref[f"tdi_{case}_index"]] = connectome_ref[f"tdi_{case}_value"]
    return Grid(
        (shape[0], shape[1], shape[2]), connectome_ref[f"tdi_{case}_affine"]
    ), image


@pytest.mark.parametrize(
    ("case", "voxel_size", "weighted", "ends_only"),
    [
        ("default", None, False, False),
        ("fine", 0.7, False, False),
        ("weighted", None, True, False),
        ("ends", None, False, True),
    ],
)
def test_density_matches_tckmap(
    connectome_ref: dict[str, np.ndarray],
    case: str,
    voxel_size: float | None,
    weighted: bool,
    ends_only: bool,
) -> None:
    tracks = load_tractogram(DATA / "tracks.tck")
    weights = read_streamline_weights(DATA / "weights.txt") if weighted else None
    grid = load_grid(DATA / "template.nii.gz")
    if voxel_size is not None:
        grid = density_grid(grid, voxel_size)
    expected_grid, expected = _tckmap(connectome_ref, case)
    # The template stores x from right to left; tckmap's image is compared in RAS.
    canonical = grid.canonical()
    assert canonical.shape == expected_grid.shape
    # tckmap stores the fine grid's affine in float32.
    np.testing.assert_allclose(canonical.affine, expected_grid.affine, atol=1e-5)

    # On the grid as tckmap stored it, with the upsampling it chose for this file.
    density = track_density(tracks, expected_grid, weights=weights, ends_only=ends_only)

    assert density.data.dtype == (np.float32 if weighted else np.uint32)
    assert expected.sum() > 0
    # tckmap sums the weights in float32.
    np.testing.assert_allclose(density.data.ravel(), expected, rtol=1e-6, atol=1e-5)


def test_compiled_density_matches_the_numpy_reference() -> None:
    tracks = load_tractogram(DATA / "tracks.tck")[:60]
    weights = np.linspace(0.1, 2.0, tracks.num_streamlines)
    # An oblique grid of anisotropic voxels, so every axis of the map is used.
    angle = np.deg2rad(20.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    affine = np.eye(4)
    affine[:3, :3] = rotation * [1.5, 1.2, 1.8]
    affine[:3, 3] = [-140.0, -130.0, 0.0]
    grid = Grid((90, 120, 70), affine)
    for ends_only in (False, True):
        density = track_density(
            tracks, grid, weights=weights, ends_only=ends_only, upsample=4
        )
        expected = reference_density.track_density(tracks, grid, 4, ends_only, weights)
        assert expected.sum() > 0
        np.testing.assert_allclose(density.data, expected, rtol=1e-6)


def test_ends_count_each_endpoint_and_points_count_once_per_streamline() -> None:
    points = np.array(
        [
            *([0.2, 0.2, 0.2], [3.0, 0.2, 0.2], [0.3, 0.2, 0.2]),  # back to its start
            [2.2, 2.2, 2.2],  # a single point
        ],
        dtype=np.float32,
    )
    # The second streamline is empty.
    tracks = Tractogram(points, np.array([0, 3, 3, 4]))
    grid = Grid((5, 5, 5), np.eye(4))
    ends = track_density(tracks, grid, ends_only=True).data
    counts = track_density(tracks, grid, upsample=1).data
    assert ends[0, 0, 0] == 2 and ends[2, 2, 2] == 1 and ends.sum() == 3
    assert counts[0, 0, 0] == 1 and counts[3, 0, 0] == 1 and counts.sum() == 3


def test_batches_and_threads_give_the_same_density() -> None:
    whole = load_tractogram(DATA / "tracks.tck")
    weights = read_streamline_weights(DATA / "weights.txt")
    grid = density_grid(load_grid(DATA / "template.nii.gz"), 1.3)
    single = track_density(whole, grid, weights=weights, n_jobs=1)
    batches = iter_tractogram(DATA / "tracks.tck", batch_size=70)
    batched = track_density(batches, grid, weights=weights, n_jobs=-1)
    np.testing.assert_array_equal(batched.data, single.data)


def _lines(step: float, count: int, y: float) -> Tractogram:
    """count straight streamlines of 11 points, step mm apart along a diagonal."""
    direction = np.array([1.0, 0.7, 0.3]) / np.linalg.norm([1.0, 0.7, 0.3])
    lines = [
        (np.arange(11)[:, None] * step * direction + [0.3, y + 0.1 * k, 0.2])
        for k in range(count)
    ]
    offsets = np.arange(count + 1) * 11
    return Tractogram(np.concatenate(lines).astype(np.float32), offsets)


def test_automatic_upsampling_fills_long_segments_among_short_ones() -> None:
    # 0.5 mm steps, then 10 mm ones: one ratio for the whole streamline, set by its
    # median spacing, would skip the voxels between the 10 mm points. Along an axis,
    # so that the line does not clip voxel corners shorter than the spacing.
    direction = np.array([1.0, 0.0, 0.0])
    distances = np.concatenate([np.arange(0.0, 20.0, 0.5), [20.0, 30.0, 40.0]])
    points = (distances[:, None] * direction + 0.3).astype(np.float32)
    tracks = Tractogram(points, np.array([0, len(points)]))
    grid = Grid((50, 50, 50), np.eye(4))
    dense = np.arange(0.0, 40.0, 0.01)[:, None] * direction + 0.3
    expected = np.zeros(grid.shape, np.uint32)
    voxels = np.unique(np.floor(dense + 0.5).astype(int), axis=0)
    expected[tuple(voxels.T)] = 1

    automatic = track_density(tracks, grid).data

    np.testing.assert_array_equal(automatic, expected)


@pytest.mark.parametrize(
    "arguments",
    [
        {"weights": np.ones(3)},
        {"weights": np.array([np.nan] * 200)},
        {"upsample": 0},
        {"n_jobs": 0},
    ],
)
def test_invalid_arguments_are_refused(arguments: dict[str, object]) -> None:
    tracks = load_tractogram(DATA / "tracks.tck")
    with pytest.raises(ValueError):
        track_density(tracks, load_grid(DATA / "template.nii.gz"), **arguments)  # type: ignore[arg-type]


def test_density_grid_keeps_the_field_of_view_corner() -> None:
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [10.0, 0.0, 0.0]
    grid = density_grid(Grid((5, 4, 3), affine), [0.7, 1.5, 3.0])
    assert grid.shape == (15, 6, 2)
    # In RAS order x starts at the reference's last voxel (x = 2 mm); each axis
    # starts at the outer corner (half an old voxel back) plus half a new voxel.
    corner = (
        np.array([10.0 - 4 * 2.0, 0.0, 0.0]) - 1.0 + 0.5 * np.array([0.7, 1.5, 3.0])
    )
    np.testing.assert_allclose(grid.affine[:3, 3], corner)
    np.testing.assert_allclose(np.diag(grid.affine)[:3], [0.7, 1.5, 3.0])
    with pytest.raises(ValueError):
        density_grid(grid, [1.0, -1.0, 1.0])
