"""Applying transforms: round trips, carried data, streaming and reorientation."""

from pathlib import Path

import numpy as np
import pytest

from larmor.io import load_tractogram, save_tractogram
from larmor.registration import (
    AffineTransform,
    DisplacementField,
    TransformChain,
    TransformPair,
    read_ants_transform,
    transform_surface,
    transform_tractogram,
    transform_tractogram_file,
    transform_volume,
)
from larmor.types import Grid, Surface, Tractogram, Volume

DATA = Path(__file__).parents[1] / "data" / "registration"


def _rotation(axis: list[float], degrees: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    angle = np.deg2rad(degrees)
    rotation: np.ndarray = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k
    return rotation


def _affine(linear: np.ndarray, shift: list[float]) -> AffineTransform:
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = shift
    return AffineTransform(matrix)


def _grid(shape: tuple[int, int, int] = (12, 12, 12), size: float = 2.0) -> Grid:
    affine = np.diag([size, size, size, 1.0])
    affine[:3, 3] = -size * (np.array(shape) - 1) / 2
    return Grid(shape, affine)


def _linear_field() -> tuple[DisplacementField, np.ndarray, np.ndarray]:
    """A field d(p) = B p + c on a grid, with B and c."""
    grid = _grid((16, 16, 16), 2.0)
    b = np.array([[0.05, 0.02, -0.03], [0.01, -0.04, 0.02], [0.03, 0.0, 0.06]])
    c = np.array([1.0, -0.5, 0.8])
    ijk = np.stack(np.meshgrid(*map(np.arange, grid.shape), indexing="ij"), -1)
    world = ijk @ grid.affine[:3, :3].T + grid.affine[:3, 3]
    return DisplacementField(grid, world @ b.T + c), b, c


def _tractogram() -> Tractogram:
    rng = np.random.default_rng(1)
    points = rng.uniform(-5, 5, (9, 3))
    return Tractogram(
        points,
        np.array([0, 4, 6, 9]),
        point_data={"fa": np.arange(9.0)},
        streamline_data={"cluster": np.array([3, 1, 2])},
    )


def test_image_there_and_back_returns_a_linear_image() -> None:
    # Linear interpolation reproduces a linear function exactly, so going through
    # a transform and its inverse gives the input wherever both steps land inside.
    grid = _grid()
    ijk = np.stack(np.meshgrid(*map(np.arange, grid.shape), indexing="ij"), -1)
    world = ijk @ grid.affine[:3, :3].T + grid.affine[:3, 3]
    image = Volume((world @ [1.0, -2.0, 0.5] + 100).astype(np.float64), grid.affine)
    transform = read_ants_transform(DATA / "ants_affine.mat")
    back = transform_volume(
        transform_volume(image, transform, grid), transform.inverse(), grid
    )
    inner = (slice(3, -3),) * 3
    np.testing.assert_allclose(back.data[inner], image.data[inner], atol=1e-9)


def test_points_there_and_back_through_a_warp_chain() -> None:
    chain = TransformChain(
        [
            read_ants_transform(DATA / "ants_warp.nii.gz"),
            read_ants_transform(DATA / "ants_affine.mat"),
        ]
    )
    points = np.loadtxt(DATA / "points_fixed.txt")
    back = chain.inverse().map_points(chain.map_points(points))
    np.testing.assert_allclose(back, points, atol=2e-4)


def test_inverse_of_a_field_and_its_jacobian() -> None:
    field, b, c = _linear_field()
    y = np.array([[1.0, 2.0, -3.0], [-4.0, 0.5, 2.0]])
    expected = np.linalg.solve(np.eye(3) + b, (y - c).T).T
    np.testing.assert_allclose(field.inverse().map_points(y), expected, atol=1e-4)
    np.testing.assert_allclose(
        field.inverse().jacobians(y),
        np.broadcast_to(np.linalg.inv(np.eye(3) + b), (2, 3, 3)),
        atol=1e-9,
    )
    assert field.inverse().inverse() is field


def test_field_does_not_move_points_far_outside_its_grid() -> None:
    field, _, _ = _linear_field()
    far = np.array([[100.0, 0.0, 0.0]])
    np.testing.assert_array_equal(field.map_points(far), far)


def test_pair_inverse_is_the_given_inverse() -> None:
    forward = _affine(np.eye(3), [1.0, 0.0, 0.0])
    backward = _affine(np.eye(3), [-5.0, 0.0, 0.0])
    pair = TransformPair(forward, backward)
    assert pair.inverse().forward is backward


def test_tractogram_keeps_its_data_and_takes_the_reference() -> None:
    tractogram = _tractogram()
    reference = _grid()
    moved = transform_tractogram(
        tractogram, _affine(np.eye(3), [1.0, 2.0, 3.0]), reference=reference
    )
    np.testing.assert_allclose(moved.points, tractogram.points + [1, 2, 3], atol=1e-6)
    np.testing.assert_array_equal(moved.offsets, tractogram.offsets)
    np.testing.assert_array_equal(moved.point_data["fa"], tractogram.point_data["fa"])
    np.testing.assert_array_equal(
        moved.streamline_data["cluster"], tractogram.streamline_data["cluster"]
    )
    assert moved.reference is reference


def test_tractogram_point_outside_a_deformation_is_an_error() -> None:
    field, _, _ = _linear_field()
    displacements = field.displacements.copy()
    displacements[...] = np.nan
    undefined = DisplacementField(field.grid, displacements)
    with pytest.raises(ValueError, match="undefined"):
        transform_tractogram(_tractogram(), undefined)


@pytest.mark.parametrize("suffix", [".tck", ".trk", ".vtk"])
def test_tractogram_file_streamed_or_whole_equals_in_memory(
    tmp_path: Path, suffix: str
) -> None:
    tractogram = _tractogram()
    tractogram.point_data.clear()
    tractogram.streamline_data.clear()
    source = tmp_path / "in.tck"
    save_tractogram(tractogram, source)
    transform = _affine(_rotation([1, 2, 3], 20), [3.0, -1.0, 2.0])
    reference = _grid()
    out = tmp_path / f"out{suffix}"
    count = transform_tractogram_file(
        source, out, transform, reference=reference, batch_size=2
    )
    assert count == 3
    expected = transform_tractogram(load_tractogram(source), transform)
    np.testing.assert_allclose(load_tractogram(out).points, expected.points, atol=1e-4)


def test_surface_reflected_keeps_its_normals_out() -> None:
    surface = Surface(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
        vertex_data={"curv": np.arange(4.0)},
    )
    mirrored = transform_surface(surface, _affine(np.diag([-1.0, 1, 1]), [0, 0, 0]))
    np.testing.assert_array_equal(mirrored.faces, surface.faces[:, ::-1])
    np.testing.assert_array_equal(mirrored.vertex_data["curv"], np.arange(4.0))
    turned = transform_surface(surface, _affine(_rotation([0, 0, 1], 30), [0, 0, 0]))
    np.testing.assert_array_equal(turned.faces, surface.faces)


def _uniform(values: list[float], grid: Grid) -> Volume:
    data = np.broadcast_to(
        np.asarray(values, dtype=np.float32), (*grid.shape, len(values))
    )
    return Volume(np.ascontiguousarray(data), grid.affine)


def test_peaks_are_turned_back_by_the_transform() -> None:
    # The transform maps reference points to the image, p -> A p, so what lies
    # along v in the image lies along A^-1 v in the reference space, its length
    # kept. With a scaling in A, A^-1 differs from A^T.
    grid = _grid()
    linear = _rotation([1, 2, 2], 40) @ np.diag([1.3, 0.8, 1.1])
    v = np.array([0.0, 0.6, 0.8]) * 2.5
    peaks = _uniform([*v, np.nan, np.nan, np.nan], grid)
    out = transform_volume(
        peaks,
        _affine(linear, [0, 0, 0]),
        grid,
        interpolation="nearest",
        reorient="peaks",
    )
    turned = np.linalg.solve(linear, v)
    expected = turned / np.linalg.norm(turned) * 2.5
    np.testing.assert_allclose(out.data[6, 6, 6, :3], expected, atol=1e-5)
    assert np.isnan(out.data[6, 6, 6, 3:]).all()


def test_tensors_take_only_the_rotation_finite_strain() -> None:
    grid = _grid()
    rotation = _rotation([2, -1, 1], 35)
    linear = rotation @ np.diag([1.2, 0.9, 1.1])
    tensor = np.array([[1.7, 0.1, 0.0], [0.1, 0.3, 0.05], [0.0, 0.05, 0.2]]) * 1e-3
    elements = tensor[[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
    out = transform_volume(
        _uniform(list(elements), grid),
        _affine(linear, [0, 0, 0]),
        grid,
        reorient="tensor",
    )
    turned = rotation.T @ tensor @ rotation
    expected = turned[[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
    np.testing.assert_allclose(out.data[6, 6, 6], expected, rtol=1e-5)


def test_peaks_need_nearest_interpolation() -> None:
    grid = _grid()
    with pytest.raises(ValueError, match="nearest"):
        transform_volume(
            _uniform([1.0, 0, 0], grid),
            _affine(np.eye(3), [0, 0, 0]),
            grid,
            reorient="peaks",
        )


def test_chain_maps_and_differentiates_in_order() -> None:
    first = _affine(_rotation([1, 0, 0], 30), [1.0, 0, 0])
    second = _affine(np.diag([1.0, 2.0, 0.5]) @ _rotation([0, 1, 0], 20), [0, 2.0, 0])
    chain = TransformChain([first, second])
    combined = second.matrix @ first.matrix
    point = np.array([[1.0, -2.0, 3.0]])
    np.testing.assert_allclose(
        chain.map_points(point)[0], combined[:3, :3] @ point[0] + combined[:3, 3]
    )
    np.testing.assert_allclose(chain.jacobians(point)[0], combined[:3, :3])


@pytest.mark.parametrize("nearest", [False, True], ids=["linear", "nearest"])
def test_resampled_image_takes_the_reference_xform_codes(nearest: bool) -> None:
    image = Volume(np.arange(27, dtype=np.float32).reshape(3, 3, 3), np.eye(4))
    template = Grid((3, 3, 3), np.eye(4), xform_codes=(4, 4))

    out = transform_volume(
        image,
        AffineTransform(np.eye(4)),
        template,
        interpolation="nearest" if nearest else "linear",
    )

    assert out.xform_codes == (4, 4)
