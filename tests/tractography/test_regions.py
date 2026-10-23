import warnings

import numpy as np
import numpy.typing as npt
import pytest

from larmor.tractography import (
    Sphere,
    crop,
    passes_in_order,
    passes_through,
    points_inside,
)
from larmor.tractography import regions as regions_module
from larmor.types import Tractogram, Volume


def _line_mask(
    values: list[float], axis: int, dtype: npt.DTypeLike = np.float32
) -> Volume:
    # The values lie along one axis, the other two axes have one voxel.
    shape = [1, 1, 1]
    shape[axis] = len(values)
    data = np.asarray(values, dtype=dtype).reshape(shape)
    return Volume(data=data, affine=np.eye(4))


def _on_axis(coordinates: list[float], axis: int) -> np.ndarray:
    points = np.zeros((len(coordinates), 3), dtype=np.float32)
    points[:, axis] = coordinates
    return points


def test_mask_value_is_inside_from_one_half_in_magnitude() -> None:
    mask = _line_mask([0.49, 0.5, -0.5, -0.49, 0.0, np.nan], axis=0)
    inside = points_inside(_on_axis([0, 1, 2, 3, 4, 5], axis=0), mask)
    np.testing.assert_array_equal(inside, [False, True, True, False, False, False])


@pytest.mark.parametrize(
    ("values", "slope", "intercept", "expected"),
    [
        ([0, 3, -1], 1.0, 0.0, [False, True, True]),
        ([1, 2, -2], 0.25, 0.0, [False, True, True]),
        ([0, -1, 2], 1.0, 0.6, [True, False, True]),
    ],
)
def test_integer_mask_is_tested_on_its_scaled_values(
    values: list[int], slope: float, intercept: float, expected: list[bool]
) -> None:
    data = np.asarray(values, dtype=np.int16).reshape(3, 1, 1)
    mask = Volume(data=data, affine=np.eye(4), slope=slope, intercept=intercept)
    inside = points_inside(_on_axis([0, 1, 2], axis=0), mask)
    np.testing.assert_array_equal(inside, expected)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_points_round_to_the_nearest_voxel_halves_away_from_zero(axis: int) -> None:
    # Voxel 1 is empty, so a half rounded to even (0.5 -> 0) shows; -0.5 and 2.5
    # round to the voxels -1 and 3, outside the image.
    mask = _line_mask([1, 0, 1], axis=axis, dtype=np.uint8)
    coordinates = [-0.5, -0.49, 0.5, 1.5, 2.49, 2.5]
    inside = points_inside(_on_axis(coordinates, axis), mask)
    np.testing.assert_array_equal(inside, [False, True, False, True, True, False])


def test_mask_must_be_3d() -> None:
    mask = Volume(data=np.ones((2, 2, 2, 2), dtype=np.uint8), affine=np.eye(4))
    with pytest.raises(ValueError, match="3D"):
        points_inside(np.zeros((1, 3), dtype=np.float32), mask)


def test_points_must_be_p_by_3() -> None:
    with pytest.raises(ValueError, match="shape"):
        points_inside(np.zeros((4, 2)), Sphere(np.zeros(3), 1.0))


def test_sphere_contains_points_on_its_surface() -> None:
    sphere = Sphere(np.array([1.0, 2.0, 3.0]), 2.0)
    points = np.array([[3, 2, 3], [1, 2, 1], [1, 4.01, 3], [1, 2, 3]], dtype=np.float32)
    np.testing.assert_array_equal(
        points_inside(points, sphere), [True, True, False, True]
    )


@pytest.mark.parametrize(
    ("centre", "radius"),
    [
        ([0.0, 0.0], 1.0),
        ([0.0, np.nan, 0.0], 1.0),
        ([0.0, np.inf, 0.0], 1.0),
        ([0.0, 0.0, 0.0], -1.0),
        ([0.0, 0.0, 0.0], np.inf),
        ([0.0, 0.0, 0.0], np.nan),
    ],
)
def test_sphere_rejects_invalid_centre_or_radius(
    centre: list[float], radius: float
) -> None:
    with pytest.raises(ValueError, match="sphere"):
        Sphere(np.array(centre), radius)


def test_sphere_centre_is_a_read_only_copy() -> None:
    centre = np.array([1.0, 2.0, 3.0])
    sphere = Sphere(centre, 0.0)
    centre[0] = 9.0
    assert sphere.centre[0] == 1.0
    with pytest.raises(ValueError, match="read-only"):
        sphere.centre[0] = 5.0


def _oblique_mask() -> Volume:
    rng = np.random.default_rng(1)
    affine = np.array(
        [[0, -2.0, 0.2, 10], [1.5, 0, 0, -8], [0, 0.3, -2.5, 6], [0, 0, 0, 1]]
    )
    return Volume(data=rng.uniform(-1, 1, size=(6, 7, 5)), affine=affine)


@pytest.mark.parametrize("operation", ["points_inside", "passes", "order", "crop"])
def test_mask_is_prepared_once_per_pass_not_per_chunk(
    operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mask = _oblique_mask()
    # Voxel positions in and around the image, mapped to world coordinates.
    voxels = np.random.default_rng(2).uniform(-1, 7, size=(60, 3))
    points = voxels @ mask.affine[:3, :3].T + mask.affine[:3, 3]
    tractogram = Tractogram(points=points, offsets=np.arange(0, 61, 6))

    def run() -> np.ndarray:
        if operation == "points_inside":
            return points_inside(points, mask)
        if operation == "passes":
            return passes_through(tractogram, mask)
        if operation == "order":
            return passes_in_order(tractogram, [mask])
        return crop(tractogram, mask).points

    calls = []
    prepare = regions_module._nonzero_voxels

    def counted(volume: Volume) -> np.ndarray:
        calls.append(volume)
        return prepare(volume)

    monkeypatch.setattr(regions_module, "_nonzero_voxels", counted)
    whole = run()
    calls_whole = len(calls)
    monkeypatch.setattr(regions_module, "CHUNK_POINTS", 4)
    chunked = run()

    # Once per pass over the points, however many chunks the pass has.
    assert len(calls) - calls_whole == calls_whole
    np.testing.assert_array_equal(chunked, whole)


def test_nan_points_are_outside_a_mask_without_warnings() -> None:
    mask = Volume(data=np.ones((3, 3, 3), dtype=np.uint8), affine=np.eye(4))
    points = np.array(
        [[np.nan, 1, 1], [1, np.nan, 1], [1, 1, np.nan], [1, 1, 1]], dtype=np.float32
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        inside = points_inside(points, mask)
    np.testing.assert_array_equal(inside, [False, False, False, True])


def test_region_of_another_type_is_a_type_error() -> None:
    sphere = Sphere(np.zeros(3), 1.0)
    tractogram = Tractogram(points=np.zeros((2, 3)), offsets=np.array([0, 2]))
    with pytest.raises(TypeError, match="Sphere or a mask Volume"):
        passes_through(tractogram, [sphere])  # type: ignore[arg-type]


def test_mask_with_one_volume_on_a_fourth_axis_is_a_3d_mask() -> None:
    data = np.zeros((3, 3, 3), dtype=np.float32)
    data[1, 1, 1] = 0.7
    points = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.float32)
    flat = points_inside(points, Volume(data=data, affine=np.eye(4)))
    stacked = points_inside(points, Volume(data=data[..., None], affine=np.eye(4)))
    np.testing.assert_array_equal(stacked, flat)
    np.testing.assert_array_equal(stacked, [True, False])
