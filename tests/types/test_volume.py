import copy
import itertools
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from larmor.types import Volume

SHAPE = (2, 3, 4)
SHAPE_4D = (2, 3, 4, 3)

# Voxel sizes 2, 2, 3 mm, axes along L, P, S, with a translation so that a flip
# that forgets to move the origin is caught.
LPS = np.array(
    [
        [-2.0, 0.0, 0.0, 10.0],
        [0.0, -2.0, 0.0, -20.0],
        [0.0, 0.0, 3.0, 5.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Axes along A, S, L: a cyclic permutation, which is not its own inverse, so a
# permutation applied the wrong way round is caught.
ASL = np.array(
    [
        [0.0, 0.0, -2.0, 1.0],
        [2.0, 0.0, 0.0, 2.0],
        [0.0, 3.0, 0.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _rotation_z(degrees: float) -> npt.NDArray[np.float64]:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    rotation = np.eye(4)
    rotation[:2, :2] = [[c, -s], [s, c]]
    return rotation


def _rotation_x(degrees: float) -> npt.NDArray[np.float64]:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    rotation = np.eye(4)
    rotation[1:3, 1:3] = [[c, -s], [s, c]]
    return rotation


# LPS turned 20 degrees about z: still closest to L, P, S, with a remaining rotation.
OBLIQUE_LPS = _rotation_z(20.0) @ LPS
# RAS turned 70 degrees about z: axis 0 now points closer to A than to R.
OBLIQUE_ALS = _rotation_z(70.0) @ np.diag([2.0, 2.0, 3.0, 1.0])
# ASL turned 20 degrees about z and 10 about x: oblique in two planes, with every
# column's largest component well clear of the next, so no orientation is a tie.
OBLIQUE_ASL = _rotation_x(10.0) @ _rotation_z(20.0) @ ASL

# Double oblique: axes 0 and 1 both have their largest component along x, so
# choosing a world axis for each voxel axis on its own gives both x. The small turns
# about x and z leave no ties (each column's largest component beats the next by at
# least 0.09), so the answer does not depend on rounding. nibabel's aff2axcodes
# gives S, L, P too.
_H = np.sqrt(0.5)
_DOUBLE_OBLIQUE_BASE = np.eye(4)
_DOUBLE_OBLIQUE_BASE[:3, :3] = np.array(
    [[_H, 0.5, 0.5], [-_H, 0.5, 0.5], [0.0, -_H, _H]]
).T
DOUBLE_OBLIQUE = _rotation_z(5.0) @ _rotation_x(10.0) @ _DOUBLE_OBLIQUE_BASE

# Real-world affine from nibabel issue #1449: in the third row, columns 0 and 2 are
# both the largest, and a labelling that goes column by column calls the RAS result
# S, A, L. nibabel's test_io_orientation_column_strength_regression expects I, A, R.
NIBABEL_1449 = np.array(
    [
        [2.08759499e00, 7.70245194e-02, 1.12271041e-01, -1.67531357e01],
        [-1.04219818e00, 1.58019245e-01, -5.34135476e-02, 1.22405062e01],
        [-2.33361936e00, -1.66752085e-03, 1.24289364e-01, -1.09963446e01],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)

# Every orientation: each of the 6 axis orders with each of the 8 sign choices.
ALL_ORIENTATIONS = [
    "".join(letters)
    for axes in itertools.permutations(["RL", "AP", "SI"])
    for letters in itertools.product(*axes)
]


def _volume(affine: npt.NDArray[np.float64], shape: tuple[int, ...] = SHAPE) -> Volume:
    # Distinct values, so every voxel can be traced after a reorientation.
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return Volume(data=data, affine=affine.copy())


def _world(
    affine: npt.NDArray[np.float64], ijk: tuple[int, ...]
) -> npt.NDArray[np.float64]:
    point: npt.NDArray[np.float64] = affine @ np.array([*ijk, 1.0])
    return point


def _assert_same_anatomy(before: Volume, after: Volume) -> None:
    """Every voxel of `before` has the same values at the same world point in `after`.

    For 4D data the whole series is compared, so `data[..., n]` must hold the same
    anatomy before and after.
    """
    assert after.data.ndim == before.data.ndim
    assert after.data.shape[3:] == before.data.shape[3:]
    inverse = np.linalg.inv(after.affine)
    for ijk in itertools.product(*(range(n) for n in before.data.shape[:3])):
        new = inverse @ _world(before.affine, ijk)
        new_ijk = tuple(int(i) for i in np.rint(new[:3]))
        np.testing.assert_allclose(new[:3], new_ijk, atol=1e-9)
        # Negative indices would wrap around silently.
        assert all(0 <= i < n for i, n in zip(new_ijk, after.data.shape, strict=False))
        np.testing.assert_array_equal(after.data[new_ijk], before.data[ijk])


# Shapes


def test_3d_data_stays_3d() -> None:
    volume = Volume(data=np.zeros(SHAPE), affine=np.eye(4))
    assert volume.data.shape == SHAPE


@pytest.mark.parametrize("n", [1, 3])
def test_4d_data_stays_4d(n: int) -> None:
    volume = Volume(data=np.zeros((*SHAPE, n)), affine=np.eye(4))
    assert volume.data.shape == (*SHAPE, n)


def test_5d_data_is_rejected() -> None:
    with pytest.raises(ValueError, match="5D"):
        Volume(data=np.zeros((*SHAPE, 2, 2)), affine=np.eye(4))


@pytest.mark.parametrize("data_shape", [(), (2,), (2, 3)], ids=["0D", "1D", "2D"])
def test_fewer_than_3_dimensions_is_rejected(data_shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        Volume(data=np.zeros(data_shape), affine=np.eye(4))


# Affine. The checks are shared with Grid and tested in detail in test_grid.py;
# these cases only show that Volume runs them.


def test_affine_that_is_not_finite_is_rejected() -> None:
    affine = LPS.copy()
    affine[1, 3] = np.nan
    with pytest.raises(ValueError):
        Volume(data=np.zeros(SHAPE), affine=affine)


def test_singular_affine_is_rejected() -> None:
    affine = LPS.copy()
    affine[:3, 1] = 2.0 * affine[:3, 0]
    with pytest.raises(ValueError, match="singular"):
        Volume(data=np.zeros(SHAPE), affine=affine)


def test_integer_affine_becomes_float64() -> None:
    volume = Volume(data=np.zeros(SHAPE), affine=LPS.astype(np.int64))
    assert volume.affine.dtype == np.float64
    np.testing.assert_array_equal(volume.affine, LPS)


def test_changing_callers_affine_does_not_change_volume() -> None:
    affine = LPS.copy()
    volume = Volume(data=np.zeros(SHAPE), affine=affine)
    affine[0, 3] = 99.0
    np.testing.assert_array_equal(volume.affine, LPS)


def test_volumes_compare_by_identity() -> None:
    volume = _volume(LPS)
    assert volume == volume
    assert volume != copy.copy(volume)


# Read-only arrays


def _write_into(volume: Volume, name: str) -> None:
    array: np.ndarray = getattr(volume, name)
    array[(0,) * array.ndim] = 1.0


@pytest.mark.parametrize("name", ["data", "affine"])
def test_arrays_are_read_only(name: str) -> None:
    volume = _volume(LPS)
    with pytest.raises(ValueError, match="read-only"):
        _write_into(volume, name)


@pytest.mark.parametrize("name", ["data", "affine"])
def test_arrays_of_with_data_volume_are_read_only(name: str) -> None:
    volume = _volume(LPS).with_data(np.ones(SHAPE))
    with pytest.raises(ValueError, match="read-only"):
        _write_into(volume, name)


# to_orientation and to_closest_canonical only call these two, so each array is
# checked after each of them.
@pytest.mark.parametrize("name", ["data", "affine"])
@pytest.mark.parametrize("method", ["flip", "permute"])
def test_arrays_stay_read_only_after_reorientation(name: str, method: str) -> None:
    volume = _volume(LPS)
    if method == "flip":
        volume.flip_axis(0)
    else:
        volume.permute_axes((2, 0, 1))
    with pytest.raises(ValueError, match="read-only"):
        _write_into(volume, name)


REORIENTATIONS: dict[str, Callable[[Volume], None]] = {
    "flip": lambda v: v.flip_axis(0),
    "permute": lambda v: v.permute_axes((2, 0, 1)),
    "to_orientation": lambda v: v.to_orientation("RAS"),
}


@pytest.mark.parametrize("reorient", REORIENTATIONS.values(), ids=REORIENTATIONS)
def test_reorienting_does_not_change_a_copy_taken_before(
    reorient: Callable[[Volume], None],
) -> None:
    volume = _volume(LPS)
    original = volume.data.copy()
    before = copy.copy(volume)
    reorient(volume)
    np.testing.assert_array_equal(before.data, original)
    np.testing.assert_array_equal(before.affine, LPS)


@pytest.mark.parametrize("reorient", REORIENTATIONS.values(), ids=REORIENTATIONS)
def test_reorienting_does_not_change_affine_of_volume_sharing_it(
    reorient: Callable[[Volume], None],
) -> None:
    volume = _volume(LPS)
    other = volume.with_data(np.zeros(SHAPE))
    reorient(volume)
    np.testing.assert_array_equal(other.affine, LPS)


@pytest.mark.parametrize("reorient", REORIENTATIONS.values(), ids=REORIENTATIONS)
def test_reorientation_does_not_copy_the_data(
    reorient: Callable[[Volume], None],
) -> None:
    volume = _volume(ASL)
    original = volume.data
    reorient(volume)
    assert np.shares_memory(volume.data, original)


# Derived properties


# ASL's row and column norms differ, so reading rows instead of columns is caught.
@pytest.mark.parametrize(
    ("affine", "sizes", "codes"),
    [
        (np.diag([1.5, 1.5, 1.5, 1.0]), [1.5, 1.5, 1.5], ("R", "A", "S")),
        (LPS, [2.0, 2.0, 3.0], ("L", "P", "S")),
        (ASL, [2.0, 3.0, 2.0], ("A", "S", "L")),
        (OBLIQUE_LPS, [2.0, 2.0, 3.0], ("L", "P", "S")),
        (OBLIQUE_ALS, [2.0, 2.0, 3.0], ("A", "L", "S")),
        (DOUBLE_OBLIQUE, [1.0, 1.0, 1.0], ("S", "L", "P")),
    ],
    ids=["RAS", "LPS", "ASL", "oblique-LPS", "oblique-ALS", "double-oblique"],
)
def test_voxel_sizes_and_orientation_follow_affine_columns(
    affine: npt.NDArray[np.float64], sizes: list[float], codes: tuple[str, str, str]
) -> None:
    volume = _volume(affine)
    np.testing.assert_allclose(volume.voxel_sizes, sizes)
    assert volume.orientation == codes


# At 45 degrees about z, axes 0 and 1 are equally close to x and y, and which one
# wins depends on the last bit of cos and sin. Either answer is right, as long as
# no world axis is used twice.
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_orientation_at_45_degrees_uses_each_world_axis_once(scale: float) -> None:
    affine = _rotation_z(45.0) @ np.diag([scale, scale, scale, 1.0])
    affine[:3, 3] = [10.0, -20.0, 5.0]
    codes = _volume(affine).orientation
    assert sorted("RLAPSI".index(c) // 2 for c in codes) == [0, 1, 2]
    assert codes[2] == "S"


# 4D, so a grid that takes the shape of the whole array is caught.
def test_grid_has_spatial_shape_and_affine() -> None:
    grid = _volume(LPS, SHAPE_4D).grid
    assert grid.shape == SHAPE
    np.testing.assert_array_equal(grid.affine, LPS)


# Reorientation


@pytest.mark.parametrize(
    ("axis", "expected"),
    [(0, ("R", "P", "S")), (1, ("L", "A", "S")), (2, ("L", "P", "I"))],
)
def test_flip_axis_reverses_data_and_orientation_letter(
    axis: int, expected: tuple[str, str, str]
) -> None:
    volume = _volume(LPS)
    original = volume.data.copy()
    volume.flip_axis(axis)
    np.testing.assert_array_equal(volume.data, np.flip(original, axis=axis))
    assert volume.orientation == expected


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_flip_axis_keeps_world_position_of_each_voxel(axis: int) -> None:
    before = _volume(LPS)
    after = _volume(LPS)
    after.flip_axis(axis)
    _assert_same_anatomy(before, after)


@pytest.mark.parametrize("axis", [3, -1])
def test_flip_axis_rejects_non_spatial_axis(axis: int) -> None:
    with pytest.raises(ValueError):
        _volume(LPS, SHAPE_4D).flip_axis(axis)


# The permutation tests use 4D data: the series axis must stay last.
def test_permute_axes_transposes_data_and_reorders_orientation() -> None:
    volume = _volume(LPS, SHAPE_4D)
    original = volume.data.copy()
    volume.permute_axes((2, 0, 1))
    np.testing.assert_array_equal(volume.data, original.transpose(2, 0, 1, 3))
    assert volume.orientation == ("S", "L", "P")


@pytest.mark.parametrize("order", [(1, 0, 2), (2, 0, 1), (1, 2, 0)])
def test_permute_axes_keeps_world_positions(order: tuple[int, int, int]) -> None:
    before = _volume(LPS, SHAPE_4D)
    after = _volume(LPS, SHAPE_4D)
    after.permute_axes(order)
    _assert_same_anatomy(before, after)


@pytest.mark.parametrize("order", [(0, 0, 1), (0, 1, 3), (0, 1)])
def test_permute_axes_rejects_invalid_order(order: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        _volume(LPS).permute_axes(order)


def test_to_orientation_reaches_every_orientation(subtests: pytest.Subtests) -> None:
    before = _volume(OBLIQUE_ASL, SHAPE_4D)
    for target in ALL_ORIENTATIONS:
        with subtests.test(target=target):
            after = _volume(OBLIQUE_ASL, SHAPE_4D)
            after.to_orientation(target)
            assert after.orientation == tuple(target)
            _assert_same_anatomy(before, after)


def test_to_orientation_current_orientation_changes_nothing() -> None:
    volume = _volume(LPS)
    original = volume.data.copy()
    volume.to_orientation("LPS")
    np.testing.assert_array_equal(volume.data, original)
    np.testing.assert_array_equal(volume.affine, LPS)


@pytest.mark.parametrize("codes", ["RRS", "RAX", "RLS", "RA", "RASL"])
def test_to_orientation_rejects_invalid_codes(codes: str) -> None:
    with pytest.raises(ValueError):
        _volume(LPS).to_orientation(codes)


def test_as_closest_canonical_gives_ras() -> None:
    before = _volume(ASL)
    after = _volume(ASL)
    after.to_closest_canonical()
    assert after.orientation == ("R", "A", "S")
    _assert_same_anatomy(before, after)


def test_orientation_of_nibabel_1449_affine() -> None:
    assert _volume(NIBABEL_1449).orientation == ("I", "A", "R")


def test_as_closest_canonical_of_nibabel_1449_affine_gives_ras() -> None:
    before = _volume(NIBABEL_1449)
    after = _volume(NIBABEL_1449)
    after.to_closest_canonical()
    assert after.orientation == ("R", "A", "S")
    assert after.data.shape == (4, 3, 2)
    _assert_same_anatomy(before, after)


def test_as_closest_canonical_of_nibabel_1449_affine_is_idempotent() -> None:
    volume = _volume(NIBABEL_1449)
    volume.to_closest_canonical()
    data, affine = volume.data.copy(), volume.affine.copy()
    volume.to_closest_canonical()
    np.testing.assert_array_equal(volume.data, data)
    np.testing.assert_array_equal(volume.affine, affine)
    assert volume.orientation == ("R", "A", "S")


# Scaling

INT_DATA = np.arange(-12, 12, dtype=np.int16).reshape(SHAPE)


def test_as_float_float32_applies_slope_and_intercept() -> None:
    volume = Volume(data=INT_DATA, affine=LPS.copy(), slope=2.5, intercept=-10.0)
    result = volume.as_float(dtype=np.float32)
    assert result.dtype == np.float32
    # Slope, intercept and data are exact in float32, so the comparison is exact.
    np.testing.assert_array_equal(result, INT_DATA.astype(np.float64) * 2.5 - 10.0)


def test_as_float_float64_applies_slope_and_intercept() -> None:
    volume = Volume(data=INT_DATA, affine=LPS.copy(), slope=0.1, intercept=-10.3)
    result = volume.as_float(dtype=np.float64)
    assert result.dtype == np.float64
    np.testing.assert_allclose(
        result, INT_DATA.astype(np.float64) * 0.1 - 10.3, rtol=1e-15, atol=1e-14
    )


@pytest.mark.parametrize(("slope", "intercept"), [(1.0, 5.0), (0.5, 0.0)])
def test_as_float_applies_partly_trivial_scaling(
    slope: float, intercept: float
) -> None:
    volume = Volume(data=INT_DATA, affine=LPS.copy(), slope=slope, intercept=intercept)
    np.testing.assert_array_equal(
        volume.as_float(dtype=np.float32), INT_DATA * slope + intercept
    )


# The intercept is not 0, so applying it alone (or a constant image) is caught.
@pytest.mark.parametrize("slope", [0.0, np.nan, np.inf], ids=["zero", "nan", "inf"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_zero_or_nan_slope_means_no_scaling(
    slope: float, dtype: type[np.floating]
) -> None:
    volume = Volume(data=INT_DATA, affine=LPS.copy(), slope=slope, intercept=3.0)
    result = volume.as_float(dtype=dtype)
    assert result.dtype == dtype
    np.testing.assert_array_equal(result, INT_DATA)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_matching_type_and_trivial_scaling_returns_data_without_copy(
    dtype: type[np.floating],
) -> None:
    data = np.arange(24, dtype=dtype).reshape(SHAPE)
    volume = Volume(data=data, affine=LPS.copy())
    assert np.shares_memory(volume.as_float(dtype=dtype), volume.data)


# Each case breaks one of the conditions for returning the data as is.
@pytest.mark.parametrize(
    ("dtype", "slope", "intercept"),
    [(np.float64, 1.0, 0.0), (np.float32, 2.0, 0.0), (np.float32, 1.0, 1.0)],
    ids=["other-type", "slope", "intercept"],
)
def test_as_float_converts_when_data_cannot_be_returned_as_is(
    dtype: type[np.floating], slope: float, intercept: float
) -> None:
    data = np.arange(24, dtype=dtype).reshape(SHAPE)
    volume = Volume(data=data, affine=LPS.copy(), slope=slope, intercept=intercept)
    result = volume.as_float(dtype=np.float32)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, data * slope + intercept)


def test_as_float_defaults_to_float32() -> None:
    volume = Volume(data=INT_DATA, affine=LPS.copy(), slope=2.0)
    assert volume.as_float().dtype == np.float32


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.bool_])
def test_as_float_rejects_non_float_type(dtype: type[np.generic]) -> None:
    volume = Volume(data=INT_DATA, affine=LPS.copy(), slope=0.5)
    with pytest.raises(ValueError, match="floating-point"):
        volume.as_float(dtype=dtype)


@pytest.mark.parametrize("intercept", [np.nan, np.inf])
def test_non_finite_intercept_with_a_slope_is_rejected(intercept: float) -> None:
    with pytest.raises(ValueError, match="intercept"):
        Volume(data=INT_DATA, affine=LPS.copy(), slope=2.0, intercept=intercept)


@pytest.mark.parametrize("new_shape", [(9, 9, 9), (3, 2, 4), (2, 3, 4 * 3)])
def test_with_data_rejects_other_spatial_shape(new_shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="spatial shape"):
        _volume(LPS).with_data(np.zeros(new_shape))


# 4D data on a 3D volume: only the spatial shape has to agree.
def test_with_data_keeps_affine_and_resets_scaling() -> None:
    volume = Volume(data=np.zeros(SHAPE), affine=LPS.copy(), slope=2.5, intercept=-1.0)
    new_data = np.arange(np.prod(SHAPE_4D), dtype=np.float32).reshape(SHAPE_4D)
    new = volume.with_data(new_data)
    np.testing.assert_array_equal(new.affine, LPS)
    assert new.data.shape == SHAPE_4D
    np.testing.assert_array_equal(new.data, new_data)
    assert (new.slope, new.intercept) == (1.0, 0.0)
    np.testing.assert_array_equal(new.as_float(dtype=np.float32), new_data)


def test_as_float_refuses_complex_data() -> None:
    volume = Volume(data=np.ones(SHAPE, dtype=np.complex64), affine=LPS.copy())
    with pytest.raises(ValueError, match="complex"):
        volume.as_float()
