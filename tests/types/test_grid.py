import copy
import dataclasses
import itertools
from collections.abc import Callable

import numpy as np
import pytest

from larmor.types import Grid, Volume

SHAPE = (2, 3, 4)

AFFINE = np.array(
    [
        [-2.0, 0.0, 0.0, 10.0],
        [0.0, -2.0, 0.0, -20.0],
        [0.0, 0.0, 3.0, 5.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

_c, _s = np.cos(np.radians(20.0)), np.sin(np.radians(20.0))
# AFFINE turned 20 degrees about z: canonicalisation keeps the remaining rotation.
OBLIQUE = (
    np.array([[_c, -_s, 0, 0], [_s, _c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ AFFINE
)


@pytest.mark.parametrize(
    "shape", [(2, 3), (2, 3, 4, 5), (0, 3, 4), (2, -3, 4), (2.0, 3, 4), (True, 3, 4)]
)
def test_shape_that_is_not_three_positive_ints_is_rejected(
    shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="shape"):
        Grid(shape, AFFINE)  # type: ignore[arg-type]


def test_shape_of_numpy_ints_becomes_python_ints() -> None:
    grid = Grid((np.int64(2), np.int64(3), np.int64(4)), AFFINE)  # type: ignore[arg-type]
    assert grid.shape == SHAPE
    assert all(type(n) is int for n in grid.shape)


def _with(row: int, col: int, value: float) -> np.ndarray:
    affine = AFFINE.copy()
    affine[row, col] = value
    return affine


def _singular() -> np.ndarray:
    # Columns 0 and 1 parallel and both non-zero: no axis has zero length, yet the
    # three axes span only a plane.
    affine = AFFINE.copy()
    affine[:3, 1] = -1.5 * affine[:3, 0]
    return affine


# Grid and Volume share these checks; test_volume.py runs one of them through Volume.
@pytest.mark.parametrize(
    "affine",
    [
        np.eye(3),
        np.ones((3, 4)),
        np.ones((4, 3)),
        np.ones((4, 4, 1)),
        _with(0, 0, np.nan),
        _with(1, 3, np.inf),
        _with(3, 3, 2.0),
        _with(3, 0, 1.0),
        np.diag([1.0, 0.0, 1.0, 1.0]),
        _singular(),
    ],
    ids=[
        "3x3",
        "3x4",
        "4x3",
        "4x4x1",
        "nan-rotation",
        "inf-offset",
        "last-row-scale",
        "last-row-shear",
        "zero-axis",
        "singular",
    ],
)
def test_invalid_affine_is_rejected(affine: np.ndarray) -> None:
    with pytest.raises(ValueError):
        Grid(SHAPE, affine)


def test_singular_affine_error_names_the_problem() -> None:
    with pytest.raises(ValueError, match="singular"):
        Grid(SHAPE, _singular())


@pytest.mark.parametrize("dtype", [np.int64, np.float32])
def test_affine_becomes_float64(dtype: type[np.generic]) -> None:
    grid = Grid(SHAPE, AFFINE.astype(dtype))
    assert grid.affine.dtype == np.float64
    np.testing.assert_array_equal(grid.affine, AFFINE)


def test_changing_callers_affine_does_not_change_grid() -> None:
    affine = AFFINE.copy()
    grid = Grid(SHAPE, affine)
    affine[0, 3] = 99.0
    np.testing.assert_array_equal(grid.affine, AFFINE)


def test_identical_grids_match() -> None:
    assert Grid(SHAPE, AFFINE.copy()).matches(Grid(SHAPE, AFFINE.copy()))


# A TRK header stores its affine in float32: an oblique grid must still match itself.
def test_oblique_grid_matches_its_float32_copy() -> None:
    rounded = OBLIQUE.astype(np.float32).astype(np.float64)
    assert not np.array_equal(rounded, OBLIQUE)
    assert Grid(SHAPE, OBLIQUE).matches(Grid(SHAPE, rounded))


def test_grids_with_different_shape_do_not_match() -> None:
    assert not Grid(SHAPE, AFFINE).matches(Grid((2, 3, 5), AFFINE))


# The offsets are 0.5 * atol and 2 * atol, so the result depends on atol and not on
# numpy's default tolerance.
@pytest.mark.parametrize(
    ("offset", "atol", "expected"),
    [(0.5e-3, 1e-3, True), (2e-3, 1e-3, False), (0.5e-1, 1e-1, True)],
)
def test_affine_match_uses_atol(offset: float, atol: float, expected: bool) -> None:
    shifted = AFFINE.copy()
    shifted[0, 3] += offset
    grid = Grid(SHAPE, AFFINE)
    assert grid.matches(Grid(SHAPE, shifted), atol=atol) == expected


def test_default_atol_is_1e_3_mm() -> None:
    shifted = AFFINE.copy()
    shifted[0, 3] += 2e-3
    assert not Grid(SHAPE, AFFINE).matches(Grid(SHAPE, shifted))
    shifted[0, 3] = AFFINE[0, 3] + 0.5e-3
    assert Grid(SHAPE, AFFINE).matches(Grid(SHAPE, shifted))


def test_affine_is_read_only() -> None:
    grid = Grid(SHAPE, AFFINE.copy())
    with pytest.raises(ValueError, match="read-only"):
        grid.affine[0, 0] = 1.0


def test_grid_is_frozen() -> None:
    grid = Grid(SHAPE, AFFINE.copy())
    with pytest.raises(dataclasses.FrozenInstanceError):
        grid.shape = (1, 1, 1)  # type: ignore[misc]


def test_grid_of_volume_has_read_only_affine() -> None:
    grid = Volume(data=np.zeros(SHAPE), affine=AFFINE.copy()).grid
    with pytest.raises(ValueError, match="read-only"):
        grid.affine[0, 0] = 1.0


def _permute_and_flip(volume: Volume) -> None:
    volume.permute_axes((1, 2, 0))
    volume.flip_axis(1)


REORIENTATIONS: dict[str, Callable[[Volume], None]] = {
    "flip-0": lambda v: v.flip_axis(0),
    "flip-2": lambda v: v.flip_axis(2),
    "swap": lambda v: v.permute_axes((1, 0, 2)),
    "cyclic": lambda v: v.permute_axes((2, 0, 1)),
    "cyclic-and-flip": _permute_and_flip,
    "canonical": lambda v: v.to_closest_canonical(),
}


# A flip, a permutation that is not its own inverse, and both together.
NEGATIVE = {
    name: REORIENTATIONS[name] for name in ["flip-0", "cyclic", "cyclic-and-flip"]
}


def _grids(affine: np.ndarray, reorient: Callable[[Volume], None]) -> tuple[Grid, Grid]:
    # 4D data, as the grid takes only the spatial axes.
    volume = Volume(data=np.zeros((*SHAPE, 2), dtype=np.float32), affine=affine.copy())
    reoriented = copy.copy(volume)
    reorient(reoriented)
    return volume.grid, reoriented.grid


@pytest.mark.parametrize("affine", [AFFINE, OBLIQUE], ids=["straight", "oblique"])
@pytest.mark.parametrize("reorient", REORIENTATIONS.values(), ids=REORIENTATIONS)
def test_grids_differing_only_in_axis_order_and_direction_match(
    affine: np.ndarray, reorient: Callable[[Volume], None]
) -> None:
    original, reoriented = _grids(affine, reorient)
    assert original.matches(reoriented)
    assert reoriented.matches(original)


@pytest.mark.parametrize("reorient", NEGATIVE.values(), ids=NEGATIVE)
def test_reoriented_grid_with_other_voxel_sizes_does_not_match(
    reorient: Callable[[Volume], None],
) -> None:
    original, reoriented = _grids(AFFINE, reorient)
    scaled = reoriented.affine.copy()
    scaled[:3, 0] *= 1.1
    assert not original.matches(Grid(reoriented.shape, scaled))


@pytest.mark.parametrize("reorient", NEGATIVE.values(), ids=NEGATIVE)
def test_reoriented_grid_with_other_field_of_view_does_not_match(
    reorient: Callable[[Volume], None],
) -> None:
    original, reoriented = _grids(AFFINE, reorient)
    bigger = (reoriented.shape[0] + 1, reoriented.shape[1], reoriented.shape[2])
    assert not original.matches(Grid(bigger, reoriented.affine))


@pytest.mark.parametrize("reorient", NEGATIVE.values(), ids=NEGATIVE)
def test_reoriented_grid_at_other_position_does_not_match(
    reorient: Callable[[Volume], None],
) -> None:
    original, reoriented = _grids(AFFINE, reorient)
    moved = reoriented.affine.copy()
    moved[:3, 3] += [0.0, 1.0, 0.0]
    assert not original.matches(Grid(reoriented.shape, moved))


# Permuting the affine without the shape changes the field of view; flipping a
# column without moving the origin moves the grid. Neither is the same voxels.
@pytest.mark.parametrize(
    "columns", [[1, 0, 2, 3], [2, 0, 1, 3]], ids=["swap", "cyclic"]
)
def test_affine_permuted_without_shape_does_not_match(columns: list[int]) -> None:
    assert not Grid(SHAPE, AFFINE).matches(Grid(SHAPE, AFFINE[:, columns]))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_axis_flipped_without_moving_origin_does_not_match(axis: int) -> None:
    flipped = AFFINE.copy()
    flipped[:3, axis] *= -1
    assert not Grid(SHAPE, AFFINE).matches(Grid(SHAPE, flipped))


def _rotation(axis: int, degrees: float) -> np.ndarray:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    i, j = [k for k in range(3) if k != axis]
    rotation = np.eye(4)
    rotation[[i, i, j, j], [i, j, i, j]] = [c, -s, s, c]
    return rotation


# 45 degrees about z, then 45 about x: every column is an exact tie between two
# world axes, so the closest orientation depends on the last bit of cos and sin and
# flips under float32 rounding. Comparing the grids through their orientations then
# pairs the wrong axes. The voxel sizes differ, so pairing the wrong axes also
# gives a different affine.
TIE = _rotation(0, 45.0) @ _rotation(2, 45.0) @ np.diag([1.0, 2.0, 3.0, 1.0])
TIE[:3, 3] = [10.0, -20.0, 5.0]


@pytest.mark.parametrize("rounded", [False, True], ids=["exact", "float32"])
def test_grid_at_orientation_tie_matches_every_reorientation(
    rounded: bool, subtests: pytest.Subtests
) -> None:
    original = Grid(SHAPE, TIE)
    for order in itertools.permutations(range(3)):
        for flips in itertools.product([False, True], repeat=3):
            volume = Volume(data=np.zeros(SHAPE), affine=TIE)
            volume.permute_axes((order[0], order[1], order[2]))
            for axis in np.flatnonzero(flips):
                volume.flip_axis(int(axis))
            affine = volume.affine
            if rounded:
                affine = affine.astype(np.float32).astype(np.float64)
            other = Grid(volume.grid.shape, affine)
            with subtests.test(order=order, flips=flips):
                assert original.matches(other)
                assert other.matches(original)


@pytest.mark.parametrize("codes", ["RAS", "LPS", "ASR", "IPL"])
def test_grid_orientation_matches_the_volume_on_it(codes: str) -> None:
    vol = Volume(np.zeros((2, 3, 4)), np.diag([2.0, 3.0, 4.0, 1.0]))
    vol.to_orientation(codes)
    assert vol.grid.orientation == tuple(codes)
    assert vol.orientation == vol.grid.orientation
