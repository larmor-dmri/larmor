# The numpy reference's trilinear interpolation and grid test, against scipy and
# by hand; the grid test also against the compiled start check, where a seed starts
# exactly when it lies on the FOD's grid.
import itertools

import numpy as np
import pytest
from numpy.typing import NDArray
from reference_tracking import on_grid, trilinear
from scipy.ndimage import map_coordinates

from larmor import _core

SHAPE = (4, 5, 3)
# float64 sums of 8 weighted terms: rounding stays near 1e-15 of the values.
ATOL = 1e-12


def _coeffs(num_coeffs: int, seed: int = 0) -> NDArray[np.float64]:
    out: NDArray[np.float64] = np.random.default_rng(seed).normal(
        size=(*SHAPE, num_coeffs)
    )
    return out


def _affine() -> NDArray[np.float64]:
    # Anisotropic, x flipped, offset and oblique about two axes, so every
    # off-diagonal term is non-zero: a swapped axis, a transposed matrix or a
    # dropped translation changes the result.
    a, b = np.deg2rad(20.0), np.deg2rad(35.0)
    about_y = np.array(
        [[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]]
    )
    about_z = np.array(
        [[np.cos(b), -np.sin(b), 0.0], [np.sin(b), np.cos(b), 0.0], [0.0, 0.0, 1.0]]
    )
    affine = np.eye(4)
    affine[:3, :3] = about_z @ about_y @ np.diag([-2.0, 1.25, 3.0])
    affine[:3, 3] = [30.0, -8.0, 4.5]
    return affine


def _reference(
    coeffs: NDArray[np.floating], vox: NDArray[np.float64]
) -> NDArray[np.float64]:
    # Linear interpolation that repeats the edge voxel outside the grid.
    return np.array(
        [
            map_coordinates(
                coeffs[..., k].astype(np.float64), vox[:, None], order=1, mode="nearest"
            )[0]
            for k in range(coeffs.shape[-1])
        ]
    )


def _at(
    coeffs: NDArray[np.floating],
    vox: NDArray[np.float64],
    affine: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    affine = np.eye(4) if affine is None else affine
    world = (affine @ np.append(vox, 1.0))[:3]
    out: NDArray[np.float64] = trilinear(
        coeffs, np.ascontiguousarray(np.linalg.inv(affine)), world
    )
    return out


@pytest.mark.parametrize("num_coeffs", [1, 45])
def test_voxel_centres_give_the_voxel_values(num_coeffs: int) -> None:
    coeffs = _coeffs(num_coeffs)
    for idx in itertools.product(*(range(n) for n in SHAPE)):
        np.testing.assert_allclose(
            _at(coeffs, np.array(idx, dtype=np.float64)),
            coeffs[idx],
            rtol=0,
            atol=ATOL,
        )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_midpoint_between_two_voxels_is_their_mean(axis: int) -> None:
    coeffs = _coeffs(45)
    a = np.array([1, 1, 1])
    b = a.copy()
    b[axis] += 1
    got = _at(coeffs, (a + b) / 2.0)
    np.testing.assert_allclose(
        got, (coeffs[tuple(a)] + coeffs[tuple(b)]) / 2, rtol=0, atol=ATOL
    )


def test_cube_centre_is_the_mean_of_its_8_corners() -> None:
    coeffs = _coeffs(45)
    got = _at(coeffs, np.array([1.5, 2.5, 0.5]))
    np.testing.assert_allclose(
        got, coeffs[1:3, 2:4, 0:2].mean(axis=(0, 1, 2)), rtol=0, atol=ATOL
    )


def test_interior_point_is_the_weighted_sum_of_its_corners() -> None:
    coeffs = _coeffs(45)
    vox = np.array([1.3, 2.85, 0.6])
    expected = np.zeros(45)
    for i, j, k in itertools.product((1, 2), (2, 3), (0, 1)):
        w = (1 - abs(vox[0] - i)) * (1 - abs(vox[1] - j)) * (1 - abs(vox[2] - k))
        expected += w * coeffs[i, j, k]
    np.testing.assert_allclose(_at(coeffs, vox), expected, rtol=0, atol=ATOL)


def test_random_interior_points_match_linear_interpolation() -> None:
    coeffs = _coeffs(45)
    rng = np.random.default_rng(1)
    for vox in rng.uniform(0, np.array(SHAPE) - 1, size=(50, 3)):
        np.testing.assert_allclose(
            _at(coeffs, vox), _reference(coeffs, vox), rtol=0, atol=ATOL
        )


def _beyond_edges() -> list[NDArray[np.float64]]:
    # Within half a voxel past every face, edge and corner, with an interior
    # fraction on the axes that stay inside.
    hi = np.array(SHAPE) - 1.0
    points = []
    for offsets in itertools.product((-1, 0, 1), repeat=3):
        if offsets == (0, 0, 0):
            continue
        vox = np.array([1.3, 2.6, 0.7])
        for axis, o in enumerate(offsets):
            if o == -1:
                vox[axis] = -0.45
            elif o == 1:
                vox[axis] = hi[axis] + 0.45
        points.append(vox)
    return points


@pytest.mark.parametrize("vox", _beyond_edges())
def test_points_up_to_half_a_voxel_outside_use_edge_values(
    vox: NDArray[np.float64],
) -> None:
    coeffs = _coeffs(3)
    clipped = np.clip(vox, 0, np.array(SHAPE) - 1.0)
    np.testing.assert_allclose(
        _at(coeffs, vox), _reference(coeffs, clipped), rtol=0, atol=ATOL
    )


def test_oblique_affine_samples_at_matching_world_positions() -> None:
    coeffs = _coeffs(45)
    rng = np.random.default_rng(2)
    for vox in rng.uniform(-0.4, np.array(SHAPE) - 0.6, size=(30, 3)):
        # The inverse affine and back adds ~1e-15 mm to the voxel position.
        np.testing.assert_allclose(
            _at(coeffs, vox, _affine()),
            _reference(coeffs, np.clip(vox, 0, np.array(SHAPE) - 1.0)),
            rtol=0,
            atol=1e-10,
        )


def test_float32_volume_gives_float64_values() -> None:
    coeffs = _coeffs(45).astype(np.float32)
    vox = np.array([2.2, 0.4, 1.9])
    got = _at(coeffs, vox)
    assert got.dtype == np.float64
    # The float32 values are exact in float64; only the sum rounds.
    np.testing.assert_allclose(got, _reference(coeffs, vox), rtol=0, atol=ATOL)


# on_grid


def _exact_affine() -> NDArray[np.float64]:
    # Flipped, anisotropic and offset with powers of two, so the round trip to world
    # coordinates and back is exact and the faces can be hit exactly.
    affine = np.diag([-2.0, 0.5, 4.0, 1.0])
    affine[:3, 3] = [10.0, -3.0, 2.0]
    return affine


def _on_grid_at(
    vox: NDArray[np.float64],
    affine: NDArray[np.float64],
    shape: tuple[int, ...] = SHAPE,
) -> bool:
    world = (affine @ np.append(vox, 1.0))[:3]
    inv = np.ascontiguousarray(np.linalg.inv(affine))
    reference = bool(on_grid(shape, inv, world))
    fod = np.zeros((*shape[:3], shape[3] if len(shape) > 3 else 1), dtype=np.float32)
    counts, _, _, _ = _core.track_batch(
        fod,
        inv,
        None,
        world[None],
        np.array([[1.0, 0.0, 0.0]]),
        np.zeros(1, dtype=np.int64),
        0,
        0.1,
        1,
        False,
        _core.TestAlgorithm("straight"),
        np.zeros((1, 2, 3), dtype=np.float32),
        1,
    )
    assert bool(counts[0] > 0) is reference
    return reference


_FACES = [(axis, side) for axis in range(3) for side in (-1, 1)]
_SHAPES = [SHAPE, (*SHAPE, 1), (*SHAPE, 45)]


@pytest.mark.parametrize("affine", [np.eye(4), _exact_affine()], ids=["eye", "exact"])
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize(("axis", "side"), _FACES)
@pytest.mark.parametrize(
    ("outward", "expected"), [(-1e-9, True), (0.0, True), (1e-9, False)]
)
def test_on_grid_includes_each_face_and_nothing_beyond(
    affine: NDArray[np.float64],
    shape: tuple[int, ...],
    axis: int,
    side: int,
    outward: float,
    expected: bool,
) -> None:
    vox = np.array([1.3, 2.6, 0.7])
    face = -0.5 if side < 0 else SHAPE[axis] - 0.5
    vox[axis] = face + side * outward
    assert _on_grid_at(vox, affine, shape) is expected


@pytest.mark.parametrize(("axis", "side"), _FACES)
@pytest.mark.parametrize(("outward", "expected"), [(-1e-6, True), (1e-6, False)])
def test_on_grid_faces_under_oblique_affine(
    axis: int, side: int, outward: float, expected: bool
) -> None:
    # The inverse affine and back moves the position by ~1e-15 voxel, far below 1e-6.
    vox = np.array([1.3, 2.6, 0.7])
    vox[axis] = (-0.5 if side < 0 else SHAPE[axis] - 0.5) + side * outward
    assert _on_grid_at(vox, _affine()) is expected


@pytest.mark.parametrize("affine", [np.eye(4), _affine()], ids=["eye", "oblique"])
@pytest.mark.parametrize(("axis", "side"), _FACES)
@pytest.mark.parametrize("beyond", [0.55, 10.0, 500.0])
def test_on_grid_is_false_far_outside(
    affine: NDArray[np.float64], axis: int, side: int, beyond: float
) -> None:
    vox = np.array([1.3, 2.6, 0.7])
    vox[axis] = -beyond if side < 0 else SHAPE[axis] - 1 + beyond
    assert _on_grid_at(vox, affine) is False


@pytest.mark.parametrize("affine", [np.eye(4), _affine()], ids=["eye", "oblique"])
def test_on_grid_holds_at_every_voxel_centre_and_corner(
    affine: NDArray[np.float64],
) -> None:
    for idx in itertools.product(*(range(n) for n in SHAPE)):
        assert _on_grid_at(np.array(idx, dtype=np.float64), affine)
    for corner in itertools.product(*((-0.49, n - 0.51) for n in SHAPE)):
        assert _on_grid_at(np.array(corner), affine)


def test_on_grid_ignores_the_series_size() -> None:
    # A z of 2.6 is off a grid 3 voxels deep whatever the fourth size is.
    vox = np.array([1.0, 1.0, 2.6])
    assert not _on_grid_at(vox, np.eye(4), (*SHAPE, 100))
    assert _on_grid_at(np.array([1.0, 1.0, 2.4]), np.eye(4), (4, 5, 3, 1))
