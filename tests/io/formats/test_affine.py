import numpy as np
import pytest

from larmor.io.formats._affine import apply_affine

AFFINE = np.array(
    [
        [0.0, -2.0, 0.0, 10.5],
        [1.5, 0.0, 0.0, -3.25],
        [0.0, 0.0, 3.0, 0.75],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def test_points_are_mapped_by_the_affine() -> None:
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 7.0]])
    expected = points @ AFFINE[:3, :3].T + AFFINE[:3, 3]
    np.testing.assert_allclose(apply_affine(AFFINE, points), expected, rtol=1e-15)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_points_keep_their_precision(dtype: type[np.floating]) -> None:
    points = np.array([[1.0, 2.0, 3.0]], dtype=dtype)
    assert apply_affine(AFFINE, points).dtype == dtype


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
def test_integer_points_are_not_truncated(dtype: type[np.integer]) -> None:
    points = np.array([[1, 2, 3], [0, 0, 0]], dtype=dtype)
    out = apply_affine(AFFINE, points)
    assert np.issubdtype(out.dtype, np.floating)
    assert out.dtype.itemsize >= 4
    # Every value is exact in float32: quarters and halves of small integers.
    np.testing.assert_array_equal(out, [[6.5, -1.75, 9.75], [10.5, -3.25, 0.75]])
