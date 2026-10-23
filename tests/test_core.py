import numpy as np
import pytest
from numpy.typing import NDArray

from larmor._core import norms


@pytest.fixture
def rows() -> NDArray[np.float64]:
    return np.random.default_rng(0).normal(size=(1000, 3))


def test_norms_match_numpy(rows: NDArray[np.float64]) -> None:
    # _core and numpy may sum in a different order: a few ulps apart at most.
    np.testing.assert_allclose(norms(rows, 2), np.linalg.norm(rows, axis=1), rtol=1e-15)


def test_norms_do_not_depend_on_thread_count(rows: NDArray[np.float64]) -> None:
    np.testing.assert_array_equal(norms(rows, 1), norms(rows, 4))


def test_empty_array_gives_empty_norms() -> None:
    out = norms(np.empty((0, 3)), 2)
    assert out.shape == (0,)
    assert out.dtype == np.float64


@pytest.mark.parametrize(
    "v",
    [
        np.asfortranarray(np.ones((4, 3))),
        np.ones((8, 3))[::2],
        np.ones((4, 6))[:, :3],
    ],
    ids=["fortran", "row-strided", "column-sliced"],
)
def test_non_contiguous_array_is_refused(v: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match="C-contiguous"):
        norms(v, 1)


@pytest.mark.parametrize("ncols", [2, 4])
def test_wrong_column_count_is_refused(ncols: int) -> None:
    with pytest.raises(ValueError, match=f"expected 3 columns, got {ncols}"):
        norms(np.ones((4, ncols)), 1)


def test_zero_threads_is_refused() -> None:
    with pytest.raises(ValueError, match="n_threads"):
        norms(np.ones((4, 3)), 0)


@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_other_dtypes_are_refused(dtype: type[np.generic]) -> None:
    with pytest.raises(TypeError):
        norms(np.ones((4, 3), dtype=dtype), 1)
