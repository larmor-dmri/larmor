import numpy as np
import pytest
from scipy.ndimage import convolve

from larmor.dwi import estimate_s0
from larmor.types import GradientTable, Volume

GTAB = GradientTable(
    np.array([0.0, 0.0, 1000.0, 1000.0]),
    np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
)
# Mean b0 10 below a mean DW signal of 50.
DARK = (10.0, 50.0)


def _assert_close(actual: np.ndarray, expected: list[float]) -> None:
    # The window means are divided by the window size and back, so rounding only.
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def _line(*voxels: tuple[float, float]) -> Volume:
    """A row of voxels along x, each given as (b0, DW signal)."""
    data = np.array([[b0, b0, dw, dw] for b0, dw in voxels], dtype=np.float32)
    return Volume(data.reshape(len(voxels), 1, 1, 4), np.eye(4))


def _s0(volume: Volume, mask: list[bool] | None = None) -> np.ndarray:
    n = volume.data.shape[0]
    inside = np.ones(n, dtype=bool) if mask is None else np.array(mask)
    return estimate_s0(volume, GTAB, inside.reshape(n, 1, 1))[:, 0, 0]


def test_plausible_voxel_keeps_its_mean_b0() -> None:
    data = np.array([90.0, 110.0, 40.0, 60.0], dtype=np.float32).reshape(1, 1, 1, 4)
    s0 = estimate_s0(Volume(data, np.eye(4)), GTAB, np.ones((1, 1, 1), dtype=bool))
    assert s0.dtype == np.float64
    assert s0[0, 0, 0] == 100.0


def test_dark_voxel_takes_the_mean_s0_of_its_3x3x3_neighbours() -> None:
    # The voxels two steps away lie outside the 3x3x3 window.
    volume = _line((1000, 50), (100, 50), DARK, (300, 50), (2000, 50))
    _assert_close(_s0(volume), [1000, 100, 200, 300, 2000])


def test_dark_voxel_falls_back_to_5x5x5_only_without_a_3x3x3_neighbour() -> None:
    # The middle dark voxel has only dark 3x3x3 neighbours; repaired voxels do not
    # serve as neighbours. The 5x5x5 window of the last dark voxel would add 700.
    volume = _line((100, 50), DARK, DARK, DARK, (300, 50), (700, 50))
    _assert_close(_s0(volume), [100, 100, 200, 300, 300, 700])


def test_dark_voxel_without_plausible_5x5x5_neighbour_gets_nan() -> None:
    volume = _line(DARK, DARK, DARK, (100, 50))
    _assert_close(_s0(volume), [np.nan, 100, 100, 100])


def test_voxels_outside_the_mask_are_not_neighbours_and_keep_their_mean_b0() -> None:
    volume = _line((100, 50), DARK, (300, 50), DARK)
    s0 = _s0(volume, mask=[False, True, True, False])
    _assert_close(s0, [100, 300, 300, 10])


def test_mean_b0_equal_to_mean_dw_signal_is_implausible() -> None:
    volume = _line((100, 50), (50, 50), (300, 50))
    _assert_close(_s0(volume), [100, 200, 300])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_voxels_are_neither_repaired_nor_neighbours(bad: float) -> None:
    # A mean b0 of 1000 above a DW signal of -inf would pass as plausible.
    volume = _line((1000, bad), DARK, (300, 50), (10, bad))
    _assert_close(_s0(volume), [1000, 300, 300, 10])


def test_estimate_s0_refuses_data_without_b0() -> None:
    volume = Volume(np.ones((1, 1, 1, 3), np.float32), np.eye(4))
    gtab = GradientTable(np.full(3, 1000.0), np.eye(3))
    with pytest.raises(ValueError, match="No b0"):
        estimate_s0(volume, gtab, np.ones((1, 1, 1), dtype=bool))


def test_sparse_plausible_voxels_in_3d_match_an_exact_count() -> None:
    # Most windows hold no plausible voxel: a filtered count of nearly zero must
    # not pass for a neighbour.
    rng = np.random.default_rng(0)
    shape = (30, 30, 30)
    plausible = rng.random(shape) < 0.02
    b0 = np.where(plausible, rng.uniform(300, 4000, shape), DARK[0])
    data = np.stack([b0, b0, np.full(shape, DARK[1]), np.full(shape, DARK[1])], -1)
    volume = Volume(data.astype(np.float32), np.eye(4))
    b0_stored = data[..., 0].astype(np.float32).astype(np.float64)

    expected = np.full(shape, np.nan)
    expected[plausible] = b0_stored[plausible]
    # The 3x3x3 result overwrites the 5x5x5 one where it has a neighbour.
    for width in (5, 3):
        ones = np.ones((width,) * 3)
        count = convolve(
            plausible.astype(np.int64), ones.astype(np.int64), mode="constant"
        )
        total = convolve(np.where(plausible, b0_stored, 0.0), ones, mode="constant")
        found = ~plausible & (count > 0)
        expected[found] = total[found] / count[found]

    s0 = estimate_s0(volume, GTAB, np.ones(shape, dtype=bool))
    np.testing.assert_allclose(s0, expected, rtol=1e-9)
