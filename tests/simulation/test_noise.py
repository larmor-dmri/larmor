import numpy as np
import pytest

from larmor.simulation import Phantom, Protocol, TissueMap, add_noise, simulate
from larmor.types import GradientTable, Grid, Volume


def test_zero_signal_gives_the_rician_floor() -> None:
    sigma = 0.5
    zero = Volume(np.zeros((40, 40, 40, 4), np.float32), np.eye(4))
    (noisy,) = add_noise([zero], sigma, seed=0)
    values = noisy.data.astype(np.float64)
    assert values.mean() == pytest.approx(sigma * np.sqrt(np.pi / 2), rel=0.01)
    assert values.std() == pytest.approx(sigma * np.sqrt(2 - np.pi / 2), rel=0.01)


def test_one_sigma_for_every_tissue_and_echo() -> None:
    # Flat regions of GM, CSF and background, at two echo times; at this SNR the
    # noise of a bright region is close to Gaussian with standard deviation sigma.
    grid = Grid((30, 30, 30), np.eye(4))
    gm, csf = np.zeros(grid.shape), np.zeros(grid.shape)
    gm[:10], csf[10:20] = 1.0, 1.0
    maps = (
        TissueMap("GM", Volume(gm, np.eye(4))),
        TissueMap("CSF", Volume(csf, np.eye(4))),
    )
    gtab = GradientTable(np.zeros(8), np.zeros((8, 3)))
    protocol = Protocol(gtab, 15.0, 30.0, [60.0, 120.0], 3000.0)
    result = simulate(Phantom(grid, tissue_maps=maps), protocol, snr=50, seed=1)
    assert result.sigma is not None
    for echo in result.echoes:
        for region in (slice(0, 10), slice(10, 20)):
            assert echo.data[region].std() == pytest.approx(result.sigma, rel=0.03)
        floor = echo.data[20:].mean()
        assert floor == pytest.approx(result.sigma * np.sqrt(np.pi / 2), rel=0.03)


def test_same_seed_gives_the_same_noise() -> None:
    volume = Volume(np.ones((5, 5, 5, 3), np.float32), np.eye(4))
    first = add_noise([volume, volume], 0.1, seed=4)
    second = add_noise([volume, volume], 0.1, seed=4)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a.data, b.data)
    assert not np.array_equal(first[0].data, first[1].data)
