import numpy as np
import numpy.typing as npt
import pytest

from larmor.sd import NoddiRF
from larmor.sd.rf import _legendre_gaussian_integrals, _watson_sh_coefficients
from larmor.sd.sphere import DIRS_300

# Rows of the "gradient" table: three axes and a diagonal per b-value.
_AXES = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_GRADIENTS = np.vstack([_AXES, np.ones((1, 3)) / np.sqrt(3)])


def _table(
    b_values: list[float],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    bvals = np.repeat(np.asarray(b_values, dtype=np.float64), len(_GRADIENTS))
    bvecs = np.tile(_GRADIENTS, (len(b_values), 1))
    return bvals, bvecs


def test_signal_is_s0_at_b0_for_every_direction() -> None:
    bvals, bvecs = _table([0.0])
    signal = NoddiRF(s0=2.5).signal(bvals, bvecs, DIRS_300)
    np.testing.assert_allclose(signal, 2.5, rtol=1e-12)


def test_free_water_only_is_isotropic_exponential() -> None:
    bvals, bvecs = _table([0.0, 1000.0, 3000.0])
    rf = NoddiRF(iso_fraction=1.0, iso_diffusivity=3e-3, s0=2.0)
    signal = rf.signal(bvals, bvecs, DIRS_300)
    expected = 2.0 * np.exp(-bvals * 3e-3)
    np.testing.assert_allclose(signal, np.broadcast_to(expected[:, None], signal.shape))


@pytest.mark.parametrize("concentration", [0.05, 1.0, 200.0])
def test_signal_is_symmetric_under_axis_flip(concentration: float) -> None:
    bvals, bvecs = _table([1000.0, 3000.0])
    rf = NoddiRF(concentration=concentration)
    np.testing.assert_allclose(
        rf.signal(bvals, bvecs, -DIRS_300), rf.signal(bvals, bvecs, DIRS_300)
    )


def test_concentrated_signal_is_lowest_along_the_axis() -> None:
    bvals, bvecs = _table([1000.0, 3000.0])
    signal = NoddiRF(concentration=200.0).signal(bvals, bvecs, _AXES[:1])[:, 0]
    along_x = signal[[0, 4]]
    across = signal[[1, 2, 5, 6]].reshape(2, 2)
    assert np.all(along_x[:, None] < across)


@pytest.mark.parametrize("kappa", [0.0, 0.05, 1.0, 30.0, 200.0])
def test_watson_degree_zero_coefficient_is_two_sqrt_pi(kappa: float) -> None:
    assert _watson_sh_coefficients(kappa)[0] == pytest.approx(2 * np.sqrt(np.pi))


def test_legendre_gaussian_integrals_at_zero() -> None:
    expected = np.zeros(7)
    expected[0] = 2.0
    np.testing.assert_array_equal(
        _legendre_gaussian_integrals(np.array([0.0]))[0], expected
    )
