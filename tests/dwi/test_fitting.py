import numpy as np
import numpy.typing as npt
import pytest

from larmor import _core
from larmor.dwi.fitting import fit_log_signal

Array = npt.NDArray[np.float64]


def _design() -> Array:
    """Log-signal design of a diffusion tensor: 1 and -b g_i g_j, on b0, 1000, 1500."""
    rng = np.random.default_rng(3)
    g = rng.normal(size=(40, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    b = np.repeat([1000.0, 1500.0], 20)
    x, y, z = g.T
    dw = -b[:, None] * np.column_stack(
        [x * x, 2 * x * y, 2 * x * z, y * y, 2 * y * z, z * z]
    )
    rows = np.vstack([np.zeros((3, 6)), dw])
    return np.column_stack([np.ones(len(rows)), rows])


DESIGN = _design()
# log S0 and xx, xy, xz, yy, yz, zz of three tensors (mm^2/s).
PARAMS = np.array(
    [
        [np.log(1000.0), 1.2e-3, 0.3e-3, -0.1e-3, 0.6e-3, 0.05e-3, 0.4e-3],
        [np.log(500.0), 0.8e-3, 0.0, 0.0, 0.8e-3, 0.0, 0.8e-3],
        [np.log(2000.0), 0.4e-3, -0.2e-3, 0.1e-3, 1.5e-3, 0.2e-3, 0.5e-3],
    ]
).T


@pytest.mark.parametrize("weighted", [False, True], ids=["ols", "wls"])
def test_noise_free_signal_gives_its_parameters(weighted: bool) -> None:
    signal = np.exp(DESIGN @ PARAMS).T
    params, _ = fit_log_signal(DESIGN, signal, weighted)
    np.testing.assert_allclose(params, PARAMS, rtol=1e-9, atol=1e-15)


def test_wls_is_the_weighted_fit_with_weights_from_its_own_prediction() -> None:
    rng = np.random.default_rng(4)
    clean = np.exp(DESIGN @ PARAMS).T
    noisy = np.abs(clean + rng.normal(scale=30.0, size=clean.shape))
    params, _ = fit_log_signal(DESIGN, noisy, weighted=True)
    log_s = np.log(noisy)
    for voxel in range(noisy.shape[0]):
        p = params[:, voxel]
        weights = np.exp(2 * (DESIGN @ p))
        lhs = DESIGN.T @ (weights[:, None] * DESIGN)
        rhs = DESIGN.T @ (weights * log_s[voxel])
        fixed_point = np.linalg.solve(lhs, rhs)
        # The iteration stops once no predicted log signal moves by more than 1e-6;
        # converging linearly, it is then well within that of its fixed point.
        np.testing.assert_allclose(DESIGN @ fixed_point, DESIGN @ p, rtol=0, atol=1e-6)
    # Noise makes the weighting matter: the two fits differ by far more than that.
    ols, _ = fit_log_signal(DESIGN, noisy, weighted=False)
    assert np.abs(DESIGN @ (params - ols)).max() > 1e-3


def test_voxel_whose_weighted_fit_breaks_down_takes_the_ordinary_fit() -> None:
    # Predicted diffusion-weighted signals of exp(-420) to exp(-630) give weights
    # S^2 that underflow to 0 against the b0s: only the b0 rows are left, and the
    # weighted normal equations are singular.
    broken = np.array([0.0, 0.42, 0.0, 0.0, 0.42, 0.0, 0.42])
    params = np.column_stack([PARAMS[:, 0], broken])
    signal = np.exp(DESIGN @ params).T
    wls, _ = fit_log_signal(DESIGN, signal, weighted=True)
    ols, _ = fit_log_signal(DESIGN, signal, weighted=False)
    assert np.isfinite(wls).all()
    np.testing.assert_array_equal(wls[:, 1], ols[:, 1])


@pytest.mark.parametrize(
    ("x", "g", "log_s", "n_threads", "message"),
    [
        (DESIGN, np.zeros((0, 6)), np.zeros((2, 43)), 1, "must be"),
        (DESIGN, np.zeros((0, 7)), np.zeros((2, 40)), 1, "must be"),
        (DESIGN, np.zeros((0, 7)), np.zeros((2, 43)), 0, "n_threads"),
        (DESIGN, np.zeros((0, 7)), np.zeros((43, 2)).T, 1, "C-contiguous"),
    ],
    ids=["constraints", "measurements", "threads", "strides"],
)
def test_compiled_fit_refuses_mismatched_inputs(
    x: Array, g: Array, log_s: Array, n_threads: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _core.fit_log_signal(x, g, log_s, True, 20, 1e-6, n_threads)
