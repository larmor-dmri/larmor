from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from reference_solvers import DRL_INIT, drl_solver

from larmor import _core
from larmor.sd import DRL, AnisotropicRF
from larmor.sd.kernel import Kernel
from larmor.sd.solvers import Stage, calibrate_drl_nu, fit_single_tissue
from larmor.sd.sphere import DIRS_300

N_ITER = 200
# More than two compiled blocks of 128 voxels, the last one partly filled.
N_VOXELS = 300
# The C++ products sum in another order than numpy's; rounding stays far below this.
RTOL = 1e-10

# grl holds scipy loadmat output, which has no fixed structure.


@pytest.fixture(scope="module")
def bvals(grl: dict[str, Any]) -> NDArray[np.float64]:
    return np.asarray(grl["bvals"], dtype=np.float64)


@pytest.fixture(scope="module")
def kernel(grl: dict[str, Any], bvals: NDArray[np.float64]) -> Kernel:
    wm = AnisotropicRF(eigen_vals=[1.7e-3, 0.2e-3, 0.2e-3], mean_kurtosis=0.0)
    return Kernel(
        [wm],
        bvals=bvals,
        bvecs=grl["bvecs"].astype(np.float64),
        dirs=DIRS_300,
        weights=grl["shell_weight"],
    )


@pytest.fixture(scope="module")
def nu(kernel: Kernel, bvals: NDArray[np.float64]) -> float:
    return calibrate_drl_nu(kernel, bvals)


@pytest.fixture(scope="module")
def signals(kernel: Kernel, bvals: NDArray[np.float64]) -> NDArray[np.float64]:
    # Noisy one- to three-fibre voxels run all N_ITER iterations; isotropic voxels
    # converge within a few, so rows leave a block while others still iterate.
    rng = np.random.default_rng(0)
    n_cols = kernel.k_aniso.shape[1]
    n_fibre = 2 * N_VOXELS // 3
    fods = np.zeros((n_fibre, n_cols))
    for fod in fods:
        cols = rng.choice(n_cols, size=rng.integers(1, 4), replace=False)
        fod[cols] = rng.dirichlet(np.ones(len(cols)))
    csf = rng.uniform(0.0, 0.3, size=(n_fibre, 1))
    fibre = (1 - csf) * fods @ kernel.k_aniso.T + csf * np.exp(-bvals * 3e-3)
    fibre += rng.uniform(0.0, 0.05, (n_fibre, 1)) * rng.standard_normal(fibre.shape)
    adc = rng.uniform(0.5e-3, 3e-3, size=(N_VOXELS - n_fibre, 1))
    iso = np.exp(-bvals * adc)
    out = np.vstack([fibre, iso])
    return np.ascontiguousarray(out[rng.permutation(N_VOXELS)])


def _products(
    k: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return np.ascontiguousarray(k.T), np.ascontiguousarray(k.T @ k)


def _drl(
    signals: NDArray[np.float64],
    k: NDArray[np.float64],
    init: float,
    nu: float,
    n_iter: int,
    n_threads: int,
) -> NDArray[np.float64]:
    k_t, k_t_k = _products(k)
    solver = _core.FodSolver.drl(k_t, k_t_k, nu=nu, n_iter=n_iter, init=init)
    return _core.fit_single_tissue(signals, solver, n_threads)


def _reference(
    signals: NDArray[np.float64],
    k: NDArray[np.float64],
    init: float,
    nu: float,
) -> NDArray[np.float64]:
    k_t, k_t_k = _products(k)
    fod_init = np.full(k_t.shape[0], init)
    return np.array([drl_solver(fod_init, s, k_t, k_t_k, nu, N_ITER) for s in signals])


@pytest.fixture(scope="module")
def single_solver(kernel: Kernel, bvals: NDArray[np.float64]) -> _core.FodSolver:
    return DRL()._prepare(kernel, bvals, Stage.SINGLE)


@pytest.mark.parametrize(
    ("init", "damped"),
    [("uniform", False), ("uniform", True), ("floor", True)],
    ids=["calibration", "single-tissue", "multi-tissue-start"],
)
def test_drl_solver_matches_the_numpy_solver(
    kernel: Kernel,
    signals: NDArray[np.float64],
    nu: float,
    init: str,
    damped: bool,
) -> None:
    start = 1.0 / kernel.k.shape[1] if init == "uniform" else DRL_INIT
    damping = nu if damped else 0.0
    expected = _reference(signals, kernel.k, start, damping)
    out = _drl(signals, kernel.k, start, damping, N_ITER, 2)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=0)


def test_test_signals_converge_at_different_iterations(
    kernel: Kernel, signals: NDArray[np.float64], nu: float
) -> None:
    # A voxel that converged by half the iterations gives the same FOD with all of
    # them; the agreement above needs some of each.
    init = 1.0 / kernel.k.shape[1]
    half = _drl(signals, kernel.k, init, nu, N_ITER // 2, 1)
    full = _drl(signals, kernel.k, init, nu, N_ITER, 1)
    converged = np.all(half == full, axis=1).sum()
    assert 0 < converged < N_VOXELS


@pytest.mark.parametrize("n_threads", [2, 3, 8])
def test_drl_solver_does_not_depend_on_thread_count(
    kernel: Kernel, signals: NDArray[np.float64], nu: float, n_threads: int
) -> None:
    init = 1.0 / kernel.k.shape[1]
    one = _drl(signals, kernel.k, init, nu, N_ITER, 1)
    many = _drl(signals, kernel.k, init, nu, N_ITER, n_threads)
    np.testing.assert_array_equal(many, one)


def test_single_stage_drl_matches_the_per_voxel_solver(
    kernel: Kernel,
    signals: NDArray[np.float64],
    nu: float,
    single_solver: _core.FodSolver,
) -> None:
    # The single stage solves the weighted anisotropic columns from a uniform start
    # for all n_iter iterations, damped by the kernel's calibrated nu.
    init = 1.0 / kernel.w_k_aniso.shape[1]
    expected = _reference(signals, kernel.w_k_aniso, init, nu)
    out = fit_single_tissue(signals, single_solver, n_threads=2)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=0)


def test_fit_single_tissue_accepts_a_fortran_order_array(
    signals: NDArray[np.float64], single_solver: _core.FodSolver
) -> None:
    fortran = np.asfortranarray(signals)
    out = fit_single_tissue(fortran, single_solver, n_threads=1)
    expected = fit_single_tissue(signals, single_solver, n_threads=1)
    np.testing.assert_array_equal(out, expected)


def test_solver_reports_its_directions_and_measurements(
    kernel: Kernel, single_solver: _core.FodSolver
) -> None:
    assert (single_solver.n_meas, single_solver.n_dirs) == kernel.w_k_aniso.shape


def test_empty_signals_give_an_empty_fit(single_solver: _core.FodSolver) -> None:
    empty = np.empty((0, single_solver.n_meas))
    out = _core.fit_single_tissue(empty, single_solver, 2)
    assert out.shape == (0, single_solver.n_dirs)
    assert out.dtype == np.float64


@pytest.mark.parametrize("layout", ["fortran", "row-strided"])
def test_non_contiguous_signals_are_refused(
    signals: NDArray[np.float64], single_solver: _core.FodSolver, layout: str
) -> None:
    bad = np.asfortranarray(signals) if layout == "fortran" else signals[::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        _core.fit_single_tissue(bad, single_solver, 1)


def test_drl_refuses_a_k_t_k_that_is_not_square_in_the_directions(
    kernel: Kernel,
) -> None:
    k_t, k_t_k = _products(kernel.k)
    with pytest.raises(ValueError, match="shapes do not match"):
        _core.FodSolver.drl(
            k_t, np.ascontiguousarray(k_t_k[:, :-1]), nu=0.1, n_iter=10, init=0.1
        )


def test_signals_without_the_solver_measurements_are_refused(
    signals: NDArray[np.float64], single_solver: _core.FodSolver
) -> None:
    short = np.ascontiguousarray(signals[:, :-1])
    with pytest.raises(ValueError, match="measurements"):
        _core.fit_single_tissue(short, single_solver, 1)


def test_zero_threads_is_refused(
    signals: NDArray[np.float64], single_solver: _core.FodSolver
) -> None:
    with pytest.raises(ValueError, match="n_threads"):
        _core.fit_single_tissue(signals, single_solver, 0)


@pytest.mark.parametrize(
    "settings",
    [
        {"num_iter": 0},
        {"hr_num_iter": 0},
        {"num_iter": -1},
        {"target_adc": 0.0},
        {"target_adc": float("nan")},
        {"nu_scale": 0.0},
        {"nu_scale": float("inf")},
    ],
    ids=[
        "num_iter-0",
        "hr_num_iter-0",
        "num_iter-neg",
        "adc-0",
        "adc-nan",
        "nu-0",
        "nu-inf",
    ],
)
def test_drl_refuses_bad_settings(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be"):
        DRL(**settings)
