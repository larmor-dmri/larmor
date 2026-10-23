import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import nnls

from larmor import _core
from larmor.sd import L2NNLS, AnisotropicRF, IsotropicRF
from larmor.sd.kernel import Kernel
from larmor.sd.solvers import Stage, fit_single_tissue
from larmor.sd.sphere import get_dirs

N_MEAS = 60
N_DIRS = 90
TOL_X = 1e-12


def objective(
    k: NDArray[np.float64], s: NDArray[np.float64], f: NDArray[np.float64], reg: float
) -> float:
    return float(np.sum((k @ f - s) ** 2) + reg**2 * np.sum(f**2))


def fit(
    k: NDArray[np.float64], signals: NDArray[np.float64], reg: float, n_threads: int = 1
) -> NDArray[np.float64]:
    return _core.fit_single_tissue(
        signals, _core.FodSolver.l2nnls(k, reg, TOL_X), n_threads
    )


@pytest.fixture(scope="module")
def k() -> NDArray[np.float64]:
    return np.random.default_rng(0).uniform(0.0, 1.0, (N_MEAS, N_DIRS))


def random_signals(n_voxels: int) -> NDArray[np.float64]:
    # Gaussian signals lie mostly outside the cone of the positive kernel, so even
    # the reg 0 residual is far from 0 and objectives compare in relative terms.
    return np.random.default_rng(1).standard_normal((n_voxels, N_MEAS)) + 1.0


@pytest.mark.parametrize("reg", [0.0, 0.1])
def test_fods_minimise_the_same_objective_as_scipy_nnls(
    k: NDArray[np.float64], reg: float
) -> None:
    signals = random_signals(5)
    fods = fit(k, signals, reg)
    augmented = np.vstack([k, reg * np.eye(N_DIRS)])
    for s, f in zip(signals, fods, strict=True):
        expected, _ = nnls(augmented, np.concatenate([s, np.zeros(N_DIRS)]))
        assert np.all(f >= 0.0)
        assert objective(k, s, f, reg) == pytest.approx(
            objective(k, s, expected, reg), rel=1e-10
        )


def test_positive_reg_shrinks_the_fod(k: NDArray[np.float64]) -> None:
    signals = random_signals(5)
    plain = np.linalg.norm(fit(k, signals, 0.0), axis=1)
    ridge = np.linalg.norm(fit(k, signals, 0.5), axis=1)
    assert np.all(ridge < plain)


@pytest.mark.parametrize("reg", [0.0, 0.3])
def test_sizes_are_those_of_the_kernel(k: NDArray[np.float64], reg: float) -> None:
    solver = _core.FodSolver.l2nnls(k, reg, TOL_X)
    assert (solver.n_meas, solver.n_dirs) == (N_MEAS, N_DIRS)


@pytest.mark.parametrize(
    ("reg", "tol_x", "match"),
    [(-0.1, TOL_X, "reg"), (float("nan"), TOL_X, "reg"), (0.1, 0.0, "tol_x")],
)
def test_invalid_settings_are_refused(
    k: NDArray[np.float64], reg: float, tol_x: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _core.FodSolver.l2nnls(k, reg, tol_x)


def test_signals_of_the_wrong_length_are_refused(k: NDArray[np.float64]) -> None:
    solver = _core.FodSolver.l2nnls(k, 0.1, TOL_X)
    with pytest.raises(ValueError):
        _core.fit_single_tissue(np.ones((2, N_MEAS + N_DIRS)), solver, 1)


def test_result_does_not_depend_on_n_threads(k: NDArray[np.float64]) -> None:
    # More voxels than one compiled block of 128, so the threads share the work.
    signals = random_signals(300)
    np.testing.assert_array_equal(fit(k, signals, 0.1, 1), fit(k, signals, 0.1, 3))


@pytest.fixture(scope="module")
def two_shells() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    shell = get_dirs(30)
    bvals = np.concatenate([np.full(30, 1000.0), np.full(30, 3000.0)])
    return bvals, np.vstack([shell, shell])


@pytest.fixture(scope="module")
def tissue_kernel(
    two_shells: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> Kernel:
    bvals, bvecs = two_shells
    wm = AnisotropicRF(eigen_vals=[1.7e-3, 0.2e-3, 0.2e-3], mean_kurtosis=0.0)
    # Unequal weights and an isotropic column make w_k_aniso differ from k, w_k and
    # k_aniso, so a solver built from any of those gives other FODs.
    weights = np.where(bvals == 1000.0, 0.3, 1.0)
    return Kernel(
        [wm, IsotropicRF(diffusivity=3e-3)],
        bvals,
        bvecs,
        dirs=get_dirs(40),
        weights=weights,
    )


@pytest.mark.parametrize(
    ("reg", "tol_x"),
    [
        (-0.1, 1e-6),
        (float("nan"), 1e-6),
        (float("inf"), 1e-6),
        (0.1, 0.0),
        (0.1, float("nan")),
    ],
    ids=["reg-negative", "reg-nan", "reg-inf", "tol-zero", "tol-nan"],
)
def test_solver_refuses_invalid_settings(reg: float, tol_x: float) -> None:
    with pytest.raises(ValueError):
        L2NNLS(reg=reg, tol_x=tol_x)


@pytest.mark.parametrize(
    ("stage", "uses_reg"),
    [(Stage.LOW_RES, False), (Stage.HIGH_RES, True), (Stage.SINGLE, True)],
)
def test_prepared_solver_is_plain_nnls_only_in_the_multi_tissue_loop(
    tissue_kernel: Kernel,
    two_shells: tuple[NDArray[np.float64], NDArray[np.float64]],
    stage: Stage,
    uses_reg: bool,
) -> None:
    bvals, _ = two_shells
    settings = L2NNLS(reg=0.5, tol_x=1e-10)
    k = np.ascontiguousarray(tissue_kernel.w_k_aniso)
    rng = np.random.default_rng(2)
    signals = np.ascontiguousarray(
        rng.uniform(0.0, 1.0, (4, k.shape[1])) @ k.T
        + 0.05 * rng.standard_normal((4, k.shape[0]))
    )

    def fods(reg: float) -> NDArray[np.float64]:
        solver = _core.FodSolver.l2nnls(k, reg, settings.tol_x)
        return fit_single_tissue(signals, solver, 1)

    expected, other = (fods(0.5), fods(0.0)) if uses_reg else (fods(0.0), fods(0.5))
    assert not np.allclose(expected, other)
    solver = settings._prepare(tissue_kernel, bvals, stage)
    np.testing.assert_array_equal(fit_single_tissue(signals, solver, 1), expected)
    assert (solver.n_meas, solver.n_dirs) == k.shape
