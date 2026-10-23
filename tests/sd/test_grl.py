from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from reference_solvers import fit_multi_tissue_voxel, matlab_drl
from reference_solvers import lsqnonneg as reference_lsqnonneg
from scipy.optimize import nnls

from larmor import _core
from larmor.sd import DRL, AnisotropicRF, IsotropicRF, ResponseFunction
from larmor.sd.kernel import Kernel
from larmor.sd.model import normalize_by_b0
from larmor.sd.solvers import Stage, calibrate_drl_nu, fit_multi_tissue
from larmor.sd.sphere import DIRS_300, get_dirs
from larmor.types import GradientTable

Fit = tuple[NDArray[np.float64], NDArray[np.float64]]
# Signals, the two kernels, their nu and the solvers built from them.
Case = tuple[
    NDArray[np.float64], Kernel, Kernel, float, float, _core.FodSolver, _core.FodSolver
]
# A compiled block of 128 voxels and a partly filled one.
N_SYNTHETIC = 150
# The C++ products sum in another order than numpy's, and its low-resolution loop
# runs lsqnonneg at MATLAB's default TolX where the reference runs scipy's exact
# nnls. Measured: FODs within a relative 4.2e-13, fractions within 5e-15, the same
# zero fractions. A voxel leaving the loop at another round would differ by ~1e-3.
FOD_RTOL = 1e-10
FRACTION_ATOL = 1e-12

# grl holds scipy loadmat output, which has no fixed structure.


@pytest.fixture(scope="module")
def bvals(grl: dict[str, Any]) -> NDArray[np.float64]:
    return np.asarray(grl["bvals"], dtype=np.float64)


def _kernels(grl: dict[str, Any], rfs: list[ResponseFunction]) -> tuple[Kernel, Kernel]:
    bvals = grl["bvals"].astype(np.float64)
    bvecs = grl["bvecs"].astype(np.float64)
    weights = grl["shell_weight"]
    lr_dirs = get_dirs(min(len(bvals), 45))
    lr = Kernel(rfs, bvals, bvecs, dirs=lr_dirs, weights=weights)
    hr = Kernel(rfs, bvals, bvecs, dirs=DIRS_300, weights=weights)
    return lr, hr


@pytest.fixture(scope="module")
def two_fibre_kernels(grl: dict[str, Any]) -> tuple[Kernel, Kernel]:
    # Two anisotropic responses: the final fit has one column per block. The second
    # is planar; with these 33 measurements a fatter prolate one cannot be told
    # apart from the first and its fraction is always 0.
    rfs: list[ResponseFunction] = [
        AnisotropicRF(eigen_vals=[1.7e-3, 0.2e-3, 0.2e-3], mean_kurtosis=0.0),
        AnisotropicRF(eigen_vals=[1.5e-3, 0.9e-3, 0.1e-3], mean_kurtosis=0.0),
        IsotropicRF(diffusivity=0.7e-3),
        IsotropicRF(diffusivity=3e-3),
    ]
    return _kernels(grl, rfs)


@pytest.fixture(scope="module")
def matlab_case(grl: dict[str, Any], kernel_lr: Kernel, kernel_hr: Kernel) -> Case:
    voxels = np.argwhere(grl["mask"] == 1)
    xs, ys, zs = voxels[:, 0], voxels[:, 1], voxels[:, 2]
    is_b0 = GradientTable(grl["bvals"], grl["bvecs"]).b0_mask
    signals = normalize_by_b0(grl["img"][xs, ys, zs].astype(np.float64), is_b0)
    lr_nu, hr_nu = float(grl["NN_L"]), float(grl["NN_H"])
    return (
        np.ascontiguousarray(signals),
        kernel_lr,
        kernel_hr,
        lr_nu,
        hr_nu,
        matlab_drl(kernel_lr, lr_nu, 200),
        matlab_drl(kernel_hr, hr_nu, 600),
    )


@pytest.fixture(scope="module")
def synthetic_case(
    two_fibre_kernels: tuple[Kernel, Kernel], bvals: NDArray[np.float64]
) -> Case:
    # One to three fibres of each anisotropic response, two isotropic tissues; a
    # fifth of the voxels miss one tissue entirely, so the final fit zeroes it.
    lr, hr = two_fibre_kernels
    rng = np.random.default_rng(0)
    n_dirs = len(DIRS_300)
    signals = np.empty((N_SYNTHETIC, len(bvals)))
    for signal in signals:
        fod = np.zeros(n_dirs)
        cols = rng.choice(n_dirs, size=rng.integers(1, 4), replace=False)
        fod[cols] = rng.dirichlet(np.ones(len(cols)))
        f = rng.dirichlet(np.ones(4))
        if rng.random() < 0.2:
            f[rng.integers(4)] = 0.0
            f /= f.sum()
        signal[:] = (
            f[0] * hr.k_aniso[:, :n_dirs] @ fod
            + f[1] * hr.k_aniso[:, n_dirs:] @ fod
            + f[2] * np.exp(-bvals * 0.7e-3)
            + f[3] * np.exp(-bvals * 3e-3)
        )
        signal += rng.uniform(0.0, 0.03) * rng.standard_normal(len(bvals))
    # DRL's own solvers, so the comparison also covers how it prepares each stage.
    lr_nu, hr_nu = calibrate_drl_nu(lr, bvals), calibrate_drl_nu(hr, bvals)
    lr_solver = DRL()._prepare(lr, bvals, Stage.LOW_RES)
    hr_solver = DRL()._prepare(hr, bvals, Stage.HIGH_RES)
    return signals, lr, hr, lr_nu, hr_nu, lr_solver, hr_solver


CASES = pytest.mark.parametrize("case", ["matlab_case", "synthetic_case"])


def _reference(case: Case) -> Fit:
    signals, lr, hr, lr_nu, hr_nu, _, _ = case
    fits = [fit_multi_tissue_voxel(s, lr, hr, lr_nu, hr_nu) for s in signals]
    return np.array([f[0] for f in fits]), np.array([f[1] for f in fits])


def _compiled(case: Case, n_threads: int) -> Fit:
    signals, lr, hr, _, _, lr_solver, hr_solver = case
    return fit_multi_tissue(signals, lr, hr, lr_solver, hr_solver, n_threads=n_threads)


@CASES
def test_fit_multi_tissue_matches_the_numpy_reference(
    request: pytest.FixtureRequest, case: str
) -> None:
    data: Case = request.getfixturevalue(case)
    if case == "synthetic_case":
        # The Python reference is slow: use every third voxel.
        data = (data[0][::3], *data[1:])
    ref_fods, ref_fracs = _reference(data)
    fods, fracs = _compiled(data, n_threads=2)
    np.testing.assert_allclose(fods, ref_fods, rtol=FOD_RTOL, atol=0)
    np.testing.assert_array_equal(fracs == 0, ref_fracs == 0)
    np.testing.assert_allclose(fracs, ref_fracs, rtol=0, atol=FRACTION_ATOL)


def test_synthetic_voxels_include_zero_fractions(synthetic_case: Case) -> None:
    # The comparison above needs zeros in the zero pattern and a non-zero fraction of
    # every tissue, the second anisotropic one included.
    _, fracs = _compiled(synthetic_case, n_threads=1)
    assert (fracs == 0).any(axis=0)[1:].all()
    assert (fracs > 0.05).any(axis=0).all()


def test_fit_multi_tissue_does_not_depend_on_thread_count(matlab_case: Case) -> None:
    signals, lr, hr, lr_nu, hr_nu, lr_solver, hr_solver = matlab_case
    # The threads of the call take the blocks of 128 voxels in turn: 300 voxels make
    # three blocks, all run by one thread or shared among two or eight.
    many = np.ascontiguousarray(np.tile(signals, (2, 1))[:300])
    data: Case = (many, lr, hr, lr_nu, hr_nu, lr_solver, hr_solver)
    one_fods, one_fracs = _compiled(data, n_threads=1)
    for n_threads in (2, 8):
        fods, fracs = _compiled(data, n_threads=n_threads)
        np.testing.assert_array_equal(fods, one_fods, err_msg=f"{n_threads} threads")
        np.testing.assert_array_equal(fracs, one_fracs, err_msg=f"{n_threads} threads")


def _fraction_problems() -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    # Decaying exponentials, as the isotropic tissue columns, a third of the true
    # fractions 0, and noise that makes the fit inexact.
    rng = np.random.default_rng(1)
    b = np.repeat([0.0, 1000.0, 2000.0, 3000.0], 8)
    problems = []
    for _ in range(200):
        n = int(rng.integers(2, 7))
        c = np.exp(-np.outer(b, np.sort(rng.uniform(0.1e-3, 3e-3, n))))
        x = rng.dirichlet(np.ones(n))
        x[rng.random(n) < 0.3] = 0.0
        d = c @ x + rng.uniform(0.0, 0.02) * rng.standard_normal(len(b))
        problems.append((np.ascontiguousarray(c), d))
    return problems


def _matlab_default_tol_x(c: NDArray[np.float64]) -> float:
    return float(10 * np.finfo(np.float64).eps * np.linalg.norm(c, 1) * max(c.shape))


# Close diffusivities make the columns nearly collinear, so solutions by QR (C++),
# SVD (the port) and scipy differ by the conditioning times eps. Measured on these
# problems: 1.6e-11 against scipy, 1.5e-11 against the port at the default TolX,
# 1.8e-15 at TolX 1e-2 (the small coefficients go to 0 there).
@pytest.mark.parametrize(
    ("oracle", "tol_x", "atol"),
    [("scipy", None, 1e-10), ("port", None, 1e-10), ("port", 1e-2, 1e-13)],
    ids=["scipy-default", "port-default", "port-1e-2"],
)
def test_lsqnonneg_matches_the_reference_nnls(
    oracle: str, tol_x: float | None, atol: float
) -> None:
    n_zeros = 0
    for c, d in _fraction_problems():
        x = _core.lsqnonneg(c, d, tol_x)
        if oracle == "scipy":
            expected, _ = nnls(c, d)
        else:
            tol = _matlab_default_tol_x(c) if tol_x is None else tol_x
            expected = reference_lsqnonneg(c, d, tol)
        np.testing.assert_array_equal(x == 0, expected == 0)
        np.testing.assert_allclose(x, expected, rtol=0, atol=atol)
        n_zeros += int((x == 0).sum())
    assert n_zeros > 0


def test_lsqnonneg_tol_x_zeroes_a_small_coefficient() -> None:
    # A 0.5% compartment is kept at the default TolX and dropped at 1e-2, as in
    # MATLAB's final fraction fit.
    b = np.repeat([0.0, 1000.0, 2000.0, 3000.0], 8)
    c = np.ascontiguousarray(np.exp(-np.outer(b, [0.7e-3, 3e-3])))
    d = c @ np.array([0.995, 0.005])
    assert (_core.lsqnonneg(c, d) > 0).all()
    x = _core.lsqnonneg(c, d, 1e-2)
    assert x[1] == 0
    np.testing.assert_allclose(x[0], 1.0, atol=1e-2)


def test_lsqnonneg_refuses_a_right_hand_side_of_another_length() -> None:
    with pytest.raises(ValueError, match="c must be"):
        _core.lsqnonneg(np.ones((32, 3)), np.ones(31))


def _deconv_args(kernel: Kernel) -> dict[str, Any]:
    return {
        "k_aniso": np.ascontiguousarray(kernel.k_aniso),
        "k_iso": np.ascontiguousarray(kernel.k_iso),
        "w_k_iso": np.ascontiguousarray(kernel.w_k_iso),
        "aniso_blocks": [(0, kernel.k_aniso.shape[1])],
    }


def test_deconv_kernel_accepts_a_kernel(kernel_lr: Kernel) -> None:
    _core.DeconvKernel(**_deconv_args(kernel_lr))


@pytest.mark.parametrize("which", ["k_aniso", "k_iso", "w_k_iso"])
def test_deconv_kernel_refuses_mismatched_shapes(kernel_lr: Kernel, which: str) -> None:
    args = _deconv_args(kernel_lr)
    args[which] = np.ascontiguousarray(args[which][:-1])
    with pytest.raises(ValueError, match="shapes do not match"):
        _core.DeconvKernel(**args)


BAD_BLOCKS: dict[str, Callable[[int], list[tuple[int, int]]]] = {
    "none": lambda ca: [],
    "empty": lambda ca: [(0, 0)],
    "reversed": lambda ca: [(10, 5)],
    "past-the-end": lambda ca: [(0, ca + 1)],
    "second-past-the-end": lambda ca: [(0, 20), (20, ca + 1)],
}


@pytest.mark.parametrize("blocks", BAD_BLOCKS.values(), ids=BAD_BLOCKS.keys())
def test_deconv_kernel_refuses_bad_blocks(
    kernel_lr: Kernel, blocks: Callable[[int], list[tuple[int, int]]]
) -> None:
    args = _deconv_args(kernel_lr)
    args["aniso_blocks"] = blocks(kernel_lr.k_aniso.shape[1])
    with pytest.raises(ValueError, match="aniso_blocks"):
        _core.DeconvKernel(**args)


def test_deconv_kernel_takes_more_than_eight_tissues(kernel_lr: Kernel) -> None:
    # Two isotropic tissues plus seven one-column anisotropic blocks.
    args = _deconv_args(kernel_lr)
    args["aniso_blocks"] = [(i, i + 1) for i in range(7)]
    _core.DeconvKernel(**args)


def _core_fit(
    signals: NDArray[np.float64],
    weights: NDArray[np.float64],
    lr: _core.DeconvKernel,
    hr: _core.DeconvKernel,
    solvers: tuple[_core.FodSolver, _core.FodSolver],
    n_threads: int = 1,
) -> Fit:
    lr_solver, hr_solver = solvers
    return _core.fit_multi_tissue(
        signals, weights, lr, lr_solver, hr, hr_solver, 50, 1e-3, 1e-2, n_threads
    )


def _core_kernels(case: Case) -> tuple[_core.DeconvKernel, _core.DeconvKernel]:
    _, lr, hr, *_ = case
    return (
        _core.DeconvKernel(**_deconv_args(lr)),
        _core.DeconvKernel(**_deconv_args(hr)),
    )


def test_core_fit_multi_tissue_refuses_signals_of_another_length(
    matlab_case: Case,
) -> None:
    signals, lr, _, _, _, lr_solver, hr_solver = matlab_case
    lr_k, hr_k = _core_kernels(matlab_case)
    short = np.ascontiguousarray(signals[:, :-1])
    with pytest.raises(ValueError, match="do not match"):
        _core_fit(short, lr.weights, lr_k, hr_k, (lr_solver, hr_solver))
    with pytest.raises(ValueError, match="do not match"):
        _core_fit(signals, lr.weights[:-1], lr_k, hr_k, (lr_solver, hr_solver))


def test_fit_multi_tissue_refuses_signals_of_another_length(
    matlab_case: Case,
) -> None:
    signals, lr, hr, _, _, lr_solver, hr_solver = matlab_case
    with pytest.raises(ValueError, match="do not match"):
        fit_multi_tissue(signals[:, :-1], lr, hr, lr_solver, hr_solver)


@pytest.mark.parametrize(
    ("mismatch", "resolution"),
    [("directions", "low-resolution"), ("measurements", "high-resolution")],
)
def test_fit_multi_tissue_refuses_a_solver_that_does_not_match_its_kernel(
    matlab_case: Case, mismatch: str, resolution: str
) -> None:
    signals, lr, hr, lr_nu, hr_nu, lr_solver, hr_solver = matlab_case
    if mismatch == "directions":
        # The high-resolution solver has 300 directions, the low-resolution kernel 45.
        lr_solver = hr_solver
    else:
        k = hr.w_k_aniso[:-1]
        hr_solver = _core.FodSolver.drl(
            np.ascontiguousarray(k.T), k.T @ k, nu=hr_nu, n_iter=10, init=0.1
        )
    with pytest.raises(ValueError, match=f"the {resolution} solver .* does not match"):
        fit_multi_tissue(signals, lr, hr, lr_solver, hr_solver)


def test_fit_multi_tissue_refuses_kernels_with_other_isotropic_tissues(
    matlab_case: Case,
) -> None:
    signals, lr, hr, _, _, lr_solver, hr_solver = matlab_case
    args = _deconv_args(hr)
    args["k_iso"] = np.ascontiguousarray(args["k_iso"][:, :1])
    args["w_k_iso"] = np.ascontiguousarray(args["w_k_iso"][:, :1])
    lr_k, hr_k = _core.DeconvKernel(**_deconv_args(lr)), _core.DeconvKernel(**args)
    with pytest.raises(ValueError, match="isotropic"):
        _core_fit(signals, lr.weights, lr_k, hr_k, (lr_solver, hr_solver))


def test_fit_multi_tissue_refuses_non_contiguous_signals_and_zero_threads(
    matlab_case: Case,
) -> None:
    signals, lr, _, _, _, lr_solver, hr_solver = matlab_case
    lr_k, hr_k = _core_kernels(matlab_case)
    solvers = (lr_solver, hr_solver)
    with pytest.raises(ValueError, match="C-contiguous"):
        _core_fit(np.asfortranarray(signals), lr.weights, lr_k, hr_k, solvers)
    with pytest.raises(ValueError, match="n_threads"):
        _core_fit(signals, lr.weights, lr_k, hr_k, solvers, n_threads=0)
