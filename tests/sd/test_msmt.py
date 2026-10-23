"""MSMT-CSD: the forward model, the constrained fit and MSMTCSDModel."""

import numpy as np
import pytest
from scipy.optimize import nnls

from larmor.sd import MSMTCSDModel, ZonalSHRF, find_peaks, fit_sh
from larmor.sd.msmt import (
    NORM_REG,
    constraint_matrix,
    fit_msmt,
    forward_matrix,
    prepare_msmt,
)
from larmor.sd.sh import build_basis_matrix
from larmor.sd.sphere import DIRS_300, get_dirs, get_theta_phi
from larmor.types import GradientTable, Volume

SHELLS = np.array([0.0, 1000.0, 2000.0, 3000.0])
FIBRE = np.array([0.6, 0.0, 0.8])


def _gradients() -> GradientTable:
    """Two b0s and 40 directions on each of three shells, rotated between shells."""
    dirs = get_dirs(40)
    bvecs = [np.zeros((2, 3))]
    for k in range(3):
        angle = 0.4 * k
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        bvecs.append(dirs @ rotation.T)
    bvals = np.concatenate([[0.0, 0.0], np.repeat(SHELLS[1:], 40)])
    return GradientTable(bvals, np.vstack(bvecs))


def _tensor_response(s0: float, parallel: float, perpendicular: float) -> ZonalSHRF:
    """Zonal coefficients up to lmax 8 of an axially symmetric tensor's signal."""
    nodes, weights = np.polynomial.legendre.leggauss(64)
    theta = np.arccos(nodes)
    rows = []
    for b in SHELLS:
        profile = s0 * np.exp(
            -b * (perpendicular + (parallel - perpendicular) * nodes**2)
        )
        basis = build_basis_matrix(theta, np.zeros_like(theta), 8)
        zonal = basis[:, [0, 3, 10, 21, 36]]  # the m = 0 columns of l = 0..8
        rows.append(2 * np.pi * (weights * profile) @ zonal)
    return ZonalSHRF(bvals=SHELLS, coefficients=np.array(rows))


def _isotropic_response(s0: float, diffusivity: float) -> ZonalSHRF:
    signal = s0 * np.exp(-SHELLS * diffusivity)
    return ZonalSHRF(bvals=SHELLS, coefficients=np.sqrt(4 * np.pi) * signal[:, None])


def test_forward_model_of_a_unit_peak_gives_back_the_response() -> None:
    # A narrow FOD of unit integral along d convolves to the response turned to d.
    gtab = _gradients()
    wm = _tensor_response(1000.0, 1.7e-3, 0.3e-3)
    forward = forward_matrix([wm], [8], gtab.bvals, gtab.bvecs)
    theta, phi = get_theta_phi(FIBRE[None])
    delta = build_basis_matrix(theta, phi, 8)[0]
    np.testing.assert_allclose(
        forward @ delta,
        wm.signal(gtab.bvals, gtab.bvecs, FIBRE[None])[:, 0],
        rtol=1e-10,
    )


def test_wm_and_csf_mixture_recovers_fractions_and_direction() -> None:
    # The FOD 0.7 * 9 / (4 pi) (d . u)^8 is non-negative, has degree 8 and integral
    # 0.7, so the constrained fit can recover it exactly.
    gtab = _gradients()
    wm = _tensor_response(1000.0, 1.7e-3, 0.2e-3)
    csf = _isotropic_response(2000.0, 3e-3)
    fod_true = fit_sh(0.7 * 9 / (4 * np.pi) * (DIRS_300 @ FIBRE) ** 8, DIRS_300, 8)
    forward = forward_matrix([wm, csf], [8, 0], gtab.bvals, gtab.bvecs)
    signal = forward @ np.concatenate([fod_true, [0.3 / np.sqrt(4 * np.pi)]])
    volume = Volume(np.tile(signal, (2, 1, 1, 1)), np.eye(4))

    fit = MSMTCSDModel([csf, wm]).fit(volume, gtab, n_jobs=1)

    assert fit.tissue_fractions is not None
    np.testing.assert_allclose(fit.tissue_fractions[0, 0, 0], (0.7, 0.3), atol=1e-6)
    np.testing.assert_allclose(fit.fods[0][0, 0, 0], fod_true, atol=1e-6)
    peak = find_peaks(fit.fods[0][0, 0, 0][None], num_peaks=1).directions[0, 0]
    assert abs(peak @ FIBRE) > np.cos(np.radians(0.1))


def test_fit_keeps_amplitudes_and_isotropic_coefficients_non_negative() -> None:
    gtab = _gradients()
    rfs = [_tensor_response(1000.0, 1.7e-3, 0.2e-3), _isotropic_response(2000.0, 3e-3)]
    design = prepare_msmt(rfs, [8, 0], gtab.bvals, gtab.bvecs, DIRS_300)
    rng = np.random.default_rng(0)
    noisy = 300.0 + 200.0 * rng.standard_normal((50, len(gtab.bvals)))
    coefficients, converged = fit_msmt(noisy, design, 2)
    theta, phi = get_theta_phi(DIRS_300)
    amplitudes = coefficients[:, :45] @ build_basis_matrix(theta, phi, 8).T
    assert converged.all()
    assert amplitudes.min() > -1e-9 and coefficients[:, 45].min() > -1e-9


def test_fit_meets_the_optimality_conditions() -> None:
    # Noisy voxels keep many constraints active. x minimises
    # ||A x - s||^2 + ridge^2 ||x||^2 subject to G x >= 0 exactly when its gradient
    # is G^T lambda with lambda >= 0 on the constraints it meets with equality (KKT),
    # which scipy's NNLS checks independently of the fit.
    gtab = _gradients()
    rfs = [_tensor_response(1000.0, 1.7e-3, 0.2e-3), _isotropic_response(2000.0, 3e-3)]
    design = prepare_msmt(rfs, [8, 0], gtab.bvals, gtab.bvecs, DIRS_300)
    forward = forward_matrix(rfs, [8, 0], gtab.bvals, gtab.bvecs)
    ridge_squared = NORM_REG * np.max(np.sum(forward**2, axis=0))
    constraints = constraint_matrix([8, 0], DIRS_300)
    rng = np.random.default_rng(1)
    noisy = 300.0 + 200.0 * rng.standard_normal((4, len(gtab.bvals)))
    ours, converged = fit_msmt(noisy, design, 1)
    assert converged.all()

    for signal, x in zip(noisy, ours, strict=True):
        values = constraints @ x
        assert values.min() > -1e-9
        active = values < 1e-9 * np.abs(values).max()
        assert active.sum() > 5
        gradient = forward.T @ (forward @ x - signal) + ridge_squared * x
        _, residual = nnls(constraints[active].T, gradient)
        assert residual < 1e-8 * np.linalg.norm(forward.T @ signal)


def test_refuses_more_isotropic_tissues_than_the_b_values_can_separate() -> None:
    # b = 0 and one shell cannot tell two isotropic tissues and an FOD's l = 0 apart.
    gtab = _gradients()
    one_shell = gtab.bvals <= 1000
    single = GradientTable(gtab.bvals[one_shell], gtab.bvecs[one_shell])
    rfs = [
        _tensor_response(1000.0, 1.7e-3, 0.2e-3),
        _isotropic_response(1300.0, 0.8e-3),
        _isotropic_response(2000.0, 3e-3),
    ]
    with pytest.raises(ValueError, match="determine only"):
        prepare_msmt(rfs, [8, 0, 0], single.bvals, single.bvecs, DIRS_300)


def test_needs_an_anisotropic_response_and_a_mask_to_estimate_responses() -> None:
    with pytest.raises(ValueError, match="lmax above 0"):
        MSMTCSDModel([_isotropic_response(1.0, 3e-3)])
    with pytest.raises(ValueError, match="basis"):
        MSMTCSDModel([_tensor_response(1000.0, 1.7e-3, 0.2e-3)], basis="foo")
    gtab = _gradients()
    volume = Volume(np.ones((2, 2, 2, len(gtab.bvals)), np.float32), np.eye(4))
    with pytest.raises(ValueError, match="mask"):
        MSMTCSDModel().fit(volume, gtab)
