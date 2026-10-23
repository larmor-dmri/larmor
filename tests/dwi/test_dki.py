import dataclasses
import itertools
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from reference_fitting import constrain, ols_fit, wls_fit
from scipy.optimize import LinearConstraint, minimize

from larmor.dwi import DKIFit, DKIModel
from larmor.dwi.dki import (
    KT_INDICES,
    KT_MULTIPLICITY,
    N_CONSTRAINT_DIRS,
    _constraint_matrix,
    _design_matrix,
    _fibonacci_half_sphere,
    compute_kurtosis_metrics,
)
from larmor.dwi.fitting import fit_log_signal
from larmor.io import load_dwi
from larmor.types import GradientTable, Volume

Array = npt.NDArray[np.float64]
DKI_DIR = Path(__file__).parents[1] / "data" / "dki"
S0 = 1000.0


def _unit(v: Array) -> Array:
    out: Array = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return out


def _rotation(seed: int) -> Array:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    rot: Array = q
    return rot


def _tensor(eigen_vals: list[float], seed: int = 0) -> Array:
    r = _rotation(seed)
    d: Array = r @ np.diag(eigen_vals) @ r.T
    return d


def _isotropic_w(k: float) -> Array:
    i = np.eye(3)
    w: Array = (
        k
        / 3
        * (
            np.einsum("ij,kl->ijkl", i, i)
            + np.einsum("ik,jl->ijkl", i, i)
            + np.einsum("il,jk->ijkl", i, i)
        )
    )
    return w


def _general_w(seed: int, spread: float = 0.15, k: float = 0.8) -> Array:
    """A fully symmetric kurtosis tensor: a symmetrised random one plus an isotropic."""
    a = np.random.default_rng(seed).normal(size=(3, 3, 3, 3)) * spread
    sym = sum(np.transpose(a, p) for p in itertools.permutations(range(4))) / 24
    w: Array = sym + _isotropic_w(k)
    return w


def _d6(d: Array) -> Array:
    return d[[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]


def _w15(w: Array) -> Array:
    return np.array([w[q] for q in KT_INDICES])


def _gradients() -> GradientTable:
    rng = np.random.default_rng(7)
    bvals = np.concatenate([np.zeros(3), np.full(30, 1000.0), np.full(45, 2000.0)])
    bvecs = np.vstack([np.zeros((3, 3)), _unit(rng.normal(size=(75, 3)))])
    return GradientTable(bvals, bvecs)


GTAB = _gradients()
D_WM = _tensor([1.7e-3, 0.5e-3, 0.4e-3])
W_WM = _general_w(1, spread=0.1, k=0.6)
D_ISO = np.eye(3) * 0.8e-3
W_ISO = _isotropic_w(0.9)


def _signal(d: Array, w: Array, s0: float = S0) -> Array:
    """Jensen's DKI signal on GTAB."""
    g, b = GTAB.bvecs, GTAB.bvals
    md = np.trace(d) / 3
    dn = np.einsum("ni,ij,nj->n", g, d, g)
    wn = np.einsum("ni,nj,nk,nl,ijkl->n", g, g, g, g, w)
    s: Array = s0 * np.exp(-b * dn + b**2 / 6 * md**2 * wn)
    # Below S0 everywhere, so the cap at S0 leaves the synthetic signal alone.
    assert (s[b > 0] < s0).all()
    return s


def _volume(signals: Array) -> Volume:
    """A volume of shape (X, 1, 1, N) from signals (X, N)."""
    return Volume(np.ascontiguousarray(signals[:, None, None, :]), np.eye(4))


def _fit_wm(fit_method: str) -> DKIFit:
    signals = np.stack([_signal(D_WM, W_WM), _signal(D_ISO, W_ISO)])
    return DKIModel(fit_method=fit_method).fit(
        _volume(signals), GTAB, np.ones((2, 1, 1), bool)
    )


@pytest.mark.parametrize("fit_method", ["ols", "wls"])
def test_noise_free_fit_recovers_the_tensors(fit_method: str) -> None:
    fit = _fit_wm(fit_method)
    # The maps are float32: diffusivities (~1e-3) hold about 1e-10, W (~1) 1e-7.
    for x, d, w in ((0, D_WM, W_WM), (1, D_ISO, W_ISO)):
        np.testing.assert_allclose(fit.dt[x, 0, 0], _d6(d), rtol=0, atol=1e-9)
        np.testing.assert_allclose(fit.kt[x, 0, 0], _w15(w), rtol=0, atol=1e-6)
        np.testing.assert_allclose(fit.s0[x, 0, 0], S0, rtol=1e-6)


def test_isotropic_tensors_give_equal_kurtoses_and_no_anisotropy() -> None:
    fit = _fit_wm("wls")
    for k in (fit.mk, fit.ak, fit.rk):
        np.testing.assert_allclose(k[1, 0, 0], 0.9, rtol=1e-6)
    assert abs(fit.ka[1, 0, 0]) < 1e-6
    assert fit.ka[0, 0, 0] > 0.01


def _bad_table(case: str) -> GradientTable:
    rng = np.random.default_rng(8)
    dirs = _unit(rng.normal(size=(30, 3)))
    if case == "one shell":
        bvals = np.concatenate([np.zeros(3), np.full(30, 1000.0)])
        return GradientTable(bvals, np.vstack([np.zeros((3, 3)), dirs]))
    if case == "no b0":
        bvals = np.repeat([1000.0, 2000.0], 30)
        return GradientTable(bvals, np.vstack([dirs, dirs]))
    # Ten directions repeated on both shells: their quartic terms span ten of 15.
    bvals = np.concatenate([np.zeros(3), np.repeat([1000.0, 2000.0], 10)])
    return GradientTable(bvals, np.vstack([np.zeros((3, 3)), dirs[:10], dirs[:10]]))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("one shell", "at least two non-zero"),
        ("no b0", "cannot determine"),
        ("ten directions", "cannot determine"),
    ],
)
def test_table_that_cannot_determine_the_tensors_is_refused(
    case: str, message: str
) -> None:
    gtab = _bad_table(case)
    volume = Volume(np.ones((1, 1, 1, len(gtab.bvals))), np.eye(4))
    with pytest.raises(ValueError, match=message):
        DKIModel().fit(volume, gtab, np.ones((1, 1, 1), bool))


def _cube_with_dark_centre() -> Volume:
    data = np.broadcast_to(_signal(D_WM, W_WM), (3, 3, 3, len(GTAB.bvals))).copy()
    data[1, 1, 1, GTAB.b0_mask] = 0.0
    return Volume(data, np.eye(4))


@pytest.mark.parametrize("fit_method", ["ols", "wls"])
def test_voxel_with_zero_b0s_is_fitted_with_its_neighbours_s0(fit_method: str) -> None:
    fit = DKIModel(fit_method=fit_method).fit(
        _cube_with_dark_centre(), GTAB, np.ones((3, 3, 3), bool)
    )
    np.testing.assert_allclose(fit.s0[1, 1, 1], S0, rtol=1e-6)
    np.testing.assert_allclose(fit.dt[1, 1, 1], _d6(D_WM), rtol=0, atol=1e-9)
    np.testing.assert_allclose(fit.kt[1, 1, 1], _w15(W_WM), rtol=0, atol=1e-6)


def _volume_with_an_unusable_s0(case: str) -> Volume:
    signals = np.tile(_signal(D_WM, W_WM), (6, 1))
    if case == "nan b0":
        signals[0, 0] = np.nan
    else:
        # Voxels 0 to 2 are dark; voxel 0 has no plausible voxel within two.
        signals[:3][:, GTAB.b0_mask] = 0.0
    return _volume(signals)


@pytest.mark.parametrize("case", ["nan b0", "no plausible neighbour"])
def test_voxel_without_a_usable_s0_is_zero_in_every_map(
    case: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="larmor"):
        fit = DKIModel().fit(
            _volume_with_an_unusable_s0(case), GTAB, np.ones((6, 1, 1), bool)
        )
    assert "1 mask voxels have no usable S0" in caplog.text
    for field in dataclasses.fields(fit):
        values = getattr(fit, field.name)
        assert not values[0].any(), field.name
        # The non-positive-definite flag is False in every fitted voxel here.
        assert field.name == "npd" or values[1:].any(), field.name
    np.testing.assert_allclose(fit.dt[1:, 0, 0], np.tile(_d6(D_WM), (5, 1)), atol=1e-9)


def test_measurement_above_s0_fits_as_if_it_were_at_s0() -> None:
    above, at = (np.stack([_signal(D_WM, W_WM)]) for _ in range(2))
    # S0 is the mean of identical b0s, exactly S0.
    above[0, 10] = 5 * S0
    at[0, 10] = S0
    mask = np.ones((1, 1, 1), bool)
    fit_above = DKIModel().fit(_volume(above), GTAB, mask)
    fit_at = DKIModel().fit(_volume(at), GTAB, mask)
    for name in ("dt", "kt", "s0"):
        np.testing.assert_array_equal(getattr(fit_above, name), getattr(fit_at, name))


# Brute force: Gauss-Legendre in cos(theta) by equal azimuths on the whole sphere,
# converged to 1e-10 for these tensors; 128 equal steps on the radial circle.
_T, _TW = np.polynomial.legendre.leggauss(64)
_PHI = 2 * np.pi * np.arange(128) / 128
_SPHERE = np.stack(
    [
        np.outer(np.sqrt(1 - _T**2), np.cos(_PHI)).ravel(),
        np.outer(np.sqrt(1 - _T**2), np.sin(_PHI)).ravel(),
        np.repeat(_T, len(_PHI)),
    ],
    axis=1,
)
_SPHERE_WEIGHTS = np.repeat(_TW, len(_PHI)) / (2 * len(_PHI))


def _apparent_kurtosis(d: Array, w: Array, n: Array) -> Array:
    md = np.trace(d) / 3
    wn = np.einsum("ni,nj,nk,nl,ijkl->n", n, n, n, n, w)
    k: Array = md**2 * wn / np.einsum("ni,ij,nj->n", n, d, n) ** 2
    return k


def _brute_force(d: Array, w: Array) -> Array:
    """MK, AK, RK and KA from their definitions as means of K(n)."""
    k = _apparent_kurtosis(d, w, _SPHERE)
    mk = k @ _SPHERE_WEIGHTS
    ka = np.sqrt(((k - mk) ** 2) @ _SPHERE_WEIGHTS)
    _, vecs = np.linalg.eigh(d)
    circle = np.outer(np.cos(_PHI), vecs[:, 0]) + np.outer(np.sin(_PHI), vecs[:, 1])
    ak = _apparent_kurtosis(d, w, vecs[:, 2:].T)[0]
    rk = _apparent_kurtosis(d, w, circle).mean()
    return np.array([mk, ak, rk, ka])


def _metrics(d: Array, w: Array) -> Array:
    return np.array(
        [m[0] for m in compute_kurtosis_metrics(_d6(d)[:, None], _w15(w)[:, None])]
    )


GAP = 1e-5
CASES = {
    "general": [1.7, 0.4, 0.3],
    "gap just above the extrapolation": [1.0, 1.0 - 7e-4, 0.4],
    "prolate": [1.7, 0.3, 0.3],
    "nearly prolate": [1.7, 0.3, 0.3 * (1 - GAP)],
    "nearly oblate": [1.5 * (1 + GAP), 1.5, 0.3],
    "nearly isotropic": [0.8 * (1 + GAP), 0.8, 0.8 * (1 - GAP)],
}


@pytest.mark.parametrize("eigen_vals", CASES.values(), ids=CASES.keys())
def test_kurtosis_metrics_match_their_definitions(eigen_vals: list[float]) -> None:
    d = _tensor([e * 1e-3 for e in eigen_vals], seed=2)
    w = _general_w(3)
    # MK, AK and RK: closed form, and where eigenvalues are within 3e-4 MD a
    # quadratic extrapolation (measured error below 3e-8). KA: a 16 x 32 quadrature
    # on the half sphere, which resolves K(n) of these tensors to about 1e-4.
    expected = _brute_force(d, w)
    np.testing.assert_allclose(_metrics(d, w)[:3], expected[:3], rtol=1e-7)
    np.testing.assert_allclose(_metrics(d, w)[3], expected[3], rtol=1e-4)


@pytest.mark.parametrize(
    "eigen_vals", [[1.5, 1.5, 0.3], [0.8, 0.8, 0.8]], ids=["oblate", "isotropic"]
)
def test_mean_kurtosis_and_anisotropy_hold_where_e1_is_not_unique(
    eigen_vals: list[float],
) -> None:
    # AK and RK depend on which vector of the degenerate plane is e1; MK and KA do not.
    d = _tensor([e * 1e-3 for e in eigen_vals], seed=2)
    w = _general_w(3)
    mk, _, _, ka = _metrics(d, w)
    expected = _brute_force(d, w)
    np.testing.assert_allclose(mk, expected[0], rtol=1e-7)
    np.testing.assert_allclose(ka, expected[3], rtol=1e-4)


def test_very_anisotropic_tensor_keeps_closed_form_metrics_exact() -> None:
    d = _tensor([2.5e-3, 0.15e-3, 0.1e-3], seed=2)
    w = _general_w(3)
    np.testing.assert_allclose(_metrics(d, w)[:3], _brute_force(d, w)[:3], rtol=1e-7)


def test_isotropic_d_and_w_give_zero_kurtosis_anisotropy() -> None:
    mk, ak, rk, ka = _metrics(D_ISO, W_ISO)
    # MK and RK of equal eigenvalues are extrapolated.
    np.testing.assert_allclose([mk, ak, rk], 0.9, rtol=1e-7)
    assert abs(ka) < 1e-12


@pytest.mark.parametrize(
    "eigen_vals",
    [[1.7e-3, 0.4e-3, -0.1e-3], [1.7e-3, 0.4e-3, 0.0], [np.nan, 0.4e-3, 0.3e-3]],
    ids=["negative", "zero", "nan"],
)
def test_metrics_are_zero_where_d_is_not_positive_definite(
    eigen_vals: list[float],
) -> None:
    # Diagonal, so that the zero eigenvalue is not rounded off zero by a rotation.
    d = np.diag(eigen_vals)
    np.testing.assert_array_equal(_metrics(d, _general_w(3)), 0.0)


# Errors relative to the voxel's largest |eigenvalue| (DT) or largest |W| (KT).
# Measured on the 511 voxels with every measurement positive and below S0: at most
# 3.0e-6 (DT) and 2.0e-5 (KT) for OLS, and on the 453 of them where MRtrix's WLS
# has settled, 1.0e-6 and 1.2e-5 for WLS, the size of the float32 data and outputs
# carried through the fit. WLS differs in its stopping rule (MRtrix: 10
# reweightings, Larmor: until the fit settles), so its comparison leaves out the
# voxels where MRtrix had not settled.
DT_RTOL = 1e-5
KT_RTOL = 5e-5


@pytest.mark.parametrize("fit_method", ["ols", "wls"])
def test_fit_matches_mrtrix_dwi2tensor_on_the_block(fit_method: str) -> None:
    volume, gtab = load_dwi(
        DKI_DIR / "dwi.nii.gz",
        bvals=DKI_DIR / "bvals",
        bvecs=DKI_DIR / "bvecs",
        frame="fsl",
    )
    ref = np.load(DKI_DIR / "mrtrix_dkt_ref.npz")
    mask = ref["mask"]
    # MRtrix fits without constraints.
    fit = DKIModel(fit_method=fit_method, constrained=False).fit(volume, gtab, mask)

    s = volume.as_float()[mask].astype(np.float64)
    is_b0 = gtab.b0_mask
    # Larmor caps measurements at S0 and floors non-positive ones; MRtrix does
    # neither. The block's one b0 is plausible, so S0 is that b0.
    s0 = s[:, is_b0].mean(axis=1)
    kept = (s > 0).all(axis=1) & (s[:, ~is_b0] <= s0[:, None]).all(axis=1)
    assert kept.sum() == 511
    if fit_method == "wls":
        kept &= ref["wls_settled"]

    ref_dt = ref[f"dt_{fit_method}"][kept].astype(np.float64)
    ref_kt = ref[f"kt_{fit_method}"][kept].astype(np.float64)
    matrices = ref_dt[:, [0, 1, 2, 1, 3, 4, 2, 4, 5]].reshape(-1, 3, 3)
    dt_scale = np.abs(np.linalg.eigvalsh(matrices)).max(axis=1)
    kt_scale = np.abs(ref_kt).max(axis=1)
    dt_err = np.abs(fit.dt[mask][kept] - ref_dt).max(axis=1) / dt_scale
    kt_err = np.abs(fit.kt[mask][kept] - ref_kt).max(axis=1) / kt_scale
    assert dt_err.max() < DT_RTOL
    assert kt_err.max() < KT_RTOL


# Constrained fit. A thin tensor (radial diffusivity 1e-4) with a kurtosis inside
# the bound K(n) <= 3 / (b_max D(n)); noise on its diffusion-weighted signal only,
# so that S0 is exact and no measurement reaches it.
D_THIN = _tensor([1.7e-3, 0.1e-3, 0.1e-3], seed=4)
W_THIN = _isotropic_w(0.2)


def _thin_signals(n: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    s = np.tile(_signal(D_THIN, W_THIN), (n, 1))
    dw = GTAB.bvals > 0
    s[:, dw] = np.abs(s[:, dw] + rng.normal(scale=20.0, size=(n, int(dw.sum()))))
    assert (s[:, dw] < S0).all()
    return s


def _model_rows() -> tuple[Array, Array]:
    """Rows of D(n) on dt and of W(n) on kt, along the constraint directions."""
    # The directions are the fit's own: the constraints hold only along them.
    n = _fibonacci_half_sphere(N_CONSTRAINT_DIRS)
    x, y, z = n.T
    d_rows = np.stack([x * x, 2 * x * y, 2 * x * z, y * y, 2 * y * z, z * z], axis=1)
    w_rows = np.stack(
        [
            m * n[:, i] * n[:, j] * n[:, k] * n[:, q]
            for (i, j, k, q), m in zip(KT_INDICES, KT_MULTIPLICITY, strict=True)
        ],
        axis=1,
    )
    return d_rows, w_rows


def _quartic_rows(g: Array) -> Array:
    rows: Array = np.stack(
        [
            m * g[:, i] * g[:, j] * g[:, k] * g[:, q]
            for (i, j, k, q), m in zip(KT_INDICES, KT_MULTIPLICITY, strict=True)
        ],
        axis=1,
    )
    return rows


def _constraints() -> Array:
    """G p >= 0 for p = (log S0, D, MD^2 W): D(n), V(n), 3 D(n) / b_max - V(n)."""
    d_rows, w_rows = _model_rows()
    zero = np.zeros((len(d_rows), 1))
    bound = 3 / GTAB.bvals.max()
    g: Array = np.vstack(
        [
            np.hstack([zero, d_rows, 0 * w_rows]),
            np.hstack([zero, 0 * d_rows, w_rows]),
            np.hstack([zero, bound * d_rows, -w_rows]),
        ]
    )
    return g


def _design() -> Array:
    """Jensen's log-signal model on GTAB, linear in p = (log S0, D, MD^2 W)."""
    g, b = GTAB.bvecs, GTAB.bvals
    x, y, z = g.T
    dt_rows = np.stack([x * x, 2 * x * y, 2 * x * z, y * y, 2 * y * z, z * z], axis=1)
    design: Array = np.column_stack(
        [np.ones(len(b)), -b[:, None] * dt_rows, b[:, None] ** 2 / 6 * _quartic_rows(g)]
    )
    return design


def _params(fit: DKIFit) -> Array:
    """p = (log S0, D, MD^2 W) of every voxel of a (X, 1, 1) fit, shape (X, 22)."""
    md = fit.md[:, 0, 0, None].astype(np.float64)
    return np.column_stack(
        [
            np.log(fit.s0[:, 0, 0].astype(np.float64)),
            fit.dt[:, 0, 0].astype(np.float64),
            md**2 * fit.kt[:, 0, 0].astype(np.float64),
        ]
    )


def _true_params(d: Array, w: Array) -> Array:
    return np.concatenate([[np.log(S0)], _d6(d), (np.trace(d) / 3) ** 2 * _w15(w)])


def _same_voxels(a: DKIFit, b: DKIFit) -> npt.NDArray[np.bool_]:
    same = np.ones(len(a.md), bool)
    for field in dataclasses.fields(a):
        x, y = getattr(a, field.name), getattr(b, field.name)
        same &= (x == y).reshape(len(x), -1).all(axis=1)
    return same


def _both_fits(
    fit_method: str, signals: Array, caplog: pytest.LogCaptureFixture
) -> tuple[DKIFit, DKIFit, int]:
    """Constrained and plain fits, and how many voxels the log says were constrained."""
    volume, mask = _volume(signals), np.ones((len(signals), 1, 1), bool)
    with caplog.at_level(logging.INFO, logger="larmor"):
        constrained = DKIModel(fit_method=fit_method).fit(volume, GTAB, mask)
    count = int(caplog.text.split(" of ")[0].split()[-1])
    plain = DKIModel(fit_method=fit_method, constrained=False).fit(volume, GTAB, mask)
    return constrained, plain, count


@pytest.mark.parametrize("fit_method", ["ols", "wls"])
def test_constrained_fit_meets_every_constraint_where_the_plain_fit_broke_one(
    fit_method: str, caplog: pytest.LogCaptureFixture
) -> None:
    constrained, plain, _ = _both_fits(fit_method, _thin_signals(20, 9), caplog)
    changed = ~_same_voxels(constrained, plain)
    g = _constraints()
    assert changed.any()
    assert ((_params(plain)[changed] @ g.T) < 0).any(axis=1).all()
    # The maps are float32: D(n) (~1e-4) holds about 1e-11, V(n) (~1e-7) 1e-14.
    values = _params(constrained)[changed] @ g.T
    n = N_CONSTRAINT_DIRS
    assert values[:, :n].min() > -1e-10
    assert values[:, n:].min() > -1e-12


@pytest.mark.parametrize("fit_method", ["ols", "wls"])
def test_voxels_meeting_every_constraint_keep_the_plain_fit_exactly(
    fit_method: str, caplog: pytest.LogCaptureFixture
) -> None:
    signals = np.vstack([_signal(D_WM, W_WM), _thin_signals(20, 9)])
    constrained, plain, count = _both_fits(fit_method, signals, caplog)
    same = _same_voxels(constrained, plain)
    feasible = ((_params(plain) @ _constraints().T) >= 0).all(axis=1)
    assert same[0] and feasible[0]
    np.testing.assert_array_equal(same, feasible)
    assert count == int((~feasible).sum()) > 0


def _objective(
    design: Array, y: Array, weights: Array
) -> tuple[Callable[[Array], float], Callable[[Array], Array]]:
    def value(p: Array) -> float:
        r = design @ p - y
        return float(weights @ r**2)

    def gradient(p: Array) -> Array:
        grad: Array = 2 * design.T @ (weights * (design @ p - y))
        return grad

    return value, gradient


@pytest.mark.parametrize("fit_method", ["ols", "wls"])
def test_constrained_solution_is_the_constrained_least_squares_optimum(
    fit_method: str, caplog: pytest.LogCaptureFixture
) -> None:
    signals = _thin_signals(1, 11)
    constrained, plain, count = _both_fits(fit_method, signals, caplog)
    assert count == 1
    design, g = _design(), _constraints()
    # WLS keeps the weights of the plain weighted fit's predicted signal.
    weights = np.ones(len(design))
    if fit_method == "wls":
        pred = design @ _params(plain)[0]
        weights = np.exp(2 * (pred - pred.max()))
    # Columns scaled to unit norm, as the b^2 terms are 1e6 times the b0 column.
    scale = np.linalg.norm(design, axis=0)
    value, gradient = _objective(design / scale, np.log(signals[0]), weights)
    truth = _true_params(D_THIN, W_THIN) * scale
    assert (g / scale @ truth >= 0).all()
    reference = minimize(
        value,
        truth,
        jac=gradient,
        method="SLSQP",
        constraints=LinearConstraint(g / scale, lb=0.0),
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    assert reference.success, reference.message
    fitted = value(_params(constrained)[0] * scale)
    assert fitted <= value(truth)
    # Measured within 3e-9: the float32 maps move the objective that little.
    np.testing.assert_allclose(fitted, reference.fun, rtol=1e-7)


@pytest.mark.parametrize("fit_method", ["ols", "wls"])
def test_thin_noisy_tensors_get_bounded_kurtoses(fit_method: str) -> None:
    volume, mask = _volume(_thin_signals(20, 9)), np.ones((20, 1, 1), bool)
    plain = DKIModel(fit_method=fit_method, constrained=False).fit(volume, GTAB, mask)
    fit = DKIModel(fit_method=fit_method).fit(volume, GTAB, mask)
    # Unconstrained, noise drives some MK negative. Constrained, 0 <= K(n) <=
    # 3 / (b_max D(n)) along the constraint directions, so the means of K lie
    # within [0, 3 / (b_max l3)], with a margin for the directions in between.
    assert (plain.mk < 0).any()
    assert (fit.eigen_vals[..., 2] > 0).all()
    bound = 1.05 * 3 / (GTAB.bvals.max() * fit.eigen_vals[..., 2])
    for k in (fit.mk, fit.ak, fit.rk):
        assert (k >= 0).all() and (k <= bound).all()


def _block_signals() -> tuple[Array, GradientTable]:
    """The test block's mask voxels with every measurement positive, and its table."""
    volume, gtab = load_dwi(
        DKI_DIR / "dwi.nii.gz",
        bvals=DKI_DIR / "bvals",
        bvecs=DKI_DIR / "bvecs",
        frame="fsl",
    )
    mask = np.load(DKI_DIR / "mrtrix_dkt_ref.npz")["mask"]
    s = volume.as_float()[mask].astype(np.float64)
    return s[(s > 0).all(axis=1)], gtab


@pytest.mark.parametrize("constrained", [False, True], ids=["plain", "constrained"])
@pytest.mark.parametrize("fit_method", ["ols", "wls"])
def test_compiled_fit_matches_the_numpy_reference_on_the_block(
    fit_method: str, constrained: bool
) -> None:
    s, gtab = _block_signals()
    design = _design_matrix(gtab)
    weighted = fit_method == "wls"
    g = _constraint_matrix(gtab) if constrained else None
    params, broke = fit_log_signal(design, s, weighted, g, n_threads=2)

    expected = (wls_fit if weighted else ols_fit)(design, s)
    expected_broke = np.zeros(len(s), bool)
    if constrained:
        expected, expected_broke = constrain(
            design, np.log(s), expected, gtab, weighted
        )
    np.testing.assert_array_equal(broke, expected_broke)
    assert broke.sum() > 100 if constrained else not broke.any()
    # Parameters times the column norms are of order 1 to 10 (log S0). The fits
    # differ in summation order and, constrained, in the NNLS solver; measured
    # within 6e-12.
    scale = np.linalg.norm(design, axis=0)[:, None]
    np.testing.assert_allclose(params * scale, expected * scale, rtol=0, atol=1e-10)


def test_compiled_fit_gives_the_same_bits_on_any_number_of_threads() -> None:
    s, gtab = _block_signals()
    design, g = _design_matrix(gtab), _constraint_matrix(gtab)
    one, _ = fit_log_signal(design, s, True, g, n_threads=1)
    four, _ = fit_log_signal(design, s, True, g, n_threads=4)
    np.testing.assert_array_equal(one, four)
