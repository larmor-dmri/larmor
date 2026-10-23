"""Plain numpy least-squares fits of the log signal, the reference of the compiled fit.

Ordinary and iterated weighted least squares, vectorised over voxels, and the
constrained DKI fit one voxel at a time through scipy's nnls.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls

from larmor.dwi.dki import _constraint_matrix
from larmor.types import GradientTable

# Reweighting stops in a voxel once no fitted log signal moves by more than this
# between two iterations (a relative change of the predicted signal of 1e-6).
LOG_SIGNAL_TOL = 1e-6
# Reweightings after the first, unweighted fit, at most.
MAX_REWEIGHTINGS = 20
# Memory, in bytes, for the per-voxel arrays of one chunk of voxels.
CHUNK_BYTES = 64 * 2**20


def ols_fit(design: NDArray[np.float64], s: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ordinary linear least squares of the log signal, voxel by voxel.

    Every measurement has the same weight: the parameters minimise
    ||log(s) - design @ p||^2 in each voxel.

    Args:
        design (np.ndarray): Design matrix of the log signal, shape (N, P).
        s (np.ndarray): Positive signal, shape (n_voxels, N).

    Returns:
        np.ndarray: Parameters, shape (P, n_voxels).
    """
    scale = _column_scale(design)
    p: NDArray[np.float64] = np.linalg.pinv(design / scale) @ np.log(s).T
    return p / scale[:, None]


def wls_fit(design: NDArray[np.float64], s: NDArray[np.float64]) -> NDArray[np.float64]:
    """Weighted linear least squares of the log signal, voxel by voxel.

    The noise of log(S) has a standard deviation of about sigma / S, so each
    measurement is weighted by the square of its signal. The weights come from the
    signal the fit predicts, not from the measured signal, whose noise would bias
    the estimate (Veraart et al., 2013): an ordinary least-squares fit gives the
    first prediction, and each weighted fit gives the weights of the next. A voxel
    stops once no predicted log signal changes by more than 1e-6 between two
    iterations, or after 20 weighted fits. A voxel whose result is not finite
    falls back to ordinary least squares. Voxels are fitted in chunks that bound
    the memory used.

    References:
        Veraart J, Sijbers J, Sunaert S, Leemans A, Jeurissen B. Weighted linear
        least squares estimation of diffusion MRI parameters: strengths,
        limitations, and pitfalls. NeuroImage 81 (2013) 335-346.

    Args:
        design (np.ndarray): Design matrix of the log signal, shape (N, P).
        s (np.ndarray): Positive signal, shape (n_voxels, N).

    Returns:
        np.ndarray: Parameters, shape (P, n_voxels).
    """
    n_meas, n_par = design.shape
    # Columns of b and b^2 differ in scale by orders of magnitude; scaling them to
    # unit norm keeps the weighted normal equations well conditioned.
    scale = _column_scale(design)
    x = design / scale
    pinv = np.linalg.pinv(x)
    log_s = np.log(s)
    params = np.empty((n_par, len(s)))
    chunk = max(1, CHUNK_BYTES // (8 * n_meas * (n_par + 4)))
    for start in range(0, len(s), chunk):
        y = log_s[start : start + chunk]
        ols = y @ pinv.T
        p = _reweight(x, y, ols)
        bad = ~np.isfinite(p).all(axis=1)
        p[bad] = ols[bad]
        params[:, start : start + chunk] = p.T
    result: NDArray[np.float64] = params / scale[:, None]
    return result


def _reweight(
    x: NDArray[np.float64], y: NDArray[np.float64], p: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Iterated weighted fits of log signals y (n, N) from parameters p (n, P)."""
    p = p.copy()
    active = np.isfinite(p).all(axis=1)
    pred = p @ x.T
    for _ in range(MAX_REWEIGHTINGS):
        idx = np.flatnonzero(active)
        if len(idx) == 0:
            break
        # Weights S^2, relative to the voxel's largest predicted signal
        # so that exp cannot overflow; a common factor does not change the fit.
        a_pred = pred[idx]
        w = np.exp(2 * (a_pred - a_pred.max(axis=1, keepdims=True)))
        lhs = np.einsum("vn,np,nq->vpq", w, x, x)
        rhs = np.einsum("vn,np->vp", w * y[idx], x)
        new = _solve(lhs, rhs)
        new_pred = new @ x.T
        finite = np.isfinite(new).all(axis=1)
        change = np.where(finite, np.abs(new_pred - a_pred).max(axis=1), np.inf)
        p[idx] = new
        pred[idx] = new_pred
        # A voxel whose fit broke down stops here and takes the fallback.
        active[idx] = finite & (change > LOG_SIGNAL_TOL)
    return p


def _solve(lhs: NDArray[np.float64], rhs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solves lhs (n, P, P) p = rhs (n, P) per voxel; NaN where singular."""
    try:
        sol: NDArray[np.float64] = np.linalg.solve(lhs, rhs[..., None])[..., 0]
        return sol
    except np.linalg.LinAlgError:
        out = np.full(rhs.shape, np.nan)
        for i in range(len(rhs)):
            try:
                out[i] = np.linalg.solve(lhs[i], rhs[i])
            except np.linalg.LinAlgError:
                pass
        return out


def _column_scale(design: NDArray[np.float64]) -> NDArray[np.float64]:
    """The norm of each design column, 1 for a zero column."""
    norms = np.linalg.norm(design, axis=0)
    scale: NDArray[np.float64] = np.where(norms > 0, norms, 1.0)
    return scale


def constrain(
    design: np.ndarray,
    log_s: np.ndarray,
    params: np.ndarray,
    gtab: GradientTable,
    weighted: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Parameters (P, n), constrained where a constraint is broken, and those voxels."""
    # Unit-norm columns keep the triangular solves well conditioned.
    norms = np.linalg.norm(design, axis=0)
    scale = np.where(norms > 0, norms, 1.0)
    x = design / scale
    g = _constraint_matrix(gtab) / scale
    p = params * scale[:, None]
    finite = np.isfinite(p).all(axis=0)
    broke = finite & ((g @ np.where(finite, p, 0.0)) < 0).any(axis=0)
    violating = np.flatnonzero(broke)
    out = params.copy()
    unweighted = None if weighted else np.linalg.qr(x)
    rhs = np.zeros(x.shape[1] + 1)
    rhs[-1] = 1.0
    for v in violating:
        y = log_s[v]
        if unweighted is None:
            pred = x @ p[:, v]
            sqrt_w = np.exp(pred - pred.max())
            q, r = np.linalg.qr(sqrt_w[:, None] * x)
            y = sqrt_w * y
        else:
            q, r = unweighted
        p_free = np.linalg.solve(r, q.T @ y)
        # Least distance: min ||z|| s.t. a z >= c, with z = r (p - p_free), solved
        # through the NNLS problem min ||[a^T; c^T] u - e|| over u >= 0.
        a = np.linalg.solve(r.T, g.T).T
        c = -(g @ p_free)
        u, _ = nnls(np.vstack([a.T, c]), rhs, maxiter=50 * len(c))
        res = np.vstack([a.T, c]) @ u - rhs
        if res[-1] >= -1e-12:
            continue
        z = -res[:-1] / res[-1]
        out[:, v] = (p_free + np.linalg.solve(r, z)) / scale
    return out, broke
