"""Plain numpy deconvolution solvers, the reference of the compiled fit.

Plain numpy and scipy, one voxel at a time: damped Richardson-Lucy, MATLAB's
lsqnonneg and the multi-tissue (GRL) voxel fit, whose low-resolution loop uses
scipy's exact nnls.
"""

from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import nnls

from larmor import _core
from larmor.sd import IsotropicRF
from larmor.sd.kernel import Kernel
from larmor.sd.solvers import (
    FINAL_FRACTION_TOL_X,
    FRACTION_TOL,
    MAX_ROUNDS,
    MULTI_TISSUE_DRL_INIT,
)

FOD_FLOOR = 1e-6
DRL_INIT = 1e-6
DRL_POWER = 8.0
# FOD values this close to the median count as equal to it. The low-resolution scheme
# contains the +z direction twice; exact arithmetic keeps both FOD values equal, but
# rounding can put one just below the median and the other just above.
MEDIAN_TIE_RTOL = 1e-9


def drl_solver(
    fod_init: np.ndarray,
    signal: np.ndarray,
    k_t: np.ndarray,
    k_t_k: np.ndarray,
    nu: float,
    n_iter: int = 200,
) -> np.ndarray:
    """Damped Richardson-Lucy deconvolution of a single voxel.

    Args:
        fod_init: Initial FOD amplitudes, one per kernel column, shape (n_cols,).
        signal: Signal of the voxel, shape (n_meas,).
        k_t: Transpose of the kernel K, shape (n_cols, n_meas).
        k_t_k: Product K^T K, shape (n_cols, n_cols).
        nu: Damping threshold, to avoid fitting noise: FOD amplitudes below nu are
            treated as noise and their updates are damped.
        n_iter: Maximum number of iterations.

    Returns:
        The FOD amplitudes, same shape as fod_init.
    """
    fod = fod_init.copy()
    f_zero = FOD_FLOOR
    # Effectively, this defines how sharp the transition from valid signal to invalid
    # is.
    V = DRL_POWER
    nuV = nu**V

    kernel_s = k_t @ signal
    # MATLAB's std() normalises by N - 1, numpy's by N.
    n = signal.shape[0]
    mu = max(0.0, 1.0 - 4.0 * np.std(signal) * np.sqrt(n / (n - 1)))

    last_fod = fod.copy()
    eps = 1e-4 * np.max(signal)

    for _ in range(n_iter):
        dem = k_t_k @ fod
        fodV = fod**V

        Rk = 1.0 - (fodV / (fodV + nuV))
        Uk = 1.0 - mu * Rk

        RL_factor = 1.0 + Uk * ((kernel_s - dem) / (dem + 1e-16))
        fod = np.maximum(f_zero, fod * RL_factor)
        if np.sum(np.abs(fod - last_fod)) <= eps:
            break
        last_fod = fod.copy()

    return fod


def lsqnonneg(
    C: npt.NDArray[np.float64],  # noqa: N803 (MATLAB's name for the design matrix)
    d: npt.NDArray[np.float64],
    tol_x: float,
) -> npt.NDArray[np.float64]:
    """Non-negative least squares, ported from MATLAB's lsqnonneg (Lawson-Hanson).

    Unlike scipy's nnls it has MATLAB's TolX: a column enters the solution only when its
    Lagrange multiplier exceeds tol_x, and a coefficient that falls below tol_x is set
    back to 0. Used to reproduce MATLAB's final fraction fit exactly.

    Args:
        C: Design matrix, shape (n_measurements, n_columns).
        d: Signal, shape (n_measurements,).
        tol_x: MATLAB's TolX.

    Returns:
        The non-negative coefficients, shape (n_columns,).
    """
    n = C.shape[1]
    in_set = np.zeros(n, dtype=bool)  # MATLAB's P: columns in the solution
    out_set = np.ones(n, dtype=bool)  # MATLAB's Z: columns held at 0
    x = np.zeros(n)
    w = C.T @ (d - C @ x)
    n_iter, max_iter = 0, 3 * n
    while out_set.any() and (w[out_set] > tol_x).any():
        w_out = np.where(out_set, w, -np.inf)
        t = int(np.argmax(w_out))
        in_set[t], out_set[t] = True, False
        z = np.zeros(n)
        z[in_set] = np.linalg.lstsq(C[:, in_set], d, rcond=None)[0]
        while (z[in_set] <= 0).any():
            n_iter += 1
            if n_iter > max_iter:
                return z
            q = (z <= 0) & in_set
            alpha = np.min(x[q] / (x[q] - z[q]))
            x = x + alpha * (z - x)
            out_set = ((np.abs(x) < tol_x) & in_set) | out_set
            in_set = ~out_set
            z = np.zeros(n)
            z[in_set] = np.linalg.lstsq(C[:, in_set], d, rcond=None)[0]
        x = z
        w = C.T @ (d - C @ x)
    return x


def estimate_fractions(
    fod: np.ndarray,
    kernel: Kernel,
    signal: np.ndarray,
    per_block: bool,
    tol_x: float | None = None,
) -> np.ndarray:
    """Estimates the signal fractions by NNLS of the unweighted signal.

    The NNLS design matrix has one column per anisotropic block (the kernel times
    that block's FOD, scaled to max 1), followed by the isotropic columns.

    The FOD is median thresholded first: this removes the low floor dRL spreads over all
    directions, which would otherwise produce an almost isotropic signal counted as WM.
    With per_block=False all anisotropic blocks are lumped into one column.

    With tol_x=None the NNLS is exact; otherwise it uses MATLAB's lsqnonneg with that
    TolX (see lsqnonneg).
    """
    fod = fod.copy()
    fod[fod < np.median(fod) * (1 - MEDIAN_TIE_RTOL)] = 0.0
    cols = []

    if per_block:
        aniso_blocks = [
            blk
            for rf, blk in zip(kernel.rfs, kernel.blocks, strict=True)
            if not isinstance(rf, IsotropicRF)
        ]
    else:
        aniso_blocks = [slice(0, kernel.k_aniso.shape[1])]

    for blk in aniso_blocks:
        col = kernel.k_aniso[:, blk] @ fod[blk]
        if np.max(col) > 0:
            col = col / np.max(col)
        cols.append(col)

    # Only multi-tissue fits get here, so the kernel has isotropic columns.
    k_iso = cast("np.ndarray", kernel.k_iso)
    Y = np.column_stack([*cols, k_iso])
    if tol_x is None:
        p, _ = nnls(Y, signal)
        return p
    return lsqnonneg(Y, signal, tol_x)


def _low_res_fit(
    lr_kernel: Kernel,
    signal: np.ndarray,
    signal_w: np.ndarray,
    nu: float,
    n_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_aniso_lr = lr_kernel.k_aniso.shape[1]

    p_iso = np.zeros(lr_kernel.num_iso)
    p_old = np.inf * np.ones(1 + lr_kernel.num_iso)
    # Only multi-tissue fits get here, so the kernel has isotropic columns.
    w_k_iso = cast("np.ndarray", lr_kernel.w_k_iso)

    for _ in range(MAX_ROUNDS):
        iso_signal = w_k_iso @ p_iso
        aniso_signal = np.maximum(signal_w - iso_signal, 0)

        fod_lr = drl_solver(
            np.full(n_aniso_lr, DRL_INIT),
            aniso_signal,
            lr_kernel.w_k_aniso.T,
            lr_kernel.w_k_aniso.T @ lr_kernel.w_k_aniso,
            nu,
            n_iter,
        )
        p = estimate_fractions(fod_lr, lr_kernel, signal, per_block=False)
        p_iso = p[-lr_kernel.num_iso :]

        if np.all(np.abs(p - p_old) < FRACTION_TOL):
            break
        p_old = p

    return (p, p_iso)  # type: ignore[possibly-undefined]  # the loop always runs


def _high_res_fit(
    hr_kernel: Kernel,
    p_iso: np.ndarray,
    signal: np.ndarray,
    signal_w: np.ndarray,
    nu: float,
    n_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_aniso_hr = hr_kernel.k_aniso.shape[1]

    iso_signal = hr_kernel.w_k_iso @ p_iso
    aniso_signal = np.maximum(signal_w - iso_signal, 0)

    fod_hr = drl_solver(
        np.full(n_aniso_hr, DRL_INIT),
        aniso_signal,
        hr_kernel.w_k_aniso.T,
        hr_kernel.w_k_aniso.T @ hr_kernel.w_k_aniso,
        nu,
        n_iter,
    )

    p = estimate_fractions(
        fod_hr, hr_kernel, signal, per_block=True, tol_x=FINAL_FRACTION_TOL_X
    )
    fractions = p / (np.sum(p) + 1e-16)

    return fod_hr, fractions


def fit_multi_tissue_voxel(
    signal: np.ndarray,
    lr_kernel: Kernel,
    hr_kernel: Kernel,
    lr_nu: float,
    hr_nu: float,
    lr_n_iter: int = 200,
    hr_n_iter: int = 600,
) -> tuple[np.ndarray, np.ndarray]:
    """GRL fit of a single voxel.

    Args:
        signal: (n_meas,) signal normalised by b0, unweighted (the shell weights are
            taken from the kernels).
        lr_kernel: Low-resolution kernel, built with the same RFs as hr_kernel.
        hr_kernel: High-resolution kernel.
        lr_nu: Damping threshold of the low-resolution dRL fits.
        hr_nu: Damping threshold of the high-resolution dRL fit.
        lr_n_iter: dRL iterations asked for at low resolution.
        hr_n_iter: dRL iterations asked for at high resolution.

    Returns:
        fod: (n_aniso_cols,) high-resolution FOD amplitudes estimated.
        fractions: (n_aniso_rfs + num_iso,) signal fractions in kernel order, summing
            to 1.
    """
    signal_w = signal * lr_kernel.weights

    # MATLAB starts dRL from 0/0 = NaN, which its first iteration turns into DRL_INIT
    # (MATLAB's max ignores NaN): the effective iteration count is one less than
    # requested.
    _p, p_iso = _low_res_fit(lr_kernel, signal, signal_w, lr_nu, lr_n_iter - 1)
    return _high_res_fit(hr_kernel, p_iso, signal, signal_w, hr_nu, hr_n_iter - 1)


def matlab_drl(kernel: Kernel, nu: float, n_iter: int) -> _core.FodSolver:
    """The compiled dRL solver of a multi-tissue stage with a given nu, as in MATLAB.

    MATLAB starts dRL from 0/0 = NaN, which its first iteration turns into the floor:
    it runs one iteration less than asked, from MULTI_TISSUE_DRL_INIT.
    """
    k = kernel.w_k_aniso
    return _core.FodSolver.drl(
        np.ascontiguousarray(k.T),
        np.ascontiguousarray(k.T @ k),
        nu=nu,
        n_iter=n_iter - 1,
        init=MULTI_TISSUE_DRL_INIT,
    )
