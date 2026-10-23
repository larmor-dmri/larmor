"""Least-squares fit of the log signal, shared by the DWI models."""

import numpy as np
from numpy.typing import NDArray

from larmor import _core

# Reweighting stops once no predicted log signal changes by more than this.
LOG_SIGNAL_TOL = 1e-6
# Maximum number of weighted fits after the first, unweighted one.
MAX_REWEIGHTINGS = 20
# Voxels per call to the compiled fit.
CHUNK_VOXELS = 2**15


def fit_log_signal(
    design: NDArray[np.float64],
    s: NDArray[np.float64],
    weighted: bool,
    constraints: NDArray[np.float64] | None = None,
    n_threads: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Fits a linear model to the log signal of each voxel by least squares.

    The weighted fit weights each measurement by the square of the signal the fit
    predicts (Veraart et al., 2013). It starts from the ordinary fit and refits
    until no predicted log signal changes by more than 1e-6, or after 20 weighted
    fits. A voxel whose weighted fit is not finite keeps the ordinary fit.

    With constraints G, a voxel whose solution breaks G @ p >= 0 is fitted again
    subject to it, with the same weights, as a least-squares problem with linear
    inequality constraints (Lawson and Hanson, 1974). Where no parameters meet the
    constraints or the solve fails, the voxel keeps its unconstrained fit.

    Args:
        design: Design matrix of the log signal, shape (N, P), of full column
            rank.
        s: Positive signal, shape (n_voxels, N).
        weighted: Whether the fit is weighted.
        constraints: The constraint matrix G, shape (M, P). Defaults to None, no
            constraints.
        n_threads: Number of threads. Defaults to 1.

    Returns:
        tuple[np.ndarray, np.ndarray]: The parameters, shape (P, n_voxels), and
        whether each voxel's unconstrained fit broke a constraint, shape
        (n_voxels,).

    References:
        Veraart J, Sijbers J, Sunaert S, Leemans A, Jeurissen B. Weighted linear
        least squares estimation of diffusion MRI parameters: strengths,
        limitations, and pitfalls. NeuroImage 81 (2013) 335-346.
        Lawson CL, Hanson RJ. Solving Least Squares Problems. Prentice-Hall,
        1974 (chapter 23).
    """
    design = np.ascontiguousarray(design, dtype=np.float64)
    if constraints is None:
        constraints = np.zeros((0, design.shape[1]))
    constraints = np.ascontiguousarray(constraints, dtype=np.float64)
    params = np.empty((design.shape[1], len(s)))
    constrained = np.zeros(len(s), dtype=bool)
    for start in range(0, len(s), CHUNK_VOXELS):
        part = slice(start, start + CHUNK_VOXELS)
        log_s = np.ascontiguousarray(np.log(s[part]), dtype=np.float64)
        p, broke = _core.fit_log_signal(
            design,
            constraints,
            log_s,
            weighted,
            MAX_REWEIGHTINGS,
            LOG_SIGNAL_TOL,
            n_threads,
        )
        params[:, part] = p.T
        constrained[part] = broke
    return params, constrained
