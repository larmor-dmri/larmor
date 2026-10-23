"""The constrained least-squares problem of multi-tissue CSD and its compiled fit."""

import numpy as np
import numpy.typing as npt

from larmor import _core
from larmor.sd.rf import ZonalSHRF
from larmor.sd.sh import build_basis_matrix, get_sh_indices
from larmor.sd.sphere import get_theta_phi

# Ridge added to the normal matrix A^T A, relative to its largest diagonal entry.
NORM_REG = 1e-10


def _sh_size(lmax: int) -> int:
    """Returns the number of even-degree SH coefficients up to lmax."""
    return (lmax + 1) * (lmax + 2) // 2


def forward_matrix(
    rfs: list[ZonalSHRF], lmaxes: list[int], bvals: np.ndarray, bvecs: np.ndarray
) -> npt.NDArray[np.float64]:
    """Builds the matrix that maps the SH coefficients of every tissue to the signal.

    The signal is the spherical convolution of each tissue's distribution with its
    response. On a shell where the response has zonal coefficients r_l,
    coefficients f_lm give sum_lm sqrt(4 pi / (2l + 1)) r_l f_lm Y_lm(g) along
    gradient g, so a distribution concentrated along z with unit integral gives
    back the response.

    Args:
        rfs: One response per tissue.
        lmaxes: The SH degree of each tissue's distribution.
        bvals: b-value of each measurement in s/mm^2, shape (n_meas,).
        bvecs: Unit gradient direction of each measurement, or zero for none, shape
            (n_meas, 3).

    Returns:
        The matrix, shape (n_meas, sum of each tissue's number of coefficients), the
        tissues' columns side by side in the order of rfs.
    """
    has_direction = np.linalg.norm(bvecs, axis=1) > 0
    theta, phi = get_theta_phi(np.where(has_direction[:, np.newaxis], bvecs, [0, 0, 1]))
    blocks = []
    for rf, tissue_lmax in zip(rfs, lmaxes, strict=True):
        degrees, _ = get_sh_indices(tissue_lmax)
        basis = build_basis_matrix(theta, phi, tissue_lmax)
        # Without a direction (a b0) only the isotropic part of the signal is defined.
        basis[np.ix_(~has_direction, degrees > 0)] = 0.0
        response = np.array([rf.shell_coefficients(b) for b in bvals])
        padded = np.zeros((len(bvals), tissue_lmax // 2 + 1))
        n_columns = min(padded.shape[1], response.shape[1])
        padded[:, :n_columns] = response[:, :n_columns]
        scale = np.sqrt(4 * np.pi / (2 * degrees + 1))
        blocks.append(basis * scale * padded[:, degrees // 2])
    return np.hstack(blocks)


def constraint_matrix(lmaxes: list[int], dirs: np.ndarray) -> npt.NDArray[np.float64]:
    """Builds the constraints G x >= 0 that keep every tissue non-negative.

    Each tissue with lmax above 0 gets one row per direction, its amplitude there;
    an isotropic tissue gets one row, its single coefficient.

    Returns:
        The matrix, shape (n_constraints, sum of each tissue's number of
        coefficients).
    """
    theta, phi = get_theta_phi(dirs)
    n_columns = sum(_sh_size(tissue_lmax) for tissue_lmax in lmaxes)
    rows = []
    start = 0
    for tissue_lmax in lmaxes:
        size = _sh_size(tissue_lmax)
        if tissue_lmax == 0:
            block = np.zeros((1, n_columns))
            block[0, start] = 1.0
        else:
            block = np.zeros((len(dirs), n_columns))
            block[:, start : start + size] = build_basis_matrix(theta, phi, tissue_lmax)
        rows.append(block)
        start += size
    return np.vstack(rows)


def prepare_msmt(
    rfs: list[ZonalSHRF],
    lmaxes: list[int],
    bvals: np.ndarray,
    bvecs: np.ndarray,
    dirs: np.ndarray,
) -> _core.LsiDesign:
    """Builds the compiled constrained least-squares problem of an MSMT-CSD fit.

    Returns:
        The problem, for fit_msmt.

    Raises:
        ValueError: If the measurements cannot determine every coefficient.
    """
    forward = forward_matrix(rfs, lmaxes, bvals, bvecs)
    # The ridge would make any design solvable; a rank-deficient one means the data
    # cannot tell some coefficients apart (too few b-values for the tissues, or too
    # few directions for lmax).
    rank = int(np.linalg.matrix_rank(forward))
    if rank < forward.shape[1]:
        raise ValueError(
            f"The measurements determine only {rank} of the {forward.shape[1]} SH "
            f"coefficients of {len(rfs)} tissues at lmax {max(lmaxes)}: they need more "
            "distinct b-values or directions, or fewer tissues or a lower lmax."
        )
    constraints = constraint_matrix(lmaxes, dirs)
    ridge = float(np.sqrt(NORM_REG * np.max(np.sum(forward**2, axis=0))))
    return _core.LsiDesign(
        np.ascontiguousarray(forward), np.ascontiguousarray(constraints), ridge
    )


def fit_msmt(
    signals: npt.NDArray[np.float64], design: _core.LsiDesign, n_threads: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Fits the SH coefficients of every tissue in each voxel, in C++.

    Args:
        signals: Signals in the units of the responses, one voxel per row, shape
            (n_voxels, n_meas).
        design: The problem from prepare_msmt.
        n_threads: Number of threads.

    Returns:
        The coefficients, tissues side by side in tournier order, shape
        (n_voxels, n_coefficients), and whether each voxel's solver converged,
        (n_voxels,).
    """
    return _core.fit_lsi(
        np.ascontiguousarray(signals, dtype=np.float64), design, n_threads=n_threads
    )
