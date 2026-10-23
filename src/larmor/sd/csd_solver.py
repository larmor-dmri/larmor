"""The solve of constrained spherical deconvolution (CSD) of one shell, run in C++."""

from dataclasses import dataclass, field
from math import isfinite

import numpy as np
import numpy.typing as npt

from larmor import _core
from larmor._log import LarmorLogger
from larmor.sd.sh import build_basis_matrix, get_sh_indices
from larmor.sd.sphere import DIRS_300, get_theta_phi
from larmor.types import GradientTable

# Scale of the weight of the non-negativity constraint,
# neg_lambda * CONSTRAINT_SCALE * c_0 / n_directions, with c_0 the response's l = 0
# coefficient.
CONSTRAINT_SCALE = 50.0
# Scale of the weight of the penalty on ||f||^2, norm_lambda * NORM_SCALE times the
# first diagonal entry of M^T M.
NORM_SCALE = 2e-4


@dataclass(frozen=True, kw_only=True)
class CSDSolver:
    """The settings of constrained spherical deconvolution (CSD), for `CSDModel`.

    Each voxel's FOD starts from a linear deconvolution up to init_lmax. Each
    iteration then penalises the FOD on the constraint directions where it is below
    the threshold, until that set of directions repeats. Both penalties are scaled
    by the response, so their strength does not depend on the signal's units.

    Attributes:
        neg_lambda: Strength of the non-negativity constraint (lambda in the paper).
            Defaults to 1.0.
        norm_lambda: Strength of the penalty on the norm of the FOD. Defaults to 1.0.
        threshold: FOD amplitude below which a direction is constrained (tau in the
            paper), in the FOD's units. Defaults to 0.0.
        max_iter: Maximum number of constrained solves per voxel; 0 gives the linear
            estimate. Defaults to 50.
        init_lmax: Highest degree of the linear estimate the iterations start from.
            Defaults to 4.
        directions: Unit vectors the constraint is applied on, shape (n, 3). Defaults
            to None, Larmor's built-in set of 300 directions.

    References:
        Tournier JD, Calamante F, Connelly A. Robust determination of the fibre
        orientation distribution in diffusion MRI: non-negativity constrained
        super-resolved spherical deconvolution. NeuroImage 35 (2007) 1459-1472.
        doi:10.1016/j.neuroimage.2007.02.016

        Defaults and weight scaling as in MRtrix3's dwi2fod csd: Tournier JD, Smith R,
        Raffelt D, et al. MRtrix3: A fast, flexible and open software framework for
        medical image processing and visualisation. NeuroImage 202 (2019) 116137.
        doi:10.1016/j.neuroimage.2019.116137
    """

    neg_lambda: float = 1.0
    norm_lambda: float = 1.0
    threshold: float = 0.0
    max_iter: int = 50
    init_lmax: int = 4
    directions: np.ndarray | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Checks the settings.

        Raises:
            ValueError: If a lambda is negative or not finite, the threshold is not
                finite, max_iter is negative, init_lmax is odd or negative, or the
                directions are not an (n, 3) array of finite, non-zero vectors.
        """
        for name in ("neg_lambda", "norm_lambda"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite number >= 0, got {value}")
        if not isfinite(self.threshold):
            raise ValueError(f"threshold must be finite, got {self.threshold}")
        if self.max_iter < 0:
            raise ValueError(f"max_iter must be at least 0, got {self.max_iter}")
        if self.init_lmax < 0 or self.init_lmax % 2:
            raise ValueError(f"init_lmax must be even and >= 0, got {self.init_lmax}")
        if self.directions is not None:
            dirs = np.asarray(self.directions)
            if dirs.ndim != 2 or dirs.shape[1] != 3 or len(dirs) == 0:
                raise ValueError(
                    f"directions must be an (n, 3) array, got shape {dirs.shape}"
                )
            lengths = np.linalg.norm(dirs, axis=1)
            if not (np.isfinite(lengths) & (lengths > 0)).all():
                raise ValueError(
                    "directions must be finite, non-zero vectors; rows "
                    f"{np.flatnonzero(~(np.isfinite(lengths) & (lengths > 0)))} are not"
                )


def csd_shell(gtab: GradientTable) -> float:
    """Returns the b-value of the shell CSD fits, the highest.

    Returns:
        float: The shell's b-value in s/mm^2, as `GradientTable.shells` gives it.

    Raises:
        ValueError: If every volume is a b0.
    """
    shells = gtab.shells
    if len(shells) == 0:
        raise ValueError("CSD needs a diffusion-weighted shell; every volume is a b0.")
    return float(shells[-1])


def _rotational_harmonics(zonal: np.ndarray, lmax: int) -> np.ndarray:
    """Returns the convolution factor c_l / sqrt((2l + 1) / 4 pi) of each even degree.

    Convolving an FOD with an axially symmetric response multiplies each coefficient
    of degree l by this factor; degrees the response lacks get 0.

    Returns:
        The factors of l = 0, 2, ..., lmax, shape (lmax / 2 + 1,).
    """
    factors = np.zeros(lmax // 2 + 1)
    n = min(len(factors), len(zonal))
    degrees = np.arange(0, 2 * n, 2)
    factors[:n] = zonal[:n] / np.sqrt((2 * degrees + 1) / (4 * np.pi))
    return factors


def _basis(dirs: np.ndarray, lmax: int) -> np.ndarray:
    """Returns the tournier SH basis up to lmax on unit vectors, shape (n, C)."""
    theta, phi = get_theta_phi(dirs)
    return build_basis_matrix(theta, phi, lmax, "tournier")


def _initial_matrix(
    bvecs: np.ndarray, zonal: np.ndarray, lmax: int, init_lmax: int
) -> np.ndarray:
    """Builds the linear deconvolution the CSD iterations start from, (C0, n_meas).

    The signal is fitted by least squares in SH up to the highest degree that the
    measurements, the response and lmax all allow, each coefficient is divided by the
    response's convolution factor, and the degrees above init_lmax are dropped.

    Returns:
        The matrix that maps a signal to the starting FOD's first C0 coefficients.
    """
    n_meas = len(bvecs)
    data_lmax = 0
    while (data_lmax + 3) * (data_lmax + 4) // 2 <= n_meas:
        data_lmax += 2
    response_lmax = 2 * (len(zonal) - 1)
    fit_lmax = min(data_lmax, response_lmax, lmax)
    keep_lmax = min(fit_lmax, init_lmax)
    degrees = get_sh_indices(fit_lmax)[0]
    factors = _rotational_harmonics(zonal, fit_lmax)[degrees // 2]
    keep = degrees <= keep_lmax
    initial: np.ndarray = (np.linalg.pinv(_basis(bvecs, fit_lmax)) / factors[:, None])[
        keep
    ]
    return initial


def fit_csd(
    signals: npt.NDArray[np.float64],
    bvecs: np.ndarray,
    zonal: np.ndarray,
    lmax: int,
    solver: CSDSolver,
    n_threads: int,
) -> npt.NDArray[np.float64]:
    """Fits the FOD of many voxels of one shell by CSD, in C++.

    Args:
        signals: Signal of each voxel on the shell, one per row, shape
            (n_voxels, n_meas), in the units of the response; negative values are
            fitted as 0.
        bvecs: Unit gradient direction of each measurement, shape (n_meas, 3).
        zonal: The response's zonal SH coefficients on this shell, l = 0, 2, ...
        lmax: Highest degree of the FOD.
        solver: The CSD settings.
        n_threads: Number of threads.

    Returns:
        The FODs in the tournier basis, shape (n_voxels, C).

    Raises:
        ValueError: If the shell has fewer than 6 measurements, the response's l = 0
            coefficient is not positive or it has no anisotropic degree, or
            norm_lambda is 0 while the data or the response do not determine every
            coefficient (more coefficients than measurements, or lmax above the
            response's degrees), which leaves the solve singular.
    """
    if len(bvecs) < 6:
        raise ValueError(
            f"CSD needs at least 6 measurements on the shell, got {len(bvecs)}."
        )
    if not zonal[0] > 0 or len(zonal) < 2:
        raise ValueError(
            "The response needs a positive l = 0 coefficient and at least one "
            "anisotropic degree (l = 2)."
        )
    if lmax > 2 * (len(zonal) - 1):
        LarmorLogger.warning(
            f"The response has degrees up to {2 * (len(zonal) - 1)} only; the FOD's "
            f"degrees up to {lmax} above it are set by the constraint alone."
        )
    degrees = get_sh_indices(lmax)[0]
    if solver.norm_lambda == 0 and (
        len(degrees) > len(bvecs) or lmax > 2 * (len(zonal) - 1)
    ):
        raise ValueError(
            f"norm_lambda 0 leaves the solve singular: lmax {lmax} has "
            f"{len(degrees)} coefficients for {len(bvecs)} measurements and a "
            f"response up to degree {2 * (len(zonal) - 1)}; give norm_lambda > 0."
        )
    forward = _basis(bvecs, lmax) * _rotational_harmonics(zonal, lmax)[degrees // 2]
    directions = DIRS_300 if solver.directions is None else solver.directions
    directions = np.asarray(directions, dtype=np.float64)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    weight = solver.neg_lambda * CONSTRAINT_SCALE * zonal[0] / len(directions)
    constraint = _basis(directions, lmax) * weight
    norm_weight = solver.norm_lambda * NORM_SCALE * float(forward[:, 0] @ forward[:, 0])
    initial = _initial_matrix(bvecs, zonal, lmax, solver.init_lmax)
    fods, converged = _core.fit_csd(
        np.ascontiguousarray(np.maximum(signals, 0.0), dtype=np.float64),
        np.ascontiguousarray(initial),
        np.ascontiguousarray(forward),
        np.ascontiguousarray(constraint),
        norm_weight=norm_weight,
        threshold=solver.threshold * weight,
        max_iter=solver.max_iter,
        n_threads=n_threads,
    )
    if not converged.all():
        LarmorLogger.warning(
            f"{int((~converged).sum())} voxels did not converge within "
            f"{solver.max_iter} iterations; their FODs are those of the last one."
        )
    result: npt.NDArray[np.float64] = fods
    return result
