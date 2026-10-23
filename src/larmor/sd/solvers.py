"""FOD solvers, and the single- and multi-tissue fits that run them in C++."""

from dataclasses import dataclass
from enum import Enum, auto
from math import isfinite

import numpy as np
import numpy.typing as npt

from larmor import _core
from larmor.sd.kernel import Kernel

# Maximum number of rounds of the multi-tissue loop.
MAX_ROUNDS = 50
# Default change in every fraction below which a voxel leaves the multi-tissue loop.
FRACTION_TOL = 1e-3
# In the final fraction fit, a tissue worth less than about this share of the signal
# gets a fraction of 0.
FINAL_FRACTION_TOL_X = 1e-2
# Every direction of a multi-tissue dRL solve starts from this value.
MULTI_TISSUE_DRL_INIT = 1e-6


class Stage(Enum):
    """The solve of a fit a solver is prepared for."""

    SINGLE = auto()  # the only solve of a single-tissue fit
    LOW_RES = auto()  # each solve of the multi-tissue loop
    HIGH_RES = auto()  # the final solve of a multi-tissue fit


def _rl_products(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns K^T and K^T K, the matrices of a Richardson-Lucy solve."""
    return np.ascontiguousarray(k.T), np.ascontiguousarray(k.T @ k)


def calibrate_drl_nu(
    kernel: Kernel,
    bvals: np.ndarray,
    target_adc: float = 0.7e-3,
    scale: float = 0.2,
    n_iter: int = 200,
) -> float:
    """Computes dRL's damping threshold nu for a kernel.

    nu is scale times the largest FOD amplitude that an undamped deconvolution with
    the whole kernel gives for a purely isotropic tissue, exp(-b_max target_adc) on
    every measurement, so that dRL damps amplitudes at the level of that background.

    Args:
        kernel: Kernel the fit will use.
        bvals: b-value of each measurement in s/mm^2, shape (n_meas,).
        target_adc: Diffusivity in mm^2/s of the isotropic tissue. Defaults to
            0.7e-3.
        scale: Fraction of the largest isotropic amplitude that becomes nu.
            Defaults to 0.2.
        n_iter: Iterations of the undamped fit. Defaults to 200.

    Returns:
        The damping threshold nu.
    """
    signal = np.full((1, len(bvals)), np.exp(-max(bvals) * target_adc))
    k_t, k_t_k = _rl_products(kernel.k)
    solver = _core.FodSolver.drl(
        k_t, k_t_k, nu=0.0, n_iter=n_iter, init=1.0 / kernel.k.shape[1]
    )
    fake_fod = _core.fit_single_tissue(signal, solver, n_threads=1)
    return float(scale * np.max(fake_fod))


@dataclass(frozen=True, kw_only=True)
class DRL:
    """Damped Richardson-Lucy (dRL) deconvolution, an FOD solver for `SDModel`.

    It is the default solver of the GRL method. The FOD stays non-negative. The
    damping slows the update of directions whose amplitude is below a threshold nu,
    so that the isotropic background does not grow into spurious peaks; nu is
    nu_scale times the largest FOD amplitude that an undamped deconvolution gives for
    a purely isotropic tissue of diffusivity target_adc. A voxel stops early when its
    FOD no longer changes.

    Attributes:
        num_iter: Maximum number of iterations of a single-tissue fit, and of each
            FOD solve in the loop that alternates FOD and tissue-fraction fits in a
            multi-tissue fit. Defaults to 200.
        hr_num_iter: Maximum number of iterations of the final FOD solve of a
            multi-tissue fit. Defaults to 600.
        target_adc: Diffusivity in mm^2/s of the isotropic tissue nu is set from.
            Defaults to 0.7e-3.
        nu_scale: Fraction of that tissue's largest FOD amplitude that becomes nu.
            Defaults to 0.2.

    References:
        Dell'Acqua F, Scifo P, Rizzo G, et al. A modified damped Richardson-Lucy
        algorithm to reduce isotropic background effects in spherical deconvolution.
        NeuroImage 49 (2010) 1446-1458. doi:10.1016/j.neuroimage.2009.09.033

        Follows MRIToolkit's implementation (MRTTrack).
    """

    num_iter: int = 200
    hr_num_iter: int = 600
    target_adc: float = 0.7e-3
    nu_scale: float = 0.2

    def __post_init__(self) -> None:
        """Checks the settings.

        Raises:
            ValueError: If an iteration count is below 1, or target_adc or nu_scale
                is not a positive number.
        """
        if self.num_iter < 1 or self.hr_num_iter < 1:
            raise ValueError(
                f"num_iter and hr_num_iter must be at least 1, got {self.num_iter} "
                f"and {self.hr_num_iter}"
            )
        for name in ("target_adc", "nu_scale"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a positive number, got {value}")

    def _prepare(
        self, kernel: Kernel, bvals: np.ndarray, stage: Stage
    ) -> _core.FodSolver:
        """Builds the compiled solver for one stage of a fit with this kernel.

        Args:
            kernel: Kernel of that stage; the solver fits its weighted anisotropic
                columns.
            bvals: b-value of each measurement in s/mm^2, shape (n_meas,).
            stage: Which solve of the fit this is.

        Returns:
            The solver, with its matrices and calibrated nu.
        """
        nu = calibrate_drl_nu(kernel, bvals, self.target_adc, self.nu_scale)
        k_t, k_t_k = _rl_products(kernel.w_k_aniso)
        if stage is Stage.SINGLE:
            init = 1.0 / kernel.w_k_aniso.shape[1]
            return _core.FodSolver.drl(
                k_t, k_t_k, nu=nu, n_iter=self.num_iter, init=init
            )
        n_iter = self.num_iter if stage is Stage.LOW_RES else self.hr_num_iter
        # A multi-tissue solve counts its near-zero start as the first iteration.
        return _core.FodSolver.drl(
            k_t, k_t_k, nu=nu, n_iter=n_iter - 1, init=MULTI_TISSUE_DRL_INIT
        )


@dataclass(frozen=True, kw_only=True)
class L2NNLS:
    """Non-negative least squares (NNLS) with a ridge (L2) penalty, an FOD solver.

    It is the default solver of the mFOD method. With K the kernel (the signals of
    the response functions along the FOD directions) and s a voxel's signal, the FOD
    f minimises ||K f - s||^2 + reg^2 ||f||^2 subject to f >= 0. In a multi-tissue
    fit, the FOD solves of the loop that alternates FOD and tissue-fraction fits use
    plain NNLS (reg 0); the final solve, and the only solve of a single-tissue fit,
    use the penalty.

    Attributes:
        reg: Weight of the penalty; 0 gives plain NNLS. Defaults to 0.1.
        tol_x: A coefficient enters the solution only when its gradient exceeds this.
            Defaults to 1e-6.

    References:
        De Luca A, Guo F, Froeling M, Leemans A. Spherical deconvolution with
        tissue-specific response functions and multi-shell diffusion MRI to estimate
        multiple fiber orientation distributions (mFODs). NeuroImage 222 (2020)
        117206. doi:10.1016/j.neuroimage.2020.117206

        Lawson CL, Hanson RJ. Solving Least Squares Problems. Prentice-Hall, 1974.

        Follows MRIToolkit's L2LSQ method (MRTTrack).
    """

    reg: float = 0.1
    tol_x: float = 1e-6

    def __post_init__(self) -> None:
        """Checks the settings.

        Raises:
            ValueError: If reg is not a finite number >= 0, or tol_x is not positive.
        """
        if not isfinite(self.reg) or self.reg < 0:
            raise ValueError(f"reg must be a finite number >= 0, got {self.reg}")
        if not self.tol_x > 0:
            raise ValueError(f"tol_x must be positive, got {self.tol_x}")

    def _prepare(
        self, kernel: Kernel, bvals: np.ndarray, stage: Stage
    ) -> _core.FodSolver:
        """Builds the compiled solver for one stage of a fit with this kernel.

        Args:
            kernel: Kernel of that stage; the solver fits its weighted anisotropic
                columns.
            bvals: Not used.
            stage: Which solve of the fit this is; the multi-tissue loop gets
                plain NNLS.

        Returns:
            The solver, ready to fit the kernel.
        """
        reg = 0.0 if stage is Stage.LOW_RES else self.reg
        return _core.FodSolver.l2nnls(
            np.ascontiguousarray(kernel.w_k_aniso), reg, self.tol_x
        )


@dataclass(frozen=True, kw_only=True)
class Lasso:
    """Elastic-net (LASSO-type) regression of the signal, an FOD solver.

    The signal s is regressed, without an intercept, on the columns of the kernel,
    the signals of the response functions along the FOD directions, each scaled to
    unit root mean square (K0). With m measurements, the coefficients b minimise
    (1 / 2m) ||s - K0 b||^2 + reg (alpha ||b||_1 + (1 - alpha) / 2 ||b||^2), and the
    FOD is b divided by the column scales. In a multi-tissue fit, s is the signal
    left after the isotropic tissues are subtracted. Every FOD solve of a fit uses
    the same penalty.

    The FOD is not constrained to be non-negative. In a multi-tissue fit, FOD values
    below their median are set to zero before the tissue fractions are fitted, but
    the FOD itself keeps its negative values. When reg is large enough that no
    direction explains enough of the signal, the FOD is zero.

    Attributes:
        reg: Overall weight of the penalty, lambda. Defaults to 0.01.
        alpha: Share of the penalty on the L1 norm, in (0, 1]; 1 is the plain LASSO.
            Defaults to 0.9.
        tol: Coordinate descent stops when no coefficient changes by more than this,
            relative to 1 + its size, over a sweep. Defaults to 1e-4.
        max_iter: Maximum number of sweeps over the columns. A solve that reaches
            it without converging keeps its last sweep, and the fit warns how many
            solves did. Defaults to 100_000.

    References:
        Tibshirani R. Regression shrinkage and selection via the lasso. J R Stat Soc B
        58 (1996) 267-288.

        Zou H, Hastie T. Regularization and variable selection via the elastic net.
        J R Stat Soc B 67 (2005) 301-320. doi:10.1111/j.1467-9868.2005.00503.x

        The method follows MRIToolkit's LASSO method (MRTTrack, version 1.8), whose
        default Alpha is 0.9.
    """

    reg: float = 0.01
    alpha: float = 0.9
    tol: float = 1e-4
    max_iter: int = 100_000

    def __post_init__(self) -> None:
        """Checks the settings.

        Raises:
            ValueError: If reg is not a finite number >= 0, alpha is not in (0, 1], tol
                is not positive or max_iter is below 1.
        """
        if not isfinite(self.reg) or self.reg < 0:
            raise ValueError(f"reg must be a finite number >= 0, got {self.reg}")
        if not 0 < self.alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if not self.tol > 0:
            raise ValueError(f"tol must be positive, got {self.tol}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {self.max_iter}")

    def _prepare(
        self, kernel: Kernel, bvals: np.ndarray, stage: Stage
    ) -> _core.FodSolver:
        """Builds the compiled solver for one stage of a fit with this kernel.

        Args:
            kernel: Kernel of that stage; the solver fits its weighted anisotropic
                columns.
            bvals: Not used.
            stage: Not used.

        Returns:
            The solver, ready to fit the kernel.
        """
        return _core.FodSolver.lasso(
            np.ascontiguousarray(kernel.w_k_aniso),
            self.reg,
            self.alpha,
            self.tol,
            self.max_iter,
        )


#: The FOD solvers `SDModel` accepts, `DRL`, `L2NNLS` and `Lasso`.
type SDSolver = DRL | L2NNLS | Lasso


def fit_single_tissue(
    signals: npt.NDArray[np.float64], solver: _core.FodSolver, n_threads: int
) -> npt.NDArray[np.float64]:
    """Fits the FOD of many voxels with one solver, in C++.

    Args:
        signals: Signals of the voxels normalised by b0, one per row, shape
            (n_voxels, n_meas).
        solver: Solver prepared for Stage.SINGLE.
        n_threads: Number of threads.

    Returns:
        The FOD amplitudes, one row per voxel and one column per direction, shape
        (n_voxels, n_dirs).
    """
    return _core.fit_single_tissue(
        np.ascontiguousarray(signals, dtype=np.float64), solver, n_threads=n_threads
    )


def _deconv_kernel(kernel: Kernel) -> _core.DeconvKernel:
    """Returns the matrices of one resolution of a multi-tissue kernel, for C++."""
    return _core.DeconvKernel(
        np.ascontiguousarray(kernel.k_aniso),
        np.ascontiguousarray(kernel.k_iso),
        np.ascontiguousarray(kernel.w_k_iso),
        [(block.start, block.stop) for block in kernel.aniso_blocks],
    )


def fit_multi_tissue(
    signals: npt.NDArray[np.float64],
    lr_kernel: Kernel,
    hr_kernel: Kernel,
    lr_solver: _core.FodSolver,
    hr_solver: _core.FodSolver,
    n_threads: int = 1,
    fraction_tol: float = FRACTION_TOL,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fits the FODs and tissue fractions of many voxels, in C++.

    Each voxel alternates a low-resolution FOD fit of the signal left after removing
    the isotropic tissues with a non-negative least-squares fit of the fractions,
    until the fractions settle. With the isotropic fractions found there, it then
    fits the high-resolution FOD and the final fractions, one per response function.

    Args:
        signals: Signals normalised by b0, unweighted (the shell weights come from
            the kernels), one voxel per row, shape (n_voxels, n_meas).
        lr_kernel: Low-resolution kernel, built with the same response functions
            and weights as hr_kernel.
        hr_kernel: High-resolution kernel.
        lr_solver: Solver prepared from lr_kernel for Stage.LOW_RES.
        hr_solver: Solver prepared from hr_kernel for Stage.HIGH_RES.
        n_threads: Number of threads. Defaults to 1.
        fraction_tol: A voxel leaves the loop when no fraction changes by more than
            this from one round to the next, or after MAX_ROUNDS rounds. Defaults
            to FRACTION_TOL.

    Returns:
        The high-resolution FOD amplitudes, shape (n_voxels, n_aniso_cols), and the
        fractions in kernel order, summing to 1, shape (n_voxels, n_rfs).
    """
    return _core.fit_multi_tissue(
        np.ascontiguousarray(signals, dtype=np.float64),
        np.ascontiguousarray(lr_kernel.weights, dtype=np.float64),
        _deconv_kernel(lr_kernel),
        lr_solver,
        _deconv_kernel(hr_kernel),
        hr_solver,
        max_outer=MAX_ROUNDS,
        fraction_tol=fraction_tol,
        final_tol_x=FINAL_FRACTION_TOL_X,
        n_threads=n_threads,
    )
