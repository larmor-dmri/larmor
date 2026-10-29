"""Spherical deconvolution of a DWI volume into FODs, single- or multi-tissue."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from larmor._log import LarmorLogger
from larmor._threads import n_threads
from larmor.dwi import estimate_s0, shell_mask
from larmor.dwi.inputs import as_mask, check_dwi
from larmor.dwi.mask import unmask
from larmor.sd.kernel import Kernel, order_rfs
from larmor.sd.method import SDMethod
from larmor.sd.peaks import PEAK_RELATIVE_THRESHOLD, SD_MODEL_PEAK_RELATIVE_THRESHOLD
from larmor.sd.progress import report_voxels
from larmor.sd.rf import IsotropicRF, ResponseFunction
from larmor.sd.sh import BASES, fit_sh
from larmor.sd.solvers import (
    L2NNLS,
    Lasso,
    SDSolver,
    Stage,
    fit_multi_tissue,
    fit_single_tissue,
)
from larmor.sd.sphere import DIRS_300, get_dirs
from larmor.types import GradientTable, Volume

# Voxels per call into the compiled code.
FIT_CHUNK = 16_384


MERGING_METHOD = ("ws", "majority")


# A function that fits a chunk of voxels and returns their FOD amplitudes, the
# fractions of a multi-tissue fit, and the number of solves that did not converge.
type _ChunkFit = Callable[
    [npt.NDArray[np.float64]],
    tuple[npt.NDArray[np.float64], np.ndarray | None, int],
]


@dataclass(frozen=True)
class SDFit:
    """Result of a spherical deconvolution fit.

    (X, Y, Z) is the spatial shape of the DWI and N the number of response
    functions. The FODs have even degrees only, so they have
    C = (lmax + 1)(lmax + 2) / 2 spherical harmonic (SH) coefficients, 28 for
    lmax 6.

    Attributes:
        fods: One FOD per anisotropic response function, in the order given to
            the model, each as SH coefficients in `basis`, shape (X, Y, Z, C),
            zero outside the mask. With the L2NNLS and Lasso solvers, c00 is set
            to the sum of the FOD's values on the fit's directions divided by
            sqrt(4 pi), so that its integral over the sphere is exact.
        tissue_fractions: Fraction of each response function per voxel, shape
            (X, Y, Z, N), summing to 1 inside the mask. The anisotropic response
            functions come first, then the isotropic ones, each group in the
            order given to `SDModel`. None for a single-tissue fit (one
            anisotropic response function and no isotropic one).
        basis: SH basis of the FODs, "tournier" (MRtrix, readable by mrview),
            "leemans" (ExploreDTI and MRIToolkit) or "descoteaux".
        lmax: Maximum SH degree actually used. It can be lower than the lmax
            given to the model when the highest shell has too few directions.
        peak_relative_threshold: The relative threshold to give `find_peaks` for
            these FODs; 0.2 for the FODs of `SDModel`, whose truncation to lmax
            leaves side lobes where fibres cross, and 0.1 for CSD and MSMT-CSD.
    """

    fods: list[np.ndarray]
    tissue_fractions: np.ndarray | None
    basis: str
    lmax: int
    peak_relative_threshold: float = PEAK_RELATIVE_THRESHOLD


class SDModel:
    """Spherical deconvolution of DWI data into FODs with the GRL or mFOD method.

    With a single anisotropic response function and no isotropic one, the fit is
    single-tissue, one FOD solve per voxel. Otherwise it is multi-tissue, and each
    voxel alternates between fitting the FODs and the tissue fractions until the
    fractions settle. b-values are in s/mm^2 and the response functions'
    diffusivities in mm^2/s. FODs come out in the frame of the DWI's b-vectors.

    References:
        GRL: Guo F, Leemans A, Viergever MA, Dell'Acqua F, De Luca A. Generalized
        Richardson-Lucy (GRL) for analyzing multi-shell diffusion MRI data.
        NeuroImage 218 (2020) 116948. doi:10.1016/j.neuroimage.2020.116948

        mFOD: De Luca A, Guo F, Froeling M, Leemans A. Spherical deconvolution with
        tissue-specific response functions and multi-shell diffusion MRI to estimate
        multiple fiber orientation distributions (mFODs). NeuroImage 222 (2020)
        117206. doi:10.1016/j.neuroimage.2020.117206

        The solvers (dRL, L2NNLS, LASSO) cite their own papers: see `larmor.sd.DRL`,
        `larmor.sd.L2NNLS` and `larmor.sd.Lasso`.
    """

    def __init__(
        self,
        rfs: list[ResponseFunction],
        *,
        method: SDMethod | str = SDMethod.GRL,
        solver: SDSolver | None = None,
        inner_shell_weight: float | None = None,
        lmax: int = 8,
        basis: str = "tournier",
        progress: bool = False,
    ) -> None:
        """Sets up the deconvolution.

        Args:
            rfs: Response functions; reordered with the anisotropic ones first.
            method: Deconvolution method, a member of SDMethod or its value in any
                case. Defaults to GRL.
            solver: How each FOD is solved for. Defaults to None, the method's own
                solver.
            inner_shell_weight: Weight of the measurements outside the highest
                shell in a multi-tissue fit; the highest shell has weight 1.
                Defaults to None, the method's own weight.
            lmax: Maximum SH degree. Defaults to 8.
            basis: SH basis of the FODs `fit` returns, "tournier" (MRtrix),
                "leemans" (ExploreDTI) or "descoteaux". Defaults to "tournier".
            progress: Whether to show a progress bar over the voxels. Defaults
                to False.

        Raises:
            ValueError: If the method is unknown or does not take these response
                functions, lmax is odd or below 2, inner_shell_weight is not in
                (0, 1], or the basis is unknown.
        """
        if basis not in BASES:
            raise ValueError(
                f"Invalid basis {basis}, expected one of {', '.join(BASES)}"
            )
        if lmax < 2 or lmax % 2 != 0:
            raise ValueError(f"lmax must be even and at least 2, got {lmax}")
        if inner_shell_weight is not None and not 0 < inner_shell_weight <= 1:
            raise ValueError(
                f"inner_shell_weight must be in (0, 1], got {inner_shell_weight}"
            )
        self.rfs = order_rfs(rfs)
        self.method = SDMethod(method)
        self.method._check_rfs(self.rfs)
        self.solver = solver
        self.inner_shell_weight = (
            self.method.inner_shell_weight
            if inner_shell_weight is None
            else inner_shell_weight
        )
        self.lmax = lmax
        self.basis = basis
        self.num_iso = sum(1 for rf in self.rfs if isinstance(rf, IsotropicRF))
        n_aniso = len(self.rfs) - self.num_iso
        self.is_multi_tissue = self.num_iso > 0 or n_aniso > 1
        if inner_shell_weight is not None and not self.is_multi_tissue:
            LarmorLogger.warning(
                "inner_shell_weight only applies to a multi-tissue fit; this "
                "single-tissue fit does not weight the shells."
            )
        self.progress = progress

    def fit(
        self,
        volume: Volume,
        gtab: GradientTable,
        mask: Volume | np.ndarray | None = None,
        *,
        n_jobs: int = -1,
    ) -> SDFit:
        """Fits the FODs of every voxel of the mask.

        Each voxel's signal is divided by its S0 from `estimate_s0` and capped at
        1. Voxels whose signal is not finite, or whose S0 is NaN, are not fitted;
        their FODs and fractions are NaN.

        Args:
            volume: Diffusion-weighted volume to fit, shape (X, Y, Z, N).
            gtab: Its gradient table; its `b0_mask` selects the b0s the signal is
                normalised by.
            mask: Voxels to fit (nonzero inside), on the volume's grid. Defaults to
                None, which fits every voxel.
            n_jobs: Number of threads for the fit, at most the number of cores; -1
                uses every core. Defaults to -1.

        Returns:
            SDFit: The FODs as SH coefficients in the model's `basis`, in the frame
            of the gradient table (world coordinates), with the tissue fractions of a
            multi-tissue fit and the lmax actually used.

        Raises:
            ValueError: If n_jobs is 0 or below -1.
        """
        solver = self.solver
        if solver is None:
            solver = self.method.default_solver()
        threads = n_threads(n_jobs)
        check_dwi(volume, gtab)
        bvals, bvecs = gtab.bvals, gtab.bvecs

        lmax = self._resolve_lmax(bvals)
        if mask is None:
            mask = np.ones(volume.data.shape[:-1], dtype=bool)
        else:
            mask = as_mask(volume, mask)

        # Scaled in float32 as the image is read, then float64 for the solvers.
        s0 = estimate_s0(volume, gtab, mask)[mask]
        signals = volume.as_float()[mask].astype(np.float64) / s0[:, np.newaxis]

        # The solvers would turn a NaN signal into a small, plausible FOD, so such
        # voxels are left out and given NaN. Checked before the cap, which would
        # turn +inf into 1.
        finite = np.isfinite(signals).all(axis=1)
        # Diffusion weighting only lowers the signal, so a value above S0 is noise.
        np.minimum(signals, 1.0, out=signals)
        if not finite.all():
            LarmorLogger.warning(
                f"{int((~finite).sum())} voxels have a NaN or infinite signal or "
                "no S0; their FODs and fractions are NaN."
            )
        hr_kernel, fit_chunk = self._prepare(solver, bvals, bvecs, threads)
        LarmorLogger.info(
            f"Processing {int(finite.sum())} voxels across {threads} cores "
            f"using {self.method} with {solver}..."
        )
        fitted_fods, fitted_fractions = self._run_chunks(
            signals[finite], hr_kernel.k_aniso.shape[1], fit_chunk
        )

        amplitudes = np.full((len(signals), fitted_fods.shape[1]), np.nan)
        amplitudes[finite] = fitted_fods
        exact_integral = isinstance(solver, (L2NNLS, Lasso))
        fods = [
            unmask(
                _to_sh(
                    amplitudes[:, block],
                    lmax,
                    self.basis,
                    exact_integral=exact_integral,
                ),
                mask,
            )
            for block in hr_kernel.aniso_blocks
        ]
        fractions = None
        if fitted_fractions is not None:
            fractions = np.full((len(signals), fitted_fractions.shape[1]), np.nan)
            fractions[finite] = fitted_fractions
            fractions = unmask(fractions, mask)
        return SDFit(
            fods=fods,
            tissue_fractions=fractions,
            basis=self.basis,
            lmax=lmax,
            peak_relative_threshold=SD_MODEL_PEAK_RELATIVE_THRESHOLD,
        )

    def _resolve_lmax(self, bvals: np.ndarray) -> int:
        """Returns the model's lmax, lowered to what the highest shell supports."""
        lmax = compute_max_lmax(bvals, self.lmax)
        if lmax < self.lmax:
            LarmorLogger.warning(
                f"lmax {self.lmax} needs more directions than the highest shell "
                f"has, lowering to {lmax}"
            )
        return lmax

    def _prepare(
        self,
        solver: SDSolver,
        bvals: np.ndarray,
        bvecs: np.ndarray,
        n_threads: int,
    ) -> tuple[Kernel, _ChunkFit]:
        """Builds the kernels and solvers of the fit.

        Returns:
            The high-resolution kernel, whose anisotropic blocks split the fitted FOD
            columns per response function, and the function that fits a chunk.
        """
        if not self.is_multi_tissue:
            kernel = Kernel(self.rfs, bvals, bvecs, dirs=DIRS_300)
            single = solver._prepare(kernel, bvals, Stage.SINGLE)

            def fit_single(
                signals: npt.NDArray[np.float64],
            ) -> tuple[npt.NDArray[np.float64], None, int]:
                before = single.n_unconverged
                fods = fit_single_tissue(signals, single, n_threads)
                return fods, None, single.n_unconverged - before

            return kernel, fit_single

        weights = compute_shell_weights(bvals, self.inner_shell_weight)
        hr = Kernel(self.rfs, bvals, bvecs, dirs=DIRS_300, weights=weights)
        lr_dirs = low_resolution_dirs(self.rfs, len(bvals))
        lr = Kernel(self.rfs, bvals, bvecs, dirs=lr_dirs, weights=weights)
        lr_solver = solver._prepare(lr, bvals, Stage.LOW_RES)
        hr_solver = solver._prepare(hr, bvals, Stage.HIGH_RES)

        def fit_multi(
            signals: npt.NDArray[np.float64],
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
            before = lr_solver.n_unconverged + hr_solver.n_unconverged
            fods, fractions = fit_multi_tissue(
                signals,
                lr,
                hr,
                lr_solver,
                hr_solver,
                n_threads,
                fraction_tol=self.method._fraction_tolerance,
            )
            after = lr_solver.n_unconverged + hr_solver.n_unconverged
            return fods, fractions, after - before

        return hr, fit_multi

    def _run_chunks(
        self, signals: np.ndarray, n_cols: int, fit_chunk: _ChunkFit
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Fits the voxels in chunks of FIT_CHUNK, with a progress bar.

        Warns when some solves stopped at the solver's iteration limit.

        Returns:
            The FOD amplitudes, and the fractions of a multi-tissue fit or None.
        """
        fods = np.empty((len(signals), n_cols))
        fractions = (
            np.empty((len(signals), len(self.rfs))) if self.is_multi_tissue else None
        )
        unconverged = 0
        starts = range(0, len(signals), FIT_CHUNK)
        report_voxels(0, len(signals))
        for start in tqdm(starts, desc="Fitting voxels", disable=not self.progress):
            stop = min(start + FIT_CHUNK, len(signals))
            fods[start:stop], chunk_fractions, chunk_unconverged = fit_chunk(
                signals[start:stop]
            )
            unconverged += chunk_unconverged
            if fractions is not None and chunk_fractions is not None:
                fractions[start:stop] = chunk_fractions
            report_voxels(stop, len(signals))
        # Only the elastic net counts solves that stop at their limit.
        if unconverged:
            LarmorLogger.warning(
                f"{unconverged} elastic net solves did not converge within max_iter "
                "sweeps; their FODs are those of the last sweep."
            )
        return fods, fractions


def low_resolution_dirs(rfs: list[ResponseFunction], n_meas: int) -> list[np.ndarray]:
    """Returns the directions of each anisotropic response in the low-resolution kernel.

    Each is the spiral of `get_dirs` with the response's `low_resolution_dirs`
    points, or n_meas when the data have fewer measurements, without its last point,
    which repeats the first.

    Returns:
        One array of unit vectors per anisotropic response function, in the order of
        `rfs`.
    """
    return [
        get_dirs(min(n_meas, rf.low_resolution_dirs))[:-1]
        for rf in rfs
        if not isinstance(rf, IsotropicRF)
    ]


def _to_sh(
    fods: np.ndarray, lmax: int, basis: str, *, exact_integral: bool = False
) -> np.ndarray:
    """Projects FOD amplitudes on DIRS_300 onto the SH basis up to lmax.

    The coefficients are a least-squares fit of the SH basis to the amplitudes,
    scaled by 300 / (4 pi), so that the integral of a smooth FOD over the sphere is
    the sum of the amplitudes, the tissue's signal fraction. When a few directions
    hold all the weight, as with the sparse solvers, that integral can be off by up
    to about 9%; `exact_integral` then sets c00 to the sum of the amplitudes divided
    by sqrt(4 pi).

    Args:
        fods: Amplitudes on DIRS_300, shape (n_voxels, 300).
        lmax: Maximum SH degree.
        basis: SH basis convention, see build_basis_matrix.
        exact_integral: Whether to set c00 from the sum of the amplitudes.
            Defaults to False.

    Returns:
        SH coefficients, shape (n_voxels, num_coeffs).
    """
    # n / (4 pi) turns the weights of n directions into a density on the sphere.
    density = len(DIRS_300) / (4.0 * np.pi)
    coeffs = fit_sh(fods, DIRS_300, lmax, basis=basis) * density
    if exact_integral:
        # Y00 is 1 / sqrt(4 pi) in every basis, so the integral is c00 sqrt(4 pi).
        coeffs[:, 0] = fods.sum(axis=1) / np.sqrt(4.0 * np.pi)
    return coeffs


def compute_max_lmax(bvals: np.ndarray, proposed_lmax: int) -> int:
    """Returns the largest lmax the highest shell can support.

    It is the largest even lmax up to proposed_lmax whose (lmax + 1)(lmax + 2) / 2
    SH coefficients do not exceed the number of directions in the highest shell.

    Args:
        bvals: b-value of each measurement in s/mm^2, shape (N,).
        proposed_lmax: The lmax asked for.

    Returns:
        The lmax to use.

    Raises:
        ValueError: If the highest shell cannot support even lmax = 2.
    """
    n_dirs = np.sum(shell_mask(bvals, np.max(bvals)))
    lmax = proposed_lmax

    while lmax > 0:
        n_params = (lmax + 1) * (lmax + 2) // 2
        if n_dirs >= n_params:
            return lmax
        lmax -= 2

    raise ValueError(f"Not enough gradient directions ({n_dirs}) to fit even lmax=2.")


def compute_shell_weights(bvals: np.ndarray, inner_weight: float = 0.2) -> np.ndarray:
    """Returns the weight 1 for the highest shell and inner_weight for the others.

    Args:
        bvals: b-value of each measurement in s/mm^2, shape (N,).
        inner_weight: Weight of the measurements outside the highest shell.
            Defaults to 0.2.

    Returns:
        The weight of each measurement, shape (N,).
    """
    weights = np.full(bvals.shape, inner_weight, dtype=np.float64)
    weights[shell_mask(bvals, np.max(bvals))] = 1.0
    return weights


def normalize_by_b0(signal: np.ndarray, is_b0: np.ndarray) -> np.ndarray:
    """Divides the signal of each voxel by its mean b0 signal.

    Args:
        signal: Signal, shape (n_voxels, n_meas).
        is_b0: Which measurements are b0s, shape (n_meas,) bool (the gradient
            table's `b0_mask`).

    Returns:
        Normalised signal, same shape. Unchanged if there are no b0 measurements; a
        zero b0 mean is replaced by 1e-16.
    """
    n_voxels = signal.shape[0]
    b0_means = np.mean(signal[:, is_b0], axis=1) if np.any(is_b0) else np.ones(n_voxels)
    b0_means[b0_means == 0] = 1e-16
    signal_norm: npt.NDArray[np.floating] = signal / b0_means[:, np.newaxis]
    return signal_norm


def merge_fods(
    fods: list[npt.NDArray[np.floating]],
    tissue_fractions: npt.NDArray[np.floating],
    method: Literal["ws", "majority"] = "ws",
) -> npt.NDArray[np.floating]:
    """Merges two or more fods into a single one using the selected strategy.

    Args:
        fods: The list of fods to merge
        tissue_fractions: tissue fractions to use for fod merging.
        method: "ws" for the weighted sum, "majority" for the FOD of the largest
            anisotropic fraction in each voxel.


    Returns:
        npt.NDArray[np.floating]: The merged fods.

    Raises:
        ValueError: Incorrect method or non matching fods/tissue shapes.
    """
    if tissue_fractions.shape[-1] < len(fods):
        raise ValueError("Mismatched number of fods and tissue fractions.")

    for f in fods:
        if f.shape != fods[0].shape:
            raise ValueError("Mismatched shapes between fods in list.")

    if method not in MERGING_METHOD:
        raise ValueError(
            f"Expected one method of {','.join(MERGING_METHOD)}, got {method}."
        )

    fod = np.zeros_like(fods[0])

    if method == "ws":
        for i, c_fod in enumerate(fods):
            fod += c_fod * tissue_fractions[:, :, :, i, np.newaxis]
    elif method == "majority":
        winner = tissue_fractions[..., : len(fods)].argmax(axis=-1)

        for i, c_fod in enumerate(fods):
            chosen = winner == i
            fod[chosen] = c_fod[chosen]

    return fod
