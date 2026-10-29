"""Multi-shell multi-tissue constrained spherical deconvolution (MSMT-CSD)."""

import numpy as np
from tqdm import tqdm

from larmor._log import LarmorLogger
from larmor._threads import n_threads
from larmor.dwi.inputs import as_mask, check_dwi
from larmor.dwi.mask import unmask
from larmor.sd.model import FIT_CHUNK, SDFit
from larmor.sd.msmt import fit_msmt, prepare_msmt
from larmor.sd.progress import report_voxels
from larmor.sd.rf import ZonalSHRF
from larmor.sd.sh import BASES, convert_sh_basis
from larmor.sd.sphere import DIRS_300
from larmor.sd.tissue_responses import estimate_tissue_responses
from larmor.types import GradientTable, Volume

# Keeps the normalisation of a voxel's fractions finite when they are all 0.
FRACTION_SUM_OFFSET = 1e-16


class MSMTCSDModel:
    """Multi-shell multi-tissue constrained spherical deconvolution (MSMT-CSD).

    Each tissue has a response given by its zonal SH coefficients on every shell
    (a `ZonalSHRF`, in the format MRtrix3's dwi2response writes). A response with
    lmax above 0 gives an FOD, and one with lmax 0 an isotropic tissue. In each
    voxel, all tissues are fitted at once to every measurement, b = 0 included,
    with their FOD amplitudes and isotropic coefficients kept non-negative. The
    signal is fitted as measured, not divided by S0, so an FOD's integral,
    c00 sqrt(4 pi), is its tissue's share of the signal relative to the response.
    The least squares has a ridge of 1e-10 times the largest diagonal entry of the
    normal matrix. b-values are in s/mm^2; FODs come out in the frame of the DWI's
    b-vectors.

    References:
        Jeurissen B, Tournier JD, Dhollander T, Connelly A, Sijbers J. Multi-tissue
        constrained spherical deconvolution for improved analysis of multi-shell
        diffusion MRI data. NeuroImage 103 (2014) 411-426.
        doi:10.1016/j.neuroimage.2014.07.061

        The defaults (lmax 8 or the response's, all shells weighted equally, the
        ridge) follow MRtrix3's dwi2fod msmt_csd: Tournier JD, Smith R, Raffelt D, et
        al. MRtrix3: A fast, flexible and open software framework for medical image
        processing and visualisation. NeuroImage 202 (2019) 116137.
        doi:10.1016/j.neuroimage.2019.116137

        Lawson CL, Hanson RJ. Solving Least Squares Problems. Prentice-Hall, 1974
        (chapter 23, least squares with inequality constraints).
    """

    def __init__(
        self,
        rfs: list[ZonalSHRF] | None = None,
        *,
        lmax: int = 8,
        basis: str = "tournier",
        directions: np.ndarray | None = None,
        progress: bool = False,
    ) -> None:
        """Sets up the deconvolution.

        Args:
            rfs: One response per tissue; those with lmax above 0 come first in the
                fit's FODs and fractions, then the isotropic ones, each group in the
                order given. Defaults to None, which estimates white matter, grey
                matter and CSF responses from the data and mask given to `fit`
                with `estimate_tissue_responses`.
            lmax: Maximum SH degree of the FODs, lowered to the lowest lmax of the
                anisotropic responses. Defaults to 8.
            basis: SH basis of the FODs `fit` returns, "tournier" (MRtrix),
                "leemans" (ExploreDTI) or "descoteaux". Defaults to "tournier".
            directions: Unit vectors, shape (n, 3), on which the FOD amplitudes are
                kept non-negative; a hemisphere is enough. Defaults to None,
                Larmor's built-in set of 300 directions.
            progress: Whether to show a progress bar over the voxels. Defaults to
                False.

        Raises:
            ValueError: If lmax is odd or below 2, the basis is unknown, or no response
                has lmax above 0.
        """
        if lmax < 2 or lmax % 2 != 0:
            raise ValueError(f"lmax must be even and at least 2, got {lmax}")
        if basis not in BASES:
            raise ValueError(
                f"Invalid basis {basis}, expected one of {', '.join(BASES)}"
            )
        if rfs is not None and not any(rf.lmax > 0 for rf in rfs):
            raise ValueError("MSMT-CSD needs at least one response with lmax above 0.")
        self.rfs = None if rfs is None else _anisotropic_first(rfs)
        self.lmax = lmax
        self.basis = basis
        self.directions = DIRS_300 if directions is None else np.asarray(directions)
        self.progress = progress

    def fit(
        self,
        volume: Volume,
        gtab: GradientTable,
        mask: Volume | np.ndarray | None = None,
        *,
        n_jobs: int = -1,
    ) -> SDFit:
        """Fits the FODs and tissue fractions of every voxel of the mask.

        Args:
            volume: Diffusion-weighted volume to fit, shape (X, Y, Z, N).
            gtab: Its gradient table.
            mask: Voxels to fit (nonzero inside), on the volume's grid. Needed when
                the responses are estimated; defaults to None, every voxel.
            n_jobs: Number of threads, at most the number of cores; -1 uses every
                core. Defaults to -1.

        Returns:
            SDFit: One FOD per response with lmax above 0, as SH coefficients in the
            model's basis, and, with more than one response, the fraction of each
            tissue per voxel, its c00 sqrt(4 pi) normalised to sum to 1. Voxels
            whose signal is not finite are NaN.

        Raises:
            ValueError: If n_jobs is 0 or below -1, a b-value is not near any shell
                of a response, the measurements cannot determine every SH
                coefficient, or the responses are to be estimated without a mask.
        """
        threads = n_threads(n_jobs)
        check_dwi(volume, gtab)
        if mask is None and self.rfs is None:
            raise ValueError("Estimating the tissue responses needs a brain mask.")
        if mask is None:
            mask = np.ones(volume.data.shape[:-1], dtype=bool)
        else:
            mask = as_mask(volume, mask)
        if self.rfs is None:
            estimated = estimate_tissue_responses(volume, gtab, mask, n_jobs=threads)
            rfs = [estimated.wm, estimated.gm, estimated.csf]
        else:
            rfs = self.rfs

        n_aniso = sum(1 for rf in rfs if rf.lmax > 0)
        lmax = min(self.lmax, *(rf.lmax for rf in rfs[:n_aniso]))
        if lmax < self.lmax:
            LarmorLogger.warning(
                f"lmax {self.lmax} is above the lmax of an anisotropic response, "
                f"lowering to {lmax}"
            )
        lmaxes = [lmax] * n_aniso + [0] * (len(rfs) - n_aniso)
        sizes = [(degree + 1) * (degree + 2) // 2 for degree in lmaxes]
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)
        design = prepare_msmt(rfs, lmaxes, gtab.bvals, gtab.bvecs, self.directions)

        signals = volume.as_float()[mask].astype(np.float64)
        finite = np.isfinite(signals).all(axis=1)
        if not finite.all():
            LarmorLogger.warning(
                f"{int((~finite).sum())} voxels have a NaN or infinite signal; their "
                "FODs and fractions are NaN."
            )
        LarmorLogger.info(
            f"Processing {int(finite.sum())} voxels across {threads} cores using "
            "MSMT-CSD..."
        )
        fitted = signals[finite]
        fitted_coefficients = np.empty((len(fitted), int(sum(sizes))))
        converged = np.empty(len(fitted), dtype=bool)
        chunks = range(0, len(fitted), FIT_CHUNK)
        report_voxels(0, len(fitted))
        for start in tqdm(chunks, desc="Fitting voxels", disable=not self.progress):
            stop = min(start + FIT_CHUNK, len(fitted))
            fitted_coefficients[start:stop], converged[start:stop] = fit_msmt(
                fitted[start:stop], design, threads
            )
            report_voxels(stop, len(fitted))
        if not converged.all():
            LarmorLogger.warning(
                f"The constrained fit of {int((~converged).sum())} voxels stopped at "
                "its iteration limit; their FODs may be slightly negative."
            )
        coefficients = np.full((len(signals), fitted_coefficients.shape[1]), np.nan)
        coefficients[finite] = fitted_coefficients

        fods, fractions = _fods_and_fractions(
            coefficients, mask, starts, sizes, n_aniso, self.basis
        )
        return SDFit(fods=fods, tissue_fractions=fractions, basis=self.basis, lmax=lmax)


def _fods_and_fractions(
    coefficients: np.ndarray,
    mask: np.ndarray,
    starts: np.ndarray,
    sizes: list[int],
    n_aniso: int,
    basis: str,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Splits the fitted coefficients into FODs on the grid and tissue fractions.

    Args:
        coefficients: The coefficients of every tissue, one row per masked voxel.
        mask: The fitted voxels.
        starts: The first column of each tissue in `coefficients`.
        sizes: The number of columns of each tissue.
        n_aniso: The number of tissues with lmax above 0, which come first.
        basis: The SH basis of the FODs returned.

    Returns:
        The FOD of each anisotropic tissue in `basis`, and the fraction of each
        tissue, or None with one tissue.
    """
    fods = [
        unmask(
            convert_sh_basis(coefficients[:, start : start + size], "tournier", basis),
            mask,
        )
        for start, size in zip(starts[:n_aniso], sizes[:n_aniso], strict=True)
    ]
    fractions = None
    if len(starts) > 1:
        shares = np.maximum(coefficients[:, starts] * np.sqrt(4 * np.pi), 0.0)
        total = shares.sum(axis=1, keepdims=True) + FRACTION_SUM_OFFSET
        fractions = unmask(shares / total, mask)
    return fods, fractions


def _anisotropic_first(rfs: list[ZonalSHRF]) -> list[ZonalSHRF]:
    """Returns the responses with lmax above 0 first, then the isotropic ones."""
    return [rf for rf in rfs if rf.lmax > 0] + [rf for rf in rfs if rf.lmax == 0]
