"""Constrained spherical deconvolution (CSD) of one shell into an FOD."""

import numpy as np

from larmor._log import LarmorLogger
from larmor._threads import n_threads
from larmor.dwi.inputs import as_mask, check_dwi, finite_rows, signal_chunks
from larmor.dwi.mask import unmask
from larmor.sd.csd_rf import estimate_csd_rf
from larmor.sd.csd_solver import (
    CSDSolver,
    csd_problem,
    csd_shell,
    solve_csd,
    warn_unconverged,
)
from larmor.sd.model import SDFit
from larmor.sd.progress import RESPONSE_STAGE, report_stage, report_voxels
from larmor.sd.rf import ZonalSHRF
from larmor.sd.sh import BASES, convert_sh_basis
from larmor.types import GradientTable, Volume


class CSDModel:
    """Constrained spherical deconvolution (CSD) of one shell into an FOD.

    The fit uses the shell with the highest b-value and leaves out the others, b0s
    included. Each voxel's signal on that shell is fitted as it is, in the
    response's units (not divided by b0), so the FOD's integral (c00 sqrt(4 pi)) is
    the voxel's signal relative to the response's. Negative signal values are
    fitted as 0. lmax is kept as given, even with more SH coefficients than the
    shell has directions. FODs come out in the frame of the DWI's b-vectors.

    References:
        Tournier JD, Calamante F, Connelly A. Robust determination of the fibre
        orientation distribution in diffusion MRI: non-negativity constrained
        super-resolved spherical deconvolution. NeuroImage 35 (2007) 1459-1472.
        doi:10.1016/j.neuroimage.2007.02.016

        The defaults follow MRtrix3's dwi2fod csd: Tournier JD, Smith R, Raffelt D,
        et al. MRtrix3: A fast, flexible and open software framework for medical image
        processing and visualisation. NeuroImage 202 (2019) 116137.
        doi:10.1016/j.neuroimage.2019.116137
    """

    def __init__(
        self,
        response: ZonalSHRF | None = None,
        *,
        solver: CSDSolver | None = None,
        lmax: int = 8,
        basis: str = "tournier",
    ) -> None:
        """Sets up the deconvolution.

        Args:
            response: The white matter response, with a row for the shell to fit.
                Defaults to None, which estimates it in `fit` from the voxels of the
                mask with `estimate_csd_rf` and its defaults.
            solver: The CSD settings. Defaults to None, CSDSolver's defaults.
            lmax: Maximum degree of the FOD's SH coefficients. Defaults to 8.
            basis: SH basis of the FODs `fit` returns, "tournier" (MRtrix),
                "leemans" (ExploreDTI) or "descoteaux". Defaults to "tournier".

        Raises:
            ValueError: If the response is not a ZonalSHRF, lmax is odd or below 2,
                or the basis is unknown.
        """
        if response is not None and not isinstance(response, ZonalSHRF):
            raise ValueError(
                f"CSD takes a ZonalSHRF response, got {type(response).__name__}."
            )
        if lmax < 2 or lmax % 2 != 0:
            raise ValueError(f"lmax must be even and at least 2, got {lmax}")
        if basis not in BASES:
            raise ValueError(
                f"Invalid basis {basis}, expected one of {', '.join(BASES)}"
            )
        self.response = response
        self.solver = CSDSolver() if solver is None else solver
        self.lmax = lmax
        self.basis = basis

    def fit(
        self,
        volume: Volume,
        gtab: GradientTable,
        mask: Volume | np.ndarray | None = None,
        *,
        n_jobs: int = -1,
    ) -> SDFit:
        """Fits the FOD of every voxel of the mask.

        Args:
            volume: Diffusion-weighted volume to fit, shape (X, Y, Z, N).
            gtab: Its gradient table.
            mask: Voxels to fit (nonzero inside), on the volume's grid. An
                estimated response is estimated from these voxels too. Defaults to
                None, which fits every voxel.
            n_jobs: Number of threads for the fit, at most the number of cores; -1
                uses every core. Defaults to -1.

        Returns:
            SDFit: One FOD as SH coefficients in the model's basis, in world
            coordinates, zero outside the mask, and no tissue fractions. Voxels
            whose signal on the shell is not finite get a NaN FOD.

        Raises:
            ValueError: If n_jobs is 0 or below -1, the data have no
                diffusion-weighted shell, or the response has no row near its
                b-value.
        """
        threads = n_threads(n_jobs)
        check_dwi(volume, gtab)
        mask = (
            np.ones(volume.data.shape[:-1], dtype=bool)
            if mask is None
            else as_mask(volume, mask)
        )
        bval = csd_shell(gtab)
        on_shell = gtab.shell_index == int(np.argmin(np.abs(gtab.shells - bval)))
        response = self.response
        if response is None:
            report_stage(RESPONSE_STAGE)
            response = estimate_csd_rf(volume, gtab, mask, n_jobs=n_jobs).rf
        zonal = response.shell_coefficients(bval)
        if gtab.count_shells() > 1:
            LarmorLogger.warning(
                f"CSD fits one shell: using b = {bval:g} s/mm^2 "
                f"({int(on_shell.sum())} volumes) and leaving out the others."
            )
        voxels = np.nonzero(mask)
        finite = finite_rows(volume, voxels, on_shell)
        if not finite.all():
            LarmorLogger.warning(
                f"{int((~finite).sum())} voxels have a NaN or infinite signal; their "
                "FODs are NaN."
            )
        LarmorLogger.info(
            f"Processing {int(finite.sum())} voxels across {threads} cores using CSD "
            f"on b = {bval:g} s/mm^2..."
        )
        n_fitted = int(finite.sum())
        report_voxels(0, n_fitted)
        problem = csd_problem(gtab.bvecs[on_shell], zonal, self.lmax, self.solver)
        coeffs = np.full((len(finite), problem.forward.shape[1]), np.nan)
        rows_of_fitted = np.flatnonzero(finite)
        unconverged = 0
        fitted_voxels = tuple(axis[finite] for axis in voxels)
        for rows, signals in signal_chunks(volume, fitted_voxels, on_shell):
            fods, converged = solve_csd(signals, problem, threads)
            coeffs[rows_of_fitted[rows]] = fods
            unconverged += int((~converged).sum())
        warn_unconverged(unconverged, self.solver.max_iter)
        report_voxels(n_fitted, n_fitted)
        if self.basis != "tournier":
            coeffs = convert_sh_basis(coeffs, "tournier", self.basis)
        return SDFit(
            fods=[unmask(coeffs, mask)],
            tissue_fractions=None,
            basis=self.basis,
            lmax=self.lmax,
        )
