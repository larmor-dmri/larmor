"""The NODDI model, started from a signal dictionary and fitted by Rician likelihood."""

from dataclasses import dataclass
from math import sqrt

import numpy as np
from tqdm import tqdm

from larmor import _core
from larmor._log import LarmorLogger
from larmor._threads import n_threads
from larmor.dwi.dti import DTIModel
from larmor.dwi.inputs import as_mask, check_dwi, mean_b0, signal_rows
from larmor.dwi.mask import create_brain_mask
from larmor.types import GradientTable, Volume

# The neurite densities and Watson concentrations of the dictionary's atoms.
V_IC = np.linspace(0.1, 1, 12)
KAPPA = np.linspace(0, 20, 12)
# The diffusivities NODDI fixes, along the neurites and of free water, mm^2/s.
D_PAR = 1.7e-3
D_ISO = 3.0e-3
# Weights of the ridge (L2) and L1 penalties of the dictionary's sparse solve.
LAMBDA = 0.001
GAMMA = 0.5
# Voxels per call to the compiled fit.
CHUNK_VOXELS = 4096


@dataclass(frozen=True)
class NoddiFit:
    """Holds the maps of a NODDI fit, on the image grid and zero outside the mask.

    Each map has the spatial shape (X, Y, Z) of the DWI, with a last axis of 3
    for mu, and is float32. The fractions are signal fractions, which differ from
    volume fractions when the compartments have different T2 or proton density.

    Attributes:
        v_ic: Neurite density, the neurites' share of the tissue outside free
            water.
        od: Orientation dispersion index, (2 / pi) arctan(1 / kappa), from 0
            (aligned neurites) to 1 (isotropic); 0 where v_ic is 0.
        kappa: Watson concentration of the neurites about mu.
        v_iso: Free-water fraction.
        mu: Fitted mean neurite direction in world coordinates (RAS+); its sign
            is arbitrary.
    """

    v_ic: np.ndarray
    od: np.ndarray
    kappa: np.ndarray
    v_iso: np.ndarray
    mu: np.ndarray


class NoddiModel:
    """NODDI model, which fits neurite density, dispersion and free water per voxel.

    `fit` returns a `NoddiFit`. The fit is that of the NODDI Matlab Toolbox, by
    maximum likelihood with Rician noise, but it starts from a dictionary solve
    (Daducci et al. 2015) rather than a grid search.

    References:
        Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC. NODDI: Practical
        in vivo neurite orientation dispersion and density imaging of the human
        brain. NeuroImage 61 (2012) 1000-1016.
        Daducci A, Canales-Rodriguez EJ, Zhang H, Dyrby TB, Alexander DC, Thiran
        JP. Accelerated Microstructure Imaging via Convex Optimization (AMICO)
        from diffusion MRI data. NeuroImage 105 (2015) 32-44.
        NODDI Matlab Toolbox 1.05, https://www.nitrc.org/projects/noddi_toolbox
    """

    def fit(
        self,
        volume: Volume,
        gtab: GradientTable,
        mask: Volume | np.ndarray | None = None,
        mu: np.ndarray | None = None,
        *,
        n_jobs: int = -1,
        progress: bool = False,
    ) -> NoddiFit:
        """Fits NODDI in every voxel of the mask.

        Each voxel is fitted on its own, so the result does not depend on n_jobs.
        A voxel with a signal that is not finite, without a positive mean b0, or
        with fewer than six positive diffusion-weighted measurements is not fitted
        and gets zeros, and a warning gives the number of such voxels.

        Args:
            volume: Diffusion-weighted volume, shape (X, Y, Z, N), with b0 volumes.
            gtab: Its gradient table, with at least two non-zero shells.
            mask: Voxels to fit (nonzero inside), on the volume's grid. Defaults
                to None, which computes a brain mask from the mean b0 volume.
            mu: The starting direction of each voxel, (X, Y, Z, 3), in world
                coordinates; it need not be unit length. Defaults to None, which
                takes the principal eigenvector of a DTI fit.
            n_jobs: Number of threads for the fit, at most the number of cores;
                -1 uses every core. Defaults to -1.
            progress: Whether to show a progress bar. Defaults to False.

        Returns:
            NoddiFit: The maps, zero outside the mask.

        Raises:
            ValueError: If the table or the mask does not match the volume, the
                table has fewer than two non-zero shells, mu does not have the
                shape (X, Y, Z, 3) or is zero or not finite in a mask voxel, or
                n_jobs is 0 or below -1.
        """
        threads = n_threads(n_jobs)
        check_dwi(volume, gtab)
        if gtab.count_shells() < 2:
            raise ValueError(
                "NODDI needs at least two non-zero b-value shells; the gradient table "
                f"has {gtab.count_shells()}."
            )
        shape = volume.data.shape[:3]
        if mu is not None and mu.shape != (*shape, 3):
            raise ValueError(
                f"mu must have the shape {(*shape, 3)} of the volume, got {mu.shape}."
            )
        mask = (
            create_brain_mask(mean_b0(volume, gtab))
            if mask is None
            else as_mask(volume, mask)
        )
        if mu is None:
            mu = DTIModel().fit(volume, gtab, mask, n_jobs=n_jobs).pe
        else:
            norms = np.linalg.norm(mu[mask], axis=-1)
            if not (np.isfinite(norms) & (norms > 0)).all():
                raise ValueError("mu must be finite and nonzero in every mask voxel.")
        voxels = np.nonzero(mask)
        n_voxels = len(voxels[0])
        values = np.zeros((n_voxels, 6))
        fitted = np.zeros(n_voxels, dtype=bool)
        bvals = np.ascontiguousarray(gtab.bvals, dtype=np.float64)
        bvecs = np.ascontiguousarray(gtab.bvecs, dtype=np.float64)
        is_b0 = np.ascontiguousarray(gtab.b0_mask)

        with tqdm(total=n_voxels, desc="Fitting voxels", disable=not progress) as bar:
            for start in range(0, n_voxels, CHUNK_VOXELS):
                # Each chunk is read and made float64 on its own, so the whole
                # masked image is never copied.
                chunk = slice(start, min(start + CHUNK_VOXELS, n_voxels))
                index = tuple(axis[chunk] for axis in voxels)
                values[chunk], fitted[chunk] = _core.fit_noddi(
                    np.ascontiguousarray(signal_rows(volume, voxels, chunk)),
                    np.ascontiguousarray(mu[index], dtype=np.float64),
                    bvals,
                    bvecs,
                    is_b0,
                    V_IC,
                    KAPPA,
                    D_PAR,
                    D_ISO,
                    sqrt(LAMBDA),
                    GAMMA,
                    threads,
                )
                bar.update(chunk.stop - chunk.start)
        if not fitted.all():
            LarmorLogger.warning(
                "%d mask voxels have a signal that is not finite, no positive b0 or "
                "too few positive measurements; they are not fitted and come out as 0.",
                int((~fitted).sum()),
            )

        v_ic = np.zeros(shape, dtype=np.float32)
        kappa = np.zeros(shape, dtype=np.float32)
        v_iso = np.zeros(shape, dtype=np.float32)
        fitted_mu = np.zeros((*shape, 3), dtype=np.float32)
        v_ic[mask], kappa[mask], v_iso[mask] = values[:, :3].T
        fitted_mu[mask] = values[:, 3:]

        # kappa = 0 is fully dispersed, OD = 1; without neurites OD stays 0.
        od = np.zeros(shape, dtype=np.float32)
        neurites = mask & (v_ic > 0)
        with np.errstate(divide="ignore", over="ignore"):
            od[neurites] = (2 / np.pi) * np.arctan(1 / kappa[neurites])
        return NoddiFit(v_ic=v_ic, od=od, kappa=kappa, v_iso=v_iso, mu=fitted_mu)
