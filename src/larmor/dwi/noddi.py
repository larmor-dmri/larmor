"""The NODDI model, fitted with a dictionary of signals and refined by least squares."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import sqrt

import numpy as np
import numpy.typing as npi
from scipy.optimize import least_squares
from scipy.special import dawsn
from tqdm import tqdm

from larmor import _core
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
# Model codes of the compiled engine.
WATSON = 5
ZEPPELIN = 1
BALL = 2
# Weights of the ridge (L2) and L1 penalties of the sparse solve.
LAMBDA = 0.001
GAMMA = 0.5
# Largest kappa of the refinement, OD 0.01, tighter than any tissue.
KAPPA_MAX = 64
# Voxels per task of the thread pool.
CHUNK_VOXELS = 256


def _watson_tau(kappa: float) -> float:
    """Returns tau1, the mean squared cosine of a Watson density to its mean direction.

    tau1 follows Zhang et al. 2012, Eq. 7 by quadrature up to kappa = 1, where
    Eq. 8 loses digits, and Eq. 8 above; it is 1/3 at kappa = 0.
    """
    if kappa <= 1:
        nodes, weights = np.polynomial.legendre.leggauss(32)
        u = (nodes + 1) / 2
        density = weights * np.exp(kappa * u**2)
        return float(density @ u**2 / density.sum())
    return float(-1 / (2 * kappa) + 1 / (2 * dawsn(sqrt(kappa)) * sqrt(kappa)))


def _compartments() -> tuple[npi.NDArray[np.int64], npi.NDArray[np.float64]]:
    """Returns the model codes and parameters of the compartments of every atom.

    The first 12 rows are the Watson sticks of each KAPPA, which every V_IC
    shares; the next 144 are the zeppelins of the atoms in dictionary order, with
    the diffusivities of Zhang et al. 2012, Eqs. 5 and 6.
    """
    v_ic = np.repeat(V_IC, 12)
    tau1 = np.tile([_watson_tau(float(kappa)) for kappa in KAPPA], 12)
    models = np.r_[np.full(12, WATSON), np.full(144, ZEPPELIN)].astype(np.int64)
    parameters = np.zeros((156, 4))
    parameters[:12, 0] = D_PAR
    parameters[:12, 3] = KAPPA
    parameters[12:, 0] = D_PAR - D_PAR * v_ic * (1 - tau1)
    parameters[12:, 1] = D_PAR - D_PAR * v_ic * (1 + tau1) / 2
    return models, parameters


_MODELS, _PARAMETERS = _compartments()


def _dictionary(
    bvals: npi.NDArray[np.float64],
    bvecs: npi.NDArray[np.float64],
    mu: npi.NDArray[np.float64],
) -> npi.NDArray[np.float64]:
    """Returns the dictionary for one mean direction mu, one column per atom, (N, 145).

    Column 12 i + j is the tissue atom of V_IC[i] and KAPPA[j], v_ic sticks and
    1 - v_ic zeppelin; the last column is free water.
    """
    # The pulse timings do not change these models.
    attenuation = _core.compartment_attenuation(
        _MODELS,
        _PARAMETERS,
        np.ascontiguousarray(bvals, dtype=np.float64),
        np.ascontiguousarray(bvecs @ mu, dtype=np.float64),
        20.0,
        40.0,
    )
    sticks = np.tile(attenuation[:12], (12, 1))
    v_ic = np.repeat(V_IC, 12)[:, None]
    tissue = v_ic * sticks + (1 - v_ic) * attenuation[12:]
    return np.ascontiguousarray(np.vstack([tissue, np.exp(-bvals * D_ISO)]).T)


def _signal(
    v_ic: float,
    kappa: float,
    v_iso: float,
    bvals: npi.NDArray[np.float64],
    cos_g_mu: npi.NDArray[np.float64],
) -> npi.NDArray[np.float64]:
    """Returns the NODDI signal of v_ic, kappa and v_iso, one value per volume.

    The signal is Zhang et al. 2012, Eq. 1, about the mean direction mu, with
    cos_g_mu holding g . mu per volume.
    """
    tau1 = _watson_tau(kappa)
    d_par_ec = D_PAR - D_PAR * v_ic * (1 - tau1)
    d_perp_ec = D_PAR - D_PAR * v_ic * (1 + tau1) / 2
    models = np.array([WATSON, ZEPPELIN, BALL], dtype=np.int64)
    parameters = np.array(
        [[D_PAR, 0, 0, kappa], [d_par_ec, d_perp_ec, 0, 0], [D_ISO, 0, 0, 0]]
    )
    a = _core.compartment_attenuation(
        models,
        parameters,
        np.ascontiguousarray(bvals, dtype=np.float64),
        np.ascontiguousarray(cos_g_mu, dtype=np.float64),
        20.0,
        40.0,
    )
    weights = np.array(
        [(1 - v_iso) * v_ic, (1 - v_iso) * (1 - v_ic), v_iso], np.float64
    )
    return np.asarray(weights @ a)


def _fit_voxel(
    signal: npi.NDArray[np.float64],
    gtab: GradientTable,
    mu: npi.NDArray[np.float64],
) -> tuple[float, float, float]:
    """Returns v_ic, kappa and v_iso of one voxel's signal, with mu its mean direction.

    The dictionary estimate is then refined by least squares, with mu fixed. A
    voxel without tissue (v_iso = 1) gets v_ic = kappa = 0, and one whose mean b0
    is not positive gets zeros.
    """
    b0 = signal[gtab.b0_mask].mean()
    if not b0 > 0:
        return 0.0, 0.0, 0.0
    y = signal / b0
    phi = _dictionary(gtab.bvals, gtab.bvecs, mu)
    tissue = np.ascontiguousarray(phi[:, :144])

    # Free water from a plain NNLS on every atom (no ridge, no L1).
    v_iso = float(_core.ridge_l1_nnls(phi, y, 0.0, 0.0)[144])
    y_t = y - phi[:, 144] * v_iso

    # The tissue atoms the signal needs, then their weights without penalties.
    x_t = _core.ridge_l1_nnls(tissue, y_t, sqrt(LAMBDA), GAMMA)
    keep = x_t > 0.0
    support = np.ascontiguousarray(tissue[:, keep])
    x_t[keep] = _core.ridge_l1_nnls(support, y_t, 0.0, 0.0)

    total = x_t.sum()
    if total == 0:
        return 0.0, 0.0, v_iso

    v_ic = float((x_t @ np.repeat(V_IC, 12)) / total)
    kappa = float((x_t @ np.tile(KAPPA, 12)) / total)
    cos_g_mu = gtab.bvecs @ mu

    def residuals(p: npi.NDArray[np.float64]) -> npi.NDArray[np.float64]:
        return np.asarray(_signal(p[0], p[1], p[2], gtab.bvals, cos_g_mu) - y)

    # least_squares needs a start inside the bounds; AMICO can give 0 or 1.
    start = [
        np.clip(v_ic, 0.01, 0.99),
        np.clip(kappa, 0.01, KAPPA_MAX - 1),
        np.clip(v_iso, 0.001, 0.999),
    ]
    result = least_squares(residuals, start, bounds=([0, 0, 0], [1, KAPPA_MAX, 1]))
    v_ic, kappa, v_iso = result.x
    return float(v_ic), float(kappa), float(v_iso)


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
        mu: Mean neurite direction in world coordinates (RAS+); its sign is
            arbitrary.
    """

    v_ic: np.ndarray
    od: np.ndarray
    kappa: np.ndarray
    v_iso: np.ndarray
    mu: np.ndarray


class NoddiModel:
    """NODDI model, which fits neurite density, dispersion and free water per voxel.

    `fit` returns a `NoddiFit`. Each voxel's signal is first matched by a
    non-negative mix of precomputed NODDI signals about the voxel's DTI direction,
    as in AMICO, and the parameters of that mix are then refined by least squares
    with the direction fixed. The refinement differs from AMICO: without it, the
    parameters averaged over the mix underestimate OD.

    References:
        Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC. NODDI: Practical
        in vivo neurite orientation dispersion and density imaging of the human
        brain. NeuroImage 61 (2012) 1000-1016.
        Daducci A, Canales-Rodriguez EJ, Zhang H, Dyrby TB, Alexander DC, Thiran
        JP. Accelerated Microstructure Imaging via Convex Optimization (AMICO)
        from diffusion MRI data. NeuroImage 105 (2015) 32-44.
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

        Args:
            volume: Diffusion-weighted volume, shape (X, Y, Z, N), with b0 volumes.
            gtab: Its gradient table.
            mask: Voxels to fit (nonzero inside), on the volume's grid. Defaults
                to None, which computes a brain mask from the mean b0 volume.
            mu: The mean direction of each voxel, (X, Y, Z, 3), in world
                coordinates. Defaults to None, which takes the principal
                eigenvector of a DTI fit.
            n_jobs: Number of threads for the fit, at most the number of cores;
                -1 uses every core. Defaults to -1.
            progress: Whether to show a progress bar. Defaults to False.

        Returns:
            NoddiFit: The maps, zero outside the mask.

        Raises:
            ValueError: If the table or the mask does not match the volume, or mu
                does not have the shape (X, Y, Z, 3), or n_jobs is 0 or below -1.
        """
        threads = n_threads(n_jobs)
        check_dwi(volume, gtab)
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
        mean_directions: np.ndarray = mu
        voxels = np.nonzero(mask)
        n_voxels = len(voxels[0])
        values = np.zeros((n_voxels, 3))

        def fit_chunk(start: int) -> None:
            # Each chunk is read and made float64 on its own, so the whole masked
            # image is never copied.
            chunk = slice(start, min(start + CHUNK_VOXELS, n_voxels))
            signals = signal_rows(volume, voxels, chunk)
            index = tuple(axis[chunk] for axis in voxels)
            directions = mean_directions[index].astype(np.float64)
            for row in range(len(signals)):
                values[chunk.start + row] = _fit_voxel(
                    signals[row], gtab, directions[row]
                )

        # The compiled calls release the GIL, so threads share the chunks.
        with (
            ThreadPoolExecutor(threads) as pool,
            tqdm(total=n_voxels, desc="Fitting voxels", disable=not progress) as bar,
        ):
            starts = range(0, n_voxels, CHUNK_VOXELS)
            futures = {pool.submit(fit_chunk, start): start for start in starts}
            for future in as_completed(futures):
                future.result()
                bar.update(min(CHUNK_VOXELS, n_voxels - futures[future]))

        v_ic = np.zeros(shape, dtype=np.float32)
        kappa = np.zeros(shape, dtype=np.float32)
        v_iso = np.zeros(shape, dtype=np.float32)
        v_ic[mask], kappa[mask], v_iso[mask] = values.T

        # kappa = 0 is fully dispersed, OD = 1; without neurites OD stays 0.
        od = np.zeros(shape, dtype=np.float32)
        neurites = mask & (v_ic > 0)
        with np.errstate(divide="ignore"):
            od[neurites] = (2 / np.pi) * np.arctan(1 / kappa[neurites])
        mu = np.where(mask[..., None], mu, 0).astype(np.float32)
        return NoddiFit(v_ic=v_ic, od=od, kappa=kappa, v_iso=v_iso, mu=mu)
