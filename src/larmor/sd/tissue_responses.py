"""Estimation of white matter, grey matter and CSF responses from the data alone."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.special import eval_legendre, lpmv

from larmor import _core
from larmor._log import LarmorLogger
from larmor._threads import n_threads
from larmor.dwi.inputs import as_mask, check_dwi
from larmor.sd.msmt import fit_msmt, prepare_msmt
from larmor.sd.peaks import find_peaks
from larmor.sd.rf import ZonalSHRF
from larmor.sd.sphere import DIRS_300
from larmor.types import GradientTable, Volume

# Factor that turns a median absolute deviation into a robust standard deviation.
_MAD_TO_SD = 1.4826
# Signal decay metrics are capped at this value.
_MAX_SDM = 10.0
# Degree of the white matter response on the diffusion-weighted shells.
_WM_RESPONSE_LMAX = 10
# Angles from the fibre, every degree from 0 to 90, at which the white matter
# response is kept non-negative and rising.
_CONSTRAINT_ANGLES = np.radians(np.arange(91.0))
# Ridge of the response fits, relative to the largest diagonal entry of the normal
# matrix.
_RESPONSE_RIDGE = 1e-10


@dataclass(frozen=True)
class TissueResponses:
    """Holds the three responses `estimate_tissue_responses` returns and their voxels.

    Each response has a row per shell as `response_shells` gives them, ready for
    `MSMTCSDModel`. The voxel masks have shape (X, Y, Z) and are bool.

    Attributes:
        wm: Single-fibre white matter response, with lmax 10 on the
            diffusion-weighted shells and lmax 0 on b0.
        gm: Grey matter response, isotropic.
        csf: CSF response, isotropic.
        wm_voxels: The single-fibre white matter voxels.
        gm_voxels: The grey matter voxels.
        csf_voxels: The CSF voxels.
    """

    wm: ZonalSHRF
    gm: ZonalSHRF
    csf: ZonalSHRF
    wm_voxels: np.ndarray
    gm_voxels: np.ndarray
    csf_voxels: np.ndarray


def response_shells(gtab: GradientTable) -> np.ndarray:
    """Returns the b-values of the rows of an MRtrix response for these data.

    A multi-shell response has one row per shell, in increasing b-value.

    Args:
        gtab: The data's gradient table.

    Returns:
        0 when the data have b0 volumes, then each shell of `gtab.shells`, in
        s/mm^2.
    """
    shells: np.ndarray = gtab.shells
    if gtab.b0_mask.any():
        shells = np.concatenate([[0.0], shells])
    return shells


def build_shell_response(coefficients: npt.ArrayLike, gtab: GradientTable) -> ZonalSHRF:
    """Turns the rows of an MRtrix3 response file into a response for these data.

    A response file holds only zonal SH coefficients, one row per shell, without the
    b-values. Each row gets the b-value of a shell of the data, so that the response
    can be passed to `MSMTCSDModel`.

    Args:
        coefficients: The response, one row per shell in increasing b-value, b0 first
            when the data have b0s, as `read_mrtrix_response` returns it.
        gtab: The gradient table of the data the response belongs to.

    Returns:
        ZonalSHRF: The response with the b-value of each row (`response_shells`).

    Raises:
        ValueError: If there is not one row per shell of the data.
    """
    rows = np.atleast_2d(np.asarray(coefficients, dtype=np.float64))
    shells = response_shells(gtab)
    if len(rows) != len(shells):
        raise ValueError(
            f"The response has {len(rows)} rows, but the data have {len(shells)} "
            "shells (b0 included); give one row per shell."
        )
    return ZonalSHRF(bvals=shells, coefficients=rows)


def estimate_tissue_responses(
    volume: Volume,
    gtab: GradientTable,
    mask: Volume | np.ndarray,
    *,
    erode: int = 3,
    fa_threshold: float = 0.2,
    wm_percent: float = 0.5,
    gm_percent: float = 2.0,
    csf_percent: float = 10.0,
    n_jobs: int = -1,
) -> TissueResponses:
    """Estimates white matter, grey matter and CSF responses without a segmentation.

    The responses are those `MSMTCSDModel` needs, estimated from the diffusion data
    alone, without a T1 image or a tissue segmentation. The voxels of the eroded
    brain mask are split into white matter, grey matter and CSF by their FA and by
    how fast their signal falls with b; a share of each class is kept, and each
    response is fitted to the signal of its voxels.

    Args:
        volume: Diffusion-weighted volume, shape (X, Y, Z, N), with b0 volumes and at
            least one shell.
        gtab: Its gradient table.
        mask: Brain mask (nonzero inside), on the volume's grid.
        erode: Number of erosions of the brain mask, each by one voxel and its 6
            neighbours. Defaults to 3.
        fa_threshold: FA that separates white matter from the rest. Defaults to 0.2.
        wm_percent: Share of the refined white matter voxels (after outlier
            removal) kept as single-fibre voxels, in percent. Defaults to 0.5.
        gm_percent: Share of the refined grey matter voxels (after outlier
            removal) kept, in percent. Defaults to 2.0.
        csf_percent: Share of the refined CSF voxels (after thresholding) kept, in
            percent. Defaults to 10.0.
        n_jobs: Number of threads; -1 uses every core. Defaults to -1.

    Returns:
        TissueResponses: The three responses in the units of the signal, on the shells
        of `response_shells(gtab)`, and the voxels each was estimated from.

    Raises:
        ValueError: If the data have no b0 volume or no shell, or a step finds no
            voxel.

    References:
        Dhollander T, Raffelt D, Connelly A. Unsupervised 3-tissue response function
        estimation from single-shell or multi-shell diffusion MR data without a
        co-registered T1 image. ISMRM Workshop on Breaking the Barriers of Diffusion
        MRI (2016) 5.

        Dhollander T, Mito R, Raffelt D, Connelly A. Improved white matter response
        function estimation for 3-tissue constrained spherical deconvolution. Proc
        Intl Soc Mag Reson Med 27 (2019) 555.

        Its steps and defaults follow MRtrix3's dwi2response dhollander, with the
        white matter response fitted as in its amp2response (Smith RE, Dhollander T,
        Connelly A. Constrained linear least squares estimation of anisotropic
        response function for spherical deconvolution. ISMRM Workshop on Breaking
        the Barriers of Diffusion MRI (2016) 23). Tournier JD et al. MRtrix3: A
        fast, flexible and open software framework for medical image processing and
        visualisation. NeuroImage 202 (2019) 116137.
        doi:10.1016/j.neuroimage.2019.116137

        The automatic thresholds maximise the correlation between the image and its
        thresholded mask: Ridgway GR, Omar R, Ourselin S, Hill DL, Warren JD, Fox NC.
        Issues with threshold masking in voxel-based morphometry of atrophied brains.
        NeuroImage 44 (2009) 99-111.
    """
    threads = n_threads(n_jobs)
    check_dwi(volume, gtab)
    brain = as_mask(volume, mask)
    if not gtab.b0_mask.any() or gtab.count_shells() == 0:
        raise ValueError(
            "Estimating tissue responses needs b0 volumes and at least one shell."
        )
    shell_bvals = response_shells(gtab)
    # Row of each volume in the responses: 0 for the b0s, then the shells.
    rows = np.where(gtab.b0_mask, 0, gtab.shell_index + 1)

    eroded = _erode(brain, erode)
    signal = volume.as_float()[eroded].astype(np.float64)

    sdm, valid = _signal_decay_metric(signal, rows, len(shell_bvals))
    fa, fibre_dirs = _tensor_fa_and_direction(signal[valid], gtab)
    # The metric is kept as the float32 values an image would hold, compared in
    # float64.
    sdm = sdm[valid].astype(np.float64)
    signal = signal[valid]
    safe = _flat_to_mask(eroded, valid)
    LarmorLogger.info(
        f"Mask: {int(brain.sum())} -> {int(eroded.sum())} -> {int(safe.sum())} voxels"
    )

    finite_fa = np.isfinite(fa)
    crude_wm = finite_fa & (fa.astype(np.float64) > fa_threshold)
    crude_other = finite_fa & ~crude_wm
    crude_csf = _crude_csf(sdm, crude_other)
    crude_gm = crude_other & ~crude_csf

    wm_outliers = _wm_outliers(sdm, crude_wm)
    refined_wm = crude_wm & ~wm_outliers
    refined_gm = _refined_gm(sdm, crude_gm)
    refined_csf = _refined_csf(sdm, crude_csf, wm_outliers)
    LarmorLogger.info(
        f"Refined: {int(refined_wm.sum())} WM, {int(refined_gm.sum())} GM, "
        f"{int(refined_csf.sum())} CSF voxels"
    )

    csf_voxels = refined_csf & _top(
        np.where(refined_csf, sdm, 0), round(refined_csf.sum() * csf_percent / 100)
    )
    gm_centre = _median(sdm[refined_gm])
    gm_distance = np.where(refined_gm, np.abs(sdm - gm_centre) + 1, 0).astype(
        np.float32
    )
    gm_voxels = refined_gm & _bottom(
        gm_distance, round(refined_gm.sum() * gm_percent / 100)
    )
    n_shells = len(shell_bvals)
    csf = ZonalSHRF(
        bvals=shell_bvals,
        coefficients=_isotropic_response(signal[csf_voxels], rows, n_shells),
    )
    gm = ZonalSHRF(
        bvals=shell_bvals,
        coefficients=_isotropic_response(signal[gm_voxels], rows, n_shells),
    )

    n_wm = round(refined_wm.sum() * wm_percent / 100)
    wm_voxels = _single_fibre_voxels(
        signal, refined_wm, csf, gtab, shell_bvals, n_wm, threads
    )
    wm = ZonalSHRF(
        bvals=shell_bvals,
        coefficients=_anisotropic_response(
            signal[wm_voxels], fibre_dirs[wm_voxels], gtab, rows, n_shells
        ),
    )
    LarmorLogger.info(
        f"Response voxels: {int(wm_voxels.sum())} WM, {int(gm_voxels.sum())} GM, "
        f"{int(csf_voxels.sum())} CSF"
    )
    return TissueResponses(
        wm=wm,
        gm=gm,
        csf=csf,
        wm_voxels=_flat_to_mask(safe, wm_voxels),
        gm_voxels=_flat_to_mask(safe, gm_voxels),
        csf_voxels=_flat_to_mask(safe, csf_voxels),
    )


def _erode(brain: np.ndarray, erode: int) -> np.ndarray:
    """Erodes the brain mask `erode` times by one voxel and its 6 neighbours."""
    if erode <= 0:
        return brain
    # A voxel on the edge of the grid counts as outside.
    structure = generate_binary_structure(3, 1)
    eroded: np.ndarray = binary_erosion(
        brain, structure, iterations=erode, border_value=0
    )
    return eroded


def _wm_outliers(sdm: np.ndarray, crude_wm: np.ndarray) -> np.ndarray:
    """Returns the white matter voxels whose decay metric is far above the median.

    Returns:
        np.ndarray: The crude white matter voxels more than two robust standard
        deviations (from the median absolute deviation) above the median.
    """
    wm_median = _median(sdm[crude_wm])
    wm_mad = _median(np.abs(sdm[crude_wm] - wm_median).astype(np.float32))
    outliers: np.ndarray = crude_wm & (sdm > wm_median + _MAD_TO_SD * wm_mad * 2.0)
    return outliers


def _flat_to_mask(mask: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Returns the voxels of mask that `selected`, one entry per True voxel, keeps."""
    out = np.zeros(mask.shape, dtype=bool)
    out[mask] = selected
    return out


def _median(values: np.ndarray) -> float:
    """Returns the median of the values taken as float32."""
    return float(np.median(values.astype(np.float32)))


def _shell_means(signal: np.ndarray, rows: np.ndarray, n_shells: int) -> np.ndarray:
    """Returns the mean of each shell's volumes, negatives as 0, float32."""
    clipped = np.maximum(signal, 0.0)
    return np.column_stack(
        [clipped[:, rows == k].mean(axis=1) for k in range(n_shells)]
    ).astype(np.float32)


def _signal_decay_metric(
    signal: np.ndarray, rows: np.ndarray, n_shells: int
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the signal decay metric of each voxel, and which voxels have a valid one.

    The metric is the log of the mean b0 signal over the mean signal of each shell,
    averaged over the shells weighted by their number of volumes.

    Returns:
        The metric, capped at 10, float32 (n_voxels,), and whether every shell mean
        and the metric are finite and positive (n_voxels,).
    """
    means = _shell_means(signal, rows, n_shells)
    valid = (np.isfinite(means) & (means > 0)).all(axis=1)
    counts = np.array([(rows == k).sum() for k in range(1, n_shells)], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_shell = np.log(
            means[:, :1].astype(np.float64) / means[:, 1:].astype(np.float64)
        ).astype(np.float32)
        full = (per_shell.astype(np.float64) @ counts / counts.sum()).astype(np.float32)
    valid &= np.isfinite(full) & (full > 0)
    return np.minimum(full, np.float32(_MAX_SDM)), valid


def _tensor_fa_and_direction(
    signal: np.ndarray, gtab: GradientTable
) -> tuple[np.ndarray, np.ndarray]:
    """Fits a tensor in each voxel and returns its FA and principal eigenvector.

    The log signal is fitted by weighted least squares with the signal as weights,
    then twice more with the predicted signal; values below 1e-6 of a voxel's largest
    are raised to it first.

    Returns:
        FA, float32 (n_voxels,), and the unit eigenvectors, (n_voxels, 3).
    """
    g, b = gtab.bvecs, gtab.bvals
    design = np.column_stack(
        [
            -b * g[:, 0] ** 2,
            -b * g[:, 1] ** 2,
            -b * g[:, 2] ** 2,
            -2 * b * g[:, 0] * g[:, 1],
            -2 * b * g[:, 0] * g[:, 2],
            -2 * b * g[:, 1] * g[:, 2],
            np.ones(len(b)),
        ]
    )
    floor = 1e-6 * signal.max(axis=1, keepdims=True)
    raised = np.maximum(signal, floor)
    log_signal = np.log(raised)

    def weighted_fit(weights: np.ndarray) -> np.ndarray:
        squared = weights**2
        normal = np.einsum("mi,vm,mj->vij", design, squared, design)
        rhs = np.einsum("mi,vm,vm->vi", design, squared, log_signal)
        solution: np.ndarray = np.linalg.solve(normal, rhs[..., np.newaxis])[..., 0]
        return solution

    params = weighted_fit(raised)
    for _ in range(2):
        params = weighted_fit(np.exp(params @ design.T))
    xx, yy, zz, xy, xz, yz = params[:, :6].T
    tensors = np.stack(
        [
            np.stack([xx, xy, xz], axis=-1),
            np.stack([xy, yy, yz], axis=-1),
            np.stack([xz, yz, zz], axis=-1),
        ],
        axis=1,
    )
    values, vectors = np.linalg.eigh(tensors)
    mean = values.mean(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        fa = np.sqrt(1.5 * ((values - mean) ** 2).sum(axis=1) / (values**2).sum(axis=1))
    return fa.astype(np.float32), vectors[:, :, 2]


def _golden_section_minimum(
    cost: _ThresholdCost, low: float, start: float, high: float, tol: float = 0.01
) -> float:
    """Finds the minimum of cost on [low, high] by golden-section search from start."""
    ratio = 0.61803399
    x0, x3 = low, high
    if abs(high - start) > abs(start - low):
        x1, x2 = start, start + (1 - ratio) * (high - start)
    else:
        x1, x2 = start - (1 - ratio) * (start - low), start
    f1, f2 = cost(x1), cost(x2)
    while tol * (abs(x1) + abs(x2)) < abs(x3 - x0):
        if f2 < f1:
            x0, x1 = x1, x2
            x2 = ratio * x1 + (1 - ratio) * x3
            f1, f2 = f2, cost(x2)
        else:
            x3, x2 = x2, x1
            x1 = ratio * x2 + (1 - ratio) * x0
            f2, f1 = f1, cost(x1)
    return x1 if f1 < f2 else x2


class _ThresholdCost:
    """Gives a threshold's cost, which is minus the values' correlation with a mask.

    The mask holds the values above the threshold.
    """

    def __init__(self, values: np.ndarray) -> None:
        """Stores the values, their mean and their standard deviation."""
        self.values = values.astype(np.float64)
        self.mean = self.values.mean()
        self.std = np.sqrt(
            (np.sum(self.values**2) - self.values.sum() * self.mean) / len(values)
        )

    def __call__(self, threshold: float) -> float:
        """Returns the cost of a threshold."""
        above = self.values > threshold
        count = len(self.values)
        n_above = float(above.sum())
        covariance = self.values[above].sum() / count - n_above / count * self.mean
        mask_std = np.sqrt((n_above - n_above * n_above / count) / count)
        return float(-covariance / (self.std * mask_std))


def _optimal_threshold(values: np.ndarray) -> float:
    """Returns the threshold that best correlates the values with their mask."""
    values = values.astype(np.float32)
    low, high = float(values.min()), float(values.max())
    span = high - low
    return float(
        np.float32(
            _golden_section_minimum(
                _ThresholdCost(values),
                low + 0.001 * span,
                0.5 * (low + high),
                high - 0.001 * span,
            )
        )
    )


def _crude_csf(sdm: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Returns the voxels of `other` whose SDM is above the automatic threshold."""
    shifted = np.where(other, sdm - np.float32(_median(sdm[other])), 0).astype(
        np.float32
    )
    return other & (shifted >= _optimal_threshold(shifted[other]))


def _refined_gm(sdm: np.ndarray, crude_gm: np.ndarray) -> np.ndarray:
    """Returns grey matter without the voxels far from its median SDM."""
    centre = np.float32(_median(sdm[crude_gm]))
    high = crude_gm & (sdm > centre)
    low = crude_gm & ~high
    above = np.where(high, sdm - centre, 0).astype(np.float32)
    below = np.where(low, centre - sdm, 0).astype(np.float32)
    keep_high = high & (above < _optimal_threshold(above[high])) if high.any() else high
    keep_low = low & (below < _optimal_threshold(below[low])) if low.any() else low
    return keep_high | keep_low


def _refined_csf(
    sdm: np.ndarray, crude_csf: np.ndarray, wm_outliers: np.ndarray
) -> np.ndarray:
    """Returns CSF and the white matter outliers above its lowest SDM, thresholded."""
    lowest = np.float32(sdm[crude_csf].min())
    candidates = crude_csf | (wm_outliers & (sdm > lowest))
    shifted = np.where(candidates, sdm - lowest, 0).astype(np.float32)
    return candidates & (shifted >= _optimal_threshold(shifted[candidates]))


# The end of the error message when a tissue has too few voxels.
_TOO_FEW = "the mask holds too few voxels of a tissue to estimate its response."


def _top(values: np.ndarray, count: int) -> np.ndarray:
    """Returns the voxels at or above the count-th largest finite non-zero value."""
    data = values[np.isfinite(values) & (values != 0)]
    if not 0 < count <= len(data):
        raise ValueError(f"Cannot select {count} of {len(data)} voxels: {_TOO_FEW}")
    threshold = np.partition(data, len(data) - count)[len(data) - count]
    selected: np.ndarray = values >= threshold
    return selected


def _bottom(values: np.ndarray, count: int) -> np.ndarray:
    """Returns the voxels at or below the count-th smallest finite non-zero value."""
    data = values[np.isfinite(values) & (values != 0)]
    if not 0 < count <= len(data):
        raise ValueError(f"Cannot select {count} of {len(data)} voxels: {_TOO_FEW}")
    threshold = np.partition(data, count - 1)[count - 1]
    selected: np.ndarray = values <= threshold
    return selected


def _isotropic_response(
    signal: np.ndarray, rows: np.ndarray, n_shells: int
) -> np.ndarray:
    """Returns sqrt(4 pi) times the mean signal of each shell, (n_shells, 1)."""
    means = np.array([signal[:, rows == k].mean() for k in range(n_shells)])
    response: np.ndarray = np.sqrt(4 * np.pi) * means[:, np.newaxis]
    return response


def _zonal_basis(theta: np.ndarray, lmax: int) -> np.ndarray:
    """Returns Y_l0 at polar angles theta for l = 0, 2, ..., lmax, (n, lmax / 2 + 1)."""
    return np.column_stack(
        [
            np.sqrt((2 * degree + 1) / (4 * np.pi))
            * eval_legendre(degree, np.cos(theta))
            for degree in range(0, lmax + 1, 2)
        ]
    )


def _zonal_basis_derivative(theta: np.ndarray, lmax: int) -> np.ndarray:
    """Returns the derivative of Y_l0 in theta for l = 0, 2, ..., lmax."""
    # d P_l(cos t) / dt = -sin(t) P_l'(cos t), which is P_l^1(cos t) with the
    # Condon-Shortley phase that lpmv includes.
    return np.column_stack(
        [
            np.sqrt((2 * degree + 1) / (4 * np.pi)) * lpmv(1, degree, np.cos(theta))
            for degree in range(0, lmax + 1, 2)
        ]
    )


def _solve_normal_equations(
    normal: np.ndarray, rhs: np.ndarray, constraints: np.ndarray
) -> np.ndarray:
    """Returns the x minimising x^T N x - 2 r^T x with a small ridge, where G x >= 0."""
    # The ridge also keeps the factorisation below possible when the voxels' angles
    # leave some degree undetermined.
    regularised = normal + _RESPONSE_RIDGE * np.max(np.diag(normal)) * np.eye(
        len(normal)
    )
    # With N = L L^T, ||L^T x - L^-1 r||^2 = x^T N x - 2 r^T x + const.
    factor = np.linalg.cholesky(regularised)
    data = np.linalg.solve(factor, rhs)
    design = _core.LsiDesign(
        np.ascontiguousarray(factor.T), np.ascontiguousarray(constraints), 0.0
    )
    solutions, converged = _core.fit_lsi(
        np.ascontiguousarray(data[np.newaxis]), design, n_threads=1
    )
    if not converged[0]:
        LarmorLogger.warning(
            "The constrained fit of the white matter response stopped at its "
            "iteration limit."
        )
    solution: np.ndarray = solutions[0]
    return solution


def _anisotropic_response(
    signal: np.ndarray,
    fibre_dirs: np.ndarray,
    gtab: GradientTable,
    rows: np.ndarray,
    n_shells: int,
) -> np.ndarray:
    """Fits the white matter response to the voxels' signal.

    On each shell, the zonal coefficients fit every voxel's signal against the angle
    between its gradients and its fibre direction, subject to a non-negative response
    that rises from the fibre to the plane across it; the b0s give lmax 0.

    Returns:
        The zonal coefficients, shape (n_shells, 6).

    Raises:
        ValueError: If there is no voxel.
    """
    if len(signal) == 0:
        raise ValueError(
            "No single-fibre white matter voxel to estimate a response from."
        )
    n_columns = _WM_RESPONSE_LMAX // 2 + 1
    response = np.zeros((n_shells, n_columns))
    amplitude = _zonal_basis(_CONSTRAINT_ANGLES, _WM_RESPONSE_LMAX)
    slope = _zonal_basis_derivative(_CONSTRAINT_ANGLES, _WM_RESPONSE_LMAX)
    constraints = np.vstack([amplitude, slope])
    for k in range(n_shells):
        volumes = rows == k
        values = signal[:, volumes]
        if k == 0:
            response[0, 0] = np.sqrt(4 * np.pi) * values.mean()
            continue
        cosines = np.clip(np.abs(fibre_dirs @ gtab.bvecs[volumes].T), 0.0, 1.0)
        basis = _zonal_basis(np.arccos(cosines.ravel()), _WM_RESPONSE_LMAX)
        normal = basis.T @ basis
        rhs = basis.T @ values.ravel()
        response[k] = _solve_normal_equations(normal, rhs, constraints)
    return response


def _single_fibre_voxels(
    signal: np.ndarray,
    refined_wm: np.ndarray,
    csf: ZonalSHRF,
    gtab: GradientTable,
    shell_bvals: np.ndarray,
    count: int,
    threads: int,
) -> np.ndarray:
    """Returns the white matter voxels whose largest FOD peak holds most of the signal.

    A two-tissue fit (a generic white matter response and the CSF response) at
    lmax 2 over the refined white matter keeps the 2 * count voxels with the
    largest peak amplitude over the summed l = 0 coefficients; a fit at lmax 6 over
    those keeps the count best.

    Returns:
        Which voxels (entries of signal) are kept, (n_voxels,) bool.
    """
    level = np.sqrt(4 * np.pi) * _median(
        signal[refined_wm].mean(axis=1).astype(np.float32)
    )
    generic = np.tile([level, -level, level, -level], (len(shell_bvals), 1))
    generic[0, 1:] = 0.0
    generic_wm = ZonalSHRF(bvals=shell_bvals, coefficients=generic)
    selected = refined_wm
    for lmax, n_keep in ((2, 2 * count), (6, count)):
        design = prepare_msmt(
            [generic_wm, csf], [lmax, 0], gtab.bvals, gtab.bvecs, DIRS_300
        )
        # Selection only ranks the voxels, so one that stopped at the iteration limit
        # still counts with its last fit.
        coefficients, _ = fit_msmt(signal[selected], design, threads)
        n_fod = (lmax + 1) * (lmax + 2) // 2
        peaks = find_peaks(coefficients[:, :n_fod], num_peaks=1, n_jobs=threads)
        total = coefficients[:, 0] + coefficients[:, n_fod]
        metric = np.full(len(signal), 0.0, dtype=np.float32)
        metric[selected] = (peaks.amplitudes[:, 0] / total).astype(np.float32)
        selected = selected & _top(metric, n_keep)
    return selected
