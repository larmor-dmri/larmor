"""Response functions, which give the signal of a single tissue compartment."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from scipy.special import erf, erfi, eval_legendre

from larmor._log import LarmorLogger
from larmor.dwi.fitting import fit_log_signal
from larmor.dwi.inputs import as_mask, signal_chunks
from larmor.dwi.shells import SHELL_TOLERANCE
from larmor.sd.sphere import get_theta_phi
from larmor.types import GradientTable, Volume


@dataclass
class ResponseFunction(ABC):
    """Base class of the response functions, the signal model of each tissue.

    An anisotropic response is the signal of a single fibre, and an isotropic one
    that of a tissue such as grey matter or CSF.
    """

    # Number of spiral points that give an anisotropic response's directions in the
    # coarse loop of a multi-tissue fit; fewer when the data have fewer measurements.
    low_resolution_dirs: ClassVar[int] = 45

    @abstractmethod
    def signal(
        self, bvals: np.ndarray, bvecs: np.ndarray, dirs: np.ndarray
    ) -> np.ndarray:
        """Returns the signal of this response function for each measurement.

        Args:
            bvals: b-value of each measurement in s/mm^2, shape (n_meas,).
            bvecs: Unit gradient direction of each measurement, shape (n_meas, 3).
            dirs: Unit vectors to orient the response along, shape (n_dirs, 3), in
                the frame of bvecs.

        Returns:
            Signal with one column per orientation, shape (n_meas, n_cols), where
            n_cols is n_dirs for an anisotropic response and 1 for an isotropic one.
        """
        pass


@dataclass
class IsotropicRF(ResponseFunction):
    """Isotropic response, exp(-b D), the same along every direction.

    Attributes:
        diffusivity: Isotropic diffusivity D in mm^2/s.
    """

    diffusivity: float

    def signal(
        self,
        bvals: np.ndarray,
        bvecs: np.ndarray,
        dirs: np.ndarray,
    ) -> np.ndarray:
        """Returns the signal exp(-b D), shape (n_meas, 1).

        Args:
            bvals: b-value of each measurement in s/mm^2, shape (n_meas,).
            bvecs: Not used.
            dirs: Not used.

        Returns:
            The signal, shape (n_meas, 1).
        """
        s = np.exp(-bvals * self.diffusivity)
        return s[:, np.newaxis]


@dataclass
class AnisotropicRF(ResponseFunction):
    """Response of a single fibre, a diffusion tensor with an isotropic kurtosis term.

    The signal along gradient g at b-value b is
    exp(-b g^T D g + b^2 MD^2 K / 6) + offset, with D the tensor turned so that its
    first eigenvector points along the fibre, MD its mean eigenvalue and K the mean
    kurtosis.

    Attributes:
        eigen_vals: Tensor eigenvalues in mm^2/s; the first is along the fibre.
        mean_kurtosis: Mean kurtosis K, dimensionless.
        signal_offset: Constant added to the signal, which is otherwise 1 at b = 0.
            Defaults to 0.

    References:
        Jensen JH, Helpern JA, Ramani A, Lu H, Kaczynski K. Diffusional kurtosis
        imaging: the quantification of non-Gaussian water diffusion by means of
        magnetic resonance imaging. Magn Reson Med 53 (2005) 1432-1440.
        doi:10.1002/mrm.20508

        The kernel follows MRIToolkit's (MRTTrack).
    """

    eigen_vals: np.ndarray | list[float]
    mean_kurtosis: float
    signal_offset: float = field(default=0.0, kw_only=True)

    def signal(
        self, bvals: np.ndarray, bvecs: np.ndarray, dirs: np.ndarray
    ) -> np.ndarray:
        """Returns the signal with the fibre along each direction.

        Args:
            bvals: b-value of each measurement in s/mm^2, shape (n_meas,).
            bvecs: Unit gradient direction of each measurement, shape (n_meas, 3).
            dirs: Unit vectors to orient the response along, shape (n_dirs, 3), in the
                frame of bvecs.

        Returns:
            The signal with the fibre along each direction, shape (n_meas, n_dirs).
        """
        l1, l2, l3 = self.eigen_vals
        A = np.diag([l2, l3, l1])

        theta, phi = get_theta_phi(dirs)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cos_p, sin_p = np.cos(phi), np.sin(phi)
        zeros = np.zeros_like(theta)

        R = np.stack(
            [
                np.stack([cos_p * cos_t, -sin_p, cos_p * sin_t], axis=-1),
                np.stack([sin_p * cos_t, cos_p, sin_p * sin_t], axis=-1),
                np.stack([-sin_t, zeros, cos_t], axis=-1),
            ],
            axis=-2,
        )

        d_rot = R @ A @ np.swapaxes(R, -1, -2)

        gDg = np.einsum("mi, nij, mj -> mn", bvecs, d_rot, bvecs)
        md_squared = np.mean(self.eigen_vals) ** 2
        b_vals_col = bvals[:, np.newaxis]

        s: npt.NDArray[np.floating] = (
            np.exp(
                -b_vals_col * gDg
                + (1 / 6) * (b_vals_col**2) * md_squared * self.mean_kurtosis
            )
            + self.signal_offset
        )

        return s


# The white matter response is estimated from the voxels with FA above this.
WM_MIN_FA = 0.7
# The b-value, in s/mm^2, above which the data needs two shells for the kurtosis to
# be fitted; with fewer, the estimated kurtosis is 0.
KURTOSIS_MIN_B = 500.0
# Voxel kurtosis values outside this range are left out of the mean.
KURTOSIS_RANGE = (0.0, 4.0)


def estimate_wm_rf(
    volume: Volume, gtab: GradientTable, mask: Volume | np.ndarray
) -> AnisotropicRF:
    """Estimates the white matter response from the voxels with FA > 0.7.

    In every voxel of the mask whose signal is positive and finite, a tensor is
    fitted by iterated weighted least squares, with an isotropic kurtosis term when
    the data has at least two shells above b = 500. The response's eigenvalues are the
    medians over the voxels with FA > 0.7, the second and third then averaged so
    that the response is axially symmetric. Its kurtosis is the mean of the voxel
    kurtosis values that lie in (0, 4).

    Args:
        volume: Diffusion-weighted volume, shape (X, Y, Z, N).
        gtab: Its gradient table.
        mask: Voxels to consider (nonzero inside), on the volume's grid.

    Returns:
        AnisotropicRF: Eigenvalues [l1, l23, l23] in mm^2/s and the mean
        kurtosis; the kurtosis is 0 with fewer than two shells above b = 500.

    Raises:
        ValueError: If the mask does not match the volume, or no usable voxel in it
            has FA > 0.7.

    References:
        Adapted from MRIToolkit's Eigenval_IsoK_WM_FromData.
    """
    mask = as_mask(volume, mask)
    fit_kurtosis = gtab.count_shells(above=KURTOSIS_MIN_B) >= 2

    columns = [np.ones(len(gtab.bvals)), -gtab.b_matrix]
    if fit_kurtosis:
        columns.append((gtab.bvals * 1e-3) ** 2 / 6)
    design = np.column_stack(columns)

    fitted = []
    for _, s in signal_chunks(volume, np.nonzero(mask)):
        usable = np.isfinite(s).all(axis=1) & (s > 0).all(axis=1)
        fitted.append(fit_log_signal(design, s[usable], weighted=True)[0])
    params = (
        np.concatenate(fitted, axis=1) if fitted else np.empty((design.shape[1], 0))
    )

    e_vals = _tensor_eigenvalues(params[1:7])
    fa = _fractional_anisotropy(e_vals)
    selected = fa > WM_MIN_FA
    if not selected.any():
        raise ValueError(
            f"No usable voxel in the mask has FA > {WM_MIN_FA} to estimate the white "
            "matter response from."
        )

    medians = np.median(e_vals[selected], axis=0)
    l1, l23 = float(medians[0]), float(np.mean(medians[1:]))

    kurtosis = 0.0
    if fit_kurtosis:
        k_voxels = 1e-6 * params[7] / np.mean(e_vals, axis=1) ** 2
        low, high = KURTOSIS_RANGE
        k_kept = k_voxels[selected & (k_voxels > low) & (k_voxels < high)]
        if len(k_kept) > 0:
            kurtosis = float(np.mean(k_kept))
        else:
            LarmorLogger.warning(
                "No voxel with FA > %s has a kurtosis in (%s, %s); the white matter "
                "response gets kurtosis 0.",
                WM_MIN_FA,
                low,
                high,
            )

    LarmorLogger.info(
        f"Estimated WM eigenvalues: [{l1:.2e}, {l23:.2e}, {l23:.2e}], "
        f"kurtosis {kurtosis:.3f}, from {int(selected.sum())} voxels."
    )
    return AnisotropicRF(eigen_vals=[l1, l23, l23], mean_kurtosis=kurtosis)


def _tensor_eigenvalues(d: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Returns the eigenvalues of tensors given as their 6 elements, largest first.

    Args:
        d: Elements xx, xy, xz, yy, yz, zz of each tensor, shape (6, n).

    Returns:
        np.ndarray: Eigenvalues, shape (n, 3), negative ones kept.
    """
    tensors = np.empty((d.shape[1], 3, 3))
    for row, (i, j) in zip(
        d, [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)], strict=True
    ):
        tensors[:, i, j] = row
        tensors[:, j, i] = row
    return np.linalg.eigvalsh(tensors)[:, ::-1]


def _fractional_anisotropy(e_vals: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Returns the FA of each set of eigenvalues, (n, 3) to (n,); 0 where all are 0."""
    mean = e_vals.mean(axis=1, keepdims=True)
    num = np.sqrt(1.5 * np.sum((e_vals - mean) ** 2, axis=1))
    den = np.sqrt(np.sum(e_vals**2, axis=1))
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def build_response_functions(
    volume: Volume,
    gtab: GradientTable,
    mask: Volume | np.ndarray,
    *,
    wm_eigen_vals: Sequence[float] | None = None,
    mean_kurtosis: float | None = None,
    gm_diffusivity: float | None = None,
    csf_diffusivity: float | None = None,
    gm_noddi: bool = False,
) -> list[ResponseFunction]:
    """Builds the response functions of an FOD fit with `SDModel`.

    The white matter response is either built from given tensor eigenvalues or
    estimated from the data with `estimate_wm_rf`. Grey matter is added when its
    diffusivity is given or gm_noddi is set, and CSF when its diffusivity is given.
    The responses come in the order white matter, grey matter, CSF.

    Args:
        volume: Diffusion-weighted volume, shape (X, Y, Z, N).
        gtab: Its gradient table.
        mask: Voxels used to estimate the white matter response (nonzero inside),
            on the volume's grid. Only used when wm_eigen_vals is None.
        wm_eigen_vals: White matter tensor eigenvalues in mm^2/s, the first along
            the fibre. Defaults to None, which estimates the response from the
            data with `estimate_wm_rf`.
        mean_kurtosis: Mean kurtosis of the white matter response. Defaults to
            None, which takes the estimated kurtosis when the response is
            estimated, and 0 when wm_eigen_vals is given; a number replaces either.
        gm_diffusivity: Grey matter isotropic diffusivity in mm^2/s. Defaults to
            None, which leaves grey matter out.
        csf_diffusivity: CSF isotropic diffusivity in mm^2/s. Defaults to None,
            which leaves CSF out.
        gm_noddi: Whether to model grey matter with the anisotropic NODDI response
            (`NoddiRF` with its defaults) instead of an isotropic one, as the mFOD
            method does. Defaults to False.

    Returns:
        list[ResponseFunction]: White matter, grey matter and CSF, in that order,
        without the ones left out. With any isotropic response, or NODDI grey
        matter, the fit is multi-tissue.

    Raises:
        ValueError: If the white matter response is estimated and no usable voxel in
            the mask has FA > 0.7, or if both gm_diffusivity and gm_noddi are given.
    """
    if gm_noddi and gm_diffusivity is not None:
        raise ValueError("Give either a grey matter diffusivity or gm_noddi, not both.")
    if wm_eigen_vals is None:
        wm = estimate_wm_rf(volume, gtab, mask)
        if mean_kurtosis is not None:
            wm.mean_kurtosis = mean_kurtosis
    else:
        kurtosis = 0.0 if mean_kurtosis is None else mean_kurtosis
        wm = AnisotropicRF(eigen_vals=list(wm_eigen_vals), mean_kurtosis=kurtosis)

    rfs: list[ResponseFunction] = [wm]
    if gm_noddi:
        rfs.append(NoddiRF())
    isotropic = [d for d in (gm_diffusivity, csf_diffusivity) if d is not None]
    return [*rfs, *(IsotropicRF(diffusivity=d) for d in isotropic)]


# Coefficients of the polynomials in log(kappa / 30) that give the Watson SH
# coefficients of degrees 2..12 for kappa > 30, one row per degree, constant first.
# fmt: off
_WATSON_LARGE_KAPPA_FIT = np.array([
    [7.52308, 0.411538, -0.214588, 0.0784091, -0.023981, 0.00731537, -0.0026467],
    [8.93718, 1.62147, -0.733421, 0.191568, -0.0202906, -0.00779095, 0.00574847],
    [8.87905, 3.35689, -1.15935, 0.0673053, 0.121857, -0.066642, 0.0180215],
    [7.84352, 5.03178, -1.0193, -0.426362, 0.328816, -0.0688176, -0.0229398],
    [6.30113, 6.09914, -0.16088, -1.05578, 0.338069, 0.0937157, -0.106935],
    [4.65678, 6.30069, 1.13754, -1.38393, -0.0134758, 0.331686, -0.105954],
])
# fmt: on


def _watson_sh_coefficients(kappa: float) -> npt.NDArray[np.float64]:
    """Returns the zonal SH coefficients of a Watson distribution, degrees 0 to 12.

    They are computed in closed form for 0.1 < kappa <= 30, by a power series in
    kappa below that and by fitted polynomials in log(kappa) above.

    Returns:
        The coefficients of degrees 0, 2, ..., 12, shape (7,).
    """
    coefficients = np.zeros(7)
    sqrt_pi = np.sqrt(np.pi)
    coefficients[0] = 2 * sqrt_pi
    if kappa <= 0.1:
        coefficients[1] = (4 / 3 * kappa + 8 / 63 * kappa**2) * np.sqrt(np.pi / 5)
        coefficients[2] = (8 / 21 * kappa**2 + 32 / 693 * kappa**3) * 0.2 * sqrt_pi
        coefficients[3] = (16 / 693 * kappa**3 + 32 / 10395 * kappa**4) * np.sqrt(
            np.pi / 13
        )
        coefficients[4] = 32 / 19305 * kappa**4 * np.sqrt(np.pi / 17)
        coefficients[5] = 64 * np.sqrt(np.pi / 21) * kappa**5 / 692835
        coefficients[6] = 128 * sqrt_pi * kappa**6 / 152108775
    elif kappa <= 30:
        kappa_pow = [kappa**power for power in range(7)]  # kappa_pow[n] = kappa^n
        sqrt_kappa = np.sqrt(kappa)
        # root_kappa_pow[n] = sqrt(kappa) * kappa^n
        root_kappa_pow = [sqrt_kappa * kappa**power for power in range(7)]
        erfi_value = erfi(sqrt_kappa)
        exp_kappa = np.exp(kappa)
        dawson = 0.5 * sqrt_pi * erfi_value / exp_kappa

        coefficients[1] = (
            np.sqrt(5)
            * (3 * root_kappa_pow[0] - (3 + 2 * kappa_pow[1]) * dawson)
            * exp_kappa
            / erfi_value
            / kappa_pow[1]
        )
        coefficients[2] = (
            0.375
            * (
                (105 + 60 * kappa_pow[1] + 12 * kappa_pow[2]) * dawson
                - 105 * root_kappa_pow[0]
                + 10 * root_kappa_pow[1]
            )
            * exp_kappa
            / kappa_pow[2]
            / erfi_value
        )
        coefficients[3] = (
            (
                (-3465 - 1890 * kappa_pow[1] - 420 * kappa_pow[2] - 40 * kappa_pow[3])
                * dawson
                + 3465 * root_kappa_pow[0]
                - 420 * root_kappa_pow[1]
                + 84 * root_kappa_pow[2]
            )
            * np.sqrt(13 * np.pi)
            / 64
            / kappa_pow[3]
            / dawson
        )
        coefficients[4] = (
            np.sqrt(17)
            * (
                (
                    675675
                    + 360360 * kappa_pow[1]
                    + 83160 * kappa_pow[2]
                    + 10080 * kappa_pow[3]
                    + 560 * kappa_pow[4]
                )
                * dawson
                - 675675 * root_kappa_pow[0]
                + 90090 * root_kappa_pow[1]
                - 23100 * root_kappa_pow[2]
                + 744 * root_kappa_pow[3]
            )
            * exp_kappa
            / 512
            / kappa_pow[4]
            / erfi_value
        )
        coefficients[5] = (
            np.sqrt(21 * np.pi)
            * (
                (
                    -43648605
                    - 22972950 * kappa_pow[1]
                    - 5405400 * kappa_pow[2]
                    - 720720 * kappa_pow[3]
                    - 55440 * kappa_pow[4]
                    - 2016 * kappa_pow[5]
                )
                * dawson
                + 43648605 * root_kappa_pow[0]
                - 6126120 * root_kappa_pow[1]
                + 1729728 * root_kappa_pow[2]
                - 82368 * root_kappa_pow[3]
                + 5104 * root_kappa_pow[4]
            )
            / 4096
            / kappa_pow[5]
            / dawson
        )
        coefficients[6] = (
            5
            * (
                (
                    7027425405
                    + 3666482820 * kappa_pow[1]
                    + 872972100 * kappa_pow[2]
                    + 122522400 * kappa_pow[3]
                    + 10810800 * kappa_pow[4]
                    + 576576 * kappa_pow[5]
                    + 14784 * kappa_pow[6]
                )
                * dawson
                - 7027425405 * root_kappa_pow[0]
                + 1018467450 * root_kappa_pow[1]
                - 302630328 * root_kappa_pow[2]
                + 17153136 * root_kappa_pow[3]
                - 1553552 * root_kappa_pow[4]
                + 25376 * root_kappa_pow[5]
            )
            * exp_kappa
            / 16384
            / kappa_pow[6]
            / erfi_value
        )
    else:
        log_ratio = np.log(kappa) - np.log(30)
        powers = log_ratio ** np.arange(7)
        coefficients[1:] = _WATSON_LARGE_KAPPA_FIT @ powers
    return coefficients


def _legendre_gaussian_integrals(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Computes the integrals of exp(-x t^2) P_2l(t) over t in [-1, 1], for l = 0..6.

    They are computed in closed form, by a recursion on the integrals of
    exp(-x t^2) t^2j, and by a power series in x for x <= 0.05.

    Args:
        x: Values of x >= 0, shape (n,).

    Returns:
        The integrals, shape (n, 7), column l for P_2l.
    """
    integrals = np.zeros((len(x), 7))
    exact = x > 0.05

    x_exact = x[exact]
    moments = np.zeros((len(x_exact), 7))  # moments[:, j] = I_j
    sqrt_x = np.sqrt(x_exact)
    moments[:, 0] = np.sqrt(np.pi) * erf(sqrt_x) / sqrt_x
    for j in range(1, 7):
        moments[:, j] = (-np.exp(-x_exact) + (j - 0.5) * moments[:, j - 1]) / x_exact
    integrals[exact] = moments @ _EVEN_LEGENDRE_POWERS.T

    x_small = x[~exact]
    x_pow = [x_small**power for power in range(7)]  # x_pow[n] = x^n
    integrals[~exact, 0] = (
        2 - 2 * x_pow[1] / 3 + x_pow[2] / 5 - x_pow[3] / 21 + x_pow[4] / 108
    )
    integrals[~exact, 1] = (
        -4 * x_pow[1] / 15 + 4 * x_pow[2] / 35 - 2 * x_pow[3] / 63 + 2 * x_pow[4] / 297
    )
    integrals[~exact, 2] = 8 * x_pow[2] / 315 - 8 * x_pow[3] / 693 + 4 * x_pow[4] / 1287
    integrals[~exact, 3] = -16 * x_pow[3] / 9009 + 16 * x_pow[4] / 19305
    integrals[~exact, 4] = 32 * x_pow[4] / 328185
    integrals[~exact, 5] = -64 * x_pow[5] / 14549535
    integrals[~exact, 6] = 128 * x_pow[6] / 760543875
    return integrals


def _even_legendre_powers() -> npt.NDArray[np.float64]:
    """Returns the coefficients of t^0, t^2, ..., t^12 in P_2l, one row per l."""
    rows = np.zeros((7, 7))
    for degree in range(7):
        unit = np.zeros(2 * degree + 1)
        unit[-1] = 1.0
        monomial = np.polynomial.legendre.leg2poly(unit)
        rows[degree, : degree + 1] = monomial[::2]
    return rows


_EVEN_LEGENDRE_POWERS = _even_legendre_powers()


def _watson_hindered_diffusivities(
    parallel: float, perpendicular: float, kappa: float
) -> tuple[float, float]:
    """Averages a tensor's diffusivities over axes spread by a Watson distribution.

    Args:
        parallel: The diffusivity along the tensor's axis.
        perpendicular: The diffusivity across it.
        kappa: The concentration of the Watson distribution.

    Returns:
        The averaged parallel and perpendicular diffusivities.
    """
    difference = parallel - perpendicular
    if kappa < 1e-5:
        mean = (parallel + 2 * perpendicular) / 3
        return (
            mean + 4 * difference * kappa / 45 + 8 * difference * kappa**2 / 945,
            mean - 2 * difference * kappa / 45 - 4 * difference * kappa**2 / 945,
        )
    sqrt_kappa = np.sqrt(kappa)
    dawson = 0.5 * np.exp(-kappa) * np.sqrt(np.pi) * erfi(sqrt_kappa)
    factor = sqrt_kappa / dawson
    averaged_parallel = (
        -difference + 2 * perpendicular * kappa + difference * factor
    ) / (2 * kappa)
    averaged_perpendicular = (
        difference + 2 * (parallel + perpendicular) * kappa - difference * factor
    ) / (4 * kappa)
    return float(averaged_parallel), float(averaged_perpendicular)


@dataclass(kw_only=True)
class NoddiRF(ResponseFunction):
    """Response of the NODDI model of dispersed neurites and free water.

    The neurites are sticks with diffusivity d along them and none across, their
    orientations spread around the response's axis by a Watson distribution of
    concentration kappa. The space around them is a tensor with the same parallel
    diffusivity and d (1 - f_in) across, averaged over the same distribution. The
    signal is accurate to about 1e-7 or better for kappa between 0.3 and 30 and for
    b d = 0 or above 0.2; near kappa 0.1 and b d 0.05, and above kappa 30, the
    high-degree terms lose precision.

    Attributes:
        intra_fraction: Volume fraction of the neurites, f_in.
        parallel_diffusivity: Diffusivity d along the neurites, in mm^2/s.
        concentration: Watson concentration kappa; larger is less dispersed.
        iso_fraction: Volume fraction of free water, f_iso.
        iso_diffusivity: Diffusivity of free water d_iso, in mm^2/s.
        s0: Signal at b = 0.

    References:
        Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC. NODDI: practical in
        vivo neurite orientation dispersion and density imaging of the human brain.
        NeuroImage 61 (2012) 1000-1016. doi:10.1016/j.neuroimage.2012.03.072

        Follows the NODDI toolbox (SynthMeasWatsonSHStickTortIsoV_B0), as MRIToolkit's
        mFOD uses it for grey matter.
    """

    # Number of spiral points of the coarse directions, more than for a sharp response.
    low_resolution_dirs: ClassVar[int] = 90

    intra_fraction: float = 0.4
    parallel_diffusivity: float = 1.7e-3
    concentration: float = 1.0
    iso_fraction: float = 0.0
    iso_diffusivity: float = 3e-3
    s0: float = 1.0

    def signal(
        self, bvals: np.ndarray, bvecs: np.ndarray, dirs: np.ndarray
    ) -> np.ndarray:
        """Returns the signal with the axis along each direction.

        Args:
            bvals: b-value of each measurement in s/mm^2, shape (n_meas,).
            bvecs: Unit gradient direction of each measurement, shape (n_meas, 3).
            dirs: Unit vectors to orient the response along, shape (n_dirs, 3), in the
                frame of bvecs.

        Returns:
            The signal with the axis along each direction, shape (n_meas, n_dirs).
        """
        cosines = np.clip(bvecs @ dirs.T, -1.0, 1.0)
        b_column = bvals[:, np.newaxis]

        integrals = _legendre_gaussian_integrals(bvals * self.parallel_diffusivity)
        weights = integrals * _watson_sh_coefficients(self.concentration)
        intra = np.zeros(cosines.shape)
        for degree in range(7):
            harmonic = np.sqrt((4 * degree + 1) / (4 * np.pi)) * eval_legendre(
                2 * degree, cosines
            )
            intra += weights[:, degree, np.newaxis] * harmonic
        intra *= 0.5
        _floor_non_positive(intra)

        tortuous = self.parallel_diffusivity * (1 - self.intra_fraction)
        averaged_parallel, averaged_perpendicular = _watson_hindered_diffusivities(
            self.parallel_diffusivity, tortuous, self.concentration
        )
        extra = np.exp(
            -b_column
            * (
                (averaged_parallel - averaged_perpendicular) * cosines**2
                + averaged_perpendicular
            )
        )

        neurite_tissue = (1 - self.intra_fraction) * extra + self.intra_fraction * intra
        free_water = np.exp(-b_column * self.iso_diffusivity)
        s: npt.NDArray[np.floating] = self.s0 * (
            (1 - self.iso_fraction) * neurite_tissue + self.iso_fraction * free_water
        )
        return s


def _floor_non_positive(intra: npt.NDArray[np.float64]) -> None:
    """Replaces values <= 0 in each column by a tenth of its smallest positive value.

    The truncated series for the dispersed sticks can dip to or below zero at large b
    and wide dispersion, where the true signal is small but positive.
    """
    for column in intra.T:
        bad = column <= 0
        if bad.any() and (~bad).any():
            column[bad] = 0.1 * column[~bad].min()


@dataclass(kw_only=True)
class ZonalSHRF(ResponseFunction):
    """Response given by its zonal spherical harmonic coefficients on each shell.

    On a shell of b-value b, the signal along a gradient at angle t to the response's
    axis is sum_l c_l(b) sqrt((2l + 1) / (4 pi)) P_l(cos t) over the even degrees l,
    where c_l are the m = 0 coefficients of the response in the tournier basis. This
    is the format MRtrix3's dwi2response writes, in the units of the signal (not
    divided by b0). A measurement takes the coefficients of the shell nearest its
    b-value.

    Attributes:
        bvals: b-value of each shell in s/mm^2, shape (n_shells,).
        coefficients: Coefficients of l = 0, 2, 4, ... on each shell, one row per
            shell, shape (n_shells, n_degrees).

    References:
        Tournier JD, Smith R, Raffelt D, et al. MRtrix3: A fast, flexible and open
        software framework for medical image processing and visualisation. NeuroImage
        202 (2019) 116137. doi:10.1016/j.neuroimage.2019.116137
    """

    bvals: np.ndarray
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        """Checks the shapes and stores the arrays as float64.

        Raises:
            ValueError: If coefficients is not one row per b-value, or holds a
                non-finite number.
        """
        self.bvals = np.atleast_1d(np.asarray(self.bvals, dtype=np.float64))
        self.coefficients = np.atleast_2d(
            np.asarray(self.coefficients, dtype=np.float64)
        )
        if self.bvals.ndim != 1 or self.coefficients.shape[0] != len(self.bvals):
            raise ValueError(
                f"Expected one row of coefficients per b-value, got {len(self.bvals)} "
                f"b-values and coefficients of shape {self.coefficients.shape}."
            )
        if not np.isfinite(self.coefficients).all():
            raise ValueError("The response coefficients must be finite numbers.")

    @property
    def lmax(self) -> int:
        """The highest SH degree of the coefficients."""
        return 2 * (int(self.coefficients.shape[1]) - 1)

    def shell_coefficients(self, bval: float) -> np.ndarray:
        """Returns the coefficients of the shell nearest a b-value.

        Args:
            bval: The b-value in s/mm^2.

        Returns:
            np.ndarray: The coefficients of l = 0, 2, ..., shape (n_degrees,).

        Raises:
            ValueError: If no shell is within 100 s/mm^2 of the b-value.
        """
        distances = np.abs(self.bvals - bval)
        nearest = int(np.argmin(distances))
        if distances[nearest] >= SHELL_TOLERANCE:
            shells = ", ".join(f"{b:g}" for b in self.bvals)
            raise ValueError(
                f"The response has no shell near b = {bval:g} s/mm^2; its shells are "
                f"{shells}."
            )
        coefficients: np.ndarray = self.coefficients[nearest]
        return coefficients

    def signal(
        self, bvals: np.ndarray, bvecs: np.ndarray, dirs: np.ndarray
    ) -> np.ndarray:
        """Returns the signal with the response's axis along each direction.

        Args:
            bvals: b-value of each measurement in s/mm^2, shape (n_meas,).
            bvecs: Unit gradient direction of each measurement, shape (n_meas, 3).
            dirs: Unit vectors to orient the response along, shape (n_dirs, 3), in the
                frame of bvecs.

        Returns:
            The signal, shape (n_meas, n_dirs).

        Raises:
            ValueError: If a measurement has no shell of the response near its
                b-value.
        """
        cosines = np.clip(bvecs @ dirs.T, -1.0, 1.0)
        degrees = np.arange(0, self.lmax + 1, 2)
        per_meas = np.array([self.shell_coefficients(b) for b in bvals])
        weights = per_meas * np.sqrt((2 * degrees + 1) / (4 * np.pi))
        s = np.zeros(cosines.shape)
        for column, degree in enumerate(degrees):
            s += weights[:, column, np.newaxis] * eval_legendre(degree, cosines)
        return s
