"""Real spherical harmonic bases and least-squares SH fits."""

import math

import numpy as np
import numpy.typing as npt
from scipy.special import lpmv

from larmor._log import LarmorLogger
from larmor.sd.sphere import get_theta_phi

# The square root of 2.
_SQRT2: np.float64 = np.sqrt(2)

# The SH basis conventions.
BASES = ("tournier", "descoteaux", "leemans")


def get_sh_indices(lmax: int, full: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Computes the degree l and order m of each coefficient up to lmax.

    Only even degrees are included unless full is True.

    Returns:
        The degrees and the orders, each shape (num_coeffs,), ordered by l and then
        m from -l to l. With even degrees only, num_coeffs = (lmax + 1)(lmax + 2) / 2;
        with all degrees, (lmax + 1)^2.
    """
    step = 1 if full else 2
    ls = []
    ms = []

    for ell in range(0, lmax + 1, step):
        for m in range(-ell, ell + 1):
            ls.append(ell)
            ms.append(m)

    return np.array(ls), np.array(ms)


def get_sh_coeffs(
    l: int,  # noqa: E741 (public parameter name, renaming it is an API change)
    m: int,
    theta: np.ndarray,
    phi: np.ndarray,
    basis: str = "tournier",
) -> np.ndarray:
    """Evaluates the basis function (l, m) at the given directions.

    The associated Legendre functions include the Condon-Shortley phase (-1)^m;
    with it, "tournier" matches MRtrix's basis.

    Args:
        l: Degree l.
        m: Order m, with -l <= m <= l.
        theta: Polar angle (from +z) of the directions, in radians.
        phi: Azimuthal angle (around z, from +x) of the directions, in radians.
        basis: Basis convention, "tournier" (MRtrix), "leemans" (ExploreDTI and
            MRIToolkit) or "descoteaux". Defaults to "tournier".

    Returns:
        np.ndarray: The value of Y_lm at each direction, same shape as theta.

    Raises:
        ValueError: If the basis is not a known one.
    """
    if basis == "leemans":
        # The tournier functions in a frame rotated by 90 degrees about z, evaluated
        # at (-y, x, z), so phi_leemans = phi - pi/2.
        return get_sh_coeffs(l, m, theta, phi - np.pi / 2, "tournier")

    abs_m = abs(m)

    norm: np.float64 = np.sqrt(
        ((2 * l + 1) / (4.0 * np.pi))
        * (math.factorial(l - abs_m) / math.factorial(l + abs_m))
    )
    legendre = lpmv(abs_m, l, np.cos(theta))
    sh_val = norm * legendre

    if basis == "tournier":
        if m < 0:
            return _SQRT2 * sh_val * np.sin(abs_m * phi)
        if m == 0:
            return sh_val
        if m > 0:
            return _SQRT2 * sh_val * np.cos(m * phi)

    if basis == "descoteaux":
        if m < 0:
            sign: int = (-1) ** abs_m
            return _SQRT2 * sign * sh_val * np.cos(abs_m * phi)
        if m == 0:
            return sh_val
        if m > 0:
            return _SQRT2 * sh_val * np.sin(m * phi)

    raise ValueError(f"Unsupported basis {basis}")


def build_basis_matrix(
    theta: np.ndarray,
    phi: np.ndarray,
    lmax: int,
    basis: str = "tournier",
    full: bool = False,
) -> np.ndarray:
    """Builds the SH basis matrix up to lmax at the given directions.

    Args:
        theta: Polar angle (from +z) of each direction, in radians, shape
            (n_pts,).
        phi: Azimuthal angle (around z, from +x) of each direction, in radians,
            shape (n_pts,).
        lmax: Maximum degree.
        basis: Basis convention, see get_sh_coeffs.
        full: Whether odd degrees are included too. Defaults to False.

    Returns:
        Basis matrix, shape (n_pts, num_coeffs), columns ordered as get_sh_indices.
    """
    l_vals, m_vals = get_sh_indices(lmax, full)
    n_pts = len(theta)
    num_coeffs = len(l_vals)

    B = np.zeros((n_pts, num_coeffs), dtype=np.float64)

    for j, (ell, m) in enumerate(zip(l_vals, m_vals, strict=True)):
        B[:, j] = get_sh_coeffs(ell, m, theta, phi, basis)

    return B


def fit_sh(
    amplitudes: npt.ArrayLike,
    directions: npt.ArrayLike,
    lmax: int,
    *,
    basis: str = "tournier",
    full: bool = False,
) -> npt.NDArray[np.float64]:
    """Fits SH coefficients to amplitudes sampled on unit directions.

    Args:
        amplitudes: (..., N) amplitudes, one per direction along the last axis.
        directions: (N, 3) unit vectors, in the frame of the coefficients.
        lmax: The maximum SH degree.
        basis: 'tournier' (MRtrix), 'descoteaux' or 'leemans'. Defaults to
            'tournier'.
        full: Whether to fit all degrees, (lmax + 1)^2 coefficients, for an
            asymmetric function, rather than the even degrees only,
            (lmax + 1)(lmax + 2) / 2. Defaults to False.

    Returns:
        The least-squares coefficients, (..., C).

    Raises:
        ValueError: If the amplitudes' last axis does not have one value per
            direction.
    """
    values = np.asarray(amplitudes, dtype=np.float64)
    dirs = np.asarray(directions, dtype=np.float64)
    if (
        values.ndim == 0
        or dirs.ndim != 2
        or dirs.shape[1] != 3
        or values.shape[-1] != len(dirs)
    ):
        raise ValueError(
            f"Expected amplitudes (..., N) for N directions (N, 3), got amplitudes "
            f"{values.shape} and directions {dirs.shape}."
        )
    theta, phi = get_theta_phi(dirs)
    matrix = build_basis_matrix(theta, phi, lmax, basis, full=full)
    flat = values.reshape(-1, values.shape[-1])
    coeffs: npt.NDArray[np.float64] = np.linalg.lstsq(matrix, flat.T)[0].T
    return coeffs.reshape(*values.shape[:-1], matrix.shape[1])


def _sign(k: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
    """Returns (-1)^k for integers k, negative k included."""
    return 1 - 2 * (k % 2)


def _tournier_perm_sign(
    lmax: int, full: bool, basis: str
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.int8]]:
    """Returns the map from tournier coefficients t to coefficients x in `basis`.

    x[i] = t[perm[i]] * sign[i]. Each perm swaps (l, m) with (l, -m), its own
    inverse, so t[j] = x[perm[j]] * sign[perm[j]].

    Returns:
        The permutation `perm` and the signs `sign`.

    Raises:
        ValueError: If the basis is not a known one.
    """
    _, ms = get_sh_indices(lmax, full)
    idx = np.arange(len(ms), dtype=np.intp)
    a = np.abs(ms)

    if basis == "tournier":
        perm = idx
        sign = np.ones(len(ms), dtype=np.int8)
    elif basis == "descoteaux":
        perm = idx - 2 * ms
        sign = np.where(ms < 0, _sign(a), 1).astype(np.int8)
    elif basis == "leemans":
        odd = a % 2 == 1
        perm = np.where(odd, idx - 2 * ms, idx)
        sign = np.where(
            odd, np.where(ms < 0, -1, 1) * _sign((a - 1) // 2), _sign(a // 2)
        ).astype(np.int8)
    else:
        raise ValueError(f"Invalid basis {basis}, expected one of {', '.join(BASES)}")

    return perm, sign


def convert_sh_basis(
    coeffs: npt.NDArray[np.floating],
    from_basis: str,
    to_basis: str,
) -> npt.NDArray[np.floating]:
    """Converts SH coefficients from one basis to another.

    The layout, symmetric or asymmetric, is read from the number of coefficients
    with `sh_layout`.

    Args:
        coeffs: Coefficients, shape (..., C).
        from_basis: Their basis, 'tournier', 'descoteaux' or 'leemans'.
        to_basis: The target basis, 'tournier', 'descoteaux' or 'leemans'.

    Returns:
        npt.NDArray[np.floating]: The coefficients in the target basis.

    Raises:
        ValueError: If a basis is unknown, or the number of coefficients is neither
            a symmetric nor a full SH layout.
    """
    if from_basis not in BASES:
        raise ValueError(
            f"Invalid basis {from_basis}, expected one of {', '.join(BASES)}"
        )

    if to_basis not in BASES:
        raise ValueError(
            f"Invalid basis {to_basis}, expected one of {', '.join(BASES)}"
        )

    C = coeffs.shape[-1]
    lmax, full = sh_layout(C)
    if full:
        LarmorLogger.info(
            f"Reading the SH coefficients as asymmetric: lmax {lmax}, "
            f"all degrees, {C} coefficients"
        )

    if from_basis == to_basis:
        return np.copy(coeffs)

    p_from, s_from = _tournier_perm_sign(lmax, full, from_basis)
    p_to, s_to = _tournier_perm_sign(lmax, full, to_basis)
    # The int8 signs keep the input's float type.
    tournier: npt.NDArray[np.floating] = coeffs[..., p_from] * s_from[p_from]
    converted: npt.NDArray[np.floating] = tournier[..., p_to] * s_to
    return converted


def sh_layout(num_coeffs: int) -> tuple[int, bool]:
    """Returns the lmax of num_coeffs SH coefficients and whether all degrees are kept.

    A symmetric FOD stores the even degrees, (lmax + 1)(lmax + 2) / 2 coefficients;
    an asymmetric one all degrees, (lmax + 1)^2. A count that fits both is read as
    symmetric, which makes 1 coefficient symmetric lmax 0.

    Args:
        num_coeffs: Number of coefficients.

    Returns:
        tuple[int, bool]: lmax, and whether all degrees are stored (asymmetric).

    Raises:
        ValueError: If the count fits neither layout.
    """
    no_layout = ValueError(
        f"{num_coeffs} SH coefficients fit no layout: a symmetric FOD has "
        f"(lmax + 1)(lmax + 2) / 2 (1, 6, 15, 28, 45, ...), an asymmetric one "
        f"(lmax + 1)^2 (4, 9, 16, 25, ...)."
    )
    if num_coeffs < 1:
        raise no_layout
    lmax = (-3 + math.isqrt(8 * num_coeffs + 1)) // 2

    if (lmax + 1) * (lmax + 2) // 2 == num_coeffs and lmax % 2 == 0:
        return lmax, False

    lmax = math.isqrt(num_coeffs) - 1

    if (lmax + 1) ** 2 == num_coeffs:
        return lmax, True

    raise no_layout
