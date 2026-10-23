"""Diffusion kurtosis imaging (DKI) fit and the metrics derived from it."""

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.special import elliprd, elliprf

from larmor._log import LarmorLogger
from larmor._threads import n_threads
from larmor.dwi.dti import compute_dti_metrics
from larmor.dwi.fitting import fit_log_signal
from larmor.dwi.inputs import (
    as_mask,
    check_dwi,
    estimate_s0,
    floor_bad_samples,
    mean_b0,
)
from larmor.dwi.mask import create_brain_mask, unmask
from larmor.types import GradientTable, Volume

# The fit methods: ordinary and weighted least squares.
FIT_METHODS = ("ols", "wls")

# The 15 unique elements of the kurtosis tensor as index quadruples (0 = x), in the
# order of DKIFit.kt.
KT_INDICES = (
    (0, 0, 0, 0),
    (1, 1, 1, 1),
    (2, 2, 2, 2),
    (0, 0, 0, 1),
    (0, 0, 0, 2),
    (1, 1, 1, 0),
    (1, 1, 1, 2),
    (2, 2, 2, 0),
    (2, 2, 2, 1),
    (0, 0, 1, 1),
    (0, 0, 2, 2),
    (1, 1, 2, 2),
    (0, 0, 1, 2),
    (1, 1, 0, 2),
    (2, 2, 0, 1),
)
# The number of distinct permutations of each quadruple in KT_INDICES.
KT_MULTIPLICITY = (1, 1, 1, 4, 4, 4, 4, 4, 4, 6, 6, 6, 12, 12, 12)
# Eigenvalue gap, relative to MD, below which MK and RK are extrapolated instead of
# computed in closed form.
DEGENERATE_GAP = 3e-4
# Constant C of the upper bound on the apparent kurtosis, K(n) <= C / (b_max D(n)).
KURTOSIS_BOUND_C = 3.0
# Number of directions on the half sphere along which the constraints are imposed.
N_CONSTRAINT_DIRS = 256
# Number of Gauss-Legendre nodes in cos(theta) of the kurtosis anisotropy quadrature.
KA_N_THETA = 16
# Number of equally spaced azimuths of the same quadrature.
KA_N_PHI = 32
# Voxels per chunk of the kurtosis anisotropy quadrature.
KA_CHUNK = 4096


@dataclass(frozen=True)
class DKIFit:
    """Result of a DKI fit, as maps on the image grid (zero outside the fitted mask).

    Each map has the spatial shape (X, Y, Z) of the DWI, with a last axis for dt,
    kt, pe and eigen_vals. Every map is float32, except npd. Diffusivities are in
    mm^2/s, the inverse of the units of the b-values; kurtosis values are
    dimensionless.

    The apparent kurtosis along a unit vector n is K(n) = MD^2 W(n) / D(n)^2, with
    D(n) = n^T D n and W(n) the quartic form of the kurtosis tensor.

    Attributes:
        dt: The 6 unique elements of the diffusion tensor in the order xx, xy,
            xz, yy, yz, zz, stored in world coordinates (RAS+).
        kt: Kurtosis tensor W, containing its 15 unique elements in the order
            W1111, W2222, W3333, W1112, W1113, W2221, W2223, W3331, W3332, W1122,
            W1133, W2233, W1123, W2213, W3312.
        s0: Fitted signal at b = 0, in the units of the DWI.
        fa: Fractional anisotropy, clipped to [0, 1].
        md: Mean diffusivity.
        ad: Axial diffusivity.
        rd: Radial diffusivity.
        pe: Principal eigenvector in world coordinates (RAS+); its sign is arbitrary.
        eigen_vals: Eigenvalues l1 >= l2 >= l3, as fitted (they can be
            negative).
        npd: True in the fitted voxels whose diffusion tensor is not positive
            definite (l3 <= 0), bool. The diffusion maps there are computed from
            the eigenvalues as fitted, and the kurtosis maps are 0.
        mk: Mean kurtosis, the mean of K(n) over the sphere.
        ak: Axial kurtosis, K(n) along the principal eigenvector.
        rk: Radial kurtosis, the mean of K(n) over the directions perpendicular
            to the principal eigenvector.
        ka: Kurtosis anisotropy, the standard deviation of K(n) over the sphere.
    """

    dt: np.ndarray
    kt: np.ndarray
    s0: np.ndarray
    fa: np.ndarray
    md: np.ndarray
    ad: np.ndarray
    rd: np.ndarray
    pe: np.ndarray
    eigen_vals: np.ndarray
    npd: np.ndarray
    mk: np.ndarray
    ak: np.ndarray
    rk: np.ndarray
    ka: np.ndarray


class DKIModel:
    """Diffusion kurtosis imaging (DKI) model, with a diffusion and a kurtosis tensor.

    DKI extends the diffusion tensor with a kurtosis tensor W that describes how far
    the diffusion departs from a Gaussian. It needs at least two non-zero shells.
    `fit` returns a `DKIFit` with the tensors and their maps (FA, MD, MK and
    others).

    References:
        Jensen JH, Helpern JA, Ramani A, Lu H, Kaczynski K. Diffusional kurtosis
        imaging: the quantification of non-Gaussian water diffusion by means of
        magnetic resonance imaging. Magn Reson Med 53 (2005) 1432-1440.
        Veraart J, Sijbers J, Sunaert S, Leemans A, Jeurissen B. Weighted linear
        least squares estimation of diffusion MRI parameters: strengths,
        limitations, and pitfalls. NeuroImage 81 (2013) 335-346.
    """

    def __init__(self, *, fit_method: str = "wls", constrained: bool = True) -> None:
        """Sets the fit method and whether the fit is constrained.

        Args:
            fit_method: "ols" for ordinary least squares, or "wls"
                for weighted least squares. Defaults to "wls".
            constrained: Whether the fit is constrained to
                physically plausible tensors (see `fit`). Defaults to True.

        Raises:
            ValueError: If the fit method is not one of these.
        """
        if fit_method not in FIT_METHODS:
            raise ValueError(
                f"Unknown DKI fit method {fit_method!r}: "
                f"use one of {', '.join(FIT_METHODS)}."
            )
        self.fit_method = fit_method
        self.constrained = constrained

    def fit(
        self,
        volume: Volume,
        gtab: GradientTable,
        mask: Volume | np.ndarray | None = None,
        *,
        n_jobs: int = -1,
    ) -> DKIFit:
        """Fits the diffusion and kurtosis tensors in every voxel of the mask.

        Before the fit, the data are cleaned so that every log is defined. Each
        voxel's S0 is estimated with `larmor.dwi.estimate_s0`. Where that estimate
        differs from the mean of the voxel's own b0s (because those b0s are
        implausible), it replaces the b0 measurements. Every diffusion-weighted
        measurement is then capped at S0. A voxel without a finite, positive S0 is
        not fitted and is 0 in every map, and a warning gives the number of such
        voxels. A measurement that is still not finite and positive is set to
        1e-3 S0. Without a b0 in the table nothing is replaced or capped, and such
        measurements are set to 1e-3 times the largest value of the data.

        The constrained fit (Tabesh et al., 2011) keeps the tensors physically
        plausible. It minimises the same (weighted) sum of squares subject to
        D(n) >= 0, K(n) >= 0 and K(n) <= 3 / (b_max D(n)), where b_max is the
        largest b-value of the table, and all shells are fitted. The constraints
        are imposed along 256 directions n spread over the half sphere, not along
        the acquired gradient directions. A voxel whose unconstrained solution
        breaks a constraint is fitted again as a least-squares problem with linear
        inequality constraints (Lawson and Hanson, 1974); where that finds no
        solution or does not converge, the voxel keeps its unconstrained fit. For
        "wls", the weights are those of the unconstrained weighted fit, with no
        further reweighting. The upper bound grows as D(n) goes to 0, so voxels
        with near-zero diffusivity (at the edge of the mask, for example) can give
        very large kurtosis values. Each voxel is fitted on its own, so the result
        does not depend on n_jobs.

        Args:
            volume: Diffusion-weighted volume, shape (X, Y, Z, N).
            gtab: Its gradient table, with at least two non-zero shells; the
                tensors come out in its frame, world coordinates.
            mask: Voxels to fit (nonzero inside), on the volume's grid. Defaults
                to None, which computes a brain mask from the mean b0 volume.
            n_jobs: Number of threads for the fit, at most the number of cores;
                -1 uses every core. Defaults to -1.

        Returns:
            DKIFit: The tensors and their maps, zero outside the mask.

        Raises:
            ValueError: If the table or the mask does not match the volume, the
                measurements cannot determine the tensors (fewer than two non-zero
                shells, or too few directions), if the mask is computed and
                there is no b0, or if n_jobs is 0 or below -1.

        References:
            Tabesh A, Jensen JH, Ardekani BA, Helpern JA. Estimation of tensors
            and tensor-derived measures in diffusional kurtosis imaging. Magn
            Reson Med 65 (2011) 823-836.
            Lawson CL, Hanson RJ. Solving Least Squares Problems. Prentice-Hall,
            1974 (chapter 23).
        """
        threads = n_threads(n_jobs)
        check_dwi(volume, gtab)
        design = _design_matrix(gtab)
        if mask is None:
            mask = create_brain_mask(mean_b0(volume, gtab))
        else:
            mask = as_mask(volume, mask)

        s = volume.as_float()[mask].astype(np.float64)
        fitted = mask.copy()
        is_b0 = gtab.b0_mask
        s0: np.ndarray | None = None
        if is_b0.any():
            s0 = estimate_s0(volume, gtab, mask)[mask]
            usable = np.isfinite(s0) & (s0 > 0)
            if not usable.all():
                LarmorLogger.warning(
                    "%d mask voxels have no usable S0 (no plausible b0 in them or "
                    "near them); they are not fitted and come out as 0.",
                    int((~usable).sum()),
                )
            fitted[mask] = usable
            s, s0 = s[usable], s0[usable]
            own = s[:, is_b0].mean(axis=1)
            repaired = ~np.isclose(s0, own, rtol=1e-9, atol=0.0)
            s[np.ix_(repaired, is_b0)] = s0[repaired, None]
            s[:, ~is_b0] = np.minimum(s[:, ~is_b0], s0[:, None])
        floor_bad_samples(s, s0)

        constraints = _constraint_matrix(gtab) if self.constrained else None
        params, broke = fit_log_signal(
            design, s, self.fit_method == "wls", constraints, threads
        )
        if self.constrained:
            LarmorLogger.info(
                "%d of %d voxels break a constraint.",
                int(broke.sum()),
                len(broke),
            )

        d = params[1:7]
        md = (d[0] + d[3] + d[5]) / 3
        # The fit gives MD^2 W; W is undefined where MD is not positive.
        w = np.divide(params[7:], md**2, out=np.zeros_like(params[7:]), where=md > 0)
        mk, ak, rk, ka = compute_kurtosis_metrics(d, w)
        dt, fa, md_map, ad, rd, pe, eigen_vals, npd = compute_dti_metrics(d, fitted)
        return DKIFit(
            dt=dt,
            kt=unmask(w.T, fitted),
            s0=unmask(np.exp(params[0]), fitted),
            fa=fa,
            md=md_map,
            ad=ad,
            rd=rd,
            pe=pe,
            eigen_vals=eigen_vals,
            npd=npd,
            mk=unmask(mk, fitted),
            ak=unmask(ak, fitted),
            rk=unmask(rk, fitted),
            ka=unmask(ka, fitted),
        )


def compute_kurtosis_metrics(
    d: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes MK, AK, RK and KA from diffusion and kurtosis tensors.

    The mean, axial and radial kurtosis use the closed-form expressions of Tabesh
    et al. (2011) in the eigenframe of D. Those are singular where two eigenvalues
    are equal, so where two eigenvalues are closer than 3e-4 MD, MK and RK are
    extrapolated from three slightly separated sets of eigenvalues. The kurtosis
    anisotropy is computed by quadrature over the sphere. The metrics are 0 where
    D is not positive definite.

    Args:
        d: Diffusion tensor elements xx, xy, xz, yy, yz, zz, shape (6, n).
        w: Kurtosis tensor elements in the order of `DKIFit.kt`, shape (15, n).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: MK, AK, RK and KA,
        each of shape (n,), float64.

    References:
        Tabesh A, Jensen JH, Ardekani BA, Helpern JA. Estimation of tensors and
        tensor-derived measures in diffusional kurtosis imaging. Magn Reson Med 65
        (2011) 823-836.
    """
    n = d.shape[1]
    mk, ak, rk, ka = (np.zeros(n) for _ in range(4))
    tensors = _tensor_matrices(d)
    finite = np.isfinite(tensors).all(axis=(1, 2)) & np.isfinite(w).all(axis=0)
    eigen_vals = np.zeros((n, 3))
    e_vecs = np.zeros((n, 3, 3))
    eigen_vals[finite], e_vecs[finite] = np.linalg.eigh(tensors[finite])
    eigen_vals, e_vecs = eigen_vals[:, ::-1], e_vecs[:, :, ::-1]
    ok = finite & (eigen_vals[:, 2] > 0)
    if not ok.any():
        return mk, ak, rk, ka

    lam = eigen_vals[ok]
    wt = _rotated_components(_kurtosis_tensor(w[:, ok]), e_vecs[ok])
    md = lam.mean(axis=1)
    ak[ok] = md**2 * wt[:, 0] / lam[:, 0] ** 2
    gap = np.minimum(lam[:, 0] - lam[:, 1], lam[:, 1] - lam[:, 2])
    near = gap < DEGENERATE_GAP * md
    mk_ok = np.empty(len(lam))
    rk_ok = np.empty(len(lam))
    mk_ok[~near] = _mean_kurtosis(lam[~near], wt[~near])
    rk_ok[~near] = _radial_kurtosis(lam[~near], wt[~near])
    if near.any():
        h = DEGENERATE_GAP * md[near, None] * np.array([2.0, 1.0, 0.0])
        ln, wn = lam[near], wt[near]
        mk_k = [_mean_kurtosis(ln + k * h, wn) for k in (1, 2, 3)]
        rk_k = [_radial_kurtosis(ln + k * h, wn) for k in (1, 2, 3)]
        mk_ok[near] = 3 * mk_k[0] - 3 * mk_k[1] + mk_k[2]
        rk_ok[near] = 3 * rk_k[0] - 3 * rk_k[1] + rk_k[2]
    mk[ok] = mk_ok
    rk[ok] = rk_ok
    ka[ok] = _kurtosis_anisotropy(d[:, ok], w[:, ok])
    return mk, ak, rk, ka


def _design_matrix(gtab: GradientTable) -> np.ndarray:
    """Builds the matrix that maps the DKI parameters to the log signal.

    Each row is one measurement. The columns are a constant (log S0), the six
    diffusion tensor elements and the fifteen elements of MD^2 W.

    Returns:
        np.ndarray: The design matrix, shape (N, 22).

    Raises:
        ValueError: If the table has fewer than two non-zero shells or cannot
            determine the parameters.
    """
    if gtab.count_shells() < 2:
        raise ValueError(
            "DKI needs at least two non-zero b-value shells; the gradient table has "
            f"{gtab.count_shells()}."
        )
    g = gtab.bvecs
    b2 = gtab.bvals**2 / 6
    kt_columns = [
        m * b2 * g[:, i] * g[:, j] * g[:, k] * g[:, q]
        for (i, j, k, q), m in zip(KT_INDICES, KT_MULTIPLICITY, strict=True)
    ]
    design = np.column_stack([np.ones(len(b2)), -gtab.b_matrix, *kt_columns])
    # b and b^2 columns differ in scale by orders of magnitude; the rank is that
    # of the columns scaled to unit norm.
    norms = np.linalg.norm(design, axis=0)
    scaled = design / np.where(norms > 0, norms, 1.0)
    if np.linalg.matrix_rank(scaled) < design.shape[1]:
        raise ValueError(
            "The b-values and b-vectors cannot determine the diffusion and kurtosis "
            "tensors: DKI needs a b0 or a third shell, and enough non-collinear "
            "directions (at least 15 on one of the shells)."
        )
    return design


def _constraint_matrix(gtab: GradientTable) -> np.ndarray:
    """Builds the matrix G of the constraints G p >= 0 on the 22 fit parameters.

    Its rows are the constraints D(n) >= 0, then MD^2 W(n) >= 0, then the upper
    bound on the kurtosis, each along every direction of the half sphere.

    Returns:
        np.ndarray: G, shape (3 N_CONSTRAINT_DIRS, 22).
    """
    dirs = _fibonacci_half_sphere(N_CONSTRAINT_DIRS)
    x, y, z = dirs.T
    d_rows = np.stack([x * x, 2 * x * y, 2 * x * z, y * y, 2 * y * z, z * z], axis=1)
    v_rows = np.stack(
        [
            m * dirs[:, i] * dirs[:, j] * dirs[:, k] * dirs[:, q]
            for (i, j, k, q), m in zip(KT_INDICES, KT_MULTIPLICITY, strict=True)
        ],
        axis=1,
    )
    zero_1 = np.zeros((len(dirs), 1))
    zero_d = np.zeros_like(d_rows)
    bound = KURTOSIS_BOUND_C / float(gtab.bvals.max())
    return np.vstack(
        [
            np.hstack([zero_1, d_rows, np.zeros_like(v_rows)]),
            np.hstack([zero_1, zero_d, v_rows]),
            np.hstack([zero_1, bound * d_rows, -v_rows]),
        ]
    )


def _fibonacci_half_sphere(n: int) -> np.ndarray:
    """Returns n unit vectors (n, 3) spread nearly evenly over the half sphere z > 0."""
    i = np.arange(n) + 0.5
    z = 1 - i / n
    phi = np.pi * (3 - np.sqrt(5)) * i
    r = np.sqrt(1 - z**2)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def _tensor_matrices(d: np.ndarray) -> np.ndarray:
    """Builds symmetric (n, 3, 3) matrices from the elements xx, xy, xz, yy, yz, zz."""
    tensors = np.empty((d.shape[1], 3, 3))
    for row, (i, j) in zip(
        d, [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)], strict=True
    ):
        tensors[:, i, j] = row
        tensors[:, j, i] = row
    return tensors


def _kurtosis_tensor(w: np.ndarray) -> np.ndarray:
    """Builds the full symmetric (n, 3, 3, 3, 3) tensors from their 15 elements."""
    full = np.empty((w.shape[1], 3, 3, 3, 3))
    for row, quad in zip(w, KT_INDICES, strict=True):
        for perm in set(itertools.permutations(quad)):
            full[:, perm[0], perm[1], perm[2], perm[3]] = row
    return full


def _rotated_components(full: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """Returns W1111, W2222, W3333, W1122, W1133, W2233 in the frame of vecs."""
    rot = np.einsum("nabcd,nai->nibcd", full, vecs)
    rot = np.einsum("nibcd,nbj->nijcd", rot, vecs)
    rot = np.einsum("nijcd,nck->nijkd", rot, vecs)
    rot = np.einsum("nijkd,ndl->nijkl", rot, vecs)
    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    return np.stack([rot[:, i, i, j, j] for i, j in pairs], axis=1)


def _f1(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Returns the weight of W1111 in MK for eigenvalues a, b, c (a != b, a != c)."""
    rf = elliprf(a / b, a / c, 1.0)
    rd = elliprd(a / b, a / c, 1.0)
    bc = np.sqrt(b * c)
    bracket = bc / a * rf + (3 * a**2 - a * b - a * c - b * c) / (3 * a * bc) * rd - 1
    f: np.ndarray = (a + b + c) ** 2 / (18 * (a - b) * (a - c)) * bracket
    return f


def _f2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Returns the weight of W2233 in MK for eigenvalues a, b, c (b != c)."""
    rf = elliprf(a / b, a / c, 1.0)
    rd = elliprd(a / b, a / c, 1.0)
    bc = np.sqrt(b * c)
    bracket = (b + c) / bc * rf + (2 * a - b - c) / (3 * bc) * rd - 2
    f: np.ndarray = (a + b + c) ** 2 / (3 * (b - c) ** 2) * bracket
    return f


def _mean_kurtosis(lam: np.ndarray, wt: np.ndarray) -> np.ndarray:
    """Returns MK from distinct eigenvalues (n, 3) and the rotated W components."""
    l1, l2, l3 = lam.T
    mk: np.ndarray = (
        _f1(l1, l2, l3) * wt[:, 0]
        + _f1(l2, l1, l3) * wt[:, 1]
        + _f1(l3, l2, l1) * wt[:, 2]
        + _f2(l1, l2, l3) * wt[:, 5]
        + _f2(l2, l1, l3) * wt[:, 4]
        + _f2(l3, l2, l1) * wt[:, 3]
    )
    return mk


def _radial_kurtosis(lam: np.ndarray, wt: np.ndarray) -> np.ndarray:
    """Returns RK from eigenvalues (n, 3) with l2 != l3 and rotated W components."""
    l1, l2, l3 = lam.T
    pre = (l1 + l2 + l3) ** 2 / (l2 - l3) ** 2
    g1_2 = pre / (18 * l2) * (2 * l2 + (l3**2 - 3 * l2 * l3) / np.sqrt(l2 * l3))
    g1_3 = pre / (18 * l3) * (2 * l3 + (l2**2 - 3 * l2 * l3) / np.sqrt(l2 * l3))
    g2 = pre / 3 * ((l2 + l3) / np.sqrt(l2 * l3) - 2)
    rk: np.ndarray = g1_2 * wt[:, 1] + g1_3 * wt[:, 2] + g2 * wt[:, 5]
    return rk


def _half_sphere_quadrature() -> tuple[np.ndarray, np.ndarray]:
    """Returns quadrature directions (M, 3) on the half sphere z > 0 and weights."""
    t, wt = np.polynomial.legendre.leggauss(2 * KA_N_THETA)
    upper = t > 0
    t, wt = t[upper], wt[upper]
    phi = 2 * np.pi * np.arange(KA_N_PHI) / KA_N_PHI
    sin_t = np.sqrt(1 - t**2)
    dirs = np.stack(
        [
            np.outer(sin_t, np.cos(phi)).ravel(),
            np.outer(sin_t, np.sin(phi)).ravel(),
            np.repeat(t, KA_N_PHI),
        ],
        axis=1,
    )
    weights = np.repeat(wt, KA_N_PHI)
    return dirs, weights / weights.sum()


def _kurtosis_anisotropy(d: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Returns the standard deviation of K(n) over the sphere (D positive definite)."""
    dirs, weights = _half_sphere_quadrature()
    x, y, z = dirs.T
    d_basis = np.stack([x * x, 2 * x * y, 2 * x * z, y * y, 2 * y * z, z * z], axis=1)
    w_basis = np.stack(
        [
            m * dirs[:, i] * dirs[:, j] * dirs[:, k] * dirs[:, q]
            for (i, j, k, q), m in zip(KT_INDICES, KT_MULTIPLICITY, strict=True)
        ],
        axis=1,
    )
    md2 = ((d[0] + d[3] + d[5]) / 3) ** 2
    ka = np.empty(d.shape[1])
    for start in range(0, d.shape[1], KA_CHUNK):
        part = slice(start, start + KA_CHUNK)
        k = (
            md2[part, None]
            * (w[:, part].T @ w_basis.T)
            / (d[:, part].T @ d_basis.T) ** 2
        )
        mean = k @ weights
        ka[part] = np.sqrt(np.maximum(((k - mean[:, None]) ** 2) @ weights, 0.0))
    return ka
