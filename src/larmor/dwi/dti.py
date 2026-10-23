"""Diffusion tensor imaging (DTI) fit and the metrics derived from it."""

from dataclasses import dataclass

import numpy as np

from larmor._log import LarmorLogger
from larmor._threads import n_threads
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


@dataclass(frozen=True)
class DTIFit:
    """Result of a DTI fit, as maps on the image grid (zero outside the fitted mask).

    Each map has the spatial shape (X, Y, Z) of the DWI, with a last axis for dt,
    pe and eigen_vals. Every map is float32, except npd. Diffusivities are in
    mm^2/s, the inverse of the units of the b-values.

    Attributes:
        dt: The 6 unique elements of the diffusion tensor in the order xx, xy,
            xz, yy, yz, zz, stored in world coordinates (RAS+).
        fa: Fractional anisotropy, clipped to [0, 1].
        md: Mean diffusivity.
        ad: Axial diffusivity.
        rd: Radial diffusivity.
        pe: Principal eigenvector in world coordinates (RAS+); its sign is arbitrary.
        eigen_vals: Eigenvalues l1 >= l2 >= l3, as fitted (they can be
            negative).
        npd: True in the fitted voxels whose tensor is not positive definite
            (l3 <= 0), bool. The scalar maps there are computed from the
            eigenvalues as fitted, so they describe a tensor that is not
            physical.
    """

    dt: np.ndarray
    fa: np.ndarray
    md: np.ndarray
    ad: np.ndarray
    rd: np.ndarray
    pe: np.ndarray
    eigen_vals: np.ndarray
    npd: np.ndarray


class DTIModel:
    """Diffusion tensor imaging (DTI) model, which fits a diffusion tensor per voxel.

    `fit` returns a `DTIFit` with the tensor and its maps (FA, MD, AD, RD and the
    principal eigenvector).

    References:
        Basser PJ, Mattiello J, LeBihan D. Estimation of the effective
        self-diffusion tensor from the NMR spin echo. J Magn Reson B 103 (1994)
        247-254.
        Veraart J, Sijbers J, Sunaert S, Leemans A, Jeurissen B. Weighted linear
        least squares estimation of diffusion MRI parameters: strengths,
        limitations, and pitfalls. NeuroImage 81 (2013) 335-346.
    """

    def __init__(self, *, fit_method: str = "ols") -> None:
        """Sets the fit method.

        Args:
            fit_method: "ols" for ordinary least squares, or "wls"
                for weighted least squares. Defaults to "ols".

        Raises:
            ValueError: If the fit method is not one of these.
        """
        if fit_method not in FIT_METHODS:
            raise ValueError(
                f"Unknown DTI fit method {fit_method!r}: "
                f"use one of {', '.join(FIT_METHODS)}."
            )
        self.fit_method = fit_method

    def fit(
        self,
        volume: Volume,
        gtab: GradientTable,
        mask: Volume | np.ndarray | None = None,
        *,
        n_jobs: int = -1,
    ) -> DTIFit:
        """Fits the tensor in every voxel of the mask.

        Signal values that are not positive or finite are replaced by 1e-3 S0
        before taking the log, with the voxel's S0 from `larmor.dwi.estimate_s0`;
        without a b0 in the table, or where that S0 is not positive, by 1e-3 times
        the largest value of the data. Each voxel is fitted on its own, so the
        result does not depend on n_jobs.

        Args:
            volume: Diffusion-weighted volume, shape (X, Y, Z, N).
            gtab: Its gradient table, with at least two distinct b-values (a b0 is
                not needed when a mask is given); the tensor comes out in its
                frame, world coordinates.
            mask: Voxels to fit (nonzero inside), on the volume's grid. Defaults
                to None, which computes a brain mask from the mean b0 volume.
            n_jobs: Number of threads for the fit, at most the number of cores;
                -1 uses every core. Defaults to -1.

        Returns:
            DTIFit: The tensor and its maps, zero outside the mask.

        Raises:
            ValueError: If the table or the mask does not match the volume, the
                measurements cannot determine the tensor and S0 (for example one
                shell without b0, or fewer than six non-coplanar directions), if
                the mask is computed and there is no b0, or if n_jobs is 0 or
                below -1.
        """
        threads = n_threads(n_jobs)
        check_dwi(volume, gtab)
        s = volume.as_float()
        b_mat = gtab.b_matrix

        # The fit itself needs no b0 (log S0 is its intercept); only the automatic
        # mask does.
        if mask is None:
            mask = create_brain_mask(mean_b0(volume, gtab))
        else:
            mask = as_mask(volume, mask)

        s_masked = s[mask].astype(np.float64)
        s0 = estimate_s0(volume, gtab, mask)[mask] if gtab.b0_mask.any() else None
        floor_bad_samples(s_masked, s0)

        N = s.shape[-1]
        design_matrix = np.column_stack([np.ones(N), -b_mat])
        # A rank-deficient system has no unique fit (e.g. one shell without b0).
        if np.linalg.matrix_rank(design_matrix) < design_matrix.shape[1]:
            raise ValueError(
                "The b-values and b-vectors cannot determine a tensor: DTI needs at "
                "least two distinct b-values and six non-coplanar directions."
            )
        x, _ = fit_log_signal(
            design_matrix, s_masked, self.fit_method == "wls", n_threads=threads
        )

        d_params = x[1:7, :]
        dt, fa, md, ad, rd, pe, eigen_vals, npd = compute_dti_metrics(d_params, mask)

        return DTIFit(
            dt=dt, fa=fa, md=md, ad=ad, rd=rd, pe=pe, eigen_vals=eigen_vals, npd=npd
        )


def compute_dti_metrics(d: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """Computes the tensor metrics of the fitted voxels and puts them on the grid.

    MD is a third of the trace. The other metrics come from the eigenvalues as
    fitted, without clipping; voxels whose smallest eigenvalue is not positive
    are flagged, and a warning gives their number.

    Args:
        d: The 6 unique tensor elements (xx, xy, xz, yy, yz, zz) of every fitted
            voxel, shape (6, n_voxels).
        mask: Boolean mask of the fitted voxels, shape (X, Y, Z), with n_voxels
            True values in the order of d's columns.

    Returns:
        tuple[np.ndarray, ...]: DT, FA, MD, AD, RD, PE, the eigenvalues and the
        non-positive-definite flag, as maps on the grid, zero (False) outside the
        mask, as the attributes of `DTIFit`.
    """
    N = d.shape[1]
    diff_tensor = _tensor_matrices(d)

    eigen_vals, e_vecs = np.linalg.eigh(diff_tensor)

    eigen_vals = eigen_vals[:, ::-1]
    e_vecs = e_vecs[:, :, ::-1]

    eigen_vals = np.nan_to_num(eigen_vals, nan=0.0)
    npd = eigen_vals[:, 2] <= 0.0
    if npd.any():
        LarmorLogger.warning(
            "%d of %d fitted voxels have a tensor that is not positive definite.",
            int(npd.sum()),
            N,
        )

    l1 = eigen_vals[:, 0]
    l2 = eigen_vals[:, 1]
    l3 = eigen_vals[:, 2]

    md = np.nan_to_num((d[0] + d[3] + d[5]) / 3.0, nan=0.0)

    num_sq = (l1 - l2) ** 2 + (l2 - l3) ** 2 + (l3 - l1) ** 2
    numerator = np.sqrt(np.maximum(num_sq, 0.0))
    denominator = np.sqrt(l1**2 + l2**2 + l3**2)

    fa = np.zeros_like(md)
    nonzero = denominator > 1e-12
    fa[nonzero] = np.sqrt(0.5) * (numerator[nonzero] / denominator[nonzero])

    fa = np.clip(fa, 0.0, 1.0)
    fa = np.nan_to_num(fa, nan=0.0)

    ad = l1
    rd = (l2 + l3) / 2.0
    pe = e_vecs[:, :, 0]

    fa_full = unmask(fa, mask)
    md_full = unmask(md, mask)
    ad_full = unmask(ad, mask)
    rd_full = unmask(rd, mask)
    pe_full = unmask(pe, mask)
    dt_full = unmask(d.T, mask)
    e_vals_full = unmask(eigen_vals, mask)
    npd_full = unmask(npd, mask, fill_value=False, dtype=bool)

    return dt_full, fa_full, md_full, ad_full, rd_full, pe_full, e_vals_full, npd_full


def _tensor_matrices(d: np.ndarray) -> np.ndarray:
    """Builds the symmetric 3x3 tensors from their 6 unique elements.

    Args:
        d: The elements xx, xy, xz, yy, yz, zz, shape (6, n_voxels).

    Returns:
        np.ndarray: The tensors, shape (n_voxels, 3, 3).
    """
    diff_tensor = np.zeros((d.shape[1], 3, 3), dtype=np.float64)

    diff_tensor[:, 0, 0] = d[0]
    diff_tensor[:, 0, 1] = d[1]
    diff_tensor[:, 0, 2] = d[2]
    diff_tensor[:, 1, 0] = d[1]
    diff_tensor[:, 1, 1] = d[3]
    diff_tensor[:, 1, 2] = d[4]
    diff_tensor[:, 2, 0] = d[2]
    diff_tensor[:, 2, 1] = d[4]
    diff_tensor[:, 2, 2] = d[5]
    return diff_tensor
