"""Larmor's spherical deconvolution against the MATLAB implementation (run_grl.m)."""

from pathlib import Path
from typing import Any

import numpy as np
from reference_solvers import matlab_drl
from scipy.io import loadmat

from larmor.sd import (
    AnisotropicRF,
    IsotropicRF,
    NoddiRF,
    ResponseFunction,
    SDModel,
)
from larmor.sd.kernel import Kernel
from larmor.sd.model import low_resolution_dirs, normalize_by_b0
from larmor.sd.solvers import calibrate_drl_nu, fit_multi_tissue
from larmor.sd.sphere import DIRS_300
from larmor.types import GradientTable

DATA_DIR = Path(__file__).parents[1] / "data"

# grl and deconv hold scipy loadmat output, which has no fixed structure.


def test_match_rf_hr(grl: dict[str, Any], kernel_hr: Kernel) -> None:
    np.testing.assert_allclose(kernel_hr.k, grl["HRKernel"], atol=1e-6)


def test_match_rf_lr(grl: dict[str, Any], kernel_lr: Kernel) -> None:
    np.testing.assert_allclose(kernel_lr.k, grl["LRKernel"], atol=1e-6)


def test_match_v_hr(grl: dict[str, Any], kernel_hr: Kernel) -> None:
    nu_h = calibrate_drl_nu(kernel_hr, grl["bvals"].astype(np.float64))
    np.testing.assert_allclose(grl["NN_H"], nu_h, rtol=1e-4, atol=0)


def test_match_v_lr(grl: dict[str, Any], kernel_lr: Kernel) -> None:
    nu_l = calibrate_drl_nu(kernel_lr, grl["bvals"].astype(np.float64))
    np.testing.assert_allclose(grl["NN_L"], nu_l, rtol=1e-4, atol=0)


def test_match_drl_fit(
    grl: dict[str, Any],
    deconv: dict[str, Any],
    kernel_lr: Kernel,
    kernel_hr: Kernel,
) -> None:
    """dRL FODs and fractions match MATLAB in every voxel of the mask."""
    bvals = grl["bvals"].astype(np.float64)
    voxels = np.argwhere(grl["mask"] == 1)
    xs, ys, zs = voxels[:, 0], voxels[:, 1], voxels[:, 2]
    is_b0 = GradientTable(bvals, grl["bvecs"]).b0_mask
    norm = normalize_by_b0(grl["img"][xs, ys, zs].astype(np.float64), is_b0)
    ref_fod = deconv["fod"][xs, ys, zs]
    ref_frac = deconv["fractions"][xs, ys, zs]

    fods, fracs = fit_multi_tissue(
        norm,
        kernel_lr,
        kernel_hr,
        matlab_drl(kernel_lr, float(grl["NN_L"]), 200),
        matlab_drl(kernel_hr, float(grl["NN_H"]), 600),
        n_threads=2,
    )
    for i in range(len(voxels)):
        np.testing.assert_allclose(
            fods[i], ref_fod[i], rtol=1e-3, err_msg=f"voxel {voxels[i]}"
        )
        np.testing.assert_allclose(
            fracs[i], ref_frac[i], atol=1e-3, err_msg=f"voxel {voxels[i]}"
        )


def test_match_kurtosis_kernel_on_three_shells() -> None:
    # make_grl_kurtosis_kernel_ref.m: MRIToolkit's GRL kernel with K = 0.8 on
    # b = 1000, 2000, 3000, whose kurtosis term is (1/6) b^2 MD^2 K. MATLAB keeps the
    # kernel in single precision.
    ref: dict[str, Any] = loadmat(
        DATA_DIR / "grl_kurtosis_kernel.mat", simplify_cells=True
    )["grl_kurtosis"]
    rf = AnisotropicRF(eigen_vals=list(ref["eigen_vals"]), mean_kurtosis=ref["K"])
    kernel = Kernel(
        [rf],
        bvals=ref["bvals"].astype(np.float64),
        bvecs=ref["bvecs"].astype(np.float64),
        dirs=DIRS_300,
    )
    np.testing.assert_allclose(kernel.k, ref["HRKernel"], rtol=0, atol=1e-6)


def _mfod_rfs() -> list[ResponseFunction]:
    # make_mfod_ref.m's setup: DKI white matter, NODDI grey matter, CSF.
    return [
        AnisotropicRF([1.7e-3, 0.2e-3, 0.2e-3], 0.0),
        NoddiRF(),
        IsotropicRF(3e-3),
    ]


def _load_mfod(name: str, key: str) -> dict[str, Any]:
    data: dict[str, Any] = loadmat(DATA_DIR / "mfod" / name, simplify_cells=True)[key]
    return data


def test_match_mfod_kernel_per_block() -> None:
    # make_mfod_ref.m: MRIToolkit's mFOD HRKernel (DKI, NODDI, CSF) in single
    # precision, hence atol 1e-6.
    ref = _load_mfod("mfod_kernel.mat", "kernel")
    kernel = Kernel(
        _mfod_rfs(),
        bvals=ref["bvals"].astype(np.float64),
        bvecs=ref["bvecs"].astype(np.float64),
        dirs=DIRS_300,
    )
    expected = ref["HRKernel"]
    for name, block in zip(("dki", "noddi", "csf"), kernel.blocks, strict=True):
        np.testing.assert_allclose(
            kernel.k[:, block], expected[:, block], rtol=0, atol=1e-6, err_msg=name
        )


def test_match_mfod_low_resolution_kernel() -> None:
    # 45 DKI, 90 NODDI and 1 CSF columns; single precision, hence atol 1e-6. The
    # reference repeats the +z pole as the last column of each anisotropic block,
    # which Larmor leaves out.
    ref = _load_mfod("mfod_kernel.mat", "kernel")
    rfs = _mfod_rfs()
    bvals = ref["bvals"].astype(np.float64)
    kernel = Kernel(
        rfs,
        bvals=bvals,
        bvecs=ref["bvecs"].astype(np.float64),
        dirs=low_resolution_dirs(rfs, len(bvals)),
    )
    expected = np.delete(ref["LRKernel"], [44, 134], axis=1)
    np.testing.assert_allclose(kernel.k, expected, rtol=0, atol=1e-6)


def test_match_mfod_fit() -> None:
    """mFOD FODs and fractions with the method's defaults match MRIToolkit's L2LSQ."""
    kernel_ref = _load_mfod("mfod_kernel.mat", "kernel")
    ref = _load_mfod("mfod_ref.mat", "ref")
    bvals = kernel_ref["bvals"].astype(np.float64)
    bvecs = kernel_ref["bvecs"].astype(np.float64)
    signals = normalize_by_b0(ref["signal"].astype(np.float64), bvals <= bvals.min())

    model = SDModel(_mfod_rfs(), method="mFOD", progress=False)
    hr, fit_chunk = model._prepare(
        model.method.default_solver(), bvals, bvecs, n_threads=2
    )
    fods, fractions, _ = fit_chunk(signals)

    # The reference is single precision and the NNLS active set can flip at
    # rounding level, so the bounds leave headroom over the observed 5e-5 and 2e-6.
    ref_fod = ref["fod"].astype(np.float64)
    fod_err = np.abs(fods - ref_fod).max(axis=1) / ref_fod.max(axis=1)
    assert fod_err.max() <= 1e-3, f"voxel {ref['ijk'][fod_err.argmax()]}"
    assert fractions is not None
    np.testing.assert_allclose(
        fractions, ref["fractions"].astype(np.float64), rtol=0, atol=1e-4
    )
