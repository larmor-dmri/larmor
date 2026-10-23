from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from larmor.dwi import DKIModel, DTIFit, DTIModel
from larmor.io import load_dwi
from larmor.types import GradientTable, Volume

# A white-matter-like tensor with three distinct eigenvalues, rotated so that
# its off-diagonal elements are not zero and its principal direction is not an axis.
EIGEN_VALS = np.array([1.7e-3, 0.4e-3, 0.2e-3])
SHAPE = (3, 3, 3)


def _rotation(alpha: float, beta: float, gamma: float) -> npt.NDArray[np.float64]:
    """Rotation matrix Rz(alpha) Ry(beta) Rz(gamma)."""

    def rz(t: float) -> npt.NDArray[np.float64]:
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def ry(t: float) -> npt.NDArray[np.float64]:
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    return rz(alpha) @ ry(beta) @ rz(gamma)


ROTATION = _rotation(0.4, 0.9, 0.2)
TENSOR = ROTATION @ np.diag(EIGEN_VALS) @ ROTATION.T


@pytest.fixture
def mask() -> npt.NDArray[np.bool_]:
    """Every voxel except one corner, so the fit has a background to leave at zero."""
    mask = np.ones(SHAPE, dtype=bool)
    mask[0, 0, 0] = False
    return mask


@pytest.fixture
def fit(mask: npt.NDArray[np.bool_]) -> DTIFit:
    """DTI fit of a noise-free image whose every voxel has the signal of TENSOR."""
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(30, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    bvals = np.concatenate([[0.0, 0.0], np.full(30, 1000.0)])
    bvecs = np.vstack([np.zeros((2, 3)), directions])

    signal = 1000.0 * np.exp(-bvals * np.einsum("ni,ij,nj->n", bvecs, TENSOR, bvecs))
    data = np.broadcast_to(signal, (*SHAPE, len(bvals))).astype(np.float32)

    return DTIModel().fit(
        Volume(data, np.eye(4)), GradientTable(bvals, bvecs), mask=mask
    )


def test_fit_shapes(fit: DTIFit) -> None:
    for scalar_map in (fit.fa, fit.md, fit.ad, fit.rd):
        assert scalar_map.shape == SHAPE
    assert fit.dt.shape == (*SHAPE, 6)
    assert fit.eigen_vals.shape == (*SHAPE, 3)
    assert fit.pe.shape == (*SHAPE, 3)


def test_fit_recovers_tensor(fit: DTIFit, mask: npt.NDArray[np.bool_]) -> None:
    l1, l2, l3 = EIGEN_VALS
    md = EIGEN_VALS.mean()
    fa = np.sqrt(1.5 * np.sum((EIGEN_VALS - md) ** 2) / np.sum(EIGEN_VALS**2))
    unique = TENSOR[np.triu_indices(3)]  # xx, xy, xz, yy, yz, zz
    n_voxels = int(mask.sum())

    # Results are float32, so diffusivities (~1e-3) are compared to about 1e-7.
    np.testing.assert_allclose(fit.dt[mask], np.tile(unique, (n_voxels, 1)), atol=1e-7)
    np.testing.assert_allclose(
        fit.eigen_vals[mask], np.tile(EIGEN_VALS, (n_voxels, 1)), atol=1e-7
    )
    np.testing.assert_allclose(fit.md[mask], md, atol=1e-7)
    np.testing.assert_allclose(fit.ad[mask], l1, atol=1e-7)
    np.testing.assert_allclose(fit.rd[mask], (l2 + l3) / 2, atol=1e-7)
    np.testing.assert_allclose(fit.fa[mask], fa, atol=1e-4)

    # An eigenvector's sign is arbitrary, so only its alignment is checked.
    alignment = np.abs(fit.pe[mask] @ ROTATION[:, 0])
    np.testing.assert_allclose(alignment, 1.0, atol=1e-4)


def test_fit_is_zero_outside_mask(fit: DTIFit, mask: npt.NDArray[np.bool_]) -> None:
    outside = ~mask
    for result in (fit.dt, fit.fa, fit.md, fit.ad, fit.rd, fit.pe, fit.eigen_vals):
        assert np.all(result[outside] == 0)


@pytest.mark.parametrize("fit_method", ["ols", "wls"])
@pytest.mark.parametrize("bad", [0.0, -3.0, np.nan])
def test_sample_that_is_not_positive_is_fitted_as_1e_3_s0(
    fit_method: str, bad: float
) -> None:
    rng = np.random.default_rng(1)
    directions = rng.normal(size=(30, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    bvals = np.concatenate([[0.0, 0.0], np.full(30, 1000.0)])
    bvecs = np.vstack([np.zeros((2, 3)), directions])
    gtab = GradientTable(bvals, bvecs)
    s0 = 5000.0
    signal = s0 * np.exp(-bvals * np.einsum("ni,ij,nj->n", bvecs, TENSOR, bvecs))
    data = np.broadcast_to(signal, (*SHAPE, len(bvals))).copy()
    # The voxel's S0 is half the image's largest value, so its own S0 sets the floor.
    data[1, 1, 1] *= 0.5
    floored = data.copy()
    data[1, 1, 1, 5] = bad
    floored[1, 1, 1, 5] = 1e-3 * 0.5 * s0
    model = DTIModel(fit_method=fit_method)
    mask = np.ones(SHAPE, dtype=bool)
    got = model.fit(Volume(data, np.eye(4)), gtab, mask)
    want = model.fit(Volume(floored, np.eye(4)), gtab, mask)
    np.testing.assert_allclose(got.dt, want.dt, rtol=1e-6, atol=1e-12)


DKI_DIR = Path(__file__).parents[1] / "data" / "dki"


def _block() -> tuple[Volume, GradientTable, npt.NDArray[np.bool_]]:
    volume, gtab = load_dwi(
        DKI_DIR / "dwi.nii.gz",
        bvals=DKI_DIR / "bvals",
        bvecs=DKI_DIR / "bvecs",
        frame="fsl",
    )
    return volume, gtab, np.load(DKI_DIR / "mrtrix_dt_ref.npz")["mask"]


def test_ols_fit_is_the_least_squares_solution_on_the_block() -> None:
    volume, gtab, mask = _block()
    fit = DTIModel().fit(volume, gtab, mask)
    s = volume.as_float()[mask].astype(np.float64)
    # Voxels with a sample that is not positive are left out: the fit replaces it.
    kept = (s > 0).all(axis=1)
    design = np.column_stack([np.ones(len(gtab.bvals)), -gtab.b_matrix])
    expected = np.linalg.lstsq(design, np.log(s[kept]).T, rcond=None)[0][1:7].T
    # The maps are float32: diffusivities (~1e-3) hold about 1e-10.
    np.testing.assert_allclose(fit.dt[mask][kept], expected, rtol=0, atol=1e-9)


@pytest.mark.parametrize("fit_method", ["ols", "wls"])
def test_fit_matches_mrtrix_dwi2tensor_on_the_block(fit_method: str) -> None:
    volume, gtab, mask = _block()
    fit = DTIModel(fit_method=fit_method).fit(volume, gtab, mask)
    refs = np.load(DKI_DIR / "mrtrix_dt_ref.npz")
    ref = refs[f"dt_{fit_method}"]
    # Larmor replaces non-positive measurements before the log; MRtrix does not.
    kept = (volume.as_float()[mask] > 0).all(axis=1)
    if fit_method == "wls":
        # MRtrix stops after 10 reweightings, Larmor once the fit settles.
        kept &= refs["wls_settled"]
    ref_dt = ref[kept].astype(np.float64)
    matrices = ref_dt[:, [0, 1, 2, 1, 3, 4, 2, 4, 5]].reshape(-1, 3, 3)
    scale = np.abs(np.linalg.eigvalsh(matrices)).max(axis=1)
    err = np.abs(fit.dt[mask][kept] - ref_dt).max(axis=1) / scale
    assert err.max() < 1e-5


def test_unknown_fit_method_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown DTI fit method"):
        DTIModel(fit_method="nlls")


@pytest.mark.parametrize("model", [DTIModel(), DKIModel()], ids=["dti", "dki"])
def test_n_jobs_of_zero_is_refused(model: DTIModel | DKIModel) -> None:
    volume, gtab, mask = _block()
    with pytest.raises(ValueError, match="n_jobs"):
        model.fit(volume, gtab, mask, n_jobs=0)


def test_tensor_with_a_negative_eigenvalue_keeps_the_trace_and_is_flagged() -> None:
    from larmor.dwi.dti import compute_dti_metrics

    tensor = ROTATION @ np.diag([2.9e-3, 2.8e-3, -0.3e-3]) @ ROTATION.T
    elements = tensor[[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]][:, None]
    mask = np.ones((1, 1, 1), dtype=bool)

    _, _, md, _, _, _, eigen_vals, npd = compute_dti_metrics(elements, mask)

    np.testing.assert_allclose(md[0, 0, 0], np.trace(tensor) / 3, rtol=1e-6)
    np.testing.assert_allclose(eigen_vals[0, 0, 0, 2], -0.3e-3, atol=1e-9)
    assert npd[0, 0, 0]


def test_positive_definite_voxels_are_not_flagged(fit: DTIFit) -> None:
    assert not fit.npd.any()
