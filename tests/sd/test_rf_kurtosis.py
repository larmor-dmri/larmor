import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from larmor.dwi.fitting import fit_log_signal
from larmor.io import load_volume, read_bvals_bvecs
from larmor.sd import AnisotropicRF, build_response_functions, estimate_wm_rf
from larmor.types import GradientTable, Volume

DKI_DIR = Path(__file__).parents[1] / "data" / "dki"

# The reference is MATLAB's response estimate, whose weighted fit works on
# single-precision signal and stops by a rule of its own. On the block the voxel
# eigenvalues of the weighted fit are within 4.9e-4 times the largest (l1) of
# MATLAB's (above 1e-4 only in five noisy voxels), and its voxel
# K = 1e-6 p_K / mean(lambda^2), a small parameter over small eigenvalues, within
# 7.6e-4.
WLS_TOL = 1e-3
K_ATOL = 1e-3

L1, L2, L3 = 1.7e-3, 0.3e-3, 0.2e-3
K_TRUE = 0.6
OBLIQUE = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
X_AXIS = np.array([1.0, 0.0, 0.0])


# MATLAB reference


def _mritoolkit_input(
    mask: npt.NDArray[np.bool_],
) -> tuple[Volume, GradientTable]:
    """The block as MRIToolkit's fod pipeline sees it.

    EDTI_Data_2_MRIToolkit rounds the b-values to 100 and, in a masked voxel whose
    smallest b0 is not above its largest diffusion-weighted signal m, raises every
    b0 at or below m + 1 to m + 1.
    """
    volume = load_volume(DKI_DIR / "dwi.nii.gz")
    gtab = read_bvals_bvecs(
        DKI_DIR / "bvals", DKI_DIR / "bvecs", ref_path=DKI_DIR / "dwi.nii.gz"
    )
    gtab = GradientTable(np.round(gtab.bvals / 100) * 100, gtab.bvecs)
    data = volume.as_float().copy()
    s = data[mask]
    b0 = s[:, gtab.b0_mask]
    max_dw = s[:, ~gtab.b0_mask].max(axis=1, keepdims=True)
    raised = (b0.min(axis=1, keepdims=True) <= max_dw) & (b0 <= max_dw + 1)
    s[:, gtab.b0_mask] = np.where(raised, max_dw + 1, b0)
    data[mask] = s
    return Volume(data, volume.affine), gtab


@pytest.fixture(scope="module")
def ref_mask(wm_rf_ref: dict[str, Any]) -> npt.NDArray[np.bool_]:
    mask: npt.NDArray[np.bool_] = wm_rf_ref["mask"] > 0
    return mask


@pytest.fixture(scope="module")
def block(
    ref_mask: npt.NDArray[np.bool_],
) -> tuple[Volume, GradientTable]:
    return _mritoolkit_input(ref_mask)


def _single_shell(volume: Volume, gtab: GradientTable) -> tuple[Volume, GradientTable]:
    keep = np.isin(np.round(gtab.bvals, -2), (0.0, 1000.0))
    return (
        Volume(volume.data[..., keep], volume.affine),
        GradientTable(gtab.bvals[keep], gtab.bvecs[keep]),
    )


def _estimate_logging_count(
    caplog: pytest.LogCaptureFixture,
    volume: Volume,
    gtab: GradientTable,
    mask: npt.NDArray[np.bool_],
) -> tuple[AnisotropicRF, int]:
    """estimate_wm_rf, and the number of voxels it says it selected."""
    with caplog.at_level(logging.INFO, logger="larmor"):
        rf = estimate_wm_rf(volume, gtab, mask)
    found = re.findall(r"from (\d+) voxels", caplog.text)
    assert len(found) == 1, caplog.text
    return rf, int(found[0])


def test_voxel_eigenvalues_and_kurtosis_match_mritoolkit(
    wm_rf_ref: dict[str, Any],
    ref_mask: npt.NDArray[np.bool_],
    block: tuple[Volume, GradientTable],
) -> None:
    # estimate_wm_rf's fit, step by step: Larmor's b-matrix carries the factor 2 of
    # the off-diagonal terms, so the parameters are the tensor elements themselves.
    # K is on MATLAB's mean(lambda^2) scale here, to compare with its voxel values;
    # estimate_wm_rf itself divides by MD^2.
    volume, gtab = block
    design = np.column_stack(
        [np.ones(len(gtab.bvals)), -gtab.b_matrix, (gtab.bvals * 1e-3) ** 2 / 6]
    )
    s = volume.as_float()[ref_mask].astype(np.float64)
    usable = (s > 0).all(axis=1)
    # MATLAB skips the voxels with a zero signal: one on this block.
    fitted = wm_rf_ref["voxel"]["fitted"][ref_mask] > 0
    np.testing.assert_array_equal(usable, fitted)
    assert int((~usable).sum()) == 1

    params, _ = fit_log_signal(design, s[usable], weighted=True)
    tensors = params[[1, 2, 3, 2, 4, 5, 3, 5, 6]].T.reshape(-1, 3, 3)
    e_vals = np.linalg.eigvalsh(tensors)[:, ::-1]
    k = 1e-6 * params[7] / np.mean(e_vals**2, axis=1)

    ref_e = wm_rf_ref["voxel"]["e_vals"][ref_mask][usable]
    ref_k = wm_rf_ref["voxel"]["k"][ref_mask][usable]
    assert (np.abs(e_vals - ref_e).max(axis=1) <= WLS_TOL * ref_e[:, 0]).all()
    np.testing.assert_allclose(k, ref_k, rtol=0, atol=K_ATOL)


def test_estimate_matches_the_correct_mritoolkit_aggregate(
    caplog: pytest.LogCaptureFixture,
    wm_rf_ref: dict[str, Any],
    ref_mask: npt.NDArray[np.bool_],
    block: tuple[Volume, GradientTable],
) -> None:
    volume, gtab = block
    rf, n_selected = _estimate_logging_count(caplog, volume, gtab, ref_mask)
    correct = wm_rf_ref["correct"]

    assert n_selected == correct["n"] == 9
    lambdas = correct["lambdas"]
    np.testing.assert_allclose(
        rf.eigen_vals, lambdas, rtol=0, atol=WLS_TOL * lambdas[0]
    )
    # MATLAB's voxel K is on its mean(lambda^2) scale; Larmor's is on MD^2, the scale
    # of the response's kurtosis term. Rescaled, the same voxels in (0, 4) give K.
    # The FA and K selection is copied here on purpose: no MATLAB run gives MD^2
    # values to compare with.
    voxel = wm_rf_ref["voxel"]
    fitted = voxel["fitted"][ref_mask] > 0
    e = voxel["e_vals"][ref_mask][fitted]
    k_md = (
        voxel["k"][ref_mask][fitted] * np.mean(e**2, axis=1) / np.mean(e, axis=1) ** 2
    )
    mean = e.mean(axis=1, keepdims=True)
    fa = np.sqrt(1.5 * np.sum((e - mean) ** 2, axis=1) / np.sum(e**2, axis=1))
    kept = (fa > 0.7) & (k_md > 0) & (k_md < 4)
    np.testing.assert_allclose(rf.mean_kurtosis, k_md[kept].mean(), rtol=0, atol=K_ATOL)


def test_mritoolkit_own_estimate_differs_by_its_doubled_off_diagonals(
    wm_rf_ref: dict[str, Any],
) -> None:
    # Eigenval_IsoK_WM_FromData selects voxels by, and takes the eigenvalues of, a
    # tensor whose off-diagonal elements are twice the fitted ones. Larmor uses the
    # fitted tensor, so on this block it deliberately does not match these numbers.
    correct, mritoolkit = wm_rf_ref["correct"], wm_rf_ref["mritoolkit"]
    assert (correct["n"], mritoolkit["n"]) == (9, 136)
    assert not np.allclose(mritoolkit["lambdas"], correct["lambdas"], rtol=0.1)
    assert not np.isclose(mritoolkit["K"], correct["K"], rtol=0.1)


def test_single_shell_block_matches_mritoolkit_without_kurtosis(
    caplog: pytest.LogCaptureFixture,
    wm_rf_ref: dict[str, Any],
    ref_mask: npt.NDArray[np.bool_],
    block: tuple[Volume, GradientTable],
) -> None:
    # MATLAB fitted the b0 and b = 1000 volumes with the plain tensor design; with the
    # kurtosis column that design is singular, and the eigenvalues would differ.
    volume, gtab = _single_shell(*block)
    rf, n_selected = _estimate_logging_count(caplog, volume, gtab, ref_mask)
    correct = wm_rf_ref["single"]["correct"]

    assert n_selected == correct["n"] == 20
    lambdas = correct["lambdas"]
    np.testing.assert_allclose(
        rf.eigen_vals, lambdas, rtol=0, atol=WLS_TOL * lambdas[0]
    )
    assert rf.mean_kurtosis == 0.0 == correct["K"]


# Synthetic data


def _gtab(shells: tuple[float, ...], n_dirs: int = 30) -> GradientTable:
    """Two b0s and n_dirs random directions on each shell."""
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(n_dirs * len(shells), 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    bvals = np.concatenate([[0.0, 0.0], np.repeat(shells, n_dirs)])
    return GradientTable(bvals, np.vstack([np.zeros((2, 3)), dirs]))


def _tensor(
    e_vals: tuple[float, float, float], fibre: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """A tensor with these eigenvalues, the first along the fibre."""
    a = fibre / np.linalg.norm(fibre)
    b = np.cross(a, [0.0, 0.0, 1.0] if abs(a[2]) < 0.9 else [1.0, 0.0, 0.0])
    b /= np.linalg.norm(b)
    frame = np.column_stack([a, b, np.cross(a, b)])
    tensor: npt.NDArray[np.float64] = frame @ np.diag(e_vals) @ frame.T
    return tensor


def _signal(
    gtab: GradientTable,
    e_vals: tuple[float, float, float],
    k: float = K_TRUE,
    fibre: npt.NDArray[np.float64] = OBLIQUE,
) -> npt.NDArray[np.float64]:
    """ln S = ln S0 - b g'Dg + b^2 k MD^2 / 6, so the voxel K is k."""
    d = _tensor(e_vals, fibre)
    gdg = np.einsum("ni,ij,nj->n", gtab.bvecs, d, gtab.bvecs)
    b = gtab.bvals
    md_sq = np.mean(e_vals) ** 2
    signal: npt.NDArray[np.float64] = 1000.0 * np.exp(-b * gdg + b**2 * k * md_sq / 6)
    return signal


def _volume(signals: list[npt.NDArray[np.float64]]) -> Volume:
    """One voxel per signal, along x."""
    return Volume(np.array(signals)[:, np.newaxis, np.newaxis], np.eye(4))


def _all(volume: Volume) -> npt.NDArray[np.bool_]:
    return np.ones(volume.data.shape[:3], dtype=bool)


def _with_fa(fa: float, l23: float = 0.4e-3) -> tuple[float, float, float]:
    """Eigenvalues [l1, l23, l23] with this FA = (l1 - l23) / sqrt(l1^2 + 2 l23^2)."""
    l1 = l23 * (1 + np.sqrt(1 - (1 - fa**2) * (1 - 2 * fa**2))) / (1 - fa**2)
    return (float(l1), l23, l23)


# estimate_wm_rf reads the signal as float32 (relative 6e-8): the noise-free fit is
# good to about 1e-11 mm^2/s in the eigenvalues and 1e-6 in K.
E_ATOL = 1e-9
K_RTOL = 1e-5
EXPECTED = [L1, (L2 + L3) / 2, (L2 + L3) / 2]


@pytest.mark.parametrize("fibre", [X_AXIS, OBLIQUE], ids=["axis", "oblique"])
def test_known_tensor_and_kurtosis_are_recovered_in_any_orientation(
    fibre: npt.NDArray[np.float64],
) -> None:
    # MRIToolkit's doubled off-diagonals change the result with the orientation;
    # the fitted tensor's eigenvalues do not.
    gtab = _gtab((1000.0, 2000.0))
    volume = _volume([_signal(gtab, (L1, L2, L3), fibre=fibre)] * 3)
    rf = estimate_wm_rf(volume, gtab, _all(volume))

    np.testing.assert_allclose(rf.eigen_vals, EXPECTED, rtol=0, atol=E_ATOL)
    np.testing.assert_allclose(rf.mean_kurtosis, K_TRUE, rtol=K_RTOL)


def test_estimated_response_reproduces_the_signal_it_came_from() -> None:
    # The estimate and the kernel both use MD^2, so the response's kurtosis term is
    # the b^2 term fitted to the data. With different scales it would be off (about
    # half for these eigenvalues, 17% of the signal at b = 2000).
    e_vals = (1.7e-3, 0.25e-3, 0.25e-3)
    gtab = _gtab((1000.0, 2000.0))
    voxel = _signal(gtab, e_vals)
    rf = estimate_wm_rf(_volume([voxel] * 3), gtab, _all(_volume([voxel] * 3)))

    predicted = rf.signal(gtab.bvals, gtab.bvecs, OBLIQUE[np.newaxis])[:, 0]
    np.testing.assert_allclose(predicted, voxel / 1000.0, rtol=1e-6)


@pytest.mark.parametrize("shells", [(1000.0, 2000.0), (600.0, 1000.0)])
def test_two_shells_above_500_fit_the_kurtosis(shells: tuple[float, ...]) -> None:
    gtab = _gtab(shells)
    volume = _volume([_signal(gtab, (L1, L2, L3))])

    rf = estimate_wm_rf(volume, gtab, _all(volume))

    np.testing.assert_allclose(rf.mean_kurtosis, K_TRUE, rtol=K_RTOL)


@pytest.mark.parametrize(
    "shells",
    [(1000.0,), (300.0, 1000.0), (990.0, 1000.0, 1010.0)],
    ids=["one-shell", "one-above-500", "spread-shell"],
)
def test_fewer_than_two_shells_above_500_give_zero_kurtosis(
    shells: tuple[float, ...],
) -> None:
    gtab = _gtab(shells)
    volume = _volume([_signal(gtab, (L1, L2, L3))])

    rf = estimate_wm_rf(volume, gtab, _all(volume))

    assert rf.mean_kurtosis == 0.0


def test_single_shell_fits_the_plain_tensor() -> None:
    # With one shell the kurtosis column duplicates the trace: fitting it would
    # leave the eigenvalues undetermined.
    gtab = _gtab((1000.0,))
    volume = _volume([_signal(gtab, (L1, L2, L3), k=0.0)])

    rf = estimate_wm_rf(volume, gtab, _all(volume))

    np.testing.assert_allclose(rf.eigen_vals, EXPECTED, rtol=0, atol=E_ATOL)


@pytest.mark.parametrize(
    ("other_fa", "kept"), [(0.69, False), (0.71, True)], ids=["below", "above"]
)
def test_only_voxels_with_fa_above_0_7_set_the_eigenvalues(
    other_fa: float, kept: bool
) -> None:
    # Most voxels hold the other tensor, so they set the median when they count.
    gtab = _gtab((1000.0, 2000.0))
    other = _with_fa(other_fa)
    signals = [_signal(gtab, (L1, L2, L3))] * 2 + [_signal(gtab, other)] * 5
    volume = _volume(signals)

    rf = estimate_wm_rf(volume, gtab, _all(volume))

    expected = list(other) if kept else EXPECTED
    np.testing.assert_allclose(rf.eigen_vals, expected, rtol=0, atol=E_ATOL)


def test_no_voxel_with_fa_above_0_7_is_refused() -> None:
    gtab = _gtab((1000.0, 2000.0))
    volume = _volume([_signal(gtab, _with_fa(0.69))] * 3)

    with pytest.raises(ValueError, match="FA > 0.7"):
        estimate_wm_rf(volume, gtab, _all(volume))


@pytest.mark.parametrize("bad_k", [-0.2, 4.5], ids=["negative", "above-4"])
def test_voxel_kurtosis_outside_0_to_4_is_left_out(bad_k: float) -> None:
    gtab = _gtab((1000.0, 2000.0))
    signals = [_signal(gtab, (L1, L2, L3))] * 2 + [
        _signal(gtab, (L1, L2, L3), k=bad_k)
    ] * 3
    volume = _volume(signals)

    rf = estimate_wm_rf(volume, gtab, _all(volume))

    np.testing.assert_allclose(rf.mean_kurtosis, K_TRUE, rtol=K_RTOL)


def test_kurtosis_is_averaged_over_the_selected_voxels_only() -> None:
    gtab = _gtab((1000.0, 2000.0))
    low_fa = _signal(gtab, _with_fa(0.5), k=1.5)
    volume = _volume([_signal(gtab, (L1, L2, L3))] * 2 + [low_fa] * 3)

    rf = estimate_wm_rf(volume, gtab, _all(volume))

    np.testing.assert_allclose(rf.mean_kurtosis, K_TRUE, rtol=K_RTOL)


def test_no_voxel_kurtosis_in_range_gives_zero_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gtab = _gtab((1000.0, 2000.0))
    volume = _volume([_signal(gtab, (L1, L2, L3), k=-0.2)] * 3)

    with caplog.at_level(logging.WARNING, logger="larmor"):
        rf = estimate_wm_rf(volume, gtab, _all(volume))

    assert rf.mean_kurtosis == 0.0
    assert "kurtosis 0" in caplog.text
    np.testing.assert_allclose(rf.eigen_vals, EXPECTED, rtol=0, atol=E_ATOL)


@pytest.mark.parametrize("bad", [0.0, -5.0, np.nan, np.inf])
def test_voxels_with_a_signal_not_positive_and_finite_are_skipped(bad: float) -> None:
    # The broken voxels hold another tensor and outnumber the good ones: counted,
    # they would set the median (or break the fit).
    gtab = _gtab((1000.0, 2000.0))
    broken = _signal(gtab, _with_fa(0.9))
    broken[5] = bad
    volume = _volume([_signal(gtab, (L1, L2, L3))] * 2 + [broken] * 5)

    rf = estimate_wm_rf(volume, gtab, _all(volume))

    np.testing.assert_allclose(rf.eigen_vals, EXPECTED, rtol=0, atol=E_ATOL)
    np.testing.assert_allclose(rf.mean_kurtosis, K_TRUE, rtol=K_RTOL)


# build_response_functions


@pytest.fixture
def multi_shell() -> tuple[Volume, GradientTable]:
    gtab = _gtab((1000.0, 2000.0))
    return _volume([_signal(gtab, (L1, L2, L3))] * 3), gtab


def test_estimated_response_keeps_its_kurtosis_by_default(
    multi_shell: tuple[Volume, GradientTable],
) -> None:
    volume, gtab = multi_shell
    (wm,) = build_response_functions(volume, gtab, _all(volume))

    assert isinstance(wm, AnisotropicRF)
    np.testing.assert_allclose(wm.mean_kurtosis, K_TRUE, rtol=K_RTOL)


@pytest.mark.parametrize("given", [0.0, 0.9])
def test_given_kurtosis_replaces_the_estimated_one(
    multi_shell: tuple[Volume, GradientTable], given: float
) -> None:
    volume, gtab = multi_shell
    (wm,) = build_response_functions(volume, gtab, _all(volume), mean_kurtosis=given)

    assert isinstance(wm, AnisotropicRF)
    assert wm.mean_kurtosis == given
    np.testing.assert_allclose(wm.eigen_vals, EXPECTED, rtol=0, atol=E_ATOL)


def test_given_eigenvalues_have_zero_kurtosis_by_default(
    multi_shell: tuple[Volume, GradientTable],
) -> None:
    volume, gtab = multi_shell
    (wm,) = build_response_functions(
        volume, gtab, _all(volume), wm_eigen_vals=(1.5e-3, 0.4e-3, 0.4e-3)
    )

    assert wm == AnisotropicRF(eigen_vals=[1.5e-3, 0.4e-3, 0.4e-3], mean_kurtosis=0.0)
