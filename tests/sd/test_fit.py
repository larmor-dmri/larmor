import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import larmor.sd.model as sd_module
from larmor.dwi import DTIModel, mean_b0
from larmor.dwi.inputs import as_mask
from larmor.sd import (
    DRL,
    L2NNLS,
    AnisotropicRF,
    IsotropicRF,
    Lasso,
    NoddiRF,
    ResponseFunction,
    SDMethod,
    SDModel,
    SDSolver,
    estimate_wm_rf,
    find_peaks,
)
from larmor.sd.kernel import Kernel
from larmor.sd.sphere import DIRS_300, get_dirs
from larmor.types import GradientTable, Volume

N_SH = 28

DWI = tuple[Volume, GradientTable, npt.NDArray[np.bool_]]

# grl holds scipy loadmat output, which has no fixed structure.


@pytest.fixture
def small_dwi(grl: dict[str, Any]) -> DWI:
    """A 4x4x4 corner of the reference block, its gradient table and a mask."""
    block = grl["img"][:4, :4, :4].astype(np.float32)
    gtab = GradientTable(
        grl["bvals"].astype(np.float64), grl["bvecs"].astype(np.float64)
    )
    mask = grl["mask"][:4, :4, :4].astype(bool)
    mask[:2] = False
    return Volume(block, np.eye(4)), gtab, mask


def _wm_rf() -> ResponseFunction:
    return AnisotropicRF(eigen_vals=[1.7e-3, 0.2e-3, 0.2e-3], mean_kurtosis=0.0)


def test_fit_multi_tissue(small_dwi: DWI) -> None:
    volume, gtab, mask = small_dwi
    sd = SDModel(
        [_wm_rf(), IsotropicRF(diffusivity=0.7e-3), IsotropicRF(diffusivity=3e-3)]
    )

    results = sd.fit(volume, gtab, mask=mask, n_jobs=1)
    sh = results.fods[0]
    frac = results.tissue_fractions

    assert sh.shape == mask.shape + (N_SH,)
    assert results.lmax == 6
    assert results.basis == "tournier"
    assert frac is not None
    assert frac.shape == mask.shape + (3,)

    assert np.all(sh[~mask] == 0) and np.all(frac[~mask] == 0)
    assert np.isfinite(sh[mask]).all() and np.isfinite(frac[mask]).all()
    np.testing.assert_allclose(frac[mask].sum(axis=-1), 1.0, atol=1e-5)


def test_fit_single_tissue(small_dwi: DWI) -> None:
    volume, gtab, mask = small_dwi
    sd = SDModel([_wm_rf()])
    results = sd.fit(volume, gtab, mask=mask, n_jobs=1)

    sh = results.fods[0]
    assert isinstance(sh, np.ndarray)
    assert sh.shape == mask.shape + (N_SH,)
    assert results.tissue_fractions is None
    assert np.all(sh[~mask] == 0)
    assert np.isfinite(sh[mask]).all()


def test_single_tissue_fit_does_not_depend_on_chunks_or_cores(
    small_dwi: DWI, monkeypatch: pytest.MonkeyPatch
) -> None:
    volume, gtab, mask = small_dwi
    sd = SDModel([_wm_rf()], progress=False)
    expected = sd.fit(volume, gtab, mask=mask, n_jobs=1).fods[0]
    np.testing.assert_array_equal(
        sd.fit(volume, gtab, mask=mask, n_jobs=3).fods[0], expected
    )
    # The mask holds 32 voxels: chunks of 5 leave a partial last chunk.
    monkeypatch.setattr(sd_module, "FIT_CHUNK", 5)
    np.testing.assert_array_equal(
        sd.fit(volume, gtab, mask=mask, n_jobs=2).fods[0], expected
    )


def test_multi_tissue_fit_does_not_depend_on_chunks_or_cores(
    small_dwi: DWI, monkeypatch: pytest.MonkeyPatch
) -> None:
    volume, gtab, mask = small_dwi
    rfs = [_wm_rf(), IsotropicRF(diffusivity=0.7e-3), IsotropicRF(diffusivity=3e-3)]
    sd = SDModel(rfs, progress=False)
    expected = sd.fit(volume, gtab, mask=mask, n_jobs=1)
    many_cores = sd.fit(volume, gtab, mask=mask, n_jobs=3)
    # The mask holds 32 voxels: chunks of 5 leave a partial last chunk.
    monkeypatch.setattr(sd_module, "FIT_CHUNK", 5)
    small_chunks = sd.fit(volume, gtab, mask=mask, n_jobs=2)
    for fit in (many_cores, small_chunks):
        np.testing.assert_array_equal(fit.fods[0], expected.fods[0])
        assert fit.tissue_fractions is not None
        assert expected.tissue_fractions is not None
        np.testing.assert_array_equal(fit.tissue_fractions, expected.tissue_fractions)


@pytest.mark.parametrize("solver", [None, DRL()], ids=["default", "drl"])
def test_mfod_fits_one_fod_per_anisotropic_response(
    small_dwi: DWI, solver: DRL | None
) -> None:
    volume, gtab, mask = small_dwi
    # Three voxels: the checks are on shapes and fractions, and each voxel is slow.
    mask = np.zeros_like(mask)
    mask[2, 0, :3] = True
    rfs: list[ResponseFunction] = [
        _wm_rf(),
        AnisotropicRF(eigen_vals=[1.5e-3, 0.9e-3, 0.1e-3], mean_kurtosis=0.0),
        IsotropicRF(diffusivity=3e-3),
    ]
    sd = SDModel(rfs, method=SDMethod.MFOD, solver=solver, progress=False)

    fit = sd.fit(volume, gtab, mask=mask, n_jobs=1)

    assert len(fit.fods) == 2
    for sh in fit.fods:
        assert sh.shape == mask.shape + (N_SH,)
        assert np.isfinite(sh[mask]).all()
    assert not np.array_equal(fit.fods[0], fit.fods[1])
    assert fit.tissue_fractions is not None
    assert fit.tissue_fractions.shape == mask.shape + (3,)
    np.testing.assert_allclose(fit.tissue_fractions[mask].sum(axis=-1), 1.0, atol=1e-5)


def _b_vals(n_dirs: int) -> npt.NDArray[np.float64]:
    # b=0 and lower-shell volumes must not count towards the highest shell.
    return np.concatenate([np.zeros(3), np.full(50, 1000.0), np.full(n_dirs, 3000.0)])


@pytest.mark.parametrize(
    ("proposed", "n_dirs", "expected"),
    [(10, 60, 8), (8, 100, 8), (8, 20, 4), (2, 6, 2)],
)
def test_max_lmax_is_the_largest_even_degree_the_highest_shell_fits(
    proposed: int, n_dirs: int, expected: int
) -> None:
    assert sd_module.compute_max_lmax(_b_vals(n_dirs), proposed) == expected


def test_max_lmax_refuses_a_shell_too_small_for_lmax_2() -> None:
    with pytest.raises(ValueError, match="lmax=2"):
        sd_module.compute_max_lmax(_b_vals(5), 8)


@pytest.mark.parametrize(
    ("asked", "n_dirs", "used", "warning"),
    [(10, 60, 8, "lowering to 8"), (8, 100, 8, None)],
)
def test_lowered_lmax_is_reported_with_the_degree_used(
    asked: int,
    n_dirs: int,
    used: int,
    warning: str | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="larmor"):
        assert SDModel([_wm_rf()], lmax=asked)._resolve_lmax(_b_vals(n_dirs)) == used
    if warning is None:
        assert "lowering" not in caplog.text
    else:
        assert warning in caplog.text


@pytest.mark.parametrize("lmax", [9, 1, 0])
def test_sd_model_refuses_an_odd_lmax_or_one_below_2(lmax: int) -> None:
    with pytest.raises(ValueError, match="lmax must be even and at least 2"):
        SDModel([_wm_rf()], lmax=lmax)


@pytest.mark.parametrize("n_meas", [33, 200])
def test_low_resolution_directions_are_all_different(n_meas: int) -> None:
    rfs = [_wm_rf(), NoddiRF()]
    for dirs in sd_module.low_resolution_dirs(rfs, n_meas):
        cosines = np.abs(dirs @ dirs.T)
        np.fill_diagonal(cosines, 0.0)
        assert cosines.max() < 1 - 1e-6


def test_sd_model_refuses_an_unknown_basis_when_it_is_created() -> None:
    with pytest.raises(ValueError, match="Invalid basis mrtrix"):
        SDModel([_wm_rf()], basis="mrtrix")


@pytest.mark.parametrize("wm_fraction", [1.0, 0.6])
def test_fod_integral_is_the_white_matter_signal_fraction(wm_fraction: float) -> None:
    wm, csf = _wm_rf(), IsotropicRF(diffusivity=3e-3)
    shell = get_dirs(64)
    bvecs = np.concatenate([np.zeros((2, 3)), shell, shell])
    bvals = np.concatenate([np.zeros(2), np.full(64, 1000.0), np.full(64, 3000.0)])
    # Off the fit's directions, oblique, and on one of them.
    fibres = np.array([[1.0, 0.0, 0.0], [0.3, 0.5, 0.8], DIRS_300[17]])
    fibres /= np.linalg.norm(fibres, axis=1, keepdims=True)
    signal = wm_fraction * wm.signal(bvals, bvecs, fibres).T
    signal += (1 - wm_fraction) * csf.signal(bvals, bvecs, fibres).T
    rfs = [wm] if wm_fraction == 1.0 else [wm, csf]
    volume = Volume(signal.reshape(1, 1, 3, -1).astype(np.float32), np.eye(4))

    fit = SDModel(rfs, progress=False).fit(
        volume, GradientTable(bvals, bvecs), n_jobs=1
    )

    # dRL stops after a fixed number of iterations, so its weights sum to the
    # fraction only to about 2% here.
    integral = fit.fods[0][0, 0, :, 0] * np.sqrt(4 * np.pi)
    np.testing.assert_allclose(integral, wm_fraction, rtol=0.03)


@pytest.mark.parametrize("solver", [L2NNLS(), Lasso()], ids=["l2nnls", "lasso"])
def test_sparse_fod_integral_is_the_sum_of_its_weights(solver: SDSolver) -> None:
    wm = _wm_rf()
    shell = get_dirs(64)
    bvecs = np.concatenate([np.zeros((2, 3)), shell, shell])
    bvals = np.concatenate([np.zeros(2), np.full(64, 1000.0), np.full(64, 3000.0)])
    fibres = np.array([[1.0, 0.0, 0.0], [0.3, 0.5, 0.8], DIRS_300[17]])
    fibres /= np.linalg.norm(fibres, axis=1, keepdims=True)
    volume = Volume(
        wm.signal(bvals, bvecs, fibres).T.reshape(1, 1, 3, -1).astype(np.float32),
        np.eye(4),
    )
    model = SDModel([wm], solver=solver, progress=False)
    gtab = GradientTable(bvals, bvecs)
    signals = volume.data.reshape(3, -1).astype(np.float64)
    signals /= signals[:, :2].mean(axis=1, keepdims=True)
    _, fit_chunk = model._prepare(solver, bvals, bvecs, 1)
    weights, *_ = fit_chunk(signals)

    fit = model.fit(volume, gtab, n_jobs=1)

    integral = fit.fods[0][0, 0, :, 0] * np.sqrt(4 * np.pi)
    np.testing.assert_allclose(integral, weights.sum(axis=1), rtol=1e-6)


def test_lasso_fod_keeps_the_isotropic_part_of_the_signal() -> None:
    wm = _wm_rf()
    shell = get_dirs(64)
    bvecs = np.concatenate([np.zeros((2, 3)), shell, shell])
    bvals = np.concatenate([np.zeros(2), np.full(64, 1000.0), np.full(64, 3000.0)])
    fibre = np.array([[0.3, 0.5, 0.8]]) / np.linalg.norm([0.3, 0.5, 0.8])
    volume = Volume(
        wm.signal(bvals, bvecs, fibre).T.reshape(1, 1, 1, -1).astype(np.float32),
        np.eye(4),
    )

    fit = SDModel([wm], solver=Lasso(), progress=False).fit(
        volume, GradientTable(bvals, bvecs), n_jobs=1
    )

    # The penalty shrinks the FOD a little; the integral of a fit that drops the
    # constant part of the signal was about 0.78.
    integral = fit.fods[0][0, 0, 0, 0] * np.sqrt(4 * np.pi)
    assert 0.85 < integral <= 1.0


def test_default_lasso_separates_a_45_degree_crossing_among_isotropic_tissues() -> None:
    wm = _wm_rf()
    shell = get_dirs(64)
    bvecs = np.concatenate([np.zeros((4, 3)), shell, shell])
    bvals = np.concatenate([np.zeros(4), np.full(64, 1000.0), np.full(64, 3000.0)])
    first = np.array([1.0, 0.0, 0.3]) / np.linalg.norm([1.0, 0.0, 0.3])
    a = np.radians(45)
    second = np.array([np.cos(a), np.sin(a), 0.3])
    second /= np.linalg.norm(second)
    fibres = wm.signal(bvals, bvecs, np.stack([first, second])).mean(axis=1)
    signal = 0.7 * fibres + 0.2 * np.exp(-bvals * 0.8e-3) + 0.1 * np.exp(-bvals * 3e-3)
    volume = Volume(signal.reshape(1, 1, 1, -1).astype(np.float32), np.eye(4))
    model = SDModel(
        [wm, IsotropicRF(0.8e-3), IsotropicRF(3e-3)], solver=Lasso(), progress=False
    )

    fit = model.fit(volume, GradientTable(bvals, bvecs), n_jobs=1)

    peaks = find_peaks(
        fit.fods[0], relative_threshold=fit.peak_relative_threshold, num_peaks=4
    )
    found = peaks.directions[0, 0, 0][np.isfinite(peaks.amplitudes[0, 0, 0])]
    assert len(found) == 2
    errors = np.degrees(np.arccos(np.abs(found @ np.stack([first, second]).T).max(0)))
    assert np.all(errors < 5.0)


@pytest.mark.parametrize("multi", [False, True], ids=["single", "multi"])
def test_lasso_reports_voxels_that_did_not_converge(
    small_dwi: DWI, multi: bool, caplog: pytest.LogCaptureFixture
) -> None:
    volume, gtab, mask = small_dwi
    rfs = [_wm_rf(), IsotropicRF(diffusivity=3e-3)] if multi else [_wm_rf()]
    model = SDModel(rfs, solver=Lasso(reg=1e-4, tol=1e-12, max_iter=1))

    with caplog.at_level(logging.WARNING):
        model.fit(volume, gtab, mask=mask, n_jobs=2)

    assert "elastic net solves did not converge" in caplog.text


@pytest.mark.parametrize("method", ["GRL", "mFOD"])
def test_crossing_gives_two_peaks_at_the_fits_relative_threshold(method: str) -> None:
    wm = _wm_rf()
    shell = get_dirs(64)
    bvecs = np.concatenate([np.zeros((2, 3)), shell, shell])
    bvals = np.concatenate([np.zeros(2), np.full(64, 1000.0), np.full(64, 3000.0)])
    fibres = np.array([[1.0, 0.0, 0.3], [0.0, 1.0, 0.3]])
    fibres /= np.linalg.norm(fibres, axis=1, keepdims=True)
    signal = wm.signal(bvals, bvecs, fibres).mean(axis=1)
    volume = Volume(signal.reshape(1, 1, 1, -1).astype(np.float32), np.eye(4))

    fit = SDModel([wm], method=method, progress=False).fit(
        volume, GradientTable(bvals, bvecs), n_jobs=1
    )
    peaks = find_peaks(
        fit.fods[0], num_peaks=6, relative_threshold=fit.peak_relative_threshold
    )

    # At lmax 8 the sharp FOD rings, and the rings of the two fibres add up to
    # 12-14% of the main lobes in four places.
    assert fit.peak_relative_threshold == 0.2
    assert np.isfinite(peaks.amplitudes).sum() == 2


def test_fod_weights_summing_to_a_fraction_give_that_integral() -> None:
    n = len(DIRS_300)
    weights = np.full((1, n), 0.7 / n)

    sh = sd_module._to_sh(weights, 8, "tournier")

    # A constant is fitted exactly, so only rounding separates the two.
    assert sh[0, 0] * np.sqrt(4 * np.pi) == pytest.approx(0.7, rel=1e-12)


TISSUES = pytest.mark.parametrize(
    "rfs",
    [
        [_wm_rf()],
        [_wm_rf(), IsotropicRF(diffusivity=0.7e-3), IsotropicRF(diffusivity=3e-3)],
    ],
    ids=["single", "multi"],
)


@TISSUES
def test_voxels_with_a_nan_or_inf_signal_get_nan_and_the_rest_is_unchanged(
    small_dwi: DWI, rfs: list[ResponseFunction], caplog: pytest.LogCaptureFixture
) -> None:
    volume, gtab, mask = small_dwi
    data = volume.data.copy()
    dwi_index = int(np.flatnonzero(~gtab.b0_mask)[0])
    b0_index = int(np.flatnonzero(gtab.b0_mask)[0])
    bad = [tuple(int(i) for i in v) for v in np.argwhere(mask)[:3]]
    # A NaN in a b0 reaches every measurement through the b0 mean.
    data[bad[0] + (dwi_index,)] = np.nan
    data[bad[1] + (b0_index,)] = np.nan
    data[bad[2] + (dwi_index,)] = np.inf
    bad_mask = np.zeros_like(mask)
    for v in bad:
        bad_mask[v] = True
    sd = SDModel(rfs, progress=False)

    with caplog.at_level(logging.WARNING, logger="larmor"):
        fit = sd.fit(volume.with_data(data), gtab, mask=mask, n_jobs=1)
    expected = sd.fit(volume, gtab, mask=mask & ~bad_mask, n_jobs=1)

    assert "3 voxels have a NaN or infinite signal" in caplog.text
    assert np.isnan(fit.fods[0][bad_mask]).all()
    # Voxels are fitted independently, so the good ones match bit for bit.
    np.testing.assert_array_equal(fit.fods[0][~bad_mask], expected.fods[0][~bad_mask])
    if len(rfs) > 1:
        assert fit.tissue_fractions is not None
        assert expected.tissue_fractions is not None
        assert np.isnan(fit.tissue_fractions[bad_mask]).all()
        np.testing.assert_array_equal(
            fit.tissue_fractions[~bad_mask], expected.tissue_fractions[~bad_mask]
        )
    else:
        assert fit.tissue_fractions is None


@TISSUES
def test_a_volume_of_nan_gives_nan_without_error(
    small_dwi: DWI, rfs: list[ResponseFunction]
) -> None:
    volume, gtab, mask = small_dwi
    nan_volume = volume.with_data(np.full_like(volume.data, np.nan))

    fit = SDModel(rfs, progress=False).fit(nan_volume, gtab, mask=mask, n_jobs=1)

    assert np.isnan(fit.fods[0][mask]).all()
    assert np.all(fit.fods[0][~mask] == 0)
    if len(rfs) > 1:
        assert fit.tissue_fractions is not None
        assert np.isnan(fit.tissue_fractions[mask]).all()


@pytest.mark.parametrize("n_jobs", [0, -2])
def test_fit_refuses_n_jobs_of_zero_or_below_minus_one(
    small_dwi: DWI, n_jobs: int
) -> None:
    volume, gtab, mask = small_dwi
    with pytest.raises(ValueError, match="n_jobs must be -1"):
        SDModel([_wm_rf()]).fit(volume, gtab, mask=mask, n_jobs=n_jobs)


def test_fit_accepts_a_non_boolean_mask_or_a_mask_volume(small_dwi: DWI) -> None:
    # A uint8 mask (as read from a NIfTI file) must select the same voxels.
    volume, gtab, mask = small_dwi
    sd = SDModel([_wm_rf()])
    expected = sd.fit(volume, gtab, mask=mask, n_jobs=1).fods[0]

    as_uint8 = mask.astype(np.uint8)
    np.testing.assert_array_equal(
        sd.fit(volume, gtab, mask=as_uint8, n_jobs=1).fods[0], expected
    )
    np.testing.assert_array_equal(
        sd.fit(volume, gtab, mask=volume.with_data(as_uint8), n_jobs=1).fods[0],
        expected,
    )


def test_fit_uses_the_scaled_values(small_dwi: DWI) -> None:
    # The fit sees data * slope + intercept, not the stored numbers.
    volume, gtab, mask = small_dwi
    sd = SDModel([_wm_rf()])
    scaled = Volume(volume.data / 2, np.eye(4), slope=2.0)
    np.testing.assert_allclose(
        sd.fit(scaled, gtab, mask=mask, n_jobs=1).fods[0],
        sd.fit(volume, gtab, mask=mask, n_jobs=1).fods[0],
        rtol=1e-5,
        atol=1e-7,
    )


def test_models_refuse_a_table_of_another_length(small_dwi: DWI) -> None:
    volume, gtab, mask = small_dwi
    short = GradientTable(gtab.bvals[:-1], gtab.bvecs[:-1])
    with pytest.raises(ValueError, match="entries"):
        SDModel([_wm_rf()]).fit(volume, short, mask=mask, n_jobs=1)
    with pytest.raises(ValueError, match="entries"):
        DTIModel().fit(volume, short, mask)
    with pytest.raises(ValueError, match="4D"):
        DTIModel().fit(volume.with_data(volume.data[..., 0]), gtab, mask)


def test_kernel_needs_dirs_for_an_anisotropic_response() -> None:
    bvals = np.array([0.0, 1000.0, 1000.0])
    bvecs = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    with pytest.raises(ValueError, match="dirs"):
        Kernel([_wm_rf()], bvals, bvecs, dirs=None)
    # Isotropic responses have no direction, so they need none.
    assert Kernel([IsotropicRF(diffusivity=3e-3)], bvals, bvecs, dirs=None).k.shape == (
        3,
        1,
    )


def test_mean_b0_refuses_data_without_b0() -> None:
    volume = Volume(np.ones((2, 2, 2, 3), np.float32), np.eye(4))
    gtab = GradientTable(np.full(3, 1000.0), np.eye(3))

    with pytest.raises(ValueError, match="No b0"):
        mean_b0(volume, gtab)


def test_mean_b0_uses_the_table_threshold() -> None:
    data = np.stack([np.full((2, 2, 2), v) for v in (10.0, 20.0, 30.0)], axis=-1)
    volume = Volume(data.astype(np.float32), np.eye(4))
    bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    bvals = np.array([0.0, 80.0, 1000.0])
    np.testing.assert_allclose(mean_b0(volume, GradientTable(bvals, bvecs)), 10.0)
    np.testing.assert_allclose(
        mean_b0(volume, GradientTable(bvals, bvecs, b0_threshold=100.0)), 15.0
    )


def test_dti_fits_data_without_b0_given_a_mask() -> None:
    # Two shells without b0: the fit gets log S0 as its intercept and needs no b0
    # (one shell alone would not separate S0 from the trace of D). Only the
    # automatic mask needs a b0.
    dirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    bvecs = np.vstack([dirs, dirs])
    bvals = np.repeat([500.0, 1000.0], 6)
    signal = 1000.0 * np.exp(-bvals * 0.8e-3)
    data = np.broadcast_to(signal, (2, 2, 2, 12)).astype(np.float32)
    volume = Volume(data, np.eye(4))
    gtab = GradientTable(bvals, bvecs)

    fit = DTIModel().fit(volume, gtab, np.ones((2, 2, 2), dtype=bool))

    np.testing.assert_allclose(fit.md, 0.8e-3, atol=1e-7)
    with pytest.raises(ValueError, match="No b0"):
        DTIModel().fit(volume, gtab)  # the automatic mask needs a b0

    # One shell alone cannot separate S0 from the trace of D.
    one_shell = GradientTable(np.full(12, 1000.0), bvecs)
    with pytest.raises(ValueError, match="cannot determine a tensor"):
        DTIModel().fit(volume, one_shell, np.ones((2, 2, 2), dtype=bool))


def _fits(volume: Volume, gtab: GradientTable, mask: Volume | np.ndarray) -> None:
    DTIModel().fit(volume, gtab, mask)


def _sd_fits(volume: Volume, gtab: GradientTable, mask: Volume | np.ndarray) -> None:
    SDModel([_wm_rf()]).fit(volume, gtab, mask=mask, n_jobs=1)


def _rf(volume: Volume, gtab: GradientTable, mask: Volume | np.ndarray) -> None:
    estimate_wm_rf(volume, gtab, mask)


MASK_USERS = pytest.mark.parametrize(
    "use", [_fits, _sd_fits, _rf], ids=["dti", "sd", "wm_rf"]
)


@MASK_USERS
def test_mask_volume_in_another_orientation_is_reoriented_with_a_warning(
    small_dwi: DWI,
    use: Callable[[Volume, GradientTable, Volume], None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    volume, gtab, mask = small_dwi
    flipped = volume.with_data(mask.astype(np.uint8))
    flipped.flip_axis(0)
    with caplog.at_level(logging.WARNING, logger="larmor"):
        use(volume, gtab, flipped)
    assert "stored LAS, the volume RAS; reorienting" in caplog.text


def test_a_reoriented_mask_selects_the_same_voxels(small_dwi: DWI) -> None:
    # The same voxels stored with x flipped: using its data as it is would mirror
    # the mask onto the DWI.
    volume, _, mask = small_dwi
    mask = mask.copy()
    mask[0, 0, 3] = True  # not symmetric in x, so a mirrored mask would differ
    flipped = volume.with_data(mask.astype(np.uint8))
    flipped.flip_axis(0)
    np.testing.assert_array_equal(as_mask(volume, flipped), mask)


@MASK_USERS
def test_mask_volume_on_another_grid_is_refused(
    small_dwi: DWI, use: Callable[[Volume, GradientTable, Volume], None]
) -> None:
    volume, gtab, mask = small_dwi
    shifted = np.eye(4)
    shifted[0, 3] = 5.0
    with pytest.raises(ValueError, match="not on the volume.s grid"):
        use(volume, gtab, Volume(mask.astype(np.uint8), shifted))


@MASK_USERS
@pytest.mark.parametrize("shape", [(4, 4, 3), (4, 4, 4, 1), (4, 4)])
def test_mask_array_of_another_shape_is_refused(
    small_dwi: DWI,
    use: Callable[[Volume, GradientTable, np.ndarray], None],
    shape: tuple[int, ...],
) -> None:
    volume, gtab, _ = small_dwi
    with pytest.raises(ValueError, match="mask has shape"):
        use(volume, gtab, np.ones(shape, dtype=bool))


def test_inner_shell_weight_changes_a_multi_tissue_fit() -> None:
    wm, csf = _wm_rf(), IsotropicRF(diffusivity=3e-3)
    shell = get_dirs(64)
    bvecs = np.concatenate([np.zeros((2, 3)), shell, shell])
    bvals = np.concatenate([np.zeros(2), np.full(64, 1000.0), np.full(64, 3000.0)])
    fibres = np.array([[1.0, 0.0, 0.0], [0.3, 0.5, 0.8]])
    fibres /= np.linalg.norm(fibres, axis=1, keepdims=True)
    signal = 0.7 * wm.signal(bvals, bvecs, fibres).T
    signal += 0.3 * csf.signal(bvals, bvecs, fibres).T
    # Noise keeps the shells from agreeing exactly, so their weights matter.
    signal += 0.02 * np.random.default_rng(0).standard_normal(signal.shape)
    volume = Volume(signal.reshape(1, 1, 2, -1).astype(np.float32), np.eye(4))
    gtab = GradientTable(bvals, bvecs)

    def fit(weight: float) -> npt.NDArray[np.floating]:
        model = SDModel([wm, csf], inner_shell_weight=weight, progress=False)
        return model.fit(volume, gtab, n_jobs=1).fods[0]

    assert not np.allclose(fit(1.0), fit(0.2))


@pytest.mark.parametrize("b0_scale", [0.0, 0.01], ids=["zero_b0s", "dark_b0s"])
def test_voxels_with_dark_b0s_get_a_plausible_fod_and_the_rest_is_unchanged(
    b0_scale: float,
) -> None:
    wm = _wm_rf()
    shell = get_dirs(64)
    bvecs = np.concatenate([np.zeros((2, 3)), shell])
    bvals = np.concatenate([np.zeros(2), np.full(64, 1000.0)])
    gtab = GradientTable(bvals, bvecs)
    fibres = np.array([[1.0, 0.0, 0.0], [0.3, 0.5, 0.8], [0.0, 0.0, 1.0]])
    fibres /= np.linalg.norm(fibres, axis=1, keepdims=True)
    healthy = 1000.0 * wm.signal(bvals, bvecs, fibres).T
    dark = healthy[:1].copy()
    dark[:, gtab.b0_mask] *= b0_scale
    data = np.concatenate([healthy, dark]).reshape(1, 1, 4, -1).astype(np.float32)
    sd = SDModel([wm], progress=False)

    fit = sd.fit(Volume(data, np.eye(4)), gtab, n_jobs=1).fods[0][0, 0]
    alone = sd.fit(Volume(data[:, :, :3], np.eye(4)), gtab, n_jobs=1).fods[0]

    # Without the clean-up its integral is about 90 (dark) or 9e18 (zero).
    assert 0 < fit[3, 0] * np.sqrt(4 * np.pi) < 2
    np.testing.assert_array_equal(fit[:3], alone[0, 0])


def _one_fibre_signal() -> tuple[npt.NDArray[np.float64], GradientTable]:
    shell = get_dirs(64)
    bvecs = np.concatenate([np.zeros((2, 3)), shell])
    bvals = np.concatenate([np.zeros(2), np.full(64, 1000.0)])
    signal = 1000.0 * _wm_rf().signal(bvals, bvecs, np.array([[1.0, 0.0, 0.0]])).T
    return signal, GradientTable(bvals, bvecs)


def test_a_measurement_above_s0_is_fitted_as_one_at_s0() -> None:
    signal, gtab = _one_fibre_signal()
    dw_index = int(np.flatnonzero(~gtab.b0_mask)[0])
    above, at = signal.copy(), signal.copy()
    above[0, dw_index] = 2000.0
    at[0, dw_index] = 1000.0
    data = np.concatenate([above, at]).reshape(1, 1, 2, -1).astype(np.float32)

    fod = SDModel([_wm_rf()], progress=False).fit(
        Volume(data, np.eye(4)), gtab, n_jobs=1
    )

    np.testing.assert_array_equal(fod.fods[0][0, 0, 0], fod.fods[0][0, 0, 1])


@pytest.mark.parametrize("b0_scale", [0.0, 0.01], ids=["zero_b0s", "dark_b0s"])
def test_dark_b0_voxel_without_plausible_neighbour_gets_nan(
    b0_scale: float, caplog: pytest.LogCaptureFixture
) -> None:
    signal, gtab = _one_fibre_signal()
    dark = signal.copy()
    dark[:, gtab.b0_mask] *= b0_scale
    # The plausible voxel is three steps away, outside even the 5x5x5 window, and
    # the voxels between are outside the mask.
    data = np.concatenate([signal, signal, signal, dark]).reshape(4, 1, 1, -1)
    mask = np.array([True, False, False, True]).reshape(4, 1, 1)

    with caplog.at_level(logging.WARNING, logger="larmor"):
        fit = SDModel([_wm_rf()], progress=False).fit(
            Volume(data.astype(np.float32), np.eye(4)), gtab, mask=mask, n_jobs=1
        )

    assert "1 voxels have a NaN or infinite signal or no S0" in caplog.text
    assert np.isnan(fit.fods[0][3]).all()
    assert np.isfinite(fit.fods[0][0]).all()
