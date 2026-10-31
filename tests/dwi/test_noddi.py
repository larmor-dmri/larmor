from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from larmor.dwi import NoddiFit, NoddiModel, noddi
from larmor.io import load_volume, read_bvals_bvecs, read_matlab
from larmor.types import GradientTable, Volume

NODDI_DIR = Path(__file__).parents[1] / "data" / "noddi"
MAPS = ("v_ic", "od", "v_iso")
# The toolbox's names of the same maps.
TOOLBOX_MAPS = {"v_ic": "ficvf", "od": "odi", "v_iso": "fiso"}


def _mask() -> np.ndarray:
    return load_volume(NODDI_DIR / "mask.nii.gz").data > 0


def _gradients() -> GradientTable:
    return read_bvals_bvecs(
        NODDI_DIR / "bvals", NODDI_DIR / "bvecs", ref_path=NODDI_DIR / "dwi.nii.gz"
    )


@pytest.fixture(scope="module")
def truth() -> dict[str, np.ndarray]:
    return dict(np.load(NODDI_DIR / "truth.npz"))


@pytest.fixture(scope="module")
def toolbox() -> dict[str, dict[str, np.ndarray]]:
    ref = cast(dict[str, Any], read_matlab(NODDI_DIR / "noddi_ref.mat")["ref"])
    return {
        name: {
            ours: np.asarray(ref[name][theirs]) for ours, theirs in TOOLBOX_MAPS.items()
        }
        for name in ("noise_free", "noisy")
    }


@pytest.fixture(scope="module")
def fits() -> dict[str, NoddiFit]:
    gtab = _gradients()
    return {
        name: NoddiModel().fit(load_volume(NODDI_DIR / f"{file}.nii.gz"), gtab, _mask())
        for name, file in (("noise_free", "dwi"), ("noisy", "dwi_noisy"))
    }


def _mean_error(
    maps: NoddiFit | dict[str, np.ndarray], truth: dict[str, np.ndarray], name: str
) -> float:
    value = getattr(maps, name) if isinstance(maps, NoddiFit) else maps[name]
    return float(np.abs(value - truth[name]).mean())


@pytest.mark.parametrize("name", MAPS)
def test_noise_free_fit_recovers_the_phantom(
    fits: dict[str, NoddiFit], truth: dict[str, np.ndarray], name: str
) -> None:
    error = np.abs(getattr(fits["noise_free"], name) - truth[name])

    assert error.max() < 1e-3


@pytest.mark.parametrize("name", MAPS)
def test_noisy_fit_errors_are_those_of_the_toolbox(
    fits: dict[str, NoddiFit],
    truth: dict[str, np.ndarray],
    toolbox: dict[str, dict[str, np.ndarray]],
    name: str,
) -> None:
    ours = _mean_error(fits["noisy"], truth, name)
    theirs = _mean_error(toolbox["noisy"], truth, name)

    assert ours < 1.1 * theirs, (ours, theirs)


@pytest.mark.parametrize("name", MAPS)
def test_noisy_fit_matches_the_toolbox_voxel_by_voxel(
    fits: dict[str, NoddiFit], toolbox: dict[str, dict[str, np.ndarray]], name: str
) -> None:
    # Both reach the best fit of the same model; they differ in the noise model.
    assert _mean_error(fits["noisy"], toolbox["noisy"], name) < 0.01


def test_toolbox_reference_recovers_the_noise_free_phantom(
    toolbox: dict[str, dict[str, np.ndarray]], truth: dict[str, np.ndarray]
) -> None:
    # Checks the reference itself, and that phantom and toolbox share conventions.
    for name in MAPS:
        assert _mean_error(toolbox["noise_free"], truth, name) < 0.001, name


def test_od_follows_from_kappa(fits: dict[str, NoddiFit]) -> None:
    fit = fits["noise_free"]

    with np.errstate(divide="ignore"):
        expected = (2 / np.pi) * np.arctan(1 / fit.kappa.astype(np.float64))
    expected[fit.v_ic == 0] = 0
    np.testing.assert_allclose(fit.od, expected, atol=1e-6)


def test_fit_is_zero_outside_the_mask() -> None:
    volume = load_volume(NODDI_DIR / "dwi.nii.gz")
    mask = np.zeros(volume.data.shape[:3], dtype=bool)
    mask[4, 2, :] = True

    fit = NoddiModel().fit(volume, _gradients(), mask)

    for name in ("v_ic", "od", "kappa", "v_iso", "mu"):
        value = getattr(fit, name)
        assert np.all(value[~mask] == 0), name
        assert np.any(value[mask] != 0), name


def test_free_water_voxel_has_no_neurites() -> None:
    gtab = _gradients()
    signal = 1000 * np.exp(-gtab.bvals * 3.0e-3)
    volume = Volume(np.tile(signal, (1, 1, 1, 1)).astype(np.float32), np.eye(4))
    mu = np.zeros((1, 1, 1, 3))
    mu[..., 2] = 1

    fit = NoddiModel().fit(volume, gtab, np.ones((1, 1, 1), dtype=bool), mu=mu)

    assert fit.v_iso[0, 0, 0] == pytest.approx(1, abs=1e-6)
    assert fit.v_ic[0, 0, 0] == 0
    assert fit.od[0, 0, 0] == 0


def test_voxel_without_a_positive_b0_gets_zeros() -> None:
    gtab = _gradients()
    volume = Volume(np.zeros((1, 1, 1, len(gtab.bvals)), dtype=np.float32), np.eye(4))
    mu = np.zeros((1, 1, 1, 3))
    mu[..., 2] = 1

    fit = NoddiModel().fit(volume, gtab, np.ones((1, 1, 1), dtype=bool), mu=mu)

    for name in ("v_ic", "od", "kappa", "v_iso"):
        assert getattr(fit, name)[0, 0, 0] == 0, name


def test_mu_of_the_wrong_shape_is_refused() -> None:
    volume = load_volume(NODDI_DIR / "dwi.nii.gz")

    with pytest.raises(ValueError, match="mu must have the shape"):
        NoddiModel().fit(volume, _gradients(), mu=np.zeros((2, 2, 2, 3)))


def test_fit_does_not_depend_on_the_number_of_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Chunks small enough that the phantom's voxels are shared among the threads.
    monkeypatch.setattr(noddi, "CHUNK_VOXELS", 16)
    volume = load_volume(NODDI_DIR / "dwi_noisy.nii.gz")
    gtab = _gradients()

    one = NoddiModel().fit(volume, gtab, _mask(), n_jobs=1)
    many = NoddiModel().fit(volume, gtab, _mask(), n_jobs=4, progress=True)

    for name in ("v_ic", "od", "kappa", "v_iso", "mu"):
        np.testing.assert_array_equal(getattr(one, name), getattr(many, name), name)


def test_n_jobs_of_zero_is_refused() -> None:
    volume = load_volume(NODDI_DIR / "dwi.nii.gz")

    with pytest.raises(ValueError, match="n_jobs"):
        NoddiModel().fit(volume, _gradients(), _mask(), n_jobs=0)
