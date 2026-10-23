"""Constrained spherical deconvolution (CSD) and its response, against MRtrix3 too."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from larmor.io import load_dwi, load_volume, read_mrtrix_response
from larmor.sd import (
    CSDModel,
    CSDSolver,
    IsotropicRF,
    ZonalSHRF,
    build_csd_response,
    estimate_csd_rf,
    find_peaks,
)
from larmor.sd.sh import build_basis_matrix
from larmor.sd.sphere import DIRS_300, get_dirs, get_theta_phi
from larmor.types import GradientTable, Volume

DATA = Path(__file__).parents[1] / "data"
BLOCK = DATA / "grl" / "block_dwi.nii.gz"
BLOCK_MASK = DATA / "grl" / "block_dwi_brain_mask.nii.gz"
BVALS, BVECS = DATA / "bvals.txt", DATA / "bvecs.txt"

needs_mrtrix = pytest.mark.skipif(
    shutil.which("dwi2fod") is None or shutil.which("dwi2response") is None,
    reason="MRtrix3 (dwi2fod, dwi2response) is not on PATH",
)

# A white matter response at b = 1000 in signal units: zonal SH l = 0, 2, 4, 6, 8.
RESPONSE = np.array([1800.0, -700.0, 180.0, -35.0, 5.0])


def _gradients(n_dirs: int = 60) -> GradientTable:
    rng = np.random.default_rng(1)
    dirs = rng.normal(size=(n_dirs, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return GradientTable(np.r_[0.0, [1000.0] * n_dirs], np.r_[np.zeros((1, 3)), dirs])


def _voxels(fibre_sets: list[np.ndarray], noise: float = 0.0) -> Volume:
    """One voxel per set of fibres (k, 3), each holding equal parts of them."""
    gtab = _gradients()
    rf = ZonalSHRF(bvals=np.array([1000.0]), coefficients=RESPONSE[None])
    rows = [
        np.r_[2000.0, rf.signal(gtab.bvals[1:], gtab.bvecs[1:], f).mean(axis=1)]
        for f in fibre_sets
    ]
    data = np.array(rows)
    data[:, 1:] += np.random.default_rng(2).normal(0, noise, data[:, 1:].shape)
    return Volume(data.reshape(len(rows), 1, 1, -1), np.eye(4))


def _fit(fibres: np.ndarray, noise: float = 0.0, max_iter: int = 50) -> np.ndarray:
    rf = ZonalSHRF(bvals=np.array([1000.0]), coefficients=RESPONSE[None])
    model = CSDModel(rf, lmax=8, solver=CSDSolver(max_iter=max_iter))
    fod = model.fit(_voxels([fibres], noise), _gradients(), n_jobs=1).fods[0]
    voxel: np.ndarray = fod[0, 0, 0].astype(np.float64)
    return voxel


@pytest.mark.parametrize("angle", [90.0, 60.0])
def test_one_fibre_and_a_crossing_recover_their_directions(angle: float) -> None:
    rad = np.radians(angle)
    crossing = np.array([[1.0, 0.0, 0.0], [np.cos(rad), np.sin(rad), 0.0]])
    for fibres in (crossing[:1], crossing):
        peaks = find_peaks(_fit(fibres), num_peaks=3).directions
        found = peaks[np.isfinite(peaks[:, 0])]
        assert len(found) == len(fibres)
        cosines = np.abs(found @ fibres.T).max(axis=0)
        assert np.degrees(np.arccos(cosines.clip(max=1))).max() < 2.0


def test_constraint_suppresses_negative_lobes() -> None:
    fibres = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    basis = build_basis_matrix(*get_theta_phi(get_dirs(2000)), 8)
    linear = basis @ _fit(fibres, noise=40.0, max_iter=0)
    constrained = basis @ _fit(fibres, noise=40.0)

    # The constraint is a penalty, not a hard bound: small negative values remain.
    negative_linear = -linear.min() / linear.max()
    negative_constrained = -constrained.min() / constrained.max()
    assert negative_constrained < negative_linear / 3


def test_csd_takes_only_a_zonal_response() -> None:
    with pytest.raises(ValueError, match="ZonalSHRF"):
        CSDModel(IsotropicRF(3e-3))  # type: ignore[arg-type]


def test_settings_that_cannot_be_solved_are_refused() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        CSDSolver(directions=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))
    # lmax 10 has 66 coefficients for the 60 measurements: singular without the norm.
    rf = ZonalSHRF(bvals=np.array([1000.0]), coefficients=RESPONSE[None])
    model = CSDModel(rf, lmax=10, solver=CSDSolver(norm_lambda=0.0))
    with pytest.raises(ValueError, match="singular"):
        model.fit(_voxels([np.eye(3)[:1]]), _gradients(), n_jobs=1)


def test_estimated_response_is_that_of_the_single_fibres() -> None:
    rng = np.random.default_rng(3)
    singles = rng.normal(size=(60, 3))
    singles /= np.linalg.norm(singles, axis=1, keepdims=True)
    crossings = [np.array([d, np.cross(d, [0.0, 0.0, 1.0])]) for d in singles[:40]]
    crossings = [c / np.linalg.norm(c, axis=1, keepdims=True) for c in crossings]
    # Shuffled, so that a selection by position rather than by score shows.
    voxels = [d[None] for d in singles] + crossings
    order = rng.permutation(len(voxels))
    volume = _voxels([voxels[i] for i in order])
    is_single = order < len(singles)

    estimate = estimate_csd_rf(
        volume, _gradients(), np.ones(volume.data.shape[:3]), num_voxels=30, n_jobs=1
    )

    assert estimate.converged
    chosen = estimate.voxels[:, 0, 0]
    assert chosen.sum() == 30 and not (chosen & ~is_single).any()
    found = estimate.rf.coefficients[0]
    np.testing.assert_allclose(found[:5], RESPONSE, atol=1e-3 * RESPONSE[0])
    assert np.abs(found[5:]).max() < 1e-3 * RESPONSE[0]


def _block() -> tuple[Volume, GradientTable, Volume]:
    volume, gtab = load_dwi(BLOCK, BVALS, BVECS)
    return volume, gtab, load_volume(BLOCK_MASK)


def _same_grid(path: Path, like: Volume) -> np.ndarray:
    """An MRtrix output on the grid and in the orientation of `like`."""
    result = load_volume(path)
    result.to_orientation(like.orientation)
    data: np.ndarray = result.data
    return data


@needs_mrtrix
@pytest.mark.parametrize("lmax", [8, 6])
def test_matches_dwi2fod_csd(tmp_path: Path, lmax: int) -> None:
    # The ISMRM 2015 block, with the response dwi2response tournier gives on the
    # whole image; both tools apply the constraint on the same 300 directions.
    response = DATA / "csd" / "wm_response.txt"
    theta, phi = get_theta_phi(DIRS_300 / np.linalg.norm(DIRS_300, axis=1)[:, None])
    directions = tmp_path / "dirs.txt"
    np.savetxt(directions, np.c_[phi, theta], fmt="%.10f")
    out = tmp_path / "fod.nii"
    subprocess.run(
        [
            "dwi2fod", "csd", str(BLOCK), str(response), str(out),
            "-fslgrad", str(BVECS), str(BVALS), "-mask", str(BLOCK_MASK),
            "-lmax", str(lmax), "-directions", str(directions), "-quiet",
        ],
        check=True,
    )  # fmt: skip

    volume, gtab, mask = _block()
    rf = build_csd_response(read_mrtrix_response(response), gtab)
    fit = CSDModel(rf, lmax=lmax).fit(volume, gtab, mask=mask)
    inside = mask.data > 0
    ours = fit.fods[0][inside].astype(np.float64)
    ref = _same_grid(out, volume)[inside].astype(np.float64)
    rel = np.linalg.norm(ours - ref, axis=1) / np.linalg.norm(ref, axis=1)
    # Both are single precision: the difference is rounding.
    assert rel.max() < 1e-5


@needs_mrtrix
def test_estimated_response_matches_dwi2response_tournier(tmp_path: Path) -> None:
    response, voxels = tmp_path / "response.txt", tmp_path / "voxels.nii"
    subprocess.run(
        [
            "dwi2response", "tournier", str(BLOCK), str(response),
            "-fslgrad", str(BVECS), str(BVALS), "-mask", str(BLOCK_MASK),
            "-number", "20", "-iter_voxels", "60", "-voxels", str(voxels),
            "-scratch", str(tmp_path), "-quiet",
        ],
        check=True,
    )  # fmt: skip

    volume, gtab, mask = _block()
    estimate = estimate_csd_rf(volume, gtab, mask, num_voxels=20, num_candidates=60)
    ref_voxels = _same_grid(voxels, volume) > 0
    overlap = (estimate.voxels & ref_voxels).sum() / ref_voxels.sum()
    assert overlap >= 0.9
    ref = read_mrtrix_response(response)[0]
    found = estimate.rf.coefficients[0]
    assert np.abs(found - ref).max() < 0.01 * ref[0]
