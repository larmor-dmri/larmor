"""Tissue response estimation, on a synthetic three-tissue phantom."""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from larmor.io import (
    load_volume,
    read_mrtrix_response,
    save_volume,
    write_bvals_bvecs,
)
from larmor.sd import TissueResponses, estimate_tissue_responses
from larmor.sd.sphere import get_dirs
from larmor.types import GradientTable, Volume

needs_mrtrix = pytest.mark.skipif(
    shutil.which("dwi2response") is None, reason="MRtrix3 is not on PATH"
)

SHAPE = (26, 26, 16)
SHELLS = (1000.0, 2000.0, 3000.0)
S0 = {"wm": 1000.0, "gm": 1300.0, "csf": 3000.0}


@dataclass(frozen=True)
class Phantom:
    volume: Volume
    gtab: GradientTable
    mask: np.ndarray
    fractions: dict[str, np.ndarray]
    single_fibre: np.ndarray


def _gradients() -> GradientTable:
    """Four b0s and 30 directions on each shell, turned about z from shell to shell."""
    dirs = get_dirs(30)
    bvecs = [np.zeros((4, 3))]
    for k in range(len(SHELLS)):
        angle = 0.5 * k
        c, s = np.cos(angle), np.sin(angle)
        bvecs.append(dirs @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T)
    bvals = np.concatenate([np.zeros(4), np.repeat(SHELLS, 30)])
    return GradientTable(bvals, np.vstack(bvecs))


def _step(x: np.ndarray, edge: float) -> np.ndarray:
    """A smooth step from 0 to 1 across edge, a few voxels wide."""
    return 1 / (1 + np.exp(-(x - edge) / 0.04))


def _make_phantom() -> Phantom:
    """An ellipsoid with CSF at its centre, white matter around it and grey matter.

    The white matter fibres run around the z axis; above the middle slice a second
    fibre population along z crosses them, so the single-fibre voxels lie below it.
    """
    gtab = _gradients()
    grid = np.stack(
        np.meshgrid(*[np.arange(n) - (n - 1) / 2 for n in SHAPE], indexing="ij"), -1
    )
    radius = np.linalg.norm(grid / np.array([12.0, 12.0, 7.0]), axis=-1)
    csf = 1 - _step(radius, 0.3)
    gm = _step(radius, 0.65)
    wm = 1 - csf - gm
    mask = radius < 1.0

    around = np.stack([-grid[..., 1], grid[..., 0], np.zeros(SHAPE)], -1)
    around /= np.linalg.norm(around, axis=-1, keepdims=True).clip(1e-9)
    crossing = grid[..., 2] > 0
    b, g = gtab.bvals, gtab.bvecs

    def fibre(direction: np.ndarray) -> np.ndarray:
        cos2 = (direction @ g.T) ** 2
        decay: np.ndarray = np.exp(-b * (0.3e-3 + 1.4e-3 * cos2))
        return decay

    wm_signal = fibre(around)
    along_z = fibre(np.array([0.0, 0.0, 1.0]))
    wm_signal[crossing] = 0.5 * wm_signal[crossing] + 0.5 * along_z
    signal = (
        S0["wm"] * wm[..., None] * wm_signal
        + S0["gm"] * gm[..., None] * np.exp(-b * 0.8e-3)
        + S0["csf"] * csf[..., None] * np.exp(-b * 3e-3)
    )
    rng = np.random.default_rng(0)
    noisy = np.abs(signal + 15.0 * rng.standard_normal(signal.shape)).astype(np.float32)
    affine = np.diag([2.5, 2.5, 2.5, 1.0])
    return Phantom(
        volume=Volume(noisy * mask[..., None], affine),
        gtab=gtab,
        mask=mask,
        fractions={"wm": wm, "gm": gm, "csf": csf},
        single_fibre=~crossing,
    )


@pytest.fixture(scope="module")
def phantom() -> Phantom:
    return _make_phantom()


@pytest.fixture(scope="module")
def estimated(phantom: Phantom) -> TissueResponses:
    return estimate_tissue_responses(
        phantom.volume, phantom.gtab, phantom.mask, n_jobs=2
    )


def test_each_tissue_is_estimated_from_its_own_voxels(
    phantom: Phantom, estimated: TissueResponses
) -> None:
    assert (phantom.fractions["csf"][estimated.csf_voxels] > 0.9).all()
    # The eroded mask keeps only the inner, partial-volume edge of the grey matter.
    assert (phantom.fractions["gm"][estimated.gm_voxels] > 0.75).all()
    assert (phantom.fractions["wm"][estimated.wm_voxels] > 0.9).all()
    assert phantom.single_fibre[estimated.wm_voxels].all()


def test_responses_follow_the_tissues_signal(estimated: TissueResponses) -> None:
    # Rows b = 0 and 1000, where the signal is well above the noise.
    root = np.sqrt(4 * np.pi)
    for rf, s0, diffusivity in (
        (estimated.csf, S0["csf"], 3e-3),
        (estimated.gm, S0["gm"], 0.8e-3),
    ):
        want = root * s0 * np.exp(-np.array([0.0, 1000.0]) * diffusivity)
        np.testing.assert_allclose(rf.coefficients[:2, 0], want, rtol=0.03)
    assert estimated.wm.coefficients[0, 0] == pytest.approx(root * S0["wm"], rel=0.03)
    # The white matter signal falls along the fibre: negative l = 2 coefficients.
    assert (estimated.wm.coefficients[1:, 1] < 0).all()


@needs_mrtrix
def test_matches_dwi2response_dhollander(
    phantom: Phantom, estimated: TissueResponses, tmp_path: Path
) -> None:
    dwi, mask = tmp_path / "dwi.nii", tmp_path / "mask.nii"
    save_volume(phantom.volume, dwi)
    save_volume(Volume(phantom.mask.astype(np.uint8), phantom.volume.affine), mask)
    write_bvals_bvecs(
        phantom.gtab, tmp_path / "bvals", tmp_path / "bvecs", ref_path=dwi
    )
    out = [tmp_path / f"{tissue}.txt" for tissue in ("wm", "gm", "csf")]
    voxels = tmp_path / "voxels.nii"
    subprocess.run(
        [
            "dwi2response",
            "dhollander",
            str(dwi),
            *map(str, out),
            "-mask",
            str(mask),
            "-fslgrad",
            str(tmp_path / "bvecs"),
            str(tmp_path / "bvals"),
            "-voxels",
            str(voxels),
            "-nthreads",
            "0",
            "-quiet",
        ],
        check=True,
    )
    chosen = load_volume(voxels).data > 0  # CSF, GM, WM along the last axis
    for k, (rf, path, ours) in enumerate(
        (
            (estimated.csf, out[2], estimated.csf_voxels),
            (estimated.gm, out[1], estimated.gm_voxels),
            (estimated.wm, out[0], estimated.wm_voxels),
        )
    ):
        theirs = chosen[..., k]
        dice = 2 * (ours & theirs).sum() / (ours.sum() + theirs.sum())
        assert dice > 0.9
        reference = read_mrtrix_response(path)
        scale = np.abs(reference[:, :1])
        np.testing.assert_allclose(
            rf.coefficients / scale, reference / scale, atol=0.01
        )
