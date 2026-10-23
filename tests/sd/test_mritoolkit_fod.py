from pathlib import Path

import numpy as np
import pytest

from larmor.dwi import DTIModel
from larmor.io import load_grid, load_volume, read_bvals_bvecs, read_mritoolkit_frame
from larmor.sd import find_peaks, read_mritoolkit_fod
from larmor.sd.sh import build_basis_matrix
from larmor.sd.sphere import DIRS_300, get_theta_phi

DATA = Path(__file__).parents[1] / "data" / "mritoolkit"
DWI = DATA / "dwi.nii.gz"


@pytest.mark.parametrize("name", ["grl.mat", "grl_v7.mat"])
def test_fod_fractions_and_residuals_on_the_fit_grid(name: str) -> None:
    result = read_mritoolkit_fod(DATA / name, DATA / "dki_fit.mat")
    grid = load_grid(DWI)
    assert len(result.fods) == 1
    assert result.fods[0].data.shape == (*grid.shape, 45)
    np.testing.assert_array_equal(result.fods[0].affine, grid.affine)
    exported = load_volume(DATA / "grl_fractions.nii.gz").as_float()
    np.testing.assert_array_equal(result.tissue_fractions.data, exported)
    assert result.rss is not None and result.rss.data.shape == grid.shape


def test_the_fod_is_mritoolkits_export_on_larmors_scale() -> None:
    # MRIToolkit's SaveOutputToNii export: MRtrix-basis SH in its own frame (NIfTI
    # voxel axes j, i, k), scaled by its kernel; the reader's FOD is in world
    # coordinates, scaled by 300 / (4 pi). Compared as amplitudes on the 300
    # directions, each taken to its frame.
    fod = read_mritoolkit_fod(DATA / "grl.mat", DWI).fods[0].data
    export = load_volume(DATA / "grl_CSD_FOD.nii.gz").as_float()
    mask = np.abs(export).sum(axis=-1) > 0
    frame = read_mritoolkit_frame(DWI)
    ours = _amplitudes(fod[mask], DIRS_300 @ frame.T)
    theirs = _amplitudes(export[mask], DIRS_300)
    factor = (ours * theirs).sum() / (ours * ours).sum()
    assert np.abs(ours * factor - theirs).max() < 1e-5 * np.abs(theirs).max()
    # MRIToolkit's kernel scale for this fit (its kernel summed over the 33
    # directions of the outer shell), 11.3769, against Larmor's density.
    assert factor == pytest.approx(11.3769 / (300 / (4 * np.pi)), rel=1e-4)


def _amplitudes(coefficients: np.ndarray, directions: np.ndarray) -> np.ndarray:
    theta, phi = get_theta_phi(directions)
    basis = build_basis_matrix(theta, phi, 8, "tournier")
    values: np.ndarray = coefficients @ basis.T
    return values


def test_fod_peaks_follow_the_tensor_in_world_coordinates() -> None:
    fod = read_mritoolkit_fod(DATA / "grl.mat", DWI).fods[0]
    dwi = load_volume(DWI)
    mask = load_volume(DATA / "mask.nii.gz").data > 0
    gtab = read_bvals_bvecs(DATA / "bvals", DATA / "bvecs", ref_path=DWI)
    tensor = DTIModel().fit(dwi, gtab, mask)
    fibres = mask & (tensor.fa > 0.5)
    peaks = find_peaks(fod.data, fibres, num_peaks=1)
    cosines = np.abs((peaks.directions[..., 0, :] * tensor.pe).sum(axis=-1))[fibres]
    assert np.median(cosines) > 0.99


def test_the_struct_is_found_or_named() -> None:
    result = read_mritoolkit_fod(DATA / "grl.mat", DWI, variable="GRL_Results")
    assert len(result.fods) == 1
    with pytest.raises(ValueError, match="no FOD"):
        read_mritoolkit_fod(DATA / "grl.mat", DWI, variable="Other")
    with pytest.raises(ValueError, match="0 structs"):
        read_mritoolkit_fod(DATA / "dti_fit.mat", DWI)
