from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from larmor.dwi import DKIModel, DTIModel
from larmor.dwi.dti import compute_dti_metrics
from larmor.io import (
    load_grid,
    load_tractogram,
    load_volume,
    read_bvals_bvecs,
    read_matlab,
    read_mritoolkit_fit,
    read_mritoolkit_grid,
    read_mritoolkit_tracts,
    read_mritoolkit_volume,
)
from larmor.tractography import sample_volume
from larmor.types import GradientTable, Volume

DATA = Path(__file__).parents[1] / "data" / "mritoolkit"
DWI = DATA / "dwi.nii.gz"


@pytest.fixture(scope="module")
def dwi() -> Volume:
    return load_volume(DWI)


@pytest.fixture(scope="module")
def gtab() -> GradientTable:
    return read_bvals_bvecs(DATA / "bvals", DATA / "bvecs", ref_path=DWI)


@pytest.fixture(scope="module")
def mask() -> np.ndarray:
    return load_volume(DATA / "mask.nii.gz").data > 0


def _metrics(dt: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, ...]:
    return compute_dti_metrics(dt[mask].T.astype(np.float64), mask)


@pytest.mark.parametrize("name", ["dti_fit.mat", "dti_fit_v7.mat"])
def test_fit_lies_on_the_nifti_grid_with_its_volumes(name: str, dwi: Volume) -> None:
    fit = read_mritoolkit_fit(DATA / name)
    assert read_mritoolkit_grid(DATA / name).matches(load_grid(DWI))
    np.testing.assert_array_equal(fit.dt.affine, dwi.affine)
    # The file sorts the volumes by b-value; each is one of the NIfTI's, unchanged.
    stored = fit.dwi.data.reshape(-1, fit.dwi.data.shape[-1]).T
    original = {v.tobytes() for v in dwi.data.reshape(-1, dwi.data.shape[-1]).T}
    assert all(v.tobytes() in original for v in stored.astype(dwi.data.dtype))
    assert fit.kt is None


def test_gradients_are_the_tables_in_world_coordinates(
    dwi: Volume, gtab: GradientTable
) -> None:
    fit = read_mritoolkit_fit(DATA / "dti_fit.mat")
    flat = dwi.data.reshape(-1, dwi.data.shape[-1])
    order = [
        int(np.flatnonzero((flat == v[:, None]).all(axis=0))[0])
        for v in fit.dwi.data.reshape(-1, fit.dwi.data.shape[-1]).T
    ]
    np.testing.assert_allclose(fit.gradients.bvals, gtab.bvals[order], atol=0.01)
    weighted = gtab.bvals[order] > 50
    cosines = np.abs((fit.gradients.bvecs * gtab.bvecs[order]).sum(axis=1))
    np.testing.assert_allclose(cosines[weighted], 1.0, atol=1e-8)


def test_tensor_in_world_matches_larmors_fit_and_the_stored_maps(
    dwi: Volume, gtab: GradientTable, mask: np.ndarray
) -> None:
    fit = read_mritoolkit_fit(DATA / "dti_fit.mat")
    assert np.array_equal(fit.mask.data, mask)
    own = DTIModel().fit(dwi, gtab, mask)
    for c in range(6):
        corr = np.corrcoef(fit.dt.data[mask][:, c], own.dt[mask][:, c])[0, 1]
        assert corr > 0.99
    _, fa, md, _, _, pe, *_ = _metrics(fit.dt.data, mask)
    np.testing.assert_allclose(fa, fit.maps["FA"].data, atol=1e-6)
    np.testing.assert_allclose(md, fit.maps["MD"].data, atol=1e-9)
    anisotropic = mask & (fa > 0.1)
    cosines = np.abs((pe * fit.maps["FE"].data).sum(axis=-1))[anisotropic]
    np.testing.assert_allclose(cosines, 1.0, atol=1e-5)
    assert np.abs((pe * own.pe).sum(axis=-1))[anisotropic].min() > 0.95


def test_kurtosis_tensor_in_world_matches_larmors_fit(
    dwi: Volume, gtab: GradientTable, mask: np.ndarray
) -> None:
    fit = read_mritoolkit_fit(DATA / "dki_fit.mat")
    assert fit.kt is not None and fit.kt.data.shape == (*mask.shape, 15)
    own = DKIModel(fit_method="ols", constrained=False).fit(dwi, gtab, mask)
    read, fitted = fit.kt.data[mask], own.kt[mask]
    # Each element matches its own and no other: the order is settled, not just
    # close (W1112 and W1113, say, could correlate almost as well swapped).
    corr = np.corrcoef(read.T, fitted.T)[:15, 15:]
    off_diagonal = np.where(np.eye(15, dtype=bool), -np.inf, corr).max(axis=1)
    assert np.diag(corr).min() > 0.995 and off_diagonal.max() < 0.9
    # The two fits differ in their b0 handling; most voxels agree closely.
    spread = np.percentile(np.abs(read - fitted), 99, axis=0)
    assert (spread / np.abs(fitted).max(axis=0)).max() < 0.03


def test_a_file_that_is_not_a_fit_names_the_readers() -> None:
    with pytest.raises(ValueError, match="read_mritoolkit_tracts"):
        read_mritoolkit_fit(DATA / "tracts.mat")


@pytest.fixture(scope="module")
def tensor_fit() -> Volume:
    return read_mritoolkit_fit(DATA / "dti_fit.mat").dt


@pytest.mark.parametrize("name", ["tracts.mat", "tracts_v73.mat"])
def test_tracts_lie_where_their_fa_was_sampled(name: str, tensor_fit: Volume) -> None:
    tracts = read_mritoolkit_tracts(DATA / name, DATA / "dti_fit.mat")
    assert tracts.num_streamlines == 112
    assert tracts.reference is not None and tracts.reference.matches(load_grid(DWI))
    np.testing.assert_array_equal(tracts.streamline_data["L"], np.diff(tracts.offsets))
    # FE: the stored absolute components, from the frame (j, i, k) to (i, j, k).
    stored = np.concatenate(read_matlab(DATA / name).read("TractFE"))  # type: ignore[arg-type]
    np.testing.assert_array_equal(tracts.point_data["FE"], stored[:, [1, 0, 2]])
    # MRIToolkit's FA along a tract is that of the trilinearly interpolated tensor.
    elements = np.stack(
        [
            sample_volume(tracts, Volume(tensor_fit.data[..., c], tensor_fit.affine))
            for c in range(6)
        ]
    )
    _, fa, *_ = compute_dti_metrics(elements, np.ones((tracts.num_points, 1, 1), bool))
    np.testing.assert_allclose(fa.ravel(), tracts.point_data["FA"], atol=1e-5)


def test_a_tract_length_list_that_does_not_fit_is_left_out(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "tracts.mat"
    cells = np.empty((1, 2), dtype=object)
    cells[0, 0], cells[0, 1] = np.ones((3, 3)), np.ones((2, 3))
    lengths = np.empty((1, 1), dtype=object)
    lengths[0, 0] = np.array([[3.0]])
    savemat(path, {"Tracts": cells, "TractL": lengths})
    tracts = read_mritoolkit_tracts(path, DATA / "dti_fit.mat")
    assert "L" not in tracts.streamline_data
    assert "TractL has 1 entries for 2 streamlines" in caplog.text


def test_tracts_can_be_read_without_their_values() -> None:
    tracts = read_mritoolkit_tracts(
        DATA / "tracts.mat", DATA / "dti_fit.mat", values=False
    )
    with_values = read_mritoolkit_tracts(DATA / "tracts.mat", DATA / "dti_fit.mat")
    assert tracts.point_data == {} and tracts.streamline_data == {}
    assert with_values.point_data and with_values.streamline_data
    np.testing.assert_array_equal(tracts.points, with_values.points)


def test_every_reference_kind_gives_the_same_points() -> None:
    by_fit = read_mritoolkit_tracts(DATA / "tracts.mat", DATA / "dti_fit.mat")
    for reference in (DWI, load_grid(DWI)):
        other = read_mritoolkit_tracts(DATA / "tracts.mat", reference)
        np.testing.assert_array_equal(other.points, by_fit.points)


def test_load_tractogram_reads_mat_tracts_with_a_reference() -> None:
    grid = load_grid(DWI)
    tracts = load_tractogram(DATA / "tracts.mat", reference=grid)
    assert tracts.num_streamlines == 112
    assert (
        load_tractogram(
            DATA / "tracts.mat", reference=grid, max_streamlines=10
        ).num_streamlines
        == 10
    )
    with pytest.raises(ValueError, match="reference"):
        load_tractogram(DATA / "tracts.mat")
    with pytest.raises(ValueError, match="read_mritoolkit_fit"):
        load_tractogram(DATA / "dti_fit.mat", reference=grid)


def test_volume_reader_checks_the_grid() -> None:
    fa = read_mritoolkit_volume(DATA / "dti_fit.mat", "FA")
    assert fa.data.shape == load_grid(DWI).shape
    dt = read_mritoolkit_volume(DATA / "dti_fit.mat", "DT")
    assert dt.data.shape == (*load_grid(DWI).shape, 6)
    with pytest.raises(ValueError, match="shape"):
        read_mritoolkit_volume(DATA / "dti_fit.mat", "VDims")
