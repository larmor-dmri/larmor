import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from larmor.io import (
    read_matlab,
    read_mritoolkit_fit,
    read_mritoolkit_frame,
    read_mritoolkit_map,
    read_mritoolkit_tracts,
    read_mritoolkit_volume,
)
from larmor.sd.sh import build_basis_matrix
from larmor.sd.sphere import get_theta_phi
from larmor.types import Tractogram, Volume
from larmor.viewer import Category, LoadAs, MatlabPreset, OptionName, Viewer
from larmor.viewer.model.matlab_import import (
    MatlabPart,
    PartKind,
    find_reference,
    read_part,
    suggest_preset,
)
from larmor.viewer.model.surface import SurfaceData

DATA = Path(__file__).parents[2] / "data" / "mritoolkit"
FIT = DATA / "dti_fit.mat"


@pytest.mark.parametrize(
    ("name", "preset"),
    [
        ("dti_fit.mat", MatlabPreset.TENSOR_FIT),
        ("dki_fit.mat", MatlabPreset.TENSOR_FIT),
        ("grl.mat", MatlabPreset.FOD_FIT),
        ("tracts.mat", MatlabPreset.TRACTOGRAPHY),
    ],
)
def test_the_preset_that_fits_is_suggested(name: str, preset: MatlabPreset) -> None:
    assert suggest_preset(read_matlab(DATA / name)) == preset


def test_a_file_without_a_grid_finds_the_fit_beside_it(tmp_path: Path) -> None:
    shutil.copy(DATA / "grl.mat", tmp_path / "grl.mat")
    assert find_reference(tmp_path / "grl.mat") is None
    shutil.copy(FIT, tmp_path / "dwi.mat")
    assert find_reference(tmp_path / "grl.mat") == tmp_path / "dwi.mat"
    assert find_reference(tmp_path / "dwi.mat") == tmp_path / "dwi.mat"


def test_tensor_fit_adds_its_maps_with_dti_colours_and_fa_shown() -> None:
    viewer = Viewer()
    items = viewer.load_matlab(FIT, MatlabPreset.TENSOR_FIT)
    names = {i.name: i for i in items}
    assert set(names) == {
        f"dti_fit {n}" for n in ("FA", "MD", "eigenvalues", "FE", "DWI")
    }
    fa, fe = names["dti_fit FA"], names["dti_fit FE"]
    assert viewer.scene.view.main_volume == fa.id
    assert fe.options.dti and fe.options.modulate == fa.id  # type: ignore[attr-defined]
    assert isinstance(fa.data, Volume)
    np.testing.assert_array_equal(fa.data.data, read_mritoolkit_map(FIT, "FA").data)
    assert names["dti_fit DWI"].data.data.shape[-1] == 133  # type: ignore[attr-defined]


def test_fod_fit_needs_a_reference_and_adds_fods_fractions_and_rss() -> None:
    viewer = Viewer()
    with pytest.raises(ValueError, match="reference"):
        viewer.load_matlab(DATA / "grl.mat", MatlabPreset.FOD_FIT)
    items = viewer.load_matlab(DATA / "grl.mat", MatlabPreset.FOD_FIT, reference=FIT)
    assert [(i.name, i.category) for i in items] == [
        ("grl FOD", Category.FODS),
        ("grl fractions", Category.VOLUMES),
        ("grl RSS", Category.VOLUMES),
    ]
    assert items[0].data.data.shape == (6, 7, 5, 45)  # type: ignore[attr-defined]


def test_a_preset_that_does_not_fit_is_refused() -> None:
    with pytest.raises(ValueError, match="Tracts"):
        Viewer().load_matlab(FIT, MatlabPreset.TRACTOGRAPHY)


def test_manual_import_reads_images_and_refuses_shapes_that_do_not_fit() -> None:
    viewer = Viewer()
    choices = {"DT{1}": LoadAs.OVERLAY, "eigval": LoadAs.VOLUME, "FE": LoadAs.PEAKS}
    items = viewer.load_matlab(FIT, MatlabPreset.MANUAL, variables=choices)
    assert [(i.name, i.category) for i in items] == [
        ("dti_fit DT{1}", Category.OVERLAYS),
        ("dti_fit eigval", Category.VOLUMES),
        ("dti_fit FE", Category.PEAKS),
    ]
    with pytest.raises(ValueError, match="fewer than three dimensions"):
        viewer.load_matlab(FIT, MatlabPreset.MANUAL, variables={"VDims": LoadAs.VOLUME})
    with pytest.raises(ValueError, match="3 values per voxel is neither"):
        viewer.load_matlab(
            DATA / "grl.mat",
            MatlabPreset.MANUAL,
            reference=FIT,
            variables={"GRL_Results.fractions": LoadAs.FODS},
        )
    with pytest.raises(ValueError, match="Needs a reference grid"):
        viewer.load_matlab(
            DATA / "grl.mat",
            MatlabPreset.MANUAL,
            variables={"GRL_Results.RSS": LoadAs.VOLUME},
        )


def test_manual_import_of_sh_streamlines_and_a_surface(tmp_path: Path) -> None:
    # A file of the user's own, version 5, on the fixture's grid (7 x 6 x 5 stored).
    path = tmp_path / "own.mat"
    savemat(
        path,
        {
            "sh": np.ones((7, 6, 5, 45), np.float32),
            "lines": np.array([np.ones((4, 3)), np.ones((2, 3)) * 2], dtype=object),
            "mesh": {
                "vertices": np.eye(3),
                "faces": np.array([[1, 2, 3]], dtype=np.float64),
            },
        },
    )
    viewer = Viewer()
    choices = {
        "sh": LoadAs.FODS,
        "lines": LoadAs.TRACTOGRAM,
        "mesh": LoadAs.SURFACE,
    }
    items = viewer.load_matlab(
        path, MatlabPreset.MANUAL, reference=FIT, variables=choices
    )
    assert [i.category for i in items] == [
        Category.FODS,
        Category.TRACTOGRAMS,
        Category.SURFACES,
    ]
    assert isinstance(items[1].data, Tractogram) and items[1].data.num_streamlines == 2
    surface = items[2].data
    assert isinstance(surface, SurfaceData)
    np.testing.assert_array_equal(surface.surface.faces, [[0, 1, 2]])


def test_manual_peaks_and_fods_are_taken_to_world_coordinates(tmp_path: Path) -> None:
    # Arrays whose x, y and z differ, in MRIToolkit's frame (voxel axes j, i, k)
    # on the oblique grid of the fit, which has no symmetry to hide a mistake.
    rng = np.random.default_rng(3)
    peaks = rng.normal(size=(7, 6, 5, 6)).astype(np.float32)
    sh = rng.normal(size=(7, 6, 5, 45)).astype(np.float32)
    path = tmp_path / "own.mat"
    savemat(path, {"peaks": peaks, "sh": sh})
    choices = {"peaks": LoadAs.PEAKS, "sh": LoadAs.FODS}
    items = Viewer().load_matlab(
        path, MatlabPreset.MANUAL, reference=FIT, variables=choices
    )
    frame = read_mritoolkit_frame(FIT)
    world_peaks = items[0].data.data  # type: ignore[attr-defined]
    stored = read_mritoolkit_volume(path, "peaks", reference=FIT).data
    expected = stored.reshape(6, 7, 5, 2, 3) @ frame.T
    np.testing.assert_allclose(world_peaks, expected.reshape(6, 7, 5, 6), atol=1e-5)
    # The FOD's value along a world direction is the stored one along the same
    # direction in MRIToolkit's frame.
    directions = rng.normal(size=(20, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    world = items[1].data.data  # type: ignore[attr-defined]
    stored_sh = read_mritoolkit_volume(path, "sh", reference=FIT).data
    np.testing.assert_allclose(
        _evaluate(world, directions @ frame.T),
        _evaluate(stored_sh, directions),
        atol=1e-4,
    )


def _evaluate(coefficients: np.ndarray, directions: np.ndarray) -> np.ndarray:
    basis = build_basis_matrix(*get_theta_phi(directions), 8, "tournier")
    values: np.ndarray = coefficients @ basis.T
    return values


def test_values_per_point_colour_a_loaded_tractogram() -> None:
    viewer = Viewer()
    [tracts] = viewer.load_matlab(
        DATA / "tracts.mat", MatlabPreset.TRACTOGRAPHY, reference=FIT
    )
    viewer.load_matlab(
        DATA / "tracts.mat",
        MatlabPreset.MANUAL,
        reference=FIT,
        variables={"TractMD": LoadAs.POINT_VALUES},
        tractogram_id=tracts.id,
    )
    values = viewer._tsf_point_values(tracts.id)
    assert isinstance(tracts.data, Tractogram)
    assert values is not None
    stored = read_mritoolkit_tracts(DATA / "tracts.mat", FIT).point_data["MD"]
    np.testing.assert_allclose(values, stored)


def test_the_tractography_preset_reads_the_points_only() -> None:
    # The per-point values of a large tractography take far longer to read than
    # the points, and the viewer never shows them.
    part = MatlabPart(PartKind.TRACTS, "Tracts", 0, FIT)
    tracts = read_part(DATA / "tracts.mat", part, None, None)
    assert isinstance(tracts, Tractogram)
    assert tracts.point_data == {} and tracts.streamline_data == {}


def test_values_per_point_may_be_rows_and_a_missing_one_adds_nothing(
    tmp_path: Path,
) -> None:
    viewer = Viewer()
    [tracts] = viewer.load_matlab(
        DATA / "tracts.mat", MatlabPreset.TRACTOGRAPHY, reference=FIT
    )
    assert isinstance(tracts.data, Tractogram)
    counts = np.diff(tracts.data.offsets)
    rows = np.empty((1, len(counts)), dtype=object)
    for s, n in enumerate(counts):
        rows[0, s] = np.arange(n, dtype=float)[None, :]  # 1 x N, a MATLAB row
    path = tmp_path / "values.mat"
    savemat(path, {"rows": rows, "mesh": {"vertices": np.eye(3), "faces": [[1, 2, 3]]}})
    viewer.load_matlab(
        path,
        MatlabPreset.MANUAL,
        variables={"rows": LoadAs.POINT_VALUES},
        tractogram_id=tracts.id,
    )
    values = viewer._tsf_point_values(tracts.id)
    assert values is not None
    np.testing.assert_array_equal(
        values, np.concatenate([np.arange(n) for n in counts])
    )
    before = len(viewer.scene.items)
    choices = {"mesh": LoadAs.SURFACE, "nothing_here": LoadAs.POINT_VALUES}
    with pytest.raises(ValueError, match="nothing_here"):
        viewer.load_matlab(
            path, MatlabPreset.MANUAL, variables=choices, tractogram_id=tracts.id
        )
    assert len(viewer.scene.items) == before


def test_fit_maps_are_zero_outside_the_fit_mask(tmp_path: Path) -> None:
    # A 2 x 3 x 2 fit whose tensor is NaN in one voxel, where the other maps are
    # finite, and whose eigenvalues are NaN in another.
    stored = (3, 2, 2)  # MRIToolkit's layout of a 2 x 3 x 2 grid
    dt = np.full(stored, 1e-3)
    dt[0, 0, 0] = np.nan
    eigval = np.ones((*stored, 3)) * [3.0, 2.0, 1.0]
    eigval[1, 1, 1, 2] = np.nan

    def cell(arrays: list[np.ndarray]) -> np.ndarray:
        result = np.empty((1, len(arrays)), dtype=object)
        for n, array in enumerate(arrays):
            result[0, n] = array
        return result

    path = tmp_path / "fit.mat"
    savemat(
        path,
        {
            "DT": cell([dt] * 6),
            "DWI": cell([np.ones(stored)] * 2),
            "b": np.array([np.zeros(6), [1000.0, 0, 0, 0, 0, 0]]),
            "FA": np.ones(stored),
            "eigval": eigval,
            "FE": np.ones((*stored, 3)),
            "DWIB0": np.ones(stored),
            "hdr": {
                "dime": {"dim": [3, 2, 3, 2, 1, 1, 1, 1], "pixdim": [1, 1, 1, 1]},
                "hist": {
                    "sform_code": 1.0,
                    "qform_code": 0.0,
                    "srow_x": [1, 0, 0, 0],
                    "srow_y": [0, 1, 0, 0],
                    "srow_z": [0, 0, 1, 0],
                },
            },
        },
    )
    fit = read_mritoolkit_fit(path)
    outside = ~fit.mask.data
    assert outside.sum() == 1
    for name, volume in fit.maps.items():
        assert not volume.data[outside].any(), name
        assert volume.data[~outside].any(), name
    # A NaN eigenvalue is 0, not a NaN in the means.
    assert np.isfinite(fit.maps["MD"].data).all()
    np.testing.assert_allclose(
        read_mritoolkit_map(path, "b0").data, fit.maps["b0"].data
    )


def test_a_session_reopens_every_import_from_its_variables(tmp_path: Path) -> None:
    folder = tmp_path / "project"
    shutil.copytree(DATA, folder)
    viewer = Viewer()
    viewer.load_matlab(folder / "dti_fit.mat", MatlabPreset.TENSOR_FIT)
    viewer.load_matlab(
        folder / "grl.mat", MatlabPreset.FOD_FIT, reference=folder / "dti_fit.mat"
    )
    [tracts] = viewer.load_matlab(
        folder / "tracts.mat",
        MatlabPreset.TRACTOGRAPHY,
        reference=folder / "dti_fit.mat",
    )
    viewer.set(tracts.id, OptionName.VALUES_FILE, f"{folder / 'tracts.mat'}:TractFA")
    viewer.load_matlab(
        folder / "dti_fit.mat", MatlabPreset.MANUAL, variables={"DT{2}": LoadAs.VOLUME}
    )
    viewer.save_session(folder / "view.lar.json")
    in_place = Viewer()
    assert in_place.load_session(folder / "view.lar.json").problems == ()
    tract = in_place.scene.in_category(Category.TRACTOGRAMS)[0]
    assert in_place._tsf_point_values(tract.id) is not None
    # The folder moves: the files and the references are found from the session's;
    # a values file is an absolute path, as a TSF file's.
    moved = tmp_path / "moved"
    folder.rename(moved)
    reopened = Viewer()
    report = reopened.load_session(moved / "view.lar.json")
    assert [p for p in report.problems if "TSF file" not in p] == []
    before = [(i.name, i.category, i.part) for i in viewer.scene.items]
    after = [(i.name, i.category) for i in reopened.scene.items]
    assert after == [(n, c) for n, c, _ in before]
    fod = reopened.scene.in_category(Category.FODS)[0]
    assert fod.part == MatlabPart(PartKind.FOD, "GRL_Results", 0, moved / "dti_fit.mat")
    fe = next(i for i in reopened.scene.items if i.name == "dti_fit FE")
    fa = next(i for i in reopened.scene.items if i.name == "dti_fit FA")
    assert fe.options.modulate == fa.id  # type: ignore[attr-defined]


def test_a_part_round_trips_through_json() -> None:
    part = MatlabPart(PartKind.FOD, "GRL_Results", 1, Path("/data/dwi.mat"))
    assert MatlabPart.from_json(part.to_json()) == part
    with pytest.raises(ValueError):
        MatlabPart.from_json({"kind": "nonsense"})
