import logging
from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from nibabel.gifti.gifti import GiftiDataArray, GiftiImage

import larmor.io
from larmor.io import load_surface, save_surface
from larmor.io.formats import (
    read_freesurfer_values,
    write_freesurfer_values,
    write_gifti_values,
)
from larmor.types import Grid, Surface

# Small FreeSurfer files committed for the tests: see tests/data/README.md.
SURF = Path(__file__).parents[1] / "data" / "surfaces" / "surf"

# A tetrahedron: 4 vertices, 4 faces, every index used.
VERTICES = np.array(
    [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
)
FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
CURV = np.array([0.5, -1.25, 3.0, -0.125], dtype=np.float32)

# A conformed LIA grid shifted off the origin, so a lost reference moves vertices.
REFERENCE = Grid(
    (16, 16, 16),
    np.array(
        [
            [-1.0, 0.0, 0.0, 18.0],
            [0.0, 0.0, 1.0, -28.0],
            [0.0, -1.0, 0.0, 13.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
)


def _tetrahedron(tmp_path: Path, name: str = "lh.white") -> Path:
    path = tmp_path / name
    save_surface(Surface(VERTICES, FACES), path)
    return path


def test_freesurfer_values_are_named_after_the_hemisphere() -> None:
    files = [SURF / "lh.curv", SURF / "lh.thickness"]
    surface = load_surface(SURF / "lh.white", values=files)
    assert surface.num_vertices == 642
    assert list(surface.vertex_data) == ["curv", "thickness"]
    for name, file in zip(["curv", "thickness"], files, strict=True):
        np.testing.assert_array_equal(
            surface.vertex_data[name], read_freesurfer_values(file)
        )


@pytest.mark.parametrize("name", ["lh.mesh_v2", "mesh"])
def test_freesurfer_surface_is_recognised_by_its_bytes(
    tmp_path: Path, name: str
) -> None:
    surface = load_surface(_tetrahedron(tmp_path, name))
    np.testing.assert_allclose(surface.vertices, VERTICES)
    np.testing.assert_array_equal(surface.faces, FACES)


def test_gifti_round_trip_keeps_vertex_data(tmp_path: Path) -> None:
    path = tmp_path / "lh.white.gii"
    save_surface(Surface(VERTICES, FACES, vertex_data={"curv": CURV}), path)
    again = load_surface(path)
    np.testing.assert_allclose(again.vertices, VERTICES)
    np.testing.assert_array_equal(again.faces, FACES)
    np.testing.assert_array_equal(again.vertex_data["curv"], CURV)


def test_freesurfer_round_trip_keeps_the_reference(tmp_path: Path) -> None:
    path = tmp_path / "lh.white"
    save_surface(Surface(VERTICES, FACES, reference=REFERENCE), path)
    again = load_surface(path)
    # float32 on disk, after a shift into tkregister space and back.
    np.testing.assert_allclose(again.vertices, VERTICES, atol=1e-4)
    np.testing.assert_array_equal(again.faces, FACES)
    assert again.reference is not None
    assert again.reference.matches(REFERENCE)


def test_gifti_values_file_gives_its_inner_names(tmp_path: Path) -> None:
    values = tmp_path / "lh.measures.func.gii"
    write_gifti_values({"curv": CURV, "sulc": -CURV}, values)
    surface = load_surface(_tetrahedron(tmp_path), values=[values])
    assert list(surface.vertex_data) == ["curv", "sulc"]
    np.testing.assert_array_equal(surface.vertex_data["sulc"], -CURV)


def _per_vertex_as_surface(tmp_path: Path) -> None:
    path = tmp_path / "lh.curv"
    write_freesurfer_values(CURV, path)
    load_surface(path)


def _text_file(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("not a surface\n")
    load_surface(path)


def _duplicate_across_files(tmp_path: Path) -> None:
    write_freesurfer_values(CURV, tmp_path / "lh.curv")
    write_gifti_values({"curv": CURV}, tmp_path / "curv.func.gii")
    load_surface(
        _tetrahedron(tmp_path),
        values=[tmp_path / "lh.curv", tmp_path / "curv.func.gii"],
    )


def _duplicate_of_own_data(tmp_path: Path) -> None:
    path = tmp_path / "lh.white.gii"
    save_surface(Surface(VERTICES, FACES, vertex_data={"curv": CURV}), path)
    write_freesurfer_values(CURV, tmp_path / "lh.curv")
    load_surface(path, values=[tmp_path / "lh.curv"])


def _values_for_another_mesh(tmp_path: Path) -> None:
    write_freesurfer_values(np.zeros(5, dtype=np.float32), tmp_path / "lh.curv")
    load_surface(_tetrahedron(tmp_path), values=[tmp_path / "lh.curv"])


def _mesh_as_values(tmp_path: Path) -> None:
    mesh = tmp_path / "lh.pial.surf.gii"
    save_surface(Surface(VERTICES, FACES), mesh)
    load_surface(_tetrahedron(tmp_path), values=mesh)


@pytest.mark.parametrize(
    ("action", "match"),
    [
        (_per_vertex_as_surface, "values per vertex .*quad"),
        (_text_file, "unsupported surface"),
        (_duplicate_across_files, "already has values 'curv'"),
        (_duplicate_of_own_data, "already has values 'curv'"),
        (_values_for_another_mesh, r"lh\.curv holds 5 values per vertex .* 4 vertices"),
        (_mesh_as_values, r"lh\.pial\.surf\.gii holds no values per vertex"),
    ],
)
def test_load_surface_refuses(
    tmp_path: Path, action: Callable[[Path], None], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        action(tmp_path)


@pytest.mark.parametrize("name", ["x.nii.gz", "x.nii", "x.mif", "x.tck", "x.vtk"])
def test_save_surface_refuses_volume_and_tractogram_names(
    tmp_path: Path, name: str
) -> None:
    with pytest.raises(ValueError, match="volume or tractogram"):
        save_surface(Surface(VERTICES, FACES), tmp_path / name)
    assert not (tmp_path / name).exists()


def test_load_and_save_surface_are_exported() -> None:
    assert {"load_surface", "save_surface"} <= set(larmor.io.__all__)


@pytest.mark.parametrize("data", [{"curv": CURV}, {}], ids=["values", "mesh-only"])
def test_freesurfer_save_warns_only_when_values_are_dropped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, data: dict[str, np.ndarray]
) -> None:
    with caplog.at_level(logging.WARNING, logger="larmor"):
        save_surface(Surface(VERTICES, FACES, vertex_data=data), tmp_path / "lh.white")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    if data:
        assert len(warnings) == 1
        assert "curv" in warnings[0].getMessage()
    else:
        assert warnings == []


# HCP surfaces are in scanner RAS already, without VolGeom metadata, so they must
# land on the brain of the T1 they were made from. A few vertices
# round onto voxels the brain mask cut (6 of 130879); a 10 mm shift puts ~20000
# outside. HCP's data may not be redistributed, so it runs only with
# LARMOR_TEST_DATA naming a folder that holds HCP/100307/T1w.
def test_real_hcp_surface_lies_inside_its_t1_brain(external_data: Path) -> None:
    hcp = external_data / "HCP" / "100307" / "T1w"
    hcp_white = hcp / "Native" / "100307.L.white.native.surf.gii"
    hcp_t1 = hcp / "T1w_acpc_dc_restore_brain.nii.gz"
    if not (hcp_white.exists() and hcp_t1.exists()):
        pytest.skip(f"HCP 100307 not found in {external_data}")
    surface = load_surface(hcp_white)
    assert surface.num_vertices == 130879
    assert surface.reference is None
    image = nib.load(hcp_white)
    assert isinstance(image, GiftiImage)
    np.testing.assert_array_equal(
        surface.vertices, image.agg_data("NIFTI_INTENT_POINTSET")
    )
    t1 = nib.load(hcp_t1)
    assert isinstance(t1, nib.Nifti1Image)
    voxels = np.rint(
        nib.affines.apply_affine(np.linalg.inv(t1.affine), surface.vertices)
    ).astype(int)
    brain = np.asarray(t1.dataobj)
    assert np.all(voxels >= 0)
    assert np.all(voxels < brain.shape)
    assert np.count_nonzero(brain[tuple(voxels.T)] == 0) < 0.001 * len(voxels)


def _unnamed_values(path: Path, *arrays: GiftiDataArray) -> Path:
    nib.save(GiftiImage(darrays=list(arrays)), path)
    return path


def test_unnamed_gifti_values_are_named_after_their_file(tmp_path: Path) -> None:
    a = _unnamed_values(tmp_path / "lh.a.shape.gii", GiftiDataArray(CURV))
    b = _unnamed_values(tmp_path / "lh.b.shape.gii", GiftiDataArray(-CURV))
    surface = load_surface(_tetrahedron(tmp_path), values=[a, b])
    assert list(surface.vertex_data) == ["a", "b"]
    np.testing.assert_array_equal(surface.vertex_data["a"], CURV)
    np.testing.assert_array_equal(surface.vertex_data["b"], -CURV)


def test_several_unnamed_gifti_values_get_their_index(tmp_path: Path) -> None:
    path = _unnamed_values(
        tmp_path / "lh.c.func.gii",
        GiftiDataArray(CURV),
        GiftiDataArray(CURV * 2, meta={"Name": "kept"}),
        GiftiDataArray(CURV * 3),
    )
    surface = load_surface(_tetrahedron(tmp_path), values=[path])
    assert list(surface.vertex_data) == ["c0", "kept", "c2"]
    np.testing.assert_array_equal(surface.vertex_data["kept"], CURV * 2)
    np.testing.assert_array_equal(surface.vertex_data["c2"], CURV * 3)


@pytest.mark.parametrize("kind", [str, Path])
def test_a_single_values_file_is_accepted(
    tmp_path: Path, kind: Callable[[Path], str | Path]
) -> None:
    write_freesurfer_values(CURV, tmp_path / "lh.curv")
    surface = load_surface(_tetrahedron(tmp_path), values=kind(tmp_path / "lh.curv"))
    assert list(surface.vertex_data) == ["curv"]
    np.testing.assert_array_equal(surface.vertex_data["curv"], CURV)
