import base64
import gzip
import re
from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from nibabel.freesurfer.mghformat import MGHHeader
from nibabel.gifti.gifti import GiftiCoordSystem, GiftiDataArray, GiftiImage

from larmor.io.formats import (
    read_gifti_surface,
    read_gifti_values,
    write_gifti_surface,
    write_gifti_values,
)
from larmor.types import Surface

# A tetrahedron: 4 vertices, 4 faces, every index used.
VERTICES = np.array(
    [[0.0, 0.0, 0.0], [10.5, 0.0, 0.0], [0.0, -10.25, 0.0], [0.0, 0.0, 10.125]],
    dtype=np.float32,
)
FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
CURV = np.array([0.5, -1.25, 3.0, -0.125], dtype=np.float32)

# Small GIfTI files committed for the tests: see tests/data/README.md.
SURFACES = Path(__file__).parents[2] / "data" / "surfaces"


def _mesh_arrays(encoding: str = "B64GZ") -> list[GiftiDataArray]:
    return [
        GiftiDataArray(VERTICES, intent="NIFTI_INTENT_POINTSET", encoding=encoding),
        GiftiDataArray(FACES, intent="NIFTI_INTENT_TRIANGLE", encoding=encoding),
    ]


def _save(path: Path, arrays: list[GiftiDataArray]) -> None:
    nib.save(GiftiImage(darrays=arrays), path)


def _array_xml(
    data: str,
    dims: tuple[int, ...] = (4,),
    intent: str = "NIFTI_INTENT_SHAPE",
    datatype: str = "NIFTI_TYPE_FLOAT32",
    encoding: str = "ASCII",
    endian: str = "LittleEndian",
    order: str = "RowMajorOrder",
) -> str:
    dim_attributes = " ".join(f'Dim{i}="{n}"' for i, n in enumerate(dims))
    return (
        f'<DataArray Intent="{intent}" DataType="{datatype}" '
        f'ArrayIndexingOrder="{order}" Dimensionality="{len(dims)}" '
        f'{dim_attributes} Encoding="{encoding}" Endian="{endian}" '
        f'ExternalFileName="" ExternalFileOffset="0"><MetaData/>'
        f"<Data>{data}</Data></DataArray>"
    )


def _write_xml(path: Path, *arrays: str, root: str = "GIFTI") -> None:
    path.write_text(
        f'<?xml version="1.0" encoding="UTF-8"?><{root} Version="1.0" '
        f'NumberOfDataArrays="{len(arrays)}"><MetaData/><LabelTable/>'
        f"{''.join(arrays)}</{root}>"
    )


@pytest.mark.parametrize("encoding", ["ASCII", "B64BIN", "B64GZ"])
def test_nibabel_surface_is_read(tmp_path: Path, encoding: str) -> None:
    path = tmp_path / "lh.white.surf.gii"
    curv = GiftiDataArray(
        CURV, intent="NIFTI_INTENT_SHAPE", encoding=encoding, meta={"Name": "curv"}
    )
    _save(path, [*_mesh_arrays(encoding), curv])
    surface = read_gifti_surface(path)
    # nibabel's ASCII writes six decimals.
    np.testing.assert_allclose(surface.vertices, VERTICES, atol=1e-6)
    np.testing.assert_array_equal(surface.faces, FACES)
    assert list(surface.vertex_data) == ["curv"]
    np.testing.assert_allclose(surface.vertex_data["curv"], CURV, atol=1e-6)
    assert surface.reference is None


def test_values_are_named_by_name_entry_then_index(tmp_path: Path) -> None:
    path = tmp_path / "lh.white.surf.gii"
    arrays = [
        *_mesh_arrays(),
        GiftiDataArray(CURV, meta={"Description": "first", "Name": "curv"}),
        GiftiDataArray(CURV + 1),
        GiftiDataArray(CURV + 2, meta={"Name": "curv"}),
    ]
    _save(path, arrays)
    data = read_gifti_surface(path).vertex_data
    assert list(data) == ["curv", "data3", "curv4"]
    np.testing.assert_array_equal(data["curv"], CURV)
    np.testing.assert_array_equal(data["data3"], CURV + 1)
    np.testing.assert_array_equal(data["curv4"], CURV + 2)


# nibabel always writes little-endian, so the big-endian array is built by hand.
def test_big_endian_column_major_values_are_read(tmp_path: Path) -> None:
    values = np.arange(8, dtype=np.float32).reshape(4, 2) * 1.5 - 2
    raw = values.astype(">f4").tobytes(order="F")
    path = tmp_path / "lh.func.gii"
    _write_xml(
        path,
        _array_xml(
            base64.b64encode(raw).decode(),
            dims=(4, 2),
            encoding="Base64Binary",
            endian="BigEndian",
            order="ColumnMajorOrder",
        ),
    )
    read = read_gifti_values(path)["data0"]
    assert read.dtype == np.float32
    np.testing.assert_array_equal(read, values)


def test_gzip_compressed_values_are_read(tmp_path: Path) -> None:
    raw = gzip.compress(CURV.astype("<f4").tobytes())
    path = tmp_path / "lh.shape.gii"
    _write_xml(
        path, _array_xml(base64.b64encode(raw).decode(), encoding="GZipBase64Binary")
    )
    np.testing.assert_array_equal(read_gifti_values(path)["data0"], CURV)


def test_values_file_gives_every_non_mesh_array(tmp_path: Path) -> None:
    path = tmp_path / "lh.white.surf.gii"
    thickness = GiftiDataArray(CURV * 2, meta={"Name": "thickness"})
    _save(
        path, [*_mesh_arrays(), GiftiDataArray(CURV, meta={"Name": "curv"}), thickness]
    )
    values = read_gifti_values(path)
    assert list(values) == ["curv", "thickness"]
    np.testing.assert_array_equal(values["thickness"], CURV * 2)


def test_written_surface_is_read_by_nibabel(tmp_path: Path) -> None:
    path = tmp_path / "lh.white.surf.gii"
    data = {
        "curv": CURV.astype(np.float64),
        "mask": np.array([True, False, True, True]),
        "label": np.array([3, -7, 0, 2**31 - 1], dtype=np.int64),
    }
    write_gifti_surface(Surface(VERTICES, FACES, vertex_data=data), path)
    image = nib.load(path)
    assert isinstance(image, GiftiImage)
    points, triangles, curv, mask, label = image.darrays

    assert nib.nifti1.intent_codes.niistring[points.intent] == "NIFTI_INTENT_POINTSET"
    assert points.data.dtype == np.float32
    np.testing.assert_array_equal(points.data, VERTICES)
    assert points.coordsys.dataspace == 1
    assert points.coordsys.xformspace == 1
    np.testing.assert_array_equal(points.coordsys.xform, np.eye(4))

    assert (
        nib.nifti1.intent_codes.niistring[triangles.intent] == "NIFTI_INTENT_TRIANGLE"
    )
    assert triangles.data.dtype == np.int32
    np.testing.assert_array_equal(triangles.data, FACES)

    expected = [
        (curv, "curv", "NIFTI_INTENT_SHAPE", np.float32),
        (mask, "mask", "NIFTI_INTENT_NONE", np.uint8),
        (label, "label", "NIFTI_INTENT_NONE", np.int32),
    ]
    for array, name, intent, dtype in expected:
        assert array.meta["Name"] == name
        assert nib.nifti1.intent_codes.niistring[array.intent] == intent
        assert array.data.dtype == dtype
        np.testing.assert_array_equal(array.data, data[name])
        assert array.encoding == 3  # GZipBase64Binary
        assert array.endian == 2  # LittleEndian


def test_written_surface_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "lh.white.surf.gii"
    data = {
        "curv": CURV,
        "label": np.array([1, 2, 3, 4], dtype=np.int32),
        "mask": np.array([True, False, False, True]),
    }
    write_gifti_surface(Surface(VERTICES, FACES, vertex_data=data), path)
    surface = read_gifti_surface(path)
    np.testing.assert_array_equal(surface.vertices, VERTICES)
    np.testing.assert_array_equal(surface.faces, FACES)
    assert list(surface.vertex_data) == ["curv", "label", "mask"]
    np.testing.assert_array_equal(surface.vertex_data["curv"], CURV)
    np.testing.assert_array_equal(surface.vertex_data["label"], data["label"])
    assert surface.vertex_data["mask"].dtype == np.uint8
    np.testing.assert_array_equal(surface.vertex_data["mask"], [1, 0, 0, 1])


def test_written_values_are_read_by_nibabel(tmp_path: Path) -> None:
    path = tmp_path / "lh.func.gii"
    series = np.arange(8, dtype=np.float64).reshape(4, 2) / 4
    write_gifti_values({"curv": CURV, "series": series}, path)
    image = nib.load(path)
    assert isinstance(image, GiftiImage)
    assert [a.meta["Name"] for a in image.darrays] == ["curv", "series"]
    np.testing.assert_array_equal(image.darrays[0].data, CURV)
    np.testing.assert_array_equal(image.darrays[1].data, series)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.float32(1.0), "scalar"),
        (np.array([0, 2**31], dtype=np.int64), "int32"),
        (np.array([-(2**31) - 1, 0], dtype=np.int64), "int32"),
    ],
    ids=["scalar", "above-int32", "below-int32"],
)
def test_unwritable_values_are_refused(
    tmp_path: Path, values: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        write_gifti_values({"bad": values}, tmp_path / "lh.func.gii")


def _check_labels(path: Path, count: int, first: str) -> None:
    values = read_gifti_values(path)
    image = nib.load(path)
    assert isinstance(image, GiftiImage)
    assert len(values) == count
    assert next(iter(values)) == first
    for read, expected in zip(values.values(), image.darrays, strict=True):
        np.testing.assert_array_equal(read, expected.data)


def test_label_file_matches_nibabel() -> None:
    _check_labels(SURFACES / "gifti" / "lh.labels.label.gii", 2, "Lobes")


def test_real_caret_labels_match_nibabel(freesurfer_subjects: Path) -> None:
    path = freesurfer_subjects / "fsaverage" / "label" / "lh.PALS_B12.labels.gii"
    _check_labels(path, 15, "Lobes (PALS atlas)")


_MESH = (
    _array_xml(
        " ".join(str(x) for x in VERTICES.ravel()),
        dims=(4, 3),
        intent="NIFTI_INTENT_POINTSET",
    ),
    _array_xml(
        " ".join(str(x) for x in FACES.ravel()),
        dims=(4, 3),
        intent="NIFTI_INTENT_TRIANGLE",
        datatype="NIFTI_TYPE_INT32",
    ),
)
_CURV = " ".join(str(x) for x in CURV)


@pytest.mark.parametrize(
    ("arrays", "root", "match"),
    [
        ((*_MESH,), "NIFTI", "GIfTI file"),
        (
            (*_MESH, _array_xml(_CURV, datatype="NIFTI_TYPE_FLOAT64")),
            "GIFTI",
            "data type",
        ),
        (
            (*_MESH, _array_xml(_CURV, encoding="ExternalFileBinary")),
            "GIFTI",
            "encoding",
        ),
        ((*_MESH, _array_xml(_CURV, dims=(5,))), "GIFTI", "holds 4 values"),
        ((_array_xml(_CURV),), "GIFTI", "one pointset and one triangle"),
        ((*_MESH, _MESH[0]), "GIFTI", "one pointset and one triangle"),
    ],
    ids=["root", "float64", "external", "count", "values-only", "two-pointsets"],
)
def test_bad_surface_file_is_rejected(
    tmp_path: Path, arrays: tuple[str, ...], root: str, match: str
) -> None:
    path = tmp_path / "lh.white.surf.gii"
    _write_xml(path, *arrays, root=root)
    with pytest.raises(ValueError, match=match):
        read_gifti_surface(path)


# An oblique, anisotropic, non-cubic volume, as FreeSurfer's VolGeom describes it.
_ANGLE = 0.2
VOLGEOM_DIRECTIONS = np.array(
    [
        [np.cos(_ANGLE), -np.sin(_ANGLE), 0.0],
        [np.sin(_ANGLE), np.cos(_ANGLE), 0.0],
        [0.0, 0.0, 1.0],
    ]
) @ np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
VOLGEOM_SHAPE = (176, 240, 256)
VOLGEOM_SIZES = np.array([1.2, 0.9, 1.0])
VOLGEOM_CRAS = np.array([3.0, -17.0, 22.0])


def _volgeom() -> dict[str, str]:
    # Same keys and number format as FreeSurfer's surfa writes.
    meta = {
        name: f"{n:d}"
        for name, n in zip(("Width", "Height", "Depth"), VOLGEOM_SHAPE, strict=True)
    }
    meta |= {
        f"{axis}size": f"{size:.15e}"
        for axis, size in zip("XYZ", VOLGEOM_SIZES, strict=True)
    }
    for j, axis in enumerate("XYZ"):
        for k, ras in enumerate("RAS"):
            meta[f"{axis}_{ras}"] = f"{VOLGEOM_DIRECTIONS[k, j]:.15e}"
    for k, ras in enumerate("RAS"):
        meta[f"C_{ras}"] = f"{VOLGEOM_CRAS[k]:.15e}"
    return {f"VolGeom{name}": value for name, value in meta.items()}


def _save_with_meta(path: Path, meta: dict[str, str]) -> None:
    _save(
        path,
        [
            GiftiDataArray(VERTICES, intent="NIFTI_INTENT_POINTSET", meta=meta),
            GiftiDataArray(FACES, intent="NIFTI_INTENT_TRIANGLE"),
        ],
    )


def test_volgeom_metadata_moves_vertices_to_scanner_ras(tmp_path: Path) -> None:
    header = MGHHeader()
    header.set_data_shape(VOLGEOM_SHAPE)
    header.set_zooms(VOLGEOM_SIZES)
    header["Mdc"] = VOLGEOM_DIRECTIONS.T
    header["Pxyz_c"] = VOLGEOM_CRAS
    norig, torig = header.get_affine(), header.get_vox2ras_tkr()
    path = tmp_path / "lh.white.surf.gii"
    _save_with_meta(path, _volgeom())
    surface = read_gifti_surface(path)
    expected = nib.affines.apply_affine(norig @ np.linalg.inv(torig), VERTICES)
    np.testing.assert_allclose(surface.vertices, expected, atol=1e-4)
    assert surface.reference is not None
    assert surface.reference.shape == VOLGEOM_SHAPE
    np.testing.assert_allclose(surface.reference.affine, norig, atol=1e-9)


# DataSpace 1 is NIFTI_XFORM_SCANNER_ANAT: the vertices are scanner RAS already.
def test_scanner_dataspace_keeps_vertices_but_sets_the_reference(
    tmp_path: Path,
) -> None:
    path = tmp_path / "lh.white.surf.gii"
    points = GiftiDataArray(
        VERTICES,
        intent="NIFTI_INTENT_POINTSET",
        meta=_volgeom(),
        coordsys=GiftiCoordSystem(dataspace=1, xformspace=1),
    )
    _save(path, [points, GiftiDataArray(FACES, intent="NIFTI_INTENT_TRIANGLE")])
    surface = read_gifti_surface(path)
    np.testing.assert_array_equal(surface.vertices, VERTICES)
    assert surface.reference is not None
    assert surface.reference.shape == VOLGEOM_SHAPE


# fMRIPrep strips some VolGeom keys and leaves scanner RAS vertices.
def test_partial_volgeom_metadata_keeps_vertices_as_stored(tmp_path: Path) -> None:
    meta = _volgeom()
    del meta["VolGeomC_R"]
    path = tmp_path / "lh.white.surf.gii"
    _save_with_meta(path, meta)
    surface = read_gifti_surface(path)
    np.testing.assert_array_equal(surface.vertices, VERTICES)
    assert surface.reference is None


def test_non_numeric_volgeom_metadata_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "lh.white.surf.gii"
    _save_with_meta(path, _volgeom() | {"VolGeomXsize": "one"})
    with pytest.raises(ValueError, match="Malformed VolGeom"):
        read_gifti_surface(path)


@pytest.mark.parametrize(
    ("count", "names"), [(1, ["thickness"]), (2, ["thickness0", "thickness1"])]
)
def test_unnamed_arrays_take_the_given_name(
    tmp_path: Path, count: int, names: list[str]
) -> None:
    path = tmp_path / "lh.func.gii"
    _save(path, [GiftiDataArray(CURV + i) for i in range(count)])
    values = read_gifti_values(path, unnamed="thickness")
    assert list(values) == names
    np.testing.assert_array_equal(values[names[-1]], CURV + count - 1)


@pytest.mark.parametrize("reader", [read_gifti_surface, read_gifti_values])
@pytest.mark.parametrize(
    "text", ["<GIFTI><DataArray", "not xml at all\n"], ids=["cut", "text"]
)
def test_non_xml_file_is_rejected_with_its_path(
    tmp_path: Path, reader: Callable[[Path], object], text: str
) -> None:
    path = tmp_path / "lh.broken.gii"
    path.write_text(text)
    with pytest.raises(ValueError, match=f"^{re.escape(str(path))}: "):
        reader(path)


def test_surface_errors_start_with_the_path(tmp_path: Path) -> None:
    path = tmp_path / "lh.white.surf.gii"
    _write_xml(path, _array_xml(_CURV))
    with pytest.raises(ValueError, match=f"^{re.escape(str(path))}: Expected one"):
        read_gifti_surface(path)
