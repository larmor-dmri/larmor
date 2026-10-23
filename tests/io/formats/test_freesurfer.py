import re
import warnings
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest
from nibabel.freesurfer.io import (
    read_geometry,
    read_label,
    read_morph_data,
    write_geometry,
    write_morph_data,
)
from nibabel.freesurfer.mghformat import MGHHeader

from larmor.io.formats import (
    read_freesurfer_label,
    read_freesurfer_surface,
    read_freesurfer_values,
    write_freesurfer_surface,
    write_freesurfer_values,
)
from larmor.types import Surface

# A tetrahedron: 4 vertices, 4 faces, every index used.
VERTICES = np.array(
    [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
)
FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

# Small FreeSurfer files committed for the tests: see tests/data/README.md.
SURFACES = Path(__file__).parents[2] / "data" / "surfaces"


def _volume_info(
    cras: list[float], size: float = 1.0, n: int = 256
) -> OrderedDict[str, Any]:
    # Same layout nibabel returns for fsaverage lh.white: a conformed LIA volume.
    return OrderedDict(
        head=np.array([2, 0, 20], dtype=np.int32),
        valid="1  # volume info valid",
        filename="../mri/orig.mgz",
        volume=np.array([n, n, n]),
        voxelsize=np.array([size, size, size]),
        xras=np.array([-1.0, 0.0, 0.0]),
        yras=np.array([0.0, 0.0, -1.0]),
        zras=np.array([0.0, 1.0, 0.0]),
        cras=np.array(cras),
    )


def _check_white_surface(path: Path, num_vertices: int, has_reference: bool) -> None:
    vertices, faces = read_geometry(path)
    surface = read_freesurfer_surface(path)
    assert surface.num_vertices == num_vertices
    assert surface.num_faces == 2 * num_vertices - 4
    # These surfaces have cras 0, so scanner and tkregister RAS coincide.
    np.testing.assert_allclose(surface.vertices, vertices, atol=1e-4)
    np.testing.assert_array_equal(surface.faces, faces)
    if has_reference:
        assert surface.reference is not None
        assert surface.reference.shape == (256, 256, 256)
    else:
        assert surface.reference is None


@pytest.mark.parametrize(
    ("name", "has_reference"), [("lh.white", True), ("rh.white", False)]
)
def test_white_surface_matches_nibabel(name: str, has_reference: bool) -> None:
    _check_white_surface(SURFACES / "surf" / name, 642, has_reference)


@pytest.mark.parametrize(
    ("subject", "num_vertices", "has_reference"),
    [("fsaverage5", 10242, False), ("fsaverage", 163842, True)],
)
def test_fsaverage_white_surface_matches_nibabel(
    freesurfer_subjects: Path, subject: str, num_vertices: int, has_reference: bool
) -> None:
    path = freesurfer_subjects / subject / "surf" / "lh.white"
    _check_white_surface(path, num_vertices, has_reference)


# For a conformed LIA volume, scanner RAS = tkregister RAS + cras, whatever the
# voxel size.
@pytest.mark.parametrize(("size", "n"), [(1.0, 256), (0.8, 320)], ids=["1mm", "0.8mm"])
def test_valid_volume_info_shifts_vertices_by_cras(
    tmp_path: Path, size: float, n: int
) -> None:
    cras = [10.0, -20.0, 5.0]
    path = tmp_path / "lh.white"
    write_geometry(path, VERTICES, FACES, volume_info=_volume_info(cras, size, n))
    surface = read_freesurfer_surface(path)
    np.testing.assert_allclose(surface.vertices, VERTICES + cras, atol=1e-4)
    np.testing.assert_array_equal(surface.faces, FACES)
    assert surface.reference is not None
    assert surface.reference.shape == (n, n, n)
    centre = surface.reference.affine @ [n / 2, n / 2, n / 2, 1.0]
    np.testing.assert_allclose(centre[:3], cras, atol=1e-9)


def test_file_without_footer_has_no_reference(tmp_path: Path) -> None:
    path = tmp_path / "lh.white"
    write_geometry(path, VERTICES, FACES)
    surface = read_freesurfer_surface(path)
    assert surface.reference is None
    np.testing.assert_array_equal(surface.vertices, VERTICES)
    np.testing.assert_array_equal(surface.faces, FACES)


def test_wrong_magic_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "lh.white"
    write_geometry(path, VERTICES, FACES)
    path.write_bytes(b"\xff\xff\xfd" + path.read_bytes()[3:])
    with pytest.raises(ValueError, match="header"):
        read_freesurfer_surface(path)


def test_header_without_blank_line_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "lh.white"
    path.write_bytes(b"\xff\xff\xfecreated by test\n")
    with pytest.raises(ValueError, match="(?i)truncated"):
        read_freesurfer_surface(path)


# numpy's own "buffer is smaller" ValueError must not count: the match checks
# that the reader catches the short file itself. The last 104 bytes are the two
# counts, the vertices and the faces, so that cut ends right after the header.
@pytest.mark.parametrize(
    "cut", [60, 4, 104], ids=["mid-vertices", "last-face", "before-counts"]
)
def test_truncated_file_is_rejected(tmp_path: Path, cut: int) -> None:
    path = tmp_path / "lh.white"
    write_geometry(path, VERTICES, FACES)
    path.write_bytes(path.read_bytes()[:-cut])
    with pytest.raises(ValueError, match=f"{re.escape(str(path))}.*(?i:truncated)"):
        read_freesurfer_surface(path)


VALUES = np.array([0.5, -1.25, 3.0, -0.125], dtype=np.float32)


@pytest.mark.parametrize("name", ["lh.curv", "lh.thickness", "lh.sulc"])
def test_real_values_match_nibabel(name: str) -> None:
    path = SURFACES / "surf" / name
    values = read_freesurfer_values(path)
    assert values.shape == (642,)
    assert values.dtype == np.float32
    assert values.flags.writeable
    np.testing.assert_array_equal(values, read_morph_data(path))


def test_written_values_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "lh.curv"
    write_morph_data(path, VALUES, fnum=len(FACES))
    np.testing.assert_array_equal(read_freesurfer_values(path), VALUES)


def test_real_values_attach_to_their_surface() -> None:
    surf = SURFACES / "surf"
    white = read_freesurfer_surface(surf / "lh.white")
    curv = read_freesurfer_values(surf / "lh.curv")
    surface = Surface(white.vertices, white.faces, vertex_data={"curv": curv})
    np.testing.assert_array_equal(surface.vertex_data["curv"], curv)


def test_old_curv_format_is_rejected(tmp_path: Path) -> None:
    # The old format starts with a 3-byte vertex count instead of the magic.
    path = tmp_path / "lh.curv"
    path.write_bytes(b"\x00\x00\x04\x00\x00\x04" + b"\x00\x00" * len(VALUES))
    with pytest.raises(ValueError, match="header"):
        read_freesurfer_values(path)


def test_several_values_per_vertex_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "lh.curv"
    write_morph_data(path, VALUES)
    data = path.read_bytes()
    path.write_bytes(data[:11] + np.array(2, ">i4").tobytes() + data[15:])
    with pytest.raises(ValueError, match="per vertex.*quad"):
        read_freesurfer_values(path)


# A 5-byte file stops inside the header, before the reader can check the size.
@pytest.mark.parametrize("keep", [-2, 5], ids=["mid-values", "mid-header"])
def test_truncated_values_are_rejected(tmp_path: Path, keep: int) -> None:
    path = tmp_path / "lh.curv"
    write_morph_data(path, VALUES)
    path.write_bytes(path.read_bytes()[:keep])
    with pytest.raises(ValueError, match="(?i)truncated"):
        read_freesurfer_values(path)


CRAS = [10.0, -20.0, 5.0]


def _surface_with_reference(tmp_path: Path) -> Surface:
    path = tmp_path / "source.white"
    write_geometry(path, VERTICES, FACES, volume_info=_volume_info(CRAS))
    return read_freesurfer_surface(path)


def test_written_surface_round_trips(tmp_path: Path) -> None:
    surface = _surface_with_reference(tmp_path)
    path = tmp_path / "lh.white"
    write_freesurfer_surface(surface, path)
    again = read_freesurfer_surface(path)
    np.testing.assert_allclose(again.vertices, surface.vertices, atol=1e-4)
    np.testing.assert_array_equal(again.faces, surface.faces)
    assert surface.reference is not None
    assert again.reference is not None
    assert again.reference.shape == surface.reference.shape
    np.testing.assert_allclose(again.reference.affine, surface.reference.affine)


# nibabel does not apply cras, so it must see the tkregister vertices back.
def test_written_surface_is_read_by_nibabel_in_tkregister_space(
    tmp_path: Path,
) -> None:
    path = tmp_path / "lh.white"
    write_freesurfer_surface(_surface_with_reference(tmp_path), path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vertices, faces, info = read_geometry(path, read_metadata=True)
    np.testing.assert_allclose(vertices, VERTICES, atol=1e-4)
    np.testing.assert_array_equal(faces, FACES)
    np.testing.assert_allclose(info["cras"], CRAS, atol=1e-9)
    np.testing.assert_array_equal(info["volume"], [256, 256, 256])


def test_surface_without_reference_is_written_as_is(tmp_path: Path) -> None:
    path = tmp_path / "lh.white"
    write_freesurfer_surface(Surface(VERTICES + 1.5, FACES), path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vertices, faces, info = read_geometry(path, read_metadata=True)
    np.testing.assert_array_equal(vertices, VERTICES + 1.5)
    np.testing.assert_array_equal(faces, FACES)
    assert len(info) == 0


def test_real_surface_is_rewritten_unchanged(tmp_path: Path) -> None:
    source = SURFACES / "surf" / "lh.white"
    path = tmp_path / "lh.white"
    write_freesurfer_surface(read_freesurfer_surface(source), path)
    vertices, faces = read_geometry(path)
    expected_vertices, expected_faces = read_geometry(source)
    np.testing.assert_array_equal(vertices, expected_vertices)
    np.testing.assert_array_equal(faces, expected_faces)


def test_written_values_are_read_by_both_readers(tmp_path: Path) -> None:
    path = tmp_path / "lh.curv"
    write_freesurfer_values(VALUES, path, num_faces=len(FACES))
    np.testing.assert_array_equal(read_morph_data(path), VALUES)
    np.testing.assert_array_equal(read_freesurfer_values(path), VALUES)
    assert int(np.frombuffer(path.read_bytes()[7:11], ">i4")[0]) == len(FACES)


def test_multi_dimensional_values_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="multi-dimensional"):
        write_freesurfer_values(VALUES.reshape(2, 2), tmp_path / "lh.curv")


# Oblique, anisotropic and non-cubic, so swapped axes or sizes in the tkregister
# affine move the vertices.
_ANGLE = 0.2
OBLIQUE_DIRECTIONS = np.array(
    [
        [np.cos(_ANGLE), -np.sin(_ANGLE), 0.0],
        [np.sin(_ANGLE), np.cos(_ANGLE), 0.0],
        [0.0, 0.0, 1.0],
    ]
) @ np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
OBLIQUE_SHAPE = (176, 240, 256)
OBLIQUE_SIZES = np.array([1.2, 0.9, 1.0])
OBLIQUE_CRAS = np.array([3.0, -17.0, 22.0])


def _nibabel_affines() -> tuple[np.ndarray, np.ndarray]:
    header = MGHHeader()
    header.set_data_shape(OBLIQUE_SHAPE)
    header.set_zooms(OBLIQUE_SIZES)
    header["Mdc"] = OBLIQUE_DIRECTIONS.T
    header["Pxyz_c"] = OBLIQUE_CRAS
    return header.get_affine(), header.get_vox2ras_tkr()


def _oblique_file(tmp_path: Path, head: list[int] | None = None) -> Path:
    path = tmp_path / "lh.white"
    info = OrderedDict(
        head=np.array([2, 0, 20], dtype=np.int32),
        valid="1  # volume info valid",
        filename="x",
        volume=np.array(OBLIQUE_SHAPE),
        voxelsize=OBLIQUE_SIZES,
        xras=OBLIQUE_DIRECTIONS[:, 0],
        yras=OBLIQUE_DIRECTIONS[:, 1],
        zras=OBLIQUE_DIRECTIONS[:, 2],
        cras=OBLIQUE_CRAS,
    )
    write_geometry(path, VERTICES, FACES, volume_info=info)
    if head is not None:
        raw = path.read_bytes()
        i = raw.rindex(np.array([2, 0, 20], ">i4").tobytes())
        path.write_bytes(raw[:i] + np.array(head, ">i4").tobytes() + raw[i + 12 :])
    return path


def _oblique_scanner_vertices() -> np.ndarray:
    norig, torig = _nibabel_affines()
    expected: np.ndarray = nib.affines.apply_affine(
        norig @ np.linalg.inv(torig), VERTICES
    )
    return expected


# [20] alone is the head of older files, which hold tkregister RAS too.
@pytest.mark.parametrize("head", [None, [20]], ids=["2-0-20", "20"])
def test_oblique_footer_converts_vertices_as_nibabel_mgh(
    tmp_path: Path, head: list[int] | None
) -> None:
    surface = read_freesurfer_surface(_oblique_file(tmp_path, head))
    np.testing.assert_allclose(surface.vertices, _oblique_scanner_vertices(), atol=1e-4)
    assert surface.reference is not None
    assert surface.reference.shape == OBLIQUE_SHAPE
    np.testing.assert_allclose(
        surface.reference.affine, _nibabel_affines()[0], atol=1e-9
    )


def test_oblique_surface_round_trips(tmp_path: Path) -> None:
    surface = read_freesurfer_surface(_oblique_file(tmp_path))
    path = tmp_path / "again.white"
    write_freesurfer_surface(surface, path)
    again = read_freesurfer_surface(path)
    np.testing.assert_allclose(again.vertices, _oblique_scanner_vertices(), atol=1e-4)
    assert again.reference is not None
    np.testing.assert_allclose(again.reference.affine, _nibabel_affines()[0], atol=1e-9)


def test_real_ras_footer_keeps_vertices_as_stored(tmp_path: Path) -> None:
    surface = read_freesurfer_surface(_oblique_file(tmp_path, [2, 1, 20]))
    np.testing.assert_allclose(surface.vertices, VERTICES, atol=1e-6)
    assert surface.reference is not None
    np.testing.assert_allclose(
        surface.reference.affine, _nibabel_affines()[0], atol=1e-9
    )


@pytest.mark.parametrize(
    "footer",
    [b"valid = 1\nfilename = x\n", b"valid = 1\nfilename = x\nvolume = 1 2 a\n"],
    ids=["missing-field", "non-numeric"],
)
def test_malformed_valid_footer_is_rejected(tmp_path: Path, footer: bytes) -> None:
    path = _oblique_file(tmp_path)
    raw = path.read_bytes()
    i = raw.rindex(np.array([2, 0, 20], ">i4").tobytes())
    path.write_bytes(raw[: i + 12] + footer)
    with pytest.raises(ValueError, match="Malformed volume info"):
        read_freesurfer_surface(path)


def _wrong_magic_surface(path: Path) -> None:
    path.write_bytes(b"\xff\xff\xfd" + bytes(20))
    read_freesurfer_surface(path)


def _truncated_surface(path: Path) -> None:
    write_geometry(path, VERTICES, FACES)
    path.write_bytes(path.read_bytes()[:-4])
    read_freesurfer_surface(path)


def _wrong_magic_values(path: Path) -> None:
    path.write_bytes(b"\xff\xff\xfe" + bytes(20))
    read_freesurfer_values(path)


def _truncated_values(path: Path) -> None:
    write_morph_data(path, VALUES)
    path.write_bytes(path.read_bytes()[:-2])
    read_freesurfer_values(path)


@pytest.mark.parametrize(
    "action",
    [_wrong_magic_surface, _truncated_surface, _wrong_magic_values, _truncated_values],
)
def test_errors_name_the_file(tmp_path: Path, action: Callable[[Path], None]) -> None:
    with pytest.raises(ValueError, match="rh.subject42"):
        action(tmp_path / "rh.subject42")


def test_real_labels_match_nibabel_and_split_the_hemisphere() -> None:
    label = SURFACES / "label"
    cortex = read_freesurfer_label(label / "lh.cortex.label")
    medial = read_freesurfer_label(label / "lh.Medial_wall.label")
    assert cortex.dtype == np.int64
    np.testing.assert_array_equal(cortex, read_label(label / "lh.cortex.label"))
    np.testing.assert_array_equal(medial, read_label(label / "lh.Medial_wall.label"))
    np.testing.assert_array_equal(
        np.sort(np.concatenate([cortex, medial])), np.arange(642)
    )


LABEL_LINES = ["#!ascii label", "2", "7 1.0 2.0 3.0 0.0", "3 -1.5 0.5 9.0 1.0"]


def test_label_lines_after_the_count_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "lh.test.label"
    path.write_text("\n".join([*LABEL_LINES, "this is not a vertex line"]) + "\n")
    np.testing.assert_array_equal(read_freesurfer_label(path), [7, 3])


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (LABEL_LINES[:1], "vertex count"),
        (["#", "two", *LABEL_LINES[2:]], "vertex count"),
        (LABEL_LINES[:3], "truncated"),
        ([*LABEL_LINES[:3], "3 -1.5 0.5 9.0"], "5 numbers"),
        ([*LABEL_LINES[:3], "3.5 -1.5 0.5 9.0 1.0"], "not an integer"),
        (["#", "-1"], "negative"),
    ],
    ids=[
        "no-count",
        "non-numeric-count",
        "truncated",
        "four-fields",
        "float-index",
        "negative-count",
    ],
)
def test_malformed_label_is_rejected(
    tmp_path: Path, lines: list[str], message: str
) -> None:
    path = tmp_path / "lh.test.label"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=f"{re.escape(str(path))}.*{message}"):
        read_freesurfer_label(path)


def test_binary_file_as_label_is_rejected_with_its_path(tmp_path: Path) -> None:
    path = tmp_path / "lh.white"
    path.write_bytes(b"\xff\xff\xfe created by test\n\n")
    with pytest.raises(ValueError, match=f"{re.escape(str(path))} is not a label file"):
        read_freesurfer_label(path)
