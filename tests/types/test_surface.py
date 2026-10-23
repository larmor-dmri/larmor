from pathlib import Path

import numpy as np
import pytest
from nibabel.freesurfer.io import read_geometry, read_morph_data

import larmor
import larmor.types
from larmor.types import Surface

# A tetrahedron: 4 vertices, 4 faces, every index used.
VERTICES = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

# Small FreeSurfer files committed for the tests: see tests/data/README.md.
SURF = Path(__file__).parents[1] / "data" / "surfaces" / "surf"


@pytest.mark.parametrize(
    "vertices", [VERTICES, VERTICES.astype(int).tolist()], ids=["float64", "int-list"]
)
@pytest.mark.parametrize(
    "faces",
    [
        FACES.astype(np.int64),
        FACES.astype(np.int16),
        FACES.astype(np.uint8),
        FACES.tolist(),
    ],
    ids=["int64", "int16", "uint8", "nested-list"],
)
def test_vertices_become_float32_and_faces_uint32(
    vertices: np.ndarray | list[list[int]], faces: np.ndarray | list[list[int]]
) -> None:
    surface = Surface(vertices=vertices, faces=faces)  # type: ignore[arg-type]  # lists on purpose
    assert surface.vertices.dtype == np.float32
    assert surface.faces.dtype == np.uint32
    np.testing.assert_array_equal(surface.vertices, VERTICES)
    np.testing.assert_array_equal(surface.faces, FACES)
    assert surface.num_vertices == 4
    assert surface.num_faces == 4


@pytest.mark.parametrize(
    "vertices", [np.zeros((4, 2)), np.zeros(12), np.zeros((4, 3, 1))]
)
def test_vertices_that_are_not_v_by_3_are_rejected(vertices: np.ndarray) -> None:
    with pytest.raises(ValueError, match="vertices"):
        Surface(vertices=vertices, faces=FACES)


@pytest.mark.parametrize(
    "faces", [np.zeros((4, 2), int), np.zeros(12, int), np.zeros((4, 3, 1), int)]
)
def test_faces_that_are_not_f_by_3_are_rejected(faces: np.ndarray) -> None:
    with pytest.raises(ValueError, match="faces"):
        Surface(vertices=VERTICES, faces=faces)


def test_float_faces_are_rejected() -> None:
    with pytest.raises(ValueError, match="integers"):
        Surface(vertices=VERTICES, faces=FACES.astype(np.float64))


# np.empty((0, 3)) is float64 by default; an empty mesh must not need a cast.
def test_empty_float_faces_give_an_empty_uint32_mesh() -> None:
    surface = Surface(vertices=np.empty((0, 3)), faces=np.empty((0, 3)))
    assert surface.faces.dtype == np.uint32
    assert surface.num_vertices == surface.num_faces == 0


# -1 would wrap to 4294967295 in uint32 if checked after the cast.
@pytest.mark.parametrize("bad_index", [-1, 4], ids=["negative", "equal-to-V"])
def test_face_index_outside_vertices_is_rejected(bad_index: int) -> None:
    faces = FACES.copy()
    faces[2, 1] = bad_index
    with pytest.raises(ValueError, match="Face indices"):
        Surface(vertices=VERTICES, faces=faces)


def test_vertex_data_given_as_list_becomes_array() -> None:
    surface = Surface(
        vertices=VERTICES,
        faces=FACES,
        vertex_data={"curv": [0.1, 0.2, 0.3, 0.4]},  # type: ignore[dict-item]  # a list on purpose
    )
    assert isinstance(surface.vertex_data["curv"], np.ndarray)
    np.testing.assert_array_equal(surface.vertex_data["curv"], [0.1, 0.2, 0.3, 0.4])


# 4 entries along the second axis but 3 along the first; and a scalar.
@pytest.mark.parametrize(
    "values",
    [np.zeros(3), np.zeros((3, 4)), np.float64(1.0)],
    ids=["short", "transposed", "scalar"],
)
def test_vertex_data_without_v_entries_is_rejected(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="vertex_data"):
        Surface(vertices=VERTICES, faces=FACES, vertex_data={"curv": values})


def test_surfaces_compare_by_identity() -> None:
    first = Surface(vertices=VERTICES, faces=FACES)
    second = Surface(vertices=VERTICES, faces=FACES)
    assert first == first
    assert first != second


def test_surface_is_exported_from_types_and_top_level() -> None:
    assert larmor.types.Surface is Surface
    assert larmor.Surface is Surface
    assert "Surface" in larmor.types.__all__
    assert "Surface" in larmor.__all__


def test_freesurfer_white_surface_with_curvature() -> None:
    vertices, faces = read_geometry(SURF / "lh.white")
    curv = read_morph_data(SURF / "lh.curv")
    surface = Surface(vertices=vertices, faces=faces, vertex_data={"curv": curv})
    assert surface.num_vertices == 642
    assert surface.num_faces == 1280
    # The geometry file stores float32 coordinates, so the cast is exact.
    np.testing.assert_array_equal(surface.vertices, vertices)
    np.testing.assert_array_equal(surface.faces, faces)
    np.testing.assert_array_equal(surface.vertex_data["curv"], curv)
