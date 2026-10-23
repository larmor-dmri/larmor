from pathlib import Path

import numpy as np
import pytest

from larmor.io import load_surface
from larmor.io.formats import read_freesurfer_label, read_freesurfer_surface
from larmor.surfaces import concatenate_surfaces, keep_vertices, vertex_normals
from larmor.types import Grid, Surface

# Small FreeSurfer files committed for the tests: see tests/data/README.md.
SUBJECT = Path(__file__).parents[1] / "data" / "surfaces"

# One face per octant, counter-clockwise seen from outside. By symmetry each
# vertex normal is exactly its own direction.
OCTAHEDRON_VERTICES = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float
)
OCTAHEDRON_FACES = np.array(
    [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ]
)

# A tetrahedron with values on its vertices, for the joining and cutting tests.
VERTICES = np.array(
    [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
)
FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
GRID = Grid((16, 16, 16), np.diag([2.0, 2.0, 2.0, 1.0]))


def _tetrahedron(reference: Grid | None = None) -> Surface:
    return Surface(
        VERTICES,
        FACES,
        vertex_data={
            "curv": np.arange(4, dtype=np.float32),
            "rgb": np.arange(12, dtype=np.float32).reshape(4, 3),
        },
        reference=reference,
    )


def test_octahedron_normals_are_vertex_directions() -> None:
    surface = Surface(OCTAHEDRON_VERTICES * 5.0, OCTAHEDRON_FACES)
    normals = vertex_normals(surface)
    np.testing.assert_allclose(normals, OCTAHEDRON_VERTICES, atol=1e-7)
    flipped = Surface(surface.vertices, OCTAHEDRON_FACES[:, ::-1])
    np.testing.assert_allclose(vertex_normals(flipped), -OCTAHEDRON_VERTICES, atol=1e-7)


# Two triangles of very different areas share vertex 0, so an unweighted mean
# of the face directions would point elsewhere.
def test_normals_are_weighted_by_face_area() -> None:
    vertices = np.array(
        [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 1]], dtype=np.float64
    )
    faces = np.array([[0, 1, 2], [0, 3, 1]])
    normal = vertex_normals(Surface(vertices, faces))[0]
    expected = np.array([0.0, 10.0, 100.0])
    np.testing.assert_allclose(normal, expected / np.linalg.norm(expected), atol=1e-6)


def test_vertex_in_no_face_gets_zero_normal() -> None:
    vertices = np.vstack([OCTAHEDRON_VERTICES, [[3.0, 3.0, 3.0]]])
    normals = vertex_normals(Surface(vertices, OCTAHEDRON_FACES))
    assert normals.dtype == np.float32
    assert normals.shape == (7, 3)
    np.testing.assert_array_equal(normals[6], [0.0, 0.0, 0.0])


# The sphere is a faceted icosphere of radius ~100, so its normals only
# approach position / radius.
def test_sphere_normals_point_out_along_the_radius() -> None:
    sphere = read_freesurfer_surface(SUBJECT / "surf" / "lh.sphere")
    positions = sphere.vertices.astype(np.float64)
    radial = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    normals = vertex_normals(sphere)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)
    # The sphere has only 642 vertices: area weighting tilts a few normals by 0.011.
    np.testing.assert_allclose(normals, radial, atol=0.02)
    flipped = Surface(sphere.vertices, sphere.faces[:, ::-1])
    np.testing.assert_allclose(vertex_normals(flipped), -radial, atol=0.02)


def test_concatenate_shifts_faces_and_keeps_shared_values() -> None:
    first = _tetrahedron()
    second = Surface(
        VERTICES + 100.0,
        FACES[:2],
        vertex_data={"thickness": np.ones(4), "curv": np.full(4, 9.0)},
    )
    joined = concatenate_surfaces([first, second])
    np.testing.assert_array_equal(
        joined.vertices, np.vstack([VERTICES, VERTICES + 100])
    )
    np.testing.assert_array_equal(joined.faces, np.vstack([FACES, FACES[:2] + 4]))
    assert list(joined.vertex_data) == ["curv"]
    np.testing.assert_array_equal(joined.vertex_data["curv"], [0, 1, 2, 3, 9, 9, 9, 9])


@pytest.mark.parametrize(
    ("references", "kept"),
    [
        ([GRID, GRID], True),
        ([GRID, Grid((16, 16, 16), np.diag([1.0, 1.0, 1.0, 1.0]))], False),
        ([GRID, None], False),
        ([None, GRID], False),
        ([None, None], False),
    ],
    ids=["same", "different", "second-none", "first-none", "both-none"],
)
def test_concatenate_keeps_reference_only_when_all_match(
    references: list[Grid | None], kept: bool
) -> None:
    joined = concatenate_surfaces([_tetrahedron(r) for r in references])
    assert (joined.reference is GRID) if kept else (joined.reference is None)


def test_concatenate_refuses_empty_list() -> None:
    with pytest.raises(ValueError, match="empty"):
        concatenate_surfaces([])


def test_concatenate_hemispheres() -> None:
    surf = SUBJECT / "surf"
    lh = load_surface(
        surf / "lh.white", values=[surf / "lh.curv", surf / "lh.thickness"]
    )
    rh = load_surface(surf / "rh.white", values=surf / "rh.curv")
    joined = concatenate_surfaces([lh, rh])
    assert joined.num_vertices == 1284
    assert joined.num_faces == 2560
    assert list(joined.vertex_data) == ["curv"]
    np.testing.assert_array_equal(
        joined.vertices[joined.faces[-1]], rh.vertices[rh.faces[-1]]
    )
    np.testing.assert_array_equal(
        joined.vertices[joined.faces[lh.num_faces :]], rh.vertices[rh.faces]
    )
    assert joined.reference is None


def test_keep_vertices_drops_faces_touching_removed_vertices() -> None:
    surface = _tetrahedron(GRID)
    mask = np.array([True, False, True, True])
    kept = keep_vertices(surface, mask)
    np.testing.assert_array_equal(kept.vertices, VERTICES[mask])
    # Only face [0, 3, 2] avoids vertex 1; old 3 and 2 become 2 and 1.
    np.testing.assert_array_equal(kept.faces, [[0, 2, 1]])
    np.testing.assert_array_equal(kept.vertex_data["curv"], [0, 2, 3])
    np.testing.assert_array_equal(
        kept.vertex_data["rgb"], surface.vertex_data["rgb"][mask]
    )
    assert kept.reference is GRID


@pytest.mark.parametrize(
    "mask",
    [np.array([0, 2, 3]), np.ones(4, dtype=int), np.ones(3, dtype=bool)],
    ids=["int-indices", "int-ones", "wrong-length"],
)
def test_keep_vertices_refuses_non_boolean_or_wrong_length_mask(
    mask: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="boolean mask"):
        keep_vertices(_tetrahedron(), mask)


def test_keep_vertices_removes_the_medial_wall() -> None:
    surf = SUBJECT / "surf"
    white = load_surface(surf / "lh.white", values=surf / "lh.thickness")
    mask = np.zeros(white.num_vertices, dtype=bool)
    mask[read_freesurfer_label(SUBJECT / "label" / "lh.cortex.label")] = True
    cortex = keep_vertices(white, mask)
    assert cortex.num_vertices == 578
    whole_faces = white.faces[mask[white.faces].all(axis=1)]
    assert 0 < cortex.num_faces == len(whole_faces) < white.num_faces
    np.testing.assert_array_equal(
        cortex.vertices[cortex.faces], white.vertices[whole_faces]
    )
    thickness = cortex.vertex_data["thickness"]
    np.testing.assert_array_equal(thickness, white.vertex_data["thickness"][mask])
    # The medial wall is where FreeSurfer gives no thickness.
    assert np.count_nonzero(white.vertex_data["thickness"] == 0) > 0
    assert np.count_nonzero(thickness == 0) == 0
