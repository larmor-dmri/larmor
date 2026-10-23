"""Operations on triangle meshes."""

from collections.abc import Sequence

import numpy as np

from larmor.types import Surface


def vertex_normals(surface: Surface) -> np.ndarray:
    """Computes the unit normal of every vertex, pointing out of the surface.

    Args:
        surface: The surface, with V vertices.

    Returns:
        The normals, shape (V, 3), float32. A vertex that is in no face has a
        normal of zeros.
    """
    v: np.ndarray = surface.vertices.astype(np.float64)
    f: np.ndarray = surface.faces.astype(np.intp)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    face_normals = np.cross(b - a, c - a)
    sums = np.zeros_like(v)

    for corner in range(3):
        np.add.at(sums, f[:, corner], face_normals)
    lengths = np.linalg.norm(sums, axis=1, keepdims=True)
    normals: np.ndarray = np.divide(
        sums, lengths, out=np.zeros_like(sums), where=lengths > 0
    ).astype(np.float32)
    return normals


def concatenate_surfaces(surfaces: Sequence[Surface]) -> Surface:
    """Joins surfaces into one mesh, such as lh and rh.

    Only the value names every surface has are kept, and the reference only when
    all the surfaces have matching grids.

    Args:
        surfaces: The surfaces to join. Their faces are renumbered, in the order
            given.

    Returns:
        Surface: The joined surface.

    Raises:
        ValueError: If `surfaces` is empty.
    """
    if not surfaces:
        raise ValueError("Cannot concatenate an empty list.")

    counts = [s.num_vertices for s in surfaces]
    shifts: np.ndarray = np.cumsum([0, *counts[:-1]])
    vertices = np.concatenate([s.vertices for s in surfaces])
    faces = np.concatenate(
        [
            s.faces.astype(np.int64) + shift
            for s, shift in zip(surfaces, shifts, strict=True)
        ]
    )
    names = [
        n
        for n in surfaces[0].vertex_data
        if all(n in s.vertex_data for s in surfaces[1:])
    ]
    vertex_data = {
        n: np.concatenate([s.vertex_data[n] for s in surfaces]) for n in names
    }
    first = surfaces[0].reference
    same = first is not None and all(
        s.reference is not None and s.reference.matches(first) for s in surfaces
    )

    return Surface(
        vertices, faces, vertex_data=vertex_data, reference=first if same else None
    )


def keep_vertices(surface: Surface, mask: np.ndarray) -> Surface:
    """Keeps only the vertices where `mask` is True, such as the cortex of a label.

    Faces that touch a removed vertex are removed, the kept faces are renumbered,
    and `vertex_data` is sliced to match; the reference is kept.

    Args:
        surface: The surface to cut.
        mask: (V,) bool, True for the vertices to keep.

    Returns:
        Surface: The kept part. Its vertex numbering differs from `surface`'s.

    Raises:
        ValueError: If `mask` is not a boolean array with one entry per vertex.
    """
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != (surface.num_vertices,):
        raise ValueError(
            f"Expected a boolean mask of shape ({surface.num_vertices},), got "
            f"{mask.dtype} of shape {mask.shape}."
        )

    faces: np.ndarray = surface.faces.astype(np.intp)
    kept_faces = faces[mask[faces].all(axis=1)]
    new_index = np.full(surface.num_vertices, -1, dtype=np.intp)
    new_index[mask] = np.arange(np.count_nonzero(mask))

    return Surface(
        surface.vertices[mask],
        new_index[kept_faces],
        vertex_data={
            name: values[mask] for name, values in surface.vertex_data.items()
        },
        reference=surface.reference,
    )
