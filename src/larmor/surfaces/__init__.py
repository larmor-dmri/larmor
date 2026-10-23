"""Operations on surface meshes, such as joining, cutting and vertex normals."""

from larmor.surfaces.mesh import concatenate_surfaces, keep_vertices, vertex_normals

__all__ = [
    "concatenate_surfaces",
    "keep_vertices",
    "vertex_normals",
]
