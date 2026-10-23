"""The triangle mesh type."""

from dataclasses import dataclass, field

import numpy as np

from larmor.types._arrays import check_data
from larmor.types.grid import Grid


@dataclass(eq=False)
class Surface:
    """A triangle mesh in world coordinates, with named values per vertex.

    Surfaces compare by identity (`==` is `is`); compare the arrays to compare
    contents.

    Args:
        vertices: The vertices (V, 3) in world coordinates (RAS+) in mm, stored as
            float32.
        faces: The vertex indices of each triangle (F, 3), counter-clockwise seen
            from outside so that normals point out, stored as uint32.
        vertex_data: Named values per vertex, each with first axis V, such as
            "curv" or "thickness". Empty by default.
        reference: The voxel grid of the volume the surface was made on, if
            known. Defaults to None.

    Raises:
        ValueError: If `vertices` is not (V, 3) or `faces` is not (F, 3), the faces
            are not integers or an index is outside [0, V), or a `vertex_data` value
            does not have V entries along its first axis.
    """

    vertices: np.ndarray
    faces: np.ndarray
    vertex_data: dict[str, np.ndarray] = field(default_factory=dict)
    reference: Grid | None = None

    def __post_init__(self) -> None:
        """Checks the shapes and the face indices, and converts the arrays' types."""
        vertices = np.asarray(self.vertices, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(
                f"Expected vertices of shape (V, 3), got {vertices.shape}."
            )

        faces = np.asarray(self.faces)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"Expected faces of shape (F, 3), got {faces.shape}.")
        if faces.size > 0:
            if not np.issubdtype(faces.dtype, np.integer):
                raise ValueError(f"Faces must be integers, got {faces.dtype}.")
            min_f, max_f = faces.min(), faces.max()
            if min_f < 0 or max_f >= len(vertices):
                V = vertices.shape[0]
                raise ValueError(
                    f"Face indices must be in [0, {V}), got [{min_f}, {max_f}]."
                )

        self.vertices = vertices
        self.faces = faces.astype(np.uint32)
        self.vertex_data = check_data(
            self.vertex_data, "vertex_data", self.num_vertices
        )

    @property
    def num_vertices(self) -> int:
        """The number of vertices, V."""
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        """The number of faces, F."""
        return len(self.faces)
