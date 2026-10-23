"""Reads and writes FreeSurfer surfaces, values per vertex and labels."""

from pathlib import Path

import numpy as np

from larmor.io.formats._affine import apply_affine
from larmor.types import Grid, Surface

# The first bytes of a FreeSurfer surface and of a FreeSurfer per-vertex file.
TRIANGLE_MAGIC = b"\xff\xff\xfe"
CURV_MAGIC = b"\xff\xff\xff"


def read_freesurfer_surface(path: str | Path) -> Surface:
    """Reads a FreeSurfer triangle surface, such as `lh.white` or `lh.pial`.

    FreeSurfer stores vertices in tkregister RAS, centred on the volume the surface
    was made on, and describes that volume in a footer. With a valid footer the
    vertices are converted to scanner RAS, unless the footer says they already are,
    and the footer's volume becomes the reference. Without one they are kept as
    stored and the reference is None.

    Args:
        path: The file.

    Returns:
        Surface: The mesh, with no values per vertex.

    Raises:
        ValueError: If the file is not a FreeSurfer triangle file or is truncated.
    """
    data: bytes = Path(path).read_bytes()
    if data[:3] != TRIANGLE_MAGIC:
        raise ValueError(
            f"{path} is not a FreeSurfer triangle surface: its header does not "
            "start with FF FF FE."
        )
    end: int = data.find(b"\n\n", 3)

    if end == -1:
        raise ValueError(
            f"{path} is truncated: no blank line ends the header's comment."
        )
    offset = end + 2
    if len(data) < offset + 8:
        raise ValueError(f"{path} is truncated: it ends before its vertex count.")
    v, f = (int(n) for n in np.frombuffer(data, ">i4", count=2, offset=offset))

    offset += 8
    if len(data) < offset + 12 * v + 12 * f:
        raise ValueError(f"{path} is truncated: it ends before its {f} faces.")

    vertices = np.frombuffer(data, ">f4", 3 * v, offset).reshape(v, 3)
    offset += 12 * v
    faces = np.frombuffer(data, ">i4", 3 * f, offset).reshape(f, 3)
    offset += 12 * f

    try:
        info = _read_volume_info(data, offset)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    if info is None:
        return Surface(vertices, faces)
    reference, scanner = info
    if not scanner:
        vertices = tkregister_to_scanner(vertices, reference)
    return Surface(vertices, faces, reference=reference)


def read_freesurfer_values(path: str | Path) -> np.ndarray:
    """Reads a FreeSurfer file of values per vertex, such as `lh.curv`.

    Args:
        path: The per-vertex file, such as `lh.curv`.

    Returns:
        np.ndarray: (V,) float32, one value per vertex.

    Raises:
        ValueError: If the file is not in the "new curv" format, holds more than
            one value per vertex, or is truncated.
    """
    data: bytes = Path(path).read_bytes()
    if data[:3] != CURV_MAGIC:
        raise ValueError(
            f"{path} is not a FreeSurfer per-vertex file in the new curv format: "
            "its header does not start with FF FF FF."
        )
    if len(data) < 15:
        raise ValueError(f"{path} is truncated: it ends inside the header.")

    v, f, per_vertex = (int(n) for n in np.frombuffer(data, ">i4", count=3, offset=3))

    if per_vertex != 1:
        raise ValueError(
            f"{path}: expected 1 value per vertex, got {per_vertex} (an old quad "
            "surface also starts with FF FF FF, and is not supported)."
        )

    if len(data) < 15 + 4 * v:
        raise ValueError(f"{path} is truncated: it ends before its {v} values.")

    values: np.ndarray = np.frombuffer(data, ">f4", count=v, offset=15).astype(
        np.float32
    )
    return values


def write_freesurfer_surface(surface: Surface, path: str | Path) -> None:
    """Writes a surface as a FreeSurfer triangle file.

    With a reference, the vertices are converted back to tkregister RAS and the
    reference is written in the footer. Values per vertex are not written; use
    `write_freesurfer_values`.

    Args:
        surface: The surface to write.
        path: The file to write.
    """
    vertices: np.ndarray = surface.vertices.astype(np.float64)
    faces: np.ndarray = surface.faces.astype(np.int32)
    ref = surface.reference
    if ref is not None:
        vertices = apply_affine(_tkr_affine(ref) @ np.linalg.inv(ref.affine), vertices)

    V, F = len(vertices), len(faces)

    parts: list[bytes] = [
        TRIANGLE_MAGIC,
        b"created by larmor\n\n",
        np.array([V, F], dtype=">i4").tobytes(),
        vertices.astype(">f4").tobytes(),
        faces.astype(">i4").tobytes(),
    ]

    if ref is not None:
        parts.append(_volume_info_bytes(ref))
    Path(path).write_bytes(b"".join(parts))


def write_freesurfer_values(
    values: np.ndarray, path: str | Path, *, num_faces: int = 0
) -> None:
    """Writes values per vertex as a FreeSurfer "new curv" file, such as `lh.curv`.

    Args:
        values: (V,) values, written as float32.
        path: The file.
        num_faces: The number of faces of the mesh, stored in the header.
            Defaults to 0.

    Raises:
        ValueError: If `values` is not one-dimensional.
    """
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("Cannot save multi-dimensional property as freesurfer data.")

    header = CURV_MAGIC + np.array([len(values), num_faces, 1], dtype=">i4").tobytes()
    Path(path).write_bytes(header + values.astype(">f4").tobytes())


def read_freesurfer_label(path: str | Path) -> np.ndarray:
    """Reads the vertex indices of a FreeSurfer label file, such as `lh.cortex.label`.

    Only the indices are returned, not the coordinates and values of the file.

    Args:
        path: The label file.

    Returns:
        np.ndarray: (N,) int64 vertex indices, in the file's order.

    Raises:
        ValueError: If the file is not text, the count is not a number or is
            negative, or the file has fewer vertex lines than its count or a
            malformed one.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{path} is not a label file: it is not text.") from error
    try:
        count = int(lines[1])
    except (IndexError, ValueError) as error:
        raise ValueError(
            f"{path}: expected the vertex count on the second line of a label file."
        ) from error
    if count < 0:
        raise ValueError(f"{path}: the vertex count is negative ({count}).")
    rows = [line.split() for line in lines[2 : 2 + count]]
    if len(rows) < count:
        raise ValueError(f"{path} is truncated: {len(rows)} of {count} vertex lines.")
    if any(len(row) != 5 for row in rows):
        raise ValueError(f"{path}: every vertex line must hold 5 numbers.")
    try:
        indices = np.array([int(row[0]) for row in rows], dtype=np.int64)
    except ValueError as error:
        raise ValueError(f"{path}: a vertex index is not an integer.") from error
    return indices


def _read_volume_info(data: bytes, offset: int) -> tuple[Grid, bool] | None:
    """Returns the grid in the volume info footer at `offset`, or None if not valid.

    The flag says whether the vertices are already in scanner RAS (FreeSurfer's
    useRealRAS) rather than tkregister RAS.

    Raises:
        ValueError: If a valid footer lacks a field or holds a malformed one.
    """
    if len(data) < offset + 4:
        return None
    if np.frombuffer(data, ">i4", count=1, offset=offset)[0] == 20:
        scanner, offset = False, offset + 4
    elif len(data) >= offset + 12:
        head = np.frombuffer(data, ">i4", count=3, offset=offset).tolist()
        if head not in ([2, 0, 20], [2, 1, 20]):
            return None
        scanner, offset = head[1] == 1, offset + 12
    else:
        return None
    fields: dict[str, str] = {}
    for line in data[offset:].split(b"\n")[:8]:
        key, _, value = line.decode("ascii", errors="replace").partition("=")
        fields[key.strip()] = value.strip()
    if fields.get("valid", "").split()[:1] != ["1"]:
        return None

    try:
        c, r, s = (int(n) for n in fields["volume"].split())
        sizes, x, y, z, cras = (
            np.array(fields[k].split(), dtype=np.float64)
            for k in ("voxelsize", "xras", "yras", "zras", "cras")
        )
        grid = grid_from_volume_info((c, r, s), sizes, np.column_stack([x, y, z]), cras)
    except (KeyError, ValueError) as error:
        raise ValueError(f"Malformed volume info in the footer: {error}.") from error
    return grid, scanner


def grid_from_volume_info(
    shape: tuple[int, int, int],
    sizes: np.ndarray,
    directions: np.ndarray,
    centre: np.ndarray,
) -> Grid:
    """Returns the grid of a FreeSurfer volume geometry, whose affine is `Norig`.

    Args:
        shape: The volume's size in voxels.
        sizes: (3,) voxel sizes in mm.
        directions: (3, 3) unit directions of the voxel axes in RAS, as columns
            (FreeSurfer's xras, yras, zras).
        centre: (3,) RAS of the volume's centre (FreeSurfer's cras).

    Raises:
        ValueError: If the arrays do not have these shapes.
    """
    if np.shape(sizes) != (3,) or np.shape(directions) != (3, 3):
        raise ValueError("Expected 3 voxel sizes and 3 directions of 3 values.")
    if np.shape(centre) != (3,):
        raise ValueError("Expected a centre of 3 values.")
    rotation = np.asarray(directions, dtype=np.float64) * np.asarray(sizes)
    affine = np.eye(4)
    affine[:3, :3] = rotation
    affine[:3, 3] = np.asarray(centre) - rotation @ (np.array(shape) / 2)
    return Grid(shape, affine)


def tkregister_to_scanner(points: np.ndarray, grid: Grid) -> np.ndarray:
    """Returns (N, 3) `points` moved from `grid`'s tkregister RAS to scanner RAS."""
    return apply_affine(
        grid.affine @ np.linalg.inv(_tkr_affine(grid)), points.astype(np.float64)
    )


def _volume_info_bytes(grid: Grid) -> bytes:
    """Returns the volume info footer that describes `grid`."""
    sizes = np.linalg.norm(grid.affine[:3, :3], axis=0)
    directions = grid.affine[:3, :3] / sizes
    cras = grid.affine[:3, :3] @ (np.array(grid.shape) / 2) + grid.affine[:3, 3]
    c, r, s = grid.shape
    lines: list[str] = [
        "valid = 1  # volume info valid",
        "filename = ",
        f"volume = {c} {r} {s}",
        f"voxelsize = {_floats(sizes)}",
        f"xras   = {_floats(directions[:, 0])}",
        f"yras   = {_floats(directions[:, 1])}",
        f"zras   = {_floats(directions[:, 2])}",
        f"cras   = {_floats(cras)}",
    ]
    return np.array([2, 0, 20], dtype=">i4").tobytes() + "".join(
        line + "\n" for line in lines
    ).encode("ascii")


def _floats(values: np.ndarray) -> str:
    """Returns the numbers as text in FreeSurfer's style, with 15 digits."""
    return " ".join(f"{x:.15e}" for x in values)


def _tkr_affine(grid: Grid) -> np.ndarray:
    """Returns the voxel to tkregister RAS affine of a volume of `grid`'s size."""
    c, r, s = grid.shape
    xs, ys, zs = np.linalg.norm(grid.affine[:3, :3], axis=0)

    return np.asarray(
        [
            [-xs, 0, 0, xs * c / 2],
            [0, 0, zs, -zs * s / 2],
            [0, -ys, 0, ys * r / 2],
            [0, 0, 0, 1],
        ]
    )
