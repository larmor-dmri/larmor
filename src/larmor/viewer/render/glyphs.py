"""Builds the FOD glyphs of a slice as one mesh, with numpy alone."""

import threading
from dataclasses import dataclass
from functools import cache

import numpy as np
from scipy.spatial import ConvexHull

from larmor.sd import sh_layout
from larmor.sd.sh import build_basis_matrix
from larmor.sd.sphere import get_theta_phi
from larmor.types import Volume
from larmor.viewer.render.colour_space import physical_colours

# Step of the finite differences that give the SH's gradient on the sphere.
_EPSILON = 1e-4
# Voxels evaluated at once when the image's high amplitude is computed.
_CHUNK = 20_000
# Share of white mixed into the glyph colours, for bright, slightly pastel hues.
_WHITE = 0.3
# Percentile of the voxels' largest amplitudes that whole-image scaling and the
# threshold are relative to, as the image's maximum is often a lone outlier.
_PERCENTILE = 90.0


@dataclass(frozen=True)
class Sphere:
    """Holds a geodesic sphere's unit vertices and outward triangles."""

    vertices: np.ndarray  # (V, 3) float32
    faces: np.ndarray  # (F, 3) uint32, counter-clockwise seen from outside


@dataclass(frozen=True)
class SphereBasis:
    """Holds an SH basis evaluated on a sphere's vertices, for a slice's glyphs."""

    amplitudes: np.ndarray  # (C, V) float32, maps coefficients to amplitudes
    # (C, 3V) float32, maps coefficients to each vertex's unnormalised normal.
    normals: np.ndarray


@dataclass(frozen=True)
class Glyphs:
    """Holds the glyphs of a slice as one mesh, in world mm."""

    positions: np.ndarray  # (n V, 3) float32
    normals: np.ndarray  # (n V, 3) float32
    # (n V, 3) float32, the vertex's absolute direction as linear light
    colours: np.ndarray
    faces: np.ndarray  # (n F, 3) uint32
    count: int  # the number of glyphs, n


@cache
def sphere(detail: int) -> Sphere:
    """Returns an icosahedron with each face cut into detail^2 triangles."""
    g = (1 + 5**0.5) / 2
    ico = np.array(
        [(-1, g, 0), (1, g, 0), (-1, -g, 0), (1, -g, 0)]
        + [(0, -1, g), (0, 1, g), (0, -1, -g), (0, 1, -g)]
        + [(g, 0, -1), (g, 0, 1), (-g, 0, -1), (-g, 0, 1)],
        dtype=np.float64,
    )
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    points: list[list[np.ndarray]] = []
    triangles: list[list[int]] = []
    # Points (i, j) of a face, i + j <= detail, numbered row by row.
    grid = [(i, j) for i in range(detail + 1) for j in range(detail + 1 - i)]
    number = {p: k for k, p in enumerate(grid)}
    for face in ConvexHull(ico).simplices:
        a, b, c = ico[face]
        if np.dot(np.cross(b - a, c - a), a + b + c) < 0:  # outward
            b, c = c, b
        start = len(points) * len(grid)
        points.append(
            [a + (b - a) * i / detail + (c - a) * j / detail for i, j in grid]
        )
        for i, j in grid:
            if i + j < detail:
                up = (number[i, j], number[i + 1, j], number[i, j + 1])
                triangles.append([start + k for k in up])
            if i + j < detail - 1:
                down = (number[i + 1, j], number[i + 1, j + 1], number[i, j + 1])
                triangles.append([start + k for k in down])
    flat = np.concatenate(points)
    flat /= np.linalg.norm(flat, axis=1, keepdims=True)
    # The faces' edges share their points, so keep one of each.
    _, first, inverse = np.unique(
        np.round(flat, 6), axis=0, return_index=True, return_inverse=True
    )
    vertices = flat[first]
    faces = inverse.reshape(-1)[np.array(triangles)]
    return Sphere(vertices.astype(np.float32), faces.astype(np.uint32))


@cache
def sphere_basis(lmax: int, full: bool, basis: str, detail: int) -> SphereBasis:
    """Returns the basis on the sphere of `detail`, and the matrix giving normals."""
    u = sphere(detail).vertices.astype(np.float64)
    # Two tangents per vertex, with (t1, t2, u) right-handed.
    axis = np.eye(3)[np.argmin(np.abs(u), axis=1)]
    t1 = np.cross(u, axis)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(u, t1)

    def evaluate(dirs: np.ndarray) -> np.ndarray:
        """Returns the basis at directions, normalised first, as (C, V)."""
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        theta, phi = get_theta_phi(dirs)
        return build_basis_matrix(theta, phi, lmax, basis, full=full).T  # (C, V)

    b = evaluate(u)
    arc = 2 * np.arctan(_EPSILON)
    d1 = (evaluate(u + _EPSILON * t1) - evaluate(u - _EPSILON * t1)) / arc
    d2 = (evaluate(u + _EPSILON * t2) - evaluate(u - _EPSILON * t2)) / arc
    # For the surface r(u) u, the normal is r u - (dr/dt1) t1 - (dr/dt2) t2, linear
    # in the coefficients like r itself.
    normals = b[:, :, None] * u - d1[:, :, None] * t1 - d2[:, :, None] * t2
    return SphereBasis(
        b.astype(np.float32),
        normals.reshape(len(b), -1).astype(np.float32),
    )


@cache
def sphere_colours(detail: int) -> np.ndarray:
    """Returns a glyph's vertex colours, each absolute direction as linear light.

    Each colour is scaled so its largest component is 1, then mixed with some
    white, so lobes between the axes are as bright as lobes along them.
    """
    rgb = np.abs(sphere(detail).vertices).astype(np.float64)
    rgb /= rgb.max(axis=1, keepdims=True)
    colours = physical_colours((1 - _WHITE) * rgb + _WHITE)
    colours.flags.writeable = False
    return colours


class _Repeated:
    """Keeps the colours and faces of n glyphs, as prefixes of growing arrays."""

    def __init__(self) -> None:
        """Creates the empty arrays."""
        self._lock = threading.Lock()
        self._arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def get(self, detail: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns the colours (n V, 3) and faces (n F, 3) of n glyphs of a sphere."""
        s = sphere(detail)
        v, f = len(s.vertices), len(s.faces)
        with self._lock:
            colours, faces = self._arrays.get(detail, (np.zeros((0, 3)),) * 2)
            if len(faces) < n * f:
                size = max(n, 2 * len(faces) // f, 64)
                colours = np.tile(sphere_colours(detail), (size, 1))
                offsets = (np.arange(size, dtype=np.uint32) * v)[:, None, None]
                faces = (s.faces[None] + offsets).reshape(-1, 3)
                self._arrays[detail] = (colours, faces)
        return colours[: n * v], faces[: n * f]


_REPEATED = _Repeated()


def repeated(detail: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns the colours (n V, 3) and faces (n F, 3) of n glyphs, for any slice.

    Returns:
        The colours, as linear light, and the faces.
    """
    return _REPEATED.get(detail, n)


def _coefficients(volume: Volume, plane: np.ndarray) -> np.ndarray:
    """Returns stored values as SH coefficients, float32."""
    coeffs = plane.astype(np.float32)
    if volume.slope != 1.0 or volume.intercept != 0.0:
        coeffs *= volume.slope
        coeffs += volume.intercept
    return coeffs


def slice_glyphs(
    volume: Volume,
    axis: int,
    index: int,
    basis: str,
    detail: int,
    radius: float,
    threshold: float = 0.0,
    image_amplitude: float | None = None,
    anisotropy: float = 0.0,
    cursor_plane: tuple[int, float] | None = None,
    crop: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> Glyphs:
    """Returns the glyphs of one slice of an FOD, on voxel `axis` at `index`.

    Each glyph's largest amplitude reaches `radius` mm or, with `image_amplitude`,
    an amplitude of `image_amplitude` does (no glyphs if it is not above 0). A
    voxel gets no glyph if its largest amplitude is not above `threshold` and 0,
    or its anisotropy is below `anisotropy`. Negative amplitudes are drawn as 0.
    With `cursor_plane`, a world axis and a coordinate on it, the glyphs' centres
    are moved onto that plane, where the views draw the slices. With `crop`, the
    first and past-the-last voxel along each of the slice's two other axes (the
    lower axis first), only the glyphs inside are built.

    Returns:
        Glyphs: The glyphs.
    """
    s = sphere(detail)
    v = len(s.vertices)
    data = volume.data if volume.data.ndim == 4 else volume.data[..., None]
    layout = sh_layout(data.shape[3])
    if not 0 <= index < data.shape[axis]:
        return empty_glyphs()
    if image_amplitude is not None and image_amplitude <= 0:
        return empty_glyphs()  # an image with no positive high amplitude
    # The slice copied in its memory order, as in an F-ordered (NIfTI) array the
    # coefficients of one voxel lie a whole image apart.
    where: list[int | slice] = [slice(None)] * 4
    where[axis] = index
    plane = data[tuple(where)].copy(order="K")  # (a, b, C)
    flat = plane.reshape(-1, data.shape[3])
    chosen = _nonzero_voxels(flat, plane.shape[1], crop)
    coeffs = _coefficients(volume, flat[chosen])
    if anisotropy > 0:
        kept = sh_anisotropy(coeffs) >= anisotropy
        chosen, coeffs = chosen[kept], coeffs[kept]
    b = sphere_basis(*layout, basis, detail)
    amplitudes = coeffs @ b.amplitudes  # (n, V)
    largest = amplitudes.max(axis=1, initial=0.0)
    kept = largest > max(threshold, 0.0)
    if not kept.all():
        chosen, coeffs = chosen[kept], coeffs[kept]
        amplitudes, largest = amplitudes[kept], largest[kept]
    n = len(chosen)
    if n == 0:
        return empty_glyphs()

    centres = _glyph_centres(volume.affine, axis, index, chosen, plane.shape[:2])
    if cursor_plane is not None:
        centres[:, cursor_plane[0]] = cursor_plane[1]

    np.maximum(amplitudes, 0.0, out=amplitudes)
    if image_amplitude is None:
        amplitudes *= (radius / largest)[:, None]
    else:
        amplitudes *= np.float32(radius / image_amplitude)
    positions = amplitudes[:, :, None] * s.vertices
    positions += centres[:, None, :]
    normals = coeffs @ b.normals  # (n, 3V)
    colours, faces = _REPEATED.get(detail, n)
    return Glyphs(
        positions.reshape(n * v, 3), normals.reshape(n * v, 3), colours, faces, n
    )


def _nonzero_voxels(
    flat: np.ndarray,
    columns: int,
    crop: tuple[tuple[int, int], tuple[int, int]] | None,
) -> np.ndarray:
    """Returns the rows of a slice's (n, C) coefficients that are not all zero.

    With crop, only those inside it; `columns` is the slice's second size.
    """
    chosen = np.flatnonzero(np.any(flat != 0, axis=1))
    if crop is not None:
        (a0, a1), (b0, b1) = crop
        rows, cols = np.divmod(chosen, columns)
        chosen = chosen[(rows >= a0) & (rows < a1) & (cols >= b0) & (cols < b1)]
    return chosen


def _glyph_centres(
    affine: np.ndarray,
    axis: int,
    index: int,
    chosen: np.ndarray,
    plane_shape: tuple[int, ...],
) -> np.ndarray:
    """Returns the world centres, (n, 3) float32, of the chosen voxels of a slice.

    chosen holds flat indices into the slice's plane, of shape plane_shape.
    """
    others = [d for d in range(3) if d != axis]
    voxels = np.empty((len(chosen), 3), np.float32)
    voxels[:, others[0]], voxels[:, others[1]] = np.unravel_index(chosen, plane_shape)
    voxels[:, axis] = index
    a = affine.astype(np.float32)
    centres: np.ndarray = voxels @ a[:3, :3].T + a[:3, 3]
    return centres


def voxel_glyph(
    volume: Volume,
    voxel: tuple[int, int, int],
    basis: str,
    detail: int,
    radius: float = 1.0,
) -> Glyphs:
    """Returns one voxel's glyph centred at the origin, its largest amplitude at radius.

    Negative amplitudes are drawn as 0, and a voxel whose largest amplitude is not
    above 0 has no glyph.

    Returns:
        Glyphs: The glyph.
    """
    data = volume.data if volume.data.ndim == 4 else volume.data[..., None]
    layout = sh_layout(data.shape[3])
    coeffs = _coefficients(volume, np.asarray(data[voxel])[None])  # (1, C)
    b = sphere_basis(*layout, basis, detail)
    amplitudes = (coeffs @ b.amplitudes)[0]
    largest = float(amplitudes.max(initial=0.0))
    if largest <= 0:
        return empty_glyphs()
    s = sphere(detail)
    np.maximum(amplitudes, 0.0, out=amplitudes)
    positions = (amplitudes * np.float32(radius / largest))[:, None] * s.vertices
    normals = (coeffs @ b.normals).reshape(-1, 3)
    return Glyphs(positions, normals, sphere_colours(detail), s.faces, 1)


def sh_anisotropy(coeffs: np.ndarray) -> np.ndarray:
    """Returns how far each row's FOD is from a sphere, from its coefficients alone.

    The value is sqrt(1 - c0^2 / sum(c^2)): 0 for a sphere or a row of zeros,
    towards 1 for a sharp FOD.

    Returns:
        The anisotropy of each row.
    """
    power: np.ndarray = np.einsum("ij,ij->i", coeffs, coeffs)
    share = np.divide(
        coeffs[:, 0] ** 2, power, out=np.ones_like(power), where=power > 0
    )
    anisotropy: np.ndarray = np.sqrt(np.clip(1 - share, 0, 1))
    return anisotropy


def image_amplitude(volume: Volume, basis: str, detail: int) -> float:
    """Returns the 90th percentile of the voxels' largest amplitudes.

    Voxels that are all zero are left out.

    Returns:
        The amplitude; 0 for an empty image.
    """
    data = volume.data if volume.data.ndim == 4 else volume.data[..., None]
    layout = sh_layout(data.shape[3])
    b = sphere_basis(*layout, basis, detail)
    largest: list[np.ndarray] = []
    # A slice at a time, and its voxels in chunks, keeps the memory small.
    for i in range(data.shape[0]):
        flat = data[i].reshape(-1, data.shape[3])
        flat = flat[np.any(flat != 0, axis=1)]
        for start in range(0, len(flat), _CHUNK):
            chunk = _coefficients(volume, flat[start : start + _CHUNK])
            largest.append((chunk @ b.amplitudes).max(axis=1))
    if not largest:
        return 0.0
    return max(float(np.percentile(np.concatenate(largest), _PERCENTILE)), 0.0)


def empty_glyphs() -> Glyphs:
    """Returns an empty set of glyphs."""
    empty = np.zeros((0, 3), np.float32)
    return Glyphs(empty, empty, empty, np.zeros((0, 3), np.uint32), 0)
