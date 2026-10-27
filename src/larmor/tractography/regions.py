"""Spheres, boxes and masks that select and crop streamlines."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from larmor.types import Volume

# Number of points tested at a time, here and in the editing functions.
CHUNK_POINTS = 1_000_000


@dataclass(frozen=True, eq=False)
class Sphere:
    """A sphere in world coordinates.

    A point is inside when its squared distance to the centre is at most the
    squared radius, computed in float32 like the points of a tractogram.

    Args:
        centre: (3,) the centre in world coordinates (RAS+), mm. Stored as a
            read-only float64 array.
        radius: The radius in mm, finite and not negative.

    Raises:
        ValueError: If `centre` is not three finite values or `radius` is negative
            or not finite.
    """

    centre: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        """Checks the centre and the radius, and makes the centre read-only."""
        centre = np.array(self.centre, dtype=np.float64)
        if centre.shape != (3,) or not np.isfinite(centre).all():
            raise ValueError(
                f"A sphere's centre must be three finite values, got {self.centre}."
            )
        radius = float(self.radius)
        if not math.isfinite(radius) or radius < 0:
            raise ValueError(
                f"A sphere's radius must be finite and not negative, got {radius}."
            )
        centre.flags.writeable = False
        object.__setattr__(self, "centre", centre)
        object.__setattr__(self, "radius", radius)


@dataclass(frozen=True, eq=False)
class Box:
    """A box in world coordinates, its edges along the world axes.

    A point is inside when it is at most half the size from the centre along
    each of x, y and z, computed in float32 like the points of a tractogram.

    Args:
        centre: (3,) the centre in world coordinates (RAS+), mm. Stored as a
            read-only float64 array.
        size: (3,) the lengths of the edges along x, y and z, mm, finite and not
            negative. Stored as a read-only float64 array.

    Raises:
        ValueError: If `centre` is not three finite values, or `size` is not three
            finite values that are not negative.
    """

    centre: np.ndarray
    size: np.ndarray

    def __post_init__(self) -> None:
        """Checks the centre and the size, and makes both read-only."""
        centre = np.array(self.centre, dtype=np.float64)
        if centre.shape != (3,) or not np.isfinite(centre).all():
            raise ValueError(
                f"A box's centre must be three finite values, got {self.centre}."
            )
        size = np.array(self.size, dtype=np.float64)
        if size.shape != (3,) or not np.isfinite(size).all() or (size < 0).any():
            raise ValueError(
                "A box's size must be three finite values, none negative, got "
                f"{self.size}."
            )
        centre.flags.writeable = False
        size.flags.writeable = False
        object.__setattr__(self, "centre", centre)
        object.__setattr__(self, "size", size)


type Region = Sphere | Box | Volume
"""A sphere, a box, or a 3D mask `Volume` on any grid (or 4D with one volume)."""


def points_inside(points: np.ndarray, region: Region) -> np.ndarray:
    """Tests which points lie inside a region.

    For a sphere, a point is inside when its squared distance to the centre is at
    most the squared radius. For a box, a point is inside when it is at most half
    the size from the centre along each axis. For a mask, a point is inside when
    its nearest voxel, rounding halves away from zero, is in the image and its
    scaled value, rounded the same way, is not zero. Only the given points are
    tested, with no interpolation between them.

    Args:
        points: (P, 3) points in world coordinates (RAS+), mm.
        region: A `Sphere`, a `Box`, or a 3D mask `Volume` on any grid.

    Returns:
        np.ndarray: (P,) bool, True where the point is inside the region.

    Raises:
        TypeError: If `region` is not a `Sphere`, a `Box` or a `Volume`.
        ValueError: If `points` is not (P, 3) or the mask is not 3D (a 4D mask
            with a single volume counts as 3D).

    References:
        Follows MRtrix3's region tests in tckedit and tckgen. Tournier, J.-D. et al.
        MRtrix3: A fast, flexible and open software framework for medical image
        processing and visualisation. NeuroImage 202, 116137 (2019).
        doi:10.1016/j.neuroimage.2019.116137
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points of shape (P, 3), got {points.shape}.")
    test = point_test(region)
    inside = np.empty(len(points), dtype=bool)
    for start in range(0, len(points), CHUNK_POINTS):
        stop = min(start + CHUNK_POINTS, len(points))
        inside[start:stop] = test(points[start:stop])
    return inside


def point_test(region: Region) -> Callable[[np.ndarray], np.ndarray]:
    """Returns the region prepared as a test of (P, 3) points, as in `points_inside`.

    The preparation is done once, so that the test can be applied to many chunks of
    points.

    Args:
        region: A `Sphere`, a `Box` or a mask `Volume`.

    Returns:
        A function taking (P, 3) points and returning a boolean array of shape (P,),
        True for the points inside.

    Raises:
        TypeError: If `region` is not a `Sphere`, a `Box` or a `Volume`.
        ValueError: If the mask is not 3D (or 4D with a single volume).
    """
    if isinstance(region, Sphere):
        return _sphere_test(region)
    if isinstance(region, Box):
        return _box_test(region)
    if isinstance(region, Volume):
        return _mask_test(region)
    raise TypeError(
        "A region must be a Sphere, a Box or a mask Volume, got "
        f"{type(region).__name__}."
    )


def _sphere_test(sphere: Sphere) -> Callable[[np.ndarray], np.ndarray]:
    """Returns the test of points at most the radius from the sphere's centre.

    The squared distances are computed in float32, like the points of a tractogram.
    """
    centre = sphere.centre.astype(np.float32)
    squared_radius = np.float32(sphere.radius) ** 2

    def test(points: np.ndarray) -> np.ndarray:
        offsets = points.astype(np.float32, copy=False) - centre
        inside: np.ndarray = np.einsum("ij,ij->i", offsets, offsets) <= squared_radius
        return inside

    return test


def _box_test(box: Box) -> Callable[[np.ndarray], np.ndarray]:
    """Returns the test of points at most half the size from the centre on each axis.

    The offsets are computed in float32, like the points of a tractogram.
    """
    centre = box.centre.astype(np.float32)
    half = (box.size / 2).astype(np.float32)

    def test(points: np.ndarray) -> np.ndarray:
        offsets = np.abs(points.astype(np.float32, copy=False) - centre)
        inside: np.ndarray = np.all(offsets <= half, axis=1)
        return inside

    return test


def _nonzero_voxels(mask: Volume) -> np.ndarray:
    """Returns the (X, Y, Z) bool voxels of `mask` whose scaled value is not 0.

    The values are rounded half away from zero first.

    Raises:
        ValueError: If the mask is neither 3D nor 4D with a single volume.
    """
    data = mask.data
    # A 4D image with a single volume is a 3D mask stored with a trailing axis.
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(
            "A mask region must be 3D (or 4D with a single volume), got a volume "
            f"of shape {mask.data.shape}."
        )
    integer = np.issubdtype(data.dtype, np.integer)
    if integer and mask.slope == 1.0 and mask.intercept == 0.0:
        return np.asarray(data != 0)
    # A value rounds to a non-zero integer exactly when its magnitude is at least
    # one half; NaN compares False and so is outside.
    values = data.astype(np.float64) * mask.slope + mask.intercept
    return np.asarray(np.abs(values) >= 0.5)


def _mask_test(mask: Volume) -> Callable[[np.ndarray], np.ndarray]:
    """Returns the test of points whose nearest voxel is a non-zero voxel of `mask`."""
    nonzero_voxels = _nonzero_voxels(mask)
    shape = nonzero_voxels.shape
    nonzero = np.ascontiguousarray(nonzero_voxels).ravel()
    world_to_voxel = np.linalg.inv(mask.affine)

    def test(points: np.ndarray) -> np.ndarray:
        within = np.ones(len(points), dtype=bool)
        flat_index = np.zeros(len(points), dtype=np.intp)
        for axis in range(3):
            # Column by column, to avoid a float64 copy of all the points.
            row = world_to_voxel[axis]
            coordinates = points[:, 0] * row[0]
            coordinates += points[:, 1] * row[1]
            coordinates += points[:, 2] * row[2]
            coordinates += row[3]
            # Halves away from zero: -0.5 rounds to -1 (outside), size - 0.5 to
            # size. NaN compares False, so it is outside too.
            on_axis = (coordinates > -0.5) & (coordinates < shape[axis] - 0.5)
            within &= on_axis
            # Points off the image (NaN included) look up voxel 0 instead, which
            # keeps the cast to integers valid; `within` drops them afterwards.
            np.putmask(coordinates, ~on_axis, 0.0)
            rounded = np.rint(coordinates)
            # rint sends halves to even; in the image they are all positive, so the
            # ones it rounded down go up instead.
            rounded += coordinates - rounded == 0.5
            flat_index *= shape[axis]
            flat_index += rounded.astype(np.intp)
        inside: np.ndarray = nonzero[flat_index] & within
        return inside

    return test
