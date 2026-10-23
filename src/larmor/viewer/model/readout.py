"""Reads the main volume's voxel and value at the cursor for the crosshair."""

from dataclasses import dataclass

import numpy as np

from larmor.types import Grid, Volume

type Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Readout:
    """The cursor's position and, with a main volume, its voxel and value there.

    Attributes:
        position: The cursor in world coordinates, mm.
        name: The main volume's name; None without one.
        voxel: The nearest voxel (with the volume index of a 4D image); None
            outside the grid.
        value: The value there in scaled units; None outside the grid.
    """

    position: Vec3
    name: str | None = None
    voxel: tuple[int, ...] | None = None
    value: float | None = None


def read_volume(
    volume: Volume, position: Vec3, volume_index: int, interpolate: bool
) -> tuple[tuple[int, ...] | None, float | None]:
    """Returns the nearest voxel and the value, trilinear if interpolate, at a point."""
    data = volume.data[..., volume_index] if volume.data.ndim == 4 else volume.data
    shape = np.array(data.shape[:3])
    inverse = np.linalg.inv(volume.affine)
    point = inverse[:3, :3] @ np.asarray(position, dtype=float) + inverse[:3, 3]
    nearest = np.floor(point + 0.5).astype(int)  # halves up, as nearest sampling
    if np.any(nearest < 0) or np.any(nearest >= shape):
        return None, None
    voxel = tuple(int(v) for v in nearest)
    if volume.data.ndim == 4:
        voxel = (*voxel, volume_index)
    if interpolate:
        raw = _trilinear(data, np.clip(point, 0, shape - 1))
    else:
        raw = float(data[nearest[0], nearest[1], nearest[2]])
    return voxel, float(raw * volume.slope + volume.intercept)


def closest_axis(grid: Grid, axis: int) -> int:
    """Returns the grid's voxel axis closest to a world axis (0, 1 or 2)."""
    return int(np.argmax(np.abs(grid.affine[axis, :3])))


def slice_index(grid: Grid, position: Vec3, axis: int) -> tuple[int, int]:
    """Returns the slice of a grid at a world point along a world axis, and the count.

    The slice is along the grid's voxel axis closest to the world axis, rounded as
    the readout's voxel is; it can lie outside 0 to count - 1.

    Returns:
        The slice index and the number of slices.
    """
    j = closest_axis(grid, axis)
    inverse = np.linalg.inv(grid.affine)
    point = inverse[:3, :3] @ np.asarray(position, dtype=float) + inverse[:3, 3]
    return int(np.floor(point[j] + 0.5)), grid.shape[j]


def _trilinear(data: np.ndarray, point: np.ndarray) -> float:
    """Interpolates the data at a voxel position inside the grid from 8 neighbours."""
    low = np.floor(point).astype(int)
    frac = point - low
    high = np.minimum(low + 1, np.array(data.shape[:3]) - 1)
    total = 0.0
    for ci, wi in ((low[0], 1 - frac[0]), (high[0], frac[0])):
        for cj, wj in ((low[1], 1 - frac[1]), (high[1], frac[1])):
            for ck, wk in ((low[2], 1 - frac[2]), (high[2], frac[2])):
                weight = wi * wj * wk
                if weight:
                    total += weight * float(data[ci, cj, ck])
    return total
