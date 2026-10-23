"""Values of an image sampled along a tractogram's streamlines."""

import numpy as np

from larmor.types import Tractogram, Volume

# Points sampled at a time.
_CHUNK = 1 << 20


def sample_volume(tractogram: Tractogram, volume: Volume) -> np.ndarray:
    """Samples an image at every point of a tractogram, with trilinear interpolation.

    A neighbour beyond the edge of the image takes the edge voxel's value, and a
    point whose nearest voxel is outside the image gets 0. The values are scaled by
    the image's slope and intercept.

    Args:
        tractogram: The streamlines, in world coordinates (mm).
        volume: A 3D image, or a 4D one with a single volume.

    Returns:
        The (P,) float32 values, one per point, in the order of `tractogram.points`.

    Raises:
        ValueError: If the image is 4D with more than one volume.

    References:
        Follows MRtrix3's `tcksample` with its default (trilinear) interpolation.
        Tournier, J.-D. et al. MRtrix3: A fast, flexible and open software
        framework for medical image processing and visualisation. NeuroImage 202,
        116137 (2019). doi:10.1016/j.neuroimage.2019.116137.
    """
    data = volume.data
    if data.ndim == 4:
        if data.shape[3] != 1:
            raise ValueError(
                f"Sampling needs a 3D image or a 4D one with one volume; got shape "
                f"{data.shape}."
            )
        data = data[..., 0]
    # The image as a flat array of voxels, float32 kept as it is and other types
    # as float64.
    kind = np.float32 if data.dtype == np.float32 else np.float64
    flat = np.ascontiguousarray(data, dtype=kind).ravel()
    shape = np.array(data.shape[:3])
    to_voxel = np.linalg.inv(volume.affine)
    values = np.empty(tractogram.num_points, np.float32)
    for start in range(0, tractogram.num_points, _CHUNK):
        points = tractogram.points[start : start + _CHUNK].astype(np.float64)
        voxel = points @ to_voxel[:3, :3].T + to_voxel[:3, 3]
        values[start : start + _CHUNK] = _trilinear(
            flat, shape, voxel, volume.slope, volume.intercept
        )
    return values


def _trilinear(
    flat: np.ndarray,
    shape: np.ndarray,
    voxel: np.ndarray,
    slope: float,
    intercept: float,
) -> np.ndarray:
    """Returns the scaled values (N,) at the voxel coordinates `voxel` (N, 3).

    `flat` is the image as a flat array and `shape` its (3,) shape; a coordinate
    whose nearest voxel is outside the image gives 0.
    """
    inside = np.all((voxel > -0.5) & (voxel < shape - 0.5), axis=1)
    voxel = voxel[inside]
    lower = np.floor(voxel)
    fraction = voxel - lower
    low = lower.astype(np.int64)
    # Neighbours beyond the edge are the edge voxel itself.
    high = np.minimum(low + 1, shape - 1)
    np.maximum(low, 0, out=low)
    strides = np.array([shape[1] * shape[2], shape[2], 1])
    low *= strides
    high *= strides
    fx, fy, fz = fraction[:, 0], fraction[:, 1], fraction[:, 2]
    # Interpolated along z, then y, then x.
    planes = []
    for x in (low[:, 0], high[:, 0]):
        rows = []
        for y in (low[:, 1], high[:, 1]):
            near = flat[x + y + low[:, 2]]
            far = flat[x + y + high[:, 2]]
            rows.append(near + fz * (far - near))
        planes.append(rows[0] + fy * (rows[1] - rows[0]))
    result = np.zeros(len(inside), np.float64)
    result[inside] = (planes[0] + fx * (planes[1] - planes[0])) * slope + intercept
    return result
