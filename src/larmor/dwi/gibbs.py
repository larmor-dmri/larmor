"""Gibbs-ringing removal by local sub-voxel shifts."""

import numpy as np

from larmor import _core
from larmor._threads import n_threads
from larmor.types import Volume

# Voxels per call to the compiled kernel, at most; each call takes whole volumes.
CHUNK_VOXELS = 2**24


def remove_gibbs_ringing(
    volume: Volume,
    *,
    slice_axis: int = 2,
    num_shifts: int = 20,
    window: tuple[int, int] = (1, 3),
    n_jobs: int = -1,
) -> Volume:
    """Removes Gibbs ringing from a 3D or 4D volume, slice by slice.

    The local sub-voxel shift method (Kellner et al., 2016) re-interpolates each
    line of a slice at num_shifts sub-voxel shifts on each side of zero, and keeps
    for each voxel the shift with the least oscillation next to it. Lines wrap
    around at the edge of the slice. The method assumes full k-space sampling;
    along a partial Fourier direction some ringing may remain. Each slice is
    computed on its own, so the result does not depend on the number of threads.

    Args:
        volume: The image, (X, Y, Z) or (X, Y, Z, N); its slope and intercept
            are applied.
        slice_axis: The voxel axis across the acquired slices (0, 1 or 2); the
            other two span each slice. Defaults to 2.
        num_shifts: Shifts tried on each side of zero. Defaults to 20.
        window: The neighbours, from window[0] to window[1] + 1 voxels from the
            voxel, over which the oscillation is measured. Defaults to (1, 3).
        n_jobs: Number of threads, -1 for all cores. Defaults to -1.

    Returns:
        Volume: The image without ringing, float32, on the same grid.

    Raises:
        ValueError: If slice_axis is not 0, 1 or 2, num_shifts is below 1, the
            window is not 0 <= first <= last, n_jobs is 0 or below -1, or the
            data is complex.

    References:
        Kellner E, Dhital B, Kiselev VG, Reisert M. Gibbs-ringing artifact removal
        based on local subvoxel-shifts. Magn Reson Med 76 (2016) 1574-1581.
        Tournier JD et al. MRtrix3: A fast, flexible and open software framework for
        medical image processing and visualisation. NeuroImage 202 (2019) 116137.
    """
    if slice_axis not in (0, 1, 2):
        raise ValueError(f"slice_axis must be 0, 1 or 2, got {slice_axis}.")
    if num_shifts < 1:
        raise ValueError(f"num_shifts must be at least 1, got {num_shifts}.")
    first, last = window
    if not 0 <= first <= last:
        raise ValueError(f"The window must have 0 <= first <= last, got {window}.")
    threads = n_threads(n_jobs)
    if np.iscomplexobj(volume.data):
        raise ValueError(
            "The volume holds complex data; pass the magnitude instead "
            "(for example np.abs(data))."
        )

    data = volume.data if volume.data.ndim == 4 else volume.data[..., np.newaxis]
    result = np.empty(data.shape, dtype=np.float32)
    voxels_per_volume = int(np.prod(data.shape[:3]))
    per_call = max(1, CHUNK_VOXELS // max(1, voxels_per_volume))
    for start in range(0, data.shape[3], per_call):
        part = slice(start, start + per_call)
        chunk = np.ascontiguousarray(data[..., part], dtype=np.float32)
        if volume.slope != 1.0 or volume.intercept != 0.0:
            chunk = chunk * np.float32(volume.slope) + np.float32(volume.intercept)
        out = np.empty_like(chunk)
        _core.remove_gibbs_ringing(
            chunk, out, slice_axis, num_shifts, first, last, threads
        )
        result[..., part] = out
    return volume.with_data(result.reshape(volume.data.shape))
