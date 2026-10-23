"""Resampling streamlines to a step size or a number of points."""

import math
import numbers

import numpy as np
from numpy.typing import NDArray

from larmor import _core
from larmor._threads import n_threads
from larmor.types import Tractogram


def resample(
    tractogram: Tractogram,
    *,
    step_size: float | None = None,
    num_points: int | None = None,
    n_jobs: int = -1,
) -> Tractogram:
    """Resamples every streamline to a step size or to a number of points.

    The new points lie on a cubic Hermite spline (tension 0.1) through each
    streamline's points. With `num_points`, they are at equal arc lengths of the
    polyline. With `step_size`, they start from the middle point and go outward to
    each end, each at `step_size` (straight-line distance) from the previous one.
    The end points are kept, exactly with `step_size` and up to float32 rounding with
    `num_points`, unless the first or last segment has zero length. Streamlines of
    fewer than 2 points are kept as they are. Floating-point `point_data` is
    interpolated linearly and keeps its dtype; other entries take the value of the
    nearer original point. `streamline_data` and `reference` are kept. The result
    does not depend on `n_jobs`.

    Args:
        tractogram: The tractogram to resample.
        step_size: The distance between consecutive new points, in mm. Defaults to
            None.
        num_points: The number of points of every new streamline, at least 2.
            Defaults to None.
        n_jobs: Number of threads, at most the cores; -1 is all cores. Defaults to
            -1.

    Returns:
        Tractogram: A new tractogram with the resampled streamlines, in the same
            order.

    Raises:
        ValueError: If neither or both of `step_size` and `num_points` are given,
            `step_size` is not a positive finite number, `num_points` is not an
            integer >= 2, or `n_jobs` is not an integer, 0 or below -1.

    References:
        Follows MRtrix3's tckresample (version 3.0.8), options -step_size and
        -num_points. Tournier, J.-D. et al. MRtrix3: A fast, flexible and open
        software framework for medical image processing and visualisation.
        NeuroImage 202, 116137 (2019). doi:10.1016/j.neuroimage.2019.116137
    """
    if (step_size is None) == (num_points is None):
        raise ValueError("Give exactly one of step_size and num_points.")
    if step_size is not None and not (
        math.isfinite(step_size)
        and abs(step_size) <= float(np.finfo(np.float32).max)
        and np.float32(step_size) > 0
    ):
        raise ValueError(
            "step_size must be a positive number that float32 can hold, "
            f"got {step_size}."
        )
    if num_points is not None and not (
        _is_integer(num_points) and 2 <= num_points <= np.iinfo(np.int64).max
    ):
        raise ValueError(f"num_points must be an integer >= 2, got {num_points!r}.")
    if not _is_integer(n_jobs):
        raise ValueError(f"n_jobs must be an integer, got {n_jobs!r}.")
    threads = n_threads(int(n_jobs))

    points, offsets, segments, fractions = _core.resample_streamlines(
        np.ascontiguousarray(tractogram.points, dtype=np.float32),
        np.ascontiguousarray(tractogram.offsets, dtype=np.int64),
        float(step_size) if step_size is not None else 0.0,
        int(num_points) if num_points is not None else 0,
        threads,
        positions=bool(tractogram.point_data),
    )
    point_data = {
        name: _interpolate(values, segments, fractions)
        for name, values in tractogram.point_data.items()
    }
    return Tractogram(
        points,
        offsets,
        point_data=point_data,
        streamline_data=dict(tractogram.streamline_data),
        reference=tractogram.reference,
    )


def _is_integer(value: object) -> bool:
    """Returns whether value is an integer and not a bool."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _interpolate(
    values: np.ndarray, segments: NDArray[np.int64], fractions: NDArray[np.float64]
) -> np.ndarray:
    """Returns the per-point `values` at the new points.

    Each new point lies at `fractions` of the way along the original segment
    `segments`; floats are interpolated linearly, other types take the nearer
    point's value.
    """
    if len(segments) == 0:
        return values[:0].copy()
    upper = np.minimum(segments + 1, len(values) - 1)
    shape = (-1,) + (1,) * (values.ndim - 1)
    if np.issubdtype(values.dtype, np.floating):
        weight = fractions.reshape(shape)
        lower_values, upper_values = values[segments], values[upper]
        # At fraction 0 or 1 the value is the point's own, even next to a NaN.
        with np.errstate(invalid="ignore"):
            mixed = (1.0 - weight) * lower_values + weight * upper_values
        mixed = np.where(weight == 0.0, lower_values, mixed)
        mixed = np.where(weight == 1.0, upper_values, mixed)
        interpolated: np.ndarray = mixed.astype(values.dtype)
        return interpolated
    nearest = np.where(fractions <= 0.5, segments, upper)
    nearest_values: np.ndarray = values[nearest]
    return nearest_values
