"""Affine helpers shared by the file formats."""

import numpy as np


def apply_affine(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Returns (N, 3) `points` mapped by a (4, 4) `affine`, in the points' precision.

    float32 points stay float32 and float64 points give float64. Integer points of
    up to 16 bits are mapped in float32, wider ones in float64.
    """
    dtype = np.result_type(points.dtype, np.float32)
    m = affine[:3].T.astype(dtype)
    out: np.ndarray = points.astype(dtype, copy=False) @ m[:3]
    out += m[3]
    return out
