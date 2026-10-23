"""Numpy track density: the reference implementation of the compiled kernel."""

import numpy as np

from larmor.types import Grid, Tractogram

HALF_TENSION = np.float32(0.05)


def _spline_weights(ratio: int) -> list[np.ndarray]:
    """The four control-point weights at k / ratio, k = 1 to ratio - 1, in float32."""
    weights = []
    for k in range(1, ratio):
        mu = np.float32(k) / np.float32(ratio)
        mu2 = mu * mu
        mu3 = mu * mu2
        f = np.float32
        # The two fused multiply-adds of the compiled spline: a float32 product is
        # exact in float64, so each is rounded once, to float32, like a fused one.
        inner = f(f(f(1.5) + HALF_TENSION) * np.float64(mu3) + 1.0)
        w1 = f(-(f(2.5) + HALF_TENSION) * np.float64(mu2) + np.float64(inner))
        inner = f(
            f(f(2.0) + f(2.0) * HALF_TENSION) * np.float64(mu2)
            + np.float64((f(0.5) - HALF_TENSION) * mu)
        )
        w2 = f(-(f(1.5) + HALF_TENSION) * np.float64(mu3) + np.float64(inner))
        weights.append(
            np.array(
                [
                    (f(0.5) - HALF_TENSION) * (f(2.0) * mu2 - mu3 - mu),
                    w1,
                    w2,
                    (f(0.5) - HALF_TENSION) * (mu3 - mu2),
                ],
                dtype=np.float32,
            )
        )
    return weights


def upsampled(points: np.ndarray, ratio: int) -> np.ndarray:
    """The points with ratio - 1 spline points added between each consecutive pair."""
    if len(points) < 2 or ratio == 1:
        return points
    extended = np.concatenate(
        [2 * points[:1] - points[1:2], points, 2 * points[-1:] - points[-2:-1]]
    )
    out = []
    weights = _spline_weights(ratio)
    for segment in range(len(points) - 1):
        controls = extended[segment : segment + 4]
        out.append(points[segment])
        for w in weights:
            point = w[0] * controls[0] + w[1] * controls[1]
            point = point + w[2] * controls[2]
            out.append(point + w[3] * controls[3])
    out.append(points[-1])
    return np.array(out, dtype=np.float32)


def track_density(
    t: Tractogram, grid: Grid, ratio: int, ends_only: bool, weights: np.ndarray | None
) -> np.ndarray:
    """(X, Y, Z) float64: each streamline's weight added once per voxel it reaches."""
    to_voxel = np.linalg.inv(grid.affine)
    density = np.zeros(grid.shape)
    for s in range(t.num_streamlines):
        points = t[s]
        if len(points) == 0:
            continue
        if ends_only:
            points = points[[0, -1]] if len(points) > 1 else points
        else:
            points = upsampled(points, ratio)
        coordinates = points.astype(np.float64) @ to_voxel[:3, :3].T + to_voxel[:3, 3]
        inside = np.all(
            (coordinates > -0.5) & (coordinates < np.array(grid.shape) - 0.5), axis=1
        )
        voxels = np.floor(coordinates[inside] + 0.5).astype(np.int64)
        if not ends_only:
            voxels = np.unique(voxels, axis=0)
        weight = 1.0 if weights is None else weights[s]
        np.add.at(density, tuple(voxels.T), weight)
    return density
