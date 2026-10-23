"""The GPU's voxelization and ambient light in numpy, as the tests' reference.

Same steps as larmor.viewer.render.line_voxels and its shader, after Kraaijeveld
B, Jalba AC, Vilanova A, Chamberland M. Real-Time Rendering of Dynamic Line Sets
using Voxel Ray Tracing. arXiv:2510.09081 (2025).
"""

import numpy as np
from scipy.ndimage import map_coordinates

from larmor.viewer.render.geometry import Lines
from larmor.viewer.render.line_voxels import (
    CONES,
    MIN_RADIUS,
    REACH,
    cone_directions,
    cone_tan_half,
    grid_for,
    segment_starts,
)

# The segments voxelized at once, which bounds the temporaries.
_CHUNK = 100_000


def ambient_light(lines: Lines, radius: float, size: int = 128) -> np.ndarray:
    """The light at the points, before scaling; and the grids: see `light_grids`."""
    grid = grid_for(lines.points, size)
    voxels = grid.to_voxels(lines.points)
    base, _, light = light_grids(voxels, lines.offsets, radius / grid.voxel, size)
    return read(light, base > 0, voxels)


def light_grids(
    points: np.ndarray, offsets: np.ndarray, radius: float, size: int
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """The occupancy, its pyramid and the light per voxel, for points in voxels."""
    base = occupancy(points, offsets, radius, size)
    levels = pyramid(base)
    return base, levels, cone_traced(levels, base > 0)


def occupancy(
    points: np.ndarray, offsets: np.ndarray, radius: float, size: int
) -> np.ndarray:
    """The share of each voxel the capsules around the segments fill, at most 1.

    Points are in voxel coordinates (voxel i spans i to i + 1). Each segment is a
    capsule cut flat at its ends by planes across the streamline's tangent there,
    so neighbouring capsules do not overlap; its voxels are found by stepping
    along the segment's major axis and taking, at each step, the box the capsule
    spans across the two minor axes.

    Returns:
        (size, size, size) float32.
    """
    tangent = tangents(points, offsets)
    starts = segment_starts(offsets, len(points))
    total = np.zeros(size**3, np.float64)
    for at in range(0, len(starts), _CHUNK):
        i = starts[at : at + _CHUNK]
        v0, v1 = points[i], points[i + 1]
        segment, voxels = _capsule_voxels(v0, v1, max(radius, MIN_RADIUS))
        inside = np.all((voxels >= 0) & (voxels < size), axis=1)
        segment, voxels = segment[inside], voxels[inside]
        filled = _capsule_occupancy(
            voxels + 0.5,
            v0[segment],
            v1[segment],
            tangent[i][segment],
            tangent[i + 1][segment],
            radius,
        )
        index = np.ravel_multi_index(voxels.T, (size,) * 3)
        total += np.bincount(index, weights=filled, minlength=size**3)
    return np.minimum(total, 1).astype(np.float32).reshape((size,) * 3)


def tangents(points: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """The unit tangent at each point, by central differences within a streamline."""
    tangent = np.zeros_like(points)
    starts = segment_starts(offsets, len(points))
    step = points[starts + 1] - points[starts]
    np.add.at(tangent, starts, step)
    np.add.at(tangent, starts + 1, step)
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    out = np.zeros_like(tangent)
    np.divide(tangent, norm, out=out, where=norm > 0)
    return out


def _capsule_voxels(
    v0: np.ndarray, v1: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Every voxel each capsule touches, as (segment index, voxel) pairs."""
    d = v1 - v0
    n = len(d)
    rows = np.arange(n)
    major = np.argmax(np.abs(d), axis=1)
    minor1, minor2 = (major + 1) % 3, (major + 2) % 3
    # Stepped along the major axis upwards: the ends swapped where it points down.
    down = d[rows, major] < 0
    a = np.where(down[:, None], v1, v0)
    b = np.where(down[:, None], v0, v1)
    d = b - a
    along = np.maximum(d[rows, major], 1e-12)
    step = d / along[:, None]  # one voxel along the major axis
    point = along < 1e-9  # a sphere, stepped along the major axis
    step[point] = 0
    step[rows[point], major[point]] = 1
    length = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
    a = a - step * radius  # room for the round ends
    b = b + step * radius
    # The half-widths of the capsule's cut across the major axis, an ellipse: for
    # the unit axis u, r * sqrt(1 - u_j^2) / |u_major| along minor axis i (j the
    # other), at least r, so the round ends fit too.
    u = d / length[:, None]
    u_major = np.maximum(np.abs(u[rows, major]), 1e-12)
    r1 = radius * np.sqrt(np.clip(1 - u[rows, minor2] ** 2, 0, 1)) / u_major
    r2 = radius * np.sqrt(np.clip(1 - u[rows, minor1] ** 2, 0, 1)) / u_major
    r1 = np.where(along < 1e-9, radius, r1)  # a point: a sphere
    r2 = np.where(along < 1e-9, radius, r2)
    t_min, t_max = a[rows, major], b[rows, major]
    # The steps: from t_min to the next voxel boundary, then a voxel at a time.
    first = np.floor(t_min)
    count = (np.ceil(t_max) - first).astype(np.int64).clip(min=1)
    owner = np.repeat(rows, count)
    k = np.arange(len(owner)) - np.repeat(np.cumsum(count) - count, count)
    cell = first[owner] + k
    t0 = np.maximum(cell, t_min[owner])
    t1 = np.minimum(cell + 1, t_max[owner])
    p0 = a[owner] + step[owner] * (t0 - t_min[owner])[:, None]
    p1 = a[owner] + step[owner] * (t1 - t_min[owner])[:, None]
    o = np.arange(len(owner))
    m1, m2 = minor1[owner], minor2[owner]
    j_lo = np.floor(np.minimum(p0[o, m1], p1[o, m1]) - r1[owner]).astype(np.int64)
    j_hi = np.floor(np.maximum(p0[o, m1], p1[o, m1]) + r1[owner]).astype(np.int64)
    k_lo = np.floor(np.minimum(p0[o, m2], p1[o, m2]) - r2[owner]).astype(np.int64)
    k_hi = np.floor(np.maximum(p0[o, m2], p1[o, m2]) + r2[owner]).astype(np.int64)
    wide_j, wide_k = j_hi - j_lo + 1, k_hi - k_lo + 1
    boxes = wide_j * wide_k
    slot = np.repeat(np.arange(len(owner)), boxes)
    q = np.arange(len(slot)) - np.repeat(np.cumsum(boxes) - boxes, boxes)
    voxels = np.empty((len(slot), 3), np.int64)
    sel = np.arange(len(slot))
    voxels[sel, major[owner][slot]] = cell[slot].astype(np.int64)
    voxels[sel, m1[slot]] = j_lo[slot] + q // wide_k[slot]
    voxels[sel, m2[slot]] = k_lo[slot] + q % wide_k[slot]
    return owner[slot], voxels


def _capsule_occupancy(
    p: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    n0: np.ndarray,
    n1: np.ndarray,
    radius: float,
) -> np.ndarray:
    """The occupancy each clipped capsule gives the voxel centred at p."""
    # Thin capsules are widened and fade instead, so they do not alias.
    widened = max(radius, MIN_RADIUS)
    d = v1 - v0
    pv0, pv1 = p - v0, p - v1
    dd = np.maximum(np.sum(d * d, axis=1), 1e-12)
    h = np.clip(np.sum(pv0 * d, axis=1) / dd, 0, 1)
    capsule = np.linalg.norm(pv0 - d * h[:, None], axis=1) - widened
    behind = -np.sum(pv0 * n0, axis=1)
    beyond = np.sum(pv1 * n1, axis=1)
    distance = np.maximum(capsule, np.maximum(behind, beyond))
    filled: np.ndarray = np.clip(0.5 - distance, 0, 1) * (radius / widened) ** 2
    return filled


def pyramid(base: np.ndarray) -> list[np.ndarray]:
    """The grid and its averages over 2x2x2 blocks, down to one voxel."""
    levels = [base]
    while levels[-1].shape[0] > 1:
        g = levels[-1]
        s = g.shape[0] // 2
        levels.append(g.reshape(s, 2, s, 2, s, 2).mean(axis=(1, 3, 5)))
    return levels


def cone_traced(levels: list[np.ndarray], occupied: np.ndarray) -> np.ndarray:
    """The share of the cones from each occupied voxel that nothing stops."""
    centres = np.argwhere(occupied).astype(np.float64) + 0.5
    directions = cone_directions(CONES)
    # Cones as wide as the sphere's share of each: solid angle 4 pi / CONES.
    tan_half = cone_tan_half(CONES)
    light = np.zeros((len(centres), CONES))
    for c, direction in enumerate(directions):
        opacity = np.zeros(len(centres))
        distance = 1.0  # from the voxel's edge, past its own lines
        while distance < REACH:
            width = max(2 * distance * tan_half, 1.0)
            level = min(np.log2(width), len(levels) - 1)
            at = centres + direction * distance
            sample = _sample_level(levels, at, level)
            # Steps of half the width: the occupancy corrected for that length.
            sample = 1 - (1 - np.clip(sample, 0, 1)) ** 0.5
            opacity += (1 - opacity) * sample
            distance += width / 2
        light[:, c] = 1 - opacity
    out = np.zeros(occupied.shape, np.float32)
    out[occupied] = light.mean(axis=1)
    return out


def _sample_level(levels: list[np.ndarray], at: np.ndarray, level: float) -> np.ndarray:
    """The occupancy at base-voxel coordinates `at`, between two levels."""
    low = int(np.floor(level))
    high = min(low + 1, len(levels) - 1)
    weight = level - low
    values = []
    for n in (low, high):
        coords = (at / 2**n - 0.5).T
        values.append(map_coordinates(levels[n], coords, order=1, mode="grid-constant"))
    return (1 - weight) * values[0] + weight * values[1]


def read(light: np.ndarray, occupied: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """The light at points, averaged over the occupied voxels around each."""
    coords = (grid - 0.5).T
    lit = map_coordinates(light, coords, order=1, mode="grid-constant")
    share = map_coordinates(
        occupied.astype(np.float32), coords, order=1, mode="grid-constant"
    )
    out = np.ones_like(lit)
    np.divide(lit, share, out=out, where=share > 1e-6)
    return out
