# Plain numpy reference of the tracking loop (cpp/src/tracking/tracker.cpp, grid.cpp)
# with the dummy steps of _core.TestAlgorithm, one streamline at a time and written
# for clarity. The arithmetic is done in the same order as in C++, so a point it
# gives rounds to the same float32.
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from reference_random import draw_vector, stream, uniform

from larmor import _core
from larmor.tractography import EndStatus

Vector = NDArray[np.float64]
Ends = tuple[EndStatus, EndStatus]

MASK_THRESHOLD = 0.5
# The sampling counters of track_batch for algorithms that do not sample.
NO_SAMPLING = {
    "sampled_steps": 0,
    "accepted_index_sum": 0,
    "truncations": 0,
    "max_truncation": 0.0,
    "calibration_directions": 0,
}


def to_voxel(inv_affine: Vector, pos: Vector) -> tuple[float, float, float]:
    a = inv_affine
    vx = a[0, 0] * pos[0] + a[0, 1] * pos[1] + a[0, 2] * pos[2] + a[0, 3]
    vy = a[1, 0] * pos[0] + a[1, 1] * pos[1] + a[1, 2] * pos[2] + a[1, 3]
    vz = a[2, 0] * pos[0] + a[2, 1] * pos[1] + a[2, 2] * pos[2] + a[2, 3]
    return float(vx), float(vy), float(vz)


def on_grid(shape: tuple[int, ...], inv_affine: Vector, pos: Vector) -> bool:
    """Whether pos is within half a voxel of the grid's outermost centres."""
    v = to_voxel(inv_affine, pos)
    return all(-0.5 <= v[i] <= shape[i] - 0.5 for i in range(3))


def trilinear(coeffs: NDArray[np.floating], inv_affine: Vector, pos: Vector) -> Vector:
    """The (X, Y, Z, C) volume at pos, interpolated, indices clamped to the edge."""
    shape = coeffs.shape
    v = to_voxel(inv_affine, pos)
    corners = []
    for i in range(3):
        lo = math.floor(v[i])
        w = v[i] - lo
        clamp = (min(max(lo, 0), shape[i] - 1), min(max(lo + 1, 0), shape[i] - 1))
        corners.append(((clamp[0], 1.0 - w), (clamp[1], w)))
    out = np.zeros(shape[3])
    for xi, wx in corners[0]:
        for yi, wy in corners[1]:
            for zi, wz in corners[2]:
                out += (wx * wy * wz) * coeffs[xi, yi, zi].astype(np.float64)
    return out


def inside_mask(mask: NDArray[np.floating], inv_affine: Vector, pos: Vector) -> bool:
    """Whether pos's nearest voxel of the (X, Y, Z) mask is on it and >= 0.5.

    Halves round away from zero, and a coordinate at -0.5 or size - 0.5 is off.
    """
    v = to_voxel(inv_affine, pos)
    if not all(-0.5 < v[i] < mask.shape[i] - 0.5 for i in range(3)):
        return False
    index = tuple(math.floor(v[i]) + (v[i] - math.floor(v[i]) >= 0.5) for i in range(3))
    return bool(mask[index] >= MASK_THRESHOLD)


# The steps of _core.TestAlgorithm: (status, direction, random numbers drawn)


def next_direction(
    step: str, stop: EndStatus | None, previous: Vector, key: int, k: int
) -> tuple[EndStatus, Vector, int]:
    if step == "straight":
        return EndStatus.NONE, previous, k
    if step == "quarter_turn":
        return EndStatus.NONE, np.array([-previous[1], previous[0], 0.0]), k
    if step == "wander":
        d = [float(previous[i]) + 0.5 * (uniform(key, k + i) - 0.5) for i in range(3)]
        n = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        return EndStatus.NONE, np.array([c / n for c in d]), k + 3
    assert step == "refuse" and stop is not None
    return stop, previous, k


@dataclass(frozen=True)
class Scene:
    fod: NDArray[np.float32]
    fod_affine: Vector
    mask: NDArray[np.float32] | None
    mask_affine: Vector
    step_size: float
    max_steps: int
    bidirectional: bool
    step: str
    stop: EndStatus | None = None
    search_refuses_past_x: float = math.inf
    run_seed: int = 3


@dataclass
class Streamline:
    """points: (m, 3) in row order; ends: the end at points[0], then at points[-1];
    direction: the one followed from the seed; draws: random numbers used."""

    points: Vector
    ends: Ends
    direction: Vector
    draws: int


def track_reference(
    scene: Scene, seed: Vector, direction: Vector | None, index: int
) -> Streamline | None:
    """Tracks one seed; None if it does not start. direction None searches one."""
    fod_inv = np.linalg.inv(scene.fod_affine)
    mask_inv = np.linalg.inv(scene.mask_affine)

    def leaves(p: Vector) -> EndStatus:
        if not on_grid(scene.fod.shape, fod_inv, p):
            return EndStatus.LEFT_VOLUME
        if scene.mask is not None and not inside_mask(scene.mask, mask_inv, p):
            return EndStatus.LEFT_MASK
        return EndStatus.NONE

    if leaves(seed) != EndStatus.NONE:
        return None
    key = stream(scene.run_seed, index)
    if direction is None:
        # One draw, from the search's own counter range.
        found = draw_vector(np.array([key, 2**63], dtype=np.uint64))
        if seed[0] > scene.search_refuses_past_x:
            return None
        direction = found
    k = 0

    def half(points: list[Vector], d: Vector) -> EndStatus:
        nonlocal k
        # Starts from points[-1]: the seed, also after the reversal.
        pos = points[-1]
        while True:
            pos = np.array([pos[i] + scene.step_size * d[i] for i in range(3)])
            # A point off the image or outside the mask ends the half, unwritten.
            end = leaves(pos)
            if end != EndStatus.NONE:
                return end
            points.append(pos)
            if len(points) == scene.max_steps + 1:
                return EndStatus.MAX_LENGTH
            status, d, k = next_direction(scene.step, scene.stop, d, key, k)
            if status != EndStatus.NONE:
                return status

    points = [np.asarray(seed, dtype=np.float64)]
    end_a = half(points, direction)
    if not scene.bidirectional:
        ends = (EndStatus.NONE, end_a)
    else:
        points.reverse()
        if len(points) == scene.max_steps + 1:
            end_b = EndStatus.MAX_LENGTH
        else:
            end_b = half(points, -direction)
        ends = (end_a, end_b)
    return Streamline(np.array(points), ends, direction, k)


# Running the compiled loop on a scene, in the same form

SPARE_ROWS = 2


def track_core(
    scene: Scene,
    seeds: Vector,
    directions: Vector | None,
    indices: NDArray[np.int64],
    n_threads: int = 1,
) -> list[Streamline | None]:
    """_core.track_batch on the scene; also checks that it writes nowhere else."""
    n = len(seeds)
    rows = n + SPARE_ROWS
    positions = np.full((rows, scene.max_steps + 1, 3), np.nan, dtype=np.float32)
    mask = (
        None
        if scene.mask is None
        else (scene.mask, np.ascontiguousarray(np.linalg.inv(scene.mask_affine)))
    )
    algorithm = _core.TestAlgorithm(
        scene.step,
        None if scene.stop is None else int(scene.stop),
        scene.search_refuses_past_x,
    )
    counts, status, followed, stats = _core.track_batch(
        scene.fod,
        np.ascontiguousarray(np.linalg.inv(scene.fod_affine)),
        mask,
        np.ascontiguousarray(seeds, dtype=np.float64),
        None if directions is None else np.ascontiguousarray(directions),
        np.ascontiguousarray(indices, dtype=np.int64),
        scene.run_seed,
        scene.step_size,
        scene.max_steps,
        scene.bidirectional,
        algorithm,
        positions,
        n_threads,
    )
    assert counts.shape == (n,) and status.shape == (n, 2) and followed.shape == (n, 3)
    assert stats == NO_SAMPLING
    # Also catches a write past the end of a row, which lands in the next one.
    assert np.isnan(positions[n:]).all()
    out: list[Streamline | None] = []
    for i in range(n):
        count = int(counts[i])
        assert np.isnan(positions[i, count:]).all()
        if count == 0:
            assert list(status[i]) == [0, 0]
            out.append(None)
            continue
        ends = (EndStatus(int(status[i, 0])), EndStatus(int(status[i, 1])))
        points = positions[i, :count].astype(np.float64)
        out.append(Streamline(points, ends, followed[i], -1))
    return out


def assert_same(core: list[Streamline | None], ref: list[Streamline | None]) -> None:
    """The compiled float32 points are the reference's float64 ones rounded, exactly."""
    assert [r is None for r in core] == [r is None for r in ref]
    for a, b in zip(core, ref, strict=True):
        if a is not None and b is not None:
            np.testing.assert_array_equal(a.points, b.points.astype(np.float32))
            assert a.ends == b.ends
            np.testing.assert_array_equal(a.direction, b.direction)
