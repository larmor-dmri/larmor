# Plain numpy reference of probabilistic tracking (cpp/src/tracking/prob.cpp and
# calibrate.cpp), MRtrix's iFOD1: each step draws candidates uniformly in the cone
# around the previous direction and accepts one with probability amplitude / bound,
# the bound being the largest amplitude on a small calibrated grid of directions
# around the previous one, times a safety ratio. The random numbers are those of
# reference_random.py, drawn in the same order as _core, and the directions are built
# with the same arithmetic, so a streamline's float32 points are _core's. The
# amplitude is evaluated independently, through larmor.sd.sh; it differs from
# _core's by ~1e-15, which flips an acceptance only on a near-tie.
import functools
import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from reference_random import SEARCH_COUNTER, stream, uniform
from reference_tracking import Streamline, inside_mask, on_grid, trilinear

from larmor import _core
from larmor.sd.sh import build_basis_matrix
from larmor.tractography import EndStatus

Vector = NDArray[np.float64]

ELEVATION_STEP = 0.001


def amplitudes(sh: Vector, lmax: int, dirs: Vector) -> Vector:
    """The FOD sh (tournier) along each unit row of dirs."""
    dirs = np.atleast_2d(dirs)
    theta = np.arctan2(np.hypot(dirs[:, 0], dirs[:, 1]), dirs[:, 2])
    phi = np.arctan2(dirs[:, 1], dirs[:, 0])
    out: Vector = build_basis_matrix(theta, phi, lmax, "tournier") @ sh
    return out


def amplitude(sh: Vector, lmax: int, d: Vector) -> float:
    return float(amplitudes(sh, lmax, d)[0])


def rotate_to(v: Vector, d: Vector) -> Vector:
    """d rotated so that +z goes to the unit vector v, as MRtrix's rotate_direction."""
    n = math.sqrt(v[0] * v[0] + v[1] * v[1])
    if n == 0.0:
        return -d if v[2] < 0.0 else d.copy()
    m = (v[0] / n, v[1] / n)
    mp = (v[2] * m[0], v[2] * m[1], -n)
    alpha = d[2]
    beta = d[0] * m[0] + d[1] * m[1]
    return np.array(
        [
            d[0] + alpha * v[0] + beta * (mp[0] - m[0]),
            d[1] + alpha * v[1] + beta * (mp[1] - m[1]),
            d[2] + alpha * (v[2] - 1.0) + beta * (mp[2] - 0.0),
        ]
    )


@dataclass
class Draws:
    """A streamline's random numbers from its key, from counter on."""

    key: int
    counter: int = 0

    def uniform(self) -> float:
        u = uniform(self.key, self.counter)
        self.counter += 1
        return u

    def unit_vector(self) -> Vector:
        z = 2.0 * self.uniform() - 1.0
        phi = 2.0 * math.pi * self.uniform()
        r = math.sqrt(1.0 - z * z)
        return np.array([r * math.cos(phi), r * math.sin(phi), z])

    def unit_vector_in_cone(self, max_angle: float, sin_bound: float) -> Vector:
        """Uniform on the cap of half-angle max_angle around +z: phi first, then
        theta by rejection with density proportional to sin theta."""
        phi = 2.0 * math.pi * self.uniform()
        while True:
            theta = max_angle * self.uniform()
            if sin_bound * self.uniform() <= math.sin(theta):
                break
        st, ct = math.sin(theta), math.cos(theta)
        return np.array([st * math.cos(phi), st * math.sin(phi), ct])


# The calibration (MRtrix's tracking/calibrator.h)


def direction_grid(max_angle: float, spacing: float) -> Vector:
    """A hexagonal grid of directions around +z, spacing radians apart, out to
    max_angle radians."""
    extent = math.ceil(max_angle / spacing)
    i, j = np.meshgrid(np.arange(-extent, extent + 1), np.arange(-extent, extent + 1))
    x = (i + 0.5 * j).ravel()
    y = (math.sqrt(3) / 2 * j).ravel()
    r = np.hypot(x, y)
    keep = r * r <= (max_angle / spacing) ** 2
    x, y, r = x[keep], y[keep], r[keep]
    angle = spacing * r
    scale = np.divide(np.sin(angle), r, out=np.zeros_like(r), where=r > 0)
    return np.stack([scale * x, scale * y, np.cos(angle)], axis=1)


def _grid_size(max_angle: float, spacing: float) -> int:
    extent = math.ceil(max_angle / spacing)
    i, j = np.meshgrid(np.arange(-extent, extent + 1), np.arange(-extent, extent + 1))
    x, y = i + 0.5 * j, math.sqrt(3) / 2 * j
    return int(np.count_nonzero(x * x + y * y <= (max_angle / spacing) ** 2))


@functools.cache
def calibrate(lmax: int, max_angle: float) -> tuple[Vector, float]:
    """The grid and the ratio the bound uses, for a cone of max_angle radians.

    Picks the grid spacing that minimises the expected number of FOD evaluations
    per step, on a single fibre (a delta function truncated at lmax): the grid's
    size plus the expected number of candidates, which grows as the ratio does.
    """
    fibre = np.zeros((lmax + 1) * (lmax + 2) // 2)
    for ell in range(0, lmax + 1, 2):
        fibre[ell * (ell + 1) // 2] = math.sqrt((2 * ell + 1) / (4 * math.pi))
    # The elevation accumulates its step, as in MRtrix and _core.
    els = []
    el = 0.0
    while el < max_angle:
        els.append(el)
        el += ELEVATION_STEP
    els_a = np.array(els)
    amps = amplitudes(
        fibre, lmax, np.stack([np.sin(els_a), 0 * els_a, np.cos(els_a)], 1)
    )
    bad = np.flatnonzero(~np.isfinite(amps) | (amps <= 0))
    if len(bad):
        els_a, amps = els_a[: bad[0] + 1], amps[: bad[0] + 1]
    zero = els_a[-1]
    best, theta, ratio = math.inf, math.nan, math.nan
    for el_i, a_i in zip(els_a[1:], amps[1:], strict=True):
        n = max_angle**2 * (1 + amps[0] / a_i) / (2 * zero**2)
        n += _grid_size(max_angle + el_i, math.sqrt(3) * el_i)
        if 0 < n < best:
            best, theta, ratio = n, float(el_i), float(amps[0] / a_i)
    return direction_grid(max_angle + theta, math.sqrt(3) * theta), ratio


# The step


@dataclass
class Sampling:
    """The counters of _core.track_batch, for one streamline or summed."""

    sampled_steps: int = 0
    accepted_index_sum: int = 0
    truncations: int = 0
    max_truncation: float = 0.0

    def add(self, other: Sampling) -> None:
        self.sampled_steps += other.sampled_steps
        self.accepted_index_sum += other.accepted_index_sum
        self.truncations += other.truncations
        self.max_truncation = max(self.max_truncation, other.max_truncation)


@dataclass(frozen=True)
class Prob:
    cutoff: float
    max_angle_deg: float
    lmax: int
    trials: int = 1000
    attempts: int = 1000
    max_angle: float = field(init=False)
    sin_bound: float = field(init=False)

    def __post_init__(self) -> None:
        a = math.radians(self.max_angle_deg)
        object.__setattr__(self, "max_angle", a)
        object.__setattr__(self, "sin_bound", 1.0 if a >= math.pi / 2 else math.sin(a))

    def params(self) -> _core.ProbParams:
        return _core.ProbParams(
            self.cutoff, self.max_angle_deg, self.trials, self.attempts
        )

    def first_direction(self, sh: Vector, draws: Draws) -> Vector | None:
        for _ in range(self.attempts):
            d = draws.unit_vector()
            if amplitude(sh, self.lmax, d) > self.cutoff:
                return d
        return None

    def next_direction(
        self, sh: Vector, previous: Vector, draws: Draws, stats: Sampling
    ) -> tuple[EndStatus, Vector]:
        grid, ratio = calibrate(self.lmax, self.max_angle)
        on_grid_amps = amplitudes(
            sh, self.lmax, np.array([rotate_to(previous, g) for g in grid])
        )
        largest = max([0.0, *[a for a in on_grid_amps if not math.isnan(a)]])
        if largest <= 0.0:
            return EndStatus.LOW_FOD, previous
        bound = largest * ratio
        stats.sampled_steps += 1
        for n in range(self.trials):
            d = rotate_to(
                previous, draws.unit_vector_in_cone(self.max_angle, self.sin_bound)
            )
            a = amplitude(sh, self.lmax, d)
            if math.isnan(a) or a <= self.cutoff:
                continue
            if a > bound:
                stats.truncations += 1
                stats.max_truncation = max(stats.max_truncation, a / bound)
            if draws.uniform() < a / bound:
                stats.accepted_index_sum += n
                length = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
                return EndStatus.NONE, np.array([c / length for c in d])
        return EndStatus.NO_DIRECTION, previous


# The tracking loop with this step, as reference_tracking.track_reference


@dataclass(frozen=True)
class ProbScene:
    fod: NDArray[np.float32]
    fod_affine: Vector
    mask: NDArray[np.float32] | None
    mask_affine: Vector
    step_size: float
    max_steps: int
    bidirectional: bool
    prob: Prob
    run_seed: int = 3


def track_prob(
    scene: ProbScene, seed: Vector, direction: Vector | None, index: int
) -> tuple[Streamline | None, Sampling]:
    """Tracks one seed; None if it does not start. direction None searches one."""
    fod_inv = np.linalg.inv(scene.fod_affine)
    mask_inv = np.linalg.inv(scene.mask_affine)
    stats = Sampling()

    def leaves(p: Vector) -> EndStatus:
        if not on_grid(scene.fod.shape, fod_inv, p):
            return EndStatus.LEFT_VOLUME
        if scene.mask is not None and not inside_mask(scene.mask, mask_inv, p):
            return EndStatus.LEFT_MASK
        return EndStatus.NONE

    if leaves(seed) != EndStatus.NONE:
        return None, stats
    key = stream(scene.run_seed, index)
    if direction is None:
        sh = trilinear(scene.fod, fod_inv, seed)
        direction = scene.prob.first_direction(sh, Draws(key, SEARCH_COUNTER))
        if direction is None:
            return None, stats
    draws = Draws(key)

    def half(points: list[Vector], d: Vector) -> EndStatus:
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
            sh = trilinear(scene.fod, fod_inv, pos)
            status, d = scene.prob.next_direction(sh, d, draws, stats)
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
    return Streamline(np.array(points), ends, direction, draws.counter), stats


def track_prob_core(
    scene: ProbScene,
    seeds: Vector,
    directions: Vector | None,
    indices: NDArray[np.int64],
    n_threads: int = 1,
) -> tuple[list[Streamline | None], dict[str, float]]:
    """_core.track_batch with ProbParams on the scene, and its sampling counters."""
    n = len(seeds)
    positions = np.full((n, scene.max_steps + 1, 3), np.nan, dtype=np.float32)
    mask = (
        None
        if scene.mask is None
        else (scene.mask, np.ascontiguousarray(np.linalg.inv(scene.mask_affine)))
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
        scene.prob.params(),
        positions,
        n_threads,
    )
    out: list[Streamline | None] = []
    for i in range(n):
        count = int(counts[i])
        if count == 0:
            out.append(None)
            continue
        ends = (EndStatus(int(status[i, 0])), EndStatus(int(status[i, 1])))
        out.append(
            Streamline(positions[i, :count].astype(np.float64), ends, followed[i], -1)
        )
    return out, stats
