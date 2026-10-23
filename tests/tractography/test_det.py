# Deterministic tracking on synthetic FODs. A fibre is a sharp lobe
# exp(-k (1 - (u.a)^2)) projected on the SH basis with a Gauss-Legendre x uniform
# azimuth quadrature: the fitted lobe keeps its axial symmetry, so its peak is the
# axis a itself. Crossings are sums of lobes. SH directions are world directions,
# as the step compares them with world steps.
import functools
import math
import os
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from reference_random import SEARCH_COUNTER, draw_vector, search_vector, stream
from reference_tracking import NO_SAMPLING, trilinear

from larmor import _core
from larmor.sd.sh import build_basis_matrix
from larmor.tractography import (
    EndStatus,
    MaskSeeds,
    Seeds,
    Tracker,
    read_seeds,
    seeds_per_voxel,
    tracker,
)
from larmor.types import Tractogram, Volume

Vector = NDArray[np.float64]

LMAX = 8
SHARPNESS = 8.0
X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])
# The climb converges quadratically and stops below 1e-4 rad; float32 coefficients
# move the peak by ~1e-6 rad. 1e-3 degrees is ~1.7e-5 rad.
TOL_DEG = 1e-3
# Points are stored as float32: ~1e-6 mm at coordinates of ~10 mm.
POS_TOL = 1e-5


def _unit(v: Vector | list[float]) -> Vector:
    a = np.asarray(v, dtype=np.float64)
    return a / float(np.linalg.norm(a))


def _deg(a: Vector, b: Vector) -> float:
    return math.degrees(math.acos(min(1.0, float(a @ b))))


def _tilted(a: Vector, deg: float, azimuth: float = 0.0) -> Vector:
    # A unit vector deg degrees away from a, toward the azimuth around it.
    helper = Y if abs(a[1]) < 0.9 else Z
    e1 = _unit(np.cross(a, helper))
    e2 = np.cross(a, e1)
    side = math.cos(azimuth) * e1 + math.sin(azimuth) * e2
    t = math.radians(deg)
    return _unit(math.cos(t) * a + math.sin(t) * side)


@functools.cache
def _grid() -> tuple[Vector, Vector]:
    # Directions and quadrature weights, exact for polynomials of degree < 48.
    nodes, weights = np.polynomial.legendre.leggauss(24)
    phis = np.arange(48) * (2 * math.pi / 48)
    ct, ph = np.meshgrid(nodes, phis, indexing="ij")
    st = np.sqrt(1 - ct**2)
    dirs = np.stack(
        [(st * np.cos(ph)).ravel(), (st * np.sin(ph)).ravel(), ct.ravel()], axis=1
    )
    return dirs, np.repeat(weights, 48)


@functools.cache
def _fit_matrix(lmax: int) -> Vector:
    # Least squares weighted by the quadrature: a projection onto the basis, which
    # does not depend on how the lobe sits on the grid.
    dirs, weights = _grid()
    basis = build_basis_matrix(
        np.arccos(dirs[:, 2]), np.arctan2(dirs[:, 1], dirs[:, 0]), lmax, "tournier"
    )
    root = np.sqrt(weights)[:, np.newaxis]
    fit: Vector = np.linalg.pinv(root * basis) * root.T
    return fit


def _lobes(axes: Vector, lmax: int = LMAX) -> Vector:
    # One fitted lobe per row of axes, shape (n, C).
    values = np.exp(-SHARPNESS * (1 - (axes @ _grid()[0].T) ** 2))
    fitted: Vector = values @ _fit_matrix(lmax).T
    return fitted


def _lobe(axis: Vector, lmax: int = LMAX) -> Vector:
    lobe: Vector = _lobes(axis[np.newaxis], lmax)[0]
    return lobe


def _amplitude(sh: Vector, u: Vector, lmax: int = LMAX) -> float:
    basis = build_basis_matrix(
        np.array([math.acos(u[2])]), np.array([math.atan2(u[1], u[0])]), lmax
    )
    return float((basis @ sh)[0])


# One step, through the tracking loop: a single 4 mm voxel holds sh. The seed sits
# at -prev and the first step of 1 mm along prev lands exactly on the voxel centre,
# where the interpolated FOD is sh itself; the step taken from there is the one
# Det chose, read back from the next point: float32, so normalised again, which
# leaves ~1e-7 rad.
VOXEL_INV = np.diag([0.25, 0.25, 0.25, 1.0])


def _step(
    sh: Vector, prev: Vector, cutoff: float = 0.0, max_angle: float = 180.0
) -> tuple[EndStatus, Vector]:
    fod = np.ascontiguousarray(sh.reshape(1, 1, 1, -1), dtype=np.float32)
    positions = np.full((1, 3, 3), np.nan, dtype=np.float32)
    counts, status, _, stats = _core.track_batch(
        fod,
        VOXEL_INV,
        None,
        -prev[np.newaxis],
        prev[np.newaxis],
        np.zeros(1, dtype=np.int64),
        0,
        1.0,
        2,
        False,
        _core.DetParams(cutoff, max_angle),
        positions,
        1,
    )
    assert stats == NO_SAMPLING
    assert not positions[0, 1].any()
    if counts[0] == 2:
        return EndStatus(int(status[0, 1])), np.full(3, np.nan)
    assert counts[0] == 3 and status[0, 1] == EndStatus.MAX_LENGTH
    return EndStatus.NONE, _unit(positions[0, 2].astype(np.float64))


# The maximum angle


@pytest.mark.parametrize("turn", [1.0, 10.0, 30.0, 45.0])
@pytest.mark.parametrize("fibre", [X, Z, _unit([-0.3, 0.4, -0.85])])
def test_turn_just_past_the_max_angle_is_sharp_and_just_within_is_not(
    turn: float, fibre: Vector
) -> None:
    # The climb lands on the fibre to ~1e-5 degrees, far inside the 0.01 margin.
    prev = _tilted(fibre, turn, 1.0)
    assert _step(_lobe(fibre), prev, max_angle=turn - 0.01)[0] == EndStatus.SHARP_TURN
    status, direction = _step(_lobe(fibre), prev, max_angle=turn + 0.01)
    assert status == EndStatus.NONE and _deg(direction, fibre) < TOL_DEG


@pytest.mark.parametrize("turn", [30.0, 40.0])
def test_max_angle_of_180_degrees_never_ends_a_step_sharp(turn: float) -> None:
    # Further off, the climb ends on ringing of the lmax 8 lobe, not on its axis.
    prev = _tilted(X, turn, 2.0)
    assert _step(_lobe(X), prev, max_angle=turn - 1.0)[0] == EndStatus.SHARP_TURN
    status, direction = _step(_lobe(X), prev, max_angle=180.0)
    assert status == EndStatus.NONE and _deg(direction, X) < TOL_DEG


@pytest.mark.parametrize("cutoff", [0, 0.0, 3])
def test_integer_and_float_cutoffs_mean_the_same(cutoff: float) -> None:
    sh = 2.0 * _lobe(X) / _amplitude(_lobe(X), X)
    expected = EndStatus.NONE if cutoff <= 2 else EndStatus.LOW_FOD
    assert _step(sh, _tilted(X, 5.0), cutoff=cutoff)[0] == expected


# One step


FIBRES = [X, Y, Z, _unit([1.0, 2.0, 0.5]), _unit([-0.3, 0.4, -0.85])]


@pytest.mark.parametrize("fibre", FIBRES)
@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("off_deg", [5.0, 20.0, 35.0])
@pytest.mark.parametrize("azimuth", [0.0, 2.0, 4.5])
def test_step_returns_the_fibre_axis_on_the_side_of_the_previous_direction(
    fibre: Vector, sign: float, off_deg: float, azimuth: float
) -> None:
    status, direction = _step(_lobe(fibre), _tilted(sign * fibre, off_deg, azimuth))

    assert status == EndStatus.NONE
    assert _deg(direction, sign * fibre) < TOL_DEG


@pytest.mark.parametrize("fibre", FIBRES)
def test_step_below_the_cutoff_ends_low_fod(fibre: Vector) -> None:
    sh = _lobe(fibre)
    peak = _amplitude(sh, fibre)
    prev = _tilted(fibre, 10.0)

    assert _step(sh, prev, cutoff=1.01 * peak)[0] == EndStatus.LOW_FOD
    assert _step(sh, prev, cutoff=0.99 * peak)[0] == EndStatus.NONE


def test_step_with_the_peak_amplitude_as_cutoff_continues() -> None:
    # Started on an axially symmetric peak at the pole, the climb does not move,
    # so its amplitude is known to the last bit.
    sh = _lobe(Z)
    _, peak = _core.climb(sh.astype(np.float32).astype(np.float64), LMAX, Z)

    assert _step(sh, Z.copy(), cutoff=peak)[0] == EndStatus.NONE
    above = float(np.nextafter(peak, math.inf))
    assert _step(sh, Z.copy(), cutoff=above)[0] == EndStatus.LOW_FOD


@pytest.mark.parametrize("fibre", FIBRES)
@pytest.mark.parametrize("azimuth", [0.0, 2.0, 4.5])
def test_step_to_a_peak_beyond_the_max_angle_is_a_sharp_turn(
    fibre: Vector, azimuth: float
) -> None:
    # 40 degrees off a lobe this sharp is inside its basin: the climb converges.
    sh = _lobe(fibre)
    prev = _tilted(fibre, 40.0, azimuth)
    status, direction = _step(sh, prev)
    assert status == EndStatus.NONE and _deg(direction, fibre) < TOL_DEG

    assert _step(sh, prev, max_angle=30.0)[0] == EndStatus.SHARP_TURN
    assert _step(sh, prev, max_angle=45.0)[0] == EndStatus.NONE


def test_step_turning_by_exactly_the_max_angle_continues() -> None:
    # On the pole of an axially symmetric lobe the gradient is ~1e-17: the climb
    # does not move, and the turn's cosine is 1.0 to the bit. So is the cosine of
    # 1e-9 degrees, the smallest turn allowed here.
    assert math.cos(math.radians(1e-9)) == 1.0
    status, direction = _step(_lobe(Z), Z.copy(), max_angle=1e-9)

    assert status == EndStatus.NONE
    assert float(direction @ Z) == 1.0


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_step_on_a_non_finite_fod_finds_no_direction(bad: float) -> None:
    sh = _lobe(X)
    sh[3] = bad

    assert _step(sh, _tilted(X, 10.0))[0] == EndStatus.NO_DIRECTION


def _ridge() -> Vector:
    # A sharp ridge around the equator (a fan of fibres in the xy plane) with a
    # weak peak at x along it.
    values = np.exp(-4.0 * _grid()[0][:, 2] ** 2)
    ring: Vector = _fit_matrix(LMAX) @ values
    return ring + 0.01 * _lobe(X)


@pytest.mark.parametrize("azimuth_deg", [25.0, 30.0, 35.0])
@pytest.mark.parametrize("height_deg", [2.5, 3.0, 4.0])
def test_step_zigzagging_up_a_ridge_finds_no_direction(
    azimuth_deg: float, height_deg: float
) -> None:
    # Started just off the ridge, steepest ascent crosses it back and forth in
    # ever shorter steps and runs out of iterations before reaching x.
    a, h = math.radians(azimuth_deg), math.radians(height_deg)
    prev = np.array([math.cos(a) * math.cos(h), math.sin(a) * math.cos(h), math.sin(h)])

    assert _step(_ridge(), prev)[0] == EndStatus.NO_DIRECTION


@pytest.mark.parametrize("cutoff", [1e-6, 0.1])
def test_step_on_an_empty_fod_ends_low_fod(cutoff: float) -> None:
    # On a flat FOD the climb stops at once with amplitude 0: this is a low FOD,
    # not a failed climb.
    sh = np.zeros(45)

    assert _step(sh, _tilted(X, 10.0), cutoff=cutoff)[0] == EndStatus.LOW_FOD


@pytest.mark.parametrize("separation_deg", [60.0, 90.0])
@pytest.mark.parametrize("near_b", [False, True])
@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("azimuth", [0.0, 2.0, 4.5])
def test_step_in_a_crossing_returns_the_peak_nearest_the_previous_direction(
    separation_deg: float, near_b: bool, sign: float, azimuth: float
) -> None:
    s = math.radians(separation_deg)
    a, b = X, np.array([math.cos(s), math.sin(s), 0.0])
    sh = _lobe(a) + _lobe(b)
    target, other = (b, a) if near_b else (a, b)

    status, direction = _step(sh, _tilted(sign * target, 15.0, azimuth))

    assert status == EndStatus.NONE
    # Overlap pulls the peaks of a 60 degree crossing apart, by under a degree.
    assert _deg(direction, sign * target) < 1.0
    assert abs(float(direction @ other)) < math.cos(s / 2)


@pytest.mark.parametrize("lmax", [2, 4, 6, 8])
@pytest.mark.parametrize("fibre", [X, _unit([1.0, 2.0, 0.5])])
def test_step_reads_lmax_from_the_number_of_coefficients(
    lmax: int, fibre: Vector
) -> None:
    sh = _lobe(fibre, lmax)
    assert len(sh) == (lmax + 1) * (lmax + 2) // 2

    status, direction = _step(sh, _tilted(fibre, 20.0, 1.0))

    assert status == EndStatus.NONE
    assert _deg(direction, fibre) < TOL_DEG


# Whole streamlines


def _fod(coeffs: Vector, affine: Vector | None = None) -> Volume:
    return Volume(
        np.ascontiguousarray(coeffs, dtype=np.float32),
        np.eye(4) if affine is None else affine,
    )


def _uniform(shape: tuple[int, int, int], sh: Vector) -> Vector:
    return np.broadcast_to(sh, (*shape, len(sh))).copy()


def _run(
    fod: Volume,
    positions: Vector,
    directions: Vector,
    seed: int = 1,
    n_jobs: int = 1,
    **options: float,
) -> Tractogram:
    # A wide angle, so that the tilted seed directions of the tests are followed.
    settings: dict[str, float] = {
        "step_size": 0.1,
        "min_length": 0.05,
        "max_length": 100.0,
        "max_angle": 60.0,
    }
    settings.update(options)
    run = Tracker("det", progress=False, **settings).track(  # type: ignore[arg-type]
        fod, None, Seeds(positions, directions), seed=seed, n_jobs=n_jobs
    )
    assert run.tractogram is not None and run.num_streamlines == len(positions)
    return run.tractogram


def _streamline(t: Tractogram, i: int) -> NDArray[np.float32]:
    points: NDArray[np.float32] = t.points[t.offsets[i] : t.offsets[i + 1]]
    return points


def _ends(t: Tractogram, i: int) -> tuple[EndStatus, EndStatus]:
    first, last = t.streamline_data["end_status"][i]
    return EndStatus(int(first)), EndStatus(int(last))


def _off_line(points: NDArray[np.float32], origin: Vector, axis: Vector) -> float:
    rel = points.astype(np.float64) - origin
    return float(np.max(np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)))


SEED_TILT = 15.0


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_straight_bundle_is_followed_to_both_edges(sign: float) -> None:
    # The seed direction is tilted, so only the first step of each half leaves
    # the axis; after it the streamline runs parallel to the bundle.
    seed = np.array([5.3, 2.0, 2.0])
    direction = _tilted(sign * X, SEED_TILT, 1.0)
    t = _run(_fod(_uniform((12, 5, 5), _lobe(X))), seed[np.newaxis], direction[None])

    points = _streamline(t, 0)
    step = 0.1
    assert _off_line(points, seed, X) < step * math.sin(math.radians(SEED_TILT)) + 1e-5
    assert _ends(t, 0) == (EndStatus.LEFT_VOLUME, EndStatus.LEFT_VOLUME)
    # Each end is the last point on the grid, within a step of its edge.
    xs = sorted([float(points[0, 0]), float(points[-1, 0])])
    assert -0.5 <= xs[0] < -0.5 + step and 11.5 - step < xs[1] <= 11.5


def test_oblique_bundle_on_an_oblique_grid_stays_on_its_line() -> None:
    angle = math.radians(25)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([1.5, 1.5, 2.0])
    affine[:3, 3] = [-10.0, 4.0, 7.0]
    shape = (10, 10, 8)
    fibre = _unit([1.0, 2.0, 0.5])
    centre = affine[:3, :3] @ ((np.array(shape) - 1) / 2) + affine[:3, 3]
    t = _run(
        _fod(_uniform(shape, _lobe(fibre)), affine),
        centre[np.newaxis],
        _tilted(fibre, SEED_TILT, 2.0)[np.newaxis],
        step_size=0.15,
    )

    points = _streamline(t, 0)
    assert (
        _off_line(points, centre, fibre)
        < 0.15 * math.sin(math.radians(SEED_TILT)) + 1e-5
    )
    assert _ends(t, 0) == (EndStatus.LEFT_VOLUME, EndStatus.LEFT_VOLUME)


ARC_CENTRE = np.array([8.5, 8.5])
ARC_RADIUS = 5.0


def _arc_fod() -> Volume:
    # Fibres tangent to circles around the z axis through ARC_CENTRE.
    shape = (18, 18, 3)
    ijk = np.indices(shape).reshape(3, -1).T.astype(np.float64)
    rel = ijk[:, :2] - ARC_CENTRE
    tangent = np.zeros_like(ijk)
    tangent[:, 0], tangent[:, 1] = -rel[:, 1], rel[:, 0]
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = np.where(norms > 0, tangent / np.where(norms > 0, norms, 1.0), X)
    return _fod(_lobes(tangent).reshape(*shape, -1))


@pytest.mark.parametrize("step", [0.2, 0.1, 0.05])
def test_arc_bundle_is_followed_around_its_circle(step: float) -> None:
    # 0.9 of a turn; the first half takes every step, so the seed is the last
    # point. A step along the tangent moves outward: r^2 grows by step^2 each
    # time, so k steps from the seed sit at sqrt(R^2 + k step^2). Interpolating
    # the lobes between voxels adds under 0.002 at this radius.
    seed = np.array([ARC_CENTRE[0] + ARC_RADIUS, ARC_CENTRE[1], 1.0])
    length = 0.9 * 2 * math.pi * ARC_RADIUS
    t = _run(
        _arc_fod(),
        seed[np.newaxis],
        Y[np.newaxis],
        step_size=step,
        max_length=length,
    )

    points = _streamline(t, 0).astype(np.float64)[::-1]
    radius = np.linalg.norm(points[:, :2] - ARC_CENTRE, axis=1)
    euler = np.sqrt(ARC_RADIUS**2 + np.arange(len(points)) * step**2)
    np.testing.assert_allclose(radius, euler, atol=0.005)
    assert np.max(np.abs(points[:, 2] - 1.0)) < POS_TOL
    swept = np.unwrap(np.arctan2(*(points[:, 1::-1] - ARC_CENTRE[::-1]).T))
    assert np.ptp(swept) == pytest.approx(0.9 * 2 * math.pi, rel=0.05)
    assert _ends(t, 0) == (EndStatus.MAX_LENGTH, EndStatus.MAX_LENGTH)


def test_streamlines_are_the_same_whatever_the_run_seed_and_threads() -> None:
    rng = np.random.default_rng(3)
    angle = rng.uniform(0, 2 * math.pi, 40)
    radius = rng.uniform(3.0, 6.0, 40)
    positions = np.stack(
        [
            ARC_CENTRE[0] + radius * np.cos(angle),
            ARC_CENTRE[1] + radius * np.sin(angle),
            np.full(40, 1.0),
        ],
        axis=1,
    )
    directions = rng.normal(size=(40, 3)) * 0.3
    directions[:, 0] -= np.sin(angle)
    directions[:, 1] += np.cos(angle)
    fod = _arc_fod()
    threads = min(4, os.cpu_count() or 1)

    a = _run(fod, positions, directions, seed=1, n_jobs=1, max_length=20.0)
    b = _run(fod, positions, directions, seed=99, n_jobs=threads, max_length=20.0)

    np.testing.assert_array_equal(a.points, b.points)
    np.testing.assert_array_equal(a.offsets, b.offsets)
    np.testing.assert_array_equal(
        a.streamline_data["end_status"], b.streamline_data["end_status"]
    )


def _crossing_fod() -> Volume:
    # An x bundle in rows y = 4..6 and a y bundle in columns x = 4..6, which cross
    # in the middle; empty elsewhere.
    coeffs = np.zeros((11, 11, 3, 45))
    coeffs[:, 4:7] += _lobe(X)
    coeffs[4:7, :] += _lobe(Y)
    return _fod(coeffs)


@pytest.mark.parametrize("bundle", [X, Y])
def test_crossing_is_gone_straight_through(bundle: Vector) -> None:
    seed = np.array([5.0, 5.0, 1.0]) - 4.0 * bundle
    t = _run(
        _crossing_fod(),
        seed[np.newaxis],
        _tilted(bundle, SEED_TILT, 0.5)[np.newaxis],
        cutoff=0.1,
    )

    points = _streamline(t, 0)
    assert (
        _off_line(points, seed, bundle) < 0.1 * math.sin(math.radians(SEED_TILT)) + 1e-5
    )
    # Each end is the last point on the grid, within a step of its edge.
    along = points.astype(np.float64) @ bundle
    assert -0.5 <= along.min() < -0.4 and 10.4 < along.max() <= 10.5
    assert _ends(t, 0) == (EndStatus.LEFT_VOLUME, EndStatus.LEFT_VOLUME)


def test_streamline_ends_low_fod_where_the_bundle_fades() -> None:
    # Full strength up to voxel 5 and none from voxel 6: with half the peak as
    # cutoff the FOD falls below it past x = 5.5. Steps of 0.2 from 2.05 land at
    # 5.45, still above, then at 5.65, the last point.
    sh = _lobe(X)
    coeffs = _uniform((12, 3, 3), sh)
    coeffs[6:] = 0.0
    seed = np.array([[2.05, 1.0, 1.0]])
    t = _run(
        _fod(coeffs), seed, X[np.newaxis], step_size=0.2, cutoff=0.5 * _amplitude(sh, X)
    )

    points = _streamline(t, 0)
    ends = _ends(t, 0)
    far = 0 if points[0, 0] > points[-1, 0] else -1
    assert float(points[far, 0]) == pytest.approx(5.65, abs=POS_TOL)
    assert ends[far] == EndStatus.LOW_FOD
    assert ends[-1 - far] == EndStatus.LEFT_VOLUME


BEND_DEG = 40.0
BEND = np.array([math.cos(math.radians(BEND_DEG)), math.sin(math.radians(BEND_DEG)), 0])


def _bend_fod() -> Volume:
    # Along x up to voxel 4, then along BEND from voxel 5. Between them the peak
    # near x drifts by under 2 degrees, then vanishes at x = 4.7: the next peak is
    # the one along BEND, a turn of about 37 degrees in one step.
    coeffs = _uniform((12, 12, 3), _lobe(X))
    coeffs[5:] = _lobe(BEND)
    return _fod(coeffs)


def test_default_angle_stops_at_a_bend_tighter_than_its_radius() -> None:
    # About 37 degrees in one 0.1 voxel step is far tighter than a radius of 0.65
    # voxel, which allows about 8.8 degrees per step.
    seed = np.array([[2.0, 3.0, 1.0]])
    run = Tracker("det", step_size=0.1, min_length=0.05, progress=False).track(
        _bend_fod(), None, Seeds(seed, X[np.newaxis]), seed=1, n_jobs=1
    )

    assert run.tractogram is not None
    assert EndStatus.SHARP_TURN in _ends(run.tractogram, 0)


def test_bend_sharper_than_the_max_angle_ends_sharp_turn() -> None:
    seed = np.array([[2.0, 3.0, 1.0]])
    t = _run(_bend_fod(), seed, X[np.newaxis], max_angle=30.0)

    points = _streamline(t, 0)
    ends = _ends(t, 0)
    far = 0 if points[0, 0] > points[-1, 0] else -1
    assert ends[far] == EndStatus.SHARP_TURN
    assert 4.5 < float(points[far, 0]) < 5.0
    assert abs(float(points[far, 1]) - 3.0) < 0.05


def test_bend_within_the_max_angle_is_followed() -> None:
    seed = np.array([[2.0, 3.0, 1.0]])
    run = Tracker(
        "det", step_size=0.1, min_length=0.05, max_angle=60.0, progress=False
    ).track(_bend_fod(), None, Seeds(seed, X[np.newaxis]), seed=1, n_jobs=1)

    assert run.tractogram is not None
    points = _streamline(run.tractogram, 0).astype(np.float64)
    if points[0, 0] > points[-1, 0]:
        points = points[::-1]
    # The last step, far past the bend; float32 points make it good to ~1e-5 rad.
    assert _deg(_unit(points[-1] - points[-2]), BEND) < 0.01
    assert _ends(run.tractogram, 0) == (EndStatus.LEFT_VOLUME, EndStatus.LEFT_VOLUME)


# The first direction of a seed without one


# On a sharp lobe about a third of the climbs end in a peak, so many of 100 seeds
# need more than one.
INDICES = np.arange(100, dtype=np.int64)
# Climbs from new random draws before a seed is given up.
MAX_CLIMBS = 20


def _first_direction(
    coeffs: Vector, cutoff: float, run_seed: int, index: int
) -> tuple[Vector | None, int]:
    # The first climb of the seed's search draws that reaches the cutoff, and the
    # number of climbs that took; None where MAX_CLIMBS do not.
    state = np.array([stream(run_seed, index), SEARCH_COUNTER], dtype=np.uint64)
    for n in range(1, MAX_CLIMBS + 1):
        peak, amplitude = _core.climb(coeffs, LMAX, draw_vector(state))
        if not math.isnan(peak[0]) and amplitude >= cutoff:
            return peak, n
    return None, MAX_CLIMBS


def _searches(
    sh: Vector, cutoff: float, indices: NDArray[np.int64] = INDICES, run_seed: int = 0
) -> list[Vector | None]:
    # Seeds without directions at the centre of a single voxel holding sh: the
    # direction each one's search found, None where it found none.
    fod = np.ascontiguousarray(sh.reshape(1, 1, 1, -1), dtype=np.float32)
    n = len(indices)
    counts, status, followed, stats = _core.track_batch(
        fod,
        VOXEL_INV,
        None,
        np.zeros((n, 3)),
        None,
        indices,
        run_seed,
        1.0,
        1,
        False,
        _core.DetParams(cutoff, 180.0),
        np.zeros((n, 2, 3), dtype=np.float32),
        1,
    )
    assert stats == NO_SAMPLING
    out: list[Vector | None] = []
    for i in range(n):
        if counts[i] == 0:
            assert list(status[i]) == [0, 0] and not followed[i].any()
            out.append(None)
        else:
            out.append(followed[i])
    return out


def _as_stored(sh: Vector) -> Vector:
    # The FOD is float32 in the kernel.
    stored: Vector = sh.astype(np.float32).astype(np.float64)
    return stored


@pytest.mark.parametrize("fibre", FIBRES)
def test_first_step_on_one_fibre_finds_its_axis_either_way_or_nothing(
    fibre: Vector,
) -> None:
    sh = _lobe(fibre)
    cutoff = 0.5 * _amplitude(sh, fibre)
    signs = set()
    for direction in _searches(sh, cutoff):
        if direction is None:
            continue
        assert _deg(direction, np.sign(direction @ fibre) * fibre) < TOL_DEG
        assert _amplitude(_as_stored(sh), direction) >= cutoff
        signs.add(float(np.sign(direction @ fibre)))
    assert signs == {1.0, -1.0}


@pytest.mark.parametrize("fibre", FIBRES)
def test_first_step_is_the_first_climb_that_reaches_the_cutoff(fibre: Vector) -> None:
    sh = _lobe(fibre)
    cutoff = 0.5 * _amplitude(sh, fibre)
    climbs = []
    for index, direction in zip(INDICES, _searches(sh, cutoff), strict=True):
        peak, n = _first_direction(_as_stored(sh), cutoff, 0, int(index))

        assert (direction is None) == (peak is None)
        if direction is not None and peak is not None:
            # The numpy draws can differ from the kernel's in the last bit.
            np.testing.assert_allclose(direction, peak, rtol=0, atol=1e-15)
        climbs.append(n)
    # Seeds whose first climb ended below the cutoff start too.
    assert max(climbs) > 1


# Seeds of run seed 0 whose first climb to the cutoff on the X lobe is the 20th and
# the 21st, found with _first_direction.
@pytest.mark.parametrize(("index", "starts"), [(2116, True), (19833, False)])
def test_first_step_gives_up_after_max_climbs(index: int, starts: bool) -> None:
    sh = _lobe(X)
    cutoff = 0.5 * _amplitude(sh, X)
    state = np.array([stream(0, index), SEARCH_COUNTER], dtype=np.uint64)
    passes = []
    for _ in range(MAX_CLIMBS + 1):
        peak, amplitude = _core.climb(_as_stored(sh), LMAX, draw_vector(state))
        passes.append(not math.isnan(peak[0]) and amplitude >= cutoff)
    assert passes.index(True) == (MAX_CLIMBS - 1 if starts else MAX_CLIMBS)

    (direction,) = _searches(sh, cutoff, np.array([index], dtype=np.int64))

    assert (direction is not None) == starts


@pytest.mark.parametrize("separation_deg", [60.0, 90.0])
def test_first_step_in_a_crossing_finds_one_of_the_fibres_or_nothing(
    separation_deg: float,
) -> None:
    s = math.radians(separation_deg)
    a, b = X, np.array([math.cos(s), math.sin(s), 0.0])
    sh = _lobe(a) + _lobe(b)
    found = set()
    for direction in _searches(sh, 0.5 * _amplitude(sh, a)):
        if direction is None:
            continue
        # Overlap pulls the peaks of a 60 degree crossing apart, by under a degree.
        near = [i for i, f in enumerate((a, b)) if abs(float(direction @ f)) > 0.9998]
        assert len(near) == 1
        found.add(near[0])
    assert found == {0, 1}


def _near(direction: Vector, axis: Vector) -> bool:
    return abs(float(direction @ axis)) > math.cos(math.radians(TOL_DEG))


def test_first_step_skips_a_fibre_below_the_cutoff() -> None:
    sh = _lobe(X) + 0.3 * _lobe(Y)
    cutoff = 0.5 * _amplitude(sh, X)
    assert _amplitude(sh, Y) < cutoff
    without = _searches(sh, 0.0)
    assert any(d is not None and _near(d, Y) for d in without)

    found = [d for d in _searches(sh, cutoff) if d is not None]

    assert found and all(_near(d, X) for d in found)


@pytest.mark.parametrize(
    ("sh", "cutoff"),
    [
        (np.zeros(45), 1e-6),
        (np.zeros(45), 0.1),
        (_lobe(X), 1.01 * _amplitude(_lobe(X), X)),
    ],
    ids=["empty-small-cutoff", "empty", "lobe-under-cutoff"],
)
def test_first_step_finds_nothing_without_a_lobe_above_the_cutoff(
    sh: Vector, cutoff: float
) -> None:
    assert all(d is None for d in _searches(sh, cutoff))


def test_first_step_with_zero_cutoff_takes_the_first_draw_on_an_empty_fod() -> None:
    # A zero cutoff never stops for a low FOD: amplitude 0 is enough.
    for index, direction in zip(INDICES, _searches(np.zeros(45), 0.0), strict=True):
        assert direction is not None
        # The climb normalises its start, which moves the last bit.
        np.testing.assert_allclose(
            direction, search_vector(0, int(index)), rtol=0, atol=1e-15
        )


def _found(directions: list[Vector | None]) -> Vector:
    return np.array([np.full(3, np.nan) if d is None else d for d in directions])


def test_first_step_depends_only_on_the_run_seed_and_the_index() -> None:
    sh = _lobe(_unit([1.0, 2.0, 0.5])) + _lobe(Z)
    first = _found(_searches(sh, 0.1))
    reversed_order = _found(_searches(sh, 0.1, INDICES[::-1].copy())[::-1])
    other_seed = _found(_searches(sh, 0.1, run_seed=1))
    np.testing.assert_array_equal(first, reversed_order)
    assert not np.array_equal(first, other_seed, equal_nan=True)


# Tracking seeds without directions


def _fading_fod() -> tuple[Volume, float]:
    # Full strength up to voxel 5, none from voxel 6: with half the peak as cutoff
    # a seed starts up to x = 5.5 and finds nothing beyond.
    sh = _lobe(X)
    coeffs = _uniform((12, 3, 3), sh)
    coeffs[6:] = 0.0
    return _fod(coeffs), 0.5 * _amplitude(_as_stored(sh), X)


START_LIMIT_X = 5.5


class RecordingMaskSeeds(MaskSeeds):
    def __init__(self, mask: Volume) -> None:
        super().__init__(mask)
        self.drawn: list[Vector] = []

    def draw(self, n: int, rng: np.random.Generator) -> Seeds:
        seeds = super().draw(n, rng)
        self.drawn.append(seeds.positions)
        return seeds


def _seed_mask() -> Volume:
    mask = np.zeros((12, 3, 3), dtype=np.float32)
    mask[3:9, 1, 1] = 1.0
    return Volume(mask, np.eye(4))


def _searching(cutoff: float) -> Tracker:
    return Tracker(
        "det",
        step_size=0.5,
        min_length=0.1,
        max_length=4.0,
        cutoff=cutoff,
        progress=False,
    )


def _same(a: Tractogram, b: Tractogram) -> None:
    np.testing.assert_array_equal(a.points, b.points)
    np.testing.assert_array_equal(a.offsets, b.offsets)
    np.testing.assert_array_equal(
        a.streamline_data["end_status"], b.streamline_data["end_status"]
    )


def _assert_peak_directions(seeds: Seeds) -> None:
    assert seeds.directions is not None
    along = seeds.directions @ X
    assert np.all(np.abs(along) > math.cos(math.radians(TOL_DEG)))
    assert (along > 0).any() and (along < 0).any()


def test_mask_seeds_get_peak_directions_and_the_rest_do_not_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tracker, "MAX_BATCH", 16)
    fod, cutoff = _fading_fod()
    source = RecordingMaskSeeds(_seed_mask())
    seeds_file = tmp_path / "seeds.txt"
    run = _searching(cutoff).track(
        fod, None, source, num_streamlines=30, seed=1, output_seeds=seeds_file, n_jobs=1
    )

    assert run.num_streamlines == 30
    assert run.num_rejected["too_short"] == 0
    assert run.num_seeds == run.num_streamlines + run.num_rejected["no_start"]
    kept = read_seeds(seeds_file)
    assert len(kept) == 30 and kept.indices is not None
    tried = np.concatenate(source.drawn)[: run.num_seeds]
    np.testing.assert_array_equal(kept.positions, tried[kept.indices])
    started = np.zeros(run.num_seeds, dtype=bool)
    started[kept.indices] = True
    assert run.num_rejected["no_start"] == int((~started).sum())
    beyond = tried[:, 0] > START_LIMIT_X
    assert beyond.any()
    np.testing.assert_array_equal(started, ~beyond)
    _assert_peak_directions(kept)


def test_mask_seeds_tracked_again_from_their_file_give_the_same_streamlines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tracker, "MAX_BATCH", 16)
    fod, cutoff = _fading_fod()
    seeds_file = tmp_path / "seeds.txt"
    first = _searching(cutoff).track(
        fod,
        None,
        MaskSeeds(_seed_mask()),
        num_streamlines=30,
        seed=1,
        output_seeds=seeds_file,
    )
    kept = read_seeds(seeds_file)
    given = _searching(cutoff).track(fod, None, kept, seed=1)
    # Without the directions the search finds them again from the indices.
    searched = Seeds(kept.positions, None, kept.indices)
    again = _searching(cutoff).track(fod, None, searched, seed=1)

    assert first.tractogram is not None
    for run in (given, again):
        assert run.tractogram is not None and run.num_streamlines == 30
        _same(first.tractogram, run.tractogram)


def test_mask_seeds_without_directions_do_not_depend_on_the_threads() -> None:
    fod, cutoff = _fading_fod()
    threads = min(4, os.cpu_count() or 1)
    one = _searching(cutoff).track(
        fod, None, MaskSeeds(_seed_mask()), num_streamlines=40, seed=5, n_jobs=1
    )
    many = _searching(cutoff).track(
        fod, None, MaskSeeds(_seed_mask()), num_streamlines=40, seed=5, n_jobs=threads
    )

    assert one.tractogram is not None and many.tractogram is not None
    assert one.num_seeds == many.num_seeds
    _same(one.tractogram, many.tractogram)


def test_fixed_seeds_without_directions_start_where_a_search_finds_a_peak(
    tmp_path: Path,
) -> None:
    fod, cutoff = _fading_fod()
    xs = np.linspace(2.6, 8.4, 80)
    positions = np.column_stack([xs, np.full((80, 2), 1.0)])
    seeds_file = tmp_path / "seeds.txt"
    run = _searching(cutoff).track(
        fod, None, Seeds(positions), seed=2, output_seeds=seeds_file
    )

    # A seed starts iff the search keyed by its index finds a peak on the
    # interpolated FOD.
    inv = np.linalg.inv(fod.affine)
    data = np.asarray(fod.data, dtype=np.float32)
    starts = np.zeros(len(positions), dtype=bool)
    for i, p in enumerate(positions):
        peak, _ = _first_direction(trilinear(data, inv, p), cutoff, 2, i)
        starts[i] = peak is not None
    np.testing.assert_array_equal(starts, xs <= START_LIMIT_X)
    assert run.num_seeds == 80
    assert run.num_streamlines == int(starts.sum())
    assert run.num_rejected["no_start"] == int((~starts).sum())
    kept = read_seeds(seeds_file)
    assert kept.indices is not None
    np.testing.assert_array_equal(kept.indices, np.flatnonzero(starts))
    np.testing.assert_array_equal(kept.positions, positions[starts])
    _assert_peak_directions(kept)
    assert run.tractogram is not None
    for i, seed in enumerate(positions[starts]):
        assert _off_line(_streamline(run.tractogram, i), seed, X) < POS_TOL


def test_most_seeds_in_crossing_bundles_start() -> None:
    # Two bundles crossing at right angles; one climb ends under the auto cutoff
    # for about two seeds in three.
    shape = (8, 8, 3)
    across = np.zeros(shape, dtype=bool)
    across[:, 3:5] = True
    along = np.zeros(shape, dtype=bool)
    along[3:5, :] = True
    coeffs = np.zeros((*shape, _lobe(X).size))
    coeffs[across] = _lobe(X)
    coeffs[along] = _lobe(Y)
    coeffs[across & along] = _lobe(X) + _lobe(Y)
    bundles = Volume((across | along).astype(np.uint8), np.eye(4))
    seeds = seeds_per_voxel(bundles, 4, rng=0)

    run = Tracker("det", progress=False).track(
        _fod(coeffs), None, seeds, mask=bundles, seed=0, n_jobs=1
    )

    assert run.num_rejected["no_start"] <= 0.05 * run.num_seeds
