import math
import os
import re
from math import cos, pi, sin
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import reference_peaks as ref
from scipy.optimize import minimize_scalar
from scipy.special import lpmv

from larmor import _core
from larmor._core import climb, sh_derivatives
from larmor.io import load_volume
from larmor.sd import Peaks, convert_sh_basis, find_peaks
from larmor.sd.peaks import (
    MAX_LMAX,
    _build_icosphere,
    _get_amplitudes,
)
from larmor.sd.sh import build_basis_matrix, get_sh_indices

# Central differences with this step have truncation error ~h^2 * f''' ~ 1e-10
# relative and rounding error ~eps / h ~ 1e-11 relative, so 1e-6 of the
# derivative scale leaves a wide margin while still catching any wrong term.
H = 1e-5
RTOL_FD = 1e-6
LMAXES = [2, 8, 16]
# MRtrix's largest climb step, the default of _core.climb.
MAX_STEP = 0.2


def _random_sh(lmax: int, seed: int = 0) -> npt.NDArray[np.float64]:
    n = len(get_sh_indices(lmax)[0])
    sh: npt.NDArray[np.float64] = np.random.default_rng(seed).normal(size=n)
    return sh


def _random_angles(
    n: int, seed: int = 1, margin: float = 0.1
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    return rng.uniform(margin, pi - margin, n), rng.uniform(0, 2 * pi, n)


def _random_dirs(n: int = 50) -> list[tuple[float, float]]:
    theta, phi = _random_angles(n)
    return [(float(t), float(p)) for t, p in zip(theta, phi, strict=True)]


def _all(
    sh: npt.NDArray[np.float64], lmax: int, theta: float, phi: float
) -> npt.NDArray[np.float64]:
    return np.array(sh_derivatives(sh, lmax, theta, phi))


def _d_dtheta(
    sh: npt.NDArray[np.float64], lmax: int, theta: float, phi: float
) -> npt.NDArray[np.float64]:
    return (_all(sh, lmax, theta + H, phi) - _all(sh, lmax, theta - H, phi)) / (2 * H)


def _d_dphi(
    sh: npt.NDArray[np.float64], lmax: int, theta: float, phi: float
) -> npt.NDArray[np.float64]:
    return (_all(sh, lmax, theta, phi + H) - _all(sh, lmax, theta, phi - H)) / (2 * H)


def _assert_close_to_scale(
    actual: npt.NDArray[np.float64], expected: npt.NDArray[np.float64]
) -> None:
    scale = float(np.max(np.abs(expected)))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=RTOL_FD * scale)


# Legendre table and coefficient order of the reference


@pytest.mark.parametrize("lmax", [0, 1, 8, 16])
@pytest.mark.parametrize("theta", [0.0, 0.3, 1.2, pi / 2, 2.5, pi])
def test_legendre_table_matches_normalised_lpmv(lmax: int, theta: float) -> None:
    table = ref.legendre_table(lmax, theta)

    assert table.shape == (lmax + 1, lmax + 1)
    expected = np.zeros((lmax + 1, lmax + 1))
    for ell in range(lmax + 1):
        for m in range(ell + 1):
            norm = math.sqrt(
                (2 * ell + 1)
                / (4 * pi)
                * math.factorial(ell - m)
                / math.factorial(ell + m)
            )
            expected[ell, m] = norm * lpmv(m, ell, math.cos(theta))
    # Values reach ~1 at lmax 16; the recurrence loses a few ulps per degree.
    np.testing.assert_allclose(table, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("theta", [0.0, 1.2, pi])
def test_legendre_table_is_zero_above_the_diagonal(theta: float) -> None:
    table = ref.legendre_table(8, theta)
    assert np.all(np.triu(table, k=1) == 0)


def test_index_follows_get_sh_indices_order() -> None:
    ls, ms = get_sh_indices(16)
    positions = [ref.index(int(ell), int(m)) for ell, m in zip(ls, ms, strict=True)]
    assert positions == list(range(len(ls)))


# Amplitude


@pytest.mark.parametrize("lmax", [0, 2, 8, 16])
def test_amplitude_matches_tournier_basis(lmax: int) -> None:
    sh = _random_sh(lmax)
    # Full range of theta: the amplitude has no pole special case.
    theta, phi = _random_angles(300, margin=0.0)

    f = np.array(
        [sh_derivatives(sh, lmax, t, p)[0] for t, p in zip(theta, phi, strict=True)]
    )

    expected = build_basis_matrix(theta, phi, lmax, "tournier") @ sh
    np.testing.assert_allclose(f, expected, rtol=0, atol=1e-12 * np.abs(sh).sum())


# First derivatives


@pytest.mark.parametrize("lmax", LMAXES)
def test_theta_derivative_matches_finite_difference(lmax: int) -> None:
    sh = _random_sh(lmax)
    dirs = _random_dirs()

    f_t = np.array([_all(sh, lmax, t, p)[1] for t, p in dirs])
    fd = np.array([_d_dtheta(sh, lmax, t, p)[0] for t, p in dirs])

    _assert_close_to_scale(f_t, fd)


@pytest.mark.parametrize("lmax", LMAXES)
def test_phi_derivative_matches_finite_difference(lmax: int) -> None:
    sh = _random_sh(lmax)
    dirs = _random_dirs()

    f_p = np.array([_all(sh, lmax, t, p)[2] for t, p in dirs])
    fd = np.array([_d_dphi(sh, lmax, t, p)[0] / sin(t) for t, p in dirs])

    _assert_close_to_scale(f_p, fd)


# Second derivatives: first differences of the closed-form first derivatives,
# because second differences of f divide rounding error by h^2.


@pytest.mark.parametrize("lmax", LMAXES)
def test_theta_theta_derivative_matches_finite_difference(lmax: int) -> None:
    sh = _random_sh(lmax)
    dirs = _random_dirs()

    f_tt = np.array([_all(sh, lmax, t, p)[3] for t, p in dirs])
    fd = np.array([_d_dtheta(sh, lmax, t, p)[1] for t, p in dirs])

    _assert_close_to_scale(f_tt, fd)


@pytest.mark.parametrize("lmax", LMAXES)
def test_theta_phi_derivative_matches_finite_difference(lmax: int) -> None:
    sh = _random_sh(lmax)
    dirs = _random_dirs()

    f_tp = np.array([_all(sh, lmax, t, p)[4] for t, p in dirs])
    fd = np.array([_d_dphi(sh, lmax, t, p)[1] / sin(t) for t, p in dirs])

    _assert_close_to_scale(f_tp, fd)


@pytest.mark.parametrize("lmax", LMAXES)
def test_phi_phi_derivative_matches_finite_difference(lmax: int) -> None:
    sh = _random_sh(lmax)
    dirs = _random_dirs()

    # f_p is already divided by sin(theta), so one more division gives sin^2.
    f_pp = np.array([_all(sh, lmax, t, p)[5] for t, p in dirs])
    fd = np.array([_d_dphi(sh, lmax, t, p)[2] / sin(t) for t, p in dirs])

    _assert_close_to_scale(f_pp, fd)


# Poles

POLES = [0.0, 1e-6, pi - 1e-6, pi]


@pytest.mark.parametrize("lmax", LMAXES)
@pytest.mark.parametrize("theta", POLES)
def test_pole_outputs_are_finite(lmax: int, theta: float) -> None:
    assert np.all(np.isfinite(_all(_random_sh(lmax), lmax, theta, 0.7)))


@pytest.mark.parametrize("lmax", LMAXES)
@pytest.mark.parametrize("theta", POLES)
def test_pole_amplitude_matches_tournier_basis(lmax: int, theta: float) -> None:
    sh = _random_sh(lmax)
    f = sh_derivatives(sh, lmax, theta, 0.7)[0]
    expected = build_basis_matrix(np.array([theta]), np.array([0.7]), lmax) @ sh
    # lpmv takes sin(theta) as sqrt(1 - cos^2), which at theta = 1e-6 is off by
    # ~5e-5 relative; its m >= 1 terms are ~1e-6, so the reference is good to ~1e-10.
    assert f == pytest.approx(float(expected[0]), abs=1e-9 * np.abs(sh).sum())


@pytest.mark.parametrize("lmax", LMAXES)
@pytest.mark.parametrize("theta", POLES)
def test_pole_theta_derivative_matches_finite_difference(
    lmax: int, theta: float
) -> None:
    # The SH sum is analytic in theta, so a central difference across the pole
    # (theta < 0 or > pi) is still valid.
    sh = _random_sh(lmax)
    f_t = _all(sh, lmax, theta, 0.7)[1]
    fd = _d_dtheta(sh, lmax, theta, 0.7)[0]
    assert f_t == pytest.approx(fd, abs=RTOL_FD * max(abs(fd), 1.0))


@pytest.mark.parametrize("lmax", LMAXES)
def test_north_pole_phi_derivative_is_the_limit(lmax: int) -> None:
    sh = _random_sh(lmax)
    theta = 1e-6
    f_p = _all(sh, lmax, theta, 0.7)[2]
    fd = _d_dphi(sh, lmax, theta, 0.7)[0] / sin(theta)
    # The limit form drops O(theta) terms: theta = 1e-6 bounds the gap.
    assert f_p == pytest.approx(fd, abs=1e-5 * max(abs(fd), 1.0))


# Climb

LMAX_FOD = 8
# Directions of a few fibres: the north pole (where the derivatives switch to
# its pole form), near the pole, on the equator and in the southern hemisphere.
FIBRES = [
    (0.0, 0.0, 1.0),
    (0.03, 0.02, 1.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.3, 0.5, -1.0),
]
# A single apodised fibre at lmax 8 has its main lobe out to a trough near 38
# degrees and a faint ring at about 53 degrees, so starts stay within 30.
OFFSETS = [5.0, 15.0, 30.0]
AZIMUTHS = [0.0, 1.0, 2.5, 4.0, 5.5]
# The climb converges quadratically, so the final error is far below this.
TOL_DEG = 1e-3


def _unit(v: npt.ArrayLike) -> npt.NDArray[np.float64]:
    u = np.asarray(v, dtype=np.float64)
    out: npt.NDArray[np.float64] = u / np.linalg.norm(u)
    return out


def _polar(v: npt.NDArray[np.float64]) -> tuple[float, float]:
    return math.acos(max(-1.0, min(1.0, float(v[2])))), math.atan2(
        float(v[1]), float(v[0])
    )


def _amplitude(sh: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> float:
    theta, phi = _polar(v)
    b = build_basis_matrix(np.array([theta]), np.array([phi]), LMAX_FOD)
    return float((b @ sh)[0])


def _fibre_sh(direction: npt.ArrayLike) -> npt.NDArray[np.float64]:
    ls, _ = get_sh_indices(LMAX_FOD)
    apod = np.exp(-0.02 * ls * (ls + 1))
    theta, phi = _polar(_unit(direction))
    b = build_basis_matrix(np.array([theta]), np.array([phi]), LMAX_FOD)
    sh: npt.NDArray[np.float64] = b[0] * apod
    return sh


def _toward(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], angle: float
) -> npt.NDArray[np.float64]:
    # Rotates a by angle (radians) along the great circle toward b.
    e = b - (a @ b) * a
    return _unit(math.cos(angle) * a + math.sin(angle) * _unit(e))


def _start(
    fibre: npt.NDArray[np.float64], off_deg: float, azimuth: float
) -> npt.NDArray[np.float64]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(fibre[0]) < 0.9 else np.eye(3)[1]
    e1 = _unit(np.cross(fibre, helper))
    e2 = np.cross(fibre, e1)
    tangent = math.cos(azimuth) * e1 + math.sin(azimuth) * e2
    off = math.radians(off_deg)
    return _unit(math.cos(off) * fibre + math.sin(off) * tangent)


def _axis_deg(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    # d and -d are the same fibre axis.
    c = abs(float(a @ b)) / float(np.linalg.norm(a) * np.linalg.norm(b))
    return math.degrees(math.acos(min(1.0, c)))


def _crossing(
    separation_deg: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    a = np.array([1.0, 0.0, 0.0])
    s = math.radians(separation_deg)
    b = np.array([math.cos(s), math.sin(s), 0.0])
    return a, b, _fibre_sh(a) + _fibre_sh(b)


def _crossing_peak(
    sh: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    # The plane of the two fibres is a mirror plane of the FOD, so the peak near
    # a lies on the great circle from a to b; overlap pulls it off a.
    res = minimize_scalar(
        lambda t: -_amplitude(sh, _toward(a, b, t)),
        bounds=(-0.2, 0.2),
        method="bounded",
        options={"xatol": 1e-12},
    )
    return _toward(a, b, float(res.x))


@pytest.mark.parametrize("fibre", FIBRES)
@pytest.mark.parametrize("off_deg", OFFSETS)
@pytest.mark.parametrize("azimuth", AZIMUTHS)
def test_climb_reaches_single_fibre(
    fibre: tuple[float, float, float], off_deg: float, azimuth: float
) -> None:
    d = _unit(fibre)
    direction, _ = climb(_fibre_sh(d), LMAX_FOD, _start(d, off_deg, azimuth))

    assert _axis_deg(direction, d) < TOL_DEG
    assert float(np.linalg.norm(direction)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("fibre", FIBRES)
@pytest.mark.parametrize("off_deg", OFFSETS)
def test_climb_amplitude_is_the_fod_at_the_returned_direction(
    fibre: tuple[float, float, float], off_deg: float
) -> None:
    d = _unit(fibre)
    sh = _fibre_sh(d)
    direction, f = climb(sh, LMAX_FOD, _start(d, off_deg, 1.0))

    # f is taken before the last step (< 1e-4 rad) at a peak with curvature ~27,
    # so it can differ from the final amplitude by at most ~1e-7.
    assert f == pytest.approx(_amplitude(sh, direction), abs=1e-6)
    assert f == pytest.approx(_amplitude(sh, d), abs=1e-6)


@pytest.mark.parametrize("fibre", FIBRES)
def test_climb_started_on_the_peak_stays_there(
    fibre: tuple[float, float, float],
) -> None:
    d = _unit(fibre)
    sh = _fibre_sh(d)
    direction, f = climb(sh, LMAX_FOD, d)

    assert _axis_deg(direction, d) < TOL_DEG
    assert f == pytest.approx(_amplitude(sh, d), abs=1e-12)


@pytest.mark.parametrize("separation_deg", [60.0, 90.0])
@pytest.mark.parametrize("off_deg", [5.0, 15.0])
@pytest.mark.parametrize("azimuth", AZIMUTHS)
@pytest.mark.parametrize("near_b", [False, True])
def test_climb_in_a_crossing_ends_on_the_nearby_fibre(
    separation_deg: float, off_deg: float, azimuth: float, near_b: bool
) -> None:
    a, b, sh = _crossing(separation_deg)
    target, other = (b, a) if near_b else (a, b)
    peak = _crossing_peak(sh, target, other)

    direction, _ = climb(sh, LMAX_FOD, _start(target, off_deg, azimuth))

    assert _axis_deg(direction, peak) < TOL_DEG
    assert _axis_deg(direction, other) > separation_deg / 2


@pytest.mark.parametrize("separation_deg", [60.0, 90.0])
@pytest.mark.parametrize("near_b", [False, True])
def test_climb_off_the_bisector_does_not_stop_between_fibres(
    separation_deg: float, near_b: bool
) -> None:
    a, b, sh = _crossing(separation_deg)
    target, other = (b, a) if near_b else (a, b)
    peak = _crossing_peak(sh, target, other)
    start = _toward(target, other, math.radians(separation_deg / 2 - 10))

    direction, _ = climb(sh, LMAX_FOD, start)

    assert _axis_deg(direction, peak) < TOL_DEG


@pytest.mark.parametrize("off_deg", [0.01, 0.1, 1.0])
@pytest.mark.parametrize("near_b", [False, True])
def test_climb_near_a_saddle_ends_on_a_peak(off_deg: float, near_b: bool) -> None:
    # In the 60 degree crossing the lobes overlap, and the bisector is a saddle
    # (minimum along the fibre plane, maximum across it).
    a, b, sh = _crossing(60.0)
    target, other = (b, a) if near_b else (a, b)
    peak = _crossing_peak(sh, target, other)
    start = _toward(target, other, math.radians(30 - off_deg))

    direction, _ = climb(sh, LMAX_FOD, start)

    assert _axis_deg(direction, peak) < TOL_DEG


@pytest.mark.parametrize("separation_deg", [60.0, 90.0])
def test_climb_started_exactly_on_the_saddle_stays_there(
    separation_deg: float,
) -> None:
    # Behaviour shared with MRtrix's get_peak, not a goal. On the bisector the
    # gradient is zero up to rounding (~1e-16, from cos(pi/2) != 0), so the
    # Newton step |slope / curvature| is ~1e-16 and passes the tolerance at
    # once. Any start within ~1e-4 rad of the saddle stops near it the same way.
    a, b, sh = _crossing(separation_deg)
    saddle = _unit(a + b)
    theta, phi = _polar(saddle)
    _, f_t, f_p, *_ = sh_derivatives(sh, LMAX_FOD, theta, phi)
    assert math.hypot(f_t, f_p) < 1e-14

    direction, f = climb(sh, LMAX_FOD, saddle)

    np.testing.assert_allclose(direction, saddle, rtol=0, atol=1e-12)
    assert f == pytest.approx(_amplitude(sh, saddle), abs=1e-12)


@pytest.mark.parametrize("fibre", FIBRES)
def test_climb_beyond_the_main_lobe_stops_on_the_side_ring(
    fibre: tuple[float, float, float],
) -> None:
    # The climb is local: from 50 degrees it rises to the faint ring around the
    # fibre (~53 degrees, ~2.5% of the peak) instead of crossing the trough.
    d = _unit(fibre)
    sh = _fibre_sh(d)
    direction, f = climb(sh, LMAX_FOD, _start(d, 50.0, 1.0))

    assert 50.0 < _axis_deg(direction, d) < 56.0
    assert 0 < f < 0.05 * _amplitude(sh, d)


# Starts whose gradient lies along a meridian or along the equator: there the
# derivatives climb uses are those along a great circle. Elsewhere its curvature
# lacks the sphere's connection terms, as in MRtrix, so the step is not the
# great-circle Newton step. With the _start frame, azimuth 0 from +x heads to +z,
# pi / 2 along the equator and pi to -z.
ONE_STEP_STARTS = [
    ((0.0, 0.0, 1.0), 2.0),
    ((0.0, 0.0, -1.0), 4.0),
    ((1.0, 0.0, 0.0), 0.0),
    ((1.0, 0.0, 0.0), pi / 2),
    ((1.0, 0.0, 0.0), pi),
]


@pytest.mark.parametrize(("fibre", "azimuth"), ONE_STEP_STARTS)
@pytest.mark.parametrize(
    ("off_deg", "capped"), [(5.0, False), (10.0, True), (30.0, False)]
)
def test_climb_first_step_is_the_capped_newton_step(
    fibre: tuple[float, float, float],
    azimuth: float,
    off_deg: float,
    capped: bool,
) -> None:
    # A tolerance of 1 rad, above the cap, makes the climb return after one step.
    # The single-fibre FOD is symmetric about the fibre, so the gradient points
    # along the great circle to it. 5 degrees is inside the concave cap of the
    # lobe, 10 degrees has a Newton step above the 0.2 cap, and 30 degrees is on
    # the convex flank, where the sign of the step matters.
    d = _unit(fibre)
    sh = _fibre_sh(d)
    start = _start(d, off_deg, azimuth)
    off = math.radians(off_deg)

    def along(t: float) -> float:
        return _amplitude(sh, _toward(d, start, t))

    h = 1e-4
    slope = (along(off + h) - along(off - h)) / (2 * h)
    curvature = (along(off + h) - 2 * along(off) + along(off - h)) / h**2
    newton = abs(slope / curvature)
    assert (newton > MAX_STEP) == capped
    # The move is along a unit tangent then normalised, so it turns by atan(step).
    turn = math.atan(min(newton, MAX_STEP))

    direction, _ = climb(sh, LMAX_FOD, start, tol=1.0)

    # Second differences at h = 1e-4 are good to ~1e-7 relative here. A step
    # longer than the offset overshoots the fibre, hence the abs.
    assert math.radians(_axis_deg(direction, start)) == pytest.approx(turn, abs=1e-6)
    assert math.radians(_axis_deg(direction, d)) == pytest.approx(
        abs(off - turn), abs=1e-6
    )


@pytest.mark.parametrize("separation_deg", [60.0, 90.0])
@pytest.mark.parametrize(
    "start", [(0.9, 0.3, 0.4), (0.5, 0.8, -0.3), (0.2, 0.9, 0.5), (0.7, -0.2, -0.6)]
)
def test_climb_first_step_follows_get_peak(
    separation_deg: float,
    start: tuple[float, float, float],
) -> None:
    # get_peak's step from coordinate derivatives, here taken by central
    # differences of the basis so that sh_derivatives is not used. These starts
    # have a gradient with both components, so every Hessian term counts.
    _, _, sh = _crossing(separation_deg)
    s = _unit(start)
    theta, phi = _polar(s)

    def amp(t: float, p: float) -> float:
        return _amplitude(sh, np.array([sin(t) * cos(p), sin(t) * sin(p), cos(t)]))

    h = 1e-4
    sn = sin(theta)
    f0 = amp(theta, phi)
    f_t = (amp(theta + h, phi) - amp(theta - h, phi)) / (2 * h)
    f_p = (amp(theta, phi + h) - amp(theta, phi - h)) / (2 * h) / sn
    f_tt = (amp(theta + h, phi) - 2 * f0 + amp(theta - h, phi)) / h**2
    f_pp = (amp(theta, phi + h) - 2 * f0 + amp(theta, phi - h)) / h**2 / sn**2
    f_tp = (
        amp(theta + h, phi + h)
        - amp(theta + h, phi - h)
        - amp(theta - h, phi + h)
        + amp(theta - h, phi - h)
    ) / (4 * h**2 * sn)
    grad = math.hypot(f_t, f_p)
    u_t, u_p = f_t / grad, f_p / grad
    curvature = u_t**2 * f_tt + 2 * u_t * u_p * f_tp + u_p**2 * f_pp
    step = min(abs(grad / curvature), MAX_STEP)
    e_t = np.array([cos(theta) * cos(phi), cos(theta) * sin(phi), -sn])
    e_p = np.array([-sin(phi), cos(phi), 0.0])
    expected = _unit(s + step * (u_t * e_t + u_p * e_p))

    direction, f = climb(sh, LMAX_FOD, s, tol=1.0)

    # Second differences at h = 1e-4 carry ~1e-8 relative rounding error.
    np.testing.assert_allclose(direction, expected, rtol=0, atol=1e-6)
    assert f == pytest.approx(f0, abs=1e-12)


@pytest.mark.parametrize("off_deg", [15.0, 30.0])
def test_climb_returns_nan_when_iterations_run_out(off_deg: float) -> None:
    d = _unit(FIBRES[4])

    direction, f = climb(_fibre_sh(d), LMAX_FOD, _start(d, off_deg, 1.0), max_iters=1)

    assert direction.shape == (3,)
    assert np.all(np.isnan(direction))
    assert math.isnan(f)


def test_climb_that_never_converges_returns_nan() -> None:
    # No step is below a zero tolerance, so only the iteration limit ends it.
    d = _unit(FIBRES[4])

    direction, f = climb(_fibre_sh(d), LMAX_FOD, _start(d, 15.0, 1.0), tol=0.0)

    assert np.all(np.isnan(direction))
    assert math.isnan(f)


# Icosphere


def test_icosphere_has_642_unit_vertices() -> None:
    vertices, _ = _build_icosphere()

    assert vertices.shape == (642, 3)
    np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 1.0, atol=1e-12)


def test_icosphere_contains_every_antipode() -> None:
    vertices, _ = _build_icosphere()

    nearest = np.max(-vertices @ vertices.T, axis=1)
    np.testing.assert_allclose(nearest, 1.0, rtol=0, atol=1e-12)


def test_icosphere_original_vertices_have_five_neighbours_padded_with_self() -> None:
    _, neighbours = _build_icosphere()
    own = neighbours == np.arange(642)[:, None]
    distinct = np.array([len(set(row.tolist())) for row in neighbours])

    assert neighbours.shape == (642, 6)
    # The 12 icosahedron vertices come first: subdivision appends midpoints.
    np.testing.assert_array_equal(own.sum(axis=1), [1] * 12 + [0] * 630)
    np.testing.assert_array_equal(distinct, [6] * 642)


def test_icosphere_neighbours_are_adjacent_vertices() -> None:
    vertices, neighbours = _build_icosphere()
    own = neighbours == np.arange(642)[:, None]
    cosines = np.einsum("ij,ikj->ik", vertices, vertices[neighbours])[~own]
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))

    # Edges of the three-times subdivided icosahedron span 7.93 to 9.44 degrees.
    assert angles.min() > 7.9
    assert angles.max() < 9.5


def test_icosphere_neighbour_relation_is_symmetric() -> None:
    _, neighbours = _build_icosphere()

    for v, row in enumerate(neighbours):
        for n in row:
            assert v in neighbours[n]


def test_icosphere_is_built_once() -> None:
    first = _build_icosphere()
    second = _build_icosphere()

    assert first[0] is second[0]
    assert first[1] is second[1]


# Amplitudes on the icosphere


@pytest.mark.parametrize("lmax", [0, 2, 8])
def test_amplitudes_are_the_fod_at_the_vertices(lmax: int) -> None:
    coeffs = np.stack([_random_sh(lmax, seed) for seed in range(3)])
    vertices, _ = _build_icosphere()
    theta = np.arccos(np.clip(vertices[:, 2], -1.0, 1.0))
    phi = np.arctan2(vertices[:, 1], vertices[:, 0])

    amplitudes = _get_amplitudes(coeffs, lmax, full=False)

    assert amplitudes.shape == (3, 642)
    expected = coeffs @ build_basis_matrix(theta, phi, lmax, "tournier").T
    np.testing.assert_allclose(amplitudes, expected, rtol=0, atol=1e-12)


# Test FODs: sums of apodised lmax 8 fibres.

X, Y, Z = np.eye(3)
DEG60 = np.array([math.cos(math.radians(60)), math.sin(math.radians(60)), 0.0])
OBLIQUE = _unit((0.3, 0.5, -1.0))
# On the equator: an earlier filter that kept only upper-half starts missed it.
EQUATOR = np.array([0.6, 0.8, 0.0])
TEST_FODS: dict[str, list[npt.NDArray[np.float64]]] = {
    "single": [X],
    "oblique": [OBLIQUE],
    "equator": [EQUATOR],
    "crossing_90": [X, Y],
    "crossing_60": [X, DEG60],
    "three_orthogonal": [X, Y, Z],
    "oblique_crossing": [OBLIQUE, _unit(np.cross(OBLIQUE, (1.0, 0.0, 0.0)))],
}


def _fod(
    fibres: list[npt.NDArray[np.float64]], weights: list[float] | None = None
) -> npt.NDArray[np.float64]:
    w = [1.0] * len(fibres) if weights is None else weights
    sh: npt.NDArray[np.float64] = sum(
        (wi * _fibre_sh(f) for wi, f in zip(w, fibres, strict=True)),
        np.zeros(len(get_sh_indices(LMAX_FOD)[0])),
    )
    return sh


def _expected_peaks(name: str) -> list[npt.NDArray[np.float64]]:
    fibres = TEST_FODS[name]
    if name != "crossing_60":
        return fibres
    # Overlap pulls each peak ~0.7 degrees toward the other fibre.
    sh = _fod(fibres)
    a, b = fibres
    return [_crossing_peak(sh, a, b), _crossing_peak(sh, b, a)]


def _found(p: Peaks) -> npt.NDArray[np.float64]:
    out: npt.NDArray[np.float64] = p.directions[~np.isnan(p.amplitudes)]
    return out


def _rough_peaks(sh: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    _, neighbours = _build_icosphere()
    out: npt.NDArray[np.bool_] = ref.init_peaks(
        _get_amplitudes(sh[None], LMAX_FOD, full=False)[0], neighbours
    )
    return out


# Rough peaks on the icosphere, by the reference's rule; the compiled search's
# starts are tested through peaks_chunk in test_peaks_compiled.py.


@pytest.mark.parametrize("name", TEST_FODS)
def test_rough_peaks_lie_near_every_fibre_on_both_ends(name: str) -> None:
    vertices, _ = _build_icosphere()
    starts = vertices[_rough_peaks(_fod(TEST_FODS[name]))]

    # No direction is farther than ~5.4 degrees from its nearest vertex.
    for fibre in TEST_FODS[name]:
        for end in (fibre, -fibre):
            nearest = math.degrees(math.acos(min(1.0, float(np.max(starts @ end)))))
            assert nearest < 6.0


@pytest.mark.parametrize(
    "amplitudes",
    [
        np.zeros(642),
        -np.ones(642),
        # A fibre's lobe lowered below 0 (its peak is ~1.68): maxima, but negative.
        _get_amplitudes(_fibre_sh(X)[None], LMAX_FOD, full=False)[0] - 2.0,
    ],
    ids=["zero", "negative_constant", "negative_lobe"],
)
def test_rough_peaks_ignore_non_positive_amplitudes(
    amplitudes: npt.NDArray[np.float64],
) -> None:
    _, neighbours = _build_icosphere()

    assert not np.any(ref.init_peaks(amplitudes, neighbours))


def test_rough_peaks_keep_both_neighbours_of_a_tie() -> None:
    _, neighbours = _build_icosphere()
    a, b = 100, int(neighbours[100, 0])
    amplitudes = np.full(642, 0.1)
    amplitudes[[a, b]] = 1.0

    is_peak = ref.init_peaks(amplitudes, neighbours)

    assert is_peak[a]
    assert is_peak[b]
    assert not np.any(is_peak[neighbours[a]] & ~np.isin(neighbours[a], [a, b]))


@pytest.mark.parametrize("vertex", [0, 7, 11, 12, 300, 641])
def test_rough_peaks_find_a_single_maximum_at_any_vertex(vertex: int) -> None:
    # Vertices 0 to 11 have five neighbours and are padded with themselves.
    vertices, neighbours = _build_icosphere()

    is_peak = ref.init_peaks(vertices @ vertices[vertex], neighbours)

    np.testing.assert_array_equal(np.nonzero(is_peak)[0], [vertex])


# Peaks of one voxel


@pytest.mark.parametrize("name", TEST_FODS)
def test_find_peaks_returns_every_fibre_once(name: str) -> None:
    expected = _expected_peaks(name)

    found = _found(find_peaks(_fod(TEST_FODS[name])))

    assert len(found) == len(expected)
    for e in expected:
        assert min(_axis_deg(d, e) for d in found) < TOL_DEG


@pytest.mark.parametrize("fibre", FIBRES)
def test_find_peaks_single_fibre_anywhere(fibre: tuple[float, float, float]) -> None:
    d = _unit(fibre)

    found = _found(find_peaks(_fibre_sh(d)))

    assert len(found) == 1
    assert _axis_deg(found[0], d) < TOL_DEG


@pytest.mark.parametrize("name", TEST_FODS)
def test_peak_directions_are_unit_with_nonnegative_z(name: str) -> None:
    found = _found(find_peaks(_fod(TEST_FODS[name])))

    np.testing.assert_allclose(np.linalg.norm(found, axis=1), 1.0, atol=1e-12)
    assert np.all(found[:, 2] >= 0)


@pytest.mark.parametrize("fibre", FIBRES)
def test_peak_direction_points_to_upper_half(fibre: tuple[float, float, float]) -> None:
    d = _unit(fibre)
    up = d if d[2] >= 0 else -d

    found = _found(find_peaks(_fibre_sh(d)))

    # Equator fibres have z ~ 0 and either sign is a valid answer.
    if abs(up[2]) > 1e-6:
        assert float(found[0] @ up) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("name", TEST_FODS)
def test_peak_amplitudes_are_the_fod_at_their_direction(name: str) -> None:
    sh = _fod(TEST_FODS[name])
    p = find_peaks(sh)

    # As for the climb: f is taken one step (< 1e-4 rad) before the end.
    for d, a in zip(_found(p), p.amplitudes[~np.isnan(p.amplitudes)], strict=True):
        assert a == pytest.approx(_amplitude(sh, d), abs=1e-6)


WEIGHTED = ([X, Y, Z], [1.0, 0.7, 0.4])


def test_peaks_are_sorted_by_amplitude() -> None:
    fibres, weights = WEIGHTED

    p = find_peaks(_fod([Z, Y, X], weights[::-1]))

    assert np.all(np.diff(p.amplitudes) < 0)
    for d, fibre in zip(p.directions, fibres, strict=True):
        assert _axis_deg(d, fibre) < TOL_DEG


@pytest.mark.parametrize("num_peaks", [1, 2, 3, 5])
def test_n_peaks_keeps_the_strongest(num_peaks: int) -> None:
    fibres, weights = WEIGHTED

    p = find_peaks(_fod(fibres, weights), num_peaks=num_peaks)

    assert p.directions.shape == (num_peaks, 3)
    assert p.amplitudes.shape == (num_peaks,)
    kept = min(num_peaks, 3)
    for d, fibre in zip(p.directions[:kept], fibres, strict=False):
        assert _axis_deg(d, fibre) < TOL_DEG
    assert np.all(np.isnan(p.amplitudes[kept:]))


# A main fibre and a weak one at 90 degrees: the weak peak is ~6.9% of the main
# one, and each fibre's side ring (~53 degrees) peaks at ~2.7%.
MAIN_AND_WEAK = ([X, Y], [1.0, 0.05])


@pytest.mark.parametrize(
    ("relative_threshold", "count"), [(0.1, 1), (0.04, 2), (0.01, 4)]
)
def test_relative_threshold_drops_peaks_below_a_fraction_of_the_largest(
    relative_threshold: float, count: int
) -> None:
    p = find_peaks(
        _fod(*MAIN_AND_WEAK), num_peaks=4, relative_threshold=relative_threshold
    )

    assert int(np.sum(~np.isnan(p.amplitudes))) == count
    assert _axis_deg(p.directions[0], X) < TOL_DEG
    if count > 1:
        assert _axis_deg(p.directions[1], Y) < TOL_DEG


@pytest.mark.parametrize(
    ("threshold", "count"), [(0.0, 4), (0.1, 2), (0.5, 1), (2.0, 0)]
)
def test_threshold_drops_peaks_below_an_absolute_amplitude(
    threshold: float, count: int
) -> None:
    p = find_peaks(
        _fod(*MAIN_AND_WEAK), num_peaks=4, relative_threshold=0.0, threshold=threshold
    )

    assert int(np.sum(~np.isnan(p.amplitudes))) == count
    assert np.all(p.amplitudes[:count] >= threshold)


@pytest.mark.parametrize(
    "sh", [_fibre_sh(X), np.zeros(45)], ids=["single_fibre", "zero"]
)
def test_unused_peak_slots_are_nan_in_both_arrays(sh: npt.NDArray[np.float64]) -> None:
    p = find_peaks(sh, num_peaks=4)
    used = 1 if sh.any() else 0

    assert np.all(np.isfinite(p.directions[:used]))
    assert np.all(np.isfinite(p.amplitudes[:used]))
    assert np.all(np.isnan(p.directions[used:]))
    assert np.all(np.isnan(p.amplitudes[used:]))


MERGE_FODS = {name: _fod(fibres) for name, fibres in TEST_FODS.items()} | {
    f"random_{seed}": _random_sh(LMAX_FOD, seed) for seed in range(4)
}


@pytest.mark.parametrize("name", MERGE_FODS)
def test_climbs_on_one_axis_merge_into_the_largest(name: str) -> None:
    # Climbs from several rough peaks reach the same axis (from v and -v, or
    # neighbouring starts), and their amplitudes differ in the last digits: the
    # climb returns f from before its last step. The merge keeps the largest.
    sh = MERGE_FODS[name]
    vertices, _ = _build_icosphere()
    climbs = [climb(sh, LMAX_FOD, v) for v in vertices[_rough_peaks(sh)]]

    p = find_peaks(sh, num_peaks=10, relative_threshold=0.0)

    for d, a in zip(_found(p), p.amplitudes[~np.isnan(p.amplitudes)], strict=True):
        same_axis = [f for c, f in climbs if abs(float(c @ d)) > ref.DOT_THRESHOLD]
        assert a == max(same_axis)


# Volumes, masks and chunks


def _volume(shape: tuple[int, ...], seed: int = 0) -> npt.NDArray[np.float64]:
    # A different crossing in each voxel, with unequal weights, so that a voxel
    # written to the wrong place shows.
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    voxels = [
        _fod([_unit(a), _unit(b)], [1.0, w])
        for a, b, w in zip(
            rng.normal(size=(n, 3)),
            rng.normal(size=(n, 3)),
            rng.uniform(0.3, 0.9, n),
            strict=True,
        )
    ]
    out: npt.NDArray[np.float64] = np.reshape(voxels, (*shape, -1))
    return out


def _assert_same_peaks(actual: Peaks, expected: Peaks) -> None:
    np.testing.assert_array_equal(actual.directions, expected.directions)
    np.testing.assert_array_equal(actual.amplitudes, expected.amplitudes)


@pytest.mark.parametrize("num_peaks", [1, 3, 5])
def test_single_voxel_gives_per_peak_arrays(num_peaks: int) -> None:
    p = find_peaks(_fibre_sh(X), num_peaks=num_peaks)

    assert p.directions.shape == (num_peaks, 3)
    assert p.amplitudes.shape == (num_peaks,)


@pytest.mark.parametrize("shape", [(5,), (2, 2, 2)])
def test_volume_voxels_equal_their_single_voxel_peaks(shape: tuple[int, ...]) -> None:
    coeffs = _volume(shape)

    p = find_peaks(coeffs, num_peaks=3)

    assert p.directions.shape == (*shape, 3, 3)
    assert p.amplitudes.shape == (*shape, 3)
    for idx in np.ndindex(shape):
        _assert_same_peaks(
            Peaks(p.directions[idx], p.amplitudes[idx]), find_peaks(coeffs[idx])
        )


def test_masked_out_voxels_are_nan_and_others_unchanged() -> None:
    coeffs = _volume((3, 2))
    mask = np.array([[True, False], [False, True], [True, True]])

    p = find_peaks(coeffs, mask=mask)

    full = find_peaks(coeffs)
    assert np.all(np.isnan(p.directions[~mask]))
    assert np.all(np.isnan(p.amplitudes[~mask]))
    np.testing.assert_array_equal(p.directions[mask], full.directions[mask])
    np.testing.assert_array_equal(p.amplitudes[mask], full.amplitudes[mask])


def test_all_false_mask_gives_all_nan() -> None:
    p = find_peaks(_volume((2, 3)), mask=np.zeros((2, 3), dtype=bool), num_peaks=2)

    assert p.directions.shape == (2, 3, 2, 3)
    assert p.amplitudes.shape == (2, 3, 2)
    assert np.all(np.isnan(p.directions))
    assert np.all(np.isnan(p.amplitudes))


@pytest.mark.parametrize("mask_shape", [(6,), (2, 3), (3, 2, 1), (3, 2, 45)])
def test_mask_of_wrong_shape_is_refused(mask_shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=re.escape(str(mask_shape))) as info:
        find_peaks(_volume((3, 2)), mask=np.ones(mask_shape, dtype=bool))

    assert str((3, 2)) in str(info.value)


@pytest.mark.parametrize("num_peaks", [0, -1])
def test_n_peaks_below_one_is_refused(num_peaks: int) -> None:
    with pytest.raises(
        ValueError, match=f"num_peaks must be at least 1, got {num_peaks}"
    ):
        find_peaks(_volume((3, 2)), num_peaks=num_peaks)


@pytest.mark.parametrize("chunk_size", [0, -5])
def test_chunk_size_below_one_is_refused(chunk_size: int) -> None:
    with pytest.raises(
        ValueError, match=f"chunk_size must be at least 1, got {chunk_size}"
    ):
        find_peaks(_volume((3, 2)), chunk_size=chunk_size)


def test_integer_mask_selects_like_a_boolean_one() -> None:
    coeffs = _volume((3, 2))
    mask = np.array([[1, 0], [0, 1], [1, 1]])

    _assert_same_peaks(
        find_peaks(coeffs, mask=mask), find_peaks(coeffs, mask=mask.astype(bool))
    )


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 100])
def test_chunk_size_does_not_change_the_result(chunk_size: int) -> None:
    coeffs = _volume((7,))

    _assert_same_peaks(
        find_peaks(coeffs, chunk_size=chunk_size), find_peaks(coeffs, chunk_size=7)
    )


# Bases


@pytest.mark.parametrize("basis", ["descoteaux", "leemans"])
def test_peaks_do_not_depend_on_the_basis(basis: str) -> None:
    coeffs = _volume((4,))
    converted = convert_sh_basis(coeffs, "tournier", basis)
    assert not np.allclose(converted, coeffs)

    _assert_same_peaks(find_peaks(converted, basis=basis), find_peaks(coeffs))


@pytest.mark.parametrize("num_coeffs", [2, 10, 44])
def test_invalid_coefficient_count_is_refused(num_coeffs: int) -> None:
    with pytest.raises(ValueError, match=f"{num_coeffs} SH coefficients fit no layout"):
        find_peaks(np.ones((2, num_coeffs)))


def test_unknown_basis_is_refused() -> None:
    with pytest.raises(ValueError, match="Invalid basis mrtrix"):
        find_peaks(_fibre_sh(X), basis="mrtrix")


# Asymmetric FODs: all degrees


def _numpy_f(sh: npt.NDArray[np.float64], lmax: int, theta: float, phi: float) -> float:
    b = build_basis_matrix(np.array([theta]), np.array([phi]), lmax, full=True)
    return float((b @ sh)[0])


@pytest.mark.parametrize("lmax", [1, 7, 8])
def test_full_amplitude_and_theta_derivative_match_the_numpy_basis(lmax: int) -> None:
    sh = np.random.default_rng(0).normal(size=(lmax + 1) ** 2)
    dirs = _random_dirs(20)

    core = np.array([sh_derivatives(sh, lmax, t, p, full=True)[:2] for t, p in dirs])
    f = np.array([_numpy_f(sh, lmax, t, p) for t, p in dirs])
    fd = np.array(
        [
            (_numpy_f(sh, lmax, t + H, p) - _numpy_f(sh, lmax, t - H, p)) / (2 * H)
            for t, p in dirs
        ]
    )

    np.testing.assert_allclose(core[:, 0], f, rtol=0, atol=1e-12 * np.abs(sh).sum())
    _assert_close_to_scale(core[:, 1], fd)


def _in_full_layout(sh: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    ls, ms = get_sh_indices(LMAX_FOD)
    out = np.zeros((LMAX_FOD + 1) ** 2)
    out[ls * ls + ls + ms] = sh
    return out


@pytest.mark.parametrize("name", TEST_FODS)
def test_symmetric_fod_in_full_layout_gives_each_peak_and_its_opposite(
    name: str,
) -> None:
    sh = _fod(TEST_FODS[name])
    sym = find_peaks(sh)
    asym = find_peaks(_in_full_layout(sh))

    sym_dirs = _found(sym)
    sym_amps = sym.amplitudes[~np.isnan(sym.amplitudes)]
    expected_dirs = np.concatenate([sym_dirs, -sym_dirs])
    expected_amps = np.concatenate([sym_amps, sym_amps])
    asym_dirs = _found(asym)
    asym_amps = asym.amplitudes[~np.isnan(asym.amplitudes)]
    assert len(asym_dirs) == len(expected_dirs)
    # Pair each expected peak with the found one nearest in signed cos.
    order = np.argmax(expected_dirs @ asym_dirs.T, axis=1)
    assert sorted(order) == list(range(len(asym_dirs)))
    np.testing.assert_allclose(asym_dirs[order], expected_dirs, rtol=0, atol=1e-6)
    np.testing.assert_allclose(asym_amps[order], expected_amps, rtol=1e-9, atol=0)


def _one_sided_lobe(direction: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # All degrees with a positive taper: the lobe along direction adds up, the
    # opposite side nearly cancels (0.14 against 3.2), under the 0.1 threshold.
    ls, _ = get_sh_indices(LMAX_FOD, full=True)
    theta, phi = _polar(_unit(direction))
    b = build_basis_matrix(np.array([theta]), np.array([phi]), LMAX_FOD, full=True)
    sh: npt.NDArray[np.float64] = b[0] * np.exp(-0.02 * ls * (ls + 1))
    return sh


@pytest.mark.parametrize("side", [1.0, -1.0], ids=["z_below_0", "z_above_0"])
def test_one_sided_lobe_gives_one_peak_with_its_sign(side: float) -> None:
    sh = _one_sided_lobe(side * OBLIQUE)

    found = _found(find_peaks(sh))

    assert len(found) == 1
    assert found[0] @ (side * OBLIQUE) > math.cos(math.radians(TOL_DEG))


@pytest.mark.parametrize(
    ("num_coeffs", "num_peaks", "slots"),
    [(45, None, 3), (81, None, 6), (64, None, 6), (45, 5, 5), (81, 2, 2)],
)
def test_n_peaks_defaults_to_three_symmetric_and_six_asymmetric(
    num_coeffs: int, num_peaks: int | None, slots: int
) -> None:
    coeffs = np.random.default_rng(0).normal(size=(2, num_coeffs))

    found = find_peaks(coeffs, num_peaks=num_peaks)

    assert found.directions.shape == (2, slots, 3)
    assert found.amplitudes.shape == (2, slots)


# A 4x4x2 block of scilpy's asymmetric filtering of the ISMRM FOD (tests/data/asym):
# 2 to 5 peaks per voxel, with one-sided lobes.
ASYM_FOD = Path(__file__).parents[1] / "data" / "asym" / "fod_block.nii.gz"


def test_peaks_of_a_real_asymmetric_fod_are_the_maxima_on_a_fine_sphere() -> None:
    coeffs = load_volume(ASYM_FOD).data.astype(np.float64).reshape(-1, 81)
    found = find_peaks(coeffs, n_jobs=1)
    v = np.random.default_rng(0).normal(size=(50_000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    on_sphere = coeffs @ _full_basis(v).T

    assert np.all(np.isfinite(found.amplitudes[:, 0]))
    for sh, amps, dirs, peaks in zip(
        coeffs, on_sphere, found.directions, found.amplitudes, strict=True
    ):
        dirs, peaks = dirs[~np.isnan(peaks)], peaks[~np.isnan(peaks)]
        # As for the climb: f is taken one step (< 1e-4 rad) before the end, so
        # it sits ~1e-8 from the FOD at the direction and the true maximum.
        np.testing.assert_allclose(_full_basis(dirs) @ sh, peaks, rtol=0, atol=1e-6)
        assert amps.max() <= peaks[0] * (1 + 1e-6)
        for d, a in zip(dirs, peaks, strict=True):
            near = v @ d > math.cos(math.radians(10))
            assert amps[near].max() <= a * (1 + 1e-6)


def _full_basis(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    b: npt.NDArray[np.float64] = build_basis_matrix(
        np.arccos(np.clip(v[:, 2], -1, 1)), np.arctan2(v[:, 1], v[:, 0]), 8, full=True
    )
    return b


# Compiled search against the numpy reference


def test_peaks_chunk_matches_the_reference_voxel_by_voxel() -> None:
    block = np.ascontiguousarray(
        np.concatenate([_volume((4,)), np.zeros((1, 45)), _fod(*MAIN_AND_WEAK)[None]])
    )
    vertices, neighbours = _build_icosphere()
    amps = np.ascontiguousarray(_get_amplitudes(block, LMAX_FOD, full=False))
    args = (vertices, neighbours, 3, 0.04, 0.01)

    dirs, found = _core.peaks_chunk(
        block, LMAX_FOD, amps, vertices, neighbours.astype(np.int64), 3, 0.04, 0.01, 2
    )

    # Measured: ~1e-15 apart; the climbs start on the same vertices and take
    # the same few Newton steps.
    for i in range(len(block)):
        exp_dirs, exp_amps = ref.pep(block[i], LMAX_FOD, amps[i], *args)
        np.testing.assert_allclose(dirs[i], exp_dirs, rtol=0, atol=1e-12)
        np.testing.assert_allclose(found[i], exp_amps, rtol=0, atol=1e-12)


@pytest.mark.parametrize("n_jobs", [0, -2])
def test_n_jobs_of_zero_or_below_minus_one_is_refused(n_jobs: int) -> None:
    with pytest.raises(ValueError, match=f"n_jobs must be -1 .* got {n_jobs}"):
        find_peaks(_volume((3, 2)), n_jobs=n_jobs)


@pytest.mark.parametrize(
    ("n_jobs", "threads"),
    [
        (1, 1),
        (3, min(3, os.cpu_count() or 1)),
        (-1, os.cpu_count() or 1),
        # More threads than cores are capped at the number of cores.
        (10_000, os.cpu_count() or 1),
    ],
)
def test_n_jobs_is_the_thread_count_of_the_search(
    monkeypatch: pytest.MonkeyPatch, n_jobs: int, threads: int
) -> None:
    seen = []
    chunk = _core.peaks_chunk

    def counting_chunk(
        coeffs: npt.NDArray[np.float64],
        lmax: int,
        on_sphere: npt.NDArray[np.float64],
        vertices: npt.NDArray[np.float64],
        neighbours: npt.NDArray[np.int64],
        num_peaks: int,
        relative_threshold: float,
        threshold: float,
        n_threads: int,
        full: bool = False,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        seen.append(n_threads)
        return chunk(
            coeffs,
            lmax,
            on_sphere,
            vertices,
            neighbours,
            num_peaks,
            relative_threshold,
            threshold,
            n_threads,
            full=full,
        )

    monkeypatch.setattr(_core, "peaks_chunk", counting_chunk)

    find_peaks(_volume((3, 2)), n_jobs=n_jobs)

    assert seen == [threads]


def _lmax_fibre(lmax: int) -> npt.NDArray[np.float64]:
    ls, _ = get_sh_indices(lmax)
    theta, phi = _polar(_unit((0.3, 0.5, 1.0)))
    b = build_basis_matrix(np.array([theta]), np.array([phi]), lmax)
    sh: npt.NDArray[np.float64] = b[0] * np.exp(-0.02 * ls * (ls + 1))
    return sh


def test_lmax_of_max_lmax_is_searched() -> None:
    found = _found(find_peaks(_lmax_fibre(MAX_LMAX)))

    assert len(found) == 1
    assert _axis_deg(found[0], _unit((0.3, 0.5, 1.0))) < TOL_DEG


@pytest.mark.parametrize("lmax", [MAX_LMAX + 2, MAX_LMAX + 10])
def test_lmax_above_max_lmax_is_refused(lmax: int) -> None:
    with pytest.raises(ValueError, match=f"lmax must be at most 20, got {lmax}"):
        find_peaks(np.ones((2, (lmax + 1) * (lmax + 2) // 2)))


@pytest.mark.parametrize("relative_threshold", [1.5, -0.1, float("nan")])
def test_relative_threshold_outside_zero_to_one_is_refused(
    relative_threshold: float,
) -> None:
    # Above 1 even the largest peak would fail it, giving all NaN without an error.
    with pytest.raises(ValueError, match="relative_threshold must be between 0 and 1"):
        find_peaks(_volume((3, 2)), relative_threshold=relative_threshold)


def test_relative_threshold_of_one_keeps_only_the_largest_peak() -> None:
    found = find_peaks(_volume((3, 2)), relative_threshold=1.0)

    assert np.all(np.isfinite(found.amplitudes[..., 0]))
    assert np.all(np.isnan(found.amplitudes[..., 1:]))


def test_nan_voxels_give_nan_peaks_and_leave_others_unchanged() -> None:
    coeffs = _volume((3, 2))
    with_nan = coeffs.copy()
    with_nan[0, 1] = np.nan

    found = find_peaks(with_nan)
    expected = find_peaks(coeffs)

    assert np.all(np.isnan(found.directions[0, 1]))
    assert np.all(np.isnan(found.amplitudes[0, 1]))
    others = np.ones((3, 2), dtype=bool)
    others[0, 1] = False
    _assert_same_peaks(
        Peaks(found.directions[others], found.amplitudes[others]),
        Peaks(expected.directions[others], expected.amplitudes[others]),
    )
