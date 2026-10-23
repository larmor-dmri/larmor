# Plain numpy reference of the FOD peak search: the previous Python
# implementation of find_peaks, kept line for line so that the compiled search can be
# compared with it. Slow; use it on a few voxels only.
from math import acos, atan2, cos, isnan, pi, sin, sqrt

import numpy as np
from numpy.typing import NDArray

MAX_ITERS = 50
MAX_STEP = 0.2
TOL = 1e-4
DOT_THRESHOLD = 0.99


def legendre_table(lmax: int, theta: float) -> NDArray[np.float64]:
    # P[l, m] for 0 <= m <= l, normalised, with the Condon-Shortley sign as in
    # get_sh_coeffs; 0 above the diagonal.
    x = cos(theta)
    s = sin(theta)
    p = np.zeros((lmax + 1, lmax + 1))

    double_factorial_ratio = 1.0
    for m in range(0, lmax + 1):
        if m > 0:
            double_factorial_ratio *= (2 * m - 1) / (2 * m)
        p[m, m] = (
            (-1) ** m * sqrt((2 * m + 1) / (4 * pi) * double_factorial_ratio) * s**m
        )
        if m < lmax:
            p[m + 1, m] = sqrt(2 * m + 3) * x * p[m, m]
        for ell in range(m + 2, lmax + 1):
            f_l = sqrt((4 * ell**2 - 1) / (ell**2 - m**2))
            f_l_prev = sqrt((4 * (ell - 1) ** 2 - 1) / ((ell - 1) ** 2 - m**2))
            p[ell, m] = f_l * (x * p[ell - 1, m] - p[ell - 2, m] / f_l_prev)

    return p


def _up(ell: int, m: int) -> float:
    return sqrt((ell - m) * (ell + m + 1))


def _down(ell: int, m: int) -> float:
    return sqrt((ell + m) * (ell - m + 1))


def _delta_plm(p: NDArray[np.float64], ell: int, m: int) -> float:
    if ell == 0:
        return 0.0
    if m == 0:
        return _up(ell, 0) * float(p[ell, 1])
    above = _up(ell, m) * float(p[ell, m + 1]) if m < ell else 0.0
    return (above - _down(ell, m) * float(p[ell, m - 1])) / 2


def _delta2_plm(p: NDArray[np.float64], ell: int, m: int) -> float:
    # For m == 1 the P[l, m - 2] term is P[l, -1] = -P[l, 1].
    if ell == 0:
        return 0.0
    if m == 0:
        return (
            _up(ell, 0) * _up(ell, 1) * float(p[ell, 2])
            - _up(ell, 0) ** 2 * float(p[ell, 0])
        ) / 2

    total = -(_up(ell, m) ** 2 + _down(ell, m) ** 2) * float(p[ell, m])
    if m == 1:
        total -= _down(ell, 1) * _down(ell, 0) * float(p[ell, 1])
    else:
        total += _down(ell, m) * _down(ell, m - 1) * float(p[ell, m - 2])
    if m + 2 <= ell:
        total += _up(ell, m) * _up(ell, m + 1) * float(p[ell, m + 2])
    return total / 4


def index(ell: int, m: int) -> int:
    # Position of (l, m) in get_sh_indices' even-degree order.
    return ell * (ell - 1) // 2 + ell + m


def derivatives(
    sh: NDArray[np.float64], lmax: int, theta: float, phi: float
) -> tuple[float, float, float, float, float, float]:
    # f, df/dtheta, df/dphi / sin, d2f/dtheta2, d2f/dtheta dphi / sin,
    # d2f/dphi2 / sin^2; within 1e-4 of a pole, MRtrix's limit forms.
    p = legendre_table(lmax, theta)
    f = f_t = f_p = f_tt = f_tp = f_pp = 0.0
    sin_theta = sin(theta)
    at_pole = sin_theta < 1e-4

    for ell in range(0, lmax + 1, 2):
        c = float(sh[index(ell, 0)])
        f += c * float(p[ell, 0])
        f_t += c * _delta_plm(p, ell, 0)
        f_tt += c * _delta2_plm(p, ell, 0)

        for m in range(1, ell + 1):
            cp, cm = float(sh[index(ell, m)]), float(sh[index(ell, -m)])
            A = sqrt(2) * (cp * cos(m * phi) + cm * sin(m * phi))
            dA = sqrt(2) * (cm * cos(m * phi) - cp * sin(m * phi))
            p_lm = float(p[ell, m])
            dp_lm = _delta_plm(p, ell, m)

            f += A * p_lm
            f_t += A * dp_lm
            f_tt += A * _delta2_plm(p, ell, m)

            if at_pole:
                f_p += dA * dp_lm
            else:
                f_p += m * dA * p_lm
                f_tp += m * dA * dp_lm
                f_pp -= m**2 * A * p_lm

    if not at_pole:
        f_p /= sin_theta
        f_tp /= sin_theta
        f_pp /= sin_theta**2

    return f, f_t, f_p, f_tt, f_tp, f_pp


def climb(
    sh: NDArray[np.float64],
    lmax: int,
    direction: NDArray[np.float64],
    max_iters: int = MAX_ITERS,
    tol: float = TOL,
    max_step: float = MAX_STEP,
) -> tuple[NDArray[np.float64], float]:
    # MRtrix's get_peak; NaN if it does not converge within max_iters.
    for _ in range(max_iters):
        x, y, z = direction[0], direction[1], direction[2]
        theta = acos(min(max(z, -1.0), 1.0))
        phi = atan2(y, x)
        f, f_t, f_p, f_tt, f_tp, f_pp = derivatives(sh, lmax, theta, phi)

        grad = sqrt(f_t**2 + f_p**2)
        if grad == 0:
            u_t, u_p = 0.0, 0.0
        else:
            u_t, u_p = f_t / grad, f_p / grad

        slope = u_t * f_t + u_p * f_p
        curvature = u_t**2 * f_tt + 2 * u_t * u_p * f_tp + u_p**2 * f_pp
        step = abs(slope / curvature) if curvature != 0 else 0.0
        step = min(step, max_step)

        e_t = np.asarray([cos(theta) * cos(phi), cos(theta) * sin(phi), -sin(theta)])
        e_p = np.asarray([-sin(phi), cos(phi), 0])

        direction = direction + step * (u_t * e_t + u_p * e_p)
        direction = direction / np.linalg.norm(direction)

        if step < tol:
            return direction, f

    return (np.asarray([np.nan, np.nan, np.nan]), np.nan)


def init_peaks(
    amplitudes: NDArray[np.float64], neighbours: NDArray[np.intp]
) -> NDArray[np.bool_]:
    # Vertices with a positive amplitude and no neighbour above it.
    n = len(amplitudes)
    is_peak = np.zeros(n, dtype=np.bool_)

    for v in range(0, n):
        if amplitudes[v] <= 0:
            continue

        highest = True
        for k in range(0, 6):
            if amplitudes[neighbours[v, k]] > amplitudes[v]:
                highest = False
                break
        if highest:
            is_peak[v] = True

    return is_peak


def pep(
    sh: NDArray[np.float64],
    lmax: int,
    amplitudes: NDArray[np.float64],
    vertices: NDArray[np.float64],
    neighbours: NDArray[np.intp],
    num_peaks: int,
    relative_threshold: float,
    threshold: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # The peaks of one voxel, as find_peaks returns them for it.
    if not np.isfinite(amplitudes[0]):
        return np.full((num_peaks, 3), np.nan), np.full(num_peaks, np.nan)

    starts = np.nonzero(init_peaks(amplitudes, neighbours))[0]
    found_dirs = np.zeros((len(starts), 3))
    found_amps = np.zeros((len(starts),))
    count = 0

    for s in starts:
        direction, amp = climb(sh, lmax, vertices[s])
        if isnan(amp) or amp <= 0:
            continue

        duplicate = -1
        for j in range(0, count):
            if abs(np.dot(direction, found_dirs[j])) > DOT_THRESHOLD:
                duplicate = j
                break

        if duplicate == -1:
            found_dirs[count] = direction
            found_amps[count] = amp
            count += 1
        elif amp > found_amps[duplicate]:
            found_dirs[duplicate] = direction
            found_amps[duplicate] = amp

    directions = np.full((num_peaks, 3), np.nan)
    peak_amplitudes = np.full(num_peaks, np.nan)
    if count == 0:
        return directions, peak_amplitudes

    order = np.argsort(-found_amps[:count])
    largest = found_amps[order[0]]
    written = 0

    for i in order:
        if written == num_peaks:
            break
        a = found_amps[i]
        if a < threshold or a < relative_threshold * largest:
            break
        d = found_dirs[i]
        if d[2] < 0:
            d = -d
        directions[written] = d
        peak_amplitudes[written] = a
        written += 1

    return directions, peak_amplitudes
