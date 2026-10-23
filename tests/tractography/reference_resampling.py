# Plain numpy reference of streamline resampling (cpp/src/streamlines/resample.cpp),
# one streamline and one point at a time and written for clarity. The arithmetic is
# float32 throughout, in the same order as in C++ and with the same fused
# multiply-adds, so it gives the same points.
import numpy as np
from numpy.typing import NDArray

from larmor.types import Tractogram

f32 = np.float32
Points = NDArray[np.float32]
# The position of each output point on its input streamline: the index of the first
# point of its segment and the fraction (0 to 1) of the way to the next point.
Positions = tuple[list[int], list[float]]

TENSION = f32(0.1)
HALF_TENSION = f32(0.5) * TENSION
BISECTION_TOLERANCE = f32(0.001)


def fma(a: np.float32, b: np.float32, c: np.float32) -> np.float32:
    """a * b + c rounded once to float32, through float64 (rarely rounded twice)."""
    return f32(np.float64(a) * np.float64(b) + np.float64(c))


def distance(a: Points, b: Points) -> np.float32:
    """The Euclidean distance between two points, in float32."""
    d = a - b
    return f32(np.sqrt(f32(f32(d[0] * d[0] + d[1] * d[1]) + d[2] * d[2])))


def hermite(mu: np.float32, p0: Points, p1: Points, p2: Points, p3: Points) -> Points:
    """The cubic Hermite spline (tension 0.1) between p1 and p2 at mu in [0, 1]."""
    t = HALF_TENSION
    mu2 = mu * mu
    mu3 = mu * mu2
    w0 = (f32(0.5) - t) * (f32(2.0) * mu2 - mu3 - mu)
    w1 = fma(-(f32(2.5) + t), mu2, fma(f32(1.5) + t, mu3, f32(1.0)))
    w2 = fma(
        -(f32(1.5) + t), mu3, fma(f32(2.0) + f32(2.0) * t, mu2, (f32(0.5) - t) * mu)
    )
    w3 = (f32(0.5) - t) * (mu3 - mu2)
    point: Points = (w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3).astype(np.float32)
    return point


def extend(points: Points) -> Points:
    """The points with one mirrored point added before the first and after the last."""
    first = points[0] + (points[0] - points[1])
    last = points[-1] + (points[-1] - points[-2])
    return np.concatenate([first[None], points, last[None]]).astype(np.float32)


def at_point(index: int, n_points: int) -> tuple[int, float]:
    """Input point index as a position: its segment at 0, or the last one at 1."""
    segment = min(index, n_points - 2)
    return segment, float(index - segment)


def resample_num_points(points: Points, num_points: int) -> tuple[Points, Positions]:
    """num_points points of one streamline (>= 2 points), at equal arc lengths."""
    n_points = len(points)
    steps = [distance(points[i + 1], points[i]) for i in range(n_points - 1)]
    steps.append(f32(0.0))
    length = f32(0.0)
    for step in steps[:-1]:
        length = f32(length + step)
    extended = extend(points)

    out: list[Points] = []
    segments: list[int] = []
    fractions: list[float] = []
    cumulative = f32(0.0)
    index = 0
    for output_index in range(num_points):
        target = f32(f32(length * f32(output_index)) / f32(num_points - 1))
        while index < n_points and f32(cumulative + steps[index]) < target:
            cumulative = f32(cumulative + steps[index])
            index += 1
        if index >= n_points - 1:
            out.append(points[-1])
            segment, fraction = at_point(n_points - 1, n_points)
            segments.append(segment)
            fractions.append(fraction)
            continue
        if steps[index] != 0:
            mu = f32(f32(target - cumulative) / steps[index])
        else:
            mu = f32(0.5)
        out.append(hermite(mu, *extended[index : index + 4]))
        segments.append(index)
        fractions.append(float(mu))
    return np.array(out, np.float32), (segments, fractions)


def bisect(
    controls: list[Points], previous: Points, step: np.float32, mu_lower: np.float32
) -> tuple[Points, np.float32, np.float32, np.float32, Points, bool]:
    """Bisects mu for the point at step from previous: point, mu, bracket, converged."""
    p_lower, p_upper = controls[1], controls[2]
    mu_upper = f32(1.0)
    mu = mu_lower
    point = p_lower
    first_try = True
    while True:
        next_mu = f32(f32(0.5) * f32(mu_lower + mu_upper))
        if not first_try and next_mu == mu:
            return point, mu, mu_lower, mu_upper, p_upper, False
        first_try = False
        mu = next_mu
        point = hermite(mu, *controls)
        if distance(point, previous) < step:
            mu_lower, p_lower = mu, point
        else:
            mu_upper, p_upper = mu, point
        if not distance(p_upper, p_lower) > BISECTION_TOLERANCE * step:
            return point, mu, mu_lower, mu_upper, p_upper, True


def resample_step_size(points: Points, step_size: float) -> tuple[Points, Positions]:
    """One streamline (>= 2 points) resampled to step_size, from the middle out."""
    step = f32(step_size)
    n_points = len(points)
    extended = extend(points)
    first, last = 1, n_points  # the input's end points in extended
    middle = (n_points + 2) // 2

    middle_segment, middle_fraction = at_point(middle - 1, n_points)
    halves: list[tuple[list[Points], list[int], list[float]]] = []
    for direction in (-1, 1):
        end = first if direction == -1 else last
        half: list[Points] = []
        half_segments: list[int] = []
        half_fractions: list[float] = []
        index = middle
        mu_lower = f32(0.0)
        previous = extended[middle]
        while index != end:
            while (
                index != end and distance(previous, extended[index + direction]) < step
            ):
                index += direction
                mu_lower = f32(0.0)
            if index == end:
                point = extended[end]
                segment, fraction = at_point(end - 1, n_points)
            else:
                controls = [extended[index + k * direction] for k in (-1, 0, 1, 2)]
                point, mu, mu_lower, mu_upper, p_upper, converged = bisect(
                    controls, previous, step, mu_lower
                )
                if not converged:
                    # Below float32 resolution: take the upper end and search beyond
                    # it, or move to the next segment when there is no room left.
                    if not mu_upper > mu_lower:
                        index += direction
                        mu_lower = f32(0.0)
                        if index != end:
                            continue
                        point = extended[end]
                    else:
                        point, mu, mu_lower = p_upper, mu_upper, mu_upper
                if index == end or not np.isfinite(point).all():
                    index = end  # a point that is not finite ends the pass
                    point = extended[end]
                    segment, fraction = at_point(end - 1, n_points)
                elif direction == 1:
                    segment, fraction = index - 1, float(mu)
                else:
                    segment, fraction = index - 2, 1.0 - float(mu)
            half.append(point)
            half_segments.append(segment)
            half_fractions.append(fraction)
            previous = point
        halves.append((half, half_segments, half_fractions))

    backward, forward = halves
    out = [*backward[0][::-1], extended[middle], *forward[0]]
    segments = [*backward[1][::-1], middle_segment, *forward[1]]
    fractions = [*backward[2][::-1], middle_fraction, *forward[2]]
    return np.array(out, np.float32).reshape(-1, 3), (segments, fractions)


def resample(
    tractogram: Tractogram,
    step_size: float | None = None,
    num_points: int | None = None,
) -> Tractogram:
    """The tractogram resampled streamline by streamline, point_data interpolated."""
    pieces: list[Points] = []
    lengths: list[int] = []
    sources: list[int] = []
    fractions: list[float] = []
    for s in range(len(tractogram)):
        start = int(tractogram.offsets[s])
        points = tractogram[s]
        if len(points) < 2:
            out = points.copy()
            positions = (list(range(len(points))), [0.0] * len(points))
        elif step_size is not None:
            out, positions = resample_step_size(points, step_size)
        else:
            assert num_points is not None
            out, positions = resample_num_points(points, num_points)
        pieces.append(out.reshape(-1, 3))
        lengths.append(len(out))
        sources.extend(start + segment for segment in positions[0])
        fractions.extend(positions[1])

    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    points = np.concatenate(pieces) if pieces else np.zeros((0, 3), np.float32)
    lower = np.array(sources, np.int64)
    weight = np.array(fractions, np.float64)
    point_data = {}
    for name, values in tractogram.point_data.items():
        upper = np.minimum(lower + 1, len(values) - 1)
        if np.issubdtype(values.dtype, np.floating):
            shaped = weight.reshape((-1,) + (1,) * (values.ndim - 1))
            with np.errstate(invalid="ignore"):
                mixed = (1.0 - shaped) * values[lower] + shaped * values[upper]
            mixed = np.where(shaped == 0.0, values[lower], mixed)
            mixed = np.where(shaped == 1.0, values[upper], mixed)
            point_data[name] = mixed.astype(values.dtype)
        else:
            point_data[name] = np.where(
                (weight <= 0.5).reshape((-1,) + (1,) * (values.ndim - 1)),
                values[lower],
                values[upper],
            )
    return Tractogram(
        points,
        offsets,
        point_data=point_data,
        streamline_data=dict(tractogram.streamline_data),
        reference=tractogram.reference,
    )
