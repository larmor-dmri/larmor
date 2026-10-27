"""Synthetic fibre bundles, drawn as streamlines around a smooth centreline."""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import PchipInterpolator

from larmor.types import Tractogram


@dataclass(frozen=True, eq=False)
class BundleGeometry:
    """The shape of a synthetic bundle.

    A geometry describes a bundle by a few control points along its centreline,
    a radius, and how much the bundle fans out at its ends. Each streamline
    follows the centreline (see `centreline_spline`) at a fixed offset across
    it, drawn uniformly in a disc of `radius`. The offset is scaled by
    1 + fan_start (1 - u)^p + fan_end u^p, where u in [0, 1] is the position
    along the centreline and p is fan_sharpness. A smooth random wander of
    standard deviation `dispersion`, zero at both ends, is added along the way.

    Args:
        control_points: (K, 3) points the centreline passes through, K >= 2, in
            world coordinates (RAS+) in mm.
        num_streamlines: The number of streamlines. Defaults to 1000.
        radius: The bundle's radius in mm. Defaults to 2.0.
        num_points: Points per streamline. Defaults to 100.
        fan_start: Extra spread at the first control point, where the disc has
            the radius radius (1 + fan_start). Defaults to 0.0.
        fan_end: Extra spread at the last control point, in the same way.
            Defaults to 0.0.
        fan_sharpness: p, how close to the ends the fanning stays; a larger p
            keeps it nearer the ends. Defaults to 2.0.
        dispersion: Standard deviation of the wander, mm. Defaults to 0.0.
        seed: The seed of the random offsets and wander, so that the same seed
            gives the same streamlines; None draws new ones each time. Defaults
            to None.

    Raises:
        ValueError: If the control points are not (K, 3) finite values with
            K >= 2, or a count or size is out of range.
    """

    control_points: np.ndarray
    num_streamlines: int = 1000
    radius: float = 2.0
    num_points: int = 100
    fan_start: float = 0.0
    fan_end: float = 0.0
    fan_sharpness: float = 2.0
    dispersion: float = 0.0
    seed: int | None = field(default=None)

    def __post_init__(self) -> None:
        """Checks the values and makes the control points a read-only array."""
        points = np.array(self.control_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError(
                f"control_points must be (K, 3) with K >= 2, got {points.shape}."
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("control_points must be finite.")
        if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) == 0):
            raise ValueError("Consecutive control points must differ.")
        points.flags.writeable = False
        object.__setattr__(self, "control_points", points)
        if self.num_streamlines < 1 or self.num_points < 2:
            raise ValueError("num_streamlines must be >= 1 and num_points >= 2.")
        sizes = (self.radius, self.fan_start, self.fan_end, self.dispersion)
        if not all(math.isfinite(s) and s >= 0 for s in sizes):
            raise ValueError("radius, fans and dispersion must be finite and >= 0.")
        if not self.fan_sharpness > 0:
            raise ValueError("fan_sharpness must be positive.")

    def to_dict(self) -> dict[str, object]:
        """Returns the geometry as plain values that can be saved as JSON.

        Returns:
            A dict of the fields, with the control points as lists.
        """
        values: dict[str, object] = {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }
        values["control_points"] = self.control_points.tolist()
        return values

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> BundleGeometry:
        """Builds a geometry from the values `to_dict` gives.

        Args:
            data: The dict.

        Returns:
            The geometry.

        Raises:
            ValueError: If a value is out of range.
        """
        values = dict(data)
        values["control_points"] = np.asarray(values["control_points"], dtype=float)
        return cls(**values)  # type: ignore[arg-type]


# Samples per centreline segment in the table that maps arc length to the curve.
_SAMPLES_PER_SEGMENT = 64


@dataclass(frozen=True, eq=False)
class Centreline:
    """A bundle's centreline as a curve of u in [0, 1] that follows its arc length.

    Calling it with an array of u returns the (N, 3) points there.
    """

    coefficients: np.ndarray  # (K - 1, 4, 3) cubic coefficients per segment
    u_table: np.ndarray  # u of the arc length table's samples
    g_table: np.ndarray  # segment number plus position in it, of the same samples
    length: float  # the curve's length, mm

    def __call__(self, u: np.ndarray) -> np.ndarray:
        """Returns the points at u, (N, 3)."""
        return self._evaluate(u, derivative=False)

    def derivative(self) -> Callable[[np.ndarray], np.ndarray]:
        """Returns the function that gives the curve's derivative with respect to u.

        The derivative has the curve's length as its norm, since u follows the
        arc length.
        """

        def derivative(u: np.ndarray) -> np.ndarray:
            velocity = self._evaluate(u, derivative=True)
            speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
            result: np.ndarray = velocity / np.maximum(speed, 1e-300) * self.length
            return result

        return derivative

    def _evaluate(self, u: np.ndarray, *, derivative: bool) -> np.ndarray:
        """Returns the points, or their derivatives within a segment, at u."""
        g = np.interp(np.asarray(u, dtype=np.float64), self.u_table, self.g_table)
        segment = np.clip(np.floor(g).astype(np.int64), 0, len(self.coefficients) - 1)
        s = (g - segment)[..., None]
        c = self.coefficients[segment]  # (..., 4, 3)
        result: np.ndarray
        if derivative:
            result = c[..., 1, :] + s * (2 * c[..., 2, :] + 3 * s * c[..., 3, :])
        else:
            result = c[..., 0, :] + s * (
                c[..., 1, :] + s * (c[..., 2, :] + s * c[..., 3, :])
            )
        return result


def centreline_spline(control_points: np.ndarray) -> tuple[Centreline, np.ndarray]:
    """Returns the centreline of a bundle, the curve through its control points.

    The curve is a centripetal Catmull-Rom spline, which passes through every
    control point without loops or cusps between them; mirrored points past the
    ends set its direction there, so two control points give a straight segment.

    Args:
        control_points: The control points, (K, 3), in world mm, K >= 2, with no
            two consecutive points equal.

    Returns:
        The curve, and (K,) the u of each control point.

    References:
        Yuksel, C., Schaefer, S. and Keyser, J. Parameterization and
        applications of Catmull-Rom curves. Computer-Aided Design 43(7),
        747-755 (2011). doi:10.1016/j.cad.2010.08.008
    """
    control = np.asarray(control_points, dtype=np.float64)
    ends = np.vstack(
        [2 * control[0] - control[1], control, 2 * control[-1] - control[-2]]
    )
    p0, p1, p2, p3 = ends[:-3], ends[1:-2], ends[2:-1], ends[3:]

    # Knot intervals: the square root of the distance between points (alpha = 0.5).
    def interval(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        result: np.ndarray = np.sqrt(np.linalg.norm(b - a, axis=1))[:, None]
        return result

    d01, d12, d23 = interval(p0, p1), interval(p1, p2), interval(p2, p3)
    # The tangents at both ends of each segment, scaled to a segment of length one.
    m1 = ((p1 - p0) / d01 - (p2 - p0) / (d01 + d12) + (p2 - p1) / d12) * d12
    m2 = ((p2 - p1) / d12 - (p3 - p1) / (d12 + d23) + (p3 - p2) / d23) * d12
    coefficients = np.stack(
        [p1, m1, -3 * p1 + 3 * p2 - 2 * m1 - m2, 2 * p1 - 2 * p2 + m1 + m2], axis=1
    )

    count = len(coefficients)
    s = np.linspace(0.0, 1.0, _SAMPLES_PER_SEGMENT + 1)[:-1]
    g_table = np.append((np.arange(count)[:, None] + s).ravel(), float(count))
    curve = Centreline(coefficients, np.linspace(0.0, 1.0, len(g_table)), g_table, 1.0)
    samples = curve._evaluate(curve.u_table, derivative=False)
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(samples, axis=0), axis=1))]
    )
    u_table = arc / arc[-1]
    curve = Centreline(coefficients, u_table, g_table, float(arc[-1]))
    return curve, u_table[::_SAMPLES_PER_SEGMENT].copy()


def create_bundle(geometry: BundleGeometry, *, index: int = 0) -> Tractogram:
    """Creates the streamlines of a synthetic bundle.

    Args:
        geometry: The bundle's shape and seed.
        index: The bundle's index, stored for every streamline in
            streamline_data["bundle"]. Defaults to 0.

    Returns:
        A tractogram of num_streamlines streamlines of num_points points each, in
        world coordinates (mm).
    """
    rng = np.random.default_rng(geometry.seed)
    centreline, u_control = centreline_spline(geometry.control_points)
    u = np.linspace(0.0, 1.0, geometry.num_points)
    centre = centreline(u)
    tangents = centreline.derivative()(u)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    normals, binormals = _transported_frames(tangents)

    p = geometry.fan_sharpness
    fan = 1 + geometry.fan_start * (1 - u) ** p + geometry.fan_end * u**p
    num = geometry.num_streamlines
    radius = geometry.radius * np.sqrt(rng.random(num))
    angle = 2 * np.pi * rng.random(num)
    # Each streamline's place in the disc, (S, N, 2) in the frames' coordinates.
    across = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    disc = fan[None, :, None] * across[:, None, :]
    if geometry.dispersion > 0:
        wander = geometry.dispersion * rng.standard_normal((len(u_control), num, 2))
        wander[[0, -1]] = 0.0
        disc += np.moveaxis(PchipInterpolator(u_control, wander, axis=0)(u), 0, 1)
    offsets = disc[..., :1] * normals[None] + disc[..., 1:] * binormals[None]
    points = (centre[None] + offsets).reshape(-1, 3)  # (S * num_points, 3)
    offsets_index = np.arange(num + 1, dtype=np.int64) * geometry.num_points
    return Tractogram(
        points,
        offsets_index,
        streamline_data={"bundle": np.full(num, index, dtype=np.int64)},
    )


def _transported_frames(tangents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns normals and binormals along the tangents, carried without twist.

    Each normal is the previous one carried along by parallel transport, so the
    frame does not spin around the centreline.

    Args:
        tangents: The (N, 3) unit tangents, in order along the curve.

    Returns:
        The (N, 3) unit normals and binormals.
    """
    first = tangents[0]
    helper = np.array([1.0, 0, 0]) if abs(first[0]) < 0.9 else np.array([0, 1.0, 0])
    normal = np.cross(first, helper)
    normal /= np.linalg.norm(normal)
    normals = [normal]
    for previous, current in zip(tangents[:-1], tangents[1:], strict=True):
        axis = np.cross(previous, current)
        sine = np.linalg.norm(axis)
        cosine = float(np.clip(np.dot(previous, current), -1, 1))
        if sine > 1e-12:
            axis /= sine
            # Rodrigues' rotation of the normal about axis by the tangents' angle.
            normal = (
                normal * cosine
                + np.cross(axis, normal) * sine
                + axis * np.dot(axis, normal) * (1 - cosine)
            )
        normal = normal - np.dot(normal, current) * current
        normal /= np.linalg.norm(normal)
        normals.append(normal)
    normal_array = np.array(normals)
    return normal_array, np.cross(tangents, normal_array)
