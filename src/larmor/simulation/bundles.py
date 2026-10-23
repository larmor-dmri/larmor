"""Synthetic fibre bundles, drawn as streamlines around a smooth centreline."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

from larmor.types import Tractogram


@dataclass(frozen=True, eq=False)
class BundleGeometry:
    """The shape of a synthetic bundle.

    A geometry describes a bundle by a few control points along its centreline,
    a radius, and how much the bundle fans out at its ends. Each streamline is a
    spline through the control points, shifted across the centreline by an
    offset drawn uniformly in a disc of `radius`. The offset is scaled by
    1 + fan_start (1 - u)^p + fan_end u^p, where u in [0, 1] is the position
    along the centreline and p is fan_sharpness. A random wander of standard
    deviation `dispersion` is added at the inner control points.

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


def centreline_spline(control_points: np.ndarray) -> tuple[BSpline, np.ndarray]:
    """Returns the centreline of a bundle, the spline through its control points.

    The spline is cubic, or of lower degree for fewer than four points, and
    parameterised by chord length scaled to [0, 1].

    Args:
        control_points: The control points, (K, 3), in world mm, K >= 2.

    Returns:
        The spline of u in [0, 1], and (K,) the u of each control point.
    """
    control = np.asarray(control_points, dtype=np.float64)
    chord = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(control, axis=0), axis=1))]
    )
    u_control = chord / chord[-1]
    degree = min(3, len(control) - 1)
    return make_interp_spline(u_control, control, k=degree), u_control


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
    control = geometry.control_points
    count = len(control)
    centreline, u_control = centreline_spline(control)
    degree = min(3, count - 1)
    tangents = centreline.derivative()(u_control)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    normals, binormals = _transported_frames(tangents)

    p = geometry.fan_sharpness
    fan = (
        1 + geometry.fan_start * (1 - u_control) ** p + geometry.fan_end * u_control**p
    )
    num = geometry.num_streamlines
    radius = geometry.radius * np.sqrt(rng.random(num))
    angle = 2 * np.pi * rng.random(num)
    across = (radius * np.cos(angle))[:, None, None] * normals[None] + (
        radius * np.sin(angle)
    )[:, None, None] * binormals[None]
    offsets = fan[None, :, None] * across
    if geometry.dispersion > 0:
        wander = geometry.dispersion * rng.standard_normal((num, count, 2))
        wander[:, [0, -1]] = 0.0
        offsets += wander[..., :1] * normals[None] + wander[..., 1:] * binormals[None]

    shifted = control[None] + offsets  # (S, K, 3)
    spline = make_interp_spline(u_control, np.moveaxis(shifted, 1, 0), k=degree)
    u = np.linspace(0.0, 1.0, geometry.num_points)
    points = np.moveaxis(spline(u), 0, 1).reshape(-1, 3)  # (S * num_points, 3)
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

    Returns:
        The (K, 3) unit normals and binormals.
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
