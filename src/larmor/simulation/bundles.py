"""Synthetic fibre bundles, drawn as streamlines around a smooth centreline."""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import PchipInterpolator

from larmor.types import Tractogram

# A bundle's radius, in standard deviations of the Gaussian across it.
EDGE_SD = 2.5

# How far a streamline may go towards a bend's centre, as a share of the bend's radius.
FOLD_LIMIT = 0.85
# The share of that limit up to which a streamline is left where it is.
FOLD_KNEE = 0.6
# How far a bend is averaged on each side of a point, as a share of the centreline.
FOLD_WINDOW = 0.03

# The shortest tangle, in spacings of a streamline's points.
MIN_TANGLE_SPACINGS = 2.0

# The smallest radius a landing's line gives the section, mm.
MIN_LANDING_RADIUS = 0.25


@dataclass(frozen=True, eq=False)
class Landing:
    """A line that the streamlines of one end of a bundle bend over to.

    The centreline stays as it is. The section at that end takes its first radius
    and its angle from the line, and each streamline lands at a place along the
    line given by its place along that radius. One further out than an end of the
    line lands on that end. A streamline keeps its offset along the second radius,
    unless a thickness line sets that radius and carries the offset along itself.

    Args:
        line: (P, 3) points of the line, in world coordinates in mm.
        thickness_line: (P, 3) points of a second line across the first, whose
            shape and direction count but not where it lies: the streamlines land
            on the surface made by moving the first line along it. Defaults to
            None.
        opens_from: The position u along the centreline, from 0 at its start to 1
            at its end, where the bending starts, in (0, 1). Defaults to 0.5.

    Raises:
        ValueError: If a line is not (P, 3) finite values with P >= 2 and some
            length, or opens_from is outside (0, 1).
    """

    line: np.ndarray
    thickness_line: np.ndarray | None = None
    opens_from: float = 0.5

    def __post_init__(self) -> None:
        """Checks the values and makes the lines read-only arrays."""
        object.__setattr__(self, "line", _checked_line(self.line))
        if self.thickness_line is not None:
            thickness = _checked_line(self.thickness_line)
            object.__setattr__(self, "thickness_line", thickness)
        if not 0 < self.opens_from < 1:
            raise ValueError(f"opens_from must be in (0, 1), got {self.opens_from}.")

    def to_dict(self) -> dict[str, object]:
        """Returns the landing as plain values, without a thickness line if none."""
        values: dict[str, object] = {
            "line": self.line.tolist(),
            "opens_from": self.opens_from,
        }
        if self.thickness_line is not None:
            values["thickness_line"] = self.thickness_line.tolist()
        return values


@dataclass(frozen=True, eq=False)
class BundleGeometry:
    """The shape of a synthetic bundle.

    A geometry describes a bundle by a few control points along its centreline
    and a radius. Each streamline follows the centreline (see
    `centreline_spline`) at an offset across it, drawn from a Gaussian around the
    centreline that is cut at `radius`, so the streamlines are densest at the
    centre. Streamlines on the inside of a bend whose radius is below about twice the
    bundle's are held back, so they squeeze together and fold far less, though
    they can still turn sharply. A corner that turns within a few per cent of the
    centreline's length folds as it would without this. Rings set the section
    along the bundle instead of the radius, so that it can widen, flatten and turn,
    and landings bend the streamlines of an end over to a line. A bundle with
    either fills by streamline length, not as a tube.

    Args:
        control_points: (K, 3) points the centreline passes through, K >= 2, in
            world coordinates (RAS+) in mm.
        num_streamlines: The number of streamlines. Defaults to 1000.
        radius: The bundle's radius in mm, 2.5 standard deviations of the
            Gaussian. Defaults to 2.0.
        num_points: Points per streamline. Defaults to 100.
        seed: The seed of the random offsets, so that the same seed gives the same
            streamlines; None draws new ones each time. Defaults to None.
        rings: (R, 4) rows of u, r1, r2 and angle, which replace `radius` for the
            bundle's shape. u is the position along the centreline, rising from 0
            in the first row to 1 in the last. r1 and r2 are the section's two
            radii in mm. angle turns r1, in degrees and right-handed about the
            tangent, away from the normal of the frame carried along the
            centreline without twist, whose first normal is the first tangent
            crossed with the x axis, or the y axis if the tangent is near x.
            Between rows each value follows a monotone curve, so it stays between
            its neighbours. Defaults to None.
        tangle_mm: The distance along the centreline over which each streamline
            slides from one place across the bundle to another, in mm; 0 for none.
            The slide is a regular wave, so the section repeats every 4 tangle_mm.
            The spread across the bundle keeps its standard deviation, but halfway
            through a slide about 2% of the streamlines pass the edge, by up to
            about a third of the radius. A tube's fill does not change. A straight
            round bundle's streamlines lie about 40 radius / tangle_mm degrees from
            the centreline on average, up to about 25 degrees, and they turn
            sharply at each point below about 2 sqrt(spacing radius), where
            spacing is the distance between points. Defaults to 0.0.
        landing_start: Where the streamlines bend over to at the first point, or
            None. It replaces the first radius and the angle of that end's row of
            rings, and the second radius if it has a thickness line, so that the
            section turns to the line between rows. Without rings, the section keeps
            `radius` until `opens_from` and then stretches along the line without
            turning. Defaults to None.
        landing_end: The same at the last point. Defaults to None.

    Raises:
        ValueError: If the control points are not (K, 3) finite values with
            K >= 2, a count or size is out of range, the rings are not as
            described above, a landing has neither rings nor a radius above 0, or
            tangle_mm is negative or not finite.
    """

    control_points: np.ndarray
    num_streamlines: int = 1000
    radius: float = 2.0
    num_points: int = 100
    seed: int | None = field(default=None)
    rings: np.ndarray | None = None
    tangle_mm: float = 0.0
    landing_start: Landing | None = None
    landing_end: Landing | None = None

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
        if not (math.isfinite(self.radius) and self.radius >= 0):
            raise ValueError("radius must be finite and >= 0.")
        if self.rings is not None:
            object.__setattr__(self, "rings", _checked_rings(self.rings))
        if not (math.isfinite(self.tangle_mm) and self.tangle_mm >= 0):
            raise ValueError("tangle_mm must be finite and >= 0.")
        if self.has_landing and self.rings is None and self.radius == 0:
            raise ValueError("A landing needs rings or a radius above 0.")

    @property
    def has_landing(self) -> bool:
        """Tells whether either end of the bundle has a landing."""
        return self.landing_start is not None or self.landing_end is not None

    @property
    def is_tube(self) -> bool:
        """Tells whether the bundle is a round tube of one radius all along."""
        return self.rings is None and not self.has_landing

    def median_section(self) -> float:
        """Returns the median area of the bundle's section, leaving out its landings.

        A change over less than half of the bundle's length leaves it alone, so it
        tells the size the bundle mostly has. Without rings it is the area of a
        circle of `radius`.

        Returns:
            The area in mm^2.
        """
        if self.rings is None:
            return math.pi * self.radius**2
        u = np.linspace(0.0, 1.0, self.num_points)
        radii = PchipInterpolator(self.rings[:, 0], self.rings[:, 1:3], axis=0)(u)
        return float(np.pi * np.median(np.prod(radii, axis=1)))

    def to_dict(self) -> dict[str, object]:
        """Returns the geometry as plain values that can be saved as JSON.

        Returns:
            A dict of the fields, with the arrays as lists, without the rings and
            landings if there are none and the tangle if it is 0.
        """
        values: dict[str, object] = {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }
        values["control_points"] = self.control_points.tolist()
        if self.rings is None:
            del values["rings"]
        else:
            values["rings"] = self.rings.tolist()
        if self.tangle_mm == 0:
            del values["tangle_mm"]
        for name in ("landing_start", "landing_end"):
            landing = getattr(self, name)
            del values[name]
            if landing is not None:
                values[name] = landing.to_dict()
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
        for name in ("landing_start", "landing_end"):
            if values.get(name) is not None:
                values[name] = Landing(**values[name])  # type: ignore[arg-type]
        return cls(**values)  # type: ignore[arg-type]


def _checked_line(points: object) -> np.ndarray:
    """Returns a landing's line as a read-only (P, 3) array, or raises ValueError."""
    line = np.array(points, dtype=np.float64)
    if line.ndim != 2 or line.shape[1] != 3 or len(line) < 2:
        raise ValueError(f"A landing's line must be (P, 3), got {line.shape}.")
    if not np.isfinite(line).all() or not np.ptp(line, axis=0).any():
        raise ValueError("A landing's line must be finite and have some length.")
    line.flags.writeable = False
    return line


def _checked_rings(rings: object) -> np.ndarray:
    """Returns the rings as a read-only (R, 4) array, or raises ValueError."""
    array = np.array(rings, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(
            f"rings must be (R, 4) rows of u, r1, r2 and angle, got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("rings must be finite.")
    u = array[:, 0]
    if len(u) < 2 or u[0] != 0 or u[-1] != 1 or np.any(np.diff(u) <= 0):
        raise ValueError("The rings' u must rise from 0 in the first to 1 in the last.")
    if np.any(array[:, 1:3] <= 0):
        raise ValueError("The rings' radii must be positive.")
    array.flags.writeable = False
    return array


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

    Raises:
        ValueError: If tangle_mm is under twice the spacing of the points.
    """
    rng = np.random.default_rng(geometry.seed)
    centreline, _ = centreline_spline(geometry.control_points)
    u = np.linspace(0.0, 1.0, geometry.num_points)
    centre = centreline(u)
    tangents = centreline.derivative()(u)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    normals, binormals = _transported_frames(tangents)

    num = geometry.num_streamlines
    shares = _places_along(rng, num, u * centreline.length, geometry.tangle_mm)
    rings, landed = _read_landings(geometry, tangents, normals, binormals)
    # Each streamline's place across the bundle, (S, N, 2) in the frames' coordinates.
    if rings is None:
        place = geometry.radius * shares
        for landing, end, ring in landed:
            opening = _opening(u, landing.opens_from, at_end=end == -1)
            place = place + opening[None, :, None] * _stretch(
                ring, shares, geometry.radius
            )
    else:
        place = _ring_places(rings, u, shares)
    offsets = place[..., :1] * normals[None] + place[..., 1:] * binormals[None]
    step = centreline.length / (geometry.num_points - 1)
    offsets = _squeeze_inside_bends(offsets, *_bend_limits(tangents, step))
    for landing, end, ring in landed:
        # Without rings the places are not yet along the section's radii.
        along = shares[:, end] if rings is not None else shares[:, end] @ _axes(ring)
        shift = _landing_shift(landing, ring, normals[end], binormals[end], along)
        opening = _opening(u, landing.opens_from, at_end=end == -1)
        offsets = offsets + opening[None, :, None] * (shift - centre[end])[:, None]
    points = (centre[None] + offsets).reshape(-1, 3)  # (S * num_points, 3)
    offsets_index = np.arange(num + 1, dtype=np.int64) * geometry.num_points
    return Tractogram(
        points,
        offsets_index,
        streamline_data={"bundle": np.full(num, index, dtype=np.int64)},
    )


@dataclass(frozen=True, eq=False)
class BundlePath:
    """Where a bundle's centreline runs and what its section is along it.

    Attributes:
        centreline: The curve, which takes u from 0 to 1.
        control_u: The u of each control point, (K,).
        u: The positions along the centreline, (N,).
        centre: The centreline there, (N, 3) in mm.
        shown: Where the middle of the bundle runs: the centreline bent over to the
            landings, (N, 3) in mm.
        tangents: The unit tangents, (N, 3).
        normals: The normals of the frame carried along without twist, (N, 3).
        binormals: The binormals of that frame, (N, 3).
        sections: The section's two radii in mm and its angle in degrees, (N, 3).
        rings: The rings with what the landings set in their end rows, (R, 4); None
            without rings.
        end_rings: What each landing sets at its end, as a row (u, r1, r2, angle),
            by the index of that end, 0 or -1.
    """

    centreline: Centreline
    control_u: np.ndarray
    u: np.ndarray
    centre: np.ndarray
    shown: np.ndarray
    tangents: np.ndarray
    normals: np.ndarray
    binormals: np.ndarray
    sections: np.ndarray
    rings: np.ndarray | None
    end_rings: dict[int, np.ndarray]


def bundle_path(geometry: BundleGeometry, num_points: int = 100) -> BundlePath:
    """Returns where a geometry's bundle runs and how wide it is, without streamlines.

    It follows `create_bundle`, for tools that need the shape of a bundle.

    Args:
        geometry: The geometry.
        num_points: The number of positions along the centreline.

    Returns:
        The path.
    """
    centreline, control_u = centreline_spline(geometry.control_points)
    u = np.linspace(0.0, 1.0, num_points)
    centre = centreline(u)
    tangents = centreline.derivative()(u)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    normals, binormals = _transported_frames(tangents)
    rings, landed = _read_landings(geometry, tangents, normals, binormals)
    if rings is None:
        sections = np.tile([geometry.radius, geometry.radius, 0.0], (num_points, 1))
        for landing, end, ring in landed:
            opening = _opening(u, landing.opens_from, at_end=end == -1)
            sections[:, :2] += opening[:, None] * (ring[1:3] - geometry.radius)
            sections[opening > 0, 2] = ring[3]
    else:
        sections = PchipInterpolator(rings[:, 0], rings[:, 1:], axis=0)(u)
    shown = centre.copy()
    for landing, end, ring in landed:
        middle = _landing_shift(
            landing, ring, normals[end], binormals[end], np.zeros((1, 2))
        )[0]
        opening = _opening(u, landing.opens_from, at_end=end == -1)
        shown += opening[:, None] * (middle - centre[end])
    return BundlePath(
        centreline,
        control_u,
        u,
        centre,
        shown,
        tangents,
        normals,
        binormals,
        sections,
        rings,
        {end: ring for _, end, ring in landed},
    )


def bundle_places(geometry: BundleGeometry) -> np.ndarray:
    """Returns where each streamline starts across a bundle, as shares of the radii.

    A streamline keeps these numbers through every change of the bundle's shape, so
    they tell streamlines apart. They are those `create_bundle` draws.

    Args:
        geometry: The geometry, with a seed.

    Returns:
        The places, (S, 2).
    """
    return _gaussian_places(
        np.random.default_rng(geometry.seed), geometry.num_streamlines
    )


def _gaussian_places(rng: np.random.Generator, count: int) -> np.ndarray:
    """Draws places across a bundle from a 2D Gaussian cut at the bundle's edge.

    The distance from the centre comes from the inverse of its cumulative
    distribution, so each place takes the same number of random values.

    Args:
        rng: The random generator.
        count: The number of places.

    Returns:
        The (count, 2) places, as shares of the radius, at most 1 from the centre.
    """
    kept = 1 - np.exp(-(EDGE_SD**2) / 2)
    distance = np.sqrt(-2 * np.log1p(-kept * rng.random(count))) / EDGE_SD
    angle = 2 * np.pi * rng.random(count)
    places: np.ndarray = distance[:, None] * np.stack(
        [np.cos(angle), np.sin(angle)], axis=1
    )
    return places


def _places_along(
    rng: np.random.Generator, count: int, distance: np.ndarray, tangle_mm: float
) -> np.ndarray:
    """Returns each streamline's place along the bundle, as shares of the radii.

    With a tangle, each streamline slides between two places drawn from the same
    Gaussian. Without one it keeps its first place, and only that place is drawn.

    Args:
        rng: The random generator.
        count: The number of streamlines.
        distance: The distance along the centreline of each point, (N,) in mm.
        tangle_mm: The distance of a slide from one place to the other, 0 for none.

    Returns:
        The places, (S, 1, 2) without a tangle and (S, N, 2) with one.

    Raises:
        ValueError: If the tangle is too short for the spacing of the points, which
            cannot show the slide: streamlines would zigzag or not move at all.
    """
    first = _gaussian_places(rng, count)[:, None]
    if tangle_mm == 0:
        return first
    least = MIN_TANGLE_SPACINGS * (distance[1] - distance[0])
    # The margin lets through a value of exactly the limit, whatever the rounding.
    if tangle_mm < least * (1 - 1e-9):
        raise ValueError(
            f"A tangle_mm of {tangle_mm:g} is too short for this bundle's points; "
            f"give it at least {least:.3g} mm or more num_points."
        )
    phi = (np.pi / 2) * distance / tangle_mm
    second = _gaussian_places(rng, count)[:, None]
    places: np.ndarray = first * np.cos(phi)[:, None] + second * np.sin(phi)[:, None]
    return places


def _ring_places(rings: np.ndarray, u: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """Returns each streamline's place across the bundle, from the rings.

    Args:
        rings: The rings, (R, 4).
        u: The positions along the centreline, (N,).
        shares: Each streamline's place as shares of the two radii, (S, 1 or N, 2).

    Returns:
        The places in mm in the frames' coordinates, (S, N, 2).
    """
    r1, r2, angle = PchipInterpolator(rings[:, 0], rings[:, 1:], axis=0)(u).T
    first, second = shares[..., 0] * r1, shares[..., 1] * r2
    cosine, sine = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    return np.stack(
        [first * cosine - second * sine, first * sine + second * cosine], axis=-1
    )


def _read_landings(
    geometry: BundleGeometry,
    tangents: np.ndarray,
    normals: np.ndarray,
    binormals: np.ndarray,
) -> tuple[np.ndarray | None, list[tuple[Landing, int, np.ndarray]]]:
    """Returns a bundle's rings, and for each landing its end and the ring it makes.

    The rings, (R, 4), hold what the landings set in that end's row, and are None
    if the bundle has none. A ring is (u, r1, r2, angle), with u not used without
    rings, where the landing's section starts from a circle of the bundle's radius.
    """
    rings = None if geometry.rings is None else geometry.rings.copy()
    landed = []
    for landing, end, neighbour in (
        (geometry.landing_end, -1, -2),
        (geometry.landing_start, 0, 1),
    ):
        if landing is not None:
            radius = geometry.radius
            ring = np.array([0.0, radius, radius, 0.0]) if rings is None else rings[end]
            nearest = 0.0 if rings is None else float(rings[neighbour, 3])
            across, ring[1] = _across_path(landing.line, tangents[end])
            angle = math.degrees(
                math.atan2(across @ binormals[end], across @ normals[end])
            )
            # A section looks the same turned by half a circle: take the nearest way.
            ring[3] = angle + 180 * round((nearest - angle) / 180)
            if landing.thickness_line is not None:
                ring[2] = _across_path(landing.thickness_line, tangents[end])[1]
            landed.append((landing, end, ring))
    return rings, landed


def _axes(ring: np.ndarray) -> np.ndarray:
    """Returns a ring's two radii as the columns of a 2 by 2 array, in the frame."""
    cosine, sine = math.cos(math.radians(ring[3])), math.sin(math.radians(ring[3]))
    return np.array([[cosine, -sine], [sine, cosine]])


def _stretch(ring: np.ndarray, shares: np.ndarray, radius: float) -> np.ndarray:
    """Returns what opening a round section to a ring adds to its places.

    `shares` holds the places as shares of `radius`, (S, 1 or N, 2), and so does the
    result in mm. The section stretches along the ring's radii without turning, so
    no streamline twists.
    """
    axes = _axes(ring)
    result: np.ndarray = ((shares @ axes) * (ring[1:3] - radius)) @ axes.T
    return result


def _across_path(line: np.ndarray, tangent: np.ndarray) -> tuple[np.ndarray, float]:
    """Returns the part of a line's chord across the path, and the radius it gives."""
    chord = line[-1] - line[0]
    across = chord - tangent * (chord @ tangent)
    return across, max(float(np.linalg.norm(across)) / 2, MIN_LANDING_RADIUS)


def _landing_shift(
    landing: Landing,
    ring: np.ndarray,
    normal: np.ndarray,
    binormal: np.ndarray,
    shares: np.ndarray,
) -> np.ndarray:
    """Returns each streamline's place on a landing, less its offset in the section.

    Args:
        landing: The landing.
        ring: The ring at that end, with what the landing sets in it.
        normal: The frame's normal at that end.
        binormal: The frame's binormal at that end.
        shares: Each streamline's place at that end, as shares of the ring's two
            radii, (S, 2).

    Returns:
        The differences in world coordinates, (S, 3).
    """
    (cosine, minus_sine), (sine, _) = _axes(ring)
    first = cosine * normal + sine * binormal
    shift = _on_line(landing.line, shares[:, 0], ring[1], first)
    if landing.thickness_line is not None:
        second = cosine * binormal + minus_sine * normal
        thickness = landing.thickness_line
        middle = _on_line(thickness, np.zeros(1), ring[2], second)
        shift = shift + _on_line(thickness, shares[:, 1], ring[2], second) - middle
    return shift


def _on_line(
    line: np.ndarray, shares: np.ndarray, radius: float, axis: np.ndarray
) -> np.ndarray:
    """Returns where places along a section's radius land on a line, less those places.

    `shares` holds the (S,) places as shares of `radius`, whose direction is `axis`.
    Shares of -1 and 1 land on the line's ends, and any further out land there too.
    """
    sign = 1.0 if (line[-1] - line[0]) @ axis >= 0 else -1.0
    arc = np.r_[0.0, np.linalg.norm(np.diff(line, axis=0), axis=1).cumsum()]
    t = arc[-1] * (sign * np.clip(shares, -1.0, 1.0) + 1) / 2
    lands = np.stack([np.interp(t, arc, column) for column in line.T], axis=-1)
    result: np.ndarray = lands - (shares * radius)[:, None] * axis
    return result


def _opening(u: np.ndarray, opens_from: float, *, at_end: bool) -> np.ndarray:
    """Returns how far a landing has bent the streamlines over at u, from 0 to 1."""
    t = (u - opens_from) / (1 - opens_from) if at_end else 1 - u / opens_from
    x = np.clip(t, 0.0, 1.0)
    result: np.ndarray = x * x * (3 - 2 * x)
    return result


def _bend_limits(tangents: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns the way to each bend's centre, and how far a streamline may go along it.

    Args:
        tangents: The (N, 3) unit tangents, in order along the curve.
        step: The distance between neighbouring tangents, mm.

    Returns:
        The (N, 3) unit directions to the centre of the bend at each point, and
        the (N,) limits along them in mm.
    """
    half = round(FOLD_WINDOW * (len(tangents) - 1))
    curvature = _moving_mean(np.gradient(tangents, step, axis=0), half)
    size = np.maximum(np.linalg.norm(curvature, axis=1, keepdims=True), 1e-12)
    return curvature / size, FOLD_LIMIT / size[:, 0]


def _moving_mean(values: np.ndarray, half: int) -> np.ndarray:
    """Returns the mean of each row and `half` rows on each side, fewer at the ends."""
    total = np.vstack([np.zeros(values.shape[1]), np.cumsum(values, axis=0)])
    row = np.arange(len(values))
    low, high = np.maximum(row - half, 0), np.minimum(row + half + 1, len(values))
    mean: np.ndarray = (total[high] - total[low]) / (high - low)[:, None]
    return mean


def _squeeze_inside_bends(
    offsets: np.ndarray, towards: np.ndarray, limit: np.ndarray
) -> np.ndarray:
    """Holds streamlines back on the inside of tight bends, so they fold far less.

    A streamline stays where it is up to FOLD_KNEE of the limit. Beyond that it
    approaches the limit smoothly without reaching it, so streamlines keep their
    order across the bundle.

    Args:
        offsets: Each streamline's offset from the centreline, (S, N, 3).
        towards: The directions to the bends' centres, (N, 3).
        limit: How far a streamline may go along them, (N,) in mm.

    Returns:
        The offsets, (S, N, 3).
    """
    reach = np.sum(offsets * towards[None], axis=2)
    past = np.maximum(reach - FOLD_KNEE * limit, 0.0)
    rest = (1 - FOLD_KNEE) * limit
    change = -rest * np.expm1(-past / rest) - past
    result: np.ndarray = offsets + change[..., None] * towards[None]
    return result


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
