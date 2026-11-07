"""Works out what the drawing tools do to a bundle's shape, as plain values."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Final

import numpy as np

from larmor.viewer.model.phantom import (
    BranchLink,
    BundleSettings,
    GeometrySettings,
    LandingSettings,
    Ring,
    Vec3,
    build_geometry,
    with_geometry,
)

# larmor.simulation and scipy are imported only in the functions that need them, so
# opening the viewer does not load them.

# The most control points one freehand stroke gives.
STROKE_POINTS: Final = 24

# A working copy of a bundle's rings: rows of u, r1, r2 and angle.
type Rows = list[list[float]]
# Gives the section at one position u: r1, r2 and angle.
type Profile = Callable[[float], tuple[float, float, float]]

# The positions along a bundle that the tools measure it at.
SAMPLES: Final = 100
# The narrowest a Width stroke makes a section, mm.
MIN_WIDTH: Final = 0.2
# The smallest radius any row of rings keeps, mm.
MIN_RADIUS: Final = 0.05
# How far a row may be from the shape without it, and still be taken away: at least
# this many mm, this share of the radius there, and this many degrees.
PRUNE_MM: Final = 0.03
PRUNE_SHARE: Final = 0.02
PRUNE_DEGREES: Final = 0.5
# The positions at which `prune_rows` compares shapes.
_PRUNE_SAMPLES: Final = 129
# The stretch over which a radius changed by key while drawing takes effect, mm.
RADIUS_RAMP: Final = 5.0
# How near to a place along a bundle, as a share of its length, a row counts as
# being at that place, at most.
NEAR_ROW: Final = 0.03


@dataclass(frozen=True, eq=False)
class Frame:
    """A bundle's section at one place along it.

    Attributes:
        k: The place's index in the bundle's samples.
        centre: Where the middle of the bundle runs there.
        tangent: The direction of the bundle.
        normal: The normal of the frame carried along without twist.
        binormal: The binormal of that frame.
        first: The direction of the first radius.
        second: The direction of the second radius.
        r1: The first radius, mm.
        r2: The second radius, mm.
        angle: The angle of the first radius, in degrees.
    """

    k: int
    centre: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    binormal: np.ndarray
    first: np.ndarray
    second: np.ndarray
    r1: float
    r2: float
    angle: float


@dataclass(frozen=True, eq=False)
class Shape:
    """Where a bundle runs and how wide it is, measured at `SAMPLES` places.

    A branch's stem is part of it, so that positions u count along the whole path.

    Attributes:
        geometry: The geometry measured.
        stem_points: How many of the first control points are a branch's stem.
        u: The positions along the centreline, (M,).
        base: The centreline there, (M, 3).
        centre: Where the middle of the bundle shows: the centreline bent over to
            any landing, (M, 3).
        tangents: The unit tangents, (M, 3).
        normals: The normals of the frame carried along without twist, (M, 3).
        binormals: The binormals of that frame, (M, 3).
        sections: The two radii in mm and the angle in degrees, (M, 3).
        control: The control points, (K, 3).
        control_u: The u of each control point, (K,).
        length: The length of the centreline, mm.
        stem_u: The position where a branch's stem ends; 0 for other bundles.
        first_own: The first sample past the stem.
        rows: The section as rows of rings, with what the landings set in their
            end rows, whether or not the geometry has rings.
        point: Gives the point of the centreline at one position u.
    """

    geometry: GeometrySettings
    stem_points: int
    u: np.ndarray
    base: np.ndarray
    centre: np.ndarray
    tangents: np.ndarray
    normals: np.ndarray
    binormals: np.ndarray
    sections: np.ndarray
    control: np.ndarray
    control_u: np.ndarray
    length: float
    stem_u: float
    first_own: int
    rows: Rows
    point: Callable[[float], np.ndarray]
    _widths: dict[int, np.ndarray] = field(default_factory=dict)

    def k_at(self, u: float) -> int:
        """Returns the index of the sample nearest a position u."""
        return int(np.clip(round(u * (SAMPLES - 1)), 0, SAMPLES - 1))

    def frame(self, u: float) -> Frame:
        """Returns the section at the sample nearest a position u."""
        k = self.k_at(u)
        r1, r2, angle = self.sections[k]
        cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        normal, binormal = self.normals[k], self.binormals[k]
        return Frame(
            k,
            self.centre[k],
            self.tangents[k],
            normal,
            binormal,
            cosine * normal + sine * binormal,
            cosine * binormal - sine * normal,
            float(r1),
            float(r2),
            float(angle),
        )

    def widths(self, axis: int) -> np.ndarray:
        """Returns how wide the bundle looks from along a world axis, at each sample."""
        known = self._widths.get(axis)
        if known is not None:
            return known
        view = np.eye(3)[axis]
        across = np.cross(view, self.tangents)
        size = np.linalg.norm(across, axis=1, keepdims=True)
        radians = np.radians(self.sections[:, 2])[:, None]
        cosine, sine = np.cos(radians), np.sin(radians)
        first = cosine * self.normals + sine * self.binormals
        second = cosine * self.binormals - sine * self.normals
        # Looking along the bundle, the widest way across shows.
        direction = np.where(size > 0.35, across / np.maximum(size, 1e-9), first)
        a = self.sections[:, 0] * np.sum(direction * first, axis=1)
        b = self.sections[:, 1] * np.sum(direction * second, axis=1)
        width: np.ndarray = np.hypot(a, b)
        self._widths[axis] = width
        return width


def shape_of(geometry: GeometrySettings, stem_points: int = 0) -> Shape:
    """Measures a bundle.

    Args:
        geometry: The geometry, with at least two different control points.
        stem_points: How many of its first control points are a branch's stem.

    Returns:
        The measures.

    Raises:
        ValueError: If the geometry cannot be built.
    """
    from larmor.simulation.bundles import bundle_path

    built = build_geometry(geometry)
    path = bundle_path(built, SAMPLES)
    control = np.array(geometry.control_points, dtype=np.float64)
    stem_u = float(path.control_u[stem_points - 1]) if stem_points > 1 else 0.0
    return Shape(
        geometry,
        stem_points,
        path.u,
        path.centre,
        path.shown,
        path.tangents,
        path.normals,
        path.binormals,
        path.sections,
        control,
        path.control_u,
        float(path.centreline.length),
        stem_u,
        min(SAMPLES - 1, round(stem_u * (SAMPLES - 1))),
        _rows_of(geometry, path.rings, path.end_rings),
        lambda u: np.asarray(path.centreline(np.array([u])))[0],
    )


def _rows_of(
    geometry: GeometrySettings,
    rings: np.ndarray | None,
    end_rings: dict[int, np.ndarray],
) -> Rows:
    """Returns a bundle's section as rows of rings, whether or not it has any."""
    if rings is not None:
        return [[float(v) for v in row] for row in rings]
    r = geometry.radius
    rows = [[0.0, r, r, 0.0], [1.0, r, r, 0.0]]
    # Without rings a landing stretches the section from where it opens, not turning.
    if geometry.landing_start is not None:
        ring = end_rings[0]
        rows[0] = [0.0, float(ring[1]), float(ring[2]), float(ring[3])]
        rows.insert(1, [geometry.landing_start.opens_from, r, r, float(ring[3])])
    if geometry.landing_end is not None:
        ring = end_rings[-1]
        rows[-1] = [1.0, float(ring[1]), float(ring[2]), float(ring[3])]
        opens = geometry.landing_end.opens_from
        if opens > rows[-2][0]:
            rows.insert(-1, [opens, r, r, float(ring[3])])
    return rows


def ring_profile(rows: Rows) -> Profile:
    """Returns the section at any position, from rows of rings in order of u."""
    from scipy.interpolate import PchipInterpolator

    table = np.array(rows, dtype=np.float64)
    curve = PchipInterpolator(table[:, 0], table[:, 1:], axis=0)

    def at(u: float) -> tuple[float, float, float]:
        """Returns r1, r2 and angle at u."""
        r1, r2, angle = curve(u)
        return float(r1), float(r2), float(angle)

    return at


def with_rows(geometry: GeometrySettings, rows: Rows) -> GeometrySettings:
    """Returns a geometry with these rings, their ends and order made valid."""
    ordered = sorted(rows, key=lambda r: r[0])
    kept: Rows = []
    for row in ordered:
        if kept and row[0] - kept[-1][0] < 1e-4:
            kept[-1] = row
        else:
            kept.append(row)
    kept[0][0], kept[-1][0] = 0.0, 1.0
    rings: tuple[Ring, ...] = tuple(
        (r[0], max(r[1], MIN_RADIUS), max(r[2], MIN_RADIUS), r[3]) for r in kept
    )
    return replace(geometry, rings=rings)


def simplify_stroke(
    points: Sequence[Vec3], tolerance: float, most: int = STROKE_POINTS
) -> list[Vec3]:
    """Returns control points for a freehand stroke, evenly spaced along it.

    They are the fewest evenly spaced points, from 2 to `most` and always with
    the stroke's first and last, whose bundle centreline (see
    `centreline_spline`) stays within `tolerance` of the stroke; `most` points
    when no fewer do.

    Args:
        points: The stroke, world mm, in order.
        tolerance: How far the centreline may stray from the stroke, mm.
        most: The most points returned, at least 2.

    Returns:
        The points, in order; one point for a stroke that never moves.
    """
    stroke = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(stroke) == 0:
        return []
    steps = np.linalg.norm(np.diff(stroke, axis=0), axis=1)
    if not np.any(steps > 0):
        return [vec3(stroke[0])]
    stroke = np.vstack([stroke[:1], stroke[1:][steps > 0]])
    arc = np.concatenate([[0.0], np.cumsum(steps[steps > 0])])
    limit = max(tolerance, 1e-9)
    # The stroke resampled finely enough that nearest samples measure distances.
    count = int(min(np.ceil(arc[-1] / (limit / 4)), 4000)) + 1
    dense = _along(stroke, arc, np.linspace(0.0, arc[-1], count))
    chosen = stroke[[0, -1]]
    for n in range(2, max(most, 2) + 1):
        chosen = _along(stroke, arc, np.linspace(0.0, arc[-1], n))
        chosen[0], chosen[-1] = stroke[0], stroke[-1]
        if _follows(chosen, dense, limit):
            break
    return [vec3(p) for p in chosen]


def _along(stroke: np.ndarray, arc: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Returns the points of the stroke at the arc lengths `at`, (N, 3).

    Args:
        stroke: The stroke's points, (M, 3).
        arc: The arc length at each of them, (M,), increasing.
        at: The arc lengths wanted, (N,).
    """
    return np.column_stack([np.interp(at, arc, stroke[:, k]) for k in range(3)])


def _follows(control: np.ndarray, dense: np.ndarray, tolerance: float) -> bool:
    """Returns whether the centreline through `control` keeps close to a stroke.

    Every point of the centreline must be within `tolerance` of the finely
    resampled stroke `dense`, and every point of `dense` within `tolerance` of
    the centreline.
    """
    from scipy.spatial import cKDTree

    # Imported here, so that opening the viewer does not load the simulation.
    from larmor.simulation.bundles import centreline_spline

    if np.any(np.linalg.norm(np.diff(control, axis=0), axis=1) == 0):
        return False
    centreline, _ = centreline_spline(control)
    curve = centreline(np.linspace(0.0, 1.0, len(dense)))
    stray = cKDTree(dense).query(curve)[0].max()
    missed = cKDTree(curve).query(dense)[0].max()
    return bool(max(stray, missed) <= tolerance)


def vec3(point: np.ndarray) -> Vec3:
    """Returns a point as a tuple of three floats."""
    return (float(point[0]), float(point[1]), float(point[2]))


def smoothstep(x: float) -> float:
    """Returns a smooth rise from 0 at x = 0 to 1 at x = 1."""
    t = min(1.0, max(0.0, x))
    return t * t * (3 - 2 * t)


def brush_rows(
    rows: Rows, u: float, reach_u: float, low: float = 0.0
) -> tuple[list[float], list[tuple[list[float], float]]]:
    """Makes the rows a soft brush at a place needs, and says how each one follows.

    One row stands at the place, and others out to the reach on each side hold what
    is there now. Rows already inside the reach take part of the change, so they
    never block it.

    Args:
        rows: The rows, changed in place.
        u: The position of the place.
        reach_u: How far along the change is felt, in u.
        low: The position below which rows stay as they are, a branch's stem.

    Returns:
        The row at the place, and the others that follow it with their shares of
        its change.
    """
    profile = ring_profile(rows)
    near = min(NEAR_ROW, reach_u * 0.3)
    ring = next((r for r in rows if abs(r[0] - u) <= near and r[0] >= low), None)
    if ring is None:
        ring = [u, *profile(u)]
        rows.append(ring)
    for part in (-1.0, -0.5, 0.5, 1.0):
        at = ring[0] + part * reach_u
        if low < at < 1 and not any(abs(r[0] - at) <= near for r in rows):
            rows.append([at, *profile(at)])
    rows.sort(key=lambda r: r[0])
    # A branch's stem stays as its parent has it; another bundle's first row can move.
    others = [
        (r, smoothstep(1 - abs(r[0] - ring[0]) / reach_u))
        for r in rows
        if r is not ring and (r[0] > low or low == 0)
    ]
    return ring, [(r, share) for r, share in others if share > 0]


def ensure_row(rows: Rows, u: float, reach_u: float, low: float = 0.0) -> list[float]:
    """Finds or makes the row at a place, with rows holding the old values at the reach.

    Those rows are not added where a row already stands between them and the place.

    Args:
        rows: The rows, changed in place.
        u: The position of the place.
        reach_u: How far along the change is felt, in u.
        low: The position below which rows stay as they are, a branch's stem.

    Returns:
        The row at the place.
    """
    profile = ring_profile(rows)
    near = min(NEAR_ROW, reach_u * 0.3)
    ring = next((r for r in rows if abs(r[0] - u) <= near and r[0] >= low), None)
    place = ring[0] if ring is not None else u
    for at in (place - reach_u, place + reach_u):
        if low < at < 1:
            first, last = min(at, place), max(at, place)
            if not any(first <= r[0] <= last and r[0] != place for r in rows):
                rows.append([at, *profile(at)])
    if ring is None:
        ring = [u, *profile(u)]
        rows.append(ring)
    rows.sort(key=lambda r: r[0])
    return ring


def prune_rows(rows: Rows, low: float = 0.0) -> Rows:
    """Takes away the rows the bundle would look the same without.

    Each round takes away the row that changes the shape least, measured over the
    whole bundle, until any further one would show, so that repeated strokes at one
    place leave as many rows as one stroke does.

    Args:
        rows: The rows, in order of u.
        low: The position up to which rows stay, a branch's stem.

    Returns:
        The rows kept.
    """
    grid = np.linspace(0.0, 1.0, _PRUNE_SAMPLES)
    kept = list(rows)
    while len(kept) > 2:
        wanted = _sampled(kept, grid)
        limit = np.maximum(PRUNE_MM, PRUNE_SHARE * wanted[:, :2])
        best, best_miss = -1, 1.0
        for i in range(1, len(kept) - 1):
            if kept[i][0] <= low:
                continue
            other = _sampled(kept[:i] + kept[i + 1 :], grid)
            miss = max(
                float(np.max(np.abs(other[:, :2] - wanted[:, :2]) / limit)),
                float(np.max(np.abs(other[:, 2] - wanted[:, 2]))) / PRUNE_DEGREES,
            )
            if miss <= best_miss:
                best, best_miss = i, miss
        if best < 0:
            break
        del kept[best]
    return kept


def _sampled(rows: Rows, grid: np.ndarray) -> np.ndarray:
    """Returns r1, r2 and angle of rows at the positions of a grid, (N, 3)."""
    from scipy.interpolate import PchipInterpolator

    table = np.array(rows, dtype=np.float64)
    sampled: np.ndarray = PchipInterpolator(table[:, 0], table[:, 1:], axis=0)(grid)
    return sampled


def visible_width(
    axis: int, frame: Frame, towards: np.ndarray
) -> tuple[float, np.ndarray]:
    """Returns how wide a bundle looks from along a world axis, and the way across.

    Looking along the bundle, the way across is the one towards a point, or else its
    first radius.

    Args:
        axis: The world axis.
        frame: The section.
        towards: The offset of a point from the section's centre.

    Returns:
        The width in mm, and the unit direction across the bundle in the view.
    """
    view = np.eye(3)[axis]
    across = np.cross(view, frame.tangent)
    size = float(np.linalg.norm(across))
    in_view = towards - view * float(towards @ view)
    if size > 0.35:
        direction = across / size
    elif np.linalg.norm(in_view) > 1e-3:
        direction = in_view / np.linalg.norm(in_view)
    else:
        direction = frame.first
    a = frame.r1 * float(direction @ frame.first)
    b = frame.r2 * float(direction @ frame.second)
    return math.hypot(a, b), direction


# Lines drawn across an end


@dataclass(frozen=True, eq=False)
class ReadLine:
    """A drawn line as its middle and its extent across the path.

    Attributes:
        middle: The point half way along it.
        direction: The unit direction across the path.
        half: Half its extent across the path, mm.
    """

    middle: np.ndarray
    direction: np.ndarray
    half: float


def read_line(points: Sequence[Vec3], tangent: np.ndarray) -> ReadLine:
    """Reads a line against the path's direction at the end it belongs to."""
    line = np.array(points, dtype=np.float64)
    arc = np.r_[0.0, np.linalg.norm(np.diff(line, axis=0), axis=1).cumsum()]
    middle = np.array([np.interp(arc[-1] / 2, arc, column) for column in line.T])
    chord = line[-1] - line[0]
    across = chord - tangent * float(chord @ tangent)
    half = float(np.linalg.norm(across)) / 2
    direction = across / (2 * half) if half > 1e-6 else np.zeros(3)
    return ReadLine(middle, direction, max(half, 0.25))


def scaled_line(
    points: Sequence[Vec3], direction: np.ndarray, middle: np.ndarray, factor: float
) -> tuple[Vec3, ...]:
    """Returns a line stretched along a direction about a point by a factor."""
    line = np.array(points, dtype=np.float64)
    moved = line + direction * (((line - middle) @ direction) * (factor - 1))[:, None]
    return _points(moved)


def rotated_line(
    points: Sequence[Vec3], middle: np.ndarray, axis: np.ndarray, degrees: float
) -> tuple[Vec3, ...]:
    """Returns a line turned about an axis through a point."""
    t = math.radians(degrees)
    cosine, sine = math.cos(t), math.sin(t)
    d = np.array(points, dtype=np.float64) - middle
    turned = (
        d * cosine
        + np.cross(axis, d) * sine
        + axis * ((d @ axis) * (1 - cosine))[:, None]
    )
    return _points(middle + turned)


def _points(array: np.ndarray) -> tuple[Vec3, ...]:
    """Returns (N, 3) values as a tuple of points."""
    return tuple((float(p[0]), float(p[1]), float(p[2])) for p in array)


def stroke_length(points: Sequence[Vec3] | np.ndarray) -> float:
    """Returns the length of a polyline, mm."""
    array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum())


def landing_of(geometry: GeometrySettings, at_end: bool) -> LandingSettings | None:
    """Returns the landing of a geometry's last point, or else of its first."""
    return geometry.landing_end if at_end else geometry.landing_start


def with_landing(
    geometry: GeometrySettings, at_end: bool, landing: LandingSettings | None
) -> GeometrySettings:
    """Returns a geometry with the landing of its last or else first point replaced."""
    if at_end:
        return replace(geometry, landing_end=landing)
    return replace(geometry, landing_start=landing)


# Drawing along a bundle


def rows_from_track(
    track: Sequence[tuple[float, float]], total: float, angle: float = 0.0
) -> Rows:
    """Returns rows from the radii a stroke was drawn with, in order along it.

    Args:
        track: The radius from each distance along the stroke on, mm.
        total: The stroke's length, mm.
        angle: The rows' angle, in degrees.

    Returns:
        The rows, from 0 to 1.
    """
    size = max(total, 1e-6)
    rows: Rows = [[0.0, track[0][1], track[0][1], angle]]
    last_arc, last_r = 0.0, track[0][1]
    for arc, radius in track[1:]:
        if radius == last_r:
            continue
        # The old radius holds until shortly before, so it changes from there on.
        if arc - RADIUS_RAMP > last_arc:
            rows.append([(arc - RADIUS_RAMP) / size, last_r, last_r, angle])
        rows.append([min(arc / size, 1.0), radius, radius, angle])
        last_arc, last_r = arc, radius
    rows.append([1.0, last_r, last_r, angle])
    return rows


def extended(
    shape: Shape,
    added: Sequence[Vec3],
    at_end: bool,
    length: float,
    track: Sequence[tuple[float, float]] | None = None,
) -> tuple[GeometrySettings, float]:
    """Lengthens a bundle from one end.

    The new part starts where the end shows, so a fan on that end, which goes, leaves
    no jump. A row of the same section marks the new end, and rows and the other
    end's landing keep their places along the old part.

    Args:
        shape: The bundle as it is.
        added: The new control points, from the end outwards, the first at the end.
        at_end: Whether it grows from its last point rather than its first.
        length: The length of the new part, mm.
        track: The radii the new part was drawn with, if any changed.

    Returns:
        The geometry, and the share of the new length that the old part takes.
    """
    geometry = shape.geometry
    k = shape.length / (shape.length + length)
    points = list(geometry.control_points)
    new_points = list(added[1:])
    rows = [list(r) for r in shape.rows]
    # A fan on this end goes, so the new part starts where the end shows.
    if at_end:
        landed_end = geometry.landing_end is not None
        tip = _points(shape.centre[-1:])[0] if landed_end else points[-1]
        control = (*points[:-1], tip, *new_points)
        for row in rows:
            row[0] *= k
        last = rows[-1]
        made = _new_rows(track, length, last)
        joint = len(rows) - 1
        rows += [[k + r[0] * (1 - k), r[1], r[2], r[3]] for r in made]
    else:
        landed_start = geometry.landing_start is not None
        tip = _points(shape.centre[:1])[0] if landed_start else points[0]
        control = (*reversed(new_points), tip, *points[1:])
        for row in rows:
            row[0] = 1 - (1 - row[0]) * k
        made = _new_rows(track, length, rows[0])
        grown = [[(1 - k) * (1 - r[0]), r[1], r[2], r[3]] for r in reversed(made)]
        joint = len(grown)
        rows = grown + rows
    made_geometry = with_landing(
        replace(geometry, control_points=control), at_end, None
    )
    other = landing_of(made_geometry, not at_end)
    if other is not None:
        opens = other.opens_from * k if at_end else 1 - (1 - other.opens_from) * k
        made_geometry = with_landing(
            made_geometry, not at_end, replace(other, opens_from=opens)
        )
    if geometry.rings is not None or track is not None:
        rows = _drop_same_middle(rows, joint)
        made_geometry = with_rows(made_geometry, rows)
    return made_geometry, k


def _new_rows(
    track: Sequence[tuple[float, float]] | None, total: float, end_row: list[float]
) -> Rows:
    """Returns the rows of a new part, from 0 at the old end, leaving out that end.

    Without the radii it was drawn with, it keeps the section of the old end.
    """
    if track is None:
        return [[1.0, end_row[1], end_row[2], end_row[3]]]
    return rows_from_track(track, total, end_row[3])[1:]


def _drop_same_middle(rows: Rows, middle: int) -> Rows:
    """Takes away a row between two of the same section, which adds nothing."""
    if 0 < middle < len(rows) - 1:
        before, row, after = rows[middle - 1], rows[middle], rows[middle + 1]
        if before[1:] == row[1:] == after[1:]:
            return rows[:middle] + rows[middle + 1 :]
    return rows


def redrawn(
    shape: Shape,
    world: Sequence[Vec3],
    axis: int,
    ua: float,
    ub: float,
    fit: Callable[[Sequence[Vec3]], list[Vec3]],
) -> GeometrySettings:
    """Redoes a stretch of a bundle along a stroke.

    The stroke gives the two coordinates of its view. The third stays as the bundle
    had it, plus any scrolling.

    Args:
        shape: The bundle as it is.
        world: The stroke, from the place it started.
        axis: The world axis along which its view looks.
        ua: The position along the bundle where the stroke starts.
        ub: The position along the bundle where it stops.
        fit: Makes few control points of a polyline.

    Returns:
        The geometry.
    """
    stroke = np.array(world, dtype=np.float64)
    start_depth = stroke[0, axis]
    if ua > ub:
        stroke, ua, ub = stroke[::-1], ub, ua
    walked = np.r_[0.0, np.linalg.norm(np.diff(stroke, axis=0), axis=1).cumsum()]
    total = max(float(walked[-1]), 1e-6)
    placed = []
    for p, s in zip(stroke, walked, strict=True):
        old = shape.point(ua + (ub - ua) * float(s) / total)
        q = p.copy()
        q[axis] = old[axis] + p[axis] - start_depth
        placed.append((float(q[0]), float(q[1]), float(q[2])))
    stem = shape.stem_points
    u_control = shape.control_u
    points = shape.geometry.control_points
    before = [points[i] for i in range(stem, len(points)) if u_control[i] < ua - 0.015]
    after = [points[i] for i in range(stem, len(points)) if u_control[i] > ub + 0.015]
    control = (*points[:stem], *before, *fit(placed), *after)
    return replace(shape.geometry, control_points=_distinct(control))


def _distinct(points: Sequence[Vec3]) -> tuple[Vec3, ...]:
    """Returns the points without those nearly at the point before them."""
    kept: list[Vec3] = []
    for point in points:
        if not kept or math.dist(point, kept[-1]) > 1e-6:
            kept.append(point)
    return tuple(kept)


# Branches


def stem_points_of(parent: Shape, at: float) -> tuple[Vec3, ...]:
    """Returns the parent's path up to a position, as control points."""
    points = parent.geometry.control_points
    before = [points[i] for i in range(len(points)) if parent.control_u[i] < at - 0.01]
    return (*before, _points(parent.point(at)[None])[0])


def branch_geometry(
    parent: Shape, at: float, own: Sequence[Vec3], seed: int
) -> tuple[GeometrySettings, BranchLink, int]:
    """Makes the geometry of a branch, the stem written into its points and rings.

    It takes half of the parent's streamlines, and the section its parent has where
    it leaves.

    Args:
        parent: The bundle it leaves.
        at: The position along the parent where it leaves.
        own: Its own control points, from where it leaves.
        seed: Its seed.

    Returns:
        The geometry, a link with the parent's id 0, and the streamlines it takes.
    """
    share = parent.geometry.num_streamlines // 2
    plain = parent.geometry.is_tube
    stem = stem_points_of(parent, at)
    control = _joined(stem, own)
    made = replace(
        parent.geometry,
        control_points=control,
        num_streamlines=max(share, 1),
        seed=seed,
        rings=None,
        landing_start=None,
        landing_end=None,
    )
    if not plain:
        section = ring_profile(parent.rows)(at)
        rows = _stem_rows(parent, at, shape_of(made, len(stem)).stem_u)
        rows.append([1.0, *section])
        made = with_rows(made, rows)
    return made, BranchLink(0, at, len(stem)), max(share, 1)


def _joined(stem: Sequence[Vec3], own: Sequence[Vec3]) -> tuple[Vec3, ...]:
    """Returns a stem and a branch's own points, not repeating the place they meet."""
    rest = list(own)
    if len(rest) > 1 and math.dist(rest[0], stem[-1]) < 3:
        rest = rest[1:]
    return _distinct((*stem, *rest))


def _stem_rows(parent: Shape, at: float, stem_u: float) -> Rows:
    """Returns the rows of a branch's stem, its parent's up to where it leaves."""
    profile = ring_profile(parent.rows)
    scale = stem_u / at if at > 0 else 0.0
    rows = [[r[0] * scale, r[1], r[2], r[3]] for r in parent.rows if r[0] < at - 1e-6]
    if not rows:
        rows = [[0.0, *profile(0.0)]]
    if stem_u > rows[-1][0]:
        rows.append([stem_u, *profile(at)])
    return rows


def with_new_stem(
    branch: GeometrySettings, link: BranchLink, parent: GeometrySettings
) -> tuple[GeometrySettings, BranchLink]:
    """Writes a branch's stem again from its parent, keeping its own part.

    Args:
        branch: The branch.
        link: Its link.
        parent: Its parent, as it now is.

    Returns:
        The branch's geometry and link.
    """
    old = shape_of(branch, link.stem_points)
    own = branch.control_points[link.stem_points :]
    parent_shape = shape_of(parent)
    stem = stem_points_of(parent_shape, link.at)
    control = _joined(stem, (stem[-1], *own))
    made = replace(branch, control_points=control)
    new_link = replace(link, stem_points=len(stem))
    if parent.is_tube and branch.rings is None:
        return replace(made, radius=parent.radius), new_link
    stem_u = shape_of(made, len(stem)).stem_u
    rows = _stem_rows(parent_shape, link.at, stem_u)
    for row in old.rows:
        if row[0] > old.stem_u + 1e-6:
            spare = (row[0] - old.stem_u) / max(1 - old.stem_u, 1e-6)
            rows.append([stem_u + spare * (1 - stem_u), *row[1:]])
    return with_rows(made, rows), new_link


def with_family(
    bundles: Sequence[BundleSettings], changed: dict[int, BundleSettings]
) -> dict[int, BundleSettings]:
    """Follows changed bundles with their branches, and the branches' own branches.

    Args:
        bundles: The phantom's bundles, parents before their branches.
        changed: The bundles changed, by item.

    Returns:
        The changed bundles and all that hang on them, by item.
    """
    result = dict(changed)
    for bundle in bundles:
        link = bundle.branch
        if link is None or link.parent not in result or bundle.geometry is None:
            continue
        parent = result[link.parent].geometry
        if parent is None:
            continue
        current = result.get(bundle.item_id, bundle)
        assert current.geometry is not None and current.branch is not None
        geometry, new_link = with_new_stem(current.geometry, current.branch, parent)
        result[bundle.item_id] = replace(
            with_geometry(current, geometry), branch=new_link
        )
    return result


# Bending, cutting and landing


def bent(
    shape: Shape, u: float, delta: np.ndarray, reach: float, whole: bool
) -> GeometrySettings:
    """Moves a place along a bundle, with the parts nearby following.

    Args:
        shape: The bundle as it was at the press.
        u: The position grabbed.
        delta: How far it moved, mm.
        reach: The distance along the bundle at which a part follows by 0.6, mm x 2.
        whole: Whether the whole bundle moves instead.

    Returns:
        The geometry.
    """
    geometry = shape.geometry
    s0 = u * shape.length
    weight = _follow_weight(s0, reach, whole)
    stem = shape.stem_points
    items = [
        (shape.control[i], float(shape.control_u[i]) * shape.length)
        for i in range(stem, len(shape.control))
    ]
    if not whole and not any(abs(s - s0) < reach * 0.25 for _, s in items):
        items.append((shape.point(u), s0))
        items.sort(key=lambda item: item[1])
    moved = [p + delta * weight(s) for p, s in items]
    control = (*geometry.control_points[:stem], *_points(np.array(moved)))
    made = replace(geometry, control_points=_distinct(control))
    # A fan's line goes with the end it belongs to.
    for at_end, s in ((True, shape.length), (False, 0.0)):
        landing = landing_of(made, at_end)
        if landing is not None:
            by = delta * weight(s)
            made = with_landing(made, at_end, _moved_landing(landing, by))
    return made


def _follow_weight(s0: float, reach: float, whole: bool) -> Callable[[float], float]:
    """Returns how much a part at a distance along a bundle follows a grab at s0."""
    sigma = reach / 2

    def weight(s: float) -> float:
        """Returns the share of the move at s."""
        return 1.0 if whole else math.exp(-0.5 * ((s - s0) / sigma) ** 2)

    return weight


def _moved_landing(landing: LandingSettings, by: np.ndarray) -> LandingSettings:
    """Returns a landing with its lines moved."""
    thickness = landing.thickness_line
    return replace(
        landing,
        line=_points(np.array(landing.line) + by),
        thickness_line=None if thickness is None else _points(np.array(thickness) + by),
    )


def bent_branch(
    branch: Shape,
    at: float,
    parent_length: float,
    u: float,
    delta: np.ndarray,
    reach: float,
    whole: bool,
) -> GeometrySettings:
    """Moves a branch's own points with a grab on its parent, less further along it.

    Args:
        branch: The branch as it was at the press.
        at: The position along the parent where it leaves.
        parent_length: The length of the parent, mm.
        u: The position grabbed along the parent.
        delta: How far the parent's place moved, mm.
        reach: As for `bent`.
        whole: Whether the whole parent moves.

    Returns:
        The geometry; its stem is written again from the parent after this.
    """
    w = _follow_weight(u * parent_length, reach, whole)(at * parent_length)
    sigma = reach / 2
    points = branch.control[branch.stem_points :]
    walked = 0.0
    moved = []
    for i, p in enumerate(points):
        if i:
            walked += float(np.linalg.norm(p - points[i - 1]))
        share = 1.0 if whole else w * math.exp(-0.5 * (walked / sigma) ** 2)
        moved.append(p + delta * share)
    control = (
        *branch.geometry.control_points[: branch.stem_points],
        *_points(np.array(moved)),
    )
    return replace(branch.geometry, control_points=control)


@dataclass(frozen=True, eq=False)
class Cut:
    """What cutting a bundle leaves.

    Attributes:
        geometry: The part kept.
        at: Where the bundle was cut along its old length.
        keep_start: Whether the part up to the cut was kept.
    """

    geometry: GeometrySettings
    at: float
    keep_start: bool

    def branch_at(self, at: float) -> float | None:
        """Returns where a branch at a position now leaves; None if it went too."""
        if self.keep_start:
            return None if at > self.at else at / self.at
        return None if at < self.at else (at - self.at) / (1 - self.at)


def cut(shape: Shape, u: float, keep_start: bool) -> Cut | None:
    """Cuts a bundle at a position and keeps one side.

    Rows and the kept end's landing stay at their places, and a landing on the end
    cut off goes. A branch's stem always stays.

    Args:
        shape: The bundle as it is.
        u: The position of the cut.
        keep_start: Whether to keep the part up to the cut.

    Returns:
        The part kept; None if too little is left of it.
    """
    geometry = shape.geometry
    points = geometry.control_points
    stem = shape.stem_points
    at = _points(shape.point(u)[None])[0]
    rows = None if geometry.rings is None else [list(r) for r in shape.rows]
    section = ring_profile(shape.rows)(u)
    if keep_start:
        kept = [
            points[i]
            for i in range(stem, len(points))
            if shape.control_u[i] < u - 0.005
        ]
        control = (*points[:stem], *kept, at)
        new_rows = [
            [r[0] / u, r[1], r[2], r[3]] for r in (rows or []) if r[0] < u - 0.01
        ] + [[1.0, *section]]
    else:
        kept = [
            points[i]
            for i in range(stem, len(points))
            if shape.control_u[i] > u + 0.005
        ]
        control = (*points[:stem], at, *kept)
        new_rows = [[0.0, *section]] + [
            [(r[0] - u) / (1 - u), r[1], r[2], r[3]]
            for r in (rows or [])
            if r[0] > u + 0.01
        ]
    control = _distinct(control)
    if len(control) < 2:
        return None
    made = with_landing(replace(geometry, control_points=control), keep_start, None)
    if keep_start and made.landing_start is not None:
        opens = min(made.landing_start.opens_from / u, 0.95)
        made = replace(
            made, landing_start=replace(made.landing_start, opens_from=opens)
        )
    if not keep_start and made.landing_end is not None:
        opens = max((made.landing_end.opens_from - u) / (1 - u), 0.05)
        made = replace(made, landing_end=replace(made.landing_end, opens_from=opens))
    if geometry.rings is not None:
        made = with_rows(made, new_rows)
    return Cut(made, u, keep_start)


@dataclass(frozen=True, eq=False)
class Landed:
    """What a line drawn across a bundle's end does.

    Attributes:
        geometry: The bundle with its landing.
        second: Whether the line set the fan's thickness rather than the fan.
        cut: The cut, if the line lay across the bundle well before its end.
    """

    geometry: GeometrySettings
    second: bool
    cut: Cut | None


def landed(
    shape: Shape,
    at_end: bool,
    world: Sequence[Vec3],
    tolerance: float,
    opens: float,
) -> Landed | None:
    """Makes a line drawn across a bundle's end the line its streamlines land on.

    A line closer to the sheet's thickness than to its width sets the thickness; any
    other replaces the line that was there. The path stays as drawn, so a new try
    leaves nothing behind. Only a line across the bundle, well before its end, stops
    the bundle there.

    Args:
        shape: The bundle as it was at the press.
        at_end: Whether the line belongs to the last point rather than the first.
        world: The line drawn, world mm.
        tolerance: How far the line kept may stray from the line drawn, mm.
        opens: Where the fan begins to open, as a share of the way along the bundle's
            own part towards that end.

    Returns:
        The result; None if the line has no length.
    """
    line = tuple(simplify_stroke(world, tolerance, 12))
    if len(line) < 2 or not np.ptp(np.array(line), axis=0).any():
        return None
    geometry = shape.geometry
    existing = landing_of(geometry, at_end)
    if existing is not None:
        frame = shape.frame(1.0 if at_end else 0.0)
        read = read_line(world, frame.tangent)
        if abs(read.direction @ frame.second) > abs(read.direction @ frame.first):
            made = with_landing(
                geometry, at_end, replace(existing, thickness_line=line)
            )
            return Landed(made, True, None)
    length = stroke_length(world)
    array = np.array(world, dtype=np.float64)
    walked = np.r_[0.0, np.linalg.norm(np.diff(array, axis=0), axis=1).cumsum()]
    middle = np.array([np.interp(length / 2, walked, c) for c in array.T])
    first = shape.first_own if at_end else 0
    gaps = np.linalg.norm(shape.base[first:] - middle, axis=1)
    k = int(np.argmin(gaps)) + first
    u_cut = float(shape.u[k])
    from_end = (1 - u_cut if at_end else u_cut) * shape.length
    the_cut = None
    made_shape = shape
    if gaps[k - first] < length / 2 and from_end > max(10.0, 0.08 * shape.length):
        the_cut = cut(shape, u_cut, at_end)
        if the_cut is None:
            return None
        made_shape = shape_of(the_cut.geometry, shape.stem_points)
    return Landed(_with_fan(made_shape, at_end, line, opens), False, the_cut)


def _with_fan(
    shape: Shape, at_end: bool, line: tuple[Vec3, ...], opens: float
) -> GeometrySettings:
    """Gives a bundle's end a landing, with a row holding the section where it opens."""
    geometry = shape.geometry
    own = opens if at_end else 1 - opens
    opens_at = shape.stem_u + own * (1 - shape.stem_u)
    fan = LandingSettings(line, None, opens_at)
    made = with_landing(geometry, at_end, fan)
    if geometry.rings is None:
        return made
    rows = [list(r) for r in shape.rows]
    if not any(abs(r[0] - opens_at) < 0.12 for r in rows):
        rows.append([opens_at, *ring_profile(rows)(opens_at)])
        rows.sort(key=lambda r: r[0])
    # A round row can take the line's angle unseen, so the section does not bulge
    # out of the line's plane on its way there.
    ring = shape_of(made, shape.stem_points).rows[-1 if at_end else 0]
    for row in rows:
        if abs(row[0] - opens_at) < 0.12 and abs(row[1] - row[2]) < 0.05 * row[1]:
            row[3] = ring[3]
    return with_rows(made, rows)


def with_opening(shape: Shape, at_end: bool, opens: float) -> GeometrySettings | None:
    """Moves where a bundle's fan begins to open, with the row that holds the section.

    Args:
        shape: The bundle.
        at_end: Which end's fan.
        opens: Where it begins to open, as a share of the way along the bundle's own
            part towards that end.

    Returns:
        The geometry; None if that end has no fan.
    """
    geometry = shape.geometry
    landing = landing_of(geometry, at_end)
    if landing is None:
        return None
    own = opens if at_end else 1 - opens
    opens_at = shape.stem_u + own * (1 - shape.stem_u)
    made = with_landing(geometry, at_end, replace(landing, opens_from=opens_at))
    if geometry.rings is None:
        return made
    rows = [list(r) for r in shape.rows]
    previous = landing.opens_from
    row = next((r for r in rows if abs(r[0] - previous) < 0.13 and 0 < r[0] < 1), None)
    if row is None:
        rows.append([opens_at, *ring_profile(rows)(opens_at)])
    else:
        row[0] = opens_at
    rows.sort(key=lambda r: r[0])
    return with_rows(made, rows)


def settled(geometry: GeometrySettings, stem_points: int = 0) -> GeometrySettings:
    """Writes what the landings set into the end rows, as a saved file should hold."""
    if geometry.rings is None or (
        geometry.landing_start is None and geometry.landing_end is None
    ):
        return geometry
    return with_rows(geometry, shape_of(geometry, stem_points).rows)


def crossing_of(screen: np.ndarray, centre: np.ndarray) -> tuple[float, float] | None:
    """Finds where a polyline on the screen first crosses a bundle's centre on it.

    Args:
        screen: The polyline, (P, 2) in pixels.
        centre: The centre, (C, 2) in pixels.

    Returns:
        The place along the centre, in its segments, and the place along the polyline
        in its segments; None if they do not cross.
    """
    for i in range(len(screen) - 1):
        a, b = screen[i], screen[i + 1]
        ba = b - a
        c, d = centre[:-1], centre[1:]
        dc = d - c
        den = ba[0] * dc[:, 1] - ba[1] * dc[:, 0]
        ca = c - a
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ca[:, 0] * dc[:, 1] - ca[:, 1] * dc[:, 0]) / den
            v = (ca[:, 0] * ba[1] - ca[:, 1] * ba[0]) / den
        hits = np.flatnonzero(
            (np.abs(den) > 1e-9) & (t >= 0) & (t <= 1) & (v >= 0) & (v <= 1)
        )
        if len(hits):
            j = int(hits[0])
            return j + float(v[j]), i + float(t[j])
    return None
