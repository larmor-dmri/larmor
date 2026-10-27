"""Selecting, cropping, joining and editing the streamlines of tractograms.

A selection returns a boolean mask with one entry per streamline, which
`tractogram[mask]` applies.
"""

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from larmor._log import LarmorLogger
from larmor.tractography import regions as region_module
from larmor.tractography.regions import Box, Region, Sphere, point_test
from larmor.tractography.resampling import resample
from larmor.types import Grid, Tractogram, Volume

# Points per chunk when measuring segments.
_LENGTH_CHUNK = 1_000_000


def streamline_lengths(tractogram: Tractogram) -> np.ndarray:
    """Returns the length of each streamline.

    The length is the sum of the distances between consecutive points, computed in
    float64; a streamline of 0 or 1 points has length 0.

    Args:
        tractogram: The streamlines.

    Returns:
        np.ndarray: (S,) float64, the lengths in mm.

    References:
        Follows MRtrix3's tckstats. Tournier, J.-D. et al. MRtrix3: A fast,
        flexible and open software framework for medical image processing and
        visualisation. NeuroImage 202, 116137 (2019).
        doi:10.1016/j.neuroimage.2019.116137
    """
    points = tractogram.points
    offsets = tractogram.offsets
    num_segments = max(len(points) - 1, 0)
    # Segment i joins points i and i + 1; those crossing from one streamline to the
    # next are set to 0.
    segments = np.zeros(num_segments + 1, dtype=np.float64)
    for start in range(0, num_segments, _LENGTH_CHUNK):
        stop = min(start + _LENGTH_CHUNK, num_segments)
        steps = points[start + 1 : stop + 1].astype(np.float64) - points[start:stop]
        segments[start:stop] = np.sqrt(np.einsum("ij,ij->i", steps, steps))
    boundaries = offsets[1:-1]
    segments[boundaries[(boundaries > 0) & (boundaries <= num_segments)] - 1] = 0.0

    # Each streamline of 2 points or more sums its segments from its first point
    # up to the next such streamline's first point; the segments in between are 0.
    lengths = np.zeros(tractogram.num_streamlines, dtype=np.float64)
    with_segments = np.diff(offsets) >= 2
    if with_segments.any():
        lengths[with_segments] = np.add.reduceat(segments, offsets[:-1][with_segments])
    return lengths


def _check_limits(low: float, high: float, name: str) -> None:
    """Raises a ValueError if a limit is NaN or the lower one exceeds the upper one."""
    if math.isnan(low) or math.isnan(high):
        raise ValueError(f"The {name} limits must be numbers, got {low} and {high}.")
    if low > high:
        raise ValueError(
            f"The minimum {name} ({low}) is larger than the maximum {name} ({high})."
        )


def select_length(
    tractogram: Tractogram, *, min_length: float = 0.0, max_length: float = math.inf
) -> np.ndarray:
    """Selects the streamlines whose length is within limits.

    The length is that of `streamline_lengths`; both limits are inclusive.

    Args:
        tractogram: The streamlines.
        min_length: The shortest length kept, in mm. Defaults to 0.
        max_length: The longest length kept, in mm. Defaults to infinity.

    Returns:
        np.ndarray: (S,) bool, True for the streamlines kept.

    Raises:
        ValueError: If a limit is NaN or `min_length` is larger than `max_length`.

    References:
        Follows the -minlength and -maxlength options of MRtrix3's tckedit.
        Tournier, J.-D. et al. MRtrix3: A fast, flexible and open software framework
        for medical image processing and visualisation. NeuroImage 202, 116137
        (2019). doi:10.1016/j.neuroimage.2019.116137
    """
    _check_limits(min_length, max_length, "length")
    lengths = streamline_lengths(tractogram)
    selected: np.ndarray = (lengths >= min_length) & (lengths <= max_length)
    return selected


def select_weight(
    tractogram: Tractogram,
    *,
    min_weight: float = -math.inf,
    max_weight: float = math.inf,
) -> np.ndarray:
    """Selects the streamlines whose weight is within limits.

    The weights are `streamline_data["weight"]`, compared in float64; both limits
    are inclusive. A NaN weight is never selected.

    Args:
        tractogram: The streamlines, with a "weight" entry in `streamline_data`,
            such as one read with `read_streamline_weights`.
        min_weight: The smallest weight kept. Defaults to minus infinity.
        max_weight: The largest weight kept. Defaults to infinity.

    Returns:
        np.ndarray: (S,) bool, True for the streamlines kept.

    Raises:
        ValueError: If the tractogram has no "weight" entry, the entry is not one
            value per streamline, a limit is NaN or `min_weight` is larger than
            `max_weight`.

    References:
        Follows the -minweight and -maxweight options of MRtrix3's tckedit.
        Tournier, J.-D. et al. MRtrix3: A fast, flexible and open software framework
        for medical image processing and visualisation. NeuroImage 202, 116137
        (2019). doi:10.1016/j.neuroimage.2019.116137
    """
    _check_limits(min_weight, max_weight, "weight")
    weights = tractogram.streamline_data.get("weight")
    if weights is None:
        raise ValueError(
            'The tractogram has no streamline_data["weight"]; read the weights '
            "with read_streamline_weights and store them there first."
        )
    if weights.ndim != 1:
        raise ValueError(
            'streamline_data["weight"] must hold one value per streamline, got '
            f"shape {weights.shape}."
        )
    weights = weights.astype(np.float64)
    selected: np.ndarray = (weights >= min_weight) & (weights <= max_weight)
    return selected


def concatenate_tractograms(tractograms: Sequence[Tractogram]) -> Tractogram:
    """Joins tractograms into one, their streamlines one after the other.

    The data entries (`point_data` and `streamline_data`) that every input has,
    with the same kind of dtype (float, signed int, unsigned int, ...) and the same
    shape after the first axis, are joined with a common dtype; the others are
    dropped with a warning. The reference is kept when every input has one and they
    all match (`Grid.matches`); otherwise it is None, with a warning unless no input
    has one. Inputs with no streamlines are left out of both rules unless every
    input is empty.

    Args:
        tractograms: The tractograms, in order. Any of them may be empty.

    Returns:
        Tractogram: The joined tractogram, with the reference of the first input
            that has streamlines (or of the first input, if all are empty) when it
            is kept.

    Raises:
        ValueError: If no tractogram is given.
    """
    if len(tractograms) == 0:
        raise ValueError("concatenate_tractograms needs at least one tractogram.")

    num_points = np.cumsum([0] + [t.num_points for t in tractograms])
    offsets = np.concatenate(
        [[0]]
        + [
            t.offsets[1:] + shift
            for t, shift in zip(tractograms, num_points[:-1], strict=True)
        ]
    )
    with_rows = [t for t in tractograms if t.num_streamlines > 0] or tractograms
    return Tractogram(
        points=np.concatenate([t.points for t in tractograms]),
        offsets=offsets,
        point_data=_join_data([t.point_data for t in with_rows], "point_data"),
        streamline_data=_join_data(
            [t.streamline_data for t in with_rows], "streamline_data"
        ),
        reference=_joined_reference(with_rows),
    )


def _join_data(data: list[dict[str, np.ndarray]], kind: str) -> dict[str, np.ndarray]:
    """Returns the joined entries all inputs have with one dtype kind and row shape.

    `data` holds the entries of each input and `kind` names them in the warning
    about the dropped ones.
    """
    joined: dict[str, np.ndarray] = {}
    dropped: list[str] = []
    names = list(dict.fromkeys(name for entries in data for name in entries))
    for name in names:
        arrays = [entries.get(name) for entries in data]
        present = [a for a in arrays if a is not None]
        if (
            len(present) < len(arrays)
            or len({a.shape[1:] for a in present}) > 1
            or len({a.dtype.kind for a in present}) > 1
        ):
            dropped.append(name)
            continue
        joined[name] = np.concatenate(present, dtype=np.result_type(*present))
    if dropped:
        LarmorLogger.warning(
            "Dropped the %s entries %s: not every tractogram has them with the same "
            "kind of dtype and shape.",
            kind,
            dropped,
        )
    return joined


def _joined_reference(tractograms: Sequence[Tractogram]) -> Grid | None:
    """Returns the common reference of the inputs, or None (warning if they differ)."""
    references = [t.reference for t in tractograms]
    first = references[0]
    if all(r is None for r in references):
        return None
    if first is not None and all(
        r is not None and first.matches(r) for r in references
    ):
        return first
    LarmorLogger.warning(
        "The tractograms do not all have the same reference grid; the joined "
        "tractogram has none."
    )
    return None


# ---------------------------------------------------------------------------
# Selection and cropping by region
# ---------------------------------------------------------------------------

type _PointTest = Callable[[np.ndarray], np.ndarray]


def _as_region_list(regions: Region | Sequence[Region]) -> list[Region]:
    """Returns the regions as a list, a single region becoming a list of one.

    Raises:
        ValueError: If no region is given.
    """
    if isinstance(regions, (Sphere, Box, Volume)):
        listed = [regions]
    else:
        listed = list(regions)
    if not listed:
        raise ValueError("Expected at least one region, got none.")
    return listed


def _streamline_chunks(offsets: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yields consecutive ranges [first, stop) of streamlines of about CHUNK_POINTS."""
    num_streamlines = len(offsets) - 1
    first = 0
    while first < num_streamlines:
        limit = offsets[first] + region_module.CHUNK_POINTS
        stop = int(np.searchsorted(offsets, limit, side="right")) - 1
        stop = min(max(stop, first + 1), num_streamlines)
        yield first, stop
        first = stop


def _inside_any(points: np.ndarray, tests: Sequence[_PointTest]) -> np.ndarray:
    """Returns (P,) bool, True where a point passes at least one of the region tests."""
    inside = np.zeros(len(points), dtype=bool)
    for test in tests:
        inside |= test(points)
    return inside


def _first_region(points: np.ndarray, tests: Sequence[_PointTest]) -> np.ndarray:
    """Returns (P,) the index of the first region containing each point, or -1."""
    dtype = np.min_scalar_type(-len(tests))
    assigned = np.full(len(points), -1, dtype=dtype)
    # Last region first, so that earlier regions overwrite later ones.
    for index in range(len(tests) - 1, -1, -1):
        assigned[tests[index](points)] = index
    return assigned


def _endpoints(tractogram: Tractogram) -> tuple[np.ndarray, np.ndarray]:
    """Returns the endpoints of the N non-empty streamlines and which those are.

    The endpoints are (2N, 3), first and last point of each in turn; the mask of the
    non-empty streamlines is (S,) bool.
    """
    non_empty = np.diff(tractogram.offsets) > 0
    firsts = tractogram.offsets[:-1][non_empty]
    lasts = tractogram.offsets[1:][non_empty] - 1
    points = tractogram.points[np.stack([firsts, lasts], axis=1).ravel()]
    return points, non_empty


def _any_per_streamline(inside: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Returns (S,) bool, True for the streamlines with at least one True point.

    `inside` holds one value per point and `offsets` the streamlines' offsets.
    """
    lengths = np.diff(offsets)
    result = np.zeros(len(lengths), dtype=bool)
    non_empty = lengths > 0
    if non_empty.any():
        # Without the empty streamlines the starts strictly increase, so each
        # reduction runs over exactly one streamline's points.
        result[non_empty] = np.logical_or.reduceat(inside, offsets[:-1][non_empty])
    return result


def _all_per_streamline(inside: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Returns (S,) bool, True for the non-empty streamlines with only True points."""
    lengths = np.diff(offsets)
    result = np.zeros(len(lengths), dtype=bool)
    non_empty = lengths > 0
    if non_empty.any():
        result[non_empty] = np.logical_and.reduceat(inside, offsets[:-1][non_empty])
    return result


def passes_through(
    tractogram: Tractogram, region: Region, *, ends_only: bool = False
) -> np.ndarray:
    """Tests which streamlines pass through a region.

    A streamline passes through the region when at least one of its points is
    inside it (see `points_inside`); with `ends_only`, only its first and last
    points are tested. Empty streamlines never pass.

    Args:
        tractogram: The streamlines.
        region: A `Sphere`, a `Box`, or a 3D mask `Volume` on any grid.
        ends_only: Whether to test only the two endpoints. Defaults to False.

    Returns:
        np.ndarray: (S,) bool, True for the streamlines that pass through.

    Raises:
        TypeError: If the region is not a `Sphere`, a `Box` or a `Volume`.
        ValueError: If the mask is neither 3D nor 4D with a single volume.

    References:
        Follows MRtrix3's tckedit (version 3.0.8). Tournier, J.-D. et al. MRtrix3: A
        fast, flexible and open software framework for medical image processing and
        visualisation. NeuroImage 202, 116137 (2019).
        doi:10.1016/j.neuroimage.2019.116137
    """
    test = point_test(region)
    result = np.zeros(tractogram.num_streamlines, dtype=bool)
    if ends_only:
        endpoints, non_empty = _endpoints(tractogram)
        result[non_empty] = _inside_any(endpoints, [test]).reshape(-1, 2).any(axis=1)
        return result
    for first, stop in _streamline_chunks(tractogram.offsets):
        start, end = tractogram.offsets[first], tractogram.offsets[stop]
        offsets = tractogram.offsets[first : stop + 1] - start
        result[first:stop] = _any_per_streamline(
            test(tractogram.points[start:end]), offsets
        )
    return result


def stays_within(
    tractogram: Tractogram, region: Region, *, ends_only: bool = False
) -> np.ndarray:
    """Tests which streamlines stay within a region.

    A streamline stays within the region when every one of its points is inside
    it (see `points_inside`); with `ends_only`, when both its first and last points
    are. Empty streamlines never stay within.

    Args:
        tractogram: The streamlines.
        region: A `Sphere`, a `Box`, or a 3D mask `Volume` on any grid.
        ends_only: Whether to test only the two endpoints. Defaults to False.

    Returns:
        np.ndarray: (S,) bool, True for the streamlines that stay within.

    Raises:
        TypeError: If the region is not a `Sphere`, a `Box` or a `Volume`.
        ValueError: If the mask is neither 3D nor 4D with a single volume.
    """
    test = point_test(region)
    result = np.zeros(tractogram.num_streamlines, dtype=bool)
    if ends_only:
        endpoints, non_empty = _endpoints(tractogram)
        result[non_empty] = test(endpoints).reshape(-1, 2).all(axis=1)
        return result
    for first, stop in _streamline_chunks(tractogram.offsets):
        start, end = tractogram.offsets[first], tractogram.offsets[stop]
        offsets = tractogram.offsets[first : stop + 1] - start
        result[first:stop] = _all_per_streamline(
            test(tractogram.points[start:end]), offsets
        )
    return result


def passes_in_order(
    tractogram: Tractogram, regions: Sequence[Region], *, ends_only: bool = False
) -> np.ndarray:
    """Tests which streamlines pass through the regions in the given order.

    Each point is assigned to the first region, in the given order, that contains
    it. Along the streamline, ignoring the points in no region, the assigned regions
    must start at the first region, then either stay in the same region or move to
    the next one, and end at the last region. So a streamline may leave a region
    into no region and come back to it, but never go back to an earlier region or
    skip one. With `ends_only`, only the first and then the last point are assigned,
    which allows at most two regions. Empty streamlines never pass.

    Args:
        tractogram: The streamlines.
        regions: One or more spheres or 3D masks, in order.
        ends_only: Whether to test only the two endpoints. Defaults to False.

    Returns:
        np.ndarray: (S,) bool, True for the streamlines that pass through all the
            regions in order.

    Raises:
        TypeError: If a region is not a `Sphere`, a `Box` or a `Volume`.
        ValueError: If no region is given, more than two with `ends_only`, or a
            mask is neither 3D nor 4D with a single volume.

    References:
        Follows MRtrix3's tckedit (version 3.0.8). Tournier, J.-D. et al. MRtrix3: A
        fast, flexible and open software framework for medical image processing and
        visualisation. NeuroImage 202, 116137 (2019).
        doi:10.1016/j.neuroimage.2019.116137
    """
    regions = _as_region_list(regions)
    if ends_only and len(regions) > 2:
        raise ValueError(
            "Only the two endpoints are tested with ends_only, so they cannot pass "
            f"through {len(regions)} regions in order."
        )
    tests = [point_test(region) for region in regions]
    num_regions = len(regions)
    result = np.zeros(tractogram.num_streamlines, dtype=bool)
    if ends_only:
        endpoints, non_empty = _endpoints(tractogram)
        pairs = np.arange(0, len(endpoints) + 1, 2, dtype=np.int64)
        assigned = _first_region(endpoints, tests)
        result[non_empty] = _in_order(assigned, pairs, num_regions)
        return result

    # Every region must be visited, so the ordered test runs only on the
    # streamlines through one of them, a sphere if there is one, as it is the
    # cheapest to test.
    first_test = tests[
        next((i for i, r in enumerate(regions) if isinstance(r, Sphere)), 0)
    ]
    for first, stop in _streamline_chunks(tractogram.offsets):
        start, end = tractogram.offsets[first], tractogram.offsets[stop]
        points = tractogram.points[start:end]
        offsets = tractogram.offsets[first : stop + 1] - start
        candidates = _any_per_streamline(first_test(points), offsets)
        if not candidates.any():
            continue
        lengths = np.diff(offsets)
        candidate_points = points[np.repeat(candidates, lengths)]
        candidate_offsets = np.concatenate([[0], np.cumsum(lengths[candidates])])
        assigned = _first_region(candidate_points, tests)
        in_order = _in_order(assigned, candidate_offsets, num_regions)
        result[first:stop][candidates] = in_order
    return result


def _in_order(
    assigned: np.ndarray, offsets: np.ndarray, num_regions: int
) -> np.ndarray:
    """Returns (S,) bool, the streamlines whose regions go 0, 1, ..., last in order.

    `assigned` holds the region of each point (-1 for none), as from
    `_first_region`, and `offsets` the streamlines' offsets.
    """
    num_streamlines = len(offsets) - 1
    hits = np.flatnonzero(assigned >= 0)
    if len(hits) == 0:
        return np.zeros(num_streamlines, dtype=bool)
    labels = assigned[hits]
    owners = np.searchsorted(offsets, hits, side="right") - 1
    del hits

    same_streamline = owners[1:] == owners[:-1]
    # The labels are not negative, so their differences fit their signed type.
    steps = np.diff(labels)
    bad_step = same_streamline & ((steps < 0) | (steps > 1))
    has_bad_step = np.zeros(num_streamlines, dtype=bool)
    has_bad_step[owners[1:][bad_step]] = True

    firsts = np.flatnonzero(np.concatenate([[True], ~same_streamline]))
    lasts = np.concatenate([firsts[1:], [len(labels)]]) - 1
    result = np.zeros(num_streamlines, dtype=bool)
    hit_owners = owners[firsts]
    result[hit_owners] = (
        (labels[firsts] == 0)
        & (labels[lasts] == num_regions - 1)
        & ~has_bad_step[hit_owners]
    )
    return result


def crop(
    tractogram: Tractogram,
    regions: Region | Sequence[Region],
    *,
    keep_inside: bool = True,
) -> Tractogram:
    """Cuts the streamlines to the parts inside (or outside) the regions.

    Each streamline is cut into its runs of consecutive points inside any of the
    regions (outside all of them with `keep_inside=False`); runs of fewer than 2
    points are dropped, so a streamline that leaves and re-enters gives several
    pieces, and one that is never inside gives none. The pieces come in the order
    of the input streamlines and of their points. Each piece keeps its points'
    `point_data` and its streamline's `streamline_data`.

    Args:
        tractogram: The streamlines.
        regions: One or more spheres or 3D masks.
        keep_inside: Whether to keep the parts inside the regions; False keeps the
            parts outside all of them. Defaults to True.

    Returns:
        Tractogram: The pieces, with their data and the same reference.

    Raises:
        TypeError: If a region is not a `Sphere`, a `Box` or a `Volume`.
        ValueError: If no region is given, or a mask is neither 3D nor 4D with a
            single volume.

    References:
        Follows MRtrix3's tckedit (version 3.0.8). Tournier, J.-D. et al. MRtrix3: A
        fast, flexible and open software framework for medical image processing and
        visualisation. NeuroImage 202, 116137 (2019).
        doi:10.1016/j.neuroimage.2019.116137
    """
    pieces, _ = _crop_with_sources(tractogram, regions, keep_inside)
    return pieces


def _crop_with_sources(
    tractogram: Tractogram, regions: Region | Sequence[Region], keep_inside: bool = True
) -> tuple[Tractogram, np.ndarray]:
    """Runs `crop` and also returns the (N,) int64 input streamline of each piece."""
    tests = [point_test(region) for region in _as_region_list(regions)]
    in_piece = np.zeros(tractogram.num_points, dtype=bool)
    all_begins = [np.zeros(0, dtype=np.int64)]
    all_stops = [np.zeros(0, dtype=np.int64)]
    for first, stop in _streamline_chunks(tractogram.offsets):
        start, end = tractogram.offsets[first], tractogram.offsets[stop]
        keep = _inside_any(tractogram.points[start:end], tests)
        if not keep_inside:
            np.logical_not(keep, out=keep)
        offsets = tractogram.offsets[first : stop + 1] - start
        begins, stops = _runs(keep, offsets)
        # The pieces do not overlap, so their points are those marked between a
        # begin and its stop.
        marks = np.zeros(len(keep) + 1, dtype=np.int8)
        marks[begins] = 1
        marks[stops] -= 1
        in_piece[start:end] = np.cumsum(marks[:-1], dtype=np.int8).astype(bool)
        all_begins.append(begins + start)
        all_stops.append(stops + start)

    begins = np.concatenate(all_begins)
    stops = np.concatenate(all_stops)
    sources = np.searchsorted(tractogram.offsets, begins, side="right") - 1
    offsets = np.concatenate([[0], np.cumsum(stops - begins)]).astype(np.int64)
    pieces = Tractogram(
        points=tractogram.points[in_piece],
        offsets=offsets,
        point_data={
            name: values[in_piece] for name, values in tractogram.point_data.items()
        },
        streamline_data={
            name: values[sources] for name, values in tractogram.streamline_data.items()
        },
        reference=tractogram.reference,
    )
    return pieces, sources.astype(np.int64)


def _runs(keep: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns the starts and exclusive stops of the runs of >= 2 kept points.

    Runs do not cross from one streamline to the next (`offsets`).
    """
    num_points = len(keep)
    non_empty = np.diff(offsets) > 0
    streamline_first = np.zeros(num_points, dtype=bool)
    streamline_first[offsets[:-1][non_empty]] = True
    streamline_last = np.zeros(num_points, dtype=bool)
    streamline_last[offsets[1:][non_empty] - 1] = True

    # A run begins at a kept point that starts its streamline or follows a dropped
    # point, and ends at a kept point that ends its streamline or precedes one.
    previous_dropped = np.ones(num_points, dtype=bool)
    np.logical_not(keep[:-1], out=previous_dropped[1:])
    next_dropped = np.ones(num_points, dtype=bool)
    np.logical_not(keep[1:], out=next_dropped[:-1])
    begins = np.flatnonzero(keep & (streamline_first | previous_dropped))
    stops = np.flatnonzero(keep & (streamline_last | next_dropped)) + 1

    long_enough = stops - begins >= 2
    return begins[long_enough].astype(np.int64), stops[long_enough].astype(np.int64)


# ---------------------------------------------------------------------------
# The whole edit
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class EditOptions:
    """The selections and edits of one whole edit, for `edit` and `edit_files`.

    Every field defaults to "not given", which is None, False or no regions. The
    regions may be given as any sequence and are stored as tuples; each is a
    `Sphere`, a `Box`, a mask `Volume`, or text that `load_region` reads (`x,y,z,r`
    or a mask image's path), which only `edit_files` accepts. `edit` says in which
    order the options apply.

    Args:
        min_length: The shortest length kept, in mm (inclusive).
        max_length: The longest length kept, in mm (inclusive).
        min_weight: The smallest weight kept (inclusive); needs
            `streamline_data["weight"]`.
        max_weight: The largest weight kept (inclusive).
        include: Regions every kept streamline passes through.
        include_ordered: Regions every kept streamline passes through in this
            order, see `passes_in_order`.
        exclude: Regions no kept streamline passes through.
        within: Regions every kept streamline stays within, see `stays_within`.
        ends_only: Whether to test only the two endpoints against `include`,
            `include_ordered`, `exclude` and `within`.
        inverse: Whether to keep the streamlines that fail at least one of the
            length, weight and region tests, and crop to the parts outside `mask`.
        mask: Regions to crop the streamlines to.
        skip: Number of selected streamlines to drop first.
        every: Keep every `every`-th selected streamline, starting with the first.
        number: Number of selected streamlines to keep, the first ones.
        subsample: Number of selected streamlines to keep, chosen at random.
        seed: Seed of the random choice of `subsample`.
        step_size: Step size to resample the result to, in mm.
        num_points: Number of points per streamline to resample the result to.

    Raises:
        ValueError: If both `number` and `subsample` are given, or both
            `step_size` and `num_points`; if `step_size <= 0`, `num_points < 2`,
            a minimum is above its maximum or a limit is NaN, `every < 1`, `skip`,
            `number` or `subsample` is negative, `seed` is given without
            `subsample`, or `ends_only` is given with no include, ordered,
            exclude or within region or with more than two ordered regions.
    """

    min_length: float | None = None
    max_length: float | None = None
    min_weight: float | None = None
    max_weight: float | None = None
    include: Sequence[Region | str | Path] = ()
    include_ordered: Sequence[Region | str | Path] = ()
    exclude: Sequence[Region | str | Path] = ()
    within: Sequence[Region | str | Path] = ()
    ends_only: bool = False
    inverse: bool = False
    mask: Sequence[Region | str | Path] = ()
    skip: int | None = None
    every: int | None = None
    number: int | None = None
    subsample: int | None = None
    seed: int | None = None
    step_size: float | None = None
    num_points: int | None = None

    def __post_init__(self) -> None:
        """Stores the regions as tuples and checks the options."""
        for name in ("include", "include_ordered", "exclude", "within", "mask"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        _check_limits(
            -math.inf if self.min_length is None else self.min_length,
            math.inf if self.max_length is None else self.max_length,
            "length",
        )
        _check_limits(
            -math.inf if self.min_weight is None else self.min_weight,
            math.inf if self.max_weight is None else self.max_weight,
            "weight",
        )
        if self.number is not None and self.subsample is not None:
            raise ValueError("Give at most one of number and subsample.")
        self._check_resampling_and_counts()
        self._check_ends_only()

    def _check_resampling_and_counts(self) -> None:
        """Checks the resampling options, the counts and the seed.

        Raises:
            ValueError: If an option is out of range or given without the one it
                needs.
        """
        if self.step_size is not None and self.num_points is not None:
            raise ValueError("Give at most one of step_size and num_points.")
        if self.step_size is not None and not self.step_size > 0:
            raise ValueError(f"step_size must be positive, got {self.step_size}.")
        if self.num_points is not None and self.num_points < 2:
            raise ValueError(f"num_points must be at least 2, got {self.num_points}.")
        if self.every is not None and self.every < 1:
            raise ValueError(f"every must be at least 1, got {self.every}.")
        for name in ("skip", "number", "subsample"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative, got {value}.")
        if self.seed is not None and self.subsample is None:
            raise ValueError("seed applies only with subsample.")

    def _check_ends_only(self) -> None:
        """Checks that ends_only comes with regions it can test.

        Raises:
            ValueError: If ends_only is set without regions, or with more than two
                ordered regions.
        """
        if self.ends_only and not (
            self.include or self.include_ordered or self.exclude or self.within
        ):
            raise ValueError(
                "ends_only applies to include, include_ordered, exclude and within "
                "regions; give at least one."
            )
        if self.ends_only and len(self.include_ordered) > 2:
            raise ValueError(
                "Only the two endpoints are tested with ends_only, so they cannot "
                f"pass through {len(self.include_ordered)} ordered regions."
            )

    @property
    def _selects(self) -> bool:
        """Returns whether a length, weight or region test is given."""
        return self._selects_by_weight or bool(
            self.min_length is not None
            or self.max_length is not None
            or self.include
            or self.include_ordered
            or self.exclude
            or self.within
        )

    @property
    def _selects_by_weight(self) -> bool:
        """Returns whether a weight limit is given."""
        return self.min_weight is not None or self.max_weight is not None


@dataclass
class EditCounts:
    """How far `skip`, `every` and `number` have counted, carried across batches.

    Attributes:
        skipped: Streamlines skipped so far.
        after_skip: Streamlines counted by `every` so far, after the skipped ones.
        kept: Streamlines kept so far.
    """

    skipped: int = 0
    after_skip: int = 0
    kept: int = 0

    def reached_number(self, options: EditOptions) -> bool:
        """Returns whether `number` streamlines are kept."""
        return options.number is not None and self.kept >= options.number


def edit(tractogram: Tractogram, options: EditOptions) -> Tractogram:
    """Runs a whole edit: selection, cropping, counting and resampling.

    In this order:

    1. The length and weight limits and the include, ordered include, exclude and
       within regions give one selection, which keeps a streamline when it passes
       all of them. With `inverse` the selection is flipped, keeping the
       streamlines that fail at least one test; with no test given, every
       streamline is kept.
    2. The kept streamlines are cropped to the `mask` regions (with `inverse`, to
       the parts outside them), see `crop`.
    3. `skip`, `every` and then `number` or `subsample` count the selected
       streamlines: the first `skip` are dropped, then every `every`-th is kept
       starting with the first, then the first `number` (or a random `subsample`)
       of those. A streamline cropped into several pieces counts as one, and one
       cropped to nothing is gone. When fewer streamlines are left than `number`
       or `subsample` asks for, all are kept; `subsample` logs a warning.
    4. The result is resampled when `step_size` or `num_points` is given, see
       `resample`.

    Only the tests given drop streamlines, so empty streamlines and negative weights
    are otherwise kept. Each piece keeps its points' `point_data` and its
    streamline's `streamline_data`; the reference is kept.

    Args:
        tractogram: The streamlines, with `streamline_data["weight"]` when a
            weight limit is given.
        options: The selections and edits.

    Returns:
        Tractogram: A new tractogram with the edited streamlines, in input order.

    Raises:
        ValueError: If a weight limit is given and the tractogram has no "weight"
            entry, or a mask region is neither 3D nor 4D with a single volume.
        TypeError: If a region is still text or a path (see `load_region`).

    References:
        Follows MRtrix3's tckedit and tckresample (version 3.0.8). Tournier, J.-D.
        et al. MRtrix3: A fast, flexible and open software framework for medical
        image processing and visualisation. NeuroImage 202, 116137 (2019).
        doi:10.1016/j.neuroimage.2019.116137
    """
    return edit_batch(tractogram, options, EditCounts())


def edit_batch(t: Tractogram, options: EditOptions, counts: EditCounts) -> Tractogram:
    """Runs `edit` on one batch of a larger tractogram, counting on from `counts`."""
    selected = _selection(t, options)
    if selected is not None:
        if options.inverse:
            np.logical_not(selected, out=selected)
        t = t[selected]
        del selected

    # The input streamline of each piece; the count runs over these.
    sources = None
    if options.mask:
        mask = _loaded(options.mask)
        t, sources = _crop_with_sources(t, mask, not options.inverse)
    units = (
        np.arange(t.num_streamlines, dtype=np.int64)
        if sources is None
        else np.unique(sources)
    )
    chosen = _count(units, options, counts)
    if chosen is not None:
        t = t[chosen if sources is None else np.isin(sources, chosen)]

    if options.step_size is not None or options.num_points is not None:
        t = resample(t, step_size=options.step_size, num_points=options.num_points)
    return t


def _selection(tractogram: Tractogram, options: EditOptions) -> np.ndarray | None:
    """Returns (S,) bool, the streamlines passing every test, or None without tests."""
    if not options._selects:
        return None
    selected = np.ones(tractogram.num_streamlines, dtype=bool)
    if options.min_length is not None or options.max_length is not None:
        selected &= select_length(
            tractogram,
            min_length=0.0 if options.min_length is None else options.min_length,
            max_length=math.inf if options.max_length is None else options.max_length,
        )
    if options._selects_by_weight:
        selected &= select_weight(
            tractogram,
            min_weight=-math.inf if options.min_weight is None else options.min_weight,
            max_weight=math.inf if options.max_weight is None else options.max_weight,
        )
    for region in _loaded(options.include):
        selected &= passes_through(tractogram, region, ends_only=options.ends_only)
    if options.include_ordered:
        ordered = _loaded(options.include_ordered)
        selected &= passes_in_order(tractogram, ordered, ends_only=options.ends_only)
    for region in _loaded(options.exclude):
        selected &= ~passes_through(tractogram, region, ends_only=options.ends_only)
    for region in _loaded(options.within):
        selected &= stays_within(tractogram, region, ends_only=options.ends_only)
    return selected


def _loaded(specs: Sequence[Region | str | Path]) -> list[Region]:
    """Returns the regions, checking that none is still text or a path.

    Raises:
        TypeError: If a region is given as text or a path.
    """
    regions: list[Region] = []
    for spec in specs:
        if isinstance(spec, (str, Path)):
            raise TypeError(
                f"The region {str(spec)!r} is given as text: read it with load_region "
                "first, or edit files with edit_files, which reads it."
            )
        regions.append(spec)
    return regions


def _count(
    units: np.ndarray, options: EditOptions, counts: EditCounts
) -> np.ndarray | None:
    """Returns the units that skip, every and number or subsample keep.

    `units` are the input streamlines of this batch; `counts` is updated. None
    means that every unit is kept.
    """
    if (
        options.skip is None
        and options.every is None
        and options.number is None
        and options.subsample is None
    ):
        return None
    skip_now = min(max((options.skip or 0) - counts.skipped, 0), len(units))
    counts.skipped += skip_now
    units = units[skip_now:]
    if options.every is not None:
        # The first unit of this batch kept by `every`, counting on from the
        # units after the skipped ones in earlier batches.
        first = (-counts.after_skip) % options.every
        counts.after_skip += len(units)
        units = units[first :: options.every]
    if options.number is not None:
        units = units[: max(options.number - counts.kept, 0)]
    elif options.subsample is not None:
        if len(units) < options.subsample:
            LarmorLogger.warning(
                "Asked for a subsample of %d streamlines, but only %d are left; "
                "all of them are kept.",
                options.subsample,
                len(units),
            )
        else:
            rng = np.random.default_rng(options.seed)
            picked = rng.choice(len(units), size=options.subsample, replace=False)
            units = units[np.sort(picked)]
    counts.kept += len(units)
    return units
