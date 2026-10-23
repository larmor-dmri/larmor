import logging
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from larmor.io import load_tractogram, load_volume, read_streamline_weights
from larmor.tractography import (
    Region,
    Sphere,
    concatenate_tractograms,
    crop,
    editing,
    passes_in_order,
    passes_through,
    select_length,
    select_weight,
    stays_within,
    streamline_lengths,
)
from larmor.tractography import regions as regions_module
from larmor.types import Grid, Tractogram, Volume

DATA = Path(__file__).parents[1] / "data" / "editing"

type Regions = dict[str, Region]
type Case = Callable[[Tractogram, Regions], np.ndarray]


@pytest.fixture(scope="module")
def tracks() -> Tractogram:
    return load_tractogram(DATA / "tracks.tck")


@pytest.fixture(scope="module")
def lengths_ref() -> dict[str, np.ndarray]:
    with np.load(DATA / "lengths_ref.npz") as data:
        return dict(data)


def make(
    streamlines: list[list[list[float]]],
    reference: Grid | None = None,
    with_data: bool = True,
) -> Tractogram:
    sizes = [len(s) for s in streamlines]
    points = np.array([p for s in streamlines for p in s], dtype=np.float32)
    num_points = sum(sizes)
    return Tractogram(
        points=points.reshape(-1, 3),
        offsets=np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
        point_data={"fa": np.arange(num_points, dtype=np.float32)} if with_data else {},
        streamline_data=(
            {"weight": np.arange(len(sizes), dtype=np.float32)} if with_data else {}
        ),
        reference=reference,
    )


def grid(shift: float = 0.0) -> Grid:
    affine = np.eye(4)
    affine[0, 3] = shift
    return Grid((4, 4, 4), affine)


def test_lengths_match_tckstats(
    tracks: Tractogram, lengths_ref: dict[str, np.ndarray]
) -> None:
    # tckstats prints 6 significant digits.
    np.testing.assert_allclose(
        streamline_lengths(tracks), lengths_ref["lengths"], rtol=1e-5
    )


@pytest.mark.parametrize("chunk", [2, 3])
def test_chunked_lengths_match_one_chunk(
    tracks: Tractogram, chunk: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    short = make([[], [[0, 0, 0]], [[0, 0, 0], [3, 4, 0], [3, 4, 12]], [], [[1, 1, 1]]])
    whole = [streamline_lengths(t) for t in (tracks, short)]
    monkeypatch.setattr(editing, "_LENGTH_CHUNK", chunk)
    for t, expected in zip((tracks, short), whole, strict=True):
        np.testing.assert_allclose(streamline_lengths(t), expected, rtol=1e-12)


def test_streamlines_of_zero_or_one_point_have_length_zero() -> None:
    # A 3-4-5 step, then 1, 0, 2 and (last) 0 points; the empty last streamline
    # starts past the end of the points.
    t = make(
        [
            [[0, 0, 0], [3, 4, 0], [3, 4, 12]],
            [[9, 9, 9]],
            [],
            [[1, 1, 1], [1, 1, 3]],
            [],
        ]
    )
    np.testing.assert_array_equal(streamline_lengths(t), [17.0, 0.0, 0.0, 2.0, 0.0])


@pytest.mark.parametrize("case", range(4))
def test_length_selection_matches_tckedit(
    tracks: Tractogram, lengths_ref: dict[str, np.ndarray], case: int
) -> None:
    low, high = lengths_ref["length_limits"][case]
    selected = select_length(
        tracks,
        min_length=0.0 if math.isnan(low) else float(low),
        max_length=math.inf if math.isnan(high) else float(high),
    )
    np.testing.assert_array_equal(
        np.flatnonzero(selected), lengths_ref[f"length_kept_{case}"]
    )


def test_length_limits_are_inclusive() -> None:
    t = make([[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [2, 0, 0]], [[0, 0, 0], [3, 0, 0]]])
    np.testing.assert_array_equal(
        select_length(t, min_length=1.0, max_length=2.0), [True, True, False]
    )


def test_joined_halves_select_as_tckedit_on_the_two_files(
    tracks: Tractogram, lengths_ref: dict[str, np.ndarray]
) -> None:
    split = int(lengths_ref["join_split"])
    joined = concatenate_tractograms([tracks[:split], tracks[split:]])
    np.testing.assert_array_equal(joined.points, tracks.points)
    np.testing.assert_array_equal(joined.offsets, tracks.offsets)
    low, high = lengths_ref["join_limits"]
    selected = select_length(joined, min_length=float(low), max_length=float(high))
    np.testing.assert_array_equal(np.flatnonzero(selected), lengths_ref["join_kept"])


def test_join_keeps_common_data_and_drops_the_rest_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    a = make([[[0, 0, 0], [1, 0, 0]]])
    b = make([[[5, 0, 0]], [[6, 0, 0], [7, 0, 0]]])
    a.point_data["only_a"] = np.zeros(2, dtype=np.float32)
    a.streamline_data["rgb"] = np.zeros((1, 3), dtype=np.float32)
    b.streamline_data["rgb"] = np.zeros((2, 4), dtype=np.float32)
    a.streamline_data["label"] = np.zeros(1, dtype=np.int32)
    b.streamline_data["label"] = np.zeros(2, dtype=np.float32)
    b.streamline_data["weight"] = b.streamline_data["weight"].astype(np.float64)
    with caplog.at_level(logging.WARNING, logger="larmor"):
        joined = concatenate_tractograms([a, b])
    assert "Dropped" in caplog.text
    assert set(joined.point_data) == {"fa"}
    assert set(joined.streamline_data) == {"weight"}
    assert joined.streamline_data["weight"].dtype == np.float64
    np.testing.assert_array_equal(joined.point_data["fa"], [0, 1, 0, 1, 2])
    np.testing.assert_array_equal(joined.streamline_data["weight"], [0, 0, 1])
    np.testing.assert_array_equal(joined.offsets, [0, 2, 3, 5])


@pytest.mark.parametrize(
    ("references", "kept", "warns"),
    [
        ((grid(), grid()), True, False),
        ((None, None), False, False),
        ((grid(), None), False, True),
        ((None, grid()), False, True),
        ((grid(), grid(5.0)), False, True),
    ],
    ids=["same", "none", "first-only", "second-only", "different"],
)
def test_join_reference(
    references: tuple[Grid | None, Grid | None],
    kept: bool,
    warns: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    inputs = [make([[[0, 0, 0]]], reference=r) for r in references]
    with caplog.at_level(logging.WARNING, logger="larmor"):
        joined = concatenate_tractograms(inputs)
    assert (joined.reference is references[0]) if kept else joined.reference is None
    assert ("reference grid" in caplog.text) == warns


def test_join_ignores_the_keys_of_empty_inputs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    a = make([[[0, 0, 0], [1, 0, 0]]], reference=grid())
    with caplog.at_level(logging.WARNING, logger="larmor"):
        joined = concatenate_tractograms(
            [make([], with_data=False), a, make([], with_data=False)]
        )
    assert not caplog.text
    assert joined.reference is a.reference
    np.testing.assert_array_equal(joined.point_data["fa"], a.point_data["fa"])
    np.testing.assert_array_equal(
        joined.streamline_data["weight"], a.streamline_data["weight"]
    )


def test_join_of_empty_tractograms_keeps_data_keys() -> None:
    joined = concatenate_tractograms([make([]), make([])])
    assert joined.num_streamlines == 0
    assert set(joined.point_data) == {"fa"}
    assert set(joined.streamline_data) == {"weight"}


def test_join_without_tractograms_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        concatenate_tractograms([])


def test_weight_selection_matches_tckedit(
    tracks: Tractogram, lengths_ref: dict[str, np.ndarray]
) -> None:
    weights = read_streamline_weights(DATA / "weights.txt")
    weighted = Tractogram(
        tracks.points, tracks.offsets, streamline_data={"weight": weights}
    )
    low, high = lengths_ref["weight_limits"]
    selected = select_weight(weighted, min_weight=float(low), max_weight=float(high))
    np.testing.assert_array_equal(np.flatnonzero(selected), lengths_ref["weight_kept"])
    np.testing.assert_array_equal(
        weighted[selected].streamline_data["weight"],
        read_streamline_weights(DATA / "weights_selected.txt"),
    )


@pytest.mark.parametrize(
    "select",
    [
        lambda t: select_length(t, min_length=2.0, max_length=1.0),
        lambda t: select_length(t, min_length=math.nan),
        lambda t: select_weight(t),
    ],
    ids=["min-above-max", "nan-limit", "weight-not-one-column"],
)
def test_bad_limits_or_weights_raise(
    select: Callable[[Tractogram], np.ndarray],
) -> None:
    t = make([[[0, 0, 0], [1, 0, 0]], [[5, 0, 0]]])
    t.streamline_data["weight"] = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        select(t)


def test_weight_selection_without_weights_raises() -> None:
    with pytest.raises(ValueError, match="weight"):
        select_weight(make([[[0, 0, 0]]], with_data=False))


@pytest.mark.parametrize(
    "streamlines",
    [[], [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [3, 0, 0]]]],
    ids=["empty-tractogram", "empty-selection"],
)
def test_empty_results_keep_data_keys(streamlines: list[list[list[float]]]) -> None:
    t = make(streamlines)
    assert streamline_lengths(t).shape == (len(streamlines),)
    selected = select_length(t, min_length=1.5, max_length=2.5) & select_weight(
        t, min_weight=0.0, max_weight=10.0
    )
    kept = t[selected]
    assert kept.num_streamlines == 0
    assert set(kept.point_data) == {"fa"}
    assert set(kept.streamline_data) == {"weight"}


@pytest.fixture(scope="module")
def regions_ref() -> dict[str, np.ndarray]:
    with np.load(DATA / "regions_ref.npz") as data:
        return {name: data[name] for name in data.files}


@pytest.fixture(scope="module")
def regions(regions_ref: dict[str, np.ndarray]) -> Regions:
    found: Regions = {
        name.removeprefix("sphere_"): Sphere(values[:3], float(values[3]))
        for name, values in regions_ref.items()
        if name.startswith("sphere_")
    }
    found["shell"] = load_volume(DATA / "shell_mask.nii.gz")
    found["float"] = load_volume(DATA / "float_mask.nii.gz")
    return found


def _ordered(names: str, ends_only: bool = False) -> Case:
    return lambda t, r: passes_in_order(
        t, [r[n] for n in names.split()], ends_only=ends_only
    )


# The selection tckedit made for each case of make_regions_ref.py.
SELECTIONS: dict[str, Case] = {
    "include_sphere": lambda t, r: passes_through(t, r["a"]),
    "include_mask": lambda t, r: passes_through(t, r["shell"]),
    "include_float_mask": lambda t, r: passes_through(t, r["float"]),
    "include_two": lambda t, r: passes_through(t, r["a"]) & passes_through(t, r["c"]),
    "ordered_two": _ordered("a c"),
    "ordered_two_reversed": _ordered("c a"),
    "ordered_three": _ordered("a b c"),
    "ordered_three_swapped": _ordered("a c b"),
    "ordered_overlap": _ordered("b big"),
    "ordered_overlap_reversed": _ordered("big b"),
    "ordered_mask": _ordered("a shell"),
    "exclude_sphere": lambda t, r: ~passes_through(t, r["away"]),
    "exclude_mask": lambda t, r: ~passes_through(t, r["shell"]),
    "include_exclude": lambda t, r: (
        passes_through(t, r["a"]) & ~passes_through(t, r["b"])
    ),
    "ends_include": lambda t, r: passes_through(t, r["start"], ends_only=True),
    "ends_include_two": lambda t, r: (
        passes_through(t, r["start"], ends_only=True)
        & passes_through(t, r["end"], ends_only=True)
    ),
    "ends_include_mask": lambda t, r: passes_through(t, r["shell"], ends_only=True),
    "ends_exclude": lambda t, r: ~passes_through(t, r["end"], ends_only=True),
    "ends_ordered": _ordered("start end", ends_only=True),
    "inverse_include": lambda t, r: ~passes_through(t, r["a"]),
    "inverse_exclude": lambda t, r: ~~passes_through(t, r["away"]),
}

# name: (regions, keep_inside)
CROPS: dict[str, tuple[list[str], bool]] = {
    "crop_shell": (["shell"], True),
    "crop_float": (["float"], True),
    "crop_spheres": (["a", "c"], True),
    "crop_shell_inverse": (["shell"], False),
}


@pytest.mark.parametrize("case", SELECTIONS)
def test_selection_matches_tckedit(
    case: str, tracks: Tractogram, regions: Regions, regions_ref: dict[str, np.ndarray]
) -> None:
    selected = SELECTIONS[case](tracks, regions)
    assert selected.shape == (tracks.num_streamlines,)
    np.testing.assert_array_equal(np.flatnonzero(selected), regions_ref[case])


@pytest.mark.parametrize("case", CROPS)
def test_crop_matches_tckedit(
    case: str, tracks: Tractogram, regions: Regions, regions_ref: dict[str, np.ndarray]
) -> None:
    names, keep_inside = CROPS[case]
    expected = load_tractogram(DATA / f"{case}.tck")
    # The indices as data show which input streamline and points each piece holds.
    indexed = Tractogram(
        points=tracks.points,
        offsets=tracks.offsets,
        point_data={"index": np.arange(tracks.num_points)},
        streamline_data={"source": np.arange(tracks.num_streamlines)},
    )
    pieces = crop(indexed, [regions[n] for n in names], keep_inside=keep_inside)
    np.testing.assert_array_equal(pieces.offsets, expected.offsets)
    np.testing.assert_array_equal(pieces.points, expected.points)
    np.testing.assert_array_equal(pieces.streamline_data["source"], regions_ref[case])
    np.testing.assert_array_equal(
        tracks.points[pieces.point_data["index"]], pieces.points
    )


@pytest.mark.parametrize("case", ["ordered_three", "ordered_mask", "crop_shell"])
def test_chunked_region_tests_give_the_same_result(
    case: str,
    tracks: Tractogram,
    regions: Regions,
    regions_ref: dict[str, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # About one streamline's points: chunks of several streamlines, of one, and
    # streamlines longer than a chunk.
    monkeypatch.setattr(regions_module, "CHUNK_POINTS", 40)
    if case in CROPS:
        names, keep_inside = CROPS[case]
        pieces = crop(tracks, [regions[n] for n in names], keep_inside=keep_inside)
        expected = load_tractogram(DATA / f"{case}.tck")
        np.testing.assert_array_equal(pieces.offsets, expected.offsets)
        np.testing.assert_array_equal(pieces.points, expected.points)
    else:
        selected = SELECTIONS[case](tracks, regions)
        np.testing.assert_array_equal(np.flatnonzero(selected), regions_ref[case])


def _ordered_regions() -> list[Region]:
    # Region 0 is a mask, so the sphere test that preselects the streamlines is on
    # region 1 and does not hide a streamline that never starts in region 0.
    box = Volume(
        data=np.ones((3, 3, 3), dtype=np.uint8),
        affine=np.array(
            [[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]], dtype=float
        ),
    )
    return [
        box,
        Sphere(np.array([10.0, 0, 0]), 1.0),
        Sphere(np.array([20.0, 0, 0]), 1.0),
    ]


def _through(labels: list[int]) -> Tractogram:
    # Label k is a point in region k, -1 a point in none.
    x = [10.0 * k if k >= 0 else 50.0 for k in labels]
    points = np.zeros((len(x), 3), dtype=np.float32)
    points[:, 0] = x
    return Tractogram(points=points, offsets=np.array([0, len(x)]))


@pytest.mark.parametrize(
    ("labels", "passes"),
    [
        ([0, 1, 2], True),
        ([0, 0, 1, 2, 2], True),
        ([-1, 0, -1, 1, -1, 2, -1], True),
        ([0, -1, 0, 1, -1, 1, 2], True),
        ([0, 2], False),
        ([0, 1, 0, 1, 2], False),
        ([0, 1, 2, 1], False),
        ([1, 2], False),
        ([0, 1], False),
        ([2, 1, 0], False),
    ],
)
def test_ordered_include_follows_the_stored_points(
    labels: list[int], passes: bool
) -> None:
    result = passes_in_order(_through(labels), _ordered_regions())
    np.testing.assert_array_equal(result, [passes])


@pytest.mark.parametrize(
    ("labels", "passes"), [([0, 1, 2, 3], True), ([0, 1, 3], False)]
)
def test_ordered_include_never_skips_a_region(labels: list[int], passes: bool) -> None:
    # The preselecting sphere is region 1, which [0, 1, 3] goes through, so only the
    # order test can reject the skip of region 2.
    regions = [*_ordered_regions(), Sphere(np.array([30.0, 0, 0]), 1.0)]
    result = passes_in_order(_through(labels), regions)
    np.testing.assert_array_equal(result, [passes])


@pytest.mark.parametrize(
    ("labels", "passes"),
    [([0, 1], True), ([0, -1, 1], True), ([1, 0], False), ([1, 1], False)],
)
def test_ordered_include_with_ends_only_tests_the_two_endpoints(
    labels: list[int], passes: bool
) -> None:
    result = passes_in_order(_through(labels), _ordered_regions()[:2], ends_only=True)
    np.testing.assert_array_equal(result, [passes])


@pytest.mark.parametrize(
    ("x", "passes"), [([1.5, 4.0], True), ([0.0, 4.0, 1.5], False)]
)
def test_point_in_overlapping_regions_belongs_to_the_earlier_one(
    x: list[float], passes: bool
) -> None:
    # The spheres overlap for x in [1, 2]; x = 1.5 is in region 0, not 1.
    regions: list[Region] = [
        Sphere(np.zeros(3), 2.0),
        Sphere(np.array([3.0, 0, 0]), 2.0),
    ]
    points = np.zeros((len(x), 3), dtype=np.float32)
    points[:, 0] = x
    t = Tractogram(points=points, offsets=np.array([0, len(x)]))
    np.testing.assert_array_equal(passes_in_order(t, regions), [passes])


def test_ends_only_with_more_than_two_ordered_regions_raises() -> None:
    with pytest.raises(ValueError, match="ends_only"):
        passes_in_order(_through([0, 1, 2]), _ordered_regions(), ends_only=True)


def test_no_regions_raise() -> None:
    with pytest.raises(ValueError, match="region"):
        passes_in_order(_through([0]), [])
    with pytest.raises(ValueError, match="region"):
        crop(_through([0]), [])


def test_crop_keeps_each_piece_s_point_and_streamline_data() -> None:
    reference = load_volume(DATA / "shell_mask.nii.gz").grid
    t = Tractogram(
        points=_through([0, 0, -1, 0, 0, 0, -1, 0]).points,
        offsets=np.array([0, 8]),
        point_data={"fa": np.arange(8, dtype=np.float32)},
        streamline_data={"weight": np.array([0.5], dtype=np.float32)},
        reference=reference,
    )
    pieces = crop(t, Sphere(np.zeros(3), 1.0))
    np.testing.assert_array_equal(pieces.offsets, [0, 2, 5])
    np.testing.assert_array_equal(pieces.point_data["fa"], [0, 1, 3, 4, 5])
    np.testing.assert_array_equal(pieces.streamline_data["weight"], [0.5, 0.5])
    assert pieces.reference is reference

    outside = crop(t, Sphere(np.zeros(3), 1.0), keep_inside=False)
    # The single outside points at 2 and 6 are too short to be pieces.
    assert outside.num_streamlines == 0


def test_empty_tractogram_gives_empty_results() -> None:
    t = Tractogram(points=np.zeros((0, 3)), offsets=np.array([0]))
    sphere = Sphere(np.zeros(3), 1.0)
    for ends_only in (False, True):
        assert passes_through(t, sphere, ends_only=ends_only).shape == (0,)
        assert passes_in_order(t, [sphere], ends_only=ends_only).shape == (0,)
    assert crop(t, sphere).num_streamlines == 0


@pytest.mark.parametrize("ends_only", [False, True])
def test_empty_streamlines_never_pass_and_one_point_ones_can(ends_only: bool) -> None:
    # Streamlines: empty, one point inside, two points inside, empty, one outside.
    points = np.array([[0, 0, 0], [0, 0, 0], [0.5, 0, 0], [9, 0, 0]], dtype=np.float32)
    t = Tractogram(points=points, offsets=np.array([0, 0, 1, 3, 3, 4]))
    sphere = Sphere(np.zeros(3), 1.0)
    expected = [False, True, True, False, False]
    np.testing.assert_array_equal(
        passes_through(t, sphere, ends_only=ends_only), expected
    )
    np.testing.assert_array_equal(
        passes_in_order(t, [sphere], ends_only=ends_only), expected
    )


def test_crop_drops_empty_and_one_point_streamlines() -> None:
    points = np.array([[0, 0, 0], [0, 0, 0], [0.5, 0, 0]], dtype=np.float32)
    t = Tractogram(
        points=points,
        offsets=np.array([0, 0, 1, 3, 3]),
        streamline_data={"id": np.arange(4)},
    )
    pieces = crop(t, Sphere(np.zeros(3), 1.0))
    np.testing.assert_array_equal(pieces.offsets, [0, 2])
    np.testing.assert_array_equal(pieces.streamline_data["id"], [2])


@pytest.mark.parametrize(
    ("ends_only", "expected"),
    [(False, [True, False, False]), (True, [True, True, False])],
)
def test_stays_within_needs_every_point_or_both_ends_inside(
    ends_only: bool, expected: list[bool]
) -> None:
    # Streamlines: all inside; inside but for its middle point; empty.
    points = np.array(
        [[0, 0, 0], [0.5, 0, 0], [0, 0, 0], [5, 0, 0], [0.5, 0, 0]], dtype=np.float32
    )
    t = Tractogram(points=points, offsets=np.array([0, 2, 5, 5]))
    sphere = Sphere(np.zeros(3), 1.0)
    np.testing.assert_array_equal(
        stays_within(t, sphere, ends_only=ends_only), expected
    )
