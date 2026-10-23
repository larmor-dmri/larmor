import numpy as np
import pytest

from larmor.types import Grid, Tractogram

# Streamlines of different lengths, with every point value distinct, so a wrong
# offset, a wrong gather or a selection in the wrong order gives other values.
LENGTHS = [3, 2, 4, 1]
OFFSETS = np.array([0, 3, 5, 9, 10])
POINTS = np.arange(30, dtype=np.float32).reshape(10, 3)
FA = np.arange(10, dtype=np.float64) / 10.0
LABEL = np.array([10, 20, 30, 40])
REFERENCE = Grid((2, 3, 4), np.diag([2.0, 2.0, 3.0, 1.0]))


def _tractogram() -> Tractogram:
    return Tractogram(
        points=POINTS.copy(),
        offsets=OFFSETS.copy(),
        point_data={"fa": FA.copy()},
        streamline_data={"label": LABEL.copy()},
        reference=REFERENCE,
    )


def _assert_selects(selection: Tractogram, indices: list[int]) -> None:
    """`selection` holds the streamlines `indices` of `_tractogram()`, in that order."""
    lengths = [LENGTHS[i] for i in indices]
    np.testing.assert_array_equal(selection.offsets, np.cumsum([0, *lengths]))
    assert selection.offsets.dtype == np.int64
    expected_points = [POINTS[OFFSETS[i] : OFFSETS[i + 1]] for i in indices]
    expected_fa = [FA[OFFSETS[i] : OFFSETS[i + 1]] for i in indices]
    np.testing.assert_array_equal(
        selection.points, np.concatenate([np.empty((0, 3)), *expected_points])
    )
    np.testing.assert_array_equal(
        selection.point_data["fa"], np.concatenate([np.empty(0), *expected_fa])
    )
    np.testing.assert_array_equal(selection.streamline_data["label"], LABEL[indices])
    assert selection.reference is REFERENCE


# Construction


def test_valid_tractogram_has_s_streamlines_and_p_points() -> None:
    tractogram = _tractogram()
    assert len(tractogram) == tractogram.num_streamlines == 4
    assert tractogram.num_points == 10


@pytest.mark.parametrize(
    "points", [np.zeros((10, 2)), np.zeros(30), np.zeros((10, 3, 1))]
)
def test_points_that_are_not_p_by_3_are_rejected(points: np.ndarray) -> None:
    with pytest.raises(ValueError, match="points"):
        Tractogram(points=points, offsets=OFFSETS)


@pytest.mark.parametrize("offsets", [np.array([]), OFFSETS.reshape(1, 5)])
def test_offsets_that_are_not_1d_and_non_empty_are_rejected(
    offsets: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="offsets"):
        Tractogram(points=POINTS, offsets=offsets)


@pytest.mark.parametrize(
    ("offsets", "match"),
    [
        ([1, 3, 5, 9, 10], "start at 0"),
        ([0, 5, 3, 9, 10], "never decrease"),
        ([0, 3, 5, 9], "last offset"),
        ([0, 3, 5, 9, 11], "last offset"),
    ],
    ids=["not-from-0", "decreasing", "last-below-P", "last-above-P"],
)
def test_inconsistent_offsets_are_rejected(offsets: list[int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Tractogram(points=POINTS, offsets=np.array(offsets))


def test_point_data_with_wrong_first_axis_is_rejected() -> None:
    # 4 entries is S, not P: catches a check against the wrong count.
    with pytest.raises(ValueError, match="point_data"):
        Tractogram(points=POINTS, offsets=OFFSETS, point_data={"fa": np.zeros(4)})


def test_streamline_data_with_wrong_first_axis_is_rejected() -> None:
    # 10 entries is P, not S.
    with pytest.raises(ValueError, match="streamline_data"):
        Tractogram(
            points=POINTS, offsets=OFFSETS, streamline_data={"label": np.zeros(10)}
        )


# A list is a natural way to pass values; it must work in selections, not only
# pass construction (a list cannot be indexed by the gather array).
def test_data_given_as_lists_become_arrays_and_can_be_selected() -> None:
    tractogram = Tractogram(
        points=POINTS,
        offsets=OFFSETS,
        point_data={"fa": FA.tolist()},
        streamline_data={"label": LABEL.tolist()},
        reference=REFERENCE,
    )
    assert isinstance(tractogram.point_data["fa"], np.ndarray)
    assert isinstance(tractogram.streamline_data["label"], np.ndarray)
    _assert_selects(tractogram[[2, 0]], [2, 0])


def test_scalar_data_is_rejected() -> None:
    scalar = {"x": np.float64(1.0)}
    with pytest.raises(ValueError, match="point_data"):
        Tractogram(points=POINTS, offsets=OFFSETS, point_data=scalar)  # type: ignore[arg-type]  # a scalar on purpose
    with pytest.raises(ValueError, match="streamline_data"):
        Tractogram(points=POINTS, offsets=OFFSETS, streamline_data=scalar)  # type: ignore[arg-type]  # a scalar on purpose


# Casting would truncate them to [0, 2, 5, 9, 10], which passes every other check.
def test_float_offsets_are_rejected() -> None:
    with pytest.raises(ValueError, match="integers"):
        Tractogram(points=POINTS, offsets=np.array([0.0, 2.7, 5.2, 9.9, 10.0]))


def test_tractograms_compare_by_identity() -> None:
    tractogram = _tractogram()
    assert tractogram == tractogram
    assert tractogram != _tractogram()


def test_integer_points_become_float32_and_int32_offsets_become_int64() -> None:
    tractogram = Tractogram(
        points=POINTS.astype(np.int64), offsets=OFFSETS.astype(np.int32)
    )
    assert tractogram.points.dtype == np.float32
    assert tractogram.offsets.dtype == np.int64
    np.testing.assert_array_equal(tractogram.points, POINTS)
    np.testing.assert_array_equal(tractogram.offsets, OFFSETS)


def test_empty_tractogram_is_valid() -> None:
    tractogram = Tractogram(points=np.zeros((0, 3)), offsets=np.array([0]))
    assert len(tractogram) == 0
    assert list(tractogram) == []


def test_streamline_without_points_is_valid() -> None:
    tractogram = Tractogram(points=POINTS[:5], offsets=np.array([0, 3, 3, 5]))
    assert len(tractogram) == 3
    assert tractogram[1].shape == (0, 3)
    np.testing.assert_array_equal(tractogram[2], POINTS[3:5])


# Integer indexing


@pytest.mark.parametrize(
    ("key", "s"), [(0, 0), (1, 1), (3, 3), (-1, 3), (-4, 0), (np.int64(2), 2)]
)
def test_integer_gives_that_streamlines_points(key: int, s: int) -> None:
    np.testing.assert_array_equal(
        _tractogram()[key], POINTS[OFFSETS[s] : OFFSETS[s + 1]]
    )


def test_integer_gives_a_view_into_points() -> None:
    tractogram = _tractogram()
    assert np.shares_memory(tractogram[2], tractogram.points)


@pytest.mark.parametrize("key", [4, -5])
def test_integer_out_of_range_is_rejected(key: int) -> None:
    with pytest.raises(IndexError):
        _tractogram()[key]


@pytest.mark.parametrize("key", [True, np.True_, 1.5, "a"])
def test_bool_float_and_string_keys_are_rejected(key: object) -> None:
    with pytest.raises(IndexError):
        _tractogram()[key]  # type: ignore[call-overload]


# Selection


@pytest.mark.parametrize(
    ("key", "indices"),
    [
        (slice(None), [0, 1, 2, 3]),
        (slice(1, 3), [1, 2]),
        (slice(None, None, 2), [0, 2]),
        (slice(None, None, -1), [3, 2, 1, 0]),
        (slice(2, 100), [2, 3]),
    ],
    ids=["all", "range", "step", "reversed", "past-the-end"],
)
def test_slice_selects_those_streamlines(key: slice, indices: list[int]) -> None:
    _assert_selects(_tractogram()[key], indices)


@pytest.mark.parametrize(
    "indices",
    [[2, 0, 3], [1, 1], [-1, 0], [3, 2, 1, 0]],
    ids=["unsorted", "repeated", "negative", "reversed"],
)
def test_list_of_indices_selects_in_that_order(indices: list[int]) -> None:
    expected = [i % 4 for i in indices]
    _assert_selects(_tractogram()[indices], expected)
    _assert_selects(_tractogram()[np.array(indices)], expected)


def test_range_and_tuple_select_like_a_list() -> None:
    _assert_selects(_tractogram()[range(1, 3)], [1, 2])
    _assert_selects(_tractogram()[(3, 0)], [3, 0])


def test_boolean_mask_selects_where_true() -> None:
    tractogram = _tractogram()
    _assert_selects(tractogram[tractogram.streamline_data["label"] > 15], [1, 2, 3])
    _assert_selects(tractogram[[True, False, True, False]], [0, 2])


@pytest.mark.parametrize(
    "key",
    [[], np.array([], dtype=np.int64), np.zeros(4, dtype=bool)],
    ids=["empty-list", "empty-array", "all-false-mask"],
)
def test_selecting_nothing_gives_an_empty_tractogram(key: list[int]) -> None:
    selection = _tractogram()[key]
    assert len(selection) == 0
    _assert_selects(selection, [])


@pytest.mark.parametrize(
    "key",
    [
        np.ones(3, dtype=bool),
        np.ones(5, dtype=bool),
        [0, 4],
        [-5],
        # -8 + 4 is still negative and would silently pick a streamline from the end.
        [-8],
        np.array([[0, 1]]),
        np.array([0.0, 1.0]),
    ],
    ids=[
        "short-mask",
        "long-mask",
        "too-large",
        "too-negative",
        "far-negative",
        "2D",
        "float",
    ],
)
def test_invalid_selection_is_rejected(key: np.ndarray) -> None:
    with pytest.raises(IndexError):
        _tractogram()[key]


def test_selection_leaves_the_original_unchanged() -> None:
    tractogram = _tractogram()
    selection = tractogram[[2, 0]]
    selection.points[:] = -1.0
    selection.point_data["fa"][:] = -1.0
    selection.streamline_data["label"][:] = -1
    np.testing.assert_array_equal(tractogram.points, POINTS)
    np.testing.assert_array_equal(tractogram.offsets, OFFSETS)
    np.testing.assert_array_equal(tractogram.point_data["fa"], FA)
    np.testing.assert_array_equal(tractogram.streamline_data["label"], LABEL)


# len and iteration


def test_iteration_gives_each_streamline_in_order() -> None:
    streamlines = list(_tractogram())
    assert len(streamlines) == 4
    for s, points in enumerate(streamlines):
        np.testing.assert_array_equal(points, POINTS[OFFSETS[s] : OFFSETS[s + 1]])


# subsample


@pytest.mark.parametrize("n", [0, 2, 4])
def test_subsample_keeps_n_distinct_streamlines_in_file_order(n: int) -> None:
    subset = _tractogram().subsample(n, seed=3)
    assert len(subset) == n
    # Labels are increasing in the original, so file order means increasing here,
    # and the labels identify which streamlines were chosen.
    labels = subset.streamline_data["label"]
    assert list(labels) == sorted(set(labels))
    _assert_selects(subset, [int(np.flatnonzero(LABEL == x)[0]) for x in labels])


def test_subsample_with_the_same_seed_gives_the_same_streamlines() -> None:
    tractogram = _tractogram()
    first = tractogram.subsample(2, seed=7)
    second = tractogram.subsample(2, seed=7)
    np.testing.assert_array_equal(first.points, second.points)
    np.testing.assert_array_equal(first.offsets, second.offsets)


def test_subsample_with_different_seeds_gives_different_streamlines() -> None:
    # Seeds 0 and 1 happen to choose different pairs out of the 6 possible.
    tractogram = _tractogram()
    labels_0 = tractogram.subsample(2, seed=0).streamline_data["label"]
    labels_1 = tractogram.subsample(2, seed=1).streamline_data["label"]
    assert list(labels_0) != list(labels_1)


@pytest.mark.parametrize("n", [5, -1])
def test_subsample_count_outside_0_to_s_is_rejected(n: int) -> None:
    with pytest.raises(ValueError):
        _tractogram().subsample(n, seed=0)
