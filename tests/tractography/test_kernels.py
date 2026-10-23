# The compiled tracking loop through _core.track_batch, on a 10x10x10 identity grid with
# 1 mm steps along x from (x, 5, 5), where every point and end is known by hand. The
# grid ends at voxel coordinate -0.5 and 9.5.
import math
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray
from reference_random import search_vector
from reference_tracking import NO_SAMPLING, Scene, track_reference

from larmor import _core
from larmor.tractography import EndStatus

Vector = NDArray[np.float64]

FOD = np.ones((10, 10, 10, 1), dtype=np.float32)
IDENTITY = np.eye(4)
X = np.array([1.0, 0.0, 0.0])

LV = EndStatus.LEFT_VOLUME
LM = EndStatus.LEFT_MASK
ML = EndStatus.MAX_LENGTH
NONE = EndStatus.NONE

SEARCH_REFUSES_PAST_X = 6.5


def _mask(lo: int, hi: int) -> NDArray[np.float32]:
    # Voxels lo..hi along x: inside from x = lo - 0.5 up to, not including, hi + 0.5.
    mask = np.zeros((10, 10, 10), dtype=np.float32)
    mask[lo : hi + 1] = 1.0
    return mask


@dataclass
class Batch:
    positions: NDArray[np.float32]
    counts: NDArray[np.int64]
    status: NDArray[np.uint8]
    # The directions followed: given, or found by the search.
    directions: Vector

    def points(self, row: int = 0) -> NDArray[np.float32]:
        return self.positions[row, : self.counts[row]]

    def ends(self, row: int = 0) -> tuple[EndStatus, EndStatus]:
        return EndStatus(self.status[row, 0]), EndStatus(self.status[row, 1])


def _run(
    seeds_x: list[float],
    max_steps: int = 100,
    bidirectional: bool = True,
    mask: NDArray[np.float32] | None = None,
    algorithm: _core.TestAlgorithm | None = None,
    spare_rows: int = 2,
    directions: Vector | None = None,
    searched: bool = False,
) -> Batch:
    n = len(seeds_x)
    if directions is None and not searched:
        directions = np.tile(X, (n, 1))
    positions = np.full((n + spare_rows, max_steps + 1, 3), np.nan, dtype=np.float32)
    seeds = np.column_stack([seeds_x, np.full((n, 2), 5.0)])
    counts, status, followed, stats = _core.track_batch(
        FOD,
        IDENTITY,
        None if mask is None else (mask, IDENTITY),
        seeds,
        directions,
        np.arange(n, dtype=np.int64),
        0,
        1.0,
        max_steps,
        bidirectional,
        algorithm if algorithm is not None else _core.TestAlgorithm("straight"),
        positions,
        1,
    )
    assert stats == NO_SAMPLING
    # Also catches a write past the end of a row, which lands in the next one.
    assert np.isnan(positions[n:]).all()
    for row in range(n):
        assert np.isnan(positions[row, counts[row] :]).all()
    return Batch(positions, counts, status, followed)


def _along_x(xs: list[float] | NDArray[np.float64]) -> NDArray[np.float32]:
    xs = np.asarray(xs, dtype=np.float64)
    return np.column_stack([xs, np.full((len(xs), 2), 5.0)]).astype(np.float32)


# Grid edges and directions


def test_bidirectional_line_ends_at_the_last_points_on_the_grid() -> None:
    # The points at x = 10 and -1 are off the grid, so they are not written.
    out = _run([5.0])
    np.testing.assert_array_equal(out.points(), _along_x(np.arange(9.0, -1.0, -1.0)))
    assert out.counts[0] == 10
    assert out.ends() == (LV, LV)


@pytest.mark.parametrize(
    ("seed_x", "first", "last"),
    [(9.0, 9.0, 0.0), (9.5, 9.5, -0.5), (-0.5, 9.5, -0.5), (0.0, 9.0, 0.0)],
)
def test_seed_near_the_edge_ends_at_the_last_points_on_the_grid(
    seed_x: float, first: float, last: float
) -> None:
    out = _run([seed_x])
    np.testing.assert_array_equal(
        out.points(), _along_x(np.arange(first, last - 0.5, -1.0))
    )
    assert out.ends() == (LV, LV)


def test_unidirectional_starts_at_the_seed_and_leaves_its_end_untracked() -> None:
    out = _run([5.0], bidirectional=False)
    np.testing.assert_array_equal(out.points(), _along_x(np.arange(5.0, 10.0)))
    assert out.ends() == (NONE, LV)


# Maximum number of steps, shared by both halves


@pytest.mark.parametrize(
    ("max_steps", "bidirectional", "first", "last", "ends"),
    [
        (4, True, 9.0, 5.0, (ML, ML)),
        (4, False, 5.0, 9.0, (NONE, ML)),
        # The first half leaves the grid at its 5th step, which adds no point, so
        # the second half has one step left.
        (5, True, 9.0, 4.0, (LV, ML)),
        (5, False, 5.0, 9.0, (NONE, LV)),
        (7, True, 9.0, 2.0, (LV, ML)),
        (11, True, 9.0, 0.0, (LV, LV)),
    ],
)
def test_max_steps_counts_the_steps_of_the_whole_streamline(
    max_steps: int,
    bidirectional: bool,
    first: float,
    last: float,
    ends: tuple[EndStatus, EndStatus],
) -> None:
    out = _run([5.0], max_steps=max_steps, bidirectional=bidirectional)
    step = 1.0 if last > first else -1.0
    xs = np.arange(first, last + step / 2, step)
    np.testing.assert_array_equal(out.points(), _along_x(xs))
    assert out.counts[0] <= max_steps + 1
    assert out.ends() == ends


# Mask


def test_mask_ends_at_the_last_point_inside_it_at_both_ends() -> None:
    out = _run([5.0], mask=_mask(3, 7))
    np.testing.assert_array_equal(out.points(), _along_x(np.arange(7.0, 2.0, -1.0)))
    assert out.ends() == (LM, LM)


@pytest.mark.parametrize(
    ("mask", "max_steps", "last", "end"),
    [
        # x = 10 is off the FOD grid and off the mask's, and is checked before the
        # length.
        (_mask(3, 9), 100, 9.0, LV),
        (_mask(3, 9), 5, 9.0, LV),
        # x = 8 is outside the mask at the last step there is room for.
        (_mask(3, 7), 3, 7.0, LM),
    ],
)
def test_grid_then_mask_then_length_decide_the_end(
    mask: NDArray[np.float32], max_steps: int, last: float, end: EndStatus
) -> None:
    out = _run([5.0], max_steps=max_steps, bidirectional=False, mask=mask)
    np.testing.assert_array_equal(out.points(), _along_x(np.arange(5.0, last + 0.5)))
    assert out.ends() == (NONE, end)


def test_mask_on_its_own_grid_is_read_in_world_coordinates() -> None:
    # Mask voxels of 2 mm from x = -1: voxels 2..3 cover x = 2.0 to 6.0 (edges at
    # voxel coordinates 1.5 and 3.5). x = 6 is at 3.5, which rounds to voxel 4, so
    # it is the first point outside and is not written.
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [-1.0, 0.0, 0.0]
    data = np.zeros((6, 6, 6), dtype=np.float32)
    data[2:4] = 1.0
    positions = np.full((1, 101, 3), np.nan, dtype=np.float32)
    counts, status, _, _ = _core.track_batch(
        FOD,
        IDENTITY,
        (data, np.linalg.inv(affine)),
        np.array([[5.0, 5.0, 5.0]]),
        X[None],
        np.zeros(1, dtype=np.int64),
        0,
        1.0,
        100,
        False,
        _core.TestAlgorithm("straight"),
        positions,
        1,
    )
    np.testing.assert_array_equal(positions[0, : counts[0]], _along_x([5.0]))
    assert list(status[0]) == [NONE, LM]


# Seeds that do not start


@pytest.mark.parametrize(
    ("seed_x", "mask"),
    [
        (9.6, None),
        (-0.6, None),
        (1.0, _mask(3, 7)),
        (8.0, _mask(3, 7)),
    ],
)
@pytest.mark.parametrize("searched", [False, True])
def test_seed_off_the_grid_or_outside_the_mask_does_not_start(
    seed_x: float, mask: NDArray[np.float32] | None, searched: bool
) -> None:
    # The search finds a direction everywhere, yet the seed is checked as before.
    out = _run([5.0, seed_x, 5.0], mask=mask, searched=searched)
    assert out.counts[1] == 0
    assert np.isnan(out.positions[1]).all()
    assert out.ends(1) == (NONE, NONE)
    assert out.counts[0] > 0 and out.counts[2] > 0


# The direction step


@pytest.mark.parametrize(
    "status", [EndStatus.LOW_FOD, EndStatus.SHARP_TURN, EndStatus.NO_DIRECTION]
)
@pytest.mark.parametrize("bidirectional", [True, False])
def test_refused_step_adds_no_point_and_records_its_status(
    status: EndStatus, bidirectional: bool
) -> None:
    refuse = _core.TestAlgorithm("refuse", int(status))
    out = _run([5.0], bidirectional=bidirectional, algorithm=refuse)
    if bidirectional:
        np.testing.assert_array_equal(out.points(), _along_x([6.0, 5.0, 4.0]))
        assert out.ends() == (status, status)
    else:
        np.testing.assert_array_equal(out.points(), _along_x([5.0, 6.0]))
        assert out.ends() == (NONE, status)


@pytest.mark.parametrize("status", list(EndStatus)[1:])
def test_every_stop_an_algorithm_returns_is_recorded_as_given(
    status: EndStatus,
) -> None:
    out = _run([5.0], algorithm=_core.TestAlgorithm("refuse", int(status)))
    assert out.ends() == (status, status)


def test_first_step_follows_the_seed_then_the_algorithm_its_previous_step() -> None:
    out = _run(
        [5.0],
        max_steps=4,
        bidirectional=False,
        algorithm=_core.TestAlgorithm("quarter_turn"),
    )
    expected = [[5, 5, 5], [6, 5, 5], [6, 6, 5], [5, 6, 5], [5, 5, 5]]
    np.testing.assert_array_equal(out.points(), np.array(expected, dtype=np.float32))
    assert out.ends() == (NONE, ML)


@pytest.mark.parametrize(
    ("max_steps", "bidirectional"), [(100, True), (100, False), (4, True), (7, True)]
)
def test_one_random_counter_runs_through_both_halves(
    max_steps: int, bidirectional: bool
) -> None:
    # The second half of a wander walk continues the first half's draws: the
    # reference, which counts its draws through both halves, gives the same points.
    scene = Scene(
        fod=FOD,
        fod_affine=IDENTITY,
        mask=None,
        mask_affine=IDENTITY,
        step_size=1.0,
        max_steps=max_steps,
        bidirectional=bidirectional,
        step="wander",
        run_seed=0,
    )
    seed = np.array([5.0, 5.0, 5.0])
    ref = track_reference(scene, seed, X, 0)
    assert ref is not None
    out = _run(
        [5.0],
        max_steps=max_steps,
        bidirectional=bidirectional,
        algorithm=_core.TestAlgorithm("wander"),
    )
    np.testing.assert_array_equal(out.points(), ref.points.astype(np.float32))
    assert out.ends() == ref.ends


# Batches


def test_each_seed_fills_its_own_row_and_later_rows_are_untouched() -> None:
    out = _run([5.0, 9.0, 2.0], max_steps=6, spare_rows=3)
    assert list(out.counts) == [7, 7, 7]
    np.testing.assert_array_equal(out.points(0), _along_x(np.arange(9.0, 2.0, -1)))
    np.testing.assert_array_equal(out.points(1), _along_x(np.arange(9.0, 2.0, -1)))
    np.testing.assert_array_equal(out.points(2), _along_x(np.arange(8.0, 1.0, -1)))
    assert [out.ends(i) for i in range(3)] == [(LV, ML), (LV, ML), (ML, ML)]


def test_empty_batch_returns_empty_results_and_touches_nothing() -> None:
    positions = np.full((2, 5, 3), np.nan, dtype=np.float32)
    counts, status, followed, stats = _core.track_batch(
        FOD,
        IDENTITY,
        None,
        np.empty((0, 3)),
        None,
        np.empty(0, dtype=np.int64),
        0,
        1.0,
        4,
        True,
        _core.TestAlgorithm("straight"),
        positions,
        1,
    )
    assert counts.shape == (0,) and status.shape == (0, 2) and followed.shape == (0, 3)
    assert stats == NO_SAMPLING
    assert np.isnan(positions).all()


# Seeds without directions


# Points are float32: ~5e-7 at coordinates under 10.
POINT_TOL = 2e-6


@pytest.mark.parametrize("bidirectional", [True, False])
def test_seed_without_direction_starts_along_the_direction_found(
    bidirectional: bool,
) -> None:
    out = _run([5.0, 5.0, 5.0], max_steps=3, bidirectional=bidirectional, searched=True)
    seed = np.array([5.0, 5.0, 5.0])
    for i in range(3):
        found = search_vector(0, i)
        np.testing.assert_array_equal(out.directions[i], found)
        # Bidirectionally the first half takes every step and is reversed.
        k = [3, 2, 1, 0] if bidirectional else [0, 1, 2, 3]
        expected = seed + np.outer(k, found)
        np.testing.assert_allclose(out.points(i), expected, rtol=0, atol=POINT_TOL)
    assert not np.allclose(out.directions[0], out.directions[1])


def test_seed_where_the_search_fails_does_not_start() -> None:
    algorithm = _core.TestAlgorithm(
        "straight", search_refuses_past_x=SEARCH_REFUSES_PAST_X
    )
    out = _run([5.0, 8.0, 2.0], algorithm=algorithm, searched=True)
    assert out.counts[1] == 0
    assert np.isnan(out.positions[1]).all()
    assert out.ends(1) == (NONE, NONE)
    np.testing.assert_array_equal(out.directions[1], np.zeros(3))
    for i in (0, 2):
        assert out.counts[i] > 0
        np.testing.assert_array_equal(out.directions[i], search_vector(0, i))


def test_steps_draw_the_same_numbers_whether_the_direction_was_given_or_found() -> None:
    # The search draws from its own counter range: tracking again with the
    # directions it found gives the same random walk.
    seeds_x = [5.0, 4.0, 6.0, 5.5]
    wander = _core.TestAlgorithm("wander")
    found = _run(seeds_x, max_steps=8, algorithm=wander, searched=True)
    given = _run(seeds_x, max_steps=8, algorithm=wander, directions=found.directions)
    np.testing.assert_array_equal(found.positions, given.positions)
    np.testing.assert_array_equal(found.counts, given.counts)
    np.testing.assert_array_equal(found.status, given.status)
    for i in range(len(seeds_x)):
        np.testing.assert_array_equal(found.directions[i], search_vector(0, i))
    # The walk does turn: a straight line would hide a shifted counter.
    assert not np.allclose(
        found.points(0)[1] - found.points(0)[0], found.points(0)[2] - found.points(0)[1]
    )


def test_seeds_with_directions_are_never_searched() -> None:
    # Past x = 6.5 the search fails: were it called, the seed at x = 9 would not
    # start and its direction would change.
    algorithm = _core.TestAlgorithm(
        "straight", search_refuses_past_x=SEARCH_REFUSES_PAST_X
    )
    out = _run([5.0, 9.0, 2.0], max_steps=6, algorithm=algorithm)
    assert list(out.counts) == [7, 7, 7]
    np.testing.assert_array_equal(out.directions, np.tile(X, (3, 1)))
    np.testing.assert_array_equal(out.points(1), _along_x(np.arange(9.0, 2.0, -1)))


# Argument checks of the binding


def _call(**changes: object) -> None:
    args: dict[str, object] = {
        "fod": FOD,
        "fod_inv_affine": IDENTITY,
        "mask": None,
        "seeds": np.full((2, 3), 5.0),
        "directions": np.tile(X, (2, 1)),
        "indices": np.arange(2, dtype=np.int64),
        "run_seed": 0,
        "step_size": 1.0,
        "max_steps": 4,
        "bidirectional": True,
        "algorithm": _core.TestAlgorithm("straight"),
        "positions": np.zeros((2, 5, 3), dtype=np.float32),
        "n_threads": 1,
    }
    args.update(changes)
    _core.track_batch(*args.values())  # type: ignore[arg-type]


def test_valid_call_runs() -> None:
    _call()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"seeds": np.full((2, 2), 5.0)}, "seeds and directions"),
        ({"directions": np.tile(X, (3, 1))}, "seeds and directions"),
        ({"directions": np.ones((2, 4))}, "seeds and directions"),
        ({"indices": np.arange(3, dtype=np.int64)}, "seeds and directions"),
        ({"indices": np.array([0, -1], dtype=np.int64)}, "non-negative"),
        ({"positions": np.zeros((1, 5, 3), dtype=np.float32)}, "positions must be"),
        ({"positions": np.zeros((2, 4, 3), dtype=np.float32)}, "positions must be"),
        ({"positions": np.zeros((2, 6, 3), dtype=np.float32)}, "positions must be"),
        ({"positions": np.zeros((2, 5, 4), dtype=np.float32)}, "positions must be"),
        (
            {"positions": np.zeros((2, 3, 5), dtype=np.float32).transpose(0, 2, 1)},
            "C-contiguous",
        ),
        ({"fod": np.ones((10, 10, 10, 2), np.float32)[..., :1]}, "C-contiguous"),
        ({"fod_inv_affine": np.eye(3)}, r"\(4, 4\)"),
        ({"mask": (_mask(0, 9), np.eye(3))}, r"\(4, 4\)"),
        ({"mask": (np.asfortranarray(_mask(0, 9)), IDENTITY)}, "C-contiguous"),
        ({"step_size": 0.0}, "positive step size"),
        ({"step_size": -1.0}, "positive step size"),
        ({"step_size": math.nan}, "positive step size"),
        ({"step_size": math.inf}, "positive step size"),
        ({"max_steps": 0, "positions": np.zeros((2, 1, 3), np.float32)}, "max_steps"),
        ({"n_threads": 0}, "n_threads"),
    ],
)
def test_bad_arguments_are_refused(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _call(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"mask": _mask(0, 9)},
        {"mask": (_mask(0, 9),)},
        {"fod": np.ones((10, 10, 10), dtype=np.float32)},
        {"fod": np.ones((10, 10, 10, 1), dtype=np.float64)},
        {"positions": np.zeros((2, 5, 3), dtype=np.float64)},
        {"indices": np.arange(2, dtype=np.int32)},
        {"run_seed": -1},
        {"run_seed": 2**64},
        {"algorithm": "straight"},
    ],
)
def test_wrong_types_are_refused(changes: dict[str, object]) -> None:
    with pytest.raises((TypeError, OverflowError, ValueError)):
        _call(**changes)


def test_mask_passed_as_a_tuple_is_used() -> None:
    # Outside the mask everywhere: no seed starts.
    positions = np.zeros((2, 5, 3), dtype=np.float32)
    empty = np.zeros((10, 10, 10), dtype=np.float32)
    _call(mask=(empty, IDENTITY), positions=positions)
    assert not positions.any()
    _call(mask=(_mask(0, 9), IDENTITY), positions=positions)
    assert positions.any()


@pytest.mark.parametrize(
    ("step", "stop"),
    [
        ("sideways", None),
        ("refuse", None),
        ("refuse", 0),
        ("refuse", 7),
        ("straight", 4),
        ("wander", 1),
    ],
)
def test_test_algorithm_refuses_unknown_steps_and_stops(
    step: str, stop: int | None
) -> None:
    with pytest.raises(ValueError, match="stop"):
        _core.TestAlgorithm(step, stop)


@pytest.mark.parametrize(
    ("cutoff", "max_angle", "message"),
    [
        (-0.1, 60.0, "cutoff"),
        (math.nan, 60.0, "cutoff"),
        (math.inf, 60.0, "cutoff"),
        (0.1, 0.0, "maximum angle"),
        (0.1, -10.0, "maximum angle"),
        (0.1, 180.5, "maximum angle"),
        (0.1, math.nan, "maximum angle"),
    ],
)
def test_det_params_refuse_values_out_of_range(
    cutoff: float, max_angle: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _core.DetParams(cutoff, max_angle)


@pytest.mark.parametrize(("cutoff", "max_angle"), [(0.0, 180.0), (0.1, 1e-9)])
def test_det_params_accept_the_edges_of_their_ranges(
    cutoff: float, max_angle: float
) -> None:
    _core.DetParams(cutoff, max_angle)


@pytest.mark.parametrize("num_coeffs", [0, 2, 5, 14, 16, 44, 46, 253])
def test_det_refuses_a_fod_that_is_not_a_full_even_series(num_coeffs: int) -> None:
    # 1, 6, 15, 28, 45 ... up to 231 at lmax 20 are accepted.
    fod = np.ones((3, 3, 3, num_coeffs), dtype=np.float32)
    with pytest.raises(ValueError, match="lmax|coefficients"):
        _call(fod=fod, algorithm=_core.DetParams(0.1, 60.0))


@pytest.mark.parametrize("num_coeffs", [1, 6, 45, 231])
def test_det_accepts_a_full_even_series(num_coeffs: int) -> None:
    fod = np.zeros((10, 10, 10, num_coeffs), dtype=np.float32)
    _call(fod=fod, algorithm=_core.DetParams(0.0, 60.0))
