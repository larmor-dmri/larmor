import warnings
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray
from reference_random import (
    GAMMA,
    draw_vector,
    mix,
    search_vector,
    stream,
    uniform,
    uniform_array,
)
from reference_tracking import Scene, track_core, track_reference

U64 = 2**64
TOP = U64 - 1

# SplitMix64 started at state 1234567: the published reference outputs.
SPLITMIX_STATE = 1234567
SPLITMIX_OUTPUTS = [
    6457827717110365317,
    3203168211198807973,
    9817491932198370423,
    4593380528125082431,
    16408922859458223821,
]

# Indices and counters at the edges of the int64 and uint64 ranges, where a signed
# type or a missed wrap-around would show.
INDICES = [0, 1, 2, 1000, 2**31, 2**63 - 1, 2**63, 2**63 + 1, TOP - 1, TOP]
KS = [0, 1, 2, 999, 2**32, 2**63 - 1, 2**63, TOP - 1, TOP]


def _scalar(seed: int, index: int, k: int) -> float:
    return uniform(stream(seed, index), k)


def _uint64(values: list[int]) -> NDArray[np.uint64]:
    return np.array(values, dtype=np.uint64)


def test_mix_gives_the_splitmix64_reference_outputs() -> None:
    inputs = [
        (SPLITMIX_STATE + (k + 1) * int(GAMMA)) % U64
        for k in range(len(SPLITMIX_OUTPUTS))
    ]
    assert mix(_uint64(inputs)).tolist() == SPLITMIX_OUTPUTS


def test_numpy_twin_gives_the_numbers_the_cpp_tests_pin() -> None:
    # The values asserted in cpp/tests/tracking/test_rng.cpp: both sides agree on them.
    assert stream(42, 7) == 15985126342852841771
    out = uniform_array(42, [7, 7, 7], _uint64([0, 1, 2**63]))
    assert out.tolist() == [0.8940814753778417, 0.5331080354362024, 0.45263234128120544]


@pytest.mark.parametrize("seed", [0, 42, TOP])
def test_scalar_and_array_draws_give_the_same_numbers(seed: int) -> None:
    expected = np.array([[_scalar(seed, i, k) for k in KS] for i in INDICES])
    out = uniform_array(seed, _uint64(INDICES)[:, None], _uint64(KS)[None, :])
    assert out.shape == (len(INDICES), len(KS))
    assert np.array_equal(out, expected)


def test_equal_shaped_indices_and_counters_pair_elementwise() -> None:
    expected = [_scalar(42, 7, 0), _scalar(42, 7, 1), _scalar(42, 8, 0)]
    out = uniform_array(42, _uint64([7, 7, 8]), _uint64([0, 1, 0]))
    assert np.array_equal(out, expected)


def test_lists_and_scalars_are_accepted() -> None:
    from_lists = uniform_array(42, [7, 7, 8], [0, 1, 0])
    from_arrays = uniform_array(42, _uint64([7, 7, 8]), _uint64([0, 1, 0]))
    assert np.array_equal(from_lists, from_arrays)
    scalar = uniform_array(TOP, TOP, 3)
    assert np.ndim(scalar) == 0
    assert float(scalar) == _scalar(TOP, TOP, 3)


def test_a_draw_does_not_depend_on_the_rest_of_the_batch() -> None:
    grid = uniform_array(
        5, np.arange(20, dtype=np.uint64)[:, None], np.arange(30, dtype=np.uint64)
    )
    alone = uniform_array(5, _uint64([13]), _uint64([21]))
    assert np.array_equal(grid[13, 21], alone[0])


def test_values_are_uniform_in_the_unit_interval() -> None:
    num_streamlines, n_draws = 1000, 1000
    out = uniform_array(
        2024,
        np.arange(num_streamlines, dtype=np.uint64)[:, None],
        np.arange(n_draws, dtype=np.uint64),
    ).ravel()
    assert out.min() >= 0.0
    assert out.max() < 1.0
    # 1e6 draws: the standard errors are about 3e-4 for the mean, 8e-5 for the
    # variance and 300 counts per decile, so these bounds sit beyond 10 sigma.
    assert out.mean() == pytest.approx(0.5, abs=3e-3)
    assert out.var() == pytest.approx(1 / 12, abs=1e-3)
    counts, _ = np.histogram(out, bins=10, range=(0.0, 1.0))
    assert np.all(np.abs(counts - out.size / 10) < 0.01 * out.size)


@pytest.mark.parametrize(("seed_a", "seed_b"), [(0, 1), (42, 43), (0, TOP)])
def test_different_seeds_give_different_sequences(seed_a: int, seed_b: int) -> None:
    ks = np.arange(1000, dtype=np.uint64)
    a = uniform_array(seed_a, _uint64([0]), ks)
    b = uniform_array(seed_b, _uint64([0]), ks)
    assert not np.intersect1d(a, b).size


@pytest.mark.parametrize("index", [0, 1, 2**63 - 1, TOP - 1])
def test_neighbouring_streamlines_share_no_numbers(index: int) -> None:
    # Disjoint value sets rule out one streamline being a shifted copy of the next.
    ks = np.arange(1000, dtype=np.uint64)
    a = uniform_array(42, _uint64([index]), ks)
    b = uniform_array(42, _uint64([index + 1]), ks)
    assert np.unique(a).size == ks.size
    assert not np.intersect1d(a, b).size


@pytest.mark.parametrize(
    ("seed", "indices", "ks"),
    [
        (TOP, TOP, TOP),
        (0, 0, 0),
        (TOP, _uint64(INDICES)[:, None], _uint64(KS)),
        (TOP, np.array(TOP, dtype=np.uint64), np.array(TOP, dtype=np.uint64)),
    ],
)
def test_draws_raise_no_overflow_warning(
    seed: int, indices: NDArray[np.uint64] | int, ks: NDArray[np.uint64] | int
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        uniform_array(seed, indices, ks)
        stream(TOP, TOP)
        uniform(TOP, TOP)
        draw_vector(np.array([TOP, TOP - 1], dtype=np.uint64))


# Random unit vectors


def _state(key: int, k: int) -> NDArray[np.uint64]:
    return np.array([key, k], dtype=np.uint64)


@pytest.mark.parametrize("k", [0, 1, 2**63, TOP - 1])
def test_draw_vector_is_a_unit_vector_and_takes_two_numbers(k: int) -> None:
    state = _state(12345, k)
    v = draw_vector(state)
    assert v.shape == (3,)
    assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-15)
    assert state.tolist() == [12345, (k + 2) % U64]


def test_draw_vector_depends_only_on_the_state() -> None:
    first, again = _state(99, 2**63), _state(99, 2**63)
    np.testing.assert_array_equal(draw_vector(first), draw_vector(again))
    # A second draw is the draw of the state two counters on.
    np.testing.assert_array_equal(
        draw_vector(first), draw_vector(_state(99, 2**63 + 2))
    )
    assert not np.array_equal(draw_vector(_state(99, 0)), draw_vector(_state(100, 0)))


def test_draw_vector_is_uniform_on_the_sphere() -> None:
    # For 20000 uniform unit vectors each component has mean 0 (sd 0.004) and
    # second moment 1/3 (sd 0.002); each octant holds 1/8 (sd 0.0023).
    n = 20_000
    state = _state(stream(SPLITMIX_STATE, 0), 0)
    v = np.array([draw_vector(state) for _ in range(n)])
    np.testing.assert_allclose(v.mean(axis=0), 0.0, atol=0.02)
    np.testing.assert_allclose((v**2).mean(axis=0), 1 / 3, atol=0.01)
    octant = (v > 0) @ np.array([1, 2, 4])
    np.testing.assert_allclose(np.bincount(octant, minlength=8) / n, 1 / 8, atol=0.01)


# The compiled generator against the twin, through the tracking loop


def _open_space(step: str, max_steps: int = 50) -> Scene:
    # Two voxels of 1 m: no walk of 50 steps of 1 mm from the middle leaves them.
    return Scene(
        fod=np.ones((2, 2, 2, 1), dtype=np.float32),
        fod_affine=np.diag([1000.0, 1000.0, 1000.0, 1.0]),
        mask=None,
        mask_affine=np.eye(4),
        step_size=1.0,
        max_steps=max_steps,
        bidirectional=False,
        step=step,
    )


MIDDLE = np.array([[500.0, 500.0, 500.0]])


@pytest.mark.parametrize("run_seed", [0, 42, 2**63, TOP])
@pytest.mark.parametrize("index", [0, 7, 2**31, 2**63 - 1])
def test_compiled_walk_draws_the_numbers_of_the_numpy_twin(
    run_seed: int, index: int
) -> None:
    # A wander step adds three draws to the direction: 50 steps show any draw that
    # differs, keyed or counted another way.
    scene = replace(_open_space("wander"), run_seed=run_seed)
    direction = np.array([[0.0, 0.6, 0.8]])
    indices = np.array([index], dtype=np.int64)
    core = track_core(scene, MIDDLE, direction, indices)[0]
    ref = track_reference(scene, MIDDLE[0], direction[0], index)
    assert core is not None and ref is not None
    assert ref.draws == 3 * 49
    assert len(core.points) == 51
    np.testing.assert_array_equal(core.points, ref.points.astype(np.float32))


@pytest.mark.parametrize("run_seed", [0, 42, TOP])
def test_compiled_search_draws_the_first_vector_of_the_search_counter(
    run_seed: int,
) -> None:
    scene = replace(_open_space("straight", max_steps=1), run_seed=run_seed)
    indices = np.array([0, 1, 2, 1000, 2**31, 2**63 - 1], dtype=np.int64)
    seeds = np.repeat(MIDDLE, len(indices), axis=0)
    out = track_core(scene, seeds, None, indices)
    for streamline, index in zip(out, indices, strict=True):
        assert streamline is not None
        expected = search_vector(run_seed, int(index))
        np.testing.assert_array_equal(streamline.direction, expected)
