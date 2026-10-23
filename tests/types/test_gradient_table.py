import logging

import numpy as np
import pytest

from larmor.types import GradientTable

NAN = float("nan")

# Oblique, with every component distinct and non-zero, so a flip or permutation
# that touches the wrong component is caught.
G = (0.48, 0.60, 0.64)


def _unit(n: int, seed: int = 0) -> np.ndarray:
    vectors = np.random.default_rng(seed).normal(size=(n, 3))
    unit: np.ndarray = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return unit


def _table(bvals: list[float], **kwargs: float) -> GradientTable:
    return GradientTable(np.array(bvals), np.tile(G, (len(bvals), 1)), **kwargs)


# Construction


@pytest.mark.parametrize("bvals_shape", [(), (2, 1)], ids=["0D", "2D"])
def test_bvals_that_are_not_one_dimensional_are_rejected(
    bvals_shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        GradientTable(np.full(bvals_shape, 1000.0), np.ones((2, 3)))


@pytest.mark.parametrize(
    "bvecs_shape",
    [(2, 4), (3, 2), (1, 3), (6,)],
    ids=["4-columns", "transposed", "too-few-rows", "flat"],
)
def test_bvecs_that_are_not_n_by_3_are_rejected(
    bvecs_shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="shape"):
        GradientTable(np.full(2, 1000.0), np.ones(bvecs_shape))


def test_nan_and_infinite_bvals_are_rejected_naming_the_volumes() -> None:
    with pytest.raises(ValueError, match=r"\[1, 3\]"):
        GradientTable(np.array([1000.0, NAN, 1000.0, np.inf]), np.tile(G, (4, 1)))


def test_empty_table_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one volume"):
        GradientTable(np.zeros(0), np.zeros((0, 3)))


# -0.0 is not negative, so a b0 written as -0 is still accepted.
def test_negative_bvals_are_rejected_naming_the_volumes() -> None:
    with pytest.raises(ValueError, match=r"Volumes \[2\] have a negative"):
        GradientTable(np.array([-0.0, 1000.0, -1000.0]), np.tile(G, (3, 1)))


# Unlike NaN (no direction), infinity is corrupt: normalising it would give b = inf
# and a NaN direction. The b0 row catches a check limited to weighted volumes.
@pytest.mark.parametrize("b", [1000.0, 0.0], ids=["weighted", "b0"])
def test_infinite_bvec_component_is_rejected_naming_the_volume(b: float) -> None:
    bvecs = np.array([G, (np.inf, 0.0, 0.0)])
    with pytest.raises(ValueError, match=r"Volumes \[1\] have an infinite"):
        GradientTable(np.array([0.0, b]), bvecs)


# The partial NaN row catches a check written with all() instead of any(); the
# b just above the threshold catches a check that uses >= on the wrong side.
@pytest.mark.parametrize(
    ("row", "b"),
    [((0.0, 0.0, 0.0), 1000.0), ((NAN, NAN, NAN), 1000.0), ((NAN, 1.0, 0.0), 50.5)],
    ids=["zero", "nan", "partial-nan"],
)
def test_weighted_volume_without_direction_is_rejected_naming_it(
    row: tuple[float, float, float], b: float
) -> None:
    with pytest.raises(ValueError, match=r"Volumes \[1\]"):
        GradientTable(np.array([0.0, b]), np.array([G, row]))


@pytest.mark.parametrize(
    ("row", "b"),
    [((NAN, NAN, NAN), 50.0), ((NAN, 1.0, 0.0), 5.0), ((0.0, 0.0, 0.0), 50.0)],
    ids=["nan-at-threshold", "partial-nan", "zero-at-threshold"],
)
def test_b0_without_direction_becomes_zero_vector_at_b_zero(
    row: tuple[float, float, float], b: float
) -> None:
    # Normalising scales b by the squared length, 0 for a vector with no direction.
    table = GradientTable(np.array([b, 1000.0]), np.array([row, G]))
    np.testing.assert_array_equal(table.bvecs[0], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(table.bvals, [0.0, 1000.0])
    np.testing.assert_array_equal(table.b0_mask, [True, False])


def test_integer_inputs_become_float64() -> None:
    table = GradientTable(np.array([0, 1000]), np.array([[0, 0, 0], [0, 1, 0]]))
    assert table.bvals.dtype == np.float64
    assert table.bvecs.dtype == np.float64
    np.testing.assert_array_equal(table.bvals, [0.0, 1000.0])


def test_callers_arrays_are_not_changed() -> None:
    bvals = np.array([0.0, 2800.0])
    bvecs = np.array([[NAN, NAN, NAN], [0.3, 0.4, 0.0]])
    table = GradientTable(bvals, bvecs)
    # Flipping in place also catches a table that shares the caller's array.
    table.flip(0)
    np.testing.assert_array_equal(bvals, [0.0, 2800.0])
    np.testing.assert_array_equal(bvecs, [[NAN, NAN, NAN], [0.3, 0.4, 0.0]])


# Normalisation


@pytest.mark.parametrize(
    ("b", "vector", "expected_b", "expected_vector"),
    [
        (2800.0, (0.3, 0.4, 0.0), 700.0, (0.6, 0.8, 0.0)),
        (
            1000.0,
            (0.9993 * 0.6, 0.0, -0.9993 * 0.8),
            1000.0 * 0.9993**2,
            (0.6, 0.0, -0.8),
        ),
        (5.0, (0.0, -1.0, 0.0), 5.0, (0.0, -1.0, 0.0)),
    ],
    ids=["shortened", "rounded", "unit-b0"],
)
def test_b_is_scaled_by_squared_length_and_vector_made_unit_keeping_sign(
    b: float,
    vector: tuple[float, float, float],
    expected_b: float,
    expected_vector: tuple[float, float, float],
) -> None:
    table = GradientTable(np.array([b]), np.array([vector]))
    np.testing.assert_allclose(table.bvals, [expected_b], rtol=1e-12)
    np.testing.assert_allclose(table.bvecs, [expected_vector], atol=1e-12)


@pytest.mark.parametrize("length", [1e-3, 0.0999])
def test_weighted_volume_with_a_vector_shorter_than_0_1_is_rejected(
    length: float,
) -> None:
    with pytest.raises(ValueError, match=r"Volumes \[1\].*shorter than 0.1"):
        GradientTable(np.array([0.0, 1000.0]), np.array([G, np.array(G) * length]))


def test_short_vector_of_a_b0_volume_is_accepted() -> None:
    table = GradientTable(np.array([0.0, 1000.0]), np.array([np.array(G) * 1e-3, G]))
    np.testing.assert_array_equal(table.b0_mask, [True, False])


def test_weighted_volume_that_scaling_turns_into_a_b0_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="larmor"):
        table = GradientTable(
            np.array([0.0, 1000.0, 1000.0]), np.array([G, G, np.array(G) * 0.2])
        )
    np.testing.assert_array_equal(table.b0_mask, [True, False, True])
    assert "Volumes [2]" in caplog.text
    assert "b0" in caplog.text


def test_normalising_a_normalised_table_changes_nothing() -> None:
    lengths = np.random.default_rng(1).uniform(0.5, 1.0, size=50)
    bvals = np.random.default_rng(2).uniform(0.0, 3000.0, size=50)
    table = GradientTable(bvals, _unit(50) * lengths[:, None])
    again = GradientTable(table.bvals, table.bvecs)
    # Bit for bit: unit vectors are not renormalised, so a table written by Larmor and
    # read back is identical.
    np.testing.assert_array_equal(again.bvals, table.bvals)
    np.testing.assert_array_equal(again.bvecs, table.bvecs)


# b0 mask


def test_b0_mask_includes_the_threshold() -> None:
    table = _table([0.0, 49.9, 50.0, 50.1, 1000.0])
    np.testing.assert_array_equal(table.b0_mask, [True, True, True, False, False])


def test_b0_mask_uses_a_custom_threshold() -> None:
    table = _table([0.0, 10.0, 10.5, 50.0], b0_threshold=10.0)
    np.testing.assert_array_equal(table.b0_mask, [True, True, False, False])


# Shells


def test_shells_of_spread_bvals_with_interleaved_b0s() -> None:
    table = _table([0.0, 1000.0, 2995.0, 5.0, 1005.0, 3000.0, 995.0])
    np.testing.assert_array_equal(table.shell_index, [-1, 0, 1, -1, 0, 1, 0])
    np.testing.assert_allclose(table.shells, [1000.0, 2997.5])


def test_shells_are_ascending_when_volumes_are_descending() -> None:
    table = _table([3000.0, 2000.0, 1000.0])
    np.testing.assert_array_equal(table.shell_index, [2, 1, 0])
    np.testing.assert_allclose(table.shells, [1000.0, 2000.0, 3000.0])


def test_shells_encoded_in_vector_lengths_are_separated() -> None:
    short = np.sqrt(1.0 / 3.0) * np.array(G)
    table = GradientTable(np.full(4, 3000.0), np.array([G, short, G, short]))
    np.testing.assert_array_equal(table.shell_index, [1, 0, 1, 0])
    np.testing.assert_allclose(table.shells, [1000.0, 3000.0])


@pytest.mark.parametrize(("gap", "n_shells"), [(80.0, 1), (80.5, 2)])
def test_only_a_gap_larger_than_shell_epsilon_starts_a_shell(
    gap: float, n_shells: int
) -> None:
    assert len(_table([1000.0, 1000.0 + gap]).shells) == n_shells


def test_custom_shell_epsilon_splits_closer_bvals() -> None:
    table = _table([1000.0, 1050.0], shell_epsilon=30.0)
    np.testing.assert_array_equal(table.shell_index, [0, 1])


def test_table_of_only_b0s_has_no_shells() -> None:
    table = _table([0.0, 5.0, 50.0])
    np.testing.assert_array_equal(table.shell_index, [-1, -1, -1])
    assert table.shells.shape == (0,)


def test_wide_shell_is_reported_once(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="larmor"):
        table = _table(list(np.arange(100.0, 381.0, 20.0)))
        # Reading the shells again (as each model does) must not repeat it.
        assert len(table.shells) == 1
        _ = table.shells
    assert caplog.text.count("Shell 0") == 1


def test_normal_shells_are_not_reported(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="larmor"):
        table = _table([0.0, 1000.0, 2995.0, 5.0, 1005.0, 3000.0, 995.0])
        _ = table.shells
    assert caplog.text == ""


def test_spread_bvals_count_as_one_shell_each() -> None:
    assert _table([995.0, 1000.0, 1005.0, 1990.0, 2010.0]).count_shells() == 2


def test_b0s_never_count_as_a_shell() -> None:
    assert _table([0.0, 5.0, 50.0, 1000.0]).count_shells() == 1


def test_table_of_only_b0s_counts_no_shell() -> None:
    assert _table([0.0, 5.0, 50.0]).count_shells() == 0


@pytest.mark.parametrize(
    ("above", "n_shells"), [(0.0, 3), (300.0, 2), (999.0, 2), (1000.0, 1)]
)
def test_only_shells_above_the_threshold_count(above: float, n_shells: int) -> None:
    table = _table([0.0, 300.0, 995.0, 1005.0, 2000.0])
    assert table.count_shells(above=above) == n_shells


# b-matrix


def test_b_matrix_of_one_volume_by_hand() -> None:
    table = GradientTable(np.array([1000.0]), np.array([[0.6, 0.8, 0.0]]))
    np.testing.assert_allclose(
        table.b_matrix, [[360.0, 960.0, 0.0, 640.0, 0.0, 0.0]], atol=1e-9
    )


def test_b_matrix_makes_the_tensor_model_linear() -> None:
    # All six components non-zero, so a wrong order, a missing factor 2 or a
    # repeated component changes the result.
    rng = np.random.default_rng(3)
    a = rng.normal(size=(3, 3))
    d = (a + a.T) * 1e-3
    bvals = rng.uniform(500.0, 3000.0, size=20)
    g = _unit(20, seed=4)
    table = GradientTable(bvals, g)
    d6 = d[[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
    assert table.b_matrix.shape == (20, 6)
    np.testing.assert_allclose(
        table.b_matrix @ d6, bvals * np.einsum("ni,ij,nj->n", g, d, g), rtol=1e-12
    )


def test_b_matrix_equals_b_g_gt_of_the_values_as_read() -> None:
    # Non-unit vectors: MRIToolkit's b-matrix is b g g^T with g as written, so the
    # normalised table must give the same numbers.
    bvals = np.array([0.0, 1000.0, 2800.0, 3000.0])
    lengths = np.array([1.0, 0.9993, 0.5, 1.0])
    bvecs = _unit(4, seed=5) * lengths[:, None]
    table = GradientTable(bvals, bvecs)
    outer = bvals[:, None, None] * np.einsum("ni,nj->nij", bvecs, bvecs)
    rows, cols = np.triu_indices(3)  # xx, xy, xz, yy, yz, zz
    expected = outer[:, rows, cols] * np.where(rows == cols, 1.0, 2.0)
    np.testing.assert_allclose(table.b_matrix, expected, rtol=1e-12, atol=1e-9)


# Correcting a wrong table


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_flip_negates_one_component(axis: int) -> None:
    table = _table([0.0, 1000.0])
    table.flip(axis)
    expected = np.array(G)
    expected[axis] = -expected[axis]
    np.testing.assert_allclose(table.bvecs, [expected, expected])
    np.testing.assert_array_equal(table.bvals, [0.0, 1000.0])


def test_flip_twice_is_the_identity() -> None:
    table = GradientTable(np.full(5, 1000.0), _unit(5))
    before = table.bvecs.copy()
    table.flip(1)
    table.flip(1)
    np.testing.assert_array_equal(table.bvecs, before)


@pytest.mark.parametrize("axis", [3, -1])
def test_flip_of_an_invalid_axis_is_rejected(axis: int) -> None:
    with pytest.raises(ValueError):
        _table([1000.0]).flip(axis)


def test_permute_takes_new_component_j_from_old_order_j() -> None:
    # A cyclic order is not its own inverse, so an inverted permutation is caught.
    table = _table([0.0, 1000.0])
    table.permute((1, 2, 0))
    np.testing.assert_allclose(table.bvecs, [[G[1], G[2], G[0]]] * 2)
    np.testing.assert_array_equal(table.bvals, [0.0, 1000.0])


@pytest.mark.parametrize("order", [(0, 0, 1), (0, 1, 3), (0, 1)])
def test_permute_with_an_invalid_order_is_rejected(order: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        _table([1000.0]).permute(order)  # type: ignore[arg-type]  # wrong length on purpose
