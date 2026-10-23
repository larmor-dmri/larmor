import warnings

import numpy as np
import pytest

from larmor.simulation import Protocol, PulseSequence, create_gradient_table
from larmor.types import GradientTable

GTAB = GradientTable(np.array([0.0, 1000.0]), np.array([[0, 0, 0], [1.0, 0, 0]]))


def protocol(**changes: object) -> Protocol:
    values: dict[str, object] = {
        "small_delta_ms": 15.0,
        "big_delta_ms": 30.0,
        "te_ms": 80.0,
        "tr_ms": 3000.0,
    } | changes
    return Protocol(GTAB, **values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"small_delta_ms": 0.0}, "delta must be positive"),
        ({"small_delta_ms": 40.0, "big_delta_ms": 30.0}, "must not exceed Delta"),
        ({"te_ms": 45.0}, "shorter than the echo time"),
        ({"tr_ms": 70.0}, "TR"),
        ({"te_ms": [80.0, 3500.0]}, "TR"),
    ],
)
def test_each_timing_check_rejects_its_case(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        protocol(**changes)


def test_gradient_echo_also_ends_its_diffusion_block_before_the_echo() -> None:
    with pytest.raises(ValueError, match="shorter than the echo time"):
        protocol(te_ms=40.0, sequence=PulseSequence.GRADIENT_ECHO)


def test_unknown_sequence_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown pulse sequence"):
        protocol(sequence="stimulated_echo")


def test_b_values_in_the_wrong_unit_warn() -> None:
    small = GradientTable(np.array([0.0, 3.0]), np.array([[0, 0, 0], [1.0, 0, 0]]))
    with pytest.warns(UserWarning, match="wrong unit"):
        Protocol(small, 15.0, 30.0, 80.0, 3000.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        protocol()


def test_round_trip_through_a_dict() -> None:
    table = GradientTable(
        np.array([20.0, 1000.0]),
        np.array([[0, 0, 0], [1.0, 0, 0]]),
        b0_threshold=30.0,
        shell_epsilon=60.0,
    )
    original = Protocol(
        table, 15.0, 30.0, [80.0, 120.0], 3000.0, PulseSequence.GRADIENT_ECHO
    )
    copy = Protocol.from_dict(original.to_dict())
    assert copy.to_dict() == original.to_dict()
    assert copy.sequence is PulseSequence.GRADIENT_ECHO
    assert copy.gradients.b0_threshold == 30.0
    assert copy.gradients.shell_epsilon == 60.0


def test_protocol_keeps_a_read_only_copy_of_the_table() -> None:
    table = GradientTable(np.array([0.0, 1000.0]), np.array([[0, 0, 0], [1.0, 0, 0]]))
    p = Protocol(table, 15.0, 30.0, 80.0, 3000.0)
    table.flip(0)
    assert p.gradients.bvecs[1, 0] == 1.0
    with pytest.raises(ValueError, match="read-only"):
        p.gradients.flip(0)


def test_gradient_table_has_unit_shells_without_antipodal_pairs() -> None:
    table = create_gradient_table([(1000, 30), (3000, 60)], num_b0=3, seed=4)
    assert table.bvals.tolist() == [0.0] * 3 + [1000.0] * 30 + [3000.0] * 60
    for b, count in ((1000, 30), (3000, 60)):
        dirs = table.bvecs[table.bvals == b]
        np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-12)
        cosines = np.abs(dirs @ dirs.T)[np.triu_indices(count, k=1)]
        # Well spread: no two directions within 10 degrees, up to sign.
        assert cosines.max() < np.cos(np.radians(10))
    np.testing.assert_array_equal(table.bvecs[:3], 0.0)


def test_gradient_table_is_deterministic_for_a_seed() -> None:
    first = create_gradient_table([(1000, 20)], seed=7)
    second = create_gradient_table([(1000, 20)], seed=7)
    np.testing.assert_array_equal(first.bvecs, second.bvecs)
