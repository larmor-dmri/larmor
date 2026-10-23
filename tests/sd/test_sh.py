import itertools
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from larmor.sd import convert_sh_basis, fit_sh, sh_layout
from larmor.sd.sh import BASES, build_basis_matrix, get_sh_indices
from larmor.sd.sphere import get_theta_phi

REF = np.load(Path(__file__).parents[1] / "data" / "sh_tournier_ref.npz")


def test_tournier_matches_mrtrix() -> None:
    """Every basis function up to lmax 8 against MRtrix's sh2amp
    (see data/make_sh_ref.py)."""
    theta, phi = get_theta_phi(REF["dirs"])
    basis = build_basis_matrix(theta, phi, lmax=int(REF["lmax"]), basis="tournier")
    np.testing.assert_allclose(basis, REF["basis"], atol=1e-6)  # sh2amp writes float32


def test_leemans_is_tournier_in_the_explore_dti_frame() -> None:
    """MRIToolkit converts ExploreDTI (EDTI) SH to MRtrix by evaluating the same
    functions at (-y, x, z) (MRTTrack.SHFitMatrix): a leemans FOD at direction R @ g
    must equal a tournier FOD at g, with R the 90 degree rotation about z."""
    dirs = REF["dirs"]
    rotated = np.column_stack([-dirs[:, 1], dirs[:, 0], dirs[:, 2]])

    leemans = build_basis_matrix(*get_theta_phi(rotated), lmax=8, basis="leemans")
    tournier = build_basis_matrix(*get_theta_phi(dirs), lmax=8, basis="tournier")
    np.testing.assert_allclose(leemans, tournier, atol=1e-12)


# Conversion between bases

PAIRS = list(itertools.permutations(BASES, 2))
# (lmax, full): every even-only size and full sizes with odd degrees and a
# degree high enough to have several orders of each parity.
SIZES = [(0, False), (2, False), (4, False), (8, False), (16, False)]
SIZES += [(1, True), (3, True), (8, True)]


def _random_dirs(n: int, seed: int = 0) -> npt.NDArray[np.float64]:
    dirs = np.random.default_rng(seed).normal(size=(n, 3))
    unit: npt.NDArray[np.float64] = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    return unit


def _n_coeffs(lmax: int, full: bool) -> int:
    return len(get_sh_indices(lmax, full)[0])


@pytest.mark.parametrize(("lmax", "full"), SIZES)
@pytest.mark.parametrize(("src", "dst"), PAIRS)
def test_converted_coefficients_give_the_same_amplitudes(
    src: str, dst: str, lmax: int, full: bool
) -> None:
    theta, phi = get_theta_phi(_random_dirs(300))
    coeffs = np.random.default_rng(1).normal(size=_n_coeffs(lmax, full))

    converted = convert_sh_basis(coeffs, src, dst)

    before = build_basis_matrix(theta, phi, lmax, src, full) @ coeffs
    after = build_basis_matrix(theta, phi, lmax, dst, full) @ converted
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-12 * len(coeffs))


@pytest.mark.parametrize(("lmax", "full"), [(8, False), (3, True), (8, True)])
@pytest.mark.parametrize(("src", "dst"), PAIRS)
def test_converting_there_and_back_returns_the_coefficients(
    src: str, dst: str, lmax: int, full: bool
) -> None:
    coeffs = np.random.default_rng(2).normal(size=_n_coeffs(lmax, full))

    back = convert_sh_basis(convert_sh_basis(coeffs, src, dst), dst, src)

    # A permutation with sign flips: exact, no rounding.
    np.testing.assert_array_equal(back, coeffs)


@pytest.mark.parametrize("basis", BASES)
def test_same_basis_returns_an_equal_copy(basis: str) -> None:
    coeffs = np.random.default_rng(3).normal(size=15)

    converted = convert_sh_basis(coeffs, basis, basis)

    np.testing.assert_array_equal(converted, coeffs)
    assert not np.shares_memory(converted, coeffs)


@pytest.mark.parametrize("leading", [(), (7,), (2, 3, 4)])
def test_any_leading_shape_converts_each_coefficient_vector(
    leading: tuple[int, ...],
) -> None:
    coeffs = np.random.default_rng(4).normal(size=(*leading, 15))

    converted = convert_sh_basis(coeffs, "leemans", "descoteaux")

    assert converted.shape == coeffs.shape
    flat = coeffs.reshape(-1, 15)
    expected = [convert_sh_basis(c, "leemans", "descoteaux") for c in flat]
    np.testing.assert_array_equal(converted.reshape(-1, 15), expected)


@pytest.mark.parametrize(("src", "dst"), [*PAIRS, ("leemans", "leemans")])
def test_float32_stays_float32(src: str, dst: str) -> None:
    coeffs = np.random.default_rng(5).normal(size=(2, 15)).astype(np.float32)
    assert convert_sh_basis(coeffs, src, dst).dtype == np.float32


@pytest.mark.parametrize(
    ("src", "dst"), [("mrtrix", "tournier"), ("tournier", "mrtrix")]
)
def test_unknown_basis_is_refused(src: str, dst: str) -> None:
    with pytest.raises(ValueError, match="mrtrix"):
        convert_sh_basis(np.zeros(15), src, dst)


@pytest.mark.parametrize(
    ("num_coeffs", "layout"),
    [
        (1, (0, False)),
        (4, (1, True)),
        (6, (2, False)),
        (9, (2, True)),
        (45, (8, False)),
        (81, (8, True)),
        (231, (20, False)),
        (441, (20, True)),
    ],
)
def test_sh_layout_reads_lmax_and_full_from_the_count(
    num_coeffs: int, layout: tuple[int, bool]
) -> None:
    assert sh_layout(num_coeffs) == layout


# 2 and 7 are no size at all, 10 is lmax 3 with even degrees only (odd lmax).
@pytest.mark.parametrize("num_coeffs", [0, -3, 2, 7, 10])
def test_sh_layout_refuses_a_count_of_no_basis(num_coeffs: int) -> None:
    with pytest.raises(ValueError, match=re.escape(str(num_coeffs))):
        sh_layout(num_coeffs)


@pytest.mark.parametrize("num_coeffs", [0, 10])
def test_invalid_number_of_coefficients_is_refused(num_coeffs: int) -> None:
    with pytest.raises(ValueError):
        convert_sh_basis(np.zeros(num_coeffs), "tournier", "leemans")


@pytest.mark.parametrize(("num_coeffs", "full"), [(81, True), (45, False)])
def test_only_a_full_size_array_is_reported_as_asymmetric(
    num_coeffs: int, full: bool, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="larmor"):
        convert_sh_basis(np.zeros(num_coeffs), "tournier", "leemans")
    line = "Reading the SH coefficients as asymmetric: lmax 8, all degrees, 81"
    if full:
        assert line in caplog.text
    else:
        assert "asymmetric" not in caplog.text


def test_leemans_fit_converted_to_tournier_matches_a_tournier_fit() -> None:
    # Two crossing fibres, sharp enough to need lmax 8, sampled densely.
    dirs = _random_dirs(2000, seed=6)
    fibres = np.array([[1.0, 0, 0], [0, 0.6, 0.8]])
    amplitudes = np.exp(10 * (dirs @ fibres.T) ** 2).sum(axis=1)

    leemans = fit_sh(amplitudes, dirs, 8, basis="leemans")
    tournier = fit_sh(amplitudes, dirs, 8, basis="tournier")

    converted = convert_sh_basis(leemans, "leemans", "tournier")
    np.testing.assert_allclose(
        converted, tournier, rtol=0, atol=1e-9 * np.abs(tournier).max()
    )


def test_mritoolkit_fod_converted_to_tournier_matches_mritoolkit(
    leemans_tournier_ref: dict[str, Any],
) -> None:
    # A real MRIToolkit FOD as it writes it to disk, and the same FOD refitted by
    # MRIToolkit in its "mrtrix" basis: an external check of leemans -> tournier.
    converted = convert_sh_basis(
        leemans_tournier_ref["fod_edti"], "leemans", "tournier"
    )
    np.testing.assert_allclose(
        converted, leemans_tournier_ref["fod_mrtrix"], rtol=0, atol=1e-12
    )


def test_fit_sh_is_the_least_squares_fit_of_the_basis_at_the_directions() -> None:
    dirs = _random_dirs(40)
    amplitudes = np.random.default_rng(1).random((5, 40))
    matrix = build_basis_matrix(*get_theta_phi(dirs), 4, "descoteaux")
    expected = np.linalg.lstsq(matrix, amplitudes.T)[0].T
    assert np.allclose(fit_sh(amplitudes, dirs, 4, basis="descoteaux"), expected)


def test_fit_sh_keeps_the_leading_axes() -> None:
    dirs = _random_dirs(60)
    amplitudes = np.random.default_rng(1).random((2, 3, 60))
    coeffs = fit_sh(amplitudes, dirs, 4)
    assert coeffs.shape == (2, 3, 15)
    assert np.allclose(coeffs[1, 2], fit_sh(amplitudes[1, 2], dirs, 4))


@pytest.mark.parametrize(("full", "num_coeffs"), [(False, 45), (True, 81)])
def test_fit_sh_returns_one_coefficient_per_degree_and_order(
    full: bool, num_coeffs: int
) -> None:
    amplitudes = np.ones(300)
    assert fit_sh(amplitudes, _random_dirs(300), 8, full=full).shape == (num_coeffs,)


def test_fit_sh_recovers_the_coefficients_of_an_sh_function() -> None:
    dirs = _random_dirs(300)
    matrix = build_basis_matrix(*get_theta_phi(dirs), 4)
    coeffs = np.random.default_rng(2).normal(size=15)
    amplitudes = matrix @ coeffs
    fitted = fit_sh(amplitudes, dirs, 4)
    assert np.allclose(fitted, coeffs)
    assert np.allclose(matrix @ fitted, amplitudes)


def test_fit_sh_does_not_depend_on_the_number_of_directions() -> None:
    few = fit_sh(np.full(60, 2.0), _random_dirs(60), 4)
    many = fit_sh(np.full(600, 2.0), _random_dirs(600, seed=1), 4)
    assert np.allclose(few, many)


@pytest.mark.parametrize(
    ("amp_shape", "dir_shape"),
    [((2, 30), (40, 3)), ((40,), (40, 2)), ((40,), (120,)), ((), (40, 3))],
)
def test_fit_sh_rejects_amplitudes_without_one_value_per_direction(
    amp_shape: tuple[int, ...], dir_shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="amplitudes") as err:
        fit_sh(np.ones(amp_shape), np.ones(dir_shape), 4)
    assert str(amp_shape) in str(err.value)
    assert str(dir_shape) in str(err.value)
