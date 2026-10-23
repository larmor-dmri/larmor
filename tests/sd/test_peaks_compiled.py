# The compiled peak search (_core.sh_derivatives, _core.climb, _core.peaks_chunk and
# find_peaks on top of them) against the numpy reference in reference_peaks.py, and
# the checks of the bindings.
import math
import pathlib
from typing import Any, TypedDict, Unpack

import numpy as np
import numpy.typing as npt
import pytest
import reference_peaks as ref

from larmor import _core
from larmor.io import load_volume
from larmor.sd import find_peaks
from larmor.sd.peaks import _build_icosphere, _get_amplitudes
from larmor.sd.sh import build_basis_matrix, get_sh_indices

DATA = pathlib.Path(__file__).parents[1] / "data" / "peaks"
LMAX = 8
X, Y, Z = np.eye(3)

Array = npt.NDArray[np.float64]


def _unit(v: npt.ArrayLike) -> Array:
    u = np.asarray(v, dtype=np.float64)
    out: Array = u / np.linalg.norm(u)
    return out


def _random_sh(lmax: int, seed: int) -> Array:
    n = len(get_sh_indices(lmax)[0])
    sh: Array = np.random.default_rng(seed).normal(size=n)
    return sh


def _random_starts(n: int, seed: int) -> Array:
    starts = np.random.default_rng(seed).normal(size=(n, 3))
    out: Array = starts / np.linalg.norm(starts, axis=1, keepdims=True)
    return out


def _fibre_sh(direction: Array) -> Array:
    ls, _ = get_sh_indices(LMAX)
    theta = math.acos(float(np.clip(direction[2], -1.0, 1.0)))
    phi = math.atan2(float(direction[1]), float(direction[0]))
    b = build_basis_matrix(np.array([theta]), np.array([phi]), LMAX)
    sh: Array = b[0] * np.exp(-0.02 * ls * (ls + 1))
    return sh


@pytest.fixture(scope="module")
def fod_block() -> Array:
    # The 8x8x8 block the sh2peaks reference tests use, one voxel per row.
    fod = load_volume(DATA / "fod.mif")
    fod.to_closest_canonical()
    coeffs: Array = np.ascontiguousarray(
        fod.as_float(dtype=np.float64).reshape(-1, fod.data.shape[-1])
    )
    return coeffs


# Derivatives


@pytest.mark.parametrize("lmax", range(0, 21, 2))
def test_sh_derivatives_match_the_reference(lmax: int) -> None:
    rng = np.random.default_rng(lmax)
    # Poles and near-poles included: there both switch to the limit forms.
    theta = np.concatenate(
        [rng.uniform(0, math.pi, 40), [0.0, 1e-6, 5e-5, 2e-4, math.pi - 1e-6, math.pi]]
    )
    phi = rng.uniform(-math.pi, math.pi, len(theta))

    for seed in range(3):
        sh = _random_sh(lmax, seed)
        for t, p in zip(theta, phi, strict=True):
            expected = np.array(ref.derivatives(sh, lmax, float(t), float(p)))
            actual = np.array(_core.sh_derivatives(sh, lmax, float(t), float(p)))
            # Measured: within 5e-15 of the largest of the six; the sums run in a
            # different order, so a component near 0 has no relative bound.
            scale = float(np.abs(expected).max())
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13 * scale)


# Climb

# Arbitrary starts on the block's FODs and on random SH: most climbs take a few
# Newton steps and agree to ~1e-16. A few wander for tens of iterations over a
# flat region, where each step is poorly conditioned: there the reference itself
# ends up to ~4e-10 elsewhere when its start moves by one ulp, and so did the
# previous implementation.
CLIMB_DIR_ATOL = 1e-9
# The amplitude is taken at a near-stationary point, so it is insensitive to
# where the climb ends: measured within 2e-16 of sum |sh|.
CLIMB_AMP_ATOL = 1e-14


class ClimbSettings(TypedDict, total=False):
    max_iters: int
    tol: float
    max_step: float


def _assert_same_climb(
    sh: Array, lmax: int, start: Array, **settings: Unpack[ClimbSettings]
) -> None:
    d_core, f_core = _core.climb(sh, lmax, start, **settings)
    d_ref, f_ref = ref.climb(sh, lmax, start, **settings)

    assert math.isnan(f_core) == math.isnan(f_ref)
    np.testing.assert_array_equal(np.isnan(d_core), np.isnan(d_ref))
    if not math.isnan(f_ref):
        np.testing.assert_allclose(d_core, d_ref, rtol=0, atol=CLIMB_DIR_ATOL)
        assert f_core == pytest.approx(
            f_ref, rel=0, abs=CLIMB_AMP_ATOL * np.abs(sh).sum()
        )


def test_climb_matches_the_reference_from_arbitrary_starts_on_real_fods(
    fod_block: Array,
) -> None:
    rng = np.random.default_rng(0)
    voxels = rng.choice(len(fod_block), 20, replace=False)
    starts = _random_starts(20, seed=1)

    for i in voxels:
        for start in starts:
            _assert_same_climb(fod_block[i], LMAX, start)


@pytest.mark.parametrize("lmax", [2, 8, 16, 20])
def test_climb_matches_the_reference_from_arbitrary_starts_on_random_sh(
    lmax: int,
) -> None:
    starts = _random_starts(15, seed=lmax)

    for seed in range(4):
        sh = _random_sh(lmax, seed)
        for start in starts:
            _assert_same_climb(sh, lmax, start)


def _crossing(separation_deg: float) -> tuple[Array, Array]:
    s = math.radians(separation_deg)
    b = np.array([math.cos(s), math.sin(s), 0.0])
    return _fibre_sh(X) + _fibre_sh(b), _unit(X + b)


SADDLE_60_SH, SADDLE_60 = _crossing(60.0)
SADDLE_90_SH, SADDLE_90 = _crossing(90.0)
# Stationary points, where the gradient is 0 up to rounding: a fibre's own axis at
# the pole (axially symmetric, the pole form of the derivatives), its antipode, a
# point on the ring of minima 90 degrees from it, and the saddle between two fibres.
STATIONARY = {
    "peak_at_pole": (_fibre_sh(Z), Z),
    "antipode_at_pole": (_fibre_sh(Z), -Z),
    "peak_on_equator": (_fibre_sh(X), X),
    "ring_of_minima": (_fibre_sh(X), Y),
    "ring_at_pole": (_fibre_sh(X), Z),
    "saddle_60": (SADDLE_60_SH, SADDLE_60),
    "saddle_90": (SADDLE_90_SH, SADDLE_90),
}


@pytest.mark.parametrize("name", STATIONARY)
def test_climb_matches_the_reference_on_stationary_points(name: str) -> None:
    sh, start = STATIONARY[name]

    d_core, f_core = _core.climb(sh, LMAX, start)
    d_ref, f_ref = ref.climb(sh, LMAX, start)

    # One step of ~1e-16 and out: no room for rounding to grow.
    np.testing.assert_allclose(d_core, d_ref, rtol=0, atol=1e-14)
    np.testing.assert_allclose(d_core, start, rtol=0, atol=1e-12)
    assert f_core == pytest.approx(f_ref, rel=0, abs=1e-14)


# Each setting is tested on climbs that stop after a step or a few, where the two
# agree to rounding (measured 2e-14): one step with tol above the cap, a smaller
# cap, and too few iterations, which gives NaN for most starts.
SETTINGS: list[ClimbSettings] = [
    {"tol": 1.0},
    {"tol": 1.0, "max_step": 0.05},
    {"max_iters": 3},
    {"tol": 0.0},
]


@pytest.mark.parametrize("settings", SETTINGS, ids=str)
def test_climb_settings_match_the_reference(
    fod_block: Array, settings: ClimbSettings
) -> None:
    starts = _random_starts(10, seed=2)

    for i in range(0, len(fod_block), 37):
        for start in starts:
            _assert_same_climb(fod_block[i], LMAX, start, **settings)


# Peak search


# Measured with the default thresholds: directions within 5e-14, amplitudes within
# 1e-15 relative. Without a relative threshold the weak lobes count too: their
# climbs are long and poorly conditioned, as for arbitrary starts (measured 2e-11
# and 4e-13).
@pytest.mark.parametrize(
    ("num_peaks", "relative_threshold", "threshold", "dir_atol", "amp_rtol"),
    [
        (3, 0.1, 0.0, 1e-12, 1e-12),
        (2, 0.3, 0.2, 1e-12, 1e-12),
        (6, 0.0, 0.0, 1e-9, 1e-11),
    ],
)
def test_find_peaks_matches_the_reference_on_a_real_fod_block(
    fod_block: Array,
    num_peaks: int,
    relative_threshold: float,
    threshold: float,
    dir_atol: float,
    amp_rtol: float,
) -> None:
    vertices, neighbours = _build_icosphere()
    amps = _get_amplitudes(fod_block, LMAX, full=False)

    found = find_peaks(
        fod_block,
        num_peaks=num_peaks,
        relative_threshold=relative_threshold,
        threshold=threshold,
    )

    assert np.isfinite(found.amplitudes).any()
    for i in range(0, len(fod_block), 4):
        exp_dirs, exp_amps = ref.pep(
            fod_block[i],
            LMAX,
            amps[i],
            vertices,
            neighbours,
            num_peaks,
            relative_threshold,
            threshold,
        )
        # Same peaks, in the same slots.
        np.testing.assert_array_equal(np.isnan(found.amplitudes[i]), np.isnan(exp_amps))
        np.testing.assert_allclose(found.directions[i], exp_dirs, rtol=0, atol=dir_atol)
        np.testing.assert_allclose(found.amplitudes[i], exp_amps, rtol=amp_rtol)


@pytest.mark.parametrize("n_jobs", [2, 4, -1])
def test_find_peaks_does_not_depend_on_the_thread_count(
    fod_block: Array, n_jobs: int
) -> None:
    one = find_peaks(fod_block, num_peaks=5, relative_threshold=0.0, n_jobs=1)

    many = find_peaks(fod_block, num_peaks=5, relative_threshold=0.0, n_jobs=n_jobs)

    np.testing.assert_array_equal(many.directions, one.directions)
    np.testing.assert_array_equal(many.amplitudes, one.amplitudes)


# The search's starts: peaks_chunk takes the sphere amplitudes from the caller, so
# a chosen amplitude pattern decides which vertices the climbs start from.


def _chunk(
    sh: Array,
    on_sphere: Array,
    vertices: Array | None = None,
    neighbours: npt.NDArray[np.int64] | None = None,
    num_peaks: int = 4,
    relative_threshold: float = 0.0,
    threshold: float = 0.0,
) -> tuple[Array, Array]:
    if vertices is None or neighbours is None:
        vertices, icosphere_neighbours = _build_icosphere()
        neighbours = icosphere_neighbours.astype(np.int64)
    return _core.peaks_chunk(
        np.ascontiguousarray(sh[None]),
        LMAX,
        np.ascontiguousarray(on_sphere[None]),
        vertices,
        neighbours,
        num_peaks,
        relative_threshold,
        threshold,
        1,
    )


def _two_starts(a: Array, b: Array) -> tuple[Array, npt.NDArray[np.int64]]:
    # A two-vertex sphere, each the other's only neighbour (padded with itself).
    neighbours = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1]], dtype=np.int64)
    return np.array([a, b]), neighbours


@pytest.mark.parametrize(
    "on_sphere",
    [
        np.zeros(642),
        -np.ones(642),
        _get_amplitudes(_fibre_sh(X)[None], LMAX, full=False)[0] - 2.0,
    ],
    ids=["zero", "negative_constant", "negative_lobe"],
)
def test_search_does_not_start_from_non_positive_maxima(on_sphere: Array) -> None:
    # The FOD has a positive peak along x; only the starts are non-positive.
    dirs, amps = _chunk(_fibre_sh(X), on_sphere)

    assert np.all(np.isnan(dirs))
    assert np.all(np.isnan(amps))


def test_search_starts_from_both_vertices_of_a_tie() -> None:
    # A 90 degree crossing: the climbs from x and y end on different fibres.
    vertices, neighbours = _two_starts(X, Y)
    sh = _fibre_sh(X) + _fibre_sh(Y)

    tie, _ = _chunk(sh, np.array([1.0, 1.0]), vertices, neighbours)
    lower, _ = _chunk(sh, np.array([1.0, 0.9]), vertices, neighbours)

    assert int(np.sum(np.isfinite(tie[0, :, 0]))) == 2
    assert int(np.sum(np.isfinite(lower[0, :, 0]))) == 1


@pytest.mark.parametrize("vertex", [0, 7, 11, 12, 300, 641])
def test_search_starts_only_from_the_single_maximum(vertex: int) -> None:
    # Vertices 0 to 11 have five neighbours and are padded with themselves. Three
    # orthogonal fibres: starting anywhere else would find more than one.
    vertices, _ = _build_icosphere()
    sh = _fibre_sh(X) + _fibre_sh(Y) + _fibre_sh(Z)
    expected, f = ref.climb(sh, LMAX, vertices[vertex])
    assert f > 0

    dirs, amps = _chunk(sh, vertices @ vertices[vertex])

    assert int(np.sum(np.isfinite(amps))) == 1
    assert abs(float(dirs[0, 0] @ expected)) == pytest.approx(1.0, abs=1e-12)
    assert amps[0, 0] == pytest.approx(f, abs=1e-12)


def _on_ring(azimuth_deg: float) -> Array:
    # 50 degrees from z: inside the basin of the side ring of a fibre along z.
    t, p = math.radians(50.0), math.radians(azimuth_deg)
    return np.array([math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)])


@pytest.mark.parametrize(("azimuth_deg", "count"), [(6.0, 1), (12.0, 2)])
def test_search_merges_peaks_closer_than_0_99_in_cos(
    azimuth_deg: float, count: int
) -> None:
    # The side ring of a fibre along z is flat in azimuth, so the climbs move
    # only toward it (~53 degrees from z) and end at their start's azimuth.
    # 6 degrees of azimuth there are ~4.8 degrees apart (|cos| 0.9965), 12 degrees
    # ~9.6 (|cos| 0.986).
    vertices, neighbours = _two_starts(_on_ring(0.0), _on_ring(azimuth_deg))

    dirs, _ = _chunk(_fibre_sh(Z), np.array([1.0, 1.0]), vertices, neighbours)

    assert int(np.sum(np.isfinite(dirs[0, :, 0]))) == count


def test_search_drops_maxima_of_non_positive_amplitude() -> None:
    # Only reachable through peaks_chunk: the start is positive by fiat, but the
    # climb on this negated fibre ends on a negative maximum. A relative threshold
    # of 1 and no absolute threshold would otherwise keep the largest peak.
    sh = -_fibre_sh(X)
    start = _unit((0.0, 0.53, 0.85))
    _, f = ref.climb(sh, LMAX, start)
    assert f < 0
    vertices, neighbours = _two_starts(start, start)

    dirs, amps = _chunk(
        sh,
        np.array([1.0, 1.0]),
        vertices,
        neighbours,
        relative_threshold=1.0,
        threshold=-math.inf,
    )

    assert np.all(np.isnan(dirs))
    assert np.all(np.isnan(amps))


# Binding checks

C45 = np.zeros(45)


# An odd (unless full) or too large lmax, or a coefficient count that does not
# match the layout.
BAD_LMAX = (
    r"lmax must be at most 20, and even unless full, got|"
    r"lmax \d+ has \d+ coefficients, got"
)


@pytest.mark.parametrize(
    ("lmax", "num_coeffs", "full"),
    [
        (7, 36, False),
        (22, 276, False),
        (8, 28, False),
        (6, 45, False),
        (21, 484, True),
        # The symmetric count for the lmax, or the full count of an odd lmax
        # without full.
        (8, 45, True),
        (7, 64, False),
        (7, 36, True),
    ],
    ids=[
        "odd",
        "above_20",
        "too_few",
        "too_many",
        "full_above_20",
        "full_given_even_count",
        "full_count_without_full",
        "full_odd_too_few",
    ],
)
def test_bindings_refuse_a_bad_lmax(lmax: int, num_coeffs: int, full: bool) -> None:
    sh = np.zeros(num_coeffs)
    vertices, neighbours = _build_icosphere()

    with pytest.raises(ValueError, match=BAD_LMAX):
        _core.climb(sh, lmax, Z, full=full)
    with pytest.raises(ValueError, match=BAD_LMAX):
        _core.sh_derivatives(sh, lmax, 0.3, 0.2, full=full)
    with pytest.raises(ValueError, match=BAD_LMAX):
        _core.peaks_chunk(
            sh[None],
            lmax,
            np.zeros((1, 642)),
            vertices,
            neighbours.astype(np.int64),
            3,
            0.1,
            0.0,
            1,
            full=full,
        )


def _good_chunk_args() -> dict[str, Any]:
    vertices, neighbours = _build_icosphere()
    return {
        "coeffs": np.zeros((2, 45)),
        "lmax": LMAX,
        "on_sphere": np.zeros((2, 642)),
        "vertices": vertices,
        "neighbours": neighbours.astype(np.int64),
        "n_peaks": 3,
        "relative_threshold": 0.1,
        "threshold": 0.0,
        "n_threads": 1,
    }


def _run_chunk(args: dict[str, Any]) -> tuple[Array, Array]:
    return _core.peaks_chunk(**args)


def test_good_chunk_arguments_are_accepted() -> None:
    dirs, amps = _run_chunk(_good_chunk_args())

    assert dirs.shape == (2, 3, 3)
    assert amps.shape == (2, 3)


def _neighbours_with(value: int) -> npt.NDArray[np.int64]:
    _, neighbours = _build_icosphere()
    out = neighbours.astype(np.int64)
    out[5, 2] = value
    return out


VERTICES, NEIGHBOURS = _build_icosphere()
BAD_CHUNK_ARGS: dict[str, tuple[str, Any, str]] = {
    "vertices_4_columns": (
        "vertices",
        np.zeros((642, 4)),
        "shapes do not match",
    ),
    "neighbours_5_columns": (
        "neighbours",
        NEIGHBOURS[:, :5].astype(np.int64).copy(),
        "shapes do not match",
    ),
    "neighbours_fewer_rows": (
        "neighbours",
        NEIGHBOURS[:600].astype(np.int64),
        "shapes do not match",
    ),
    "on_sphere_fewer_voxels": (
        "on_sphere",
        np.zeros((1, 642)),
        "shapes do not match",
    ),
    "on_sphere_fewer_vertices": (
        "on_sphere",
        np.zeros((2, 600)),
        "shapes do not match",
    ),
    "negative_neighbour": (
        "neighbours",
        _neighbours_with(-1),
        "neighbour index is outside the sphere",
    ),
    "neighbour_past_the_end": (
        "neighbours",
        _neighbours_with(642),
        "neighbour index is outside the sphere",
    ),
    "zero_peaks": ("n_peaks", 0, "n_peaks must be at least 1"),
    "zero_threads": ("n_threads", 0, "n_threads must be at least 1"),
    "fortran_coeffs": (
        "coeffs",
        np.asfortranarray(np.zeros((2, 45))),
        "C-contiguous",
    ),
    "strided_on_sphere": ("on_sphere", np.zeros((4, 642))[::2], "C-contiguous"),
}


@pytest.mark.parametrize("case", BAD_CHUNK_ARGS)
def test_peaks_chunk_refuses_bad_arguments(case: str) -> None:
    name, value, message = BAD_CHUNK_ARGS[case]
    args = _good_chunk_args()
    args[name] = value

    with pytest.raises(ValueError, match=message):
        _run_chunk(args)
