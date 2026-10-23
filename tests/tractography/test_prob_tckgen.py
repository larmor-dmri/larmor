# Probabilistic tracking against tckgen -algorithm iFOD1 (MRtrix 3.0.8) on the
# synthetic FOD of tests/data/make_tckgen_det_ref.py, from the 256 grid seeds of
# tests/data/make_tckgen_prob_ref.py along +x. The walks are random, so the two
# tools are compared through distributions, with tolerances from sampling noise:
# a two-sample Kolmogorov-Smirnov distance below its critical value at
# alpha = 0.001, means within 4 standard errors, and tckgen's -info figures within
# the spread of its ten runs with different random seeds (prob_ref_info.json).
import functools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from larmor.io import load_tractogram, load_volume
from larmor.tractography import EndStatus, Seeds, Tracker, TractographyRun

Points = NDArray[np.float64]

DATA = Path(__file__).parents[1] / "data" / "tracking"
CUTOFF = 0.3
MIN_LENGTH = 5.0
MAX_LENGTH = 80.0
N_SEEDS = 256
RUN_SEED = 1
# The Kolmogorov-Smirnov coefficient c(alpha) for alpha = 0.001.
KS_C = math.sqrt(-math.log(0.001 / 2) / 2)
LEAVING = (EndStatus.LEFT_MASK, EndStatus.LEFT_VOLUME)


@dataclass(frozen=True)
class Halves:
    """Each streamline split at its seed: away along the seed direction, back
    along its opposite, both starting at the seed."""

    away: list[Points]
    back: list[Points]


@dataclass(frozen=True)
class Pair:
    larmor: Halves
    ref: Halves
    run: TractographyRun
    status: NDArray[np.uint8]  # Larmor's EndStatus of (away, back)


def _split(points: Points, k: int) -> tuple[Points, Points]:
    return points[: k + 1][::-1], points[k:]


@functools.cache
def _pair() -> Pair:
    ref = load_tractogram(DATA / "prob_ref.tck")
    rows = np.loadtxt(DATA / "prob_ref_seeds.txt", delimiter=",", usecols=range(5))
    np.testing.assert_array_equal(rows[:, 0], np.arange(N_SEEDS))
    refs = [
        ref.points[ref.offsets[i] : ref.offsets[i + 1]].astype(np.float64)
        for i in range(N_SEEDS)
    ]
    seeds = rows[:, 2:5]
    run = Tracker(
        "prob",
        cutoff=CUTOFF,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        progress=False,
    ).track(
        load_volume(DATA / "fod.nii.gz"),
        None,
        Seeds(seeds, np.tile([1.0, 0.0, 0.0], (N_SEEDS, 1))),
        mask=load_volume(DATA / "mask.nii.gz"),
        seed=RUN_SEED,
        n_jobs=1,
    )
    t = run.tractogram
    assert t is not None and run.num_streamlines == N_SEEDS
    status = t.streamline_data["end_status"]
    away, back = [], []
    for i in range(N_SEEDS):
        points = t.points[t.offsets[i] : t.offsets[i + 1]].astype(np.float64)
        at = np.flatnonzero(np.all(points == seeds[i].astype(np.float32), axis=1))
        a, b = _split(points, int(at[0]))
        away.append(a)
        back.append(b)
    ref_halves = [_split(p, int(k)) for p, k in zip(refs, rows[:, 1], strict=True)]
    return Pair(
        larmor=Halves(away, back),
        ref=Halves([h[0] for h in ref_halves], [h[1] for h in ref_halves]),
        run=run,
        status=status,
    )


@functools.cache
def _info() -> dict[str, Any]:
    info: dict[str, Any] = json.loads(
        (DATA / "prob_ref_info.json").read_text(encoding="utf-8")
    )
    return info


def _runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = _info()["runs"]
    assert len(runs) == 10
    return runs


def _header() -> dict[str, str]:
    fields: dict[str, str] = {}
    with open(DATA / "prob_ref.tck", "rb") as f:
        for raw in f:
            line = raw.decode("latin-1").strip()
            if line == "END":
                return fields
            key, _, value = line.partition(": ")
            fields.setdefault(key, value)
    raise AssertionError("no END in the header")


def _length(points: Points) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _assert_same_distribution(larmor: Points, ref: Points, what: str) -> None:
    d = stats.ks_2samp(larmor, ref).statistic
    n, m = len(larmor), len(ref)
    critical = KS_C * math.sqrt((n + m) / (n * m))
    quartiles = np.percentile(larmor, [25, 50, 75]), np.percentile(ref, [25, 50, 75])
    assert d < critical, f"{what}: KS {d:.3f} >= {critical:.3f}, {quartiles}"


def _assert_same_mean(larmor: Points, ref: Points, what: str) -> None:
    se = math.sqrt(
        np.var(larmor, ddof=1) / len(larmor) + np.var(ref, ddof=1) / len(ref)
    )
    diff = abs(float(np.mean(larmor) - np.mean(ref)))
    assert diff < 4 * se, f"{what}: {np.mean(larmor)} vs {np.mean(ref)}, se {se}"


def test_reference_was_made_with_larmor_defaults_and_the_parameters_tracked_here() -> (
    None
):
    # Also checks that Larmor's defaults for prob are iFOD1's: the step and the angle.
    header = _header()
    parameters = _pair().run.parameters

    assert header["mrtrix_version"] == "3.0.8"
    assert _info()["mrtrix_version"] == "tckgen 3.0.8"
    assert header["method"] == "iFOD1"
    assert header["init_direction"] == "1,0,0"
    assert header["sh_precomputed"] == "0"
    assert header["fod_power"] == "1"
    assert header["rk4"] == "0" and header["unidirectional"] == "0"
    assert header["max_trials"] == header["max_seed_attempts"] == "1000"
    assert float(header["step_size"]) == pytest.approx(parameters["step_size"])
    assert float(header["max_angle"]) == parameters["max_angle"] == 15.0
    assert float(header["threshold"]) == pytest.approx(CUTOFF)
    assert float(header["min_dist"]) == MIN_LENGTH
    assert float(header["max_dist"]) == MAX_LENGTH
    assert int(header["count"]) == N_SEEDS


@pytest.mark.parametrize("half", ["whole", "away", "back"])
def test_lengths_have_the_same_distribution(half: str) -> None:
    pair = _pair()

    def lengths(h: Halves) -> Points:
        if half == "whole":
            return np.array(
                [_length(a) + _length(b) for a, b in zip(h.away, h.back, strict=True)]
            )
        return np.array([_length(p) for p in getattr(h, half)])

    _assert_same_distribution(lengths(pair.larmor), lengths(pair.ref), half)


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
@pytest.mark.parametrize("half", ["away", "back"])
def test_end_points_have_the_same_distribution(half: str, axis: int) -> None:
    # Away, the ends spread along the arc and the fading bundle; back, they sit at
    # the mask's edge, spread across the bundle.
    pair = _pair()
    larmor = np.array([p[-1, axis] for p in getattr(pair.larmor, half)])
    ref = np.array([p[-1, axis] for p in getattr(pair.ref, half)])

    _assert_same_distribution(larmor, ref, f"{half} end {'xyz'[axis]}")


def _turns(points: Points) -> Points:
    # The angle (degrees) between consecutive steps, without the first step, which
    # Larmor takes along the seed direction and tckgen samples.
    steps = np.diff(points[1:], axis=0)
    steps /= np.linalg.norm(steps, axis=1, keepdims=True)
    cos = np.clip(np.sum(steps[1:] * steps[:-1], axis=1), -1.0, 1.0)
    out: Points = np.degrees(np.arccos(cos))
    return out


def test_steps_turn_by_the_same_angles() -> None:
    # The turn per step is how the sampled directions spread in the cone: its mean
    # per streamline (independent between streamlines) is compared, and the
    # distribution of all the turns.
    pair = _pair()
    larmor = [_turns(p) for p in pair.larmor.away + pair.larmor.back if len(p) > 3]
    ref = [_turns(p) for p in pair.ref.away + pair.ref.back if len(p) > 3]

    _assert_same_mean(
        np.array([t.mean() for t in larmor]), np.array([t.mean() for t in ref]), "turn"
    )
    all_larmor, all_ref = np.concatenate(larmor), np.concatenate(ref)
    assert np.all(all_larmor <= 15.0 + 1e-3)
    for q in [25, 50, 75, 95]:
        assert np.percentile(all_larmor, q) == pytest.approx(
            np.percentile(all_ref, q), rel=0.05
        )


@pytest.mark.parametrize("mm", [5.0, 20.0])
def test_direction_away_from_the_seed_spreads_the_same(mm: float) -> None:
    # The direction from the seed to the point mm along the away half: its
    # deviation from +x and its mean.
    pair = _pair()
    n = round(mm / float(pair.run.parameters["step_size"]))  # type: ignore[arg-type]

    def chords(halves: list[Points]) -> Points:
        c = np.array([p[n] - p[0] for p in halves if len(p) > n])
        out: Points = c / np.linalg.norm(c, axis=1, keepdims=True)
        return out

    larmor, ref = chords(pair.larmor.away), chords(pair.ref.away)
    _assert_same_distribution(
        np.degrees(np.arccos(larmor[:, 0])), np.degrees(np.arccos(ref[:, 0])), "angle"
    )
    for axis in range(3):
        _assert_same_mean(larmor[:, axis], ref[:, axis], f"chord {'xyz'[axis]}")


def test_ends_run_out_of_candidates_as_often_as_in_tckgen() -> None:
    # tckgen's "diffusion model sub-threshold" is NO_DIRECTION; the other ends of
    # both tools leave the mask.
    percents = [
        r["termination_percent"]["Diffusion model sub-threshold"] for r in _runs()
    ]
    status = _pair().status
    larmor = 100 * np.count_nonzero(status == EndStatus.NO_DIRECTION) / status.size

    assert set(np.unique(status).tolist()) <= {
        EndStatus.NO_DIRECTION,
        *LEAVING,
    }
    assert abs(larmor - np.mean(percents)) < 4 * np.std(percents, ddof=1)


def test_sampling_figures_are_tckgens() -> None:
    sampling = _pair().run.sampling
    runs = _runs()
    assert sampling is not None
    means = [r["mean_samples_per_step"] for r in runs]
    between = [r["steps_per_truncation"] for r in runs if r["steps_per_truncation"]]
    largest = [r["max_truncation"] for r in runs if r["max_truncation"]]

    assert {r["calibration_directions"] for r in runs} == {7}
    assert sampling["calibration_directions"] == 7
    mean = sampling["mean_samples_per_step"]
    assert mean is not None
    assert abs(mean - np.mean(means)) < 4 * np.std(means, ddof=1)
    # Truncations are rare in tckgen: none in some runs, else one every 15000 steps
    # or more, a candidate at most 1.06 times the bound. Larmor's are as rare and as
    # small, within a factor of 2.
    steps, worst = sampling["steps_per_truncation"], sampling["max_truncation"]
    assert steps is None or steps > min(between) / 2
    assert (steps is None) == (worst is None)
    assert worst is None or worst < 1 + 2 * (max(largest) - 1)
