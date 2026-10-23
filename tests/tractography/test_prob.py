# Probabilistic tracking (iFOD1-like): the binding's checks, the compiled step against
# the numpy reference in reference_prob.py on a smooth synthetic FOD, and the
# tracker's behaviour with it. The reference draws the same random numbers in the
# same order, so whole streamlines, their ends and the sampling counters must
# match _core's, not only their statistics.
import functools
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from reference_prob import (
    Draws,
    Prob,
    ProbScene,
    Sampling,
    amplitude,
    calibrate,
    track_prob,
    track_prob_core,
)
from reference_random import SEARCH_COUNTER, stream
from reference_tracking import Streamline, trilinear

from larmor import _core
from larmor.io import load_volume
from larmor.sd.sh import build_basis_matrix
from larmor.tractography import (
    EndStatus,
    Seeds,
    Tracker,
    TractographyRun,
    read_seeds,
)
from larmor.types import Tractogram, Volume

Vector = NDArray[np.float64]

LMAX = 8
CORES = os.cpu_count() or 1
DATA = Path(__file__).parents[1] / "data" / "tracking"


@functools.cache
def _fit() -> tuple[Vector, Vector]:
    # A quadrature-weighted projection onto the SH basis, and its directions: a
    # fitted lobe keeps its axial symmetry, so its peak is its axis.
    nodes, weights = np.polynomial.legendre.leggauss(24)
    phis = np.arange(48) * (2 * math.pi / 48)
    ct, ph = np.meshgrid(nodes, phis, indexing="ij")
    st = np.sqrt(1 - ct**2)
    dirs = np.stack(
        [(st * np.cos(ph)).ravel(), (st * np.sin(ph)).ravel(), ct.ravel()], axis=1
    )
    basis = build_basis_matrix(
        np.arccos(dirs[:, 2]), np.arctan2(dirs[:, 1], dirs[:, 0]), LMAX
    )
    root = np.sqrt(np.repeat(weights, 48))[:, np.newaxis]
    fit: Vector = np.linalg.pinv(root * basis) * root.T
    return fit, dirs


def _lobes(axes: Vector, sharpness: float) -> Vector:
    fit, dirs = _fit()
    out: Vector = np.exp(-sharpness * (1 - (axes @ dirs.T) ** 2)) @ fit.T
    return out


def _unit(v: Vector) -> Vector:
    out: Vector = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return out


# The compiled step against the reference


def _rotation(axis: Vector, degrees: float) -> Vector:
    axis = _unit(axis)
    r = math.radians(degrees)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    out: Vector = np.eye(3) + math.sin(r) * k + (1 - math.cos(r)) * (k @ k)
    return out


SMOOTH_SHAPE = (8, 8, 8)
SMOOTH_AFFINE = np.eye(4)
SMOOTH_AFFINE[:3, :3] = 1.5 * _rotation(np.array([1.0, 2.0, 2.0]), 35.0)
SMOOTH_AFFINE[:3, 3] = [-4.0, 2.0, 1.0]


def _axes(vox: Vector) -> Vector:
    # The lobe axis (world) at voxel coordinates vox (n, 3): it turns slowly in the
    # voxel grid's first two axes.
    t = 0.25 * vox[:, 0] + 0.15 * vox[:, 1]
    axes = np.stack([np.cos(t), np.sin(t), 0.3 + 0 * t], axis=1)
    return _unit(axes @ SMOOTH_AFFINE[:3, :3].T)


@functools.cache
def _smooth_fod() -> NDArray[np.float32]:
    # One broad lobe per voxel along _axes, faded in the upper half of the third
    # voxel axis: the walks curve, and run out of candidates above the cutoff
    # where the lobe has faded.
    vox = np.indices(SMOOTH_SHAPE).reshape(3, -1).T.astype(np.float64)
    scale = np.clip(1.6 - 0.2 * vox[:, 2], 0.0, 1.0)[:, np.newaxis]
    fod = (scale * _lobes(_axes(vox), 4.0)).reshape(*SMOOTH_SHAPE, -1)
    return np.ascontiguousarray(fod, dtype=np.float32)


def _ball_mask() -> NDArray[np.float32]:
    grid = np.moveaxis(np.indices(SMOOTH_SHAPE), 0, -1)
    ball = np.linalg.norm(grid - 3.5, axis=-1) < 4.2
    return np.ascontiguousarray(ball, dtype=np.float32)


N_SEEDS = 12
INDICES = np.arange(100, 100 + N_SEEDS, dtype=np.int64)
# The Python reference is slow: the comparison with it uses the first seeds.
N_COMPARED = 8


def _seeds() -> tuple[Vector, Vector]:
    # Voxel coordinates up to just past the grid's edge, so a few do not start;
    # directions within 10 degrees of the lobe's axis, either way.
    rng = np.random.default_rng(11)
    vox = rng.uniform(-0.8, 7.8, size=(N_SEEDS, 3))
    seeds = vox @ SMOOTH_AFFINE[:3, :3].T + SMOOTH_AFFINE[:3, 3]
    sign = rng.choice([-1.0, 1.0], size=(N_SEEDS, 1))
    tilt = math.tan(math.radians(10)) / math.sqrt(3) * rng.uniform(-1, 1, (N_SEEDS, 3))
    return seeds, _unit(sign * _axes(np.round(vox)) + tilt)


def _scene(prob: Prob, masked: bool, bidirectional: bool = True) -> ProbScene:
    return ProbScene(
        fod=_smooth_fod(),
        fod_affine=SMOOTH_AFFINE,
        mask=_ball_mask() if masked else None,
        mask_affine=SMOOTH_AFFINE,
        step_size=0.5,
        max_steps=20,
        bidirectional=bidirectional,
        prob=prob,
    )


def _reference(
    scene: ProbScene, seeds: Vector, directions: Vector | None
) -> tuple[list[Streamline | None], Sampling]:
    total = Sampling()
    out = []
    for i, (s, index) in enumerate(zip(seeds, INDICES[: len(seeds)], strict=True)):
        line, stats = track_prob(
            scene, s, None if directions is None else directions[i], int(index)
        )
        out.append(line)
        total.add(stats)
    return out, total


def _assert_same_walks(
    core: list[Streamline | None], ref: list[Streamline | None]
) -> None:
    # The compiled float32 points are the reference's rounded, exactly. A searched
    # direction can differ in its last bit: the C++ sine and cosine of the drawn
    # azimuth are not always math.sin's and math.cos's; float32 hides it.
    assert [r is None for r in core] == [r is None for r in ref]
    for a, b in zip(core, ref, strict=True):
        if a is not None and b is not None:
            np.testing.assert_array_equal(a.points, b.points.astype(np.float32))
            assert a.ends == b.ends
            np.testing.assert_allclose(a.direction, b.direction, rtol=0, atol=1e-15)


def _assert_same_sampling(
    core: dict[str, float], ref: Sampling, calibration_directions: int
) -> None:
    assert core["sampled_steps"] == ref.sampled_steps
    assert core["accepted_index_sum"] == ref.accepted_index_sum
    assert core["truncations"] == ref.truncations
    # a / bound from amplitudes that differ by ~1e-15.
    assert core["max_truncation"] == pytest.approx(ref.max_truncation, rel=1e-9)
    assert core["calibration_directions"] == calibration_directions


# (max angle, cutoff, trials, attempts): the cone below and above 90 degrees,
# where the theta bound switches to 1; a cutoff of 0; trials and attempts few
# enough that streamlines run out of candidates and seeds find no direction.
CASES = {
    "15deg": (15.0, 0.3, 1000, 1000),
    "45deg-no-cutoff": (45.0, 0.0, 1000, 1000),
    "120deg": (120.0, 0.2, 1000, 1000),
    "few-trials": (30.0, 0.4, 3, 3),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("searched", [False, True], ids=["given", "searched"])
@pytest.mark.parametrize("bidirectional", [True, False], ids=["bi", "uni"])
def test_streamlines_and_counters_match_the_reference(
    case: str, searched: bool, bidirectional: bool
) -> None:
    angle, cutoff, trials, attempts = CASES[case]
    prob = Prob(cutoff, angle, LMAX, trials, attempts)
    scene = _scene(prob, masked=case != "120deg", bidirectional=bidirectional)
    seeds, directions = _seeds()
    seeds, directions = seeds[:N_COMPARED], directions[:N_COMPARED]
    given = None if searched else directions

    core, stats = track_prob_core(scene, seeds, given, INDICES[:N_COMPARED])
    ref, ref_stats = _reference(scene, seeds, given)

    _assert_same_walks(core, ref)
    _assert_same_sampling(
        stats, ref_stats, len(calibrate(LMAX, math.radians(angle))[0])
    )
    assert sum(r is not None for r in core) >= 2


def test_reference_cases_reach_every_kind_of_end() -> None:
    # Guards the comparison above: walks that all ended the same way, or searches
    # that all succeeded, would test little of the step.
    scenes = {
        case: _scene(Prob(c, a, LMAX, t, n), masked=True)
        for case, (a, c, t, n) in CASES.items()
    }
    ends: set[EndStatus] = set()
    for case in ["15deg", "few-trials"]:
        core, _ = track_prob_core(scenes[case], *_seeds(), INDICES)
        ends |= {e for r in core if r is not None for e in r.ends}
    searched, _ = track_prob_core(scenes["few-trials"], _seeds()[0], None, INDICES)

    assert {
        EndStatus.LEFT_VOLUME,
        EndStatus.LEFT_MASK,
        EndStatus.MAX_LENGTH,
        EndStatus.NO_DIRECTION,
    } <= ends
    assert any(s is None for s in searched) and any(s is not None for s in searched)


def test_truncations_are_counted_as_the_reference_does() -> None:
    # Truncations are rare (about 1 in 40000 steps on this FOD), so the streamline
    # with one is found by tracking the tckgen reference's seeds one at a time.
    fod = load_volume(DATA / "fod.nii.gz")
    seeds = np.loadtxt(DATA / "prob_ref_seeds.txt", delimiter=",", usecols=(2, 3, 4))
    scene = ProbScene(
        fod=np.ascontiguousarray(fod.data, dtype=np.float32),
        fod_affine=fod.affine,
        mask=None,
        mask_affine=fod.affine,
        step_size=0.2,
        max_steps=400,
        bidirectional=True,
        prob=Prob(0.3, 15.0, LMAX),
        run_seed=1,
    )
    x = np.array([[1.0, 0.0, 0.0]])
    truncated = []
    for i, s in enumerate(seeds):
        index = np.array([i], dtype=np.int64)
        _, stats = track_prob_core(scene, s[np.newaxis], x, index)
        if stats["truncations"]:
            truncated.append((i, stats))
    assert truncated

    i, stats = truncated[0]
    line, ref_stats = track_prob(scene, seeds[i], x[0], i)
    core, _ = track_prob_core(scene, seeds[i][np.newaxis], x, np.array([i]))
    _assert_same_walks(core, [line])
    _assert_same_sampling(stats, ref_stats, 7)
    assert ref_stats.max_truncation > 1.0


def test_first_direction_draws_until_one_is_above_the_cutoff() -> None:
    # A seed's search draws uniform directions from its own counter range and keeps
    # the first above the cutoff; the reference predicts which, and how many
    # attempts it took.
    scene = _scene(Prob(0.5, 15.0, LMAX, attempts=1000), masked=False)
    seeds, _ = _seeds()
    core, _ = track_prob_core(scene, seeds, None, INDICES)
    inv = np.linalg.inv(SMOOTH_AFFINE)
    tries = []
    for s, index, r in zip(seeds, INDICES, core, strict=True):
        if r is None:
            continue
        sh = trilinear(scene.fod, inv, s)
        draws = Draws(stream(scene.run_seed, int(index)), SEARCH_COUNTER)
        while amplitude(sh, LMAX, d := draws.unit_vector()) <= 0.5:
            pass
        np.testing.assert_array_equal(r.direction, d)
        tries.append((draws.counter - SEARCH_COUNTER) // 2)
    # Several seeds needed more than one attempt.
    assert len(tries) >= N_SEEDS // 2 and max(tries) > 1


# The binding's checks


def _track_batch(fod: NDArray[np.float32], params: _core.ProbParams) -> None:
    _core.track_batch(
        fod,
        np.eye(4),
        None,
        np.full((1, 3), 1.0),
        np.array([[1.0, 0.0, 0.0]]),
        np.zeros(1, dtype=np.int64),
        0,
        0.5,
        4,
        True,
        params,
        np.zeros((1, 5, 3), dtype=np.float32),
        1,
    )


@pytest.mark.parametrize("cutoff", [-0.1, math.nan, math.inf, -math.inf])
def test_prob_params_refuse_a_bad_cutoff(cutoff: float) -> None:
    with pytest.raises(ValueError, match="cutoff"):
        _core.ProbParams(cutoff, 15.0)


# 0.057 degrees is just under 0.001 rad, the calibration's elevation step.
@pytest.mark.parametrize("max_angle", [0.0, -15.0, 0.057, 180.01, math.nan, math.inf])
def test_prob_params_refuse_a_max_angle_out_of_range(max_angle: float) -> None:
    with pytest.raises(ValueError, match="maximum angle"):
        _core.ProbParams(0.1, max_angle)


@pytest.mark.parametrize(("trials", "attempts"), [(0, 1000), (1000, 0), (0, 0)])
def test_prob_params_refuse_zero_trials_or_attempts(trials: int, attempts: int) -> None:
    with pytest.raises(ValueError, match="trials and attempts"):
        _core.ProbParams(0.1, 15.0, trials, attempts)


@pytest.mark.parametrize(
    ("cutoff", "max_angle", "trials", "attempts"),
    [(0.0, 0.058, 1, 1), (0.1, 180.0, 1, 1), (0.1, 90.0, 1000, 1000)],
)
def test_prob_params_accept_the_edges_of_their_ranges(
    cutoff: float, max_angle: float, trials: int, attempts: int
) -> None:
    fod = np.zeros((3, 3, 3, 45), dtype=np.float32)
    _track_batch(fod, _core.ProbParams(cutoff, max_angle, trials, attempts))


@pytest.mark.parametrize("num_coeffs", [0, 2, 5, 14, 16, 44, 46, 253])
def test_prob_refuses_a_fod_that_is_not_a_full_even_series(num_coeffs: int) -> None:
    fod = np.ones((3, 3, 3, num_coeffs), dtype=np.float32)
    with pytest.raises(ValueError, match="lmax|coefficients"):
        _track_batch(fod, _core.ProbParams(0.1, 15.0))


@pytest.mark.parametrize("num_coeffs", [1, 6, 45, 231])
def test_prob_accepts_a_full_even_series(num_coeffs: int) -> None:
    fod = np.zeros((3, 3, 3, num_coeffs), dtype=np.float32)
    _track_batch(fod, _core.ProbParams(0.0, 15.0))


# Through the tracker, on the tckgen reference's FOD: a 5 mm wide bundle along +x
# through a crossing, an arc and a fade (tests/data/make_tckgen_det_ref.py), with
# an isotropic background of 0.05 below the cutoff


CUTOFF = 0.3


@functools.cache
def _bundle() -> tuple[Volume, Volume, Vector]:
    seeds = np.loadtxt(DATA / "prob_ref_seeds.txt", delimiter=",", usecols=(2, 3, 4))
    return load_volume(DATA / "fod.nii.gz"), load_volume(DATA / "mask.nii.gz"), seeds


def _prob(**kwargs: float) -> Tracker:
    options: dict[str, float] = {
        "cutoff": CUTOFF,
        "min_length": 1.0,
        "max_length": 80.0,
    }
    options.update(kwargs)
    return Tracker("prob", progress=False, **options)  # type: ignore[arg-type]


def _track(
    seeds: Seeds,
    seed: int = 1,
    n_jobs: int = 1,
    output: Path | None = None,
    tracker: Tracker | None = None,
    output_seeds: Path | None = None,
) -> TractographyRun:
    fod, mask, _ = _bundle()
    return (tracker or _prob()).track(
        fod,
        output,
        seeds,
        mask=mask,
        seed=seed,
        n_jobs=n_jobs,
        output_seeds=output_seeds,
    )


def _along_x(n: int = 256) -> Seeds:
    positions = _bundle()[2][:n]
    return Seeds(positions, np.tile([1.0, 0.0, 0.0], (len(positions), 1)))


def _streamlines(run: TractographyRun) -> Tractogram:
    assert run.tractogram is not None
    return run.tractogram


def test_same_seed_gives_the_same_file_with_one_and_all_threads(tmp_path: Path) -> None:
    files = [tmp_path / f"{n}.tck" for n in ("one", "again", "all")]
    _track(_along_x(), n_jobs=1, output=files[0])
    _track(_along_x(), n_jobs=1, output=files[1])
    _track(_along_x(), n_jobs=CORES, output=files[2])

    one = files[0].read_bytes()
    assert files[1].read_bytes() == one
    assert files[2].read_bytes() == one


def test_different_seeds_give_different_streamlines() -> None:
    a = _streamlines(_track(_along_x(32), seed=1))
    b = _streamlines(_track(_along_x(32), seed=2))

    # The seeds are the same; the walks from them are not.
    assert not (a.points.shape == b.points.shape and np.array_equal(a.points, b.points))
    assert not np.array_equal(np.diff(a.offsets), np.diff(b.offsets))


def test_seeds_without_directions_start_where_the_fod_is_above_the_cutoff(
    tmp_path: Path,
) -> None:
    # The bundle's seeds start, each along a direction its search drew from the
    # run seed; seeds in the background, below the cutoff in every direction, do
    # not.
    fod, _, positions = _bundle()
    background = np.array([[-8.0, 9.0, 0.3], [5.0, -1.0, 0.3], [-9.0, 5.0, -2.0]])
    seeds = Seeds(np.concatenate([positions[:40], background]))
    out = tmp_path / "seeds.txt"
    run = _track(seeds, seed=5, output_seeds=out)
    again = _track(seeds, seed=5)

    assert run.num_streamlines == 40
    assert run.num_rejected == {"no_start": 3, "too_short": 0}
    started = read_seeds(out)
    assert started.indices is not None and started.directions is not None
    np.testing.assert_array_equal(started.indices, np.arange(40))
    inv = np.linalg.inv(fod.affine)
    for p, d, index in zip(
        started.positions, started.directions, started.indices, strict=True
    ):
        sh = trilinear(fod.data, inv, p)
        assert amplitude(sh, LMAX, d) > CUTOFF
        draws = Draws(stream(5, int(index)), SEARCH_COUNTER)
        while amplitude(sh, LMAX, expected := draws.unit_vector()) <= CUTOFF:
            pass
        np.testing.assert_allclose(d, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(_streamlines(run).points, _streamlines(again).points)


def test_a_fod_below_the_cutoff_everywhere_starts_no_seed() -> None:
    run = _track(Seeds(_bundle()[2][:20]), tracker=_prob(cutoff=50.0))

    assert run.num_streamlines == 0
    assert run.num_rejected == {"no_start": 20, "too_short": 0}


def test_streamlines_stop_only_when_out_of_candidates_or_leaving() -> None:
    # Candidates below the cutoff are skipped, not a reason to stop, and the
    # background keeps every cone positive: no LOW_FOD, and no SHARP_TURN at all.
    status = _streamlines(_track(_along_x())).streamline_data["end_status"]
    ends = set(np.unique(status).tolist())

    assert ends <= {EndStatus.NO_DIRECTION, EndStatus.LEFT_MASK, EndStatus.LEFT_VOLUME}
    assert EndStatus.NO_DIRECTION in ends


def _fading_fod() -> Volume:
    # A lobe along +x at full amplitude in the voxels x <= 5, zero beyond.
    lobe = _lobes(np.array([[1.0, 0.0, 0.0]]), 8.0)[0]
    data = np.zeros((12, 5, 5, len(lobe)), dtype=np.float32)
    data[:6] = lobe
    return Volume(data, np.eye(4))


@pytest.mark.parametrize(
    ("cutoff", "end"),
    [(0.0, EndStatus.LOW_FOD), (0.1, EndStatus.NO_DIRECTION)],
    ids=["no-cutoff", "cutoff"],
)
def test_a_fading_fod_ends_in_no_direction_unless_the_cone_has_none(
    cutoff: float, end: EndStatus
) -> None:
    # With a cutoff, the faded lobe stops the walk where every candidate is below
    # it. Without one, acceptance does not depend on the FOD's scale, so the walk
    # goes on until the FOD is zero in the whole cone, past the last lobe voxel.
    seeds = Seeds(np.array([[2.0, 2.0, 2.0]] * 8), np.tile([1.0, 0.0, 0.0], (8, 1)))
    run = Tracker(
        "prob",
        step_size=0.25,
        cutoff=cutoff,
        min_length=0.1,
        max_length=50.0,
        bidirectional=False,
        progress=False,
    ).track(_fading_fod(), None, seeds, seed=3)
    t = _streamlines(run)
    last_x = t.points[t.offsets[1:] - 1, 0]

    assert (t.streamline_data["end_status"][:, 1] == end).all()
    if end == EndStatus.LOW_FOD:
        assert (last_x >= 6.0).all()
    else:
        assert (last_x > 5.0).all() and (last_x < 6.0).all()


def _halves_steps(t: Tractogram, n: int) -> list[int]:
    # Steps of each half, from the seed's position in each row.
    positions = _bundle()[2][:n].astype(np.float32)
    out = []
    for i in range(n):
        row = t.points[t.offsets[i] : t.offsets[i + 1]]
        at = int(np.flatnonzero(np.all(row == positions[i], axis=1))[0])
        out += [at, len(row) - 1 - at]
    return out


def test_sampling_counts_every_step_that_drew_candidates() -> None:
    # A half's first step follows the seed direction without sampling; then every
    # step after one that stays inside samples once, the accepted ones adding a
    # point unless it leaves the image or the mask, a NO_DIRECTION end none.
    run = _track(_along_x(64))
    t = _streamlines(run)
    steps = np.array(_halves_steps(t, 64))
    ends = t.streamline_data["end_status"]
    no_point = int(
        np.count_nonzero(
            np.isin(
                ends,
                [EndStatus.NO_DIRECTION, EndStatus.LEFT_VOLUME, EndStatus.LEFT_MASK],
            )
        )
    )
    sampling = run.sampling

    assert sampling is not None and run.num_streamlines == 64
    assert sampling["sampled_steps"] == int(np.sum(steps - 1)) + no_point
    assert sampling["calibration_directions"] == 7
    assert sampling["truncations"] == 0
    assert sampling["steps_per_truncation"] is None
    assert sampling["max_truncation"] is None
    mean = sampling["mean_samples_per_step"]
    assert mean is not None and 7.0 < mean < 9.0


@pytest.mark.parametrize(("angle", "directions"), [(15.0, 7), (90.0, 37)])
def test_sampling_reports_the_calibration_of_the_angle(
    angle: float, directions: int
) -> None:
    run = _track(_along_x(8), tracker=_prob(max_angle=angle))
    assert run.sampling is not None
    assert run.sampling["calibration_directions"] == directions


def test_sampling_is_none_for_det() -> None:
    run = _track(
        _along_x(8),
        tracker=Tracker("det", cutoff=CUTOFF, min_length=1.0, progress=False),
    )
    assert run.num_streamlines == 8
    assert run.sampling is None


@pytest.mark.parametrize("algorithm", ["prob", "det"])
def test_sampling_round_trips_through_json(tmp_path: Path, algorithm: str) -> None:
    tracker = Tracker(algorithm, cutoff=CUTOFF, min_length=1.0, progress=False)
    run = _track(_along_x(16), tracker=tracker)
    run.write_json(tmp_path / "run.json")
    data = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))

    assert data["sampling"] == run.sampling
    if algorithm == "prob":
        # No truncation in 16 streamlines: no steps between them to report.
        assert data["sampling"]["steps_per_truncation"] is None
    else:
        assert data["sampling"] is None
