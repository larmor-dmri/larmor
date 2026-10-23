# Tracking on FODs stored with all SH degrees. A symmetric FOD copied into the full
# layout (odd degrees zero) must track as the symmetric one; a one-sided lobe along
# +z, the m = 0 terms of every degree, is large along +z and near zero along -z.
import functools
import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from larmor import _core
from larmor.io import load_volume
from larmor.sd.sh import get_sh_indices
from larmor.tractography import EndStatus, Seeds, Tracker, TractographyRun
from larmor.types import Tractogram, Volume

DATA = Path(__file__).parents[1] / "data" / "tracking"
LMAX = 8
STEP = 0.5
SEED_POINT = np.array([[2.0, 2.0, 5.5]])
TOP = 11.5  # the last voxel centre along z of the one-sided lobe's grid


def _in_full(coeffs: NDArray[np.float32]) -> NDArray[np.float32]:
    ls, ms = get_sh_indices(LMAX)
    full = np.zeros((*coeffs.shape[:-1], (LMAX + 1) ** 2), dtype=np.float32)
    full[..., ls * ls + ls + ms] = coeffs
    return full


@functools.cache
def _bundle() -> tuple[Volume, Volume, Volume, NDArray[np.float64]]:
    fod, mask = load_volume(DATA / "fod.nii.gz"), load_volume(DATA / "mask.nii.gz")
    seeds = np.loadtxt(DATA / "prob_ref_seeds.txt", delimiter=",", usecols=(2, 3, 4))
    return fod, Volume(_in_full(fod.data), fod.affine), mask, seeds


def _one_sided() -> Volume:
    coeffs = np.zeros((LMAX + 1) ** 2, dtype=np.float32)
    for ell in range(LMAX + 1):
        coeffs[ell * ell + ell] = math.sqrt(2 * ell + 1) * math.exp(
            -ell * (ell + 1) / 30
        )
    return Volume(np.broadcast_to(coeffs, (5, 5, 12, len(coeffs))).copy(), np.eye(4))


def _streamlines(run: TractographyRun) -> Tractogram:
    assert run.tractogram is not None
    return run.tractogram


def _bundle_run(
    algorithm: str, full: bool, directions: bool = True, seed: int = 1
) -> TractographyRun:
    symmetric, asymmetric, mask, positions = _bundle()
    seeds = Seeds(
        positions,
        np.tile([1.0, 0.0, 0.0], (len(positions), 1)) if directions else None,
    )
    tracker = Tracker(
        algorithm, cutoff=0.3, min_length=1.0, max_length=80.0, progress=False
    )
    fod = asymmetric if full else symmetric
    return tracker.track(fod, None, seeds, mask=mask, seed=seed, n_jobs=1)


@pytest.mark.parametrize("directions", [True, False], ids=["given", "searched"])
def test_det_on_a_symmetric_fod_in_full_gives_the_same_streamlines(
    directions: bool,
) -> None:
    a = _streamlines(_bundle_run("det", full=False, directions=directions))
    b = _streamlines(_bundle_run("det", full=True, directions=directions))
    assert a.num_streamlines > 0
    np.testing.assert_array_equal(a.offsets, b.offsets)
    np.testing.assert_array_equal(a.points, b.points)


def test_prob_on_a_symmetric_fod_in_full_gives_the_same_distribution() -> None:
    # The two runs draw differently; over 256 seeds the mean length (~176 points,
    # spread ~45) moves by ~4 points between run seeds, and the share of each end
    # by ~0.01.
    sym_run, full_run = _bundle_run("prob", full=False), _bundle_run("prob", full=True)
    a, b = _streamlines(sym_run), _streamlines(full_run)
    assert a.num_streamlines == b.num_streamlines == 256
    mean_a, mean_b = np.diff(a.offsets).mean(), np.diff(b.offsets).mean()
    assert mean_b == pytest.approx(mean_a, rel=0.05)

    def ends(t: Tractogram) -> NDArray[np.float64]:
        status = t.streamline_data["end_status"].ravel()
        counts: NDArray[np.float64] = np.bincount(status, minlength=8) / len(status)
        return counts

    np.testing.assert_allclose(ends(a), ends(b), rtol=0, atol=0.05)
    # The full layout's bound comes from a one-sided lobe; it must stay as rarely
    # exceeded, and by as little, as the symmetric one (1 and 3 in ~44k steps here).
    for run in (sym_run, full_run):
        assert run.sampling is not None
        truncations, steps = run.sampling["truncations"], run.sampling["sampled_steps"]
        largest = run.sampling["max_truncation"]
        assert truncations is not None and steps is not None
        assert truncations <= 1e-3 * steps
        assert largest is None or largest < 1.05


@pytest.mark.parametrize("algorithm", ["det", "prob"])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_one_sided_lobe_is_followed_on_its_side_only(
    algorithm: str, sign: float
) -> None:
    tracker = Tracker(
        algorithm, step_size=STEP, min_length=0.1, max_length=50.0, progress=False
    )
    seeds = Seeds(SEED_POINT, np.array([[0.0, 0.0, sign]]))
    t = _streamlines(tracker.track(_one_sided(), None, seeds, seed=1))

    assert t.num_streamlines == 1
    z = t.points[:, 2].astype(np.float64)
    # Points run from the end of the first half to the end of the second.
    start_end, last_end = (int(s) for s in t.streamline_data["end_status"][0])
    up, up_end, down, down_end = (
        (z[0], start_end, z[-1], last_end)
        if sign > 0
        else (z[-1], last_end, z[0], start_end)
    )
    # The +z end is the last point on the grid, within a step of TOP.
    assert up > TOP - STEP and EndStatus(up_end) == EndStatus.LEFT_VOLUME
    # The -z side keeps only the seed's own step, then finds nothing to follow.
    assert down == pytest.approx(SEED_POINT[0, 2] - STEP, abs=1e-5)
    assert z.min() == down
    assert EndStatus(down_end) in (EndStatus.LOW_FOD, EndStatus.NO_DIRECTION)


@pytest.mark.parametrize("algorithm", ["det", "prob"])
def test_one_sided_lobe_tracked_one_way_along_minus_z_stops_after_the_seed_step(
    algorithm: str,
) -> None:
    tracker = Tracker(
        algorithm,
        step_size=STEP,
        min_length=0.1,
        max_length=50.0,
        bidirectional=False,
        progress=False,
    )
    seeds = Seeds(SEED_POINT, np.array([[0.0, 0.0, -1.0]]))
    t = _streamlines(tracker.track(_one_sided(), None, seeds, seed=1))
    np.testing.assert_allclose(t.points[:, 2], [5.5, 5.5 - STEP], atol=1e-5)
    assert EndStatus(int(t.streamline_data["end_status"][0, 1])) in (
        EndStatus.LOW_FOD,
        EndStatus.NO_DIRECTION,
    )


def _track_batch(
    num_coeffs: int, full: bool, params: _core.DetParams | _core.ProbParams
) -> None:
    _core.track_batch(
        np.zeros((3, 3, 3, num_coeffs), dtype=np.float32),
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
        full=full,
    )


PARAMS = [_core.DetParams(0.0, 60.0), _core.ProbParams(0.0, 15.0)]


@pytest.mark.parametrize("params", PARAMS, ids=["det", "prob"])
@pytest.mark.parametrize("num_coeffs", [4, 81, 441])
def test_track_batch_accepts_a_full_series_with_full(
    params: _core.DetParams | _core.ProbParams, num_coeffs: int
) -> None:
    _track_batch(num_coeffs, True, params)


@pytest.mark.parametrize("params", PARAMS, ids=["det", "prob"])
@pytest.mark.parametrize(("num_coeffs", "full"), [(45, True), (81, False), (484, True)])
def test_track_batch_refuses_a_count_that_does_not_fit_the_layout(
    params: _core.DetParams | _core.ProbParams, num_coeffs: int, full: bool
) -> None:
    with pytest.raises(ValueError, match="lmax|coefficients"):
        _track_batch(num_coeffs, full, params)


def _auto_cutoff(fod: Volume, point: NDArray[np.float64] = SEED_POINT) -> float:
    run = Tracker("det", cutoff="auto", progress=False).track(
        fod, None, Seeds(point, np.array([[0.0, 0.0, 1.0]])), seed=1
    )
    cutoff = run.parameters["cutoff"]
    assert isinstance(cutoff, float)
    return cutoff


def test_auto_cutoff_of_a_symmetric_fod_in_full_is_the_symmetric_one() -> None:
    symmetric, asymmetric, _, seeds = _bundle()
    # The peaks are found to ~1e-4 rad, which moves an amplitude by far less.
    assert _auto_cutoff(asymmetric, seeds[:1]) == pytest.approx(
        _auto_cutoff(symmetric, seeds[:1]), rel=1e-5
    )


def test_auto_cutoff_of_a_one_sided_lobe_is_a_fifth_of_its_peak() -> None:
    fod = _one_sided()
    peak = float(
        sum(
            math.sqrt(2 * ell + 1) * fod.data[0, 0, 0, ell * ell + ell]
            for ell in range(LMAX + 1)
        )
        / math.sqrt(4 * math.pi)
    )
    assert _auto_cutoff(fod) == pytest.approx(0.2 * peak, rel=1e-4)
