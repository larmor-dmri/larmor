# Deterministic tracking against tckgen -algorithm SD_Stream (MRtrix 3.0.8) on the
# synthetic FOD of tests/data/make_tckgen_det_ref.py: a straight bundle that goes
# through a crossing, turns along a quarter circle and fades or leaves the image,
# from 32 grid seeds along +x; going -x it leaves the mask or kinks sharply. Each
# streamline is split at its seed into the half along the seed direction ("away")
# and the half along its opposite ("back"); both tools put the away half first,
# reversed, so the seed sits between the halves.
import functools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from larmor.io import load_tractogram, load_volume
from larmor.tractography import EndStatus, Seeds, Tracker
from larmor.types import Volume

Points = NDArray[np.float64]

DATA = Path(__file__).parents[1] / "data" / "tracking"
STEP = 0.2
MAX_ANGLE = 30.0
CUTOFF = 0.3
MIN_LENGTH = 5.0
B_X = -3.0  # bundle B, which crosses the seeded bundle
MAX_LENGTHS = {"det_ref": 80.0, "det_ref_maxlen": 10.0}
# tckgen accumulates positions in float32 (pos += dir * step), rounding each step
# by up to half an ulp, ~1e-6 mm at 16 mm: up to ~1e-4 mm after 200 steps. Larmor
# steps in float64. The largest difference seen is 1.5e-5 mm.
POS_TOL = 1e-4


@dataclass(frozen=True)
class Pair:
    larmor_away: list[Points]
    larmor_back: list[Points]
    ref_away: list[Points]
    ref_back: list[Points]
    status: NDArray[np.uint8]  # Larmor's EndStatus of (away, back)


def _halves(points: Points, seed: Points) -> tuple[Points, Points]:
    at = np.flatnonzero(np.all(points == seed, axis=1))
    assert len(at) == 1
    k = int(at[0])
    return points[: k + 1][::-1], points[k:]


@functools.cache
def _pair(name: str) -> Pair:
    ref = load_tractogram(DATA / f"{name}.tck")
    track, seed_index = np.loadtxt(
        DATA / f"{name}_seeds.txt",
        delimiter=",",
        usecols=(0, 1),
        dtype=np.int64,
        unpack=True,
    )
    np.testing.assert_array_equal(track, np.arange(len(track)))
    refs = [
        ref.points[ref.offsets[i] : ref.offsets[i + 1]].astype(np.float64)
        for i in range(len(ref.offsets) - 1)
    ]
    assert len(refs) == len(seed_index) == 32
    # The seed as tckgen stored it (float32), so both start at the same point.
    seeds = np.array([r[k] for r, k in zip(refs, seed_index, strict=True)])
    # In the seed voxels every lobe is symmetric about the x axis, so +x is the
    # peak that tckgen's climb from -seed_direction 1,0,0 lands on.
    directions = np.tile([1.0, 0.0, 0.0], (len(seeds), 1))
    run = Tracker(
        "det",
        step_size=STEP,
        max_angle=MAX_ANGLE,
        cutoff=CUTOFF,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTHS[name],
        progress=False,
    ).track(
        load_volume(DATA / "fod.nii.gz"),
        None,
        Seeds(seeds, directions),
        mask=load_volume(DATA / "mask.nii.gz"),
        seed=1,
        n_jobs=1,
    )
    t = run.tractogram
    assert t is not None and run.num_streamlines == len(seeds)
    larmor = [
        t.points[t.offsets[i] : t.offsets[i + 1]].astype(np.float64)
        for i in range(len(seeds))
    ]
    larmor_halves = [_halves(p, s) for p, s in zip(larmor, seeds, strict=True)]
    ref_halves = [_halves(p, s) for p, s in zip(refs, seeds, strict=True)]
    return Pair(
        larmor_away=[h[0] for h in larmor_halves],
        larmor_back=[h[1] for h in larmor_halves],
        ref_away=[h[0] for h in ref_halves],
        ref_back=[h[1] for h in ref_halves],
        status=t.streamline_data["end_status"],
    )


@pytest.fixture(params=sorted(MAX_LENGTHS))
def pair(request: pytest.FixtureRequest) -> Pair:
    return _pair(request.param)


def _header(name: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    with open(DATA / f"{name}.tck", "rb") as f:
        for raw in f:
            line = raw.decode("latin-1").strip()
            if line == "END":
                return fields
            key, _, value = line.partition(": ")
            fields[key] = value
    raise AssertionError("no END in the header")


def _voxel(volume: Volume, point: Points) -> Points:
    inv = np.linalg.inv(volume.affine)
    voxel: Points = inv[:3, :3] @ point + inv[:3, 3]
    return voxel


def _in_mask(mask: Volume, point: Points) -> bool:
    # tckgen's rule for a mask: the nearest voxel, outside the image is outside.
    v = np.rint(_voxel(mask, point)).astype(np.int64)
    if np.any(v < 0) or np.any(v >= mask.data.shape):
        return False
    return bool(mask.data[v[0], v[1], v[2]])


@pytest.mark.parametrize("name", sorted(MAX_LENGTHS))
def test_reference_was_made_with_the_parameters_tracked_here(name: str) -> None:
    header = _header(name)

    assert header["method"] == "SDStream"
    assert header["init_direction"] == "1,0,0"
    assert float(header["step_size"]) == pytest.approx(STEP)
    assert float(header["max_angle"]) == MAX_ANGLE
    assert float(header["threshold"]) == pytest.approx(CUTOFF)
    assert float(header["min_dist"]) == MIN_LENGTH
    assert float(header["max_dist"]) == MAX_LENGTHS[name]


def test_reference_goes_through_the_crossing_and_around_the_arc() -> None:
    # Guards the reference itself: one that missed the crossing or the turn would
    # weaken the point-by-point test without failing it.
    away = _pair("det_ref").ref_away
    crossed = [bool(np.any(np.abs(a[:, 0] - B_X) < 2.5)) for a in away]
    turned = [float(a[-1, 1] - a[-2, 1]) > 0.99 * STEP for a in away]

    assert all(crossed)
    assert sum(turned) >= 16


def test_streamlines_match_tckgen_point_by_point(pair: Pair) -> None:
    for ours, theirs in [
        (pair.larmor_away, pair.ref_away),
        (pair.larmor_back, pair.ref_back),
    ]:
        for a, b in zip(ours, theirs, strict=True):
            n = min(len(a), len(b))
            np.testing.assert_allclose(a[:n], b[:n], rtol=0, atol=POS_TOL)


def _ends(pair: Pair, status: EndStatus) -> list[tuple[Points, Points]]:
    # (Larmor half, tckgen half) of every end that Larmor stopped with this status.
    away = [
        (a, b)
        for a, b, s in zip(
            pair.larmor_away, pair.ref_away, pair.status[:, 0], strict=True
        )
        if s == status
    ]
    back = [
        (a, b)
        for a, b, s in zip(
            pair.larmor_back, pair.ref_back, pair.status[:, 1], strict=True
        )
        if s == status
    ]
    return away + back


@pytest.mark.parametrize(
    "status", [EndStatus.LOW_FOD, EndStatus.SHARP_TURN], ids=lambda s: s.name
)
def test_an_end_where_no_step_is_taken_has_the_same_points_as_tckgen(
    status: EndStatus,
) -> None:
    # Neither tool adds a point when the algorithm refuses to step.
    ends = _ends(_pair("det_ref"), status)

    assert ends
    assert all(len(a) == len(b) for a, b in ends)


@pytest.mark.parametrize(
    "status", [EndStatus.LEFT_VOLUME, EndStatus.LEFT_MASK], ids=lambda s: s.name
)
def test_an_end_that_leaves_drops_the_point_outside_as_tckgen_does(
    status: EndStatus,
) -> None:
    # Leaving the FOD also leaves the mask, which is on the same grid: tckgen
    # stops with EXIT_MASK and drops the point in both cases, and so does Larmor.
    mask = load_volume(DATA / "mask.nii.gz")
    ends = _ends(_pair("det_ref"), status)

    assert ends
    for a, b in ends:
        assert len(a) == len(b)
        assert _in_mask(mask, a[-1]) and _in_mask(mask, b[-1])


def test_at_the_maximum_length_tckgen_takes_more_steps() -> None:
    # Larmor stops after ceil(max length / step) steps and does not start the back
    # half. tckgen stops a half once the streamline has 3 + ceil(max length /
    # step) points, then still takes one step of the back half.
    pair = _pair("det_ref_maxlen")
    steps = math.ceil(MAX_LENGTHS["det_ref_maxlen"] / STEP)

    assert np.all(pair.status == EndStatus.MAX_LENGTH)
    assert {len(a) for a in pair.larmor_away} == {steps + 1}
    assert {len(a) for a in pair.larmor_back} == {1}
    assert {len(a) for a in pair.ref_away} == {steps + 3}
    assert {len(a) for a in pair.ref_back} == {2}
