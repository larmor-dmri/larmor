# The tracker on a 10x10x10 FOD with dummy steps from _core.TestAlgorithm, by
# default a random walk whose seeds' direction search always fails. Lengths vary
# with where the walk leaves the grid, so the keep rule, the counts and the
# batching are exercised; the seed decides every random number, so runs can be
# compared bit for bit.
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from larmor import _core
from larmor.io import load_tractogram
from larmor.sd import convert_sh_basis
from larmor.tractography import (
    EndStatus,
    MaskSeeds,
    Seeds,
    Tracker,
    TractographyRun,
    read_seeds,
    tracker,
)
from larmor.tractography.termination import auto_cutoff
from larmor.tractography.tracker import _Algorithm, _inside_mask
from larmor.types import Tractogram, Volume

GATE = 1  # A tournier index that moves to another place in the descoteaux basis.
SEED = 7
CORES = os.cpu_count() or 1


def _algorithm(step: str = "wander", stop: EndStatus | None = None) -> _Algorithm:
    return _Algorithm(
        name="test",
        default_max_angle=45.0,
        make_params=lambda cutoff, angle: _core.TestAlgorithm(
            step, None if stop is None else int(stop), -math.inf
        ),
    )


def _tracker(
    step: str = "wander", stop: EndStatus | None = None, **kwargs: float | bool | str
) -> Tracker:
    options: dict[str, float | bool | str] = {
        "step_size": 0.5,
        "cutoff": 0.0,
        "min_length": 3.0,
        "max_length": 20.0,
        "progress": False,
    }
    options.update(kwargs)
    return Tracker(_algorithm(step, stop), **options)  # type: ignore[arg-type]


def _fod(
    zooms: tuple[float, float, float] = (1.0, 1.0, 1.0), split: bool = False
) -> Volume:
    coeffs = np.zeros((10, 10, 10, 6), dtype=np.float32)
    coeffs[..., 0] = 1.0
    coeffs[..., GATE] = 1.0
    if split:
        coeffs[6:, :, :, GATE] = -1.0
    return Volume(coeffs, np.diag([*zooms, 1.0]))


def _mask(lo: int, hi: int) -> Volume:
    data = np.zeros((10, 10, 10), dtype=np.float32)
    data[lo : hi + 1] = 1.0
    return Volume(data, np.eye(4))


def _fixed(n: int, low: float = 1.0, high: float = 8.0) -> Seeds:
    rng = np.random.default_rng(n)
    return Seeds(rng.uniform(low, high, (n, 3)), rng.normal(size=(n, 3)))


@dataclass
class BoxSeeds:
    # Endless seeds in a box wider than the grid, so some never start.
    low: float = -2.0
    high: float = 11.0
    drawn: list[NDArray[np.float64]] = field(default_factory=list)

    def draw(self, n: int, rng: np.random.Generator) -> Seeds:
        seeds = Seeds(rng.uniform(self.low, self.high, (n, 3)), rng.normal(size=(n, 3)))
        self.drawn.append(seeds.positions)
        return seeds


class DirectedMaskSeeds(MaskSeeds):
    def draw(self, n: int, rng: np.random.Generator) -> Seeds:
        return Seeds(super().draw(n, rng).positions, rng.normal(size=(n, 3)))


def _streamlines(run: TractographyRun) -> Tractogram:
    assert run.tractogram is not None
    return run.tractogram


def _assert_same(a: Tractogram, b: Tractogram) -> None:
    np.testing.assert_array_equal(a.points, b.points)
    np.testing.assert_array_equal(a.offsets, b.offsets)
    np.testing.assert_array_equal(
        a.streamline_data["end_status"], b.streamline_data["end_status"]
    )


def _adds_up(run: TractographyRun) -> bool:
    rejected = run.num_rejected["no_start"] + run.num_rejected["too_short"]
    return run.num_seeds == run.num_streamlines + rejected


def _warned(caplog: pytest.LogCaptureFixture) -> str:
    return " ".join(
        r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
    )


@pytest.fixture
def small_batches(monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(tracker, "MAX_BATCH", 7)
    return 7


# Constructor


@pytest.mark.parametrize("name", ["step_size", "min_length", "max_length"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_lengths_must_be_positive_and_finite(name: str, value: float) -> None:
    with pytest.raises(ValueError, match="must be positive and finite"):
        Tracker(_algorithm(), **{name: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("cutoff", [-1.0, math.nan, math.inf, "high"])
def test_bad_cutoff_is_refused(cutoff: float | str) -> None:
    with pytest.raises(ValueError, match="cutoff"):
        Tracker(_algorithm(), cutoff=cutoff)


@pytest.mark.parametrize("cutoff", [0.0, 0, 0.3, "auto"])
def test_zero_positive_or_auto_cutoff_is_accepted(cutoff: float | str) -> None:
    assert Tracker(_algorithm(), cutoff=cutoff).cutoff == cutoff


@pytest.mark.parametrize("angle", [0.0, -10.0, 180.5, math.nan])
def test_max_angle_out_of_range_is_refused(angle: float) -> None:
    with pytest.raises(ValueError, match="max angle"):
        Tracker(_algorithm(), max_angle=angle)


@pytest.mark.parametrize("angle", [1e-3, 90.0, 180.0])
def test_max_angle_in_range_is_accepted(angle: float) -> None:
    assert Tracker(_algorithm(), max_angle=angle).max_angle == angle


@pytest.mark.parametrize(("low", "high"), [(10.0, 10.0), (20.0, 10.0)])
def test_min_length_not_below_max_length_is_refused(low: float, high: float) -> None:
    with pytest.raises(ValueError, match="min length >= max length"):
        Tracker(_algorithm(), min_length=low, max_length=high)


def test_unknown_algorithm_name_lists_the_known_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    algo = _algorithm()
    monkeypatch.setattr(tracker, "ALGORITHMS", {"walk": algo, "b": algo})
    with pytest.raises(ValueError, match="known ones are: b, walk"):
        Tracker("ifod")
    assert Tracker("walk").algorithm is algo


def test_det_is_registered_with_core_parameters() -> None:
    det = tracker.ALGORITHMS["det"]
    assert det.name == "det" and det.default_max_angle is None
    assert det.default_min_radius == 0.65
    assert isinstance(det.make_params(0.1, 45.0), _core.DetParams)


def test_unknown_algorithm_with_empty_registry_says_none_yet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracker, "ALGORITHMS", {})
    with pytest.raises(ValueError, match=r"Unknown algorithm 'ifod'.*none yet"):
        Tracker("ifod")


# Refusals and warnings before tracking


@pytest.mark.parametrize("n", [0, -1])
def test_non_positive_target_is_refused(tmp_path: Path, n: int) -> None:
    out = tmp_path / "t.tck"
    with pytest.raises(ValueError, match="Cannot seed"):
        _tracker().track(_fod(), out, _fixed(5), num_streamlines=n)
    assert not out.exists()


def test_endless_source_without_target_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "t.tck"
    with pytest.raises(ValueError, match="need num_streamlines"):
        _tracker().track(_fod(), out, BoxSeeds())
    assert not out.exists()


def test_4d_mask_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "t.tck"
    mask = Volume(np.ones((10, 10, 10, 1), dtype=np.float32), np.eye(4))
    with pytest.raises(ValueError, match="3D mask"):
        _tracker().track(_fod(), out, _fixed(5), mask=mask)
    assert not out.exists()


@pytest.mark.parametrize(
    ("seeds", "target"),
    [(Seeds(np.full((3, 3), 5.0)), None), (MaskSeeds(_mask(2, 4)), 2)],
    ids=["fixed", "endless"],
)
def test_seeds_whose_direction_search_fails_do_not_start(
    tmp_path: Path, seeds: Seeds | MaskSeeds, target: int | None
) -> None:
    # The walk's search never finds a direction.
    seeds_out = tmp_path / "seeds.txt"
    run = _tracker().track(
        _fod(), None, seeds, num_streamlines=target, seed=SEED, output_seeds=seeds_out
    )
    assert run.num_streamlines == _streamlines(run).num_streamlines == 0
    expected = 3 if target is None else tracker.SEEDS_PER_STREAMLINE * target
    assert run.num_seeds == run.num_rejected["no_start"] == expected
    assert len(read_seeds(seeds_out)) == 0


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_bad_run_seed_is_refused_before_outputs_are_touched(
    tmp_path: Path, seed: int
) -> None:
    out, seeds_out = tmp_path / "t.tck", tmp_path / "seeds.txt"
    out.write_bytes(b"previous run")
    seeds_out.write_text("previous seeds")
    with pytest.raises(ValueError, match="run seed"):
        _tracker().track(_fod(), out, _fixed(5), seed=seed, output_seeds=seeds_out)
    assert out.read_bytes() == b"previous run"
    assert seeds_out.read_text() == "previous seeds"


def _det(cutoff: float | str = 0.0) -> Tracker:
    return Tracker(
        "det",
        step_size=0.5,
        min_length=0.5,
        max_length=5.0,
        cutoff=cutoff,
        progress=False,
    )


def _fod_of_lmax(lmax: int) -> Volume:
    coeffs = np.zeros((10, 10, 10, (lmax + 1) * (lmax + 2) // 2), dtype=np.float32)
    coeffs[..., 0] = 1.0
    coeffs[..., GATE] = 1.0
    return Volume(coeffs, np.eye(4))


@pytest.mark.parametrize("cutoff", [0.1, "auto"])
def test_fod_above_max_lmax_is_refused_before_outputs_are_touched(
    tmp_path: Path, cutoff: float | str
) -> None:
    out, seeds_out = tmp_path / "t.tck", tmp_path / "seeds.txt"
    out.write_bytes(b"previous run")
    seeds_out.write_text("previous seeds")
    fod = _fod_of_lmax(_core.MAX_LMAX + 2)
    with pytest.raises(ValueError, match="lmax must be at most"):
        _det(cutoff).track(fod, out, _fixed(2), output_seeds=seeds_out)
    assert out.read_bytes() == b"previous run"
    assert seeds_out.read_text() == "previous seeds"


def test_full_size_fod_is_tracked() -> None:
    coeffs = np.zeros((10, 10, 10, 81), dtype=np.float32)
    coeffs[..., 0] = 1.0
    coeffs[..., 6] = 1.0  # l = 2, m = 0 in the full layout: a lobe along z
    run = _det().track(Volume(coeffs, np.eye(4)), None, _fixed(2, 4.0, 6.0), seed=SEED)
    assert run.num_streamlines == 2


def test_fod_of_max_lmax_is_tracked(tmp_path: Path) -> None:
    out = tmp_path / "t.tck"
    run = _det().track(
        _fod_of_lmax(_core.MAX_LMAX), out, _fixed(2, 4.0, 6.0), seed=SEED
    )
    assert run.num_streamlines == load_tractogram(out).num_streamlines == 2


def test_numpy_run_seed_is_recorded_as_a_plain_int() -> None:
    run = _tracker().track(_fod(), None, _fixed(5), seed=np.int64(3))  # type: ignore[arg-type]
    assert type(run.parameters["seed"]) is int
    json.dumps(run.parameters)


@pytest.mark.parametrize("n", [0, -2, -10_000])
def test_thread_count_out_of_range_is_refused(n: int) -> None:
    with pytest.raises(ValueError, match="n_jobs must be -1"):
        _tracker().track(_fod(), None, _fixed(5), n_jobs=n)


@pytest.mark.parametrize(("n", "used"), [(1, 1), (CORES, CORES), (CORES + 1, CORES)])
def test_thread_count_is_capped_at_the_number_of_cores(n: int, used: int) -> None:
    run = _tracker().track(_fod(), None, _fixed(5), n_jobs=n)
    assert run.parameters["num_threads"] == used


def test_every_core_is_used_by_default() -> None:
    assert _tracker().track(_fod(), None, _fixed(5)).parameters["num_threads"] == CORES


def test_resolved_min_length_not_below_max_length_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "t.tck"
    with pytest.raises(ValueError, match="min_length"):
        Tracker(_algorithm(), cutoff=0.0, min_length=200.0).track(
            _fod(), out, _fixed(5)
        )
    assert not out.exists()


def _far_mask() -> Volume:
    affine = np.eye(4)
    affine[:3, 3] = 50.0
    return Volume(np.ones((3, 3, 3), dtype=np.float32), affine)


@pytest.mark.parametrize(
    "seeds",
    [Seeds(np.full((4, 3), 20.0), np.ones((4, 3))), DirectedMaskSeeds(_far_mask())],
    ids=["fixed", "mask"],
)
def test_no_seed_on_the_fod_grid_is_refused(
    tmp_path: Path, seeds: Seeds | MaskSeeds
) -> None:
    out = tmp_path / "t.tck"
    with pytest.raises(ValueError, match="No seed lies on the FOD's grid"):
        _tracker().track(_fod(), out, seeds, num_streamlines=5)
    assert not out.exists()


@pytest.mark.parametrize(
    "seeds",
    [_fixed(5, 1.0, 3.0), DirectedMaskSeeds(_mask(1, 3))],
    ids=["fixed", "mask"],
)
def test_no_seed_in_the_mask_is_refused(
    tmp_path: Path, seeds: Seeds | MaskSeeds
) -> None:
    out = tmp_path / "t.tck"
    with pytest.raises(ValueError, match="No seed lies in the mask"):
        _tracker().track(_fod(), out, seeds, num_streamlines=5, mask=_mask(6, 8))
    assert not out.exists()


def test_seeds_partly_off_the_grid_warn_with_their_share(
    caplog: pytest.LogCaptureFixture,
) -> None:
    positions = np.array([[5.0, 5, 5], [5, 5, 6], [5, 5, 30], [5, 30, 5]])
    seeds = Seeds(positions, np.ones((4, 3)))
    run = _tracker().track(_fod(), None, seeds)
    assert "50% of the seeds do not lie on the FOD's" in _warned(caplog)
    assert run.num_rejected["no_start"] >= 2


@pytest.mark.parametrize(
    "seeds",
    [_fixed(8, 2.0, 7.0), DirectedMaskSeeds(_mask(2, 7))],
    ids=["fixed", "mask"],
)
def test_seeds_partly_outside_the_mask_warn(
    caplog: pytest.LogCaptureFixture, seeds: Seeds | MaskSeeds
) -> None:
    target = None if isinstance(seeds, Seeds) else 3
    _tracker().track(
        _fod(), None, seeds, num_streamlines=target, mask=_mask(5, 9), seed=SEED
    )
    assert "% of the seeds do not lie in the mask" in _warned(caplog)


# Parameters


def test_defaults_are_resolved_from_the_mean_voxel_size_and_recorded() -> None:
    algo = _algorithm()
    run = Tracker(algo, cutoff=0.0, progress=False).track(
        _fod((1.0, 2.0, 3.0)), None, _fixed(3, 1.0, 5.0)
    )
    p = run.parameters
    assert p["step_size"] == pytest.approx(0.2)
    assert p["min_length"] == pytest.approx(10.0)
    assert p["max_length"] == pytest.approx(200.0)
    assert p["max_steps"] == 1000
    assert p["max_angle"] == algo.default_max_angle
    assert p["batch_size"] == tracker.BUFFER_BYTES // (1001 * 12)
    assert p["cutoff"] == 0.0 and p["cutoff_mode"] == "given"
    assert p["num_streamlines_target"] is None and p["mask"] is False
    assert p["algorithm"] == "test" and p["basis"] == "tournier"


@pytest.mark.parametrize(
    ("step", "angle"),
    [
        (None, 2 * math.degrees(math.asin(0.2 / 2.6))),
        (1.0, 2 * math.degrees(math.asin(1.0 / 2.6))),
        (3.0, 180.0),
    ],
)
def test_det_default_angle_keeps_the_radius_of_curvature_at_0_65_voxel(
    step: float | None, angle: float
) -> None:
    prep = Tracker("det", step_size=step, cutoff=0.0, progress=False)._prepare(
        _fod((2.0, 2.0, 2.0)), "tournier", None
    )
    assert prep.max_angle == pytest.approx(angle)
    if angle < 180:
        assert prep.min_radius == pytest.approx(1.3)


@pytest.mark.parametrize(("algorithm", "angle"), [("det", 60.0), ("prob", 15.0)])
def test_given_or_fixed_angle_is_kept_and_its_radius_recorded(
    algorithm: str, angle: float
) -> None:
    given = 60.0 if algorithm == "det" else None
    prep = Tracker(algorithm, max_angle=given, cutoff=0.0, progress=False)._prepare(
        _fod((2.0, 2.0, 2.0)), "tournier", None
    )
    assert prep.max_angle == angle
    assert prep.min_radius == pytest.approx(
        0.2 / (2 * math.sin(math.radians(angle / 2)))
    )


@pytest.mark.parametrize("voxel", [1.45, 2.9, 1.0, 2.5])
def test_max_steps_ignore_float_noise(voxel: float) -> None:
    # With 1.45 or 2.9 mm voxels, (100 x voxel) / (0.1 x voxel) is
    # 1000.0000000000001 in floating point, which is still 1000 steps.
    run = Tracker(_algorithm(), cutoff=0.0, progress=False).track(
        _fod((voxel, voxel, voxel)), None, _fixed(1, 0.0, 0.5)
    )
    assert run.parameters["max_steps"] == 1000


@pytest.mark.parametrize(
    ("step", "max_length"), [(0.1, 250.0), (0.05, 250.0), (1.0, 20.0), (0.2, 300.0)]
)
def test_the_buffer_stays_under_its_size(step: float, max_length: float) -> None:
    p = Tracker(
        _algorithm(),
        step_size=step,
        max_length=max_length,
        min_length=step,
        cutoff=0.0,
        progress=False,
    )._prepare(_fod(), "tournier", None)
    assert p.batch_size * (p.max_steps + 1) * 3 * 4 <= tracker.BUFFER_BYTES
    assert 1 <= p.batch_size <= tracker.MAX_BATCH


def test_fod_is_tracked_on_its_scaled_values() -> None:
    fod = _fod()
    scaled = Volume(fod.data, fod.affine, slope=2.0, intercept=0.5)
    p = _tracker()._prepare(scaled, "tournier", None)
    np.testing.assert_allclose(p.coeffs, fod.data * 2.0 + 0.5)


def test_batch_size_is_at_least_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracker, "BUFFER_BYTES", 10)
    run = _tracker().track(_fod(), None, _fixed(3))
    assert run.parameters["batch_size"] == 1
    assert run.num_seeds == 3


def test_auto_cutoff_is_measured_on_the_fod_in_tournier() -> None:
    fod = _fod(split=True)
    descoteaux = fod.with_data(convert_sh_basis(fod.data, "tournier", "descoteaux"))
    run = _tracker(cutoff="auto").track(descoteaux, None, _fixed(3), basis="descoteaux")
    assert run.parameters["cutoff"] == auto_cutoff(np.asarray(fod.data))
    assert run.parameters["cutoff_mode"] == "auto"


def test_auto_cutoff_is_measured_inside_the_tracking_mask() -> None:
    # Strong FODs in the mask (x from 2 to 5) and weak ones outside it, which
    # would pull the mean down if they were counted.
    fod = _fod()
    coeffs = np.array(fod.data)
    coeffs[2:6] *= 4.0
    mask = _mask(2, 5)
    run = _tracker(cutoff="auto").track(
        fod.with_data(coeffs), None, _fixed(3, 2.0, 5.0), mask=mask
    )
    assert run.parameters["cutoff"] == pytest.approx(auto_cutoff(coeffs[2:6]))
    assert run.parameters["cutoff"] != pytest.approx(auto_cutoff(coeffs))


def test_every_seed_of_a_one_voxel_thick_mask_starts() -> None:
    data = np.zeros((10, 10, 10), dtype=np.float32)
    data[2:8, 5, 5] = 1.0
    mask = Volume(data, np.eye(4))
    run = _tracker().track(
        _fod(), None, DirectedMaskSeeds(mask), num_streamlines=300, mask=mask
    )
    assert run.num_rejected["no_start"] == 0


def test_drawn_run_seed_is_recorded_and_reproduces_the_run() -> None:
    first = _tracker().track(_fod(), None, BoxSeeds(), num_streamlines=20)
    run_seed = first.parameters["seed"]
    assert isinstance(run_seed, int)
    again = _tracker().track(
        _fod(), None, BoxSeeds(), num_streamlines=20, seed=run_seed
    )
    _assert_same(_streamlines(first), _streamlines(again))


# Counts and the keep rule


def test_target_is_reached_exactly_and_the_counts_add_up(
    caplog: pytest.LogCaptureFixture, small_batches: int
) -> None:
    run = _tracker().track(_fod(), None, BoxSeeds(), num_streamlines=30, seed=SEED)
    assert "Stopped" not in _warned(caplog)
    assert run.num_streamlines == _streamlines(run).num_streamlines == 30
    assert run.num_rejected["no_start"] > 0 and run.num_rejected["too_short"] > 0
    assert _adds_up(run)


def test_seeds_after_the_last_kept_are_not_counted(tmp_path: Path) -> None:
    # One batch holds every seed, so the cut drops the end of it.
    seeds_file = tmp_path / "seeds.txt"
    run = _tracker().track(
        _fod(), None, BoxSeeds(), num_streamlines=30, seed=SEED, output_seeds=seeds_file
    )
    kept = read_seeds(seeds_file)
    batch_size = run.parameters["batch_size"]
    assert kept.indices is not None and isinstance(batch_size, int)
    assert run.num_seeds == kept.indices[-1] + 1 < batch_size
    assert _adds_up(run)


def test_fixed_seeds_without_target_are_all_tracked() -> None:
    run = _tracker(min_length=8.0).track(_fod(), None, _fixed(40), seed=SEED)
    assert run.num_seeds == 40
    assert run.num_rejected["too_short"] > 0
    assert _adds_up(run)


def test_running_out_of_fixed_seeds_before_the_target_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    run = _tracker().track(_fod(), None, _fixed(20), num_streamlines=1000, seed=SEED)
    assert re.search(r"seeds ran out with \d+ of 1000", _warned(caplog))
    assert run.num_seeds == 20
    assert 0 < run.num_streamlines <= 20


def test_run_stops_after_too_many_seeds_per_streamline(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    small_batches: int,
) -> None:
    # 5 x 2 = 10 seeds at most: batches of 7, then 3, not two full batches of 7.
    monkeypatch.setattr(tracker, "SEEDS_PER_STREAMLINE", 5)
    long = _tracker(min_length=50.0, max_length=60.0)
    run = long.track(_fod(), None, BoxSeeds(), num_streamlines=2, seed=SEED)
    assert "Stopped after 10 seeds with 0 of 2" in _warned(caplog)
    assert run.num_seeds == 10 and run.num_streamlines == 0
    assert _streamlines(run).num_streamlines == 0
    assert _adds_up(run)


def test_kept_streamlines_are_at_least_min_length() -> None:
    run = _tracker(min_length=8.0).track(_fod(), None, _fixed(60), seed=SEED)
    lengths = (np.diff(_streamlines(run).offsets) - 1) * 0.5
    assert run.num_rejected["too_short"] > 0
    assert lengths.min() >= 8.0


@pytest.mark.parametrize(("min_length", "kept"), [(9.0, 1), (9.5, 0)])
def test_streamline_of_exactly_min_length_is_kept(min_length: float, kept: int) -> None:
    # 1 mm steps along x from (5, 5, 5): 10 points from x = 9 to x = 0, 9 mm.
    straight = _tracker(
        "straight", step_size=1.0, min_length=min_length, max_length=50.0
    )
    seeds = Seeds(np.array([[5.0, 5.0, 5.0]]), np.array([[1.0, 0.0, 0.0]]))
    run = straight.track(_fod(), None, seeds)
    assert run.num_streamlines == kept
    assert run.num_rejected["too_short"] == 1 - kept


# Batches and reproducibility


@pytest.mark.parametrize("target", [None, 20])
def test_several_batches_give_the_streamlines_of_one(
    monkeypatch: pytest.MonkeyPatch, target: int | None
) -> None:
    seeds = _fixed(50)
    one = _tracker().track(_fod(), None, seeds, num_streamlines=target, seed=SEED)
    monkeypatch.setattr(tracker, "MAX_BATCH", 7)
    several = _tracker().track(_fod(), None, seeds, num_streamlines=target, seed=SEED)
    assert several.parameters["batch_size"] == 7
    assert one.num_seeds == several.num_seeds
    _assert_same(_streamlines(one), _streamlines(several))


def test_endless_batches_are_drawn_from_the_run_seed_and_batch_index(
    small_batches: int,
) -> None:
    source = BoxSeeds()
    _tracker().track(_fod(), None, source, num_streamlines=20, seed=SEED)
    assert len(source.drawn) > 2
    for k, positions in enumerate(source.drawn):
        expected = BoxSeeds().draw(7, np.random.default_rng([SEED, k]))
        np.testing.assert_array_equal(positions, expected.positions)


@pytest.mark.parametrize(("other", "same"), [(SEED, True), (SEED + 1, False)])
def test_run_seed_fixes_the_output(other: int, same: bool) -> None:
    a = _streamlines(
        _tracker().track(_fod(), None, BoxSeeds(), num_streamlines=20, seed=SEED)
    )
    b = _streamlines(
        _tracker().track(_fod(), None, BoxSeeds(), num_streamlines=20, seed=other)
    )
    equal = a.points.shape == b.points.shape and np.array_equal(a.points, b.points)
    assert equal == same


def test_random_numbers_follow_the_run_seed_for_fixed_seeds() -> None:
    a = _streamlines(_tracker().track(_fod(), None, _fixed(20), seed=SEED))
    b = _streamlines(_tracker().track(_fod(), None, _fixed(20), seed=SEED + 1))
    assert not (a.points.shape == b.points.shape and np.array_equal(a.points, b.points))


def test_number_of_threads_does_not_change_the_output() -> None:
    n_max = CORES
    one = _tracker().track(
        _fod(), None, BoxSeeds(), num_streamlines=40, seed=SEED, n_jobs=1
    )
    many = _tracker().track(
        _fod(), None, BoxSeeds(), num_streamlines=40, seed=SEED, n_jobs=n_max
    )
    assert (
        one.parameters["num_threads"] == 1 and many.parameters["num_threads"] == n_max
    )
    _assert_same(_streamlines(one), _streamlines(many))


# Outputs


@pytest.mark.parametrize("suffix", [".tck", ".trk"])
def test_written_tractogram_matches_the_one_in_memory(
    tmp_path: Path, small_batches: int, suffix: str
) -> None:
    fod = _fod()
    path = tmp_path / f"t{suffix}"
    written = _tracker().track(fod, path, BoxSeeds(), num_streamlines=30, seed=SEED)
    memory = _tracker().track(fod, None, BoxSeeds(), num_streamlines=30, seed=SEED)
    assert written.tractogram is None and written.path == path
    loaded = load_tractogram(path, reference=fod.grid)
    expected = _streamlines(memory)
    np.testing.assert_array_equal(loaded.offsets, expected.offsets)
    if suffix == ".tck":
        np.testing.assert_array_equal(loaded.points, expected.points)
    else:
        # TRK stores voxel mm, one float32 rounding away from the world points.
        np.testing.assert_allclose(loaded.points, expected.points, atol=1e-5)
        np.testing.assert_array_equal(
            loaded.streamline_data["end_status"],
            expected.streamline_data["end_status"],
        )


def test_seeds_file_has_one_header_and_one_row_per_streamline(
    tmp_path: Path, small_batches: int
) -> None:
    seeds_file = tmp_path / "seeds.txt"
    run = _tracker().track(
        _fod(), None, BoxSeeds(), num_streamlines=30, seed=SEED, output_seeds=seeds_file
    )
    lines = seeds_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# index x y z dx dy dz"
    assert sum(line.startswith("#") for line in lines) == 1
    assert len(lines) == 1 + run.num_streamlines == 31
    assert run.seeds_path == seeds_file


def test_seeds_file_tracked_again_gives_the_same_streamlines(
    tmp_path: Path, small_batches: int
) -> None:
    seeds_file = tmp_path / "seeds.txt"
    first = _tracker().track(
        _fod(), None, BoxSeeds(), num_streamlines=30, seed=SEED, output_seeds=seeds_file
    )
    again = _tracker().track(_fod(), None, read_seeds(seeds_file), seed=SEED)
    assert again.num_seeds == again.num_streamlines == 30
    _assert_same(_streamlines(first), _streamlines(again))


@pytest.mark.parametrize("bidirectional", [True, False])
def test_unidirectional_streamlines_start_at_their_seed(
    tmp_path: Path, bidirectional: bool
) -> None:
    seeds_file = tmp_path / "seeds.txt"
    run = _tracker(bidirectional=bidirectional).track(
        _fod(), None, _fixed(30), seed=SEED, output_seeds=seeds_file
    )
    status = _streamlines(run).streamline_data["end_status"]
    first_points = _streamlines(run).points[_streamlines(run).offsets[:-1]]
    on_seed = np.all(
        first_points == read_seeds(seeds_file).positions.astype(np.float32), axis=1
    )
    assert run.num_streamlines > 0
    if bidirectional:
        assert (status != EndStatus.NONE).all()
        assert not on_seed.any()
    else:
        assert (status[:, 0] == EndStatus.NONE).all()
        assert (status[:, 1] != EndStatus.NONE).all()
        assert on_seed.all()


@pytest.mark.parametrize("files", [False, True])
def test_run_record_round_trips_through_json(tmp_path: Path, files: bool) -> None:
    out = tmp_path / "t.tck" if files else None
    seeds_file = tmp_path / "seeds.txt" if files else None
    run = _tracker().track(
        _fod(), out, BoxSeeds(), num_streamlines=10, seed=SEED, output_seeds=seeds_file
    )
    run.write_json(tmp_path / "run.json")
    data = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert "tractogram" not in data
    assert data["path"] == (str(out) if files else None)
    assert data["seeds_path"] == (str(seeds_file) if files else None)
    assert data["parameters"] == run.parameters
    assert data["num_seeds"] == run.num_seeds
    assert data["num_streamlines"] == run.num_streamlines == 10
    assert data["num_rejected"] == run.num_rejected
    # The tracker only sees arrays; the caller records its files.
    assert run.inputs is None
    assert data["inputs"] is None


# FOD basis


def test_fod_in_another_basis_tracks_as_in_tournier() -> None:
    # Det follows the FOD's peaks, which move if a coefficient is read in the
    # wrong place.
    fod = _fod(split=True)
    descoteaux = fod.with_data(convert_sh_basis(fod.data, "tournier", "descoteaux"))
    det = Tracker(
        "det",
        step_size=0.5,
        min_length=0.5,
        max_length=20.0,
        cutoff=0.0,
        progress=False,
    )
    seeds = _fixed(20, 1.0, 5.0)
    expected = _streamlines(det.track(fod, None, seeds, seed=SEED))
    converted = det.track(descoteaux, None, seeds, seed=SEED, basis="descoteaux")
    assert expected.num_streamlines == 20
    _assert_same(expected, _streamlines(converted))
    misread = _streamlines(det.track(descoteaux, None, seeds, seed=SEED))
    assert not (
        misread.points.shape == expected.points.shape
        and np.array_equal(misread.points, expected.points)
    )


def test_python_mask_check_reads_the_nearest_voxel() -> None:
    data = np.zeros((3, 3, 3), np.float32)
    data[1, 1, 1] = 1.0
    mask = Volume(data, np.eye(4))
    points = np.array([[0.4, 1, 1], [0.6, 1, 1], [1.4, 1, 1], [1.6, 1, 1]], float)
    np.testing.assert_array_equal(
        _inside_mask(points, mask), [False, True, True, False]
    )


def test_mask_is_read_through_its_scaled_values() -> None:
    # A fractional map stored as integers with a slope of 0.01: 0.8 from x = 2 to 5,
    # 0.3 elsewhere. Read raw, every voxel would be at least 0.5.
    fractions = np.full((10, 10, 10), 0.3, dtype=np.float32)
    fractions[2:6] = 0.8
    floats = Volume(fractions, np.eye(4))
    stored = Volume(np.round(fractions * 100).astype(np.uint8), np.eye(4), slope=0.01)
    seeds = _fixed(20, 2.0, 5.0)
    a = _streamlines(_tracker().track(_fod(), None, seeds, mask=floats, seed=3))
    b = _streamlines(_tracker().track(_fod(), None, seeds, mask=stored, seed=3))
    _assert_same(a, b)
