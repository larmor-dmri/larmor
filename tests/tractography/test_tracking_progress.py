"""Progress reports of tracking runs inside track_progress."""

import numpy as np
import pytest

from larmor.tractography import MaskSeeds, Seeds, Tracker, tracker
from larmor.tractography.progress import track_progress
from larmor.types import Volume


class StopError(Exception):
    pass


def _fod() -> Volume:
    """A 10x10x10 field of FODs with one lobe along z."""
    coeffs = np.zeros((10, 10, 10, 6), dtype=np.float32)
    coeffs[..., 0] = 1.0
    coeffs[..., 3] = 1.0
    return Volume(coeffs, np.eye(4))


def _det() -> Tracker:
    return Tracker("det", step_size=0.5, min_length=0.5, max_length=5.0, cutoff=0.0)


def _seeds(n: int) -> Seeds:
    return Seeds(np.random.default_rng(n).uniform(3.0, 6.0, (n, 3)))


def test_fixed_seeds_are_reported_batch_by_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracker, "MAX_BATCH", 7)
    calls: list[tuple[int, int, str]] = []
    with track_progress(lambda *call: calls.append(call)):
        _det().track(_fod(), None, _seeds(20), seed=1)
    assert calls == [(n, 20, "Tracking seeds") for n in (0, 7, 14, 20)]


def test_a_target_is_reported_as_streamlines_kept() -> None:
    mask = Volume(np.ones((10, 10, 10), dtype=np.float32), np.eye(4))
    calls: list[tuple[int, int, str]] = []
    with track_progress(lambda *call: calls.append(call)):
        run = _det().track(_fod(), None, MaskSeeds(mask), num_streamlines=30, seed=1)
    assert calls[0] == (0, 30, "Tracking streamlines")
    assert calls[-1] == (run.num_streamlines, 30, "Tracking streamlines")


def test_an_exception_from_report_stops_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracker, "MAX_BATCH", 7)
    calls: list[int] = []

    def report(done: int, total: int, stage: str) -> None:
        calls.append(done)
        if done > 0:
            raise StopError

    with pytest.raises(StopError), track_progress(report):
        _det().track(_fod(), None, _seeds(20), seed=1)
    assert calls == [0, 7]
