# Resampling against tckresample 3.0.8 (tests/data/editing, made by
# make_resample_ref.py) and against the numpy reference in reference_resampling.py,
# which the C++ kernel must equal bit for bit.
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import reference_resampling

from larmor.io import load_tractogram
from larmor.tractography import resample
from larmor.types import Grid, Tractogram

DATA = Path(__file__).parents[1] / "data" / "editing"
MODES = [
    pytest.param({"step_size": 0.5}, id="step_size_0.5"),
    pytest.param({"step_size": 2.0}, id="step_size_2"),
    pytest.param({"num_points": 2}, id="num_points_2"),
    pytest.param({"num_points": 10}, id="num_points_10"),
    pytest.param({"num_points": 50}, id="num_points_50"),
]
EDGE_MODES = [
    pytest.param({"step_size": 0.5}, "step_size_0.5", id="step_size_0.5"),
    pytest.param({"step_size": 2.0}, "step_size_2", id="step_size_2"),
    pytest.param({"num_points": 2}, "num_points_2", id="num_points_2"),
    pytest.param({"num_points": 5}, "num_points_5", id="num_points_5"),
]


def _file_name(mode: dict[str, Any]) -> str:
    ((option, value),) = mode.items()
    return f"{option}_{value:g}"


@pytest.fixture(scope="module")
def tracks() -> Tractogram:
    return load_tractogram(DATA / "tracks.tck")


def _with_point_data(t: Tractogram) -> Tractogram:
    rng = np.random.default_rng(3)
    n = len(t.points)
    point_data = {
        "fa": rng.random(n).astype(np.float32),
        "rgb": rng.random((n, 3)),
        "label": rng.integers(0, 100, n).astype(np.int32),
    }
    return Tractogram(t.points, t.offsets, point_data=point_data)


def _random_tractogram() -> Tractogram:
    # Random walks of 0 to 12 points, some with repeated points, so every branch of
    # both modes (short streamlines, zero-length segments) is hit.
    rng = np.random.default_rng(7)
    pieces = []
    for _ in range(80):
        n = int(rng.integers(0, 13))
        steps = rng.normal(0.0, 1.0, (n, 3)) * (rng.random((n, 1)) > 0.15)
        pieces.append(np.cumsum(steps, axis=0).astype(np.float32))
    offsets = np.concatenate([[0], np.cumsum([len(p) for p in pieces])])
    return _with_point_data(Tractogram(np.concatenate(pieces), offsets))


def _assert_same(a: Tractogram, b: Tractogram) -> None:
    np.testing.assert_array_equal(a.offsets, b.offsets)
    np.testing.assert_array_equal(a.points, b.points)
    assert a.point_data.keys() == b.point_data.keys()
    for name in a.point_data:
        assert a.point_data[name].dtype == b.point_data[name].dtype
        np.testing.assert_array_equal(a.point_data[name], b.point_data[name])


@pytest.mark.parametrize("mode", MODES)
def test_matches_tckresample_on_tracks(
    tracks: Tractogram, mode: dict[str, Any]
) -> None:
    expected = load_tractogram(DATA / f"tracks_{_file_name(mode)}.tck")
    result = resample(tracks, **mode)
    np.testing.assert_array_equal(result.offsets, expected.offsets)
    np.testing.assert_array_equal(result.points, expected.points)


@pytest.mark.parametrize("mode", [{"step_size": 0.7}, {"num_points": 7}])
def test_equals_the_numpy_reference(tracks: Tractogram, mode: dict[str, Any]) -> None:
    # A tenth of tracks.tck: the reference walks point by point in Python.
    for t in (_with_point_data(tracks[::10]), _random_tractogram()):
        _assert_same(resample(t, **mode), reference_resampling.resample(t, **mode))


@pytest.mark.parametrize("mode", [{"step_size": 0.5}, {"num_points": 20}])
def test_result_does_not_depend_on_n_jobs(
    tracks: Tractogram, mode: dict[str, Any]
) -> None:
    t = _with_point_data(tracks)
    _assert_same(
        resample(t, **mode, n_jobs=1),
        resample(t, **mode, n_jobs=os.cpu_count() or 1),
    )


@pytest.mark.parametrize(("mode", "suffix"), EDGE_MODES)
def test_edge_cases_match_tckresample_but_two(
    mode: dict[str, Any], suffix: str
) -> None:
    edges = load_tractogram(DATA / "edge_cases.tck")
    expected = load_tractogram(DATA / f"edge_cases_{suffix}.tck")
    result = resample(edges, **mode)
    assert len(result) == len(edges)

    # Difference 1: tckresample empties a 1-point streamline; Larmor keeps it.
    assert len(expected[0]) == 0
    np.testing.assert_array_equal(result[0], edges[0])

    two_points = [1, 5]
    if "step_size" in mode:
        # Difference 2: with a step size tckresample turns a 2-point streamline into
        # its last point three times; Larmor walks from the last point to the first.
        for s in two_points:
            np.testing.assert_array_equal(expected[s], np.repeat(edges[s][-1:], 3, 0))
            np.testing.assert_array_equal(result[s][[0, -1]], edges[s])
            gaps = np.linalg.norm(np.diff(result[s], axis=0), axis=1)
            assert np.all(gaps <= mode["step_size"] * (1 + 1e-3))
    same = [
        s for s in range(1, len(edges)) if "num_points" in mode or s not in two_points
    ]
    for s in same:
        np.testing.assert_array_equal(result[s], expected[s])


@pytest.mark.parametrize("mode", [{"step_size": 0.5}, {"num_points": 10}])
def test_endpoints_are_kept(tracks: Tractogram, mode: dict[str, Any]) -> None:
    result = resample(tracks, **mode)
    first, last = tracks.offsets[:-1], tracks.offsets[1:] - 1
    new_first, new_last = result.offsets[:-1], result.offsets[1:] - 1
    np.testing.assert_array_equal(result.points[new_first], tracks.points[first])
    if "step_size" in mode:
        np.testing.assert_array_equal(result.points[new_last], tracks.points[last])
    else:
        # As in tckresample, the last point is the spline at a parameter of about 1,
        # whose float32 weights sum to 1 only up to rounding: a few ulps of a
        # ~100 mm coordinate (ulp 7.6e-6 mm).
        np.testing.assert_allclose(
            result.points[new_last], tracks.points[last], rtol=0, atol=1e-4
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_point_data_is_interpolated_linearly(dtype: type) -> None:
    # A straight 3 mm streamline cut into 4 equal parts: the new points sit at
    # fractions 0, 1/4, 1/2, 3/4 and 1 of its only segment.
    t = Tractogram(
        np.array([[0, 0, 0], [3, 0, 0]], np.float32),
        np.array([0, 2]),
        point_data={"value": np.array([0.0, 10.0], dtype)},
    )
    value = resample(t, num_points=5).point_data["value"]
    assert value.dtype == dtype
    np.testing.assert_array_equal(value, [0.0, 2.5, 5.0, 7.5, 10.0])


def test_integer_point_data_takes_the_nearest_point_and_the_earlier_at_a_tie() -> None:
    t = Tractogram(
        np.array([[0, 0, 0], [3, 0, 0]], np.float32),
        np.array([0, 2]),
        point_data={"label": np.array([7, 9], np.int16)},
    )
    label = resample(t, num_points=5).point_data["label"]
    assert label.dtype == np.int16
    # Fractions 0, 1/4, 1/2 (the tie), 3/4 and 1.
    np.testing.assert_array_equal(label, [7, 7, 7, 9, 9])


@pytest.mark.parametrize("mode", [{"step_size": 0.5}, {"num_points": 5}])
def test_streamlines_of_fewer_than_two_points_are_kept(mode: dict[str, Any]) -> None:
    t = Tractogram(
        np.array([[1, 2, 3], [0, 0, 0], [4, 0, 0]], np.float32),
        np.array([0, 0, 1, 1, 3]),
        point_data={"fa": np.array([0.1, 0.2, 0.3], np.float32)},
    )
    result = resample(t, **mode)
    assert [len(result[s]) for s in range(3)] == [0, 1, 0]
    np.testing.assert_array_equal(result[1], t[1])
    assert result.point_data["fa"][0] == np.float32(0.1)


def test_streamline_data_and_reference_are_kept(tracks: Tractogram) -> None:
    grid = Grid((10, 10, 10), np.eye(4))
    weight = np.arange(len(tracks), dtype=np.float32)
    t = Tractogram(
        tracks.points,
        tracks.offsets,
        streamline_data={"weight": weight},
        reference=grid,
    )
    result = resample(t, step_size=1.0)
    np.testing.assert_array_equal(result.streamline_data["weight"], weight)
    assert result.reference is grid


@pytest.mark.parametrize("mode", [{"step_size": 1.0}, {"num_points": 5}])
def test_empty_tractogram(mode: dict[str, Any]) -> None:
    t = Tractogram(
        np.zeros((0, 3), np.float32),
        np.array([0]),
        point_data={"fa": np.zeros(0, np.float64), "label": np.zeros(0, np.int32)},
        streamline_data={"weight": np.zeros(0, np.float32)},
    )
    result = resample(t, **mode)
    assert len(result) == 0
    assert result.points.shape == (0, 3)
    np.testing.assert_array_equal(result.offsets, [0])
    assert result.point_data["fa"].dtype == np.float64
    assert result.point_data["label"].dtype == np.int32
    assert result.streamline_data["weight"].shape == (0,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"step_size": 1.0, "num_points": 5},
        {"step_size": 0.0},
        {"step_size": -1.0},
        {"step_size": float("nan")},
        {"step_size": float("inf")},
        {"num_points": 1},
        {"num_points": 2.5},
        {"num_points": True},
        {"num_points": 5.0},
        {"num_points": float("inf")},
        {"num_points": "5"},
        {"num_points": 2**70},
        {"num_points": 5, "n_jobs": 2.0},
        {"num_points": 5, "n_jobs": True},
        {"num_points": 5, "n_jobs": 0},
        {"num_points": 5, "n_jobs": -2},
        # 4 * 2**62 points overflow int64, and 2**62 * 3 coordinates do already.
        {"num_points": 2**62},
        {"num_points": 2**61},
    ],
)
def test_invalid_arguments_are_refused(kwargs: dict[str, Any]) -> None:
    # Four streamlines of 2 points.
    t = Tractogram(np.zeros((8, 3), np.float32), np.array([0, 2, 4, 6, 8]))
    with pytest.raises(ValueError):
        resample(t, **kwargs)


def test_numpy_integers_are_accepted() -> None:
    t = Tractogram(np.array([[0, 0, 0], [3, 0, 0]], np.float32), np.array([0, 2]))
    # The annotations say int; numpy integers are accepted at run time too.
    num_points: Any = np.int64(4)
    n_jobs: Any = np.int32(1)
    result = resample(t, num_points=num_points, n_jobs=n_jobs)
    np.testing.assert_array_equal(result.offsets, [0, 4])


def test_num_points_on_a_not_finite_point_gives_not_finite_points() -> None:
    points = np.array([[0, 0, 0], [1, 0, 0], [np.nan, 0, 0], [3, 0, 0]], np.float32)
    result = resample(Tractogram(points, np.array([0, 4])), num_points=5)
    assert len(result.points) == 5
    assert not np.isfinite(result.points).all()


def test_step_size_beyond_float32_is_refused_in_python() -> None:
    t = Tractogram(np.zeros((2, 3), np.float32), np.array([0, 2]))
    with pytest.raises(ValueError, match="float32 can hold"):
        resample(t, step_size=1e39)


# Run in a child process with a timeout, so a pass that never ends fails the test
# instead of hanging the suite. The step sizes are a few float32 ulps of the ~100 mm
# coordinates (ulp 7.6e-6 mm), so the bisection cannot meet its tolerance; NaN and
# inf points must end too.
_TINY_STEP_SCRIPT = """
import numpy as np
import reference_resampling
from larmor.tractography import resample
from larmor.types import Tractogram

long = np.array([[100, 50, 20], [100.5, 50.2, 20.1], [101, 50.3, 20.3]], np.float32)
short = np.array(
    [[100, 50, 20], [100.004, 50.002, 20.001], [100.008, 50.003, 20.003]], np.float32
)
not_finite = np.array(
    [[100, 0, 0], [101, 0, 0], [np.nan, 0, 0], [103, 0, 0], [104, 0, 0],
     [np.inf, 0, 0], [106, 0, 0]],
    np.float32,
)
for points, step in [(long, 2e-5), (long, 1e-5), (short, 1e-5), (not_finite, 0.5)]:
    t = Tractogram(points, [0, len(points)])
    result = resample(t, step_size=step)
    np.testing.assert_array_equal(result.points[[0, -1]], points[[0, -1]])
for points, step in [(short, 2e-5), (short, 1e-5), (not_finite, 0.5)]:
    t = Tractogram(points, [0, len(points)])
    expected = reference_resampling.resample(t, step_size=step)
    np.testing.assert_array_equal(resample(t, step_size=step).points, expected.points)
"""


def test_step_sizes_near_float32_resolution_and_not_finite_points_end() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _TINY_STEP_SCRIPT],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("mode", [{"step_size": 0.5}, {"num_points": 3}])
def test_point_data_at_a_point_ignores_a_not_finite_neighbour(
    mode: dict[str, Any],
) -> None:
    # A 1-point streamline followed by one starting with NaN, and a streamline whose
    # last point follows an inf: the values at fractions 0 and 1 are the point's own.
    t = Tractogram(
        np.array([[5, 5, 5], [0, 0, 0], [2, 0, 0], [0, 1, 0], [2, 1, 0]], np.float32),
        np.array([0, 1, 3, 5]),
        point_data={"fa": np.array([5.0, np.nan, 2.0, np.inf, 4.0], np.float32)},
    )
    result = resample(t, **mode)
    fa = result.point_data["fa"]
    assert fa[result.offsets[0]] == np.float32(5.0)
    assert fa[result.offsets[3] - 1] == np.float32(4.0)
    _assert_same(result, reference_resampling.resample(t, **mode))
