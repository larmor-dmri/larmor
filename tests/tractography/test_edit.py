import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from larmor.io import (
    load_tractogram,
    load_volume,
    read_streamline_weights,
    save_tractogram,
    write_streamline_weights,
)
from larmor.io.formats import write_tck
from larmor.tractography import (
    EditOptions,
    Region,
    Sphere,
    edit,
    edit_files,
    load_region,
)
from larmor.types import Tractogram

DATA = Path(__file__).parents[1] / "data" / "editing"
NUM_STREAMLINES = 500

# The spheres and masks of make_tract_edit_ref.py.
SPHERES = {
    "a": "-124,-76,74,5",
    "b": "-116,-62,73,5",
    "c": "-108,-50,74,5",
    "away": "-100,-60,85,8",
    "start": "-129,-88,77,4",
    "end": "-105,-43,76,4",
}


def _sphere(name: str) -> Region:
    return load_region(SPHERES[name])


def _mask(name: str) -> Region:
    return load_volume(DATA / f"{name}_mask.nii.gz")


# The tckedit options of each case of make_tract_edit_ref.py, as EditOptions.
def _cases() -> dict[str, dict[str, Any]]:
    a, b, c = _sphere("a"), _sphere("b"), _sphere("c")
    shell, flt = _mask("shell"), _mask("float")
    start, end = _sphere("start"), _sphere("end")
    return {
        "includes_exclude_min_length": dict(
            include=[a, c], exclude=[_sphere("away")], min_length=45.5
        ),
        "includes_mask_max_length": dict(include=[a, shell], max_length=55.5),
        "ends_includes_length": dict(
            include=[start, end], ends_only=True, min_length=40.5, max_length=58.5
        ),
        "ends_ordered_exclude": dict(
            include_ordered=[start, end], exclude=[b], ends_only=True
        ),
        "ordered_skip_number": dict(include_ordered=[a, c], skip=10, number=25),
        "weight_include": dict(include=[b], min_weight=101, max_weight=400),
        "mask_shell_number": dict(mask=[shell], number=50),
        "mask_sphere_skip_number": dict(mask=[c], skip=20, number=30),
        "skip_differs_spheres": dict(mask=[a, c], skip=30, number=40),
        "include_mask_float_number": dict(include=[c], mask=[flt], number=60),
        "inverse_include_crop": dict(include=[a], inverse=True, mask=[flt], number=100),
        "skip_differs_inverse_crop": dict(
            inverse=True, mask=[shell], skip=5, number=30
        ),
    }


CASES = _cases()
MATCHING = [name for name in CASES if not name.startswith("skip_differs")]


@pytest.fixture(scope="module")
def tracks() -> Tractogram:
    t = load_tractogram(DATA / "tracks.tck")
    # Index + 1 as the weight, as tckedit got it: each output streamline names its
    # input streamline.
    t.streamline_data["weight"] = np.arange(1, NUM_STREAMLINES + 1, dtype=np.float32)
    return t


@pytest.fixture(scope="module")
def ref() -> dict[str, np.ndarray]:
    with np.load(DATA / "tract_edit_ref.npz") as data:
        return {name: data[name] for name in data.files}


def _sources(t: Tractogram) -> np.ndarray:
    return t.streamline_data["weight"].astype(np.int64) - 1


@pytest.mark.parametrize("case", MATCHING)
def test_edit_matches_tckedit(
    tracks: Tractogram, ref: dict[str, np.ndarray], case: str
) -> None:
    result = edit(tracks, EditOptions(**CASES[case]))

    np.testing.assert_array_equal(_sources(result), ref[case])
    np.testing.assert_array_equal(np.diff(result.offsets), ref[f"{case}_lengths"])


@pytest.mark.parametrize("case", ["skip_differs_spheres", "skip_differs_inverse_crop"])
def test_skip_counts_a_streamline_cropped_into_pieces_unlike_tckedit(
    tracks: Tractogram, ref: dict[str, np.ndarray], case: str
) -> None:
    options = CASES[case]
    skip, number = options["skip"], options["number"]
    uncounted = edit(
        tracks, EditOptions(mask=options["mask"], inverse="inverse" in options)
    )
    units, pieces = np.unique(_sources(uncounted), return_counts=True)

    result = edit(tracks, EditOptions(**options))

    # Larmor skips the first `skip` cropped streamlines whatever their pieces.
    np.testing.assert_array_equal(
        np.unique(_sources(result)), units[skip : skip + number]
    )
    # tckedit writes some of those when they were cut into several pieces.
    written_by_tckedit = np.intersect1d(ref[case], units[:skip])
    assert written_by_tckedit.size > 0
    assert np.all(pieces[np.isin(units, written_by_tckedit)] > 1)


def _write_inputs(tmp_path: Path, tracks: Tractogram) -> tuple[Path, Path]:
    tck, weights = tmp_path / "in.tck", tmp_path / "in_weights.txt"
    write_tck(tracks, tck)
    write_streamline_weights(tracks.streamline_data["weight"], weights)
    return tck, weights


LAZY_CASES = {
    **{name: CASES[name] for name in ("ordered_skip_number", "weight_include")},
    **{
        name: CASES[name]
        for name in ("mask_sphere_skip_number", "skip_differs_inverse_crop")
    },
    "mask_skip_every_number": dict(mask=[_sphere("c")], skip=3, every=3, number=10),
    "every_without_number": dict(include=[_sphere("a")], skip=2, every=4),
}


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("case", list(LAZY_CASES))
def test_edit_files_gives_the_same_as_edit(
    tmp_path: Path, tracks: Tractogram, case: str, lazy: bool
) -> None:
    tck, weights = _write_inputs(tmp_path, tracks)
    options = EditOptions(**LAZY_CASES[case])
    out, weights_out = tmp_path / "out.tck", tmp_path / "out_weights.txt"

    # Batches of 7 split the counted streamlines across many batches.
    _, num_written = edit_files(
        [tck],
        out,
        options,
        weights_in=weights,
        weights_out=weights_out,
        lazy=lazy,
        batch_size=7,
    )

    expected = edit(tracks, options)
    written = load_tractogram(out)
    assert num_written == expected.num_streamlines
    np.testing.assert_array_equal(written.offsets, expected.offsets)
    np.testing.assert_array_equal(written.points, expected.points)
    np.testing.assert_array_equal(
        read_streamline_weights(weights_out), expected.streamline_data["weight"]
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_two_inputs_are_joined_and_counted_across_the_join_as_tckedit(
    tmp_path: Path, lazy: bool
) -> None:
    out = tmp_path / "out.tck"
    options = EditOptions(include=[_sphere("c")], number=300)

    num_read, num_written = edit_files(
        [DATA / "tracks.tck"] * 2, out, options, lazy=lazy, batch_size=7
    )

    expected = load_tractogram(DATA / "tract_edit_join.tck")
    written = load_tractogram(out)
    np.testing.assert_array_equal(written.offsets, expected.offsets)
    np.testing.assert_array_equal(written.points, expected.points)
    assert num_written == 300
    # Lazy reading stops at the batch that reaches the number, near the end of
    # the second input.
    assert num_read == (990 if lazy else 2 * NUM_STREAMLINES)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(number=3, subsample=3),
        dict(step_size=1.0, num_points=5),
        dict(step_size=0.0),
        dict(step_size=-1.0),
        dict(num_points=1),
        dict(min_length=10.0, max_length=5.0),
        dict(min_weight=2.0, max_weight=1.0),
        dict(min_length=float("nan")),
        dict(every=0),
        dict(skip=-1),
        dict(number=-1),
        dict(subsample=-1),
        dict(ends_only=True, include_ordered=[Sphere(np.zeros(3), 1.0)] * 3),
    ],
)
def test_bad_options_raise(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        EditOptions(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(min_length=5.0, max_length=5.0),
        dict(skip=0, number=0, every=1),
        dict(num_points=2),
        dict(ends_only=True, include_ordered=[Sphere(np.zeros(3), 1.0)] * 2),
    ],
)
def test_options_at_their_limits_are_accepted(kwargs: dict[str, Any]) -> None:
    EditOptions(**kwargs)


def _line(xs: list[float], y: float = 0.0) -> list[list[float]]:
    return [[x, y, 0.0] for x in xs]


def _tractogram(streamlines: list[list[list[float]]]) -> Tractogram:
    sizes = [len(s) for s in streamlines]
    points = np.array([p for s in streamlines for p in s], dtype=np.float32)
    return Tractogram(
        points.reshape(-1, 3),
        np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
        streamline_data={"weight": np.arange(len(sizes), dtype=np.float32)},
    )


# A streamline along x through three spheres, cut into three pieces by them.
THREE_SPHERES = [Sphere(np.array([x, 0.0, 0.0]), 1.0) for x in (0.0, 10.0, 20.0)]
# 0: cut into three pieces; 1: missing every sphere; 2 and 3: one piece each.
CUT = _tractogram(
    [
        _line([float(x) for x in range(21)]),
        _line([float(x) for x in range(21)], y=5.0),
        _line([9.0, 10.0, 11.0]),
        _line([19.0, 20.0, 21.0]),
    ]
)


@pytest.mark.parametrize(
    ("counting", "kept"),
    [
        (dict(number=1), [0, 0, 0]),
        (dict(skip=1), [2, 3]),
        (dict(every=2), [0, 0, 0, 3]),
        (dict(skip=1, number=1), [2]),
    ],
)
def test_pieces_of_one_streamline_count_as_one(
    counting: dict[str, Any], kept: list[int]
) -> None:
    result = edit(CUT, EditOptions(mask=THREE_SPHERES, **counting))

    np.testing.assert_array_equal(result.streamline_data["weight"], kept)


def test_length_is_tested_before_cropping() -> None:
    # Streamline 0 is 20 mm long; its pieces are 2 mm.
    result = edit(CUT, EditOptions(min_length=15.0, mask=THREE_SPHERES))

    np.testing.assert_array_equal(result.streamline_data["weight"], [0, 0, 0])


def test_resampling_comes_after_cropping() -> None:
    result = edit(CUT, EditOptions(mask=THREE_SPHERES, num_points=5))

    np.testing.assert_array_equal(np.diff(result.offsets), [5] * 5)


@pytest.mark.parametrize("counting", [dict(number=10), dict(subsample=10, seed=1)])
def test_counting_more_than_are_left_keeps_all(
    caplog: pytest.LogCaptureFixture, counting: dict[str, Any]
) -> None:
    with caplog.at_level(logging.WARNING):
        result = edit(CUT, EditOptions(**counting))

    np.testing.assert_array_equal(result.streamline_data["weight"], [0, 1, 2, 3])
    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == ("subsample" in counting)


def test_empty_streamline_and_negative_weight_are_kept_without_limits() -> None:
    t = _tractogram([[], _line([0.0, 1.0])])
    t.streamline_data["weight"] = np.array([1.0, -1.0], dtype=np.float32)

    assert edit(t, EditOptions()).num_streamlines == 2
    # The empty streamline counts as one.
    assert edit(t, EditOptions(number=1)).num_streamlines == 1
    assert edit(t, EditOptions(number=1)).offsets.tolist() == [0, 0]
    assert edit(t, EditOptions(min_length=0.5)).num_streamlines == 1
    assert edit(t, EditOptions(min_weight=0.0)).num_streamlines == 1


def test_a_sphere_that_is_not_four_numbers_names_the_form() -> None:
    with pytest.raises(ValueError, match="x,y,z,r"):
        load_region("1,2,3")


def test_edit_refuses_a_region_given_as_text(tracks: Tractogram) -> None:
    with pytest.raises(TypeError, match="load_region"):
        edit(tracks, EditOptions(include=[SPHERES["a"]]))


def test_edit_files_reads_regions_given_as_text(
    tmp_path: Path, tracks: Tractogram, ref: dict[str, np.ndarray]
) -> None:
    tck, weights = _write_inputs(tmp_path, tracks)
    options = EditOptions(mask=[str(DATA / "shell_mask.nii.gz")], number=50)
    weights_out = tmp_path / "out_weights.txt"

    edit_files(
        [tck],
        tmp_path / "out.tck",
        options,
        weights_in=weights,
        weights_out=weights_out,
    )

    kept = read_streamline_weights(weights_out).astype(np.int64) - 1
    np.testing.assert_array_equal(kept, ref["mask_shell_number"])


def test_lazy_weights_replacing_a_trk_property_warn_once(
    tmp_path: Path, tracks: Tractogram, caplog: pytest.LogCaptureFixture
) -> None:
    trk = tmp_path / "in.trk"
    save_tractogram(tracks, trk, reference=load_volume(DATA / "shell_mask.nii.gz").grid)
    weights = tmp_path / "weights.txt"
    write_streamline_weights(np.ones(NUM_STREAMLINES, dtype=np.float32), weights)

    with caplog.at_level(logging.WARNING, logger="larmor"):
        edit_files(
            [trk],
            tmp_path / "out.trk",
            EditOptions(),
            weights_in=weights,
            lazy=True,
            batch_size=50,
        )

    replaced = [r for r in caplog.records if "replace" in r.getMessage()]
    assert len(replaced) == 1
