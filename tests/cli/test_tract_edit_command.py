from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner, Result

from larmor.io import (
    load_tractogram,
    read_mritoolkit_tracts,
    read_streamline_weights,
    read_track_timestamp,
    read_tsf,
    save_tractogram,
    save_volume,
    write_streamline_weights,
    write_tsf,
)
from larmor.io.formats import write_tck
from larmor.main import app
from larmor.types import Grid, Tractogram, Volume

# Voxel (i, j, k) is at world (i, j, k - 1): the streamlines lie in slice k = 1.
AFFINE = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, -1.0], [0, 0, 0, 1]])
GRID = Grid((40, 30, 3), AFFINE)
TIMESTAMP = "1234.5"


pytestmark = pytest.mark.usefixtures("keep_log_level")


def _line(start: int, stop: int, y: float) -> np.ndarray:
    step = 1 if stop >= start else -1
    xs = np.arange(start, stop + step, step, dtype=np.float32)
    return np.stack([xs, np.full_like(xs, y), np.zeros_like(xs)], axis=1)


# Straight lines along x with 1 mm steps, lengths 10, 20, 20, 20, 30 and 10 mm;
# streamline 3 is streamline 2 stored backwards.
LINES = [
    _line(0, 10, 0.0),
    _line(0, 20, 0.0),
    _line(0, 20, 10.0),
    _line(20, 0, 10.0),
    _line(0, 30, 20.0),
    _line(5, 15, 20.0),
]
SIZES = [len(line) for line in LINES]
POINTS = np.concatenate(LINES)
OFFSETS = np.concatenate([[0], np.cumsum(SIZES)]).astype(np.int64)
WEIGHTS = np.array([0.5, 1.5, -1.0, 2.0, 1.0, 0.0], dtype=np.float32)


def _run(*args: str | Path) -> Result:
    return CliRunner().invoke(
        app, ["tract-edit", *map(str, args)], env={"TERM": "dumb"}
    )


def _edit(*args: str | Path) -> Result:
    result = _run(*args)
    assert result.exit_code == 0, result.output
    return result


@pytest.fixture
def tck(tmp_path: Path) -> Path:
    path = tmp_path / "in.tck"
    write_tck(Tractogram(POINTS, OFFSETS), path, timestamp=TIMESTAMP)
    return path


@pytest.fixture
def mask(tmp_path: Path) -> Path:
    # Voxels with x 0..3 on the line y = 20: only streamline 4 reaches them.
    data = np.zeros(GRID.shape, dtype=np.uint8)
    data[0:4, 20, 1] = 1
    path = tmp_path / "mask.nii.gz"
    save_volume(Volume(data, AFFINE), path)
    return path


def _kept(path: Path) -> list[int]:
    """The input streamline each output streamline is, whole and unchanged."""
    t = load_tractogram(path)
    by_points = {line.tobytes(): i for i, line in enumerate(LINES)}
    return [
        by_points[t.points[t.offsets[s] : t.offsets[s + 1]].tobytes()]
        for s in range(t.num_streamlines)
    ]


def _xs(path: Path) -> list[list[float]]:
    t = load_tractogram(path)
    return [
        t.points[a:b, 0].tolist()
        for a, b in zip(t.offsets[:-1], t.offsets[1:], strict=True)
    ]


@pytest.mark.parametrize(
    ("args", "kept"),
    [
        (["--min-length", "15"], [1, 2, 3, 4]),
        (["--max-length", "25"], [0, 1, 2, 3, 5]),
        (["--include", "20,10,0,0.5"], [2, 3]),
        (["--include", "0,0,0,0.5", "--include", "20,0,0,0.5"], [1]),
        # Stored backwards, streamline 3 meets the regions in the other order.
        (["--include-ordered", "0,10,0,0.5", "--include-ordered", "20,10,0,0.5"], [2]),
        (["--exclude", "5,0,0,0.5"], [2, 3, 4, 5]),
        # Streamline 1 passes x = 10 between its ends.
        (["--include", "10,0,0,0.5"], [0, 1]),
        (["--include", "10,0,0,0.5", "--ends-only"], [0]),
        (["--exclude", "10,0,0,0.5", "--ends-only"], [1, 2, 3, 4, 5]),
        (["--include", "mask"], [4]),
        # Streamline 1 leaves the second sphere, which streamline 0 stays in.
        (["--within", "10,0,0,10.5"], [0, 1]),
        (["--within", "10,0,0,10.5", "--within", "0,0,0,10.5"], [0]),
        (["--exclude", "mask"], [0, 1, 2, 3, 5]),
        (["--include", "20,10,0,0.5", "--inverse"], [0, 1, 4, 5]),
        # tckedit drops nothing on --inverse with only lengths; Larmor keeps the
        # streamlines that fail them.
        (["--min-length", "15", "--inverse"], [0, 5]),
        # Skip 1, then every 2nd of 1..5, then the first 2 of 1, 3, 5.
        (["--skip", "1", "--every", "2", "--number", "2"], [1, 3]),
        (["--number", "10"], [0, 1, 2, 3, 4, 5]),
    ],
)
def test_selection_options(
    tmp_path: Path, tck: Path, mask: Path, args: list[str], kept: list[int]
) -> None:
    out = tmp_path / "out.tck"
    _edit(tck, out, *[str(mask) if a == "mask" else a for a in args])

    assert _kept(out) == kept


def test_length_limits_are_inclusive(tmp_path: Path, tck: Path) -> None:
    out = tmp_path / "out.tck"
    _edit(tck, out, "--min-length", "20", "--max-length", "20")

    # Streamlines 1, 2 and 3 are 20 mm long: sums of twenty 1 mm steps.
    assert _kept(out) == [1, 2, 3]


@pytest.mark.parametrize(
    ("args", "xs"),
    [
        (["--mask", "10,0,0,2.5"], [[8, 9, 10], [8, 9, 10, 11, 12]]),
        (["--mask", "mask"], [[0, 1, 2, 3]]),
        # With only --mask, --inverse crops every streamline to the outside.
        (
            ["--mask", "10,0,0,2.5", "--inverse"],
            [
                list(range(8)),
                list(range(8)),
                list(range(13, 21)),
                *[[float(x) for x in line[:, 0]] for line in LINES[2:]],
            ],
        ),
    ],
)
def test_mask_crops(
    tmp_path: Path, tck: Path, mask: Path, args: list[str], xs: list[list[float]]
) -> None:
    out = tmp_path / "out.tck"
    _edit(tck, out, *[str(mask) if a == "mask" else a for a in args])

    assert _xs(out) == xs


def test_weights_select_and_are_written_for_the_kept_streamlines(
    tmp_path: Path, tck: Path
) -> None:
    weights_in, weights_out = tmp_path / "w.txt", tmp_path / "kept.txt"
    write_streamline_weights(WEIGHTS, weights_in)
    out = tmp_path / "out.tck"

    _edit(
        tck,
        out,
        "--weights-in",
        weights_in,
        "--weights-out",
        weights_out,
        "--min-weight",
        "0.5",
        "--max-weight",
        "1.5",
    )

    assert _kept(out) == [0, 1, 4]
    np.testing.assert_array_equal(
        read_streamline_weights(weights_out), WEIGHTS[[0, 1, 4]]
    )


def test_negative_weights_are_kept_without_a_weight_limit(
    tmp_path: Path, tck: Path
) -> None:
    weights_in, weights_out = tmp_path / "w.txt", tmp_path / "kept.txt"
    write_streamline_weights(WEIGHTS, weights_in)

    _edit(
        tck,
        tmp_path / "out.tck",
        "--weights-in",
        weights_in,
        "--weights-out",
        weights_out,
    )

    np.testing.assert_array_equal(read_streamline_weights(weights_out), WEIGHTS)


def test_subsample_with_a_seed_repeats(tmp_path: Path, tck: Path) -> None:
    first, second = tmp_path / "a.tck", tmp_path / "b.tck"
    _edit(tck, first, "--subsample", "3", "--seed", "11")
    _edit(tck, second, "--subsample", "3", "--seed", "11")

    kept = _kept(first)
    assert kept == _kept(second)
    assert len(kept) == 3 and kept == sorted(set(kept))


def test_step_size_resamples(tmp_path: Path, tck: Path) -> None:
    out = tmp_path / "out.tck"
    _edit(tck, out, "--step-size", "0.5", "--include", "0,0,0,0.5")

    # Steps of 0.5 mm from the middle point out to each end (the bisection stops
    # within 1e-3 mm), then the end itself a fraction of a step away.
    for xs, length in zip(_xs(out), [10, 20], strict=True):
        steps = np.diff(xs)
        assert (xs[0], xs[-1]) == (0, length)
        np.testing.assert_allclose(steps[:-1], 0.5, atol=1e-3)
        assert 0 < steps[-1] < 0.5


def test_num_points_resamples(tmp_path: Path, tck: Path) -> None:
    out = tmp_path / "out.tck"
    _edit(tck, out, "--num-points", "5", "--include", "0,0,0,0.5")

    expected = np.array([[0, 2.5, 5, 7.5, 10], [0, 5, 10, 15, 20]])
    np.testing.assert_allclose(np.array(_xs(out)), expected, atol=1e-5)


def test_tsf_values_follow_the_crop_with_the_tractogram_timestamp(
    tmp_path: Path, tck: Path
) -> None:
    write_tsf(POINTS[:, 0] * 10, OFFSETS, tmp_path / "xten.tsf", timestamp=TIMESTAMP)
    out = tmp_path / "out.tck"

    _edit(
        tck,
        out,
        "--mask",
        "10,0,0,2.5",
        "--tsf-in",
        tmp_path / "xten.tsf",
        "--tsf-out",
        "xten",
    )

    t = load_tractogram(out)
    tsf = tmp_path / "out_xten.tsf"
    values, _ = read_tsf(tsf, tractogram=t)
    np.testing.assert_array_equal(values, t.points[:, 0] * 10)
    assert read_track_timestamp(out) == read_track_timestamp(tsf) == TIMESTAMP


def test_reference_gives_the_grid_of_a_trk_output(tmp_path: Path, tck: Path) -> None:
    reference = tmp_path / "ref.nii.gz"
    save_volume(Volume(np.zeros(GRID.shape, np.uint8), AFFINE), reference)
    out = tmp_path / "out.trk"

    _edit(tck, out, "--reference", reference)

    t = load_tractogram(out)
    assert t.reference is not None and t.reference.matches(GRID)
    np.testing.assert_allclose(t.points, POINTS, atol=1e-5)


def test_lazy_gives_the_same_as_whole(tmp_path: Path, tck: Path) -> None:
    whole, lazy = tmp_path / "whole.tck", tmp_path / "lazy.tck"
    args = ["--mask", "10,0,0,2.5", "--inverse", "--skip", "1", "--every", "2"]
    _edit(tck, whole, *args)
    _edit(tck, lazy, *args, "--lazy")

    assert _xs(lazy) == _xs(whole)


def test_reports_streamlines_read_and_written(tmp_path: Path, tck: Path) -> None:
    out = tmp_path / "out.tck"
    result = _edit(tck, tck, out, "--include", "20,10,0,0.5")

    # The two inputs are joined.
    assert f"Read 12 streamlines, wrote 4 to {out}." in result.output
    assert _kept(out) == [2, 3, 2, 3]


def test_force_overwrites_the_output(tmp_path: Path, tck: Path) -> None:
    out = tmp_path / "out.tck"
    out.write_text("old")

    _edit(tck, out, "--number", "1", "--force")

    assert _kept(out) == [0]


@pytest.mark.parametrize("lazy", [False, True])
def test_trk_keeps_point_and_streamline_data(tmp_path: Path, lazy: bool) -> None:
    source = Tractogram(
        POINTS,
        OFFSETS,
        point_data={"x": POINTS[:, 0].copy()},
        streamline_data={
            "weight": WEIGHTS,
            "label": np.arange(len(LINES), dtype=np.float32),
        },
        reference=GRID,
    )
    trk, out = tmp_path / "in.trk", tmp_path / "out.trk"
    save_tractogram(source, trk)

    # The weight property drives --max-weight without --weights-in.
    args = ["--mask", "10,0,0,2.5", "--mask", "10,20,0,2.5", "--max-weight", "1.5"]
    _edit(trk, out, *args, *(["--lazy"] if lazy else []))

    t = load_tractogram(out)
    np.testing.assert_array_equal(t.point_data["x"], t.points[:, 0])
    np.testing.assert_array_equal(t.streamline_data["label"], [0, 1, 4, 5])
    np.testing.assert_array_equal(t.streamline_data["weight"], WEIGHTS[[0, 1, 4, 5]])


SPHERE = "0,0,0,1"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["out.tck", "--min-weight", "1"], "needs weights"),
        (["out.tck", "--weights-out", "w.txt"], "needs weights"),
        (["out.tck", "--tsf-in", "a/fa.tsf", "--tsf-in", "b/fa.tsf"], "stem 'fa'"),
        (["out.tck", "--tsf-out", "fa"], "No values named 'fa'"),
        (["out.trk"], "needs a reference grid"),
        (["out.vtk", "--lazy"], "only uncompressed TCK and TRK"),
        (["out.tck.gz", "--lazy"], "only uncompressed TCK and TRK"),
        (["out.tck", "--lazy", "--subsample", "2"], "cannot subsample"),
        (["out.tck", "--lazy", "--tsf-in", "fa.tsf"], "TSF"),
        (["out.tck", "--number", "2", "--subsample", "2"], "number and subsample"),
        (["out.tck", "--step-size", "1", "--num-points", "5"], "step_size and num"),
        (
            ["out.tck", "--ends-only", *["--include-ordered", SPHERE] * 3],
            "cannot pass through 3 ordered regions",
        ),
    ],
)
def test_refuses_before_reading_or_writing(
    tmp_path: Path, args: list[str], message: str
) -> None:
    # The input does not exist, so a refusal that reads it would fail differently.
    result = _run(tmp_path / "missing.tck", tmp_path / args[0], *args[1:])

    assert result.exit_code == 1
    assert "ERROR: " in result.output and message in result.output, result.output
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("missing", ["missing.tck", "missing.trk"])
def test_existing_output_is_kept_without_force(tmp_path: Path, missing: str) -> None:
    # A TRK input's header is read in the checks: the output comes first.
    out = tmp_path / "out.tck"
    out.write_text("old")

    result = _run(tmp_path / missing, out)

    assert result.exit_code == 1 and "exists" in result.output, result.output
    assert out.read_text() == "old"


DATA = Path(__file__).parents[1] / "data" / "editing"


# Two cases of make_tract_edit_ref.py that tckedit and Larmor agree on, through the
# command. The 500-streamline tracks.tck stands in for the whole ISMRM tractogram.
@pytest.mark.parametrize(
    ("case", "args"),
    [
        (
            "includes_exclude_min_length",
            ["--include", "-124,-76,74,5", "--include", "-108,-50,74,5"]
            + ["--exclude", "-100,-60,85,8", "--min-length", "45.5"],
        ),
        (
            "mask_shell_number",
            ["--mask", str(DATA / "shell_mask.nii.gz")] + ["--number", "50"],
        ),
    ],
)
def test_command_matches_tckedit(tmp_path: Path, case: str, args: list[str]) -> None:
    weights_in, weights_out = tmp_path / "in.txt", tmp_path / "out.txt"
    write_streamline_weights(np.arange(1, 501, dtype=np.float32), weights_in)
    out = tmp_path / "out.tck"

    _edit(
        DATA / "tracks.tck",
        out,
        *args,
        "--weights-in",
        weights_in,
        "--weights-out",
        weights_out,
    )

    with np.load(DATA / "tract_edit_ref.npz") as ref:
        expected, lengths = ref[case], ref[f"{case}_lengths"]
    kept = read_streamline_weights(weights_out).astype(np.int64) - 1
    np.testing.assert_array_equal(kept, expected)
    np.testing.assert_array_equal(np.diff(load_tractogram(out).offsets), lengths)


def test_vtk_weights_and_point_values_are_used(tmp_path: Path) -> None:
    vtk = tmp_path / "in.vtk"
    fa = np.arange(len(POINTS), dtype=np.float32)
    save_tractogram(Tractogram(POINTS, OFFSETS, {"fa": fa}, {"weight": WEIGHTS}), vtk)
    out, weights_out = tmp_path / "out.tck", tmp_path / "out_weights.txt"

    _edit(
        vtk, out, "--min-weight", "1", "--weights-out", weights_out, "--tsf-out", "fa"
    )

    assert _kept(out) == [1, 3, 4]
    np.testing.assert_array_equal(
        read_streamline_weights(weights_out), WEIGHTS[[1, 3, 4]]
    )
    values, _ = read_tsf(tmp_path / "out_fa.tsf", tractogram=load_tractogram(out))
    expected = np.concatenate([fa[OFFSETS[s] : OFFSETS[s + 1]] for s in (1, 3, 4)])
    np.testing.assert_array_equal(values, expected)


def test_vtk_without_weights_is_refused_once_read(tmp_path: Path) -> None:
    vtk = tmp_path / "in.vtk"
    save_tractogram(Tractogram(POINTS, OFFSETS), vtk)

    result = _run(vtk, tmp_path / "out.tck", "--min-weight", "1")

    assert result.exit_code == 1 and "needs weights" in result.output, result.output
    assert not (tmp_path / "out.tck").exists()


def test_lazy_tck_output_does_not_report_the_weights_as_dropped(
    tmp_path: Path, tck: Path
) -> None:
    weights_in = tmp_path / "in_weights.txt"
    write_streamline_weights(WEIGHTS, weights_in)
    weights_out = tmp_path / "out_weights.txt"

    result = _edit(
        tck,
        tmp_path / "out.tck",
        *("--weights-in", weights_in, "--weights-out", weights_out),
        *("--lazy", "--info"),
    )

    assert "cannot hold" not in result.output, result.output


@pytest.mark.parametrize("lazy", [False, True])
def test_tck_output_does_not_report_side_file_entries_as_dropped(
    tmp_path: Path, tck: Path, lazy: bool
) -> None:
    weights_in = tmp_path / "in_weights.txt"
    write_streamline_weights(WEIGHTS, weights_in)
    side = ["--weights-in", str(weights_in)]
    if not lazy:
        values = tmp_path / "values.tsf"
        write_tsf(np.ones(len(POINTS), np.float32), OFFSETS, values)
        side += ["--tsf-in", str(values)]

    result = _edit(
        tck, tmp_path / "out.tck", *side, *(["--lazy"] if lazy else []), "--info"
    )

    assert "cannot hold" not in result.output, result.output


def test_tck_output_reports_the_inputs_own_data_as_dropped(tmp_path: Path) -> None:
    trk = tmp_path / "in.trk"
    save_tractogram(
        Tractogram(
            POINTS, OFFSETS, streamline_data={"weight": WEIGHTS}, reference=GRID
        ),
        trk,
    )

    result = _edit(trk, tmp_path / "out.tck")

    assert "cannot hold" in result.output, result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--seed", "1"], "seed applies only with subsample"),
        (["--ends-only"], "ends_only applies to include"),
    ],
)
def test_options_that_apply_to_nothing_are_refused(
    tmp_path: Path, tck: Path, args: list[str], message: str
) -> None:
    result = _run(tck, tmp_path / "out.tck", *args)

    assert result.exit_code == 1 and message in result.output, result.output
    assert not (tmp_path / "out.tck").exists()


@pytest.mark.parametrize(
    ("name", "option", "missing"),
    [
        ("out.tck", "--mask", "missing_mask.nii.gz"),
        ("out.trk", "--reference", "missing.nii.gz"),
    ],
    ids=["regions", "reference"],
)
def test_existing_output_is_refused_before_the_input_option_is_read(
    tmp_path: Path, tck: Path, name: str, option: str, missing: str
) -> None:
    out = tmp_path / name
    out.write_text("old")

    result = _run(tck, out, option, tmp_path / missing)

    assert result.exit_code == 1 and "exists" in result.output, result.output
    assert out.read_text() == "old"


MRITOOLKIT = Path(__file__).parents[1] / "data" / "mritoolkit"


def test_mritoolkit_tracts_are_edited_with_a_reference(tmp_path: Path) -> None:
    out = tmp_path / "kept.tck"
    fit = MRITOOLKIT / "dti_fit.mat"
    result = _run(MRITOOLKIT / "tracts.mat", out, "--reference", fit, "--number", "5")
    assert result.exit_code == 0, result.output
    expected = read_mritoolkit_tracts(MRITOOLKIT / "tracts.mat", fit)[:5]
    np.testing.assert_allclose(load_tractogram(out).points, expected.points, atol=1e-5)


def test_mritoolkit_tracts_without_a_reference_are_refused(tmp_path: Path) -> None:
    result = _run(MRITOOLKIT / "tracts.mat", tmp_path / "kept.tck")
    assert result.exit_code == 1
    assert "ERROR: " in result.output and "stores no grid" in result.output
