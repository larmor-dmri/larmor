import gc
import logging
import re
import shutil
import subprocess
import tracemalloc
from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from larmor.io import BatchWriter, open_tractogram_writer, save_tractogram
from larmor.io.formats import read_tck, read_trk, write_tck, write_trk
from larmor.io.formats.tck import TckWriter
from larmor.io.formats.trk import TrkWriter
from larmor.types import Grid, Tractogram

REF = Path(__file__).parents[1] / "data" / "tck"

needs_mrtrix = pytest.mark.skipif(
    shutil.which("tckinfo") is None, reason="MRtrix3 (tckinfo) is not on PATH"
)

GRID = Grid((4, 5, 6), np.diag([2.0, 2.5, 3.0, 1.0]))
OTHER_GRID = Grid((9, 9, 9), np.diag([-1.0, 1.0, 1.5, 1.0]))

# Zero-length streamlines alone and next to others, at the start of a batch and in
# the middle, so that an offset error at any batch boundary shows.
LENGTHS = (3, 0, 1, 5, 0, 0, 2, 4)
S = len(LENGTHS)

# Batch boundaries (indices into the streamlines); repeated values are empty batches.
SPLITS = {
    "one": [0, S],
    "with-empty": [0, 0, 3, 3, 4, 8, 8],
    "singles": list(range(S + 1)),
    "zero-length-only": [0, 1, 2, 4, 6, 8],
}
EACH_SPLIT = pytest.mark.parametrize("bounds", SPLITS.values(), ids=list(SPLITS))

DATA = {
    "none": ({}, {}),
    "point-1d": ({"fa": 1}, {}),
    "point-columns": ({"dir": 3}, {}),
    "streamline-1d": ({}, {"length": 1}),
    "streamline-columns": ({}, {"rgb": 3}),
    "all": ({"fa": 1, "dir": 3}, {"length": 1, "rgb": 3}),
}
EACH_DATA = pytest.mark.parametrize("data", DATA.values(), ids=list(DATA))


def _values(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    shape = (n,) if k == 1 else (n, k)
    return rng.uniform(-1, 1, shape).astype(np.float32)


def _tractogram(
    lengths: tuple[int, ...] = LENGTHS,
    data: tuple[dict[str, int], dict[str, int]] = ({}, {}),
    reference: Grid | None = None,
    seed: int = 0,
) -> Tractogram:
    rng = np.random.default_rng(seed)
    n = sum(lengths)
    point_sizes, streamline_sizes = data
    return Tractogram(
        rng.uniform(0, 10, (n, 3)).astype(np.float32),
        np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)]),
        point_data={k: _values(rng, n, v) for k, v in point_sizes.items()},
        streamline_data={
            k: _values(rng, len(lengths), v) for k, v in streamline_sizes.items()
        },
        reference=reference,
    )


def _batches(t: Tractogram, bounds: list[int]) -> list[Tractogram]:
    return [t[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True)]


def _write(writer: BatchWriter, batches: list[Tractogram]) -> None:
    with writer:
        for batch in batches:
            writer.write(batch)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


def _assert_same(got: Tractogram, want: Tractogram, exact: bool = True) -> None:
    np.testing.assert_array_equal(got.offsets, want.offsets)
    if exact:
        np.testing.assert_array_equal(got.points, want.points)
    else:
        # TRK stores voxel mm in float32: a few roundings of ~10 mm values.
        np.testing.assert_allclose(got.points, want.points, rtol=1e-6, atol=1e-5)
    assert set(got.point_data) == set(want.point_data)
    assert set(got.streamline_data) == set(want.streamline_data)
    for name, values in want.point_data.items():
        np.testing.assert_array_equal(got.point_data[name], values, strict=True)
    for name, values in want.streamline_data.items():
        np.testing.assert_array_equal(got.streamline_data[name], values, strict=True)


def _plain(t: Tractogram) -> Tractogram:
    return Tractogram(t.points, t.offsets)


# Same bytes as writing the whole tractogram


@EACH_SPLIT
@pytest.mark.parametrize("data", [DATA["none"], DATA["all"]], ids=["plain", "data"])
def test_tck_in_batches_is_byte_identical_to_write_tck(
    tmp_path: Path, bounds: list[int], data: tuple[dict[str, int], dict[str, int]]
) -> None:
    t = _tractogram(data=data)
    write_tck(t, tmp_path / "whole.tck")
    _write(TckWriter(tmp_path / "batched.tck"), _batches(t, bounds))
    assert (tmp_path / "batched.tck").read_bytes() == (
        tmp_path / "whole.tck"
    ).read_bytes()


@EACH_SPLIT
@EACH_DATA
def test_trk_in_batches_is_byte_identical_to_write_trk(
    tmp_path: Path, bounds: list[int], data: tuple[dict[str, int], dict[str, int]]
) -> None:
    t = _tractogram(data=data)
    write_trk(t, tmp_path / "whole.trk", reference=GRID)
    _write(TrkWriter(tmp_path / "batched.trk", GRID), _batches(t, bounds))
    assert (tmp_path / "batched.trk").read_bytes() == (
        tmp_path / "whole.trk"
    ).read_bytes()


@pytest.mark.parametrize("name", ["t.tck", "t.trk"])
def test_zero_batches_is_byte_identical_to_an_empty_tractogram(
    tmp_path: Path, name: str
) -> None:
    empty = _tractogram(lengths=())
    save_tractogram(empty, tmp_path / f"whole-{name}", reference=GRID)
    _write(open_tractogram_writer(tmp_path / name, reference=GRID), [])
    assert (tmp_path / name).read_bytes() == (tmp_path / f"whole-{name}").read_bytes()


@pytest.mark.parametrize("name", ["t.tck", "t.trk"])
def test_open_tractogram_writer_matches_save_tractogram(
    tmp_path: Path, name: str
) -> None:
    t = _tractogram(data=DATA["all"])
    save_tractogram(t, tmp_path / f"whole-{name}", reference=GRID)
    _write(
        open_tractogram_writer(str(tmp_path / name), reference=GRID),
        _batches(t, SPLITS["with-empty"]),
    )
    assert (tmp_path / name).read_bytes() == (tmp_path / f"whole-{name}").read_bytes()


def test_trk_data_type_may_differ_between_batches(tmp_path: Path) -> None:
    t = _tractogram(data=DATA["all"])
    first, second = _batches(t, [0, 3, S])
    as_float64 = Tractogram(
        second.points,
        second.offsets,
        {k: v.astype(np.float64) for k, v in second.point_data.items()},
        {k: v.astype(np.float64) for k, v in second.streamline_data.items()},
    )
    write_trk(t, tmp_path / "whole.trk", reference=GRID)
    _write(TrkWriter(tmp_path / "batched.trk", GRID), [first, as_float64])
    assert (tmp_path / "batched.trk").read_bytes() == (
        tmp_path / "whole.trk"
    ).read_bytes()


# Reading back


@EACH_SPLIT
def test_tck_in_batches_reads_back_without_warning(
    tmp_path: Path, bounds: list[int], caplog: pytest.LogCaptureFixture
) -> None:
    t = _tractogram()
    path = tmp_path / "t.tck"
    _write(TckWriter(path), _batches(t, bounds))
    with caplog.at_level(logging.INFO, logger="larmor"):
        got = read_tck(path)
    _assert_same(got, t)
    assert caplog.records == []


@EACH_SPLIT
@EACH_DATA
def test_trk_in_batches_reads_back_without_warning(
    tmp_path: Path,
    bounds: list[int],
    data: tuple[dict[str, int], dict[str, int]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    t = _tractogram(data=data)
    path = tmp_path / "t.trk"
    _write(TrkWriter(path, GRID), _batches(t, bounds))
    with caplog.at_level(logging.INFO, logger="larmor"):
        got = read_trk(path)
    _assert_same(got, t, exact=False)
    assert got.reference is not None
    assert got.reference.matches(GRID)
    assert caplog.records == []


def test_trk_uses_the_writer_reference_not_the_batches(tmp_path: Path) -> None:
    t = _tractogram(data=DATA["all"], reference=OTHER_GRID)
    write_trk(t, tmp_path / "whole.trk", reference=GRID)
    _write(TrkWriter(tmp_path / "batched.trk", GRID), _batches(t, [0, 4, S]))
    assert (tmp_path / "batched.trk").read_bytes() == (
        tmp_path / "whole.trk"
    ).read_bytes()


@needs_mrtrix
@pytest.mark.parametrize("finished", [True, False], ids=["finished", "unfinished"])
def test_tckinfo_counts_a_tck_written_in_batches(
    tmp_path: Path, finished: bool
) -> None:
    t = _tractogram()
    path = tmp_path / "t.tck"
    batches = _batches(t, SPLITS["with-empty"])
    if finished:
        _write(TckWriter(path), batches)
    else:
        _interrupted(TckWriter(path), batches)
    info = subprocess.run(
        ["tckinfo", "-count", str(path)], check=True, capture_output=True, text=True
    ).stdout
    count = S if finished else 0
    assert re.search(rf"^\s*count:\s*{count}\s*$", info, re.M)
    assert re.search(rf"actual count in file:\s*{S}\s*$", info, re.M)


# Unfinished files


def _interrupted(writer: BatchWriter, batches: list[Tractogram]) -> None:
    with pytest.raises(RuntimeError, match="interrupted"), writer:
        for batch in batches:
            writer.write(batch)
        raise RuntimeError("interrupted")


def _dropped(writer: BatchWriter, batches: list[Tractogram]) -> None:
    # A run killed without unwinding: the writer is never closed or finished.
    for batch in batches:
        writer.write(batch)
    del writer
    gc.collect()


STOPS = {"exception": _interrupted, "never-closed": _dropped}
EACH_STOP = pytest.mark.parametrize("stop", STOPS.values(), ids=list(STOPS))


@EACH_STOP
def test_unfinished_tck_reads_all_written_with_warning(
    tmp_path: Path,
    stop: Callable[[BatchWriter, list[Tractogram]], None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    t = _tractogram()
    path = tmp_path / "t.tck"
    stop(TckWriter(path), _batches(t, SPLITS["with-empty"]))
    with caplog.at_level(logging.WARNING, logger="larmor"):
        got = read_tck(path)
    _assert_same(got, t)
    messages = [r.getMessage() for r in _warnings(caplog)]
    assert len(messages) == 1
    assert "count is 0" in messages[0]
    assert "incomplete" in messages[0]


@EACH_STOP
def test_unfinished_trk_reads_all_written_with_warning(
    tmp_path: Path,
    stop: Callable[[BatchWriter, list[Tractogram]], None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    t = _tractogram(data=DATA["all"])
    path = tmp_path / "t.trk"
    stop(TrkWriter(path, GRID), _batches(t, SPLITS["with-empty"]))
    with caplog.at_level(logging.WARNING, logger="larmor"):
        got = read_trk(path)
    _assert_same(got, t, exact=False)
    messages = [r.getMessage() for r in _warnings(caplog)]
    assert len(messages) == 1
    assert "was not finished" in messages[0]
    assert "incomplete" in messages[0]


def test_unfinished_trk_is_read_by_nibabel(tmp_path: Path) -> None:
    # Other readers ignore Larmor's marker; n_count 0 tells them to read to the end.
    # nibabel drops zero-length streamlines, so this tractogram has none.
    t = _tractogram(lengths=(3, 1, 5, 2, 4))
    path = tmp_path / "t.trk"
    _interrupted(TrkWriter(path, GRID), _batches(t, [0, 2, 5]))
    got = nib.streamlines.load(path).streamlines
    assert len(got) == t.num_streamlines
    np.testing.assert_allclose(got.get_data(), t.points, atol=1e-4)


@EACH_STOP
@pytest.mark.parametrize("name", ["t.tck", "t.trk"])
def test_unfinished_file_without_batches_reads_as_empty(
    tmp_path: Path, stop: Callable[[BatchWriter, list[Tractogram]], None], name: str
) -> None:
    path = tmp_path / name
    stop(open_tractogram_writer(path, reference=GRID), [])
    got = read_tck(path) if name == "t.tck" else read_trk(path)
    assert got.num_streamlines == 0
    assert got.points.shape == (0, 3)


def test_unfinished_trk_without_batches_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "t.trk"
    _interrupted(TrkWriter(path, GRID), [])
    with caplog.at_level(logging.WARNING, logger="larmor"):
        read_trk(path)
    assert "was not finished" in caplog.text


# TRK data layout


MISMATCHES = {
    "point-renamed": ({"fb": 1}, {"length": 1}),
    "point-size": ({"fa": 3}, {"length": 1}),
    "point-missing": ({}, {"length": 1}),
    "point-extra": ({"fa": 1, "dir": 3}, {"length": 1}),
    "streamline-renamed": ({"fa": 1}, {"len": 1}),
    "streamline-size": ({"fa": 1}, {"length": 2}),
    "streamline-missing": ({"fa": 1}, {}),
    "streamline-extra": ({"fa": 1}, {"length": 1, "rgb": 3}),
}


@pytest.mark.parametrize("other", MISMATCHES.values(), ids=list(MISMATCHES))
def test_trk_batch_with_other_data_layout_raises(
    tmp_path: Path, other: tuple[dict[str, int], dict[str, int]]
) -> None:
    first = _tractogram(lengths=(2, 3), data=({"fa": 1}, {"length": 1}))
    with TrkWriter(tmp_path / "t.trk", GRID) as writer:
        writer.write(first)
        with pytest.raises(ValueError, match="differ from the first batch"):
            writer.write(_tractogram(lengths=(1, 4), data=other, seed=1))


def test_trk_layout_is_fixed_by_an_empty_first_batch(tmp_path: Path) -> None:
    t = _tractogram(data=({"fa": 1}, {}))
    with TrkWriter(tmp_path / "t.trk", GRID) as writer:
        writer.write(_tractogram(lengths=()))
        with pytest.raises(ValueError, match="differ from the first batch"):
            writer.write(t)


def test_trk_refused_batch_leaves_the_file_as_before(tmp_path: Path) -> None:
    t = _tractogram(data=({"fa": 1}, {"length": 1}))
    first, second = _batches(t, [0, 3, S])
    with TrkWriter(tmp_path / "batched.trk", GRID) as writer:
        writer.write(first)
        with pytest.raises(ValueError):
            writer.write(_tractogram(lengths=(2,), data=({"fb": 1}, {"length": 1})))
        writer.write(second)
    write_trk(t, tmp_path / "whole.trk", reference=GRID)
    assert (tmp_path / "batched.trk").read_bytes() == (
        tmp_path / "whole.trk"
    ).read_bytes()


def test_trk_refused_first_batch_leaves_the_writer_usable(tmp_path: Path) -> None:
    t = _tractogram(data=({"fa": 1}, {"length": 1}))
    too_many = {f"p{i}": 1 for i in range(11)}
    with TrkWriter(tmp_path / "batched.trk", GRID) as writer:
        with pytest.raises(ValueError, match="batched.trk"):
            writer.write(_tractogram(lengths=(2,), data=({}, too_many)))
        writer.write(t)
    write_trk(t, tmp_path / "whole.trk", reference=GRID)
    assert (tmp_path / "batched.trk").read_bytes() == (
        tmp_path / "whole.trk"
    ).read_bytes()


def test_trk_refused_only_batch_still_closes(tmp_path: Path) -> None:
    too_many = {f"p{i}": 1 for i in range(11)}
    path = tmp_path / "t.trk"
    writer = TrkWriter(path, GRID)
    with pytest.raises(ValueError):
        writer.write(_tractogram(lengths=(2,), data=({}, too_many)))
    writer.close()
    assert read_trk(path).num_streamlines == 0


# TCK data and points


@pytest.mark.parametrize(
    "data",
    [DATA["point-1d"], DATA["streamline-columns"], DATA["all"]],
    ids=["point_data", "streamline_data", "both"],
)
def test_tck_batches_with_data_log_one_info_and_no_warning(
    tmp_path: Path,
    data: tuple[dict[str, int], dict[str, int]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    t = _tractogram(data=data)
    with caplog.at_level(logging.INFO, logger="larmor"):
        _write(TckWriter(tmp_path / "t.tck"), _batches(t, SPLITS["singles"]))
    assert _warnings(caplog) == []
    assert [r.levelno for r in caplog.records] == [logging.INFO]
    assert "point_data or streamline_data" in caplog.text


def test_tck_batches_without_data_log_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="larmor"):
        _write(
            TckWriter(tmp_path / "t.tck"), _batches(_tractogram(), SPLITS["singles"])
        )
    assert caplog.records == []


@pytest.mark.parametrize(
    "point", [(np.nan, np.nan, np.nan), (np.inf, 0, 0), (1, np.nan, 2), (0, 0, -np.inf)]
)
def test_tck_batch_with_points_that_are_not_finite_is_refused(
    tmp_path: Path, point: tuple[float, float, float]
) -> None:
    t = _tractogram()
    first, second = _batches(t, [0, 4, S])
    bad = _tractogram(lengths=(2, 3), seed=1)
    bad.points[3] = point
    with TckWriter(tmp_path / "batched.tck") as writer:
        writer.write(first)
        with pytest.raises(ValueError, match="NaN or infinite"):
            writer.write(bad)
        writer.write(second)
    write_tck(t, tmp_path / "whole.tck")
    assert (tmp_path / "batched.tck").read_bytes() == (
        tmp_path / "whole.tck"
    ).read_bytes()


# open_tractogram_writer


@pytest.mark.parametrize(("name", "kind"), [("t.tck", TckWriter), ("t.trk", TrkWriter)])
def test_writer_is_chosen_by_extension(
    tmp_path: Path, name: str, kind: type[BatchWriter]
) -> None:
    with open_tractogram_writer(tmp_path / name, reference=GRID) as writer:
        assert type(writer) is kind


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("t.tck.gz", "compressed"),
        ("t.trk.gz", "compressed"),
        ("t.vtk", "VTK"),
        ("t.fib", "VTK"),
    ],
)
def test_formats_that_cannot_be_written_in_batches_raise(
    tmp_path: Path, name: str, message: str
) -> None:
    path = tmp_path / name
    with pytest.raises(ValueError, match=message) as error:
        open_tractogram_writer(path, reference=GRID)
    assert name in str(error.value)
    assert not path.exists()


def test_trk_without_reference_raises(tmp_path: Path) -> None:
    path = tmp_path / "t.trk"
    with pytest.raises(ValueError, match="reference"):
        open_tractogram_writer(path)
    assert not path.exists()


def test_tck_needs_no_reference(tmp_path: Path) -> None:
    t = _tractogram()
    _write(open_tractogram_writer(tmp_path / "t.tck"), [t])
    _assert_same(read_tck(tmp_path / "t.tck"), t)


@pytest.mark.parametrize("name", ["t.tck", "t.trk"])
def test_close_twice_is_harmless(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    with open_tractogram_writer(path, reference=GRID) as writer:
        writer.write(_tractogram(data=DATA["all"]))
        writer.close()
        finished = path.read_bytes()
        writer.close()
    assert path.read_bytes() == finished
    save_tractogram(
        _tractogram(data=DATA["all"]), tmp_path / f"whole-{name}", reference=GRID
    )
    assert finished == (tmp_path / f"whole-{name}").read_bytes()


# Memory


BATCH_LENGTHS = (20,) * 1000


def _peak(writer: BatchWriter, n_batches: int) -> int:
    tracemalloc.start()
    try:
        with writer:
            for i in range(n_batches):
                # A new batch each time, as the tracker makes them, dropped after.
                writer.write(
                    _tractogram(BATCH_LENGTHS, ({"fa": 1}, {"length": 1}), seed=i)
                )
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("name", ["t.tck", "t.trk"])
def test_memory_while_writing_depends_on_the_batch_not_the_total(
    tmp_path: Path, name: str
) -> None:
    one = _peak(open_tractogram_writer(tmp_path / f"one-{name}", reference=GRID), 1)
    many = _peak(open_tractogram_writer(tmp_path / name, reference=GRID), 30)
    # 30 batches of 20,000 float32 points: the payload is 30 times one batch's.
    assert (tmp_path / name).stat().st_size > 25 * (
        tmp_path / f"one-{name}"
    ).stat().st_size
    assert many < 1.5 * one
    assert many < (tmp_path / name).stat().st_size / 5


# The TCK header layout


def _header_layout(raw: bytes) -> tuple[list[str], int, int]:
    end = raw.index(b"\nEND\n") + len(b"\nEND\n")
    lines = raw[:end].decode().splitlines()
    file_at = raw.index(b"\nfile: ") + 1
    match = re.search(rb"\nfile: \. (\d+)\n", raw[:end])
    assert match is not None
    return lines, file_at, int(match.group(1))


def _round4(n: int) -> int:
    return n + -n % 4


def test_mrtrix_reference_has_the_offset_rule() -> None:
    lines, file_at, offset = _header_layout((REF / "ref.tck").read_bytes())
    assert (file_at, offset) == (171, 236)
    assert offset == _round4(file_at + 65)
    assert lines[lines.index("file: . 236") + 1] == "count: 4"


@pytest.mark.parametrize("lengths", [(), (2,), LENGTHS * 2], ids=["0", "1", "16"])
def test_write_tck_header_follows_tckgen(
    tmp_path: Path, lengths: tuple[int, ...]
) -> None:
    path = tmp_path / "t.tck"
    write_tck(_tractogram(lengths=lengths), path)
    raw = path.read_bytes()
    lines, file_at, offset = _header_layout(raw)
    assert lines == [
        "mrtrix tracks",
        "datatype: Float32LE",
        f"file: . {offset}",
        f"count: {len(lengths)}",
        "END",
    ]
    assert offset == _round4(file_at + 65)
    assert raw[raw.index(b"\nEND\n") + 5 : offset].strip(b"\0") == b""


def test_unfinished_tck_header_has_the_same_offset(tmp_path: Path) -> None:
    path = tmp_path / "t.tck"
    _interrupted(TckWriter(path), [_tractogram()])
    unfinished = path.read_bytes()
    write_tck(_tractogram(), path)
    assert _header_layout(unfinished)[2] == _header_layout(path.read_bytes())[2]


def test_tck_batches_with_timestamp_match_write_tck(tmp_path: Path) -> None:
    whole = _tractogram(lengths=(3, 0, 5, 2))
    write_tck(whole, tmp_path / "whole.tck", timestamp="1792186512.84")
    with TckWriter(tmp_path / "batches.tck", timestamp="1792186512.84") as writer:
        writer.write(whole[:2])
        writer.write(whole[2:])
    raw = (tmp_path / "batches.tck").read_bytes()
    assert b"\ntimestamp: 1792186512.84\n" in raw
    assert raw == (tmp_path / "whole.tck").read_bytes()


def test_tck_writer_refuses_a_timestamp_with_a_line_break(tmp_path: Path) -> None:
    path = tmp_path / "t.tck"
    with pytest.raises(ValueError, match="timestamp"):
        TckWriter(path, timestamp="1\n2")
    assert not path.exists()
