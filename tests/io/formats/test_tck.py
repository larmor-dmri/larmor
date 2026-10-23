import gzip
import logging
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from larmor.io.formats import read_tck, write_tck
from larmor.types import Grid, Tractogram

REF = Path(__file__).parents[2] / "data" / "tck"

needs_mrtrix = pytest.mark.skipif(
    shutil.which("tckconvert") is None or shutil.which("tckinfo") is None,
    reason="MRtrix3 (tckconvert, tckinfo) is not on PATH",
)

DATATYPES = {
    "Float32LE": "<f4",
    "Float32BE": ">f4",
    "Float64LE": "<f8",
    "Float64BE": ">f8",
}

NAN = [np.nan] * 3
INF = [np.inf] * 3

# Lengths with a single point, a long streamline and empty ones at the start, in the
# middle and at the end, so that an off-by-one at any boundary shows.
LENGTHS = [0, 4, 1, 0, 0, 3, 0]


def _streamlines(lengths: list[int], seed: int = 0) -> list[np.ndarray]:
    # Multiples of 1/8: exact in float32 and in MRtrix's 6-digit text output.
    rng = np.random.default_rng(seed)
    return [rng.integers(-8000, 8000, size=(n, 3)) / 8 for n in lengths]


def _tractogram(streamlines: list[np.ndarray]) -> Tractogram:
    lengths = [len(s) for s in streamlines]
    points = np.concatenate([np.empty((0, 3)), *streamlines])
    offsets = np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)])
    return Tractogram(points, offsets)


def _rows(streamlines: list[np.ndarray], end: bool = True) -> np.ndarray:
    rows: list[np.ndarray | list[float]] = []
    for s in streamlines:
        rows.extend(s)
        rows.append(NAN)
    if end:
        rows.append(INF)
    return np.array(rows, dtype=np.float64).reshape(-1, 3)


def _tck(
    path: Path,
    values: np.ndarray,
    datatype: str = "Float32LE",
    dtype: str = "<f4",
    offset: int = 128,
    header: str | None = None,
) -> Path:
    if header is None:
        header = f"mrtrix tracks\ndatatype: {datatype}\nfile: . {offset}\nEND\n"
    raw = header.encode()
    assert len(raw) <= offset
    path.write_bytes(raw.ljust(offset, b"\0") + values.astype(dtype).tobytes())
    return path


def _assert_same(got: Tractogram, streamlines: list[np.ndarray]) -> None:
    want = _tractogram(streamlines)
    assert got.points.dtype == np.float32
    np.testing.assert_array_equal(got.offsets, want.offsets)
    np.testing.assert_array_equal(got.points, want.points)


def _ref_streamlines() -> list[np.ndarray]:
    files = sorted(REF.glob("s-*.txt"), key=lambda p: int(p.stem[2:]))
    return [np.loadtxt(f, ndmin=2) for f in files]


def _tckconvert_to_text(path: Path, out: Path) -> list[np.ndarray]:
    out.mkdir()
    subprocess.run(
        ["tckconvert", "-quiet", "-force", str(path), str(out / "s-[].txt")],
        check=True,
        capture_output=True,
    )
    files = sorted(out.glob("s-*.txt"), key=lambda p: int(p.stem[2:]))
    # MRtrix writes an empty streamline as an empty file.
    return [
        np.loadtxt(f, ndmin=2) if f.stat().st_size else np.empty((0, 3)) for f in files
    ]


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


# Reading


def test_mrtrix_reference_file_is_read() -> None:
    got = read_tck(REF / "ref.tck")
    _assert_same(got, _ref_streamlines())


def test_read_result_has_no_reference_and_no_data() -> None:
    got = read_tck(REF / "ref.tck")
    assert got.reference is None
    assert got.point_data == {}
    assert got.streamline_data == {}


@pytest.mark.parametrize("as_str", [False, True])
def test_path_may_be_str_or_path(as_str: bool) -> None:
    path = REF / "ref.tck"
    got = read_tck(str(path) if as_str else path)
    _assert_same(got, _ref_streamlines())


@needs_mrtrix
def test_file_written_by_tckconvert_is_read(tmp_path: Path) -> None:
    streamlines = _streamlines([6, 1, 2, 9], seed=1)
    for s, points in enumerate(streamlines):
        np.savetxt(tmp_path / f"s-{s}.txt", points, fmt="%.3f")
    subprocess.run(
        ["tckconvert", "-quiet", "-force", "s-[].txt", "out.tck"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    _assert_same(read_tck(tmp_path / "out.tck"), streamlines)


@pytest.mark.parametrize("name", ["ref.tck.gz", "ref.tck"])
def test_gzip_file_is_read(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_bytes(gzip.compress((REF / "ref.tck").read_bytes()))
    _assert_same(read_tck(path), _ref_streamlines())


@pytest.mark.parametrize("datatype", list(DATATYPES))
def test_every_datatype_is_read_as_float32(tmp_path: Path, datatype: str) -> None:
    streamlines = _streamlines([3, 1, 2])
    path = _tck(tmp_path / "t.tck", _rows(streamlines), datatype, DATATYPES[datatype])
    _assert_same(read_tck(path), streamlines)


def test_float64_values_are_rounded_to_float32(tmp_path: Path) -> None:
    # 0.1 and 1/3 are not float32 values: an unconverted read keeps float64.
    streamlines = [np.array([[0.1, 1 / 3, -2.7], [1e-7, 123.456789, 5.0]])]
    path = _tck(tmp_path / "t.tck", _rows(streamlines), "Float64LE", "<f8")
    got = read_tck(path)
    assert got.points.dtype == np.float32
    np.testing.assert_array_equal(got.points, streamlines[0].astype(np.float32))


@pytest.mark.parametrize(
    ("datatype", "dtype"),
    [("float32le", "<f4"), ("FLOAT32BE", ">f4"), ("float64Be", ">f8")],
)
def test_datatype_case_is_ignored(tmp_path: Path, datatype: str, dtype: str) -> None:
    streamlines = _streamlines([2, 3])
    path = _tck(tmp_path / "t.tck", _rows(streamlines), datatype, dtype)
    _assert_same(read_tck(path), streamlines)


@needs_mrtrix
@pytest.mark.parametrize("datatype", list(DATATYPES))
def test_hand_written_file_reads_as_mrtrix_reads_it(
    tmp_path: Path, datatype: str
) -> None:
    streamlines = _streamlines(LENGTHS)
    path = _tck(tmp_path / "t.tck", _rows(streamlines), datatype, DATATYPES[datatype])
    mrtrix = _tckconvert_to_text(path, tmp_path / "text")
    _assert_same(read_tck(path), mrtrix)


@pytest.mark.parametrize(
    "datatype", ["Int32LE", "Float16LE", "Float32", "Float32XE", "Float32LEX", ""]
)
def test_unsupported_datatype_raises(tmp_path: Path, datatype: str) -> None:
    path = _tck(tmp_path / "t.tck", _rows(_streamlines([2])), datatype)
    with pytest.raises(ValueError):
        read_tck(path)


def test_trailing_spaces_after_magic_are_allowed(tmp_path: Path) -> None:
    streamlines = _streamlines([2, 1])
    header = "mrtrix tracks    \ndatatype: Float32LE\nfile: . 128\nEND\n"
    path = _tck(tmp_path / "t.tck", _rows(streamlines), header=header)
    _assert_same(read_tck(path), streamlines)


@pytest.mark.parametrize(
    "magic", ["mrtrix image", "mrtrix tracksx", "mrtrix", "datatype: Float32LE", ""]
)
def test_wrong_magic_raises(tmp_path: Path, magic: str) -> None:
    header = f"{magic}\ndatatype: Float32LE\nfile: . 128\nEND\n"
    path = _tck(tmp_path / "t.tck", _rows(_streamlines([2])), header=header)
    with pytest.raises(ValueError):
        read_tck(path)


@pytest.mark.parametrize("missing", ["datatype", "file"])
def test_missing_key_raises_naming_the_file(tmp_path: Path, missing: str) -> None:
    lines = {"datatype": "datatype: Float32LE\n", "file": "file: . 128\n"}
    del lines[missing]
    header = "mrtrix tracks\n" + "".join(lines.values()) + "END\n"
    path = _tck(tmp_path / "t.tck", _rows(_streamlines([2])), header=header)
    with pytest.raises(ValueError, match=re.escape(path.name)):
        read_tck(path)


def test_header_without_end_raises(tmp_path: Path) -> None:
    header = "mrtrix tracks\ndatatype: Float32LE\nfile: . 128\n"
    path = tmp_path / "t.tck"
    path.write_bytes(header.encode())
    with pytest.raises(ValueError):
        read_tck(path)


@pytest.mark.parametrize(
    "entries",
    [
        ["other.dat 128"],
        [".. 128"],
        [". abc"],
        [". 128.5"],
        ["."],
        [". 128", ". 128"],
    ],
)
def test_malformed_file_entry_raises(tmp_path: Path, entries: list[str]) -> None:
    files = "".join(f"file: {e}\n" for e in entries)
    header = f"mrtrix tracks\ndatatype: Float32LE\n{files}END\n"
    path = _tck(tmp_path / "t.tck", _rows(_streamlines([2])), header=header)
    with pytest.raises(ValueError):
        read_tck(path)


@pytest.mark.parametrize("offset", [61, 128, 1000])
def test_data_starts_at_the_file_offset(tmp_path: Path, offset: int) -> None:
    streamlines = _streamlines([2, 3])
    path = _tck(tmp_path / "t.tck", _rows(streamlines), offset=offset)
    _assert_same(read_tck(path), streamlines)


def test_data_after_the_first_inf_is_ignored(tmp_path: Path) -> None:
    streamlines = _streamlines([2, 3])
    after = _rows(_streamlines([4, 1], seed=1))
    path = _tck(tmp_path / "t.tck", np.concatenate([_rows(streamlines), after]))
    _assert_same(read_tck(path), streamlines)


def test_file_without_inf_is_read_without_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    streamlines = _streamlines([2, 3, 1])
    path = _tck(tmp_path / "t.tck", _rows(streamlines, end=False))
    with caplog.at_level(logging.INFO, logger="larmor"):
        got = read_tck(path)
    _assert_same(got, streamlines)
    assert _warnings(caplog) == []


def test_complete_file_is_read_without_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="larmor"):
        read_tck(REF / "ref.tck")
    assert _warnings(caplog) == []


def test_empty_streamlines_are_kept(tmp_path: Path) -> None:
    streamlines = _streamlines(LENGTHS)
    path = _tck(tmp_path / "t.tck", _rows(streamlines))
    got = read_tck(path)
    _assert_same(got, streamlines)
    assert np.diff(got.offsets).tolist() == LENGTHS


@pytest.mark.parametrize(
    "rows",
    [np.array([INF]), np.empty((0, 3)), np.array([INF, [1.0, 2, 3], NAN])],
    ids=["inf-only", "no-data", "inf-first"],
)
def test_no_streamlines_gives_empty_tractogram(
    tmp_path: Path, rows: np.ndarray
) -> None:
    got = read_tck(_tck(tmp_path / "t.tck", rows))
    assert got.points.shape == (0, 3)
    assert got.points.dtype == np.float32
    assert got.offsets.tolist() == [0]


# Cut points, in float32 values after the last complete streamline (2 points + NaN):
# whole points without a NaN, part of a point, part of the closing NaN triplet.
@pytest.mark.parametrize(
    "extra",
    [
        [1.0, 2, 3],
        [1.0, 2, 3, 4, 5, 6],
        [1.0],
        [1.0, 2],
        [1.0, 2, 3, 4],
        [1.0, 2, 3, np.nan],
        [1.0, 2, 3, np.nan, np.nan],
    ],
)
def test_file_cut_in_last_streamline_drops_it_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, extra: list[float]
) -> None:
    streamlines = _streamlines([2, 3])
    values = np.concatenate([_rows(streamlines, end=False).ravel(), extra])
    path = _tck(tmp_path / "t.tck", values)
    with caplog.at_level(logging.WARNING, logger="larmor"):
        got = read_tck(path)
    _assert_same(got, streamlines)
    assert _warnings(caplog) != []


@pytest.mark.parametrize("n_values", [1, 3, 7])
def test_file_cut_in_first_streamline_has_no_streamlines_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, n_values: int
) -> None:
    values = _streamlines([4])[0].ravel()[:n_values]
    path = _tck(tmp_path / "t.tck", values)
    with caplog.at_level(logging.WARNING, logger="larmor"):
        got = read_tck(path)
    _assert_same(got, [])
    assert _warnings(caplog) != []


@pytest.mark.parametrize("cut_bytes", [1, 2, 3, 5, 11])
def test_file_cut_inside_a_number_drops_last_streamline_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, cut_bytes: int
) -> None:
    streamlines = _streamlines([2, 3])
    whole = _rows([*streamlines, *_streamlines([2], seed=1)], end=False)
    path = _tck(tmp_path / "t.tck", whole)
    raw = path.read_bytes()
    # Cut cut_bytes into the last streamline's data, mid-number unless a multiple of 4.
    keep = 128 + _rows(streamlines, end=False).size * 4 + cut_bytes
    path.write_bytes(raw[:keep])
    with caplog.at_level(logging.WARNING, logger="larmor"):
        got = read_tck(path)
    _assert_same(got, streamlines)
    assert _warnings(caplog) != []


@pytest.mark.parametrize("n_inf", [1, 2])
def test_file_cut_in_the_inf_triplet_keeps_streamlines_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, n_inf: int
) -> None:
    streamlines = _streamlines([2, 3])
    values = np.concatenate([_rows(streamlines, end=False).ravel(), [np.inf] * n_inf])
    path = _tck(tmp_path / "t.tck", values)
    with caplog.at_level(logging.WARNING, logger="larmor"):
        got = read_tck(path)
    _assert_same(got, streamlines)
    assert _warnings(caplog) != []


def test_mrtrix_file_cut_mid_streamline_is_read_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw = (REF / "ref.tck").read_bytes()
    # The file ends with the last streamline's 2 points, a NaN and an Inf triplet, 12
    # bytes each: cutting 30 bytes leaves its first point and 1.5 values of its second.
    path = tmp_path / "cut.tck"
    path.write_bytes(raw[:-30])
    with caplog.at_level(logging.WARNING, logger="larmor"):
        got = read_tck(path)
    _assert_same(got, _ref_streamlines()[:-1])
    assert _warnings(caplog) != []


@pytest.mark.parametrize(
    "row",
    [
        [np.nan, 1, 2],
        [1, np.nan, 2],
        [1, 2, np.nan],
        [np.nan, np.nan, 2],
        [1, np.nan, np.nan],
        [np.nan, 1, np.nan],
    ],
)
def test_partly_nan_row_raises(tmp_path: Path, row: list[float]) -> None:
    streamlines = _streamlines([2, 3])
    rows = _rows(streamlines)
    rows = np.insert(rows, 1, row, axis=0)
    with pytest.raises(ValueError):
        read_tck(_tck(tmp_path / "t.tck", rows))


@pytest.mark.parametrize("row", [[1, np.nan, 2], [1, 2, np.nan]])
def test_partly_nan_row_in_unclosed_last_streamline_raises(
    tmp_path: Path, row: list[float]
) -> None:
    unclosed = np.array([[1.0, 2, 3], row, [4.0, 5, 6]])
    rows = np.concatenate([_rows(_streamlines([2]), end=False), unclosed])
    with pytest.raises(ValueError, match="some coordinates NaN"):
        read_tck(_tck(tmp_path / "t.tck", rows))


def test_partly_nan_row_after_the_end_marker_is_ignored(tmp_path: Path) -> None:
    streamlines = _streamlines([2, 3])
    after = np.array([[1, np.nan, 2], [np.nan, 1, 2]])
    path = _tck(tmp_path / "t.tck", np.concatenate([_rows(streamlines), after]))
    _assert_same(read_tck(path), streamlines)


def test_negative_infinite_x_ends_the_data(tmp_path: Path) -> None:
    streamlines = _streamlines([2, 3])
    rows = np.concatenate(
        [
            _rows(streamlines, end=False),
            [[-np.inf, np.nan, 0]],
            _rows(_streamlines([1], seed=1)),
        ]
    )
    _assert_same(read_tck(_tck(tmp_path / "t.tck", rows)), streamlines)


@pytest.mark.parametrize("count", ["", "count: n/a\n", "count: -3\n"])
def test_missing_or_non_numeric_count_is_logged_as_info(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, count: str
) -> None:
    streamlines = _streamlines([2, 3, 1])
    header = f"mrtrix tracks\ndatatype: Float32LE\n{count}file: . 128\nEND\n"
    path = _tck(tmp_path / "t.tck", _rows(streamlines), header=header)
    with caplog.at_level(logging.INFO, logger="larmor"):
        got = read_tck(path)
    _assert_same(got, streamlines)
    assert _warnings(caplog) == []
    assert [r for r in caplog.records if r.levelno == logging.INFO] != []


@pytest.mark.parametrize("count", [0, 1, 7])
def test_count_disagreeing_with_data_reads_all_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, count: int
) -> None:
    streamlines = _streamlines([2, 3, 1])
    header = f"mrtrix tracks\ndatatype: Float32LE\ncount: {count}\nfile: . 128\nEND\n"
    path = _tck(tmp_path / "t.tck", _rows(streamlines), header=header)
    with caplog.at_level(logging.INFO, logger="larmor"):
        got = read_tck(path)
    _assert_same(got, streamlines)
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "incomplete" in warnings[0].getMessage()
    assert f"count is {count}" in warnings[0].getMessage()


def test_correct_count_reads_silently(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    streamlines = _streamlines([2, 3, 1])
    header = "mrtrix tracks\ndatatype: Float32LE\ncount: 3\nfile: . 128\nEND\n"
    path = _tck(tmp_path / "t.tck", _rows(streamlines), header=header)
    with caplog.at_level(logging.INFO, logger="larmor"):
        got = read_tck(path)
    _assert_same(got, streamlines)
    assert caplog.records == []


# Writing


def _split(raw: bytes) -> tuple[list[str], int]:
    end = raw.index(b"\nEND\n") + len(b"\nEND\n")
    lines = raw[:end].decode().splitlines()
    return lines, end


def test_written_file_has_the_tck_layout(tmp_path: Path) -> None:
    streamlines = _streamlines(LENGTHS)
    path = tmp_path / "t.tck"
    write_tck(_tractogram(streamlines), path)
    raw = path.read_bytes()
    lines, end = _split(raw)

    assert lines[0] == "mrtrix tracks"
    assert lines[-1] == "END"
    assert "datatype: Float32LE" in lines
    assert f"count: {len(LENGTHS)}" in lines
    files = [line for line in lines if line.startswith("file:")]
    assert len(files) == 1
    name, offset = files[0].removeprefix("file:").split()
    assert name == "."
    assert int(offset) >= end
    values = np.frombuffer(raw[int(offset) :], dtype="<f4")
    np.testing.assert_array_equal(values, _rows(streamlines).astype("<f4").ravel())


def test_timestamp_is_written_before_the_datatype(tmp_path: Path) -> None:
    path = tmp_path / "t.tck"
    write_tck(_tractogram(_streamlines(LENGTHS)), path, timestamp="1792186512.84")
    lines, _ = _split(path.read_bytes())
    assert lines[1:3] == ["timestamp: 1792186512.84", "datatype: Float32LE"]


@pytest.mark.parametrize("timestamp", ["1#2", "1\n2", "1\r2"])
def test_timestamp_a_header_line_cannot_hold_is_refused(
    tmp_path: Path, timestamp: str
) -> None:
    path = tmp_path / "t.tck"
    with pytest.raises(ValueError, match="timestamp"):
        write_tck(_tractogram(_streamlines([2])), path, timestamp=timestamp)
    assert not path.exists()


@pytest.mark.parametrize("lengths", [LENGTHS, [1], [0], [], [2, 0, 0, 5]], ids=str)
def test_round_trip_is_exact(tmp_path: Path, lengths: list[int]) -> None:
    # Full float32 values, not multiples of 1/8, so that any rounding shows.
    rng = np.random.default_rng(3)
    streamlines = [
        (rng.standard_normal((n, 3)) * 50).astype(np.float32) for n in lengths
    ]
    path = tmp_path / "t.tck"
    write_tck(_tractogram(streamlines), path)
    _assert_same(read_tck(path), streamlines)


@needs_mrtrix
def test_mrtrix_reads_the_written_file(tmp_path: Path) -> None:
    streamlines = _streamlines(LENGTHS)
    path = tmp_path / "t.tck"
    write_tck(_tractogram(streamlines), path)
    info = subprocess.run(
        ["tckinfo", "-count", str(path)], check=True, capture_output=True, text=True
    ).stdout
    assert re.search(rf"^\s*count:\s*{len(LENGTHS)}\s*$", info, re.M)
    assert re.search(rf"actual count in file:\s*{len(LENGTHS)}\s*$", info, re.M)
    mrtrix = _tckconvert_to_text(path, tmp_path / "text")
    _assert_same(_tractogram(mrtrix), streamlines)


@pytest.mark.parametrize(
    ("point_data", "streamline_data"),
    [(True, False), (False, True), (True, True)],
)
def test_data_that_tck_cannot_hold_is_dropped_with_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    point_data: bool,
    streamline_data: bool,
) -> None:
    streamlines = _streamlines([2, 3])
    plain = _tractogram(streamlines)
    t = Tractogram(
        plain.points,
        plain.offsets,
        point_data={"fa": np.arange(5.0)} if point_data else {},
        streamline_data={"length": np.arange(2.0)} if streamline_data else {},
    )
    path = tmp_path / "t.tck"
    with caplog.at_level(logging.WARNING, logger="larmor"):
        write_tck(t, path)
    assert _warnings(caplog) != []
    _assert_same(read_tck(path), streamlines)


def test_plain_tractogram_is_written_without_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="larmor"):
        write_tck(_tractogram(_streamlines([2, 3])), tmp_path / "t.tck")
    assert _warnings(caplog) == []


def test_reference_is_not_written(tmp_path: Path) -> None:
    streamlines = _streamlines([2, 3])
    plain = _tractogram(streamlines)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    t = Tractogram(plain.points, plain.offsets, reference=Grid((4, 5, 6), affine))
    path = tmp_path / "t.tck"
    write_tck(t, path)
    with_reference = path.read_bytes()
    write_tck(plain, path)
    assert with_reference == path.read_bytes()


@pytest.mark.parametrize("as_str", [False, True])
def test_existing_file_is_overwritten(tmp_path: Path, as_str: bool) -> None:
    path = tmp_path / "t.tck"
    write_tck(_tractogram(_streamlines([9, 9, 9], seed=1)), path)
    streamlines = _streamlines([2])
    write_tck(_tractogram(streamlines), str(path) if as_str else path)
    _assert_same(read_tck(path), streamlines)


@pytest.mark.parametrize(
    "point", [(np.nan, np.nan, np.nan), (np.inf, 0, 0), (1, np.nan, 2), (0, 0, -np.inf)]
)
def test_write_refuses_points_that_are_not_finite(
    tmp_path: Path, point: tuple[float, float, float]
) -> None:
    streamlines = _streamlines([3, 2])
    streamlines[0][1] = point
    path = tmp_path / "a.tck"
    with pytest.raises(ValueError, match="NaN or infinite"):
        write_tck(_tractogram(streamlines), path)
    assert not path.exists()
