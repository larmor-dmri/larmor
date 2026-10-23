import gzip
import logging
from pathlib import Path

import numpy as np
import pytest

from larmor.io import load_tractogram, read_track_timestamp, read_tsf, write_tsf
from larmor.io.formats import write_tck
from larmor.types import Tractogram

DATA = Path(__file__).parents[2] / "data"
TRACKS = DATA / "editing" / "tracks.tck"
BZERO = DATA / "tsf" / "bzero.tsf"
TIMESTAMP = "1792186512.8427672386"

# Empty streamlines at the start, in the middle and at the end, so that an
# off-by-one at any boundary shows.
LENGTHS = [0, 4, 1, 0, 0, 3, 0]


def _values(lengths: list[int]) -> tuple[np.ndarray, np.ndarray]:
    # Multiples of 1/8: exact in float32.
    rng = np.random.default_rng(0)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return rng.integers(-8000, 8000, size=offsets[-1]) / 8, offsets


def _payload(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    parts = [
        np.append(values[a:b], np.nan)
        for a, b in zip(offsets[:-1], offsets[1:], strict=True)
    ]
    return np.concatenate([np.empty(0), *parts])


def _tsf(
    path: Path, payload: np.ndarray, datatype: str = "Float32LE", dtype: str = "<f4"
) -> Path:
    header = f"mrtrix track scalars\ndatatype: {datatype}\nfile: . 128\nEND\n"
    path.write_bytes(
        header.encode().ljust(128, b"\0") + payload.astype(dtype).tobytes()
    )
    return path


def _data_offset(raw: bytes) -> int:
    return int(raw.split(b"\nfile: . ")[1].split(b"\n")[0])


@pytest.fixture(scope="module")
def tracks() -> Tractogram:
    return load_tractogram(TRACKS)


# Reading


def test_mrtrix_file_has_one_value_per_point_of_its_tractogram(
    tracks: Tractogram,
) -> None:
    values, offsets = read_tsf(BZERO, tractogram=tracks)
    assert values.dtype == np.float32
    assert len(values) == len(tracks.points)
    np.testing.assert_array_equal(offsets, tracks.offsets)
    assert np.isfinite(values).all()


@pytest.mark.parametrize("change", ["fewer_streamlines", "one_point_moved"])
def test_values_not_matching_the_tractogram_raise(
    tracks: Tractogram, change: str
) -> None:
    if change == "fewer_streamlines":
        end = tracks.offsets[-2]
        other = Tractogram(tracks.points[:end], tracks.offsets[:-1])
        message = "499"
    else:
        offsets = tracks.offsets.copy()
        offsets[1] += 1
        other = Tractogram(tracks.points, offsets)
        message = "streamline 0"
    with pytest.raises(ValueError, match=message):
        read_tsf(BZERO, tractogram=other)


@pytest.mark.parametrize("lengths", [LENGTHS, [0], [5]])
def test_hand_written_file_gives_values_and_offsets(
    tmp_path: Path, lengths: list[int]
) -> None:
    values, offsets = _values(lengths)
    got, got_offsets = read_tsf(_tsf(tmp_path / "v.tsf", _payload(values, offsets)))
    np.testing.assert_array_equal(got_offsets, offsets)
    np.testing.assert_array_equal(got, values)


@pytest.mark.parametrize(
    ("datatype", "dtype"),
    [
        ("Float32LE", "<f4"),
        ("Float32BE", ">f4"),
        ("Float64LE", "<f8"),
        ("Float64BE", ">f8"),
    ],
)
def test_every_datatype_is_read_as_stored(
    tmp_path: Path, datatype: str, dtype: str
) -> None:
    values, offsets = _values(LENGTHS)
    path = _tsf(tmp_path / "v.tsf", _payload(values, offsets), datatype, dtype)
    got, got_offsets = read_tsf(path)
    assert got.dtype == np.dtype(dtype).newbyteorder("=")
    np.testing.assert_array_equal(got, values)
    np.testing.assert_array_equal(got_offsets, offsets)


def test_data_after_the_first_inf_is_ignored(tmp_path: Path) -> None:
    values, offsets = _values([2, 3])
    payload = np.concatenate([_payload(values, offsets), [np.inf, 7.0, np.nan]])
    got, got_offsets = read_tsf(_tsf(tmp_path / "v.tsf", payload))
    np.testing.assert_array_equal(got, values)
    np.testing.assert_array_equal(got_offsets, offsets)


@pytest.mark.parametrize("name", ["v.tsf.gz", "v.tsf"])
def test_gzip_file_is_read(tmp_path: Path, tracks: Tractogram, name: str) -> None:
    path = tmp_path / name
    path.write_bytes(gzip.compress(BZERO.read_bytes()))
    got, _ = read_tsf(path, tractogram=tracks)
    np.testing.assert_array_equal(got, read_tsf(BZERO)[0])


def test_file_cut_in_last_streamline_drops_it_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    values, offsets = _values([2, 3])
    payload = np.concatenate([_payload(values, offsets), [1.0, 2.0]])
    with caplog.at_level(logging.WARNING, logger="larmor"):
        got, got_offsets = read_tsf(_tsf(tmp_path / "v.tsf", payload))
    np.testing.assert_array_equal(got, values)
    np.testing.assert_array_equal(got_offsets, offsets)
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


@pytest.mark.parametrize("cut", ["header", "gzip"])
def test_truncated_file_raises(tmp_path: Path, cut: str) -> None:
    raw = BZERO.read_bytes()
    path = tmp_path / "v.tsf"
    if cut == "header":
        path.write_bytes(raw[: _data_offset(raw) - 100])
    else:
        packed = gzip.compress(raw)
        path.write_bytes(packed[: len(packed) // 2])
    with pytest.raises(ValueError):
        read_tsf(path)


@pytest.mark.parametrize("magic", ["mrtrix tracks", "mrtrix track scalar", ""])
def test_wrong_magic_raises(tmp_path: Path, magic: str) -> None:
    path = tmp_path / "v.tsf"
    path.write_bytes(f"{magic}\ndatatype: Float32LE\nfile: . 64\nEND\n".encode())
    with pytest.raises(ValueError, match="not a TSF file"):
        read_tsf(path)


# Writing


def test_written_data_is_byte_identical_to_mrtrix(
    tmp_path: Path, tracks: Tractogram
) -> None:
    values, offsets = read_tsf(BZERO, tractogram=tracks)
    path = tmp_path / "v.tsf"
    write_tsf(values, offsets, path, timestamp=TIMESTAMP)
    raw, ref = path.read_bytes(), BZERO.read_bytes()
    assert raw[_data_offset(raw) :] == ref[_data_offset(ref) :]
    header = raw.split(b"\nEND\n")[0].decode().splitlines()
    assert f"timestamp: {TIMESTAMP}" in header
    assert "count: 500" in header


def test_no_timestamp_writes_no_timestamp_line(tmp_path: Path) -> None:
    values, offsets = _values([2])
    write_tsf(values, offsets, tmp_path / "v.tsf")
    assert b"timestamp" not in (tmp_path / "v.tsf").read_bytes()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("lengths", [LENGTHS, [0], [6]])
def test_round_trip_is_exact(
    tmp_path: Path, dtype: type[np.floating], lengths: list[int]
) -> None:
    values, offsets = _values(lengths)
    values = (values + 1 / 3).astype(dtype)
    path = tmp_path / "v.tsf"
    write_tsf(values, offsets, str(path))
    got, got_offsets = read_tsf(str(path))
    assert got.dtype == dtype
    np.testing.assert_array_equal(got, values)
    np.testing.assert_array_equal(got_offsets, offsets)


def test_big_endian_float64_round_trips_exactly_as_float64(tmp_path: Path) -> None:
    values, offsets = _values(LENGTHS)
    values = (values + 1 / 3).astype(">f8")
    path = tmp_path / "v.tsf"
    write_tsf(values, offsets, path)
    assert b"datatype: Float64LE" in path.read_bytes()
    got, _ = read_tsf(path)
    np.testing.assert_array_equal(got, values)


def test_unsigned_offsets_are_written_as_signed_ones(tmp_path: Path) -> None:
    values, offsets = _values(LENGTHS)
    write_tsf(values, offsets, tmp_path / "signed.tsf")
    write_tsf(values, offsets.astype(np.uint64), tmp_path / "unsigned.tsf")
    assert (tmp_path / "unsigned.tsf").read_bytes() == (
        tmp_path / "signed.tsf"
    ).read_bytes()


def test_existing_file_is_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "v.tsf"
    path.write_bytes(b"x" * 10_000)
    values, offsets = _values([3])
    write_tsf(values, offsets, path)
    np.testing.assert_array_equal(read_tsf(path)[0], values)


@pytest.mark.parametrize(
    ("values", "offsets"),
    [
        (np.zeros((3, 1)), np.array([0, 3])),
        (np.zeros(3), np.array([1, 3])),
        (np.zeros(3), np.array([0, 2])),
        (np.zeros(3), np.array([0, 2, 1, 3])),
        (np.zeros(3), np.array([0, 2, 1, 3], dtype=np.uint64)),
        (np.zeros(3), np.array([0.0, 3.0])),
        (np.zeros(3), np.array([[0, 3]])),
        (np.zeros(3), np.array([], dtype=np.int64)),
        (np.array([1.0, np.nan, 2.0]), np.array([0, 3])),
        (np.array([1.0, np.inf, 2.0]), np.array([0, 3])),
        (np.zeros(3, dtype=np.complex128), np.array([0, 3])),
    ],
    ids=[
        "values_2d",
        "not_from_0",
        "not_to_end",
        "decreasing",
        "decreasing_unsigned",
        "float_offsets",
        "offsets_2d",
        "no_offsets",
        "nan",
        "inf",
        "complex",
    ],
)
def test_invalid_input_is_refused_before_writing(
    tmp_path: Path, values: np.ndarray, offsets: np.ndarray
) -> None:
    path = tmp_path / "v.tsf"
    with pytest.raises(ValueError, match="Cannot write"):
        write_tsf(values, offsets, path)
    assert not path.exists()


@pytest.mark.parametrize("timestamp", ["1#2", "1\n2", "1\r2"])
def test_timestamp_a_header_line_cannot_hold_is_refused(
    tmp_path: Path, timestamp: str
) -> None:
    path = tmp_path / "v.tsf"
    with pytest.raises(ValueError, match="timestamp"):
        write_tsf(np.zeros(2), np.array([0, 2]), path, timestamp=timestamp)
    assert not path.exists()


# Timestamps


def test_mrtrix_tsf_shares_its_tractogram_timestamp() -> None:
    timestamp = read_track_timestamp(TRACKS)
    assert timestamp is not None
    assert read_track_timestamp(BZERO) == timestamp


@pytest.mark.parametrize("gz", [False, True], ids=["plain", "gzip"])
def test_timestamp_written_by_larmor_is_read_back(
    tmp_path: Path, tracks: Tractogram, gz: bool
) -> None:
    suffix = ".gz" if gz else ""
    values, offsets = _values([2, 3])
    write_tsf(values, offsets, tmp_path / f"v.tsf{suffix}", timestamp="12.5")
    write_tck(tracks, tmp_path / f"t.tck{suffix}", timestamp="12.5")
    assert read_track_timestamp(tmp_path / f"v.tsf{suffix}") == "12.5"
    assert read_track_timestamp(str(tmp_path / f"t.tck{suffix}")) == "12.5"


def test_file_without_timestamp_gives_none(tmp_path: Path, tracks: Tractogram) -> None:
    values, offsets = _values([2])
    write_tsf(values, offsets, tmp_path / "v.tsf")
    write_tck(tracks, tmp_path / "t.tck")
    assert read_track_timestamp(tmp_path / "v.tsf") is None
    assert read_track_timestamp(tmp_path / "t.tck") is None


def test_timestamp_of_another_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "x.tsf"
    path.write_bytes(b"mrtrix image\ntimestamp: 1\nEND\n")
    with pytest.raises(ValueError, match="not a TCK or TSF file"):
        read_track_timestamp(path)
