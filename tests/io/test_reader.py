import gzip
import logging
import re
import struct
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from larmor.io.reader import (
    ReadFile,
    open_file_read,
    read_payload,
    read_struct,
    read_text_header,
)

# A small binary header: an int, a sub-array, a float and a fixed string, no padding.
HEADER = [("size", "i4"), ("dim", "i2", (4,)), ("scale", "f4"), ("name", "S4")]
HEADER_FORMAT = "i4hf4s"
HEADER_VALUES = (348, 3, 64, 64, 30, 1.5, b"lar!")

GZ = pytest.mark.parametrize("gz", [False, True], ids=["plain", "gzip"])
ORDER = pytest.mark.parametrize("order", ["<", ">"], ids=["little", "big"])


def _write(path: Path, data: bytes, gz: bool) -> Path:
    path.write_bytes(gzip.compress(data) if gz else data)
    return path


def _names_sizes(
    err: pytest.ExceptionInfo[ValueError], path: Path, *sizes: int
) -> bool:
    msg = str(err.value)
    return path.name in msg and all(re.search(rf"\b{n}\b", msg) for n in sizes)


def _header(path: Path, text: str, end: str = "END") -> dict[str, list[str]]:
    data = ("mrtrix image\n" + text).encode()
    with open_file_read(_write(path, data, gz=False)) as f:
        f.readline()
        return read_text_header(f, end)


@pytest.mark.parametrize(
    ("name", "gz"), [("data.bin", True), ("data.gz", False)], ids=["gzip", "plain"]
)
def test_gzip_is_detected_from_content_not_name(
    tmp_path: Path, name: str, gz: bool
) -> None:
    data = b"\x00\x01payload bytes"
    with open_file_read(_write(tmp_path / name, data, gz)) as f:
        assert f.read() == data
    assert f.closed


@ORDER
@GZ
def test_struct_is_one_record_in_file_byte_order(
    tmp_path: Path, order: Literal["<", ">"], gz: bool
) -> None:
    dtype = np.dtype(HEADER).newbyteorder(order)
    raw = struct.pack(order + HEADER_FORMAT, *HEADER_VALUES)
    with open_file_read(_write(tmp_path / "h.bin", raw + b"rest", gz)) as f:
        rec = read_struct(f, dtype)
        assert f.read() == b"rest"
    assert rec.shape == (1,)
    assert rec.dtype == dtype
    assert rec.tobytes() == raw
    assert rec["size"][0] == 348
    assert rec["dim"][0].tolist() == [3, 64, 64, 30]
    assert rec["scale"][0] == 1.5
    assert rec["name"][0] == b"lar!"


@GZ
def test_short_struct_raises_naming_file_and_sizes(tmp_path: Path, gz: bool) -> None:
    path = _write(tmp_path / "h.bin", b"\x00" * 10, gz)
    with open_file_read(path) as f, pytest.raises(ValueError) as err:
        read_struct(f, np.dtype(HEADER))
    assert _names_sizes(err, path, np.dtype(HEADER).itemsize, 10), err.value


def test_text_header_keys_and_values_are_stripped(tmp_path: Path) -> None:
    header = _header(tmp_path / "h.mif", "dim: 2,3,4\n  vox :  1.5,1.5,2  \nEND\n")
    assert header == {"dim": ["2,3,4"], "vox": ["1.5,1.5,2"]}


def test_text_header_splits_at_first_colon_only(tmp_path: Path) -> None:
    header = _header(tmp_path / "h.mif", "timestamp: 2026-10-03T12:34:56\nEND\n")
    assert header == {"timestamp": ["2026-10-03T12:34:56"]}


def test_text_header_repeated_key_keeps_all_values_in_order(tmp_path: Path) -> None:
    text = "command_history: a\ndim: 2\ncommand_history: b\ncommand_history: c\nEND\n"
    header = _header(tmp_path / "h.mif", text)
    assert header == {"command_history": ["a", "b", "c"], "dim": ["2"]}


def test_text_header_is_utf8(tmp_path: Path) -> None:
    header = _header(tmp_path / "h.mif", "units: µm\nsubject: Désirée\nEND\n")
    assert header == {"units": ["µm"], "subject": ["Désirée"]}


@GZ
def test_text_header_leaves_file_just_after_end_line(tmp_path: Path, gz: bool) -> None:
    data = b"mrtrix image\ndim: 2\nEND\n\x00\x01END\n"
    with open_file_read(_write(tmp_path / "h.mif", data, gz)) as f:
        f.readline()
        read_text_header(f)
        assert f.read() == b"\x00\x01END\n"


def test_text_header_stops_at_given_end_line(tmp_path: Path) -> None:
    header = _header(tmp_path / "h.tck", "count: 3\nDONE\nEND\n", end="DONE")
    assert header == {"count": ["3"]}


@pytest.mark.parametrize(
    "read",
    [read_text_header, lambda f: read_text_header(f, end_required=True)],
    ids=["default", "true"],
)
def test_text_header_without_end_raises(
    tmp_path: Path, read: Callable[[ReadFile], object]
) -> None:
    path = _write(tmp_path / "h.mif", b"mrtrix image\ndim: 2\n", gz=False)
    with open_file_read(path) as f:
        f.readline()
        with pytest.raises(ValueError, match="h.mif ends before the 'END' line"):
            read(f)


@GZ
def test_text_header_not_required_to_end_stops_at_end_of_file(
    tmp_path: Path, gz: bool
) -> None:
    path = _write(tmp_path / "h.mih", b"mrtrix image\ndim: 2\nfile: h.dat", gz)
    with open_file_read(path) as f:
        f.readline()
        assert read_text_header(f, end_required=False) == {
            "dim": ["2"],
            "file": ["h.dat"],
        }


def test_text_header_not_required_to_end_still_stops_at_end_line(
    tmp_path: Path,
) -> None:
    data = b"mrtrix image\ndim: 2\nEND\nvox: 1\n"
    with open_file_read(_write(tmp_path / "h.mih", data, gz=False)) as f:
        f.readline()
        assert read_text_header(f, end_required=False) == {"dim": ["2"]}
        assert f.read() == b"vox: 1\n"


def test_text_header_lines_are_read_as_mrtrix_reads_them(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Comments, blank lines, lines without a colon and empty keys are skipped, and
    # END may have spaces or a comment around it. Only the two malformed lines are
    # reported.
    text = (
        "# a comment line\n"
        "\n"
        "dim: 2  # trailing comment\n"
        "not a header line\n"
        ": no key\n"
        "   \n"
        "vox: 1\n"
        "  END  # done\n"
        "after: end\n"
    )
    with caplog.at_level(logging.INFO, logger="larmor"):
        assert _header(tmp_path / "h.mif", text) == {"dim": ["2"], "vox": ["1"]}
    assert len(caplog.records) == 2


def test_text_header_that_is_not_utf8_raises_naming_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "h.mif", b"mrtrix image\nname: \xff\xfe\nEND\n", gz=False)
    with open_file_read(path) as f, pytest.raises(ValueError, match="h.mif"):
        f.readline()
        read_text_header(f)


@ORDER
@GZ
def test_payload_is_flat_native_with_file_values(
    tmp_path: Path, order: Literal["<", ">"], gz: bool
) -> None:
    values = [-3.0, 0.5, 1.5, 2.0**-10, 2.0**20]
    data = b"HEAD1234" + struct.pack(f"{order}5f", *values) + b"tail"
    with open_file_read(_write(tmp_path / "p.bin", data, gz)) as f:
        arr = read_payload(f, 8, np.dtype(f"{order}f4"), count=5)
    assert arr.dtype.isnative
    assert arr.dtype == np.dtype("float32")
    assert arr.shape == (5,)
    assert arr.tolist() == values


@ORDER
def test_payload_is_read_only(tmp_path: Path, order: Literal["<", ">"]) -> None:
    data = struct.pack(f"{order}3i", 1, 2, 3)
    with open_file_read(_write(tmp_path / "p.bin", data, gz=False)) as f:
        arr = read_payload(f, 0, np.dtype(f"{order}i4"), count=3)
    with pytest.raises(ValueError, match="read-only"):
        arr[0] = 7


def test_single_byte_payload(tmp_path: Path) -> None:
    with open_file_read(_write(tmp_path / "p.bin", bytes(range(256)), gz=False)) as f:
        arr = read_payload(f, 6, np.dtype("u1"))
    assert arr.dtype == np.dtype("u1")
    assert arr.tolist() == list(range(6, 256))


@GZ
def test_short_payload_raises_naming_file_and_sizes(tmp_path: Path, gz: bool) -> None:
    path = _write(tmp_path / "p.bin", b"HEAD" + struct.pack("<3f", 1, 2, 3), gz)
    with open_file_read(path) as f, pytest.raises(ValueError) as err:
        read_payload(f, 4, np.dtype("<f4"), count=5)
    assert _names_sizes(err, path, 20, 12), err.value


@GZ
def test_payload_without_count_reads_to_end_silently(
    tmp_path: Path, gz: bool, caplog: pytest.LogCaptureFixture
) -> None:
    data = b"HEAD" + struct.pack(">4h", -1, 2, -3, 4)
    with (
        caplog.at_level(logging.WARNING, logger="larmor"),
        open_file_read(_write(tmp_path / "p.tck", data, gz)) as f,
    ):
        arr = read_payload(f, 4, np.dtype(">i2"))
    assert arr.tolist() == [-1, 2, -3, 4]
    assert caplog.records == []


@GZ
def test_incomplete_last_item_is_dropped_with_warning(
    tmp_path: Path, gz: bool, caplog: pytest.LogCaptureFixture
) -> None:
    data = b"HEAD" + struct.pack("<3f", 1, 2, 3) + b"\x00\x00"
    with (
        caplog.at_level(logging.WARNING, logger="larmor"),
        open_file_read(_write(tmp_path / "p.tck", data, gz)) as f,
    ):
        arr = read_payload(f, 4, np.dtype("<f4"))
    assert arr.tolist() == [1.0, 2.0, 3.0]
    assert [(r.name, r.levelno) for r in caplog.records] == [
        ("larmor", logging.WARNING)
    ]


@pytest.mark.parametrize("count", [None, 1])
@GZ
def test_offset_past_end_raises(tmp_path: Path, gz: bool, count: int | None) -> None:
    path = _write(tmp_path / "p.bin", b"\x00" * 8, gz)
    with open_file_read(path) as f, pytest.raises(ValueError) as err:
        read_payload(f, 9, np.dtype("<f4"), count)
    # Reading to the end names the offset and the file size; with a count, it is
    # a short read of 4 bytes that finds none.
    sizes = (9, 8) if count is None else (4, 0)
    assert _names_sizes(err, path, *sizes)


@GZ
def test_offset_at_end_without_count_gives_empty_array(
    tmp_path: Path, gz: bool
) -> None:
    with open_file_read(_write(tmp_path / "p.bin", b"\x00" * 8, gz)) as f:
        arr = read_payload(f, 8, np.dtype(">f4"))
    assert arr.shape == (0,)
    assert arr.dtype.isnative


def test_large_payload_keeps_one_copy_in_memory(tmp_path: Path) -> None:
    # Big-endian so the byte swap runs: it must happen in the buffer that was read.
    n = 25_000_000
    path = tmp_path / "p.bin"
    np.arange(n, dtype=">u4").tofile(path)
    with open_file_read(path) as f:
        tracemalloc.start()
        try:
            arr = read_payload(f, 0, np.dtype(">u4"), count=n)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    assert peak < 1.3 * 4 * n
    assert arr.dtype.isnative
    assert (arr[0], arr[n // 2], arr[-1]) == (0, n // 2, n - 1)


# Random values, so that gzip cannot squeeze them into a few bytes.
NOISE = np.random.default_rng(0).random(1000).astype("<f4").tobytes()


def _read_struct(f: ReadFile) -> object:
    return read_struct(f, np.dtype(("<f4", (1000,))))


def _read_text(f: ReadFile) -> object:
    f.readline()
    return read_text_header(f)


@pytest.mark.parametrize(
    "read",
    [
        _read_struct,
        _read_text,
        lambda f: read_payload(f, 0, np.dtype("<f4"), 1000),
        lambda f: read_payload(f, 0, np.dtype("<f4")),
    ],
    ids=["struct", "text-header", "payload-count", "payload-to-end"],
)
def test_truncated_gzip_raises_naming_file(
    tmp_path: Path, read: Callable[[ReadFile], object]
) -> None:
    # A download that stopped half-way: gzip's own EOFError would not name the file.
    data = NOISE
    if read is _read_text:
        data = ("mrtrix image\n" + "key: value\n" * 2000 + "END\n").encode()
    full = gzip.compress(data)
    path = tmp_path / "cut.bin.gz"
    path.write_bytes(full[: len(full) // 2])
    with open_file_read(path) as f, pytest.raises(ValueError, match="cut.bin.gz"):
        read(f)


@pytest.mark.parametrize("count", [1000, None])
def test_gzip_with_wrong_checksum_raises(tmp_path: Path, count: int | None) -> None:
    # The data decompresses fine, but the CRC in the last 8 bytes does not match it;
    # gzip only notices if the stream is read to its end.
    damaged = bytearray(gzip.compress(NOISE))
    damaged[-8] ^= 0xFF
    path = tmp_path / "crc.bin.gz"
    path.write_bytes(damaged)
    with open_file_read(path) as f, pytest.raises(ValueError, match="crc.bin.gz"):
        read_payload(f, 0, np.dtype("<f4"), count)
