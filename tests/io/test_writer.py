import gzip
import struct
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from larmor.io.reader import (
    open_file_read,
    read_payload,
    read_struct,
    read_text_header,
)
from larmor.io.writer import (
    open_file_write,
    write_payload,
    write_struct,
    write_text_header,
)

# A small binary header: an int, a sub-array, a float and a fixed string, no padding.
HEADER = [("size", "i4"), ("dim", "i2", (4,)), ("scale", "f4"), ("name", "S4")]
HEADER_FORMAT = "i4hf4s"
HEADER_VALUES = (348, 3, 64, 64, 30, 1.5, b"lar!")

# Exactly representable in float32, so a float64 -> '<f4' round trip is lossless.
VALUES = [-3.0, 0.5, 1.5, 2.0**-10, 2.0**20]

GZ = pytest.mark.parametrize("gz", [False, True], ids=["plain", "gzip"])
ORDER = pytest.mark.parametrize("order", ["<", ">"], ids=["little", "big"])


def _name(gz: bool) -> str:
    return "f.bin.gz" if gz else "f.bin"


def _record(order: Literal["<", ">"]) -> np.ndarray:
    dtype = np.dtype(HEADER).newbyteorder(order)
    return np.array([(348, (3, 64, 64, 30), 1.5, b"lar!")], dtype=dtype)


def _written(path: Path, payload: np.ndarray, dtype: str) -> bytes:
    with open_file_write(path) as f:
        write_payload(f, payload, np.dtype(dtype))
    return path.read_bytes()


def _text(path: Path, header: dict[str, list[str]], end: str = "END") -> bytes:
    with open_file_write(path) as f:
        write_text_header(f, "mrtrix image", header, end)
    return path.read_bytes()


@pytest.mark.parametrize(
    ("name", "gz"),
    [("a.nii.gz", True), ("a.nii", False), ("a.gz.nii", False), ("a.gzip", False)],
)
def test_gzip_is_written_only_for_names_ending_in_gz(
    tmp_path: Path, name: str, gz: bool
) -> None:
    data = b"\x00\x01payload bytes"
    path = tmp_path / name
    with open_file_write(path) as f:
        f.write(data)
    assert f.closed
    raw = path.read_bytes()
    assert (gzip.decompress(raw) if gz else raw) == data


def test_gzip_header_has_fastest_level_and_zero_mtime(tmp_path: Path) -> None:
    path = tmp_path / "a.nii.gz"
    with open_file_write(path) as f:
        f.write(b"\x00" * 1000)
    raw = path.read_bytes()
    # XFL (byte 8) is 4 only for compresslevel 1; MTIME is bytes 4..8.
    assert raw[8] == 4
    assert raw[4:8] == b"\x00\x00\x00\x00"


def test_gzip_bytes_do_not_depend_on_the_file_name(tmp_path: Path) -> None:
    written = []
    for name in ("a.nii.gz", "another_name.nii.gz"):
        with open_file_write(tmp_path / name) as f:
            f.write(b"same data")
        written.append((tmp_path / name).read_bytes())
    assert written[0] == written[1]
    assert gzip.decompress(written[0]) == b"same data"


@GZ
def test_existing_file_is_overwritten(tmp_path: Path, gz: bool) -> None:
    path = tmp_path / _name(gz)
    for data in (b"a much longer first content", b"short"):
        with open_file_write(path) as f:
            f.write(data)
    with open_file_read(path) as f:
        assert f.read() == b"short"


@ORDER
def test_struct_is_written_as_its_bytes(
    tmp_path: Path, order: Literal["<", ">"]
) -> None:
    path = tmp_path / "h.bin"
    with open_file_write(path) as f:
        write_struct(f, _record(order))
    assert path.read_bytes() == struct.pack(order + HEADER_FORMAT, *HEADER_VALUES)


def test_text_header_is_magic_then_one_line_per_value_then_end(
    tmp_path: Path,
) -> None:
    header = {
        "dim": ["2,3,4"],
        "command_history": ["mrconvert a b", "dwidenoise b c"],
        "vox": ["1.5,1.5,2"],
    }
    assert _text(tmp_path / "h.mif", header) == (
        b"mrtrix image\n"
        b"dim: 2,3,4\n"
        b"command_history: mrconvert a b\n"
        b"command_history: dwidenoise b c\n"
        b"vox: 1.5,1.5,2\n"
        b"END\n"
    )


def test_text_header_is_utf8(tmp_path: Path) -> None:
    text = _text(tmp_path / "h.mif", {"units": ["µm"]})
    assert text == "mrtrix image\nunits: µm\nEND\n".encode()


def test_text_header_ends_with_given_end_line(tmp_path: Path) -> None:
    text = _text(tmp_path / "h.tck", {"count": ["3"]}, end="DONE")
    assert text == b"mrtrix image\ncount: 3\nDONE\n"


def test_text_header_value_with_several_lines_gets_one_line_each(
    tmp_path: Path,
) -> None:
    text = _text(tmp_path / "h.mif", {"comments": ["first\nsecond"]})
    assert text == b"mrtrix image\ncomments: first\ncomments: second\nEND\n"


@pytest.mark.parametrize(
    "header",
    [
        {"": ["1"]},
        {" dim": ["1"]},
        {"a:b": ["1"]},
        {"a#b": ["1"]},
        {"a\nb": ["1"]},
        {"note": ["see #3"]},
    ],
    ids=["empty-key", "spaces", "colon", "hash", "newline", "hash-in-value"],
)
def test_text_header_that_would_not_read_back_raises(
    tmp_path: Path, header: dict[str, list[str]]
) -> None:
    path = tmp_path / "h.mif"
    with pytest.raises(ValueError):
        _text(path, header)
    assert path.read_bytes() == b""


@pytest.mark.parametrize(
    ("source", "dtype", "fmt"),
    [("float64", "<f4", "<4f"), (">i4", "<i4", "<4i"), ("<f4", ">f4", ">4f")],
    ids=["f8-to-f4", "big-to-little", "little-to-big"],
)
def test_payload_is_converted_to_dtype(
    tmp_path: Path, source: str, dtype: str, fmt: str
) -> None:
    values = [-3, 1, 2, 2**20]
    payload = np.array(values, dtype=source)
    assert _written(tmp_path / "p.bin", payload, dtype) == struct.pack(fmt, *values)


@pytest.mark.parametrize("source", ["<f4", "float64"])
def test_strided_payload_is_written_in_order(tmp_path: Path, source: str) -> None:
    payload = np.arange(12, dtype=source)[::3]
    assert _written(tmp_path / "p.bin", payload, "<f4") == struct.pack(
        "<4f", 0, 3, 6, 9
    )


@pytest.mark.parametrize(
    "layout",
    [lambda a: a, np.asfortranarray, lambda a: np.float32(1)],
    ids=["c-order-2d", "fortran-2d", "0d"],
)
def test_payload_that_is_not_flat_raises(
    tmp_path: Path, layout: Callable[[np.ndarray], np.ndarray]
) -> None:
    # Writing a 2-D array in some order would scramble a volume without an error:
    # the format module flattens it in the file's order first.
    payload = np.asarray(layout(np.arange(12, dtype="<f4").reshape(3, 4)))
    with pytest.raises(ValueError, match="1-D"):
        _written(tmp_path / "p.bin", payload, "<f4")


def test_payload_of_dtype_is_written_without_copy(tmp_path: Path) -> None:
    payload = np.ones(25_000_000, dtype="<f4")
    path = tmp_path / "p.bin"
    with open_file_write(path) as f:
        tracemalloc.start()
        try:
            write_payload(f, payload, np.dtype("<f4"))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    assert peak < 0.3 * payload.nbytes
    assert path.stat().st_size == payload.nbytes


@GZ
def test_struct_and_payload_read_back(tmp_path: Path, gz: bool) -> None:
    path = tmp_path / _name(gz)
    with open_file_write(path) as f:
        write_struct(f, _record("<"))
        write_payload(f, np.array(VALUES), np.dtype("<f4"))
    with open_file_read(path) as f:
        rec = read_struct(f, np.dtype(HEADER).newbyteorder("<"))
        arr = read_payload(f, rec.dtype.itemsize, np.dtype("<f4"))
    assert rec.tobytes() == _record("<").tobytes()
    assert arr.tolist() == VALUES


@GZ
def test_text_header_and_payload_read_back(tmp_path: Path, gz: bool) -> None:
    header = {
        "command_history": ["mrconvert a b", "dwidenoise b c"],
        "timestamp": ["2026-10-03T12:34:56"],
        "vox_units": ["µm"],
    }
    path = tmp_path / _name(gz)
    with open_file_write(path) as f:
        write_text_header(f, "mrtrix image", header)
        write_payload(f, np.array(VALUES), np.dtype("<f4"))
    with open_file_read(path) as f:
        assert f.readline() == b"mrtrix image\n"
        assert read_text_header(f) == header
        arr = read_payload(f, f.tell(), np.dtype("<f4"))
    assert arr.tolist() == VALUES
