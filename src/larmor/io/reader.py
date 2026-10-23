"""Reads headers and data from files, plain or gzip, for every format."""

import gzip
import os
import zlib
from collections.abc import Callable, Generator
from contextlib import contextmanager
from io import BufferedReader
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import DTypeLike, NDArray

from larmor._log import LarmorLogger

# The first two bytes of a gzip file.
GZIP_MAGIC = b"\x1f\x8b"

# A file opened for reading, plain or gzip.
ReadFile = BufferedReader | gzip.GzipFile

# The errors that reading a cut-off or damaged gzip file raises.
_GZIP_ERRORS = (EOFError, gzip.BadGzipFile, zlib.error)


#: A progress callback for reading a file, such as a progress bar's update.
#: It is called with (bytes_read, bytes_total) of the file on disk (compressed
#: bytes for gzip); raising ReadCancelledError in it stops the read.
Progress = Callable[[int, int], None]


class ReadCancelledError(Exception):
    """Raised by a progress callback to stop a read; the read then raises it too."""


# Bytes per read.
_CHUNK: int = 16 * 2**20


@contextmanager
def _gzip_errors_name_the_file(f: ReadFile) -> Generator[None]:
    """Re-raises the error of a truncated or corrupt gzip file `f` as a ValueError."""
    try:
        yield
    except _GZIP_ERRORS as e:
        raise ValueError(f"{f.name} is truncated or corrupt: {e}") from e


def open_file_read(path: Path) -> ReadFile:
    """Opens a file for reading, decompressing it on the fly if it is gzip.

    A file is gzip if it starts with the bytes 1f 8b, whatever its extension.

    Args:
        path: The file to read.

    Returns:
        A binary file object, to be used as a context manager.
    """
    f = path.open("rb")
    magic = f.read(2)

    if magic == GZIP_MAGIC:
        f.close()
        return gzip.open(path, "rb")

    f.seek(0)
    return f


def read_struct(f: ReadFile, dtype: DTypeLike) -> np.ndarray:
    """Reads one binary header (a fixed-size struct) from the current position.

    Args:
        f: A file from `open_file_read`.
        dtype: A structured dtype describing the header, field by field.

    Returns:
        An array of one record, shape (1,).

    Raises:
        ValueError: If the file ends before `dtype.itemsize` bytes were read, or is a
            truncated or corrupt gzip file.
    """
    cast_dtype = np.dtype(dtype)
    num_bytes = cast_dtype.itemsize
    byte_arr = bytearray(num_bytes)
    with _gzip_errors_name_the_file(f):
        bytes_read = f.readinto(byte_arr)

    if bytes_read != num_bytes:
        raise ValueError(
            f"{f.name} is too short: expected {num_bytes} bytes, got {bytes_read}."
        )

    return np.frombuffer(byte_arr, dtype=cast_dtype)


def read_text_header(
    f: ReadFile, end: str = "END", end_required: bool = True
) -> dict[str, list[str]]:
    """Reads `key: value` lines, decoded as UTF-8, up to the line `end`.

    Args:
        f: A file from `open_file_read`.
        end: The line that closes the header.
        end_required: If False, the end of the file also closes the header (MRtrix
            writes `.mih` files without `END`).

    Returns:
        The values of each key, in file order. A key that appears more than once
        (such as command_history) keeps all its values. Keys and values are split
        at the first colon and stripped of surrounding spaces.

    Raises:
        ValueError: If the file ends before the `end` line (when required), a line
            is not UTF-8, or
            the file is a truncated or corrupt gzip file.
    """
    d: dict[str, list[str]] = {}

    while True:
        with _gzip_errors_name_the_file(f):
            raw = f.readline()
        if not raw:
            if not end_required:
                break
            raise ValueError(f"{f.name} ends before the {end!r} line of its header.")

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"Header line {raw!r} in {f.name} is not UTF-8.") from e

        line = text.partition("#")[0].strip()
        if line == end:
            break
        if not line:
            continue

        key, sep, value = line.partition(":")
        if not sep or not key.strip():
            LarmorLogger.info("Ignoring malformed header line %r in %s.", line, f.name)
            continue

        d.setdefault(key.strip(), []).append(value.strip())
    return d


def read_payload(
    f: ReadFile,
    offset: int,
    dtype: DTypeLike,
    count: int | None = None,
    on_progress: Progress | None = None,
) -> np.ndarray:
    """Reads the data (payload) as a flat, read-only array in native byte order.

    Args:
        f: A file from `open_file_read`.
        offset: Where the payload starts, in bytes from the start of the
            (uncompressed) file.
        dtype: The type of each item on disk, with its byte order (such as '>f4').
        count: The number of items to read, or None to read to the end of the file.
            With a count, a gzip file is still read to its end, so that its
            checksum is checked.
        on_progress: Called after each chunk is read, see Progress.

    Returns:
        A flat array of `count` items (or as many as the file holds).

    Raises:
        ValueError: If the file holds fewer than `count` items, `offset` is past
            the end of the file, or the file is a truncated or corrupt gzip file.
    """
    cast_dtype = np.dtype(dtype)
    num_bytes = cast_dtype.itemsize

    with _gzip_errors_name_the_file(f):
        f.seek(offset)
        if count is not None:
            num_bytes *= count
            byte_arr = np.empty(num_bytes, dtype=np.uint8)
            bytes_read = _readinto_chunks(f, byte_arr, on_progress)
            if isinstance(f, gzip.GzipFile):
                # Read the rest, so that gzip checks its checksum and a damaged file
                # raises instead of returning wrong values.
                while f.read(_CHUNK):
                    _report(f, on_progress)
        else:
            # A gzip file stops at its end when asked to seek past it; a plain file
            # does not, so its size comes from the file system.
            gz = isinstance(f, gzip.GzipFile)
            end = f.tell() if gz else os.fstat(f.fileno()).st_size
            if offset > end:
                raise ValueError(
                    f"Offset {offset} is past the end of {f.name} ({end} bytes)."
                )
            buf = _read_to_end(f, on_progress)
            byte_arr = np.frombuffer(buf, dtype=np.uint8)
            bytes_read = len(buf)

    if count is not None and bytes_read != num_bytes:
        raise ValueError(
            f"{f.name} is too short: expected {num_bytes} bytes, got {bytes_read}."
        )

    n_items, n_extra = divmod(bytes_read, cast_dtype.itemsize)
    if n_extra:
        LarmorLogger.warning(
            "%s ends with an incomplete item (%d of %d bytes); it was dropped.",
            f.name,
            n_extra,
            cast_dtype.itemsize,
        )

    arr = np.frombuffer(byte_arr, dtype=cast_dtype, count=n_items)
    if not cast_dtype.isnative:
        arr.byteswap(inplace=True)
        arr = arr.view(cast_dtype.newbyteorder("="))

    arr.flags.writeable = False
    return arr


def _report(f: ReadFile, on_progress: Progress | None) -> None:
    """Calls `on_progress` with the position in and the size of the file `f` on disk."""
    if on_progress is None:
        return
    if isinstance(f, gzip.GzipFile):
        raw = cast(BufferedReader, f.fileobj)
    else:
        raw = f
    on_progress(raw.tell(), os.fstat(raw.fileno()).st_size)


def _readinto_chunks(
    f: ReadFile, buf: bytearray | NDArray[np.uint8], on_progress: Progress | None
) -> int:
    """Fills `buf` from the current position of `f`, reporting after each chunk.

    Returns:
        The number of bytes read, which can be less than len(buf) if the file ends
        first.
    """
    done = 0
    with memoryview(buf) as view:
        while done < len(view):
            n = f.readinto(view[done : done + _CHUNK])
            if n == 0:
                break
            done += n
            _report(f, on_progress)
    return done


def _read_to_end(f: ReadFile, on_progress: Progress | None) -> bytearray:
    """Reads `f` from the current position to the end, reporting after each chunk.

    Returns:
        The bytes read.
    """
    if isinstance(f, gzip.GzipFile):
        buf = bytearray()
        while chunk := f.read(_CHUNK):
            buf += chunk
            _report(f, on_progress)
        return buf
    buf = bytearray(max(os.fstat(f.fileno()).st_size - f.tell(), 0))
    n = _readinto_chunks(f, buf, on_progress)
    del buf[n:]
    return buf


def check_not_gzip(path: Path) -> None:
    """Checks that a file to be read in pieces is not gzip-compressed.

    Args:
        path: The file to check.

    Raises:
        ValueError: If the file is gzip-compressed.
    """
    with path.open("rb") as f:
        magic = f.read(2)
    if magic == GZIP_MAGIC:
        raise ValueError(
            f"{path} is gzip-compressed; only uncompressed files can be read in "
            "batches. Decompress it first, or load it whole."
        )


def read_chunk(f: BufferedReader, carry: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Reads the next chunk of a file, after the bytes `carry` left from the last one.

    Args:
        f: An uncompressed file, at the position to read from.
        carry: Bytes of the previous chunk not used yet, such as the start of an
            item that the chunk cut off.

    Returns:
        `carry` followed by up to _CHUNK new bytes, in one new array. It has no
        more bytes than `carry` once the file has ended.
    """
    buf = np.empty(len(carry) + _CHUNK, dtype=np.uint8)
    buf[: len(carry)] = carry
    n = _readinto_chunks(f, buf[len(carry) :], None)
    return buf[: len(carry) + n]


def read_all(f: ReadFile, on_progress: Progress | None = None) -> bytearray:
    """Reads a file from the current position to the end.

    Args:
        f: A file from `open_file_read`.
        on_progress: Called after each chunk is read, see Progress.

    Returns:
        The bytes read.

    Raises:
        ValueError: If the file is a truncated or corrupt gzip file.
        ReadCancelledError: If on_progress raises it.
    """
    with _gzip_errors_name_the_file(f):
        return _read_to_end(f, on_progress)
