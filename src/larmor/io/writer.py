"""Writes headers and data to files, plain or gzip, for every format."""

import gzip
from io import BufferedWriter
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np

from larmor.types import Tractogram

# A file opened for writing, plain or gzip.
WriteFile = BufferedWriter | gzip.GzipFile


def open_file_write(path: Path) -> WriteFile:
    """Opens a file for writing, compressing it on the fly if it is gzip.

    A file is gzip if its name ends in ".gz".

    Args:
        path: The file to write.

    Returns:
        A binary file object, to be used as a context manager.
    """
    if path.suffix == ".gz":
        # Fast compression, and no time stamp, so the same data gives the same bytes.
        raw = path.open("wb", buffering=0)
        gz = gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0
        )
        # So that closing the GzipFile closes the file too.
        gz.myfileobj = raw
        return gz
    else:
        return path.open("wb")


def write_struct(f: WriteFile, header: np.ndarray) -> None:
    """Writes a binary header (a fixed-size struct) at the current position.

    The bytes are written as they are in memory.

    Args:
        f: A file from `open_file_write`.
        header: The header record, with a structured dtype.
    """
    f.write(header.tobytes())


def write_text_header(
    f: WriteFile, magic: str, header: dict[str, list[str]], end: str = "END"
) -> None:
    """Writes a text header with the magic line, `key: value` lines, then `end`.

    Each value is written on its own line, and a value with several lines as
    several `key: value` lines.

    Args:
        f: A file from `open_file_write`.
        magic: The first line, which identifies the format (such as "mrtrix image").
        header: The values of each key.
        end: The line that closes the header.

    Raises:
        ValueError: If a key is empty, has surrounding spaces or contains ':', '#'
            or a line break, or a value contains '#' (read back as a comment).
    """
    for key, vals in header.items():
        if not key or key != key.strip() or any(c in key for c in ":#\n"):
            raise ValueError(f"Header key {key!r} cannot be written.")
        for v in vals:
            if "#" in v:
                raise ValueError(
                    f"Header value {v!r} of {key!r} contains '#', which would be "
                    "read back as the start of a comment."
                )

    f.write(f"{magic}\n".encode())
    for key, vals in header.items():
        for v in vals:
            for line in v.split("\n"):
                f.write(f"{key}: {line}\n".encode())
    f.write(f"{end}\n".encode())


def write_payload(f: WriteFile, payload: np.ndarray, dtype: np.dtype) -> None:
    """Writes the items of a flat `payload`, in order, as `dtype`.

    Args:
        f: A file from `open_file_write`.
        payload: The data to write, 1-D.
        dtype: The type of each item on disk, with its byte order (such as '<f4').

    Raises:
        ValueError: If `payload` is not 1-D.
    """
    if payload.ndim != 1:
        raise ValueError(
            f"The payload must be flat (1-D), got shape {payload.shape}: flatten it "
            "in the file's order first."
        )
    f.write(np.asarray(payload, dtype=dtype, order="C"))


class BatchWriter:
    """Writes a tractogram file batch by batch, finished by `close`.

    Use it as a context manager. Leaving the block normally finishes the file;
    leaving it with an exception only closes it, and the file is then read back as
    incomplete.
    """

    path: Path
    num_streamlines: int
    _file: BufferedWriter

    def __enter__(self) -> Self:
        """Returns the writer itself."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Finishes the file, or only closes it if the block raised."""
        if exc_type is None:
            self.close()
        else:
            self._file.close()

    def write(self, tractogram: Tractogram) -> None:
        """Appends the streamlines of a tractogram to the file.

        Args:
            tractogram: The streamlines to append.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Writes the final streamline count and closes the file."""
        raise NotImplementedError


def open_batch_file(path: Path, fmt: str) -> BufferedWriter:
    """Opens an uncompressed file to be written in batches, whose header is rewritten.

    Args:
        path: The file to open.
        fmt: The format's name, used in the error message.

    Returns:
        The file, opened for writing in binary mode.

    Raises:
        ValueError: If the file is gzip-compressed, which cannot be rewritten in place.
    """
    if path.suffix == ".gz":
        raise ValueError(
            f"Cannot write {path} batch by batch: a compressed {fmt} cannot be "
            "updated in place. Write it uncompressed, then compress it."
        )
    return path.open("wb")
