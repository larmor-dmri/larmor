"""Reads and writes TCK files, the tractogram format of MRtrix."""

import os
from collections.abc import Iterator
from io import SEEK_END
from pathlib import Path

import numpy as np

from larmor._log import LarmorLogger
from larmor.io.formats._track_file import (
    check_count,
    check_timestamp,
    offsets_from_delimiters,
    payload_dtype,
    payload_start,
    track_file_header,
)
from larmor.io.reader import (
    Progress,
    ReadFile,
    check_not_gzip,
    open_file_read,
    read_chunk,
    read_payload,
    read_text_header,
)
from larmor.io.writer import (
    BatchWriter,
    open_batch_file,
    open_file_write,
    write_payload,
)
from larmor.types import Tractogram

# The first line of a TCK file.
TCK_MAGIC = "mrtrix tracks"


def _split_tck_payload(
    path: Path, payload: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the float32 points and the streamline offsets of a TCK file's data.

    Raises:
        ValueError: If a point has some, but not all, of its coordinates NaN.
    """
    leftover = len(payload) % 3
    if leftover > 0:
        LarmorLogger.warning("%s ends with an incomplete point; it was dropped.", path)
    rows = payload[: len(payload) - leftover].reshape(-1, 3)
    partly_nan = f"{path} has a point with some coordinates NaN."

    # Only a row whose x is not finite can be a marker: Inf ends the data, NaN a
    # streamline.
    marked = np.flatnonzero(~np.isfinite(rows[:, 0]))
    is_end = np.isinf(rows[marked, 0])
    if is_end.any():
        first_end = int(np.argmax(is_end))
        rows, marked = rows[: marked[first_end]], marked[:first_end]
    delims = marked
    if not np.isnan(rows[delims, 1:]).all():
        raise ValueError(partly_nan)

    end = delims[-1] + 1 if len(delims) > 0 else 0
    is_point = np.ones(end, dtype=bool)
    is_point[delims] = False
    # Each row viewed as one item, so that whole rows are selected at once.
    whole_rows = np.ascontiguousarray(rows[:end])
    row_items = whole_rows.view(np.dtype((np.void, 3 * rows.itemsize)))[:, 0]
    points = row_items[is_point].view(rows.dtype).reshape(-1, 3)
    # The remaining x are finite, so any NaN left is in a point's y or z.
    if np.isnan(points).any() or np.isnan(rows[end:]).any():
        raise ValueError(partly_nan)

    if end < len(rows):
        LarmorLogger.warning(
            "%s: the last streamline has no closing NaN (interrupted tracking?); "
            "it was dropped.",
            path,
        )
    return points.astype(np.float32, copy=False), offsets_from_delimiters(delims)


def _read_tck_header(
    path: Path, f: ReadFile
) -> tuple[dict[str, list[str]], np.dtype, int]:
    """Reads and checks a TCK header, and returns it, the data's dtype and offset."""
    magic = f.readline().decode("utf-8", "replace").strip()
    if magic != TCK_MAGIC:
        raise ValueError(f"{path} is not a TCK file: no {TCK_MAGIC!r} line.")
    header = read_text_header(f)
    for key in ["datatype", "file"]:
        if key not in header:
            raise ValueError(f"{path} has no {key!r} in its header.")
    return (
        header,
        payload_dtype(path, header, "TCK"),
        payload_start(path, header, "TCK"),
    )


def read_tck(path: str | Path, *, on_progress: Progress | None = None) -> Tractogram:
    """Reads a TCK file, an MRtrix tractogram, into a `Tractogram`.

    The data is read to the end, whatever the header's `count` says; a last
    streamline cut short by an interrupted write is dropped with a warning. The
    points are float32 and the tractogram has no reference grid.

    Args:
        path: The `.tck` file.
        on_progress: Called as the file is read, see `load_volume`.

    Returns:
        The streamlines of the file.

    Raises:
        ValueError: If the file is not a TCK file Larmor can read.
        ReadCancelledError: If on_progress raises it.
    """
    path = Path(path)
    with open_file_read(path) as f:
        header, dtype, offset = _read_tck_header(path, f)
        payload = read_payload(f, offset, dtype, on_progress=on_progress)

    points, offsets = _split_tck_payload(path, payload)
    check_count(path, header, len(offsets) - 1)
    return Tractogram(points, offsets)


def read_tck_count(path: str | Path) -> int | None:
    """Reads the streamline count in a TCK header, without the streamlines.

    Args:
        path: The `.tck` file, also gzipped.

    Returns:
        The header's `count`, or None if it has none or it is not a number.

    Raises:
        ValueError: If the file is not a TCK file Larmor can read.
    """
    path = Path(path)
    with open_file_read(path) as f:
        header, _, _ = _read_tck_header(path, f)
    count = header.get("count", [""])[0]
    return int(count) if count.isdigit() else None


def _check_finite(tractogram: Tractogram, path: Path) -> None:
    """Raises ValueError if a point is NaN or infinite, which TCK uses as markers."""
    if not np.isfinite(tractogram.points).all():
        raise ValueError(
            f"Cannot write {path}: some points are NaN or infinite, which TCK uses to "
            "end streamlines and the file."
        )


def _tck_header(count: int, timestamp: str | None) -> bytes:
    """Builds the header of a TCK file, padded up to the start of the data."""
    lines = [] if timestamp is None else [f"timestamp: {timestamp}"]
    lines.append("datatype: Float32LE")
    return track_file_header(TCK_MAGIC, lines, count)


def _tck_rows(tractogram: Tractogram) -> np.ndarray:
    """Returns the points as float32 rows (N, 3), a NaN row after each streamline."""
    n_points, num_streamlines = tractogram.num_points, tractogram.num_streamlines
    lengths = np.diff(tractogram.offsets)
    rows = np.full((n_points + num_streamlines, 3), np.nan, dtype=np.float32)
    rows[np.arange(n_points) + np.repeat(np.arange(num_streamlines), lengths)] = (
        tractogram.points
    )
    return rows


# The row that ends the data of a TCK file.
_END_ROW = np.full(3, np.inf, dtype="<f4")


def write_tck(
    tractogram: Tractogram, path: str | Path, *, timestamp: str | None = None
) -> None:
    """Writes a `Tractogram` as a TCK file of float32 points.

    Only the points are written, since TCK holds no `point_data`, `streamline_data`
    or reference grid.

    Args:
        tractogram: The tractogram to write.
        path: The `.tck` file; an existing file is overwritten.
        timestamp: The header's `timestamp`, which MRtrix uses to pair the file
            with its TSF files; pass the same value to `write_tsf`. Defaults to
            None, no timestamp.

    Raises:
        ValueError: If a point is NaN or infinite, which TCK reads as the end of a
            streamline or of the file, or `timestamp` has a '#' or a line break.
    """
    path = Path(path)
    _check_finite(tractogram, path)
    check_timestamp(path, timestamp)
    if tractogram.point_data or tractogram.streamline_data:
        LarmorLogger.warning(
            "%s: TCK cannot hold point_data or streamline_data; they were not written.",
            path,
        )
    with open_file_write(path) as f:
        f.write(_tck_header(tractogram.num_streamlines, timestamp))
        write_payload(f, _tck_rows(tractogram).ravel(), np.dtype("<f4"))
        f.write(_END_ROW.tobytes())


class TckWriter(BatchWriter):
    """Writes a TCK file batch by batch, keeping only the current batch in memory.

    The header's count stays 0 until `close`, so a file that was never closed reads
    back with a warning that it is incomplete. The file is the same as the one
    `write_tck` writes for all the streamlines at once.
    """

    def __init__(self, path: str | Path, *, timestamp: str | None = None) -> None:
        """Creates the file, with its header and an empty body.

        Args:
            path: The `.tck` file; an existing file is overwritten. Not `.tck.gz`.
            timestamp: The header's `timestamp`, see `write_tck`. Defaults to
                None, no timestamp.

        Raises:
            ValueError: If the file is gzip-compressed, or `timestamp` has a '#'
                or a line break.
        """
        self.path = Path(path)
        check_timestamp(self.path, timestamp)
        self.timestamp = timestamp
        self.num_streamlines = 0
        self._dropped_data = False
        self._file = open_batch_file(self.path, "TCK")
        self._file.write(_tck_header(0, timestamp))
        self._file.write(_END_ROW.tobytes())

    def write(self, tractogram: Tractogram) -> None:
        """Appends the streamlines of a batch, in order.

        `point_data` and `streamline_data` are dropped, since TCK holds only points.

        Raises:
            ValueError: If a point is NaN or infinite.
        """
        _check_finite(tractogram, self.path)
        if (tractogram.point_data or tractogram.streamline_data) and not (
            self._dropped_data
        ):
            LarmorLogger.info(
                "%s: TCK cannot hold point_data or streamline_data; they are not "
                "written.",
                self.path,
            )
            self._dropped_data = True
        # Overwrites the end row, which is written again after the batch.
        self._file.seek(-_END_ROW.nbytes, SEEK_END)
        write_payload(self._file, _tck_rows(tractogram).ravel(), np.dtype("<f4"))
        self._file.write(_END_ROW.tobytes())
        self.num_streamlines += tractogram.num_streamlines

    def close(self) -> None:
        """Writes the streamline count into the header and closes the file."""
        if self._file.closed:
            return
        self._file.seek(0)
        self._file.write(_tck_header(self.num_streamlines, self.timestamp))
        self._file.close()


def read_tck_chunks(path: str | Path) -> Iterator[Tractogram]:
    """Reads an uncompressed TCK file piece by piece, whole streamlines at a time.

    Each piece is a tractogram with the streamlines one chunk of the file
    completes. Joined in order, the pieces are the tractogram `read_tck` returns.
    Incomplete data after the end marker is not reported. At least one piece is
    yielded, empty for a file without streamlines.

    Args:
        path: The `.tck` file, not gzip-compressed.

    Returns:
        An iterator over the pieces, each with float32 points and no reference.

    Raises:
        ValueError: If the file is gzip-compressed (at the call), or is not a TCK
            file Larmor can read (while iterating).
    """
    path = Path(path)
    check_not_gzip(path)
    return _tck_chunks(path)


def _tck_chunks(path: Path) -> Iterator[Tractogram]:
    """Yields the streamlines that each chunk of the TCK file at `path` completes."""
    with path.open("rb") as f:
        header, dtype, offset = _read_tck_header(path, f)
        size = os.fstat(f.fileno()).st_size
        if offset > size:
            raise ValueError(
                f"Offset {offset} is past the end of {path} ({size} bytes)."
            )
        f.seek(offset)

        row_bytes = 3 * dtype.itemsize
        found = 0
        carry = np.empty(0, dtype=np.uint8)
        while True:
            data = read_chunk(f, carry)
            if len(data) == len(carry):
                break
            rows = data[: len(data) - len(data) % row_bytes].view(dtype).reshape(-1, 3)
            inf_rows = np.flatnonzero(np.isinf(rows[:, 0]))
            if len(inf_rows) > 0:
                rows = rows[: inf_rows[0]]
            delims = np.flatnonzero(np.isnan(rows[:, 0]))
            end = delims[-1] + 1 if len(delims) > 0 else 0
            # The rows up to the last delimiter are whole streamlines.
            points, offsets = _split_tck_payload(path, rows[:end].ravel())
            if len(offsets) > 1:
                found += len(offsets) - 1
                yield Tractogram(points, offsets)
            if len(inf_rows) > 0:
                carry = rows[end:].ravel().view(np.uint8)
                break
            carry = data[end * row_bytes :].copy()

    # What is left is an unfinished streamline, perhaps with an incomplete point.
    extra = len(carry) % dtype.itemsize
    if extra:
        LarmorLogger.warning(
            "%s ends with an incomplete item (%d of %d bytes); it was dropped.",
            path,
            extra,
            dtype.itemsize,
        )
    _split_tck_payload(path, carry[: len(carry) - extra].view(dtype))
    check_count(path, header, found)
    if found == 0:
        yield Tractogram(np.empty((0, 3), dtype=np.float32), np.zeros(1, np.int64))
