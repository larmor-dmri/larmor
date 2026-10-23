"""Reads and writes TSF files, MRtrix track scalar files with one value per point."""

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
from larmor.io.formats.tck import TCK_MAGIC
from larmor.io.reader import (
    Progress,
    open_file_read,
    read_payload,
    read_text_header,
)
from larmor.io.writer import open_file_write, write_payload
from larmor.types import Tractogram

# The first line of a TSF file.
TSF_MAGIC = "mrtrix track scalars"


def _split_tsf_payload(
    path: Path, payload: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the values (without delimiters) and offsets of a TSF's data."""
    inf_items = np.flatnonzero(np.isinf(payload))
    if len(inf_items) > 0:
        payload = payload[: inf_items[0]]

    is_delim = np.isnan(payload)
    delims = np.flatnonzero(is_delim)
    end = delims[-1] + 1 if len(delims) > 0 else 0
    if end < len(payload):
        LarmorLogger.warning(
            "%s: the last streamline has no closing NaN (interrupted writing?); it "
            "was dropped.",
            path,
        )
        payload, is_delim = payload[:end], is_delim[:end]

    return payload[~is_delim], offsets_from_delimiters(delims)


def _check_against(path: Path, offsets: np.ndarray, tractogram: Tractogram) -> None:
    """Raises ValueError if the file's streamline lengths differ from the tractogram's.

    Raises:
        ValueError: If the numbers of streamlines differ, or a streamline has a
            different number of values than points; the message names the first.
    """
    found = len(offsets) - 1
    if found != tractogram.num_streamlines:
        raise ValueError(
            f"{path} has values for {found} streamlines, but the tractogram has "
            f"{tractogram.num_streamlines}."
        )
    lengths = np.diff(offsets)
    expected = np.diff(tractogram.offsets)
    differ = np.flatnonzero(lengths != expected)
    if len(differ) > 0:
        first = differ[0]
        raise ValueError(
            f"{path} does not match the tractogram: streamline {first} has "
            f"{lengths[first]} values but {expected[first]} points ({len(differ)} "
            "streamlines differ)."
        )


def read_tsf(
    path: str | Path,
    *,
    tractogram: Tractogram | None = None,
    on_progress: Progress | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reads a TSF file, an MRtrix track scalar file with one value per point.

    A last streamline cut short by an interrupted write is dropped with a warning.

    Args:
        path: The `.tsf` file.
        tractogram: The tractogram the values belong to. When given, the file must
            have one value per point of each of its streamlines. Defaults to None.
        on_progress: Called as the file is read, see `load_volume`.

    Returns:
        The (P,) values, float32 or float64 as stored, and the (S + 1,) int64
        offsets; streamline s has values[offsets[s]:offsets[s + 1]].

    Raises:
        ValueError: If the file is not a TSF file Larmor can read, or does not match
            `tractogram`.
        ReadCancelledError: If on_progress raises it.

    References:
        Follows MRtrix3's track scalar files (`tcksample`, `tsfinfo`).
        Tournier, J.-D. et al. MRtrix3: A fast, flexible and open software
        framework for medical image processing and visualisation. NeuroImage 202,
        116137 (2019). doi:10.1016/j.neuroimage.2019.116137.
    """
    path = Path(path)
    with open_file_read(path) as f:
        magic = f.readline().decode("utf-8", "replace").strip()
        if magic != TSF_MAGIC:
            raise ValueError(f"{path} is not a TSF file: no {TSF_MAGIC!r} line.")
        header = read_text_header(f)

        for key in ["datatype", "file"]:
            if key not in header:
                raise ValueError(f"{path} has no {key!r} in its header.")
        dtype = payload_dtype(path, header, "TSF")
        offset = payload_start(path, header, "TSF")
        payload = read_payload(f, offset, dtype, on_progress=on_progress)

    values, offsets = _split_tsf_payload(path, payload)
    check_count(path, header, len(offsets) - 1)
    if tractogram is not None:
        _check_against(path, offsets, tractogram)
    return values, offsets


def read_track_timestamp(path: str | Path) -> str | None:
    """Reads the `timestamp` in the header of a TCK or TSF file, without its data.

    A TSF file and its TCK file belong together when they hold the same timestamp;
    pass it to `write_tsf` to pair new values with a TCK file.

    Args:
        path: A `.tck` or `.tsf` file, also gzipped.

    Returns:
        The timestamp, or None if the header has none.

    Raises:
        ValueError: If the file is not a TCK or TSF file.
    """
    path = Path(path)
    with open_file_read(path) as f:
        magic = f.readline().decode("utf-8", "replace").strip()
        if magic not in (TCK_MAGIC, TSF_MAGIC):
            raise ValueError(
                f"{path} is not a TCK or TSF file: no {TCK_MAGIC!r} or "
                f"{TSF_MAGIC!r} line."
            )
        header = read_text_header(f)
    timestamps = header.get("timestamp")
    return timestamps[0] if timestamps else None


def _check_tsf_input(values: np.ndarray, offsets: np.ndarray, path: Path) -> None:
    """Raises ValueError unless `values` and `offsets` describe finite streamlines."""
    if values.dtype.kind not in "biuf":
        raise ValueError(
            f"Cannot write {path}: TSF holds real numbers, got values of type "
            f"{values.dtype}."
        )
    if values.ndim != 1:
        raise ValueError(
            f"Cannot write {path}: TSF holds one value per point, got values of "
            f"shape {values.shape}."
        )
    if (
        offsets.ndim != 1
        or not np.issubdtype(offsets.dtype, np.integer)
        or len(offsets) == 0
        or offsets[0] != 0
        or offsets[-1] != len(values)
        or (offsets[1:] < offsets[:-1]).any()
    ):
        raise ValueError(
            f"Cannot write {path}: the offsets must be 1-D integers that start at 0, "
            f"never decrease and end at the number of values ({len(values)})."
        )
    if not np.isfinite(values).all():
        raise ValueError(
            f"Cannot write {path}: some values are NaN or infinite, which TSF uses "
            "to end streamlines and the file."
        )


def write_tsf(
    values: np.ndarray,
    offsets: np.ndarray,
    path: str | Path,
    *,
    timestamp: str | None = None,
) -> None:
    """Writes one value per streamline point as a TSF file, an MRtrix track scalar file.

    float64 and longer floats are written as Float64, everything else as Float32.

    Args:
        values: (P,) one value per point, all finite.
        offsets: (S + 1,) where each streamline starts, as `Tractogram.offsets`.
        path: The `.tsf` file; an existing file is overwritten.
        timestamp: The header's `timestamp`, which MRtrix uses to pair a TSF file
            with its TCK file; some MRtrix commands refuse a TSF file without one.
            Pass the timestamp of the TCK file, from `write_tck` or
            `read_track_timestamp`. Defaults to None, no timestamp.

    Raises:
        ValueError: If `values` is not 1-D, real and finite, `offsets` do not
            match it, or `timestamp` has a '#' or a line break.

    References:
        Follows MRtrix3's track scalar files (`tcksample`, `tsfinfo`).
        Tournier, J.-D. et al. MRtrix3: A fast, flexible and open software
        framework for medical image processing and visualisation. NeuroImage 202,
        116137 (2019). doi:10.1016/j.neuroimage.2019.116137.
    """
    path = Path(path)
    values, offsets = np.asarray(values), np.asarray(offsets)
    _check_tsf_input(values, offsets, path)
    check_timestamp(path, timestamp)
    offsets = offsets.astype(np.int64, copy=False)
    is_double = values.dtype.kind == "f" and values.dtype.itemsize >= 8
    dtype = np.dtype("<f8" if is_double else "<f4")

    # A NaN ends each streamline.
    data = np.insert(values.astype(dtype, copy=False), offsets[1:], np.nan)
    num_streamlines = len(offsets) - 1

    lines = [] if timestamp is None else [f"timestamp: {timestamp}"]
    lines.append(f"datatype: {'Float64LE' if is_double else 'Float32LE'}")
    with open_file_write(path) as f:
        f.write(track_file_header(TSF_MAGIC, lines, num_streamlines))
        write_payload(f, data, dtype)
