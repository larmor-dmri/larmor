"""Header and data helpers shared by the TCK and TSF track files of MRtrix."""

from pathlib import Path

import numpy as np

from larmor._log import LarmorLogger

# The numpy codes of the data types and byte orders of a track file.
_TYPES = {"float32": "f4", "float64": "f8"}
_ORDERS = {"le": "<", "be": ">"}


def payload_dtype(path: Path, header: dict[str, list[str]], fmt: str) -> np.dtype:
    """Returns the numpy dtype of the header's datatype, such as '<f4' for Float32LE.

    Raises:
        ValueError: If the datatype is not Float32 or Float64 with LE or BE.
    """
    datatype = header["datatype"][0]
    name, order = datatype[:-2].lower(), datatype[-2:].lower()
    if name not in _TYPES or order not in _ORDERS:
        raise ValueError(
            f"{path}: unsupported datatype {datatype}; {fmt} is Float32 or Float64, "
            "with LE or BE."
        )
    return np.dtype(_ORDERS[order] + _TYPES[name])


def payload_start(path: Path, header: dict[str, list[str]], fmt: str) -> int:
    """Returns the offset of the data, from the header's `file: . <offset>`.

    Raises:
        ValueError: If there is not exactly one file entry, it names another file, or
            its offset is not a whole number.
    """
    entries = header["file"]
    if len(entries) != 1:
        raise ValueError(
            f"{path} has {len(entries)} file entries; Larmor reads exactly one."
        )
    name, _, number = entries[0].rpartition(" ")
    if name.strip() != ".":
        raise ValueError(
            f"{path}: file entry {entries[0]!r} must be '. <offset>'; {fmt} data is "
            "always in the same file."
        )
    try:
        return int(number)
    except ValueError as e:
        raise ValueError(f"{path}: file offset {number!r} is not a number.") from e


def offsets_from_delimiters(delims: np.ndarray) -> np.ndarray:
    """Returns the (S + 1,) int64 offsets of the streamlines ended by `delims`.

    `delims` holds the increasing indices of the delimiters among the items of the
    data, one item per point; the last item must be a delimiter.
    """
    # Each delimiter removed before streamline k shifts its end back by one.
    offsets = np.zeros(len(delims) + 1, dtype=np.int64)
    offsets[1:] = delims - np.arange(len(delims))
    return offsets


def check_count(path: Path, header: dict[str, list[str]], found: int) -> None:
    """Logs when the header's count is missing or disagrees with the data."""
    # MRtrix writes the count last, so a wrong count marks an interrupted write.
    count = header.get("count", ["missing"])[0]
    if not count.isdigit():
        LarmorLogger.info(
            "%s: the header's count is %s, the file has %d streamlines.",
            path,
            count,
            found,
        )
    elif int(count) != found:
        LarmorLogger.warning(
            "%s: the header's count is %s but the file has %d streamlines; it may "
            "be incomplete (writing was interrupted). All %d were read.",
            path,
            count,
            found,
            found,
        )


def track_file_header(magic: str, lines: list[str], count: int) -> bytes:
    """Builds the header of a track file, padded up to the start of the data.

    The data offset does not depend on the count, so the header can be rewritten in
    place once the count is known.

    Args:
        magic: The first line of the file.
        lines: The `key: value` lines between the magic and `file`, `datatype` last.
        count: The number of streamlines to write in the header.

    Returns:
        The header bytes, padded with zeros to the start of the data.
    """
    prefix = "".join(f"{line}\n" for line in [magic, *lines]).encode()
    offset = len(prefix) + 65
    offset += -offset % 4
    rest = f"file: . {offset}\ncount: {count}\nEND\n".encode()
    return (prefix + rest).ljust(offset, b"\0")


def check_timestamp(path: Path, timestamp: str | None) -> None:
    """Raises ValueError if the timestamp cannot be a header line's value."""
    if timestamp is not None and any(c in timestamp for c in "#\r\n"):
        raise ValueError(
            f"Cannot write {path}: the timestamp {timestamp!r} has a '#' or a line "
            "break, which a header line cannot hold."
        )
