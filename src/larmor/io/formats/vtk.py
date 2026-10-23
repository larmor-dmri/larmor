"""Reads and writes tractograms as legacy VTK polydata files (.vtk and .fib)."""

import math
import re
from pathlib import Path

import numpy as np

from larmor.io.reader import Progress, open_file_read, read_all
from larmor.io.writer import WriteFile, open_file_write
from larmor.types import Tractogram

# The first line of a legacy VTK file, with its version.
_VERSION = re.compile(r"# vtk DataFile Version (\d+)\.(\d+)", re.IGNORECASE)
# The space tag some tools put in the title line.
_SPACE_TAG = re.compile(r"\bSPACE=(RAS|LPS)\b", re.IGNORECASE)
# The spaces the points of a file can be in.
_SPACES = ("RAS", "LPS")
# In ASCII, a block of numbers ends where a line starts with a keyword.
_NEXT_KEYWORD = re.compile(rb"\n[ \t\r]*[A-Z]")

# The numpy code of each VTK type name, in lower case.
_TYPES = {
    "char": "i1",
    "unsigned_char": "u1",
    "short": "i2",
    "unsigned_short": "u2",
    "int": "i4",
    "unsigned_int": "u4",
    "long": "i8",
    "unsigned_long": "u8",
    "float": "f4",
    "double": "f8",
    "vtkidtype": "i8",
    **{f"vtktypeint{n}": f"i{n // 8}" for n in (8, 16, 32, 64)},
    **{f"vtktypeuint{n}": f"u{n // 8}" for n in (8, 16, 32, 64)},
}
# The VTK name Larmor writes for each numpy type; bool is written as unsigned_char.
_VTK_NAMES = {
    "b1": "unsigned_char",
    "i1": "char",
    "u1": "unsigned_char",
    "i2": "short",
    "u2": "unsigned_short",
    "i4": "int",
    "u4": "unsigned_int",
    "i8": "vtktypeint64",
    "u8": "vtktypeuint64",
    "f4": "float",
    "f8": "double",
}
# The polydata cells other than lines, which Larmor does not read.
_NOT_LINES = ("VERTICES", "POLYGONS", "TRIANGLE_STRIPS")


class _Cursor:
    """Reads keyword lines and blocks of numbers from a legacy VTK file in memory."""

    def __init__(self, data: bytes | bytearray, path: Path) -> None:
        self.data, self.path, self.pos = data, path, 0
        self.binary = False

    def raw_line(self) -> str | None:
        """Returns the next line, without its line ending, or None at the end."""
        if self.pos >= len(self.data):
            return None
        end = self.data.find(b"\n", self.pos)
        end = len(self.data) if end < 0 else end
        line = self.data[self.pos : end].decode("latin1").rstrip("\r")
        self.pos = end + 1
        return line

    def line(self) -> list[str] | None:
        """Returns the words of the next non-blank line, or None at the end."""
        while (line := self.raw_line()) is not None:
            if line.strip():
                return line.split()
        return None

    def peek(self) -> list[str] | None:
        """Returns the words of the next non-blank line without moving past it."""
        pos = self.pos
        words = self.line()
        self.pos = pos
        return words

    def values(self, count: int, type_name: str) -> np.ndarray:
        """Returns the next `count` numbers of type `type_name`, in native byte order.

        Raises:
            ValueError: If the type is unknown or the file holds fewer numbers.
        """
        dtype = _dtype(self.path, type_name)
        if self.binary:
            size = count * dtype.itemsize
            raw = self.data[self.pos : self.pos + size]
            if len(raw) != size:
                raise ValueError(f"{self.path} ends inside a block of {count} values.")
            self.pos += size
            return np.frombuffer(raw, dtype.newbyteorder(">")).astype(dtype)

        # The block usually runs up to the next keyword line; when it does not, as
        # before a lower-case FIELD array name, it ends after `count` numbers.
        match = _NEXT_KEYWORD.search(self.data, self.pos)
        end = match.start() if match else len(self.data)
        # numpy parses bytes as numbers, but a bytearray as a sequence of bytes.
        tokens = bytes(self.data[self.pos : end]).split()
        if len(tokens) != count:
            end = self._end_of_tokens(count)
            tokens = bytes(self.data[self.pos : end]).split()
        try:
            numbers = np.array(tokens, dtype=np.float64)
        except ValueError as e:
            raise ValueError(f"{self.path}: a block of numbers has text in it.") from e
        self.pos = end
        return numbers.astype(dtype)

    def _end_of_tokens(self, count: int) -> int:
        """Returns the end of the line holding the `count`-th token from `pos`."""
        pos, seen = self.pos, 0
        while seen < count and pos <= len(self.data):
            end = self.data.find(b"\n", pos)
            end = len(self.data) if end < 0 else end
            seen += len(self.data[pos:end].split())
            pos = end + 1
        if seen < count:
            raise ValueError(f"{self.path}: expected {count} values, found {seen}.")
        if seen > count:
            raise ValueError(f"{self.path}: a block of {count} values ends mid-line.")
        return pos

    def skip_metadata(self) -> None:
        """Skips a METADATA block (VTK 9), which ends at a blank line."""
        while (line := self.raw_line()) is not None and line.strip():
            pass


def _dtype(path: Path, type_name: str) -> np.dtype:
    """Returns the numpy dtype of a VTK type name such as "float".

    Raises:
        ValueError: If the type is not one Larmor reads.
    """
    code = _TYPES.get(type_name.lower())
    if code is None:
        raise ValueError(f"{path}: unsupported VTK type {type_name!r}.")
    return np.dtype(code)


def _words(path: Path, words: list[str] | None, n: int) -> list[str]:
    """Returns `words` after checking that there are at least `n`.

    Raises:
        ValueError: If the file ended, or the line is too short.
    """
    if words is None or len(words) < n:
        raise ValueError(f"{path}: incomplete line {' '.join(words or [])!r}.")
    return words


def _integer(path: Path, word: str) -> int:
    """Returns `word` as a non-negative integer.

    Raises:
        ValueError: If it is not one.
    """
    if not word.isdigit():
        raise ValueError(f"{path}: {word!r} is not a count.")
    return int(word)


def _legacy_lines(
    path: Path, flat: np.ndarray, n_lines: int
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the offsets and point indices of `n_lines` legacy LINES.

    `flat` holds each line's number of points followed by its point indices.

    Raises:
        ValueError: If the counts do not add up to the section's size.
    """
    counts_at = []
    pos, size = 0, len(flat)
    view = memoryview(np.ascontiguousarray(flat, dtype=np.int64))
    while pos < size:
        if view[pos] < 0:
            raise ValueError(f"{path}: a line has {view[pos]} points.")
        counts_at.append(pos)
        pos += 1 + view[pos]
    if pos != size or len(counts_at) != n_lines or (flat < 0).any():
        raise ValueError(f"{path}: the LINES section is inconsistent with its size.")
    starts = np.array(counts_at, dtype=np.int64)
    offsets = np.zeros(n_lines + 1, dtype=np.int64)
    np.cumsum(flat[starts], out=offsets[1:])
    keep = np.ones(size, dtype=bool)
    keep[starts] = False
    return offsets, flat[keep].astype(np.int64)


def _attribute(
    path: Path, cur: _Cursor, words: list[str], count: int
) -> list[tuple[str, np.ndarray]]:
    """Reads one attribute (SCALARS, VECTORS, NORMALS or FIELD) as named arrays.

    `words` is the attribute's keyword line and `count` the number of points or
    cells it covers.

    Returns:
        The (name, values) pairs, values of shape (count, k).

    Raises:
        ValueError: If the attribute is of another kind or its sizes do not match.
    """
    key = words[0].upper()
    if key == "SCALARS":
        _, name, type_name, *rest = _words(path, words, 3)
        k = _integer(path, rest[0]) if rest else 1
        lookup = cur.peek()
        if lookup and lookup[0].upper() == "LOOKUP_TABLE":
            cur.line()
        return [(name, cur.values(count * k, type_name).reshape(count, k))]
    if key in ("VECTORS", "NORMALS"):
        _, name, type_name = _words(path, words, 3)[:3]
        return [(name, cur.values(count * 3, type_name).reshape(count, 3))]
    if key == "FIELD":
        arrays = []
        for _ in range(_integer(path, _words(path, words, 3)[2])):
            name, k_word, n_word, type_name = _words(path, cur.line(), 4)[:4]
            if _integer(path, n_word) != count:
                raise ValueError(
                    f"{path}: field array {name!r} has {n_word} entries, expected "
                    f"{count}."
                )
            k = _integer(path, k_word)
            arrays.append((name, cur.values(count * k, type_name).reshape(count, k)))
            following = cur.peek()
            if following and following[0].upper() == "METADATA":
                cur.line()
                cur.skip_metadata()
        return arrays
    raise ValueError(f"{path}: VTK attribute {words[0]} is not supported.")


def read_vtk(
    path: str | Path,
    *,
    space: str | None = None,
    on_progress: Progress | None = None,
) -> Tractogram:
    """Reads a legacy VTK polydata file of lines (`.vtk`, `.fib`) into a `Tractogram`.

    The points are world coordinates in mm, gathered in line order; points no line
    uses are dropped. `POINT_DATA` and `CELL_DATA` arrays go into `point_data` and
    `streamline_data`.

    The format does not fix the space of the points: some tools write RAS and
    others LPS, which differs by the sign of x and y. The space comes from `space`
    when it is given, else from a `SPACE=RAS` or `SPACE=LPS` tag in the file's
    title line, else it is taken as RAS. LPS points are converted to RAS.

    Args:
        path: The `.vtk` or `.fib` file.
        space: "RAS" or "LPS" (any case), the space the file's points are in.
            Defaults to None, which follows the file's tag.
        on_progress: Called as the file is read, see `load_volume`.

    Returns:
        Tractogram: The streamlines, in world (RAS) coordinates.

    Raises:
        ValueError: If `space` is not RAS or LPS, or the file is not a legacy VTK
            polydata of lines Larmor can read.
        ReadCancelledError: If on_progress raises it.
    """
    path = Path(path)
    if space is not None and space.upper() not in _SPACES:
        raise ValueError(f"space must be RAS or LPS, got {space!r}.")
    with open_file_read(path) as f:
        cur = _Cursor(read_all(f, on_progress=on_progress), path)

    space = _read_header(path, cur, space)
    points, lines, data, declared = _read_sections(path, cur)

    if points is None:
        points = np.zeros((0, 3), dtype=np.float32)
    offsets, conn = (
        lines if lines is not None else (np.zeros(1, np.int64), np.zeros(0, np.int64))
    )
    if len(conn) > 0 and (conn.min() < 0 or conn.max() >= len(points)):
        raise ValueError(f"{path}: a line uses a point that does not exist.")

    expected = {"POINT_DATA": len(points), "CELL_DATA": len(offsets) - 1}
    for kind, count in declared.items():
        if count != expected[kind]:
            raise ValueError(
                f"{path}: {kind} {count}, but the file has {expected[kind]}."
            )
    in_order = len(conn) == len(points) and np.array_equal(conn, np.arange(len(points)))
    point_data, streamline_data = _data_arrays(
        path, data, len(points), len(offsets) - 1, None if in_order else conn
    )

    points = points if in_order else points[conn]
    if space.upper() == "LPS":
        points = points * np.array([-1, -1, 1], dtype=points.dtype)
    return Tractogram(points, offsets, point_data, streamline_data)


def _read_header(path: Path, cur: _Cursor, space: str | None) -> str:
    """Reads the four header lines and returns the space of the points.

    Returns:
        str: `space` when given, else the title line's tag, else RAS.

    Raises:
        ValueError: If the version, the ASCII or BINARY line or the dataset is not
            one Larmor reads.
    """
    version = _VERSION.match(cur.raw_line() or "")
    if not version:
        raise ValueError(f"{path} is not a legacy VTK file: no version line.")
    if not (2, 0) <= (int(version[1]), int(version[2])) <= (5, 1):
        raise ValueError(
            f"{path}: VTK version {version[1]}.{version[2]}; Larmor reads 2.0 to 5.1."
        )
    tag = _SPACE_TAG.search(cur.raw_line() or "")
    if space is None:
        space = tag[1] if tag else "RAS"
    mode = (cur.raw_line() or "").strip().upper()
    if mode not in ("ASCII", "BINARY"):
        raise ValueError(f"{path}: expected ASCII or BINARY, got {mode!r}.")
    dataset = [w.upper() for w in (cur.line() or [])]
    if dataset != ["DATASET", "POLYDATA"]:
        raise ValueError(f"{path}: Larmor reads DATASET POLYDATA, got {dataset}.")
    cur.binary = mode == "BINARY"
    return space


def _read_sections(
    path: Path, cur: _Cursor
) -> tuple[
    np.ndarray | None,
    tuple[np.ndarray, np.ndarray] | None,
    dict[str, dict[str, np.ndarray]],
    dict[str, int],
]:
    """Reads the sections after the header.

    Returns:
        The points (None if absent), the line offsets and point indices (None if
        absent), the arrays of POINT_DATA and CELL_DATA by name, and the counts
        the data sections declare.

    Raises:
        ValueError: If a section is malformed or holds cells other than lines.
    """
    points: np.ndarray | None = None
    lines: tuple[np.ndarray, np.ndarray] | None = None
    data: dict[str, dict[str, np.ndarray]] = {"POINT_DATA": {}, "CELL_DATA": {}}
    target: tuple[str, int] | None = None
    declared: dict[str, int] = {}

    while (words := cur.line()) is not None:
        key = words[0].upper()
        if key == "POINTS":
            points = _read_points(path, cur, words)
        elif key == "LINES":
            lines = _read_lines(path, cur, words)
        elif key in _NOT_LINES:
            raise ValueError(
                f"{path} has {key}: Larmor reads only lines (streamlines)."
            )
        elif key in data:
            target = (key, _integer(path, _words(path, words, 2)[1]))
            declared[key] = target[1]
        elif key == "METADATA":
            cur.skip_metadata()
        else:
            if target is None:
                raise ValueError(f"{path}: unexpected {words[0]} outside of data.")
            for name, values in _attribute(path, cur, words, target[1]):
                if name in data[target[0]]:
                    raise ValueError(f"{path}: {target[0]} has two arrays {name!r}.")
                data[target[0]][name] = values
    return points, lines, data, declared


def _read_points(path: Path, cur: _Cursor, words: list[str]) -> np.ndarray:
    """Reads a POINTS section.

    Returns:
        np.ndarray: The points, shape (n_points, 3).

    Raises:
        ValueError: If the points are not float or double.
    """
    _, n, type_name = _words(path, words, 3)[:3]
    if type_name.lower() not in ("float", "double"):
        raise ValueError(f"{path}: points must be float or double.")
    return cur.values(3 * _integer(path, n), type_name).reshape(-1, 3)


def _read_lines(
    path: Path, cur: _Cursor, words: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Reads a LINES section, in the 5.1 or the legacy layout.

    Returns:
        The line offsets and the point indices.

    Raises:
        ValueError: If the section is inconsistent.
    """
    n_lines, size = (_integer(path, w) for w in _words(path, words, 3)[1:3])
    following = cur.peek()
    if not (following and following[0].upper() == "OFFSETS"):
        # Legacy LINES have no type; VTK stores them as int32.
        return _legacy_lines(path, cur.values(size, "int"), n_lines)
    # In version 5.1 n_lines counts the offsets, one more than the lines.
    offsets = cur.values(n_lines, _words(path, cur.line(), 2)[1])
    conn_words = _words(path, cur.line(), 2)
    if conn_words[0].upper() != "CONNECTIVITY":
        raise ValueError(f"{path}: OFFSETS without CONNECTIVITY.")
    conn = cur.values(size, conn_words[1])
    offsets = offsets.astype(np.int64)
    if (
        len(offsets) == 0
        or offsets[0] != 0
        or offsets[-1] != size
        or (np.diff(offsets) < 0).any()
    ):
        raise ValueError(f"{path}: the line offsets are inconsistent.")
    return offsets, conn.astype(np.int64)


def _data_arrays(
    path: Path,
    data: dict[str, dict[str, np.ndarray]],
    n_points: int,
    n_lines: int,
    conn: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Returns the point and streamline arrays, checked against the counts.

    Returns:
        The point arrays and the streamline arrays by name. Point arrays are put
        in line order with `conn`, unless it is None; arrays of one column
        become 1D.

    Raises:
        ValueError: If an array's length does not match its count.
    """
    point_data: dict[str, np.ndarray] = {}
    streamline_data: dict[str, np.ndarray] = {}
    for kind, values_by_name, count, out in (
        ("POINT_DATA", data["POINT_DATA"], n_points, point_data),
        ("CELL_DATA", data["CELL_DATA"], n_lines, streamline_data),
    ):
        for name, values in values_by_name.items():
            if len(values) != count:
                raise ValueError(
                    f"{path}: {kind} has {len(values)} entries, expected {count}."
                )
            if kind == "POINT_DATA" and conn is not None:
                values = values[conn]
            out[name] = values[:, 0] if values.shape[1] == 1 else values
    return point_data, streamline_data


def _check_fields(data: dict[str, np.ndarray], label: str) -> None:
    """Checks that VTK can store each array of `data`, under its name.

    Raises:
        ValueError: If a name is empty or has whitespace, or an array has more
            than 2 dimensions or a type VTK has no name for.
    """
    for name, values in data.items():
        if not name or any(c.isspace() for c in name):
            raise ValueError(f"The {label} name {name!r} cannot be written to VTK.")
        if values.ndim > 2:
            raise ValueError(
                f"{label}[{name!r}] has shape {values.shape}; VTK stores (N,) or "
                "(N, k)."
            )
        if values.dtype.str[1:] not in _VTK_NAMES:
            raise ValueError(
                f"VTK cannot store {label}[{name!r}] of type {values.dtype}."
            )


def _write_values(f: WriteFile, values: np.ndarray, binary: bool, dtype: str) -> None:
    """Writes a block of numbers of type `dtype`, big-endian or as ASCII rows."""
    if binary:
        f.write(np.ascontiguousarray(values, dtype=">" + dtype).tobytes())
        f.write(b"\n")
    else:
        # Enough significant digits to read every float32 and float64 back exactly.
        fmt = {"f4": "%.9g", "f8": "%.17g"}.get(dtype, "%d")
        rows = values.reshape(len(values), math.prod(values.shape[1:]))
        np.savetxt(f, rows.astype(dtype, copy=False), fmt=fmt)


def _write_fields(
    f: WriteFile, kind: str, data: dict[str, np.ndarray], count: int, binary: bool
) -> None:
    """Writes `data` as one FIELD of arrays, each in its own type, under `kind`."""
    if not data:
        return
    f.write(f"{kind} {count}\nFIELD FieldData {len(data)}\n".encode())
    for name, values in data.items():
        k = math.prod(values.shape[1:])
        code = values.dtype.str[1:]
        f.write(f"{name} {k} {count} {_VTK_NAMES[code]}\n".encode())
        _write_values(f, values.reshape(count, k), binary, code)


def write_vtk(tractogram: Tractogram, path: str | Path, *, binary: bool = True) -> None:
    """Writes a `Tractogram` as a legacy VTK polydata file (`.vtk` or `.fib`).

    The points are written as float32 in world (RAS) coordinates, and the title line
    says so with the tag `SPACE=RAS`. `point_data` and `streamline_data` are
    written as FIELD arrays under `POINT_DATA` and `CELL_DATA`, each in its own
    type, with bool as unsigned_char.

    Args:
        tractogram: The tractogram to write.
        path: The `.vtk` or `.fib` file; an existing file is overwritten.
        binary: If True, write binary data; otherwise ASCII. Defaults to True.

    Raises:
        ValueError: If a data name is empty or has whitespace, a data array has more
            than 2 dimensions or a type VTK cannot store, or the tractogram is too
            large for VTK's int32 indices.
    """
    path = Path(path)
    _check_fields(tractogram.point_data, "point_data")
    _check_fields(tractogram.streamline_data, "streamline_data")
    n_points, n_lines = tractogram.num_points, tractogram.num_streamlines
    if n_points + n_lines > np.iinfo(np.int32).max:
        raise ValueError(f"Cannot write {path}: too many points for VTK's indices.")

    # Legacy LINES hold each line's number of points, then its point indices.
    lengths = np.diff(tractogram.offsets)
    flat = np.empty(n_points + n_lines, dtype=np.int64)
    counts_at = tractogram.offsets[:-1] + np.arange(n_lines)
    is_index = np.ones(len(flat), dtype=bool)
    is_index[counts_at] = False
    flat[counts_at] = lengths
    flat[is_index] = np.arange(n_points)

    with open_file_write(path) as f:
        f.write(b"# vtk DataFile Version 3.0\nlarmor tractogram SPACE=RAS\n")
        f.write(b"BINARY\n" if binary else b"ASCII\n")
        f.write(f"DATASET POLYDATA\nPOINTS {n_points} float\n".encode())
        _write_values(f, tractogram.points, binary, "f4")
        f.write(f"LINES {n_lines} {len(flat)}\n".encode())
        if binary:
            _write_values(f, flat, binary, "i4")
        else:
            bounds = np.append(counts_at, len(flat))
            for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
                f.write(" ".join(map(str, flat[start:stop])).encode() + b"\n")
        _write_fields(f, "POINT_DATA", tractogram.point_data, n_points, binary)
        _write_fields(f, "CELL_DATA", tractogram.streamline_data, n_lines, binary)
