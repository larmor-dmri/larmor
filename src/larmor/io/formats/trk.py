"""Reads and writes TRK files, the tractogram format of TrackVis."""

import math
import os
from collections.abc import Iterator, Sequence
from io import SEEK_END
from pathlib import Path
from typing import NamedTuple

import numpy as np

from larmor._log import LarmorLogger
from larmor.io.formats._affine import apply_affine
from larmor.io.reader import (
    Progress,
    ReadFile,
    check_not_gzip,
    open_file_read,
    read_chunk,
    read_payload,
    read_struct,
)
from larmor.io.writer import (
    BatchWriter,
    open_batch_file,
    open_file_write,
    write_payload,
    write_struct,
)
from larmor.types import Grid, Tractogram

# The world axis (0 = x, 1 = y, 2 = z) that each orientation letter refers to.
_AXIS = {"R": 0, "L": 0, "A": 1, "P": 1, "S": 2, "I": 2}

# The first bytes of a TRK file.
TRK_MAGIC = b"TRACK"
# The size of the header in bytes.
HEADER_SIZE = 1000
# The largest number of scalar names, and of property names, in a header.
MAX_NAMES = 10
# The size of each name in bytes.
NAME_SIZE = 20

# The TRK header, version 2.
TRK_HEADER = np.dtype(
    [
        ("id_string", "S6"),
        ("dim", "<i2", 3),
        ("voxel_size", "<f4", 3),
        ("origin", "<f4", 3),
        ("n_scalars", "<i2"),
        ("scalar_name", "S20", MAX_NAMES),
        ("n_properties", "<i2"),
        ("property_name", "S20", MAX_NAMES),
        ("vox_to_ras", "<f4", (4, 4)),
        ("reserved", "S444"),
        ("voxel_order", "S4"),
        ("pad2", "S4"),
        ("image_orientation_patient", "<f4", 6),
        ("pad1", "S2"),
        ("invert_x", "S1"),
        ("invert_y", "S1"),
        ("invert_z", "S1"),
        ("swap_xy", "S1"),
        ("swap_yz", "S1"),
        ("swap_zx", "S1"),
        ("n_count", "<i4"),
        ("version", "<i4"),
        ("hdr_size", "<i4"),
    ]
)


def _decode_names(
    path: Path, names: np.ndarray, total: int, rest: str
) -> list[tuple[str, int]]:
    """Returns (name, number of values) for each named block of the `total` values.

    A name holds its number of values after a NUL byte (`fa`, NUL, `3`). A name
    with 0 values is skipped, and values left without a name are grouped under
    `rest`.

    Raises:
        ValueError: If a name is malformed, or the names hold more than `total`
            values.
    """
    blocks = []
    for raw in names:
        if not raw:
            continue
        parts = raw.decode("latin1").rstrip("\x00").split("\x00")
        count = parts[1] if len(parts) == 2 else "1"
        if len(parts) > 2 or not count.isdigit():
            raise ValueError(f"{path}: malformed TRK name {raw!r}.")
        if int(count) > 0:
            blocks.append((parts[0], int(count)))
    named = sum(k for _, k in blocks)
    if named > total:
        raise ValueError(
            f"{path}: the {rest} names hold {named} values, but the header has {total}."
        )
    if named < total:
        blocks.append((rest, total - named))
    return blocks


def _encode_names(data: dict[str, np.ndarray], label: str) -> tuple[np.ndarray, int]:
    """Returns the encoded names of `data`, sorted, and the total number of values.

    Raises:
        ValueError: If there are more than 10 names, or one is empty, contains a NUL,
            is not latin-1 or does not fit in 20 bytes.
    """
    if len(data) > MAX_NAMES:
        raise ValueError(
            f"TRK holds at most {MAX_NAMES} {label} names, got {len(data)}."
        )
    names = np.zeros(MAX_NAMES, dtype="S20")
    total = 0
    for i, name in enumerate(sorted(data)):
        k = 1 if data[name].ndim == 1 else int(np.prod(data[name].shape[1:]))
        if not name or "\x00" in name:
            raise ValueError(f"The {label} name {name!r} cannot be written to TRK.")
        try:
            encoded = (name if k == 1 else f"{name}\x00{k}").encode("latin1")
        except UnicodeEncodeError as e:
            raise ValueError(
                f"The {label} name {name!r} is not latin-1, which TRK needs."
            ) from e
        if len(encoded) > NAME_SIZE:
            raise ValueError(
                f"The {label} name {name!r} does not fit in TRK's {NAME_SIZE} bytes."
            )
        names[i] = encoded
        total += k
    return names, total


def _as_columns(values: np.ndarray) -> np.ndarray:
    """Returns (N,) or (N, ...) `values` as (N, k), also when N is 0."""
    return values.reshape(len(values), math.prod(values.shape[1:]))


def _trackvis_to_world(
    dims: np.ndarray,
    voxel_size: np.ndarray,
    voxel_order: str,
    affine: np.ndarray,
    codes: Sequence[str],
) -> np.ndarray:
    """Returns the affine from TrackVis voxel mm to world (RAS+) mm.

    Voxel mm are divided by the voxel size, moved from the corner to the centre of
    the voxel, brought from the header's `voxel_order` to the axes of `affine`
    (whose orientation is `codes`), then mapped by `affine`.
    """
    scale = np.diag([*(1.0 / voxel_size), 1.0])
    centre = np.eye(4)
    centre[:3, 3] = -0.5

    # Row i is the header's axis i, and points at the affine axis along the same
    # world axis, flipped if the letters differ.
    reorder = np.zeros((4, 4))
    reorder[3, 3] = 1.0
    for i, c in enumerate(voxel_order):
        j = next(k for k in range(3) if _AXIS[codes[k]] == _AXIS[c])
        if codes[j] == c:
            reorder[i, j] = 1.0
        else:
            reorder[i, j] = -1.0
            reorder[i, 3] = dims[i] - 1
    return affine @ reorder @ centre @ scale


def _read_trk_header(path: Path, f: ReadFile) -> tuple[np.ndarray, str]:
    """Returns the header and the byte order of the file ('<' or '>').

    Raises:
        ValueError: If `hdr_size` is not 1000 in either byte order, or the magic is
            not TRACK.
    """
    record = read_struct(f, TRK_HEADER)
    order = "<"
    if record["hdr_size"][0] != HEADER_SIZE:
        record = record.view(TRK_HEADER.newbyteorder(">"))
        order = ">"
        if record["hdr_size"][0] != HEADER_SIZE:
            raise ValueError(
                f"{path} is not a TRK file: hdr_size is not {HEADER_SIZE}."
            )
    header = record[0]
    if not bytes(header["id_string"]).startswith(TRK_MAGIC):
        raise ValueError(f"{path} is not a TRK file: no {TRK_MAGIC!r} magic.")
    return header, order


def _header_grid(
    path: Path, header: np.ndarray, reference: Grid | None
) -> tuple[Grid, np.ndarray]:
    """Returns the tractogram's reference grid and the vox_to_ras affine to use.

    Raises:
        ValueError: If the version is unsupported, the affine is missing without a
            reference or does not match the reference, or the affine is singular.
    """
    version = int(header["version"])
    if version == 3:
        LarmorLogger.warning("%s is TRK version 3; it is read as version 2.", path)
    elif version not in (1, 2):
        raise ValueError(f"{path}: TRK version {version} is not supported.")

    affine = header["vox_to_ras"].astype(np.float64)
    if version == 1 or affine[3, 3] == 0:
        if reference is None:
            raise ValueError(
                f"{path} has no vox_to_ras (TRK version 1 or not recorded); give a "
                "reference grid to place its streamlines."
            )
        # The header's dim and voxel_size still place the points, so they must be
        # the reference's.
        sizes = np.linalg.norm(reference.affine[:3, :3], axis=0)
        if tuple(int(d) for d in header["dim"]) != reference.shape or not np.allclose(
            sizes, header["voxel_size"], rtol=0, atol=1e-3
        ):
            raise ValueError(
                f"{path}: the reference grid (shape {reference.shape}, voxel sizes "
                f"{sizes}) does not match the file's dim {header['dim']} and "
                f"voxel_size {header['voxel_size']}."
            )
        return reference, np.array(reference.affine, dtype=np.float64)

    dims = tuple(int(d) for d in header["dim"])
    try:
        grid = Grid((dims[0], dims[1], dims[2]), affine)
        grid.orientation  # noqa: B018 (raises on a singular affine)
    except ValueError as e:
        raise ValueError(f"{path}: invalid vox_to_ras or dim: {e}") from e
    if reference is not None and not reference.matches(grid):
        raise ValueError(f"{path}: the reference grid does not match the file's grid.")
    return grid, affine


def _find_trk_streamlines(
    path: Path, words: np.ndarray, step: int, n_props: int
) -> tuple[np.ndarray, int]:
    """Returns where each whole streamline of `words` starts, and where the last ends.

    `words` is TRK data as int32; each streamline is a point count n, n rows of
    `step` float32 and `n_props` float32. A streamline that runs past the end of
    `words` is left out.

    Raises:
        ValueError: If a streamline has a negative point count.
    """
    counts_at = []
    pos, size = 0, len(words)
    view = memoryview(words)
    while pos < size:
        n = view[pos]
        if n < 0:
            raise ValueError(f"{path}: a streamline has {n} points.")
        end = pos + 1 + n * step + n_props
        if end > size:
            break
        counts_at.append(pos)
        pos = end
    return np.array(counts_at, dtype=np.int64), pos


def _gather_trk_streamlines(
    words: np.ndarray, starts: np.ndarray, end: int, step: int, n_props: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns the rows, offsets, properties and counts of the streamlines at `starts`.

    `end` is where the last of them ends in `words`.
    """
    counts = words[starts].astype(np.int64)
    offsets = np.zeros(len(starts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    values = words.view(np.float32)
    props_at = starts + 1 + counts * step
    properties = values[props_at[:, None] + np.arange(n_props)]

    keep = np.ones(end, dtype=bool)
    keep[starts] = False
    keep[(props_at[:, None] + np.arange(n_props)).ravel()] = False
    rows = values[:end][keep].reshape(-1, step)
    return rows, offsets, properties, counts


def _split_trk_payload(
    path: Path, words: np.ndarray, step: int, n_props: int, cut: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns the rows (points and scalars), offsets, properties and counts of data.

    `cut` says the file ended inside a value. An incomplete last streamline is
    dropped with a warning.

    Raises:
        ValueError: If a streamline has a negative point count.
    """
    starts, end = _find_trk_streamlines(path, words, step, n_props)
    if end < len(words) or cut:
        LarmorLogger.warning(
            "%s ends inside its last streamline; it was dropped.", path
        )
    return _gather_trk_streamlines(words, starts, end, step, n_props)


class _TrkLayout(NamedTuple):
    """How a TRK file's data is laid out and placed in the world."""

    grid: Grid
    to_world: np.ndarray  # from TrackVis voxel mm to world mm
    step: int  # values per point, 3 coordinates and the scalars
    n_props: int  # values of properties per streamline
    scalars: list[tuple[str, int]]  # (name, number of values) per scalar
    properties: list[tuple[str, int]]  # (name, number of values) per property


def _trk_layout(path: Path, header: np.ndarray, reference: Grid | None) -> _TrkLayout:
    """Returns the layout of a TRK's data from its header.

    Raises:
        ValueError: If the header's grid, voxel order, voxel sizes or data names are
            invalid, or the reference does not fit the file.
    """
    grid, affine = _header_grid(path, header, reference)
    voxel_order = bytes(header["voxel_order"]).decode("latin1").strip().upper()
    if not voxel_order:
        LarmorLogger.warning(
            "%s has no voxel_order; LPS is assumed, as TrackVis.", path
        )
        voxel_order = "LPS"
    axes = sorted(_AXIS.get(c, -1) for c in voxel_order)
    if axes != [0, 1, 2]:
        raise ValueError(
            f"{path}: voxel_order {voxel_order!r} is not one of R/L, A/P and S/I each."
        )
    voxel_size = header["voxel_size"].astype(np.float64)
    if (voxel_size <= 0).any():
        raise ValueError(f"{path}: voxel sizes {voxel_size} must be positive.")

    n_scalars, n_props = int(header["n_scalars"]), int(header["n_properties"])
    if n_scalars < 0 or n_props < 0:
        raise ValueError(
            f"{path}: n_scalars ({n_scalars}) and n_properties ({n_props}) cannot be "
            "negative."
        )
    to_world = _trackvis_to_world(
        header["dim"], voxel_size, voxel_order, affine, grid.orientation
    )
    return _TrkLayout(
        grid,
        to_world,
        3 + n_scalars,
        n_props,
        _decode_names(path, header["scalar_name"], n_scalars, "scalars"),
        _decode_names(path, header["property_name"], n_props, "properties"),
    )


def _trk_tractogram(
    layout: _TrkLayout, rows: np.ndarray, offsets: np.ndarray, properties: np.ndarray
) -> Tractogram:
    """Builds the tractogram, in world coordinates, from a TRK's rows and properties."""
    points = apply_affine(layout.to_world, rows[:, :3])
    point_data, streamline_data = {}, {}
    col = 3
    for name, k in layout.scalars:
        block = rows[:, col : col + k]
        point_data[name] = block[:, 0].copy() if k == 1 else block.copy()
        col += k
    col = 0
    for name, k in layout.properties:
        block = properties[:, col : col + k]
        streamline_data[name] = block[:, 0].copy() if k == 1 else block.copy()
        col += k
    return Tractogram(points, offsets, point_data, streamline_data, layout.grid)


def _check_trk_count(path: Path, header: np.ndarray, found: int) -> None:
    """Warns if the file is unfinished or its n_count disagrees with `found`."""
    # n_count 0 means unknown.
    n_count = int(header["n_count"])
    if bytes(header["reserved"]).startswith(UNFINISHED):
        LarmorLogger.warning(
            "%s was not finished (writing was interrupted); it is incomplete. The "
            "%d streamlines written so far were read.",
            path,
            found,
        )
    elif n_count not in (0, found):
        LarmorLogger.warning(
            "%s: the header's n_count is %d but the file has %d streamlines; it may "
            "be incomplete (writing was interrupted). All %d were read.",
            path,
            n_count,
            found,
            found,
        )


def read_trk(
    path: str | Path,
    *,
    reference: Grid | None = None,
    on_progress: Progress | None = None,
) -> Tractogram:
    """Reads a TRK file, a TrackVis tractogram, into a `Tractogram`.

    Points are converted from TrackVis voxel mm to world coordinates. Scalars go
    into `point_data` and properties into `streamline_data`, named as in the
    header. The data is read to the end, whatever `n_count` says; a last
    streamline cut short by an interrupted write is dropped with a warning. A file
    whose writing was not finished, or whose `n_count` is not 0 and disagrees with
    the data, gives a warning that it may be incomplete.

    Args:
        path: The `.trk` file.
        reference: The grid of the image the tractogram belongs to. Needed when the
            file has no vox_to_ras (version 1); otherwise it must match the file's.
            Defaults to None.
        on_progress: Called as the file is read, see `load_volume`.

    Returns:
        Tractogram: The streamlines, in world coordinates, with the file's scalars and
        properties.

    Raises:
        ValueError: If the file is not a TRK file Larmor can read, has no vox_to_ras
            and no reference is given, or the reference does not match the file.
        ReadCancelledError: If on_progress raises it.

    References:
        Converts points to world coordinates as nibabel does.
    """
    path = Path(path)
    with open_file_read(path) as f:
        header, order = _read_trk_header(path, f)
        # Read as bytes, so that a file cut inside a value gets one warning below.
        raw = read_payload(f, HEADER_SIZE, np.dtype("u1"), on_progress=on_progress)
    cut = len(raw) % 4
    words = raw[: len(raw) - cut].view(order + "i4").astype("=i4", copy=False)

    layout = _trk_layout(path, header, reference)
    rows, offsets, properties, _ = _split_trk_payload(
        path, words, layout.step, layout.n_props, cut > 0
    )
    tractogram = _trk_tractogram(layout, rows, offsets, properties)
    _check_trk_count(path, header, tractogram.num_streamlines)
    return tractogram


def read_trk_contents(
    path: str | Path,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], int]:
    """Reads the names of a TRK file's data and its count, from the header alone.

    Args:
        path: The `.trk` file, also gzipped.

    Returns:
        The scalars (`point_data`) and properties (`streamline_data`) as (name,
        number of values) pairs, named as `read_trk` names them, and the header's
        n_count (0 when unknown or unfinished).

    Raises:
        ValueError: If the file is not a TRK file or its names are malformed.
    """
    path = Path(path)
    with open_file_read(path) as f:
        header, _ = _read_trk_header(path, f)
    scalars = _decode_names(
        path, header["scalar_name"], int(header["n_scalars"]), "scalars"
    )
    properties = _decode_names(
        path, header["property_name"], int(header["n_properties"]), "properties"
    )
    return scalars, properties, int(header["n_count"])


# The marker, at the start of the header's reserved field, of a file whose writing
# has not finished.
UNFINISHED = b"Larmor: unfinished"


def _layout(data: dict[str, np.ndarray]) -> tuple[tuple[str, int], ...]:
    """Returns the names of `data`, sorted, each with its number of values per item."""
    return tuple(
        (name, math.prod(values.shape[1:])) for name, values in sorted(data.items())
    )


def _trk_header(
    grid: Grid,
    point_data: dict[str, np.ndarray],
    streamline_data: dict[str, np.ndarray],
    n_count: int,
    unfinished: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds the TRK header for a grid and data names, with the streamline count.

    `unfinished` writes the marker of a file whose writing has not finished.

    Returns:
        The header, and the affine from world coordinates to TrackVis voxel mm.

    Raises:
        ValueError: If the grid's shape does not fit TRK, or there are too many or
            too long names.
    """
    if max(grid.shape) > np.iinfo(np.int16).max:
        raise ValueError(f"TRK cannot hold the shape {grid.shape}.")
    scalar_names, n_scalars = _encode_names(point_data, "point_data")
    prop_names, n_props = _encode_names(streamline_data, "streamline_data")

    affine = np.array(grid.affine, dtype=np.float64)
    header = np.zeros((), dtype=TRK_HEADER)
    header["id_string"] = TRK_MAGIC
    header["dim"] = grid.shape
    header["voxel_size"] = np.linalg.norm(affine[:3, :3], axis=0)
    header["n_scalars"] = n_scalars
    header["scalar_name"] = scalar_names
    header["n_properties"] = n_props
    header["property_name"] = prop_names
    header["vox_to_ras"] = affine
    # The orientation of the affine as stored in float32, as readers compute it;
    # near a tie the float64 orientation can differ.
    stored = header["vox_to_ras"].astype(np.float64)
    codes = Grid(grid.shape, stored).orientation
    header["voxel_order"] = "".join(codes).encode()
    header["n_count"] = n_count
    if unfinished:
        header["reserved"] = UNFINISHED
    header["version"] = 2
    header["hdr_size"] = HEADER_SIZE

    to_world = _trackvis_to_world(
        np.asarray(grid.shape),
        header["voxel_size"].astype(np.float64),
        "".join(codes),
        stored,
        codes,
    )
    return header, np.linalg.inv(to_world)


def _trk_words(tractogram: Tractogram, to_trackvis: np.ndarray) -> np.ndarray:
    """Returns the data of a tractogram as TRK writes it, as little-endian int32 words.

    Each streamline is its point count, then its rows (the point in TrackVis voxel
    mm and its scalars, sorted by name), then its properties, sorted by name.
    """
    voxmm = apply_affine(to_trackvis, tractogram.points)
    n_scalars = sum(k for _, k in _layout(tractogram.point_data))
    n_props = sum(k for _, k in _layout(tractogram.streamline_data))

    step = 3 + n_scalars
    n_points, num_streamlines = tractogram.num_points, tractogram.num_streamlines
    rows = np.empty((n_points, step), dtype=np.float32)
    rows[:, :3] = voxmm
    col = 3
    for name in sorted(tractogram.point_data):
        values = _as_columns(tractogram.point_data[name])
        rows[:, col : col + values.shape[1]] = values
        col += values.shape[1]
    props = np.empty((num_streamlines, n_props), dtype=np.float32)
    col = 0
    for name in sorted(tractogram.streamline_data):
        values = _as_columns(tractogram.streamline_data[name])
        props[:, col : col + values.shape[1]] = values
        col += values.shape[1]

    counts = np.diff(tractogram.offsets)
    out = np.empty(num_streamlines + n_points * step + num_streamlines * n_props, "<i4")
    values_out = out.view("<f4")
    starts = tractogram.offsets[:-1] * step + np.arange(num_streamlines) * (1 + n_props)
    out[starts] = counts
    is_row = np.ones(len(out), dtype=bool)
    is_row[starts] = False
    props_at = (starts + 1 + counts * step)[:, None] + np.arange(n_props)
    is_row[props_at.ravel()] = False
    values_out[is_row] = rows.ravel()
    values_out[props_at.ravel()] = props.ravel()
    return out


def _resolve_grid(path: Path, reference: Grid | None) -> Grid:
    """Returns `reference`, or raises ValueError naming the file if there is none."""
    if reference is None:
        raise ValueError(
            f"Cannot write {path}: TRK needs a reference grid; pass `reference` "
            "(for example volume.grid) or set the tractogram's reference."
        )
    return reference


def write_trk(
    tractogram: Tractogram, path: str | Path, *, reference: Grid | None = None
) -> None:
    """Writes a `Tractogram` as a TRK file, version 2.

    The header's grid comes from the reference. `point_data` is written as scalars
    and `streamline_data` as properties, at most 10 names of each.

    Args:
        tractogram: The tractogram to write.
        path: The `.trk` file; an existing file is overwritten.
        reference: The grid of the image the tractogram belongs to. Defaults to
            the tractogram's own `reference`.

    Raises:
        ValueError: If there is no reference grid, its shape does not fit TRK, or
            there are too many or too long names.

    References:
        Names the scalars and properties as nibabel does.
    """
    path = Path(path)
    grid = _resolve_grid(path, reference or tractogram.reference)
    try:
        header, to_trackvis = _trk_header(
            grid,
            tractogram.point_data,
            tractogram.streamline_data,
            tractogram.num_streamlines,
        )
    except ValueError as e:
        raise ValueError(f"Cannot write {path}: {e}") from e
    with open_file_write(path) as f:
        write_struct(f, header)
        write_payload(f, _trk_words(tractogram, to_trackvis), np.dtype("<i4"))


class TrkWriter(BatchWriter):
    """Writes a TRK file batch by batch, keeping only the current batch in memory.

    The first batch sets the names and sizes of `point_data` and `streamline_data`;
    every later batch must carry the same. A file that was never closed reads back
    with a warning that it is incomplete. The file is the same as the one
    `write_trk` writes for all the streamlines at once.
    """

    def __init__(self, path: str | Path, reference: Grid) -> None:
        """Creates the file, with a header and an empty body.

        Args:
            path: The `.trk` file; an existing file is overwritten. Not `.trk.gz`.
            reference: The grid of the image the tractogram belongs to; the
                batches' own `reference` is not used.

        Raises:
            ValueError: If the file is gzip-compressed, or the grid's shape does
                not fit TRK.
        """
        self.path = Path(path)
        self.num_streamlines = 0
        self._grid = _resolve_grid(self.path, reference)
        self._point_data: dict[str, np.ndarray] = {}
        self._streamline_data: dict[str, np.ndarray] = {}
        self._layout: tuple[tuple[tuple[str, int], ...], ...] | None = None
        try:
            header, self._to_trackvis = _trk_header(
                self._grid, {}, {}, 0, unfinished=True
            )
        except ValueError as e:
            raise ValueError(f"Cannot write {self.path}: {e}") from e
        self._file = open_batch_file(self.path, "TRK")
        write_struct(self._file, header)

    def write(self, tractogram: Tractogram) -> None:
        """Appends the streamlines of a batch, in order.

        Raises:
            ValueError: If the batch's `point_data` or `streamline_data` differ in
                names or sizes from the first batch's, or cannot be written to TRK.
        """
        layout = (_layout(tractogram.point_data), _layout(tractogram.streamline_data))
        if self._layout is None:
            # Empty arrays stand for the data in the header, kept only once the
            # header accepts them, so that a refused first batch changes nothing.
            point_data = {k: v[:0] for k, v in tractogram.point_data.items()}
            streamline_data = {k: v[:0] for k, v in tractogram.streamline_data.items()}
            try:
                header, _ = _trk_header(
                    self._grid, point_data, streamline_data, 0, unfinished=True
                )
            except ValueError as e:
                raise ValueError(f"Cannot write {self.path}: {e}") from e
            self._point_data, self._streamline_data = point_data, streamline_data
            self._file.seek(0)
            write_struct(self._file, header)
            self._file.seek(0, SEEK_END)
            self._layout = layout
        elif layout != self._layout:
            raise ValueError(
                f"Cannot write {self.path}: this batch's point_data and "
                f"streamline_data {layout} differ from the first batch's "
                f"{self._layout}."
            )
        write_payload(
            self._file, _trk_words(tractogram, self._to_trackvis), np.dtype("<i4")
        )
        self.num_streamlines += tractogram.num_streamlines

    def close(self) -> None:
        """Writes the streamline count into the header and closes the file."""
        if self._file.closed:
            return
        header, _ = _trk_header(
            self._grid, self._point_data, self._streamline_data, self.num_streamlines
        )
        self._file.seek(0)
        write_struct(self._file, header)
        self._file.close()


def read_trk_chunks(
    path: str | Path, *, reference: Grid | None = None
) -> Iterator[Tractogram]:
    """Reads an uncompressed TRK file piece by piece, whole streamlines at a time.

    Each piece is a tractogram with the streamlines one chunk of the file
    completes. Joined in order, the pieces are the tractogram `read_trk` returns.
    At least one piece is yielded, empty for a file without streamlines.

    Args:
        path: The `.trk` file, not gzip-compressed.
        reference: As for `read_trk`.

    Returns:
        An iterator over the pieces.

    Raises:
        ValueError: If the file is gzip-compressed (at the call), or is not a TRK
            file Larmor can read (while iterating), as for `read_trk`.
    """
    path = Path(path)
    check_not_gzip(path)
    return _trk_chunks(path, reference)


def _trk_chunks(path: Path, reference: Grid | None) -> Iterator[Tractogram]:
    """Yields the streamlines that each chunk of the TRK file at `path` completes."""
    with path.open("rb") as f:
        header, order = _read_trk_header(path, f)
        layout = _trk_layout(path, header, reference)
        size = os.fstat(f.fileno()).st_size
        found = 0
        carry = np.empty(0, dtype=np.uint8)
        while True:
            data = read_chunk(f, carry)
            if len(data) == len(carry):
                break
            data_start = f.tell() - len(data)
            raw = data[: len(data) - len(data) % 4].view(order + "i4")
            words = raw.astype("=i4", copy=False)
            starts, end = _find_trk_streamlines(
                path, words, layout.step, layout.n_props
            )
            if len(starts) > 0:
                rows, offsets, properties, _ = _gather_trk_streamlines(
                    words, starts, end, layout.step, layout.n_props
                )
                found += len(starts)
                yield _trk_tractogram(layout, rows, offsets, properties)
            carry = data[4 * end :].copy()
            if end < len(words):
                # A streamline that runs past the end of the file is the last one,
                # so the rest of the file is not carried chunk by chunk.
                n = int(words[end])
                words_needed = 1 + n * layout.step + layout.n_props
                if data_start + 4 * (end + words_needed) > size:
                    break

    if len(carry) > 0:
        LarmorLogger.warning(
            "%s ends inside its last streamline; it was dropped.", path
        )
    _check_trk_count(path, header, found)
    if found == 0:
        empty = np.empty(0, dtype=np.int32)
        yield _trk_tractogram(
            layout,
            *_gather_trk_streamlines(
                empty, empty.astype(np.int64), 0, layout.step, layout.n_props
            )[:3],
        )
