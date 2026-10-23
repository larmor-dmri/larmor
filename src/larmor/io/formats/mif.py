"""Reads and writes MIF files, the image format of MRtrix."""

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np

from larmor.io.reader import (
    Progress,
    open_file_read,
    read_payload,
    read_text_header,
)
from larmor.io.writer import open_file_write, write_payload, write_text_header
from larmor.types import GradientTable, Grid, Volume

# The first line of a MIF file.
MRTRIX_MAGIC_MIF = "mrtrix image"
# The header keys every MIF file must have.
REQUIRED_KEYS = ["dim", "vox", "layout", "datatype"]

# The numpy code of each MIF data type, without its byte order.
_MIF_TYPES = {
    "int8": "i1",
    "uint8": "u1",
    "int16": "i2",
    "uint16": "u2",
    "int32": "i4",
    "uint32": "u4",
    "int64": "i8",
    "uint64": "u8",
    "float32": "f4",
    "float64": "f8",
}
# The numpy byte order of each MIF suffix.
ORDER = {"le": "<", "be": ">", "": "="}

# The MIF datatype written for each numpy type, always little-endian.
_MIF_NAMES = {
    "i1": "Int8",
    "u1": "UInt8",
    "i2": "Int16LE",
    "u2": "UInt16LE",
    "i4": "Int32LE",
    "u4": "UInt32LE",
    "i8": "Int64LE",
    "u8": "UInt64LE",
    "f4": "Float32LE",
    "f8": "Float64LE",
}

# The dtype that stands for MIF's Bit type, one bit per voxel packed into bytes.
BIT = np.dtype(np.bool_)


class _MifHeader(NamedTuple):
    """The parts of a MIF header that Larmor uses."""

    dims: tuple[int, ...]
    layout: tuple[int, ...]  # each axis's rank on disk, 0 changing fastest
    backwards: tuple[bool, ...]  # axes stored in reverse ("-" in layout)
    dtype: np.dtype  # BIT for packed bits
    affine: np.ndarray
    scaling: tuple[float, float]  # (offset, scale), value = offset + scale * stored
    data_path: Path  # the file holding the data, the MIF itself or a .dat
    offset: int  # where the data starts in data_path, in bytes
    keys: dict[str, list[str]]  # every key of the header


def _numbers[T](value: str, cast: Callable[[str], T], path: Path, key: str) -> list[T]:
    """Returns the comma-separated numbers of a header value.

    Raises:
        ValueError: If one is not a number of that type.
    """
    try:
        return [cast(v) for v in value.split(",")]
    except ValueError as e:
        raise ValueError(f"{path}: {key} {value!r} is not a list of numbers.") from e


def _get_datatype(dt: str) -> np.dtype:
    """Returns the numpy dtype of a MIF datatype, or BIT for Bit.

    Raises:
        ValueError: If the datatype is complex or unknown.
    """
    t = dt.lower()
    if t == "bit":
        return BIT
    if t.startswith("cfloat"):
        raise ValueError(f"complex data ({dt}) is not supported.")

    if t.endswith(("le", "be")):
        base, suffix = t[:-2], t[-2:]
    else:
        base, suffix = t, ""

    if base not in _MIF_TYPES:
        raise ValueError(
            f"unsupported datatype {dt}; Larmor reads Bit, "
            f"{', '.join(_MIF_TYPES)} (with LE or BE)."
        )
    return np.dtype(ORDER[suffix] + _MIF_TYPES[base])


def _get_affine(transform: list[str], vox_size: list[float], path: Path) -> np.ndarray:
    """Returns the voxel-to-world affine, the transform's columns times the voxel sizes.

    Raises:
        ValueError: If transform is not 3 rows of 4 numbers.
    """
    if not transform:
        mat = np.hstack([np.eye(3), np.zeros((3, 1))])
    else:
        rows = [_numbers(row, float, path, "transform") for row in transform]
        if len(rows) != 3 or any(len(r) != 4 for r in rows):
            raise ValueError(f"{path}: transform must be 3 rows of 4 numbers.")
        mat = np.array(rows)
    aff = np.eye(4)
    # Axes padded for a 1D or 2D image get voxel size 1.
    aff[:3, :3] = mat[:, :3] * (list(vox_size[:3]) + [1.0] * 3)[:3]
    aff[:3, 3] = mat[:, 3]
    return aff


def _get_data_file(entries: list[str], path: Path) -> tuple[Path, int]:
    """Returns the data file and the offset of the data in it.

    Raises:
        ValueError: If there is not exactly one `file` entry, or its offset is not
            a whole number.
    """
    if len(entries) != 1:
        raise ValueError(
            f"{path} has {len(entries)} file entries; Larmor reads exactly one."
        )
    name, _, offset = entries[0].rpartition(" ")
    try:
        start = int(offset)
    except ValueError as e:
        if name == ".":
            raise ValueError(f"{path}: file offset {offset!r} is not a number.") from e
        # "file: name" with no offset; the name may contain spaces.
        name, start = entries[0], 0
    data_path = path if name == "." else path.parent / name
    return data_path, start


def _parse_mif_header(path: Path) -> _MifHeader:
    """Reads and checks the header of a MIF file, without the data.

    Returns:
        The parsed header.

    Raises:
        ValueError: If the file is not a MIF file Larmor can read.
    """
    with open_file_read(path) as f:
        magic = f.readline().decode("utf-8", "replace").strip()
        if magic != MRTRIX_MAGIC_MIF:
            raise ValueError(f"{path} is not a MIF file: no {MRTRIX_MAGIC_MIF!r} line.")
        hdr = read_text_header(f, end_required=not path.name.endswith(".mih"))

    for key in REQUIRED_KEYS + ["file"]:
        if key not in hdr:
            raise ValueError(f"{path} has no {key!r} in its header.")

    dims = _numbers(hdr["dim"][0], int, path, "dim")
    vox_size = _numbers(hdr["vox"][0], float, path, "vox")
    layout_text = hdr["layout"][0].split(",")
    layout = _numbers(hdr["layout"][0], int, path, "layout")
    backwards = tuple(v.strip().startswith("-") for v in layout_text)
    if not len(dims) == len(vox_size) == len(layout):
        raise ValueError(
            f"{path}: dim, vox and layout have {len(dims)}, {len(vox_size)} and "
            f"{len(layout)} entries; they must match."
        )
    if min(dims) < 1:
        raise ValueError(
            f"{path} has the dimension sizes {dims}; each must be 1 or more."
        )
    if sorted(abs(v) for v in layout) != list(range(len(layout))):
        raise ValueError(f"{path}: layout {hdr['layout'][0]} is not a valid order.")

    try:
        dtype = _get_datatype(hdr["datatype"][0])
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e

    affine = _get_affine(hdr.get("transform", []), vox_size, path)
    scaling = (0.0, 1.0)
    if "scaling" in hdr:
        values = _numbers(hdr["scaling"][0], float, path, "scaling")
        if len(values) != 2:
            raise ValueError(f"{path}: scaling must be 2 numbers, offset and scale.")
        scaling = (values[0], values[1])

    data_path, offset = _get_data_file(hdr["file"], path)
    return _MifHeader(
        tuple(dims),
        tuple(abs(v) for v in layout),
        backwards,
        dtype,
        affine,
        scaling,
        data_path,
        offset,
        hdr,
    )


def _undo_layout(
    flat: np.ndarray,
    dims: tuple[int, ...],
    layout: tuple[int, ...],
    backwards: tuple[bool, ...],
) -> np.ndarray:
    """Reorders the flat data from its order on disk to the axes of `dim`."""
    order = sorted(range(len(dims)), key=lambda a: layout[a])
    block = flat.reshape([dims[a] for a in order], order="F")
    data = np.transpose(block, np.argsort(order))
    flipped = tuple(a for a in range(len(dims)) if backwards[a])
    return np.flip(data, axis=flipped) if flipped else data


def read_mif(path: str | Path, *, on_progress: Progress | None = None) -> Volume:
    """Reads a MIF file (`.mif`, `.mif.gz` or `.mih`) into a `Volume`.

    The axes are those of the header's `dim`, whatever the layout on disk. The data
    keeps its type on disk, with Bit as bool; `scaling` is stored on the `Volume`,
    not applied.

    Args:
        path: The MIF file.
        on_progress: Called as the file is read, see `load_volume`.

    Returns:
        Volume: The volume.

    Raises:
        ValueError: If the file is not a MIF file Larmor can read, or is too short.
        ReadCancelledError: If on_progress raises it.
    """
    path = Path(path)
    h = _parse_mif_header(path)
    if len(h.dims) > 4:
        raise ValueError(f"{path} has {len(h.dims)} dimensions; Larmor reads 1 to 4.")
    n = int(np.prod(h.dims))
    with open_file_read(h.data_path) as f:
        if h.dtype == BIT:
            packed = read_payload(
                f, h.offset, np.dtype("u1"), -(-n // 8), on_progress=on_progress
            )
            flat = np.unpackbits(packed, bitorder="big")[:n].astype(np.bool_)
        else:
            flat = read_payload(f, h.offset, h.dtype, n, on_progress=on_progress)
    data = _undo_layout(flat, h.dims, h.layout, h.backwards)
    data = data.reshape(data.shape + (1,) * (3 - data.ndim))
    offset, scale = h.scaling
    try:
        return Volume(data, h.affine, scale, offset)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e


def read_mif_gradients(path: str | Path) -> GradientTable:
    """Reads the gradient table (`dw_scheme`) of a MIF file, from its header only.

    The directions are already in world coordinates, so nothing is converted.

    Args:
        path: The MIF file.

    Returns:
        GradientTable: The table.

    Raises:
        ValueError: If the header has no `dw_scheme`, a line is not 4 numbers, or
            the number of lines differs from the fourth dimension.
    """
    path = Path(path)
    h = _parse_mif_header(path)
    if "dw_scheme" not in h.keys:
        raise ValueError(f"{path} has no gradient table (dw_scheme).")
    rows = [_numbers(line, float, path, "dw_scheme") for line in h.keys["dw_scheme"]]
    if any(len(r) != 4 for r in rows):
        raise ValueError(f"{path}: each dw_scheme line must be 4 numbers, x,y,z,b.")
    if len(h.dims) < 4:
        raise ValueError(
            f"{path} has no fourth dimension; a gradient table needs a 4D image."
        )
    if len(rows) != h.dims[3]:
        raise ValueError(
            f"{path} has {len(rows)} dw_scheme lines but {h.dims[3]} volumes."
        )
    table = np.array(rows)
    return GradientTable(table[:, 3], table[:, :3])


def has_mif_gradients(path: str | Path) -> bool:
    """Checks whether a MIF file's header has a gradient table (`dw_scheme`).

    Only the presence of the table is checked, not whether it can be read.

    Args:
        path: The MIF file.

    Returns:
        True if the header has a `dw_scheme`.

    Raises:
        ValueError: If the file is not a MIF file Larmor can read.
    """
    return "dw_scheme" in _parse_mif_header(Path(path)).keys


def read_mif_header(path: str | Path) -> tuple[Grid, tuple[int, ...]]:
    """Reads only the header of a MIF file, without the data.

    Args:
        path: The `.mif`, `.mif.gz` or `.mih` file.

    Returns:
        The grid, and the full shape of the data: (X, Y, Z) or (X, Y, Z, N).

    Raises:
        ValueError: If the file is not a MIF file Larmor can read.
    """
    path = Path(path)
    h = _parse_mif_header(path)
    if len(h.dims) > 4:
        raise ValueError(f"{path} has {len(h.dims)} dimensions; Larmor reads 1 to 4.")
    shape = h.dims + (1,) * (3 - len(h.dims))
    try:
        grid = Grid((shape[0], shape[1], shape[2]), h.affine)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e
    return grid, shape


def mih_data_file(path: Path) -> Path:
    """Returns the `.dat` file that holds the data of a `.mih` header."""
    return path.with_suffix(".dat")


def _full(x: float) -> str:
    """Returns `x` as text that reads back exactly."""
    return repr(float(x))


def write_mif(
    volume: Volume, path: str | Path, *, gradients: GradientTable | None = None
) -> None:
    """Writes a `Volume`, and optionally its gradient table, as MIF.

    A `.mif` holds header and data, compressed if the name ends in ".gz"; a `.mih`
    holds the header, with the data in a `.dat` of the same name. The data keeps
    its type, with bool as Bit. No other metadata is written.

    Args:
        volume: The volume to write.
        path: The `.mif`, `.mif.gz` or `.mih` file; existing files are
            overwritten.
        gradients: Written into the header as `dw_scheme`, in world coordinates.
            Defaults to None.

    Raises:
        ValueError: If MIF cannot store the data's type (complex, float16), or
            `gradients` does not have one entry per volume.
    """
    path = Path(path)
    data = volume.data
    is_bit = data.dtype == np.bool_
    type_string = data.dtype.str[1:]
    if not is_bit and type_string not in _MIF_NAMES:
        raise ValueError(f"MIF cannot store {data.dtype} data.")
    if gradients is not None:
        if data.ndim != 4:
            raise ValueError(
                f"{path}: a gradient table needs a 4D volume, got shape {data.shape}."
            )
        if data.shape[3] != len(gradients.bvals):
            raise ValueError(
                f"{path}: the gradient table has {len(gradients.bvals)} entries, "
                f"but the volume has {data.shape[3]} volumes."
            )

    vox = np.linalg.norm(volume.affine[:3, :3], axis=0)
    rows = np.hstack([volume.affine[:3, :3] / vox, volume.affine[:3, 3:]])
    header: dict[str, list[str]] = {
        "dim": [",".join(str(n) for n in data.shape)],
        "vox": [",".join(_full(v) for v in list(vox) + [1.0] * (data.ndim - 3))],
        "layout": [",".join(f"+{a}" for a in range(data.ndim))],
        "datatype": ["Bit" if is_bit else _MIF_NAMES[type_string]],
        "transform": [",".join(_full(v) for v in row) for row in rows],
    }
    if (volume.intercept, volume.slope) != (0.0, 1.0):
        header["scaling"] = [f"{_full(volume.intercept)},{_full(volume.slope)}"]
    if gradients is not None:
        header["dw_scheme"] = [
            ",".join(_full(v) for v in (*g, b))
            for g, b in zip(gradients.bvecs, gradients.bvals, strict=True)
        ]

    # MIF stores x fastest, like NIfTI.
    flat = data.ravel(order="F")
    payload = np.packbits(flat, bitorder="big") if is_bit else flat
    disk_dtype = np.dtype("u1") if is_bit else np.dtype("<" + type_string)

    if path.name.endswith(".mih"):
        dat = mih_data_file(path)
        header["file"] = [f"{dat.name} 0"]
        with open_file_write(path) as f:
            write_text_header(f, MRTRIX_MAGIC_MIF, header)
        with open_file_write(dat) as f:
            write_payload(f, payload, disk_dtype)
        return

    # Leaves room for the file line and END, rounded up to a multiple of 4.
    lines = [MRTRIX_MAGIC_MIF] + [f"{k}: {v}" for k, vs in header.items() for v in vs]
    offset = sum(len(line.encode()) + 1 for line in lines) + 24
    offset += -offset % 4
    header["file"] = [f". {offset}"]
    with open_file_write(path) as f:
        write_text_header(f, MRTRIX_MAGIC_MIF, header)
        written = f.tell()
        if written > offset:
            raise ValueError(f"{path}: the header ({written} bytes) overran its space.")
        f.write(bytes(offset - written))
        write_payload(f, payload, disk_dtype)
