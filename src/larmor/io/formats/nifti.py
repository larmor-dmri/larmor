"""Reads and writes NIfTI-1 and NIfTI-2 images."""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np

from larmor._log import LarmorLogger
from larmor.io.reader import (
    Progress,
    ReadFile,
    open_file_read,
    read_payload,
    read_struct,
)
from larmor.io.writer import open_file_write, write_payload, write_struct
from larmor.types import Grid, Volume

# The NIfTI-1 header, 348 bytes.
NIFTI1_HEADER = np.dtype(
    [
        ("sizeof_hdr", "<i4"),  # 348; also tells the byte order
        ("data_type", "S10"),  # unused (ANALYZE 7.5)
        ("db_name", "S18"),  # unused
        ("extents", "<i4"),  # unused
        ("session_error", "<i2"),  # unused
        ("regular", "S1"),  # unused
        ("dim_info", "u1"),  # frequency, phase and slice axes
        ("dim", "<i2", (8,)),  # dim[0] = number of dimensions, then the sizes
        ("intent_p1", "<f4"),
        ("intent_p2", "<f4"),
        ("intent_p3", "<f4"),
        ("intent_code", "<i2"),
        ("datatype", "<i2"),  # code of the data type (2 = uint8, 16 = float32, ...)
        ("bitpix", "<i2"),  # bits per voxel
        ("slice_start", "<i2"),
        ("pixdim", "<f4", (8,)),  # pixdim[0] = qfac, then the voxel sizes
        ("vox_offset", "<f4"),  # where the data starts, in bytes
        ("scl_slope", "<f4"),
        ("scl_inter", "<f4"),
        ("slice_end", "<i2"),
        ("slice_code", "u1"),
        ("xyzt_units", "u1"),  # units of the voxel sizes and of time
        ("cal_max", "<f4"),
        ("cal_min", "<f4"),
        ("slice_duration", "<f4"),
        ("toffset", "<f4"),
        ("glmax", "<i4"),  # unused
        ("glmin", "<i4"),  # unused
        ("descrip", "S80"),
        ("aux_file", "S24"),
        ("qform_code", "<i2"),
        ("sform_code", "<i2"),
        ("quatern_b", "<f4"),  # qform rotation as a quaternion (a is implied)
        ("quatern_c", "<f4"),
        ("quatern_d", "<f4"),
        ("qoffset_x", "<f4"),
        ("qoffset_y", "<f4"),
        ("qoffset_z", "<f4"),
        ("srow_x", "<f4", (4,)),  # sform affine, one row each
        ("srow_y", "<f4", (4,)),
        ("srow_z", "<f4", (4,)),
        ("intent_name", "S16"),
        ("magic", "S4"),  # b"n+1\0" (one file) or b"ni1\0" (.hdr + .img)
    ]
)

# The NIfTI-2 header, 540 bytes.
NIFTI2_HEADER = np.dtype(
    [
        ("sizeof_hdr", "<i4"),  # 540
        ("magic", "S4"),  # b"n+2\0" or b"ni2\0", followed by eol_check
        ("eol_check", "i1", (4,)),  # \r \n \032 \n, to detect text-mode corruption
        ("datatype", "<i2"),
        ("bitpix", "<i2"),
        ("dim", "<i8", (8,)),
        ("intent_p1", "<f8"),
        ("intent_p2", "<f8"),
        ("intent_p3", "<f8"),
        ("pixdim", "<f8", (8,)),
        ("vox_offset", "<i8"),
        ("scl_slope", "<f8"),
        ("scl_inter", "<f8"),
        ("cal_max", "<f8"),
        ("cal_min", "<f8"),
        ("slice_duration", "<f8"),
        ("toffset", "<f8"),
        ("slice_start", "<i8"),
        ("slice_end", "<i8"),
        ("descrip", "S80"),
        ("aux_file", "S24"),
        ("qform_code", "<i4"),
        ("sform_code", "<i4"),
        ("quatern_b", "<f8"),
        ("quatern_c", "<f8"),
        ("quatern_d", "<f8"),
        ("qoffset_x", "<f8"),
        ("qoffset_y", "<f8"),
        ("qoffset_z", "<f8"),
        ("srow_x", "<f8", (4,)),
        ("srow_y", "<f8", (4,)),
        ("srow_z", "<f8", (4,)),
        ("slice_code", "<i4"),
        ("xyzt_units", "<i4"),
        ("intent_code", "<i4"),
        ("intent_name", "S16"),
        ("dim_info", "u1"),
        ("unused_str", "S15"),
    ]
)

# The byte order of a file, as numpy writes it in a dtype.
ByteOrder = Literal["<", ">"]

# The NIfTI datatype codes Larmor reads, and their numpy types without byte order.
TYPE_TABLE: dict[int, str] = {
    2: "u1",
    4: "i2",
    8: "i4",
    16: "f4",
    64: "f8",
    256: "i1",
    512: "u2",
    768: "u4",
    1024: "i8",
    1280: "u8",
}
# The NIfTI datatype code of each numpy type.
DATATYPE_OF_TYPE = {type_string: code for code, type_string in TYPE_TABLE.items()}

# The largest dimension NIfTI-1 can hold (int16); larger images are written as NIfTI-2.
NIFTI1_MAX_DIM = 32767


def _read_nifti_header(f: ReadFile) -> tuple[np.void, ByteOrder]:
    """Reads a NIfTI-1 or NIfTI-2 header from the start of the file.

    The version and the byte order both come from `sizeof_hdr`, the first 4 bytes,
    which are 348 for NIfTI-1 and 540 for NIfTI-2.

    Args:
        f: A file from `open_file_read`, at its start.

    Returns:
        The header record, in the file's byte order, and that byte order.

    Raises:
        ValueError: If the file is not a single-file NIfTI-1 or NIfTI-2.
    """
    first = read_struct(f, np.dtype("<i4"))
    orders: tuple[ByteOrder, ...] = ("<", ">")
    for order in orders:
        sizeof_hdr = int(first.view(first.dtype.newbyteorder(order))[0])
        if sizeof_hdr in (348, 540):
            break
    else:
        raise ValueError(
            f"{f.name} is not a NIfTI file: its first 4 bytes are not 348 or 540 "
            "in either byte order."
        )

    header_dtype = NIFTI1_HEADER if sizeof_hdr == 348 else NIFTI2_HEADER
    f.seek(0)
    hdr = read_struct(f, header_dtype.newbyteorder(order))[0]

    magic = bytes(hdr["magic"])
    expected = b"n+1" if sizeof_hdr == 348 else b"n+2"
    if magic in (b"ni1", b"ni2"):
        raise ValueError(
            f"{f.name} is the header of a .hdr/.img pair, which Larmor does not read; "
            "convert it to a single .nii file."
        )
    if magic != expected:
        raise ValueError(f"{f.name} has the NIfTI magic {magic!r}, not {expected!r}.")
    return hdr, order


def _shape_from_header(hdr: np.void, name: str) -> tuple[int, ...]:
    """Returns the shape of the data, (X, Y, Z) or (X, Y, Z, N).

    Files with fewer than three dimensions get size-1 axes. A 5D file with a single
    time point, such as a displacement field (X, Y, Z, 1, 3), is read as
    (X, Y, Z, 3).

    Args:
        hdr: The header record, or a mapping with its fields.
        name: The file's name, for errors.

    Raises:
        ValueError: If the file does not have 1 to 4 dimensions (or 5 with a single
            time point), or a size is below 1.
    """
    n_dims = int(hdr["dim"][0])
    single_time = n_dims == 5 and int(hdr["dim"][4]) == 1
    if not 1 <= n_dims <= 4 and not single_time:
        raise ValueError(
            f"{name} has {n_dims} dimensions; Larmor reads 1 to 4, and 5 with one "
            "time point."
        )
    sizes = [int(n) for n in hdr["dim"][1 : n_dims + 1]]
    if min(sizes) < 1:
        raise ValueError(
            f"{name} has the dimension sizes {sizes}; each must be 1 or more."
        )
    if single_time:
        # The size-1 time axis holds no data, so dropping it keeps the layout.
        return (sizes[0], sizes[1], sizes[2], sizes[4])
    return tuple(sizes + [1] * (3 - n_dims))


def affine_from_header(hdr: np.void | Mapping[str, np.ndarray]) -> np.ndarray:
    """Returns the voxel-to-world affine (4, 4) in mm.

    The affine is the sform if `sform_code > 0`, otherwise the qform if
    `qform_code > 0`, otherwise `diag(pixdim[1], pixdim[2], pixdim[3], 1)`. In the
    qform a negative voxel size is used by its absolute value, with a warning, since
    only qfac flips an axis. `xyzt_units` is ignored and lengths are taken as mm.

    Args:
        hdr: The header record, or a mapping with its fields.

    References:
        Follows MRtrix3 when rounding leaves almost nothing for the quaternion's a.
    """
    if int(hdr["sform_code"]) > 0:
        affine = np.eye(4)
        affine[0] = hdr["srow_x"]
        affine[1] = hdr["srow_y"]
        affine[2] = hdr["srow_z"]
        return affine

    zooms = np.array(hdr["pixdim"][1:4], dtype=np.float64)
    if int(hdr["qform_code"]) <= 0:
        return np.diag([zooms[0], zooms[1], zooms[2], 1.0])

    # a follows from a^2 + b^2 + c^2 + d^2 = 1. When rounding leaves almost nothing
    # for a, a is 0 and (b, c, d) is normalised, so the rotation stays orthogonal.
    bcd = np.array(
        [hdr["quatern_b"], hdr["quatern_c"], hdr["quatern_d"]], dtype=np.float64
    )
    rest = 1.0 - float(bcd @ bcd)
    if rest < 1e-7:
        a = 0.0
        bcd /= np.linalg.norm(bcd)
    else:
        a = math.sqrt(rest)
    b, c, d = bcd
    rotation = np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
        ]
    )

    if (zooms < 0).any():
        LarmorLogger.warning(
            "The NIfTI header has negative voxel sizes %s; the qform uses their "
            "absolute values.",
            zooms.tolist(),
        )
        zooms = np.abs(zooms)

    # qfac, in pixdim[0], flips the third voxel axis when negative (0 counts as 1).
    if float(hdr["pixdim"][0]) < 0:
        zooms[2] = -zooms[2]

    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag(zooms)
    affine[:3, 3] = [hdr["qoffset_x"], hdr["qoffset_y"], hdr["qoffset_z"]]
    return affine


def _read_nifti_payload(
    f: ReadFile,
    hdr: np.void,
    order: ByteOrder,
    shape: tuple[int, ...],
    on_progress: Progress | None,
) -> np.ndarray:
    """Reads the data at `vox_offset`, in its type on disk and native byte order.

    Args:
        f: The file the header was read from.
        hdr: The header record, or a mapping with its fields.
        order: The file's byte order.
        shape: The data's shape, from `_shape_from_header`.
        on_progress: Called as the payload is read, see `read_payload`.

    Returns:
        The data, read-only, shape `shape`.

    Raises:
        ValueError: If the datatype is not one Larmor reads, `bitpix` does not match
            it, `vox_offset` points inside the header, or the file is too short.
    """
    code = int(hdr["datatype"])
    if code not in TYPE_TABLE:
        raise ValueError(
            f"{f.name} has the NIfTI datatype {code}, which Larmor does not read "
            f"(it reads {sorted(TYPE_TABLE)})."
        )
    dtype = np.dtype(TYPE_TABLE[code]).newbyteorder(order)

    bitpix = int(hdr["bitpix"])
    if bitpix != dtype.itemsize * 8:
        raise ValueError(
            f"{f.name} has bitpix {bitpix}, but its datatype {code} has "
            f"{dtype.itemsize * 8} bits per voxel."
        )

    if not math.isfinite(float(hdr["vox_offset"])):
        raise ValueError(f"{f.name} has the vox_offset {hdr['vox_offset']}.")
    offset = int(hdr["vox_offset"])
    if offset < hdr.dtype.itemsize:
        raise ValueError(
            f"{f.name} has vox_offset {offset}, inside its "
            f"{hdr.dtype.itemsize}-byte header."
        )

    data = read_payload(f, offset, dtype, math.prod(shape), on_progress=on_progress)
    # NIfTI stores x fastest.
    return data.reshape(shape, order="F")


def read_nifti(path: str | Path, *, on_progress: Progress | None = None) -> Volume:
    """Reads a NIfTI-1 or NIfTI-2 file, plain or gzip, into a `Volume`.

    The data keeps its type on disk; `scl_slope` and `scl_inter` are stored on the
    `Volume`, not applied. The sform and qform codes are kept in
    `Volume.xform_codes`, with an unknown code (0) taken as 1.

    Args:
        path: The `.nii` or `.nii.gz` file.
        on_progress: Called as the file is read, see `load_volume`.

    Returns:
        Volume: The image, with its scaling not applied.

    Raises:
        ValueError: If the file is not a NIfTI file Larmor can read, or is too short.
        ReadCancelledError: If on_progress raises it.
    """
    path = Path(path)
    with open_file_read(path) as f:
        hdr, order = _read_nifti_header(f)
        shape = _shape_from_header(hdr, f.name)
        affine = affine_from_header(hdr)
        data = _read_nifti_payload(f, hdr, order, shape, on_progress=on_progress)
    # Volume's errors do not name the file, so they are caught to add it.
    codes = _xform_codes(hdr)
    try:
        return Volume(
            data,
            affine,
            float(hdr["scl_slope"]),
            float(hdr["scl_inter"]),
            xform_codes=codes,
        )
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e


def _xform_codes(hdr: np.void) -> tuple[int, int]:
    """Returns the sform and qform codes of a header, with 0 (unknown) as 1."""
    return max(int(hdr["sform_code"]), 1), max(int(hdr["qform_code"]), 1)


def read_nifti_header(path: str | Path) -> tuple[Grid, tuple[int, ...]]:
    """Reads only the header of a NIfTI file, without the data.

    Args:
        path: The `.nii` or `.nii.gz` file.

    Returns:
        The grid, and the full shape of the data: (X, Y, Z) or (X, Y, Z, N).

    Raises:
        ValueError: If the file is not a NIfTI file Larmor can read.
    """
    path = Path(path)
    with open_file_read(path) as f:
        hdr, _ = _read_nifti_header(f)
        shape = _shape_from_header(hdr, f.name)
    try:
        grid = Grid(
            (shape[0], shape[1], shape[2]),
            affine_from_header(hdr),
            _xform_codes(hdr),
        )
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e
    return grid, shape


def _quaternion_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Returns the quaternion (a, b, c, d) of a rotation matrix, with a >= 0.

    Uses the eigenvector method of Bar-Itzhack (2000), which also works for
    180-degree rotations such as an LPS image.

    References:
        Follows nibabel's `mat2quat`.
    """
    (xx, yx, zx), (xy, yy, zy), (xz, yz, zz) = rotation
    k = (
        np.array(
            [
                [xx - yy - zz, 0.0, 0.0, 0.0],
                [yx + xy, yy - xx - zz, 0.0, 0.0],
                [zx + xz, zy + yz, zz - xx - yy, 0.0],
                [yz - zy, zx - xz, xy - yx, xx + yy + zz],
            ]
        )
        / 3.0
    )
    values, vectors = np.linalg.eigh(k)
    quaternion = vectors[[3, 0, 1, 2], np.argmax(values)]
    return -quaternion if quaternion[0] < 0 else quaternion


def _qform_from_affine(affine: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Returns the qform of an affine, as (b, c, d), qfac and the voxel sizes.

    The qform holds a rotation only, so an affine with shear gets its closest
    rotation; a reflection is stored as qfac = -1, which flips the third axis.
    """
    zooms = np.linalg.norm(affine[:3, :3], axis=0)
    u, _, vt = np.linalg.svd(affine[:3, :3] / zooms)
    rotation = u @ vt
    qfac = 1.0
    if np.linalg.det(rotation) < 0:
        qfac = -1.0
        rotation[:, 2] = -rotation[:, 2]
    return _quaternion_from_rotation(rotation)[1:], qfac, zooms


def _nifti_header(volume: Volume, datatype: int, itemsize: int) -> np.ndarray:
    """Builds the header of `volume`: NIfTI-1, or NIfTI-2 if a dimension is too large.

    The affine goes into the sform exactly and into the qform as its closest
    rotation, with the volume's `xform_codes`. Every other field is zero.

    Returns:
        The header, as a structured array.
    """
    shape = volume.data.shape
    nifti2 = max(shape) > NIFTI1_MAX_DIM
    header = np.zeros(1, NIFTI2_HEADER if nifti2 else NIFTI1_HEADER)
    header["sizeof_hdr"] = header.dtype.itemsize
    header["magic"] = b"n+2" if nifti2 else b"n+1"
    if nifti2:
        header["eol_check"] = [13, 10, 26, 10]

    header["dim"] = [len(shape), *shape, *[1] * (7 - len(shape))]
    header["datatype"] = datatype
    header["bitpix"] = itemsize * 8
    # The 4-byte extension flag, "no extensions", sits between header and data.
    header["vox_offset"] = header.dtype.itemsize + 4
    header["scl_slope"] = volume.slope
    header["scl_inter"] = volume.intercept
    header["xyzt_units"] = 10  # mm and seconds

    affine = volume.affine
    quaternion, qfac, zooms = _qform_from_affine(affine)
    header["pixdim"] = [qfac, *zooms, 1.0, 1.0, 1.0, 1.0]
    sform_code, qform_code = volume.xform_codes
    header["qform_code"] = qform_code
    header["quatern_b"], header["quatern_c"], header["quatern_d"] = quaternion
    header["qoffset_x"], header["qoffset_y"], header["qoffset_z"] = affine[:3, 3]
    header["sform_code"] = sform_code
    header["srow_x"] = affine[0]
    header["srow_y"] = affine[1]
    header["srow_z"] = affine[2]
    return header


def write_nifti(volume: Volume, path: str | Path) -> None:
    """Writes a `Volume` as a NIfTI file, compressed if the name ends in ".gz".

    The file is NIfTI-1, or NIfTI-2 when a dimension is larger than 32767. The data
    keeps its type, with bool as uint8, and the header holds the volume's slope,
    intercept and sform and qform codes. No other metadata is written.

    Args:
        volume: The volume to write.
        path: The `.nii` or `.nii.gz` file; an existing file is overwritten.

    Raises:
        ValueError: If NIfTI cannot store the data's type (for example float16).

    References:
        Follows MRtrix3 in when to write NIfTI-2 and in the units written.
    """
    data = volume.data.view(np.uint8) if volume.data.dtype == np.bool_ else volume.data
    type_string = data.dtype.str[1:]
    if type_string not in DATATYPE_OF_TYPE:
        raise ValueError(
            f"NIfTI cannot store {data.dtype} data; convert it to one of "
            f"{sorted(DATATYPE_OF_TYPE)} first."
        )
    path = Path(path)
    disk_dtype = np.dtype(type_string).newbyteorder("<")
    header = _nifti_header(volume, DATATYPE_OF_TYPE[type_string], disk_dtype.itemsize)

    with open_file_write(path) as f:
        write_struct(f, header)
        f.write(bytes(4))
        # NIfTI stores x fastest.
        write_payload(f, data.ravel(order="F"), disk_dtype)
