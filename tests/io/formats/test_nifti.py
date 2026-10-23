import gzip
import io
import logging
import math
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest
from nibabel.nifti1 import Nifti1Header
from nibabel.nifti2 import Nifti2Header

from larmor.io.formats import read_nifti, read_nifti_header, write_nifti
from larmor.types import Grid, Volume

NIBABEL_DATA = Path(nib.__file__).parent / "tests" / "data"

DTYPES = ["u1", "i1", "i2", "u2", "i4", "u4", "i8", "u8", "f4", "f8"]

GZ = pytest.mark.parametrize("gz", [False, True], ids=["plain", "gzip"])
ORDER = pytest.mark.parametrize("order", ["<", ">"], ids=["little", "big"])

# Non-cubic, so that a C-order or transposed read gives other values.
SHAPE = (2, 3, 4)

# The header stores the affine as float32: entries up to ~25 mm round to ~2e-6.
F32_ATOL = 1e-5


def _rotation(z: float, x: float) -> np.ndarray:
    cz, sz, cx, sx = math.cos(z), math.sin(z), math.cos(x), math.sin(x)
    about_z = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    about_x = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]])
    rotation: np.ndarray = about_z @ about_x
    return rotation


def _affine(linear: np.ndarray, offset: tuple[float, float, float]) -> np.ndarray:
    affine = np.eye(4)
    affine[:3, :3] = linear[:3, :3]
    affine[:3, 3] = offset
    return affine


OFFSET = (-10.5, 20.25, 3.0)
# Pure rotations times voxel sizes: the qform holds each of these exactly.
ROTATIONS = {
    "oblique": _affine(_rotation(0.3, 0.2) @ np.diag([2, 3, 2.5, 1]), OFFSET),
    "reflected": _affine(_rotation(0.3, 0.2) @ np.diag([2, 3, -2.5, 1]), OFFSET),
    # x and y both flipped: a 180-degree rotation about z.
    "lps": _affine(np.diag([-2.0, -3.0, 2.5]), OFFSET),
}
SHEARED = _affine(
    _rotation(0.3, 0.2)[:3, :3] @ np.array([[2, 0.3, 0], [0, 3, 0.2], [0, 0, 2.5]]),
    OFFSET,
)
AFFINE = ROTATIONS["oblique"]


def _values(dtype: str, shape: tuple[int, ...] = SHAPE) -> np.ndarray:
    rng = np.random.default_rng(0)
    kind = np.dtype(dtype)
    if kind.kind == "f":
        return (rng.standard_normal(shape) * 1e3).astype(kind)
    info = np.iinfo(kind)
    values = rng.integers(info.min, info.max, shape, dtype=kind, endpoint=True)
    values.flat[0], values.flat[-1] = info.min, info.max
    return values


def _header(
    data: np.ndarray,
    version: int = 1,
    order: str = "<",
    affine: np.ndarray = AFFINE,
    codes: tuple[int, int] = (1, 1),
    scaling: tuple[float, float] = (math.nan, math.nan),
) -> Nifti1Header:
    if version == 1:
        header = Nifti1Header(endianness=order)
    else:
        header = Nifti2Header(endianness=order)
    header.set_data_dtype(data.dtype)
    header.set_data_shape(data.shape)
    header.set_sform(affine, code=codes[0])
    header.set_qform(affine, code=codes[1])
    header.set_data_offset(header.single_vox_offset)
    # Set directly: nibabel's set_slope_inter refuses a slope of 0.
    header["scl_slope"], header["scl_inter"] = scaling
    return header


def _raw(header: Nifti1Header, data: np.ndarray) -> bytes:
    block: bytes = header.binaryblock
    padding = bytes(int(header["vox_offset"]) - len(block))
    payload = data.astype(header.get_data_dtype()).tobytes(order="F")
    return block + padding + payload


def _save(path: Path, raw: bytes) -> Path:
    path.write_bytes(gzip.compress(raw) if path.name.endswith(".gz") else raw)
    return path


def _nib_file(path: Path, data: np.ndarray, header: Nifti1Header | None = None) -> Path:
    return _save(path, _raw(_header(data) if header is None else header, data))


def _name(gz: bool) -> str:
    return "a.nii.gz" if gz else "a.nii"


def _assert_same_as_nibabel(vol: Volume, path: Path) -> None:
    img: Any = nib.load(path)
    unscaled = np.asarray(img.dataobj.get_unscaled())
    assert vol.data.dtype == img.get_data_dtype().newbyteorder("=")
    assert vol.data.dtype.isnative
    assert not vol.data.flags.writeable
    assert vol.data.shape == unscaled.shape
    np.testing.assert_array_equal(vol.data, unscaled)
    assert (vol.slope, vol.intercept) == (img.dataobj.slope, img.dataobj.inter)
    np.testing.assert_allclose(vol.affine, img.affine, rtol=0, atol=1e-12)


# Reading


@pytest.mark.parametrize("dtype", DTYPES)
@ORDER
def test_every_datatype_and_byte_order_reads_as_nibabel(
    tmp_path: Path, dtype: str, order: str
) -> None:
    data = _values(dtype)
    path = _nib_file(tmp_path / "a.nii", data, _header(data, order=order))
    vol = read_nifti(path)
    _assert_same_as_nibabel(vol, path)
    np.testing.assert_array_equal(vol.data, data, strict=True)


@pytest.mark.parametrize("version", [1, 2], ids=["nifti1", "nifti2"])
@ORDER
@GZ
def test_both_versions_plain_and_gzip_read_as_nibabel_with_scaling(
    tmp_path: Path, version: int, order: str, gz: bool
) -> None:
    data = _values("i2", (*SHAPE, 5))
    header = _header(data, version=version, order=order, scaling=(2.5, -1.0))
    path = _nib_file(tmp_path / _name(gz), data, header)
    vol = read_nifti(path)
    _assert_same_as_nibabel(vol, path)
    assert (vol.slope, vol.intercept) == (2.5, -1.0)


@pytest.mark.parametrize(
    ("scaling", "expected"),
    [((0.5, -3.0), (0.5, -3.0)), ((0.0, 5.0), (1.0, 0.0)), ((math.nan,) * 2, (1, 0))],
    ids=["scaled", "slope-0", "slope-nan"],
)
def test_scaling_is_stored_not_applied(
    tmp_path: Path, scaling: tuple[float, float], expected: tuple[float, float]
) -> None:
    data = _values("i2")
    path = _nib_file(tmp_path / "a.nii", data, _header(data, scaling=scaling))
    vol = read_nifti(path)
    _assert_same_as_nibabel(vol, path)
    assert (vol.slope, vol.intercept) == expected
    np.testing.assert_array_equal(vol.data, data, strict=True)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((5,), (5, 1, 1)),
        ((5, 3), (5, 3, 1)),
        (SHAPE, SHAPE),
        ((*SHAPE, 2), (*SHAPE, 2)),
    ],
    ids=["1d", "2d", "3d", "4d"],
)
def test_fewer_than_three_dimensions_get_size_one_axes(
    tmp_path: Path, shape: tuple[int, ...], expected: tuple[int, ...]
) -> None:
    data = _values("f4", shape)
    vol = read_nifti(_nib_file(tmp_path / "a.nii", data))
    assert vol.data.shape == expected
    np.testing.assert_array_equal(vol.data, data.reshape(expected), strict=True)


@pytest.mark.parametrize(
    ("codes", "source"),
    [
        ((1, 1), "sform"),
        ((2, 0), "sform"),
        ((4, 3), "sform"),
        ((0, 1), "qform"),
        ((0, 2), "qform"),
        ((0, 0), "pixdim"),
    ],
)
def test_affine_is_sform_then_qform_then_voxel_sizes(
    tmp_path: Path, codes: tuple[int, int], source: str
) -> None:
    data = _values("u1")
    header = _header(data, affine=ROTATIONS["oblique"], codes=codes)
    # A different sform, so that the test sees which of the two was read.
    header.set_sform(SHEARED, code=codes[0])
    vol = read_nifti(_save(tmp_path / "a.nii", _raw(header, data)))
    expected = {
        "sform": SHEARED,
        "qform": ROTATIONS["oblique"],
        "pixdim": np.diag([2.0, 3.0, 2.5, 1.0]),
    }[source]
    np.testing.assert_allclose(vol.affine, expected, rtol=0, atol=F32_ATOL)


def test_qfac_zero_counts_as_one(tmp_path: Path) -> None:
    # The standard reads a qfac of 0 as 1; nibabel's set_qform only writes 1 or -1.
    data = _values("u1")
    header = _header(data, affine=ROTATIONS["oblique"], codes=(0, 1))
    pixdim = header["pixdim"].copy()
    pixdim[0] = 0.0
    header["pixdim"] = pixdim
    vol = read_nifti(_save(tmp_path / "a.nii", _raw(header, data)))
    np.testing.assert_allclose(vol.affine, ROTATIONS["oblique"], rtol=0, atol=F32_ATOL)


@pytest.mark.parametrize("affine", ROTATIONS.values(), ids=ROTATIONS.keys())
def test_qform_rotation_reads_as_nibabel(tmp_path: Path, affine: np.ndarray) -> None:
    data = _values("u1")
    path = _nib_file(
        tmp_path / "a.nii", data, _header(data, affine=affine, codes=(0, 1))
    )
    vol = read_nifti(path)
    img: Any = nib.load(path)
    np.testing.assert_allclose(vol.affine, img.affine, rtol=0, atol=1e-12)
    np.testing.assert_allclose(vol.affine, affine, rtol=0, atol=F32_ATOL)


@pytest.mark.parametrize("units", ["meter", "micron"])
# Only the qform and the voxel-size affine use pixdim, which the units apply to.
@pytest.mark.parametrize("codes", [(0, 1), (0, 0)], ids=["qform", "pixdim"])
def test_length_units_are_ignored(
    tmp_path: Path, units: str, codes: tuple[int, int]
) -> None:
    data = _values("u1")
    header = _header(data, codes=codes)
    plain = read_nifti(_save(tmp_path / "mm.nii", _raw(header, data)))
    header.set_xyzt_units(xyz=units)
    vol = read_nifti(_save(tmp_path / "a.nii", _raw(header, data)))
    np.testing.assert_array_equal(vol.affine, plain.affine)
    assert vol.voxel_sizes == pytest.approx([2.0, 3.0, 2.5])


@pytest.mark.parametrize("version", [1, 2], ids=["nifti1", "nifti2"])
@GZ
def test_grid_is_read_from_header_only(tmp_path: Path, version: int, gz: bool) -> None:
    data = _values("f4", (*SHAPE, 2))
    header = _header(data, version=version)
    full = read_nifti(_save(tmp_path / _name(gz), _raw(header, data)))
    cut = _save(
        tmp_path / f"cut{_name(gz)}", _raw(header, data)[: len(header.binaryblock)]
    )
    grid, _ = read_nifti_header(cut)
    assert isinstance(grid, Grid)
    assert grid.shape == full.data.shape[:3] == SHAPE
    np.testing.assert_array_equal(grid.affine, full.affine)
    with pytest.raises(ValueError, match=cut.name):
        read_nifti(cut)


# Files written by other tools, shipped with nibabel.
REAL_FILES = [
    "anatomical.nii",
    "example4d.nii.gz",
    "example_nifti2.nii.gz",
    "functional.nii",
    "standard.nii.gz",
    "reoriented_anat_moved.nii",
    "resampled_anat_moved.nii",
]


@pytest.mark.parametrize("name", REAL_FILES)
def test_real_file_reads_as_nibabel(name: str) -> None:
    path = NIBABEL_DATA / name
    vol = read_nifti(path)
    _assert_same_as_nibabel(vol, path)
    grid, _ = read_nifti_header(path)
    assert grid.shape == vol.data.shape[:3]
    np.testing.assert_array_equal(grid.affine, vol.affine)


def test_real_file_with_more_than_four_dimensions_is_refused() -> None:
    # A CIFTI-2 connectivity matrix: a NIfTI-2 file with six dimensions.
    with pytest.raises(ValueError, match="row_major.dconn.nii"):
        read_nifti(NIBABEL_DATA / "row_major.dconn.nii")


# Errors

Edit = Callable[[Nifti1Header], None]


def _set(**fields: object) -> Edit:
    def edit(header: Nifti1Header) -> None:
        for key, value in fields.items():
            header[key] = value

    return edit


# Each error names the file and says what is wrong with it.
HEADER_ERRORS = [
    pytest.param(1, "<", _set(sizeof_hdr=349), "not a NIfTI", id="sizeof-little"),
    pytest.param(1, ">", _set(sizeof_hdr=349), "not a NIfTI", id="sizeof-big"),
    pytest.param(1, "<", _set(magic=b"ni1"), ".hdr/.img", id="pair-nifti1"),
    pytest.param(2, "<", _set(magic=b"ni2"), ".hdr/.img", id="pair-nifti2"),
    pytest.param(1, "<", _set(magic=b"n+2"), "magic", id="magic-nifti1"),
    pytest.param(2, ">", _set(magic=b"abc"), "magic", id="magic-nifti2"),
    pytest.param(1, "<", _set(dim=[5, 2, 3, 4, 2, 3, 1, 1]), "dimensions", id="5d"),
    pytest.param(1, "<", _set(dim=[3, -2, 3, 4, 1, 1, 1, 1]), "sizes", id="size-0"),
    pytest.param(
        1,
        "<",
        _set(sform_code=0, qform_code=0, pixdim=[1, 0, 0, 0, 1, 1, 1, 1]),
        "zero length",
        id="pixdim-0",
    ),
    pytest.param(1, "<", _set(srow_x=[math.nan, 0, 0, 0]), "NaN", id="sform-nan"),
]


@pytest.mark.parametrize(("version", "order", "edit", "says"), HEADER_ERRORS)
@pytest.mark.parametrize("read", [read_nifti, read_nifti_header])
def test_bad_header_raises_naming_file(
    tmp_path: Path,
    version: int,
    order: str,
    edit: Edit,
    says: str,
    read: Callable[[Path], object],
) -> None:
    data = _values("i2")
    header = _header(data, version=version, order=order)
    edit(header)
    path = _save(tmp_path / "bad.nii", _raw(header, data))
    with pytest.raises(ValueError, match="bad.nii") as err:
        read(path)
    assert says in str(err.value)


def _keep(raw: bytes) -> bytes:
    return raw


PAYLOAD_ERRORS = [
    pytest.param(_set(datatype=128, bitpix=24), _keep, "a.nii", id="rgb"),
    pytest.param(_set(bitpix=8), _keep, "a.nii", id="bitpix"),
    pytest.param(_set(vox_offset=344), _keep, "a.nii", id="vox-offset"),
    pytest.param(_set(vox_offset=math.nan), _keep, "a.nii", id="vox-offset-nan"),
    pytest.param(
        _set(scl_slope=2.0, scl_inter=math.inf), _keep, "a.nii", id="inter-inf"
    ),
    pytest.param(_set(), lambda raw: raw[:-1], "a.nii", id="short"),
    pytest.param(_set(), lambda raw: raw[: len(raw) // 2], "a.nii.gz", id="cut-gzip"),
]


@pytest.mark.parametrize(("edit", "cut", "name"), PAYLOAD_ERRORS)
def test_bad_payload_raises_naming_file(
    tmp_path: Path, edit: Edit, cut: Callable[[bytes], bytes], name: str
) -> None:
    data = _values("i2")
    header = _header(data)
    payload = _raw(header, data)[len(header.binaryblock) :]
    edit(header)
    raw = header.binaryblock + payload
    path = tmp_path / name
    path.write_bytes(cut(gzip.compress(raw) if name.endswith(".gz") else raw))
    with pytest.raises(ValueError, match=name):
        read_nifti(path)


# Writing


def _volume(
    dtype: str = "f4",
    shape: tuple[int, ...] = SHAPE,
    affine: np.ndarray = AFFINE,
    scaling: tuple[float, float] = (0.5, -3.0),
) -> Volume:
    return Volume(_values(dtype, shape), affine, *scaling)


@pytest.mark.parametrize("dtype", DTYPES)
@GZ
def test_written_file_reads_back_the_same(tmp_path: Path, dtype: str, gz: bool) -> None:
    vol = _volume(dtype, (*SHAPE, 2))
    path = tmp_path / _name(gz)
    write_nifti(vol, path)
    back = read_nifti(path)
    np.testing.assert_array_equal(back.data, vol.data, strict=True)
    assert (back.slope, back.intercept) == (vol.slope, vol.intercept)
    np.testing.assert_allclose(back.affine, vol.affine, rtol=0, atol=F32_ATOL)


@pytest.mark.parametrize("dtype", DTYPES)
def test_written_file_reads_in_nibabel_with_same_data_and_scaling(
    tmp_path: Path, dtype: str
) -> None:
    vol = _volume(dtype)
    write_nifti(vol, tmp_path / "a.nii")
    assert isinstance(nib.load(tmp_path / "a.nii"), nib.Nifti1Image)
    img: Any = nib.load(tmp_path / "a.nii")
    np.testing.assert_array_equal(img.dataobj.get_unscaled(), vol.data)
    assert img.get_data_dtype() == np.dtype(dtype).newbyteorder("<")
    assert (img.dataobj.slope, img.dataobj.inter) == (vol.slope, vol.intercept)


@pytest.mark.parametrize(
    "affine", [*ROTATIONS.values(), SHEARED], ids=[*ROTATIONS.keys(), "sheared"]
)
@GZ
def test_written_sform_is_affine_and_qform_its_rotation(
    tmp_path: Path, affine: np.ndarray, gz: bool
) -> None:
    write_nifti(_volume(affine=affine), tmp_path / _name(gz))
    header: Any = nib.load(tmp_path / _name(gz)).header
    np.testing.assert_allclose(header.get_sform(), affine, rtol=0, atol=F32_ATOL)
    # nibabel's own qform of the affine: the closest rotation when there is shear.
    reference = Nifti1Header()
    reference.set_qform(affine)
    np.testing.assert_allclose(
        header.get_qform(), reference.get_qform(), rtol=0, atol=F32_ATOL
    )
    if affine is not SHEARED:
        np.testing.assert_allclose(header.get_qform(), affine, rtol=0, atol=F32_ATOL)


def test_header_has_scanner_codes_and_data_after_extension_flag(
    tmp_path: Path,
) -> None:
    write_nifti(_volume(), tmp_path / "a.nii")
    raw = (tmp_path / "a.nii").read_bytes()
    header = Nifti1Header.from_fileobj(io.BytesIO(raw))
    assert header["sizeof_hdr"] == 348
    assert header["magic"] == b"n+1"
    assert (header["sform_code"], header["qform_code"]) == (1, 1)
    assert header["vox_offset"] == 352
    assert raw[348:352] == bytes(4)
    assert len(raw) == 352 + math.prod(SHAPE) * 4


@pytest.mark.parametrize("dtype", [">i2", ">f8"])
def test_big_endian_data_is_written_little_endian(tmp_path: Path, dtype: str) -> None:
    data = _values(dtype[1:]).astype(dtype)
    write_nifti(Volume(data, AFFINE), tmp_path / "a.nii")
    raw = (tmp_path / "a.nii").read_bytes()
    assert raw[:4] == struct.pack("<i", 348)
    little = np.dtype(dtype).newbyteorder("<")
    assert raw[352:] == data.astype(little).tobytes(order="F")
    back = read_nifti(tmp_path / "a.nii")
    np.testing.assert_array_equal(back.data, data)


def test_bool_mask_is_written_as_uint8(tmp_path: Path) -> None:
    mask = _values("u1") > 127
    write_nifti(Volume(mask, AFFINE), tmp_path / "a.nii")
    back = read_nifti(tmp_path / "a.nii")
    np.testing.assert_array_equal(back.data, mask.astype(np.uint8), strict=True)
    img: Any = nib.load(tmp_path / "a.nii")
    assert img.get_data_dtype() == np.uint8


def test_float16_raises_before_writing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="float16"):
        write_nifti(
            Volume(_values("f4").astype(np.float16), AFFINE), tmp_path / "a.nii"
        )
    assert not (tmp_path / "a.nii").exists()


@pytest.mark.parametrize(
    ("size", "image", "offset"),
    [(32767, "Nifti1Image", 352), (32768, "Nifti2Image", 544)],
    ids=["nifti1-max", "nifti2"],
)
@pytest.mark.parametrize("axis", [0, 3])
def test_dimension_above_int16_is_written_as_nifti2(
    tmp_path: Path, size: int, image: str, offset: int, axis: int
) -> None:
    shape = [1, 1, 1, 1]
    shape[axis] = size
    vol = Volume(np.arange(size, dtype=np.uint8).reshape(shape), np.eye(4))
    path = tmp_path / "a.nii"
    write_nifti(vol, path)
    img: Any = nib.load(path)
    assert type(img).__name__ == image
    assert img.shape == tuple(shape)
    header = type(img.header).from_fileobj(io.BytesIO(path.read_bytes()))
    assert header["vox_offset"] == offset
    np.testing.assert_array_equal(read_nifti(path).data, vol.data, strict=True)


def test_gzip_output_does_not_depend_on_name(tmp_path: Path) -> None:
    vol = _volume()
    write_nifti(vol, tmp_path / "a.nii.gz")
    write_nifti(vol, tmp_path / "other_name.nii.gz")
    raw = (tmp_path / "a.nii.gz").read_bytes()
    assert raw[:2] == b"\x1f\x8b"
    assert raw == (tmp_path / "other_name.nii.gz").read_bytes()


@pytest.mark.parametrize("gz", [False, True])
def test_paths_can_be_strings(tmp_path: Path, gz: bool) -> None:
    vol = _volume()
    path = str(tmp_path / _name(gz))
    write_nifti(vol, path)
    np.testing.assert_array_equal(read_nifti(path).data, vol.data, strict=True)
    assert read_nifti_header(path)[1] == vol.data.shape


def test_5d_file_with_one_time_point_reads_as_4d(tmp_path: Path) -> None:
    # ITK writes displacement fields as (X, Y, Z, 1, 3).
    data = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 1, 3)
    path = tmp_path / "warp.nii"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    volume = read_nifti(path)
    np.testing.assert_array_equal(volume.data, data[:, :, :, 0, :])
    assert read_nifti_header(path)[1] == (2, 3, 4, 3)


def test_sform_and_qform_codes_survive_a_round_trip(tmp_path: Path) -> None:
    data = _values("u1")
    path = _nib_file(tmp_path / "a.nii", data, _header(data, codes=(4, 3)))
    write_nifti(read_nifti(path).with_data(data), tmp_path / "b.nii")
    header = Nifti1Header.from_fileobj(io.BytesIO((tmp_path / "b.nii").read_bytes()))
    assert (header["sform_code"], header["qform_code"]) == (4, 3)


def test_header_grid_keeps_the_xform_codes(tmp_path: Path) -> None:
    data = _values("u1")
    path = _nib_file(tmp_path / "a.nii", data, _header(data, codes=(4, 3)))
    grid, _ = read_nifti_header(path)
    assert grid.xform_codes == (4, 3)
    assert read_nifti(path).grid.xform_codes == (4, 3)
    assert grid.canonical().xform_codes == (4, 3)


def test_negative_pixdim_in_the_qform_is_taken_as_positive(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    data = _values("u1")
    header = _header(data, codes=(0, 1))
    pixdim = header["pixdim"].copy()
    pixdim[1] = -pixdim[1]
    header["pixdim"] = pixdim
    path = _save(tmp_path / "a.nii", _raw(header, data))
    with caplog.at_level(logging.WARNING, logger="larmor"):
        vol = read_nifti(path)
    img: Any = nib.load(path)
    np.testing.assert_allclose(vol.affine, img.affine, rtol=0, atol=1e-12)
    assert "negative voxel sizes" in caplog.text
