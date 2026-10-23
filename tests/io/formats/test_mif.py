import gzip
import math
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from larmor.io import read_bvals_bvecs
from larmor.io.formats import (
    read_mif,
    read_mif_gradients,
    read_mif_header,
    read_nifti,
    write_mif,
)
from larmor.types import GradientTable, Volume

REF = Path(__file__).parents[2] / "data" / "mif"

needs_mrtrix = pytest.mark.skipif(
    shutil.which("mrconvert") is None or shutil.which("mrinfo") is None,
    reason="MRtrix3 (mrconvert, mrinfo) is not on PATH",
)

DTYPES = ["u1", "i1", "i2", "u2", "i4", "u4", "i8", "u8", "f4", "f8"]

# Non-cubic, so that a transposed read gives other values.
SHAPE = (2, 3, 4)


def _rotation(z: float, x: float) -> np.ndarray:
    cz, sz, cx, sx = math.cos(z), math.sin(z), math.cos(x), math.sin(x)
    about_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    about_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rotation: np.ndarray = about_z @ about_x
    return rotation


def _affine(linear: np.ndarray, offset: tuple[float, float, float]) -> np.ndarray:
    affine = np.eye(4)
    affine[:3, :3] = linear
    affine[:3, 3] = offset
    return affine


OFFSET = (-10.5, 20.25, 3.0)
AFFINES = {
    "oblique": _affine(_rotation(0.3, 0.2) @ np.diag([2, 3, 2.5]), OFFSET),
    "reflected": _affine(_rotation(0.3, 0.2) @ np.diag([2, 3, -2.5]), OFFSET),
    "lps": _affine(np.diag([-2.0, -3.0, 2.5]), OFFSET),
    "sheared": _affine(
        _rotation(0.3, 0.2) @ np.array([[2, 0.3, 0], [0, 3, 0.2], [0, 0, 2.5]]),
        OFFSET,
    ),
}
# Axes swapped and flipped: MRtrix reorders them, so its .mif axes differ from the
# NIfTI's. Both signs of the determinant, since the FSL frame depends on it.
SWAPPED = {
    "swapped-det-neg": _affine(
        np.array([[0, -3.0, 0], [2, 0, 0], [0, 0, -2.5]]), OFFSET
    ),
    "swapped-det-pos": _affine(
        np.array([[0, -3.0, 0], [2, 0, 0], [0, 0, 2.5]]), OFFSET
    ),
    "swapped-oblique": _affine(
        _rotation(0.3, 0)[:3, :3] @ np.array([[0, -3.0, 0], [2, 0, 0], [0, 0, -2.5]]),
        OFFSET,
    ),
}


def _values(dtype: str, shape: tuple[int, ...] = SHAPE) -> np.ndarray:
    rng = np.random.default_rng(0)
    kind = np.dtype(dtype)
    if kind.kind == "b":
        return rng.random(shape) > 0.5
    if kind.kind == "f":
        return (rng.standard_normal(shape) * 1e3).astype(kind)
    info = np.iinfo(kind)
    values = rng.integers(info.min, info.max, shape, dtype=kind, endpoint=True)
    values.flat[0], values.flat[-1] = info.min, info.max
    return values


def _table(n: int) -> GradientTable:
    rng = np.random.default_rng(n)
    bvecs = rng.standard_normal((n, 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    bvecs[0] = 0
    bvals = np.array([0.0] + [1000.0, 2000.0, 3000.0] * n)[:n]
    return GradientTable(bvals, bvecs)


def _assert_same_world(got: Volume, want: Volume) -> None:
    # Compares the value at each world position: MRtrix may store the axes of an
    # image in another order and direction than the file it was converted from.
    shape = got.data.shape[:3]
    assert sorted(shape) == sorted(want.data.shape[:3])
    assert got.data.shape[3:] == want.data.shape[3:]
    ijk = np.indices(shape).reshape(3, -1)
    world = got.affine[:3, :3] @ ijk + got.affine[:3, 3:]
    other = np.linalg.solve(want.affine[:3, :3], world - want.affine[:3, 3:])
    idx = np.rint(other).astype(int)
    # Affines read from float32 NIfTI fields land within ~1e-6 voxel of the grid.
    np.testing.assert_allclose(other, idx, rtol=0, atol=1e-4)
    for axis in range(3):
        assert 0 <= idx[axis].min() and idx[axis].max() < want.data.shape[axis]
    assert len({tuple(c) for c in idx.T}) == idx.shape[1]
    np.testing.assert_array_equal(got.data[tuple(ijk)], want.data[tuple(idx)])


def _mrconvert(*args: str | Path, cwd: Path | None = None) -> None:
    subprocess.run(
        ["mrconvert", "-quiet", "-force", *map(str, args)],
        check=True,
        capture_output=True,
        cwd=cwd,
    )


def _mrinfo(path: Path, *options: str) -> list[str]:
    out = subprocess.run(
        ["mrinfo", str(path), *options], check=True, capture_output=True, text=True
    )
    return out.stdout.splitlines()


def _nib_save(data: np.ndarray, affine: np.ndarray, path: Path) -> Path:
    nib.save(nib.Nifti1Image(data, affine), path)
    return path


def _nib_volume(path: Path) -> Volume:
    img = nib.load(path)
    return Volume(np.asarray(img.dataobj), np.asarray(img.affine))  # type: ignore[attr-defined]


def _split_header(raw: bytes) -> tuple[str, int]:
    end = raw.index(b"\nEND\n") + len(b"\nEND\n")
    found = re.search(rb"^file: \. (\d+)$", raw[:end], re.MULTILINE)
    assert found is not None
    return raw[:end].decode(), int(found.group(1))


# Files written by MRtrix at test time

SOURCE = (np.arange(math.prod(SHAPE) * 5) * 37 % 101 - 50).astype(np.int16)
SOURCE_4D = SOURCE.reshape(*SHAPE, 5)
MASK = (np.arange(math.prod(SHAPE)) * 7 % 5 == 0).astype(np.uint8).reshape(SHAPE)


def _mrtrix_pair(
    tmp_path: Path, source: np.ndarray, *options: str
) -> tuple[Path, Path]:
    """Converts `source` to .mif with MRtrix, then the .mif to NIfTI."""
    nii = _nib_save(source, SWAPPED["swapped-oblique"], tmp_path / "source.nii")
    name = options[-1]
    _mrconvert(nii, *options[:-1], tmp_path / name)
    _mrconvert(tmp_path / name, tmp_path / "export.nii")
    return tmp_path / name, tmp_path / "export.nii"


@needs_mrtrix
@pytest.mark.parametrize(
    ("source", "strides"),
    [
        (SOURCE_4D, "1,2,3,4"),
        (SOURCE_4D, "-1,2,3,4"),
        (SOURCE_4D, "3,1,-2,4"),
        (SOURCE_4D, "2,3,4,1"),
        (SOURCE_4D, "-4,-3,-2,-1"),
        (SOURCE_4D[..., 0], "1,2,3"),
        (SOURCE_4D[..., 0], "-2,3,-1"),
    ],
    ids=["1234", "-1234", "31-24", "2341", "all-backwards", "3d", "3d-permuted"],
)
def test_mrtrix_layout_reads_as_its_nifti_export(
    tmp_path: Path, source: np.ndarray, strides: str
) -> None:
    mif, export = _mrtrix_pair(tmp_path, source, "-strides", strides, "a.mif")
    vol = read_mif(mif)
    _assert_same_world(vol, _nib_volume(tmp_path / "source.nii"))
    # MRtrix 3.0.8 writes a volume axis stored backwards to NIfTI in reverse order
    # (mrdump of the .mif agrees with the source, not with that NIfTI).
    if not any(v.startswith("-") for v in strides.split(",")[3:]):
        _assert_same_world(vol, read_nifti(export))
    assert vol.data.dtype == np.int16


MRTRIX_DATATYPES = {
    "int8": "i1",
    "uint8": "u1",
    "int16le": "i2",
    "int16be": "i2",
    "uint16le": "u2",
    "uint16be": "u2",
    "int32le": "i4",
    "int32be": "i4",
    "uint32le": "u4",
    "uint32be": "u4",
    "int64le": "i8",
    "int64be": "i8",
    "uint64le": "u8",
    "uint64be": "u8",
    "float32le": "f4",
    "float32be": "f4",
    "float64le": "f8",
    "float64be": "f8",
    "bit": "?",
}


@needs_mrtrix
@pytest.mark.parametrize(("datatype", "dtype"), MRTRIX_DATATYPES.items())
def test_mrtrix_datatype_reads_as_its_nifti_export(
    tmp_path: Path, datatype: str, dtype: str
) -> None:
    source = MASK if datatype == "bit" else np.abs(SOURCE_4D)
    mif, export = _mrtrix_pair(
        tmp_path, source, "-datatype", datatype, "-strides", "-1,3,2", "a.mif"
    )
    vol = read_mif(mif)
    assert vol.data.dtype == np.dtype(dtype)
    assert vol.data.dtype.isnative
    want = read_nifti(export)
    if datatype == "bit":
        want = want.with_data(want.data.astype(bool))
    _assert_same_world(vol, want)


@needs_mrtrix
@pytest.mark.parametrize("name", ["a.mif.gz", "a.mih"])
def test_mrtrix_compressed_and_split_files_read_as_nifti_export(
    tmp_path: Path, name: str
) -> None:
    mif, export = _mrtrix_pair(tmp_path, SOURCE_4D, "-strides", "3,1,-2,4", name)
    _assert_same_world(read_mif(mif), read_nifti(export))
    grid, shape = read_mif_header(mif)
    assert shape == read_mif(mif).data.shape
    np.testing.assert_array_equal(grid.affine, read_mif(mif).affine)


@needs_mrtrix
@pytest.mark.parametrize("affine", SWAPPED.values(), ids=SWAPPED.keys())
def test_mrtrix_dw_scheme_from_fsl_files_equals_larmor_conversion(
    tmp_path: Path, affine: np.ndarray
) -> None:
    table = _table(5)
    nii = _nib_save(SOURCE_4D, affine, tmp_path / "dwi.nii")
    # Any unit vectors serve as an FSL bvecs file; both tools convert them to world.
    np.savetxt(tmp_path / "bvals", table.bvals[None], fmt="%.17g")
    np.savetxt(tmp_path / "bvecs", table.bvecs.T, fmt="%.17g")
    _mrconvert(
        nii, "-fslgrad", tmp_path / "bvecs", tmp_path / "bvals", tmp_path / "a.mif"
    )
    mrtrix = read_mif_gradients(tmp_path / "a.mif")
    larmor = read_bvals_bvecs(tmp_path / "bvals", tmp_path / "bvecs", ref_path=nii)
    # MRtrix writes dw_scheme with 10 significant digits.
    np.testing.assert_allclose(mrtrix.bvals, larmor.bvals, rtol=1e-9)
    np.testing.assert_allclose(mrtrix.bvecs, larmor.bvecs, rtol=0, atol=1e-6)


# MRtrix reads files written by Larmor


@needs_mrtrix
@pytest.mark.parametrize(
    "affine",
    [*AFFINES.values(), *SWAPPED.values()],
    ids=[*AFFINES.keys(), *SWAPPED.keys()],
)
@pytest.mark.parametrize("name", ["a.mif", "a.mif.gz", "a.mih"])
def test_mrtrix_reads_larmor_file_at_same_world_positions(
    tmp_path: Path, affine: np.ndarray, name: str
) -> None:
    vol = Volume(_values("f4", (*SHAPE, 3)), affine)
    write_mif(vol, tmp_path / name)
    _mrconvert(tmp_path / name, tmp_path / "export.nii")
    _assert_same_world(read_nifti(tmp_path / "export.nii"), vol)


@needs_mrtrix
@pytest.mark.parametrize(
    ("dtype", "datatype"),
    [
        ("i1", "Int8"),
        ("u1", "UInt8"),
        ("i2", "Int16LE"),
        ("u2", "UInt16LE"),
        ("i4", "Int32LE"),
        ("u4", "UInt32LE"),
        ("i8", "Int64LE"),
        ("u8", "UInt64LE"),
        ("f4", "Float32LE"),
        ("f8", "Float64LE"),
        ("?", "Bit"),
    ],
)
def test_mrtrix_reports_larmor_size_datatype_and_strides(
    tmp_path: Path, dtype: str, datatype: str
) -> None:
    # An axis-aligned RAS affine, so that MRtrix keeps the axes as they are.
    vol = Volume(_values(dtype, (*SHAPE, 3)), np.diag([2.0, 3.0, 2.5, 1.0]))
    write_mif(vol, tmp_path / "a.mif")
    info = _mrinfo(tmp_path / "a.mif", "-size", "-datatype", "-strides")
    assert info == ["2 3 4 3", datatype, "1 2 3 4"]
    _mrconvert(tmp_path / "a.mif", tmp_path / "export.nii")
    back = read_nifti(tmp_path / "export.nii")
    np.testing.assert_array_equal(back.data.astype(vol.data.dtype), vol.data)


@needs_mrtrix
def test_mrtrix_keeps_larmor_bit_image_as_bit(tmp_path: Path) -> None:
    vol = Volume(_values("?", (5, 3, 7)), AFFINES["oblique"])
    write_mif(vol, tmp_path / "a.mif")
    _mrconvert(tmp_path / "a.mif", tmp_path / "b.mif")
    assert _mrinfo(tmp_path / "b.mif", "-datatype") == ["Bit"]
    _assert_same_world(read_mif(tmp_path / "b.mif"), vol)


@needs_mrtrix
def test_mrtrix_reads_larmor_dw_scheme_unchanged(tmp_path: Path) -> None:
    table = _table(7)
    write_mif(
        Volume(_values("i2", (*SHAPE, 7)), AFFINES["oblique"]),
        tmp_path / "a.mif",
        gradients=table,
    )
    rows = np.array(
        [line.split() for line in _mrinfo(tmp_path / "a.mif", "-dwgrad")], float
    )
    # mrinfo prints 15 significant digits.
    np.testing.assert_allclose(rows[:, :3], table.bvecs, rtol=0, atol=1e-14)
    np.testing.assert_allclose(rows[:, 3], table.bvals, rtol=1e-14)
    _mrconvert(tmp_path / "a.mif", tmp_path / "b.mif")
    back = read_mif_gradients(tmp_path / "b.mif")
    np.testing.assert_array_equal(back.bvecs, table.bvecs)
    np.testing.assert_array_equal(back.bvals, table.bvals)


# Committed files written by MRtrix (tests/data/README.md): these run everywhere.

SOURCE_REF = REF / "source.nii"


@pytest.mark.parametrize(
    ("name", "source", "dtype"),
    [
        ("layout_permuted.mif", "source.nii", "i2"),
        ("float32be.mif", "source.nii", "f4"),
        ("int16be.mif.gz", "source.nii", "i2"),
        ("mask_bit.mif", "mask.nii", "?"),
        ("split.mih", "source.nii", "i2"),
        ("dwi.mif", "source.nii", "i2"),
    ],
)
def test_committed_mrtrix_file_reads_as_its_source(
    name: str, source: str, dtype: str
) -> None:
    vol = read_mif(REF / name)
    assert vol.data.dtype == np.dtype(dtype)
    assert not vol.data.flags.writeable
    want = _nib_volume(REF / source)
    if dtype == "?":
        want = want.with_data(want.data.astype(bool))
    _assert_same_world(vol, want)
    grid, shape = read_mif_header(REF / name)
    assert shape == vol.data.shape
    np.testing.assert_array_equal(grid.affine, vol.affine)


def test_committed_dw_scheme_equals_larmor_conversion_of_fsl_files() -> None:
    mrtrix = read_mif_gradients(REF / "dwi.mif")
    larmor = read_bvals_bvecs(REF / "dwi.bval", REF / "dwi.bvec", ref_path=SOURCE_REF)
    np.testing.assert_allclose(mrtrix.bvals, larmor.bvals, rtol=1e-9)
    np.testing.assert_allclose(mrtrix.bvecs, larmor.bvecs, rtol=0, atol=1e-6)


# Round trips through Larmor


@pytest.mark.parametrize("dtype", [*DTYPES, "?"])
@pytest.mark.parametrize("shape", [SHAPE, (*SHAPE, 5)], ids=["3d", "4d"])
@pytest.mark.parametrize("name", ["a.mif", "a.mif.gz", "a.mih"])
def test_every_dtype_reads_back_the_same(
    tmp_path: Path, dtype: str, shape: tuple[int, ...], name: str
) -> None:
    vol = Volume(_values(dtype, shape), AFFINES["oblique"])
    write_mif(vol, tmp_path / name)
    back = read_mif(tmp_path / name)
    np.testing.assert_array_equal(back.data, vol.data, strict=True)
    assert (back.slope, back.intercept) == (1.0, 0.0)


@pytest.mark.parametrize("affine", AFFINES.values(), ids=AFFINES.keys())
def test_affine_reads_back(tmp_path: Path, affine: np.ndarray) -> None:
    write_mif(Volume(_values("i2"), affine), tmp_path / "a.mif")
    back = read_mif(tmp_path / "a.mif")
    # The file holds affine / voxel size and the voxel size, each exactly; the
    # product can differ from the affine in the last bit.
    np.testing.assert_allclose(back.affine, affine, rtol=0, atol=1e-13)
    grid, shape = read_mif_header(tmp_path / "a.mif")
    assert shape == SHAPE
    np.testing.assert_array_equal(grid.affine, back.affine)


@pytest.mark.parametrize("scaling", [(0.5, -3.0), (1.0, 2.5), (1 / 3, 0.0)])
def test_scaling_reads_back_and_is_not_applied(
    tmp_path: Path, scaling: tuple[float, float]
) -> None:
    vol = Volume(_values("i2"), AFFINES["oblique"], *scaling)
    write_mif(vol, tmp_path / "a.mif")
    back = read_mif(tmp_path / "a.mif")
    assert (back.slope, back.intercept) == scaling
    np.testing.assert_array_equal(back.data, vol.data, strict=True)


@pytest.mark.parametrize("name", ["a.mif", "a.mif.gz", "a.mih"])
def test_gradient_table_reads_back_identical(tmp_path: Path, name: str) -> None:
    table = _table(6)
    # Values with long decimal expansions, so that any rounding shows.
    table = GradientTable(table.bvals + 1 / 3, table.bvecs)
    write_mif(
        Volume(_values("i2", (*SHAPE, 6)), AFFINES["oblique"]),
        tmp_path / name,
        gradients=table,
    )
    back = read_mif_gradients(tmp_path / name)
    np.testing.assert_array_equal(back.bvals, table.bvals, strict=True)
    np.testing.assert_array_equal(back.bvecs, table.bvecs, strict=True)


@pytest.mark.parametrize("n", [1290, 1295, 1296, 1300])
def test_large_gradient_table_reads_back(tmp_path: Path, n: int) -> None:
    # About 100 kB of header: the data offset takes six digits. Which n overrun
    # depends on the header length modulo 4, so several are tried.
    table = _table(n)
    vol = Volume(np.zeros((1, 1, 1, n), np.float32), np.eye(4))
    write_mif(vol, tmp_path / "a.mif", gradients=table)
    back = read_mif_gradients(tmp_path / "a.mif")
    np.testing.assert_array_equal(back.bvecs, table.bvecs)
    np.testing.assert_array_equal(read_mif(tmp_path / "a.mif").data, vol.data)


@pytest.mark.parametrize("dtype", [">i2", ">f8"])
def test_big_endian_data_is_written_little_endian(tmp_path: Path, dtype: str) -> None:
    data = _values(dtype[1:]).astype(dtype)
    write_mif(Volume(data, AFFINES["oblique"]), tmp_path / "a.mif")
    raw = (tmp_path / "a.mif").read_bytes()
    _, offset = _split_header(raw)
    assert raw[offset:] == data.astype(np.dtype(dtype).newbyteorder("<")).tobytes("F")
    np.testing.assert_array_equal(read_mif(tmp_path / "a.mif").data, data)


def test_flipped_view_is_written_in_its_own_order(tmp_path: Path) -> None:
    vol = Volume(_values("i2", (*SHAPE, 2)), AFFINES["oblique"])
    vol.flip_axis(1)
    write_mif(vol, tmp_path / "a.mif")
    back = read_mif(tmp_path / "a.mif")
    np.testing.assert_array_equal(back.data, vol.data)
    np.testing.assert_allclose(back.affine, vol.affine, rtol=0, atol=1e-13)


# Written file layout


@pytest.mark.parametrize(
    ("dtype", "datatype"),
    [
        ("i1", "Int8"),
        ("u1", "UInt8"),
        ("i2", "Int16LE"),
        ("u4", "UInt32LE"),
        ("i8", "Int64LE"),
        ("f4", "Float32LE"),
        ("f8", "Float64LE"),
        ("?", "Bit"),
    ],
)
def test_written_header_lines(tmp_path: Path, dtype: str, datatype: str) -> None:
    affine = AFFINES["lps"]
    write_mif(Volume(_values(dtype, (*SHAPE, 5)), affine), tmp_path / "a.mif")
    text, _ = _split_header((tmp_path / "a.mif").read_bytes())
    lines = text.splitlines()
    assert lines[0] == "mrtrix image"
    assert "dim: 2,3,4,5" in lines
    assert "vox: 2.0,3.0,2.5,1.0" in lines
    assert "layout: +0,+1,+2,+3" in lines
    assert f"datatype: {datatype}" in lines
    assert [x for x in lines if x.startswith("transform:")] == [
        "transform: -1.0,0.0,0.0,-10.5",
        "transform: 0.0,-1.0,0.0,20.25",
        "transform: 0.0,0.0,1.0,3.0",
    ]
    assert not any(x.startswith(("scaling", "dw_scheme")) for x in lines)


@pytest.mark.parametrize("n_volumes", [None, 1, 5, 300])
@pytest.mark.parametrize("gz", [False, True], ids=["plain", "gzip"])
def test_data_starts_at_multiple_of_four_after_zero_padding(
    tmp_path: Path, n_volumes: int | None, gz: bool
) -> None:
    shape = SHAPE if n_volumes is None else (1, 1, 1, n_volumes)
    data = _values("i2", shape)
    table = None if n_volumes in (None, 1) else _table(data.shape[3])
    path = tmp_path / ("a.mif.gz" if gz else "a.mif")
    write_mif(Volume(data, AFFINES["oblique"]), path, gradients=table)
    raw = gzip.decompress(path.read_bytes()) if gz else path.read_bytes()
    text, offset = _split_header(raw)
    assert offset % 4 == 0
    assert len(text) <= offset
    assert raw[len(text) : offset] == bytes(offset - len(text))
    assert raw[offset:] == data.astype("<i2").tobytes(order="F")


@pytest.mark.parametrize("shape", [(3, 1, 1), (5, 3, 7), (2, 3, 4, 3)])
def test_bool_is_packed_most_significant_bit_first(
    tmp_path: Path, shape: tuple[int, ...]
) -> None:
    data = _values("?", shape)
    data.flat[0], data.flat[1] = True, False
    write_mif(Volume(data, np.eye(4)), tmp_path / "a.mif")
    raw = (tmp_path / "a.mif").read_bytes()
    _, offset = _split_header(raw)
    bits = np.unpackbits(np.frombuffer(raw[offset:], np.uint8), bitorder="big")
    assert len(raw) - offset == -(-data.size // 8)
    np.testing.assert_array_equal(bits[: data.size], data.ravel(order="F"))
    assert not bits[data.size :].any()


@pytest.mark.parametrize(
    ("dtype", "shape"), [("i2", (*SHAPE, 5)), ("f8", SHAPE), ("?", (5, 3, 7))]
)
def test_mih_writes_header_and_dat_of_same_name(
    tmp_path: Path, dtype: str, shape: tuple[int, ...]
) -> None:
    vol = Volume(_values(dtype, shape), AFFINES["oblique"])
    write_mif(vol, tmp_path / "scan.mih")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["scan.dat", "scan.mih"]
    lines = (tmp_path / "scan.mih").read_text().splitlines()
    assert "file: scan.dat 0" in lines
    assert lines[-1] == "END"
    size = -(-vol.data.size // 8) if dtype == "?" else vol.data.nbytes
    assert (tmp_path / "scan.dat").stat().st_size == size


@pytest.mark.parametrize("dtype", ["f2", "c8", "c16"])
def test_unsupported_dtype_raises_before_writing(tmp_path: Path, dtype: str) -> None:
    vol = Volume(_values("f4").astype(dtype), AFFINES["oblique"])
    with pytest.raises(ValueError, match=np.dtype(dtype).name):
        write_mif(vol, tmp_path / "a.mif")
    assert not (tmp_path / "a.mif").exists()


@pytest.mark.parametrize(
    ("shape", "n"),
    [((*SHAPE, 5), 4), ((*SHAPE, 5), 6)],
    ids=["fewer", "more"],
)
def test_gradient_count_mismatch_raises_before_writing(
    tmp_path: Path, shape: tuple[int, ...], n: int
) -> None:
    with pytest.raises(ValueError, match=rf"{n} entries, but the volume has 5"):
        write_mif(
            Volume(_values("i2", shape), np.eye(4)),
            tmp_path / "a.mif",
            gradients=_table(n),
        )
    assert not (tmp_path / "a.mif").exists()


def test_gradients_for_a_3d_volume_raise_before_writing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs a 4D volume"):
        write_mif(
            Volume(_values("i2", SHAPE), np.eye(4)),
            tmp_path / "a.mif",
            gradients=_table(1),
        )
    assert not (tmp_path / "a.mif").exists()


@pytest.mark.parametrize("entry", ["my scan.dat", "my scan.dat 0"])
def test_data_file_name_may_contain_spaces(tmp_path: Path, entry: str) -> None:
    (tmp_path / "my scan.dat").write_bytes(BASE_DATA.astype("<i2").tobytes("F"))
    vol = read_mif(_hand(tmp_path / "a.mif", file=entry, payload=b""))
    np.testing.assert_array_equal(vol.data, BASE_DATA, strict=True)


# Hand-written headers

BASE = {
    "dim": "2,3,4",
    "vox": "1.5,2,3",
    "layout": "+0,+1,+2",
    "datatype": "Int16LE",
}
BASE_DATA = np.arange(24, dtype=np.int16).reshape(SHAPE)
DATA_OFFSET = 1024


def _hand(
    path: Path,
    keys: dict[str, str | list[str] | None] | None = None,
    payload: bytes | None = None,
    magic: str = "mrtrix image",
    file: str | list[str] | None = "auto",
) -> Path:
    """Writes a MIF file from header values; None drops a key of BASE."""
    values: dict[str, str | list[str] | None] = {**BASE, **(keys or {})}
    lines = [magic]
    for key, value in values.items():
        if value is None:
            continue
        for v in [value] if isinstance(value, str) else value:
            lines.append(f"{key}: {v}")
    files = [f". {DATA_OFFSET}"] if file == "auto" else file
    for v in [files] if isinstance(files, str) else files or []:
        lines.append(f"file: {v}")
    text = ("\n".join(lines) + "\nEND\n").encode()
    if payload is None:
        payload = BASE_DATA.astype("<i2").tobytes(order="F")
    assert len(text) <= DATA_OFFSET
    path.write_bytes(text + bytes(DATA_OFFSET - len(text)) + payload)
    return path


def test_hand_written_file_reads_with_affine_from_transform_and_vox(
    tmp_path: Path,
) -> None:
    transform = ["0,-1,0,10", "1,0,0,-20", "0,0,1,5.5"]
    vol = read_mif(_hand(tmp_path / "a.mif", {"transform": transform}))
    np.testing.assert_array_equal(vol.data, BASE_DATA, strict=True)
    expected = np.array(
        [[0, -2.0, 0, 10], [1.5, 0, 0, -20], [0, 0, 3.0, 5.5], [0, 0, 0, 1]]
    )
    np.testing.assert_array_equal(vol.affine, expected)


def test_missing_transform_gives_voxel_sizes_and_zero_origin(tmp_path: Path) -> None:
    vol = read_mif(_hand(tmp_path / "a.mif"))
    np.testing.assert_array_equal(vol.affine, np.diag([1.5, 2.0, 3.0, 1.0]))


# Explicit cases: dims 3,2,1 and int8 bytes 0..5 stored in file order.
@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        ("+0,+1,+2", [[0, 3], [1, 4], [2, 5]]),
        ("+1,+0,+2", [[0, 1], [2, 3], [4, 5]]),
        ("-1,+0,+2", [[4, 5], [2, 3], [0, 1]]),
        ("+1,-0,+2", [[1, 0], [3, 2], [5, 4]]),
        ("-0,+1,+2", [[2, 5], [1, 4], [0, 3]]),
        ("-0,-1,+2", [[5, 2], [4, 1], [3, 0]]),
        ("+2,+0,+1", [[0, 1], [2, 3], [4, 5]]),
    ],
)
def test_layout_order_and_signs_are_undone(
    tmp_path: Path, layout: str, expected: list[list[int]]
) -> None:
    keys: dict[str, str | list[str] | None] = {
        "dim": "3,2,1",
        "vox": "1,1,1",
        "layout": layout,
        "datatype": "Int8",
    }
    vol = read_mif(_hand(tmp_path / "a.mif", keys, bytes(range(6))))
    np.testing.assert_array_equal(vol.data[..., 0], np.array(expected, np.int8))


@pytest.mark.parametrize(
    "layout",
    ["+0,+1,+2,+3", "+3,+2,+1,+0", "-1,+3,-0,+2", "+1,+2,+3,+0", "-0,-1,-2,-3"],
)
def test_four_dimensional_layout_is_undone(tmp_path: Path, layout: str) -> None:
    data = np.arange(2 * 3 * 4 * 5, dtype=np.int16).reshape(*SHAPE, 5)
    ranks = [abs(int(v)) for v in layout.split(",")]
    stored = data
    for axis, v in enumerate(layout.split(",")):
        if v.startswith("-"):
            stored = np.flip(stored, axis)
    stored = np.transpose(stored, np.argsort(ranks))
    keys: dict[str, str | list[str] | None] = {
        "dim": "2,3,4,5",
        "vox": "1,1,1,1",
        "layout": layout,
    }
    vol = read_mif(_hand(tmp_path / "a.mif", keys, stored.astype("<i2").tobytes("F")))
    np.testing.assert_array_equal(vol.data, data, strict=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("suffix", ["LE", "BE", ""], ids=["le", "be", "native"])
def test_every_datatype_and_byte_order_reads(
    tmp_path: Path, dtype: str, suffix: str
) -> None:
    names = {"u": "UInt", "i": "Int", "f": "Float"}
    datatype = f"{names[dtype[0]]}{int(dtype[1]) * 8}{suffix}"
    order = {"LE": "<", "BE": ">", "": "="}[suffix]
    data = _values(dtype)
    payload = data.astype(order + dtype).tobytes(order="F")
    vol = read_mif(_hand(tmp_path / "a.mif", {"datatype": datatype}, payload))
    assert vol.data.dtype.isnative
    np.testing.assert_array_equal(vol.data, data, strict=True)


def test_bit_is_unpacked_most_significant_bit_first(tmp_path: Path) -> None:
    keys: dict[str, str | list[str] | None] = {
        "dim": "10,1,1",
        "vox": "1,1,1",
        "datatype": "Bit",
    }
    vol = read_mif(_hand(tmp_path / "a.mif", keys, bytes([0b10110000, 0b01000000])))
    expected = np.array([1, 0, 1, 1, 0, 0, 0, 0, 0, 1], dtype=bool)
    np.testing.assert_array_equal(vol.data[:, 0, 0], expected, strict=True)


def test_scaling_is_offset_then_scale(tmp_path: Path) -> None:
    vol = read_mif(_hand(tmp_path / "a.mif", {"scaling": "2.5,0.5"}))
    assert (vol.intercept, vol.slope) == (2.5, 0.5)
    np.testing.assert_array_equal(vol.data, BASE_DATA, strict=True)


@pytest.mark.parametrize(
    ("keys", "shape"),
    [
        ({"dim": "4", "vox": "2", "layout": "+0"}, (4, 1, 1)),
        ({"dim": "4,6", "vox": "2,3", "layout": "+0,+1"}, (4, 6, 1)),
    ],
    ids=["1d", "2d"],
)
def test_fewer_than_three_dimensions_get_size_one_axes(
    tmp_path: Path, keys: dict[str, str | list[str] | None], shape: tuple[int, ...]
) -> None:
    data = np.arange(math.prod(shape), dtype=np.int16)
    path = _hand(tmp_path / "a.mif", keys, data.astype("<i2").tobytes())
    vol = read_mif(path)
    np.testing.assert_array_equal(vol.data, data.reshape(shape, order="F"), strict=True)
    sizes = [float(v) for v in str(keys["vox"]).split(",")]
    np.testing.assert_array_equal(vol.voxel_sizes[: len(sizes)], sizes)
    grid, full = read_mif_header(path)
    assert full == shape
    np.testing.assert_array_equal(grid.affine, vol.affine)


def test_data_file_without_offset_is_read_from_its_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The name is relative to the header's folder, not to the working directory.
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.dat").write_bytes(BASE_DATA.astype("<i2").tobytes("F"))
    _hand(tmp_path / "sub" / "a.mif", file="d.dat", payload=b"")
    monkeypatch.chdir(tmp_path)
    vol = read_mif(Path("sub") / "a.mif")
    np.testing.assert_array_equal(vol.data, BASE_DATA, strict=True)


def test_data_file_offset_is_honoured(tmp_path: Path) -> None:
    data = BASE_DATA.astype("<i2").tobytes("F")
    (tmp_path / "d.dat").write_bytes(b"skip me!" + data)
    vol = read_mif(_hand(tmp_path / "a.mif", file="d.dat 8", payload=b""))
    np.testing.assert_array_equal(vol.data, BASE_DATA, strict=True)


def test_mih_without_end_line_reads(tmp_path: Path) -> None:
    # MRtrix writes .mih files that stop after the file line.
    (tmp_path / "a.dat").write_bytes(BASE_DATA.astype("<i2").tobytes("F"))
    text = "mrtrix image\n" + "".join(f"{k}: {v}\n" for k, v in BASE.items())
    (tmp_path / "a.mih").write_text(text + "file: a.dat\n")
    vol = read_mif(tmp_path / "a.mih")
    np.testing.assert_array_equal(vol.data, BASE_DATA, strict=True)


def test_mif_without_end_line_raises(tmp_path: Path) -> None:
    text = "mrtrix image\n" + "".join(f"{k}: {v}\n" for k, v in BASE.items())
    (tmp_path / "a.mif").write_text(text + "file: . 200\n")
    with pytest.raises(ValueError, match="a.mif"):
        read_mif(tmp_path / "a.mif")


def test_header_is_read_without_the_data(tmp_path: Path) -> None:
    keys: dict[str, str | list[str] | None] = {
        "dim": "2,3,4,2",
        "vox": "1.5,2,3,1",
        "layout": "+0,+1,+2,+3",
        "dw_scheme": ["0,0,1,0", "1,0,0,1000"],
    }
    path = _hand(tmp_path / "a.mif", keys, payload=b"")
    grid, shape = read_mif_header(path)
    assert shape == (2, 3, 4, 2)
    assert grid.shape == SHAPE
    np.testing.assert_array_equal(grid.affine, np.diag([1.5, 2.0, 3.0, 1.0]))
    assert read_mif_gradients(path).bvals.tolist() == [0.0, 1000.0]
    with pytest.raises(ValueError, match="a.mif"):
        read_mif(path)


def test_missing_data_file_raises_naming_it(tmp_path: Path) -> None:
    path = _hand(tmp_path / "a.mif", file="gone.dat 0", payload=b"")
    with pytest.raises((OSError, ValueError), match="gone.dat"):
        read_mif(path)


HEADER_ERRORS = [
    pytest.param({"dim": None}, "auto", "dim", id="no-dim"),
    pytest.param({"vox": None}, "auto", "vox", id="no-vox"),
    pytest.param({"layout": None}, "auto", "layout", id="no-layout"),
    pytest.param({"datatype": None}, "auto", "datatype", id="no-datatype"),
    pytest.param({}, None, "file", id="no-file"),
    pytest.param({"vox": "1,2"}, "auto", "vox", id="vox-length"),
    pytest.param({"layout": "+0,+1"}, "auto", "layout", id="layout-length"),
    pytest.param({"dim": "2,3,4,5"}, "auto", "dim", id="dim-length"),
    pytest.param({"layout": "+0,+1,+1"}, "auto", "layout", id="layout-repeat"),
    pytest.param({"layout": "+0,+1,+3"}, "auto", "layout", id="layout-range"),
    pytest.param({"layout": "+0,+1,x"}, "auto", "layout", id="layout-text"),
    pytest.param({"dim": "2,0,4"}, "auto", "dim", id="dim-0"),
    pytest.param({"dim": "2,-3,4"}, "auto", "dim", id="dim-negative"),
    pytest.param({"dim": "2,3.5,4"}, "auto", "dim", id="dim-float"),
    pytest.param({"vox": "1,a,1"}, "auto", "vox", id="vox-text"),
    pytest.param(
        {"transform": ["1,0,0,0", "0,1,0,0"]}, "auto", "transform", id="2-rows"
    ),
    pytest.param(
        {"transform": ["1,0,0", "0,1,0", "0,0,1"]}, "auto", "transform", id="3-columns"
    ),
    pytest.param({"scaling": "2"}, "auto", "scaling", id="scaling-1"),
    pytest.param({"scaling": "0,1,2"}, "auto", "scaling", id="scaling-3"),
    pytest.param({}, [". 1024", "b.dat 0"], "file", id="two-files"),
    pytest.param({}, ". x", "offset", id="offset-text"),
    pytest.param({"datatype": "CFloat32LE"}, "auto", "CFloat32LE", id="cfloat32"),
    pytest.param({"datatype": "CFloat64"}, "auto", "CFloat64", id="cfloat64"),
    pytest.param({"datatype": "Float16LE"}, "auto", "Float16LE", id="float16"),
    pytest.param({"datatype": "Int24"}, "auto", "Int24", id="unknown"),
    pytest.param({"datatype": "Int16XE"}, "auto", "Int16XE", id="bad-suffix"),
]


@pytest.mark.parametrize(("keys", "file", "says"), HEADER_ERRORS)
@pytest.mark.parametrize("read", [read_mif, read_mif_header, read_mif_gradients])
def test_bad_header_raises_naming_file(
    tmp_path: Path,
    keys: dict[str, str | list[str] | None],
    file: str | list[str] | None,
    says: str,
    read: Callable[[Path], object],
) -> None:
    path = _hand(tmp_path / "bad.mif", keys, file=file)
    with pytest.raises(ValueError, match="bad.mif") as err:
        read(path)
    assert says in str(err.value)


@pytest.mark.parametrize("magic", ["mrtrix tracks", "MRtrix image", "", "\x00\x01"])
@pytest.mark.parametrize("read", [read_mif, read_mif_header, read_mif_gradients])
def test_wrong_magic_raises_naming_file(
    tmp_path: Path, magic: str, read: Callable[[Path], object]
) -> None:
    path = _hand(tmp_path / "bad.mif", magic=magic)
    with pytest.raises(ValueError, match="bad.mif") as err:
        read(path)
    assert "mrtrix image" in str(err.value)


@pytest.mark.parametrize("read", [read_mif, read_mif_header])
def test_more_than_four_dimensions_raise(
    tmp_path: Path, read: Callable[[Path], object]
) -> None:
    keys: dict[str, str | list[str] | None] = {
        "dim": "2,3,4,1,1",
        "vox": "1,1,1,1,1",
        "layout": "+0,+1,+2,+3,+4",
    }
    with pytest.raises(ValueError, match="bad.mif") as err:
        read(_hand(tmp_path / "bad.mif", keys))
    assert "5" in str(err.value).replace(str(tmp_path), "")


def test_short_data_raises_naming_file(tmp_path: Path) -> None:
    path = _hand(tmp_path / "bad.mif", payload=bytes(47))
    with pytest.raises(ValueError, match="bad.mif"):
        read_mif(path)


# Gradient table errors

GRADIENT_ERRORS = [
    pytest.param({}, "dw_scheme", id="none"),
    pytest.param({"dw_scheme": ["0,0,1,0", "1,0,0"]}, "4 numbers", id="3-numbers"),
    pytest.param(
        {"dw_scheme": ["0,0,1,0", "1,0,0,1000,5"]}, "4 numbers", id="5-numbers"
    ),
    pytest.param({"dw_scheme": ["0,0,1,0", "a,b,c,d"]}, "dw_scheme", id="text"),
    pytest.param({"dw_scheme": ["0,0,1,0"]}, "1 dw_scheme", id="fewer"),
    pytest.param({"dw_scheme": ["0,0,1,0"] * 3}, "3 dw_scheme", id="more"),
]


@pytest.mark.parametrize(("keys", "says"), GRADIENT_ERRORS)
def test_bad_dw_scheme_raises_naming_file(
    tmp_path: Path, keys: dict[str, str | list[str] | None], says: str
) -> None:
    keys = {"dim": "2,3,4,2", "vox": "1,1,1,1", "layout": "+0,+1,+2,+3", **keys}
    with pytest.raises(ValueError, match="bad.mif") as err:
        read_mif_gradients(_hand(tmp_path / "bad.mif", keys, payload=bytes(96)))
    assert says in str(err.value)


def test_dw_scheme_of_3d_image_raises(tmp_path: Path) -> None:
    path = _hand(tmp_path / "bad.mif", {"dw_scheme": ["0,0,1,0"]})
    with pytest.raises(ValueError, match="bad.mif") as err:
        read_mif_gradients(path)
    assert "fourth dimension" in str(err.value)


def test_dw_scheme_is_world_coordinates_without_conversion(tmp_path: Path) -> None:
    # A swapped, flipped transform must not change the vectors read.
    keys: dict[str, str | list[str] | None] = {
        "dim": "2,3,4,2",
        "vox": "1,1,1,1",
        "layout": "-2,+0,+1,+3",
        "transform": ["0,-1,0,0", "-1,0,0,0", "0,0,-1,0"],
        "dw_scheme": ["0.6,0,-0.8,0", "0,0.8,0.6,1000.5"],
    }
    table = read_mif_gradients(_hand(tmp_path / "a.mif", keys, payload=b""))
    np.testing.assert_array_equal(table.bvecs, [[0.6, 0, -0.8], [0, 0.8, 0.6]])
    np.testing.assert_array_equal(table.bvals, [0, 1000.5])
