import gzip
import logging
import math
from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from nibabel.streamlines.header import Field
from nibabel.streamlines.tractogram import Tractogram as NibTractogram
from nibabel.streamlines.trk import (
    TrkFile,
    get_affine_trackvis_to_rasmm,
    header_2_dtype,
)

from larmor.io.formats import read_tck, read_trk, write_tck, write_trk
from larmor.types import Grid, Tractogram

HEADER = header_2_dtype.newbyteorder("<")

# Non-cubic, so that a flip or permutation using the wrong axis' size moves points.
SHAPE = (4, 5, 6)
OFFSET = (-10.5, 20.25, 3.0)
LENGTHS = (3, 1, 5, 2)

# Points are float32 (~1e-6 relative); TRK stores voxmm and its affine in float32,
# so the conversion to world mm adds a few float32 roundings of ~30 mm values.
RTOL, ATOL = 1e-6, 1e-4
# vox_to_ras is stored as float32: entries up to ~25 mm round to ~2e-6.
F32_ATOL = 1e-5


def _rotation(z: float, x: float) -> np.ndarray:
    cz, sz, cx, sx = math.cos(z), math.sin(z), math.cos(x), math.sin(x)
    about_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    about_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rotation: np.ndarray = about_z @ about_x
    return rotation


def _affine(linear: np.ndarray) -> np.ndarray:
    affine = np.eye(4)
    affine[:3, :3] = linear
    affine[:3, 3] = OFFSET
    return affine


AFFINES = {
    "ras": _affine(np.diag([2.0, 2.5, 3.0])),
    "lps": _affine(np.diag([-2.0, -2.5, 3.0])),
    "las": _affine(np.diag([-2.0, 2.5, 3.0])),
    "oblique": _affine(_rotation(0.3, 0.2) @ np.diag([2.0, 2.5, 3.0])),
    # Voxel axes along S, R and A.
    "permuted": _affine(np.array([[0, 2.5, 0], [0, 0, 3.0], [2.0, 0, 0]])),
}
AFFINE_IDS = list(AFFINES)
EACH_AFFINE = pytest.mark.parametrize("affine", AFFINES.values(), ids=AFFINE_IDS)


def _tractogram(
    affine: np.ndarray = AFFINES["oblique"],
    lengths: tuple[int, ...] = LENGTHS,
    reference: Grid | None = None,
) -> Tractogram:
    rng = np.random.default_rng(0)
    n = sum(lengths)
    voxels = rng.uniform(0, 1, (n, 3)) * (np.array(SHAPE) - 1)
    points = voxels @ affine[:3, :3].T + affine[:3, 3]
    s = len(lengths)
    return Tractogram(
        points=points,
        offsets=np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)]),
        point_data={
            "fa": rng.uniform(0, 1, n).astype(np.float32),
            "dir": rng.uniform(-1, 1, (n, 3)).astype(np.float32),
        },
        streamline_data={
            "length": rng.uniform(10, 50, s).astype(np.float32),
            "rgb": rng.uniform(0, 1, (s, 3)).astype(np.float32),
        },
        reference=reference,
    )


def _column(values: np.ndarray) -> np.ndarray:
    # nibabel keeps a size-1 last axis that Larmor drops.
    return values if values.ndim == 2 else values[:, np.newaxis]


def _nib_save(
    path: Path,
    t: Tractogram,
    affine: np.ndarray,
    shape: tuple[int, int, int] = SHAPE,
) -> Path:
    spans = list(zip(t.offsets[:-1], t.offsets[1:], strict=True))
    tractogram = NibTractogram(
        [t.points[a:b] for a, b in spans],
        data_per_point={
            k: [_column(v)[a:b] for a, b in spans] for k, v in t.point_data.items()
        },
        data_per_streamline={k: _column(v) for k, v in t.streamline_data.items()},
        affine_to_rasmm=np.eye(4),
    )
    header = {
        Field.VOXEL_TO_RASMM: affine,
        Field.DIMENSIONS: shape,
        Field.VOXEL_SIZES: np.linalg.norm(affine[:3, :3], axis=0),
        Field.VOXEL_ORDER: "".join(nib.aff2axcodes(affine)).encode(),
    }
    TrkFile(tractogram, header).save(path)
    return path


def _header(path: Path) -> np.ndarray:
    header: np.ndarray = np.frombuffer(path.read_bytes()[:1000], HEADER)[0]
    return header


def _patch(path: Path, **fields: object) -> Path:
    raw = path.read_bytes()
    header = np.frombuffer(raw[:1000], HEADER).copy()
    for name, value in fields.items():
        header[name] = value
    path.write_bytes(header.tobytes() + raw[1000:])
    return path


def _blocks(raw: bytes) -> list[tuple[int, int]]:
    # (start, end) byte range of each streamline in a little-endian file.
    header = np.frombuffer(raw[:1000], HEADER)[0]
    per_point = 3 + int(header[Field.NB_SCALARS_PER_POINT])
    per_streamline = int(header[Field.NB_PROPERTIES_PER_STREAMLINE])
    blocks, pos = [], 1000
    while pos < len(raw):
        n = int(np.frombuffer(raw, "<i4", 1, pos)[0])
        end = pos + 4 + 4 * (n * per_point + per_streamline)
        blocks.append((pos, end))
        pos = end
    return blocks


def _big_endian(raw: bytes) -> bytes:
    header = np.frombuffer(raw[:1000], HEADER)
    out = [header.astype(header_2_dtype.newbyteorder(">")).tobytes()]
    for start, end in _blocks(raw):
        out.append(np.frombuffer(raw[start : start + 4], "<i4").astype(">i4").tobytes())
        out.append(np.frombuffer(raw[start + 4 : end], "<f4").astype(">f4").tobytes())
    return b"".join(out)


def _nib_load(path: Path) -> TrkFile:
    loaded: TrkFile = TrkFile.load(path)
    return loaded


def _assert_same_as_nibabel(t: Tractogram, loaded: TrkFile) -> None:
    lengths = [len(s) for s in loaded.streamlines]
    np.testing.assert_array_equal(t.offsets, np.concatenate([[0], np.cumsum(lengths)]))
    np.testing.assert_allclose(
        t.points, loaded.streamlines.get_data(), rtol=RTOL, atol=ATOL
    )
    per_point = loaded.tractogram.data_per_point
    per_streamline = loaded.tractogram.data_per_streamline
    assert set(t.point_data) == set(per_point)
    assert set(t.streamline_data) == set(per_streamline)
    for name, values in t.point_data.items():
        np.testing.assert_array_equal(_column(values), per_point[name].get_data())
    for name, values in t.streamline_data.items():
        np.testing.assert_array_equal(_column(values), per_streamline[name])


def _assert_same_tractogram(t: Tractogram, expected: Tractogram) -> None:
    np.testing.assert_array_equal(t.offsets, expected.offsets)
    np.testing.assert_allclose(t.points, expected.points, rtol=RTOL, atol=ATOL)
    assert set(t.point_data) == set(expected.point_data)
    assert set(t.streamline_data) == set(expected.streamline_data)
    for name, values in expected.point_data.items():
        np.testing.assert_array_equal(t.point_data[name], values, strict=True)
    for name, values in expected.streamline_data.items():
        np.testing.assert_array_equal(t.streamline_data[name], values, strict=True)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


# Reading


@EACH_AFFINE
def test_points_and_data_read_as_nibabel(
    tmp_path: Path, affine: np.ndarray, caplog: pytest.LogCaptureFixture
) -> None:
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    with caplog.at_level(logging.INFO, logger="larmor"):
        t = read_trk(path)
    _assert_same_as_nibabel(t, _nib_load(path))
    assert caplog.records == []


@EACH_AFFINE
def test_read_points_are_the_saved_world_points(
    tmp_path: Path, affine: np.ndarray
) -> None:
    original = _tractogram(affine)
    t = read_trk(_nib_save(tmp_path / "a.trk", original, affine))
    _assert_same_tractogram(t, original)


def test_read_types_and_data_shapes(tmp_path: Path) -> None:
    affine = AFFINES["oblique"]
    t = read_trk(_nib_save(tmp_path / "a.trk", _tractogram(affine), affine))
    p, s = sum(LENGTHS), len(LENGTHS)
    assert t.points.dtype == np.float32
    assert t.offsets.dtype == np.int64
    assert t.point_data["fa"].shape == (p,)
    assert t.point_data["dir"].shape == (p, 3)
    assert t.streamline_data["length"].shape == (s,)
    assert t.streamline_data["rgb"].shape == (s, 3)


@EACH_AFFINE
def test_reference_is_the_header_grid(tmp_path: Path, affine: np.ndarray) -> None:
    t = read_trk(_nib_save(tmp_path / "a.trk", _tractogram(affine), affine))
    assert t.reference is not None
    assert t.reference.shape == SHAPE
    np.testing.assert_allclose(t.reference.affine, affine, rtol=0, atol=F32_ATOL)


@pytest.mark.parametrize("order", ["<", ">"], ids=["little", "big"])
@pytest.mark.parametrize("gz", [False, True], ids=["plain", "gzip"])
def test_byte_order_and_gzip_read_as_nibabel(
    tmp_path: Path, order: str, gz: bool
) -> None:
    affine = AFFINES["oblique"]
    plain = _nib_save(tmp_path / "plain.trk", _tractogram(affine), affine)
    raw = plain.read_bytes() if order == "<" else _big_endian(plain.read_bytes())
    path = tmp_path / ("a.trk.gz" if gz else "a.trk")
    path.write_bytes(gzip.compress(raw) if gz else raw)
    _assert_same_as_nibabel(read_trk(path), _nib_load(plain))


@pytest.mark.parametrize("as_str", [False, True], ids=["Path", "str"])
def test_read_accepts_str_and_path(tmp_path: Path, as_str: bool) -> None:
    affine = AFFINES["ras"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    _assert_same_as_nibabel(read_trk(str(path) if as_str else path), _nib_load(path))


@pytest.mark.parametrize(
    "affine_id", ["ras", "lps", "oblique", "permuted"], ids=lambda a: f"affine-{a}"
)
@pytest.mark.parametrize("voxel_order", ["LPS", "RAS", "LAS", "RPI", "SRA", "ASL"])
def test_voxel_order_disagreeing_with_affine_is_corrected_as_nibabel(
    tmp_path: Path, affine_id: str, voxel_order: str
) -> None:
    affine = AFFINES[affine_id]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    _patch(path, voxel_order=voxel_order.encode())
    _assert_same_as_nibabel(read_trk(path), _nib_load(path))


def test_lowercase_voxel_order_is_read_as_nibabel(tmp_path: Path) -> None:
    affine = AFFINES["ras"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    _patch(path, voxel_order=b"lps")
    _assert_same_as_nibabel(read_trk(path), _nib_load(path))


def test_empty_voxel_order_is_read_as_lps_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    affine = AFFINES["ras"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    _patch(path, voxel_order=b"")
    with caplog.at_level(logging.WARNING, logger="larmor"):
        t = read_trk(path)
    assert len(_warnings(caplog)) == 1
    with pytest.warns(Warning, match="LPS"):
        loaded = _nib_load(path)
    _assert_same_as_nibabel(t, loaded)


# nibabel refuses each of these; reading them anyway would misplace every point.
@pytest.mark.parametrize("voxel_order", [b"XYZ", b"LP", b"LLS", b"RAS2"])
def test_invalid_voxel_order_raises(tmp_path: Path, voxel_order: bytes) -> None:
    affine = AFFINES["ras"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    _patch(path, voxel_order=voxel_order)
    with pytest.raises(ValueError):
        read_trk(path)


@pytest.mark.parametrize(
    "field", [Field.NB_SCALARS_PER_POINT, Field.NB_PROPERTIES_PER_STREAMLINE]
)
@pytest.mark.parametrize("count", [-1, -5])
def test_negative_number_of_values_raises(
    tmp_path: Path, field: str, count: int
) -> None:
    affine = AFFINES["ras"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    _patch(path, **{field: count})
    with pytest.raises(ValueError):
        read_trk(path)


def test_version_3_is_read_as_version_2_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    affine = AFFINES["lps"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    _patch(path, version=3)
    with caplog.at_level(logging.WARNING, logger="larmor"):
        t = read_trk(path)
    assert len(_warnings(caplog)) == 1
    with pytest.warns(Warning):
        loaded = _nib_load(path)
    _assert_same_as_nibabel(t, loaded)


@pytest.mark.parametrize("version", [0, 4, -1, 2 << 16])
def test_unknown_version_raises(tmp_path: Path, version: int) -> None:
    affine = AFFINES["ras"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    _patch(path, version=version)
    with pytest.raises(ValueError):
        read_trk(path)


@pytest.mark.parametrize("magic", [b"TRACX", b"track", b"", b"TRAC"])
def test_wrong_magic_raises(tmp_path: Path, magic: bytes) -> None:
    affine = AFFINES["ras"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    _patch(path, magic_number=magic)
    with pytest.raises(ValueError):
        read_trk(path)


@pytest.mark.parametrize("hdr_size", [0, 999, 1001, 1000 << 16])
@pytest.mark.parametrize("order", ["<", ">"], ids=["little", "big"])
def test_header_size_other_than_1000_in_either_byte_order_raises(
    tmp_path: Path, hdr_size: int, order: str
) -> None:
    # 1000 << 16 is neither 1000 read little-endian nor 1000 read big-endian.
    affine = AFFINES["ras"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    raw = bytearray(
        path.read_bytes() if order == "<" else _big_endian(path.read_bytes())
    )
    raw[996:1000] = np.array([hdr_size], f"{order}i4").tobytes()
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        read_trk(path)


def test_file_shorter_than_header_raises(tmp_path: Path) -> None:
    affine = AFFINES["ras"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    path.write_bytes(path.read_bytes()[:500])
    with pytest.raises(ValueError):
        read_trk(path)


def _missing_affine(path: Path, how: str) -> Path:
    if how == "version 1":
        # Version 1 has no vox_to_ras: whatever those bytes hold is not an affine.
        return _patch(path, version=1, voxel_to_rasmm=np.eye(4))
    affine = _header(path)[Field.VOXEL_TO_RASMM].copy()
    affine[3, 3] = 0
    return _patch(path, voxel_to_rasmm=affine)


MISSING_AFFINE = pytest.mark.parametrize("how", ["version 1", "unset"])


@MISSING_AFFINE
def test_missing_affine_without_reference_raises(tmp_path: Path, how: str) -> None:
    affine = AFFINES["lps"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    _missing_affine(path, how)
    with pytest.raises(ValueError):
        read_trk(path)


@MISSING_AFFINE
@pytest.mark.parametrize("affine_id", ["lps", "oblique", "permuted"])
def test_missing_affine_is_taken_from_reference(
    tmp_path: Path, how: str, affine_id: str
) -> None:
    affine = AFFINES[affine_id]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    loaded = _nib_load(path)
    _missing_affine(path, how)
    reference = Grid(SHAPE, affine)
    t = read_trk(path, reference=reference)
    _assert_same_as_nibabel(t, loaded)
    assert t.reference is not None
    assert t.reference.matches(reference)


@pytest.mark.parametrize("as_canonical", [False, True], ids=["same", "reoriented"])
def test_reference_matching_header_is_accepted(
    tmp_path: Path, as_canonical: bool
) -> None:
    affine = AFFINES["lps"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    grid = Grid(SHAPE, affine)
    t = read_trk(path, reference=grid.canonical() if as_canonical else grid)
    _assert_same_as_nibabel(t, _nib_load(path))
    assert t.reference is not None
    assert t.reference.shape == SHAPE
    np.testing.assert_allclose(t.reference.affine, affine, rtol=0, atol=F32_ATOL)


def _shifted(affine: np.ndarray) -> np.ndarray:
    moved = affine.copy()
    moved[:3, 3] += moved[:3, 0]
    return moved


@pytest.mark.parametrize(
    "reference",
    [
        Grid((4, 5, 7), AFFINES["lps"]),
        Grid(SHAPE, _shifted(AFFINES["lps"])),
        Grid(SHAPE, AFFINES["ras"]),
    ],
    ids=["shape", "position", "direction"],
)
def test_reference_not_matching_header_raises(tmp_path: Path, reference: Grid) -> None:
    affine = AFFINES["lps"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    with pytest.raises(ValueError):
        read_trk(path, reference=reference)


@pytest.mark.parametrize(
    "linear",
    [
        np.array([[1.0, 1.0, 0], [1.0, 1.0, 0], [0, 0, 1.0]]),
        np.diag([2.0, 0.0, 3.0]),
    ],
    ids=["rank-2", "zero-axis"],
)
def test_singular_affine_raises(tmp_path: Path, linear: np.ndarray) -> None:
    affine = AFFINES["ras"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    _patch(path, voxel_to_rasmm=_affine(linear))
    with pytest.raises(ValueError):
        read_trk(path)


def test_values_not_covered_by_names_are_read_as_nibabel(tmp_path: Path) -> None:
    affine = AFFINES["ras"]
    original = _tractogram(affine)
    path = _nib_save(tmp_path / "a.trk", original, affine)
    header = _header(path)
    # Sorted names: scalars are dir (3 values) then fa; properties length then rgb.
    assert header["scalar_name"][0] == b"dir\x003"
    assert header["property_name"][1] == b"rgb\x003"
    scalar_names = header["scalar_name"].copy()
    scalar_names[1] = b""
    property_names = header["property_name"].copy()
    property_names[1] = b""
    _patch(path, scalar_name=scalar_names, property_name=property_names)

    t = read_trk(path)
    _assert_same_as_nibabel(t, _nib_load(path))
    assert set(t.point_data) == {"dir", "scalars"}
    assert set(t.streamline_data) == {"length", "properties"}
    np.testing.assert_array_equal(
        _column(t.point_data["scalars"]), _column(original.point_data["fa"])
    )
    np.testing.assert_array_equal(
        t.streamline_data["properties"], original.streamline_data["rgb"]
    )


def test_values_without_any_name_go_to_scalars_and_properties(
    tmp_path: Path,
) -> None:
    affine = AFFINES["ras"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    empty = np.zeros(10, "S20")
    _patch(path, scalar_name=empty, property_name=empty)
    t = read_trk(path)
    _assert_same_as_nibabel(t, _nib_load(path))
    assert t.point_data["scalars"].shape == (sum(LENGTHS), 4)
    assert t.streamline_data["properties"].shape == (len(LENGTHS), 4)


def test_streamlines_without_points_are_kept(tmp_path: Path) -> None:
    affine = AFFINES["oblique"]
    original = _tractogram(affine)
    path = _nib_save(tmp_path / "a.trk", original, affine)
    raw = path.read_bytes()
    # A 0-point streamline is its count and its 4 properties.
    first = np.array([0], "<i4").tobytes() + np.arange(4, dtype="<f4").tobytes()
    last = np.array([0], "<i4").tobytes() + np.full(4, 7, "<f4").tobytes()
    path.write_bytes(raw[:1000] + first + raw[1000:] + last)
    _patch(path, nb_streamlines=len(LENGTHS) + 2)

    t = read_trk(path)
    np.testing.assert_array_equal(np.diff(t.offsets), [0, *LENGTHS, 0])
    np.testing.assert_allclose(t.points, original.points, rtol=RTOL, atol=ATOL)
    for name, values in original.point_data.items():
        np.testing.assert_array_equal(t.point_data[name], values)
    # Sorted property names: length (1 value) then rgb (3 values).
    np.testing.assert_array_equal(
        t.streamline_data["length"][[0, -1]], np.array([0, 7], np.float32)
    )
    np.testing.assert_array_equal(
        t.streamline_data["rgb"][[0, -1]], np.array([[1, 2, 3], [7, 7, 7]], np.float32)
    )
    np.testing.assert_array_equal(
        t.streamline_data["rgb"][1:-1], original.streamline_data["rgb"]
    )


def test_empty_file_reads_as_empty_tractogram(tmp_path: Path) -> None:
    affine = AFFINES["ras"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine, lengths=()), affine)
    t = read_trk(path)
    assert t.num_streamlines == 0
    assert t.points.shape == (0, 3)
    assert t.reference is not None


@pytest.mark.parametrize("count", [0, len(LENGTHS)], ids=["unknown", "correct"])
def test_unknown_or_correct_count_reads_every_streamline_silently(
    tmp_path: Path, count: int, caplog: pytest.LogCaptureFixture
) -> None:
    affine = AFFINES["oblique"]
    original = _tractogram(affine)
    path = _nib_save(tmp_path / "a.trk", original, affine)
    _patch(path, nb_streamlines=count)
    with caplog.at_level(logging.INFO, logger="larmor"):
        t = read_trk(path)
    _assert_same_tractogram(t, original)
    assert caplog.records == []


@pytest.mark.parametrize("count", [1, len(LENGTHS) - 1, 97], ids=["1", "S-1", "97"])
def test_wrong_count_reads_to_the_end_with_warning(
    tmp_path: Path, count: int, caplog: pytest.LogCaptureFixture
) -> None:
    affine = AFFINES["oblique"]
    original = _tractogram(affine)
    path = _nib_save(tmp_path / "a.trk", original, affine)
    _patch(path, nb_streamlines=count)
    with caplog.at_level(logging.INFO, logger="larmor"):
        t = read_trk(path)
    _assert_same_tractogram(t, original)
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert f"n_count is {count}" in caplog.text
    assert "incomplete" in caplog.text


@pytest.mark.parametrize(
    "kept",
    [lambda n: 2, lambda n: 4, lambda n: 4 + 7, lambda n: n - 2],
    ids=["in-count", "after-count", "in-points", "in-properties"],
)
def test_file_cut_inside_last_streamline_drops_it_with_warning(
    tmp_path: Path, kept: Callable[[int], int], caplog: pytest.LogCaptureFixture
) -> None:
    affine = AFFINES["oblique"]
    original = _tractogram(affine)
    path = _nib_save(tmp_path / "a.trk", original, affine)
    raw = path.read_bytes()
    start, end = _blocks(raw)[-1]
    path.write_bytes(raw[: start + kept(end - start)])
    with caplog.at_level(logging.WARNING, logger="larmor"):
        t = read_trk(path)
    _assert_same_tractogram(t, original[: len(LENGTHS) - 1])
    # The cut, and n_count (S) disagreeing with the S - 1 streamlines read.
    messages = [r.getMessage() for r in _warnings(caplog)]
    assert len(messages) == 2
    assert "ends inside its last streamline" in messages[0]
    assert f"n_count is {len(LENGTHS)}" in messages[1]


@pytest.mark.parametrize("which", [0, -1], ids=["first", "last"])
def test_negative_point_count_raises(tmp_path: Path, which: int) -> None:
    affine = AFFINES["ras"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    raw = bytearray(path.read_bytes())
    start, _ = _blocks(bytes(raw))[which]
    raw[start : start + 4] = np.array([-1], "<i4").tobytes()
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        read_trk(path)


# Writing


@EACH_AFFINE
def test_written_header_has_the_grid_and_counts(
    tmp_path: Path, affine: np.ndarray
) -> None:
    path = tmp_path / "a.trk"
    write_trk(_tractogram(affine), path, reference=Grid(SHAPE, affine))
    header = _header(path)
    assert header[Field.MAGIC_NUMBER] == b"TRACK"
    assert tuple(header[Field.DIMENSIONS]) == SHAPE
    np.testing.assert_allclose(
        header[Field.VOXEL_SIZES], np.linalg.norm(affine[:3, :3], axis=0), rtol=1e-6
    )
    np.testing.assert_array_equal(header[Field.ORIGIN], [0, 0, 0])
    np.testing.assert_allclose(
        header[Field.VOXEL_TO_RASMM], affine, rtol=0, atol=F32_ATOL
    )
    assert header[Field.VOXEL_ORDER] == "".join(nib.aff2axcodes(affine)).encode()
    assert header["version"] == 2
    assert header["hdr_size"] == 1000
    assert header[Field.NB_STREAMLINES] == len(LENGTHS)
    assert header[Field.NB_SCALARS_PER_POINT] == 4
    assert header[Field.NB_PROPERTIES_PER_STREAMLINE] == 4
    assert list(header["scalar_name"]) == [b"dir\x003", b"fa"] + [b""] * 8
    assert list(header["property_name"]) == [b"length", b"rgb\x003"] + [b""] * 8
    point_size = 4 * (3 + 4)
    assert path.stat().st_size == 1000 + 4 * len(LENGTHS) * (1 + 4) + point_size * sum(
        LENGTHS
    )


@EACH_AFFINE
def test_nibabel_reads_written_file_as_the_tractogram(
    tmp_path: Path, affine: np.ndarray
) -> None:
    original = _tractogram(affine)
    path = tmp_path / "a.trk"
    write_trk(original, path, reference=Grid(SHAPE, affine))
    _assert_same_as_nibabel(original, _nib_load(path))


@EACH_AFFINE
def test_round_trip_keeps_points_data_and_reference(
    tmp_path: Path, affine: np.ndarray
) -> None:
    original = _tractogram(affine)
    grid = Grid(SHAPE, affine)
    path = tmp_path / "a.trk"
    write_trk(original, path, reference=grid)
    t = read_trk(path)
    _assert_same_tractogram(t, original)
    assert t.reference is not None
    assert t.reference.matches(grid)


def test_world_points_survive_trk_within_float32_precision(tmp_path: Path) -> None:
    # Points as a TCK holds them: world mm, here far from the grid and the origin.
    rng = np.random.default_rng(1)
    points = rng.uniform(-120, 120, (40, 3)).astype(np.float32)
    original = Tractogram(points, np.array([0, 10, 11, 30, 40]))
    path = tmp_path / "a.trk"
    write_trk(original, path, reference=Grid(SHAPE, AFFINES["oblique"]))
    t = read_trk(path)
    np.testing.assert_array_equal(t.offsets, original.offsets)
    # float32 spacing at 120 mm is ~8e-6; a few roundings stay well under 1e-4.
    np.testing.assert_allclose(t.points, original.points, rtol=0, atol=1e-4)
    assert t.point_data == {}
    assert t.streamline_data == {}


def test_round_trip_holds_when_float32_rounding_flips_the_orientation(
    tmp_path: Path,
) -> None:
    # 45 degrees about z is a tie between two axis codes; float32 rounding of the
    # stored affine breaks it the other way, and readers use the stored affine.
    angle = math.pi / 4 + 1e-9
    affine = np.eye(4)
    affine[:3, :3] = 2.0 * _rotation(angle, 0.0)
    affine[:3, 3] = [-50, -60, -40]
    shape = (40, 50, 30)
    grid = Grid(shape, affine)
    stored = Grid(shape, affine.astype(np.float32).astype(np.float64))
    assert grid.orientation != stored.orientation
    points = np.array([[1, 2, 3], [10, -5, 7]], dtype=np.float32)
    path = tmp_path / "a.trk"
    write_trk(Tractogram(points, np.array([0, 2])), path, reference=grid)
    np.testing.assert_allclose(read_trk(path).points, points, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        _nib_load(path).streamlines.get_data(), points, rtol=0, atol=1e-4
    )


def test_tck_to_trk_and_back_keeps_the_points(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    points = rng.uniform(-120, 120, (40, 3)).astype(np.float32)
    write_tck(Tractogram(points, np.array([0, 10, 10, 30, 40])), tmp_path / "a.tck")
    from_tck = read_tck(tmp_path / "a.tck")
    write_trk(from_tck, tmp_path / "a.trk", reference=Grid(SHAPE, AFFINES["oblique"]))
    write_tck(read_trk(tmp_path / "a.trk"), tmp_path / "b.tck")
    back = read_tck(tmp_path / "b.tck")
    np.testing.assert_array_equal(back.offsets, from_tck.offsets)
    np.testing.assert_allclose(back.points, from_tck.points, rtol=0, atol=1e-4)


def test_streamlines_without_points_round_trip(tmp_path: Path) -> None:
    original = _tractogram(lengths=(0, 3, 0, 2, 0))
    path = tmp_path / "a.trk"
    write_trk(original, path, reference=Grid(SHAPE, AFFINES["oblique"]))
    _assert_same_tractogram(read_trk(path), original)


def test_empty_tractogram_round_trips(tmp_path: Path) -> None:
    original = Tractogram(np.zeros((0, 3)), np.array([0]))
    path = tmp_path / "a.trk"
    write_trk(original, path, reference=Grid(SHAPE, AFFINES["ras"]))
    t = read_trk(path)
    assert t.num_streamlines == 0
    assert t.points.shape == (0, 3)


def test_larmor_reads_nibabel_written_file_identically_to_its_own(
    tmp_path: Path,
) -> None:
    affine = AFFINES["permuted"]
    original = _tractogram(affine)
    ours = tmp_path / "ours.trk"
    write_trk(original, ours, reference=Grid(SHAPE, affine))
    theirs = _nib_save(tmp_path / "theirs.trk", original, affine)
    a, b = read_trk(ours), read_trk(theirs)
    _assert_same_tractogram(a, b)


def test_data_is_stored_as_float32(tmp_path: Path) -> None:
    original = _tractogram(lengths=(2, 3))
    original.point_data = {"label": np.arange(5, dtype=np.int64)}
    original.streamline_data = {"weight": np.array([0.1, 1 / 3])}
    path = tmp_path / "a.trk"
    write_trk(original, path, reference=Grid(SHAPE, AFFINES["oblique"]))
    t = read_trk(path)
    np.testing.assert_array_equal(
        t.point_data["label"], np.arange(5, dtype=np.float32), strict=True
    )
    np.testing.assert_array_equal(
        t.streamline_data["weight"], np.array([0.1, 1 / 3], np.float32), strict=True
    )


def test_reference_argument_is_used_over_the_tractogram_reference(
    tmp_path: Path,
) -> None:
    own = Grid((7, 8, 9), AFFINES["ras"])
    given = Grid(SHAPE, AFFINES["oblique"])
    original = _tractogram(reference=own)
    path = tmp_path / "a.trk"
    write_trk(original, path, reference=given)
    header = _header(path)
    assert tuple(header[Field.DIMENSIONS]) == SHAPE
    np.testing.assert_allclose(
        header[Field.VOXEL_TO_RASMM], given.affine, rtol=0, atol=F32_ATOL
    )
    _assert_same_tractogram(read_trk(path), original)


def test_tractogram_reference_is_used_without_reference_argument(
    tmp_path: Path,
) -> None:
    grid = Grid(SHAPE, AFFINES["lps"])
    original = _tractogram(AFFINES["lps"], reference=grid)
    path = tmp_path / "a.trk"
    write_trk(original, path)
    header = _header(path)
    assert tuple(header[Field.DIMENSIONS]) == SHAPE
    np.testing.assert_allclose(
        header[Field.VOXEL_TO_RASMM], grid.affine, rtol=0, atol=F32_ATOL
    )
    _assert_same_tractogram(read_trk(path), original)


def test_no_reference_at_all_raises_before_writing(tmp_path: Path) -> None:
    path = tmp_path / "a.trk"
    with pytest.raises(ValueError):
        write_trk(_tractogram(), path)
    assert not path.exists()


def test_read_filtered_and_written_without_reference(tmp_path: Path) -> None:
    affine = AFFINES["permuted"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    t = read_trk(path)
    mask = np.array([True, False, True, True])
    out = tmp_path / "b.trk"
    write_trk(t[mask], out)
    back = read_trk(out)
    _assert_same_tractogram(back, t[mask])
    assert back.reference is not None
    assert back.reference.matches(Grid(SHAPE, affine))


@pytest.mark.parametrize(
    ("size", "fits"), [(32767, True), (32768, False), (70000, False)]
)
def test_grid_shape_must_fit_int16(tmp_path: Path, size: int, fits: bool) -> None:
    path = tmp_path / "a.trk"
    grid = Grid((size, 2, 3), AFFINES["ras"])
    if fits:
        write_trk(_tractogram(), path, reference=grid)
        assert tuple(_header(path)[Field.DIMENSIONS]) == (size, 2, 3)
    else:
        with pytest.raises(ValueError):
            write_trk(_tractogram(), path, reference=grid)


DATA_KINDS = pytest.mark.parametrize("kind", ["point_data", "streamline_data"])


def _with_data(kind: str, data: dict[str, np.ndarray]) -> Tractogram:
    t = _tractogram()
    setattr(t, kind, data)
    return t


def _values(kind: str, k: int) -> np.ndarray:
    n = sum(LENGTHS) if kind == "point_data" else len(LENGTHS)
    values = np.arange(n * k, dtype=np.float32)
    return values if k == 1 else values.reshape(n, k)


@DATA_KINDS
@pytest.mark.parametrize(("count", "fits"), [(10, True), (11, False)])
def test_at_most_ten_names(tmp_path: Path, kind: str, count: int, fits: bool) -> None:
    data = {f"v{i:02d}": _values(kind, 1) for i in range(count)}
    t = _with_data(kind, data)
    path = tmp_path / "a.trk"
    if fits:
        write_trk(t, path, reference=Grid(SHAPE, AFFINES["oblique"]))
        assert set(getattr(read_trk(path), kind)) == set(data)
    else:
        with pytest.raises(ValueError):
            write_trk(t, path, reference=Grid(SHAPE, AFFINES["oblique"]))


@DATA_KINDS
@pytest.mark.parametrize(
    ("length", "k", "fits"),
    [(20, 1, True), (21, 1, False), (18, 2, True), (18, 10, False), (19, 2, False)],
)
def test_encoded_name_must_fit_20_bytes(
    tmp_path: Path, kind: str, length: int, k: int, fits: bool
) -> None:
    name = "n" * length
    t = _with_data(kind, {name: _values(kind, k)})
    path = tmp_path / "a.trk"
    if fits:
        write_trk(t, path, reference=Grid(SHAPE, AFFINES["oblique"]))
        np.testing.assert_array_equal(
            getattr(read_trk(path), kind)[name], _values(kind, k)
        )
        field = "scalar_name" if kind == "point_data" else "property_name"
        assert (
            _header(path)[field][0] == (name if k == 1 else f"{name}\x00{k}").encode()
        )
    else:
        with pytest.raises(ValueError):
            write_trk(t, path, reference=Grid(SHAPE, AFFINES["oblique"]))


@pytest.mark.parametrize("as_str", [False, True], ids=["Path", "str"])
def test_write_accepts_str_and_path_and_overwrites(
    tmp_path: Path, as_str: bool
) -> None:
    path = tmp_path / "a.trk"
    grid = Grid(SHAPE, AFFINES["oblique"])
    write_trk(_tractogram(lengths=(30, 30, 30)), path, reference=grid)
    small = _tractogram(lengths=(2, 1))
    write_trk(small, str(path) if as_str else path, reference=grid)
    _assert_same_tractogram(read_trk(path), small)


def test_nibabel_affine_formula_agrees_on_written_header(tmp_path: Path) -> None:
    # The written header is self-consistent: nibabel's voxmm-to-world affine applied
    # to the stored voxmm gives back the world points.
    affine = AFFINES["oblique"]
    original = _tractogram(affine)
    path = tmp_path / "a.trk"
    write_trk(original, path, reference=Grid(SHAPE, affine))
    raw = path.read_bytes()
    header = dict(zip(HEADER.names or (), _header(path), strict=True))
    to_world = get_affine_trackvis_to_rasmm(header)
    rows = [
        np.frombuffer(raw, "<f4", n * 7, start + 4).reshape(n, 7)[:, :3]
        for (start, _), n in zip(_blocks(raw), LENGTHS, strict=True)
    ]
    voxmm = np.concatenate(rows)
    world = voxmm @ to_world[:3, :3].T + to_world[:3, 3]
    np.testing.assert_allclose(world, original.points, rtol=RTOL, atol=ATOL)


def test_filtered_to_nothing_with_data_round_trips(tmp_path: Path) -> None:
    affine = AFFINES["ras"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    empty = read_trk(path)[np.zeros(len(LENGTHS), dtype=bool)]
    write_trk(empty, tmp_path / "b.trk")
    t = read_trk(tmp_path / "b.trk")
    assert t.num_streamlines == 0
    assert t.point_data["dir"].shape == (0, 3)
    assert t.point_data["fa"].shape == (0,)
    assert t.streamline_data["rgb"].shape == (0, 3)
    assert t.streamline_data["length"].shape == (0,)


@pytest.mark.parametrize("name", [b"fa\x005", b"fa\x00x", b"fa\x00-1"])
def test_names_not_fitting_the_header_raise(tmp_path: Path, name: bytes) -> None:
    affine = AFFINES["ras"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    names = _header(path)["scalar_name"].copy()
    assert names[1] == b"fa"
    names[1] = name
    _patch(path, scalar_name=names)
    with pytest.raises(ValueError, match="a.trk"):
        read_trk(path)


def test_name_with_zero_values_is_skipped_as_nibabel(tmp_path: Path) -> None:
    affine = AFFINES["ras"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    names = _header(path)["scalar_name"].copy()
    names[1] = b"fa\x000"
    _patch(path, scalar_name=names)
    t = read_trk(path)
    _assert_same_as_nibabel(t, _nib_load(path))
    assert set(t.point_data) == {"dir", "scalars"}


@pytest.mark.parametrize(
    "reference",
    [
        Grid((4, 5, 7), AFFINES["lps"]),
        Grid(SHAPE, AFFINES["lps"] @ np.diag([2.0, 1.0, 1.0, 1.0])),
    ],
    ids=["shape", "voxel size"],
)
def test_reference_not_matching_version_1_header_raises(
    tmp_path: Path, reference: Grid
) -> None:
    affine = AFFINES["lps"]
    path = _nib_save(tmp_path / "a.trk", _tractogram(affine), affine)
    _missing_affine(path, "version 1")
    with pytest.raises(ValueError, match="does not match"):
        read_trk(path, reference=reference)


@pytest.mark.parametrize("name", ["", "a\x00b", "Δ"])
@pytest.mark.parametrize("kind", ["point_data", "streamline_data"])
def test_names_trk_cannot_store_raise(tmp_path: Path, kind: str, name: str) -> None:
    t = _tractogram(AFFINES["ras"], reference=Grid(SHAPE, AFFINES["ras"]))
    data = getattr(t, kind)
    data[name] = data.pop("fa" if kind == "point_data" else "length")
    path = tmp_path / "a.trk"
    with pytest.raises(ValueError, match="name"):
        write_trk(t, path)
    assert not path.exists()
