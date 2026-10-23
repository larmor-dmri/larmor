import gzip
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from larmor.io.formats import read_tck, read_vtk, write_tck, write_vtk
from larmor.types import Tractogram

DATA = Path(__file__).parents[2] / "data"
REF_TCK = DATA / "tck" / "ref.tck"
REF_VTK = DATA / "vtk"

needs_mrtrix = pytest.mark.skipif(
    shutil.which("tckconvert") is None, reason="MRtrix3 (tckconvert) is not on PATH"
)

MODES = ["ASCII", "BINARY"]

Values = Sequence[float] | np.ndarray
Section = tuple[str, Values | None, str]

# Six stored points; the lines use them out of order, share point 2, skip points 1
# and 3, and have an empty line in the middle.
PTS = (np.arange(18.0).reshape(6, 3) - 7) / 8
LEGACY = [3, 4, 0, 2, 0, 2, 2, 5]
CONN = [4, 0, 2, 2, 5]
OFFSETS = np.array([0, 3, 3, 5])
POINTS: Section = ("POINTS 6 float", PTS, ">f4")
LINES: Section = ("LINES 3 8", LEGACY, ">i4")

PD1 = np.arange(6.0) + 10
PD3 = np.arange(18.0).reshape(6, 3) / 4 - 2
CD1 = np.array([1.5, 2.5, 3.5])
CD3 = np.arange(9.0).reshape(3, 3) - 4

# Lengths with empty streamlines at the start, in the middle and at the end.
LENGTHS = [0, 4, 1, 0, 0, 3, 0]
# MRtrix breaks on empty streamlines in VTK, so it only ever sees these.
MRTRIX_LENGTHS = [5, 1, 3, 2, 7]


def _h(header: str) -> Section:
    return (header, None, "")


def _body(mode: str, sections: list[Section]) -> bytes:
    out = []
    for header, values, dtype in sections:
        out.append(header.encode() + b"\n")
        if values is None:
            continue
        array = np.asarray(values).astype(dtype).ravel()
        if mode == "BINARY":
            out.append(array.tobytes() + b"\n")
        else:
            out.append(" ".join(repr(v) for v in array.tolist()).encode() + b"\n")
    return b"".join(out)


def _vtk(
    path: Path,
    mode: str,
    sections: list[Section],
    version: str = "3.0",
    title: str = "test file",
) -> Path:
    head = f"# vtk DataFile Version {version}\n{title}\n{mode}\nDATASET POLYDATA\n"
    path.write_bytes(head.encode() + _body(mode, sections))
    return path


def _tractogram(
    lengths: list[int],
    seed: int = 0,
    data: bool = False,
    eighths: bool = False,
) -> Tractogram:
    rng = np.random.default_rng(seed)
    n_points, n_lines = sum(lengths), len(lengths)
    if eighths:
        # Multiples of 1/8: exact in float32 and in MRtrix's 6-digit text output.
        points = rng.integers(-8000, 8000, size=(n_points, 3)) / 8
    else:
        # Full float32 values, so that any rounding in the text output shows.
        points = (rng.standard_normal((n_points, 3)) * 50).astype(np.float32)
    offsets = np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)])
    if not data:
        return Tractogram(points, offsets)

    def values(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float32)

    return Tractogram(
        points,
        offsets,
        point_data={"fa": values(n_points), "peak": values(n_points, 3)},
        streamline_data={"length": values(n_lines), "rgb": values(n_lines, 3)},
    )


def _assert_same(got: Tractogram, want: Tractogram) -> None:
    assert got.points.dtype == np.float32
    np.testing.assert_array_equal(got.offsets, want.offsets)
    np.testing.assert_array_equal(got.points, want.points)
    for got_data, want_data in (
        (got.point_data, want.point_data),
        (got.streamline_data, want.streamline_data),
    ):
        assert got_data.keys() == want_data.keys()
        for name, values in want_data.items():
            assert got_data[name].shape == values.shape
            np.testing.assert_array_equal(got_data[name], values)


def _gathered() -> Tractogram:
    return Tractogram(PTS[CONN], OFFSETS)


def _squeeze(values: np.ndarray) -> np.ndarray:
    return values[:, 0] if values.shape[1] == 1 else values


def _ref() -> Tractogram:
    return read_tck(REF_TCK)


def _tckconvert(*args: str | Path) -> None:
    subprocess.run(
        ["tckconvert", "-quiet", "-force", *map(str, args)],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )


# Reading


@pytest.mark.parametrize("name", ["ascii.vtk", "binary.vtk"])
def test_mrtrix_reference_file_is_read(name: str) -> None:
    got = read_vtk(REF_VTK / name)
    _assert_same(got, _ref())
    assert got.reference is None


@pytest.mark.parametrize("as_str", [False, True])
def test_path_may_be_str_or_path(as_str: bool) -> None:
    path = REF_VTK / "binary.vtk"
    _assert_same(read_vtk(str(path) if as_str else path), _ref())


@pytest.mark.parametrize("ref", ["ascii.vtk", "binary.vtk"])
@pytest.mark.parametrize("name", ["t.vtk.gz", "t.vtk", "t.fib"])
def test_gzip_file_is_read(tmp_path: Path, ref: str, name: str) -> None:
    path = tmp_path / name
    path.write_bytes(gzip.compress((REF_VTK / ref).read_bytes()))
    _assert_same(read_vtk(path), _ref())


@needs_mrtrix
@pytest.mark.parametrize("binary", [False, True])
def test_file_written_by_tckconvert_is_read_exactly(
    tmp_path: Path, binary: bool
) -> None:
    want = _tractogram(MRTRIX_LENGTHS, seed=4, eighths=True)
    write_tck(want, tmp_path / "t.tck")
    _tckconvert(
        *(["-binary"] if binary else []), tmp_path / "t.tck", tmp_path / "t.vtk"
    )
    assert (b"BINARY" if binary else b"ASCII") in (tmp_path / "t.vtk").read_bytes()
    _assert_same(read_vtk(tmp_path / "t.vtk"), want)


@pytest.mark.parametrize("mode", MODES)
def test_points_are_gathered_in_line_order(tmp_path: Path, mode: str) -> None:
    path = _vtk(tmp_path / "t.vtk", mode, [POINTS, LINES])
    _assert_same(read_vtk(path), _gathered())


LPS_TO_RAS = np.array([-1.0, -1.0, 1.0], dtype=np.float32)


@pytest.mark.parametrize(
    ("title", "flip"),
    [("vtk output SPACE=LPS", True), ("SPACE=RAS", False), ("test file", False)],
)
def test_space_tag_in_the_title_is_followed(
    tmp_path: Path, title: str, flip: bool
) -> None:
    path = _vtk(tmp_path / "t.vtk", "ASCII", [POINTS, LINES], title=title)
    want = _gathered().points * (LPS_TO_RAS if flip else 1)
    np.testing.assert_array_equal(read_vtk(path).points, want)


@pytest.mark.parametrize(
    ("title", "space", "flip"),
    [("test file", "LPS", True), ("SPACE=LPS", "RAS", False), ("x", "lps", True)],
)
def test_space_option_says_which_space_the_file_is_in(
    tmp_path: Path, title: str, space: str, flip: bool
) -> None:
    path = _vtk(tmp_path / "t.vtk", "ASCII", [POINTS, LINES], title=title)
    want = _gathered().points * (LPS_TO_RAS if flip else 1)
    np.testing.assert_array_equal(read_vtk(path, space=space).points, want)


def test_unknown_space_option_raises(tmp_path: Path) -> None:
    path = _vtk(tmp_path / "t.vtk", "ASCII", [POINTS, LINES])
    with pytest.raises(ValueError, match="RAS or LPS"):
        read_vtk(path, space="LAS")


@pytest.mark.parametrize("mode", MODES)
def test_points_used_in_stored_order_are_read(tmp_path: Path, mode: str) -> None:
    lines = ("LINES 3 9", [2, 0, 1, 0, 4, 2, 3, 4, 5], ">i4")
    path = _vtk(tmp_path / "t.vtk", mode, [POINTS, lines])
    _assert_same(read_vtk(path), Tractogram(PTS, np.array([0, 2, 2, 6])))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("type_name", ["vtktypeint64", "vtktypeint32"])
def test_version_5_1_lines_are_read(tmp_path: Path, mode: str, type_name: str) -> None:
    dtype = ">i8" if type_name == "vtktypeint64" else ">i4"
    sections = [
        POINTS,
        _h("LINES 4 5"),
        (f"OFFSETS {type_name}", OFFSETS, dtype),
        (f"CONNECTIVITY {type_name}", CONN, dtype),
    ]
    path = _vtk(tmp_path / "t.vtk", mode, sections, version="5.1")
    _assert_same(read_vtk(path), _gathered())


@pytest.mark.parametrize("mode", MODES)
def test_double_points_are_read_as_float32(tmp_path: Path, mode: str) -> None:
    # 0.1 and 1/3 are not float32 values: an unconverted read keeps float64.
    points = np.array([[0.1, 1 / 3, -2.7], [1e-7, 123.456789, 5.0]])
    sections: list[Section] = [
        ("POINTS 2 double", points, ">f8"),
        ("LINES 1 3", [2, 0, 1], ">i4"),
    ]
    got = read_vtk(_vtk(tmp_path / "t.vtk", mode, sections))
    _assert_same(got, Tractogram(points.astype(np.float32), np.array([0, 2])))


@pytest.mark.parametrize("mode", ["ascii", "Ascii", "binary", "Binary"])
def test_mode_case_is_ignored(tmp_path: Path, mode: str) -> None:
    body = _body(mode.upper(), [POINTS, LINES])
    head = f"# vtk DataFile Version 3.0\ntitle\n{mode}\nDATASET POLYDATA\n"
    path = tmp_path / "t.vtk"
    path.write_bytes(head.encode() + body)
    _assert_same(read_vtk(path), _gathered())


@pytest.mark.parametrize("version", ["2.0", "3.0", "4.2", "5.1"])
def test_any_version_number_is_accepted(tmp_path: Path, version: str) -> None:
    path = _vtk(tmp_path / "t.vtk", "ASCII", [POINTS, LINES], version=version)
    _assert_same(read_vtk(path), _gathered())


def test_ascii_layout_is_free(tmp_path: Path) -> None:
    separators = [" ", "\r\n", "   ", "\t", "\r\n\r\n  ", " \t "]
    numbers = [repr(v) for v in PTS.ravel().tolist()]
    spread = "".join(n + separators[i % len(separators)] for i, n in enumerate(numbers))
    text = (
        "# vtk DataFile Version 3.0\r\n"
        "  a title, with  spaces  \r\n"
        "ascii\r\n"
        "DATASET POLYDATA\r\n"
        "POINTS   6   float\r\n"
        f"{spread}\r\n\r\n"
        "LINES 3 8\r\n"
        "3\r\n4 0\r\n  2 0 2\r\n\t2 5\r\n\r\n"
        "CELL_DATA 3\r\n"
        "FIELD FieldData 1\r\n"
        "len 1 3 float\r\n"
        "1.5\r\n\r\n 2.5   3.5  \r\n"
    )
    path = tmp_path / "t.vtk"
    path.write_bytes(text.encode())
    want = Tractogram(PTS[CONN], OFFSETS, streamline_data={"len": CD1})
    _assert_same(read_vtk(path), want)


@pytest.mark.parametrize("mode", MODES)
def test_empty_lines_are_kept(tmp_path: Path, mode: str) -> None:
    points = ("POINTS 3 float", PTS[:3], ">f4")
    lines = ("LINES 6 9", [0, 2, 0, 1, 0, 0, 1, 2, 0], ">i4")
    got = read_vtk(_vtk(tmp_path / "t.vtk", mode, [points, lines]))
    _assert_same(got, Tractogram(PTS[:3], np.array([0, 0, 2, 2, 2, 3, 3])))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("with_points", [False, True])
def test_file_without_lines_has_no_streamlines(
    tmp_path: Path, mode: str, with_points: bool
) -> None:
    sections: list[Section] = [POINTS] if with_points else []
    got = read_vtk(_vtk(tmp_path / "t.vtk", mode, sections))
    _assert_same(got, Tractogram(np.zeros((0, 3)), np.zeros(1, np.int64)))


ATTRIBUTES: dict[str, tuple[str, list[Section], dict[str, np.ndarray]]] = {
    "scalars": (
        "POINT_DATA",
        [("SCALARS fa float", PD1, ">f4")],
        {"fa": PD1[:, None]},
    ),
    "scalars-k1-lookup": (
        "POINT_DATA",
        [_h("SCALARS fa float 1"), ("LOOKUP_TABLE default", PD1, ">f4")],
        {"fa": PD1[:, None]},
    ),
    "scalars-k3-lookup-double": (
        "POINT_DATA",
        [_h("SCALARS rgb double 3"), ("LOOKUP_TABLE my_table", PD3, ">f8")],
        {"rgb": PD3},
    ),
    "scalars-int": (
        "POINT_DATA",
        [("SCALARS label int 1", np.arange(6), ">i4")],
        {"label": np.arange(6)[:, None]},
    ),
    "vectors": ("POINT_DATA", [("VECTORS dir float", PD3, ">f4")], {"dir": PD3}),
    "normals": ("POINT_DATA", [("NORMALS n double", PD3, ">f8")], {"n": PD3}),
    "field": (
        "POINT_DATA",
        [
            _h("FIELD FieldData 2"),
            ("fa 1 6 float", PD1, ">f4"),
            ("rgb 3 6 double", PD3, ">f8"),
        ],
        {"fa": PD1[:, None], "rgb": PD3},
    ),
    "two-attributes": (
        "POINT_DATA",
        [("SCALARS fa float", PD1, ">f4"), ("VECTORS dir float", PD3, ">f4")],
        {"fa": PD1[:, None], "dir": PD3},
    ),
    "cell-scalars": (
        "CELL_DATA",
        [_h("SCALARS len float"), ("LOOKUP_TABLE default", CD1, ">f4")],
        {"len": CD1[:, None]},
    ),
    "cell-vectors": ("CELL_DATA", [("VECTORS dir double", CD3, ">f8")], {"dir": CD3}),
    "cell-field": (
        "CELL_DATA",
        [
            _h("FIELD FieldData 2"),
            ("len 1 3 float", CD1, ">f4"),
            ("Rgb 3 3 float", CD3, ">f4"),
        ],
        {"len": CD1[:, None], "Rgb": CD3},
    ),
}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", list(ATTRIBUTES))
def test_attribute_is_read(tmp_path: Path, mode: str, case: str) -> None:
    kind, sections, raw = ATTRIBUTES[case]
    count = 6 if kind == "POINT_DATA" else 3
    path = _vtk(
        tmp_path / "t.vtk", mode, [POINTS, LINES, _h(f"{kind} {count}"), *sections]
    )
    got = read_vtk(path)
    if kind == "POINT_DATA":
        want = {name: _squeeze(v[CONN]) for name, v in raw.items()}
        assert got.streamline_data == {}
        data = got.point_data
    else:
        want = {name: _squeeze(v) for name, v in raw.items()}
        assert got.point_data == {}
        data = got.streamline_data
    assert data.keys() == want.keys()
    for name, values in want.items():
        assert data[name].shape == values.shape
        np.testing.assert_array_equal(data[name], values)
    _assert_same(Tractogram(got.points, got.offsets), _gathered())


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cell_first", [False, True])
def test_point_and_cell_data_are_both_read(
    tmp_path: Path, mode: str, cell_first: bool
) -> None:
    point = [_h("POINT_DATA 6"), ("SCALARS fa float", PD1, ">f4")]
    cell = [_h("CELL_DATA 3"), ("VECTORS dir float", CD3, ">f4")]
    data = cell + point if cell_first else point + cell
    got = read_vtk(_vtk(tmp_path / "t.vtk", mode, [POINTS, LINES, *data]))
    want = Tractogram(
        PTS[CONN], OFFSETS, point_data={"fa": PD1[CONN]}, streamline_data={"dir": CD3}
    )
    _assert_same(got, want)


@pytest.mark.parametrize("mode", MODES)
def test_metadata_blocks_are_skipped(tmp_path: Path, mode: str) -> None:
    short = _h("METADATA\nINFORMATION 0\n")
    long = _h(
        "METADATA\nINFORMATION 1\nNAME L2_NORM_RANGE LOCATION vtkDataArray\n"
        "DATA 2 0 1.5\n"
    )
    sections = [
        POINTS,
        short,
        LINES,
        long,
        _h("POINT_DATA 6"),
        _h("FIELD FieldData 2"),
        ("fa 1 6 float", PD1, ">f4"),
        long,
        ("rgb 3 6 float", PD3, ">f4"),
        short,
        _h("CELL_DATA 3"),
        _h("FIELD FieldData 1"),
        ("len 1 3 float", CD1, ">f4"),
        long,
    ]
    got = read_vtk(_vtk(tmp_path / "t.vtk", mode, sections, version="5.1"))
    want = Tractogram(
        PTS[CONN],
        OFFSETS,
        point_data={"fa": PD1[CONN], "rgb": PD3[CONN]},
        streamline_data={"len": CD1},
    )
    _assert_same(got, want)


HEADERS = {
    "empty-file": b"",
    "no-hash": b"vtk DataFile Version 3.0\nt\nASCII\nDATASET POLYDATA\n",
    "no-version": b"# vtk DataFile\nt\nASCII\nDATASET POLYDATA\n",
    "other-first-line": b"# some file\nt\nASCII\nDATASET POLYDATA\n",
    "other-mode": b"# vtk DataFile Version 3.0\nt\nTEXT\nDATASET POLYDATA\n",
    "no-title": b"# vtk DataFile Version 3.0\nASCII\nDATASET POLYDATA\n",
    "no-dataset": b"# vtk DataFile Version 3.0\nt\nASCII\n",
    "unstructured-grid": (
        b"# vtk DataFile Version 3.0\nt\nASCII\nDATASET UNSTRUCTURED_GRID\n"
    ),
    "structured-points": (
        b"# vtk DataFile Version 3.0\nt\nASCII\nDATASET STRUCTURED_POINTS\n"
    ),
}


@pytest.mark.parametrize("case", list(HEADERS))
def test_bad_header_raises_naming_the_file(tmp_path: Path, case: str) -> None:
    path = tmp_path / "bad-q7.vtk"
    path.write_bytes(HEADERS[case] + _body("ASCII", [POINTS, LINES]))
    with pytest.raises(ValueError, match=re.escape(path.name)):
        read_vtk(path)


def _lines_51(offsets: Values, conn: Values) -> list[Section]:
    return [
        _h(f"LINES {len(offsets)} {len(conn)}"),
        ("OFFSETS vtktypeint64", offsets, ">i8"),
        ("CONNECTIVITY vtktypeint64", conn, ">i8"),
    ]


BAD_BODIES: dict[str, list[Section]] = {
    "int-points": [("POINTS 6 int", np.arange(18), ">i4"), LINES],
    "vertices": [POINTS, ("VERTICES 1 2", [1, 0], ">i4")],
    "polygons": [POINTS, ("POLYGONS 1 4", [3, 0, 1, 2], ">i4")],
    "triangle-strips": [POINTS, ("TRIANGLE_STRIPS 1 4", [3, 0, 1, 2], ">i4")],
    "counts-overrun-size": [POINTS, ("LINES 3 8", [3, 4, 0, 2, 0, 3, 2, 5], ">i4")],
    "too-few-lines-declared": [POINTS, ("LINES 2 8", LEGACY, ">i4")],
    "too-many-lines-declared": [POINTS, ("LINES 4 8", LEGACY, ">i4")],
    "size-too-large": [POINTS, ("LINES 3 9", [*LEGACY, 1], ">i4")],
    "index-too-large": [POINTS, ("LINES 3 8", [3, 4, 0, 6, 0, 2, 2, 5], ">i4")],
    "negative-index": [POINTS, ("LINES 3 8", [3, 4, 0, -1, 0, 2, 2, 5], ">i4")],
    "negative-count": [POINTS, ("LINES 3 8", [3, 4, 0, 2, -1, 2, 2, 5], ">i4")],
    "count-moving-back": [POINTS, ("LINES 3 8", [3, 4, 0, 2, -3, 2, 2, 5], ">i4")],
    "51-index-too-large": [POINTS, *_lines_51(OFFSETS, [4, 0, 6, 2, 5])],
    "51-negative-index": [POINTS, *_lines_51(OFFSETS, [4, 0, -1, 2, 5])],
    "51-first-offset-not-0": [POINTS, *_lines_51([1, 3, 3, 5], CONN)],
    "51-last-offset-not-size": [POINTS, *_lines_51([0, 3, 3, 4], CONN)],
    "51-decreasing-offsets": [POINTS, *_lines_51([0, 3, 2, 5], CONN)],
    "51-offsets-without-connectivity": [
        POINTS,
        _h("LINES 4 5"),
        ("OFFSETS vtktypeint64", OFFSETS, ">i8"),
    ],
    "point-data-count": [
        POINTS,
        LINES,
        _h("POINT_DATA 5"),
        ("SCALARS fa float", PD1[:5], ">f4"),
    ],
    "cell-data-count": [
        POINTS,
        LINES,
        _h("CELL_DATA 4"),
        ("SCALARS len float", [*CD1, 0], ">f4"),
    ],
    "field-array-count": [
        POINTS,
        LINES,
        _h("POINT_DATA 6"),
        _h("FIELD FieldData 1"),
        # Six values are there: only the declared count is wrong.
        ("fa 1 5 float", PD1, ">f4"),
    ],
    "tensors": [
        POINTS,
        LINES,
        _h("POINT_DATA 6"),
        ("TENSORS t float", [0] * 54, ">f4"),
    ],
    "color-scalars": [
        POINTS,
        LINES,
        _h("POINT_DATA 6"),
        ("COLOR_SCALARS c 3", [0.5] * 18, ">f4"),
    ],
    "texture-coordinates": [
        POINTS,
        LINES,
        _h("POINT_DATA 6"),
        ("TEXTURE_COORDINATES t 2 float", [0] * 12, ">f4"),
    ],
    "scalars-before-data": [POINTS, LINES, ("SCALARS fa float", PD1, ">f4")],
    "field-before-data": [
        _h("FIELD FieldData 1"),
        ("fa 1 6 float", PD1, ">f4"),
        POINTS,
        LINES,
    ],
}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", list(BAD_BODIES))
def test_bad_body_raises_naming_the_file(tmp_path: Path, mode: str, case: str) -> None:
    path = _vtk(tmp_path / "bad-q7.vtk", mode, BAD_BODIES[case])
    with pytest.raises(ValueError, match=re.escape(path.name)):
        read_vtk(path)


@pytest.mark.parametrize("kind", ["POINT_DATA", "CELL_DATA"])
def test_data_section_count_must_match_even_without_arrays(
    tmp_path: Path, kind: str
) -> None:
    path = _vtk(tmp_path / "bad-q7.vtk", "ASCII", [POINTS, LINES, _h(f"{kind} 5")])
    with pytest.raises(ValueError, match=re.escape(path.name)):
        read_vtk(path)


ASCII_BAD = {
    "too-few-points": "POINTS 6 float\n" + "0 " * 17 + "\nLINES 1 2\n1 0\n",
    "text-in-points": "POINTS 2 float\n0 0 0 1 x 1\nLINES 1 3\n2 0 1\n",
    # The next array's header jammed onto the line of the last value.
    "field-header-mid-line": (
        "POINTS 2 float\n0 0 0 1 1 1\nLINES 1 3\n2 0 1\n"
        "POINT_DATA 2\nFIELD FieldData 2\nfa 1 2 float\n0.5 0.25 md 1 2 float\n"
        "1 2\n"
    ),
    "field-extra-value": (
        "POINTS 2 float\n0 0 0 1 1 1\nLINES 1 3\n2 0 1\n"
        "POINT_DATA 2\nFIELD FieldData 2\nfa 1 2 float\n0.5 0.25 7\n"
        "md 1 2 float\n1 2\n"
    ),
    "field-last-with-text": (
        "POINTS 2 float\n0 0 0 1 1 1\nLINES 1 3\n2 0 1\n"
        "CELL_DATA 1\nFIELD FieldData 1\nlen 1 1 float\n0.5 extra\n"
    ),
}


@pytest.mark.parametrize("case", list(ASCII_BAD))
def test_bad_ascii_numbers_raise_naming_the_file(tmp_path: Path, case: str) -> None:
    path = tmp_path / "bad-q7.vtk"
    head = "# vtk DataFile Version 3.0\nt\nASCII\nDATASET POLYDATA\n"
    path.write_bytes((head + ASCII_BAD[case]).encode())
    with pytest.raises(ValueError, match=re.escape(path.name)):
        read_vtk(path)


@pytest.mark.parametrize("block", ["points", "lines", "offsets", "field"])
@pytest.mark.parametrize("missing", [1, 3])
def test_binary_file_ending_inside_a_block_raises(
    tmp_path: Path, block: str, missing: int
) -> None:
    sections = {
        "points": [POINTS],
        "lines": [POINTS, LINES],
        "offsets": [POINTS, _h("LINES 4 5"), ("OFFSETS vtktypeint64", OFFSETS, ">i8")],
        "field": [
            POINTS,
            LINES,
            _h("POINT_DATA 6"),
            _h("FIELD FieldData 1"),
            ("fa 1 6 float", PD1, ">f4"),
        ],
    }[block]
    path = _vtk(tmp_path / "bad-q7.vtk", "BINARY", sections)
    # Drop the newline after the block and `missing` bytes of the block.
    path.write_bytes(path.read_bytes()[: -1 - missing])
    with pytest.raises(ValueError, match=re.escape(path.name)):
        read_vtk(path)


# Writing


class _Reader:
    # Walks a written file the same way in both modes: keyword lines as words, then
    # blocks of numbers (big-endian bytes in binary, text tokens in ASCII).
    def __init__(self, raw: bytes) -> None:
        parts = raw.split(b"\n", 4)
        self.header = [p.decode() for p in parts[:4]]
        self.binary = self.header[2] == "BINARY"
        self.rest = parts[4]
        self.tokens = [] if self.binary else self.rest.decode().split()
        self.pos = 0

    def words(self, n: int) -> list[str]:
        if self.binary:
            while self.rest[self.pos : self.pos + 1] == b"\n":
                self.pos += 1
            end = self.rest.index(b"\n", self.pos)
            out = self.rest[self.pos : end].decode().split()
            self.pos = end + 1
        else:
            out = self.tokens[self.pos : self.pos + n]
            self.pos += n
        assert len(out) == n
        return out

    def block(self, count: int, dtype: str) -> np.ndarray:
        if self.binary:
            size = count * np.dtype(dtype).itemsize
            raw = self.rest[self.pos : self.pos + size]
            assert len(raw) == size
            self.pos += size
            return np.frombuffer(raw, dtype)
        out = np.array(self.tokens[self.pos : self.pos + count], dtype=np.float64)
        assert len(out) == count
        self.pos += count
        return out

    def at_end(self) -> bool:
        if self.binary:
            return self.rest[self.pos :].strip() == b""
        return self.pos == len(self.tokens)


@pytest.mark.parametrize("binary", [True, False])
@pytest.mark.parametrize("data", [True, False])
def test_written_file_has_the_legacy_layout(
    tmp_path: Path, binary: bool, data: bool
) -> None:
    t = _tractogram(LENGTHS, data=data)
    path = tmp_path / "t.vtk"
    write_vtk(t, path, binary=binary)
    r = _Reader(path.read_bytes())
    n_points, n_lines = t.num_points, t.num_streamlines

    assert r.header[0] == "# vtk DataFile Version 3.0"
    assert "SPACE=RAS" in r.header[1].split()
    assert r.header[2] == ("BINARY" if binary else "ASCII")
    assert r.header[3] == "DATASET POLYDATA"
    assert r.words(3) == ["POINTS", str(n_points), "float"]
    points = r.block(3 * n_points, ">f4").astype(np.float32).reshape(-1, 3)
    np.testing.assert_array_equal(points, t.points)
    assert r.words(3) == ["LINES", str(n_lines), str(n_lines + n_points)]
    want_lines = []
    for start, stop in zip(t.offsets[:-1], t.offsets[1:], strict=True):
        want_lines += [stop - start, *range(start, stop)]
    np.testing.assert_array_equal(r.block(n_lines + n_points, ">i4"), want_lines)

    sections = [("POINT_DATA", n_points, t.point_data)]
    sections += [("CELL_DATA", n_lines, t.streamline_data)]
    for kind, count, want in sections if data else []:
        assert r.words(2) == [kind, str(count)]
        assert r.words(3) == ["FIELD", "FieldData", str(len(want))]
        got = {}
        for _ in want:
            name, k, n, type_name = r.words(4)
            assert (n, type_name) == (str(count), "float")
            got[name] = r.block(count * int(k), ">f4").astype(np.float32)
        assert got.keys() == want.keys()
        for name, values in want.items():
            np.testing.assert_array_equal(got[name], values.reshape(count, -1).ravel())
    assert r.at_end()


@pytest.mark.parametrize("binary", [True, False])
@pytest.mark.parametrize("which", ["point", "streamline"])
def test_only_nonempty_data_is_written(
    tmp_path: Path, binary: bool, which: str
) -> None:
    t = _tractogram([2, 3], data=True)
    if which == "point":
        t.streamline_data = {}
    else:
        t.point_data = {}
    path = tmp_path / "t.vtk"
    write_vtk(t, path, binary=binary)
    raw = path.read_bytes()
    assert (b"POINT_DATA" in raw) == (which == "point")
    assert (b"CELL_DATA" in raw) == (which == "streamline")
    _assert_same(read_vtk(path), t)


@pytest.mark.parametrize("binary", [True, False])
@pytest.mark.parametrize("name", ["t.vtk", "t.fib", "t.vtk.gz"])
@pytest.mark.parametrize(
    "lengths", [LENGTHS, [1], [0], [0, 0], [], [2, 0, 0, 5]], ids=str
)
def test_round_trip_is_exact(
    tmp_path: Path, binary: bool, name: str, lengths: list[int]
) -> None:
    t = _tractogram(lengths, seed=3, data=True)
    path = tmp_path / name
    write_vtk(t, path, binary=binary)
    _assert_same(read_vtk(path), t)


@pytest.mark.parametrize("binary", [True, False])
@pytest.mark.parametrize("lengths", [LENGTHS, [0, 0], []], ids=str)
def test_round_trip_without_data_has_no_data(
    tmp_path: Path, binary: bool, lengths: list[int]
) -> None:
    t = _tractogram(lengths, seed=5)
    write_vtk(t, tmp_path / "t.vtk", binary=binary)
    _assert_same(read_vtk(tmp_path / "t.vtk"), t)


def test_default_is_binary(tmp_path: Path) -> None:
    write_vtk(_tractogram([2, 3]), tmp_path / "t.vtk")
    assert (tmp_path / "t.vtk").read_bytes().split(b"\n")[2] == b"BINARY"


@pytest.mark.parametrize("binary", [True, False])
def test_gz_name_writes_a_gzip_file(tmp_path: Path, binary: bool) -> None:
    write_vtk(_tractogram([2, 3]), tmp_path / "t.vtk.gz", binary=binary)
    raw = gzip.decompress((tmp_path / "t.vtk.gz").read_bytes())
    assert raw.startswith(b"# vtk DataFile Version 3.0\n")


@pytest.mark.parametrize("binary", [True, False])
def test_tck_to_vtk_and_back_keeps_points_and_offsets(
    tmp_path: Path, binary: bool
) -> None:
    write_vtk(_ref(), tmp_path / "t.vtk", binary=binary)
    write_tck(read_vtk(tmp_path / "t.vtk"), tmp_path / "back.tck")
    _assert_same(read_tck(tmp_path / "back.tck"), _ref())


@needs_mrtrix
@pytest.mark.parametrize("data", [False, True])
def test_mrtrix_reads_the_written_file(tmp_path: Path, data: bool) -> None:
    t = _tractogram(MRTRIX_LENGTHS, seed=6, data=data)
    write_vtk(t, tmp_path / "t.vtk")
    _tckconvert(tmp_path / "t.vtk", tmp_path / "t.tck")
    _assert_same(read_tck(tmp_path / "t.tck"), Tractogram(t.points, t.offsets))


@pytest.mark.parametrize("which", ["point_data", "streamline_data"])
@pytest.mark.parametrize("name", ["", "a b", "a\tb", "fa\n", " fa"])
def test_bad_data_name_raises_and_writes_nothing(
    tmp_path: Path, which: str, name: str
) -> None:
    plain = _tractogram([2, 3])
    count = plain.num_points if which == "point_data" else plain.num_streamlines
    data = {name: np.zeros(count)}
    t = Tractogram(
        plain.points,
        plain.offsets,
        point_data=data if which == "point_data" else {},
        streamline_data=data if which == "streamline_data" else {},
    )
    path = tmp_path / "t.vtk"
    with pytest.raises(ValueError):
        write_vtk(t, path)
    assert not path.exists()


@pytest.mark.parametrize("as_str", [False, True])
def test_existing_file_is_overwritten(tmp_path: Path, as_str: bool) -> None:
    path = tmp_path / "t.vtk"
    write_vtk(_tractogram([9, 9, 9], seed=1, data=True), path, binary=False)
    t = _tractogram([2])
    write_vtk(t, str(path) if as_str else path)
    _assert_same(read_vtk(path), t)


def test_ascii_values_written_with_capitals_are_numbers(tmp_path: Path) -> None:
    path = tmp_path / "a.vtk"
    path.write_text(
        "# vtk DataFile Version 3.0\nt\nASCII\nDATASET POLYDATA\nPOINTS 2 float\n"
        "0 0 0\nNaN Inf -Inf\nLINES 1 3\n2 0 1\n"
    )
    t = read_vtk(path)
    assert np.isnan(t.points[1, 0])
    assert t.points[1, 1:].tolist() == [np.inf, -np.inf]


@pytest.mark.parametrize("version", ["1.0", "5.2", "6.0"])
def test_versions_outside_2_0_to_5_1_raise(tmp_path: Path, version: str) -> None:
    path = tmp_path / "a.vtk"
    path.write_text(
        f"# vtk DataFile Version {version}\nt\nASCII\nDATASET POLYDATA\n"
        "POINTS 1 float\n0 0 0\nLINES 1 2\n1 0\n"
    )
    with pytest.raises(ValueError, match="a.vtk"):
        read_vtk(path)


def test_two_arrays_with_one_name_raise(tmp_path: Path) -> None:
    path = tmp_path / "a.vtk"
    path.write_text(
        "# vtk DataFile Version 3.0\nt\nASCII\nDATASET POLYDATA\nPOINTS 1 float\n"
        "0 0 0\nLINES 1 2\n1 0\nPOINT_DATA 1\nSCALARS fa float\nLOOKUP_TABLE default\n"
        "1\nFIELD FieldData 1\nfa 1 1 float\n2\n"
    )
    with pytest.raises(ValueError, match="a.vtk"):
        read_vtk(path)


@pytest.mark.parametrize("binary", [True, False], ids=["binary", "ascii"])
def test_data_keeps_its_type(tmp_path: Path, binary: bool) -> None:
    rng = np.random.default_rng(3)
    big = np.array([2**40 + 1, -(2**53) + 3, 7], dtype=np.int64)
    t = Tractogram(
        rng.standard_normal((3, 3)).astype(np.float32),
        np.array([0, 2, 3]),
        point_data={
            "label": big,
            "fa": rng.random(3),
            "dirs": rng.random((3, 3)),
            "id": np.array([1, 200, 255], dtype=np.uint8),
        },
        streamline_data={"n": np.array([2**31 - 1, -5], dtype=np.int32)},
    )
    path = tmp_path / "a.vtk"
    write_vtk(t, path, binary=binary)
    r = read_vtk(path)
    for name, values in t.point_data.items():
        assert r.point_data[name].dtype == values.dtype
        np.testing.assert_array_equal(r.point_data[name], values)
    assert r.streamline_data["n"].dtype == np.int32
    np.testing.assert_array_equal(r.streamline_data["n"], t.streamline_data["n"])


def test_bool_data_is_written_as_unsigned_char(tmp_path: Path) -> None:
    mask = np.array([True, False, True])
    t = Tractogram(np.zeros((3, 3)), np.array([0, 3]), point_data={"in_roi": mask})
    write_vtk(t, tmp_path / "a.vtk")
    r = read_vtk(tmp_path / "a.vtk")
    assert r.point_data["in_roi"].dtype == np.uint8
    np.testing.assert_array_equal(r.point_data["in_roi"], mask)


@pytest.mark.parametrize(
    "values",
    [np.zeros((3, 2, 2)), np.zeros(3, dtype=np.complex64), np.zeros(3, np.float16)],
    ids=["3d", "complex", "float16"],
)
def test_data_vtk_cannot_store_raises(tmp_path: Path, values: np.ndarray) -> None:
    t = Tractogram(np.zeros((3, 3)), np.array([0, 3]), point_data={"x": values})
    path = tmp_path / "a.vtk"
    with pytest.raises(ValueError, match="x"):
        write_vtk(t, path)
    assert not path.exists()
