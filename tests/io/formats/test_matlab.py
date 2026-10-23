from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.io import loadmat, savemat

from larmor.io import MatlabEntry, read_matlab

DATA = Path(__file__).parents[2] / "data" / "matlab"
FILES = {"v6": "values_v6.mat", "v7": "values_v7.mat", "v73": "values_v73.mat"}
VERSIONS = {"v6": "5", "v7": "5", "v73": "7.3"}
LINEAR = loadmat(DATA / "expected.mat", squeeze_me=True)["linear"]


@pytest.fixture(params=sorted(FILES))
def mat(request: pytest.FixtureRequest) -> tuple[str, Path]:
    return request.param, DATA / FILES[request.param]


def test_version_is_detected(mat: tuple[str, Path]) -> None:
    family, path = mat
    assert read_matlab(path).version == VERSIONS[family]


def test_numeric_arrays_keep_matlab_shape_class_and_order(
    mat: tuple[str, Path],
) -> None:
    f = read_matlab(mat[1])
    array3d = f["array3d"]
    assert isinstance(array3d, np.ndarray)
    assert array3d.shape == (2, 3, 4) and array3d.dtype == np.float64
    # MATLAB's linear order is column-major: a(:)' in MATLAB.
    np.testing.assert_array_equal(array3d.ravel(order="F"), LINEAR["array3d"][()])
    np.testing.assert_array_equal(
        f.read_array("uint8_mat").ravel(order="F"), LINEAR["uint8_mat"][()]
    )
    assert f.read_array("uint8_mat").dtype == np.uint8
    assert f.read_array("single2d").dtype == np.float32
    assert f.read_array("int64_col").tolist() == [[-(2**40)], [2**40 + 1]]
    assert f.read_array("scalar").shape == (1, 1)
    flags = f.read_array("flags")
    assert flags.dtype == np.bool_
    np.testing.assert_array_equal(flags.ravel(order="F"), LINEAR["flags"][()] > 0)


def test_complex_and_sparse_values(mat: tuple[str, Path]) -> None:
    f = read_matlab(mat[1])
    np.testing.assert_array_equal(f["complex_row"], [[1 + 2j, 3 - 4j, -5j]])
    matrix = f["sparse_mat"]
    assert isinstance(matrix, sparse.csc_array) and matrix.shape == (4, 3)
    np.testing.assert_array_equal(
        matrix.toarray().ravel(order="F"), LINEAR["sparse_mat"][()]
    )
    flags = f["sparse_flags"]
    assert isinstance(flags, sparse.csc_array) and flags.dtype == np.bool_
    np.testing.assert_array_equal(flags.toarray(), [[0, 1], [1, 0], [0, 0]])


def test_complex_values_inside_structs_and_cells(mat: tuple[str, Path]) -> None:
    f = read_matlab(mat[1])
    inner = f["nested.z"]
    assert isinstance(inner, np.ndarray) and inner.dtype == np.complex64
    np.testing.assert_array_equal(inner, [[1 + 1j, 2 - 3j]])
    in_cell = f["nested_cell{2}"]
    assert isinstance(in_cell, np.ndarray) and in_cell.dtype == np.complex128
    np.testing.assert_array_equal(in_cell, [[1 + 2j], [3 - 4j]])


def test_a_long_cell_is_listed_as_one_entry_and_read_whole(
    mat: tuple[str, Path],
) -> None:
    f = read_matlab(mat[1])
    paths = [e.path for e in f.entries()]
    assert "long_cell" in paths and not any(p.startswith("long_cell{") for p in paths)
    assert "long_records" in paths
    assert not any(p.startswith("long_records(") for p in paths)
    cells = f["long_cell"]
    assert isinstance(cells, list) and len(cells) == 150
    np.testing.assert_array_equal(cells[149], [[150.0]])


def test_a_nested_struct_is_listed_field_by_field(mat: tuple[str, Path]) -> None:
    listed = [e for e in read_matlab(mat[1]).entries() if e.path.startswith("record")]
    assert listed == [
        MatlabEntry("record", (1, 1), "struct"),
        MatlabEntry("record.number", (1, 1), "double"),
        MatlabEntry("record.name", (1, 1), "char"),
        MatlabEntry("record.inner", (1, 1), "struct"),
        MatlabEntry("record.inner.values", (1, 3), "double"),
        MatlabEntry("records", (1, 3), "struct"),
        MatlabEntry("records(1)", (1, 1), "struct"),
        MatlabEntry("records(1).v", (1, 1), "double"),
        MatlabEntry("records(2)", (1, 1), "struct"),
        MatlabEntry("records(2).v", (1, 2), "double"),
        MatlabEntry("records(3)", (1, 1), "struct"),
        MatlabEntry("records(3).v", (1, 1), "char"),
    ]


def test_version_4_files_are_read() -> None:
    f = read_matlab(DATA / "values_v4.mat")
    assert f.version == "4"
    np.testing.assert_array_equal(f["double2d"], [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(f["complex_row"], [[1 + 2j, 3 - 4j, -5j]])
    assert f["text"] == "hello" and f["text_rows"] == ["ab", "cd", "ef"]
    assert f.entry("sparse_mat") == MatlabEntry("sparse_mat", (4, 3), "double", True)


def test_a_version_5_variable_is_read_only_when_asked(tmp_path: Path) -> None:
    path = tmp_path / "plain.mat"
    savemat(path, {"a": np.ones((2, 2)), "b": np.zeros(3)})
    f = read_matlab(path)
    assert f.names == ["a", "b"] and f._v5 == {}
    f.read("a")
    assert list(f._v5) == ["a"]


def test_text_structs_and_cells(mat: tuple[str, Path]) -> None:
    f = read_matlab(mat[1])
    assert f["text"] == "hello"
    assert f["text_accent"] == "café"
    assert f["text_rows"] == ["ab", "cd", "ef"]
    record = f["record"]
    assert isinstance(record, dict) and list(record) == ["number", "name", "inner"]
    assert record["name"] == "x"
    np.testing.assert_array_equal(f["record.inner.values"], [[1, 2, 3]])
    records = f["records"]
    assert isinstance(records, list) and len(records) == 3
    assert records[2] == {"v": "z"}
    np.testing.assert_array_equal(f["records(2).v"], [[2, 3]])
    # A 2x3 cell, in linear order: 1, int8(6), 'two', {7, 'eight'}, [3 4 5], struct.
    cells = f["cells"]
    assert isinstance(cells, list) and len(cells) == 6
    assert cells[2] == "two" and cells[5] == {"q": np.array([[8.0]])}
    assert f["cells{2}"].dtype == np.int8  # type: ignore[union-attr]
    assert f["cells{4}{2}"] == "eight"


def test_empty_values(mat: tuple[str, Path]) -> None:
    f = read_matlab(mat[1])
    empty = f.read_array("empty_double")
    assert empty.shape == (0, 3) and empty.dtype == np.float64
    assert f["empty_text"] == ""
    assert f["empty_cell"] == []
    assert f["empty_struct"] == []


@pytest.mark.parametrize(
    ("name", "matlab_class"), [("text_object", "string"), ("handle", "function_handle")]
)
def test_objects_are_refused_by_class(
    mat: tuple[str, Path], name: str, matlab_class: str
) -> None:
    f = read_matlab(mat[1])
    assert f.entry(name).matlab_class == matlab_class
    with pytest.raises(ValueError, match=f"class '{matlab_class}'"):
        f.read(name)


def test_listings_agree_between_families() -> None:
    listings = {k: read_matlab(DATA / v).entries() for k, v in FILES.items()}
    assert listings["v6"] == listings["v7"]
    # scipy reads an empty struct like an empty cell.
    expected = [
        MatlabEntry(e.path, e.shape, "cell") if e.path == "empty_struct" else e
        for e in listings["v73"]
    ]
    assert listings["v6"] == expected
    paths = [e.path for e in listings["v73"]]
    assert {"cells{4}{2}", "records(3).v", "record.inner.values"} <= set(paths)
    assert MatlabEntry("sparse_mat", (4, 3), "double", True) in listings["v73"]


def test_top_level_listing_without_recursion(mat: tuple[str, Path]) -> None:
    f = read_matlab(mat[1])
    assert [e.path for e in f.entries(recursive=False)] == f.names
    assert "record" in f and "record.name" not in f


@pytest.mark.parametrize(
    "path", ["missing", "text.field", "cells{7}", "records.v", "text{1}", "1bad"]
)
def test_bad_paths_raise_key_error(mat: tuple[str, Path], path: str) -> None:
    with pytest.raises(KeyError):
        read_matlab(mat[1]).read(path)


@pytest.mark.parametrize(
    "content", [b"not a MATLAB file" * 10, bytes(100), b"\x00\x00\x00\x00junk" * 9]
)
def test_a_file_that_is_not_matlab_is_refused(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "x.mat"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="not a MATLAB"):
        read_matlab(path)
