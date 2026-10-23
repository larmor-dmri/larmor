import logging
import re
import shutil
import subprocess
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from larmor.io import (
    load_header,
    load_volume,
    read_b_matrix,
    read_bvals_bvecs,
    save_volume,
    write_b_matrix,
    write_bvals_bvecs,
)
from larmor.io.formats import read_nifti_header, write_nifti
from larmor.types import GradientTable, Volume

# SVD and float64 products: exact affines give rounding of a few 1e-16.
ATOL = 1e-12

# The direction (0.48, 0.60, 0.64) in the bvecs of a RAS image (det > 0, so FSL
# negates x) is this world direction.
WORLD = np.array([-0.48, 0.60, 0.64])

# Expected bvecs columns, worked out by hand, for the same anatomy re-stored with each
# affine.
CASES = {
    "ras": (np.diag([2.0, 2.0, 2.0]), [0.48, 0.60, 0.64]),
    "i_flipped": (np.diag([-2.0, 2.0, 2.0]), [0.48, 0.60, 0.64]),
    "j_flipped": (np.diag([2.0, -2.0, 2.0]), [-0.48, -0.60, 0.64]),
    "k_flipped": (np.diag([2.0, 2.0, -2.0]), [-0.48, 0.60, -0.64]),
    # Voxel i runs along world y, voxel j along world x.
    "ij_swapped": (
        np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        [0.60, -0.48, 0.64],
    ),
    # From LAS (i along -x): voxel i runs along world y, voxel j along world -x.
    "ij_swapped_from_las": (
        np.array([[0.0, -2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        [-0.60, 0.48, 0.64],
    ),
}
# Not in the table: for a tilt about x, R @ F is not symmetric, so a transposed
# conversion is caught. Integer entries (cos = 0.8, sin = 0.6, voxels 1, 5, 5 mm)
# survive the float32 header exactly. The column is F @ R.T @ WORLD, F negating x.
CASES["tilted_about_x"] = (
    np.array([[1.0, 0.0, 0.0], [0.0, 4.0, -3.0], [0.0, 3.0, 4.0]]),
    [0.48, 0.8 * 0.60 + 0.6 * 0.64, -0.6 * 0.60 + 0.8 * 0.64],
)
CONVERSION = pytest.mark.parametrize(
    ("linear", "column"), list(CASES.values()), ids=list(CASES)
)

MATLAB_DATA = Path(__file__).parents[1] / "data"

SHEAR = np.array([[2.0, 0.3, 0.0], [0.0, 2.0, 0.2], [0.0, 0.0, 2.0]])

# Unit vectors in the file, one per column, after a b0 with no direction.
UNIT = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.48, 0.60, 0.64],
        [-0.36, 0.48, 0.80],
        [0.0, -0.6, 0.8],
        [1.0, 0.0, 0.0],
    ]
)
BVALS = np.array([0.0, 1000.0, 1000.0, 2000.0, 3000.0])
IDENTITY = np.eye(3)


def _image(
    path: Path,
    linear: np.ndarray = IDENTITY,
    shape: tuple[int, ...] = (2, 2, 2, 5),
    distinct: bool = False,
) -> Path:
    affine = np.eye(4)
    affine[:3, :3] = linear
    affine[:3, 3] = (-3.0, 4.5, 1.0)
    data = np.zeros(shape, dtype=np.int16)
    if distinct:
        data = np.arange(data.size, dtype=np.int16).reshape(shape)
    write_nifti(Volume(data, affine), path)
    return path


def _files(
    tmp_path: Path, bvals: np.ndarray, bvecs: np.ndarray, newline: str = "\n"
) -> tuple[Path, Path]:
    bvals_path, bvecs_path = tmp_path / "bvals", tmp_path / "bvecs"
    np.savetxt(bvals_path, np.atleast_2d(bvals), fmt="%.17g", newline=newline)
    np.savetxt(bvecs_path, bvecs, fmt="%.17g", newline=newline)
    return bvals_path, bvecs_path


def _table() -> GradientTable:
    return GradientTable(BVALS, UNIT)


@CONVERSION
def test_fsl_column_reads_as_world_direction(
    tmp_path: Path, linear: np.ndarray, column: list[float]
) -> None:
    image = _image(tmp_path / "dwi.nii", linear, (2, 2, 2, 2))
    bvecs = np.array([[0.0, 0.0, 0.0], column]).T
    paths = _files(tmp_path, np.array([0.0, 1000.0]), bvecs)
    table = read_bvals_bvecs(*paths, frame="fsl", ref_path=image)
    np.testing.assert_allclose(table.bvecs[1], WORLD, rtol=0, atol=ATOL)
    np.testing.assert_array_equal(table.bvecs[0], [0.0, 0.0, 0.0])


@CONVERSION
def test_fsl_column_reads_as_world_direction_from_a_grid(
    tmp_path: Path, linear: np.ndarray, column: list[float]
) -> None:
    image = _image(tmp_path / "dwi.nii", linear, (2, 2, 2, 2))
    bvecs = np.array([[0.0, 0.0, 0.0], column]).T
    paths = _files(tmp_path, np.array([0.0, 1000.0]), bvecs)
    table = read_bvals_bvecs(*paths, frame="fsl", grid=load_header(image)[0])
    np.testing.assert_allclose(table.bvecs[1], WORLD, rtol=0, atol=ATOL)


@CONVERSION
def test_world_direction_writes_as_fsl_column(
    tmp_path: Path, linear: np.ndarray, column: list[float]
) -> None:
    image = _image(tmp_path / "dwi.nii", linear, (2, 2, 2, 2))
    table = GradientTable(np.array([0.0, 1000.0]), np.array([[0.0, 0.0, 0.0], WORLD]))
    bvals_path, bvecs_path = tmp_path / "bvals", tmp_path / "bvecs"
    write_bvals_bvecs(table, bvals_path, bvecs_path, frame="fsl", ref_path=image)
    np.testing.assert_allclose(np.loadtxt(bvecs_path)[:, 1], column, rtol=0, atol=ATOL)


@pytest.mark.parametrize(
    "linear",
    [*(linear for linear, _ in CASES.values()), SHEAR],
    ids=[*CASES, "shear"],
)
def test_fsl_round_trip_gives_back_the_file_numbers(
    tmp_path: Path, linear: np.ndarray
) -> None:
    image = _image(tmp_path / "dwi.nii.gz", linear)
    bvals_path, bvecs_path = _files(tmp_path, BVALS, UNIT.T)
    table = read_bvals_bvecs(bvals_path, bvecs_path, frame="fsl", ref_path=image)
    out_bvals, out_bvecs = tmp_path / "out.bval", tmp_path / "out.bvec"
    write_bvals_bvecs(table, out_bvals, out_bvecs, frame="fsl", ref_path=image)
    np.testing.assert_allclose(np.loadtxt(out_bvecs), UNIT.T, rtol=0, atol=ATOL)
    np.testing.assert_array_equal(np.loadtxt(out_bvals), BVALS)


def test_sheared_image_converts_as_mrtrix() -> None:
    # mrconvert -fslgrad on an oblique, sheared image, exported with 10 digits.
    mif = MATLAB_DATA / "mif"
    table = read_bvals_bvecs(
        mif / "sheared.bval", mif / "sheared.bvec", ref_path=mif / "sheared.nii"
    )
    rows = np.loadtxt(mif / "sheared.b")
    np.testing.assert_allclose(table.bvecs, rows[:, :3], rtol=0, atol=1e-9)
    np.testing.assert_array_equal(table.bvals, rows[:, 3])


def test_sheared_image_round_trips(tmp_path: Path) -> None:
    image = _image(tmp_path / "dwi.nii", SHEAR)
    table = read_bvals_bvecs(*_files(tmp_path, BVALS, UNIT.T), ref_path=image)
    out_bvals, out_bvecs = tmp_path / "out.bval", tmp_path / "out.bvec"
    write_bvals_bvecs(table, out_bvals, out_bvecs, ref_path=image)
    np.testing.assert_allclose(np.loadtxt(out_bvecs), UNIT.T, rtol=0, atol=ATOL)
    np.testing.assert_array_equal(np.loadtxt(out_bvals), BVALS)


@pytest.mark.parametrize("with_ref", [False, True], ids=["no_ref", "ref"])
def test_world_frame_is_not_converted(tmp_path: Path, with_ref: bool) -> None:
    # A sheared, flipped image would change the vectors if it were applied.
    image = _image(tmp_path / "dwi.nii", SHEAR @ np.diag([-1.0, 1.0, 1.0]))
    ref = image if with_ref else None
    paths = _files(tmp_path, BVALS, UNIT.T)
    table = read_bvals_bvecs(*paths, frame="world", ref_path=ref)
    np.testing.assert_array_equal(table.bvecs, _table().bvecs)

    out_bvals, out_bvecs = tmp_path / "out.bval", tmp_path / "out.bvec"
    write_bvals_bvecs(table, out_bvals, out_bvecs, frame="world", ref_path=ref)
    again = read_bvals_bvecs(out_bvals, out_bvecs, frame="world")
    np.testing.assert_array_equal(again.bvals, table.bvals)
    np.testing.assert_array_equal(again.bvecs, table.bvecs)


@pytest.mark.parametrize("layout", ["fsl", "swapped"])
def test_both_layouts_read_one_vector_per_volume(tmp_path: Path, layout: str) -> None:
    bvecs = UNIT.T if layout == "fsl" else UNIT
    table = read_bvals_bvecs(*_files(tmp_path, BVALS, bvecs), frame="world")
    np.testing.assert_array_equal(table.bvecs, UNIT)
    np.testing.assert_array_equal(table.bvals, BVALS)


@pytest.mark.parametrize("layout", ["column", "row"])
def test_single_volume_reads_as_one_vector(tmp_path: Path, layout: str) -> None:
    vector = np.array([0.0, 0.6, 0.8])
    bvecs = vector[:, None] if layout == "column" else vector[None, :]
    paths = _files(tmp_path, np.array([1000.0]), bvecs)
    table = read_bvals_bvecs(*paths, frame="world")
    assert table.bvecs.shape == (1, 3)
    np.testing.assert_array_equal(table.bvecs[0], vector)
    np.testing.assert_array_equal(table.bvals, [1000.0])


def test_three_by_three_file_is_read_as_fsl_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Rows are x, y, z: each column is one volume's vector.
    rows = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.6], [0.0, 0.0, 0.8]])
    paths = _files(tmp_path, np.array([1000.0, 1000.0, 1000.0]), rows)
    with caplog.at_level(logging.WARNING, logger="larmor"):
        table = read_bvals_bvecs(*paths, frame="world")
    np.testing.assert_array_equal(
        table.bvecs, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.6, 0.8]]
    )
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert "3x3" in caplog.text
    assert str(paths[1]) in caplog.text


@pytest.mark.parametrize("layout", ["fsl", "swapped"])
def test_unambiguous_layout_logs_no_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, layout: str
) -> None:
    bvecs = UNIT.T if layout == "fsl" else UNIT
    with caplog.at_level(logging.WARNING, logger="larmor"):
        read_bvals_bvecs(*_files(tmp_path, BVALS, bvecs), frame="world")
    assert caplog.records == []


@pytest.mark.parametrize("frame", ["fsl", "world"])
def test_windows_line_endings_read_as_unix(
    tmp_path: Path, frame: Literal["fsl", "world"]
) -> None:
    image = _image(tmp_path / "dwi.nii", np.diag([2.0, -2.0, 2.0]))
    unix = read_bvals_bvecs(
        *_files(tmp_path, BVALS, UNIT.T), frame=frame, ref_path=image
    )
    windows_dir = tmp_path / "windows"
    windows_dir.mkdir()
    paths = _files(windows_dir, BVALS, UNIT.T, newline="\r\n")
    assert b"\r\n" in paths[0].read_bytes()
    assert paths[1].read_bytes().count(b"\r\n") == 3
    windows = read_bvals_bvecs(*paths, frame=frame, ref_path=image)
    np.testing.assert_array_equal(windows.bvals, unix.bvals)
    np.testing.assert_array_equal(windows.bvecs, unix.bvecs)


def test_bvals_as_a_column_are_read(tmp_path: Path) -> None:
    bvals_path, bvecs_path = _files(tmp_path, BVALS, UNIT.T)
    np.savetxt(bvals_path, BVALS[:, None], fmt="%.17g")
    table = read_bvals_bvecs(bvals_path, bvecs_path, frame="world")
    np.testing.assert_array_equal(table.bvals, BVALS)
    np.testing.assert_array_equal(table.bvecs, UNIT)


@pytest.mark.parametrize(
    "bvecs",
    [UNIT[:4].T, UNIT[:4], np.zeros((2, 5)), np.zeros((5, 4)), np.zeros(3)],
    ids=["fsl_short", "swapped_short", "two_rows", "four_columns", "one_vector"],
)
def test_bvals_bvecs_mismatch_raises_naming_both_files(
    tmp_path: Path, bvecs: np.ndarray
) -> None:
    bvals_path, bvecs_path = _files(tmp_path, BVALS, bvecs)
    with pytest.raises(ValueError) as info:
        read_bvals_bvecs(bvals_path, bvecs_path, frame="world")
    assert str(bvals_path) in str(info.value)
    assert str(bvecs_path) in str(info.value)


@pytest.mark.parametrize(
    ("frame", "message"),
    [("fsl", "reference"), ("wrold", "wrold")],
    ids=["fsl_without_image", "unknown_frame"],
)
def test_a_frame_that_cannot_be_used_raises(
    tmp_path: Path, frame: str, message: str
) -> None:
    paths = _files(tmp_path, BVALS, UNIT.T)
    with pytest.raises(ValueError, match=message):
        read_bvals_bvecs(*paths, frame=frame)  # type: ignore[arg-type]
    out = tmp_path / "out.bval", tmp_path / "out.bvec"
    with pytest.raises(ValueError, match=message):
        write_bvals_bvecs(_table(), *out, frame=frame)  # type: ignore[arg-type]
    assert not out[0].exists() and not out[1].exists()


@pytest.mark.parametrize(
    ("operation", "frame"),
    [
        ("read", "fsl"),
        ("read", "world"),
        ("write", "fsl"),
        ("write", "world"),
        ("read", "b_matrix"),
        ("write", "b_matrix"),
    ],
)
def test_an_image_of_other_count_raises_naming_the_image(
    tmp_path: Path, operation: str, frame: str
) -> None:
    reading = operation == "read"
    image = _image(tmp_path / "dwi.nii", shape=(2, 2, 2, 4 if reading else 6))
    if frame == "b_matrix":
        path = tmp_path / "bmat.txt"
        if reading:
            write_b_matrix(_table(), path)
        with pytest.raises(ValueError, match=re.escape(str(image))):
            if reading:
                read_b_matrix(path, ref_path=image)
            else:
                write_b_matrix(_table(), path, ref_path=image)
        assert reading or not path.exists()
        return
    fr: Literal["fsl", "world"] = "fsl" if frame == "fsl" else "world"
    paths = _files(tmp_path, BVALS, UNIT.T)
    out = tmp_path / "out.bval", tmp_path / "out.bvec"
    with pytest.raises(ValueError, match=re.escape(str(image))):
        if reading:
            read_bvals_bvecs(*paths, frame=fr, ref_path=image)
        else:
            write_bvals_bvecs(_table(), *out, frame=fr, ref_path=image)
    assert reading or (not out[0].exists() and not out[1].exists())


@pytest.mark.parametrize("case", ["read_fsl", "read_world", "write_fsl", "b_matrix"])
def test_a_3d_image_raises(tmp_path: Path, case: str) -> None:
    image = _image(tmp_path / "t1.nii", shape=(2, 2, 5))
    match = "fourth dimension"
    if case == "b_matrix":
        path = tmp_path / "bmat.txt"
        write_b_matrix(_table(), path)
        with pytest.raises(ValueError, match=match):
            read_b_matrix(path, ref_path=image)
        out = tmp_path / "out.txt"
        with pytest.raises(ValueError, match=match):
            write_b_matrix(_table(), out, ref_path=image)
        assert not out.exists()
    elif case == "write_fsl":
        out_files = tmp_path / "out.bval", tmp_path / "out.bvec"
        with pytest.raises(ValueError, match=match):
            write_bvals_bvecs(_table(), *out_files, frame="fsl", ref_path=image)
    else:
        paths = _files(tmp_path, BVALS, UNIT.T)
        frame: Literal["fsl", "world"] = "fsl" if case == "read_fsl" else "world"
        with pytest.raises(ValueError, match=match):
            read_bvals_bvecs(*paths, frame=frame, ref_path=image)


@pytest.mark.parametrize("as_str", [False, True], ids=["path", "str"])
def test_str_and_path_arguments_are_accepted(tmp_path: Path, as_str: bool) -> None:
    def arg(p: Path) -> str | Path:
        return str(p) if as_str else p

    image = _image(tmp_path / "dwi.nii", np.diag([2.0, -2.0, 2.0]))
    bvals_path, bvecs_path = _files(tmp_path, BVALS, UNIT.T)
    table = read_bvals_bvecs(
        arg(bvals_path), arg(bvecs_path), ref_path=arg(image), frame="fsl"
    )
    out_bvals, out_bvecs = tmp_path / "out.bval", tmp_path / "out.bvec"
    write_bvals_bvecs(
        table, arg(out_bvals), arg(out_bvecs), ref_path=arg(image), frame="fsl"
    )
    np.testing.assert_allclose(np.loadtxt(out_bvecs), UNIT.T, rtol=0, atol=ATOL)


def test_written_files_are_one_row_and_three_rows_that_read_back_exactly(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((4, 3))
    table = GradientTable(np.array([5.0, 1000.123456789, 2000.0 / 3.0, 3e3]), vectors)
    bvals_path, bvecs_path = tmp_path / "bvals", tmp_path / "bvecs"
    write_bvals_bvecs(table, bvals_path, bvecs_path, frame="world")

    bval_lines = bvals_path.read_text().splitlines()
    bvec_lines = bvecs_path.read_text().splitlines()
    assert [len(line.split()) for line in bval_lines] == [4]
    assert [len(line.split()) for line in bvec_lines] == [4, 4, 4]
    np.testing.assert_array_equal(np.loadtxt(bvals_path), table.bvals)
    np.testing.assert_array_equal(np.loadtxt(bvecs_path), table.bvecs.T)


@pytest.mark.parametrize(
    ("shape", "name"),
    [((2, 3, 4), "t1.nii"), ((2, 3, 4, 5), "dwi.nii"), ((2, 3, 4, 5), "dwi.nii.gz")],
)
def test_header_gives_grid_and_full_shape(
    tmp_path: Path, shape: tuple[int, ...], name: str
) -> None:
    linear = np.diag([2.0, -3.0, 2.5])
    path = _image(tmp_path / name, linear, shape)
    expected = np.eye(4)
    expected[:3, :3] = linear
    expected[:3, 3] = (-3.0, 4.5, 1.0)
    for grid, full_shape in (read_nifti_header(path), load_header(path)):
        assert full_shape == shape
        assert grid.shape == shape[:3]
        np.testing.assert_array_equal(grid.affine, expected)


def _b_matrix_file(path: Path, rows: np.ndarray) -> Path:
    np.savetxt(path, np.atleast_2d(rows), fmt="%.17g")
    return path


def _assert_no_negative_zero(values: np.ndarray) -> None:
    assert not np.signbit(values[values == 0]).any()


def _oblique_table() -> GradientTable:
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((6, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    bvals = np.array([0.0, 500.0, 1000.0, 1000.0, 2000.0, 3000.0, 3000.0, 0.0])
    bvecs = np.vstack([[0.0, 0.0, 0.0], vectors, [0.0, 0.0, 0.0]])
    return GradientTable(bvals, bvecs)


def test_b_matrix_round_trip_gives_back_bvals_and_axes(tmp_path: Path) -> None:
    table = _oblique_table()
    path = tmp_path / "bmat.txt"
    write_b_matrix(table, path)
    again = read_b_matrix(path)
    # Trace of a %.17g text matrix: rounding of a few ulp of 3000.
    np.testing.assert_allclose(again.bvals, table.bvals, rtol=0, atol=1e-9)
    weighted = table.bvals > 0
    dots = np.sum(again.bvecs * table.bvecs, axis=1)
    np.testing.assert_allclose(np.abs(dots[weighted]), 1.0, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(again.bvecs[~weighted], 0.0)
    _assert_no_negative_zero(again.bvecs)


def test_b_matrix_round_trip_keeps_vectors_that_follow_the_sign_rule(
    tmp_path: Path,
) -> None:
    table = _oblique_table()
    first = np.argmax(np.abs(table.bvecs) > 0, axis=1)
    signs = np.sign(table.bvecs[np.arange(len(first)), first])
    signs[signs == 0] = 1.0
    signed = GradientTable(table.bvals, table.bvecs * signs[:, None])
    path = tmp_path / "bmat.txt"
    write_b_matrix(signed, path)
    np.testing.assert_allclose(
        read_b_matrix(path).bvecs, signed.bvecs, rtol=0, atol=1e-12
    )


def test_written_b_matrix_is_one_row_of_six_per_volume(tmp_path: Path) -> None:
    table = _oblique_table()
    path = tmp_path / "bmat.txt"
    write_b_matrix(table, path)
    lines = path.read_text().splitlines()
    assert [len(line.split()) for line in lines] == [6] * len(table.bvals)
    np.testing.assert_array_equal(np.loadtxt(path), table.b_matrix)


@pytest.mark.parametrize(
    ("vector", "expected"),
    [
        ([0.0, 0.6, -0.8], [0.0, 0.6, -0.8]),
        ([0.0, -0.6, 0.8], [0.0, 0.6, -0.8]),
        ([0.6, 0.0, -0.8], [0.6, 0.0, -0.8]),
        ([-0.6, 0.0, 0.8], [0.6, 0.0, -0.8]),
        ([0.0, 0.0, -1.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
        ([-0.48, 0.6, 0.64], [0.48, -0.6, -0.64]),
    ],
    ids=[
        "gx0_kept",
        "gx0_flipped",
        "gy0_kept",
        "gy0_flipped",
        "z_only_flipped",
        "z_only_kept",
        "oblique_flipped",
    ],
)
def test_b_matrix_vector_has_first_nonzero_component_positive(
    tmp_path: Path, vector: list[float], expected: list[float]
) -> None:
    table = GradientTable(np.array([0.0, 1000.0]), np.array([[0.0, 0.0, 0.0], vector]))
    path = tmp_path / "bmat.txt"
    write_b_matrix(table, path)
    again = read_b_matrix(path)
    np.testing.assert_allclose(again.bvecs[1], expected, rtol=0, atol=1e-12)
    _assert_no_negative_zero(again.bvecs)
    _assert_no_negative_zero(again.bvals)


@pytest.mark.parametrize(
    ("gx", "expected_y"),
    [(-1e-9, 0.6), (1e-9, 0.6)],
    ids=["negative_tiny_x", "positive_tiny_x"],
)
def test_tiny_first_component_is_ignored_by_the_sign_rule(
    tmp_path: Path, gx: float, expected_y: float
) -> None:
    # b * g g^T for g = (gx, 0.6, -0.8), b = 1000, as a text file would hold it:
    # gx^2 rounds away but the cross terms keep the sign of gx.
    gy, gz, b = 0.6, -0.8, 1000.0
    row = [0.0, 2 * b * gx * gy, 2 * b * gx * gz, b * gy**2, 2 * b * gy * gz, b * gz**2]
    again = read_b_matrix(_b_matrix_file(tmp_path / "bmat.txt", np.array(row)))
    assert again.bvecs[0, 1] == pytest.approx(expected_y, abs=1e-12)
    np.testing.assert_allclose(again.bvecs[0], [0.0, 0.6, -0.8], rtol=0, atol=1e-8)


def test_b0_row_with_negative_zeros_reads_as_positive_zeros(tmp_path: Path) -> None:
    rows = np.array([[0.0, -0.0, -0.0, 0.0, 0.0, 0.0], [1000.0, 0, 0, 0, 0, 0]])
    again = read_b_matrix(_b_matrix_file(tmp_path / "bmat.txt", rows))
    np.testing.assert_array_equal(again.bvals, [0.0, 1000.0])
    np.testing.assert_array_equal(again.bvecs, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    _assert_no_negative_zero(again.bvals)
    _assert_no_negative_zero(again.bvecs)


def test_tiny_trace_reads_as_b0(tmp_path: Path) -> None:
    # Rounding left on a b0 row by a writer with fixed decimals.
    rows = np.array([[1e-8, 0.0, 0.0, 2e-8, 0.0, -1e-8], [0.0, 0, 0, 0, 0, 1000.0]])
    again = read_b_matrix(_b_matrix_file(tmp_path / "bmat.txt", rows))
    np.testing.assert_array_equal(again.bvals, [0.0, 1000.0])
    np.testing.assert_array_equal(again.bvecs[0], [0.0, 0.0, 0.0])


def test_single_row_b_matrix_is_one_volume(tmp_path: Path) -> None:
    table = GradientTable(np.array([1000.0]), np.array([[0.0, 0.6, -0.8]]))
    path = tmp_path / "bmat.txt"
    write_b_matrix(table, path)
    assert np.loadtxt(path).shape == (6,)
    again = read_b_matrix(path)
    assert again.bvecs.shape == (1, 3)
    np.testing.assert_allclose(again.bvals, [1000.0], rtol=0, atol=1e-9)
    np.testing.assert_allclose(again.bvecs[0], [0.0, 0.6, -0.8], rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "rows",
    [np.zeros((3, 5)), np.zeros((3, 7)), np.zeros(5), np.zeros((6, 2))],
    ids=["five_columns", "seven_columns", "one_row_of_five", "transposed"],
)
def test_b_matrix_of_wrong_shape_raises_naming_the_file(
    tmp_path: Path, rows: np.ndarray
) -> None:
    path = _b_matrix_file(tmp_path / "bmat.txt", rows)
    with pytest.raises(ValueError, match=re.escape(str(path))):
        read_b_matrix(path)


@pytest.mark.parametrize(
    "text",
    ["1 2 3 4 5 6\n1 2 3\n", "a b c d e f\n"],
    ids=["ragged", "not_numbers"],
)
def test_unreadable_b_matrix_raises_naming_the_file(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bmat.txt"
    path.write_text(text)
    with pytest.raises(ValueError, match=re.escape(str(path))) as info:
        read_b_matrix(path)
    assert "cannot be read as a table of numbers" in str(info.value)


def test_missing_b_matrix_raises_naming_the_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.txt"
    with pytest.raises(ValueError, match=re.escape(f"{path} does not exist")):
        read_b_matrix(path)


@pytest.mark.parametrize("which", ["bvals", "bvecs", "b_matrix"])
def test_empty_file_raises_naming_the_file(tmp_path: Path, which: str) -> None:
    paths = _files(tmp_path, BVALS, UNIT.T)
    empty = {"bvals": paths[0], "bvecs": paths[1], "b_matrix": tmp_path / "bmat.txt"}[
        which
    ]
    empty.write_text("")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # numpy's own empty-file warning must not leak
        with pytest.raises(ValueError, match=re.escape(f"{empty} is empty")):
            if which == "b_matrix":
                read_b_matrix(empty)
            else:
                read_bvals_bvecs(*paths, frame="world")


def test_bvals_with_several_rows_and_columns_raises(tmp_path: Path) -> None:
    bvals_path, bvecs_path = _files(tmp_path, BVALS[:4], UNIT[:4].T)
    bvals_path.write_text("0 1000\n2000 3000\n")
    with pytest.raises(ValueError, match=re.escape(f"{bvals_path} has shape (2, 2)")):
        read_bvals_bvecs(bvals_path, bvecs_path, frame="world")


def test_bvecs_passed_as_bvals_raises_naming_the_bvals_file(tmp_path: Path) -> None:
    _, bvecs_path = _files(tmp_path, BVALS, UNIT.T)
    with pytest.raises(ValueError, match=re.escape(f"{bvecs_path} has shape (3, 5)")):
        read_bvals_bvecs(bvecs_path, bvecs_path, frame="world")


@pytest.mark.parametrize("as_str", [False, True], ids=["path", "str"])
def test_b_matrix_accepts_str_and_path_arguments(tmp_path: Path, as_str: bool) -> None:
    def arg(p: Path) -> str | Path:
        return str(p) if as_str else p

    image = _image(tmp_path / "dwi.nii")
    path = tmp_path / "bmat.txt"
    write_b_matrix(_table(), arg(path), ref_path=arg(image))
    again = read_b_matrix(arg(path), ref_path=arg(image))
    np.testing.assert_allclose(again.bvals, BVALS, rtol=0, atol=1e-9)


def test_matlab_b_matrix_matches_its_bvals_and_bvecs() -> None:
    # MRIToolkit wrote bmat.txt from these files, in the frame of the bvecs file.
    expected = read_bvals_bvecs(
        MATLAB_DATA / "bvals.txt", MATLAB_DATA / "bvecs.txt", frame="world"
    )
    table = read_b_matrix(MATLAB_DATA / "bmat.txt")
    # bmat.txt has 8 decimals: the trace of three entries is within 1.5e-8.
    np.testing.assert_allclose(table.bvals, expected.bvals, rtol=0, atol=1e-6)
    weighted = expected.bvals > 0
    dots = np.sum(table.bvecs * expected.bvecs, axis=1)[weighted]
    np.testing.assert_allclose(np.abs(dots), 1.0, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(table.bvecs[~weighted], 0.0)
    _assert_no_negative_zero(table.bvecs)


def test_non_rank_one_b_matrix_gives_trace_and_principal_axis(tmp_path: Path) -> None:
    # Not a b g g^T: an isotropic part is added. No warning is logged; b is the
    # trace and the vector the axis of the largest eigenvalue.
    axis = np.array([0.48, 0.6, 0.64])
    tensor = 1000.0 * np.outer(axis, axis) + 30.0 * np.eye(3)
    row = [
        tensor[0, 0],
        2 * tensor[0, 1],
        2 * tensor[0, 2],
        tensor[1, 1],
        2 * tensor[1, 2],
        tensor[2, 2],
    ]
    again = read_b_matrix(_b_matrix_file(tmp_path / "bmat.txt", np.array(row)))
    np.testing.assert_allclose(again.bvals, [1090.0], rtol=0, atol=1e-9)
    np.testing.assert_allclose(again.bvecs[0], axis, rtol=0, atol=1e-12)


# Re-storing the same anatomy with the real reorientation functions: the table
# stays in world coordinates, and only the saved file's affine changes the bvecs.
RAS = np.diag([2.0, 2.0, 2.0])
LAS = np.diag([-2.0, 2.0, 2.0])
LPS = np.diag([-2.0, -2.0, 2.0])
REORIENTED: dict[str, tuple[np.ndarray, Callable[[Volume], None], list[float]]] = {
    # FSL bvecs do not change when i is flipped: j_flipped and k_flipped are the
    # cases that catch a flip that forgets the affine.
    "i_flipped": (RAS, lambda v: v.flip_axis(0), [0.48, 0.60, 0.64]),
    "j_flipped": (RAS, lambda v: v.flip_axis(1), [-0.48, -0.60, 0.64]),
    "k_flipped": (RAS, lambda v: v.flip_axis(2), [-0.48, 0.60, -0.64]),
    "ij_swapped": (RAS, lambda v: v.permute_axes((1, 0, 2)), [0.60, -0.48, 0.64]),
    "ij_swapped_from_las": (
        LAS,
        lambda v: v.permute_axes((1, 0, 2)),
        [-0.60, 0.48, 0.64],
    ),
    # Beyond the table: F @ R.T @ WORLD for the resulting affine.
    "lps_to_canonical": (LPS, lambda v: v.to_closest_canonical(), [0.48, 0.60, 0.64]),
    "ras_to_asr": (RAS, lambda v: v.to_orientation("ASR"), [-0.60, 0.64, -0.48]),
}


def _column_in(linear: np.ndarray) -> np.ndarray:
    """The bvecs column of WORLD for an image whose 3x3 affine is `linear`."""
    flip = np.diag([-1.0, 1.0, 1.0]) if np.linalg.det(linear) > 0 else np.eye(3)
    rotation = linear / np.linalg.norm(linear, axis=0)
    column: np.ndarray = flip @ rotation.T @ WORLD
    return column


@pytest.mark.parametrize(
    ("start", "reorient", "column"), list(REORIENTED.values()), ids=list(REORIENTED)
)
def test_bvecs_follow_the_saved_reoriented_image(
    tmp_path: Path,
    start: np.ndarray,
    reorient: Callable[[Volume], None],
    column: list[float],
) -> None:
    image = _image(tmp_path / "dwi.nii", start, (2, 3, 4, 2))
    paths = _files(
        tmp_path, np.array([0.0, 1000.0]), np.c_[[0, 0, 0], _column_in(start)]
    )
    table = read_bvals_bvecs(*paths, frame="fsl", ref_path=image)

    vol = load_volume(image)
    reorient(vol)
    save_volume(vol, tmp_path / "out.nii")
    out_bvals, out_bvecs = tmp_path / "out.bval", tmp_path / "out.bvec"
    write_bvals_bvecs(
        table, out_bvals, out_bvecs, frame="fsl", ref_path=tmp_path / "out.nii"
    )

    np.testing.assert_allclose(np.loadtxt(out_bvecs)[:, 1], column, rtol=0, atol=ATOL)
    # Round trip: written and read with the same file, not a reorientation check.
    back = read_bvals_bvecs(
        out_bvals, out_bvecs, frame="fsl", ref_path=tmp_path / "out.nii"
    )
    np.testing.assert_allclose(back.bvecs, table.bvecs, rtol=0, atol=ATOL)


# MRtrix strides and the Larmor reorientation that stores the image the same way.
STRIDES: dict[str, Callable[[Volume], None]] = {
    "-1,2,3,4": lambda v: v.flip_axis(0),
    "1,-2,3,4": lambda v: v.flip_axis(1),
    "1,2,-3,4": lambda v: v.flip_axis(2),
    "2,1,3,4": lambda v: v.permute_axes((1, 0, 2)),
    "3,1,2,4": lambda v: v.permute_axes((1, 2, 0)),
}


@pytest.mark.skipif(shutil.which("mrconvert") is None, reason="MRtrix3 is not on PATH")
@pytest.mark.parametrize(("strides", "reorient"), STRIDES.items(), ids=list(STRIDES))
def test_reoriented_image_and_bvecs_match_mrtrix(
    tmp_path: Path, strides: str, reorient: Callable[[Volume], None]
) -> None:
    # Not a cube and distinct values, so the data shows any wrong flip or transpose.
    image = _image(tmp_path / "dwi.nii", RAS, (2, 3, 4, 5), distinct=True)
    bvals_path, bvecs_path = _files(tmp_path, BVALS, UNIT.T)
    mr_image, mr_bvecs = tmp_path / "mr.nii", tmp_path / "mr.bvec"
    subprocess.run(
        [
            "mrconvert",
            "-quiet",
            "-force",
            str(image),
            "-fslgrad",
            str(bvecs_path),
            str(bvals_path),
            "-strides",
            strides,
            str(mr_image),
            "-export_grad_fsl",
            str(mr_bvecs),
            str(tmp_path / "mr.bval"),
        ],
        check=True,
    )

    table = read_bvals_bvecs(bvals_path, bvecs_path, frame="fsl", ref_path=image)
    vol = load_volume(image)
    reorient(vol)
    save_volume(vol, tmp_path / "out.nii")
    out_bvecs = tmp_path / "out.bvec"
    write_bvals_bvecs(
        table,
        tmp_path / "out.bval",
        out_bvecs,
        frame="fsl",
        ref_path=tmp_path / "out.nii",
    )

    grid, shape = load_header(tmp_path / "out.nii")
    mr_grid, mr_shape = load_header(mr_image)
    assert shape == mr_shape
    np.testing.assert_allclose(grid.affine, mr_grid.affine, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(
        load_volume(tmp_path / "out.nii").data, load_volume(mr_image).data
    )
    # MRtrix writes 6 significant digits.
    np.testing.assert_allclose(
        np.loadtxt(out_bvecs), np.loadtxt(mr_bvecs), rtol=0, atol=1e-5
    )
