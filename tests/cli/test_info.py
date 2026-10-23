from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner, Result

from larmor.io import (
    TRACTOGRAM_EXTENSIONS,
    VOLUME_EXTENSIONS,
    save_tractogram,
    save_volume,
)
from larmor.main import app
from larmor.types import GradientTable, Grid, Tractogram, Volume

# Every extension Larmor reads, so a new one is covered as soon as it is added.
VOLUME_NAMES = [f"a{e}" for exts in VOLUME_EXTENSIONS.values() for e in exts]
TRACTOGRAM_NAMES = [f"t{e}" for exts in TRACTOGRAM_EXTENSIONS.values() for e in exts]

LPS = np.diag([-2.0, -2.0, 2.5, 1.0])
LPS[:3, 3] = (96.0, 110.0, -72.0)
RAS = np.diag([1.25, 1.25, 1.25, 1.0])


def _run(path: Path) -> Result:
    return CliRunner().invoke(app, ["info", str(path)], env={"TERM": "dumb"})


def _fields(result: Result) -> dict[str, str]:
    """The `label: value` lines of the output; continuation lines are skipped."""
    fields = {}
    for line in result.output.splitlines():
        label, sep, value = line.partition(":")
        if sep and not line.startswith(" "):
            fields[label] = value.strip()
    return fields


@pytest.mark.parametrize(
    ("shape", "affine", "orientation", "sizes"),
    [
        ((4, 5, 6), LPS, "LPS", "2 x 2 x 2.5 mm"),
        ((4, 5, 6, 7), LPS, "LPS", "2 x 2 x 2.5 mm"),
        ((3, 3, 3), RAS, "RAS", "1.25 x 1.25 x 1.25 mm"),
    ],
)
@pytest.mark.parametrize("name", VOLUME_NAMES)
def test_volume_shows_shape_voxel_size_orientation_and_affine(
    tmp_path: Path,
    name: str,
    shape: tuple[int, ...],
    affine: np.ndarray,
    orientation: str,
    sizes: str,
) -> None:
    path = tmp_path / name
    save_volume(Volume(np.zeros(shape, np.int16), affine), path)
    result = _run(path)

    assert result.exit_code == 0, result.output
    fields = _fields(result)
    assert fields["Shape"] == " x ".join(str(n) for n in shape)
    assert fields["Voxel size"] == sizes
    assert fields["Orientation"] == orientation
    lines = result.output.splitlines()
    first = lines.index(next(line for line in lines if line.startswith("Affine:")))
    rows = [lines[first].partition(":")[2], lines[first + 1], lines[first + 2]]
    shown = np.array([[float(v) for v in row.split()] for row in rows])
    np.testing.assert_allclose(shown, affine[:3], atol=1e-4)
    assert "Shells" not in fields


def test_volume_data_is_not_read(tmp_path: Path) -> None:
    # Only the header is needed: a file whose data was cut off still shows.
    path = tmp_path / "a.nii"
    save_volume(Volume(np.zeros((40, 40, 40, 5), np.int16), LPS), path)
    path.write_bytes(path.read_bytes()[:400])
    result = _run(path)

    assert result.exit_code == 0, result.output
    assert _fields(result)["Shape"] == "40 x 40 x 40 x 5"


@pytest.mark.parametrize("name", ["dwi.mif", "dwi.mif.gz", "dwi.mih"])
def test_mif_with_gradients_shows_b0s_and_shells(tmp_path: Path, name: str) -> None:
    bvals = np.array([0.0, 5.0, 1000.0, 1000.0, 2000.0, 2000.0])
    bvecs = np.vstack([np.zeros((2, 3)), np.eye(3)[[0, 1, 0, 2]]])
    path = tmp_path / name
    save_volume(
        Volume(np.zeros((2, 2, 2, 6), np.int16), RAS),
        path,
        gradients=GradientTable(bvals, bvecs),
    )
    fields = _fields(_run(path))

    assert fields["b0 volumes"] == "2"
    assert fields["Shells"] == "1000 (2), 2000 (2)"


def test_unreadable_gradient_table_still_shows_the_header(tmp_path: Path) -> None:
    # A dw_scheme line of three numbers: the table cannot be read, but the shape and
    # orientation are what someone checking a broken file needs.
    path = tmp_path / "dwi.mif"
    gtab = GradientTable(np.array([0.0, 1000.0]), np.array([[0, 0, 0], [1.0, 0, 0]]))
    save_volume(Volume(np.zeros((2, 2, 2, 2), np.int16), RAS), path, gradients=gtab)
    raw = path.read_bytes()
    start = raw.index(b"dw_scheme: ")
    end = raw.index(b"\n", start)
    path.write_bytes(
        raw[:start] + b"dw_scheme: 1,0,0" + b" " * (end - start - 16) + raw[end:]
    )
    result = _run(path)

    assert result.exit_code == 0, result.output
    fields = _fields(result)
    assert fields["Shape"] == "2 x 2 x 2 x 2"
    assert fields["Orientation"] == "RAS"
    assert fields["Gradients"].startswith("unreadable: ")
    assert "4 numbers" in fields["Gradients"]
    assert "Shells" not in fields


TRACTOGRAM = Tractogram(
    np.arange(30, dtype=np.float32).reshape(10, 3),
    np.array([0, 4, 4, 10]),
    point_data={"fa": np.ones(10, np.float32), "dir": np.ones((10, 3), np.float32)},
    streamline_data={"rgb": np.ones((3, 3), np.float32)},
    reference=Grid((4, 5, 6), LPS),
)


@pytest.mark.parametrize("name", TRACTOGRAM_NAMES)
def test_tractogram_shows_counts_and_points_per_line(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    save_tractogram(TRACTOGRAM, path)
    result = _run(path)

    assert result.exit_code == 0, result.output
    fields = _fields(result)
    assert fields["Streamlines"] == "3"
    assert fields["Points"] == "10"
    assert fields["Points per line"] == "0 / 3.3 / 6 (min / mean / max)"


@pytest.mark.parametrize("name", ["t.trk", "t.vtk"])
def test_tractogram_data_is_listed_with_its_width(tmp_path: Path, name: str) -> None:
    save_tractogram(TRACTOGRAM, tmp_path / name)
    fields = _fields(_run(tmp_path / name))

    assert set(fields["Point data"].split(", ")) == {"fa (1)", "dir (3)"}
    assert fields["Streamline data"] == "rgb (3)"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "t.tck",
            {"Point data": "none", "Streamline data": "none", "Reference": "none"},
        ),
        (
            "t.trk",
            {
                "Reference": "4 x 5 x 6",
                "Voxel size": "2 x 2 x 2.5 mm",
                "Orientation": "LPS",
            },
        ),
    ],
    ids=["tck_has_no_data_and_no_reference", "trk_shows_its_reference_grid"],
)
def test_the_fields_of_a_tractogram_format(
    tmp_path: Path, name: str, expected: dict[str, str]
) -> None:
    save_tractogram(TRACTOGRAM, tmp_path / name)
    fields = _fields(_run(tmp_path / name))

    for key, value in expected.items():
        assert fields[key] == value


def test_large_counts_have_thousands_separators(tmp_path: Path) -> None:
    lengths = np.full(1500, 2)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    t = Tractogram(np.zeros((3000, 3), np.float32), offsets)
    save_tractogram(t, tmp_path / "t.tck")
    fields = _fields(_run(tmp_path / "t.tck"))

    assert fields["Streamlines"] == "1,500"
    assert fields["Points"] == "3,000"


def test_empty_tractogram_has_no_points_per_line(tmp_path: Path) -> None:
    save_tractogram(Tractogram(np.zeros((0, 3)), np.array([0])), tmp_path / "e.tck")
    result = _run(tmp_path / "e.tck")

    assert result.exit_code == 0, result.output
    fields = _fields(result)
    assert fields["Streamlines"] == "0"
    assert "Points per line" not in fields


@pytest.mark.parametrize("name", ["a.txt", "a.nii.bz2", "a"])
def test_unsupported_extension_lists_both_kinds(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text("x")
    result = _run(tmp_path / name)

    assert result.exit_code == 1
    assert result.output.startswith("ERROR: ")
    assert len(result.output.strip().splitlines()) == 1
    assert ".nii.gz" in result.output and ".trk" in result.output


@pytest.mark.parametrize("name", ["a.nii", "a.tck"])
def test_unreadable_file_is_a_one_line_error(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_bytes(b"not a real file")
    result = _run(tmp_path / name)

    assert result.exit_code == 1
    assert result.output.startswith("ERROR: ")
    assert len(result.output.strip().splitlines()) == 1
