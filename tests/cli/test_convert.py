import itertools
import logging
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import pytest
from typer.testing import CliRunner, Result

from larmor.io import (
    TRACTOGRAM_EXTENSIONS,
    VOLUME_EXTENSIONS,
    has_mif_gradients,
    load_tractogram,
    load_volume,
    read_bvals_bvecs,
    save_tractogram,
    save_volume,
)
from larmor.io.formats import read_mif_gradients
from larmor.main import app
from larmor.types import GradientTable, Grid, Tractogram, Volume

VOLUME_EXTS = [e for exts in VOLUME_EXTENSIONS.values() for e in exts]
TRACTOGRAM_EXTS = [e for exts in TRACTOGRAM_EXTENSIONS.values() for e in exts]

# Oblique and LPS-like (det > 0), so FSL bvecs differ from world directions.
AFFINE = np.array(
    [[-2.0, 0.2, 0, 30], [0.1, -2.0, 0, 40], [0, 0, 2.5, -10], [0, 0, 0, 1]]
)
DATA = np.arange(2 * 3 * 4 * 5, dtype=np.int16).reshape(2, 3, 4, 5)
TABLE = GradientTable(
    np.array([0.0, 1000.0, 1000.0, 2000.0, 2000.0]),
    np.array([[0, 0, 0], [1, 0, 0], [0, 0.6, 0.8], [0.48, 0.6, 0.64], [0, 1, 0]]),
)
GRID = Grid((10, 12, 14), np.diag([2.0, 2.0, 2.0, 1.0]))
TRACTOGRAM = Tractogram(
    (np.arange(30, dtype=np.float32).reshape(10, 3) / 4) + 1,
    np.array([0, 4, 6, 10]),
    point_data={"fa": np.linspace(0, 1, 10, dtype=np.float32)},
    streamline_data={"len": np.array([4.0, 2.0, 4.0], np.float32)},
)


Arg = str | Path


def _run(*args: Arg) -> Result:
    return CliRunner().invoke(app, ["convert", *map(str, args)], env={"TERM": "dumb"})


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def _assert_same_table(a: GradientTable, b: GradientTable) -> None:
    np.testing.assert_allclose(a.bvals, b.bvals, rtol=0, atol=1e-9)
    np.testing.assert_allclose(a.bvecs, b.bvecs, rtol=0, atol=1e-6)


# Images


@pytest.mark.parametrize(("src", "dst"), list(itertools.product(VOLUME_EXTS, repeat=2)))
def test_every_image_pair_keeps_data_affine_and_scaling(
    tmp_path: Path, src: str, dst: str
) -> None:
    source = tmp_path / f"in{src}"
    save_volume(Volume(DATA, AFFINE, slope=0.5, intercept=-3.0), source)
    result = _run(source, tmp_path / f"out{dst}")

    assert result.exit_code == 0, result.output
    out = load_volume(tmp_path / f"out{dst}")
    np.testing.assert_array_equal(out.data, DATA)
    np.testing.assert_allclose(out.affine, AFFINE, rtol=0, atol=1e-5)
    assert (out.slope, out.intercept) == (0.5, -3.0)


@pytest.mark.parametrize("dst", [".mif", ".mif.gz", ".mih"])
def test_mif_table_is_kept_in_a_mif_output(tmp_path: Path, dst: str) -> None:
    save_volume(Volume(DATA, AFFINE), tmp_path / "in.mif", gradients=TABLE)
    result = _run(tmp_path / "in.mif", tmp_path / f"out{dst}")

    assert result.exit_code == 0, result.output
    _assert_same_table(read_mif_gradients(tmp_path / f"out{dst}"), TABLE)


def test_mif_table_to_nifti_is_written_to_the_requested_files(tmp_path: Path) -> None:
    save_volume(Volume(DATA, AFFINE), tmp_path / "in.mif", gradients=TABLE)
    out, bvals, bvecs = (
        tmp_path / "out.nii.gz",
        tmp_path / "b.bval",
        tmp_path / "b.bvec",
    )
    result = _run(tmp_path / "in.mif", out, "--bvals-out", bvals, "--bvecs-out", bvecs)

    assert result.exit_code == 0, result.output
    assert result.output == ""
    # FSL bvecs are relative to the image written next to them.
    _assert_same_table(read_bvals_bvecs(bvals, bvecs, ref_path=out), TABLE)


def test_mif_table_to_nifti_without_files_is_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    save_volume(Volume(DATA, AFFINE), tmp_path / "in.mif", gradients=TABLE)
    result = _run(tmp_path / "in.mif", tmp_path / "out.nii")

    assert result.exit_code == 0, result.output
    (warning,) = _warnings(caplog)
    assert "out.nii" in warning and "--bvals-out" in warning


@pytest.mark.parametrize("frame", ["fsl", "world"])
def test_nifti_with_gradient_files_becomes_a_mif_with_the_table(
    tmp_path: Path, frame: Literal["fsl", "world"]
) -> None:
    source = tmp_path / "in.nii.gz"
    save_volume(Volume(DATA, AFFINE), source)
    bvals, bvecs = tmp_path / "bvals", tmp_path / "bvecs"
    np.savetxt(bvals, TABLE.bvals[None])
    np.savetxt(bvecs, TABLE.bvecs.T)
    options: list[Arg] = ["--bvals", bvals, "--bvecs", bvecs, "--bvec-frame", frame]
    result = _run(source, tmp_path / "out.mif", *options)

    assert result.exit_code == 0, result.output
    expected = read_bvals_bvecs(bvals, bvecs, ref_path=source, frame=frame)
    _assert_same_table(read_mif_gradients(tmp_path / "out.mif"), expected)


def test_image_without_a_table_has_none_after_conversion(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    save_volume(Volume(DATA, AFFINE), tmp_path / "in.nii")
    result = _run(tmp_path / "in.nii", tmp_path / "out.mif")

    assert result.exit_code == 0, result.output
    assert not has_mif_gradients(tmp_path / "out.mif")
    assert _warnings(caplog) == []


def test_out_bvals_without_a_table_is_refused(tmp_path: Path) -> None:
    save_volume(Volume(DATA, AFFINE), tmp_path / "in.nii")
    options: list[Arg] = [
        "--bvals-out",
        tmp_path / "b.bval",
        "--bvecs-out",
        tmp_path / "b.bvec",
    ]
    result = _run(tmp_path / "in.nii", tmp_path / "out.nii", *options)

    assert result.exit_code == 1
    assert "no gradient table" in result.output
    assert not (tmp_path / "out.nii").exists()


@pytest.mark.parametrize(
    ("src", "dst", "option"),
    [
        ("missing.mif", "out.nii", "--bvals-out"),
        ("missing.mif", "out.nii", "--bvecs-out"),
        ("missing.nii", "out.mif", "--bvals"),
        ("missing.nii", "out.mif", "--bvecs"),
    ],
    ids=["out-bvals", "out-bvecs", "bvals", "bvecs"],
)
def test_one_gradient_file_alone_is_a_usage_error(
    tmp_path: Path, src: str, dst: str, option: str
) -> None:
    result = _run(src, tmp_path / dst, option, tmp_path / "b")
    assert result.exit_code == 2


@pytest.mark.parametrize(("src", "dst"), [("a.tck", "b.vtk"), ("a.vtk", "b.fib")])
def test_reference_without_a_trk_is_a_usage_error(
    tmp_path: Path, src: str, dst: str
) -> None:
    # It would change nothing: only a .trk input or output uses a grid.
    result = _run(tmp_path / src, tmp_path / dst, "--reference", "ref.nii")
    assert result.exit_code == 2
    assert "--reference" in result.output


def test_reference_for_an_image_is_a_usage_error(tmp_path: Path) -> None:
    result = _run("missing.nii", tmp_path / "out.mif", "--reference", "ref.nii")
    assert result.exit_code == 2


# Reorientation to RAS


def _oriented(codes: str) -> np.ndarray:
    """A slightly oblique affine whose closest orientation is `codes`."""
    axis = {"L": (0, -1), "R": (0, 1), "P": (1, -1), "A": (1, 1)}
    axis |= {"I": (2, -1), "S": (2, 1)}
    affine = np.eye(4)
    affine[:3, :3] = 0.05
    for voxel_axis, code in enumerate(codes):
        world_axis, sign = axis[code]
        affine[world_axis, voxel_axis] = sign * (1.5 + voxel_axis * 0.5)
    affine[:3, 3] = [12.0, -7.0, 3.0]
    return affine


def _assert_same_world_values(a: Volume, b: Volume) -> None:
    ijk = np.indices(a.data.shape[:3]).reshape(3, -1)
    world = a.affine[:3, :3] @ ijk + a.affine[:3, 3:]
    mapped = np.linalg.solve(b.affine[:3, :3], world - b.affine[:3, 3:])
    rounded = np.rint(mapped).astype(int)
    # Reorienting only permutes and flips axes, so voxel centres coincide.
    np.testing.assert_allclose(mapped, rounded, rtol=0, atol=1e-4)
    np.testing.assert_array_equal(b.data[tuple(rounded)], a.data[tuple(ijk)])


@pytest.mark.parametrize("codes", ["LAS", "LPS", "LIA"])
@pytest.mark.parametrize("shape", [(2, 3, 4), (2, 3, 4, 2)])
@pytest.mark.parametrize("dst", [".nii.gz", ".mif"])
def test_to_ras_keeps_the_value_at_every_world_position(
    tmp_path: Path, codes: str, shape: tuple[int, ...], dst: str
) -> None:
    source = tmp_path / "in.nii.gz"
    original = Volume(
        np.arange(np.prod(shape), dtype=np.int16).reshape(shape), _oriented(codes)
    )
    save_volume(original, source)
    assert "".join(nib.aff2axcodes(load_volume(source).affine)) == codes
    result = _run(source, tmp_path / f"out{dst}", "--to-ras")

    assert result.exit_code == 0, result.output
    out = load_volume(tmp_path / f"out{dst}")
    assert nib.aff2axcodes(out.affine) == ("R", "A", "S")
    assert out.data.shape[3:] == shape[3:]
    _assert_same_world_values(load_volume(source), out)


@pytest.mark.parametrize("dst", [".nii.gz", ".mif"])
def test_to_ras_leaves_a_ras_image_unchanged(tmp_path: Path, dst: str) -> None:
    affine = _oriented("RAS")
    save_volume(Volume(DATA, affine), tmp_path / "in.nii.gz")
    result = _run(tmp_path / "in.nii.gz", tmp_path / f"out{dst}", "--to-ras")

    assert result.exit_code == 0, result.output
    out = load_volume(tmp_path / f"out{dst}")
    np.testing.assert_array_equal(out.data, DATA)
    np.testing.assert_allclose(out.affine, affine, rtol=0, atol=1e-5)


# Not LAS: its FSL bvecs equal those of its RAS reorientation.
@pytest.mark.parametrize("codes", ["LPS", "LIA"])
def test_to_ras_keeps_the_world_gradient_directions(tmp_path: Path, codes: str) -> None:
    source = tmp_path / "in.nii.gz"
    save_volume(Volume(DATA, _oriented(codes)), source)
    bvals, bvecs = tmp_path / "bvals", tmp_path / "bvecs"
    np.savetxt(bvals, TABLE.bvals[None])
    np.savetxt(bvecs, TABLE.bvecs.T)
    world = read_bvals_bvecs(bvals, bvecs, ref_path=source)
    outputs = {
        "out.nii.gz": ("o.bval", "o.bvec"),
        "out.mif": ("m.bval", "m.bvec"),
    }
    for name, (out_bvals, out_bvecs) in outputs.items():
        result = _run(
            source,
            tmp_path / name,
            "--bvals",
            bvals,
            "--bvecs",
            bvecs,
            "--to-ras",
            "--bvals-out",
            tmp_path / out_bvals,
            "--bvecs-out",
            tmp_path / out_bvecs,
        )
        assert result.exit_code == 0, result.output
        written = read_bvals_bvecs(
            tmp_path / out_bvals, tmp_path / out_bvecs, ref_path=tmp_path / name
        )
        _assert_same_table(written, world)
    _assert_same_table(read_mif_gradients(tmp_path / "out.mif"), world)


def test_to_ras_with_a_tractogram_is_refused_before_writing(tmp_path: Path) -> None:
    save_tractogram(TRACTOGRAM, tmp_path / "in.tck")
    result = _run(tmp_path / "in.tck", tmp_path / "out.vtk", "--to-ras")

    assert result.exit_code == 2
    assert "--to-ras" in result.output
    assert not (tmp_path / "out.vtk").exists()


# Tractograms


def _tractogram_pairs() -> list[tuple[str, str]]:
    return [(a, b) for a in TRACTOGRAM_EXTS for b in TRACTOGRAM_EXTS]


def _holds_data(ext: str) -> bool:
    """Whether a tractogram format stores point and streamline data (TCK does not)."""
    return not ext.startswith(".tck")


@pytest.mark.parametrize(("src", "dst"), _tractogram_pairs())
def test_every_tractogram_pair_keeps_the_streamlines_and_their_data(
    tmp_path: Path, src: str, dst: str, caplog: pytest.LogCaptureFixture
) -> None:
    source = tmp_path / f"in{src}"
    save_tractogram(TRACTOGRAM, source, reference=GRID)
    caplog.clear()
    reference = tmp_path / "ref.nii.gz"
    save_volume(Volume(np.zeros(GRID.shape, np.uint8), GRID.affine), reference)
    options: list[Arg] = ["--reference", reference] if dst.startswith(".trk") else []
    result = _run(source, tmp_path / f"out{dst}", *options)

    assert result.exit_code == 0, result.output
    out = load_tractogram(tmp_path / f"out{dst}")
    np.testing.assert_array_equal(out.offsets, TRACTOGRAM.offsets)
    np.testing.assert_allclose(out.points, TRACTOGRAM.points, rtol=0, atol=1e-4)
    if dst.startswith(".trk"):
        assert out.reference is not None and out.reference.matches(GRID)

    carried = _holds_data(src) and _holds_data(dst)
    for kept, original in (
        (out.point_data, TRACTOGRAM.point_data),
        (out.streamline_data, TRACTOGRAM.streamline_data),
    ):
        assert set(kept) == (set(original) if carried else set())
        for name in kept:
            np.testing.assert_allclose(kept[name], original[name], rtol=0, atol=1e-6)
    # Data a TCK cannot hold is dropped with a warning, never silently.
    dropped = _holds_data(src) and not _holds_data(dst)
    assert any("not written" in w for w in _warnings(caplog)) == dropped


def test_trk_to_trk_and_other_formats_needs_no_reference(tmp_path: Path) -> None:
    save_tractogram(TRACTOGRAM, tmp_path / "in.trk", reference=GRID)
    for name in ("out.trk", "out.tck", "out.vtk"):
        result = _run(tmp_path / "in.trk", tmp_path / name)
        assert result.exit_code == 0, result.output
    out = load_tractogram(tmp_path / "out.trk")
    assert out.reference is not None and out.reference.matches(GRID)


def test_data_is_kept_between_formats_that_hold_it(tmp_path: Path) -> None:
    save_tractogram(TRACTOGRAM, tmp_path / "in.trk", reference=GRID)
    result = _run(tmp_path / "in.trk", tmp_path / "out.vtk")

    assert result.exit_code == 0, result.output
    out = load_tractogram(tmp_path / "out.vtk")
    np.testing.assert_allclose(out.point_data["fa"], TRACTOGRAM.point_data["fa"])
    np.testing.assert_allclose(out.streamline_data["len"], [4.0, 2.0, 4.0])


@pytest.mark.parametrize("src", [".tck", ".vtk", ".fib.gz"])
def test_trk_output_without_a_grid_needs_reference_before_reading(
    tmp_path: Path, src: str
) -> None:
    # The input does not exist: the option is checked before anything is read.
    result = _run(tmp_path / f"missing{src}", tmp_path / "out.trk")

    assert result.exit_code == 2
    assert "--reference" in result.output


def test_reference_must_match_a_trk_input(tmp_path: Path) -> None:
    save_tractogram(TRACTOGRAM, tmp_path / "in.trk", reference=GRID)
    other = tmp_path / "other.nii"
    save_volume(Volume(np.zeros((5, 5, 5), np.uint8), np.eye(4)), other)
    result = _run(tmp_path / "in.trk", tmp_path / "out.tck", "--reference", other)

    assert result.exit_code == 1
    assert "does not match" in result.output


@pytest.mark.parametrize("option", ["--bvals", "--bvecs", "--bvals-out"])
def test_gradient_options_for_a_tractogram_are_a_usage_error(
    tmp_path: Path, option: str
) -> None:
    result = _run("missing.tck", tmp_path / "out.vtk", option, "x")
    assert result.exit_code == 2


def test_vtk_space_lps_converts_the_points_to_ras(tmp_path: Path) -> None:
    (tmp_path / "in.vtk").write_bytes(
        b"# vtk DataFile Version 3.0\nno tag\nASCII\nDATASET POLYDATA\n"
        b"POINTS 2 float\n1 2 3 4 5 6\nLINES 1 3\n2 0 1\n"
    )
    result = _run(tmp_path / "in.vtk", tmp_path / "out.tck", "--vtk-space", "LPS")
    assert result.exit_code == 0, result.output
    out = load_tractogram(tmp_path / "out.tck")
    np.testing.assert_array_equal(out.points, [[-1, -2, 3], [-4, -5, 6]])


def test_vtk_space_without_a_vtk_input_is_a_usage_error(tmp_path: Path) -> None:
    save_tractogram(TRACTOGRAM, tmp_path / "in.tck")
    result = _run(tmp_path / "in.tck", tmp_path / "out.vtk", "--vtk-space", "LPS")
    assert result.exit_code == 2
    assert "--vtk-space" in result.output


# Both kinds


@pytest.mark.parametrize(("src", "dst"), [("a.nii.gz", "b.tck"), ("a.trk", "b.mif")])
def test_converting_between_kinds_is_one_line_error(
    tmp_path: Path, src: str, dst: str
) -> None:
    result = _run(tmp_path / src, tmp_path / dst)

    assert result.exit_code == 1
    assert result.output.startswith("ERROR: ")
    assert len(result.output.strip().splitlines()) == 1
    assert "same kind" in result.output


def test_unsupported_extension_is_an_error(tmp_path: Path) -> None:
    result = _run(tmp_path / "a.txt", tmp_path / "b.nii")
    assert result.exit_code == 1
    assert "unsupported extension" in result.output


@pytest.mark.parametrize("existing", ["out.mif", "b.bvec"])
def test_existing_outputs_need_force(tmp_path: Path, existing: str) -> None:
    save_volume(Volume(DATA, AFFINE), tmp_path / "in.mif", gradients=TABLE)
    (tmp_path / existing).write_bytes(b"old")
    out, bvals, bvecs = tmp_path / "out.mif", tmp_path / "b.bval", tmp_path / "b.bvec"
    args: list[Arg] = [
        tmp_path / "in.mif",
        out,
        "--bvals-out",
        bvals,
        "--bvecs-out",
        bvecs,
    ]

    refused = _run(*args)
    assert refused.exit_code == 1
    assert "exists" in refused.output
    assert (tmp_path / existing).read_bytes() == b"old"

    forced = _run(*args, "--force")
    assert forced.exit_code == 0, forced.output
    assert (tmp_path / existing).read_bytes() != b"old"
