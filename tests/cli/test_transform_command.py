from pathlib import Path

import numpy as np
from typer.testing import CliRunner, Result

from larmor.io import load_surface, load_tractogram, load_volume, save_surface
from larmor.main import app
from larmor.registration import read_mrtrix_transform
from larmor.types import Surface

DATA = Path(__file__).parents[1] / "data" / "registration"


def _run(*args: str | Path) -> Result:
    return CliRunner().invoke(app, ["transform", *map(str, args)], env={"TERM": "dumb"})


def test_image_through_an_ants_registration(tmp_path: Path) -> None:
    out = tmp_path / "out.nii.gz"
    result = _run(
        DATA / "moving.nii.gz", out, "-t", DATA / "ants_warp.nii.gz",
        "-t", DATA / "ants_affine.mat", "--reference", DATA / "fixed.nii.gz",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    expected = load_volume(DATA / "ants_chain_linear.nii.gz").data
    np.testing.assert_allclose(load_volume(out).data, expected, atol=1e-4)


def test_tractogram_follows_the_image_with_the_inverse(tmp_path: Path) -> None:
    # tcktransform moved the fixed-space tracks into the moving space with the
    # deformation that resamples moving images onto the fixed grid: for lar
    # transform that is going the other way.
    out = tmp_path / "out.tck"
    result = _run(
        DATA / "tracks_fixed.tck", out, "-t", DATA / "mrtrix_warp.mif", "--inverse"
    )
    assert result.exit_code == 0, result.output
    np.testing.assert_allclose(
        load_tractogram(out).points,
        load_tractogram(DATA / "mrtrix_warp_tracks.tck").points,
        atol=1e-5,
    )


def test_tractogram_with_the_tool_inverse_warp(tmp_path: Path) -> None:
    out = tmp_path / "out.trk"
    warps = f"{DATA / 'mrtrix_warp.mif'},{DATA / 'mrtrix_warp_inverse.mif'}"
    result = _run(
        DATA / "mrtrix_warp_tracks.tck", out, "-t", warps,
        "--reference", DATA / "fixed.nii.gz",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    np.testing.assert_allclose(
        load_tractogram(out).points,
        load_tractogram(DATA / "mrtrix_warp_inverse_tracks.tck").points,
        atol=1e-4,
    )


def test_surface_moves_with_the_inverse(tmp_path: Path) -> None:
    vertices = np.array([[0.0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]])
    source = tmp_path / "in.gii"
    save_surface(
        Surface(vertices, np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])),
        source,
    )
    out = tmp_path / "out.gii"
    result = _run(source, out, "-t", DATA / "mrtrix_affine.txt")
    assert result.exit_code == 0, result.output
    expected = (
        read_mrtrix_transform(DATA / "mrtrix_affine.txt").inverse().map_points(vertices)
    )
    np.testing.assert_allclose(load_surface(out).vertices, expected, atol=1e-4)


def test_image_without_reference_is_refused(tmp_path: Path) -> None:
    result = _run(
        DATA / "moving.nii.gz", tmp_path / "o.nii", "-t", DATA / "ants_affine.mat"
    )
    assert result.exit_code != 0
    assert "reference image" in result.output


def test_existing_output_is_kept_without_force(tmp_path: Path) -> None:
    out = tmp_path / "out.tck"
    out.write_text("keep")
    result = _run(DATA / "tracks_fixed.tck", out, "-t", DATA / "mrtrix_affine.txt")
    assert result.exit_code == 1
    assert out.read_text() == "keep"


def test_output_over_the_input_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "tracks.tck"
    path.write_bytes((DATA / "tracks_fixed.tck").read_bytes())
    result = _run(path, path, "-t", DATA / "mrtrix_affine.txt", "--force")
    assert result.exit_code == 1
    assert path.read_bytes() == (DATA / "tracks_fixed.tck").read_bytes()
