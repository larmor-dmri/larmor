from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner, Result

from larmor.io import load_volume
from larmor.main import app

DATA = Path(__file__).parents[1] / "data" / "connectome"


pytestmark = pytest.mark.usefixtures("keep_log_level")


def _run(*args: str | Path) -> Result:
    result = CliRunner().invoke(app, list(map(str, args)), env={"TERM": "dumb"})
    assert result.exit_code == 0, result.output
    return result


def test_tract_density_maps_fine_weighted_and_lazy(
    tmp_path: Path, connectome_ref: dict[str, np.ndarray]
) -> None:
    tracks, template = DATA / "tracks.tck", DATA / "template.nii.gz"
    weighted, plain, lazy, coarse = (
        tmp_path / f"{name}.nii.gz" for name in ("weighted", "plain", "lazy", "coarse")
    )
    weights = ("--weights-in", DATA / "weights.txt")
    _run("tract-density", tracks, template, weighted, "--voxel-size", "0.7", *weights)
    _run("tract-density", tracks, template, plain, "--voxel-size", "0.7,0.7,0.7")
    _run("tract-density", tracks, template, lazy, "--voxel-size", "0.7", "--lazy")
    _run("tract-density", tracks, template, coarse, "--nt", "2")

    fine = load_volume(plain)
    assert fine.data.shape == tuple(connectome_ref["tdi_fine_shape"])
    np.testing.assert_allclose(
        fine.affine, connectome_ref["tdi_fine_affine"], atol=1e-5
    )
    np.testing.assert_array_equal(load_volume(lazy).data, fine.data)
    # The weighted map holds weights where the plain map holds counts.
    sums = load_volume(weighted).data
    assert sums.dtype == np.float32
    np.testing.assert_array_equal(sums > 0, fine.data > 0)
    # On the template's own grid the map is tckmap's.
    expected = np.zeros(int(np.prod(connectome_ref["tdi_default_shape"])))
    expected[connectome_ref["tdi_default_index"]] = connectome_ref["tdi_default_value"]
    density = load_volume(coarse)
    density.to_closest_canonical()
    np.testing.assert_array_equal(density.data.ravel(), expected)


def test_tract_density_refuses_an_existing_output(tmp_path: Path) -> None:
    out = tmp_path / "tdi.nii"
    out.write_bytes(b"")
    result = CliRunner().invoke(
        app,
        [
            "tract-density",
            str(DATA / "tracks.tck"),
            str(DATA / "template.nii.gz"),
            str(out),
        ],
    )
    assert result.exit_code == 1
    assert out.read_bytes() == b""


def test_connectome_writes_named_mean_lengths_and_assignments(
    tmp_path: Path, connectome_ref: dict[str, np.ndarray]
) -> None:
    out = tmp_path / "lengths.csv"
    assignments = tmp_path / "nodes.txt"
    result = _run(
        *("connectome", DATA / "tracks.tck", DATA / "parcellation.nii.gz", out),
        *("--lut", DATA / "lut.txt", "--weights-in", DATA / "weights.txt"),
        *("--mean-length", "--assignments-out", assignments),
    )
    assert "of 200 streamlines to 60 regions" in result.output
    header = out.read_text().splitlines()[0].split(",")
    assert header[:3] == ["", "Region_01", "Region_02"] and len(header) == 61
    matrix = np.loadtxt(out, delimiter=",", skiprows=1, usecols=range(1, 61))
    np.testing.assert_allclose(matrix, connectome_ref["matrix_mean_length"], rtol=1e-6)
    np.testing.assert_array_equal(
        np.loadtxt(assignments, dtype=int), connectome_ref["nodes_default"]
    )

    plain = tmp_path / "plain.txt"
    _run(
        *("connectome", DATA / "tracks.tck", DATA / "parcellation.nii.gz", plain),
        *("--assignment", "end-voxel", "--symmetric", "--zero-diagonal"),
    )
    expected = connectome_ref["matrix_end_voxels"]
    expected = np.triu(expected, 1) + np.triu(expected, 1).T
    np.testing.assert_array_equal(np.loadtxt(plain), expected)


def _fails(*args: str | Path) -> Result:
    result = CliRunner().invoke(app, list(map(str, args)), env={"TERM": "dumb"})
    assert result.exit_code != 0
    return result


def test_connectome_refuses_existing_shared_outputs_and_a_lone_radius(
    tmp_path: Path,
) -> None:
    inputs = (DATA / "tracks.tck", DATA / "parcellation.nii.gz")
    same = tmp_path / "same.csv"
    result = _fails("connectome", *inputs, same, "--assignments-out", same)
    assert "same file" in result.output
    assert not same.exists()

    existing = tmp_path / "existing.csv"
    existing.write_text("keep\n")
    _fails("connectome", *inputs, existing)
    assert existing.read_text() == "keep\n"

    radius = tmp_path / "radius.csv"
    result = _fails(
        "connectome", *inputs, radius, "--assignment", "end-voxel", "--radius", "2"
    )
    assert "--radius" in result.output
    assert not radius.exists()
