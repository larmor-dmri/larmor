from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner, Result

from larmor.io import (
    load_tractogram,
    load_volume,
    read_bvals_bvecs,
    read_mritoolkit_fit,
    read_mritoolkit_tracts,
)
from larmor.main import app
from larmor.sd import read_mritoolkit_fod

DATA = Path(__file__).parents[1] / "data" / "mritoolkit"
FIT = DATA / "dki_fit.mat"


def _run(*args: str | Path) -> Result:
    return CliRunner().invoke(app, ["convert", *map(str, args)], env={"TERM": "dumb"})


def test_a_fit_becomes_its_dwi_with_gradients_and_its_maps(tmp_path: Path) -> None:
    out = tmp_path / "fit.nii.gz"
    bvals, bvecs = tmp_path / "fit.bval", tmp_path / "fit.bvec"
    result = _run(FIT, out, "--bvals-out", bvals, "--bvecs-out", bvecs)
    assert result.exit_code == 0, result.output
    fit = read_mritoolkit_fit(FIT)
    np.testing.assert_array_equal(load_volume(out).data, fit.dwi.data)
    table = read_bvals_bvecs(bvals, bvecs, ref_path=out)
    np.testing.assert_allclose(table.bvals, fit.gradients.bvals, atol=1e-3)
    assert fit.kt is not None
    expected = {
        "fa": fit.maps["FA"],
        "md": fit.maps["MD"],
        "ad": fit.maps["AD"],
        "rd": fit.maps["RD"],
        "pe": fit.maps["FE"],
        "dt": fit.dt,
        "kt": fit.kt,
    }
    for name, volume in expected.items():
        written = load_volume(tmp_path / f"fit_{name}.nii.gz")
        np.testing.assert_array_equal(written.data, volume.data, err_msg=name)


def test_a_grl_result_becomes_an_fod_and_its_fractions(tmp_path: Path) -> None:
    out = tmp_path / "fod.nii.gz"
    assert _run(DATA / "grl.mat", out, "--reference", FIT).exit_code == 0
    expected = read_mritoolkit_fod(DATA / "grl.mat", FIT)
    np.testing.assert_array_equal(load_volume(out).data, expected.fods[0].data)
    fractions = load_volume(tmp_path / "fod_tissue_fraction.nii.gz")
    np.testing.assert_array_equal(fractions.data, expected.tissue_fractions.data)
    assert (tmp_path / "fod_rss.nii.gz").exists()


@pytest.mark.parametrize("extension", [".tck", ".trk"])
def test_tracts_become_a_tractogram_in_world_space(
    tmp_path: Path, extension: str
) -> None:
    out = tmp_path / f"tracts{extension}"
    assert _run(DATA / "tracts.mat", out, "--reference", FIT).exit_code == 0
    expected = read_mritoolkit_tracts(DATA / "tracts.mat", FIT)
    np.testing.assert_allclose(load_tractogram(out).points, expected.points, atol=1e-5)


@pytest.mark.parametrize("name", ["tracts.mat", "grl.mat"])
def test_files_without_a_grid_need_a_reference(tmp_path: Path, name: str) -> None:
    out = tmp_path / ("out.tck" if name == "tracts.mat" else "out.nii.gz")
    result = _run(DATA / name, out)
    assert result.exit_code == 2 and "--reference" in result.output
    assert not out.exists()


@pytest.mark.parametrize(
    ("source", "output", "message"),
    [
        (FIT, "out.tck", "is an MRIToolkit tensor fit; convert it to a volume"),
        (DATA / "tracts.mat", "out.nii.gz", "tractography; convert it to a tractogram"),
        (DATA / "grl.mat", "out.trk", "FOD fit; convert it to a volume"),
        (
            Path(__file__).parents[1] / "data" / "matlab" / "values_v7.mat",
            "out.nii.gz",
            "holds no MRIToolkit fit, FOD fit or tractography",
        ),
    ],
)
def test_a_result_of_another_kind_is_refused_in_one_line(
    tmp_path: Path, source: Path, output: str, message: str
) -> None:
    result = _run(source, tmp_path / output, "--reference", FIT)
    assert result.exit_code == 1
    assert result.output.startswith("ERROR: ") and message in result.output
    assert len(result.output.strip().splitlines()) == 1
    assert not (tmp_path / output).exists()


def test_image_options_do_not_apply_to_a_mat_input(tmp_path: Path) -> None:
    result = _run(FIT, tmp_path / "x.nii.gz", "--to-ras")
    assert result.exit_code == 2 and "--to-ras" in result.output
