from pathlib import Path

import numpy as np
from typer.testing import CliRunner, Result

from larmor.dwi import NoddiModel
from larmor.io import load_volume, read_bvals_bvecs
from larmor.main import app

NODDI_DIR = Path(__file__).parents[1] / "data" / "noddi"
VOLUME, MASK = NODDI_DIR / "dwi.nii.gz", NODDI_DIR / "mask.nii.gz"
GRADIENTS = ("--bvals", str(NODDI_DIR / "bvals"), "--bvecs", str(NODDI_DIR / "bvecs"))
METRICS = ("v_ic", "od", "kappa", "v_iso", "mu")


def _run(*args: str | Path) -> Result:
    return CliRunner().invoke(app, ["noddi", *map(str, args)], env={"TERM": "dumb"})


def test_noddi_writes_the_maps_of_the_python_fit(tmp_path: Path) -> None:
    result = _run(VOLUME, tmp_path / "out", *GRADIENTS, "--mask", MASK)

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        f"out_{name}.nii.gz" for name in METRICS
    )
    gtab = read_bvals_bvecs(NODDI_DIR / "bvals", NODDI_DIR / "bvecs", ref_path=VOLUME)
    expected = NoddiModel().fit(load_volume(VOLUME), gtab, load_volume(MASK))
    for name in METRICS:
        written = load_volume(tmp_path / f"out_{name}.nii.gz").as_float()
        np.testing.assert_array_equal(written, getattr(expected, name), err_msg=name)
