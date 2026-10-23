from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner, Result

from larmor.dwi import remove_gibbs_ringing
from larmor.io import load_volume
from larmor.main import app

PHANTOM = Path(__file__).parents[1] / "data" / "gibbs" / "phantom.nii.gz"


def _run(*args: str | Path) -> Result:
    return CliRunner().invoke(app, ["degibbs", *map(str, args)], env={"TERM": "dumb"})


@pytest.mark.parametrize(
    ("options", "settings"),
    [
        ([], {}),
        (
            ["--slice-axis", "0", "--shifts", "10", "--window", "0", "2", "--nt", "1"],
            {"slice_axis": 0, "num_shifts": 10, "window": (0, 2)},
        ),
    ],
    ids=["default", "options"],
)
def test_output_equals_the_library(
    tmp_path: Path, options: list[str], settings: dict[str, object]
) -> None:
    result = _run(PHANTOM, tmp_path / "out.nii.gz", *options)

    assert result.exit_code == 0, result.output
    expected = remove_gibbs_ringing(load_volume(PHANTOM), **settings)  # type: ignore[arg-type]
    written = load_volume(tmp_path / "out.nii.gz")
    np.testing.assert_array_equal(written.data, expected.data)
    np.testing.assert_array_equal(written.affine, expected.affine)


def test_existing_output_needs_force(tmp_path: Path) -> None:
    output = tmp_path / "out.nii.gz"
    output.write_bytes(b"old")

    refused = _run(PHANTOM, output)
    assert refused.exit_code != 0
    assert output.read_bytes() == b"old"

    forced = _run(PHANTOM, output, "--force")
    assert forced.exit_code == 0, forced.output
    assert load_volume(output).data.shape == (64, 62, 2)
