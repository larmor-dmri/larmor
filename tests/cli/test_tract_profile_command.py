from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner, Result

from larmor.io import load_tractogram, read_tsf, save_tractogram, save_volume
from larmor.main import app
from larmor.types import Tractogram, Volume

pytestmark = pytest.mark.usefixtures("keep_log_level")


def _run(*args: str | Path) -> Result:
    return CliRunner().invoke(
        app, ["tract-profile", *map(str, args)], env={"TERM": "dumb"}
    )


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    # Two lines along x from 2 to 8 mm, the second stored backwards, over maps of
    # x and of 2x on a 1 mm grid.
    xs = np.linspace(2.0, 8.0, 7, dtype=np.float32)
    line = np.stack([xs, np.full(7, 2.0), np.full(7, 2.0)], 1)
    bundle = Tractogram(np.concatenate([line, line[::-1]]), np.array([0, 7, 14]))
    save_tractogram(bundle, tmp_path / "bundle.tck")
    x = np.broadcast_to(np.arange(10, dtype=np.float32)[:, None, None], (10, 5, 5))
    save_volume(Volume(x.copy(), np.eye(4)), tmp_path / "x.nii.gz")
    save_volume(Volume(2 * x, np.eye(4)), tmp_path / "double.mif")
    return tmp_path / "bundle.tck", tmp_path / "x.nii.gz", tmp_path / "double.mif"


def test_the_table_and_the_tsf_files_hold_the_profile(
    inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    out = tmp_path / "profile.csv"
    result = _run(*inputs, out, "--num-points", "3", "--tsf-prefix", tmp_path / "cst")
    assert result.exit_code == 0, result.output
    rows = [r.split(",") for r in out.read_text().splitlines()]
    assert rows[0] == ["point", "core_x", "core_y", "core_z", "x", "double"]
    np.testing.assert_allclose([float(r[1]) for r in rows[1:]], [2, 5, 8], atol=1e-4)
    # The nodes at x = 2, 5 and 8 hold the points from 2 to 3.5, 3.5 to 6.5 and
    # 6.5 to 8: the end ones lie inward.
    x = np.array([float(r[4]) for r in rows[1:]])
    assert 2 < x[0] < 3.5 and 6.5 < x[2] < 8
    assert x[1] == pytest.approx(5.0, abs=1e-4)
    np.testing.assert_allclose([float(r[5]) for r in rows[1:]], 2 * x, 1e-5)
    bundle = load_tractogram(inputs[0])
    segments, _ = read_tsf(tmp_path / "cst_segment.tsf", tractogram=bundle)
    np.testing.assert_array_equal(segments[:7], [0, 0, 1, 1, 1, 2, 2])
    np.testing.assert_array_equal(segments[7:], segments[:7][::-1])
    values, _ = read_tsf(tmp_path / "cst_x.tsf", tractogram=bundle)
    np.testing.assert_allclose(values[:7], x[segments[:7].astype(int)], rtol=1e-5)


def test_maps_with_the_same_name_are_refused(
    inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    other = tmp_path / "x.mif"
    save_volume(Volume(np.zeros((2, 2, 2), np.float32), np.eye(4)), other)
    result = _run(inputs[0], inputs[1], other, tmp_path / "profile.csv")
    assert result.exit_code != 0
    assert "names must differ" in result.output


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("no csv", "must be a .csv"),
        ("map named segment", "must differ"),
        ("output is an input", "also an input"),
    ],
)
def test_clashing_or_wrong_outputs_are_refused(
    inputs: tuple[Path, Path, Path], tmp_path: Path, case: str, message: str
) -> None:
    bundle, x_map, _ = inputs
    if case == "no csv":
        args: list[str | Path] = [bundle, x_map, tmp_path / "md.nii.gz"]
    elif case == "map named segment":
        segment = tmp_path / "segment.nii.gz"
        save_volume(Volume(np.zeros((2, 2, 2), np.float32), np.eye(4)), segment)
        args = [bundle, segment, tmp_path / "p.csv", "--tsf-prefix", tmp_path / "p"]
    else:
        args = [bundle, tmp_path / "p.csv", tmp_path / "p.csv"]
    result = _run(*args, "--force")
    assert result.exit_code != 0
    assert message in result.output
