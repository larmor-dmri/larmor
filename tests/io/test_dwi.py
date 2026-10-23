import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from larmor.io import has_mif_gradients, load_dwi, read_bvals_bvecs, save_volume
from larmor.types import GradientTable, Volume

# An oblique LPS-like affine (det > 0), so the FSL and world frames differ.
AFFINE = np.array(
    [[-2.0, 0.2, 0, 30], [0.1, -2.0, 0, 40], [0, 0, 2.5, -10], [0, 0, 0, 1]]
)
DATA = np.arange(2 * 3 * 4 * 5, dtype=np.int16).reshape(2, 3, 4, 5)
BVALS = np.array([0.0, 1000.0, 1000.0, 2000.0, 2000.0])
BVECS = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 0.6, 0.8], [0.48, 0.6, 0.64], [-0.36, 0.48, 0.8]]
)
HEADER_TABLE = GradientTable(BVALS, BVECS)
# Different numbers from the header's, so the source of a table is visible.
FILE_BVALS = np.array([0.0, 500.0, 500.0, 1500.0, 1500.0])
MIF_NAMES = ["dwi.mif", "dwi.mif.gz", "dwi.mih"]


def _volume(path: Path, gradients: GradientTable | None = None) -> Path:
    save_volume(Volume(DATA, AFFINE), path, gradients=gradients)
    return path


def _files(tmp_path: Path) -> tuple[Path, Path]:
    bvals, bvecs = tmp_path / "bvals", tmp_path / "bvecs"
    np.savetxt(bvals, FILE_BVALS[None])
    np.savetxt(bvecs, BVECS[:, [1, 2, 0]].T)
    return bvals, bvecs


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.parametrize("name", MIF_NAMES)
def test_mif_with_gradients_provides_them(tmp_path: Path, name: str) -> None:
    path = _volume(tmp_path / name, HEADER_TABLE)
    volume, gtab = load_dwi(path)
    np.testing.assert_array_equal(volume.data, DATA)
    np.testing.assert_array_equal(gtab.bvals, BVALS)
    np.testing.assert_allclose(gtab.bvecs, HEADER_TABLE.bvecs, rtol=0, atol=1e-15)


@pytest.mark.parametrize("name", [*MIF_NAMES, "dwi.nii", "dwi.nii.gz"])
def test_without_gradients_and_files_raises(tmp_path: Path, name: str) -> None:
    path = _volume(tmp_path / name)
    with pytest.raises(ValueError, match="no gradient table: give the bvals and bvecs"):
        load_dwi(path)


@pytest.mark.parametrize("name", ["dwi.nii", "dwi.nii.gz", *MIF_NAMES])
@pytest.mark.parametrize("frame", ["fsl", "world"])
def test_files_give_the_table_in_their_frame(
    tmp_path: Path,
    name: str,
    frame: Literal["fsl", "world"],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A NIfTI file, or a MIF without a table: the files, read in the given frame.
    path = _volume(tmp_path / name)
    bvals, bvecs = _files(tmp_path)
    _, gtab = load_dwi(path, bvals=bvals, bvecs=bvecs, frame=frame)
    expected = read_bvals_bvecs(bvals, bvecs, ref_path=path, frame=frame)
    np.testing.assert_array_equal(gtab.bvals, FILE_BVALS)
    np.testing.assert_array_equal(gtab.bvecs, expected.bvecs)
    assert _warnings(caplog) == []


def test_fsl_and_world_frames_differ_for_this_affine(tmp_path: Path) -> None:
    # Guards the test above: with this affine the frame changes the directions.
    path = _volume(tmp_path / "dwi.nii")
    bvals, bvecs = _files(tmp_path)
    fsl = load_dwi(path, bvals=bvals, bvecs=bvecs, frame="fsl")[1].bvecs
    world = load_dwi(path, bvals=bvals, bvecs=bvecs, frame="world")[1].bvecs
    assert not np.allclose(fsl[1:], world[1:])


@pytest.mark.parametrize("name", MIF_NAMES)
def test_files_override_the_header_with_a_warning(
    tmp_path: Path, name: str, caplog: pytest.LogCaptureFixture
) -> None:
    path = _volume(tmp_path / name, HEADER_TABLE)
    bvals, bvecs = _files(tmp_path)
    _, gtab = load_dwi(path, bvals=bvals, bvecs=bvecs)
    np.testing.assert_array_equal(gtab.bvals, FILE_BVALS)
    (warning,) = _warnings(caplog)
    assert name in warning.getMessage()
    assert "override" in warning.getMessage()


@pytest.mark.parametrize("which", ["bvals", "bvecs"])
@pytest.mark.parametrize("name", ["dwi.nii.gz", "dwi.mif"])
def test_only_one_file_raises(tmp_path: Path, which: str, name: str) -> None:
    path = _volume(tmp_path / name, HEADER_TABLE if name == "dwi.mif" else None)
    bvals, bvecs = _files(tmp_path)
    with pytest.raises(ValueError, match="both the bvals and the bvecs"):
        if which == "bvals":
            load_dwi(path, bvals=bvals)
        else:
            load_dwi(path, bvecs=bvecs)


def test_missing_volume_is_reported_before_the_gradients(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing.nii"):
        load_dwi(
            tmp_path / "missing.nii",
            bvals=tmp_path / "no_bvals",
            bvecs=tmp_path / "no_bvecs",
        )


def test_one_file_alone_is_reported_before_a_missing_volume(tmp_path: Path) -> None:
    # The argument mistake comes first, then the volume, then the gradients.
    with pytest.raises(ValueError, match="both the bvals and the bvecs"):
        load_dwi(tmp_path / "missing.nii", bvals=tmp_path / "bvals")


def test_has_mif_gradients_checks_only_the_header(tmp_path: Path) -> None:
    assert has_mif_gradients(_volume(tmp_path / "a.mif", HEADER_TABLE))
    assert not has_mif_gradients(_volume(tmp_path / "b.mif"))
