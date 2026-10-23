from pathlib import Path

import numpy as np
import pytest

from larmor.io import (
    read_bvals_bvecs,
    read_mrtrix_gradients,
    write_mrtrix_gradients,
)
from larmor.types import GradientTable

MIF = Path(__file__).parents[1] / "data" / "mif"

# MRtrix writes 10 significant digits and maps the FSL table through the float32
# header affine, so its world vectors differ from Larmor's by about 1e-9.
VEC_ATOL = 1e-8
B_RTOL = 1e-8


def test_mrtrix_file_matches_the_fsl_table_it_was_exported_from() -> None:
    got = read_mrtrix_gradients(MIF / "dwi.b")
    want = read_bvals_bvecs(
        MIF / "dwi.bval", MIF / "dwi.bvec", ref_path=MIF / "source.nii"
    )
    np.testing.assert_allclose(got.bvecs, want.bvecs, rtol=0, atol=VEC_ATOL)
    np.testing.assert_allclose(got.bvals, want.bvals, rtol=B_RTOL)


def test_hand_written_file_reads_as_mrtrix_reads_it() -> None:
    # Comments, commas, semicolons, tabs, a b = 0 row, b = 5 with a direction and
    # a half-length direction at b = 2800, which MRtrix exports as unit at b = 700.
    got = read_mrtrix_gradients(MIF / "grad_odd.b")
    want = np.loadtxt(MIF / "grad_odd_export.b")
    np.testing.assert_allclose(got.bvecs, want[:, :3], rtol=0, atol=1e-10)
    np.testing.assert_allclose(got.bvals, want[:, 3], rtol=1e-12)


def test_written_file_is_the_one_mrtrix_exports_back_to_the_fsl_table(
    tmp_path: Path,
) -> None:
    gtab = read_bvals_bvecs(
        MIF / "dwi.bval", MIF / "dwi.bvec", ref_path=MIF / "source.nii"
    )
    path = tmp_path / "dwi.b"
    write_mrtrix_gradients(gtab, path)
    np.testing.assert_allclose(
        np.loadtxt(path), np.loadtxt(MIF / "larmor_dwi.b"), rtol=0, atol=1e-15
    )
    # larmor_dwi.bvec/.bval are MRtrix's FSL export of that file.
    mrtrix = read_bvals_bvecs(
        MIF / "larmor_dwi.bval", MIF / "larmor_dwi.bvec", ref_path=MIF / "source.nii"
    )
    np.testing.assert_allclose(mrtrix.bvecs, gtab.bvecs, rtol=0, atol=VEC_ATOL)
    np.testing.assert_allclose(mrtrix.bvals, gtab.bvals, rtol=B_RTOL)


def test_round_trip_is_exact(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    bvecs = rng.normal(size=(7, 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    bvecs[0] = 0.0
    bvals = np.array([0.0, 1000.0, 1000.0, 2000.0, 2000.0, 3000.0, 3000.0])
    gtab = GradientTable(bvals, bvecs)
    path = tmp_path / "g.b"
    write_mrtrix_gradients(gtab, str(path))
    got = read_mrtrix_gradients(str(path))
    np.testing.assert_array_equal(got.bvecs, gtab.bvecs)
    np.testing.assert_array_equal(got.bvals, gtab.bvals)


@pytest.mark.parametrize(
    "text",
    [
        "0 0 0 0\n1 0 0\n",
        "0 0 0 0\n1 0 0 1000 5\n",
        "0 0 0 0\n1 0 x 1000\n",
        "0 0 0 0 0 0 0 0\n",
        "# only a comment\n\n",
        "",
    ],
    ids=["three", "five", "text", "one_row_of_eight", "comments_only", "empty"],
)
def test_table_that_is_not_n_by_4_raises(tmp_path: Path, text: str) -> None:
    path = tmp_path / "g.b"
    path.write_text(text)
    with pytest.raises(ValueError, match="g.b"):
        read_mrtrix_gradients(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        read_mrtrix_gradients(tmp_path / "missing.b")


@pytest.mark.parametrize("n_rows", [4, 6])
@pytest.mark.parametrize("write", [False, True])
def test_table_not_matching_the_image_raises(
    tmp_path: Path, n_rows: int, write: bool
) -> None:
    # source.nii has 5 volumes.
    gtab = GradientTable(np.full(n_rows, 1000.0), np.tile([0.0, 0.0, 1.0], (n_rows, 1)))
    path = tmp_path / "g.b"
    if write:
        with pytest.raises(ValueError):
            write_mrtrix_gradients(gtab, path, ref_path=MIF / "source.nii")
        assert not path.exists()
    else:
        write_mrtrix_gradients(gtab, path)
        with pytest.raises(ValueError):
            read_mrtrix_gradients(path, ref_path=MIF / "source.nii")


def test_table_matching_the_image_is_read_and_written(tmp_path: Path) -> None:
    gtab = read_mrtrix_gradients(MIF / "dwi.b", ref_path=MIF / "source.nii")
    write_mrtrix_gradients(gtab, tmp_path / "g.b", ref_path=MIF / "source.nii")
    assert len(np.loadtxt(tmp_path / "g.b")) == 5


def test_non_utf8_byte_in_a_comment_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "g.b"
    path.write_bytes(
        "# r\xe9sum\xe9 from the scanner\n0 0 0 0\n1 0 0 1000\n".encode("latin-1")
    )
    got = read_mrtrix_gradients(path)
    np.testing.assert_array_equal(got.bvals, [0.0, 1000.0])
