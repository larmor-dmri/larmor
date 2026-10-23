import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest

from larmor.io import (
    TRACTOGRAM_EXTENSIONS,
    VOLUME_EXTENSIONS,
    file_kind,
    load_grid,
    load_header,
    load_tractogram,
    load_tractogram_subset,
    load_volume,
    output_files,
    save_tractogram,
    save_volume,
)
from larmor.io.formats import read_mif, read_mif_gradients
from larmor.io.reader import GZIP_MAGIC, open_file_read
from larmor.types import GradientTable, Grid, Tractogram, Volume

AFFINE = np.array(
    [[2.0, 0, 0, -10], [0, 3.0, 0, 20], [0, 0, 2.5, 3], [0, 0, 0, 1]], dtype=np.float64
)
VOLUME = Volume(np.arange(2 * 3 * 4 * 5, dtype=np.int16).reshape(2, 3, 4, 5), AFFINE)
# Unit vectors with long decimal expansions, so that any rounding shows.
_UNIT = np.array([[0.0, 0, 0], *[[1 / 3, 2 / 3, 2 / 3]] * 6])


def _is_gzip(path: Path) -> bool:
    return path.read_bytes()[:2] == GZIP_MAGIC


def _save(path: Path) -> None:
    save_volume(VOLUME, path)


FUNCTIONS = pytest.mark.parametrize(
    "call", [load_volume, load_grid, _save], ids=["load_volume", "load_grid", "save"]
)


def _table(n: int) -> GradientTable:
    bvals = np.array([0.0] + [1000.0] * (n - 1))
    bvecs = np.tile([0.0, 0.6, 0.8], (n, 1))
    return GradientTable(bvals, bvecs)


@pytest.mark.parametrize("name", ["a.nii", "a.nii.gz"])
@pytest.mark.parametrize("as_str", [False, True], ids=["path", "str"])
def test_nifti_extensions_save_and_load_as_nifti(
    tmp_path: Path, name: str, as_str: bool
) -> None:
    path: str | Path = str(tmp_path / name) if as_str else tmp_path / name
    save_volume(VOLUME, path)
    img: Any = nib.load(path)
    np.testing.assert_array_equal(img.dataobj.get_unscaled(), VOLUME.data)
    vol = load_volume(path)
    np.testing.assert_array_equal(vol.data, VOLUME.data, strict=True)
    np.testing.assert_array_equal(vol.affine, AFFINE)
    grid = load_grid(path)
    assert grid.shape == (2, 3, 4)
    np.testing.assert_array_equal(grid.affine, AFFINE)


MIF_NAMES = pytest.mark.parametrize("name", ["a.mif", "a.mif.gz", "a.mih"])


def _is_mif(path: Path) -> bool:
    with open_file_read(path) as f:
        return f.readline() == b"mrtrix image\n"


@MIF_NAMES
@pytest.mark.parametrize("as_str", [False, True], ids=["path", "str"])
def test_mif_extensions_save_and_load_as_mif(
    tmp_path: Path, name: str, as_str: bool
) -> None:
    path: str | Path = str(tmp_path / name) if as_str else tmp_path / name
    save_volume(VOLUME, path)
    assert _is_mif(Path(path))
    assert name.endswith(".gz") == _is_gzip(Path(path))
    np.testing.assert_array_equal(read_mif(path).data, VOLUME.data, strict=True)
    vol = load_volume(path)
    np.testing.assert_array_equal(vol.data, VOLUME.data, strict=True)
    np.testing.assert_array_equal(vol.affine, AFFINE)
    grid, shape = load_header(path)
    assert (grid.shape, shape) == ((2, 3, 4), (2, 3, 4, 5))
    np.testing.assert_array_equal(grid.affine, AFFINE)
    np.testing.assert_array_equal(load_grid(path).affine, AFFINE)


def test_mih_saves_a_dat_of_the_same_name(tmp_path: Path) -> None:
    save_volume(VOLUME, tmp_path / "a.mih")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.dat", "a.mih"]
    assert (tmp_path / "a.dat").stat().st_size == VOLUME.data.nbytes


def test_mif_written_by_mrtrix_loads() -> None:
    path = Path(__file__).parents[1] / "data" / "mif" / "split.mih"
    vol = load_volume(path)
    np.testing.assert_array_equal(vol.data, read_mif(path).data, strict=True)
    grid, shape = load_header(path)
    assert shape == vol.data.shape
    np.testing.assert_array_equal(grid.affine, vol.affine)


@MIF_NAMES
def test_mif_save_writes_gradients_into_header(tmp_path: Path, name: str) -> None:
    table = GradientTable(np.array([0.0, 1000, 2000, 1000, 3000]), _UNIT[:5])
    save_volume(VOLUME, tmp_path / name, gradients=table)
    back = read_mif_gradients(tmp_path / name)
    np.testing.assert_array_equal(back.bvals, table.bvals, strict=True)
    np.testing.assert_array_equal(back.bvecs, table.bvecs, strict=True)


@MIF_NAMES
def test_mif_save_without_gradients_writes_no_dw_scheme(
    tmp_path: Path, name: str
) -> None:
    save_volume(VOLUME, tmp_path / name)
    with pytest.raises(ValueError, match="dw_scheme"):
        read_mif_gradients(tmp_path / name)


@FUNCTIONS
@pytest.mark.parametrize("name", ["a.img", "a.nii.bz2", "a_nii"])
def test_unknown_extension_raises_listing_supported_ones(
    tmp_path: Path, call: Callable[[Path], object], name: str
) -> None:
    with pytest.raises(ValueError) as err:
        call(tmp_path / name)
    message = str(err.value)
    assert name in message
    for extension in (".nii", ".nii.gz", ".mif", ".mif.gz", ".mih"):
        assert re.search(rf"{re.escape(extension)}\b", message), message
    assert not (tmp_path / name).exists()


@pytest.mark.parametrize("name", ["a.nii", "a.nii.gz"])
def test_gradients_are_refused_for_nifti(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="bvals"):
        save_volume(VOLUME, tmp_path / name, gradients=_table(5))
    assert not (tmp_path / name).exists()


@pytest.mark.parametrize(("shape", "n"), [((2, 3, 4, 5), 4), ((2, 3, 4, 5), 6)])
def test_gradients_must_have_one_entry_per_volume(
    tmp_path: Path, shape: tuple[int, ...], n: int
) -> None:
    vol = Volume(np.zeros(shape, dtype=np.int16), AFFINE)
    with pytest.raises(ValueError) as err:
        save_volume(vol, tmp_path / "a.mif", gradients=_table(n))
    # Both counts, outside the path, which may hold digits of its own.
    message = str(err.value).replace(str(tmp_path), "")
    assert re.search(rf"\b{n}\b", message), message
    assert re.search(rf"\b{shape[3]}\b", message), message
    assert not (tmp_path / "a.mif").exists()


@pytest.mark.parametrize(("name", "says"), [("a.mif", "4D volume"), ("a.nii", "bvals")])
def test_gradients_are_refused_for_a_3d_volume(
    tmp_path: Path, name: str, says: str
) -> None:
    # Even a one-entry table: a gradient table needs a fourth dimension.
    vol = Volume(np.zeros((2, 3, 4), dtype=np.int16), AFFINE)
    with pytest.raises(ValueError, match=says):
        save_volume(vol, tmp_path / name, gradients=_table(1))
    assert not (tmp_path / name).exists()


@pytest.mark.parametrize(
    ("name", "files"),
    [
        ("a.nii", ["a.nii"]),
        ("a.nii.gz", ["a.nii.gz"]),
        ("a.mif", ["a.mif"]),
        ("a.mif.gz", ["a.mif.gz"]),
        ("a.mih", ["a.mih", "a.dat"]),
    ],
)
def test_output_files_lists_every_file_a_save_writes(
    tmp_path: Path, name: str, files: list[str]
) -> None:
    assert output_files(str(tmp_path / name)) == [tmp_path / f for f in files]
    save_volume(VOLUME, tmp_path / name)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(files)


def test_output_files_refuses_an_unknown_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported extension"):
        output_files(tmp_path / "a.txt")


# Tractograms

TRACTOGRAM = Tractogram(
    np.arange(15, dtype=np.float32).reshape(5, 3) / 4, np.array([0, 2, 2, 5])
)
GRID = Grid((10, 12, 14), AFFINE)
# Each extension and the first bytes the matching writer puts in the file.
TRACTOGRAM_FILES = {
    "a.tck": b"mrtrix tracks",
    "a.tck.gz": GZIP_MAGIC,
    "a.trk": b"TRACK",
    "a.trk.gz": GZIP_MAGIC,
    "a.vtk": b"# vtk DataFile",
    "a.vtk.gz": GZIP_MAGIC,
    "a.fib": b"# vtk DataFile",
    "a.fib.gz": GZIP_MAGIC,
}


@pytest.mark.parametrize(("name", "start"), TRACTOGRAM_FILES.items())
def test_tractogram_extensions_save_and_load_their_format(
    tmp_path: Path, name: str, start: bytes
) -> None:
    path = tmp_path / name
    save_tractogram(TRACTOGRAM, path, reference=GRID)
    assert path.read_bytes().startswith(start)
    with open_file_read(path) as f:
        assert (
            f.read(len(TRACTOGRAM_FILES[name.removesuffix(".gz")]))
            == (TRACTOGRAM_FILES[name.removesuffix(".gz")])
        )
    t = load_tractogram(path)
    np.testing.assert_allclose(t.points, TRACTOGRAM.points, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(t.offsets, TRACTOGRAM.offsets)


@pytest.mark.parametrize("name", ["a.tck", "a.vtk", "a.fib"])
def test_reference_is_attached_to_formats_without_a_grid(
    tmp_path: Path, name: str
) -> None:
    save_tractogram(TRACTOGRAM, tmp_path / name)
    assert load_tractogram(tmp_path / name).reference is None
    assert load_tractogram(tmp_path / name, reference=GRID).reference is GRID


@pytest.mark.parametrize("name", ["a.vtk", "a.fib"])
def test_space_option_reaches_the_vtk_reader(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_bytes(
        b"# vtk DataFile Version 3.0\nno tag\nASCII\nDATASET POLYDATA\n"
        b"POINTS 2 float\n1 2 3 4 5 6\nLINES 1 3\n2 0 1\n"
    )
    t = load_tractogram(path, space="LPS")
    np.testing.assert_array_equal(t.points, [[-1, -2, 3], [-4, -5, 6]])


def test_space_option_is_refused_for_formats_that_have_a_space(
    tmp_path: Path,
) -> None:
    save_tractogram(TRACTOGRAM, tmp_path / "a.tck")
    with pytest.raises(ValueError, match="VTK"):
        load_tractogram(tmp_path / "a.tck", space="LPS")


def test_tck_loaded_with_a_reference_saves_as_trk(tmp_path: Path) -> None:
    save_tractogram(TRACTOGRAM, tmp_path / "a.tck")
    with pytest.raises(ValueError, match="reference"):
        save_tractogram(load_tractogram(tmp_path / "a.tck"), tmp_path / "a.trk")

    save_tractogram(
        load_tractogram(tmp_path / "a.tck", reference=GRID), tmp_path / "a.trk"
    )
    t = load_tractogram(tmp_path / "a.trk")
    assert t.reference is not None
    assert t.reference.matches(GRID)
    np.testing.assert_allclose(t.points, TRACTOGRAM.points, rtol=0, atol=1e-5)


def test_trk_reference_must_match_the_file(tmp_path: Path) -> None:
    save_tractogram(TRACTOGRAM, tmp_path / "a.trk", reference=GRID)
    other = Grid((10, 12, 15), AFFINE)
    with pytest.raises(ValueError, match="does not match"):
        load_tractogram(tmp_path / "a.trk", reference=other)


TRACTOGRAM_CALLS = pytest.mark.parametrize(
    "call",
    [load_tractogram, lambda p: save_tractogram(TRACTOGRAM, p, reference=GRID)],
    ids=["load_tractogram", "save_tractogram"],
)


@TRACTOGRAM_CALLS
@pytest.mark.parametrize("name", ["a.trx", "a.tck.bz2", "a_tck"])
def test_unknown_tractogram_extension_raises_listing_supported_ones(
    tmp_path: Path, call: Callable[[Path], object], name: str
) -> None:
    with pytest.raises(ValueError) as err:
        call(tmp_path / name)
    message = str(err.value)
    assert name in message
    for extension in TRACTOGRAM_FILES:
        assert extension.removeprefix("a") in message
    assert not (tmp_path / name).exists()


@TRACTOGRAM_CALLS
@pytest.mark.parametrize("name", ["a.nii.gz", "a.mif", "a.mih"])
def test_volume_given_to_tractogram_functions_is_named(
    tmp_path: Path, call: Callable[[Path], object], name: str
) -> None:
    with pytest.raises(ValueError, match="is a volume, not a tractogram"):
        call(tmp_path / name)


@FUNCTIONS
@pytest.mark.parametrize("name", ["a.tck", "a.trk.gz", "a.fib"])
def test_tractogram_given_to_volume_functions_is_named(
    tmp_path: Path, call: Callable[[Path], object], name: str
) -> None:
    with pytest.raises(ValueError, match="is a tractogram, not a volume"):
        call(tmp_path / name)


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        *((f"a{e}", "volume") for exts in VOLUME_EXTENSIONS.values() for e in exts),
        *(
            (f"a{e}", "tractogram")
            for e in (n.removeprefix("a") for n in TRACTOGRAM_FILES)
        ),
    ],
)
def test_file_kind_follows_the_extension(name: str, kind: str) -> None:
    assert file_kind(name) == kind
    assert file_kind(Path("/some/dir") / name) == kind


@pytest.mark.parametrize("name", ["a.txt", "a.nii.bz2", "a_nii", "a"])
def test_file_kind_of_an_unknown_extension_lists_both_kinds(name: str) -> None:
    with pytest.raises(ValueError) as err:
        file_kind(name)
    message = str(err.value)
    assert name in message
    for exts in (*VOLUME_EXTENSIONS.values(), *TRACTOGRAM_EXTENSIONS.values()):
        for extension in exts:
            assert extension in message


EDITING_TRACKS = Path(__file__).parents[1] / "data" / "editing" / "tracks.tck"


def test_load_tractogram_subset_keeps_what_load_tractogram_keeps() -> None:
    limited = load_tractogram(EDITING_TRACKS, max_streamlines=100, seed=3)
    subset, indices, total = load_tractogram_subset(EDITING_TRACKS, 100, seed=3)
    whole = load_tractogram(EDITING_TRACKS)
    assert total == whole.num_streamlines == 500
    assert np.all(np.diff(indices) > 0)
    for kept in (subset, whole[indices]):
        assert np.array_equal(kept.points, limited.points)
        assert np.array_equal(kept.offsets, limited.offsets)


def test_load_tractogram_subset_of_a_file_within_the_limit_keeps_all() -> None:
    subset, indices, total = load_tractogram_subset(EDITING_TRACKS, 500)
    assert total == 500 and np.array_equal(indices, np.arange(500))
    assert subset.num_streamlines == 500
