import gzip
from pathlib import Path

import numpy as np
import pytest

from larmor.io import (
    Progress,
    ReadCancelledError,
    load_tractogram,
    load_volume,
    reader,
    save_tractogram,
    save_volume,
)
from larmor.io.formats import write_vtk
from larmor.io.reader import open_file_read, read_all
from larmor.types import Grid, Tractogram, Volume

AFFINE = np.array(
    [[2.0, 0, 0, -10], [0, 2.5, 0, 20], [0, 0, 3.0, 3], [0, 0, 0, 1]], dtype=np.float64
)
GRID = Grid((10, 12, 14), AFFINE)
# Random values, so that gzip cannot shrink the files below a few chunks.
VOLUME = Volume(
    np.random.default_rng(0).standard_normal((4, 5, 6, 2)).astype(np.float32), AFFINE
)
LENGTHS = [3, 1, 5, 2, 4, 6, 2, 3, 1, 4]
# Small enough that every test file takes several reads.
CHUNK = 64

VOLUME_NAMES = ["a.nii", "a.nii.gz", "a.mif", "a.mif.gz"]
# "ascii" in a name writes the VTK as ASCII; save_tractogram writes binary.
TRACTOGRAM_NAMES = [
    "a.tck",
    "a.tck.gz",
    "a.trk",
    "a.trk.gz",
    "a.vtk",
    "a.vtk.gz",
    "ascii.vtk",
    "ascii.vtk.gz",
]
# Formats that store point and streamline data.
DATA_NAMES = ["a.trk", "a.trk.gz", "a.vtk", "ascii.vtk.gz"]


def _tractogram(data: bool) -> Tractogram:
    rng = np.random.default_rng(1)
    n_points, n_lines = sum(LENGTHS), len(LENGTHS)
    voxels = rng.uniform(0, 1, (n_points, 3)) * (np.array(GRID.shape) - 1)
    points = (voxels @ AFFINE[:3, :3].T + AFFINE[:3, 3]).astype(np.float32)
    offsets = np.concatenate([[0], np.cumsum(LENGTHS, dtype=np.int64)])
    if not data:
        return Tractogram(points, offsets)
    return Tractogram(
        points,
        offsets,
        point_data={
            "fa": rng.uniform(0, 1, n_points).astype(np.float32),
            "dir": rng.uniform(-1, 1, (n_points, 3)).astype(np.float32),
        },
        streamline_data={
            "length": rng.uniform(10, 50, n_lines).astype(np.float32),
            "rgb": rng.uniform(0, 1, (n_lines, 3)).astype(np.float32),
        },
    )


def _save_tractogram(path: Path, data: bool = False) -> Path:
    t = _tractogram(data)
    if path.name.startswith("ascii"):
        write_vtk(t, path, binary=False)
    else:
        save_tractogram(t, path, reference=GRID)
    return path


def _save_volume(path: Path) -> Path:
    save_volume(VOLUME, path)
    return path


def _load(path: Path, progress: Progress | None) -> Volume | Tractogram:
    if path.name.endswith((".nii", ".nii.gz", ".mif", ".mif.gz")):
        return load_volume(path, on_progress=progress)
    return load_tractogram(path, on_progress=progress)


def _save(path: Path) -> Path:
    if path.name in VOLUME_NAMES:
        return _save_volume(path)
    return _save_tractogram(path)


def _assert_same(got: Tractogram, want: Tractogram) -> None:
    np.testing.assert_array_equal(got.offsets, want.offsets, strict=True)
    np.testing.assert_array_equal(got.points, want.points, strict=True)
    for got_data, want_data in (
        (got.point_data, want.point_data),
        (got.streamline_data, want.streamline_data),
    ):
        assert got_data.keys() == want_data.keys()
        for name, values in want_data.items():
            np.testing.assert_array_equal(got_data[name], values, strict=True)


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def __call__(self, done: int, total: int) -> None:
        self.calls.append((done, total))


def _cancel(done: int, total: int) -> None:
    raise ReadCancelledError


@pytest.fixture
def small_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reader, "_CHUNK", CHUNK)


def _assert_progress(calls: list[tuple[int, int]], path: Path) -> None:
    size = path.stat().st_size
    assert len(calls) > 1
    done = [d for d, _ in calls]
    assert done == sorted(done)
    assert {t for _, t in calls} == {size}
    assert done[-1] == size


# Progress


@pytest.mark.usefixtures("small_chunks")
@pytest.mark.parametrize("name", VOLUME_NAMES + TRACTOGRAM_NAMES)
def test_every_format_reports_progress_up_to_the_file_size(
    tmp_path: Path, name: str
) -> None:
    path = _save(tmp_path / name)
    assert path.stat().st_size > 2 * CHUNK
    progress = _Recorder()
    _load(path, progress)
    _assert_progress(progress.calls, path)


@pytest.mark.usefixtures("small_chunks")
@pytest.mark.parametrize("name", [n for n in TRACTOGRAM_NAMES if n.endswith(".gz")])
def test_a_gzip_file_read_to_its_end_is_decompressed_once(
    tmp_path: Path, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Seeking to the end of a gzip stream decompresses all of it without reporting
    # progress, so a read that does it shows nothing for half of the load.
    path = _save(tmp_path / name)
    seek = gzip.GzipFile.seek

    def no_seek_to_end(self: gzip.GzipFile, offset: int, whence: int = 0) -> int:
        assert whence != 2, "the read seeked to the end of the gzip stream"
        return seek(self, offset, whence)

    monkeypatch.setattr(gzip.GzipFile, "seek", no_seek_to_end)
    progress = _Recorder()
    _load(path, progress)
    _assert_progress(progress.calls, path)


@pytest.mark.parametrize("name", VOLUME_NAMES + TRACTOGRAM_NAMES)
def test_cancelling_in_progress_stops_the_read(tmp_path: Path, name: str) -> None:
    path = _save(tmp_path / name)
    with pytest.raises(ReadCancelledError):
        _load(path, _cancel)


@pytest.mark.usefixtures("small_chunks")
@pytest.mark.parametrize("name", VOLUME_NAMES)
def test_volume_loaded_with_progress_equals_one_loaded_without(
    tmp_path: Path, name: str
) -> None:
    path = _save_volume(tmp_path / name)
    with_progress = load_volume(path, on_progress=_Recorder())
    without = load_volume(path)
    np.testing.assert_array_equal(with_progress.data, without.data, strict=True)
    np.testing.assert_array_equal(with_progress.data, VOLUME.data, strict=True)
    np.testing.assert_array_equal(with_progress.affine, without.affine)


@pytest.mark.usefixtures("small_chunks")
@pytest.mark.parametrize("name", DATA_NAMES)
def test_tractogram_loaded_with_progress_equals_one_loaded_without(
    tmp_path: Path, name: str
) -> None:
    path = _save_tractogram(tmp_path / name, data=True)
    _assert_same(load_tractogram(path, on_progress=_Recorder()), load_tractogram(path))


# read_all


@pytest.mark.usefixtures("small_chunks")
@pytest.mark.parametrize("gz", [False, True], ids=["plain", "gzip"])
def test_read_all_returns_every_byte_and_reports_progress(
    tmp_path: Path, gz: bool
) -> None:
    # Few distinct bytes: compressible, so the compressed and uncompressed positions
    # differ, but not so much that the gzip file fits in one chunk.
    data = np.random.default_rng(2).integers(0, 4, 40 * CHUNK + 7, np.uint8).tobytes()
    path = tmp_path / ("a.bin.gz" if gz else "a.bin")
    path.write_bytes(gzip.compress(data) if gz else data)
    progress = _Recorder()
    with open_file_read(path) as f:
        assert read_all(f, progress) == data
    _assert_progress(progress.calls, path)


def test_read_all_without_progress_returns_every_byte(tmp_path: Path) -> None:
    data = b"abc" * 100
    path = tmp_path / "a.bin.gz"
    path.write_bytes(gzip.compress(data))
    with open_file_read(path) as f:
        assert read_all(f) == data


# Subsets

SUBSET_NAMES = ["a.tck", "a.tck.gz", "a.trk", "a.trk.gz", "a.vtk", "ascii.vtk.gz"]


def _saved(tmp_path: Path, name: str) -> Path:
    return _save_tractogram(tmp_path / name, data=".tck" not in name)


@pytest.mark.parametrize("name", SUBSET_NAMES)
@pytest.mark.parametrize("seed", [0, 7])
def test_subset_equals_loading_then_subsampling(
    tmp_path: Path, name: str, seed: int
) -> None:
    path = _saved(tmp_path, name)
    got = load_tractogram(path, max_streamlines=4, seed=seed)
    assert got.num_streamlines == 4
    _assert_same(got, load_tractogram(path).subsample(4, seed=seed))


@pytest.mark.parametrize("name", SUBSET_NAMES)
def test_same_seed_gives_the_same_subset(tmp_path: Path, name: str) -> None:
    path = _saved(tmp_path, name)
    _assert_same(
        load_tractogram(path, max_streamlines=3, seed=5),
        load_tractogram(path, max_streamlines=3, seed=5),
    )


@pytest.mark.parametrize("name", SUBSET_NAMES)
@pytest.mark.parametrize("extra", [0, 1, 100])
def test_max_streamlines_at_least_the_count_keeps_all(
    tmp_path: Path, name: str, extra: int
) -> None:
    path = _saved(tmp_path, name)
    got = load_tractogram(path, max_streamlines=len(LENGTHS) + extra, seed=3)
    _assert_same(got, load_tractogram(path))


@pytest.mark.parametrize("name", SUBSET_NAMES)
def test_max_streamlines_zero_gives_an_empty_tractogram(
    tmp_path: Path, name: str
) -> None:
    got = load_tractogram(_saved(tmp_path, name), max_streamlines=0)
    assert got.num_streamlines == 0
    assert got.num_points == 0


def test_negative_max_streamlines_raises_before_reading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_streamlines"):
        load_tractogram(tmp_path / "missing.tck", max_streamlines=-1)


@pytest.mark.parametrize("name", ["a.tck", "a.vtk"])
def test_subset_of_a_format_without_a_grid_gets_the_given_reference(
    tmp_path: Path, name: str
) -> None:
    path = _saved(tmp_path, name)
    assert load_tractogram(path, max_streamlines=3, seed=0).reference is None
    got = load_tractogram(path, reference=GRID, max_streamlines=3, seed=0)
    assert got.reference is GRID


def test_subset_of_a_trk_keeps_its_own_grid(tmp_path: Path) -> None:
    path = _saved(tmp_path, "a.trk")
    got = load_tractogram(path, max_streamlines=3, seed=0)
    assert got.reference is not None
    assert got.reference.matches(GRID)
