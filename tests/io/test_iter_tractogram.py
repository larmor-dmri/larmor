import gc
import gzip
import logging
import math
import warnings
from collections.abc import Callable
from io import BufferedReader
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from larmor.io import iter_tractogram, load_tractogram, reader
from larmor.io.formats import trk, write_tck, write_trk
from larmor.io.formats.trk import HEADER_SIZE, TRK_HEADER
from larmor.types import Grid, Tractogram

DATA = Path(__file__).parents[1] / "data"
REF_TCK = DATA / "tck" / "ref.tck"
EDITING_TCK = DATA / "editing" / "tracks.tck"

GRID = Grid((4, 5, 6), np.diag([2.0, 2.5, 3.0, 1.0]))
# Zero-length streamlines at the start, middle and end, so that an off-by-one at a
# boundary shows.
LENGTHS = [0, 3, 1, 5, 0, 0, 2, 4, 0]
# Smaller than a TCK row (12 bytes) and not a multiple of 4, so that chunks end
# inside values, rows and streamlines.
TINY_CHUNK = 7


def _tractogram(lengths: list[int], with_data: bool) -> Tractogram:
    rng = np.random.default_rng(0)
    n, s = sum(lengths), len(lengths)
    offsets = np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)])
    points = rng.uniform(0, 8, (n, 3)).astype(np.float32)
    if not with_data:
        return Tractogram(points, offsets)
    return Tractogram(
        points,
        offsets,
        {
            "fa": rng.uniform(0, 1, n).astype(np.float32),
            "dir": rng.uniform(-1, 1, (n, 3)).astype(np.float32),
        },
        {
            "length": rng.uniform(10, 50, s).astype(np.float32),
            "rgb": rng.uniform(0, 1, (s, 3)).astype(np.float32),
        },
        GRID,
    )


def _trk(tmp_path: Path, lengths: list[int] = LENGTHS) -> Path:
    path = tmp_path / "a.trk"
    write_trk(_tractogram(lengths, with_data=True), path)
    return path


def _tck(tmp_path: Path, lengths: list[int] = LENGTHS) -> Path:
    path = tmp_path / "a.tck"
    write_tck(_tractogram(lengths, with_data=False), path)
    return path


def _tck_float64_big_endian(tmp_path: Path) -> Path:
    path = _tck(tmp_path)
    raw = path.read_bytes()
    end = raw.index(b"\nEND\n") + len(b"\nEND\n")
    header = raw[:end].decode()
    offset = int(header.split("file: . ")[1].split()[0])
    payload = np.frombuffer(raw[offset:], "<f4").astype(">f8")
    header = header.replace("Float32LE", "Float64BE")
    path.write_bytes(header.encode().ljust(offset, b"\0") + payload.tobytes())
    return path


def _trk_big_endian(tmp_path: Path) -> Path:
    path = _trk(tmp_path)
    raw = path.read_bytes()
    header = np.frombuffer(raw[:HEADER_SIZE], TRK_HEADER)
    swapped = header.astype(TRK_HEADER.newbyteorder(">"))
    payload = np.frombuffer(raw[HEADER_SIZE:], "<i4").astype(">i4")
    path.write_bytes(swapped.tobytes() + payload.tobytes())
    return path


FILES: dict[str, Callable[[Path], Path]] = {
    "ref-tck": lambda _: REF_TCK,
    "editing-tck": lambda _: EDITING_TCK,
    "trk": _trk,
    "tck-float64-big-endian": _tck_float64_big_endian,
    "trk-big-endian": _trk_big_endian,
}


def _join(batches: list[Tractogram]) -> Tractogram:
    counts = np.concatenate([np.diff(b.offsets) for b in batches])
    first = batches[0]
    return Tractogram(
        np.concatenate([b.points for b in batches]),
        np.concatenate([[0], np.cumsum(counts)]),
        {
            k: np.concatenate([b.point_data[k] for b in batches])
            for k in first.point_data
        },
        {
            k: np.concatenate([b.streamline_data[k] for b in batches])
            for k in first.streamline_data
        },
        first.reference,
    )


def _assert_identical(got: Tractogram, want: Tractogram) -> None:
    assert got.points.dtype == want.points.dtype
    np.testing.assert_array_equal(got.points, want.points)
    np.testing.assert_array_equal(got.offsets, want.offsets)
    for got_data, want_data in [
        (got.point_data, want.point_data),
        (got.streamline_data, want.streamline_data),
    ]:
        assert list(got_data) == list(want_data)
        for name, values in want_data.items():
            assert got_data[name].dtype == values.dtype
            np.testing.assert_array_equal(got_data[name], values)
    if want.reference is None:
        assert got.reference is None
    else:
        assert got.reference is not None
        assert got.reference.shape == want.reference.shape
        np.testing.assert_array_equal(got.reference.affine, want.reference.affine)


@pytest.fixture(params=[False, True], ids=["16MB-chunks", "tiny-chunks"])
def tiny_chunks(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> bool:
    if request.param:
        monkeypatch.setattr(reader, "_CHUNK", TINY_CHUNK)
    return bool(request.param)


@pytest.mark.parametrize("batch_size", [1, 4, 10**6])
@pytest.mark.parametrize("make", FILES.values(), ids=list(FILES))
@pytest.mark.usefixtures("tiny_chunks")
def test_joined_batches_equal_the_whole_file(
    tmp_path: Path, make: Callable[[Path], Path], batch_size: int
) -> None:
    path = make(tmp_path)
    batches = list(iter_tractogram(path, batch_size=batch_size))
    whole = load_tractogram(path)
    _assert_identical(_join(batches), whole)
    for batch in batches:
        assert batch.offsets.dtype == np.int64
        if whole.reference is None:
            assert batch.reference is None
        else:
            assert batch.reference is not None
            assert batch.reference.matches(whole.reference)


@pytest.mark.parametrize("batch_size", [1, 2, 4, len(LENGTHS), len(LENGTHS) + 1])
@pytest.mark.parametrize("make", [_tck, _trk], ids=["tck", "trk"])
@pytest.mark.usefixtures("tiny_chunks")
def test_every_batch_but_the_last_has_batch_size_streamlines(
    tmp_path: Path, make: Callable[[Path], Path], batch_size: int
) -> None:
    # len(LENGTHS) and its divisors put a batch boundary at the end of the file.
    sizes = [
        b.num_streamlines
        for b in iter_tractogram(make(tmp_path), batch_size=batch_size)
    ]
    n = len(LENGTHS)
    assert len(sizes) == math.ceil(n / batch_size)
    assert sizes[:-1] == [batch_size] * (len(sizes) - 1)
    assert sum(sizes) == n


@pytest.mark.usefixtures("tiny_chunks")
def test_empty_tck_gives_one_empty_batch(tmp_path: Path) -> None:
    batches = list(iter_tractogram(_tck(tmp_path, []), batch_size=3))
    assert len(batches) == 1
    assert batches[0].num_streamlines == 0
    assert batches[0].points.shape == (0, 3)


@pytest.mark.usefixtures("tiny_chunks")
def test_empty_trk_gives_one_empty_batch_with_data_names_and_grid(
    tmp_path: Path,
) -> None:
    path = _trk(tmp_path, [])
    batches = list(iter_tractogram(path, batch_size=3))
    assert len(batches) == 1
    _assert_identical(batches[0], load_tractogram(path))
    assert set(batches[0].point_data) == {"fa", "dir"}
    assert set(batches[0].streamline_data) == {"length", "rgb"}
    reference = batches[0].reference
    assert reference is not None
    assert reference.matches(GRID)


def _set_count(path: Path, count: str | None) -> None:
    raw = path.read_bytes()
    line = next(x for x in raw.split(b"\n") if x.startswith(b"count:"))
    new = b"" if count is None else f"count: {count}".encode()
    # Same header length, so that the data offset in the header stays right.
    path.write_bytes(raw.replace(line + b"\n", new.ljust(len(line)) + b"\n", 1))


def _wrong_count(path: Path) -> None:
    _set_count(path, "3")


def _missing_count(path: Path) -> None:
    _set_count(path, None)


def _no_end_marker(path: Path) -> None:
    path.write_bytes(path.read_bytes()[:-12])


def _cut_in_last_streamline(path: Path) -> None:
    # Drops the end marker and the empty last streamline, and cuts the file inside
    # the NaN row closing the streamline before it.
    path.write_bytes(path.read_bytes()[: -12 - 12 - 2 - 8])


def _data_after_end_marker(path: Path) -> None:
    rows = np.array([[1, 2, 3], [4, 5, 6], [np.nan] * 3], dtype="<f4")
    path.write_bytes(path.read_bytes() + rows.tobytes())


def _unclosed_before_end_marker(path: Path) -> None:
    # Drops the NaN rows of the last two streamlines (the last is empty), keeping
    # the end marker after the last point.
    raw = path.read_bytes()
    path.write_bytes(raw[:-36] + raw[-12:])


def _cut_trk(path: Path) -> None:
    path.write_bytes(path.read_bytes()[:-6])


def _huge_last_count(path: Path) -> None:
    # The last streamline has no points, so its count is followed by its 4
    # properties only.
    raw = bytearray(path.read_bytes())
    raw[-5 * 4 : -4 * 4] = np.array([2**30], "<i4").tobytes()
    path.write_bytes(bytes(raw))


# The file maker, the damage and whether the whole-file reader logs anything (a
# missing end marker or data after it is not reported).
DAMAGES: dict[str, tuple[Callable[[Path], Path], Callable[[Path], None], bool]] = {
    "tck-wrong-count": (_tck, _wrong_count, True),
    "tck-missing-count": (_tck, _missing_count, True),
    "tck-no-end-marker": (_tck, _no_end_marker, False),
    "tck-cut-in-last-streamline": (_tck, _cut_in_last_streamline, True),
    "tck-data-after-end-marker": (_tck, _data_after_end_marker, False),
    "tck-unclosed-before-end-marker": (_tck, _unclosed_before_end_marker, True),
    "trk-cut-in-last-streamline": (_trk, _cut_trk, True),
    "trk-huge-last-count": (_trk, _huge_last_count, True),
}


@pytest.mark.parametrize("make, damage, logs", DAMAGES.values(), ids=list(DAMAGES))
@pytest.mark.usefixtures("tiny_chunks")
def test_damaged_file_reads_and_logs_as_the_whole_file_reader(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    make: Callable[[Path], Path],
    damage: Callable[[Path], None],
    logs: bool,
) -> None:
    path = make(tmp_path)
    damage(path)
    with caplog.at_level(logging.INFO, logger="larmor"):
        whole = load_tractogram(path)
        whole_log = [(r.levelno, r.getMessage()) for r in caplog.records]
        caplog.clear()
        batches = list(iter_tractogram(path, batch_size=2))
        batched_log = [(r.levelno, r.getMessage()) for r in caplog.records]
    _assert_identical(_join(batches), whole)
    assert batched_log == whole_log
    assert (whole_log != []) == logs


@pytest.mark.parametrize("name", ["a.vtk", "a.fib", "a.tck.gz", "a.trk.gz"])
def test_vtk_and_gz_names_are_refused_before_reading(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="uncompressed"):
        iter_tractogram(tmp_path / name)


@pytest.mark.parametrize("name", ["a.tck", "a.trk"])
def test_gzip_content_is_refused_at_the_call(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_bytes(gzip.compress(REF_TCK.read_bytes()))
    with pytest.raises(ValueError, match="gzip"):
        iter_tractogram(path)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_batch_size_below_one_is_refused(tmp_path: Path, batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        iter_tractogram(tmp_path / "a.tck", batch_size=batch_size)


def test_reference_is_given_to_every_tck_batch(tmp_path: Path) -> None:
    batches = list(iter_tractogram(_tck(tmp_path), batch_size=2, reference=GRID))
    assert len(batches) > 1
    assert all(b.reference is GRID for b in batches)


@pytest.mark.parametrize("make", [_tck, _trk], ids=["tck", "trk"])
def test_an_iterator_never_iterated_leaves_no_open_file(
    tmp_path: Path, make: Callable[[Path], Path]
) -> None:
    path = make(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        batches = iter_tractogram(path, batch_size=2)
        del batches
        gc.collect()
    assert [w for w in caught if issubclass(w.category, ResourceWarning)] == []


def test_trk_count_past_the_end_stops_the_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _trk(tmp_path)
    raw = bytearray(path.read_bytes())
    raw[HEADER_SIZE : HEADER_SIZE + 4] = np.array([2**30], "<i4").tobytes()
    path.write_bytes(bytes(raw))
    monkeypatch.setattr(reader, "_CHUNK", TINY_CHUNK)
    calls = []
    read_chunk = reader.read_chunk

    def counting_read_chunk(
        f: BufferedReader, carry: NDArray[np.uint8]
    ) -> NDArray[np.uint8]:
        calls.append(1)
        return read_chunk(f, carry)

    monkeypatch.setattr(trk, "read_chunk", counting_read_chunk)
    batches = list(iter_tractogram(path, batch_size=2))
    assert [b.num_streamlines for b in batches] == [0]
    # The first count is known after one chunk; the rest of the file is not read.
    assert len(calls) <= 2


@pytest.mark.parametrize("make", [_tck, _trk], ids=["tck", "trk"])
def test_batches_own_their_arrays(tmp_path: Path, make: Callable[[Path], Path]) -> None:
    for batch in iter_tractogram(make(tmp_path), batch_size=1):
        assert batch.points.base is None
        assert batch.offsets.base is None
        for values in [*batch.point_data.values(), *batch.streamline_data.values()]:
            assert values.base is None
