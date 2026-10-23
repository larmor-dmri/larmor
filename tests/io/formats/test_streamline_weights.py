from pathlib import Path

import numpy as np
import pytest

from larmor.io import read_streamline_weights, write_streamline_weights

DATA = Path(__file__).parents[2] / "data" / "editing"


def test_weights_round_trip_exactly(tmp_path: Path) -> None:
    weights = np.random.default_rng(0).uniform(-3, 3, 50).astype(np.float32)
    weights[:3] = [0.1, 1e-30, 3.4e38]
    write_streamline_weights(weights, tmp_path / "w.txt")
    np.testing.assert_array_equal(read_streamline_weights(tmp_path / "w.txt"), weights)


@pytest.mark.parametrize(
    ("name", "count", "first"),
    [
        ("weights_selected.txt", 233, 0.702532411),
        ("sift2_weights.txt", 500, 0.009879153131),
    ],
    ids=["tckedit", "tcksift2-with-comment"],
)
def test_reads_mrtrix_weights_files(name: str, count: int, first: float) -> None:
    weights = read_streamline_weights(DATA / name)
    assert weights.dtype == np.float32
    assert weights.shape == (count,)
    assert weights[0] == np.float32(first)


def test_written_file_has_the_layout_of_tckedit(tmp_path: Path) -> None:
    expected = (DATA / "weights_selected.txt").read_text()
    write_streamline_weights(
        read_streamline_weights(DATA / "weights_selected.txt"), tmp_path / "w.txt"
    )
    assert (tmp_path / "w.txt").read_text() == expected


def test_appended_batches_read_back_as_one_array(tmp_path: Path) -> None:
    weights = np.arange(7, dtype=np.float32) / 3
    write_streamline_weights(np.array([9.0]), tmp_path / "w.txt")
    write_streamline_weights(weights[:4], tmp_path / "w.txt")
    write_streamline_weights(weights[4:], tmp_path / "w.txt", append=True)
    np.testing.assert_array_equal(read_streamline_weights(tmp_path / "w.txt"), weights)


@pytest.mark.parametrize(
    ("text", "message"),
    [("1\n0.5 abc\n", r"w\.txt, line 2"), ("1 2\n3 4\n", "table")],
    ids=["not-a-number", "table"],
)
def test_bad_weights_files_raise(tmp_path: Path, text: str, message: str) -> None:
    (tmp_path / "w.txt").write_text(text)
    with pytest.raises(ValueError, match=message):
        read_streamline_weights(tmp_path / "w.txt")
