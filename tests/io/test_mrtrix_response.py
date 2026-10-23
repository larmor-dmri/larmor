"""Reading MRtrix3 response function files."""

from pathlib import Path

import numpy as np
import pytest

from larmor.io import read_mrtrix_response


def test_reads_one_row_per_shell_skipping_comments(tmp_path: Path) -> None:
    path = tmp_path / "response.txt"
    path.write_text("# Shells: 0,1000\n3000 0 0\n1200, -400, 90  # wm\n\n")

    np.testing.assert_array_equal(
        read_mrtrix_response(path), [[3000, 0, 0], [1200, -400, 90]]
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [("1 2 3\n4 5\n", "same degrees"), ("1 x 3\n", "not a row"), ("# none\n", "no")],
)
def test_rejects_malformed_files(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "response.txt"
    path.write_text(text)

    with pytest.raises(ValueError, match=message):
        read_mrtrix_response(path)
