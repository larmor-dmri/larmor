import dataclasses
import re
from pathlib import Path

import pytest

from larmor.io import read_lookup_table
from larmor.types import Label

# FreeSurfer's layout: a header, blank lines, aligned columns and a transparency
# column.
FREESURFER = """\
#$Id: FreeSurferColorLUT.txt
# No. Label Name:                R   G   B   A

0   Unknown                       0   0   0   0
2   Left-Cerebral-White-Matter  245 245 245   0

  # an indented comment
3   Left-Cerebral-Cortex        205  62  78   0
1000 ctx-lh-unknown              25   5  25   0
3   Left-Cortex-Again            10  20  30   0
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "lut.txt"
    path.write_text(text)
    return path


def test_a_freesurfer_table_gives_each_index_its_name_and_colour(
    tmp_path: Path,
) -> None:
    table = read_lookup_table(_write(tmp_path, FREESURFER))
    assert list(table) == [0, 2, 3, 1000]
    assert table[0] == Label("Unknown", (0, 0, 0))
    assert table[2] == Label("Left-Cerebral-White-Matter", (245, 245, 245))
    assert table[1000] == Label("ctx-lh-unknown", (25, 5, 25))


def test_a_later_line_for_an_index_replaces_the_earlier(tmp_path: Path) -> None:
    table = read_lookup_table(_write(tmp_path, FREESURFER))
    assert table[3] == Label("Left-Cortex-Again", (10, 20, 30))


# MRtrix's layout, as in its fs_default.txt: a short and a long name, alpha 255.
MRTRIX = """\
# MRtrix lookup table
0    Unknown   Unknown                         0   0   0     0
1    L.LOF     ctx-lh-lateralorbitofrontal    35  75  50   255
2    L.PORB    ctx-lh-parsorbitalis          220 180 140   255
"""


def test_an_mrtrix_table_gives_each_index_its_long_name_and_colour(
    tmp_path: Path,
) -> None:
    assert read_lookup_table(_write(tmp_path, MRTRIX)) == {
        0: Label("Unknown", (0, 0, 0)),
        1: Label("ctx-lh-lateralorbitofrontal", (35, 75, 50)),
        2: Label("ctx-lh-parsorbitalis", (220, 180, 140)),
    }


def test_a_line_without_the_last_column_still_reads(tmp_path: Path) -> None:
    path = _write(tmp_path, "7 Seven 1 2 3\n")
    assert read_lookup_table(path) == {7: Label("Seven", (1, 2, 3))}


def test_one_file_can_mix_the_layouts(tmp_path: Path) -> None:
    text = "1 One 1 2 3 0\n2 L.Two Left-Two 4 5 6 255\n3 Three 7 8 9\n"
    assert read_lookup_table(_write(tmp_path, text)) == {
        1: Label("One", (1, 2, 3)),
        2: Label("Left-Two", (4, 5, 6)),
        3: Label("Three", (7, 8, 9)),
    }


def test_the_path_can_be_text(tmp_path: Path) -> None:
    path = _write(tmp_path, "7 Seven 1 2 3 0\n")
    assert read_lookup_table(str(path)) == {7: Label("Seven", (1, 2, 3))}


@pytest.mark.parametrize("rgb", ["0 0 0", "255 255 255"])
def test_the_ends_of_the_colour_range_are_accepted(rgb: str, tmp_path: Path) -> None:
    table = read_lookup_table(_write(tmp_path, f"1 One {rgb} 0\n"))
    assert table[1].rgb == tuple(int(c) for c in rgb.split())


def test_a_missing_file_is_refused_naming_it(tmp_path: Path) -> None:
    path = tmp_path / "missing.txt"
    with pytest.raises(ValueError, match=re.escape(f"{path} cannot be read")):
        read_lookup_table(path)


def test_a_binary_file_is_refused_naming_it(tmp_path: Path) -> None:
    path = tmp_path / "lut.bin"
    path.write_bytes(b"\xff\xfe\x00\x81")
    with pytest.raises(ValueError, match=re.escape(f"{path} cannot be read")):
        read_lookup_table(path)


# The bad entry is on line 3, after a comment and a good entry.
@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("x Bad 1 2 3 0", "invalid literal"),
        ("1.5 Bad 1 2 3 0", "invalid literal"),
        ("4 Bad 1 2", "expected index, name, R, G, B"),
        ("4 Bad 1 2.5 3 0", "invalid literal"),
        ("4 Bad 256 0 0 0", "colours must be between 0 and 255"),
        ("4 Bad 0 -1 0 0", "colours must be between 0 and 255"),
        ("4 Bad 0 0 300 0", "colours must be between 0 and 255"),
        # Text after the last column is read as part of the colour.
        ("4 Bad 1 2 3 0 extra words", "invalid literal"),
    ],
)
def test_a_bad_entry_is_refused_naming_the_file_and_line(
    line: str, reason: str, tmp_path: Path
) -> None:
    path = _write(tmp_path, f"# header\n1 Good 1 2 3 0\n{line}\n")
    with pytest.raises(ValueError, match=re.escape(f"{path}, line 3: ")) as error:
        read_lookup_table(path)
    assert reason in str(error.value)
    assert repr(line) in str(error.value)


@pytest.mark.parametrize("text", ["", "\n\n", "# only\n   # comments\n"])
def test_a_file_without_entries_is_refused(text: str, tmp_path: Path) -> None:
    path = _write(tmp_path, text)
    with pytest.raises(ValueError, match=re.escape(f"{path} holds no lookup-table")):
        read_lookup_table(path)


def test_a_label_is_frozen() -> None:
    label = Label("One", (1, 2, 3))
    with pytest.raises(dataclasses.FrozenInstanceError):
        label.name = "Two"  # type: ignore[misc]
