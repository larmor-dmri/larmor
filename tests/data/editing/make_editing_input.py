"""Generates tests/data/editing/tracks.tck, the input of the tractogram editing tests.

Run from the repository root with the ISMRM MRT/tracks.tck (6555 streamlines):
uv run python tests/data/editing/make_editing_input.py <path to MRT/tracks.tck>
"""

import pathlib
import subprocess
import sys

SOURCE = pathlib.Path(sys.argv[1])
OUT = pathlib.Path("tests/data/editing")

subprocess.run(
    ["tckedit", "-quiet", "-force", "-number", "500", SOURCE, OUT / "tracks.tck"],
    check=True,
)

version = subprocess.run(["tckedit", "-version"], capture_output=True, text=True)
print("saved", OUT / "tracks.tck", version.stdout.splitlines()[0])  # noqa: T201 (script output)
