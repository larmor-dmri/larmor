"""Generates the tckresample outputs that the resampling tests compare with.

Resamples tests/data/editing/tracks.tck, and a few hand-made edge cases written here
to edge_cases.tck, with MRtrix3's tckresample to step sizes and numbers of points.

Run from the repository root: uv run python tests/data/editing/make_resample_ref.py
"""

import pathlib
import subprocess

import numpy as np

from larmor.io.formats import write_tck
from larmor.types import Tractogram

OUT = pathlib.Path("tests/data/editing")

# One point; two points; a bend; coincident points inside; all points coincident; two
# points closer than any step; a short curve; a first point repeated; a last point
# repeated.
EDGE_CASES = [
    [[0, 0, 0]],
    [[0, 0, 0], [3, 0, 0]],
    [[0, 0, 0], [1, 0, 0], [1, 2, 0]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 1, 0], [2, 1, 0], [3, 1, 1]],
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    [[0, 0, 0], [0.3, 0, 0]],
    [[0, 0, 0], [0.5, 0.1, 0], [1.5, 0.4, 0.2], [2.0, 1.0, 0.5]],
    [[0, 0, 0], [0, 0, 0], [1, 0, 0], [2, 1, 0]],
    [[0, 0, 0], [1, 0, 0], [2, 1, 0], [2, 1, 0]],
]

RUNS = {
    "tracks": [("-step_size", "0.5"), ("-step_size", "2"), ("-num_points", "2"),
               ("-num_points", "10"), ("-num_points", "50")],
    "edge_cases": [("-step_size", "0.5"), ("-step_size", "2"), ("-num_points", "2"),
                   ("-num_points", "5")],
}  # fmt: skip

points = np.concatenate([np.array(s, np.float32) for s in EDGE_CASES])
offsets = np.concatenate([[0], np.cumsum([len(s) for s in EDGE_CASES])])
write_tck(Tractogram(points, offsets), OUT / "edge_cases.tck")

for stem, runs in RUNS.items():
    for option, value in runs:
        out = OUT / f"{stem}_{option.lstrip('-')}_{value}.tck"
        subprocess.run(
            [
                "tckresample",
                "-quiet",
                "-force",
                option,
                value,
                OUT / f"{stem}.tck",
                out,
            ],
            check=True,
        )
        print("saved", out)  # noqa: T201 (script output)

version = subprocess.run(["tckresample", "-version"], capture_output=True, text=True)
print(version.stdout.splitlines()[0])  # noqa: T201 (script output)
