"""Generates the MRtrix-written reference files in tests/data/tck/ with tckconvert.

Run from the repository root: uv run python tests/data/make_tck_ref.py
"""

import pathlib
import subprocess

import numpy as np

OUT = pathlib.Path("tests/data/tck")
OUT.mkdir(exist_ok=True)

# Multiples of 1/8 below 1000 in magnitude: exact in float32 and printed exactly by
# tckconvert's 6 significant digits, so the text files are the exact points.
rng = np.random.default_rng(0)
for s, n in enumerate([5, 1, 3, 2]):
    points = rng.integers(-8000, 8000, size=(n, 3)) / 8
    np.savetxt(OUT / f"s-{s}.txt", points, fmt="%.3f")

subprocess.run(
    ["tckconvert", "-quiet", "-force", "s-[].txt", "ref.tck"], cwd=OUT, check=True
)

version = subprocess.run(["tckconvert", "-version"], capture_output=True, text=True)
print("saved", OUT, version.stdout.splitlines()[0])  # noqa: T201 (script output)
