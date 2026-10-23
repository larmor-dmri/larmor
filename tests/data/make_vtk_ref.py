"""Generates the MRtrix-written reference files in tests/data/vtk/ with tckconvert.

Run from the repository root, after make_tck_ref.py:
uv run python tests/data/make_vtk_ref.py
"""

import pathlib
import subprocess

OUT = pathlib.Path("tests/data/vtk")
OUT.mkdir(exist_ok=True)
TCK = "../tck/ref.tck"

for name, extra in (("ascii.vtk", []), ("binary.vtk", ["-binary"])):
    subprocess.run(
        ["tckconvert", "-quiet", "-force", *extra, TCK, name], cwd=OUT, check=True
    )

version = subprocess.run(["tckconvert", "-version"], capture_output=True, text=True)
print("saved", OUT, version.stdout.splitlines()[0])  # noqa: T201 (script output)
