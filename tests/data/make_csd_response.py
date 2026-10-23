"""Generates the white matter response of the ISMRM 2015 data in tests/data/csd/.

Runs MRtrix3 on the whole ISMRM image (`fetch_ismrm2015`): `mrconvert` with its FSL
gradient files, `dwi2mask`, then `dwi2response tournier` in that mask, and writes the
response file (one row, zonal SH coefficients l = 0 to 10, at b = 1000) without
MRtrix's command history.

Run from the repository root:

    uv run python tests/data/make_csd_response.py
"""

import pathlib
import subprocess
import tempfile

from larmor.data import fetch_ismrm2015

OUT = pathlib.Path("tests/data/csd/wm_response.txt")

files = fetch_ismrm2015()
with tempfile.TemporaryDirectory() as tmp:
    dwi, mask, response = (
        pathlib.Path(tmp) / n for n in ("dwi.mif", "mask.mif", "r.txt")
    )
    subprocess.run(
        ["mrconvert", str(files["dwi"]), str(dwi), "-fslgrad", str(files["bvecs"]),
         str(files["bvals"]), "-quiet"],
        check=True,
    )  # fmt: skip
    subprocess.run(["dwi2mask", str(dwi), str(mask), "-quiet"], check=True)
    subprocess.run(
        ["dwi2response", "tournier", str(dwi), str(response), "-mask", str(mask),
         "-scratch", tmp, "-quiet"],
        check=True,
    )  # fmt: skip
    version = subprocess.run(
        ["dwi2fod", "-version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0]
    rows = [
        line
        for line in response.read_text().splitlines()
        if not line.startswith("# command_history")
    ]
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join([f"# {version}", *rows]) + "\n")
print(f"Wrote {OUT} ({version})")  # noqa: T201 (a script run by hand)
