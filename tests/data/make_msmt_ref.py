"""Generates the tissue responses of the MSMT-CSD tests in tests/data/msmt/.

Runs MRtrix3's `dwi2mask` and `dwi2response dhollander -shells 0,600,1000,1400,2000`
on the whole CFIN multi-shell data (`larmor.data.fetch_cfin_multishell`), so the
responses have one row for each shell of the `dki/` block, and writes them to
`tests/data/msmt/{wm,gm,csf}.txt` as dwi2response writes them.

Run from the repository root:

    uv run python tests/data/make_msmt_ref.py
"""

import pathlib
import shutil
import subprocess
import tempfile

from larmor.data import fetch_cfin_multishell

OUT = pathlib.Path("tests/data/msmt")
SHELLS = "0,600,1000,1400,2000"

OUT.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    scratch = pathlib.Path(tmp)
    files = fetch_cfin_multishell(folder=scratch / "data")
    dwi = scratch / "dwi.mif"
    mask = scratch / "mask.mif"
    subprocess.run(
        [
            "mrconvert",
            str(files["dwi"]),
            str(dwi),
            "-fslgrad",
            str(files["bvecs"]),
            str(files["bvals"]),
            "-quiet",
        ],
        check=True,
    )
    subprocess.run(["dwi2mask", str(dwi), str(mask), "-quiet"], check=True)
    responses = [scratch / f"{tissue}.txt" for tissue in ("wm", "gm", "csf")]
    subprocess.run(
        [
            "dwi2response",
            "dhollander",
            str(dwi),
            *map(str, responses),
            "-mask",
            str(mask),
            "-shells",
            SHELLS,
            "-quiet",
        ],
        check=True,
    )
    for response in responses:
        shutil.copy(response, OUT / response.name)
print("saved", sorted(p.name for p in OUT.iterdir()))  # noqa: T201 (script output)
