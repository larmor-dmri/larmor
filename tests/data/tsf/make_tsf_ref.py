"""Generates the TSF (track scalar file) references in tests/data/tsf/.

Run from the repository root, with MRtrix 3.0.8 on the PATH and the ISMRM data:

    uv run python tests/data/tsf/make_tsf_ref.py <ISMRM Data folder>

where the folder holds the ISMRM b0 image, bzero.nii.gz.

which runs, in a scratch folder holding links named tracks.tck (to
tests/data/editing/tracks.tck) and bzero.nii.gz (to the ISMRM image), so that no
local path ends up in the files' command_history:

    tcksample tracks.tck bzero.nii.gz bzero.tsf
    tsfsmooth bzero.tsf bzero_smooth.tsf

and copies both files to tests/data/tsf/.

bzero.tsf holds the b0 image sampled at every point of the 500 streamlines
(Float32LE, with the tractogram's timestamp). The script then writes the same
values with Larmor, as Float32 and as Float64 with the tractogram's timestamp, and
checks that MRtrix reads both files: tsfvalidate against the tractogram passes,
tsfinfo -count finds 500 streamlines, tsfsmooth gives bzero_smooth.tsf and
tsfdivide gives the values over the smoothed ones. These checks are not saved:
the script stops at the first that fails.
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from larmor.io import (
    load_tractogram,
    read_track_timestamp,
    read_tsf,
    write_tsf,
)

OUT = pathlib.Path("tests/data/tsf")
TRACKS = pathlib.Path("tests/data/editing/tracks.tck")
IMAGE = pathlib.Path(sys.argv[1]).resolve() / "bzero.nii.gz"


def mrtrix(*args: str | pathlib.Path, cwd: str | None = None) -> str:
    """Runs an MRtrix command and returns what it printed."""
    result = subprocess.run(
        [str(a) for a in args], capture_output=True, text=True, check=True, cwd=cwd
    )
    return result.stdout + result.stderr


with tempfile.TemporaryDirectory() as folder:
    (pathlib.Path(folder) / "tracks.tck").symlink_to(TRACKS.resolve())
    (pathlib.Path(folder) / "bzero.nii.gz").symlink_to(IMAGE)
    mrtrix("tcksample", "-quiet", "tracks.tck", "bzero.nii.gz", "bzero.tsf", cwd=folder)
    mrtrix("tsfsmooth", "-quiet", "bzero.tsf", "bzero_smooth.tsf", cwd=folder)
    for name in ("bzero.tsf", "bzero_smooth.tsf"):
        shutil.copyfile(pathlib.Path(folder) / name, OUT / name)

tractogram = load_tractogram(TRACKS)
values, offsets = read_tsf(OUT / "bzero.tsf", tractogram)
smooth, _ = read_tsf(OUT / "bzero_smooth.tsf", tractogram)
timestamp = read_track_timestamp(TRACKS)

with tempfile.TemporaryDirectory() as scratch:
    for dtype in (np.float32, np.float64):
        name = np.dtype(dtype).name
        larmor_file = pathlib.Path(scratch) / f"{name}.tsf"
        write_tsf(values.astype(dtype), offsets, larmor_file, timestamp=timestamp)
        validate = mrtrix("tsfvalidate", larmor_file, TRACKS)
        count = mrtrix("tsfinfo", "-count", larmor_file)
        smoothed = pathlib.Path(scratch) / f"{name}_smooth.tsf"
        mrtrix("tsfsmooth", "-quiet", larmor_file, smoothed)
        smoothed_values, _ = read_tsf(smoothed, tractogram)
        divided = pathlib.Path(scratch) / f"{name}_divided.tsf"
        mrtrix("tsfdivide", "-quiet", larmor_file, smoothed, divided)
        divided_values, _ = read_tsf(divided, tractogram)
        assert "checked OK" in validate, validate
        assert "actual count in file: 500" in count, count
        assert np.array_equal(smoothed_values, smooth)
        assert np.allclose(divided_values, values / smooth, rtol=1e-6)
version = subprocess.run(["tcksample", "-version"], capture_output=True, text=True)
print("saved", OUT, version.stdout.splitlines()[0])  # noqa: T201 (script output)
