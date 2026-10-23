"""Generates the references for sampling an image along streamlines.

Run from the repository root, with MRtrix 3.0.8 on the PATH and the ISMRM data:

    uv run python tests/data/tsf/make_sampling_ref.py <ISMRM Data folder>

where the folder holds the ISMRM b0 image, bzero.nii.gz.

It writes, in tests/data/tsf/:

- bzero_crop.nii.gz: the ISMRM bzero.nii.gz cropped to the box of the points of
  tests/data/editing/tracks.tck plus a margin of 2 voxels, with the affine moved
  so every voxel keeps its world position. Sampling it at those points gives the
  values of bzero.tsf (made from the whole image by make_tsf_ref.py).
- edges.nii, edges.tck, edges.tsf: a 5 x 4 x 3 image of 2 x 1 x 1 mm voxels whose
  value is 10 + x (plus 100 on the row y = 0), one streamline of points at the
  image's edges and outside them, and `tcksample edges.tck edges.nii edges.tsf`.
  The run is in a scratch folder, so no local path is in command_history.
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from larmor.io import load_tractogram, load_volume, save_tractogram, save_volume
from larmor.types import Tractogram, Volume

OUT = pathlib.Path("tests/data/tsf")
TRACKS = pathlib.Path("tests/data/editing/tracks.tck")
IMAGE = pathlib.Path(sys.argv[1]).resolve() / "bzero.nii.gz"
MARGIN = 2  # voxels

# The bzero image cropped around the streamlines.
tractogram = load_tractogram(TRACKS)
image = load_volume(IMAGE)
to_voxel = np.linalg.inv(image.affine)
voxels = tractogram.points @ to_voxel[:3, :3].T + to_voxel[:3, 3]
low = np.maximum(np.floor(voxels.min(axis=0)).astype(int) - MARGIN, 0)
high = np.minimum(
    np.ceil(voxels.max(axis=0)).astype(int) + MARGIN + 1, image.data.shape[:3]
)
cropped = np.asarray(image.data[low[0] : high[0], low[1] : high[1], low[2] : high[2]])
affine = image.affine.copy()
affine[:3, 3] = image.affine[:3, :3] @ low + image.affine[:3, 3]
save_volume(
    Volume(cropped, affine, image.slope, image.intercept), OUT / "bzero_crop.nii.gz"
)

# A small image, and points at and beyond its edges.
x = np.arange(5, dtype=np.float32)
data = np.broadcast_to((10 + x)[:, None, None], (5, 4, 3)).copy()
data[:, 0, :] += 100
edge_voxels = [
    (-0.6, 1, 1),
    (-0.5, 1, 1),
    (-0.49, 1, 1),
    (-0.4, 1, 1),
    (0.0, 1, 1),
    (0.3, 1, 1),
    (2.25, 1.5, 1),
    (3.7, 1, 1),
    (4.0, 1, 1),
    (4.4, 1, 1),
    (4.49, 1, 1),
    (4.5, 1, 1),
    (4.6, 1, 1),
    (2, -0.3, 1),
    (2, 0.5, 1),
    (2, 1, -0.5),
    (2, 1, 2.49),
    (2, 1, 2.51),
]
points = np.array(edge_voxels, np.float32) * np.array([2, 1, 1], np.float32)
with tempfile.TemporaryDirectory() as folder:
    scratch = pathlib.Path(folder)
    save_volume(Volume(data, np.diag([2.0, 1.0, 1.0, 1.0])), scratch / "edges.nii")
    save_tractogram(
        Tractogram(points, np.array([0, len(points)])), scratch / "edges.tck"
    )
    subprocess.run(
        ["tcksample", "-quiet", "edges.tck", "edges.nii", "edges.tsf"],
        check=True,
        cwd=folder,
    )
    for name in ("edges.nii", "edges.tck", "edges.tsf"):
        shutil.copyfile(scratch / name, OUT / name)
