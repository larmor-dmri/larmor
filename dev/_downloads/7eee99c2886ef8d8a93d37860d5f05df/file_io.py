"""Reading and writing files
=========================

This example makes a small diffusion image, saves it with its gradients in several
formats and reads it back, does the same with a tractogram, and converts one format
to another.

Larmor reads and writes images as NIfTI and MRtrix MIF files, gradient tables as FSL
bvals and bvecs files or MRtrix gradient files, and tractograms as TCK, TRK and VTK
files. Each file is read into one of Larmor's types, in world coordinates (RAS+, in
mm), whatever the file's own convention, and written from it. The file's extension
chooses the format.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np

from larmor.io import (
    load_dwi,
    load_header,
    load_tractogram,
    load_volume,
    read_mrtrix_gradients,
    save_tractogram,
    save_volume,
    write_bvals_bvecs,
    write_mrtrix_gradients,
)
from larmor.types import GradientTable, Tractogram, Volume

folder = Path(tempfile.mkdtemp())

# %%
# An image and its gradients
# --------------------------
# The image is 4D, with 10 x 12 x 8 voxels of 2 mm and 7 volumes: one b=0 volume and
# six directions at b=1000 s/mm^2. Its gradient table holds the b-values in s/mm^2
# and the directions as unit vectors in world coordinates.

rng = np.random.default_rng(0)
affine = np.diag([2.0, 2.0, 2.0, 1.0])
dwi = Volume(rng.uniform(100, 1000, (10, 12, 8, 7)).astype(np.float32), affine)
directions = np.array(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float
)
directions /= np.linalg.norm(directions, axis=1, keepdims=True)
gtab = GradientTable(np.r_[0.0, [1000.0] * 6], np.vstack([np.zeros(3), directions]))

# %%
# NIfTI with bvals and bvecs
# --------------------------
# The bvecs file is written in the FSL convention, in the frame of the image's
# voxel axes, so it needs the image it belongs to. `load_dwi` reads the image and
# its gradient files together.

save_volume(dwi, folder / "dwi.nii.gz")
write_bvals_bvecs(
    gtab, folder / "dwi.bval", folder / "dwi.bvec", ref_path=folder / "dwi.nii.gz"
)
image, table = load_dwi(folder / "dwi.nii.gz", folder / "dwi.bval", folder / "dwi.bvec")
print(
    f"NIfTI: shape {image.data.shape}, same data: {np.allclose(image.data, dwi.data)}"
)
print(f"bvecs read back: {np.allclose(table.bvecs, gtab.bvecs)}")

# %%
# MIF, with the gradients in its header
# -------------------------------------
# A MIF file can hold the gradient table in its header, in world coordinates;
# `load_dwi` then needs no gradient files. A MRtrix gradient file (`.b`) holds the
# same table on its own.

save_volume(dwi, folder / "dwi.mif", gradients=gtab)
image, table = load_dwi(folder / "dwi.mif")
write_mrtrix_gradients(gtab, folder / "dwi.b")
print(f"MIF: {len(table.bvals)} gradients in the header")
from_b = read_mrtrix_gradients(folder / "dwi.b")
print(f".b file read back: {np.allclose(from_b.bvecs, gtab.bvecs)}")

# %%
# The header alone
# ----------------
# `load_header` reads the grid and the shape without the data, which is fast for
# large files.

grid, shape = load_header(folder / "dwi.nii.gz")
print(f"Header: shape {shape}, voxel size {np.abs(np.diag(grid.affine)[:3])} mm")

# %%
# Tractograms
# -----------
# A tractogram holds one array of all the points and the offsets where each
# streamline starts. TCK files store the points in world coordinates. TRK files store
# them in the voxel frame of a reference image, so writing one needs the image's
# grid. Saving to another extension converts the file.

line = np.linspace([-8.0, 0.0, 0.0], [8.0, 0.0, 0.0], 17, dtype=np.float32)
points = np.concatenate([line + [0.0, offset, 4.0] for offset in (-2.0, 0.0, 2.0)])
tracts = Tractogram(points, np.array([0, 17, 34, 51]))
save_tractogram(tracts, folder / "tracts.tck")
save_tractogram(
    load_tractogram(folder / "tracts.tck"), folder / "tracts.trk", reference=grid
)
back = load_tractogram(folder / "tracts.trk")
same = np.allclose(back.points, points)
print(f"TRK: {len(back.offsets) - 1} streamlines, same points: {same}")
print(f"Volume read back from MIF: {load_volume(folder / 'dwi.mif').data.shape}")

shutil.rmtree(folder)
