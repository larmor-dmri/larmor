"""Bringing an atlas region into diffusion space
=============================================

This example brings a region of a template atlas into the diffusion space of a
subject with a registration as ANTs writes it, and selects the streamlines that pass
through the region. Everything is synthetic, so the example needs no data files.

An atlas lives in a template space, while streamlines live in the diffusion space of
one subject. A registration between the two, computed by ANTs, FSL or MRtrix3, links
them. Larmor reads such a registration and applies it, in either direction.
"""

import tempfile
from pathlib import Path

import numpy as np

from larmor.registration import read_ants_transform, transform_volume
from larmor.tractography import passes_through
from larmor.types import Grid, Tractogram, Volume

# %%
# The atlas and the diffusion grid
# --------------------------------
# The atlas is a 2 mm template with two regions, and the diffusion data has a grid of
# 2.5 mm voxels. A grid maps voxel indices to world coordinates (RAS+) in mm.

template = Grid(
    (40, 40, 40),
    np.diag([2.0, 2.0, 2.0, 1.0]) + np.c_[np.zeros((4, 3)), [-39, -39, -39, 0]],
)
ijk = np.stack(np.meshgrid(*map(np.arange, template.shape), indexing="ij"), -1)
world = ijk @ template.affine[:3, :3].T + template.affine[:3, 3]
atlas_data = np.zeros(template.shape, dtype=np.int16)
atlas_data[np.linalg.norm(world - [15.0, 0.0, 0.0], axis=-1) < 6] = 1
atlas_data[np.linalg.norm(world - [-15.0, 0.0, 0.0], axis=-1) < 6] = 2
atlas = Volume(atlas_data, template.affine)

diffusion = Grid(
    (32, 32, 32),
    np.diag([2.5, 2.5, 2.5, 1.0]) + np.c_[np.zeros((4, 3)), [-40, -40, -40, 0]],
)

# %%
# The registration
# ----------------
# Registering the subject's diffusion image (the moving image) to the template (the
# fixed image) with antsRegistration gives `0GenericAffine.mat`. Here its text form
# is written by hand, with a rotation of 10 degrees about z and a shift. ITK
# transform files, which ANTs writes, are in LPS coordinates, and Larmor converts
# them to RAS+ as it reads them. The transform maps points of the template to the
# diffusion space. This is the direction ANTs uses to resample the diffusion image
# onto the template grid.

angle = np.deg2rad(10)
parameters = [
    np.cos(angle),
    -np.sin(angle),
    0,
    np.sin(angle),
    np.cos(angle),
    0,
    0,
    0,
    1,
    4,
    -3,
    2,
]
folder = Path(tempfile.mkdtemp())
(folder / "affine.txt").write_text(
    "#Insight Transform File V1.0\n#Transform 0\n"
    "Transform: AffineTransform_double_3_3\n"
    f"Parameters: {' '.join(map(str, parameters))}\n"
    "FixedParameters: 0 0 0\n"
)
template_to_diffusion = read_ants_transform(folder / "affine.txt")

# %%
# The atlas region in diffusion space
# -----------------------------------
# Bringing the atlas onto the diffusion grid is the other direction: each
# diffusion voxel looks up its position in the template, so the transform is the
# inverse. Labels are resampled with nearest-neighbour interpolation, which keeps
# them whole numbers.

atlas_in_diffusion = transform_volume(
    atlas, template_to_diffusion.inverse(), diffusion, interpolation="nearest"
)
region = atlas_in_diffusion.with_data((atlas_in_diffusion.data == 1).astype(np.uint8))
print(f"region 1 voxels in diffusion space: {int(region.data.sum())}")

# %%
# Selecting a bundle
# ------------------
# Two synthetic bundles in diffusion space run along y, one through the place where
# region 1 lands and one through region 2. Each region's centre in diffusion space is
# its centre in the template, mapped by the template-to-diffusion transform.

centre_1 = template_to_diffusion.map_points(np.array([[15.0, 0.0, 0.0]]))[0]
centre_2 = template_to_diffusion.map_points(np.array([[-15.0, 0.0, 0.0]]))[0]
rng = np.random.default_rng(0)
streamlines = []
for centre in (centre_1, centre_2):
    for _ in range(20):
        start = centre + rng.normal(0, 1.5, 3) - [0, 25, 0]
        streamlines.append(start + np.outer(np.linspace(0, 50, 51), [0, 1, 0]))
offsets = np.concatenate([[0], np.cumsum([len(s) for s in streamlines])])
tractogram = Tractogram(np.concatenate(streamlines), offsets, reference=diffusion)

selected = tractogram[passes_through(tractogram, region)]
total = tractogram.num_streamlines
print(f"streamlines through region 1: {selected.num_streamlines} of {total}")
assert selected.num_streamlines == 20
