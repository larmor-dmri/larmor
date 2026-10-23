"""Whole-brain tractography of a simulated brain
=============================================

This example tracks streamlines through the whole simulated brain of the ISMRM 2015
Tractography Challenge: it fits FODs, seeds every voxel of the white matter, follows
the FODs with the deterministic algorithm, and shows the streamlines as lit tubes in
the 3D view, then with three slices of the T1-like image.

The data is simulated: diffusion-weighted images at 2 mm with 32 directions at
b=1000 s/mm^2, and a T1-like image, from the challenge's Zenodo record
(https://doi.org/10.5281/zenodo.572345, CC BY 4.0). Please cite: Maier-Hein KH et al.
The challenge of mapping the human connectome based on diffusion tractography.
Nature Communications 8 (2017) 1349.
"""

# needs: example-data (the tests run it only when LARMOR_DATA_DIR is set)

import numpy as np

from larmor.data import fetch_ismrm2015
from larmor.dwi import DTIModel, create_brain_mask, mean_b0
from larmor.io import load_dwi
from larmor.sd import SDModel, build_response_functions
from larmor.tractography import Tracker, seeds_per_voxel, select_length
from larmor.viewer import (
    CameraPreset,
    Category,
    OptionName,
    TractStyle,
    Viewer,
    ViewKind,
)

# %%
# FODs and seeds
# --------------
# The FODs are fitted up to degree 6, the most that 32 directions support. The
# seeds are one random point in each voxel whose FA is above 0.25, a simple white
# matter mask.

files = fetch_ismrm2015()
dwi, gtab = load_dwi(files["dwi"], files["bvals"], files["bvecs"])
mask = create_brain_mask(mean_b0(dwi, gtab))
responses = build_response_functions(dwi, gtab, mask)
fit = SDModel(responses, lmax=6).fit(dwi, gtab, mask=mask)
fod = dwi.with_data(fit.fods[0].astype(np.float32))
fa = DTIModel().fit(dwi, gtab, mask=mask).fa
white_matter = dwi.with_data((fa > 0.25).astype(np.uint8))
seeds = seeds_per_voxel(white_matter, 1, rng=0)

# %%
# Tracking
# --------
# With no output file, the streamlines stay in memory and come back in the run's
# record; a larger run writes them to a TCK or TRK file batch by batch. Tracking
# stops where the FOD's amplitude falls below the cutoff, chosen from the data, or
# where the streamline leaves the brain mask. Streamlines shorter than 40 mm are
# dropped.

tracker = Tracker("det", min_length=40.0)
run = tracker.track(fod, None, seeds, mask=dwi.with_data(mask.astype(np.uint8)), seed=0)
assert run.tractogram is not None
print(f"{run.num_streamlines} streamlines from {run.num_seeds} seeds")

# %%
# Tubes in 3D
# -----------
# Streamlines are drawn as lit tubes, coloured by direction, here seen from the
# front and a little from the side. The screenshots hide the text, so they show
# only the data.

viewer = Viewer()
t1 = viewer.load(files["t1"], Category.VOLUMES)
tracts = viewer.add(run.tractogram, Category.TRACTOGRAMS, name="whole brain")
viewer.set(t1.id, OptionName.SLICES_3D, False)
viewer.set(t1.id, OptionName.OPACITY_3D, 0.0)
viewer.set(tracts.id, OptionName.STYLE, TractStyle.TUBES)
viewer.set(tracts.id, OptionName.SUBSET, 30.0)
viewer.set_camera(ViewKind.THREE_D, CameraPreset.CORONAL)
viewer.rotate_camera(30.0, 15.0)
viewer.zoom(ViewKind.THREE_D, 1.4)
viewer.screenshot(
    ViewKind.THREE_D, "tracking_ismrm_tubes.png", size=(900, 600), overlay=False
)

# %%
# With the T1-like image
# ----------------------
# The 3D view also shows the three slices through the cursor. With the cursor near a
# corner of the image, the sagittal and coronal slices lie at its edges, out of the
# way, and the axial slice lies below most of the brain, a floor under the
# streamlines. Here a tenth of the streamlines of 60 mm or more are drawn as thin
# lines, seen from above and in front.

shape = np.array(dwi.data.shape[:3])
centre = dwi.affine[:3, :3] @ ((shape - 1) / 2) + dwi.affine[:3, 3]
long = run.tractogram[select_length(run.tractogram, min_length=60.0)]
viewer.set(tracts.id, OptionName.VISIBLE, False)
lines = viewer.add(long, Category.TRACTOGRAMS, name="60 mm or more")
viewer.set(lines.id, OptionName.SUBSET, 10.0)
viewer.move_cursor(tuple(centre + [-88.0, -106.0, -22.0]))
viewer.toggle_crosshair()
viewer.set(t1.id, OptionName.SLICES_3D, True)
viewer.set(t1.id, OptionName.MINIMUM, 100.0)
viewer.set(t1.id, OptionName.MAXIMUM, 700.0)
viewer.set_camera(ViewKind.THREE_D, CameraPreset.CORONAL)
viewer.rotate_camera(40.0, -30.0)
viewer.screenshot(
    ViewKind.THREE_D, "tracking_ismrm_slices.png", size=(900, 600), overlay=False
)
