"""Reading MRIToolkit results from MATLAB files
============================================

This example writes a small synthetic DTI fit in MRIToolkit's layout with scipy,
reads it back with Larmor, refits the tensor from the same data, and reads a tract.

MRIToolkit and ExploreDTI, two MATLAB toolboxes for diffusion MRI, save their DTI and
DKI fits, FOD fits and tractography as MATLAB ``.mat`` files. ``read_matlab`` lists
and reads any such file, and the ``read_mritoolkit_*`` readers turn MRIToolkit's
results into Larmor's types, in world coordinates.
"""

import tempfile
from pathlib import Path

import numpy as np
from scipy.io import savemat

from larmor.dwi import DTIModel
from larmor.io import read_matlab, read_mritoolkit_fit, read_mritoolkit_tracts

# %%
# A synthetic fit
# ---------------
# The fit is on a grid of 8 x 10 x 6 voxels of 2 mm, and its tensor points along
# world x. MRIToolkit stores images with the first two NIfTI axes swapped and both
# reversed. It stores vectors and tensors in the frame of the NIfTI voxel axes j, i,
# k. With this affine that frame is (y, x, z) in world, so the fibre lies along the
# frame's second axis.

nx, ny, nz = 8, 10, 6
affine = np.diag([2.0, 2.0, 2.0, 1.0])
tensor = np.diag([0.3e-3, 1.7e-3, 0.3e-3])  # in the frame (j, i, k)

directions = np.array(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float
)
directions /= np.linalg.norm(directions, axis=1, keepdims=True)
bvalue = 1000.0
gx, gy, gz = directions.T
b_matrix = bvalue * np.column_stack(
    [gx * gx, 2 * gx * gy, 2 * gx * gz, gy * gy, 2 * gy * gz, gz * gz]
)
b_matrix = np.vstack([np.zeros(6), b_matrix])  # a b0 first
signal = 1000 * np.exp(
    -np.einsum("ni,ij,nj->n", directions, tensor, directions) * bvalue
)
signal = np.concatenate([[1000.0], signal])


def cell(arrays: list[np.ndarray]) -> np.ndarray:
    """A 1 x N MATLAB cell array holding the arrays, as scipy writes one."""
    result = np.empty((1, len(arrays)), dtype=object)
    for n, array in enumerate(arrays):
        result[0, n] = array
    return result


stored = (ny, nx, nz)  # the shape of an image in MRIToolkit's layout
elements = [tensor[i, j] for i, j in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]]
header = {
    "dime": {
        "dim": [4, nx, ny, nz, len(signal), 1, 1, 1],
        "pixdim": [1, 2, 2, 2, 0, 0, 0, 0],
    },
    "hist": {
        "sform_code": 1.0,
        "qform_code": 0.0,
        "srow_x": affine[0],
        "srow_y": affine[1],
        "srow_z": affine[2],
    },
}
folder = Path(tempfile.mkdtemp())
savemat(
    folder / "fit.mat",
    {
        "DT": cell([np.full(stored, e) for e in elements]),
        "DWI": cell([np.full(stored, s) for s in signal]),
        "b": b_matrix,
        "hdr": header,
    },
)

# %%
# Listing and reading
# -------------------
# ``read_matlab`` lists every variable, struct fields and cells included, with its
# MATLAB size and class, without reading the data.

for entry in read_matlab(folder / "fit.mat").entries()[:4]:
    print(entry.path, entry.shape, entry.matlab_class)

fit = read_mritoolkit_fit(folder / "fit.mat")
centre = (nx // 2, ny // 2, nz // 2)
print("Tensor in world coordinates (xx, xy, xz, yy, yz, zz):", fit.dt.data[centre])

# %%
# Refitting with Larmor
# ---------------------
# The DWI and its gradient table, both in world coordinates, give Larmor's own fit,
# whose principal direction is along world x as well.

own = DTIModel().fit(fit.dwi, fit.gradients, np.ones((nx, ny, nz), bool))
print(f"FA {own.fa[centre]:.3f}, principal direction {np.round(own.pe[centre], 3)}")

# %%
# A tract
# -------
# Tract points are stored in mm along the axes of MRIToolkit's layout; the reader
# places them on the fit's grid.

points = np.array([[10.0, 4.0, 6.0], [10.0, 8.0, 6.0], [10.0, 12.0, 6.0]])
savemat(folder / "tracts.mat", {"Tracts": cell([points])})
tracts = read_mritoolkit_tracts(folder / "tracts.mat", folder / "fit.mat")
print("Tract points in world mm:\n", tracts.points)
