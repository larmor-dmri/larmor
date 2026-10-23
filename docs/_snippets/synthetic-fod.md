:::{dropdown} The data on this page
The examples on this page run on a small synthetic FOD image, which the code below writes to the current folder.
`fod.nii.gz` holds SH coefficients up to degree 8 in the tournier basis, and `mask.nii.gz` covers the voxels of
its two bundles. The image has 20 x 20 x 4 voxels of 2 mm. One bundle runs along x and one along y, and where they
cross the FOD has both lobes. To work on your own data, use your files instead.

```python
import numpy as np

from larmor.io import save_volume
from larmor.sd import fit_sh
from larmor.types import Volume

n = 600
z = 1 - (2 * np.arange(n) + 1) / n
azimuth = np.pi * (3 - np.sqrt(5)) * np.arange(n)
radius = np.sqrt(1 - z**2)
sphere = np.stack([radius * np.cos(azimuth), radius * np.sin(azimuth), z], axis=1)


def fod(*fibres):
    """SH coefficients of a narrow lobe along each fibre."""
    cosines = sphere @ np.array(fibres).T
    return fit_sh(np.exp(-8 * (1 - cosines**2)).sum(axis=1), sphere, lmax=8)


x, y = np.eye(3)[0], np.eye(3)[1]
along_x = np.zeros((20, 20, 4), bool)
along_x[:, 8:12] = True
along_y = np.zeros((20, 20, 4), bool)
along_y[8:12, :] = True
coeffs = np.zeros((20, 20, 4, 45), np.float32)
coeffs[along_x] = fod(x)
coeffs[along_y] = fod(y)
coeffs[along_x & along_y] = fod(x, y)

affine = np.diag([2.0, 2.0, 2.0, 1.0])
save_volume(Volume(coeffs, affine), "fod.nii.gz")
save_volume(Volume((along_x | along_y).astype(np.uint8), affine), "mask.nii.gz")
```
:::
