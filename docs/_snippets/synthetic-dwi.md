:::{dropdown} The data on this page
The examples on this page run on a small synthetic image, which the code below writes to the current folder as
`dwi.nii.gz`, with its gradient files `dwi.bval` and `dwi.bvec` and a mask, `mask.nii.gz`. The image has
12 x 12 x 2 voxels of 2 mm, two b=0 images and 30 directions at each of b=1000 and 2000 s/mm^2. Two bundles cross
in its middle, one along x and one along y, and the other voxels hold isotropic diffusion. To work on your own
data, use your files instead.

```python
import numpy as np

from larmor.io import save_volume, write_bvals_bvecs
from larmor.sd import AnisotropicRF
from larmor.types import GradientTable, Volume

rng = np.random.default_rng(0)
directions = rng.normal(size=(60, 3))
directions /= np.linalg.norm(directions, axis=1, keepdims=True)
bvals = np.r_[0, 0, [1000] * 30, [2000] * 30]
bvecs = np.r_[np.zeros((2, 3)), directions]
gtab = GradientTable(bvals, bvecs)

fibre = AnisotropicRF([1.7e-3, 0.3e-3, 0.3e-3], mean_kurtosis=0.5)
along_x, along_y = fibre.signal(bvals, gtab.bvecs, np.eye(3)[:2]).T
isotropic = np.exp(-bvals * 0.8e-3)

signal = np.broadcast_to(isotropic, (12, 12, 2, len(bvals))).copy()
signal[:, 4:8] = along_x
signal[4:8, :] = along_y
signal[4:8, 4:8] = (along_x + along_y) / 2
data = 1000 * signal + rng.normal(0, 10, signal.shape)

affine = np.diag([2.0, 2.0, 2.0, 1.0])
save_volume(Volume(data.astype(np.float32), affine), "dwi.nii.gz")
write_bvals_bvecs(gtab, "dwi.bval", "dwi.bvec", ref_path="dwi.nii.gz")
save_volume(Volume(np.ones((12, 12, 2), np.uint8), affine), "mask.nii.gz")
```
:::
