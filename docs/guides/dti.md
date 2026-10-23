---
myst:
  html_meta:
    description: "How to fit diffusion tensor imaging (DTI) with Larmor, from Python and with lar dti, to get FA, MD, AD, RD and principal-direction maps."
---

# DTI

Diffusion tensor imaging (DTI) gives maps of the fractional anisotropy (FA), the mean, axial and radial
diffusivities (MD, AD, RD) and the principal direction of diffusion. It describes the diffusion in each voxel with
a 3x3 tensor D, fitted to the log of the signal S measured at b-value b along direction g,
log(S) = log(S0) - b g^T D g (Basser et al. 1994). The maps come from the tensor's eigenvalues and its first
eigenvector. [Citing](../about/citing.md#models) lists the papers, and
[Conventions](../background/conventions.md#arrays-and-units-in-python) gives the units and the tensor's element
order.

```{include} ../_snippets/synthetic-dwi.md
```

## Fitting the tensor

::::{tab-set}
:sync-group: interface

:::{tab-item} Python
:sync: python
```python
from larmor.dwi import DTIModel
from larmor.io import load_dwi, load_volume, save_volume

dwi, gtab = load_dwi("dwi.nii.gz", "dwi.bval", "dwi.bvec")
mask = load_volume("mask.nii.gz")
fit = DTIModel().fit(dwi, gtab, mask=mask)

save_volume(dwi.with_data(fit.fa), "fa.nii.gz")
save_volume(dwi.with_data(fit.md), "md.nii.gz")
print(f"FA in the bundles: {fit.fa[0, 5, 0]:.2f}, elsewhere: {fit.fa[0, 0, 0]:.2f}")
```

The result, a {py:class}`~larmor.dwi.DTIFit`, holds the maps as arrays on the image's grid, with zeros outside the
mask. Besides `fa`, `md`, `ad` and `rd`, it holds the principal eigenvector `pe`, the eigenvalues `eigen_vals`, the
tensor `dt` with its six elements in the order xx, xy, xz, yy, yz, zz, and the map `npd` described under
[Options](#options). Diffusivities are in mm^2/s, since b-values are in s/mm^2. The principal eigenvector and the
tensor are in world coordinates (RAS+), like the gradient table, not in the voxel frame of the image.
:::

:::{tab-item} Command line
:sync: cli
```bash
lar dti dwi.nii.gz dti --bvals dwi.bval --bvecs dwi.bvec -m mask.nii.gz
```

The second argument is the prefix of the output files. The command writes `dti_fa.nii.gz`, `dti_md.nii.gz`,
`dti_ad.nii.gz` and `dti_rd.nii.gz`, the principal eigenvector `dti_pe.nii.gz`, and `dti_npd.nii.gz`, which is 1
in the voxels whose tensor is not positive definite. Diffusivities are in mm^2/s, since b-values are in s/mm^2,
and the principal eigenvector is in world coordinates (RAS+), not in the voxel frame of the image.
:::
::::

## Options

- **Fit method.** The tensor is fitted by ordinary least squares by default. `DTIModel(fit_method="wls")` or
  `--fit wls` fits it by weighted least squares instead, with weights from the predicted signal (Veraart et al.
  2013).
- **Tensors that are not positive definite.** Noise or artefacts can give a tensor with an eigenvalue at or below
  zero. Larmor does not correct these voxels. Their `md` is still a third of the tensor's trace, and `fa`, `ad`,
  `rd` and `eigen_vals` come from the eigenvalues as fitted, with FA clipped to [0, 1]. The voxels are marked True
  in `npd` (1 in `dti_npd.nii.gz`) and their number is logged as a warning, so that you can leave them out of an
  analysis.
- **Mask.** Voxels outside the mask are not fitted. Without a mask, Larmor computes a brain mask from the mean b=0
  image, in the same way as {py:func}`~larmor.dwi.create_brain_mask` and `lar mask`. On a small crop this mask can
  come out empty, and the fit then stops with an error; pass a mask, or compute one with a smaller `morph`.
- **Shells.** The fit needs at least two distinct b-values. To fit one shell of a multi-shell acquisition, keep
  its volumes first, as in [Gradient tables and shells](gradient-tables.md#keeping-one-shell).
- **Threads.** The voxels are fitted in parallel on every core. `n_jobs` in Python and `--nt` on the command line
  set the number of threads, and the result does not depend on it.

## Viewing the maps

<!-- not run: opens the viewer -->
```bash
lar bv -v dti_fa.nii.gz -v dti_pe.nii.gz
```

The viewer can draw the principal eigenvector as direction colours weighted by FA, as shown in
[DTI colours in the viewer](../gallery/viewer/viewer_dti.rst).

## See also

- Gallery: [Fitting DTI to a synthetic tensor](../gallery/models/dti_synthetic_tensor.rst),
  [DTI of a simulated brain](../gallery/models/dti_ismrm.rst).
- API: {py:class}`~larmor.dwi.DTIModel`, {py:class}`~larmor.dwi.DTIFit`.
- Command: [`lar dti`](../cli.md#lar-dti).

## References

- Basser PJ, Mattiello J, LeBihan D. Estimation of the effective self-diffusion tensor from the NMR spin echo. J
  Magn Reson B 103 (1994) 247-254.
- Veraart J, Sijbers J, Sunaert S, Leemans A, Jeurissen B. Weighted linear least squares estimation of diffusion
  MRI parameters: strengths, limitations, and pitfalls. NeuroImage 81 (2013) 335-346.
