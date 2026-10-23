---
myst:
  html_meta:
    description: "How to fit diffusion kurtosis imaging (DKI) with Larmor, from Python and with lar dki, to get the MK, AK, RK and KA maps along with the DTI maps."
---

# DKI

Diffusion kurtosis imaging (DKI) measures how far the diffusion in each direction departs from a Gaussian, and
gives maps of the mean, axial and radial kurtosis besides the DTI maps. It adds a kurtosis tensor W to the
diffusion tensor D, and fits both to the log signal,
log(S) = log(S0) - b g^T D g + b^2 MD^2 W(g) / 6 (Jensen et al. 2005). Kurtosis shows as the curvature of the log
signal in b, so DKI needs at least two non-zero shells. [Citing](../about/citing.md#models) lists the papers.

```{include} ../_snippets/synthetic-dwi.md
```

## Fitting the tensors

::::{tab-set}
:sync-group: interface

:::{tab-item} Python
:sync: python
```python
from larmor.dwi import DKIModel
from larmor.io import load_dwi, load_volume, save_volume

dwi, gtab = load_dwi("dwi.nii.gz", "dwi.bval", "dwi.bvec")
fit = DKIModel().fit(dwi, gtab, mask=load_volume("mask.nii.gz"))

save_volume(dwi.with_data(fit.mk), "mk.nii.gz")
print(f"MK in the bundles: {fit.mk[0, 5, 0]:.2f}")
```

The result, a {py:class}`~larmor.dwi.DKIFit`, holds the same maps as a DTI fit (`fa`, `md`, `ad`, `rd`, `pe`,
`eigen_vals`, `dt`, `npd`). It also holds the mean, axial and radial kurtosis (`mk`, `ak`, `rk`), the kurtosis
anisotropy (`ka`), the kurtosis tensor `kt` with its 15 elements, and the fitted signal at b=0, `s0`. All maps are
zero outside the mask.
:::

:::{tab-item} Command line
:sync: cli
```bash
lar dki dwi.nii.gz dki --bvals dwi.bval --bvecs dwi.bvec -m mask.nii.gz
```

The command writes the DTI maps (`dki_fa.nii.gz`, `dki_md.nii.gz`, `dki_ad.nii.gz`, `dki_rd.nii.gz`,
`dki_pe.nii.gz`, `dki_npd.nii.gz`) and the kurtosis maps `dki_mk.nii.gz`, `dki_ak.nii.gz`, `dki_rk.nii.gz` and
`dki_ka.nii.gz`.
:::
::::

## Options

- **Fit method.** The tensors are fitted by weighted least squares by default, with weights from the predicted
  signal (Veraart et al. 2013). `DKIModel(fit_method="ols")` or `--fit ols` uses ordinary least squares instead.
- **Constraints.** By default the fit keeps the tensors physically plausible (Tabesh et al. 2011). In each
  direction n, the diffusivity D(n) and the kurtosis K(n) must satisfy D(n) >= 0, K(n) >= 0 and
  K(n) <= 3 / (b_max D(n)), where b_max is the largest b-value. Larmor imposes these constraints along 256
  directions spread over the half sphere, whereas the paper uses the acquired gradient directions. A voxel whose
  unconstrained fit already meets them keeps that fit. `DKIModel(constrained=False)` or `--unconstrained` fits
  without the constraints. Voxels with a diffusivity near 0, such as those at the edge of the mask, can still give
  very large kurtosis values.
- **S0.** Before the fit, each voxel's b=0 signal is checked against its diffusion-weighted signal, and implausible
  b=0 values are replaced from neighbouring voxels, as {py:func}`~larmor.dwi.estimate_s0` describes.
- **Tensors that are not positive definite.** As for [DTI](dti.md#options), these voxels are marked in `npd`, and
  `md` is a third of the trace of the fitted diffusion tensor, the same MD by which the kurtosis tensor is
  normalised. The kurtosis maps are 0 in these voxels, since K(n) is not defined in every direction.
- **Mask and threads.** As for [DTI](dti.md#options).

## See also

- Gallery: [Fitting DKI to a synthetic signal](../gallery/models/dki_synthetic_kurtosis.rst),
  [Fitting DKI to a real brain](../gallery/models/dki_real_data.rst).
- API: {py:class}`~larmor.dwi.DKIModel`, {py:class}`~larmor.dwi.DKIFit`.
- Command: [`lar dki`](../cli.md#lar-dki).

## References

- Jensen JH, Helpern JA, Ramani A, Lu H, Kaczynski K. Diffusional kurtosis imaging: the quantification of
  non-Gaussian water diffusion by means of magnetic resonance imaging. Magn Reson Med 53 (2005) 1432-1440.
- Tabesh A, Jensen JH, Ardekani BA, Helpern JA. Estimation of tensors and tensor-derived measures in diffusional
  kurtosis imaging. Magn Reson Med 65 (2011) 823-836.
- Veraart J, Sijbers J, Sunaert S, Leemans A, Jeurissen B. Weighted linear least squares estimation of diffusion
  MRI parameters: strengths, limitations, and pitfalls. NeuroImage 81 (2013) 335-346.
