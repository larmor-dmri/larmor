---
myst:
  html_meta:
    description: "How to estimate fibre orientation distributions (FODs) with Larmor, from Python and with lar fod, using GRL, mFOD, CSD or MSMT-CSD, and how to convert FODs between SH bases."
---

# FODs

Spherical deconvolution estimates a fibre orientation distribution (FOD) in each voxel, which says how much of
the voxel's fibres run in each direction. It needs a response function, the signal of a single bundle of fibres
that all run the same way. The signal of a voxel is modelled as a sum of copies of this response, each turned
along one fibre direction and weighted by the share of the fibres that run that way. The fit finds the weights that
best reproduce the measured signal, and these weights are the FOD. Larmor stores FODs as spherical harmonic (SH)
coefficients. [Citing](../about/citing.md#spherical-deconvolution) lists the papers of the methods, and
[Conventions](../background/conventions.md#spherical-harmonics) gives the SH bases.

```{include} ../_snippets/synthetic-dwi.md
```

## Fitting FODs

The fit always needs a white matter response. Larmor estimates it from the voxels with FA > 0.7, or you can give
its eigenvalues yourself. Isotropic responses for grey matter and CSF are optional. With them, the fit is
multi-tissue and also gives the fraction of each voxel that each tissue fills.

::::{tab-set}
:sync-group: interface

:::{tab-item} Python
:sync: python
```python
from larmor.io import load_dwi, load_volume, save_volume
from larmor.sd import SDModel, build_response_functions

dwi, gtab = load_dwi("dwi.nii.gz", "dwi.bval", "dwi.bvec")
mask = load_volume("mask.nii.gz")
responses = build_response_functions(
    dwi, gtab, mask, gm_diffusivity=0.7e-3, csf_diffusivity=3e-3
)
fit = SDModel(responses, lmax=8).fit(dwi, gtab, mask=mask)

save_volume(dwi.with_data(fit.fods[0]), "fod.nii.gz")
save_volume(dwi.with_data(fit.tissue_fractions), "fractions.nii.gz")
print(f"lmax {fit.lmax}, basis {fit.basis}, shape {fit.fods[0].shape}")
```

`build_response_functions` returns the responses in the order white matter, grey matter, CSF. Without
`gm_diffusivity` and `csf_diffusivity` it returns the white matter response alone, for a single-tissue fit, and
`wm_eigen_vals=[1.7e-3, 0.2e-3, 0.2e-3]` gives the white matter response instead of estimating it.
:::

:::{tab-item} Command line
:sync: cli
```bash
lar fod dwi.nii.gz mask.nii.gz fod --bvals dwi.bval --bvecs dwi.bvec
```

The command writes the FOD to `fod.nii.gz` and the tissue fractions to `fod_tissue_fraction.nii.gz`. By default
it fits white matter, grey matter (isotropic, 0.7e-3 mm^2/s, with GRL) and CSF (3e-3 mm^2/s).
`--gm-rf none` and `--csf-rf none` leave grey matter and CSF out, and `--wm-rf 1.7e-3 0.2e-3 0.2e-3` gives the
white matter eigenvalues instead of estimating them.
:::
::::

The FODs are in world coordinates. A fit with even degrees up to `lmax` has (lmax + 1)(lmax + 2) / 2
coefficients, which is 28 for degree 6 and 45 for degree 8. The fit starts at `lmax` (8 by default) and lowers it
until there are no more coefficients than directions in the highest shell, so the data on this page, with 30
directions per shell, give degree 6.

## Methods and solvers

| Method | Responses | Default solver | Weight of the inner shells in a multi-tissue fit |
| --- | --- | --- | --- |
| GRL, the default (Guo et al. 2020) | one anisotropic (white matter), any isotropic | damped Richardson-Lucy, `DRL` (Dell'Acqua et al. 2010) | 0.2 |
| mFOD (De Luca et al. 2020) | one or more anisotropic, one FOD each, and any isotropic | non-negative least squares with a ridge penalty, `L2NNLS` | 1 |
| CSD (Tournier et al. 2007), with `CSDModel` | one, given by its zonal SH coefficients (`ZonalSHRF`) or estimated | iterated non-negativity constraint, `CSDSolver` | one shell only |
| MSMT-CSD (Jeurissen et al. 2014), with `MSMTCSDModel` | `ZonalSHRF` responses, one FOD per anisotropic one, or white matter, grey matter and CSF estimated | least squares with inequality constraints | 1 |

GRL and mFOD can use any of three solvers. {py:class}`~larmor.sd.DRL` is damped Richardson-Lucy,
{py:class}`~larmor.sd.L2NNLS` is non-negative least squares with a ridge penalty (`reg` 0.1 by default), and
{py:class}`~larmor.sd.Lasso` is an elastic net (`reg` 0.01 and `alpha` 0.9 by default; Tibshirani 1996; Zou and
Hastie 2005). With `alpha` 0.9, the elastic net puts 90% of its penalty on the L1 norm, which favours sparse FODs,
and 10% on the L2 norm. It fits no intercept, since the FOD model has no constant term. In a multi-tissue fit, the
elastic net fits the signal left after the isotropic tissues are subtracted, and the tissue fractions come from a
separate fit in which the FOD's signal is normalised, so the scale of the elastic-net FOD does not set them. A
stronger penalty can merge fibres that cross at small angles and drop the weaker fibre of a crossing, while a weaker
one lets more noise into the FOD. The solvers find the FOD on 300 fixed directions, where DRL and L2NNLS give
non-negative values, while the elastic net can give negative ones.

mFOD can also model grey matter with an anisotropic response, the NODDI response
({py:class}`~larmor.sd.NoddiRF`; Zhang et al. 2012), and then gives a second FOD for grey matter:

::::{tab-set}
:sync-group: interface

:::{tab-item} Python
:sync: python
```python
from larmor.sd import L2NNLS, SDMethod, merge_fods

responses = build_response_functions(
    dwi, gtab, mask, gm_noddi=True, csf_diffusivity=3e-3
)
model = SDModel(responses, method=SDMethod.MFOD, solver=L2NNLS(reg=0.1))
fit = model.fit(dwi, gtab, mask=mask)
print(len(fit.fods), "FODs")
fod = merge_fods(fit.fods, fit.tissue_fractions, "ws")  # one FOD to track
```
:::

:::{tab-item} Command line
:sync: cli
```bash
lar fod dwi.nii.gz mask.nii.gz mfod --bvals dwi.bval --bvecs dwi.bvec \
    --method mfod
```

With mFOD, grey matter defaults to the NODDI response (`--gm-rf noddi`) and the solver to L2NNLS. This merges the
two FODs into one, `mfod.nii.gz`, and writes `mfod_tissue_fraction.nii.gz`. With `--fod-merge none` it writes one
FOD per anisotropic response instead, `mfod_fod1.nii.gz` (white matter) and `mfod_fod2.nii.gz` (grey matter).
:::
::::

A tracker follows one FOD, so the two FODs are merged into one with {py:func}`~larmor.sd.merge_fods`
(`--fod-merge`). The weighted sum, `"ws"` (mFOD-WS, the default of `lar fod`), multiplies each FOD by its tissue
fraction and adds them, so the white matter FOD leads in the white matter and the grey matter FOD near the cortex.
`"majority"` keeps, in each voxel, the FOD of the anisotropic tissue with the largest fraction. The isotropic
fractions, such as the CSF's, count in neither.

`inner_shell_weight` (`--inner-shell-weight`) changes the weight of the measurements outside the highest shell in
a multi-tissue fit.

GRL and mFOD find the FOD's values on 300 fixed directions and then fit SH coefficients to those values. With the
L2NNLS and Lasso solvers, the l = 0 coefficient is then set to the sum of the values divided by sqrt(4 pi), so that
the FOD's integral does not depend on the direction of the fibre.

The FOD on the 300 directions is sharper than SH coefficients up to lmax 8 can hold, so the fitted FOD shows rings
around each lobe, about 51 degrees away from it and at about 6 to 8% of its height. Where two fibres cross, the
rings of the two fibres add up to about 12 to 14% of the main lobes, in four places around the crossing. A peak
search with a relative threshold of 0.1 would report these as four extra fibres. The FODs of GRL and mFOD therefore
use a relative threshold of 0.2: `fit.peak_relative_threshold` is 0.2 for them, and `lar peaks` uses 0.2 unless
`--fod-method` says the FOD comes from CSD or MSMT-CSD. In simulations this removed every ring peak and kept the
smaller fibre of 70/30 and 80/20 crossings, but a real fibre below a fifth of the largest one in its voxel is
dropped too. CSD and MSMT-CSD fit the SH coefficients directly, do not show these rings, and keep the threshold of
0.1.

## Constrained spherical deconvolution

CSD ({py:class}`~larmor.sd.CSDModel`) fits only the shell with the highest b-value, with one white matter
response. The response is given by its zonal SH coefficients (l = 0, 2, 4, ...) in the units of the signal. The
fit keeps the FOD from going negative by penalising it, on 300 directions, wherever it is negative. This constraint
lets the fit use a higher `lmax` than the shell's directions alone would support, so CSD keeps the `lmax` it is
given. The signal is not divided by the b=0 signal, so the FOD's integral is the voxel's signal relative to the
response's. {py:class}`~larmor.sd.CSDSolver` holds the solver's settings.

Without a response, the fit estimates one from the single-fibre voxels of the mask with
{py:func}`~larmor.sd.estimate_csd_rf`. It repeatedly deconvolves the voxels, keeps the 300 whose largest FOD peak is
large and whose second peak is small, and fits a new response to their signals, until the selected voxels no
longer change.

::::{tab-set}
:sync-group: interface

:::{tab-item} Python
:sync: python
```python
from larmor.sd import CSDModel, estimate_csd_rf

estimate = estimate_csd_rf(dwi, gtab, mask, num_voxels=50)
print(estimate.rf.coefficients, estimate.voxels.sum(), "voxels")
fit = CSDModel(estimate.rf, lmax=8).fit(dwi, gtab, mask=mask)
save_volume(dwi.with_data(fit.fods[0]), "csd.nii.gz")
```

`CSDModel(lmax=8)` without a response estimates it in `fit` with the defaults. To use a response file written by
MRtrix3's `dwi2response`, read it with {py:func}`~larmor.io.read_mrtrix_response` and give it the b-values of the
data's shells with {py:func}`~larmor.sd.build_csd_response`, as in
`CSDModel(build_csd_response(read_mrtrix_response("response.txt"), gtab))`.
:::

:::{tab-item} Command line
:sync: cli
```bash
lar fod dwi.nii.gz mask.nii.gz csd --bvals dwi.bval --bvecs dwi.bvec --method csd
```

This estimates the response in the mask and writes the FOD to `csd.nii.gz`. CSD gives no tissue fractions.
`--response response.txt` gives a response in MRtrix3's format instead.
:::
::::

## Multi-shell multi-tissue CSD

{py:class}`~larmor.sd.MSMTCSDModel` is multi-shell multi-tissue constrained spherical deconvolution (MSMT-CSD;
Jeurissen et al. 2014). It fits a white matter FOD and the amounts of isotropic grey matter and CSF to every
measurement of a voxel at once, b = 0 included. Each tissue has a response, a {py:class}`~larmor.sd.ZonalSHRF`,
given by its zonal SH coefficients on every shell in the units of the signal; the grey matter and CSF responses are
isotropic (lmax 0). The FOD is kept non-negative on 300 directions, and the isotropic amounts are kept
non-negative. Three tissues need at least three distinct b-values, b = 0 among them.

You can read the responses from files written by MRtrix3's `dwi2response` with
{py:func}`~larmor.io.read_mrtrix_response` (one row per shell), with the shells' b-values from
{py:func}`~larmor.sd.response_shells`. Or you can estimate them from the data and a brain mask with
{py:func}`~larmor.sd.estimate_tissue_responses`, which needs no tissue segmentation (Dhollander et al. 2016,
2019). `MSMTCSDModel()` without responses estimates all three in `fit`. The estimate first erodes the mask by
three voxels. The data on this page have only two slices, so the example below does not erode the mask, and since
the data hold a single isotropic tissue, it fits white matter and CSF only.

::::{tab-set}
:sync-group: interface

:::{tab-item} Python
:sync: python
```python
import numpy as np

from larmor.sd import MSMTCSDModel, estimate_tissue_responses

responses = estimate_tissue_responses(dwi, gtab, mask, erode=0)
model = MSMTCSDModel([responses.wm, responses.csf], lmax=8)
fit = model.fit(dwi, gtab, mask=mask)
print(fit.lmax, fit.tissue_fractions.shape)

np.savetxt("wm.txt", responses.wm.coefficients)
np.savetxt("csf.txt", responses.csf.coefficients)
```

`fit.fods[0]` is the white matter FOD. Its integral, c00 sqrt(4 pi), is the white matter's share of the signal
relative to its response. `fit.tissue_fractions` holds these shares for every tissue, white matter first,
normalised to sum to 1.
:::

:::{tab-item} Command line
:sync: cli
<!-- not run: needs whole-brain data or response files -->
```bash
lar fod dwi.nii.gz mask.nii.gz msmt --bvals dwi.bval --bvecs dwi.bvec --method msmt-csd
lar fod dwi.nii.gz mask.nii.gz msmt --bvals dwi.bval --bvecs dwi.bvec --method msmt-csd \
    --wm-response wm.txt --gm-response gm.txt --csf-response csf.txt --force
```

The first command estimates the three responses from the data and the mask, and the second reads them from files
such as those `dwi2response dhollander` writes. Both write the white matter FOD to `msmt.nii.gz` and the fractions
to `msmt_tissue_fraction.nii.gz`.
:::
::::

## SH bases

Larmor reads and writes three real SH bases, which hold the same functions in a different order, with different
signs or in a different frame. They are `"tournier"`, the default and the basis of MRtrix3, `"descoteaux"`, and
`"leemans"`, the basis of ExploreDTI and MRIToolkit. For Tournier and Descoteaux, Larmor supports the current
versions of the bases, not the legacy ones.

An image file does not record its basis, so you must give the basis whenever an FOD is read: with `basis=` in
Python, and with `--basis` on the command line (`lar fod`, `lar peaks`, `lar track`). A wrong basis gives wrong
directions without any error. The fit can write its FODs in another basis with `SDModel(..., basis="leemans")` or
`lar fod --basis leemans`, and existing FODs are converted as follows:

::::{tab-set}
:sync-group: interface

:::{tab-item} Python
:sync: python
```python
from larmor.io import load_volume, save_volume
from larmor.sd import convert_sh_basis, sh_layout

fod = load_volume("fod.nii.gz")
leemans = convert_sh_basis(fod.data, "tournier", "leemans")
save_volume(fod.with_data(leemans), "fod_leemans.nii.gz")
print(sh_layout(fod.data.shape[-1]))
```

{py:func}`~larmor.sd.sh_layout` returns the degree implied by the number of coefficients, and whether all degrees
are stored, as for an asymmetric FOD.
:::

:::{tab-item} Command line
:sync: cli
```bash
lar sh-convert fod.nii.gz fod_leemans.nii.gz --from tournier --to leemans
```
:::
::::

## See also

- Next: [Peaks](peaks.md) and [Tractography](tractography.md).
- Gallery: [Fitting FODs to a synthetic signal](../gallery/sd/fod_fit_synthetic.rst),
  [FODs, peaks and asymmetric FODs of a simulated brain](../gallery/sd/fods_ismrm.rst),
  [FOD glyphs in the viewer](../gallery/viewer/viewer_fod_glyphs.rst).
- API: {py:class}`~larmor.sd.SDModel`, {py:class}`~larmor.sd.SDFit`, {py:class}`~larmor.sd.MSMTCSDModel`,
  {py:func}`~larmor.sd.estimate_tissue_responses`,
  {py:func}`~larmor.sd.build_response_functions`, {py:func}`~larmor.sd.estimate_wm_rf`,
  {py:class}`~larmor.sd.CSDModel`, {py:class}`~larmor.sd.CSDSolver`, {py:func}`~larmor.sd.estimate_csd_rf`,
  {py:class}`~larmor.sd.ZonalSHRF`, {py:func}`~larmor.io.read_mrtrix_response`,
  {py:func}`~larmor.sd.convert_sh_basis`, {py:func}`~larmor.sd.fit_sh`.
- Commands: [`lar fod`](../cli.md#lar-fod), [`lar sh-convert`](../cli.md#lar-sh-convert).

## References

- Guo F, Leemans A, Viergever MA, Dell'Acqua F, De Luca A. Generalized Richardson-Lucy (GRL) for analyzing
  multi-shell diffusion MRI data. NeuroImage 218 (2020) 116948.
- De Luca A, Guo F, Froeling M, Leemans A. Spherical deconvolution with tissue-specific response functions and
  multi-shell diffusion MRI to estimate multiple fiber orientation distributions (mFODs). NeuroImage 222 (2020)
  117206.
- Dell'Acqua F, Scifo P, Rizzo G, et al. A modified damped Richardson-Lucy algorithm to reduce isotropic
  background effects in spherical deconvolution. NeuroImage 49 (2010) 1446-1458.
- Tibshirani R. Regression shrinkage and selection via the lasso. J R Stat Soc B 58 (1996) 267-288.
- Zou H, Hastie T. Regularization and variable selection via the elastic net. J R Stat Soc B 67 (2005) 301-320.
- Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC. NODDI: practical in vivo neurite orientation
  dispersion and density imaging of the human brain. NeuroImage 61 (2012) 1000-1016.
- Tournier JD, Calamante F, Connelly A. Robust determination of the fibre orientation distribution in diffusion
  MRI: non-negativity constrained super-resolved spherical deconvolution. NeuroImage 35 (2007) 1459-1472.
- Tournier JD, Calamante F, Connelly A. Determination of the appropriate b value and number of gradient directions
  for high-angular-resolution diffusion-weighted imaging. NMR Biomed 26 (2013) 1775-1786.
- Jeurissen B, Tournier JD, Dhollander T, Connelly A, Sijbers J. Multi-tissue constrained spherical deconvolution
  for improved analysis of multi-shell diffusion MRI data. NeuroImage 103 (2014) 411-426.
- Dhollander T, Raffelt D, Connelly A. Unsupervised 3-tissue response function estimation from single-shell or
  multi-shell diffusion MR data without a co-registered T1 image. ISMRM Workshop on Breaking the Barriers of
  Diffusion MRI (2016) 5.
- Dhollander T, Mito R, Raffelt D, Connelly A. Improved white matter response function estimation for 3-tissue
  constrained spherical deconvolution. Proc Intl Soc Mag Reson Med 27 (2019) 555.
- Lawson CL, Hanson RJ. Solving Least Squares Problems. Prentice-Hall, 1974.
- GRL, mFOD and their solvers are reimplementations of those in MRIToolkit (De Luca,
  [github.com/delucaal/MRIToolkit](https://github.com/delucaal/MRIToolkit)). CSD, the estimate of its response,
  their defaults and the response file format follow MRtrix3's `dwi2fod csd` and `dwi2response tournier`, and
  MSMT-CSD and its tissue responses follow `dwi2fod msmt_csd` and `dwi2response dhollander` (Tournier JD et al.
  MRtrix3: A fast, flexible and open software framework for medical image processing and visualisation.
  NeuroImage 202 (2019) 116137).
