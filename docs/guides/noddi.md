---
myst:
  html_meta:
    description: "How to fit NODDI with Larmor, from Python and with lar noddi, to get maps of neurite density, orientation dispersion and free water."
---

# NODDI

Neurite orientation dispersion and density imaging (NODDI) separates two things that both lower the fractional
anisotropy of DTI: how many neurites a voxel holds, and how widely their directions spread. It models the signal
of each voxel with three kinds of water (Zhang et al. 2012). Water inside axons and dendrites moves only along
them, as sticks whose directions follow a Watson distribution about a mean direction; water between them is
hindered, more along that direction than across it; and free water, such as cerebrospinal fluid, diffuses
equally in every direction. The fit gives the neurite density v_ic, the orientation dispersion index OD, from
0 for aligned neurites to 1 for neurites in every direction, and the free-water fraction v_iso. NODDI needs at
least two non-zero shells. [Citing](../about/citing.md#models) lists the papers.

```{include} ../_snippets/synthetic-dwi.md
```

## Fitting the model

::::{tab-set}
:sync-group: interface

:::{tab-item} Python
:sync: python
```python
from larmor.dwi import NoddiModel
from larmor.io import load_dwi, load_volume, save_volume

dwi, gtab = load_dwi("dwi.nii.gz", "dwi.bval", "dwi.bvec")
fit = NoddiModel().fit(dwi, gtab, mask=load_volume("mask.nii.gz"))

save_volume(dwi.with_data(fit.od), "od.nii.gz")
print(f"OD in a bundle: {fit.od[0, 5, 0]:.2f}, where they cross: {fit.od[5, 5, 0]:.2f}")
```

The result, a {py:class}`~larmor.dwi.NoddiFit`, holds the neurite density `v_ic`, the orientation dispersion
index `od`, the Watson concentration `kappa` it comes from, the free-water fraction `v_iso` and the fitted mean
direction `mu`. All maps are zero outside the mask.
:::

:::{tab-item} Command line
:sync: cli
```bash
lar noddi dwi.nii.gz noddi --bvals dwi.bval --bvecs dwi.bvec -m mask.nii.gz
```

The command writes `noddi_v_ic.nii.gz`, `noddi_od.nii.gz`, `noddi_kappa.nii.gz`, `noddi_v_iso.nii.gz` and the
mean direction `noddi_mu.nii.gz`.
:::
::::

## How it is fitted

Each voxel is fitted in two steps. The first matches its signal with a non-negative mix of NODDI signals computed
in advance for a grid of neurite densities and concentrations about the voxel's DTI direction, which gives a
starting point quickly and without a search (Daducci et al. 2015). The second fits v_ic, kappa, v_iso and the
direction from there by maximum likelihood with Rician noise, with the model, the noise level from the b=0
images, and the two stages of the NODDI Matlab Toolbox: first the fractions alone, then all the parameters. The
diffusivities are fixed at the toolbox's in vivo values, 1.7e-3 mm^2/s along the neurites and 3.0e-3 mm^2/s for
free water.

## Options and caveats

- **Signal fractions.** v_ic and v_iso are fractions of the signal, not of the volume. They differ when the
  compartments have different T2 or proton density, as free water does.
- **Voxels of free water.** Where v_iso is high, little signal is left to measure the neurites, and v_ic is not
  reliable there; it often reaches 1. Read v_ic and OD where v_iso is low.
- **Voxels that are not fitted.** A voxel without a positive b=0 signal, or with fewer than six positive
  diffusion-weighted measurements, gets zeros. A voxel whose signal the first step explains with free water alone
  gets v_ic = 0 and OD = 0.
- **Starting direction.** By default the fit starts from the principal eigenvector of a DTI fit. From Python,
  `fit(..., mu=directions)` starts from directions of your own, one unit vector per voxel.
- **Mask and threads.** As for [DTI](dti.md#options).

## See also

- API: {py:class}`~larmor.dwi.NoddiModel`, {py:class}`~larmor.dwi.NoddiFit`.
- Command: [`lar noddi`](../cli.md#lar-noddi).

## References

- Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC. NODDI: Practical in vivo neurite orientation
  dispersion and density imaging of the human brain. NeuroImage 61 (2012) 1000-1016.
- Daducci A, Canales-Rodriguez EJ, Zhang H, Dyrby TB, Alexander DC, Thiran JP. Accelerated Microstructure Imaging
  via Convex Optimization (AMICO) from diffusion MRI data. NeuroImage 105 (2015) 32-44.
- NODDI Matlab Toolbox 1.05, [nitrc.org/projects/noddi_toolbox](https://www.nitrc.org/projects/noddi_toolbox).
