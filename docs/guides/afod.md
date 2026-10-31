---
myst:
  html_meta:
    description: "How to make asymmetric FODs from symmetric ones with Larmor, from Python and with lar afod, by filtering each FOD with its neighbouring voxels."
---

# Asymmetric FODs

The diffusion signal cannot tell one end of a fibre from the other, so the FODs that spherical deconvolution fits
are symmetric: a fibre gives a lobe at both of its ends. A bundle that continues on both sides of a voxel and one
that ends in it give the same FOD. Asymmetric FODs (aFODs) let the two halves of a lobe differ, which shows where
fibres end, bend or fan out. The information comes from the neighbouring voxels: each half of a lobe is compared
with the voxels it points at. Where they have fibres in the same direction, that half is kept; where they do not,
as at the end of a bundle, it shrinks. Where fibres continue on both sides, the FOD stays symmetric. Larmor makes
aFODs by filtering (Poirier and Descoteaux 2024).

```{include} ../_snippets/synthetic-fod.md
```

## Making asymmetric FODs

::::{tab-set}
:sync-group: interface

:::{tab-item} Python
:sync: python
```python
import numpy as np

from larmor.io import load_volume, save_volume
from larmor.sd import filter_afod, find_peaks

fod = load_volume("fod.nii.gz")
mask = load_volume("mask.nii.gz").data > 0
# Empty the bundle along x from i = 14, away from the crossing, so that it ends at i = 13.
data = fod.data.copy()
data[14:, 8:12] = 0

afod = filter_afod(data, mask, fod.affine)
save_volume(fod.with_data(afod), "afod.nii.gz")

peaks = find_peaks(afod, mask=mask, num_peaks=2, relative_threshold=0)
for name, voxel in (("middle", (4, 10, 2)), ("end", (13, 10, 2))):
    x, amplitude = peaks.directions[voxel][:, 0], peaks.amplitudes[voxel]
    print(f"{name}: {amplitude[x < 0][0]:.2f} back along -x, {amplitude[x > 0][0]:.2f} on along +x")
```

In the middle of the bundle, both halves of the lobe keep the same amplitude. At its end, the half that points
back along the bundle stays, and the half that points into the empty voxels shrinks. The result has the SH
coefficients of every degree, odd ones included, up to the input's lmax, in the input's basis. It is zero outside
the mask.
:::

:::{tab-item} Command line
:sync: cli
```bash
lar afod fod.nii.gz afod.nii.gz --mask mask.nii.gz
```

The output has the SH coefficients of every degree up to the input's lmax, in the input's basis (`--basis`).
:::
::::

## How it works

Each direction of a voxel's FOD gets a new amplitude: the weighted mean of the amplitude along that direction in
the voxel itself and in its 26 neighbours. A neighbour counts for the directions that point at it, and not for
those that point away, so the two halves of a lobe average different voxels. Its weight also falls with its
distance, as a Gaussian of width `sigma` in mm. The new amplitudes are then fitted with the SH of every degree.

## Options and caveats

- **Strength.** `sigma` (`--sigma`, 1.25 mm by default) sets how much the neighbours count compared with the voxel
  itself, which counts as 1: larger values give stronger asymmetry and more smoothing. Only the 26 neighbours are
  used, so values above about two voxel sizes make them count nearly equally.
- **Cone.** `sharpness` (`--sharpness`, 2 by default) is the exponent of the cosine between a direction and the
  direction to a neighbour. Larger values make each direction read only the neighbours closest to it.
- **Mask.** Neighbours outside the mask are not used, so the edge of the mask does not make FODs asymmetric. Keep
  the tissue where bundles end, such as grey matter, inside the mask. Without a mask, every voxel whose FOD is not
  all zero is used.
- **Basis.** `basis` (`--basis`) is the SH basis of the input, `"tournier"` by default; the output is in the same
  basis.
- **Input.** The FOD must be symmetric. `lar afod` refuses an FOD that already has every degree, and
  {py:func}`~larmor.sd.filter_afod` returns it unchanged.
- **What it can see.** The asymmetry comes from one voxel around each voxel. A fibre that turns within a voxel, as
  where fibres enter the cortex, does not show.

## Using asymmetric FODs

`lar peaks` and {py:func}`~larmor.sd.find_peaks` read an aFOD by its number of coefficients and give the two halves
of a lobe as separate peaks ([Peaks](peaks.md#options)). `lar track` reads it too, and follows its amplitude in the
direction a streamline is going, so the half of a lobe that points back does not count.

## See also

- Guides: [FODs](fods.md), [Peaks](peaks.md), [Tractography](tractography.md).
- API: {py:func}`~larmor.sd.filter_afod`.
- Command: [`lar afod`](../cli.md#lar-afod).

## References

- Poirier C, Descoteaux M. A unified filtering method for estimating asymmetric orientation distribution
  functions. NeuroImage 287 (2024) 120516. Its filters are in scilpy, as `scil_sh_to_aodf`.
