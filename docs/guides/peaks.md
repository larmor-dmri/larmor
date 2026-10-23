---
myst:
  html_meta:
    description: "How to find the peaks of FODs with Larmor, from Python and with lar peaks, and how the thresholds, the number of peaks and asymmetric FODs affect them."
---

# Peaks

The peaks of an FOD are its local maxima, which give the directions of the fibres in a voxel, each with an
amplitude. {py:func}`~larmor.sd.find_peaks` finds them from the FOD's SH coefficients, sorts them largest first, and
drops those far smaller than the largest. [Conventions](../background/conventions.md#spherical-harmonics) gives the
SH bases and the layout of the peaks image.

```{include} ../_snippets/synthetic-fod.md
```

## Finding the peaks

::::{tab-set}
:sync-group: interface

:::{tab-item} Python
:sync: python
```python
import numpy as np

from larmor.io import load_volume, save_volume
from larmor.sd import find_peaks

fod = load_volume("fod.nii.gz")
mask = load_volume("mask.nii.gz").data > 0
peaks = find_peaks(fod.data, mask=mask, num_peaks=3, relative_threshold=0.1)

crossing = np.isfinite(peaks.amplitudes[10, 10, 0])
print(np.round(peaks.directions[10, 10, 0][crossing], 2))

vectors = peaks.directions * peaks.amplitudes[..., None]
save_volume(fod.with_data(vectors.reshape(*fod.data.shape[:3], -1)), "peaks.nii.gz")
```

The result, a {py:class}`~larmor.sd.Peaks`, holds the unit `directions`, of shape (X, Y, Z, P, 3), and their
`amplitudes`, of shape (X, Y, Z, P), largest first. Where a voxel has fewer than P peaks, the remaining entries are
NaN, and so are all entries outside the mask. The last two lines write the same peaks image as `lar peaks`.
:::

:::{tab-item} Command line
:sync: cli
```bash
lar peaks fod.nii.gz peaks.nii.gz --mask mask.nii.gz --num 3 --relative-threshold 0.1
```

The peaks image has 3 volumes per peak, largest peak first. They hold the x, y and z of the peak's direction
scaled by its amplitude, in world coordinates, and are NaN where a voxel has fewer peaks.
:::
::::

## Options

- **Number of peaks.** 3 by default (`num_peaks`, `--num`), or 6 for an asymmetric FOD.
- **Thresholds.** A peak is dropped when it is below `relative_threshold` times the voxel's largest peak, or below
  the absolute `threshold` (0 by default), and so is any maximum with an amplitude of 0 or below. The right
  relative threshold depends on the method that made the FOD. CSD and MSMT-CSD FODs use 0.1, the default of
  `find_peaks`. GRL and mFOD FODs need 0.2, because they show small side lobes that would otherwise be reported as
  fibres ([FODs](fods.md#methods-and-solvers) explains why); their fit gives this value as
  `fit.peak_relative_threshold`. `lar peaks` cannot tell from the file which method made the FOD, so you say it
  with `--fod-method`. Its default is GRL, as for `lar fod`, which gives 0.2, while CSD and MSMT-CSD give 0.1.
  `--relative-threshold` overrides both.
- **Basis.** `basis` (`--basis`) gives the FOD's SH basis, `"tournier"` by default; see
  [FODs](fods.md#sh-bases).
- **Symmetric and asymmetric FODs.** A symmetric FOD, with even degrees only, has a lobe at both ends of a fibre,
  so each fibre gives one peak, pointing up (z >= 0). An asymmetric FOD has all degrees, and is recognised by its
  number of coefficients. In an asymmetric FOD, a peak and its opposite are separate peaks that keep their sign, so
  a straight fibre gives two.
- **Close fibres.** Where two fibres cross at a small angle, their lobes merge into one peak.

## Viewing the peaks

<!-- not run: opens the viewer -->
```bash
lar bv -p peaks.nii.gz -f fod.nii.gz
```

The viewer draws each peak as a line through its voxel, over the FOD glyphs.

## See also

- Gallery: [Finding the peaks of FODs](../gallery/sd/peaks_synthetic.rst),
  [FODs, peaks and asymmetric FODs of a simulated brain](../gallery/sd/fods_ismrm.rst),
  [Peaks in the viewer](../gallery/viewer/viewer_peaks.rst).
- API: {py:func}`~larmor.sd.find_peaks`, {py:class}`~larmor.sd.Peaks`.
- Command: [`lar peaks`](../cli.md#lar-peaks).

## References

- The search follows `sh2peaks` of MRtrix3: Tournier JD et al. MRtrix3: A fast, flexible and open software
  framework for medical image processing and visualisation. NeuroImage 202 (2019) 116137.
