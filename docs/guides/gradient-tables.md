---
myst:
  html_meta:
    description: "How Larmor reads and writes gradient tables (bvals and bvecs, MRtrix and b-matrix files), how it groups b-values into shells, and how to keep the volumes of one shell."
---

# Gradient tables and shells

A {py:class}`~larmor.types.GradientTable` holds the diffusion weighting of each volume of a diffusion image, in
the order of the volumes. Each volume has a b-value in s/mm^2 and a unit direction in world coordinates (RAS+).

Vectors that are not of unit length change the b-value. Each b-value is multiplied by the squared length of its
vector and the vector is made a unit vector, so a vector of length 0.5 at b=2800 becomes a unit vector at b=700.
A diffusion-weighted volume whose vector is shorter than 0.1 is refused as corrupt. When the scaling brings a
weighted volume down to the b=0 threshold or below, a warning names it, and it then counts as a b=0 volume.

```{include} ../_snippets/synthetic-dwi.md
```

## Reading and writing

FSL bvecs files store the directions in the frame of the image's voxel axes, with x flipped when the affine's
determinant is positive. Larmor calls this frame `"fsl"` and uses it by default. Reading or writing such a file
therefore needs the image it belongs to, whose affine turns the directions into world coordinates and back, so a
table read and written back gives the file it came from. For an image with shear, the converted directions are
rescaled to the length they had in the file, so unit vectors stay unit and the b-values do not change. MRIToolkit
files are converted in the same way.

Some bvecs files hold the directions in world coordinates instead; read them with `frame="world"`, since reading
them as FSL files gives wrong directions without any error. MRtrix gradient files (`.b`, one row `x y z b` per
volume) are always in world coordinates.

::::{tab-set}
:sync-group: interface

:::{tab-item} Python
:sync: python
```python
from larmor.io import read_bvals_bvecs, read_mrtrix_gradients, write_mrtrix_gradients

gtab = read_bvals_bvecs("dwi.bval", "dwi.bvec", ref_path="dwi.nii.gz")
write_mrtrix_gradients(gtab, "dwi.b")
same = read_mrtrix_gradients("dwi.b")
```
:::

:::{tab-item} Command line
:sync: cli
The commands read bvals and bvecs files with `--bvals` and `--bvecs`, and `--bvec-frame world` for b-vectors in
world coordinates. `lar convert` writes the table into the header of a MIF image, or to new bvals and bvecs files:

```bash
lar convert dwi.nii.gz dwi.mif --bvals dwi.bval --bvecs dwi.bvec
lar convert dwi.mif dwi_copy.nii.gz --bvals-out dwi_copy.bval --bvecs-out dwi_copy.bvec
```
:::
::::

{py:func}`~larmor.io.write_bvals_bvecs`, {py:func}`~larmor.io.read_b_matrix` and
{py:func}`~larmor.io.write_b_matrix` read and write the other formats, which [File formats](../reference/file-formats.md)
lists.

## Shells

Volumes at or below `b0_threshold` (50 s/mm^2 by default) count as b=0. Larmor groups the other b-values into
shells by sorting them and starting a new shell wherever two neighbours differ by more than `shell_epsilon`
(80 s/mm^2 by default), so b-values spread around a shell, such as 995, 1000 and 1005, count as one.

::::{tab-set}
:sync-group: interface

:::{tab-item} Python
:sync: python
```python
print(gtab.shells)
print(gtab.count_shells(), gtab.b0_mask.sum())
print(gtab.shell_index[:5])
```

`shells` gives the b-value of each shell, the mean b-value of its volumes, and `shell_index` gives the shell of
each volume, with -1 for a b=0 volume.
:::

:::{tab-item} Command line
:sync: cli
`lar info` prints the number of b=0 volumes and the shells, with the number of volumes in each, from the table in
a MIF image's header:

```bash
lar info dwi.mif
```
:::
::::

## Keeping one shell

To fit a model to one shell of a multi-shell acquisition, keep its volumes and the b=0 volumes.
{py:func}`~larmor.dwi.shell_mask` selects the volumes whose b-value is within a tolerance of the one you give
(100 s/mm^2 by default). The example below keeps the b=0 volumes and the b=1000 shell:

```python
import numpy as np

from larmor.dwi import shell_mask
from larmor.io import load_dwi, save_volume
from larmor.types import GradientTable

dwi, gtab = load_dwi("dwi.nii.gz", "dwi.bval", "dwi.bvec")
keep = gtab.b0_mask | shell_mask(gtab.bvals, 1000)
b1000 = dwi.with_data(dwi.data[..., keep])
b1000_gtab = GradientTable(gtab.bvals[keep], gtab.bvecs[keep])
save_volume(b1000, "dwi_b1000.mif", gradients=b1000_gtab)
```

## Correcting a table

Sometimes the directions do not fit the data, for example when the scanner's export reversed the sign of y.
{py:meth}`~larmor.types.GradientTable.flip` then negates one component of every direction, and
{py:meth}`~larmor.types.GradientTable.permute` reorders the components. Because the table is in world coordinates,
reorienting an image never calls for these corrections. To check a table, look at the direction colours of a DTI
fit (see [DTI](dti.md)): the corpus callosum should run left to right.

## See also

- Background: [Conventions](../background/conventions.md#gradient-tables).
- Gallery: [Reading and writing files](../gallery/files/file_io.rst).
- API: {py:class}`~larmor.types.GradientTable`, {py:func}`~larmor.io.load_dwi`,
  {py:func}`~larmor.io.read_bvals_bvecs`, {py:func}`~larmor.dwi.shell_mask`.
