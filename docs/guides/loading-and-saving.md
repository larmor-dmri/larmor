---
myst:
  html_meta:
    description: "How to load and save images, diffusion images with their gradients, tractograms and surfaces with Larmor, from Python and with lar info and lar convert."
---

# Loading and saving data

Larmor reads an image into a {py:class}`~larmor.types.Volume`, a gradient table into a
{py:class}`~larmor.types.GradientTable`, a tractogram into a {py:class}`~larmor.types.Tractogram` and a surface into
a {py:class}`~larmor.types.Surface`. The file extension chooses the format. Whatever convention the file uses,
positions in Larmor are in world coordinates (RAS+, in mm). [File formats](../reference/file-formats.md)
lists every format, and [Conventions](../background/conventions.md#coordinates-and-spaces) explains the coordinates.

```{include} ../_snippets/synthetic-dwi.md
```

## Images

::::{tab-set}
:sync-group: interface

:::{tab-item} Python
:sync: python
```python
from larmor.io import load_header, load_volume, save_volume

mask = load_volume("mask.nii.gz")
print(mask.data.shape, mask.voxel_sizes, mask.orientation)
save_volume(mask, "mask.mif")

grid, shape = load_header("dwi.nii.gz")
print(shape)
```
:::

:::{tab-item} Command line
:sync: cli
```bash
lar info mask.nii.gz
lar convert mask.nii.gz mask.mif
```
:::
::::

Larmor reads and writes NIfTI (`.nii`, `.nii.gz`) and MRtrix (`.mif`, `.mif.gz`, `.mih`) images. A `Volume` holds
the values as stored in the file (`data`) and the affine that maps voxel indices to world coordinates (`affine`).
{py:meth}`~larmor.types.Volume.as_float` returns the values with the file's scaling applied, and
{py:func}`~larmor.io.load_header` reads only the grid and the shape, without the data. `lar convert --to-ras` also
reorients an image to its closest RAS orientation, which only permutes and flips the voxel axes.

A NIfTI file names the space its affine maps into (scanner, MNI or another template) with its sform and qform
codes. Larmor keeps these codes in `Volume.xform_codes` and writes them back when it saves the volume as NIfTI.
Maps made with {py:meth}`~larmor.types.Volume.with_data` keep the codes of their image, so a map of an MNI-space
image stays labelled MNI. An image resampled onto a reference grid ({py:func}`~larmor.registration.transform_volume`,
`lar transform`) takes the codes of the reference, and a volume made in memory is written as scanner space.

## A diffusion image with its gradients

{py:func}`~larmor.io.load_dwi` reads a diffusion image together with its gradient table. A NIfTI image needs its
bvals and bvecs files, while a MIF image can hold the table in its header and then needs none.

::::{tab-set}
:sync-group: interface

:::{tab-item} Python
:sync: python
```python
from larmor.io import load_dwi, save_volume

dwi, gtab = load_dwi("dwi.nii.gz", "dwi.bval", "dwi.bvec")
save_volume(dwi, "dwi.mif", gradients=gtab)
dwi, gtab = load_dwi("dwi.mif")
print(dwi.data.shape, gtab.shells)
```
:::

:::{tab-item} Command line
:sync: cli
```bash
lar convert dwi.nii.gz dwi.mif --bvals dwi.bval --bvecs dwi.bvec
lar info dwi.mif
```
:::
::::

Every `lar` command that takes a diffusion image has the options `--bvals` and `--bvecs`, which a MIF image with a
table does not need. [Gradient tables and shells](gradient-tables.md) covers the tables themselves.

## Tractograms

A tractogram stores the points of all its streamlines in one array, in world coordinates, together with the
offset at which each streamline starts. The example below builds three straight streamlines and saves them as a TCK
file:

```python
import numpy as np

from larmor.io import save_tractogram
from larmor.types import Tractogram

line = np.linspace([2.0, 0.0, 2.0], [22.0, 0.0, 2.0], 11)
points = np.concatenate([line + [0.0, y, 0.0] for y in (8.0, 10.0, 12.0)])
save_tractogram(Tractogram(points, np.array([0, 11, 22, 33])), "tracks.tck")
```

TCK and VTK files store world coordinates, while a TRK file stores its points in the voxel frame of an image, so
writing a TRK needs that image's grid. A VTK or `.fib` file can hold its points in RAS or in LPS, where x and y have
the opposite sign, and the file does not always say which. Larmor follows a `SPACE=RAS` or `SPACE=LPS` tag in the
file's title line and reads a file without one as RAS. When you know that such a file is LPS, load it with
`load_tractogram(path, space="LPS")` or `lar convert --vtk-space LPS`, and the points are converted to RAS.

::::{tab-set}
:sync-group: interface

:::{tab-item} Python
:sync: python
```python
from larmor.io import load_grid, load_tractogram, save_tractogram

tracts = load_tractogram("tracks.tck")
print(tracts.num_streamlines, tracts.num_points)
save_tractogram(tracts, "tracks.trk", reference=load_grid("dwi.nii.gz"))
save_tractogram(tracts, "tracks.vtk")
```
:::

:::{tab-item} Command line
:sync: cli
```bash
lar info tracks.tck
lar convert tracks.tck tracks.trk --reference dwi.nii.gz
lar convert tracks.tck tracks.vtk
```
:::
::::

For large files, `load_tractogram(path, max_streamlines=n, seed=0)` keeps a random subset of `n` streamlines, and
{py:func}`~larmor.io.iter_tractogram` reads an uncompressed TCK or TRK file in batches.

## Surfaces

{py:func}`~larmor.io.load_surface` and {py:func}`~larmor.io.save_surface` read and write GIfTI and FreeSurfer
surfaces, as shown in [Surfaces](surfaces.md).

## See also

- Gallery: [Reading and writing files](../gallery/files/file_io.rst), and
  [Reading MRIToolkit results from MATLAB files](../gallery/files/mritoolkit_files.rst).
- API: {py:mod}`larmor.io`, {py:mod}`larmor.types`.
- Commands: [`lar info`](../cli.md#lar-info), [`lar convert`](../cli.md#lar-convert).
