---
myst:
  html_meta:
    description: "The coordinates, gradient table frames, spherical harmonic bases, tractogram files, array axes and units that Larmor uses, and how its viewer displays images."
---

# Conventions

This page lists the coordinates, units, axis orders and bases that Larmor uses, so that you can check your data
against them. Each section names the functions and classes that follow its conventions.

## Coordinates and spaces

| Term | Meaning |
| --- | --- |
| Voxel coordinates | Indices (i, j, k) along the image's voxel axes, counted from 0. Index (i, j, k) is the centre of that voxel. |
| World coordinates | Scanner coordinates in mm, RAS+: x increases to the subject's right, y to anterior, z to superior. |
| Affine | The (4, 4) float64 matrix that maps voxel indices to world coordinates. Its last row is [0, 0, 0, 1]. |
| Orientation | For each voxel axis, the world direction it is closest to: `("R", "A", "S")` means i increases to the right, j to anterior, k to superior. |

Every {py:class}`~larmor.types.Volume` and {py:class}`~larmor.types.Grid` carries its affine; the voxel sizes are
the lengths of its first three columns. The points of a {py:class}`~larmor.types.Tractogram`, the positions of
{py:class}`~larmor.tractography.Seeds`, the directions of a {py:class}`~larmor.types.GradientTable` and the
viewer's cursor are all in world coordinates, whatever the file's own convention.

**Reading the affine.** {py:func}`~larmor.io.formats.read_nifti` takes the sform when `sform_code > 0`, otherwise
the qform when `qform_code > 0`, and otherwise `diag(pixdim[1], pixdim[2], pixdim[3], 1)`. In the qform, a negative
`qfac` in `pixdim[0]` flips the third axis, and the voxel sizes `pixdim[1..3]` are taken as positive, with a
warning when one is negative. The two codes are kept in `Volume.xform_codes`. `xyzt_units` is ignored, and lengths
are always taken as mm. {py:func}`~larmor.io.formats.read_mif` builds the affine from the header's `transform` and
`vox`, and returns the axes in the order of `dim` whatever their order and direction on disk (the `layout`).

**Writing the affine.** {py:func}`~larmor.io.formats.write_nifti` writes the affine exactly into the sform and its
closest rotation into the qform. It writes the codes in `Volume.xform_codes`, which are those of the file the
volume was read from, those of the reference grid for a resampled image, or 1 (scanner) for a volume made in
memory. {py:func}`~larmor.io.formats.write_mif` writes `layout: +0,+1,+2` (x fastest) and the affine as `transform`
and `vox`.

**Orientation.** {py:attr}`Volume.orientation <larmor.types.Volume.orientation>` gives the orientation codes.
{py:meth}`~larmor.types.Volume.to_orientation` and {py:meth}`~larmor.types.Volume.to_closest_canonical` (or
`lar convert --to-ras`) only permute and flip voxel axes and update the affine, so every voxel keeps its world
position and nothing is resampled. Reorienting an image never changes its gradient table, which is in world
coordinates.

**Masks.** A mask passed as a `Volume` to {py:meth}`DTIModel.fit <larmor.dwi.DTIModel.fit>`,
{py:meth}`DKIModel.fit <larmor.dwi.DKIModel.fit>`, {py:meth}`SDModel.fit <larmor.sd.SDModel.fit>`,
{py:func}`~larmor.sd.estimate_wm_rf` or {py:func}`~larmor.dwi.denoise_mppca` must be on the image's grid. A mask
stored in another orientation of that grid (axes permuted or flipped, the same voxels in the world) is reoriented
to the image's, with a warning, and a mask on another grid is refused. A mask passed as an array must have the
image's spatial shape. Nonzero values are inside.

## Gradient tables

A {py:class}`~larmor.types.GradientTable` holds one b-value and one direction per volume, in the order of the
volumes.

| Item | Convention |
| --- | --- |
| b-values | s/mm^2 |
| Directions | Unit vectors in world coordinates (RAS+) |
| Vectors not of unit length | Made unit length, with b multiplied by the squared length: 0.5 at b=2800 becomes 1 at b=700. A vector shorter than 0.1 on a volume above the b=0 threshold is refused, and a volume that the scaling brings to the threshold or below is reported |
| b=0 volumes | b at or below `b0_threshold`, 50 s/mm^2 by default |
| Shells | The other b-values, sorted; a gap larger than `shell_epsilon` (80 s/mm^2 by default) starts a new shell. A shell's b-value is the mean of its volumes |
| Selecting a shell | {py:func}`~larmor.dwi.shell_mask` keeps the b-values within `tol` of the shell's (100 s/mm^2 by default) |
| b-matrix | One row per volume: b * (gx^2, 2 gx gy, 2 gx gz, gy^2, 2 gy gz, gz^2), in world coordinates |

Files store the directions in different frames, and the `frame` argument of
{py:func}`~larmor.io.read_bvals_bvecs`, {py:func}`~larmor.io.write_bvals_bvecs` and {py:func}`~larmor.io.load_dwi`
(`--bvec-frame` on the command line) says which one a file uses:

| File | Frame | Python |
| --- | --- | --- |
| FSL bvecs, `frame="fsl"` (the default) | The image's voxel axes (the affine's columns divided by the voxel sizes), with x negated when their determinant is positive; each converted vector is rescaled to its length in the file, which matters only for an image with shear. Converting needs the image. | {py:func}`~larmor.io.read_bvals_bvecs`, {py:func}`~larmor.io.write_bvals_bvecs` |
| bvecs, `frame="world"` | World coordinates, nothing converted | The same, with `frame="world"` |
| MRtrix3 gradient file (`.b`), MIF `dw_scheme` | World coordinates, rows `x y z b` | {py:func}`~larmor.io.read_mrtrix_gradients`, {py:func}`~larmor.io.write_mrtrix_gradients`, {py:func}`~larmor.io.formats.read_mif_gradients` |

A bvecs file is read as (3, N) or (N, 3) and written as (3, N); bvals are written as one row. A b-matrix does not
store the sign of a direction, so {py:func}`~larmor.io.read_b_matrix` returns each with its first non-zero
component positive.

See [Gradient tables and shells](../guides/gradient-tables.md).

## Spherical harmonics

FODs are stored as real spherical harmonic (SH) coefficients along the last axis of a 4D image, in world
coordinates like the gradient table. A direction is given by theta, its angle from +z, and phi, its angle around z
from +x.

| Item | Convention |
| --- | --- |
| Bases | `"tournier"` (the default), `"descoteaux"`, `"leemans"` |
| Coefficient order | By degree l, then by order m from -l to l |
| Symmetric FOD | Even degrees only: (lmax + 1)(lmax + 2) / 2 coefficients (1, 6, 15, 28, 45 for lmax 0 to 8) |
| Asymmetric FOD | All degrees: (lmax + 1)^2 coefficients (4, 9, 16, 25, ...) |
| Telling them apart | By the number of coefficients ({py:func}`~larmor.sd.sh_layout`); a count that fits both is symmetric |
| Normalisation | Orthonormal on the sphere. N(l, m) = sqrt((2l + 1) / (4 pi) * (l - \|m\|)! / (l + \|m\|)!), and the associated Legendre function P(l, \|m\|) includes the Condon-Shortley phase |
| FOD scale | The integral over the sphere, c00 * sqrt(4 pi), is the tissue's signal fraction relative to S0 |
| lmax of a fit | Even and at least 2, 8 by default; lowered by 2 until the highest shell has at least as many volumes as coefficients |

The three bases are defined below in terms of the normalised Legendre part of each harmonic,
Y(l, m) = N(l, m) P(l, |m|)(cos theta):

| Basis | m < 0 | m = 0 | m > 0 |
| --- | --- | --- | --- |
| `tournier` | sqrt(2) Y(l, m) sin(\|m\| phi) | Y(l, 0) | sqrt(2) Y(l, m) cos(m phi) |
| `descoteaux` | sqrt(2) (-1)^m Y(l, m) cos(\|m\| phi) | Y(l, 0) | sqrt(2) Y(l, m) sin(m phi) |
| `leemans` | `tournier` with phi replaced by phi - pi/2 | `tournier` | `tournier` with phi replaced by phi - pi/2 |

An image file does not record its basis, so the basis must be given whenever an FOD is read: with `basis=` in
Python, with `--basis` on the command line, and in the FOD's options when the viewer loads it.
{py:class}`~larmor.sd.SDModel` writes symmetric FODs; {py:func}`~larmor.sd.find_peaks`, the
{py:class}`~larmor.tractography.Tracker`, `lar sh-convert` and the viewer read both symmetric and asymmetric ones.
{py:func}`~larmor.sd.convert_sh_basis` converts between the available bases, and {py:func}`~larmor.sd.fit_sh` fits
coefficients to amplitudes on a set of directions.

The peaks of a symmetric FOD have z >= 0, while those of an asymmetric FOD keep their sign. `lar peaks` writes
each peak as its unit direction times its amplitude, with three values per peak along the last axis.

See [FODs](../guides/fods.md) and [Peaks](../guides/peaks.md).

## Tractograms

A {py:class}`~larmor.types.Tractogram` holds the points of all streamlines as one (P, 3) float32 array in world mm,
with (S + 1,) int64 offsets, so that streamline s is `points[offsets[s]:offsets[s + 1]]`. Values per point are in
`point_data` (first axis P) and values per streamline in `streamline_data` (first axis S).

| File | Points on disk | Python |
| --- | --- | --- |
| TCK | World mm, float32, a NaN row after each streamline and an Inf row at the end. No grid, no values. | {py:func}`~larmor.io.formats.read_tck`, {py:func}`~larmor.io.formats.write_tck` |
| TRK | TrackVis voxel mm: mm from the corner of the first voxel, along the axes of the header's `voxel_order`. The header's `vox_to_ras`, `dim` and `voxel_size` convert them to world mm. Scalars become `point_data` and properties `streamline_data`. | {py:func}`~larmor.io.formats.read_trk`, {py:func}`~larmor.io.formats.write_trk` |
| VTK, `.fib` | World mm, in RAS or LPS: the format does not fix which, and the reader converts LPS to RAS. The space comes from the `space` argument (`--vtk-space` in `lar convert`) when it is given, else from a `SPACE=RAS` or `SPACE=LPS` tag in the title line, else it is taken as RAS. The writer writes RAS with the tag `SPACE=RAS`. `POINT_DATA` and `CELL_DATA` arrays become `point_data` and `streamline_data`. | {py:func}`~larmor.io.formats.read_vtk`, {py:func}`~larmor.io.formats.write_vtk` |
| TSF | One value per point, in the order of the points, a NaN after each streamline. Paired with its TCK by the header's `timestamp`. | {py:func}`~larmor.io.read_tsf`, {py:func}`~larmor.io.write_tsf` |

A TRK file of version 2 is placed by its own `vox_to_ras`; one of version 1, or without `vox_to_ras`, needs a
reference grid. A TRK without `voxel_order` is read as LPS. Writing a TRK needs a reference grid (the `reference`
argument, or the tractogram's own), which sets `dim`, `voxel_size`, `vox_to_ras` and `voxel_order`.

**Point order.** The {py:class}`~larmor.tractography.Tracker` keeps every point it steps to inside the image and
the tracking mask. A step that would leave the image or the mask ends the streamline without adding its point, so
every point of a streamline lies inside both. A streamline tracked one way starts at its seed. One tracked both
ways (the default) runs from the end reached along the seed's first direction, through the seed, to the end reached
along the opposite direction. Seed files ({py:func}`~larmor.tractography.read_seeds`,
{py:func}`~larmor.tractography.write_seeds`) hold world mm.

See [Tractography](../guides/tractography.md) and [Loading and saving data](../guides/loading-and-saving.md).

## Arrays and units in Python

| Data | Shape | Units and type |
| --- | --- | --- |
| `Volume.data` | (X, Y, Z) or (X, Y, Z, N): voxel axes i, j, k, then the series (DWI volumes, SH coefficients) | As stored in the file, without the scaling |
| `Volume.slope`, `Volume.intercept` | Scalars | value = data * slope + intercept; a slope of 0, NaN or Inf means no scaling. {py:meth}`~larmor.types.Volume.as_float` applies it, as float32 by default, and refuses complex data, so denoising, Gibbs removal and the fits take magnitude images only |
| `Volume.affine`, `Grid.affine` | (4, 4) | float64, voxel indices to world mm |
| `GradientTable.bvals`, `GradientTable.bvecs` | (N,), (N, 3) | s/mm^2, and unit vectors in world coordinates; float64 |
| `Tractogram.points`, `Tractogram.offsets` | (P, 3), (S + 1,) | World mm as float32; int64 |
| DTI and DKI diffusivities | (X, Y, Z) | mm^2/s, float32 ({py:class}`~larmor.dwi.DTIFit`, {py:class}`~larmor.dwi.DKIFit`) |
| Diffusion tensor | (X, Y, Z, 6), in the order xx, xy, xz, yy, yz, zz | mm^2/s, float32, in world coordinates |
| Principal eigenvector, peak directions | (X, Y, Z, 3), (..., P, 3) | Unit vectors in world coordinates; an eigenvector's sign is arbitrary |
| FODs | (X, Y, Z, C) | SH coefficients, see [Spherical harmonics](#spherical-harmonics) |

## The viewer

| Item | Convention |
| --- | --- |
| Display | Neurological (RAS) by default: the subject's right on the right of the screen. Radiological (LAS) puts it on the left in the axial and coronal views; the sagittal view is the same in both. Set it with Display convention in Settings, or {py:meth}`Viewer.set_convention <larmor.viewer.Viewer.set_convention>` and {py:class}`~larmor.viewer.Convention`. |
| Direction colours | Red, green and blue are the absolute x, y and z components of a unit direction in world coordinates: red for left-right, green for anterior-posterior, blue for superior-inferior. This holds for direction maps, streamlines (the tangent at each point), peaks and FOD glyphs. A direction map's channels are clipped to [0, 1], and can be weighted by a map such as FA. |
| Crosshair readout | The cursor's position in world mm, and the main volume's name, nearest voxel (i, j, k, with the volume index of a 4D image) and value there, with the scaling applied, interpolated if the volume is shown interpolated. Outside the volume's grid there is no voxel or value. {py:meth}`Viewer.readout <larmor.viewer.Viewer.readout>` returns the same. |
| Slice numbers | Counted from 0, as voxel indices of the main volume. |

See [The 2D views](../viewer/2d-views.md) and [Settings](../viewer/settings.md).
