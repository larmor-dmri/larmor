---
myst:
  html_meta:
    description: "How to track streamlines through FODs with Larmor, from Python and with lar track, deterministically or probabilistically, and how seeding, stopping and the run seed work."
---

# Tractography

Tractography reconstructs fibre pathways as streamlines by following the fibre directions of the FODs, step by
step, from seed points. {py:class}`~larmor.tractography.Tracker` has two algorithms. The deterministic one, `"det"`,
moves along the FOD peak nearest the current direction, and the probabilistic one, `"prob"`, draws each direction at
random according to the FOD. By default each streamline is tracked both ways from its seed.
[Citing](../about/citing.md#tractography-and-analysis) lists the papers, and
[Conventions](../background/conventions.md#tractograms) gives the point order and the file formats.

```{include} ../_snippets/synthetic-fod.md
```

## Tracking

::::{tab-set}
:sync-group: interface

:::{tab-item} Python
:sync: python
```python
from larmor.io import load_volume
from larmor.tractography import MaskSeeds, Tracker

fod = load_volume("fod.nii.gz")
mask = load_volume("mask.nii.gz")
tracker = Tracker("det")
run = tracker.track(
    fod, "tracks.tck", MaskSeeds(mask), num_streamlines=200, mask=mask, seed=0
)
print(f"{run.num_streamlines} streamlines from {run.num_seeds} seeds")
```

{py:meth}`~larmor.tractography.Tracker.track` writes the tractogram batch by batch and returns a
{py:class}`~larmor.tractography.TractographyRun`, a record of the run with the parameters used and the number of
seeds and streamlines. `num_rejected` counts the seeds that gave no streamline, by reason. A seed counts as
`no_start` when it lies outside the tracking mask or no direction above the cutoff was found there, and as
`too_short` when its streamline is shorter than the minimum length. With `None` as the output path, nothing is
written and the streamlines are kept in `run.tractogram` instead, which suits small runs only.

Take care when the same fractional map serves as seed image and tracking mask. Seeds are drawn in every voxel above
0, but a point is inside the tracking mask only where its nearest voxel is at least 0.5, so seeds in voxels between
0 and 0.5 count as `no_start`. Both images are compared on their scaled values, after the file's slope and
intercept are applied.
:::

:::{tab-item} Command line
:sync: cli
```bash
lar track fod.nii.gz tracks.tck --seed-image mask.nii.gz --select 200 --mask mask.nii.gz --seed 0
```

The progress bar counts the seeds tracked and those rejected. `--json run.json` saves the record of the run, with
every parameter used, the counts and the software versions.
:::
::::

## Seeding

| Seeds | Python | Command line |
| --- | --- | --- |
| Random positions in a mask's voxels, until a number of streamlines are kept | `MaskSeeds(mask)` with `num_streamlines` | `--seed-image mask.nii.gz --select 200` |
| N random positions in every voxel of a mask | `seeds_per_voxel(mask, n, rng=0)` | `--seed-random-per-voxel mask.nii.gz 2` |
| Seeds saved from an earlier run | `read_seeds("seeds.txt")` | `--seed-file seeds.txt` |

Fixed seeds ({py:class}`~larmor.tractography.Seeds`) are all tracked unless `num_streamlines` is given.
`output_seeds=` or `--output-seeds` saves the seeds of the kept streamlines in output order, and tracking from that
file again with the same run seed gives the same streamlines.

::::{tab-set}
:sync-group: interface

:::{tab-item} Python
:sync: python
```python
from larmor.tractography import seeds_per_voxel

seeds = seeds_per_voxel(mask, 2, rng=0)
run = Tracker("prob").track(fod, "prob.tck", seeds, mask=mask, seed=0)
```
:::

:::{tab-item} Command line
:sync: cli
```bash
lar track fod.nii.gz prob.tck --algorithm prob \
    --seed-random-per-voxel mask.nii.gz 2 --mask mask.nii.gz --seed 0
```
:::
::::

## Stopping

A streamline stops when one of the following happens.

- It leaves the tracking mask (`mask=`, `--mask`) or the image. The point outside is not kept, so the streamline
  ends at its last point inside.
- The FOD amplitude falls below the cutoff (`cutoff`, `--cutoff`). The default, `"auto"`, is 0.2 times the mean
  largest FOD peak in up to 5000 voxels drawn inside the tracking mask, or anywhere in the image without a mask.
- It turns by more than the maximum angle between steps (`max_angle`, `--angle`), whose defaults are given below.
- It reaches the maximum length (`max_length`, `--max-length`), 100 times the voxel size by default.

Together with the step size, the angle limit sets how tight a curve a streamline can follow, since the smallest
radius of curvature is step / (2 sin(angle / 2)). The same angle therefore allows a different curvature at a
different step size. When no angle is given, `det` sets it from the step size so that the smallest radius of
curvature is 0.65 times the voxel size. The angle is then 2 asin(step / (1.3 voxel)), about 8.8 degrees at the
default step of 0.1 times the voxel size, and a step longer than 1.3 voxels has no angle limit. `prob` uses 15
degrees, which at the default step allows a radius of about 0.38 times the voxel size. The run logs the smallest
radius its angle allows and keeps it in the run record as `min_radius`, in mm.

A point is inside the tracking mask when the mask's voxel nearest to it has a value of at least 0.5, so the mask's
boundary runs along the faces of its voxels. Seeding and the regions of `lar tract-edit` use the same rule, so a
seed drawn in a voxel of a mask that is also the tracking mask always starts inside it, even in a structure one
voxel thick.

Streamlines shorter than `min_length` (`--min-length`, 5 times the voxel size by default) are rejected. The step
size (`step_size`, `--step-size`) is 0.1 times the voxel size by default, and `bidirectional=False`
(`--unidirectional`) tracks each seed one way only. The reason each end of a streamline stopped is kept as its
`end_status` ({py:class}`~larmor.tractography.EndStatus`), which TRK files store and TCK files cannot.

## Reproducible runs

The run seed (`seed=`, `--seed`, an integer in [0, 2^64)) fixes the seed points drawn and every other random
number, so the same run seed gives the same tractogram whatever the number of threads. Without one, a seed is drawn
and kept in the run record.

## Output formats

`track` writes TCK or TRK files. For other formats, convert the result, for example to VTK:

```bash
lar convert tracks.tck tracks.vtk
```

[Loading and saving data](loading-and-saving.md#tractograms) covers the formats.
[Isolating a bundle](isolating-a-bundle.md) and [Tract profiles](tract-profiles.md) show what to do next.

## Viewing the streamlines

<!-- not run: opens the viewer -->
```bash
lar bv --tracts tracks.tck -f fod.nii.gz
```

## See also

- Gallery: [Tracking streamlines through crossing bundles](../gallery/tractography/tracking_synthetic.rst),
  [Whole-brain tractography of a simulated brain](../gallery/tractography/tracking_ismrm.rst),
  [Writing a tractogram batch by batch](../gallery/tractography/tractogram_batch_writing.rst).
- API: {py:class}`~larmor.tractography.Tracker`, {py:class}`~larmor.tractography.TractographyRun`,
  {py:class}`~larmor.tractography.MaskSeeds`, {py:func}`~larmor.tractography.seeds_per_voxel`,
  {py:func}`~larmor.tractography.read_seeds`, {py:func}`~larmor.tractography.streamline_lengths`.
- Command: [`lar track`](../cli.md#lar-track).
