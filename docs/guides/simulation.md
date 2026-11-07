---
myst:
  html_meta:
    description: "How to build a phantom of tissues and fibre bundles with Larmor, save it as a phantom file, simulate diffusion-weighted images from it with lar simulate, and read the ground truth it writes."
---

# Simulating phantoms

Simulated images with a known ground truth let you test a method against the answer it should give. Larmor
simulates them from a phantom, an object to scan made of tissues that fibre bundles and tissue maps place on a voxel
grid. Each tissue is a sum of compartments, such as a stick for the water inside axons or a ball for free water,
each with its own diffusion model. {py:func}`~larmor.simulation.simulate` turns a phantom and a scan protocol into
diffusion-weighted images, and keeps the ground truth: how much of each voxel each tissue and bundle fills, and
where the fibres point. [Simulating diffusion MRI](../background/simulation.md) describes what is computed and what
is not.

For a first run with no code, see the [quick start](../quick-start.md).

## Building a phantom

The phantom below is one straight bundle on a grid of 24 x 24 x 24 voxels of 1 mm. The bundle's
{py:class}`~larmor.simulation.BundleGeometry` is a smooth curve through control points in world coordinates (mm), with
`num_streamlines` streamlines around it. Their density across the bundle is a Gaussian cut at the given `radius`,
which is 2.5 standard deviations, and the seed makes the streamlines the same on every run. A phantom has the tissues
WM, GM and CSF by default, and a bundle is filled with WM unless you name another tissue.

```python
import numpy as np

from larmor.simulation import (
    Bundle,
    BundleGeometry,
    Phantom,
    Protocol,
    create_gradient_table,
    write_phantom,
)
from larmor.types import Grid

grid = Grid((24, 24, 24), np.eye(4))
line = BundleGeometry(
    np.array([[3.0, 12.0, 12.0], [21.0, 12.0, 12.0]]),
    num_streamlines=300,
    radius=2.5,
    seed=1,
)
phantom = Phantom(grid, bundles=(Bundle(line),))

gradients = create_gradient_table([(1000, 32)], num_b0=1, seed=0)
protocol = Protocol(gradients, 15.0, 30.0, 80.0, 3000.0)
write_phantom(phantom, "straight.phantom.json", protocol=protocol)
```

The protocol holds the gradient table and the timings, which here are the pulse duration delta (15 ms), the
time between the pulses Delta (30 ms), the echo time (80 ms) and the repetition time (3000 ms).
`create_gradient_table` spreads the directions of each shell evenly over the sphere, and a table read from bvals
and bvecs files works as well (see [Gradient tables](gradient-tables.md)).

A phantom has four parts.

- **Tissues.** Each tissue has a name, by which bundles and maps refer to it. The following functions make the
  predefined tissues, and their arguments set the fractions, diffusivities and relaxation times.
  - {py:func}`~larmor.simulation.standard_model_tissue` makes white matter as a Watson stick and a zeppelin, with
    optional free water. This differs slightly from the usual stick, zeppelin and ball model. By using a Watson
    stick, the simulated signal more closely resembles a real acquisition, because it drops the assumption that
    all fibres in a voxel have exactly the same direction. Its `kappa` argument sets how tightly the sticks follow
    the fibre direction, and `kappa=None` gives a plain stick and zeppelin.
  - {py:func}`~larmor.simulation.noddi_tissue` makes white or grey matter in the NODDI model, with
    Watson-dispersed sticks, a zeppelin whose diffusivities follow the neurite density and dispersion, and free
    water.
  - {py:func}`~larmor.simulation.sandi_tissue` makes grey matter.
  - {py:func}`~larmor.simulation.csf_tissue` and {py:func}`~larmor.simulation.free_water_tissue` make free water,
    named CSF and FW.

  A tissue can also be built from {py:class}`~larmor.simulation.Compartment` objects of any model, including the
  cylinder and the dot. The signal of the stick, the zeppelin, the Watson-dispersed sticks and the cylinder depends
  on a fibre direction, which only a bundle gives, so a tissue with one of these compartments can fill bundles
  only. Other tissues can also fill tissue maps and the background.
- **Bundles.** A bundle's shape is a {py:class}`~larmor.simulation.BundleGeometry` or a tractogram of your own,
  such as one from `load_tractogram`. The fraction of a voxel that a bundle fills is its occupancy. A geometry of
  one radius fills each voxel by the share of the voxel inside its tube, however dense its streamlines are there. In
  a tractogram, a geometry with rings or a landing, or a geometry given a `streamline_area`, each streamline stands
  for a piece of the bundle's cross-section, so the occupancy follows the length of streamline in the voxel. Without a
  `streamline_area`, that piece is set so that a tractogram's dense voxels have an occupancy of `densest_fill`, 0.7 by
  default. Where bundles cross and their occupancies add up to more than the voxel, they are scaled down to fit.
- **Tissue maps.** A {py:class}`~larmor.simulation.TissueMap` is a volume of fractions in [0, 1] on the grid,
  such as a grey matter probability. The maps share what the bundles leave of each voxel.
- **Background.** What is left after the bundles and maps goes to the tissue named by `background`, or gives no
  signal when there is none.

## Changing the section along a bundle

A round bundle has one `radius` all along. `rings` set the section along the bundle instead, so that it can widen,
flatten and turn. Each row holds a position `u` along the centreline, from 0 at the start to 1 at the end, the
section's two radii in mm, and the angle of the first radius in degrees. Between rows each value follows a monotone
curve, so it stays between its neighbours.

```python
ribbon = BundleGeometry(
    np.array([[3.0, 12.0, 12.0], [21.0, 12.0, 12.0]]),
    num_streamlines=300,
    seed=1,
    rings=np.array([[0.0, 2.0, 2.0, 0.0], [0.5, 6.0, 2.0, 0.0], [1.0, 6.0, 1.5, 45.0]]),
)
```

The rings replace `radius`. The angle is measured in a frame carried
along the centreline without twist, described in {py:class}`~larmor.simulation.BundleGeometry`. The density across
the section is the same Gaussian, stretched to the two radii.

A bundle with rings fills by streamline length. By default its streamlines together stand for 3 times its median
section, the size it has over most of its length, so its core is full and its edge is soft. Its total fill is about 85
to 95% of a tube's when its radius is at least 3 voxels, and about 110 to 115% at 1.5 voxels, where every voxel it
touches can fill up. A change of width over less than half of its length leaves the median alone, so the same
streamlines thin out where the bundle widens: widening a 3 mm bundle to 6 mm over a short stretch fills about 8 mm of
its 12 mm width at least half, with a faint edge beyond. A larger `streamline_area` fills more of it, and a smaller one
makes the whole bundle sparser. `densest_fill` does not apply to rings.

## Tangling a bundle

`tangle_mm` makes the streamlines lose their places across the bundle. Each one slides between two places drawn from
the same Gaussian, reaching the second after `tangle_mm` mm along the centreline, so that a value of 10 means a
streamline has moved to an unrelated place every 10 mm. It works with a plain radius, rings or a landing. The slide is a
regular wave, not a random walk: all streamlines share one wavelength of 4 `tangle_mm`, and the whole section repeats
after that distance.

The spread across the bundle keeps its standard deviation, but halfway through a slide about 2% of the streamlines
pass the edge, by up to about a third of the radius. A bundle of one radius still fills its tube, exactly as without
a tangle. A bundle that fills by streamline length fills up to about 5% more, as the streamlines are longer and reach
further.

What matters is the bundle's radius against `tangle_mm`. In a straight round bundle, the streamlines lie about 40
radius / `tangle_mm` degrees from the centreline on average, up to about 25 degrees. In a test with the default white
matter, with no noise, a radius / `tangle_mm` of 0.15 lowers the FA from 0.86 to 0.85, 0.3 to 0.83, 0.6 to 0.72 and
1.2 to 0.37. How well a crossing is resolved depends strongly on the response. At 90 degrees and SNR 30, with a
response estimated from an equally tangled bundle, 99% of the voxels show exactly two peaks up to a ratio of 0.4, 92%
at 0.6 and 65% at 0.86. With a response from an untangled bundle that is 68% at 0.4 and 2% at 0.6.

A `tangle_mm` under twice the distance between a streamline's points is refused, because the points cannot show the
slide. Above that, streamlines turn sharply at each point below about 2 sqrt(spacing radius), where spacing is that
distance, and more `num_points` cures this. A tangle shorter than the radius makes streamlines more than a quarter
longer than the centreline.

## Landing a bundle

A landing bends the streamlines of one end over to a line, while the centreline stays as it is. `landing_start` and
`landing_end` each take a {py:class}`~larmor.simulation.Landing`, which holds the line in world coordinates (mm) and
the position `opens_from` along the centreline where the bending starts, from 0 at the centreline's start to 1 at its
end, whichever end the landing is at.

```python
from larmor.simulation import Landing

spreading = BundleGeometry(
    np.array([[3.0, 12.0, 12.0], [21.0, 12.0, 12.0]]),
    num_streamlines=300,
    radius=2.0,
    seed=1,
    landing_end=Landing(np.array([[21.0, 4.0, 12.0], [21.0, 20.0, 12.0]]), opens_from=0.6),
)
```

Each streamline lands at a place along the line given by its place along the section's first radius, so the density
along the line is the same cut Gaussian. The line sets that radius and the section's angle at that end, from its extent
across the path. A streamline further out than an end of the line, which only a tangle can make, lands on that end.

With one line, a streamline keeps its offset along the second radius, so the streamlines land on a band around the
line, as thick as that radius on each side. A `thickness_line` across the first sets that radius too, and the
streamlines then land on the surface made by moving the first line along it, wherever the thickness line lies.

Without rings, the section keeps `radius` until `opens_from` and then stretches along the line without turning. With
rings, the landing replaces the values of that end's row, and the section follows the rings between rows, so with only
two rows it changes along the whole bundle. It also turns to the line's angle between those rows while it widens, which
makes it bulge out of the line's plane on the way unless the row before already has that angle. A bundle with a landing fills by streamline length, with the same default as rings: its
streamlines stand for 3 times its median section, which leaves out the landing and is pi r^2 without rings. It fills the
part before `opens_from` as the same bundle does without a landing. Over the spread, fewer voxels fill up, and more
voxels are partly filled.

The streamlines are held back on the inside of a tight bend before they bend over to the line, so near such a bend they
can land a little off it. An `opens_from` close to the end makes them turn sharply: on a 60 mm bundle landing on a line
40 mm long, most streamlines have a step over 45 degrees when the bending starts within about 1 mm of the end, at 100
points or 400. A long line along the path's direction, not across it, can make streamlines go back along the path, as
some must land behind where they would otherwise be.

## Drawing the shape in the viewer

The viewer reads and writes `rings`, `tangle_mm`, `landing_start` and `landing_end`, in a phantom file and in a session.
Its Width and Twist tools write rings, its Fan tool writes landings, and its fields set `tangle_mm` and
`streamline_area`, as [Phantoms](../viewer/phantoms.md) explains. A branch drawn there has no field of its own here: its
geometry holds the path and the section it shares with its parent, and only the viewer block of the file says which
bundle it left.

## The phantom file

`write_phantom` stores the grid, the tissues, the bundles (their geometry, or the path of their tractogram), the
paths of the tissue maps and the protocol in one JSON file, `*.phantom.json`, which you can read and edit. A bundle
geometry without a seed gets a random one, written to the file, so that the file always gives the same streamlines.
A file with the `fan_start`, `fan_end` and `fan_sharpness` of earlier versions still opens, with its fan read as rings of
the same width, and a `dispersion` in it is ignored. Both log a warning.
{py:func}`~larmor.simulation.read_phantom` reads the file back.

## Running a phantom file

::::{tab-set}
:sync-group: interface

:::{tab-item} Python
:sync: python
```python
from larmor.simulation import read_phantom, save_simulation, simulate

loaded = read_phantom("straight.phantom.json")
dwi, truth = simulate(loaded.phantom, loaded.protocol, snr=30, seed=1)
files = save_simulation(dwi, truth, "sim")
print(files.dwi[0])
```

`dwi.echoes` holds one 4D volume per echo time, with Rician noise when `snr` is given, and `dwi.noise_free` holds
the same without noise. `truth` is the ground truth, a {py:class}`~larmor.simulation.GroundTruth`, described below.
The noise level is the same in every voxel,
with a standard deviation equal to the b = 0 signal of a voxel filled with the `noise_reference` tissue ("WM" by
default), divided by the SNR. The same `seed` gives the same images on any number of threads (`n_jobs`).
:::

:::{tab-item} Command line
:sync: cli
```bash
lar simulate straight.phantom.json sim --snr 30 --seed 1
```

`lar simulate PHANTOM OUTPUT` reads the phantom file and the protocol stored in it, simulates, and writes the
output folder, which it creates if needed. In place of a file you can give the name of a phantom that ships with
Larmor, such as `crossing`; see the [quick start](../quick-start.md).
:::
::::

## The output folder

The folder is self-contained. It holds the images, the ground truth, and what is needed to simulate again.

| File | Holds |
| --- | --- |
| `dwi.nii.gz`, `dwi.bval`, `dwi.bvec` | The images and their gradients, the b-vectors in the FSL frame. With several echo times there is one image per echo, `dwi_te60ms.nii.gz`, and the gradient files are shared |
| `dwi_noise_free.nii.gz` | The same without noise; only when there is noise |
| `tissue_fractions.nii.gz` | 4D, the volume fraction of each voxel that each tissue fills, in the order of the tissues in `simulation.json` |
| `background_fraction.nii.gz` | The volume fraction no tissue fills |
| `free_water_fraction.nii.gz` | The volume fraction of each voxel that free water fills, summed over the free-water compartments of all tissues, background included |
| `bundle_fractions.nii.gz` | 4D, the volume fraction of each voxel that each bundle fills |
| `compartment_fractions.nii.gz` | 4D, the volume fraction of each voxel that each compartment fills: its tissue's fraction times its share of the tissue, in the order listed under `compartments` in `simulation.json` |
| `signal_fractions.nii.gz` | 4D, each compartment's share of the voxel's signal at b = 0, in the order of `compartment_fractions.nii.gz`. With several echo times there is one per echo, `signal_fractions_te60ms.nii.gz` |
| `free_water_signal_fraction.nii.gz` | The share of the voxel's signal at b = 0 that comes from free water; one per echo time, as above |
| `peaks.nii.gz` | The true fibre directions, in the layout `lar peaks` writes (below) |
| `mask_<tissue>.nii.gz` | One mask per tissue: the voxels the tissue fills by at least half, background included |
| `mask_bundles.nii.gz` | The voxels that the bundles fill by at least half together, so that a crossing of bundles of different tissues is not left out |
| `bundle_<n>.tck` | The streamlines of each bundle |
| `tissue_map_<tissue>.nii.gz` | A copy of each tissue map of the phantom |
| `phantom.json` | The phantom file, with the protocol; its bundles are the geometry with the seed that was used, or the `.tck` of the folder |
| `simulation.json` | The protocol, SNR, noise level, seed, tissue order and the name of every file |

The free-water maps count only the free-water compartment, which the free-water, CSF and NODDI presets and the
standard model's optional free water use. A plain ball does not count as free water, whatever its diffusivity.

Compare a fitted model with the signal-fraction maps, not with the volume fractions. The tissue, background,
bundle, compartment and free-water maps are volume fractions, the share of the voxel's volume that a tissue,
bundle or compartment fills. A model fitted to the images, such as free-water imaging, NODDI or the standard model,
recovers signal fractions instead, in which each compartment is weighted by its proton density and its relaxation.
The signal fraction of compartment c is f_c w_c / (sum over the compartments of f w), where f_c is its volume
fraction and w_c = PD (1 - exp(-TR / T1)) exp(-TE / T2) is its relaxation weight (with T2* for a gradient echo), so
it changes with the echo time. With the default scan (TE 80 ms, TR 3000 ms), for example, the NODDI preset's
free-water volume fraction of 0.05 makes up about 0.11 of the b = 0 signal, since free water has a proton density
of 1 and a long T2, while the neurites and the space around them have 0.7 and 69 ms. Use the maps of the echo time
of the images that were fitted.

The ground-truth peaks give the true fibre directions, to compare with the peaks of a fit, such as `lar fod`
followed by `lar peaks`. They are built from the streamlines, not fitted to the images. In each voxel, the
streamline directions of each bundle are grouped, from the largest down, into peaks: each joins the first peak of
the same bundle within 20 degrees of it, or else starts a new one. Different bundles always give different peaks,
even where they cross at a small angle that no fit could resolve, since they are different fibres. A peak's direction is the weighted mean of its fibres, and its amplitude is the
share of the voxel they fill. Peaks below a tenth of the voxel's largest are dropped, and a bundle whose tissue has
no fibre direction, such as one filled with CSF, gives none. `peaks.nii.gz` has 3 volumes
per peak, largest peak first, which hold the x, y and z of the direction scaled by the amplitude, in world
coordinates, and are NaN where a voxel has fewer peaks.

In Python, the `truth` that `simulate` returns holds the same ground truth as these files. Its maps are NumPy arrays
on the phantom's grid: `tissue_fractions`, `background` and `bundle_fractions`, in which `bundle_fractions[..., k] >
0` is the mask of bundle k. It also holds each bundle's `streamlines` and the `peaks`.
{py:func}`~larmor.simulation.read_ground_truth` reads the same object back from a saved folder, so that an analysis
of the images can be scored against the ground truth long after the simulation ran:

```python
from larmor.simulation import read_ground_truth

truth = read_ground_truth("sim")
print(truth.bundle_fractions.shape, len(truth.streamlines))
```

Running `phantom.json` again with the same seed gives the same images. Without `--seed`, a seed is drawn and
written to `simulation.json`; give it to `--seed` to repeat the run.

```bash
lar simulate sim/phantom.json again --snr 30 --seed 1
```

## Options of `lar simulate`

- `--snr` adds noise, and `--noise-reference` names the tissue it refers to. Without `--snr` there is no noise.
- `--seed` fixes the noise, and the streamlines of bundles that have no seed.
- `--shell B:N` replaces the gradients of the file with a shell of N directions at b = B s/mm^2. Repeat it for
  several shells, and add `--num-b0` to set the number of b = 0 volumes. `--bvals` and `--bvecs`, with
  `--bvec-frame`, read the gradients from files instead, and for these files the frame `fsl` refers to the axes of
  the phantom's grid.
- `--nt` sets the number of threads.
- `--force` is needed to write into a folder that is not empty; without it, the command stops before simulating.
  With it, the files listed in the folder's `simulation.json` are removed first, so that none is left over from the
  earlier run, while other files in the folder stay. `save_simulation` does the same, and refuses a non-empty
  folder without a `simulation.json` unless `force=True`.

The timings of the protocol always come from the phantom file.

## The signal of one voxel

To see where the signal of a voxel comes from, {py:func}`~larmor.simulation.signal_breakdown` splits the
noise-free signal of one voxel in one volume into the factors the simulation multiplies. These are each tissue's
share of the voxel, each compartment's fraction and proton density, the T1 and T2 relaxation, and the diffusion
attenuation. It needs the phantom's voxel partition, which {py:func}`~larmor.simulation.partition_voxels` makes
once for all voxels. {py:func}`~larmor.plot.plot_signal_breakdown` draws the result as three stacked bars.

```python
from larmor.plot import plot_signal_breakdown
from larmor.simulation import partition_voxels, signal_breakdown

partition = partition_voxels(phantom)
breakdown = signal_breakdown(phantom, protocol, partition, (12, 12, 12), 1, snr=30)
for c in breakdown.compartments:
    print(f"{c.label}: {c.no_decay:.3f} -> {c.after_decay:.3f} -> {c.signal:.3f}")
ax = plot_signal_breakdown(breakdown)
ax.figure.savefig("breakdown.png")
```

`breakdown.signal` is the voxel's value in the noise-free image of a run. With `snr`, the breakdown also gives
the noise floor and the mean of the noisy magnitude.

## Viewing the result

<!-- not run: opens the viewer -->
```bash
lar bv --simulation sim
```

The viewer opens the images, the tissue masks over them, the bundles' streamlines and the true peaks. It also
opens the phantom in its Phantom panel, where you can change the phantom and run it again, as described in
[Phantoms](../viewer/phantoms.md).

## See also

- Gallery: [Simulating two crossing bundles](../gallery/simulation/simulate_crossing_bundles.rst) and
  [The signal of one voxel, step by step](../gallery/simulation/signal_breakdown.rst).
- Background: [Simulating diffusion MRI](../background/simulation.md).
- API: [`larmor.simulation`](../reference/api/simulation.rst).
- Command: [`lar simulate`](../cli.md#lar-simulate).

## References

- The simulation builds on SymDWI, a simulator by Larmor's author, relicensed under the LGPL-3.0-only. The papers
  of the models are listed in [Citing](../about/citing.md#simulation).
