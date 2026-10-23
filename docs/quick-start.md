---
myst:
  html_meta:
    description: "Install Larmor, simulate a phantom of two crossing bundles with one command, and open the result in the BrainViewer viewer."
---

# Quick start

This page shows how to install Larmor, simulate a small phantom and look at the result in the viewer. It takes a
few minutes and needs no data download.

## Install

Larmor needs Python 3.14 or newer. The following command installs it together with the viewer:

```sh
pip install "larmor-dmri[viewer]"
```

If you do not have Python 3.14, [uv](https://docs.astral.sh/uv/) can install it for you along with Larmor:
`uv tool install "larmor-dmri[viewer]"`. The [Install](install.md) page describes the other ways.

## Simulate

A phantom describes a piece of tissue, and Larmor simulates the diffusion MRI signal it would give. Because the
phantom is known, so is the true answer in every voxel. Larmor comes with a phantom called `crossing`, in
which two fibre bundles cross at 90 degrees, one along x and one along y, each in its own tissue. Its grid has
40 x 40 x 40 voxels of 1 mm. The following command simulates a scan of it and writes the result into a new folder:

```bash
lar simulate crossing crossing-sim --snr 30 --seed 1
```

`--snr 30` adds Rician noise, with a signal-to-noise ratio of 30 in the b = 0 signal of a voxel full of white
matter. `--seed 1` fixes the random numbers, so every run gives the same images. The simulation takes a few
seconds.

The folder `crossing-sim` now holds the diffusion-weighted images in `dwi.nii.gz`, with their gradients in
`dwi.bval` and `dwi.bvec`. It also holds the ground truth: how much of each voxel each tissue and bundle fills,
the masks, and the true fibre directions in `peaks.nii.gz`. The bundles are saved as `.tck` files, and the
phantom is saved too, so that `lar simulate` can run it again. The [simulation guide](guides/simulation.md)
describes each file.

## View

<!-- not run: opens the viewer -->
```bash
lar bv --simulation crossing-sim
```

This opens the folder in BrainViewer, the viewer that comes with Larmor. It shows the diffusion images, with the
masks as coloured overlays on top. Each tissue has a mask of the voxels it fills by at least half. The mask
`mask_bundles.nii.gz` covers both bundles together, including the crossing, where neither tissue alone fills half
of a voxel. The viewer also draws the streamlines of the bundles and the true fibre directions as short lines. It
opens the phantom in the Phantom editor too, where you can change its settings and run the simulation again.

Click in a 2D view to move the crosshair to that point, and use the **Volume index** option in the panel on the
right to choose which diffusion volume is shown. The [viewer's guide](viewer/getting-started.md) describes the
window.

## Next

- [Simulating phantoms](guides/simulation.md) shows how to build your own phantom in Python, describes the phantom
  file, and lists the options of `lar simulate`.
- [Simulating diffusion MRI](background/simulation.md) explains what the simulation computes.
- [Simulating two crossing bundles](gallery/simulation/simulate_crossing_bundles.rst) does the same as this page
  in Python.
- [Getting started](getting-started.md) introduces the rest of Larmor on a simulated brain.
