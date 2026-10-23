---
html_theme.sidebar_secondary.remove: true
myst:
  html_meta:
    description: "Larmor is a Python library, command line and viewer for diffusion MRI. It simulates phantoms, preprocesses and fits the data, and runs tractography."
---

# Larmor

```{rst-class} lar-tagline
A Python library and viewer for simulating and processing diffusion MRI
```

<!-- One slide per line: file in _static/carousel/ | alt text | caption. -->
```{carousel}
tubes.webp | Whole-brain streamlines drawn as lit tubes in 3D, seen from the front, coloured by direction | Whole-brain tractography drawn as tubes, coloured by direction.
simulated-fods.webp | FOD glyphs fitted on a simulated phantom of two bundles crossing at right angles: red lobes along the horizontal bundle, green along the vertical one, and both in the crossing | FODs fitted on a simulated crossing of two bundles.
phantom-editor.webp | The guided phantom editor on its Tissue step: the five steps listed on the left with check marks, and the compartments of a white matter tissue on the right | The phantom editor, which guides a first simulation step by step.
signal-breakdown.webp | The viewer with two simulated bundles crossing in the axial view and the Signal breakdown panel beside it: stacked bars of the signal of the crossing voxel with no decay, after the T1 and T2 decay and after the diffusion weighting | The signal of one simulated voxel, split into its factors.
fod-glyphs.webp | FOD glyphs on a coronal slice: the corpus callosum arching left to right in red, the corticospinal tracts running up and down in blue | FOD glyphs on a coronal slice, with the round FODs hidden.
fod-asymmetric.webp | Asymmetric FOD glyphs on a coronal slice, some lobes one-sided where fibres turn or end | Asymmetric FOD glyphs.
peaks.webp | FOD peaks drawn as short coloured lines on a coronal slice over the T1-like image | FOD peaks drawn as lines, each as long as its amplitude.
dti-colours.webp | Colours of the principal diffusion direction on an axial slice, brighter where FA is high | DTI colours of the principal direction, modulated by FA.
t1-fa.webp | Sagittal slice of the T1-like image with the FA map over it in a hot colormap | FA over the T1-like image on a sagittal slice.
slices-3d.webp | 3D view from above: long streamlines as thin lines over an axial slice of the T1-like image | Streamlines over a slice of the T1-like image in the 3D view.
```

```{rst-class} lar-carousel-note
These images were drawn by BrainViewer, from a phantom simulated with Larmor and from the ISMRM 2015 Tractography Challenge data, both processed with Larmor.
```

```{include} _snippets/ismrm-credit.md
```

::::{container} lar-actions
:::{button-ref} getting-started
:ref-type: doc
:color: primary

Get started
:::
:::{button-ref} gallery/index
:ref-type: doc
:color: primary
:outline:

Gallery
:::
::::

::::{grid} 2 2 5 5
:gutter: 2
:class-container: lar-areas

:::{grid-item-card} Preprocess
:link: preprocess
:link-type: ref

Denoising and Gibbs removal
:::

:::{grid-item-card} Model
:link: model
:link-type: ref

DTI, DKI and fibre orientations
:::

:::{grid-item-card} Track
:link: track
:link-type: ref

Tractography and connectomes
:::

:::{grid-item-card} Simulate
:link: simulation
:link-type: doc

Phantoms with a known ground truth
:::

:::{grid-item-card} View
:link: viewer/index
:link-type: doc

BrainViewer in 2D and 3D
:::
::::

## Install

::::{tab-set}
:class: lar-install

:::{tab-item} Library
```bash
pip install larmor-dmri
```
:::

:::{tab-item} Viewer only
```bash
uv tool install "larmor-dmri[viewer]"
lar bv --install-launcher
```
:::

:::{tab-item} From source
```bash
git clone https://github.com/larmor-dmri/larmor.git
cd larmor
uv sync
```
:::
::::

```{rst-class} lar-install-note
The [Install](install.md) page explains each of these in more detail, and how to run the viewer on a remote server or without a GPU.
```

## Cite Larmor

If you use Larmor, please cite it, together with the papers of the methods you used. The
[Citing](about/citing.md) page lists them.

```{include} _snippets/cite-larmor.md
```

```{toctree}
:hidden:
:maxdepth: 2

Install <install>
Quick start <quick-start>
Simulation <simulation>
Gallery <gallery/index>
User guide <guides/index>
Viewer <viewer/index>
Background <background/index>
Reference <reference/index>
Agents <agents/index>
About <about/index>
```
