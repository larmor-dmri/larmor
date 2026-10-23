---
myst:
  html_meta:
    description: "The Larmor Python API for diffusion MRI: the public classes and functions of each package, generated from their docstrings."
---

# Python API

One page per public package, generated from the docstrings. A name is public when it is in its package's
`__all__`; these pages list exactly those names. Import a name from its package, for example
`from larmor.sd import find_peaks`, not from the module that defines it.

| Package | What it holds |
| --- | --- |
| [larmor](larmor.rst) | The names for the most common tasks, from the packages below |
| [larmor.types](types.rst) | The core data types: grids, volumes, tractograms, surfaces, gradient tables and labels |
| [larmor.io](io.rst) | Loading and saving by path, gradient files, MATLAB files and MRIToolkit results |
| [larmor.io.formats](io-formats.rst) | The reader and writer of each file format |
| [larmor.dwi](dwi.rst) | Brain masks, denoising, Gibbs ringing removal, and the DTI and DKI models |
| [larmor.sd](sd.rst) | Spherical deconvolution: response functions, FODs and their peaks |
| [larmor.tractography](tractography.rst) | Seeds, the tracker, editing tractograms, track density, connectomes and tract profiles |
| [larmor.registration](registration.rst) | Applying and inverting ANTs, FSL and MRtrix3 registrations |
| [larmor.simulation](simulation.rst) | Phantoms, scan protocols, compartment models and the instant simulation engine |
| [larmor.plot](plot.rst) | Plots drawn with matplotlib, such as the signal breakdown of a simulated voxel |
| [larmor.surfaces](surfaces.rst) | Operations on surface meshes |
| [larmor.data](data.rst) | Public example datasets, downloaded on first use |
| [larmor.viewer](viewer.rst) | BrainViewer from Python |

The files each function reads and writes are listed in [File formats](../file-formats.md).

```{toctree}
:hidden:

larmor <larmor>
larmor.types <types>
larmor.io <io>
larmor.io.formats <io-formats>
larmor.dwi <dwi>
larmor.sd <sd>
larmor.tractography <tractography>
larmor.registration <registration>
larmor.simulation <simulation>
larmor.plot <plot>
larmor.surfaces <surfaces>
larmor.data <data>
larmor.viewer <viewer>
```
