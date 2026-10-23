---
myst:
  html_meta:
    description: "BrainViewer, Larmor's viewer for diffusion MRI: volumes, parcellations, tractograms, FODs, peaks and surfaces in 2D and 3D, and a phantom editor for simulation."
---

# Viewer

BrainViewer, Larmor's viewer, shows volumes, parcellations, tractograms, FODs, peaks and cortical surfaces in 2D
and 3D. It runs as a desktop application (`lar bv`), from Python, off screen in scripts, or driven by an agent through
the Model Context Protocol (MCP, see [Agents](../agents/index.md)).

The viewer also has a phantom editor for Larmor's [simulation](../simulation.md). A phantom is a synthetic brain
made of tissues and fibre bundles. In the editor you draw fibre bundles in the 2D views, choose their tissues,
set the scan and run the simulation, whose images then open in the viewer. The editor's guided layout takes you
through a first simulation one step at a time. The Signal breakdown panel shows how the signal of one voxel of
the phantom comes about, and it follows each change you make in the editor.

The guides below cover one feature each. The viewer shows the same guides in its Help panel, which you open with
Help > Help or F1.

- [Getting started](getting-started.md): opening files, installing the viewer on its own, the panels and views
- [The 2D views](2d-views.md): moving around, the crosshair and the readout
- [Volumes and overlays](volumes-and-overlays.md): colormaps, thresholds, 4D images and DTI colour maps
- [Parcellations](parcellations.md): regions in their own colours, lookup tables and labels in 3D
- [Tractograms](tractograms.md): colouring by value, filtering by regions, and tubes
- [The 3D view](3d-view.md): what it shows, and the camera
- [FOD glyphs](fod-glyphs.md): symmetric and asymmetric FODs, and one FOD shown large
- [Peaks](peaks.md): an FOD's peaks as lines
- [Settings](settings.md): the viewer's preferences
- [Screenshots](screenshots.md): images, separate views and rotation videos
- [Sessions](sessions.md): saving what the viewer shows, to open it again later
- [Surfaces](surfaces.md): cortical surfaces in 3D and in the 2D views
- [MATLAB files](matlab-files.md): opening MRIToolkit and ExploreDTI results
- [Phantoms](phantoms.md): the phantom editor and its guided layout, drawing bundles, tissues and the scan, the signal
  breakdown of a voxel, and running a simulation
- [Keyboard shortcuts](shortcuts.md): every key and mouse control

The [gallery](../gallery/viewer/index.rst) drives the viewer from Python, off screen, and saves screenshots of its
views.

```{toctree}
:hidden:

Getting started <getting-started>
2d-views
volumes-and-overlays
parcellations
tractograms
3d-view
fod-glyphs
peaks
settings
screenshots
sessions
surfaces
matlab-files
phantoms
shortcuts
```
