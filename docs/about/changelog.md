---
myst:
  html_meta:
    description: "What changed in each version of Larmor, with a link to the full release notes on GitHub."
---

# Changelog

The release notes of each version are on the repository's
[GitHub Releases page](https://github.com/larmor-dmri/larmor/releases). They list every change, with the breaking
ones first.

## Versions so far

- **2.2**: Fixes across the viewer and the simulation. The viewer no longer aborts when it closes;
  phantom files keep their exact gradient table and bundles loaded through a registration; thin bundles
  are sampled finely enough to keep their volume; New and Open ask before discarding a phantom; several
  bundles or items can be removed at once; a finished run says so. Picked bvecs files are now read in the
  FSL frame by default, and ground-truth peaks and the bundle mask count only the bundles' own fibres.
- **2.1**: A new viewer layout with View and Simulate workspaces, a scene tree and an inspector;
  pointer tools for sphere and box regions in 2D and 3D; detail windows that show a rectangle of a slice
  without smoothing and export it as a figure; export presets; and simulated bundles of even width
  around a centreline that follows their control points.
- **2.0**: The first public release.
