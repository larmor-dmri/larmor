---
myst:
  html_meta:
    description: "An overview of the methods in Larmor, from preprocessing and model fitting to tractography, simulation, file formats and the viewer, with the command that runs each."
---

# What Larmor does

Larmor is a Python library for diffusion MRI, with a viewer called BrainViewer. This page lists what it can do
and names the `lar` command that runs each method. The [command-line reference](../cli.md) describes the commands
in full. The papers behind the methods are listed
under [Citing](../about/citing.md#methods-and-papers).

(preprocess)=
## Preprocessing

- Denoising diffusion images with MP-PCA (`lar denoise`).
- Removing Gibbs ringing (`lar degibbs`).
- Computing a brain mask from a diffusion image (`lar mask`).

(model)=
## Models

- DTI and DKI, fitted by ordinary or weighted least squares (`lar dti`, `lar dki`).
- NODDI, for neurite density, orientation dispersion and free water (`lar noddi`).
- Fibre orientation distributions (FODs) by spherical deconvolution (`lar fod`), with generalized
  Richardson-Lucy (GRL), with mFOD, which fits one FOD for each anisotropic tissue, or with constrained spherical
  deconvolution (CSD). GRL and mFOD can be solved by damped Richardson-Lucy (dRL), non-negative least squares or
  LASSO.
- Multi-shell multi-tissue CSD (MSMT-CSD), with white matter, grey matter and CSF responses read from MRtrix3 files
  or estimated from the data (`lar fod --method msmt-csd`).
- Conversion of FODs between spherical harmonic (SH) bases (`lar sh-convert`).
- The peaks of the FODs (`lar peaks`).
- Asymmetric FODs, made by filtering symmetric ones with their neighbouring voxels (`lar afod`).

(track)=
## Tractography

- Deterministic and probabilistic tracking through an FOD image (`lar track`).
- Editing tractograms: selecting streamlines by regions, length and weight (`lar tract-edit`).
- Sampling maps such as FA along a bundle into a tract profile (`lar tract-profile`).
- Computing track density images (`lar tract-density`) and structural connectomes (`lar connectome`).
- Applying registrations computed by ANTs, FSL and MRtrix3 to images, tractograms and surfaces (`lar transform`).

## Simulation

- Simulating diffusion-weighted images of phantoms made of tissues and fibre bundles, together with their ground
  truth (`lar simulate`).

## Files

Larmor reads and writes NIfTI, MRtrix, TCK, TRK, VTK, FreeSurfer and GIfTI files, and reads MRIToolkit results
saved as MATLAB files (`lar convert`, `lar info`); [File formats](../reference/file-formats.md) lists them all.

## The viewer

BrainViewer shows volumes, parcellations, tractograms, FODs, peaks and cortical surfaces in 2D and 3D (`lar bv`).
See [Viewer](../viewer/index.md) and [Agents](../agents/index.md).
