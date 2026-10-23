<p align="center">
  <img src="resources/larmor-logo.svg" alt="Larmor" width="360">
</p>

# Larmor

A Python library and viewer for diffusion MRI.

[![CI](https://github.com/larmor-dmri/larmor/actions/workflows/ci.yml/badge.svg)](https://github.com/larmor-dmri/larmor/actions/workflows/ci.yml)
[![License: LGPL-3.0-only](https://img.shields.io/badge/license-LGPL--3.0--only-blue)](COPYING.LESSER)
[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue)](pyproject.toml)
[![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](docs/install.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23604584.svg)](https://doi.org/10.5281/zenodo.23604584)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

<p align="center">
  <img src="src/larmor/viewer/resources/guides/images/window.png" alt="BrainViewer showing axial, coronal and sagittal slices of a T1-like image with an FA overlay, and a 3D view of whole-brain streamlines" width="800">
</p>

Data: ISMRM 2015 Tractography Challenge (Maier-Hein et al., Nat Commun 8 (2017) 1349), CC BY 4.0.

## What it does
- Preprocessing: denoising and Gibbs ringing removal.
- Models: diffusion tensor (DTI) and diffusion kurtosis (DKI) fits.
- Spherical deconvolution (multi-tissue mFOD, and CSD) and tractography, with tract profiles and connectomes.
- Simulation: Allows to simulate dwMRI data and generate hand-drawn or tractography derived phantoms.
- File formats: NIfTI and MIF images, TCK, TRK and VTK tractograms, surfaces, and MRIToolkit results saved as MATLAB files.
- BrainViewer, a viewer for volumes, overlays, FODs, peaks, tractograms and surfaces in 2D and 3D.
- A Python library and the `lar` command line, with the same operations in both.

## Install
```sh
pip install larmor-dmri
pip install "larmor-dmri[viewer]"
```

From source, with [uv](https://docs.astral.sh/uv/) and a C++20 compiler (GCC or Clang on Linux, the Command Line
Tools on macOS, Visual Studio's C++ build tools on Windows):

```sh
git clone https://github.com/larmor-dmri/larmor.git
cd larmor
uv sync
```

[Install](docs/install.md) has detailed explanation of how to install the whole library, or only some of its components.

## Example
This script generates a brain mask, then computes DTI maps inside it, from a `.mif` file that holds its gradient table:

```sh
lar mask dwi.mif mask.nii.gz
lar dti dwi.mif dti -m mask.nii.gz
```

The same can be done in Python, with the mean FA of the white matter:

```python
from larmor.dwi import DTIModel, create_brain_mask, mean_b0
from larmor.io import load_dwi, save_volume

dwi, gtab = load_dwi("dwi.mif")
mask = create_brain_mask(mean_b0(dwi, gtab))
fit = DTIModel().fit(dwi, gtab, mask=mask)
print(f"Mean FA where it is above 0.3: {fit.fa[fit.fa > 0.3].mean():.2f}")
save_volume(dwi.with_data(fit.fa), "fa.nii.gz")
```

[Getting started](docs/getting-started.md) uses the example data of the ISMRM 2015 Tractography Challenge and opens the
viewer with `lar bv`.

## Links
- [Documentation](https://larmor.org)
- [Contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md)
- [Security](SECURITY.md): report a vulnerability privately
- Citing: [CITATION.cff](CITATION.cff) and the [citing page](docs/about/citing.md), which also lists the papers of the
  methods to cite
- Contact: contact@larmor.org, for anything that is not an issue or a pull request

## License
Larmor is distributed under the GNU Lesser General Public License, version 3 only ([COPYING.LESSER](COPYING.LESSER),
which builds on [COPYING](COPYING)). [NOTICE](NOTICE) lists the third-party code and data credits.
