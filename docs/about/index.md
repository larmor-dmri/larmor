---
myst:
  html_meta:
    description: "Who writes Larmor, how to cite it, what changed and what comes next, how to contribute, its licence and the credits for the data on this site."
---

# About

Larmor is written by Antonio Di Meglio. Its spherical deconvolution methods are reimplementations of those in
MRIToolkit, by Alberto De Luca.

- [Citing](citing.md) says how to cite Larmor, and lists the original papers of each method.
- [Changelog](changelog.md) says what changed in each version.
- [Roadmap](roadmap.md) says what comes next.
- [Contributing](contributing.md) explains how to report a problem or propose a change.

```{toctree}
:hidden:

Citing <citing>
Changelog <changelog>
Roadmap <roadmap>
Contributing <contributing>
```

## Licence

Larmor is free software, distributed under the GNU Lesser General Public License, version 3 only
(LGPL-3.0-only), which builds on the GNU General Public License, version 3. The licence texts are in the files
`COPYING.LESSER` and `COPYING`, in the repository and in every installed copy.

Parts of Larmor are reimplementations of methods in MRIToolkit (LGPL-3.0); the file `NOTICE` lists them. The compiled module includes Eigen (MPL-2.0), nanobind and pocketfft (both BSD 3-clause), unmodified.

## Data credits

Every brain image on this site is drawn from the ISMRM 2015 Tractography Challenge data, or from Larmor's
processing of it.

Maier-Hein KH, Neher PF, Houde JC, et al. The challenge of mapping the human connectome based on diffusion
tractography. Nature Communications 8 (2017) 1349.
[doi:10.1038/s41467-017-01285-x](https://doi.org/10.1038/s41467-017-01285-x). The data is published on Zenodo,
[doi:10.5281/zenodo.572345](https://doi.org/10.5281/zenodo.572345), under the
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence.

The other images are drawn from synthetic data made by the examples.
