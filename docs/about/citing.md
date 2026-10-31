---
myst:
  html_meta:
    description: "How to cite Larmor, and the original papers of each method it implements, from denoising and model fitting to tractography and simulation."
---

# Citing

## Citing Larmor

If you use Larmor in your work, please cite it, together with the papers of the methods you used, listed below.
There is no paper about Larmor yet, so cite the software itself, with the version you used (`lar --version`
prints it). The DOI 10.5281/zenodo.23604584 is that of version 2.0.0. Cite it as follows:

```{include} ../_snippets/cite-larmor.md
```

In BibTeX:

```bibtex
@software{larmor,
  author  = {Di Meglio, Antonio},
  title   = {Larmor: a Python library and viewer for diffusion MRI},
  version = {x.y.z},
  year    = {YYYY},
  doi     = {10.5281/zenodo.23604584},
  url     = {https://larmor.org},
}
```

The same information is in the repository's `CITATION.cff` file, which GitHub and reference managers read.

## Methods and papers

### Preprocessing

- **MP-PCA denoising.** Veraart J, Novikov DS, Christiaens D, Ades-aron B, Sijbers J, Fieremans E. Denoising of
  diffusion MRI using random matrix theory. NeuroImage 142 (2016) 394-406.
- **Gibbs-ringing removal.** Kellner E, Dhital B, Kiselev VG, Reisert M. Gibbs-ringing artifact removal based on
  local subvoxel-shifts. Magn Reson Med 76 (2016) 1574-1581.

### Models

- **DTI.** Basser PJ, Mattiello J, LeBihan D. Estimation of the effective self-diffusion tensor from the NMR spin
  echo. J Magn Reson B 103 (1994) 247-254.
- **DKI.** Jensen JH, Helpern JA, Ramani A, Lu H, Kaczynski K. Diffusional kurtosis imaging: the quantification
  of non-Gaussian water diffusion by means of magnetic resonance imaging. Magn Reson Med 53 (2005) 1432-1440.
  [doi:10.1002/mrm.20508](https://doi.org/10.1002/mrm.20508)
- **Constrained DKI fit and its measures.** Tabesh A, Jensen JH, Ardekani BA, Helpern JA. Estimation of tensors
  and tensor-derived measures in diffusional kurtosis imaging. Magn Reson Med 65 (2011) 823-836.
- **Weighted fits of DTI and DKI.** Veraart J, Sijbers J, Sunaert S, Leemans A, Jeurissen B. Weighted linear
  least squares estimation of diffusion MRI parameters: strengths, limitations, and pitfalls. NeuroImage 81
  (2013) 335-346.
- **NODDI.** Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC. NODDI: Practical in vivo neurite
  orientation dispersion and density imaging of the human brain. NeuroImage 61 (2012) 1000-1016.
- **The start of the NODDI fit,** a dictionary solved by convex optimisation. Daducci A, Canales-Rodriguez EJ,
  Zhang H, Dyrby TB, Alexander DC, Thiran JP. Accelerated Microstructure Imaging via Convex Optimization (AMICO)
  from diffusion MRI data. NeuroImage 105 (2015) 32-44.
- **The NODDI model and fit as Larmor computes them** follow the NODDI Matlab Toolbox 1.05,
  [nitrc.org/projects/noddi_toolbox](https://www.nitrc.org/projects/noddi_toolbox).

### Spherical deconvolution

- **GRL.** Guo F, Leemans A, Viergever MA, Dell'Acqua F, De Luca A. Generalized Richardson-Lucy (GRL) for
  analyzing multi-shell diffusion MRI data. NeuroImage 218 (2020) 116948.
  [doi:10.1016/j.neuroimage.2020.116948](https://doi.org/10.1016/j.neuroimage.2020.116948)
- **mFOD**, and its solver, non-negative least squares with a ridge penalty (L2NNLS). De Luca A, Guo F, Froeling
  M, Leemans A. Spherical deconvolution with tissue-specific response functions and multi-shell diffusion MRI to
  estimate multiple fiber orientation distributions (mFODs). NeuroImage 222 (2020) 117206.
  [doi:10.1016/j.neuroimage.2020.117206](https://doi.org/10.1016/j.neuroimage.2020.117206)
- **CSD.** Tournier JD, Calamante F, Connelly A. Robust determination of the fibre orientation distribution in
  diffusion MRI: non-negativity constrained super-resolved spherical deconvolution. NeuroImage 35 (2007)
  1459-1472. [doi:10.1016/j.neuroimage.2007.02.016](https://doi.org/10.1016/j.neuroimage.2007.02.016)
- **MSMT-CSD.** Jeurissen B, Tournier JD, Dhollander T, Connelly A, Sijbers J. Multi-tissue constrained spherical
  deconvolution for improved analysis of multi-shell diffusion MRI data. NeuroImage 103 (2014) 411-426.
  [doi:10.1016/j.neuroimage.2014.07.061](https://doi.org/10.1016/j.neuroimage.2014.07.061). Its constrained least
  squares is solved as a least-distance problem by NNLS (Lawson and Hanson 1974, chapter 23).
- **Tissue responses without a segmentation.** Dhollander T, Raffelt D, Connelly A. Unsupervised 3-tissue response
  function estimation from single-shell or multi-shell diffusion MR data without a co-registered T1 image. ISMRM
  Workshop on Breaking the Barriers of Diffusion MRI (2016) 5. Dhollander T, Mito R, Raffelt D, Connelly A.
  Improved white matter response function estimation for 3-tissue constrained spherical deconvolution. Proc Intl
  Soc Mag Reson Med 27 (2019) 555. The white matter response's constrained fit: Smith RE, Dhollander T, Connelly A.
  Constrained linear least squares estimation of anisotropic response function for spherical deconvolution. ISMRM
  Workshop on Breaking the Barriers of Diffusion MRI (2016) 23. Its automatic thresholds: Ridgway GR, Omar R,
  Ourselin S, Hill DL, Warren JD, Fox NC. Issues with threshold masking in voxel-based morphometry of atrophied
  brains. NeuroImage 44 (2009) 99-111.
- **Damped Richardson-Lucy (dRL) solver.** Dell'Acqua F, Scifo P, Rizzo G, et al. A modified damped
  Richardson-Lucy algorithm to reduce isotropic background effects in spherical deconvolution. NeuroImage 49
  (2010) 1446-1458. [doi:10.1016/j.neuroimage.2009.09.033](https://doi.org/10.1016/j.neuroimage.2009.09.033)
- **LASSO solver.** Tibshirani R. Regression shrinkage and selection via the lasso. J R Stat Soc B 58 (1996)
  267-288. Zou H, Hastie T. Regularization and variable selection via the elastic net. J R Stat Soc B 67 (2005)
  301-320. [doi:10.1111/j.1467-9868.2005.00503.x](https://doi.org/10.1111/j.1467-9868.2005.00503.x)
- **Non-negative least squares (NNLS),** used by the solvers and the constrained DKI fit. Lawson CL, Hanson RJ.
  Solving Least Squares Problems. Prentice-Hall, 1974 (chapter 23).
- **NODDI response function.** Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC. NODDI: practical in vivo
  neurite orientation dispersion and density imaging of the human brain. NeuroImage 61 (2012) 1000-1016.
  [doi:10.1016/j.neuroimage.2012.03.072](https://doi.org/10.1016/j.neuroimage.2012.03.072)

- **Asymmetric FODs by filtering.** Poirier C, Descoteaux M. A unified filtering method for estimating
  asymmetric orientation distribution functions. NeuroImage 287 (2024) 120516.
  [doi:10.1016/j.neuroimage.2024.120516](https://doi.org/10.1016/j.neuroimage.2024.120516)

### Tractography and analysis

- **Deterministic tracking.** Tournier J-D, Calamante F, Connelly A. MRtrix: diffusion tractography in crossing
  fiber regions. Int J Imaging Syst Technol 22 (2012) 53-66.
  [doi:10.1002/ima.22005](https://doi.org/10.1002/ima.22005)
- **Probabilistic tracking.** Tournier J-D, Calamante F, Connelly A. MRtrix: diffusion tractography in crossing
  fiber regions. Int J Imaging Syst Technol 22 (2012) 53-66.
  [doi:10.1002/ima.22005](https://doi.org/10.1002/ima.22005)
- **Track density images.** Calamante F, Tournier JD, Jackson GD, Connelly A. Track-density imaging (TDI):
  super-resolution white matter imaging using whole-brain track-density mapping. NeuroImage 53 (2010) 1233-1243.
- **Structural connectomes.** Hagmann P, Cammoun L, Gigandet X, Meuli R, Honey CJ, Wedeen VJ, Sporns O. Mapping
  the structural core of human cerebral cortex. PLoS Biol 6 (2008) e159. For the radial search that assigns
  endpoints to regions: Smith RE, Tournier JD, Calamante F, Connelly A. The effects of SIFT on the
  reproducibility and biological accuracy of the structural connectome. NeuroImage 104 (2015) 253-265.
- **Tract profiles.** Yeatman JD, Dougherty RF, Myall NJ, Wandell BA, Feldman HM. Tract profiles of white matter
  properties: automating fiber-tract quantification. PLoS One 7 (2012) e49790.
  [doi:10.1371/journal.pone.0049790](https://doi.org/10.1371/journal.pone.0049790). Chandio BQ, Risacher SL,
  Pestilli F, et al. Bundle analytics, a computational framework for investigating the shapes and profiles of
  brain pathways across populations. Sci Rep 10 (2020) 17149.
  [doi:10.1038/s41598-020-74054-4](https://doi.org/10.1038/s41598-020-74054-4)

### Registration

- **Reorienting tensors and peaks.** Alexander DC, Pierpaoli C, Basser PJ, Gee JC. Spatial transformations of
  diffusion tensor magnetic resonance images. IEEE Trans Med Imaging 20 (2001) 1131-1139.

### Simulation

The simulation library builds on SymDWI, a simulator by Larmor's author.

- **Stick.** Behrens TEJ, Woolrich MW, Jenkinson M, et al. Characterization and propagation of uncertainty in
  diffusion-weighted MR imaging. Magn Reson Med 50 (2003) 1077-1088.
- **Powder-averaged stick.** Kroenke CD, Ackerman JJH, Yablonskiy DA. On the nature of the NAA diffusion
  attenuated MR signal in the central nervous system. Magn Reson Med 52 (2004) 1052-1059. Kaden E, Kelm ND,
  Carson RP, Does MD, Alexander DC. Multi-compartment microscopic diffusion imaging. NeuroImage 139 (2016) 346-359.
- **NODDI and Watson-dispersed sticks.** Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC. NODDI:
  Practical in vivo neurite orientation dispersion and density imaging of the human brain. NeuroImage 61 (2012)
  1000-1016.
- **Cylinder.** Van Gelderen P, DesPres D, van Zijl PCM, Moonen CTW. Evaluation of restricted diffusion in
  cylinders. Phosphocreatine in rabbit leg muscle. J Magn Reson B 103 (1994) 255-260. De Santis S, Jones DK,
  Roebroeck A. Including diffusion time dependence in the extra-axonal space improves in vivo estimates of axonal
  diameter and density in human white matter. NeuroImage 130 (2016) 91-103.
- **Ball and zeppelin.** Novikov DS, Fieremans E, Jespersen SN, Kiselev VG. Quantifying brain microstructure with
  diffusion MRI: Theory and parameter estimation. NMR Biomed 32 (2019) e3998. Kaden E, Kelm ND, Carson RP, Does MD,
  Alexander DC. Multi-compartment microscopic diffusion imaging. NeuroImage 139 (2016) 346-359.
- **Dot.** Panagiotaki E, Schneider T, Siow B, Hall MG, Lythgoe MF, Alexander DC. Compartment models of the
  diffusion MR signal in brain white matter: A taxonomy and comparison. NeuroImage 59 (2012) 2241-2254.
- **Free water.** Pasternak O, Sochen N, Gur Y, Intrator N, Assaf Y. Free water elimination and mapping from
  diffusion MRI. Magn Reson Med 62 (2009) 717-730.
- **SANDI.** Palombo M, Ianus A, Guerreri M, et al. SANDI: a compartment-based model for non-invasive apparent soma
  and neurite imaging by diffusion MRI. NeuroImage 215 (2020) 116835.
- **Standard model of white matter.** Novikov DS, Fieremans E, Jespersen SN, Kiselev VG. Quantifying brain
  microstructure with diffusion MRI: Theory and parameter estimation. NMR Biomed 32 (2019) e3998.
- **Default parameters.** Dhital B, Reisert M, Kellner E, Kiselev VG. Intra-axonal diffusivity in brain white
  matter. NeuroImage 189 (2019) 543-550. Veraart J, Novikov DS, Fieremans E. TE dependent Diffusion Imaging
  (TEdDI) distinguishes between compartmental T2 relaxation times. NeuroImage 182 (2018) 360-369. Pasternak O,
  Sochen N, Gur Y, Intrator N, Assaf Y. Free water elimination and mapping from diffusion MRI. Magn Reson Med 62
  (2009) 717-730. Wansapura JP, Holland SK, Dunn RS, Ball WS. NMR relaxation times in the human brain at 3.0
  tesla. J Magn Reson Imaging 9 (1999) 531-538. Lu H, Nagae-Poetscher LM, Golay X, Lin D, Pomper M, van Zijl PCM.
  Routine clinical brain MRI sequences for use at 3.0 Tesla. J Magn Reson Imaging 22 (2005) 13-22.
  [doi:10.1002/jmri.20356](https://doi.org/10.1002/jmri.20356)
- **Gradient directions by electrostatic repulsion.** Jones DK, Horsfield MA, Simmons A. Optimal strategies for
  measuring diffusion in anisotropic systems by magnetic resonance imaging. Magn Reson Med 42 (1999) 515-525.
- **Rician noise.** Gudbjartsson H, Patz S. The Rician distribution of noisy MRI data. Magn Reson Med 34 (1995)
  910-914.

### Toolboxes

The spherical deconvolution methods and their solvers (GRL, mFOD, dRL and L2NNLS) are reimplementations of those
in MRIToolkit, by Alberto De Luca ([github.com/delucaal/MRIToolkit](https://github.com/delucaal/MRIToolkit)), and
the LASSO solver is based on MRIToolkit's LASSO method.
CSD and the estimate of its response, with their defaults and the response file format, follow MRtrix3's
`dwi2fod csd` and `dwi2response tournier`; MSMT-CSD and its tissue responses follow `dwi2fod msmt_csd` and
`dwi2response dhollander`. Denoising, Gibbs-ringing removal, tractogram editing, track density images and
connectomes follow MRtrix3's `dwidenoise`, `mrdegibbs`, `tckedit`, `tckmap` and `tck2connectome`: Tournier JD et
al. MRtrix3: A fast,
flexible and open software framework for medical image processing and visualisation. NeuroImage 202 (2019)
116137. [doi:10.1016/j.neuroimage.2019.116137](https://doi.org/10.1016/j.neuroimage.2019.116137)
