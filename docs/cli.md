# `lar`

Command line tools of Larmor, for diffusion MRI analysis, tractography and simulation.

**Usage**:

```console
$ lar [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--version`: Show the version and exit.
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `dti`: Fit the diffusion tensor (DTI) in every...
* `dki`: Fit diffusion kurtosis imaging (DKI) to...
* `denoise`: Denoise a DWI volume with MP-PCA and...
* `mask`: Compute a binary brain mask from the mean...
* `degibbs`: Remove Gibbs ringing from an image by...
* `fod`: Estimate fibre orientation distributions...
* `info`: Print a short summary of an image or a...
* `convert`: Convert an image or a tractogram to...
* `sh-convert`: Convert an FOD image from one spherical...
* `peaks`: Find the peaks (local maxima) of an FOD...
* `track`: Track streamlines through an FOD image.
* `tract-edit`: Select, crop and resample the streamlines...
* `tract-profile`: Write the profile of maps (FA, MD and so...
* `tract-density`: Compute a track density image, the number...
* `connectome`: Count the streamlines between each pair of...
* `transform`: Apply an ANTs, FSL or MRtrix3 registration...
* `simulate`: Simulate the diffusion-weighted images of...
* `bv`: Open the Larmor viewer, with the given...

## `lar dti`

Fit the diffusion tensor (DTI) in every voxel and write its maps.

**Usage**:

```console
$ lar dti [OPTIONS] {volume} {output}
```

**Arguments**:

* `volume`: Path to the diffusion volume (.nii, .nii.gz, .mif, .mif.gz or .mih).  [required]
* `output`: Prefix of the output maps. &#x27;out&#x27; writes the fractional anisotropy to out_fa.nii.gz, the mean, axial and radial diffusivity in mm^2/s to out_md, out_ad and out_rd, the principal eigenvector to out_pe and a mask of the voxels whose tensor is not positive definite to out_npd, all as .nii.gz. The eigenvector is in world coordinates (RAS), not in the voxel frame of the image.  [required]

**Options**:

* `--bvals <str>`: Path to the b-values file. It is not needed for a .mif file with a gradient table in its header; given with such a file, --bvals and --bvecs replace that table.
* `--bvecs <str>`: Path to the b-vectors file. It is not needed for a .mif file with a gradient table in its header; given with such a file, --bvals and --bvecs replace that table.
* `--bvec-frame <fsl|world>`: Coordinate frame of the b-vectors. &#x27;fsl&#x27; means the frame of FSL bvecs files, along the image axes with x flipped when the affine determinant is positive, and &#x27;world&#x27; means scanner coordinates (RAS).  [default: fsl]
* `-m, --mask <str>`: Brain mask; voxels outside it are not fitted. Without it, a mask is computed from the mean b = 0 image.
* `--fit <ols|wls>`: Fitting method. &#x27;ols&#x27; is ordinary least squares, and &#x27;wls&#x27; is iterated weighted least squares, weighted by the predicted signal.  [default: ols]
* `--force`: Overwrite existing output files. Without it, the command stops before any work when an output exists.
* `--nt <int>`: Number of threads; -1 uses all CPU cores.  [default: -1]
* `--quiet`: Show only errors.
* `--info`: Show progress and informational messages.
* `--debug`: Show debug messages and full tracebacks.
* `--help`: Show this message and exit.

## `lar dki`

Fit diffusion kurtosis imaging (DKI) to multi-shell data and write its maps.

**Usage**:

```console
$ lar dki [OPTIONS] {volume} {output}
```

**Arguments**:

* `volume`: Path to the diffusion volume (.nii, .nii.gz, .mif, .mif.gz or .mih).  [required]
* `output`: Prefix of the output maps. &#x27;out&#x27; writes the tensor maps of lar dti (out_fa, out_md, out_ad, out_rd, out_pe and out_npd) and the mean, axial and radial kurtosis and the kurtosis anisotropy to out_mk, out_ak, out_rk and out_ka, all as .nii.gz.  [required]

**Options**:

* `--bvals <str>`: Path to the b-values file. It is not needed for a .mif file with a gradient table in its header; given with such a file, --bvals and --bvecs replace that table.
* `--bvecs <str>`: Path to the b-vectors file. It is not needed for a .mif file with a gradient table in its header; given with such a file, --bvals and --bvecs replace that table.
* `--bvec-frame <fsl|world>`: Coordinate frame of the b-vectors. &#x27;fsl&#x27; means the frame of FSL bvecs files, along the image axes with x flipped when the affine determinant is positive, and &#x27;world&#x27; means scanner coordinates (RAS).  [default: fsl]
* `-m, --mask <str>`: Brain mask; voxels outside it are not fitted. Without it, a mask is computed from the mean b = 0 image.
* `--fit <ols|wls>`: Fitting method. &#x27;wls&#x27; is iterated weighted least squares, weighted by the predicted signal, and &#x27;ols&#x27; is ordinary least squares.  [default: wls]
* `--unconstrained`: Fit without constraints. By default, the diffusivity D(n) and the kurtosis K(n) along every direction n are kept within D(n) &gt;= 0 and 0 &lt;= K(n) &lt;= 3 / (b_max D(n)).
* `--force`: Overwrite existing output files. Without it, the command stops before any work when an output exists.
* `--nt <int>`: Number of threads; -1 uses all CPU cores.  [default: -1]
* `--quiet`: Show only errors.
* `--info`: Show progress and informational messages.
* `--debug`: Show debug messages and full tracebacks.
* `--help`: Show this message and exit.

## `lar denoise`

Denoise a DWI volume with MP-PCA and estimate its noise level.

MP-PCA (Marchenko-Pastur principal component analysis) removes, in each
patch of voxels, the signal components that cannot be told apart from noise.

**Usage**:

```console
$ lar denoise [OPTIONS] {volume} {output}
```

**Arguments**:

* `volume`: Path to the diffusion volume (.nii, .nii.gz, .mif, .mif.gz or .mih).  [required]
* `output`: Path of the denoised volume. A .mif output keeps the gradient table of a .mif input; any other output has none.  [required]

**Options**:

* `--noise <str>`: Also write the noise map, the estimated noise standard deviation (sigma) of each voxel, to this path.
* `-m, --mask <str>`: Denoise only the voxels inside this mask and set the others to 0. Without it, every voxel is denoised.
* `--patch-size <int>`: Width in voxels, odd, of the cubic patch denoised around each voxel. Defaults to the smallest odd width whose cube is at least the number of volumes (5 for up to 125 volumes, 7 for up to 343).
* `--force`: Overwrite existing output files. Without it, the command stops before any work when an output exists.
* `--nt <int>`: Number of threads; -1 uses all CPU cores.  [default: -1]
* `--quiet`: Show only errors.
* `--info`: Show progress and informational messages.
* `--debug`: Show debug messages and full tracebacks.
* `--help`: Show this message and exit.

## `lar mask`

Compute a binary brain mask from the mean b = 0 image of a DWI volume.

**Usage**:

```console
$ lar mask [OPTIONS] {volume} {output}
```

**Arguments**:

* `volume`: Path to the diffusion volume (.nii, .nii.gz, .mif, .mif.gz or .mih).  [required]
* `output`: Path of the mask image to write.  [required]

**Options**:

* `--bvals <str>`: Path to the b-values file. It is not needed for a .mif file with a gradient table in its header; given with such a file, --bvals and --bvecs replace that table.
* `--bvecs <str>`: Path to the b-vectors file. It is not needed for a .mif file with a gradient table in its header; given with such a file, --bvals and --bvecs replace that table.
* `--bvec-frame <fsl|world>`: Coordinate frame of the b-vectors. &#x27;fsl&#x27; means the frame of FSL bvecs files, along the image axes with x flipped when the affine determinant is positive, and &#x27;world&#x27; means scanner coordinates (RAS).  [default: fsl]
* `--force`: Overwrite existing output files. Without it, the command stops before any work when an output exists.
* `--quiet`: Show only errors.
* `--info`: Show progress and informational messages.
* `--debug`: Show debug messages and full tracebacks.
* `--help`: Show this message and exit.

## `lar degibbs`

Remove Gibbs ringing from an image by local sub-voxel shifts.

Each voxel is resampled at the sub-voxel shift that minimises the ringing
around it.

**Usage**:

```console
$ lar degibbs [OPTIONS] {volume} {output}
```

**Arguments**:

* `volume`: Image to correct, 3D or 4D (.nii, .nii.gz, .mif, .mif.gz or .mih).  [required]
* `output`: Path of the corrected image.  [required]

**Options**:

* `--slice-axis <int range>`: Voxel axis (0, 1 or 2) along which the slices were acquired. The ringing is removed within each slice, along the other two axes.  [default: 2; 0&lt;=x&lt;=2]
* `--shifts <int range>`: Number of sub-voxel shifts tried on each side of zero, spaced 1 / (2 x shifts) voxel apart.  [default: 20; x&gt;=1]
* `--window <int int>...`: Distances in voxels of the nearest and the farthest neighbours over which the ringing around each voxel is measured.  [default: 1, 3]
* `--force`: Overwrite existing output files. Without it, the command stops before any work when an output exists.
* `--nt <int>`: Number of threads; -1 uses all CPU cores.  [default: -1]
* `--quiet`: Show only errors.
* `--info`: Show progress and informational messages.
* `--debug`: Show debug messages and full tracebacks.
* `--help`: Show this message and exit.

## `lar fod`

Estimate fibre orientation distributions (FODs) by spherical deconvolution.

**Usage**:

```console
$ lar fod [OPTIONS] {volume} {mask} {output}
```

**Arguments**:

* `volume`: Path to the diffusion volume (.nii, .nii.gz, .mif, .mif.gz or .mih).  [required]
* `mask`: Path to a brain mask on the grid of the diffusion volume.  [required]
* `output`: Prefix of the output files. &#x27;out&#x27; writes the FOD to out.nii.gz. A fit with a NODDI grey matter response (--gm-rf noddi, the default for mFOD) writes two FODs instead, to out_fod1.nii.gz and out_fod2.nii.gz. A multi-tissue fit also writes the tissue fractions to out_tissue_fraction.nii.gz.  [required]

**Options**:

* `--bvals <str>`: Path to the b-values file. It is not needed for a .mif file with a gradient table in its header; given with such a file, --bvals and --bvecs replace that table.
* `--bvecs <str>`: Path to the b-vectors file. It is not needed for a .mif file with a gradient table in its header; given with such a file, --bvals and --bvecs replace that table.
* `--method <grl|mfod|csd|msmt-csd>`: Spherical deconvolution method. GRL is generalised Richardson-Lucy, mFOD fits one FOD per anisotropic response, CSD is constrained spherical deconvolution of one shell and MSMT-CSD is multi-shell multi-tissue CSD.  [default: GRL]
* `--solver <drl|l2nnls|lasso>`: Solver of GRL and mFOD. dRL is damped Richardson-Lucy, L2NNLS is non-negative least squares with an L2 penalty, and LASSO has an L1 penalty. Defaults to dRL for GRL and L2NNLS for mFOD.
* `--reg <float>`: Weight of the penalty of the L2NNLS and LASSO solvers. Defaults to the solver&#x27;s own value.
* `--inner-shell-weight <float>`: Weight, above 0 and at most 1, of the measurements outside the highest shell in a multi-tissue GRL or mFOD fit; the highest shell has weight 1. Defaults to 0.2 for GRL and 1 for mFOD.
* `--bvec-frame <fsl|world>`: Coordinate frame of the b-vectors. &#x27;fsl&#x27; means the frame of FSL bvecs files, along the image axes with x flipped when the affine determinant is positive, and &#x27;world&#x27; means scanner coordinates (RAS).  [default: fsl]
* `--lmax <int>`: Maximum spherical harmonic (SH) degree of the FOD, even and at least 2. GRL and mFOD lower it when the highest shell has too few directions, and MSMT-CSD lowers it to that of the white matter response; CSD keeps it.  [default: 8]
* `--wm-rf <float float float>...`: Eigenvalues in mm^2/s of the white matter response of GRL and mFOD, the first along the fibre, such as 1.7e-3 0.2e-3 0.2e-3. Without it, the response is estimated from the data in the mask.
* `--gm-rf <float|none|noddi|auto>`: Grey matter response of GRL and mFOD: an isotropic diffusivity in mm^2/s, &#x27;none&#x27; to leave grey matter out, or &#x27;noddi&#x27; for an anisotropic NODDI response with its own FOD. &#x27;auto&#x27; means &#x27;noddi&#x27; for mFOD and 0.7e-3 for GRL.  [default: auto]
* `--csf-rf <float|none>`: Isotropic diffusivity in mm^2/s of the CSF response of GRL and mFOD, or &#x27;none&#x27; to leave CSF out.  [default: 0.003]
* `--mk <float>`: Mean kurtosis of the white matter response of GRL and mFOD. Defaults to the kurtosis estimated with the response (0 with fewer than two shells above b = 500), or to 0 with --wm-rf.
* `--response <file>`: White matter response for CSD, a text file of zonal SH coefficients with one row per shell, as MRtrix3&#x27;s dwi2response writes it. Without it, the response is estimated from the single-fibre voxels in the mask.
* `--wm-response <file>`: White matter response for MSMT-CSD, a text file with one row per shell, as MRtrix3&#x27;s dwi2response writes it. Without any of the three responses, all three are estimated from the data in the mask; --gm-response and --csf-response need --wm-response.
* `--gm-response <file>`: Grey matter response for MSMT-CSD, a text file with one row per shell, as MRtrix3&#x27;s dwi2response writes it. Without any of the three responses, all three are estimated from the data in the mask; --gm-response and --csf-response need --wm-response.
* `--csf-response <file>`: CSF response for MSMT-CSD, a text file with one row per shell, as MRtrix3&#x27;s dwi2response writes it. Without any of the three responses, all three are estimated from the data in the mask; --gm-response and --csf-response need --wm-response.
* `--basis <tournier|descoteaux|leemans>`: SH basis to write the FOD in. The file does not record it, so give the same basis to the commands that read the FOD.  [default: tournier]
* `--force`: Overwrite existing output files. Without it, the command stops before any work when an output exists.
* `--nt <int>`: Number of threads; -1 uses all CPU cores.  [default: -1]
* `--quiet`: Show only errors.
* `--info`: Show progress and informational messages.
* `--debug`: Show debug messages and full tracebacks.
* `--help`: Show this message and exit.

## `lar info`

Print a short summary of an image or a tractogram.

For an image, the summary comes from the header: its shape, voxel size,
orientation and affine, and the shells of a MIF gradient table. For a
tractogram, the whole file is read to count its streamlines and points and
list the data stored with them.

**Usage**:

```console
$ lar info [OPTIONS] {path}
```

**Arguments**:

* `path`: Path to an image or tractogram file.  [required]

**Options**:

* `--quiet`: Show only errors.
* `--info`: Show progress and informational messages.
* `--debug`: Show debug messages and full tracebacks.
* `--help`: Show this message and exit.

## `lar convert`

Convert an image or a tractogram to another file format of the same kind.

**Usage**:

```console
$ lar convert [OPTIONS] {input_path} {output_path}
```

**Arguments**:

* `input_path`: Path to the image or tractogram to convert, or a MATLAB .mat file with an MRIToolkit result (a DTI or DKI fit, a GRL or mFOD result, or tractography).  [required]
* `output_path`: Path of the converted file, of the same kind (image or tractogram) as the input; its extension sets the format. A .mat input also writes files next to it, with the same extension. From a DTI or DKI fit, &#x27;out.nii.gz&#x27; receives the DWI, and out_fa, out_md, out_ad, out_rd, out_pe, out_dt and, for DKI, out_kt receive the maps. From a GRL or mFOD result, &#x27;out.nii.gz&#x27; receives the FOD in the tournier basis (out_fod1, out_fod2 and so on for several FODs), and out_tissue_fraction and out_rss receive the tissue fractions and the residual sum of squares.  [required]

**Options**:

* `--bvals <str>`: b-values file of the input image, given with --bvecs. The gradient table goes into the header of a .mif output, and to --bvals-out and --bvecs-out if they are given.
* `--bvecs <str>`: b-vectors file of the input image, given with --bvals.
* `--bvec-frame <fsl|world>`: Coordinate frame of the b-vectors. &#x27;fsl&#x27; means the frame of FSL bvecs files, along the image axes with x flipped when the affine determinant is positive, and &#x27;world&#x27; means scanner coordinates (RAS).  [default: fsl]
* `--bvals-out <path>`: Write the b-values of the gradient table to this file, in FSL format. Give it with --bvecs-out.
* `--bvecs-out <path>`: Write the b-vectors of the gradient table to this file, in FSL format. Give it with --bvals-out.
* `--reference <path>`: Image whose grid a .trk output is written for, or that a .trk input is checked against. It is needed for a .trk output unless the input is a .trk, and for MRIToolkit tracts and GRL or mFOD results (.mat), which store no grid; for these, give the fit (.mat) they came from, or a NIfTI or MIF file of its image.
* `--to-ras`: Reorient an image to the closest RAS orientation by permuting and flipping its voxel axes; every voxel keeps its world position.
* `--vtk-space <RAS|LPS>`: Space of the points of a .vtk or .fib input, RAS or LPS (x and y negated); LPS is converted to RAS. Defaults to the SPACE= tag of the file&#x27;s title line, or to RAS for a file without one.
* `--force`: Overwrite existing output files. Without it, the command stops before any work when an output exists.
* `--quiet`: Show only errors.
* `--info`: Show progress and informational messages.
* `--debug`: Show debug messages and full tracebacks.
* `--help`: Show this message and exit.

## `lar sh-convert`

Convert an FOD image from one spherical harmonic (SH) basis to another.

**Usage**:

```console
$ lar sh-convert [OPTIONS] {input_path} {output_path}
```

**Arguments**:

* `input_path`: Path to the FOD image, with its spherical harmonic (SH) coefficients along the 4th axis.  [required]
* `output_path`: Path of the converted FOD image.  [required]

**Options**:

* `--from <tournier|descoteaux|leemans>`: SH basis the input is stored in, which the file does not record.  [required]
* `--to <tournier|descoteaux|leemans>`: SH basis to convert to.  [required]
* `--force`: Overwrite existing output files. Without it, the command stops before any work when an output exists.
* `--quiet`: Show only errors.
* `--info`: Show progress and informational messages.
* `--debug`: Show debug messages and full tracebacks.
* `--help`: Show this message and exit.

## `lar peaks`

Find the peaks (local maxima) of an FOD image in every voxel.

**Usage**:

```console
$ lar peaks [OPTIONS] {fod_path} {output_path}
```

**Arguments**:

* `fod_path`: Path to the FOD image, with its spherical harmonic (SH) coefficients along the 4th axis. An FOD that stores all SH degrees, not only the even ones, is read as asymmetric.  [required]
* `output_path`: Path of the peaks image. Each peak takes three volumes, the x, y and z of its direction in world coordinates (RAS), scaled by its amplitude, largest peak first; unused volumes are NaN. For an asymmetric FOD, a peak and its opposite are two peaks, and the sign of a direction matters.  [required]

**Options**:

* `-m, --mask <path>`: Only find peaks inside this mask; voxels outside it are NaN.
* `--basis <tournier|descoteaux|leemans>`: SH basis the FOD is stored in, which the file does not record.  [default: tournier]
* `--num <int>`: Maximum number of peaks per voxel. Defaults to 3, or to 6 for an asymmetric FOD.
* `--relative-threshold <float>`: Drop peaks whose amplitude is below this fraction, from 0 to 1, of the voxel&#x27;s largest peak. Defaults to 0.2 for GRL and mFOD FODs and 0.1 for CSD and MSMT-CSD ones (see --fod-method).
* `--fod-method <grl|mfod|csd|msmt-csd>`: Method that fitted the FOD, as given to lar fod --method, which the file does not record. It sets the default --relative-threshold. Give CSD for FODs from other software.  [default: GRL]
* `--threshold <float>`: Drop peaks whose amplitude is below this value.  [default: 0.0]
* `--force`: Overwrite existing output files. Without it, the command stops before any work when an output exists.
* `--nt <int>`: Number of threads; -1 uses all CPU cores.  [default: -1]
* `--quiet`: Show only errors.
* `--info`: Show progress and informational messages.
* `--debug`: Show debug messages and full tracebacks.
* `--help`: Show this message and exit.

## `lar track`

Track streamlines through an FOD image.

**Usage**:

```console
$ lar track [OPTIONS] {fod_path} {output_path}
```

**Arguments**:

* `fod_path`: Path to the FOD image, with its spherical harmonic (SH) coefficients along the 4th axis. An FOD that stores all SH degrees, not only the even ones, is read as asymmetric.  [required]
* `output_path`: Path of the tractogram, .tck or .trk, written while tracking runs. A .trk file also stores why each streamline ended.  [required]

**Options**:

* `--algorithm <det|prob>`: Tracking algorithm. &#x27;det&#x27; (deterministic) follows the FOD peak nearest the current direction, and &#x27;prob&#x27; (probabilistic) draws each new direction at random from the FOD.  [default: det]
* `--seed-image <path>`: Start streamlines at random positions inside this mask until --select streamlines are accepted. Give exactly one of --seed-image, --seed-random-per-voxel and --seed-file.
* `--seed-random-per-voxel <path int>...`: Takes a mask image and a number N, and starts streamlines from N random positions in every voxel of the mask.
* `--seed-file <path>`: Start streamlines from the seeds in a file written by --output-seeds. With the same --seed, it tracks the same streamlines again.
* `--select <int>`: Track until this many streamlines are accepted. It is required with --seed-image; without it, the other seed options track every seed.
* `-m, --mask <path>`: Tracking mask. A streamline stops where it leaves the mask, and a point is inside where the mask&#x27;s nearest voxel is at least 0.5.
* `--unidirectional`: Track from each seed in one direction only, not both.
* `--step-size <float>`: Step size in mm. Defaults to 0.1 x the voxel size.
* `--angle <float>`: Maximum angle between two consecutive steps, in degrees. The same angle allows tighter curves at a smaller step. Defaults to 15 for prob, and for det to the angle that allows curves down to a radius of 0.65 x the voxel size, 2 asin(step / (1.3 x voxel)), about 8.8 degrees at the default step.
* `--min-length <float>`: Reject streamlines shorter than this length, in mm. Defaults to 5 x the voxel size.
* `--max-length <float>`: Stop streamlines when they reach this length, in mm. Defaults to 100 x the voxel size.
* `--cutoff <str>`: FOD amplitude below which a streamline stops, a number or &#x27;auto&#x27;. &#x27;auto&#x27; is 0.2 x the mean largest FOD peak of a fixed random sample of up to 5000 voxels, inside the tracking mask if one is given.  [default: auto]
* `--basis <tournier|descoteaux|leemans>`: SH basis the FOD is stored in, which the file does not record.  [default: tournier]
* `--seed <int>`: Seed of the random number generator, from 0 to 2^64 - 1. The same seed gives the same tractogram, whatever the number of threads. Without it, a seed is drawn at random; --info shows it and --json saves it.
* `--json <path>`: Save a record of the run as JSON: every parameter used, the counts of seeds and streamlines, and the software versions.
* `--output-seeds <path>`: Save the seed point of each accepted streamline, in the order of the output, to a file that --seed-file reads.
* `--force`: Overwrite existing output files. Without it, the command stops before any work when an output exists.
* `--nt <int>`: Number of threads; -1 uses all CPU cores.  [default: -1]
* `--quiet`: Show only errors.
* `--info`: Show progress and informational messages.
* `--debug`: Show debug messages and full tracebacks.
* `--help`: Show this message and exit.

## `lar tract-edit`

Select, crop and resample the streamlines of one or more tractograms.

**Usage**:

```console
$ lar tract-edit [OPTIONS] {INPUT... OUTPUT}
```

**Arguments**:

* `INPUT... OUTPUT`: One or more input tractograms, joined in the order given, then the output tractogram. Each file can be .tck, .trk, .vtk or .fib, also gzipped (.gz); an input can also be MRIToolkit tracts (.mat).  [required]

**Options**:

* `--include <str>`: Keep the streamlines that pass through this region. A region is a sphere x,y,z,r in mm (world coordinates) or a mask image; repeat the option for several regions.
* `--include-ordered <str>`: Keep the streamlines that pass through these regions in the order given. A region is a sphere x,y,z,r in mm (world coordinates) or a mask image; repeat the option for several regions.
* `--exclude <str>`: Drop the streamlines that pass through this region. A region is a sphere x,y,z,r in mm (world coordinates) or a mask image; repeat the option for several regions.
* `--within <str>`: Keep the streamlines that stay entirely inside this region. A region is a sphere x,y,z,r in mm (world coordinates) or a mask image; repeat the option for several regions.
* `-m, --mask <str>`: Crop the streamlines to their parts inside these regions. A region is a sphere x,y,z,r in mm (world coordinates) or a mask image; repeat the option for several regions.
* `--ends-only`: Test only the two endpoints against --include, --include-ordered, --exclude and --within.
* `--inverse`: Invert the selection: keep the streamlines that fail at least one length, weight or region test, and crop them to their parts outside --mask.
* `--min-length <float>`: Keep streamlines at least this long, in mm.
* `--max-length <float>`: Keep streamlines at most this long, in mm.
* `--min-weight <float>`: Keep streamlines with at least this weight. The weights come from --weights-in or from the weight property of .trk inputs.
* `--max-weight <float>`: Keep streamlines with at most this weight.
* `--skip <int>`: Drop the first N selected streamlines.
* `--every <int>`: Keep every N-th selected streamline, starting with the first one after --skip.
* `--number <int>`: Keep the first N selected streamlines.
* `--subsample <int>`: Keep N selected streamlines chosen at random. It cannot be combined with --number.
* `--seed <int>`: Seed of the random choice of --subsample, and valid only with it. The same seed gives the same subset.
* `--step-size <float>`: Resample the output streamlines to this step size, in mm. It cannot be combined with --num-points.
* `--num-points <int>`: Resample the output streamlines to this many points each, at least 2.
* `--weights-in <path>`: Text file with one weight per streamline of a single input, such as the output of tcksift2. It replaces the weight property of a .trk input.
* `--weights-out <path>`: Write the weights of the output streamlines to this file.
* `--tsf-in <path>`: TSF file with one value per point of a single input. The values are named after the file name without its extension. Repeat the option for several files.
* `--tsf-out <str>`: Name of the per-point values to write for the output streamlines, a --tsf-in name or a .trk scalar. They go to &lt;output&gt;_&lt;name&gt;.tsf, where &lt;output&gt; is the output path without its extension. Repeat the option for several names.
* `--reference <path>`: Image whose grid a .trk output is written for; it is needed unless every input is a .trk. MRIToolkit tracts (.mat) store no grid and need it too; for them, give the fit (.mat) they came from, or a NIfTI or MIF file of its image.
* `--lazy`: Read, edit and write the streamlines in batches, for files too large for memory. It works with uncompressed .tck and .trk files only, and not with --subsample or TSF files.
* `--force`: Overwrite existing output files. Without it, the command stops before any work when an output exists.
* `--quiet`: Show only errors.
* `--info`: Show progress and informational messages.
* `--debug`: Show debug messages and full tracebacks.
* `--help`: Show this message and exit.

## `lar tract-profile`

Write the profile of maps (FA, MD and so on) along a bundle as a CSV table.

**Usage**:

```console
$ lar tract-profile [OPTIONS] {BUNDLE MAP... OUTPUT}
```

**Arguments**:

* `BUNDLE MAP... OUTPUT`: The bundle (.tck, .trk, .vtk or .fib, also gzipped), then one or more maps (.nii, .nii.gz, .mif, .mif.gz or .mih), then the output table (.csv). Each map&#x27;s column in the table is named after its file name.  [required]

**Options**:

* `--num-points <int>`: Number of points, at least 2, at which the profile samples the bundle.  [default: 100]
* `--weighting <gaussian|mean>`: How the streamlines&#x27; values are averaged at each profile point: &#x27;gaussian&#x27; weights each streamline by its distance from the bundle&#x27;s core, and &#x27;mean&#x27; weights all streamlines the same.  [default: gaussian]
* `--start <str>`: A point x,y,z in mm (world coordinates) near the end of the bundle where the profile starts. By default, the profile starts at the left, posterior or inferior end along the bundle&#x27;s main axis.
* `--tsf-prefix <str>`: Also write TSF files with one value per point of the bundle. &lt;prefix&gt;_segment.tsf holds the index of the profile point each point belongs to, and &lt;prefix&gt;_&lt;map&gt;.tsf holds the map&#x27;s profile value there.
* `--reference <path>`: Image whose grid MRIToolkit tracts (.mat) are read on, since they store none; give the fit (.mat) they came from, or a NIfTI or MIF file of its image.
* `--force`: Overwrite existing output files. Without it, the command stops before any work when an output exists.
* `--nt <int>`: Number of threads; -1 uses all CPU cores.  [default: -1]
* `--quiet`: Show only errors.
* `--info`: Show progress and informational messages.
* `--debug`: Show debug messages and full tracebacks.
* `--help`: Show this message and exit.

## `lar tract-density`

Compute a track density image, the number of streamlines in each voxel.

**Usage**:

```console
$ lar tract-density [OPTIONS] {tractogram} {reference} {output}
```

**Arguments**:

* `tractogram`: Input tractogram (.tck, .trk, .vtk, .fib, each also .gz).  [required]
* `reference`: Image whose grid the density is computed on (.nii, .nii.gz, .mif, .mif.gz, .mih); only its header is read.  [required]
* `output`: Path of the density image (.nii, .nii.gz, .mif, .mif.gz, .mih).  [required]

**Options**:

* `--voxel-size <str>`: Voxel size of the output in mm, one value or three as x,y,z. The output grid then covers the reference&#x27;s field of view at this voxel size.
* `--ends-only`: Count the two endpoints of the streamlines in each voxel instead (endpoint density).
* `--weights-in <path>`: Text file with one weight per streamline, such as the output of tcksift2. Each streamline then adds its weight instead of 1.
* `--upsample <int>`: Number of pieces each segment between two points of a streamline is split into, at least 1 (1 adds no points). By default, each segment gets the fewest pieces that are shorter than a third of the smallest voxel size. Not used with --ends-only.
* `--lazy`: Read the tractogram in batches, for files too large for memory. It works with uncompressed .tck and .trk files only.
* `--nt <int>`: Number of threads; -1 uses all CPU cores.  [default: -1]
* `--force`: Overwrite existing output files. Without it, the command stops before any work when an output exists.
* `--quiet`: Show only errors.
* `--info`: Show progress and informational messages.
* `--debug`: Show debug messages and full tracebacks.
* `--help`: Show this message and exit.

## `lar connectome`

Count the streamlines between each pair of regions (a structural connectome).

**Usage**:

```console
$ lar connectome [OPTIONS] {tractogram} {parcellation} {output}
```

**Arguments**:

* `tractogram`: Input tractogram (.tck, .trk, .vtk, .fib, each also .gz).  [required]
* `parcellation`: Parcellation image of integer region labels, with 0 for no region (.nii, .nii.gz, .mif, .mif.gz or .mih). The matrix has a row for every label from 1 to the largest, so relabel sparse labels first.  [required]
* `output`: Path of the matrix. A .csv file is comma-separated, with the region names in the first row and column; any other extension gives a space-separated file with the names on a comment line.  [required]

**Options**:

* `--lut <path>`: Lookup table, in FreeSurfer or MRtrix layout, whose names label the rows and columns. Without it, they are labelled with the label numbers.
* `--assignment <radial|end-voxel>`: How each streamline endpoint is assigned to a region. &#x27;radial&#x27; takes the labelled voxel whose centre is nearest the endpoint, if it is within --radius (the endpoint&#x27;s own voxel included), and no region otherwise; &#x27;end-voxel&#x27; takes the endpoint&#x27;s own voxel only.  [default: radial]
* `--radius <float>`: Search radius of --assignment radial, in mm. Defaults to 4. It cannot be combined with --assignment end-voxel.
* `--weights-in <path>`: Text file with one weight per streamline, such as the output of tcksift2. Each entry is then the sum of the weights instead of the number of streamlines.
* `--mean-length`: Make each entry the mean length in mm of the streamlines between the two regions, weighted by --weights-in if given. It cannot be combined with --mean-values.
* `--mean-values <path>`: File with one value per streamline, in the format of a weights file. Each entry is then the mean of these values over the streamlines between the two regions, weighted by --weights-in if given.
* `--symmetric`: Write the full symmetric matrix. By default, only the upper triangle is filled and the lower triangle is 0.
* `--zero-diagonal`: Set the diagonal (streamlines within one region) to 0.
* `--assignments-out <path>`: Write the two regions assigned to the endpoints of each streamline to this file, one streamline per line.
* `--nt <int>`: Number of threads; -1 uses all CPU cores.  [default: -1]
* `--force`: Overwrite existing output files. Without it, the command stops before any work when an output exists.
* `--quiet`: Show only errors.
* `--info`: Show progress and informational messages.
* `--debug`: Show debug messages and full tracebacks.
* `--help`: Show this message and exit.

## `lar transform`

Apply an ANTs, FSL or MRtrix3 registration to an image, tracts or surface.

An image is resampled onto the reference grid. The points of a tractogram or
surface are moved into the same space.

**Usage**:

```console
$ lar transform [OPTIONS] {input_path} {output_path}
```

**Arguments**:

* `input_path`: File to bring into the reference space, an image, a tractogram (.tck, .trk, .vtk or .fib) or a surface (GIfTI .gii or FreeSurfer).  [required]
* `output_path`: Path of the output file, of the same kind as the input.  [required]

**Options**:

* `-t, --transform <str>`: A transform file. Repeat the option for several transforms, in the order in which they map points from the reference space, so the first one applies to the reference grid; for an ANTs registration, this is -t 1Warp.nii.gz -t 0GenericAffine.mat. The files can be ANTs affines (.mat, .txt) and warps (5D NIfTI), FLIRT matrices (.mat), FSL warp fields, and MRtrix3 linear transforms (.txt) and deformation fields. A warp can be given with its inverse as &#x27;warp.nii.gz,inverse_warp.nii.gz&#x27;; otherwise it is inverted numerically where needed.  [required]
* `--reference <path>`: An image of the reference space, usually the fixed image of the registration. Images are resampled onto its grid, and a tractogram or surface takes it as its reference grid. It is needed for an image input and for a .trk output.
* `--inverse`: Apply the transforms the other way: bring the input from the reference space into the space of the registration&#x27;s moving image.
* `--interp <nearest|linear|cubic>`: Interpolation used to resample an image: &#x27;nearest&#x27; (for labels), &#x27;linear&#x27; or &#x27;cubic&#x27; (B-spline).  [default: linear]
* `--reorient <peaks|tensor>`: Reorient the directions stored in an image. &#x27;peaks&#x27; is for a peaks image as lar peaks writes it and needs --interp nearest, and &#x27;tensor&#x27; is for a diffusion tensor image of the 6 elements xx, xy, xz, yy, yz and zz.
* `--format <ants|fsl|mrtrix>`: Tool that wrote the transforms, for files that do not tell, such as a 4D NIfTI warp or a 4x4 text matrix not named .mat and without an MRtrix3 command_history comment, which can come from FSL or MRtrix3.
* `--warp-type <deformation|displacement|relative|absolute>`: How the warp fields store where each voxel comes from. MRtrix3 fields store positions (&#x27;deformation&#x27;, the default) or offsets (&#x27;displacement&#x27;), and FSL fields relative offsets (&#x27;relative&#x27;, the default) or absolute positions (&#x27;absolute&#x27;). The type is checked against the field&#x27;s values.
* `--fsl-input <path>`: Input image of the FLIRT or FNIRT run, for FSL transforms. Defaults to the input file when it is an image, or to --reference with --inverse.
* `--fsl-reference <path>`: Reference image of the FLIRT run, for FLIRT matrices. Defaults to --reference, or to the input file with --inverse.
* `--force`: Overwrite existing output files. Without it, the command stops before any work when an output exists.
* `--quiet`: Show only errors.
* `--info`: Show progress and informational messages.
* `--debug`: Show debug messages and full tracebacks.
* `--help`: Show this message and exit.

## `lar simulate`

Simulate the diffusion-weighted images of a phantom.

**Usage**:

```console
$ lar simulate [OPTIONS] {phantom_path} {output_path}
```

**Arguments**:

* `phantom_path`: Path to the phantom file (.phantom.json), which describes a voxel grid, its tissues, fibre bundles and a scan protocol. It can also be the name of a phantom that ships with Larmor (crossing), used when no file has that name.  [required]
* `output_path`: Output folder, created if needed. It receives the DWI with its bvals and bvecs, the ground truth (tissue and bundle fractions, fibre directions and masks), the bundles as .tck files, a copy of the phantom file and simulation.json, which together are enough to run the simulation again.  [required]

**Options**:

* `--shell <str>`: A shell as B:N, its b-value B in s/mm^2 and its number of directions N, such as 1000:32. Repeat the option for several shells. The shells replace the gradients of the phantom file.
* `--num-b0 <int>`: Number of b = 0 volumes added to the --shell gradients. Defaults to 1.
* `--bvals <path>`: Path to a b-values file, given with --bvecs. The two files replace the gradients of the phantom file.
* `--bvecs <path>`: Path to a b-vectors file, given with --bvals.
* `--bvec-frame <fsl|world>`: Coordinate frame of the --bvecs file. &#x27;fsl&#x27; means the frame of FSL bvecs files, along the axes of the phantom&#x27;s grid with x flipped when the affine determinant is positive, and &#x27;world&#x27; means RAS coordinates. Defaults to fsl.
* `--snr <float>`: Add Rician noise with this signal-to-noise ratio, relative to the b = 0 signal of the --noise-reference tissue. Without it, no noise is added.
* `--noise-reference <str>`: Name of the phantom tissue whose b = 0 signal --snr refers to.  [default: WM]
* `--seed <int>`: Seed of the random numbers of the noise, the --shell directions and the bundles that have no seed of their own. The same seed gives the same result.
* `--force`: Overwrite existing output files. Without it, the command stops before any work when an output exists.
* `--nt <int>`: Number of threads; -1 uses all CPU cores.  [default: -1]
* `--quiet`: Show only errors.
* `--info`: Show progress and informational messages.
* `--debug`: Show debug messages and full tracebacks.
* `--help`: Show this message and exit.

## `lar bv`

Open the Larmor viewer, with the given files loaded.

**Usage**:

```console
$ lar bv [OPTIONS] [files]...
```

**Arguments**:

* `files...`: Files to open, sorted by extension into tractograms (.tck, .trk, .vtk, .fib), a session (.lar.json) and volumes (any other file). Other .json files are skipped.

**Options**:

* `-v, --volumes <path>`: Volume files, such as a T1 or an FA map. Repeat the option for several files.
* `-o, --overlays <path>`: Overlay files shown over the volumes, such as masks, maps and parcellations. Repeat the option for several files.
* `--tracts <path>`: Tractogram files. Repeat the option for several files.
* `-f, --fods <path>`: FOD files, drawn as glyphs in the 2D views, with their SH coefficients along the 4th axis. They are read in the tournier basis unless another basis is chosen in the viewer. Repeat the option for several files.
* `-p, --peaks <path>`: Peaks files as lar peaks writes them, drawn as lines in the 2D views. Repeat the option for several files.
* `--surfaces <path>`: Surface files, GIfTI (.gii) or FreeSurfer. A FreeSurfer surface such as lh.white is read together with its lh.curv and ../label/lh.cortex.label when they exist. Repeat the option for several files.
* `--max-streamlines <int range>`: A tractogram with more streamlines is reduced to a random subset of this size when read, the same subset every time. Defaults to the read limit in the viewer&#x27;s Settings (1,000,000 unless changed).  [x&gt;=0]
* `-s, --session <path>`: Session saved from the viewer (File &gt; Save Session). It is opened first, and the files of the other options are added to it.
* `--phantom <path>`: Phantom file (.phantom.json) to open in the Simulate workspace, to edit and simulate.
* `--simulation <path>`: Simulation folder written by lar simulate. The viewer opens its DWI, tissue masks, ground-truth peaks and bundles, and shows its phantom in the Simulate workspace.
* `--mcp`: Let AI agents control the viewer through the Model Context Protocol (MCP), served at http://127.0.0.1:&lt;port&gt;/mcp. The URL and the access token are printed.
* `--mcp-port <int range>`: Port for --mcp; giving it also turns on --mcp. Defaults to the port in the viewer&#x27;s Settings (8765 unless changed).  [1024&lt;=x&lt;=65535]
* `--install-launcher`: Add a launcher for the viewer of this Larmor installation to the desktop (the application menu on Linux, ~/Applications on macOS, the Start menu on Windows), then exit.
* `--uninstall-launcher`: Remove what --install-launcher added, then exit.
* `--quiet`: Show only errors.
* `--info`: Show progress and informational messages.
* `--debug`: Show debug messages and full tracebacks.
* `--help`: Show this message and exit.
