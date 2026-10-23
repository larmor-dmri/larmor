---
myst:
  html_meta:
    description: "The file formats Larmor reads and writes, such as NIfTI, MIF, TCK, TRK, GIfTI and FreeSurfer files, gradient tables, MATLAB files and registration transforms."
---

# File formats

This page lists the formats Larmor reads and writes, with the Python functions for each. The `load_` and `save_`
functions of {py:mod}`larmor.io`, and the `lar` commands, choose the format from the end of the file name. The
`read_` and `write_` functions of {py:mod}`larmor.io.formats` each read or write one format.

## Images

| Format | File names | Read | Write | Python |
| --- | --- | --- | --- | --- |
| NIfTI-1 and NIfTI-2 | `.nii`, `.nii.gz` | yes | yes (NIfTI-2 only when a dimension needs it) | {py:func}`~larmor.io.load_volume`, {py:func}`~larmor.io.save_volume`, {py:func}`~larmor.io.formats.read_nifti`, {py:func}`~larmor.io.formats.write_nifti` |
| MRtrix image (MIF) | `.mif`, `.mif.gz`, `.mih` with its `.dat` | yes | yes | {py:func}`~larmor.io.load_volume`, {py:func}`~larmor.io.save_volume`, {py:func}`~larmor.io.formats.read_mif`, {py:func}`~larmor.io.formats.write_mif` |

A MIF header can hold the gradient table (`dw_scheme`), which {py:func}`~larmor.io.load_dwi` reads with the image
and {py:func}`~larmor.io.save_volume` writes into it.

## Gradient tables

| Format | File names | Read | Write | Python |
| --- | --- | --- | --- | --- |
| FSL bvals and bvecs | any, often `.bval` and `.bvec` | yes | yes | {py:func}`~larmor.io.read_bvals_bvecs`, {py:func}`~larmor.io.write_bvals_bvecs` |
| MRtrix gradient table | any, often `.b` | yes | yes | {py:func}`~larmor.io.read_mrtrix_gradients`, {py:func}`~larmor.io.write_mrtrix_gradients` |
| b-matrix, one row per volume | any | yes | yes | {py:func}`~larmor.io.read_b_matrix`, {py:func}`~larmor.io.write_b_matrix` |
| MRtrix response function (zonal SH per shell) | any, often `.txt` | yes | no | {py:func}`~larmor.io.read_mrtrix_response` |

## Tractograms

| Format | File names | Read | Write | Python |
| --- | --- | --- | --- | --- |
| MRtrix tracks (TCK) | `.tck`, `.tck.gz` | yes | yes | {py:func}`~larmor.io.load_tractogram`, {py:func}`~larmor.io.save_tractogram`, {py:func}`~larmor.io.formats.read_tck`, {py:func}`~larmor.io.formats.write_tck` |
| TrackVis (TRK) | `.trk`, `.trk.gz` | yes | yes (version 2) | {py:func}`~larmor.io.load_tractogram`, {py:func}`~larmor.io.save_tractogram`, {py:func}`~larmor.io.formats.read_trk`, {py:func}`~larmor.io.formats.write_trk` |
| Legacy VTK polydata | `.vtk`, `.fib`, each also `.gz` | yes | yes | {py:func}`~larmor.io.load_tractogram`, {py:func}`~larmor.io.save_tractogram`, {py:func}`~larmor.io.formats.read_vtk`, {py:func}`~larmor.io.formats.write_vtk` |
| MRIToolkit tractography | `.mat` | yes | no | {py:func}`~larmor.io.load_tractogram`, {py:func}`~larmor.io.read_mritoolkit_tracts` |

Large uncompressed TCK and TRK files can be read and written batch by batch, with {py:func}`~larmor.io.iter_tractogram` and
{py:func}`~larmor.io.open_tractogram_writer`.

## Values along streamlines

| Format | File names | Read | Write | Python |
| --- | --- | --- | --- | --- |
| MRtrix track scalar file (TSF): one value per point | `.tsf` | yes | yes | {py:func}`~larmor.io.read_tsf`, {py:func}`~larmor.io.write_tsf` |
| Streamline weights, as tcksift2 writes them: one value per streamline | any text file | yes | yes | {py:func}`~larmor.io.read_streamline_weights`, {py:func}`~larmor.io.write_streamline_weights` |

## Surfaces

| Format | File names | Read | Write | Python |
| --- | --- | --- | --- | --- |
| GIfTI surface | `.gii`, such as `.surf.gii` | yes | yes | {py:func}`~larmor.io.load_surface`, {py:func}`~larmor.io.save_surface`, {py:func}`~larmor.io.formats.read_gifti_surface`, {py:func}`~larmor.io.formats.write_gifti_surface` |
| GIfTI values per vertex | `.shape.gii`, `.func.gii`, `.label.gii` | yes | yes (`.shape.gii`, `.func.gii`) | {py:func}`~larmor.io.formats.read_gifti_values`, {py:func}`~larmor.io.formats.write_gifti_values` |
| FreeSurfer surface | no extension, such as `lh.white`, `lh.pial` | yes | yes | {py:func}`~larmor.io.load_surface`, {py:func}`~larmor.io.save_surface`, {py:func}`~larmor.io.formats.read_freesurfer_surface`, {py:func}`~larmor.io.formats.write_freesurfer_surface` |
| FreeSurfer values per vertex ("new curv") | such as `lh.curv`, `lh.thickness` | yes | yes | {py:func}`~larmor.io.formats.read_freesurfer_values`, {py:func}`~larmor.io.formats.write_freesurfer_values` |
| FreeSurfer label | `.label`, such as `lh.cortex.label` | yes | no | {py:func}`~larmor.io.formats.read_freesurfer_label` |

## Registrations

Larmor reads transforms but does not write them. {py:func}`~larmor.registration.load_transform` works out the
format from the file where it can. It reads a text `.mat` file as a FLIRT matrix, and a text file with MRtrix3's
`# command_history:` comment, or with 3 rows of 4 numbers, as an MRtrix3 transform. Any other 4x4 text matrix
could be either, and the two map different coordinates, so you must give the format: `format="fsl"` or
`"mrtrix"` in Python, `--format` on the command line.

| Format | File names | Read | Write | Python |
| --- | --- | --- | --- | --- |
| ANTs (ITK) affine | binary `.mat` (such as `0GenericAffine.mat`), text `.txt`, `.tfm` | yes | no | {py:func}`~larmor.registration.read_ants_transform` |
| ANTs displacement field | NIfTI, such as `1Warp.nii.gz` | yes | no | {py:func}`~larmor.registration.read_ants_transform` |
| FSL FLIRT matrix | text, such as `.mat` from `-omat` | yes | no | {py:func}`~larmor.registration.read_flirt_transform` |
| FSL warp field | NIfTI (fnirt `--fout`, convertwarp, invwarp) | yes | no | {py:func}`~larmor.registration.read_fsl_warp` |
| MRtrix3 linear transform | text (mrregister, transformconvert) | yes | no | {py:func}`~larmor.registration.read_mrtrix_transform` |
| MRtrix3 deformation or displacement field | `.mif` or NIfTI | yes | no | {py:func}`~larmor.registration.read_mrtrix_warp` |

## MATLAB and MRIToolkit

| Format | File names | Read | Write | Python |
| --- | --- | --- | --- | --- |
| MATLAB file, versions 4, 5 to 7 and 7.3 | `.mat` | yes | no | {py:func}`~larmor.io.read_matlab` |
| MRIToolkit DTI or DKI fit | `.mat` | yes | no | {py:func}`~larmor.io.read_mritoolkit_fit`, {py:func}`~larmor.io.read_mritoolkit_map` |
| MRIToolkit image, its values as stored | `.mat` | yes | no | {py:func}`~larmor.io.read_mritoolkit_volume` |
| MRIToolkit FODs | `.mat` | yes | no | {py:func}`~larmor.sd.read_mritoolkit_fod`, {py:func}`~larmor.sd.read_mritoolkit_sh` |
| MRIToolkit peaks | `.mat` | yes | no | {py:func}`~larmor.io.read_mritoolkit_peaks` |

## Tables and other text files

| Format | File names | Read | Write | Python |
| --- | --- | --- | --- | --- |
| Lookup table, FreeSurfer or MRtrix layout | text, such as `FreeSurferColorLUT.txt` | yes | no | {py:func}`~larmor.io.read_lookup_table` |
| Connectome matrix with region names | `.csv`, or text separated by spaces | no | yes | {py:func}`~larmor.tractography.write_connectome` |
| Regions of each streamline's endpoints, as tck2connectome writes them | text | no | yes | {py:func}`~larmor.tractography.write_node_assignments` |
| Seeds, one row per seed | text | yes | yes | {py:func}`~larmor.tractography.read_seeds`, {py:func}`~larmor.tractography.write_seeds` |
| Tract profile, one row per point | `.csv` | no | yes | {py:meth}`TractProfile.write_csv <larmor.tractography.TractProfile.write_csv>` |
| BrainViewer session | `.lar.json` | yes | yes | {py:meth}`Viewer.load_session <larmor.viewer.Viewer.load_session>`, {py:meth}`Viewer.save_session <larmor.viewer.Viewer.save_session>` |
