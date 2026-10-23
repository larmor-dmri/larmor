# Examples

Each example is a Python script that runs on its own and is explained by its comments. The tests run every
example, so a broken example fails CI. The scripts follow the format of sphinx-gallery and make up the gallery of the
documentation website: there is one folder per section, and each folder has a `GALLERY_HEADER.txt`
(reStructuredText) that introduces it.

Run an example from the repository root with `uv run python examples/<folder>/<name>.py`. Most examples build
synthetic data and need no files. The examples on a brain download the simulated data of the ISMRM 2015 Tractography
Challenge (about 41 MB) on first use, and two download the CFIN multi-shell dataset; see
[Example data](../docs/example-data.md) for the datasets, their licences and the cache folder. These examples carry
the line `# needs: example-data`, and the tests and the website run them only when `LARMOR_DATA_DIR` is set.

## Preprocessing

| Example | Shows |
| --- | --- |
| [dwi_denoise_mppca.py](preprocessing/dwi_denoise_mppca.py) | Denoising a DWI with MP-PCA and reading the noise level |
| [gibbs_ringing_removal.py](preprocessing/gibbs_ringing_removal.py) | Denoising a DWI with MP-PCA, then removing its Gibbs ringing |

## Models

| Example | Shows |
| --- | --- |
| [dti_synthetic_tensor.py](models/dti_synthetic_tensor.py) | Fitting DTI and reading fractional anisotropy |
| [dki_synthetic_kurtosis.py](models/dki_synthetic_kurtosis.py) | Fitting DKI to two shells and reading mean kurtosis |
| [dti_ismrm.py](models/dti_ismrm.py) | DTI of the ISMRM 2015 simulated brain, with direction colours modulated by FA |
| [dki_real_data.py](models/dki_real_data.py) | Fitting DKI to a real multi-shell scan, downloaded with `larmor.data` |

## Spherical deconvolution and peaks

| Example | Shows |
| --- | --- |
| [fod_fit_synthetic.py](sd/fod_fit_synthetic.py) | Fitting FODs to the simulated signal of one, two and three fibres |
| [peaks_synthetic.py](sd/peaks_synthetic.py) | Finding the peaks of FODs of fibres crossing at several angles |
| [fods_ismrm.py](sd/fods_ismrm.py) | FODs and peaks of the ISMRM 2015 simulated brain, and asymmetric FODs made ad hoc for illustration |
| [msmt_csd_cfin.py](sd/msmt_csd_cfin.py) | Multi-tissue FODs (MSMT-CSD) of a real multi-shell scan, with the white matter, grey matter and CSF fractions |

## Tractography

| Example | Shows |
| --- | --- |
| [tracking_synthetic.py](tractography/tracking_synthetic.py) | Tracking streamlines through two crossing bundles |
| [tracking_ismrm.py](tractography/tracking_ismrm.py) | Whole-brain tractography of the ISMRM 2015 simulated brain, drawn as tubes |
| [tractogram_batch_writing.py](tractography/tractogram_batch_writing.py) | Writing a TCK file batch by batch, as tractography does |
| [track_density_synthetic.py](tractography/track_density_synthetic.py) | Track density images on a reference grid and a finer one, weighted and of endpoints |
| [connectome_synthetic.py](tractography/connectome_synthetic.py) | A structural connectome: endpoints assigned to regions, counts and mean lengths, saved with names |
| [registration_atlas_region.py](tractography/registration_atlas_region.py) | Bringing an atlas region into diffusion space with an ANTs registration, and selecting a bundle with it |
| [tract_profiles.py](tractography/tract_profiles.py) | The FA profile along a bundle, printed and drawn on the bundle |

## Simulation

| Example | Shows |
| --- | --- |
| [simulate_crossing_bundles.py](simulation/simulate_crossing_bundles.py) | Simulating two bundles that cross at 90 degrees with different tissue parameters, and saving the images with their ground truth |
| [signal_breakdown.py](simulation/signal_breakdown.py) | The signal of one voxel split into its factors and drawn as stacked bars, before and after a longer echo time |
| [pipeline_ground_truth.py](simulation/pipeline_ground_truth.py) | Checking FOD peaks fitted on a simulated crossing against the true fibre directions, with the angular error |

## Files

| Example | Shows |
| --- | --- |
| [file_io.py](files/file_io.py) | Reading and writing images, gradient tables and tractograms, and converting between formats |
| [mritoolkit_files.py](files/mritoolkit_files.py) | Reading an MRIToolkit fit and tract from MATLAB files, and refitting the tensor |

## The viewer

These examples drive the viewer from Python and save screenshots of its views as PNG files, and one WebP video,
in the current folder. They draw off screen, with no window, so they need the `[viewer]` extra and a GPU or a
software renderer, but no display. They build their images in memory and need no data files, except the brain
example, which downloads the ISMRM 2015 data. The sessions example writes its own files to a temporary folder.

| Example | Shows |
| --- | --- |
| [viewer_volumes_and_overlays.py](viewer/viewer_volumes_and_overlays.py) | A volume with an overlay: colormap, threshold, opacity, colour bars |
| [viewer_parcellation.py](viewer/viewer_parcellation.py) | A parcellation shown as labels, with built-in colours and with a lookup table |
| [viewer_4d_volumes.py](viewer/viewer_4d_volumes.py) | Stepping through the volumes of a 4D image |
| [viewer_2d_views.py](viewer/viewer_2d_views.py) | The crosshair and its readout, slices, zoom and pan, the display convention |
| [viewer_tractograms.py](viewer/viewer_tractograms.py) | Tractograms: the slab in the 2D views, colouring, a subset in the 3D view |
| [viewer_tubes.py](viewer/viewer_tubes.py) | Streamlines drawn as lit tubes in the 3D view, coloured by direction and in one colour |
| [viewer_peaks.py](viewer/viewer_peaks.py) | FOD peaks as lines in the 2D views, with lengths by amplitude and without |
| [viewer_surfaces.py](viewer/viewer_surfaces.py) | A surface shaded in the 3D view and coloured by curvature, and its cut line in the 2D views |
| [viewer_3d_view.py](viewer/viewer_3d_view.py) | The 3D view: slices, blended volume rendering, a mask's surface, camera positions |
| [viewer_fod_glyphs.py](viewer/viewer_fod_glyphs.py) | FOD glyphs: crossing fibres, an asymmetric FOD, hiding round FODs by their anisotropy |
| [viewer_dti.py](viewer/viewer_dti.py) | DTI colours from a principal eigenvector, plain and modulated by FA |
| [viewer_screenshot_modes.py](viewer/viewer_screenshot_modes.py) | The 2D views as separate images, and the 3D view turning as a WebP video |
| [viewer_sessions.py](viewer/viewer_sessions.py) | Saving a session, moving its project folder and opening it again |
| [viewer_real_brain.py](viewer/viewer_real_brain.py) | The ISMRM 2015 simulated brain: FA from DTI over the T1-like image |

Every screenshot an example saves needs an alt text, a short description of the image for readers who cannot see
it, in the `ALT_TEXT` dictionary of `docs/_ext/gallery_images.py`; without one, the website build fails.
