# MATLAB files

The viewer opens MATLAB `.mat` files, such as the results MRIToolkit and ExploreDTI save. One file holds many
values, so the viewer always asks what to load from it, and as what.

**File > Open...**, with the **MATLAB files** filter, opens the import dialog. It offers four ways to load a
file, and selects the one that fits the file. The ones that do not fit are greyed out, with the reason as their
tip.

- **Load as tensor fit** loads a DTI or DKI fit (`DT`, `FA`, `eigval`, `FE`, `DWI`, `hdr`). FA, MD and the
  eigenvalues are added as volumes, the first eigenvector as a volume with DTI colours modulated by the FA, and
  the DWI as a volume too.
- **Load as FOD fit** loads the result of a GRL or mFOD fit, a struct with `FOD`, `fractions` and `RSS`. Each FOD
  goes into FODs, in the MRtrix basis, and the fractions and residuals go into Volumes.
- **Load as tractography** loads tracts (`Tracts`) as a tractogram in world coordinates.
- **Load manually** shows a table of every variable, struct field and cell, with its size, MATLAB class and
  bytes, and you choose for each row what to load it as.

In the manual table, each row can become a Volume, an Overlay, FODs (SH coefficients), Peaks, a Tractogram (a
cell of N x 3 point arrays), Values per point, a Surface (a struct with `vertices` and 1-based `faces`) or
Nothing. Values per point give one value per point of a loaded tractogram, which is then coloured by them. A row
whose shape cannot be that type says why, and Load stays off until every row fits. Peaks and FODs are taken to be
in MRIToolkit's frame, as its own results are, and are turned to world coordinates. Peaks are read as x, y, z per
peak, and FODs either as 300 amplitudes on MRIToolkit's directions or as SH coefficients in the MRtrix basis.

## Reference grid

FOD results and tracts store no image grid, so the viewer needs a reference image to place them. The dialog fills
in the fit beside the file (`dwi.mat` first) when there is one. Otherwise, choose the fit's `.mat` or a NIfTI of
the same image. Images are read in MRIToolkit's layout (its first two axes swapped and reversed) and placed on
that grid.

## Sessions and Python

Sessions store, for each item, the variable and the type it came from, so a session reopens without the dialog.
From Python and for agents, `viewer.load_matlab` takes the same choices:

```python
viewer.load_matlab("dwi.mat", MatlabPreset.TENSOR_FIT)
viewer.load_matlab("tracks.mat", MatlabPreset.TRACTOGRAPHY, reference="dwi.mat")
viewer.load_matlab("dwi.mat", MatlabPreset.MANUAL, variables={"FA": LoadAs.OVERLAY})
```

`lar convert` writes the same results to NIfTI, TCK or TRK for other tools.
