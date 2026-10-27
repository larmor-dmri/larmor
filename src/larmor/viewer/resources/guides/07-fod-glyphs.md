# FOD glyphs

A fibre orientation distribution (FOD) describes, for each voxel, how much fibre runs in each direction. It is
stored as a 4D image of spherical harmonic (SH) coefficients, such as MRtrix's `wmfod.mif`. Open it in the
**FODs** category. The 2D views then draw one glyph per voxel of the slice, a shape whose distance from the voxel
centre in each direction is the FOD's amplitude in that direction, coloured by direction (red left to right,
green front to back, blue up and down).

![FOD glyphs over the FA in a coronal slice, zoomed in on the left hemisphere's white matter, where bundles cross](images/fod-glyphs.png)

## Symmetric and asymmetric

A symmetric FOD has the same lobe at both ends of each fibre and uses the even SH degrees only, which takes
(lmax + 1)(lmax + 2) / 2 coefficients, 45 for lmax 8. An asymmetric FOD can have a lobe at one end of a fibre
only, and uses all degrees, which takes (lmax + 1)^2 coefficients, 81 for lmax 8. The viewer tells the two apart
by the number of coefficients, and refuses a file with neither number.

## Their options

- **Basis** is the SH basis of the coefficients, which the file does not record: tournier (MRtrix's basis, the
  default), descoteaux, or leemans (ExploreDTI's basis). A wrong basis gives wrong glyphs without an error.
- **Scale** sets the glyphs' size. At 1, the largest lobe reaches the edge of its voxel; with whole_image
  scaling, a lobe of the image's high amplitude does.
- **Scaling** sets what a glyph's size is relative to. With **whole_image** (the default), all glyphs are scaled
  by the image's high amplitude, the 90th percentile of the voxels' largest amplitudes, so a glyph's size shows
  its amplitude. With **per_voxel**, each glyph is scaled on its own, so round FODs look as large as sharp ones.
- **Threshold** is a share of the image's high amplitude. A voxel whose largest amplitude is below it gets no
  glyph. It starts at 0.05, which leaves out the small noise of the background.
- **Minimum anisotropy** hides the FODs that are rounder than this. Anisotropy is 0 for a round FOD and near 1
  for a sharp one, so raising the minimum hides the round FODs of grey matter and CSF.
- **Sphere detail** sets how smooth each glyph is, 8 by default. Higher is smoother and slower.
- **Colouring** colours the glyphs by direction, or in the single colour that the Colour option sets.

## One FOD, large

To look at one voxel's FOD closely, select the FOD in the scene tree and press **Show at the cursor** in its
options, or open the inspector's Cursor tab. The tab shows the FOD at the crosshair as one large, smooth glyph,
with the R/L, A/P and S/I axes, and follows the crosshair as you click around. Drag to turn it, use the wheel to
zoom, and press **Reset view** to start again.

The glyph uses the FOD's basis and colouring, but not the threshold, the minimum anisotropy or the scale, so it
always shows the whole FOD. Outside the FOD's grid it says that there is no FOD there. The tab follows the FOD
selected in the scene tree, and keeps showing it while another kind of item is selected.
