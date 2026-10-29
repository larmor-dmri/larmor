# Dissecting a bundle

The Dissect workspace (Ctrl+3, Cmd+3 on macOS) takes you from a diffusion-weighted image, a FOD file or a
tractogram to a bundle: the streamlines of a whole-brain tractogram that pass through regions you draw. Its steps
are listed on the left, each with a tick once it is done and a line that says where it stands, and the page of the
step you are on is on the right, with the fields that matter, a **More** fold for the rest, and Back and Next at
the bottom.

## Start, FODs and Track

The **Start** step asks what you have. Starting from a tractogram reads it and goes straight to the Regions step;
the FODs and Track steps then stay in the list, greyed out and marked as skipped. Starting from a FOD file takes you
to the FODs step, where you choose the file's SH basis, which the file does not record, and load it. Starting from a
DWI opens it as a volume and takes you to the FODs step with the DWI filled in, and with its bvals and bvecs files
when they sit beside it with the same name.

The **FODs** step either loads a FOD file or estimates the FODs from the DWI. Estimating needs a brain mask, chosen
from the overlays and volumes loaded, or made with **New mask...**, and offers the method and the highest SH degree
on the page; the grey matter and CSF responses, response files for CSD and MSMT-CSD, the solver, its penalty weight,
the inner-shell weight and the frame of the bvecs file are under More. The defaults are those of `lar fod`, and the
same choices give the same FODs. The fit runs in the background with a progress bar and Cancel, and the FODs (and
the tissue fractions of a multi-tissue fit) are added as items.

The **Track** step tracks the whole brain through those FODs, with the library's defaults for every field left on
auto. Under More you choose where streamlines start, the whole seed mask until the number of streamlines is kept or
a number of seeds in every voxel, the seed and tracking masks, which default to the mask of the fit, and the random
seed. The run is kept with the tractogram it made, with every setting and the seed it drew, so that it can be run
again.

## Regions and Save

The **Regions** step lists the regions with a tick for those that filter the tractogram, and says how many
streamlines they keep. The Sphere ROI (O), Box ROI (B) and Label region (L) tools place regions in the views, and
regions are moved and resized only in this workspace; they are still drawn, and still filter, in View.
**Open regions...** adds the regions of a regions file saved earlier.

When an overlay shown as labels is loaded, such as a FreeSurfer `aparc+aseg`, the top of the Regions page shows a
**Parcellation** section that keeps streamlines by its labels. **From A to B** keeps the streamlines with one end
in the labels of set A and the other end in those of set B; an end counts for a label when a voxel of it lies within
the **End radius** (4 mm by default), so a streamline that stops just short of the grey matter still counts.
**Stay within** keeps the streamlines that never leave the labels of set A. Add a label with **+ label**, which
searches the names of the parcellation's lookup table, or pick the Label region tool (L) and click the label on a
slice; Shift+click removes it, and the button beside each set chooses the set the tool adds to. The labels chosen
are drawn on the slices and as surfaces in 3D, the count of kept streamlines follows every change, and the label
region combines with the spheres, boxes and masks like any other region.

## Making a mask

Every mask field in Dissect, and the Regions page, has a **New mask...** link that opens the mask sheet. On its
left you choose what the mask is made from: the brain mask of the DWI, as `lar mask` computes it from the mean b0,
with the b-values taken from the Start step; a threshold of an image, such as an FA map above 0.2; or labels of a
parcellation. The middle shows the mask on one slice of the image as you change the settings, with its number of
voxels and its volume, and scrolling moves through the slices. On the right you can clean the mask up, by
dilating or eroding it by a number of voxels, filling its holes and keeping only its largest piece, and combine it
with another mask by union, intersection or subtraction. **Create mask** adds it as an overlay and puts it in the
field the sheet was opened from; from the Regions page it is added as a mask region that filters the tractogram.
**Save...** also writes it to a file. Painting a mask by hand is not available yet.

The **Save** step writes, into the folder you choose and under the name you give, the kept streamlines as
`name.tck`, the regions as `name.regions.json`, and, when the tractogram was tracked here, the run as
`name_run.json`. The page lists every file it will write and asks before replacing one. It also lists the items the
dissection made that have no file yet, such as the FODs, the tissue fractions, the whole-brain tractogram and new
masks, each with a tick box; those the regions file or the run record name are always saved, since the files must
point to them.

To reproduce a bundle, open the whole-brain tractogram, start the Dissect workspace from it and use **Open
regions...** on the regions file: the same regions keep the same streamlines. The run record holds the input files
and the settings of the run, and `lar track` with those files, settings and `--seed` tracks the same whole-brain
tractogram again.

Every item Dissect makes can also be saved on its own: the FODs and Track pages have a **Save...** button once their
result exists, and File > Save As... saves the item selected in the scene tree. Saving a session while some items
have no file asks whether to save them next to the session first, as the session can only reopen items from files.
