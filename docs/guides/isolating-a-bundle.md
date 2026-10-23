---
myst:
  html_meta:
    description: "How to keep one bundle of a whole-brain tractogram with Larmor, by selecting streamlines with include and exclude regions, from the command line and from Python."
---

# Isolating a bundle

A whole-brain tractogram holds every bundle at once. To keep one of them, select the streamlines that pass through
regions the bundle passes through (include regions) and drop those that pass through regions it does not reach
(exclude regions). `lar tract-edit` does this for TCK, TRK and VTK files.

## Regions

A region is either a sphere or a mask image. A sphere is written `x,y,z,r`, with its centre and radius in mm in
world coordinates, which are the coordinates the viewer shows at the cursor. A mask image (`.nii`, `.nii.gz`,
`.mif`) can be on any grid. A streamline passes through a region when at least one of its points is inside it.

## An include and an exclude region

The left corticospinal tract, for example, runs from the motor cortex down through the brainstem. The command
below keeps it with two include regions, a mask of the left precentral gyrus and a sphere in the left cerebral
peduncle, and one exclude region, a mask of the right hemisphere:

```bash
lar tract-edit tracks.tck left_cst.tck \
    --include lh_precentral.nii.gz \
    --include -12,-20,-14,6 \
    --exclude rh_hemisphere.nii.gz
```

A streamline is kept when it passes through every `--include` region and through no `--exclude` region. The command
reports how many streamlines it read and wrote. Open the result over a T1 image to check it:

```bash
lar bv -v t1.nii.gz --tracts left_cst.tck
```

## Refining the selection

- `--ends-only` tests only the two endpoints of each streamline, for bundles defined by where they start and end.
- `--include-ordered`, repeated, requires the regions to be passed in the order given.
- `--within`, which can be repeated, keeps only the streamlines whose points all lie inside a region, such as a
  white matter mask.
- `--min-length` and `--max-length` drop streamlines that are too short or too long, in mm.
- `--mask` cuts the kept streamlines to the parts inside a region, such as a brain mask.
- `--inverse` keeps the streamlines that the selection drops.

The same edit from Python:

```python
from larmor.tractography import EditOptions, edit_files, load_region

options = EditOptions(
    include=[load_region("lh_precentral.nii.gz"), load_region("-12,-20,-14,6")],
    exclude=[load_region("rh_hemisphere.nii.gz")],
)
read, written = edit_files(["tracks.tck"], "left_cst.tck", options)
```

`edit` runs the same edit on a `Tractogram` in memory. The [command-line reference](../cli.md#lar-tract-edit)
lists every option.
