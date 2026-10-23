---
myst:
  html_meta:
    description: "How to sample maps such as FA along a bundle into a tract profile with Larmor, from the command line and from Python, and how to colour the bundle by its profile in the viewer."
---

# Tract profiles

A tract profile describes a bundle by the values of a map, such as FA or MD, at points along its length, so that
subjects or groups can be compared point by point rather than by one mean over the whole bundle (Yeatman et al.
2012; Chandio et al. 2020). It needs one bundle, such as one isolated with `lar tract-edit` (see
[Isolating a bundle](isolating-a-bundle.md)), and maps on any grid:

```bash
lar tract-profile left_cst.tck fa.nii.gz md.nii.gz left_cst_profile.csv
```

The output is a CSV table with one row per profile point, from the start of the bundle to its end. Each row holds
the point's index, its position on the bundle's centreline (`core_x`, `core_y`, `core_z`, in mm), and one column
per map, named after its file (`fa`, `md`). In Python, `tract_profile(bundle, {"fa": fa})` from
`larmor.tractography` returns the same profile.

## How it is computed

Larmor first makes all streamlines run the same way. By default the profile starts at the end of the bundle that is
left, posterior or inferior along the bundle's main axis, such as the brainstem end of a corticospinal tract or the
left end of the corpus callosum. With `--start x,y,z` (in mm), each streamline starts instead at its end nearer to
that point. When the main axis lies about halfway between two of x, y and z, a small change in the bundle can move
the start to the other end, and Larmor warns. In group studies, pass `--start` with the same anatomical point in
each subject, so that every profile starts at the same end.

The profile points are `--num-points` nodes (100 by default), evenly spaced along a centreline of the bundle.
Following Chandio et al. (2020), every point of every streamline is assigned to its nearest node, so node n holds
the same place along the bundle for every streamline. A streamline that covers only part of the bundle, because it
stopped early or the bundle fans out, adds to the nodes it reaches only, instead of being stretched over the whole
bundle.

Each map is sampled at the streamlines' points with trilinear interpolation, and a streamline's value at a node is
the mean of its values at the points assigned to that node. Points outside a map and non-finite values are left
out, so they do not pull the profile's ends towards 0, and a node where no streamline has a value is NaN. With a
map that changes along the bundle, the values of the first and last nodes lie about a quarter of the node spacing
inward.

The profile at a node is a weighted mean over the streamlines that reach it. With `--weighting gaussian`, the
default, streamlines near the bundle's core count most and stray ones little. Each streamline is weighted by
exp(-d^2 / 2), where d is its Mahalanobis distance from the mean of the streamlines' positions at that node. With
`--weighting mean`, all streamlines count the same.

## Colouring the bundle in the viewer

The viewer can colour streamlines by a TSF file (track scalar file), which holds one value per point of a
tractogram. With `--tsf-prefix out/left_cst`, `lar tract-profile` also writes such files for the bundle.
`out/left_cst_segment.tsf` gives, for each point, the index of the profile point it belongs to, and
`out/left_cst_fa.tsf`, one file per map, gives the profile's value there. To see them, open the bundle and choose
one of them as its TSF file:

```bash
lar bv -v fa.nii.gz --tracts left_cst.tck
```

## References

- Yeatman JD, Dougherty RF, Myall NJ, Wandell BA, Feldman HM. Tract profiles of white matter properties:
  automating fiber-tract quantification. PLoS One 7 (2012) e49790.
- Chandio BQ, Risacher SL, Pestilli F, et al. Bundle analytics, a computational framework for investigating the
  shapes and profiles of brain pathways across populations. Sci Rep 10 (2020) 17149.
