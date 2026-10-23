---
myst:
  html_meta:
    description: "How to read and write FreeSurfer and GIfTI surfaces and their values per vertex with Larmor, cut and join meshes, move a surface into diffusion space, and view it."
---

# Surfaces

Cortical surfaces, such as the white matter surface from FreeSurfer, can be shown with tractograms and brought
into diffusion space. In Larmor a {py:class}`~larmor.types.Surface` is a triangle mesh in world coordinates (RAS+,
in mm), with named values per vertex, such as curvature or thickness, in `vertex_data`. Larmor reads and writes
FreeSurfer surfaces (`lh.white`, `lh.pial`, ...) and GIfTI surfaces (`.surf.gii`).

## A surface and its values

A surface is built from its vertices and its faces. Each face lists the vertex indices of a triangle,
counter-clockwise as seen from outside, so that the normals point out. The example below builds an octahedron of
radius 20 mm with a thickness value per vertex, and saves it both as a FreeSurfer surface with a values file beside
it and as a GIfTI surface:

```python
import numpy as np

from larmor.io import save_surface
from larmor.io.formats import write_freesurfer_values
from larmor.types import Surface

vertices = 20 * np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float
)
top = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4]]
bottom = [[2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]
faces = np.array(top + bottom)
thickness = np.array([2.0, 2.5, 3.0, 2.2, 2.8, 2.4])
surface = Surface(vertices, faces, vertex_data={"thickness": thickness})

save_surface(surface, "lh.white")
write_freesurfer_values(thickness, "lh.thickness", num_faces=surface.num_faces)
save_surface(surface, "lh.white.surf.gii")
```

A FreeSurfer surface file holds the mesh only, so its values go into files of their own, such as `lh.thickness`,
while a GIfTI surface keeps the values in the same file. {py:func}`~larmor.io.load_surface` reads either kind. It
takes values files with `values` and adds each to `vertex_data` under the part of its file name after the
hemisphere, so `lh.thickness` becomes `thickness`:

```python
from larmor.io import load_surface

white = load_surface("lh.white", values="lh.thickness")
print(white.num_vertices, white.num_faces, sorted(white.vertex_data))
```

## Normals, cutting and joining

```python
from larmor.surfaces import concatenate_surfaces, keep_vertices, vertex_normals

normals = vertex_normals(white)
upper = keep_vertices(white, white.vertices[:, 2] >= 0)
both = concatenate_surfaces([white, load_surface("lh.white.surf.gii")])
print(upper.num_vertices, both.num_vertices)
```

{py:func}`~larmor.surfaces.vertex_normals` gives a unit normal per vertex, pointing out.
{py:func}`~larmor.surfaces.keep_vertices` keeps the vertices where a mask is True, with the faces between them. With
the cortex label read by {py:func}`~larmor.io.formats.read_freesurfer_label`, for example, it removes the medial
wall. {py:func}`~larmor.surfaces.concatenate_surfaces` joins surfaces into one mesh, such as the two hemispheres.

## Moving a surface into another space

To draw a surface over diffusion data, move it with a registration from the T1 image to the diffusion image,
computed by ANTs, FSL or MRtrix3. The example uses an MRtrix3 linear transform, a 4x4 matrix in world coordinates,
and writes one first. A 4x4 text matrix could also be a FLIRT matrix, so you must give the format with it:

```python
np.savetxt("t1_to_dwi.txt", [[1, 0, 0, 5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
```

::::{tab-set}
:sync-group: interface

:::{tab-item} Python
:sync: python
```python
from larmor.io import load_surface, save_surface
from larmor.registration import apply_transform, load_registration

transform = load_registration(["t1_to_dwi.txt"], format="mrtrix")
moved = apply_transform(load_surface("lh.white"), transform)
save_surface(moved, "lh.white.dwi.surf.gii")
```
:::

:::{tab-item} Command line
:sync: cli
```bash
lar transform lh.white lh.white.dwi.surf.gii -t t1_to_dwi.txt --format mrtrix
```
:::
::::

The transform files are given in the same way as for an image, and the vertices move so that the surface lands
where the registered image does.
[Bringing an atlas region into diffusion space](../gallery/tractography/registration_atlas_region.rst) shows a
registration on images.

## Viewing surfaces

<!-- not run: opens the viewer -->
```bash
lar bv --surfaces lh.white
```

The viewer draws a surface as a shaded mesh in 3D and as the line where each slice cuts it in 2D, and can colour it
by its values per vertex. A FreeSurfer surface opens with its `lh.curv` and `../label/lh.cortex.label` when these
exist.

## See also

- Gallery: [Surfaces in the viewer](../gallery/viewer/viewer_surfaces.rst).
- File formats: [Surfaces](../reference/file-formats.md#surfaces).
- API: {py:class}`~larmor.types.Surface`, {py:func}`~larmor.io.load_surface`, {py:func}`~larmor.io.save_surface`,
  {py:mod}`larmor.surfaces`, {py:func}`~larmor.registration.apply_transform`.
- Command: [`lar transform`](../cli.md#lar-transform).
