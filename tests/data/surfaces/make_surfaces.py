"""Generates the small surface files in tests/data/surfaces/, from numbers alone.

A subject folder in FreeSurfer's layout, written with nibabel's FreeSurfer and
GIfTI writers, so the format tests run without FreeSurfer:

- `surf/lh.white`: a bumpy ellipsoid (an icosphere subdivided 3 times: 642
  vertices, 1280 faces) on the left, with a valid volume-geometry footer of a
  conformed 256^3 LIA volume with cras 0; `surf/rh.white`: its mirror image,
  without a footer; `surf/lh.sphere`: the icosphere at radius 100 mm.
- `surf/lh.curv`, `surf/lh.sulc`, `surf/lh.thickness`, `surf/rh.curv`: smooth
  per-vertex values; the thickness is 0 on the medial wall, as FreeSurfer's.
- `label/lh.cortex.label`, `label/lh.Medial_wall.label`: ASCII labels that
  split the left hemisphere, the medial wall being its 64 vertices nearest the
  midline, with their white-surface coordinates.
- `gifti/lh.white.surf.gii`: the left white surface as GIfTI (base64, gzip);
  `gifti/lh.labels.label.gii`: two named label arrays with a label table.

Run from the repository root:

    uv run python tests/data/surfaces/make_surfaces.py
"""

import pathlib
from collections import OrderedDict

import nibabel as nib
import numpy as np
from nibabel.freesurfer.io import write_geometry, write_morph_data
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable

OUT = pathlib.Path("tests/data/surfaces")
N_MEDIAL = 64
# nibabel would otherwise write the user name and the time into each surface.
STAMP = "created by tests/data/surfaces/make_surfaces.py"


def _middle(
    vertices: list[np.ndarray], middles: dict[tuple[int, int], int], a: int, b: int
) -> int:
    """The index of the edge's midpoint on the unit sphere, added once per edge."""
    key = (min(a, b), max(a, b))
    if key not in middles:
        m = vertices[a] + vertices[b]
        vertices.append(m / np.linalg.norm(m))
        middles[key] = len(vertices) - 1
    return middles[key]


def icosphere(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit icosphere: the icosahedron with each face split in four, order times."""
    t = (1 + 5**0.5) / 2
    points = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
        [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]  # fmt: skip
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9],
        [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2],
        [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
        [8, 6, 7], [9, 8, 1],
    ]  # fmt: skip
    vertices = [np.array(p, dtype=np.float64) / np.linalg.norm(p) for p in points]
    for _ in range(order):
        middles: dict[tuple[int, int], int] = {}
        split = []
        for a, b, c in faces:
            ab, bc, ca = (
                _middle(vertices, middles, *e) for e in ((a, b), (b, c), (c, a))
            )
            split += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = split
    return np.array(vertices), np.array(faces, dtype=np.int32)


def write_label(path: pathlib.Path, indices: np.ndarray, coords: np.ndarray) -> None:
    """An ASCII FreeSurfer label: a header, the count, then index x y z value."""
    lines = ["#!ascii label , from subject synth vox2ras=TkReg", str(len(indices))]
    lines += [
        f"{i}  {x:.3f}  {y:.3f}  {z:.3f} 0.0000000000"
        for i, (x, y, z) in zip(indices, coords[indices], strict=True)
    ]
    path.write_text("\n".join(lines) + "\n")


unit, faces = icosphere(3)
x, y, z = unit.T
bumps = 1 + 0.08 * np.sin(5 * x) * np.cos(4 * y)
lh = (unit * [30.0, 60.0, 45.0] * bumps[:, None] + [-35.0, -15.0, 10.0]).astype(
    np.float32
)
rh = lh * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
sphere = (100 * unit).astype(np.float32)

# The medial wall: the vertices nearest the midline (largest x on the left).
medial = np.sort(np.argsort(-lh[:, 0], kind="stable")[:N_MEDIAL])
cortex = np.setdiff1d(np.arange(len(lh)), medial)

# A conformed LIA volume with cras 0, as for fsaverage.
volume_info = OrderedDict(
    head=np.array([2, 0, 20], dtype=np.int32),
    valid="1  # volume info valid",
    filename="../mri/orig.mgz",
    volume=np.array([256, 256, 256]),
    voxelsize=np.array([1.0, 1.0, 1.0]),
    xras=np.array([-1.0, 0.0, 0.0]),
    yras=np.array([0.0, 0.0, -1.0]),
    zras=np.array([0.0, 1.0, 0.0]),
    cras=np.array([0.0, 0.0, 0.0]),
)

surf = OUT / "surf"
label = OUT / "label"
gifti = OUT / "gifti"
for folder in (surf, label, gifti):
    folder.mkdir(parents=True, exist_ok=True)

write_geometry(
    surf / "lh.white", lh, faces, create_stamp=STAMP, volume_info=volume_info
)
write_geometry(surf / "rh.white", rh, faces, create_stamp=STAMP)
write_geometry(surf / "lh.sphere", sphere, faces, create_stamp=STAMP)

curv = (0.2 * np.sin(6 * x) * np.cos(5 * z)).astype(np.float32)
thickness = (2.5 + 0.5 * z).astype(np.float32)
thickness[medial] = 0.0
write_morph_data(surf / "lh.curv", curv, fnum=len(faces))
write_morph_data(surf / "lh.sulc", (4 * y).astype(np.float32), fnum=len(faces))
write_morph_data(surf / "lh.thickness", thickness, fnum=len(faces))
write_morph_data(surf / "rh.curv", curv[::-1].copy(), fnum=len(faces))

write_label(label / "lh.cortex.label", cortex, lh)
write_label(label / "lh.Medial_wall.label", medial, lh)

points = GiftiDataArray(
    lh, intent="NIFTI_INTENT_POINTSET", datatype="NIFTI_TYPE_FLOAT32"
)
triangles = GiftiDataArray(
    faces, intent="NIFTI_INTENT_TRIANGLE", datatype="NIFTI_TYPE_INT32"
)
nib.save(GiftiImage(darrays=[points, triangles]), gifti / "lh.white.surf.gii")

table = GiftiLabelTable()
for key, (name, rgba) in enumerate(
    [
        ("???", (0.0, 0.0, 0.0, 0.0)),
        ("frontal", (0.9, 0.3, 0.3, 1.0)),
        ("parietal", (0.3, 0.6, 0.9, 1.0)),
        ("occipital", (0.4, 0.8, 0.4, 1.0)),
    ]
):
    entry = GiftiLabel(key, *rgba)
    entry.label = name
    table.labels.append(entry)
lobes = np.where(y > 0.3, 1, np.where(y > -0.4, 2, 3)).astype(np.int32)
lobes[medial] = 0
sides = (z > 0).astype(np.int32) + 1
arrays = [
    GiftiDataArray(
        data,
        intent="NIFTI_INTENT_LABEL",
        datatype="NIFTI_TYPE_INT32",
        meta={"Name": name},
    )
    for name, data in (("Lobes", lobes), ("Upper and lower", sides))
]
nib.save(GiftiImage(darrays=arrays, labeltable=table), gifti / "lh.labels.label.gii")

for path in sorted(OUT.rglob("*")):
    if path.is_file() and path.suffix != ".py":
        print(path, path.stat().st_size, "bytes")  # noqa: T201 (script output)
