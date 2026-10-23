"""Finds the surfaces of masks and parcellation labels, in voxel coordinates."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import find_objects, gaussian_filter
from skimage.measure import marching_cubes

from larmor.types import Volume

# Blur of a label's mask before its surface is found, in voxels, enough to round
# off the voxel steps without losing a thin structure.
LABEL_SMOOTHING = 1.0


@dataclass(frozen=True)
class LabelMesh:
    """Holds a triangle mesh, with vertices and normals (voxels, float32) and faces."""

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray


def mask_surface(data: np.ndarray, level: float) -> LabelMesh | None:
    """Returns the surface where 3D `data` crosses `level`; None if it never does."""
    lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
    if not lo < level < hi:
        return None
    # A border below the level closes the surface where the mask meets the edges.
    padded = np.pad(
        np.nan_to_num(data.astype(np.float32), nan=lo), 1, constant_values=lo
    )
    vertices, faces, normals, _ = marching_cubes(padded, level)
    return LabelMesh(
        (vertices - 1).astype(np.float32),
        # Wound the other way round, as pygfx takes a counter-clockwise face as
        # facing the viewer.
        np.ascontiguousarray(faces[:, ::-1], dtype=np.int32),
        normals.astype(np.float32),
    )


def label_surfaces(
    volume: Volume, index: int, labels: Iterable[int]
) -> dict[int, LabelMesh | None]:
    """Returns the surface of each label (scaled units) in a volume of a parcellation.

    Each label's mask is cut to the label's box and lightly blurred first; a label
    the volume does not hold has None.

    Returns:
        The surfaces by label.
    """
    wanted = sorted(labels)
    if not wanted:
        return {}
    data = volume.data[..., index] if volume.data.ndim == 4 else volume.data
    if np.issubdtype(data.dtype, np.integer) and (volume.slope, volume.intercept) == (
        1.0,
        0.0,
    ):
        image = np.asarray(data)
    else:
        scaled = np.nan_to_num(data * volume.slope + volume.intercept)
        image = np.rint(scaled).astype(np.int32)
    positive = [label for label in wanted if label > 0]
    # One pass over the image finds every positive label's box.
    boxes = find_objects(image, max_label=max(positive)) if positive else []
    surfaces: dict[int, LabelMesh | None] = {}
    for label in wanted:
        if label > 0:
            box = boxes[label - 1]
        else:  # find_objects skips them; rare enough to search the whole image
            where = np.nonzero(image == label)
            box = (
                tuple(slice(int(w.min()), int(w.max()) + 1) for w in where)
                if where[0].size
                else None
            )
        surfaces[label] = None if box is None else _label_surface(image, label, box)
    return surfaces


def _label_surface(
    image: np.ndarray, label: int, box: tuple[slice, ...]
) -> LabelMesh | None:
    """Returns the surface of one label, found within its box."""
    # Two voxels of border, room for the blur and a closed surface at the edges.
    mask = np.pad((image[box] == label).astype(np.float32), 2)
    smooth = gaussian_filter(mask, LABEL_SMOOTHING)
    if smooth.max() > 0.5:  # a label of a voxel or two would vanish in the blur
        mask = smooth
    surface = mask_surface(mask, 0.5)
    if surface is None:
        return None
    offset = np.array([b.start - 2 for b in box], dtype=np.float32)
    return LabelMesh(surface.vertices + offset, surface.faces, surface.normals)


@dataclass(frozen=True)
class ColouredSurface:
    """Holds several surfaces in one mesh, with a colour (RGB, 0 to 1) per vertex."""

    surface: LabelMesh
    colours: np.ndarray


def merge_surfaces(
    surfaces: Sequence[LabelMesh], colours: Sequence[tuple[float, float, float]]
) -> ColouredSurface | None:
    """Returns one mesh of all the surfaces, each in its colour; None for none."""
    if not surfaces:
        return None
    starts = np.cumsum([0] + [len(s.vertices) for s in surfaces[:-1]])
    vertices = np.concatenate([s.vertices for s in surfaces])
    normals = np.concatenate([s.normals for s in surfaces])
    faces = np.concatenate(
        [s.faces + np.int32(start) for s, start in zip(surfaces, starts, strict=True)]
    )
    rgb = np.concatenate(
        [
            np.broadcast_to(np.asarray(c, np.float32), (len(s.vertices), 3))
            for s, c in zip(surfaces, colours, strict=True)
        ]
    )
    return ColouredSurface(LabelMesh(vertices, faces, normals), rgb)
