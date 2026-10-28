"""Describes the detail windows, each a rectangle of a 2D view shown on its own."""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np

from larmor.types import Grid
from larmor.viewer.model.camera import Convention, Vec3, screen_axes
from larmor.viewer.model.item import VIEW_NAMES, ViewKind
from larmor.viewer.model.readout import closest_axis

#: The number that names a detail window in its viewer, as in `Detail.id`.
type DetailId = int
# A rectangle of voxels in a 2D view's plane: lowest i, lowest j, size along i and
# size along j, in the main volume's two voxel axes in that plane, lowest first.
type VoxelBox = tuple[int, int, int, int]

# The world axis across each 2D view's slice (x, y, z).
VIEW_AXIS: Final = {ViewKind.SAGITTAL: 0, ViewKind.CORONAL: 1, ViewKind.AXIAL: 2}
# The sphere detail of a detail window's FOD glyphs at first.
GLYPH_DETAIL: Final = 12
# The longer side of a detail window's picture at first, logical pixels.
DETAIL_SIZE: Final = 600
# The box the right-click menu opens, voxels on a side.
MENU_BOX: Final = 12


class FigureBackground(StrEnum):
    """The colour around the pictures of a detail's figure."""

    WHITE = "white"
    BLACK = "black"


class DetailOption(StrEnum):
    """The settings of a detail window, as `Viewer.set_detail` names them."""

    LINKED = "linked"  # follows the crosshair's slice
    VOXEL_GRID = "voxel_grid"  # draws the voxels' edges
    SCALE_BAR = "scale_bar"  # draws a bar of a round length in mm
    GLYPH_DETAIL = "glyph_detail"  # the sphere detail of its FOD glyphs, 2 to 16


@dataclass(frozen=True)
class Detail:
    """One detail window: a rectangle of a 2D view, drawn larger on its own.

    Attributes:
        id: The detail's id in its viewer.
        view: The 2D view it is a rectangle of.
        box: The rectangle, in the voxels of the main volume or, with none, of
            the first visible FOD (see `VoxelBox`).
        position: The world point (mm) whose slice it shows while not linked.
        linked: Whether it follows the crosshair's slice instead.
        voxel_grid: Whether the voxels' edges are drawn.
        scale_bar: Whether a scale bar is drawn.
        glyph_detail: The sphere detail of its FOD glyphs.
        size: Its picture's width and height in logical pixels.
    """

    id: DetailId
    view: ViewKind
    box: VoxelBox
    position: Vec3
    linked: bool = True
    voxel_grid: bool = True
    scale_bar: bool = True
    glyph_detail: int = GLYPH_DETAIL
    size: tuple[int, int] = (DETAIL_SIZE, DETAIL_SIZE)


def plane_axes(grid: Grid, view: ViewKind) -> tuple[int, int, int]:
    """Returns a grid's voxel axis across a 2D view, then its two axes in the plane.

    Returns:
        The axis across, then the lower and the higher in-plane axis.
    """
    across = closest_axis(grid, VIEW_AXIS[view])
    first, second = (a for a in range(3) if a != across)
    return across, first, second


def picture_size(grid: Grid, view: ViewKind, box: VoxelBox) -> tuple[int, int]:
    """Returns the size of a detail's picture, its longer side `DETAIL_SIZE`.

    The picture has the rectangle's shape in mm, so the voxels stay square.

    Returns:
        The width and height, logical pixels.
    """
    w, h = screen_millimetres(grid, view, box)
    scale = DETAIL_SIZE / max(w, h)
    return (max(round(w * scale), 1), max(round(h * scale), 1))


def box_millimetres(grid: Grid, view: ViewKind, box: VoxelBox) -> tuple[float, float]:
    """Returns a rectangle's sides in mm, along its first and second voxel axis."""
    _, first, second = plane_axes(grid, view)
    sizes = np.linalg.norm(grid.affine[:3, :3], axis=0)
    return (box[2] * float(sizes[first]), box[3] * float(sizes[second]))


def screen_millimetres(
    grid: Grid, view: ViewKind, box: VoxelBox
) -> tuple[float, float]:
    """Returns a rectangle's sides in mm, along the screen's right and up."""
    width, height = box_millimetres(grid, view, box)
    right, _ = screen_axes(view, Convention.RAS)
    _, first, _ = plane_axes(grid, view)
    # The first voxel axis runs along the screen's right or along its up.
    column = grid.affine[:3, first] / np.linalg.norm(grid.affine[:3, first])
    along_right = abs(float(column @ np.asarray(right))) > 0.5
    return (width, height) if along_right else (height, width)


def size_text(grid: Grid, view: ViewKind, box: VoxelBox) -> str:
    """Returns a rectangle's size as a notice says it, in voxels and in mm."""
    a, b = box_millimetres(grid, view, box)
    return f"{box[2]} x {box[3]} voxels, {a:.1f} x {b:.1f} mm"


def detail_field(
    grid: Grid, view: ViewKind, box: VoxelBox, position: Vec3
) -> tuple[Vec3, Vec3]:
    """Returns the world box a detail shows: its rectangle, at the voxels' edges.

    Args:
        grid: The main volume's grid.
        view: The 2D view.
        box: The rectangle of voxels.
        position: A point of the slice shown, mm.

    Returns:
        The lowest and highest corner, mm, as a 2D view's field of view takes it.
    """
    across, first, second = plane_axes(grid, view)
    inverse = np.linalg.inv(grid.affine)
    voxel = inverse[:3, :3] @ np.asarray(position, dtype=float) + inverse[:3, 3]
    corners = []
    for i in (box[0] - 0.5, box[0] + box[2] - 0.5):
        for j in (box[1] - 0.5, box[1] + box[3] - 0.5):
            for k in (voxel[across] - 0.5, voxel[across] + 0.5):
                point = np.zeros(3)
                point[first], point[second], point[across] = i, j, k
                corners.append(grid.affine[:3, :3] @ point + grid.affine[:3, 3])
    lo, hi = np.min(corners, axis=0), np.max(corners, axis=0)
    return (
        (float(lo[0]), float(lo[1]), float(lo[2])),
        (float(hi[0]), float(hi[1]), float(hi[2])),
    )


def voxel_box(
    grid: Grid,
    view: ViewKind,
    start: Vec3,
    end: Vec3,
    *,
    square: bool = False,
    from_centre: bool = False,
) -> VoxelBox:
    """Returns the rectangle of voxels a drag from start to end covers.

    Its sides lie on the voxels' edges, so it holds whole voxels, at least one.

    Args:
        grid: The main volume's grid.
        view: The 2D view dragged in.
        start: Where the drag began, world mm.
        end: Where it is now, world mm.
        square: Make the rectangle a square, as large as its longer side.
        from_centre: Grow it from start as its centre, rather than its corner.

    Returns:
        The rectangle; it may reach outside the grid.
    """
    _, first, second = plane_axes(grid, view)
    inverse = np.linalg.inv(grid.affine)
    a = inverse[:3, :3] @ np.asarray(start, dtype=float) + inverse[:3, 3]
    b = inverse[:3, :3] @ np.asarray(end, dtype=float) + inverse[:3, 3]
    a, b = a[[first, second]], b[[first, second]]
    reach = np.abs(b - a)
    if square:
        reach[:] = reach.max()
    sign = np.where(b >= a, 1.0, -1.0)
    if from_centre:
        low, high = a - reach, a + reach
    else:
        far = a + sign * reach
        low, high = np.minimum(a, far), np.maximum(a, far)
    # Voxel k spans k - 0.5 to k + 0.5: the edges are at whole numbers minus a half.
    lo_edge = np.floor(low + 0.5).astype(int)
    hi_edge = np.ceil(high + 0.5).astype(int)
    size = np.maximum(hi_edge - lo_edge, 1)
    if square:
        size[:] = size.max()
    return (int(lo_edge[0]), int(lo_edge[1]), int(size[0]), int(size[1]))


def centred_box(grid: Grid, view: ViewKind, centre: Vec3, voxels: int) -> VoxelBox:
    """Returns a square of voxels on a side centred on the voxel at a point."""
    _, first, second = plane_axes(grid, view)
    inverse = np.linalg.inv(grid.affine)
    voxel = inverse[:3, :3] @ np.asarray(centre, dtype=float) + inverse[:3, 3]
    nearest = np.floor(voxel + 0.5).astype(int)
    half = voxels // 2
    return (int(nearest[first]) - half, int(nearest[second]) - half, voxels, voxels)


def clipped_box(grid: Grid, view: ViewKind, box: VoxelBox) -> VoxelBox | None:
    """Returns the part of a rectangle inside the grid; None if none is."""
    _, first, second = plane_axes(grid, view)
    shape = grid.shape
    i0, j0 = max(box[0], 0), max(box[1], 0)
    i1 = min(box[0] + box[2], shape[first])
    j1 = min(box[1] + box[3], shape[second])
    if i1 <= i0 or j1 <= j0:
        return None
    return (i0, j0, i1 - i0, j1 - j0)


def regrid_box(
    old: Grid, new: Grid, view: ViewKind, box: VoxelBox, position: Vec3
) -> VoxelBox | None:
    """Returns a rectangle of one grid's voxels as the nearest of another's.

    The rectangle keeps its place in the world, its sides on the new grid's
    nearest voxel edges, clipped to the new grid.

    Args:
        old: The grid the rectangle is in.
        new: The grid to give it in.
        view: The 2D view it is a rectangle of.
        box: The rectangle, in voxels of old.
        position: A point of the slice shown, mm.

    Returns:
        The rectangle in voxels of new, or None if it lies outside new.
    """
    _, first, second = plane_axes(new, view)
    inverse = np.linalg.inv(new.affine)
    corners = np.array(detail_field(old, view, box, position))
    voxels = corners @ inverse[:3, :3].T + inverse[:3, 3]
    low = voxels.min(axis=0)[[first, second]]
    high = voxels.max(axis=0)[[first, second]]
    # Voxel k spans k - 0.5 to k + 0.5: the nearest edge, halves up.
    lo_edge = np.floor(low + 1.0).astype(int)
    size = np.maximum(np.floor(high + 1.0).astype(int) - lo_edge, 1)
    moved = (int(lo_edge[0]), int(lo_edge[1]), int(size[0]), int(size[1]))
    return clipped_box(new, view, moved)


def detail_title(
    name: str | None, view: ViewKind, slice_text: str, box: VoxelBox
) -> str:
    """Returns a detail window's title, such as "Detail - wmfod.mif - coronal y 60".

    The title ends with the rectangle's size in voxels.
    """
    parts = ["Detail"]
    if name:
        parts.append(name)
    parts.append(f"{VIEW_NAMES[view].lower()} {slice_text}")
    parts.append(f"{box[2]} x {box[3]} voxels")
    return " - ".join(parts)


def scale_bar_length(width_mm: float) -> float:
    """Returns a round length, 1, 2 or 5 times a power of ten, near a fifth of width."""
    if not width_mm > 0:
        return 1.0
    target = width_mm / 5
    power = 10.0 ** math.floor(math.log10(target))
    for step in (5.0, 2.0, 1.0):
        if step * power <= target:
            return step * power
    return power
