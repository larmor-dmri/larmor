"""Builds the region items' data and filters, and their shapes in the views."""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from larmor.types import Tractogram, Volume
from larmor.viewer.model.camera import Vec3
from larmor.viewer.model.item import Category, Item, ItemId
from larmor.viewer.model.options import (
    RegionOptions,
    RegionRole,
    RegionShape,
    TractogramOptions,
)
from larmor.viewer.model.scene import Scene

if TYPE_CHECKING:
    from larmor.tractography import Region

# larmor.tractography loads scipy, so it is imported only in the functions that
# need it.

# The values a selection by regions depends on.
type RegionKey = tuple[object, ...]
# How far the slice may be from a sphere's centre for the centre to count as on
# it, mm.
ON_SLICE = 1e-3


@dataclass(frozen=True)
class MaskRegion:
    """The data of a mask region: the image item whose non-zero voxels it is.

    Attributes:
        mask: The image item's id; None once the image is removed.
    """

    mask: ItemId | None


def region_data(options: RegionOptions) -> object:
    """Returns a region item's data for its options.

    Returns:
        A `Sphere` or a `Box` of `larmor.tractography`, or a `MaskRegion`.
    """
    if options.shape == RegionShape.MASK:
        return MaskRegion(options.mask)
    from larmor.tractography import Box, Sphere

    if options.shape == RegionShape.BOX:
        return Box(np.array(options.centre), np.array(options.size))
    return Sphere(np.array(options.centre), options.radius)


def applied_regions(scene: Scene, item: Item) -> list[Item]:
    """Returns the region items a tractogram applies, in id order; none for others."""
    o = item.options
    if not isinstance(o, TractogramOptions):
        return []
    return [scene.item(i) for i in o.regions]


def region_key(scene: Scene, item: Item) -> RegionKey:
    """Returns the values that decide which of a tractogram's streamlines are kept."""
    o = item.options
    assert isinstance(o, TractogramOptions)
    shapes = []
    for region in applied_regions(scene, item):
        r = region.options
        assert isinstance(r, RegionOptions)
        shapes.append((region.id, r.shape, r.role, r.centre, r.radius, r.size, r.mask))
    return (o.region_filter, o.ends_only, tuple(shapes))


def mask_ids(scene: Scene, item: Item) -> tuple[ItemId, ...]:
    """Returns the ids of the images a tractogram's mask regions name."""
    ids = []
    for region in applied_regions(scene, item):
        r = region.options
        assert isinstance(r, RegionOptions)
        if r.shape == RegionShape.MASK and r.mask is not None:
            ids.append(r.mask)
    return tuple(ids)


@dataclass(frozen=True, eq=False)
class RegionFilter:
    """The regions a streamline must pass through, must not, and must stay within."""

    include: tuple[Region, ...]
    exclude: tuple[Region, ...]
    within: tuple[Region, ...] = ()
    ends_only: bool = False


def region_filter(scene: Scene, item: Item) -> RegionFilter | None:
    """Returns the filter of the regions a tractogram applies.

    Returns:
        The filter; None if the tractogram keeps every streamline (the filter is
        off, or no region applies).
    """
    o = item.options
    assert isinstance(o, TractogramOptions)
    if not o.region_filter:
        return None
    lists: dict[RegionRole, list[Region]] = {role: [] for role in RegionRole}
    for region in applied_regions(scene, item):
        r = region.options
        assert isinstance(r, RegionOptions)
        library = _library_region(scene, region)
        if library is not None:
            lists[r.role].append(library)
    include, exclude, within = (tuple(lists[role]) for role in RegionRole)
    if not include and not exclude and not within:
        return None
    return RegionFilter(include, exclude, within, o.ends_only)


def _library_region(scene: Scene, region: Item) -> Region | None:
    """Returns a region item as the library's region; None for a mask without image."""
    data = region.data
    if isinstance(data, MaskRegion):
        if data.mask is None:
            return None
        image = scene.item(data.mask).data
        assert isinstance(image, Volume)
        return image
    return data  # type: ignore[return-value]  # a Sphere or a Box


def passes_regions(tractogram: Tractogram, regions: RegionFilter) -> np.ndarray:
    """Tests which streamlines pass the regions.

    A streamline is kept when it passes through every include region, through no
    exclude one, and stays within every within region.

    Args:
        tractogram: The streamlines to test.
        regions: The regions, and whether only the end points are tested.

    Returns:
        (S,) bool, True for each streamline kept.
    """
    from larmor.tractography import passes_through, stays_within

    kept = np.ones(tractogram.num_streamlines, dtype=bool)
    for region in regions.include:
        kept &= passes_through(tractogram, region, ends_only=regions.ends_only)
    for region in regions.exclude:
        kept &= ~passes_through(tractogram, region, ends_only=regions.ends_only)
    for region in regions.within:
        kept &= stays_within(tractogram, region, ends_only=regions.ends_only)
    return kept


def mask_regions_of(scene: Scene, image_id: ItemId) -> list[Item]:
    """Returns the mask regions that name an image."""
    return [
        region
        for region in scene.in_category(Category.REGIONS)
        if isinstance(region.data, MaskRegion) and region.data.mask == image_id
    ]


# Shapes in the views


@dataclass(frozen=True)
class Circle:
    """A sphere's cross-section with a slice.

    Attributes:
        centre: The circle's centre, in the slice's plane, world mm.
        radius: Its radius, mm.
        on_slice: Whether the sphere's own centre lies in the slice.
    """

    centre: Vec3
    radius: float
    on_slice: bool


@dataclass(frozen=True)
class Rectangle:
    """A box's cross-section with a slice, its lowest and highest corner in mm."""

    low: Vec3
    high: Vec3


def cross_section(
    options: RegionOptions, axis: int, coordinate: float
) -> Circle | Rectangle | None:
    """Returns where a slice cuts a sphere or a box region.

    Args:
        options: The region's options.
        axis: The world axis across the slice (0 x, 1 y, 2 z).
        coordinate: The slice's place on that axis, mm.

    Returns:
        The circle or rectangle in the slice's plane; None if the slice misses
        the region, or for a mask region.
    """
    centre = np.array(options.centre, dtype=float)
    offset = coordinate - centre[axis]
    centre[axis] = coordinate
    if options.shape == RegionShape.SPHERE:
        if abs(offset) > options.radius:
            return None
        radius = math.sqrt(max(options.radius**2 - offset**2, 0.0))
        return Circle(_vec(centre), radius, abs(offset) <= ON_SLICE)
    if options.shape == RegionShape.BOX:
        half = np.array(options.size, dtype=float) / 2
        if abs(offset) > half[axis]:
            return None
        half[axis] = 0.0
        return Rectangle(_vec(centre - half), _vec(centre + half))
    return None


class Handle(StrEnum):
    """The part of a region's cross-section under the pointer."""

    CENTRE = "centre"  # the centre dot, which moves the region
    EDGE = "edge"  # a sphere's ring or a box's side
    CORNER = "corner"  # a corner of a box, which resizes it


def hit_region(
    options: RegionOptions,
    axis: int,
    coordinate: float,
    point: Vec3,
    tolerance: float,
) -> Handle | None:
    """Returns the part of a region's cross-section near a point of the slice.

    Args:
        options: The region's options.
        axis: The world axis across the slice.
        coordinate: The slice's place on that axis, mm.
        point: The point, world mm, in the slice's plane.
        tolerance: How near counts, mm.

    Returns:
        The handle, or None if the point is near none of them.
    """
    shape = cross_section(options, axis, coordinate)
    p = np.array(point, dtype=float)
    p[axis] = coordinate
    if isinstance(shape, Circle):
        distance = float(np.linalg.norm(p - shape.centre))
        if distance <= tolerance:
            return Handle.CENTRE
        if abs(distance - shape.radius) <= tolerance:
            return Handle.EDGE
        return None
    if isinstance(shape, Rectangle):
        low, high = np.array(shape.low), np.array(shape.high)
        plane = [a for a in range(3) if a != axis]
        corners = []
        for x in (low[plane[0]], high[plane[0]]):
            for y in (low[plane[1]], high[plane[1]]):
                corner = p.copy()
                corner[plane[0]], corner[plane[1]] = x, y
                corners.append(corner)
        if any(np.linalg.norm(p - c) <= tolerance for c in corners):
            return Handle.CORNER
        if np.linalg.norm(p - (low + high) / 2) <= tolerance:
            return Handle.CENTRE
        inside = all(low[a] - tolerance <= p[a] <= high[a] + tolerance for a in plane)
        near = any(
            min(abs(p[a] - low[a]), abs(p[a] - high[a])) <= tolerance for a in plane
        )
        return Handle.EDGE if inside and near else None
    return None


def sphere_radius_at(
    options: RegionOptions, axis: int, coordinate: float, point: Vec3
) -> float:
    """Returns the radius that puts a sphere's cross-section through a point.

    The point is in the slice at `coordinate` on `axis`; the sphere keeps its
    centre, so its radius grows with the slice's distance from the centre.

    Returns:
        The radius, mm.
    """
    centre = np.array(options.centre, dtype=float)
    p = np.array(point, dtype=float)
    p[axis] = centre[axis]
    across = float(np.linalg.norm(p - centre))
    offset = coordinate - centre[axis]
    return math.sqrt(across**2 + offset**2)


def box_between(
    options: RegionOptions, axis: int, first: Vec3, second: Vec3
) -> tuple[Vec3, Vec3]:
    """Returns a box's centre and size with two opposite corners in a slice.

    The box keeps its centre and size along `axis`, across the slice.

    Returns:
        The centre and the size, mm.
    """
    a, b = np.array(first, dtype=float), np.array(second, dtype=float)
    centre = (a + b) / 2
    size = np.abs(a - b)
    centre[axis] = options.centre[axis]
    size[axis] = options.size[axis]
    return _vec(centre), _vec(size)


def ray_hit(
    options: RegionOptions, origin: np.ndarray, direction: np.ndarray
) -> float | None:
    """Returns how far along a ray it first meets a sphere or box region.

    Args:
        options: The region's options.
        origin: (3,) where the ray starts, world mm.
        direction: (3,) its direction, unit.

    Returns:
        The distance from the origin, mm, at least 0; None if the ray misses
        the region, or for a mask region.
    """
    centre = np.array(options.centre, dtype=float)
    if options.shape == RegionShape.SPHERE:
        offset = origin - centre
        b = float(offset @ direction)
        c = float(offset @ offset) - options.radius**2
        disc = b * b - c
        if disc < 0:
            return None
        far = -b + math.sqrt(disc)
        if far < 0:
            return None
        return max(-b - math.sqrt(disc), 0.0)
    if options.shape == RegionShape.BOX:
        half = np.array(options.size, dtype=float) / 2
        near, far = 0.0, math.inf
        for a in range(3):
            low, high = centre[a] - half[a], centre[a] + half[a]
            if direction[a] == 0:
                if not low <= origin[a] <= high:
                    return None
                continue
            t1 = (low - origin[a]) / direction[a]
            t2 = (high - origin[a]) / direction[a]
            near, far = max(near, min(t1, t2)), min(far, max(t1, t2))
        return near if near <= far else None
    return None


def _vec(values: np.ndarray) -> Vec3:
    """Returns three numbers as a tuple of floats."""
    return (float(values[0]), float(values[1]), float(values[2]))
