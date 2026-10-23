"""Builds the region filter a tractogram's options ask for and applies it."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from larmor.types import Tractogram, Volume
from larmor.viewer.model.item import Item
from larmor.viewer.model.options import TractogramOptions
from larmor.viewer.model.scene import Scene

if TYPE_CHECKING:
    from larmor.tractography import Region

# larmor.tractography loads scipy, so it is imported only in the functions that
# need it.


# The values of the options a selection by regions depends on.
type RegionKey = tuple[object, ...]


def region_key(options: TractogramOptions) -> RegionKey:
    """Returns the values of the options that decide which streamlines are kept."""
    return (
        options.region_filter,
        options.ends_only,
        options.include_masks,
        options.exclude_masks,
        options.within_masks,
        options.include_spheres,
        options.exclude_spheres,
        options.within_spheres,
    )


def mask_ids(item: Item) -> tuple[int, ...]:
    """Returns the ids of the masks a tractogram is filtered by; none for others."""
    o = item.options
    if not isinstance(o, TractogramOptions):
        return ()
    return (*o.include_masks, *o.exclude_masks, *o.within_masks)


@dataclass(frozen=True, eq=False)
class RegionFilter:
    """The regions a streamline must pass through, must not, and must stay within."""

    include: tuple[Region, ...]
    exclude: tuple[Region, ...]
    within: tuple[Region, ...] = ()
    ends_only: bool = False


def region_filter(scene: Scene, item: Item) -> RegionFilter | None:
    """Returns the filter a tractogram's options ask for.

    None if the options keep every streamline.

    Returns:
        The filter, or None.
    """
    o = item.options
    assert isinstance(o, TractogramOptions)
    if not o.region_filter:
        return None
    include = _masks(scene, o.include_masks) + _spheres(o.include_spheres)
    exclude = _masks(scene, o.exclude_masks) + _spheres(o.exclude_spheres)
    within = _masks(scene, o.within_masks) + _spheres(o.within_spheres)
    if not include and not exclude and not within:
        return None
    return RegionFilter(include, exclude, within, o.ends_only)


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


def _masks(scene: Scene, ids: tuple[int, ...]) -> tuple[Region, ...]:
    """Returns the images of the mask items."""
    masks: list[Region] = []
    for item_id in ids:
        data = scene.item(item_id).data
        assert isinstance(data, Volume)
        masks.append(data)
    return tuple(masks)


def _spheres(
    spheres: tuple[tuple[float, float, float, float], ...],
) -> tuple[Region, ...]:
    """Returns each (x, y, z, radius), in mm, as a Sphere."""
    from larmor.tractography import Sphere

    return tuple(Sphere(np.array((x, y, z)), radius) for x, y, z, radius in spheres)
