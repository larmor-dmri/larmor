"""Makes new mask images from a DWI, a threshold of an image or parcellation labels."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import numpy as np

from larmor.types import Volume
from larmor.viewer.model.item import Category, Item, ItemId
from larmor.viewer.model.options import VolumeOptions
from larmor.viewer.model.viewer import Viewer

# larmor.dwi loads scipy, so it is imported only in the functions that need it.

# b-values at or below this count as b0, s/mm^2, as in a gradient table.
B0_THRESHOLD: Final = 50.0


class MaskSource(StrEnum):
    """What a new mask is made from."""

    DWI = "dwi"  # the brain mask of a DWI, as `lar mask` computes it
    THRESHOLD = "threshold"  # the voxels of an image between two values
    LABELS = "labels"  # the voxels of some labels of a parcellation


class Combination(StrEnum):
    """How a new mask combines with another mask."""

    UNION = "union"
    INTERSECT = "intersect"
    SUBTRACT = "subtract"


@dataclass(frozen=True)
class MaskRecipe:
    """How a new mask is made: its source, its clean-up and its combination.

    Attributes:
        source: What it is made from.
        image: The source image item: the DWI, the image thresholded or the
            parcellation; None for none chosen.
        bvals: For a 4D DWI, its b-values file, whose b0 volumes are averaged;
            "" takes the first volume as the b0.
        lower: The lowest value kept by a threshold, included.
        upper: The highest value kept by a threshold, included; None for none.
        labels: The parcellation's labels kept.
        dilate: Voxels the mask grows by.
        erode: Voxels it then shrinks by.
        fill_holes: Whether the voxels it encloses are added.
        largest: Whether only its largest connected piece is kept.
        combine_with: Another mask on the same grid; None for none.
        combination: How it combines with that mask.
    """

    source: MaskSource = MaskSource.DWI
    image: ItemId | None = None
    bvals: str = ""
    lower: float = 0.0
    upper: float | None = None
    labels: tuple[int, ...] = ()
    dilate: int = 0
    erode: int = 0
    fill_holes: bool = False
    largest: bool = False
    combine_with: ItemId | None = None
    combination: Combination = Combination.UNION


class MaskMaker:
    """Makes masks from recipes on a viewer's items, keeping the last source mask.

    The source mask (the slow part for a DWI) is kept while the source and its
    settings stay the same, so that changing the clean-up is quick.
    """

    def __init__(self, viewer: Viewer) -> None:
        """Creates a maker for the viewer's items."""
        self.viewer = viewer
        self._kept: tuple[tuple[object, ...], np.ndarray] | None = None

    def sources(self, source: MaskSource) -> list[Item]:
        """Returns the items a source can be made from, in scene tree order."""
        scene = self.viewer.scene
        if source == MaskSource.DWI:
            return scene.in_category(Category.VOLUMES)
        images = scene.in_category(Category.VOLUMES) + scene.in_category(
            Category.OVERLAYS
        )
        if source == MaskSource.LABELS:
            return [i for i in images if i.volume_count == 1]
        return images

    def masks_like(self, image_id: ItemId) -> list[Item]:
        """Returns the images of one volume on the same grid as an image."""
        grid = self._volume(image_id)
        return [
            item
            for item in self.sources(MaskSource.LABELS)
            if item.id != image_id and _same_grid(self._volume(item.id), grid)
        ]

    def source_mask(self, recipe: MaskRecipe) -> np.ndarray:
        """Returns the (X, Y, Z) bool mask of the source, before clean-up.

        Raises:
            ValueError: If no image is chosen, or the brain mask is empty.
            OSError: If the b-values file cannot be read.
        """
        from larmor.dwi import create_brain_mask, label_mask, threshold_mask

        if recipe.image is None:
            raise ValueError("Choose the image the mask is made from.")
        volume = self._volume(recipe.image)
        key: tuple[object, ...] = (recipe.source, volume, recipe.image)
        if recipe.source == MaskSource.DWI:
            key += (recipe.bvals,)
        elif recipe.source == MaskSource.THRESHOLD:
            key += (recipe.lower, recipe.upper, self._index(recipe.image))
        else:
            key += (recipe.labels,)
        # Volumes compare by identity, so new data is a new key.
        if self._kept is not None and self._kept[0] == key:
            return self._kept[1]
        if recipe.source == MaskSource.DWI:
            mask = create_brain_mask(b0_of(volume, recipe.bvals))
        elif recipe.source == MaskSource.THRESHOLD:
            mask = threshold_mask(
                _scaled(volume, self._index(recipe.image)), recipe.lower, recipe.upper
            )
        else:
            mask = label_mask(_scaled(volume, 0), recipe.labels)
        self._kept = (key, mask)
        return mask

    def make(self, recipe: MaskRecipe) -> np.ndarray:
        """Returns the (X, Y, Z) bool mask a recipe makes.

        The clean-up runs in the order dilate, erode, fill holes and keep the
        largest piece, then the combination.

        Raises:
            ValueError: If no image is chosen, the brain mask is empty, or the
                mask to combine with is not on the same grid.
            OSError: If the b-values file cannot be read.
        """
        from larmor.dwi import (
            combine_masks,
            dilate_mask,
            erode_mask,
            fill_mask_holes,
            largest_component,
        )

        mask = self.source_mask(recipe)
        mask = dilate_mask(mask, recipe.dilate)
        mask = erode_mask(mask, recipe.erode)
        if recipe.fill_holes:
            mask = fill_mask_holes(mask)
        if recipe.largest:
            mask = largest_component(mask)
        if recipe.combine_with is not None:
            assert recipe.image is not None
            other = self._volume(recipe.combine_with)
            if not _same_grid(other, self._volume(recipe.image)):
                raise ValueError("The mask to combine with is on another grid.")
            mask = combine_masks(mask, _scaled(other, 0) != 0, recipe.combination.value)
        return mask

    def create(self, recipe: MaskRecipe, name: str) -> ItemId:
        """Adds the mask a recipe makes as an overlay, on its source's grid.

        The overlay's data is uint8, 1 inside and 0 outside, with the source's
        affine, so it can be saved as NIfTI or MIF and read back the same.

        Returns:
            The new overlay's id.

        Raises:
            ValueError: As `make`, or if the name is empty.
        """
        if not name.strip():
            raise ValueError("A mask needs a name.")
        mask = self.make(recipe)
        assert recipe.image is not None
        volume = self._volume(recipe.image)
        data = Volume(mask.astype(np.uint8), volume.affine.copy())
        return self.viewer.add(data, Category.OVERLAYS, name=name.strip()).id

    def _volume(self, item_id: ItemId) -> Volume:
        """Returns an item's volume."""
        data = self.viewer.scene.item(item_id).data
        assert isinstance(data, Volume)
        return data

    def _index(self, item_id: ItemId) -> int:
        """Returns the volume an image shows, 0 for a 3D one."""
        o = self.viewer.scene.item(item_id).options
        return o.volume_index if isinstance(o, VolumeOptions) else 0


def b0_of(volume: Volume, bvals: str | Path = "") -> np.ndarray:
    """Returns the b0 image of a DWI, (X, Y, Z) float64.

    Args:
        volume: The DWI, 4D, or a 3D image taken as the b0 itself.
        bvals: Its b-values file; the volumes at or below 50 s/mm^2 are
            averaged. "" takes the first volume.

    Raises:
        ValueError: If the file does not have one value per volume, or no
            volume is a b0.
        OSError: If the file cannot be read.
    """
    if volume.data.ndim == 3:
        return _scaled(volume, 0)
    if not str(bvals):
        return _scaled(volume, 0)
    values = np.loadtxt(bvals, ndmin=1).ravel()
    count = volume.data.shape[3]
    if len(values) != count:
        raise ValueError(
            f"{Path(bvals).name} has {len(values)} b-values for {count} volumes."
        )
    b0s = np.flatnonzero(values <= B0_THRESHOLD)
    if not len(b0s):
        raise ValueError(f"No b0 volume: no b-value is at or below {B0_THRESHOLD:g}.")
    stack = np.stack([_scaled(volume, int(i)) for i in b0s], axis=-1)
    mean: np.ndarray = stack.mean(axis=-1)
    return mean


def voxel_volume_ml(volume: Volume) -> float:
    """Returns the volume of one voxel of an image, mL."""
    return float(abs(np.linalg.det(volume.affine[:3, :3]))) / 1000.0


def _scaled(volume: Volume, index: int) -> np.ndarray:
    """Returns one volume of an image in scaled units, float64."""
    data = volume.data[..., index] if volume.data.ndim == 4 else volume.data
    values: np.ndarray = data * float(volume.slope) + float(volume.intercept)
    return values.astype(np.float64, copy=False)


def _same_grid(a: Volume, b: Volume) -> bool:
    """Tells whether two images have the same voxels: shape and affine."""
    return a.data.shape[:3] == b.data.shape[:3] and np.allclose(
        a.affine, b.affine, atol=1e-4
    )
