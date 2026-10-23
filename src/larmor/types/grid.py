"""The voxel grid type."""

from dataclasses import dataclass

import numpy as np

from larmor.types._affine import (
    check_affine,
    flip_affine,
    orientation_of,
    permute_affine,
    reorientation,
)
from larmor.types._arrays import read_only


@dataclass(frozen=True, eq=False)
class Grid:
    """The voxel grid of an image, with its shape and where its voxels are in the world.

    Compare grids with `matches`, not `==`.

    Attributes:
        shape: The number of voxels along each spatial axis.
        affine: The (4, 4) float64 matrix, read only, from voxel indices to world
            coordinates (RAS+) in mm.
        xform_codes: The NIfTI sform and qform codes, which name the space the
            affine maps to (1 scanner, 2 aligned, 3 Talairach, 4 MNI). An image
            resampled onto the grid takes them.
    """

    shape: tuple[int, int, int]
    affine: np.ndarray
    xform_codes: tuple[int, int] = (1, 1)

    @property
    def orientation(self) -> tuple[str, str, str]:
        """Closest RAS+ direction of each voxel axis, such as ("R", "A", "S")."""
        return orientation_of(self.affine)

    def __post_init__(self) -> None:
        """Checks the shape and the affine, and makes the affine read-only.

        Raises:
            ValueError: If `shape` is not three positive integers, or `affine` is not
                a finite 4x4 matrix with last row [0, 0, 0, 1] and non-zero axes.
        """
        # Typed loosely, because the caller may pass anything.
        shape: tuple[object, ...] = tuple(self.shape)
        sizes = tuple(
            int(n)
            for n in shape
            if isinstance(n, (int, np.integer)) and not isinstance(n, bool) and n > 0
        )
        if len(shape) != 3 or len(sizes) != 3:
            raise ValueError(
                f"A grid's shape must be three positive integers, got {shape}."
            )
        object.__setattr__(self, "shape", (sizes[0], sizes[1], sizes[2]))
        object.__setattr__(self, "affine", read_only(check_affine(self.affine)))

    def canonical(self) -> Grid:
        """Returns the same voxels with the axes permuted and flipped to RAS.

        Returns:
            Grid: The grid with RAS axes.
        """
        order, flips = reorientation(orientation_of(self.affine), "RAS")
        shape = (self.shape[order[0]], self.shape[order[1]], self.shape[order[2]])
        affine = permute_affine(self.affine, order)
        for axis, flip in enumerate(flips):
            if flip:
                affine = flip_affine(affine, axis, shape[axis])
        return Grid(shape, affine, self.xform_codes)

    def matches(self, other: Grid, *, atol: float = 1e-3) -> bool:
        """Checks whether two grids are the same voxels in the world.

        Grids that differ only in the order and direction of their axes, such as a
        volume and the same volume reoriented, match.

        Args:
            other: The grid to compare against.
            atol: Tolerance on the affines, in mm. Defaults to 1e-3, because TRK files
                store their affine in float32.

        Returns:
            True if `other` is this grid with its axes permuted and flipped: same
            shape after the permutation, and affines equal within `atol`.
        """
        # Maps `other`'s voxels to this grid's: a signed permutation for the same
        # voxels. Read directly rather than via RAS, which can tie for oblique affines.
        to_self = np.linalg.solve(self.affine, other.affine)[:3, :3]
        source = [int(np.argmax(np.abs(to_self[:, k]))) for k in range(3)]
        if sorted(source) != [0, 1, 2]:
            return False
        order = (source[0], source[1], source[2])

        shape = (self.shape[order[0]], self.shape[order[1]], self.shape[order[2]])
        if shape != other.shape:
            return False
        affine = permute_affine(self.affine, order)
        for axis in range(3):
            if to_self[order[axis], axis] < 0:
                affine = flip_affine(affine, axis, shape[axis])
        return bool(np.allclose(affine, other.affine, rtol=0, atol=atol))
