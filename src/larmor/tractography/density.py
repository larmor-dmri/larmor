"""Track density images, which count the streamlines through each voxel of a grid."""

import math
from collections.abc import Iterable, Sequence

import numpy as np

from larmor import _core
from larmor._threads import n_threads
from larmor.types import Grid, Tractogram, Volume

# Fraction of the smallest voxel size that the automatic upsampling brings the
# spacing of the points below.
_SPACING_PER_VOXEL = 1.0 / 3.0


def density_grid(reference: Grid, voxel_size: float | Sequence[float]) -> Grid:
    """Makes a grid of smaller or larger voxels over the field of view of a reference.

    Along each axis the new grid starts at the same outer corner of the field of
    view and covers it, overshooting it by less than one new voxel at the far end.

    Args:
        reference: The grid whose field of view the new grid covers.
        voxel_size: The new voxel size in mm: one value for isotropic voxels, or one
            per axis (in RAS order: x, y, z).

    Returns:
        The new grid, in RAS axis order, with the reference's axis directions.

    Raises:
        ValueError: If `voxel_size` is not one or three positive finite values.

    References:
        The grid of MRtrix3's tckmap -template with -vox (version 3.0.8).
    """
    sizes = np.atleast_1d(np.asarray(voxel_size, dtype=np.float64))
    if sizes.ndim != 1 or len(sizes) not in (1, 3) or not np.all(np.isfinite(sizes)):
        raise ValueError(
            f"voxel_size must be one or three finite values, got {voxel_size}."
        )
    if np.any(sizes <= 0):
        raise ValueError(f"voxel_size must be positive, got {voxel_size}.")
    sizes = np.broadcast_to(sizes, (3,))
    canonical = reference.canonical()
    old_sizes = np.linalg.norm(canonical.affine[:3, :3], axis=0)
    directions = canonical.affine[:3, :3] / old_sizes
    shape = tuple(
        math.ceil(n * old / new)
        for n, old, new in zip(canonical.shape, old_sizes, sizes, strict=True)
    )
    affine = np.eye(4)
    affine[:3, :3] = directions * sizes
    # The first voxel's outer corner stays where the reference's is.
    affine[:3, 3] = canonical.affine[:3, 3] + directions @ (0.5 * (sizes - old_sizes))
    return Grid((shape[0], shape[1], shape[2]), affine)


def track_density(
    tractogram: Tractogram | Iterable[Tractogram],
    grid: Grid,
    *,
    weights: np.ndarray | None = None,
    ends_only: bool = False,
    upsample: int | None = None,
    n_jobs: int = -1,
) -> Volume:
    """Counts the streamlines that pass through each voxel of a grid.

    Each streamline is first upsampled on the cubic Hermite spline that `resample`
    uses. Every point is mapped to its nearest voxel, and the streamline adds 1, or
    its weight, once to each voxel it reaches, however many of its points fall
    there. Points outside the grid are skipped. With `ends_only` the image is the
    endpoint density instead: each of the two endpoints adds 1 (or the weight) to
    its voxel, so a streamline that ends twice in one voxel adds 2 there.

    The grid may be finer than the image the streamlines were tracked on (see
    `density_grid`), which gives super-resolution track density imaging. By
    default each segment is split into enough pieces to bring the spacing of the
    points below a third of the smallest voxel size, so a fine grid has no gaps
    along a streamline. The result does not depend on `n_jobs`.

    Args:
        tractogram: The streamlines, in world coordinates (mm); or an iterable of
            tractograms, such as the batches of `iter_tractogram`, whose densities
            are added up.
        grid: The grid of the image to fill.
        weights: (S,) one weight per streamline, in the order of the streamlines
            (across all batches), such as SIFT2's. Defaults to None, every
            streamline counting 1.
        ends_only: Whether to map only the two endpoints of each streamline.
            Defaults to False.
        upsample: The number of pieces every segment is split into, at least 1 (1
            maps only the stored points). Defaults to None, each segment's own
            number from its length and the grid. Not used with `ends_only`.
        n_jobs: Number of threads, at most the cores; -1 is all cores. Defaults to
            -1.

    Returns:
        A 3D volume on `grid`: uint32 counts without weights, float32 sums of the
        weights with them.

    Raises:
        ValueError: If `weights` does not have one finite value per streamline,
            `upsample` is not an integer of at least 1, or `n_jobs` is 0 or below -1.

    References:
        Calamante F, Tournier JD, Jackson GD, Connelly A. Track-density imaging
        (TDI): super-resolution white matter imaging using whole-brain track-density
        mapping. NeuroImage 53 (2010) 1233-1243. The mapping follows MRtrix3's
        tckmap (version 3.0.8) with its default TDI contrast, and -ends_only.
    """
    threads = n_threads(n_jobs)
    if upsample is not None and (
        isinstance(upsample, bool) or not isinstance(upsample, (int, np.integer))
    ):
        raise ValueError(f"upsample must be an integer, got {upsample!r}.")
    if upsample is not None and upsample < 1:
        raise ValueError(f"upsample must be at least 1, got {upsample}.")
    all_weights = None
    if weights is not None:
        all_weights = np.ascontiguousarray(weights, dtype=np.float64)
        if all_weights.ndim != 1 or not np.all(np.isfinite(all_weights)):
            raise ValueError("weights must be a 1D array of finite values.")
    batches = [tractogram] if isinstance(tractogram, Tractogram) else tractogram

    # Mapped in RAS axis order, so a point halfway between two voxels goes to the
    # same one whatever the order of the grid's axes.
    canonical = grid.canonical()
    density = np.zeros(int(np.prod(canonical.shape)), dtype=np.float64)
    world_to_voxel = np.ascontiguousarray(np.linalg.inv(canonical.affine)[:3])
    shape = np.array(canonical.shape, dtype=np.int64)
    ratio = 1 if ends_only or upsample is None else upsample
    max_spacing = 0.0
    if upsample is None and not ends_only:
        smallest_voxel = float(np.linalg.norm(grid.affine[:3, :3], axis=0).min())
        max_spacing = _SPACING_PER_VOXEL * smallest_voxel
    done = 0
    for batch in batches:
        batch_weights = np.empty(0, dtype=np.float64)
        if all_weights is not None:
            batch_weights = all_weights[done : done + batch.num_streamlines]
            if len(batch_weights) != batch.num_streamlines:
                raise ValueError(
                    f"weights has {len(all_weights)} values, fewer than the "
                    f"streamlines ({done + batch.num_streamlines} or more)."
                )
        _core.add_track_density(
            np.ascontiguousarray(batch.points, dtype=np.float32),
            np.ascontiguousarray(batch.offsets, dtype=np.int64),
            world_to_voxel,
            shape,
            ratio,
            max_spacing,
            ends_only,
            batch_weights,
            density,
            threads,
        )
        done += batch.num_streamlines
    if all_weights is not None and done != len(all_weights):
        raise ValueError(
            f"weights has {len(all_weights)} values but there are {done} streamlines."
        )
    dtype = np.uint32 if all_weights is None else np.float32
    image = Volume(density.reshape(canonical.shape).astype(dtype), canonical.affine)
    image.to_orientation(grid.orientation)
    return Volume(np.ascontiguousarray(image.data), grid.affine.copy())
