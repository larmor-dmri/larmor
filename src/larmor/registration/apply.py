"""Applies transforms to images, tractograms and surfaces."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np

from larmor._log import LarmorLogger
from larmor.io import (
    TRACTOGRAM_EXTENSIONS,
    VOLUME_EXTENSIONS,
    iter_tractogram,
    load_grid,
    load_surface,
    load_tractogram,
    load_volume,
    open_tractogram_writer,
    save_surface,
    save_tractogram,
    save_volume,
)
from larmor.registration.files import load_registration
from larmor.registration.sampling import (
    Interpolation,
    cubic_coefficients,
    nearest_dtype,
    sample_image,
)
from larmor.registration.transforms import Transform
from larmor.types import Grid, Surface, Tractogram, Volume

# How the directions in an image are stored.
Reorientation = Literal["peaks", "tensor"]

# Output voxels resampled at once.
_CHUNK_VOXELS = 1 << 20


def transform_points(points: np.ndarray, transform: Transform) -> np.ndarray:
    """Maps points with a transform.

    Args:
        points: (N, 3) points in world coordinates (RAS+) in mm.
        transform: The transform; each point p becomes transform(p).

    Returns:
        (N, 3) float64, the mapped points.
    """
    return transform.map_points(np.asarray(points, dtype=np.float64).reshape(-1, 3))


def transform_volume(
    volume: Volume,
    transform: Transform,
    reference: Grid,
    *,
    interpolation: Interpolation = "linear",
    reorient: Reorientation | None = None,
    fill: float = 0.0,
) -> Volume:
    """Resamples an image onto a reference grid through a transform.

    Each voxel of the reference grid, at world position p, takes the image's value
    at transform(p). The transform therefore maps points of the reference space to
    the image's space, the direction transform files store, so the same file and
    reference image give the result of antsApplyTransforms, `flirt -applyxfm`,
    applywarp or mrtransform.

    A point within half a voxel beyond the image's outer voxel centres takes the
    edge value; a point further out gets `fill`.

    Directions stored in the image are reoriented when `reorient` says how they
    are stored, following Alexander et al. (2001):

    - "peaks": vectors in groups of 3 per voxel, shape (X, Y, Z, 3K), each turned
      by the local transform and kept at its length. Needs
      `interpolation="nearest"`.
    - "tensor": diffusion tensors as their 6 elements (xx, xy, xz, yy, yz, zz),
      interpolated element by element, then rotated by finite strain.

    FODs are not reoriented.

    Args:
        volume: The image, 3D or 4D; each volume of a 4D image is resampled alike.
        transform: Maps reference points to image points, in world mm (RAS+).
        reference: The output grid. The result takes its NIfTI sform and qform
            codes, so an image resampled onto a template is labelled in the
            template's space.
        interpolation: "nearest" (for labels; keeps the data type), "linear" (the
            default) or "cubic" (B-spline of order 3).
        reorient: How directions are stored, if they are: "peaks" or "tensor".
        fill: The value outside the image.

    Returns:
        The resampled image on `reference`, float32 (float64 for a float64 image)
        unless the interpolation is "nearest", which keeps the stored values and
        their scaling.

    Raises:
        ValueError: If `reorient` does not match the image's last axis, or "peaks"
            is asked for with an interpolation other than "nearest".

    References:
        Alexander DC, Pierpaoli C, Basser PJ, Gee JC. Spatial transformations of
        diffusion tensor magnetic resonance images. IEEE Trans Med Imaging 20
        (2001) 1131-1139.
    """
    channels = 1 if volume.data.ndim == 3 else volume.data.shape[3]
    _check_reorient(reorient, channels, interpolation)

    if interpolation == "nearest":
        data = volume.data
        out_dtype = nearest_dtype(data.dtype, fill)
    else:
        wide = volume.data.dtype == np.float64
        out_dtype = np.dtype(np.float64 if wide else np.float32)
        data = volume.as_float(dtype=np.float64)
    stack = data if data.ndim == 4 else data[..., None]
    if interpolation == "cubic":
        stack = np.stack(
            [cubic_coefficients(stack[..., c]) for c in range(channels)], axis=-1
        )

    to_voxels = np.linalg.inv(volume.affine)
    nx, ny, nz = reference.shape
    out = np.empty((nx, ny, nz, channels), dtype=out_dtype)
    slices = max(1, _CHUNK_VOXELS // (nx * ny))
    for start in range(0, nz, slices):
        stop = min(nz, start + slices)
        ijk = np.stack(
            np.meshgrid(
                np.arange(nx), np.arange(ny), np.arange(start, stop), indexing="ij"
            ),
            axis=-1,
        ).reshape(-1, 3)
        world = ijk @ reference.affine[:3, :3].T + reference.affine[:3, 3]
        source = transform.map_points(world)
        voxels = source @ to_voxels[:3, :3].T + to_voxels[:3, 3]
        values = np.stack(
            [
                sample_image(stack[..., c], voxels, interpolation, fill)
                for c in range(channels)
            ],
            axis=-1,
        )
        if reorient is not None:
            values = _reoriented(values, transform.jacobians(world), reorient)
        out[:, :, start:stop] = values.reshape(nx, ny, stop - start, channels)

    result = out if volume.data.ndim == 4 else out[..., 0]
    codes = reference.xform_codes
    if interpolation == "nearest":
        return Volume(
            result, reference.affine, volume.slope, volume.intercept, xform_codes=codes
        )
    return Volume(result, reference.affine, xform_codes=codes)


def transform_tractogram(
    tractogram: Tractogram, transform: Transform, *, reference: Grid | None = None
) -> Tractogram:
    """Moves every point of a tractogram with a transform.

    Each point p becomes transform(p). This is the opposite direction of
    `transform_volume`: to bring a tractogram into the space an image is resampled
    into, pass `transform.inverse()`.

    Args:
        tractogram: The streamlines, in world coordinates (RAS+) in mm.
        transform: Maps the tractogram's space to the output space.
        reference: The grid of the output space, stored as the result's
            `reference` (a TRK file needs one). None leaves it unknown.

    Returns:
        A new tractogram with the moved points and the same offsets,
        `point_data` and `streamline_data`.

    Raises:
        ValueError: If a point lands where the transform is undefined (NaN).
    """
    points = transform.map_points(tractogram.points.astype(np.float64))
    undefined = int((~np.isfinite(points).all(axis=1)).sum())
    if undefined:
        raise ValueError(
            f"{undefined} of {len(points)} points fall where the transform is "
            "undefined (outside its warp field, NaN)."
        )
    return Tractogram(
        points=points,
        offsets=tractogram.offsets,
        point_data=dict(tractogram.point_data),
        streamline_data=dict(tractogram.streamline_data),
        reference=reference,
    )


def transform_tractogram_file(
    input_path: str | Path,
    output_path: str | Path,
    transform: Transform,
    *,
    reference: Grid | None = None,
    batch_size: int = 100_000,
) -> int:
    """Moves the points of a tractogram file with a transform, into a new file.

    As in `transform_tractogram`, each point p becomes transform(p). Uncompressed
    `.tck` and `.trk` files are read and written in batches; other formats are
    read whole. The streamlines keep their order, so weights and TSF files of the
    input still match the output.

    Args:
        input_path: The tractogram file.
        output_path: The file to write; its extension chooses the format.
        transform: Maps the input's space to the output space.
        reference: The grid of the output space; a `.trk` output needs it.
        batch_size: Streamlines per batch when streaming.

    Returns:
        The number of streamlines written.

    Raises:
        ValueError: As `transform_tractogram`, or if a `.trk` output has no
            reference.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    if _streams(input_path) and _streams(output_path):
        count = 0
        with open_tractogram_writer(output_path, reference=reference) as writer:
            for batch in iter_tractogram(input_path, batch_size=batch_size):
                writer.write(
                    transform_tractogram(batch, transform, reference=reference)
                )
                count += batch.num_streamlines
        return count
    moved = transform_tractogram(
        load_tractogram(input_path), transform, reference=reference
    )
    save_tractogram(moved, output_path, reference=reference)
    return moved.num_streamlines


def transform_surface(
    surface: Surface, transform: Transform, *, reference: Grid | None = None
) -> Surface:
    """Moves every vertex of a surface with a transform.

    Each vertex p becomes transform(p), the same direction as
    `transform_tractogram` and the opposite of `transform_volume`. Where the
    transform reflects at every vertex, such as an affine with a flip, the
    triangles' vertex order is reversed so that normals still point out.

    Args:
        surface: The mesh, in world coordinates (RAS+) in mm.
        transform: Maps the surface's space to the output space.
        reference: The grid of the output space, stored as the result's
            `reference`. None leaves it unknown.

    Returns:
        A new surface with the moved vertices and the same `vertex_data`.

    Raises:
        ValueError: If a vertex lands where the transform is undefined (NaN).
    """
    vertices = transform.map_points(surface.vertices.astype(np.float64))
    if not np.isfinite(vertices).all():
        raise ValueError(
            "Some vertices fall where the transform is undefined (outside its warp "
            "field, NaN)."
        )
    faces = surface.faces
    if len(vertices) > 0:
        signs = np.sign(np.linalg.det(transform.jacobians(surface.vertices)))
        if (signs < 0).all():
            faces = faces[:, ::-1]
        elif (signs < 0).any():
            LarmorLogger.warning(
                "The transform reflects the surface at some vertices only; the "
                "normals point in where it does."
            )
    return Surface(
        vertices=vertices,
        faces=np.ascontiguousarray(faces),
        vertex_data=dict(surface.vertex_data),
        reference=reference,
    )


def apply_transform(
    data: Volume | Tractogram | Surface,
    transform: Transform,
    *,
    reference: Grid | None = None,
    interpolation: Interpolation = "linear",
    reorient: Reorientation | None = None,
) -> Volume | Tractogram | Surface:
    """Brings an image, tractogram or surface into a reference space.

    `transform` maps reference points to the data's space, as `load_registration`
    returns it. An image is resampled with it; the points of a tractogram or
    surface move with its inverse, so that every kind of data lands in the same
    space.

    Args:
        data: A `Volume`, `Tractogram` or `Surface`, in world coordinates (RAS+).
        transform: Maps reference points to the data's points, in mm.
        reference: The reference grid; needed for an image, stored as the
            `reference` of a tractogram or surface.
        interpolation: For an image, as in `transform_volume`.
        reorient: For an image, as in `transform_volume`.

    Returns:
        The data in the reference space, of the same type.

    Raises:
        ValueError: If an image has no reference grid, or as the function used.
    """
    if isinstance(data, Volume):
        if reference is None:
            raise ValueError("An image needs the reference grid to resample onto.")
        return transform_volume(
            data, transform, reference, interpolation=interpolation, reorient=reorient
        )
    if isinstance(data, Tractogram):
        return transform_tractogram(data, transform.inverse(), reference=reference)
    return transform_surface(data, transform.inverse(), reference=reference)


def transform_file(
    input_path: str | Path,
    output_path: str | Path,
    files: Sequence[str | Path | tuple[str | Path, str | Path]],
    *,
    reference_path: str | Path | None = None,
    inverse: bool = False,
    interpolation: Interpolation = "linear",
    reorient: Reorientation | None = None,
    format: str | None = None,
    warp_type: str | None = None,
    fsl_input_path: str | Path | None = None,
    fsl_reference_path: str | Path | None = None,
) -> None:
    """Brings an image, tractogram or surface file into a reference space.

    The transform files are read with `load_registration` and applied as
    `apply_transform` does, so an image is resampled as antsApplyTransforms,
    flirt, applywarp or mrtransform would with the same files and reference.

    Args:
        input_path: An image, a tractogram, or a surface (`.gii` or FreeSurfer).
        output_path: Where to write the result, of the same kind.
        files: The transform files, as in `load_registration`.
        reference_path: An image of the reference space; needed for images and
            for a `.trk` output.
        inverse: Go from the reference space into the space of the files' inputs.
        interpolation: For images, as in `transform_volume`.
        reorient: For images, as in `transform_volume`.
        format: "ants", "fsl" or "mrtrix", when the files alone do not tell.
        warp_type: The warp type of the warp fields, as in `load_transform`.
        fsl_input_path: An image of the FSL run's input, if not the default.
        fsl_reference_path: An image of the FSL run's reference, if not the
            default.

    Raises:
        ValueError: If the output is the input, an image has no reference, or as
            the functions used.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        # A streamed tractogram would overwrite the batches still to be read.
        raise ValueError(f"The output {output_path} is the input; give another path.")
    is_image = input_path.name.endswith(_IMAGE_EXTENSIONS)
    if is_image and reference_path is None:
        raise ValueError(
            f"{input_path} is an image: it needs a reference image (--reference)."
        )
    reference = None if reference_path is None else load_grid(reference_path)
    transform = load_registration(
        files,
        data_grid=load_grid(input_path) if is_image else None,
        reference_grid=reference,
        inverse=inverse,
        format=format,
        warp_type=warp_type,
        fsl_input_grid=None if fsl_input_path is None else load_grid(fsl_input_path),
        fsl_reference_grid=(
            None if fsl_reference_path is None else load_grid(fsl_reference_path)
        ),
    )
    if is_image:
        volume = load_volume(input_path)
        assert reference is not None
        moved = transform_volume(
            volume, transform, reference, interpolation=interpolation, reorient=reorient
        )
        save_volume(moved, output_path)
    elif input_path.name.endswith(_TRACT_EXTENSIONS):
        transform_tractogram_file(
            input_path, output_path, transform.inverse(), reference=reference
        )
    else:
        surface = transform_surface(
            load_surface(input_path), transform.inverse(), reference=reference
        )
        save_surface(surface, output_path)


# The file extensions of images and of tractograms.
_IMAGE_EXTENSIONS = tuple(e for exts in VOLUME_EXTENSIONS.values() for e in exts)
_TRACT_EXTENSIONS = tuple(e for exts in TRACTOGRAM_EXTENSIONS.values() for e in exts)


def _streams(path: Path) -> bool:
    """Checks whether a tractogram file can be read or written in batches."""
    return path.name.endswith((".tck", ".trk"))


def _check_reorient(
    reorient: Reorientation | None, channels: int, interpolation: Interpolation
) -> None:
    """Checks that `reorient` fits the number of channels and the interpolation.

    Raises:
        ValueError: If it does not.
    """
    if reorient == "peaks":
        if channels % 3 != 0:
            raise ValueError(
                f"Peaks need 3 values per peak along the last axis, got {channels}."
            )
        if interpolation != "nearest":
            raise ValueError(
                "Peaks are resampled with nearest-neighbour interpolation: averaging "
                "the vectors of neighbouring voxels would mix different peaks."
            )
    elif reorient == "tensor" and channels != 6:
        raise ValueError(
            f"A tensor image has 6 values per voxel (xx, xy, xz, yy, yz, zz), got "
            f"{channels}."
        )


def _reoriented(
    values: np.ndarray, jacobians: np.ndarray, reorient: Reorientation
) -> np.ndarray:
    """Returns the values (N, C) with their directions reoriented at each point.

    `jacobians` (N, 3, 3) are those of the transform; the directions are turned by
    their inverses.
    """
    with np.errstate(invalid="ignore"):
        local = np.linalg.inv(jacobians)
    if reorient == "peaks":
        vectors = values.reshape(len(values), -1, 3)
        moved = np.einsum("nij,nkj->nki", local, vectors)
        old = np.linalg.norm(vectors, axis=2, keepdims=True)
        new = np.linalg.norm(moved, axis=2, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            scaled = np.where(new > 0, moved * (old / new), vectors)
        out: np.ndarray = scaled.reshape(values.shape).astype(values.dtype)
        return out
    xx, xy, xz, yy, yz, zz = values.T
    tensors = np.stack([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], axis=0).transpose(
        2, 0, 1
    )
    # R = (F F^T)^(-1/2) F = U V^T from F = U S V^T; NaN where F is undefined.
    rotation = np.full_like(local, np.nan)
    defined = np.isfinite(local).all(axis=(1, 2))
    u, _, vt = np.linalg.svd(local[defined])
    rotation[defined] = u @ vt
    turned = rotation @ tensors @ rotation.transpose(0, 2, 1)
    index = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
    result: np.ndarray = turned[:, index[0], index[1]].astype(values.dtype)
    return result
