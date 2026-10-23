"""Reads the transform files of ANTs, FSL and MRtrix3."""

import gzip
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np

from larmor.io import VOLUME_EXTENSIONS, load_volume, read_matlab
from larmor.registration.transforms import (
    AffineTransform,
    DisplacementField,
    Transform,
    TransformChain,
    TransformPair,
)
from larmor.types import Grid

# The tools whose transform files are read.
TransformFormat = Literal["ants", "fsl", "mrtrix"]
# The warp types of each format. MRtrix3 fields hold positions (deformation) or
# offsets (displacement); FSL fields hold offsets (relative) or positions (absolute).
WARP_TYPES = {
    "mrtrix": ("deformation", "displacement"),
    "fsl": ("relative", "absolute"),
}

# ITK's physical space is LPS: x and y point the other way from RAS+.
_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
# The first line of an ITK text transform file.
_ITK_HEADER = "#Insight Transform File"
# The comment MRtrix3 writes at the top of its transform files.
_MRTRIX_HISTORY = "# command_history:"
# The ITK transform types read as an affine.
_ITK_AFFINE = re.compile(
    r"^(AffineTransform|MatrixOffsetTransformBase)_(double|float)_3_3$"
)


def read_ants_transform(path: str | Path) -> AffineTransform | DisplacementField:
    """Reads an affine or a displacement field written by ANTs.

    The transform maps points of the fixed image (the reference grid, `-r` of
    antsApplyTransforms) to the moving image, the direction antsApplyTransforms
    uses for images as listed with `-t`, and antsApplyTransformsToPoints for
    points. Bringing streamlines from the moving into the fixed space needs its
    inverse (for a warp, the `InverseWarp` file antsRegistration writes beside it).

    The transform is converted from ITK's LPS coordinates to RAS+ as read.

    Args:
        path: An affine as a binary `.mat` file (`0GenericAffine.mat`, MATLAB
            version 4) or an ITK text file (`.txt`, `.tfm`), of type
            AffineTransform or MatrixOffsetTransformBase in 3D; or a displacement
            field as a NIfTI file (`1Warp.nii.gz`, (X, Y, Z, 1, 3), mm, LPS).

    Returns:
        The affine or the displacement field, in world coordinates (RAS+) in mm.

    Raises:
        ValueError: If the file is not one of those, or holds another transform
            type (convert it with `ConvertTransformFile --convertToAffineType`).
    """
    path = Path(path)
    if _is_image(path):
        warp = load_volume(path)
        if warp.data.ndim != 4 or warp.data.shape[3] != 3:
            raise ValueError(
                f"{path}: an ANTs warp has 3 displacements per voxel, got shape "
                f"{warp.data.shape}."
            )
        displacements = warp.as_float(dtype=np.float64) * [-1.0, -1.0, 1.0]
        return DisplacementField(warp.grid, displacements)

    if _is_itk_text(path):
        kind, parameters, centre = _read_itk_text(path)
    else:
        kind, parameters, centre = _read_itk_mat(path)
    if not _ITK_AFFINE.match(kind):
        raise ValueError(
            f"{path} holds an ITK {kind}; Larmor reads AffineTransform and "
            "MatrixOffsetTransformBase in 3D. Convert it with ConvertTransformFile "
            "3 <in> <out>.mat --convertToAffineType."
        )
    if parameters.size != 12 or centre.size != 3:
        raise ValueError(
            f"{path}: expected 12 parameters and a 3D centre, got {parameters.size} "
            f"and {centre.size}."
        )
    # ITK maps x to M (x - c) + t + c, with M row by row in the first 9 parameters.
    matrix = parameters[:9].reshape(3, 3)
    lps = np.eye(4)
    lps[:3, :3] = matrix
    lps[:3, 3] = parameters[9:] + centre - matrix @ centre
    return AffineTransform(_LPS @ lps @ _LPS)


def read_flirt_transform(
    path: str | Path, input_grid: Grid, reference_grid: Grid
) -> AffineTransform:
    """Reads an FSL FLIRT matrix (`-omat`) as a transform of world points.

    FLIRT's matrix works in FSL's scaled voxel coordinates of the input and
    reference images, so it needs both grids. The transform returned maps points of
    the reference image to the input image, the inverse of FLIRT's matrix and the
    direction `flirt -applyxfm` uses to resample the input. Moving points from the
    input into the reference space needs its inverse.

    Args:
        path: The text file of 4 rows of 4 numbers.
        input_grid: The grid of FLIRT's `-in` image.
        reference_grid: The grid of FLIRT's `-ref` image.

    Returns:
        The affine, reference to input, in world coordinates (RAS+) in mm.

    Raises:
        ValueError: If the file is not a 4x4 matrix of numbers.
    """
    flirt = _read_matrix(path)
    if flirt.shape != (4, 4):
        raise ValueError(f"{path}: a FLIRT matrix has 4 rows of 4, got {flirt.shape}.")
    to_input = _fsl_to_world(input_grid) @ np.linalg.inv(flirt)
    return AffineTransform(to_input @ np.linalg.inv(_fsl_to_world(reference_grid)))


def read_fsl_warp(
    path: str | Path, input_grid: Grid, *, relative: bool = True
) -> DisplacementField:
    """Reads an FSL warp field (fnirt `--fout`, convertwarp, invwarp) as a transform.

    The field lies on the reference image's grid and holds where each reference
    voxel comes from in the input image, in FSL's scaled voxel coordinates, as a
    displacement (relative, FSL's default) or as the coordinates themselves
    (absolute, `--abs`). The transform maps points of the reference image to the
    input image, the direction applywarp uses to resample the input. Moving points
    the other way needs its inverse (invwarp's output, or the numerical inverse).

    Fields of spline coefficients (fnirt `--cout`) are not read; convert them with
    `fnirtfileutils --withaff` or `convertwarp` first.

    Args:
        path: The NIfTI file, (X, Y, Z, 3).
        input_grid: The grid of the input image (applywarp's `-i`).
        relative: Whether the field holds displacements (True) or absolute
            coordinates. The values are checked against it: absolute coordinates
            lie near each voxel's own coordinates, displacements near zero.

    Returns:
        The displacement field, in world coordinates (RAS+) in mm.

    Raises:
        ValueError: If the file is not a 4D image with 3 volumes, or its values
            look like the other kind of field.
    """
    warp = load_volume(path)
    if warp.data.ndim != 4 or warp.data.shape[3] != 3:
        raise ValueError(
            f"{path}: an FSL warp field has 3 volumes, got shape {warp.data.shape}."
        )
    grid = warp.grid
    voxels = _voxel_centres(grid.shape)
    field = warp.as_float(dtype=np.float64).reshape(-1, 3)
    reference_fsl = _apply(np.linalg.inv(_fsl_to_world(grid)) @ grid.affine, voxels)
    _check_field_type(path, field, reference_fsl, not relative, "absolute", "relative")
    input_fsl = reference_fsl + field if relative else field
    positions = _apply(_fsl_to_world(input_grid), input_fsl)
    displacements = positions - _apply(grid.affine, voxels)
    return DisplacementField(grid, displacements.reshape(*grid.shape, 3))


def read_mrtrix_transform(path: str | Path) -> AffineTransform:
    """Reads an MRtrix3 linear transform (mrregister, transformconvert).

    The matrix is in scanner coordinates (RAS+, mm) and, as mrtransform `-linear`
    uses it, maps points of the template (reference) image to the input image.
    Moving points from the input into the template space needs its inverse.

    Args:
        path: A text file of 3 or 4 rows of 4 numbers; `#` starts a comment.

    Returns:
        The affine, in world coordinates (RAS+) in mm.

    Raises:
        ValueError: If the file is not a 3x4 or 4x4 matrix of numbers.
    """
    matrix = _read_matrix(path)
    if matrix.shape == (3, 4):
        matrix = np.vstack([matrix, [0.0, 0.0, 0.0, 1.0]])
    if matrix.shape != (4, 4):
        raise ValueError(
            f"{path}: an MRtrix3 transform has 3 or 4 rows of 4, got {matrix.shape}."
        )
    return AffineTransform(matrix)


def read_mrtrix_warp(
    path: str | Path, *, displacement: bool = False
) -> DisplacementField:
    """Reads an MRtrix3 deformation or displacement field as a transform.

    A deformation field holds, at each voxel of the template grid, the position
    in the input image it comes from (scanner coordinates, RAS+, mm); a
    displacement field holds that position minus the voxel's own. Either way the
    transform maps template points to input points, as mrtransform `-warp` uses it
    to resample an image; tcktransform moves streamlines from the template space
    into the input space with the same `map_points`. NaN marks voxels without a
    position.

    mrregister's 5D warps (both halves of a symmetric registration) are not read;
    convert one with `warpconvert -type warpfull2deformation` first.

    Args:
        path: The image (`.mif`, `.nii`, ...), (X, Y, Z, 3).
        displacement: Whether the field holds displacements instead of positions.
            The values are checked against it: positions lie near each voxel's
            own position, displacements near zero.

    Returns:
        The displacement field, in world coordinates (RAS+) in mm.

    Raises:
        ValueError: If the image is not (X, Y, Z, 3), or its values look like
            the other kind of field.
    """
    warp = load_volume(path)
    if warp.data.ndim != 4 or warp.data.shape[3] != 3:
        raise ValueError(
            f"{path}: an MRtrix3 warp has 3 volumes, got shape {warp.data.shape}."
        )
    values = warp.as_float(dtype=np.float64)
    grid = warp.grid
    own = _apply(grid.affine, _voxel_centres(grid.shape))
    _check_field_type(
        path,
        values.reshape(-1, 3),
        own,
        not displacement,
        "deformation",
        "displacement",
    )
    if displacement:
        return DisplacementField(grid, values)
    return DisplacementField(grid, values - own.reshape(*grid.shape, 3))


def load_transform(
    path: str | Path,
    *,
    format: str | None = None,
    warp_type: str | None = None,
    fsl_input_grid: Grid | None = None,
    fsl_reference_grid: Grid | None = None,
) -> Transform:
    """Reads a transform file of any supported format, chosen from the file if needed.

    Without `format`: an ITK text header or a binary `.mat` is ANTs, a text `.mat`
    is a FLIRT matrix, and other text files are MRtrix3 when they have MRtrix3's
    `# command_history:` comment or 3 rows of 4 numbers. Any other 4x4 text matrix
    can be FLIRT or MRtrix3, so it needs `format`. A 5D NIfTI is an ANTs warp and a
    `.mif` an MRtrix3 deformation field; a 4D NIfTI warp can come from FSL or
    MRtrix3, so it needs `format` too. Every transform maps reference points to
    input points, as the tools use them for images.

    Args:
        path: The transform file.
        format: "ants", "fsl" or "mrtrix", when the file alone does not tell.
        warp_type: For a warp field, how it stores where each voxel comes from:
            "deformation" (positions, MRtrix3's default) or "displacement" for
            MRtrix3, "relative" (FSL's default) or "absolute" for FSL. Checked
            against the field's values. ANTs warps are always displacements.
        fsl_input_grid: The grid of the FSL run's input image, needed for FSL
            files.
        fsl_reference_grid: The grid of the FSL run's reference image, needed for
            a FLIRT matrix.

    Returns:
        The transform.

    Raises:
        ValueError: If the format is unknown or cannot be told, an FSL file
            lacks a grid, or the warp type does not fit the format or the data.
    """
    path = Path(path)
    image = _is_image(path)
    if format is None:
        format = _guess_format(path, image)
    if format not in ("ants", "fsl", "mrtrix"):
        raise ValueError(f"Unknown transform format {format!r}: ants, fsl or mrtrix.")
    if warp_type is not None and image and warp_type not in WARP_TYPES.get(format, ()):
        allowed = " or ".join(WARP_TYPES.get(format, ())) or "a displacement field"
        raise ValueError(f"{path}: a {format} warp is {allowed}, not {warp_type!r}.")
    if format == "ants":
        return read_ants_transform(path)
    if format == "mrtrix":
        if image:
            return read_mrtrix_warp(path, displacement=warp_type == "displacement")
        return read_mrtrix_transform(path)
    if fsl_input_grid is None or (fsl_reference_grid is None and not image):
        raise ValueError(
            f"{path} is an FSL transform, which needs the grid of the input image "
            "(and for a FLIRT matrix the reference image's) it was made for."
        )
    if image:
        return read_fsl_warp(path, fsl_input_grid, relative=warp_type != "absolute")
    assert fsl_reference_grid is not None
    return read_flirt_transform(path, fsl_input_grid, fsl_reference_grid)


def _load_transforms(
    files: Sequence[str | Path | tuple[str | Path, str | Path]],
    format: str | None,
    warp_type: str | None,
    input_grid: Grid | None,
    reference_grid: Grid | None,
) -> TransformChain:
    """Reads several transform files into one chain, each as `load_transform` does.

    A tuple entry is a pair of a transform file and its inverse file.

    Returns:
        The chain, in the order of `files`.
    """
    transforms: list[Transform] = []
    for entry in files:
        if isinstance(entry, tuple):
            forward, backward = entry
            # The inverse goes from the registration's reference to its input.
            transforms.append(
                TransformPair(
                    load_transform(
                        forward,
                        format=format,
                        warp_type=warp_type,
                        fsl_input_grid=input_grid,
                        fsl_reference_grid=reference_grid,
                    ),
                    load_transform(
                        backward,
                        format=format,
                        warp_type=warp_type,
                        fsl_input_grid=reference_grid,
                        fsl_reference_grid=input_grid,
                    ),
                )
            )
        else:
            transforms.append(
                load_transform(
                    entry,
                    format=format,
                    warp_type=warp_type,
                    fsl_input_grid=input_grid,
                    fsl_reference_grid=reference_grid,
                )
            )
    return TransformChain(transforms)


def load_registration(
    files: Sequence[str | Path | tuple[str | Path, str | Path]],
    *,
    data_grid: Grid | None = None,
    reference_grid: Grid | None = None,
    inverse: bool = False,
    format: str | None = None,
    warp_type: str | None = None,
    fsl_input_grid: Grid | None = None,
    fsl_reference_grid: Grid | None = None,
) -> Transform:
    """Reads the files of a registration as one transform into a reference space.

    Pass the result to `apply_transform` to bring an image, tractogram or surface
    into the reference space. The result maps reference points to data points,
    the direction `transform_volume` takes.

    The files are those that resample an image into the reference space, listed
    in the order antsApplyTransforms takes its `-t` options, so the first file
    takes the reference grid's points. A warp can be given as a pair (file,
    inverse_file) with the inverse its tool wrote (ANTs' `InverseWarp`,
    `invwarp`, `warpinvert`), which is then used instead of inverting the warp
    numerically. With `inverse` the data goes the other way, from the reference
    space into the space the files bring images from.

    FSL files need the grids of the FLIRT or FNIRT run's input and reference
    images. They default to the data's grid and `reference_grid`, swapped with
    `inverse`; data without a grid (streamlines, surfaces) needs
    `fsl_input_grid` going forward and `fsl_reference_grid` with `inverse`.

    Args:
        files: The files, or pairs of a warp and the inverse its tool wrote.
        data_grid: The grid of the data, if it is an image.
        reference_grid: The grid of the reference space.
        inverse: Go from the reference space into the space of the files' inputs.
        format: "ants", "fsl" or "mrtrix", when the files alone do not tell.
        warp_type: The warp type of the warp fields, as in `load_transform`.
        fsl_input_grid: The FSL run's input image grid, if not the default.
        fsl_reference_grid: The FSL run's reference image grid, if not the
            default.

    Returns:
        The transform, reference points to data points.

    Raises:
        ValueError: As `load_transform`.
    """
    run_input, run_reference = (
        (reference_grid, data_grid) if inverse else (data_grid, reference_grid)
    )
    chain = _load_transforms(
        files,
        format,
        warp_type,
        fsl_input_grid if fsl_input_grid is not None else run_input,
        fsl_reference_grid if fsl_reference_grid is not None else run_reference,
    )
    return chain.inverse() if inverse else chain


def _check_field_type(
    path: str | Path,
    values: np.ndarray,
    own: np.ndarray,
    absolute: bool,
    absolute_name: str,
    relative_name: str,
) -> None:
    """Raises if a field read as one warp type has values typical of the other.

    `values` and `own` are the field's values and each voxel's own position, (N, 3);
    positions lie near the voxel's own, offsets near zero, by the median.

    Raises:
        ValueError: If the values look like the other type.
    """
    finite = np.isfinite(values).all(axis=1)
    if not finite.any():
        return
    as_positions = np.median(np.linalg.norm(values[finite] - own[finite], axis=1))
    as_offsets = np.median(np.linalg.norm(values[finite], axis=1))
    if (as_positions < as_offsets) != absolute:
        read, other = (
            (absolute_name, relative_name)
            if absolute
            else (relative_name, absolute_name)
        )
        raise ValueError(
            f"{path} was read as a field of type {read!r}, but its values look like "
            f"type {other!r} (median {as_positions:.3g} mm from each voxel's own "
            f"position, {as_offsets:.3g} mm from zero). Say it is {other} (warp "
            f"type {other!r}, --warp-type on the command line)."
        )


def _guess_format(path: Path, image: bool) -> TransformFormat:
    """Returns the format of a transform file from its name and first bytes.

    Raises:
        ValueError: If the file can be an FSL or an MRtrix3 transform.
    """
    if image:
        if path.name.endswith(VOLUME_EXTENSIONS["mif"]):
            return "mrtrix"
        if _nifti_dimensions(path) == 5:
            return "ants"
        raise ValueError(
            f"{path}: a 4D NIfTI warp can be an FSL or an MRtrix3 field; say which "
            "(format 'fsl' or 'mrtrix', --format on the command line)."
        )
    if _is_itk_text(path):
        return "ants"
    if path.suffix == ".mat":
        return "fsl" if _is_text(path) else "ants"
    if not _is_text(path):
        return "mrtrix"
    text = path.read_text(encoding="utf-8", errors="replace")
    if _MRTRIX_HISTORY in text:
        return "mrtrix"
    try:
        shape = _read_matrix(path).shape
    except ValueError:
        return "mrtrix"  # the MRtrix3 reader says what is wrong with the file
    if shape == (4, 4):
        raise ValueError(
            f"{path}: a 4x4 text matrix can be a FLIRT or an MRtrix3 transform, "
            "which map different coordinates; say which (format 'fsl' or 'mrtrix', "
            "--format on the command line)."
        )
    return "mrtrix"


def _is_image(path: Path) -> bool:
    """Checks whether `path` has the extension of an image."""
    return any(path.name.endswith(exts) for exts in VOLUME_EXTENSIONS.values())


def _is_text(path: Path) -> bool:
    """Checks whether the first bytes of the file are ASCII text."""
    head = path.read_bytes()[:512]
    try:
        head.decode("ascii")
    except UnicodeDecodeError:
        return False
    return b"\0" not in head


def _is_itk_text(path: Path) -> bool:
    """Checks whether the file is an ITK text transform file."""
    return _is_text(path) and path.read_text(errors="replace").startswith(_ITK_HEADER)


def _nifti_dimensions(path: Path) -> int | None:
    """Returns the number of dimensions in a NIfTI-1 header, or None if not NIfTI-1."""
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as f:
        head = f.read(348)
    if len(head) < 348:
        return None
    for order in ("<", ">"):
        if int(np.frombuffer(head, dtype=f"{order}i4", count=1)[0]) == 348:
            return int(np.frombuffer(head, dtype=f"{order}i2", count=1, offset=40)[0])
    return None


def _read_itk_text(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    """Returns the type, parameters and centre of the transform in an ITK text file."""
    fields: dict[str, str] = {}
    transforms = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#Transform "):
            transforms += 1
        elif ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    if transforms > 1:
        raise ValueError(
            f"{path} holds {transforms} transforms; Larmor reads one per file. "
            "Split it with CompositeTransformUtil --disassemble."
        )
    try:
        return (
            fields["Transform"],
            np.array(fields["Parameters"].split(), dtype=np.float64),
            np.array(fields.get("FixedParameters", "0 0 0").split(), dtype=np.float64),
        )
    except (KeyError, ValueError) as e:
        raise ValueError(f"{path} is not an ITK transform file: {e}") from e


def _read_itk_mat(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    """Returns the type, parameters and centre of a binary ITK transform (MATLAB v4)."""
    mat = read_matlab(path)
    kinds = [n for n in mat.names if n != "fixed"]
    if len(kinds) != 1 or "fixed" not in mat:
        raise ValueError(
            f"{path} is not an ITK transform file: expected a transform and "
            f"'fixed', got {mat.names}."
        )
    parameters = np.asarray(mat.read_array(kinds[0]), dtype=np.float64).ravel()
    centre = np.asarray(mat.read_array("fixed"), dtype=np.float64).ravel()
    return kinds[0], parameters, centre


def _read_matrix(path: str | Path) -> np.ndarray:
    """Reads a matrix from a text file of rows of numbers; `#` starts a comment."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].replace(",", " ").strip()
        if text:
            try:
                rows.append([float(v) for v in text.split()])
            except ValueError as e:
                raise ValueError(f"{path} is not a matrix of numbers: {e}") from e
    if not rows or len({len(r) for r in rows}) != 1:
        raise ValueError(f"{path} is not a matrix: rows of the same length expected.")
    return np.array(rows, dtype=np.float64)


def _fsl_to_world(grid: Grid) -> np.ndarray:
    """Returns the 4x4 matrix from FSL's scaled voxel coordinates of a grid to world.

    FSL scales voxel indices by the voxel sizes and, for an affine with a positive
    determinant, counts the first axis from its far end.
    """
    sizes = np.linalg.norm(grid.affine[:3, :3], axis=0)
    voxels_to_fsl = np.diag([*sizes, 1.0])
    if np.linalg.det(grid.affine[:3, :3]) > 0:
        voxels_to_fsl[0, 0] = -sizes[0]
        voxels_to_fsl[0, 3] = (grid.shape[0] - 1) * sizes[0]
    result: np.ndarray = grid.affine @ np.linalg.inv(voxels_to_fsl)
    return result


def _voxel_centres(shape: tuple[int, int, int]) -> np.ndarray:
    """Returns the (X * Y * Z, 3) voxel indices of a grid, the last axis fastest."""
    axes = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    return np.stack(axes, axis=-1).reshape(-1, 3).astype(np.float64)


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Returns the (N, 3) points mapped by the 4x4 affine `matrix`."""
    result: np.ndarray = points @ matrix[:3, :3].T + matrix[:3, 3]
    return result
