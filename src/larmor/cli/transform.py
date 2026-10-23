"""The `lar transform` command, which applies a registration to a file."""

import typing
from enum import StrEnum
from pathlib import Path

import typer

from larmor.cli.options import ForceOpt
from larmor.cli.outputs import check_outputs


class Interpolation(StrEnum):
    """Image interpolation of `lar transform`, as in `transform_volume`."""

    nearest = "nearest"
    linear = "linear"
    cubic = "cubic"


class Reorient(StrEnum):
    """Kinds of directions an image can store, as in `transform_volume`."""

    peaks = "peaks"
    tensor = "tensor"


class WarpType(StrEnum):
    """Ways a warp field can store where each voxel comes from."""

    deformation = "deformation"
    displacement = "displacement"
    relative = "relative"
    absolute = "absolute"


class TransformFormat(StrEnum):
    """Tools whose transform files are read."""

    ants = "ants"
    fsl = "fsl"
    mrtrix = "mrtrix"


InputArg = typing.Annotated[
    Path,
    typer.Argument(
        help="File to bring into the reference space, an image, a tractogram "
        "(.tck, .trk, .vtk or .fib) or a surface (GIfTI .gii or FreeSurfer)."
    ),
]
OutputArg = typing.Annotated[
    Path,
    typer.Argument(help="Path of the output file, of the same kind as the input."),
]
TransformOpt = typing.Annotated[
    list[str],
    typer.Option(
        "--transform",
        "-t",
        help="A transform file. Repeat the option for several transforms, in the "
        "order in which they map points from the reference space, so the first one "
        "applies to the reference grid; for an ANTs registration, this is "
        "-t 1Warp.nii.gz -t 0GenericAffine.mat. The files can be ANTs affines "
        "(.mat, .txt) and warps (5D NIfTI), FLIRT matrices (.mat), FSL warp fields, "
        "and MRtrix3 linear transforms (.txt) and deformation fields. A warp can be "
        "given with its inverse as 'warp.nii.gz,inverse_warp.nii.gz'; otherwise it "
        "is inverted numerically where needed.",
    ),
]
ReferenceOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--reference",
        help="An image of the reference space, usually the fixed image of the "
        "registration. Images are resampled onto its grid, and a tractogram or "
        "surface takes it as its reference grid. It is needed for an image input "
        "and for a .trk output.",
    ),
]
InverseOpt = typing.Annotated[
    bool,
    typer.Option(
        "--inverse",
        help="Apply the transforms the other way: bring the input from the "
        "reference space into the space of the registration's moving image.",
    ),
]
InterpolationOpt = typing.Annotated[
    Interpolation,
    typer.Option(
        "--interp",
        help="Interpolation used to resample an image: 'nearest' (for labels), "
        "'linear' or 'cubic' (B-spline).",
    ),
]
ReorientOpt = typing.Annotated[
    Reorient | None,
    typer.Option(
        "--reorient",
        help="Reorient the directions stored in an image. 'peaks' is for a peaks "
        "image as lar peaks writes it and needs --interp nearest, and 'tensor' is "
        "for a diffusion tensor image of the 6 elements xx, xy, xz, yy, yz and zz.",
    ),
]
FormatOpt = typing.Annotated[
    TransformFormat | None,
    typer.Option(
        "--format",
        help="Tool that wrote the transforms, for files that do not tell, such as "
        "a 4D NIfTI warp or a 4x4 text matrix not named .mat and without an MRtrix3 "
        "command_history comment, which can come from FSL or MRtrix3.",
    ),
]
FslInputOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--fsl-input",
        help="Input image of the FLIRT or FNIRT run, for FSL transforms. Defaults "
        "to the input file when it is an image, or to --reference with --inverse.",
    ),
]
FslReferenceOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--fsl-reference",
        help="Reference image of the FLIRT run, for FLIRT matrices. Defaults to "
        "--reference, or to the input file with --inverse.",
    ),
]


WarpTypeOpt = typing.Annotated[
    WarpType | None,
    typer.Option(
        "--warp-type",
        help="How the warp fields store where each voxel comes from. MRtrix3 fields "
        "store positions ('deformation', the default) or offsets ('displacement'), "
        "and FSL fields relative offsets ('relative', the default) or absolute "
        "positions ('absolute'). The type is checked against the field's values.",
    ),
]


def transform(
    input_path: InputArg,
    output_path: OutputArg,
    transforms: TransformOpt,
    reference_path: ReferenceOpt = None,
    inverse: InverseOpt = False,
    interpolation: InterpolationOpt = Interpolation.linear,
    reorient: ReorientOpt = None,
    format: FormatOpt = None,
    warp_type: WarpTypeOpt = None,
    fsl_input: FslInputOpt = None,
    fsl_reference: FslReferenceOpt = None,
    force: ForceOpt = False,
) -> None:
    """Apply an ANTs, FSL or MRtrix3 registration to an image, tracts or surface.

    An image is resampled onto the reference grid. The points of a tractogram or
    surface are moved into the same space.
    """
    # Imported here, so that starting lar does not load scipy.
    from larmor.registration import transform_file

    check_outputs([output_path], force)
    transform_file(
        input_path,
        output_path,
        [_parse(spec) for spec in transforms],
        reference_path=reference_path,
        inverse=inverse,
        interpolation=interpolation.value,
        reorient=None if reorient is None else reorient.value,
        format=None if format is None else format.value,
        warp_type=None if warp_type is None else warp_type.value,
        fsl_input_path=fsl_input,
        fsl_reference_path=fsl_reference,
    )


def _parse(spec: str) -> str | tuple[str, str]:
    """Parses one --transform value, a file or a file and its inverse."""
    paths = spec.split(",")
    if len(paths) > 2 or not all(paths):
        raise typer.BadParameter(
            f"expected a file or 'file,inverse_file', got {spec!r}",
            param_hint="--transform",
        )
    return paths[0] if len(paths) == 1 else (paths[0], paths[1])
