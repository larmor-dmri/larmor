"""Argument and option types shared by several CLI commands."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

InfoArg = Annotated[
    Path,
    typer.Argument(help="Path to an image or tractogram file."),
]

DiffusionArg = Annotated[
    str,
    typer.Argument(
        help="Path to the diffusion volume (.nii, .nii.gz, .mif, .mif.gz or .mih)."
    ),
]

# A .mif with a gradient table needs neither file; given, they replace its table.
BValsOpt = Annotated[
    str | None,
    typer.Option(
        "--bvals",
        help="Path to the b-values file. It is not needed for a .mif file with a "
        "gradient table in its header; given with such a file, --bvals and --bvecs "
        "replace that table.",
    ),
]
BVecsOpt = Annotated[
    str | None,
    typer.Option(
        "--bvecs",
        help="Path to the b-vectors file. It is not needed for a .mif file with a "
        "gradient table in its header; given with such a file, --bvals and --bvecs "
        "replace that table.",
    ),
]


class BvecFrame(StrEnum):
    """Coordinate frame in which the b-vectors are given."""

    fsl = "fsl"
    world = "world"


class SHBasis(StrEnum):
    """Spherical harmonic basis of a set of SH coefficients."""

    tournier = "tournier"
    descoteaux = "descoteaux"
    leemans = "leemans"


BvecFrameOpt = Annotated[
    BvecFrame,
    typer.Option(
        "--bvec-frame",
        help="Coordinate frame of the b-vectors. 'fsl' means the frame of FSL "
        "bvecs files, along the image axes with x flipped when the affine "
        "determinant is positive, and 'world' means scanner coordinates (RAS).",
    ),
]

ForceOpt = Annotated[
    bool,
    typer.Option(
        "--force",
        help="Overwrite existing output files. Without it, the command stops "
        "before any work when an output exists.",
    ),
]

QuietOpt = Annotated[bool, typer.Option("--quiet", help="Show only errors.")]
InfoOpt = Annotated[
    bool, typer.Option("--info", help="Show progress and informational messages.")
]
DebugOpt = Annotated[
    bool, typer.Option("--debug", help="Show debug messages and full tracebacks.")
]

ThreadsOpt = Annotated[
    int,
    typer.Option(
        "--nt",
        help="Number of threads; -1 uses all CPU cores.",
    ),
]
