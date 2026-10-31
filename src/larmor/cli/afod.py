"""The `lar afod` command, which turns symmetric FODs into asymmetric ones."""

import typing
from enum import StrEnum
from pathlib import Path

import numpy as np
import typer

from larmor._log import LarmorLogger
from larmor.cli.options import ForceOpt, SHBasis
from larmor.cli.outputs import check_outputs
from larmor.io import load_volume, save_volume


class AfodMethod(StrEnum):
    """How the asymmetric FODs are made."""

    filter = "filter"


FodArg = typing.Annotated[
    Path,
    typer.Argument(
        help="Path to the symmetric FOD image, with its spherical harmonic (SH) "
        "coefficients of the even degrees along the 4th axis."
    ),
]
OutputArg = typing.Annotated[
    Path,
    typer.Argument(
        help="Path of the asymmetric FOD image, with the SH coefficients of every "
        "degree up to the input's lmax, in the input's basis."
    ),
]
MaskOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--mask",
        "-m",
        help="Voxels to make asymmetric; neighbours outside the mask are not used, "
        "and voxels outside it are 0. Without it, every voxel whose FOD is not all "
        "zero.",
    ),
]
MethodOpt = typing.Annotated[
    AfodMethod,
    typer.Option(
        "--method",
        help="filter: each direction of a voxel's FOD is averaged with the same "
        "direction in the neighbours it points at (Poirier and Descoteaux 2024).",
    ),
]
SigmaOpt = typing.Annotated[
    float,
    typer.Option(
        "--sigma",
        help="Width in mm of the Gaussian that weights the neighbours by distance.",
    ),
]
SharpnessOpt = typing.Annotated[
    float,
    typer.Option(
        "--sharpness",
        help="Exponent of the weight cos^n of the angle between a direction and a "
        "neighbour; higher values use only the neighbours closest to it.",
    ),
]
BasisOpt = typing.Annotated[
    SHBasis,
    typer.Option(
        "--basis",
        help="SH basis the FOD is stored in, which the file does not record; the "
        "output is in the same basis.",
    ),
]


def afod(
    fod_path: FodArg,
    output_path: OutputArg,
    mask_path: MaskOpt = None,
    method: MethodOpt = AfodMethod.filter,
    sigma: SigmaOpt = 1.25,
    sharpness: SharpnessOpt = 2.0,
    basis: BasisOpt = SHBasis.tournier,
    force: ForceOpt = False,
) -> None:
    """Make asymmetric FODs from symmetric ones."""
    from larmor.sd import filter_afod, sh_layout

    if not sigma > 0:
        raise typer.BadParameter("must be positive", param_hint="--sigma")
    if not sharpness > 0:
        raise typer.BadParameter("must be positive", param_hint="--sharpness")
    check_outputs([output_path], force)

    fod = load_volume(fod_path)
    if fod.data.ndim != 4:
        raise typer.BadParameter(
            f"expected a 4D image of SH coefficients, got shape {fod.data.shape}",
            param_hint="FOD_PATH",
        )
    _, full = sh_layout(fod.data.shape[3])
    if full:
        raise typer.BadParameter(
            f"has {fod.data.shape[3]} coefficients, every SH degree: it is already "
            "asymmetric",
            param_hint="FOD_PATH",
        )
    mask = None if mask_path is None else load_volume(mask_path).data.astype(bool)

    # The only method so far; the option leaves room for estimating them.
    assert method is AfodMethod.filter
    data = filter_afod(
        fod.as_float(dtype=np.float32),
        mask,
        fod.affine,
        sigma=sigma,
        sharpness=sharpness,
        basis=basis.value,
    )
    save_volume(fod.with_data(data), output_path)
    LarmorLogger.info(f"Saved the asymmetric FODs to {output_path}")
