"""The `lar sh-convert` command, which converts FODs between SH bases."""

import typing
from pathlib import Path

import numpy as np
import typer

from larmor.cli.options import ForceOpt, SHBasis
from larmor.cli.outputs import check_outputs
from larmor.cli.sh_input import read_layout
from larmor.io import load_volume, save_volume

InputArg = typing.Annotated[
    Path,
    typer.Argument(
        help="Path to the FOD image, with its spherical harmonic (SH) coefficients "
        "along the 4th axis."
    ),
]

OutputArg = typing.Annotated[
    Path,
    typer.Argument(help="Path of the converted FOD image."),
]

BasisFromOpt = typing.Annotated[
    SHBasis,
    typer.Option(
        "--from",
        help="SH basis the input is stored in, which the file does not record.",
    ),
]

BasisToOpt = typing.Annotated[
    SHBasis,
    typer.Option("--to", help="SH basis to convert to."),
]


def sh_convert(
    input_path: InputArg,
    output_path: OutputArg,
    from_basis: BasisFromOpt,
    to_basis: BasisToOpt,
    force: ForceOpt = False,
) -> None:
    """Convert an FOD image from one spherical harmonic (SH) basis to another."""
    from larmor.sd import convert_sh_basis

    check_outputs([output_path], force)

    volume = load_volume(input_path)
    if volume.data.ndim != 4:
        raise typer.BadParameter(
            f"expected a 4D image of SH coefficients, got shape {volume.data.shape}",
            param_hint="INPUT_PATH",
        )
    read_layout(input_path, volume.data.shape[3])

    # Read scaled, since flipping the sign of raw values is wrong with an intercept.
    # The conversion is exact, so float64 stays float64; integers become float32.
    dtype = (
        volume.data.dtype
        if np.issubdtype(volume.data.dtype, np.floating)
        else np.float32
    )
    converted = convert_sh_basis(
        volume.as_float(dtype=dtype),
        from_basis.value,
        to_basis.value,
    )
    save_volume(volume.with_data(converted), output_path)
