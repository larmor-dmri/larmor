"""The `lar denoise` command, which denoises a DWI with MP-PCA."""

import logging
import typing
from pathlib import Path

import typer

from larmor._log import LarmorLogger
from larmor.cli.options import DiffusionArg, ForceOpt, ThreadsOpt
from larmor.cli.outputs import check_outputs
from larmor.io import (
    VOLUME_EXTENSIONS,
    has_mif_gradients,
    load_dwi,
    load_volume,
    save_volume,
)

OutArg = typing.Annotated[
    str,
    typer.Argument(
        help="Path of the denoised volume. A .mif output keeps the gradient table "
        "of a .mif input; any other output has none."
    ),
]

NoiseOpt = typing.Annotated[
    str | None,
    typer.Option(
        "--noise",
        help="Also write the noise map, the estimated noise standard deviation "
        "(sigma) of each voxel, to this path.",
    ),
]

MaskOpt = typing.Annotated[
    str | None,
    typer.Option(
        "--mask",
        "-m",
        help="Denoise only the voxels inside this mask and set the others to 0. "
        "Without it, every voxel is denoised.",
    ),
]

PatchOpt = typing.Annotated[
    int | None,
    typer.Option(
        "--patch-size",
        help="Width in voxels, odd, of the cubic patch denoised around each "
        "voxel. Defaults to the smallest odd width whose cube is at least the "
        "number of volumes (5 for up to 125 volumes, 7 for up to 343).",
    ),
]


def denoise(
    volume: DiffusionArg,
    output: OutArg,
    noise: NoiseOpt = None,
    mask: MaskOpt = None,
    patch_size: PatchOpt = None,
    force: ForceOpt = False,
    jobs: ThreadsOpt = -1,
) -> None:
    """Denoise a DWI volume with MP-PCA and estimate its noise level.

    MP-PCA (Marchenko-Pastur principal component analysis) removes, in each
    patch of voxels, the signal components that cannot be told apart from noise.
    """
    from larmor.dwi import denoise_mppca

    outputs = [Path(output)] + ([Path(noise)] if noise is not None else [])
    check_outputs(outputs, force)

    # A MIF input's gradient table goes into a MIF output for the next command.
    gtab = None
    if volume.endswith(VOLUME_EXTENSIONS["mif"]) and has_mif_gradients(volume):
        image, gtab = load_dwi(volume)
    else:
        image = load_volume(volume)
    mask_volume = load_volume(mask) if mask is not None else None
    result = denoise_mppca(
        image,
        mask_volume,
        patch_size=patch_size,
        n_jobs=jobs,
        progress=LarmorLogger.isEnabledFor(logging.WARNING),
    )
    keeps_table = gtab is not None and output.endswith(VOLUME_EXTENSIONS["mif"])
    save_volume(result.volume, output, gradients=gtab if keeps_table else None)
    LarmorLogger.info(f"Saved the denoised volume to {output}")
    if not keeps_table:
        LarmorLogger.info(
            f"{output} has no gradient table: use the input's gradients with it "
            "(its bvals and bvecs files, or its MIF header)."
        )
    if noise is not None:
        save_volume(image.with_data(result.noise), noise)
        LarmorLogger.info(f"Saved the noise map to {noise}")
