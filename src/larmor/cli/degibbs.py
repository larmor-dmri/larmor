"""The `lar degibbs` command, which removes Gibbs ringing from an image."""

import typing
from pathlib import Path

import typer

from larmor._log import LarmorLogger
from larmor.cli.options import ForceOpt, ThreadsOpt
from larmor.cli.outputs import check_outputs
from larmor.io import load_volume, save_volume

VolumeArg = typing.Annotated[
    str,
    typer.Argument(
        help="Image to correct, 3D or 4D (.nii, .nii.gz, .mif, .mif.gz or .mih)."
    ),
]
OutArg = typing.Annotated[str, typer.Argument(help="Path of the corrected image.")]

SliceAxisOpt = typing.Annotated[
    int,
    typer.Option(
        "--slice-axis",
        min=0,
        max=2,
        help="Voxel axis (0, 1 or 2) along which the slices were acquired. The "
        "ringing is removed within each slice, along the other two axes.",
    ),
]
ShiftsOpt = typing.Annotated[
    int,
    typer.Option(
        "--shifts",
        min=1,
        help="Number of sub-voxel shifts tried on each side of zero, spaced "
        "1 / (2 x shifts) voxel apart.",
    ),
]
WindowOpt = typing.Annotated[
    tuple[int, int],
    typer.Option(
        "--window",
        help="Distances in voxels of the nearest and the farthest neighbours over "
        "which the ringing around each voxel is measured.",
    ),
]


def degibbs(
    volume: VolumeArg,
    output: OutArg,
    slice_axis: SliceAxisOpt = 2,
    shifts: ShiftsOpt = 20,
    window: WindowOpt = (1, 3),
    force: ForceOpt = False,
    jobs: ThreadsOpt = -1,
) -> None:
    """Remove Gibbs ringing from an image by local sub-voxel shifts.

    Each voxel is resampled at the sub-voxel shift that minimises the ringing
    around it.
    """
    from larmor.dwi import remove_gibbs_ringing

    out_path = Path(output)
    check_outputs([out_path], force)

    image = load_volume(volume)
    corrected = remove_gibbs_ringing(
        image, slice_axis=slice_axis, num_shifts=shifts, window=window, n_jobs=jobs
    )
    save_volume(corrected, out_path)
    LarmorLogger.info(f"Saved the corrected image to {out_path}")
