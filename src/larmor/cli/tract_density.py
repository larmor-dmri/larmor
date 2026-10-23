"""The `lar tract-density` command, which maps the track density of a tractogram."""

import typing
from pathlib import Path

import typer

from larmor.cli.options import ForceOpt, ThreadsOpt
from larmor.cli.outputs import check_outputs

TractogramArg = typing.Annotated[
    Path,
    typer.Argument(
        help="Input tractogram (.tck, .trk, .vtk, .fib, each also .gz).",
        show_default=False,
    ),
]
ReferenceArg = typing.Annotated[
    Path,
    typer.Argument(
        help="Image whose grid the density is computed on (.nii, .nii.gz, .mif, "
        ".mif.gz, .mih); only its header is read.",
        show_default=False,
    ),
]
OutputArg = typing.Annotated[
    Path,
    typer.Argument(
        help="Path of the density image (.nii, .nii.gz, .mif, .mif.gz, .mih).",
        show_default=False,
    ),
]
VoxelSizeOpt = typing.Annotated[
    str | None,
    typer.Option(
        "--voxel-size",
        help="Voxel size of the output in mm, one value or three as x,y,z. The "
        "output grid then covers the reference's field of view at this voxel size.",
    ),
]
EndsOnlyOpt = typing.Annotated[
    bool,
    typer.Option(
        "--ends-only",
        help="Count the two endpoints of the streamlines in each voxel instead "
        "(endpoint density).",
    ),
]
WeightsInOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--weights-in",
        help="Text file with one weight per streamline, such as the output of "
        "tcksift2. Each streamline then adds its weight instead of 1.",
    ),
]
UpsampleOpt = typing.Annotated[
    int | None,
    typer.Option(
        "--upsample",
        help="Number of pieces each segment between two points of a streamline is "
        "split into, at least 1 (1 adds no points). By default, each segment gets "
        "the fewest pieces that are shorter than a third of the smallest voxel "
        "size. Not used with --ends-only.",
    ),
]
LazyOpt = typing.Annotated[
    bool,
    typer.Option(
        "--lazy",
        help="Read the tractogram in batches, for files too large for memory. It "
        "works with uncompressed .tck and .trk files only.",
    ),
]


def tract_density(
    tractogram: TractogramArg,
    reference: ReferenceArg,
    output: OutputArg,
    voxel_size: VoxelSizeOpt = None,
    ends_only: EndsOnlyOpt = False,
    weights_in: WeightsInOpt = None,
    upsample: UpsampleOpt = None,
    lazy: LazyOpt = False,
    nt: ThreadsOpt = -1,
    force: ForceOpt = False,
) -> None:
    """Compute a track density image, the number of streamlines in each voxel."""
    from larmor.io import (
        iter_tractogram,
        load_grid,
        load_tractogram,
        read_streamline_weights,
        save_volume,
    )
    from larmor.tractography import density_grid, track_density

    check_outputs([output], force)
    grid = load_grid(reference)
    if voxel_size is not None:
        try:
            sizes = [float(v) for v in voxel_size.split(",")]
        except ValueError:
            raise typer.BadParameter(
                f"expected one or three numbers, got {voxel_size!r}",
                param_hint="--voxel-size",
            ) from None
        grid = density_grid(grid, sizes)
    weights = None if weights_in is None else read_streamline_weights(weights_in)
    streamlines = iter_tractogram(tractogram) if lazy else load_tractogram(tractogram)
    density = track_density(
        streamlines,
        grid,
        weights=weights,
        ends_only=ends_only,
        upsample=upsample,
        n_jobs=nt,
    )
    save_volume(density, output)
