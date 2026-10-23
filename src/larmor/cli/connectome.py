"""The `lar connectome` command, which builds a structural connectome."""

import typing
from enum import StrEnum
from pathlib import Path

import typer

from larmor.cli.options import ForceOpt, ThreadsOpt
from larmor.cli.outputs import check_outputs


class Assignment(StrEnum):
    """Ways of assigning a streamline's endpoint to a region."""

    radial = "radial"
    end_voxel = "end-voxel"


TractogramArg = typing.Annotated[
    Path,
    typer.Argument(
        help="Input tractogram (.tck, .trk, .vtk, .fib, each also .gz).",
        show_default=False,
    ),
]
ParcellationArg = typing.Annotated[
    Path,
    typer.Argument(
        help="Parcellation image of integer region labels, with 0 for no region "
        "(.nii, .nii.gz, .mif, .mif.gz or .mih). The matrix has a row for every "
        "label from 1 to the largest, so relabel sparse labels first.",
        show_default=False,
    ),
]
OutputArg = typing.Annotated[
    Path,
    typer.Argument(
        help="Path of the matrix. A .csv file is comma-separated, with the region "
        "names in the first row and column; any other extension gives a "
        "space-separated file with the names on a comment line.",
        show_default=False,
    ),
]
LutOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--lut",
        help="Lookup table, in FreeSurfer or MRtrix layout, whose names label the "
        "rows and columns. Without it, they are labelled with the label numbers.",
    ),
]
AssignmentOpt = typing.Annotated[
    Assignment,
    typer.Option(
        "--assignment",
        help="How each streamline endpoint is assigned to a region. 'radial' takes "
        "the labelled voxel whose centre is nearest the endpoint, if it is within "
        "--radius (the endpoint's own voxel included), and no region otherwise; "
        "'end-voxel' takes the endpoint's own voxel only.",
    ),
]
RadiusOpt = typing.Annotated[
    float | None,
    typer.Option(
        "--radius",
        help="Search radius of --assignment radial, in mm. Defaults to 4. It "
        "cannot be combined with --assignment end-voxel.",
    ),
]
WeightsInOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--weights-in",
        help="Text file with one weight per streamline, such as the output of "
        "tcksift2. Each entry is then the sum of the weights instead of the number "
        "of streamlines.",
    ),
]
MeanLengthOpt = typing.Annotated[
    bool,
    typer.Option(
        "--mean-length",
        help="Make each entry the mean length in mm of the streamlines between the "
        "two regions, weighted by --weights-in if given. It cannot be combined "
        "with --mean-values.",
    ),
]
MeanValuesOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--mean-values",
        help="File with one value per streamline, in the format of a weights file. "
        "Each entry is then the mean of these values over the streamlines between "
        "the two regions, weighted by --weights-in if given.",
    ),
]
SymmetricOpt = typing.Annotated[
    bool,
    typer.Option(
        "--symmetric",
        help="Write the full symmetric matrix. By default, only the upper triangle "
        "is filled and the lower triangle is 0.",
    ),
]
ZeroDiagonalOpt = typing.Annotated[
    bool,
    typer.Option(
        "--zero-diagonal",
        help="Set the diagonal (streamlines within one region) to 0.",
    ),
]
AssignmentsOutOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--assignments-out",
        help="Write the two regions assigned to the endpoints of each streamline "
        "to this file, one streamline per line.",
    ),
]


def connectome(
    tractogram: TractogramArg,
    parcellation: ParcellationArg,
    output: OutputArg,
    lut: LutOpt = None,
    assignment: AssignmentOpt = Assignment.radial,
    radius: RadiusOpt = None,
    weights_in: WeightsInOpt = None,
    mean_length: MeanLengthOpt = False,
    mean_values: MeanValuesOpt = None,
    symmetric: SymmetricOpt = False,
    zero_diagonal: ZeroDiagonalOpt = False,
    assignments_out: AssignmentsOutOpt = None,
    nt: ThreadsOpt = -1,
    force: ForceOpt = False,
) -> None:
    """Count the streamlines between each pair of regions (a structural connectome)."""
    from larmor.io import (
        load_tractogram,
        load_volume,
        read_lookup_table,
        read_streamline_weights,
    )
    from larmor.tractography import (
        build_connectome,
        endpoint_nodes,
        node_names,
        streamline_lengths,
        write_connectome,
        write_node_assignments,
    )

    if mean_length and mean_values is not None:
        raise typer.BadParameter(
            "give at most one of --mean-length and --mean-values",
            param_hint="--mean-length",
        )
    if radius is not None and assignment == Assignment.end_voxel:
        raise typer.BadParameter(
            "--radius applies only to --assignment radial", param_hint="--radius"
        )
    outputs = [output] if assignments_out is None else [output, assignments_out]
    check_outputs(outputs, force)
    table = None if lut is None else read_lookup_table(lut)
    parcels = load_volume(parcellation)
    streamlines = load_tractogram(tractogram)
    nodes = endpoint_nodes(
        streamlines,
        parcels,
        radius=(4.0 if radius is None else radius)
        if assignment == Assignment.radial
        else None,
        n_jobs=nt,
    )
    weights = None if weights_in is None else read_streamline_weights(weights_in)
    values = None
    if mean_length:
        values = streamline_lengths(streamlines)
    elif mean_values is not None:
        values = read_streamline_weights(mean_values)
    names = node_names(parcels, lookup_table=table)
    num_nodes = len(names)
    matrix = build_connectome(
        nodes,
        num_nodes,
        weights=weights,
        values=values,
        symmetric=symmetric,
        zero_diagonal=zero_diagonal,
    )
    write_connectome(matrix, output, names=names)
    if assignments_out is not None:
        write_node_assignments(nodes, assignments_out)
    assigned = int((nodes > 0).all(axis=1).sum())
    typer.echo(
        f"Assigned {assigned} of {len(nodes)} streamlines to {num_nodes} regions."
    )
