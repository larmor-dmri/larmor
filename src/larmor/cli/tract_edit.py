"""The `lar tract-edit` command, which selects, crops and resamples streamlines."""

import typing
from pathlib import Path

import typer

from larmor.cli.options import ForceOpt

PathsArg = typing.Annotated[
    list[Path],
    typer.Argument(
        help="One or more input tractograms, joined in the order given, then the "
        "output tractogram. Each file can be .tck, .trk, .vtk or .fib, also "
        "gzipped (.gz); an input can also be MRIToolkit tracts (.mat).",
        metavar="INPUT... OUTPUT",
        show_default=False,
    ),
]

_REGION = (
    "A region is a sphere x,y,z,r in mm (world coordinates) or a mask image; "
    "repeat the option for several regions"
)

IncludeOpt = typing.Annotated[
    list[str] | None,
    typer.Option(
        "--include",
        help=f"Keep the streamlines that pass through this region. {_REGION}.",
    ),
]
IncludeOrderedOpt = typing.Annotated[
    list[str] | None,
    typer.Option(
        "--include-ordered",
        help="Keep the streamlines that pass through these regions in the order "
        f"given. {_REGION}.",
    ),
]
ExcludeOpt = typing.Annotated[
    list[str] | None,
    typer.Option(
        "--exclude",
        help=f"Drop the streamlines that pass through this region. {_REGION}.",
    ),
]
WithinOpt = typing.Annotated[
    list[str] | None,
    typer.Option(
        "--within",
        help=f"Keep the streamlines that stay entirely inside this region. {_REGION}.",
    ),
]
MaskOpt = typing.Annotated[
    list[str] | None,
    typer.Option(
        "--mask",
        "-m",
        help=f"Crop the streamlines to their parts inside these regions. {_REGION}.",
    ),
]
EndsOnlyOpt = typing.Annotated[
    bool,
    typer.Option(
        "--ends-only",
        help="Test only the two endpoints against --include, --include-ordered, "
        "--exclude and --within.",
    ),
]
InverseOpt = typing.Annotated[
    bool,
    typer.Option(
        "--inverse",
        help="Invert the selection: keep the streamlines that fail at least one "
        "length, weight or region test, and crop them to their parts outside "
        "--mask.",
    ),
]
MinLengthOpt = typing.Annotated[
    float | None,
    typer.Option("--min-length", help="Keep streamlines at least this long, in mm."),
]
MaxLengthOpt = typing.Annotated[
    float | None,
    typer.Option("--max-length", help="Keep streamlines at most this long, in mm."),
]
MinWeightOpt = typing.Annotated[
    float | None,
    typer.Option(
        "--min-weight",
        help="Keep streamlines with at least this weight. The weights come from "
        "--weights-in or from the weight property of .trk inputs.",
    ),
]
MaxWeightOpt = typing.Annotated[
    float | None,
    typer.Option("--max-weight", help="Keep streamlines with at most this weight."),
]
SkipOpt = typing.Annotated[
    int | None,
    typer.Option(
        "--skip",
        help="Drop the first N selected streamlines.",
    ),
]
EveryOpt = typing.Annotated[
    int | None,
    typer.Option(
        "--every",
        help="Keep every N-th selected streamline, starting with the first one "
        "after --skip.",
    ),
]
NumberOpt = typing.Annotated[
    int | None,
    typer.Option("--number", help="Keep the first N selected streamlines."),
]
SubsampleOpt = typing.Annotated[
    int | None,
    typer.Option(
        "--subsample",
        help="Keep N selected streamlines chosen at random. It cannot be combined "
        "with --number.",
    ),
]
SeedOpt = typing.Annotated[
    int | None,
    typer.Option(
        "--seed",
        help="Seed of the random choice of --subsample, and valid only with it. The "
        "same seed gives the same subset.",
    ),
]
StepSizeOpt = typing.Annotated[
    float | None,
    typer.Option(
        "--step-size",
        help="Resample the output streamlines to this step size, in mm. It cannot "
        "be combined with --num-points.",
    ),
]
NumPointsOpt = typing.Annotated[
    int | None,
    typer.Option(
        "--num-points",
        help="Resample the output streamlines to this many points each, at least 2.",
    ),
]
WeightsInOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--weights-in",
        help="Text file with one weight per streamline of a single input, such as "
        "the output of tcksift2. It replaces the weight property of a .trk input.",
    ),
]
WeightsOutOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--weights-out",
        help="Write the weights of the output streamlines to this file.",
    ),
]
TsfInOpt = typing.Annotated[
    list[Path] | None,
    typer.Option(
        "--tsf-in",
        help="TSF file with one value per point of a single input. The values are "
        "named after the file name without its extension. Repeat the option for "
        "several files.",
    ),
]
TsfOutOpt = typing.Annotated[
    list[str] | None,
    typer.Option(
        "--tsf-out",
        help="Name of the per-point values to write for the output streamlines, a "
        "--tsf-in name or a .trk scalar. They go to <output>_<name>.tsf, where "
        "<output> is the output path without its extension. Repeat the option for "
        "several names.",
    ),
]
ReferenceOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--reference",
        help="Image whose grid a .trk output is written for; it is needed unless "
        "every input is a .trk. MRIToolkit tracts (.mat) store no grid and need it "
        "too; for them, give the fit (.mat) they came from, or a NIfTI or MIF file "
        "of its image.",
    ),
]
LazyOpt = typing.Annotated[
    bool,
    typer.Option(
        "--lazy",
        help="Read, edit and write the streamlines in batches, for files too large "
        "for memory. It works with uncompressed .tck and .trk files only, and not "
        "with --subsample or TSF files.",
    ),
]


def tract_edit(
    paths: PathsArg,
    include: IncludeOpt = None,
    include_ordered: IncludeOrderedOpt = None,
    exclude: ExcludeOpt = None,
    within: WithinOpt = None,
    mask: MaskOpt = None,
    ends_only: EndsOnlyOpt = False,
    inverse: InverseOpt = False,
    min_length: MinLengthOpt = None,
    max_length: MaxLengthOpt = None,
    min_weight: MinWeightOpt = None,
    max_weight: MaxWeightOpt = None,
    skip: SkipOpt = None,
    every: EveryOpt = None,
    number: NumberOpt = None,
    subsample: SubsampleOpt = None,
    seed: SeedOpt = None,
    step_size: StepSizeOpt = None,
    num_points: NumPointsOpt = None,
    weights_in: WeightsInOpt = None,
    weights_out: WeightsOutOpt = None,
    tsf_in: TsfInOpt = None,
    tsf_out: TsfOutOpt = None,
    reference: ReferenceOpt = None,
    lazy: LazyOpt = False,
    force: ForceOpt = False,
) -> None:
    """Select, crop and resample the streamlines of one or more tractograms."""
    from larmor.tractography import EditOptions, edit_files

    if len(paths) < 2:
        raise typer.BadParameter(
            "give at least one input and the output", param_hint="INPUT... OUTPUT"
        )
    *inputs, output = paths
    options = EditOptions(
        min_length=min_length,
        max_length=max_length,
        min_weight=min_weight,
        max_weight=max_weight,
        include=include or [],
        include_ordered=include_ordered or [],
        exclude=exclude or [],
        within=within or [],
        ends_only=ends_only,
        inverse=inverse,
        mask=mask or [],
        skip=skip,
        every=every,
        number=number,
        subsample=subsample,
        seed=seed,
        step_size=step_size,
        num_points=num_points,
    )
    num_read, num_written = edit_files(
        inputs,
        output,
        options,
        weights_in=weights_in,
        weights_out=weights_out,
        tsf_in=tsf_in or [],
        tsf_out=tsf_out or [],
        reference=reference,
        force=force,
        lazy=lazy,
    )
    typer.echo(f"Read {num_read} streamlines, wrote {num_written} to {output}.")
