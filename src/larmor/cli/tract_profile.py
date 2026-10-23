"""The `lar tract-profile` command, which samples maps along a bundle."""

import typing
from enum import StrEnum
from pathlib import Path

import typer

from larmor.cli.options import ForceOpt, ThreadsOpt
from larmor.cli.outputs import check_outputs


class Weighting(StrEnum):
    """Ways of averaging the streamlines' values at one profile point."""

    gaussian = "gaussian"
    mean = "mean"


PathsArg = typing.Annotated[
    list[Path],
    typer.Argument(
        help="The bundle (.tck, .trk, .vtk or .fib, also gzipped), then one or "
        "more maps (.nii, .nii.gz, .mif, .mif.gz or .mih), then the output table "
        "(.csv). Each map's column in the table is named after its file name.",
        metavar="BUNDLE MAP... OUTPUT",
        show_default=False,
    ),
]
NumPointsOpt = typing.Annotated[
    int,
    typer.Option(
        "--num-points",
        help="Number of points, at least 2, at which the profile samples the bundle.",
    ),
]
WeightingOpt = typing.Annotated[
    Weighting,
    typer.Option(
        "--weighting",
        help="How the streamlines' values are averaged at each profile point: "
        "'gaussian' weights each streamline by its distance from the bundle's core, "
        "and 'mean' weights all streamlines the same.",
    ),
]
StartOpt = typing.Annotated[
    str | None,
    typer.Option(
        "--start",
        help="A point x,y,z in mm (world coordinates) near the end of the bundle "
        "where the profile starts. By default, the profile starts at the left, "
        "posterior or inferior end along the bundle's main axis.",
    ),
]
TsfPrefixOpt = typing.Annotated[
    str | None,
    typer.Option(
        "--tsf-prefix",
        help="Also write TSF files with one value per point of the bundle. "
        "<prefix>_segment.tsf holds the index of the profile point each point "
        "belongs to, and <prefix>_<map>.tsf holds the map's profile value there.",
    ),
]
ReferenceOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--reference",
        help="Image whose grid MRIToolkit tracts (.mat) are read on, since they "
        "store none; give the fit (.mat) they came from, or a NIfTI or MIF file of "
        "its image.",
    ),
]


def tract_profile(
    paths: PathsArg,
    num_points: NumPointsOpt = 100,
    weighting: WeightingOpt = Weighting.gaussian,
    start: StartOpt = None,
    tsf_prefix: TsfPrefixOpt = None,
    reference: ReferenceOpt = None,
    force: ForceOpt = False,
    jobs: ThreadsOpt = -1,
) -> None:
    """Write the profile of maps (FA, MD and so on) along a bundle as a CSV table."""
    import os

    from larmor.io import (
        load_grid,
        load_tractogram,
        load_volume,
        write_tsf,
    )
    from larmor.tractography import tract_profile

    if len(paths) < 3:
        raise typer.BadParameter(
            "give the bundle, at least one map and the output",
            param_hint="BUNDLE MAP... OUTPUT",
        )
    bundle_path, *map_paths, output = paths
    if output.suffix.lower() != ".csv":
        raise typer.BadParameter(
            f"the output must be a .csv file, got {output}",
            param_hint="BUNDLE MAP... OUTPUT",
        )
    names = [_map_name(p) for p in map_paths]
    if len(set(names)) != len(names):
        raise typer.BadParameter(
            f"the maps' names must differ, got {', '.join(names)}", param_hint="MAP"
        )
    if num_points < 2:
        raise typer.BadParameter("must be at least 2", param_hint="--num-points")
    n_cores = os.cpu_count() or 1
    if jobs == 0 or jobs < -1 or jobs > n_cores:
        raise typer.BadParameter(
            f"must be -1 (all) or between 1 and {n_cores}", param_hint="--nt"
        )
    start_point = None if start is None else _point(start)
    tsf_paths = []
    if tsf_prefix is not None:
        tsf_paths = [Path(f"{tsf_prefix}_{n}.tsf") for n in ("segment", *names)]
    outputs = [output, *tsf_paths]
    inputs = [bundle_path, *map_paths, *([reference] if reference else [])]
    _check_files_differ(outputs, inputs)
    check_outputs(outputs, force)

    grid = None if reference is None else load_grid(reference)
    bundle = load_tractogram(bundle_path, reference=grid)
    maps = {n: load_volume(p) for n, p in zip(names, map_paths, strict=True)}
    profile = tract_profile(
        bundle,
        maps,
        num_points=num_points,
        weighting=weighting.value,
        start=start_point,
        n_jobs=jobs,
    )
    profile.write_csv(output)
    if tsf_paths:
        segments = profile.segments(bundle)
        write_tsf(segments.astype("float32"), bundle.offsets, tsf_paths[0])
        for name, path in zip(names, tsf_paths[1:], strict=True):
            per_point = profile.values[name][segments].astype("float32")
            write_tsf(per_point, bundle.offsets, path)
    typer.echo(
        f"Profiled {bundle.num_streamlines} streamlines at {num_points} points "
        f"into {output}."
    )


def _check_files_differ(outputs: list[Path], inputs: list[Path]) -> None:
    """Checks that the outputs differ from each other and from the inputs."""
    written = [p.resolve() for p in outputs]
    if len(set(written)) != len(written):
        raise typer.BadParameter(
            "the output files must differ: a map named 'segment' gives the same TSF "
            "file as the segments",
            param_hint="--tsf-prefix",
        )
    if clash := {p.resolve() for p in inputs}.intersection(written):
        raise typer.BadParameter(
            f"an output is also an input: {', '.join(sorted(map(str, clash)))}",
            param_hint="BUNDLE MAP... OUTPUT",
        )


def _map_name(path: Path) -> str:
    """Returns a map's column name, its file name without the image extension."""
    for extension in (".nii.gz", ".nii", ".mif.gz", ".mif", ".mih"):
        if path.name.endswith(extension):
            return path.name.removesuffix(extension)
    return path.stem


def _point(text: str) -> tuple[float, float, float]:
    """Parses x,y,z in mm."""
    try:
        x, y, z = (float(v) for v in text.split(","))
    except ValueError:
        raise typer.BadParameter(
            f"expected x,y,z in mm, got {text!r}", param_hint="--start"
        ) from None
    return x, y, z
