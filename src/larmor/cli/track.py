"""The `lar track` command, which tracks streamlines through an FOD image."""

import typing
from enum import StrEnum
from pathlib import Path

import typer

from larmor.cli.options import ForceOpt, SHBasis, ThreadsOpt
from larmor.cli.outputs import check_outputs
from larmor.cli.sh_input import read_layout

if typing.TYPE_CHECKING:
    from larmor.tractography import Seeds, SeedStrategy, TractographyRun

FodArg = typing.Annotated[
    Path,
    typer.Argument(
        help="Path to the FOD image, with its spherical harmonic (SH) coefficients "
        "along the 4th axis. An FOD that stores all SH degrees, not only the even "
        "ones, is read as asymmetric."
    ),
]
OutputArg = typing.Annotated[
    Path,
    typer.Argument(
        help="Path of the tractogram, .tck or .trk, written while tracking runs. "
        "A .trk file also stores why each streamline ended."
    ),
]


class TrackingAlgorithm(StrEnum):
    """Tracking algorithms."""

    det = "det"
    prob = "prob"


AlgorithmOpt = typing.Annotated[
    TrackingAlgorithm,
    typer.Option(
        "--algorithm",
        help="Tracking algorithm. 'det' (deterministic) follows the FOD peak "
        "nearest the current direction, and 'prob' (probabilistic) draws each new "
        "direction at random from the FOD.",
    ),
]

SeedImageOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--seed-image",
        help="Start streamlines at random positions inside this mask until "
        "--select streamlines are accepted. Give exactly one of --seed-image, "
        "--seed-random-per-voxel and --seed-file.",
    ),
]
SeedRandomPerVoxelOpt = typing.Annotated[
    tuple[Path, int] | None,
    typer.Option(
        "--seed-random-per-voxel",
        help="Takes a mask image and a number N, and starts streamlines from N "
        "random positions in every voxel of the mask.",
    ),
]
SeedFileOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--seed-file",
        help="Start streamlines from the seeds in a file written by "
        "--output-seeds. With the same --seed, it tracks the same streamlines "
        "again.",
    ),
]
SelectOpt = typing.Annotated[
    int | None,
    typer.Option(
        "--select",
        help="Track until this many streamlines are accepted. It is required "
        "with --seed-image; without it, the other seed options track every seed.",
    ),
]

MaskOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--mask",
        "-m",
        help="Tracking mask. A streamline stops where it leaves the mask, and a "
        "point is inside where the mask's nearest voxel is at least 0.5.",
    ),
]
UnidirectionalOpt = typing.Annotated[
    bool,
    typer.Option(
        "--unidirectional",
        help="Track from each seed in one direction only, not both.",
    ),
]
StepOpt = typing.Annotated[
    float | None,
    typer.Option(
        "--step-size", help="Step size in mm. Defaults to 0.1 x the voxel size."
    ),
]
AngleOpt = typing.Annotated[
    float | None,
    typer.Option(
        "--angle",
        help="Maximum angle between two consecutive steps, in degrees. The same "
        "angle allows tighter curves at a smaller step. Defaults to 15 for prob, "
        "and for det to the angle that allows curves down to a radius of 0.65 x "
        "the voxel size, 2 asin(step / (1.3 x voxel)), about 8.8 degrees at the "
        "default step.",
    ),
]
MinLengthOpt = typing.Annotated[
    float | None,
    typer.Option(
        "--min-length",
        help="Reject streamlines shorter than this length, in mm. Defaults to 5 x "
        "the voxel size.",
    ),
]
MaxLengthOpt = typing.Annotated[
    float | None,
    typer.Option(
        "--max-length",
        help="Stop streamlines when they reach this length, in mm. Defaults to "
        "100 x the voxel size.",
    ),
]
CutoffOpt = typing.Annotated[
    str,
    typer.Option(
        "--cutoff",
        help="FOD amplitude below which a streamline stops, a number or 'auto'. "
        "'auto' is 0.2 x the mean largest FOD peak of a fixed random sample of up to "
        "5000 voxels, "
        "inside the tracking mask if one is given.",
    ),
]
BasisOpt = typing.Annotated[
    SHBasis,
    typer.Option(
        "--basis",
        help="SH basis the FOD is stored in, which the file does not record.",
    ),
]
RunSeedOpt = typing.Annotated[
    int | None,
    typer.Option(
        "--seed",
        help="Seed of the random number generator, from 0 to 2^64 - 1. The same "
        "seed gives the same tractogram, whatever the number of threads. Without "
        "it, a seed is drawn at random; --info shows it and --json saves it.",
    ),
]
JsonOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--json",
        help="Save a record of the run as JSON: every parameter used, the counts "
        "of seeds and streamlines, and the software versions.",
    ),
]
OutputSeedsOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--output-seeds",
        help="Save the seed point of each accepted streamline, in the order of the "
        "output, to a file that --seed-file reads.",
    ),
]


def track(
    fod_path: FodArg,
    output_path: OutputArg,
    algorithm: AlgorithmOpt = TrackingAlgorithm.det,
    seed_image: SeedImageOpt = None,
    seed_random_per_voxel: SeedRandomPerVoxelOpt = None,
    seed_file: SeedFileOpt = None,
    select: SelectOpt = None,
    mask_path: MaskOpt = None,
    unidirectional: UnidirectionalOpt = False,
    step_size: StepOpt = None,
    angle: AngleOpt = None,
    min_length: MinLengthOpt = None,
    max_length: MaxLengthOpt = None,
    cutoff: CutoffOpt = "auto",
    basis: BasisOpt = SHBasis.tournier,
    run_seed: RunSeedOpt = None,
    json_path: JsonOpt = None,
    output_seeds: OutputSeedsOpt = None,
    force: ForceOpt = False,
    jobs: ThreadsOpt = -1,
) -> None:
    """Track streamlines through an FOD image."""
    import dataclasses
    import logging
    import secrets

    from larmor._log import LarmorLogger
    from larmor.io import load_volume
    from larmor.tractography import Tracker

    _check_options(
        output_path,
        select,
        jobs,
        run_seed,
        seed_image,
        seed_random_per_voxel,
        seed_file,
    )
    tracker = Tracker(
        algorithm.value,
        step_size=step_size,
        max_angle=angle,
        cutoff=_cutoff(cutoff),
        min_length=min_length,
        max_length=max_length,
        bidirectional=not unidirectional,
        progress=LarmorLogger.isEnabledFor(logging.WARNING),
    )
    extra = [p for p in (json_path, output_seeds) if p is not None]
    check_outputs([output_path, *extra], force)

    # Drawn here, so that the seeds per voxel and the tracker share it.
    seed = run_seed if run_seed is not None else secrets.randbits(64)
    seeds = _seeds(seed_image, seed_random_per_voxel, seed_file, seed)
    fod = load_volume(fod_path)
    if fod.data.ndim == 4:
        read_layout(fod_path, fod.data.shape[3])
    mask = None if mask_path is None else load_volume(mask_path)

    run = tracker.track(
        fod,
        output_path,
        seeds,
        num_streamlines=select,
        mask=mask,
        basis=basis.value,
        seed=seed,
        output_seeds=output_seeds,
        n_jobs=jobs,
    )
    inputs = _run_inputs(
        fod_path, mask_path, seed_image, seed_random_per_voxel, seed_file
    )
    run = dataclasses.replace(run, inputs=inputs)
    if json_path is not None:
        run.write_json(json_path)

    _log_run(run, output_path)


def _check_options(
    output_path: Path,
    select: int | None,
    jobs: int,
    run_seed: int | None,
    seed_image: Path | None,
    seed_random_per_voxel: tuple[Path, int] | None,
    seed_file: Path | None,
) -> None:
    """Checks the output path, the counts and that exactly one seed source is given."""
    import os

    if output_path.suffix not in (".tck", ".trk"):
        raise typer.BadParameter("must be a .tck or .trk file", param_hint="OUTPUT")
    if select is not None and select < 1:
        raise typer.BadParameter("must be at least 1", param_hint="--select")
    n_cores = os.cpu_count() or 1
    if jobs == 0 or jobs < -1 or jobs > n_cores:
        raise typer.BadParameter(
            f"must be -1 (all) or between 1 and {n_cores}", param_hint="--nt"
        )
    if run_seed is not None and not 0 <= run_seed < 2**64:
        raise typer.BadParameter("must be in [0, 2^64)", param_hint="--seed")
    sources = [seed_image, seed_random_per_voxel, seed_file]
    if sum(source is not None for source in sources) != 1:
        raise typer.BadParameter(
            "give exactly one of --seed-image, --seed-random-per-voxel and --seed-file",
            param_hint="seeds",
        )
    if seed_random_per_voxel is not None and seed_random_per_voxel[1] < 1:
        raise typer.BadParameter(
            "must be at least 1 seed per voxel", param_hint="--seed-random-per-voxel"
        )
    if seed_image is not None and select is None:
        raise typer.BadParameter(
            "--seed-image draws seeds without end, so it needs --select",
            param_hint="--select",
        )


def _run_inputs(
    fod_path: Path,
    mask_path: Path | None,
    seed_image: Path | None,
    seed_random_per_voxel: tuple[Path, int] | None,
    seed_file: Path | None,
) -> dict[str, object]:
    """Returns the absolute paths of the input files, as the run record keeps them."""
    return {
        "fod": str(fod_path.resolve()),
        "mask": None if mask_path is None else str(mask_path.resolve()),
        "seed_image": None if seed_image is None else str(seed_image.resolve()),
        "seed_random_per_voxel": None
        if seed_random_per_voxel is None
        else [str(seed_random_per_voxel[0].resolve()), seed_random_per_voxel[1]],
        "seed_file": None if seed_file is None else str(seed_file.resolve()),
    }


def _log_run(run: TractographyRun, output_path: Path) -> None:
    """Logs the parameters of the run and the counts of streamlines and seeds."""
    from larmor._log import LarmorLogger

    for name, value in run.parameters.items():
        LarmorLogger.info(f"{name}: {value}")
    if run.sampling is not None:
        for name, value in run.sampling.items():
            LarmorLogger.info(f"{name}: {value}")
    rejected = ", ".join(f"{n} {reason}" for reason, n in run.num_rejected.items())
    LarmorLogger.info(
        f"Saved {run.num_streamlines} streamlines to {output_path} from "
        f"{run.num_seeds} seeds ({rejected})"
    )


def _cutoff(value: str) -> float | str:
    """Parses --cutoff into 'auto' or a non-negative number."""
    if value == "auto":
        return value
    try:
        number = float(value)
    except ValueError:
        raise typer.BadParameter(
            f"expected 'auto' or a number, got {value!r}", param_hint="--cutoff"
        ) from None
    if not number >= 0 or number == float("inf"):
        raise typer.BadParameter(
            f"must be non-negative and finite, got {value}", param_hint="--cutoff"
        )
    return number


def _seeds(
    seed_image: Path | None,
    per_voxel: tuple[Path, int] | None,
    seed_file: Path | None,
    run_seed: int,
) -> Seeds | SeedStrategy:
    """Returns the seeds of the one seed source given."""
    from larmor.io import load_volume
    from larmor.tractography import MaskSeeds, read_seeds, seeds_per_voxel

    if seed_image is not None:
        return MaskSeeds(load_volume(seed_image))
    if per_voxel is not None:
        image, n = per_voxel
        return seeds_per_voxel(load_volume(image), n, run_seed)
    assert seed_file is not None
    return read_seeds(seed_file)
