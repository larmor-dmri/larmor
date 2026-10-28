"""The `lar simulate` command, which simulates the DWI of a phantom."""

import typing
from pathlib import Path

import typer

from larmor._log import LarmorLogger
from larmor.cli.options import BvecFrame, ForceOpt, ThreadsOpt

if typing.TYPE_CHECKING:
    from larmor.simulation import Phantom, PhantomFile, Protocol

PhantomArg = typing.Annotated[
    str,
    typer.Argument(
        help="Path to the phantom file (.phantom.json), which describes a voxel "
        "grid, its tissues, fibre bundles and a scan protocol. It can also be the "
        "name of a phantom that ships with Larmor (crossing), used when no file has "
        "that name."
    ),
]
OutputArg = typing.Annotated[
    Path,
    typer.Argument(
        help="Output folder, created if needed. It receives the DWI with its bvals "
        "and bvecs, the ground truth (tissue and bundle fractions, fibre directions "
        "and masks), the bundles as .tck files, a copy of the phantom file and "
        "simulation.json, which together are enough to run the simulation again."
    ),
]
ShellOpt = typing.Annotated[
    list[str] | None,
    typer.Option(
        "--shell",
        help="A shell as B:N, its b-value B in s/mm^2 and its number of "
        "directions N, such as 1000:32. Repeat the option for several shells. The "
        "shells replace the gradients of the phantom file.",
        show_default=False,
    ),
]
NumB0Opt = typing.Annotated[
    int | None,
    typer.Option(
        "--num-b0",
        help="Number of b = 0 volumes added to the --shell gradients. Defaults to 1.",
        show_default=False,
    ),
]
BValsOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--bvals",
        help="Path to a b-values file, given with --bvecs. The two files replace "
        "the gradients of the phantom file.",
    ),
]
BVecsOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--bvecs",
        help="Path to a b-vectors file, given with --bvals.",
    ),
]
BvecFrameOpt = typing.Annotated[
    BvecFrame | None,
    typer.Option(
        "--bvec-frame",
        help="Coordinate frame of the --bvecs file. 'fsl' means the frame of FSL "
        "bvecs files, along the axes of the phantom's grid with x flipped when the "
        "affine determinant is positive, and 'world' means RAS coordinates. "
        "Defaults to fsl.",
        show_default=False,
    ),
]
SnrOpt = typing.Annotated[
    float | None,
    typer.Option(
        "--snr",
        help="Add Rician noise with this signal-to-noise ratio, relative to the "
        "b = 0 signal of the --noise-reference tissue. Without it, no noise is "
        "added.",
        show_default=False,
    ),
]
NoiseReferenceOpt = typing.Annotated[
    str,
    typer.Option(
        "--noise-reference",
        help="Name of the phantom tissue whose b = 0 signal --snr refers to.",
    ),
]
SeedOpt = typing.Annotated[
    int | None,
    typer.Option(
        "--seed",
        help="Seed of the random numbers of the noise, the --shell directions and "
        "the bundles that have no seed of their own. The same seed gives the same "
        "result.",
        show_default=False,
    ),
]


def simulate(
    phantom_path: PhantomArg,
    output_path: OutputArg,
    shells: ShellOpt = None,
    num_b0: NumB0Opt = None,
    bvals: BValsOpt = None,
    bvecs: BVecsOpt = None,
    bvec_frame: BvecFrameOpt = None,
    snr: SnrOpt = None,
    noise_reference: NoiseReferenceOpt = "WM",
    seed: SeedOpt = None,
    force: ForceOpt = False,
    jobs: ThreadsOpt = -1,
) -> None:
    """Simulate the diffusion-weighted images of a phantom."""
    from larmor import simulation

    _check_options(shells, num_b0, bvals, bvecs, bvec_frame, jobs)
    loaded = _load_phantom(phantom_path)
    phantom, protocol = loaded.phantom, loaded.protocol

    if protocol is None:
        raise ValueError(
            f"{phantom_path} has no scan (gradients and timings); write the phantom "
            "file with a protocol."
        )
    protocol = _replace_gradients(
        protocol, phantom, shells, num_b0, bvals, bvecs, bvec_frame, seed
    )
    _check_output_folder(output_path, force)

    simulated = simulation.simulate(
        phantom,
        protocol,
        snr=snr,
        noise_reference=noise_reference,
        seed=seed,
        n_jobs=jobs,
    )
    simulation.save_simulation(
        simulated, output_path, force=force, bundle_names=loaded.bundle_names
    )
    LarmorLogger.info("Wrote the simulation to %s", output_path)


def _check_options(
    shells: list[str] | None,
    num_b0: int | None,
    bvals: Path | None,
    bvecs: Path | None,
    bvec_frame: BvecFrame | None,
    jobs: int,
) -> None:
    """Checks that the gradient and thread options go together."""
    if jobs == 0 or jobs < -1:
        raise typer.BadParameter("must be -1 (all) or at least 1", param_hint="--nt")
    if (bvals is None) != (bvecs is None):
        raise typer.BadParameter("--bvals and --bvecs go together")
    if bvals is not None and shells:
        raise typer.BadParameter("use either --shell or --bvals and --bvecs")
    if num_b0 is not None and not shells:
        raise typer.BadParameter("--num-b0 needs --shell")
    if bvec_frame is not None and bvals is None:
        raise typer.BadParameter("--bvec-frame needs --bvals and --bvecs")


def _load_phantom(phantom_path: str) -> PhantomFile:
    """Reads the phantom file at phantom_path, or the example phantom of that name."""
    from larmor import simulation

    path = Path(phantom_path)
    if path.exists():
        return simulation.read_phantom(path)
    if phantom_path in simulation.EXAMPLE_PHANTOMS:
        return simulation.example_phantom(phantom_path)
    raise FileNotFoundError(
        f"{phantom_path} does not exist and is not the name of an example "
        f"phantom ({', '.join(simulation.EXAMPLE_PHANTOMS)})."
    )


def _replace_gradients(
    protocol: Protocol,
    phantom: Phantom,
    shells: list[str] | None,
    num_b0: int | None,
    bvals: Path | None,
    bvecs: Path | None,
    bvec_frame: BvecFrame | None,
    seed: int | None,
) -> Protocol:
    """Returns the protocol with the gradients of --shell or --bvals and --bvecs.

    The protocol is returned unchanged when neither is given. The new gradients
    keep the b = 0 threshold and shell tolerance of the protocol's.
    """
    from larmor import simulation
    from larmor.io import read_bvals_bvecs
    from larmor.types import GradientTable

    gradients = None
    if shells:
        gradients = simulation.create_gradient_table(
            [_shell(s) for s in shells],
            num_b0=1 if num_b0 is None else num_b0,
            seed=seed,
        )
    elif bvals is not None and bvecs is not None:
        gradients = read_bvals_bvecs(
            bvals, bvecs, frame=(bvec_frame or BvecFrame.fsl).value, grid=phantom.grid
        )
    if gradients is None:
        return protocol
    old = protocol.gradients
    return simulation.Protocol(
        GradientTable(
            gradients.bvals,
            gradients.bvecs,
            b0_threshold=old.b0_threshold,
            shell_epsilon=old.shell_epsilon,
        ),
        protocol.small_delta_ms,
        protocol.big_delta_ms,
        protocol.te_ms,
        protocol.tr_ms,
        protocol.sequence,
    )


def _check_output_folder(output_path: Path, force: bool) -> None:
    """Checks that output_path is a folder that may be written into."""
    if output_path.exists() and not output_path.is_dir():
        raise NotADirectoryError(f"{output_path} exists and is not a folder.")
    if output_path.is_dir() and any(output_path.iterdir()) and not force:
        raise FileExistsError(
            f"Folder {output_path} exists and is not empty. Use --force to replace "
            "the simulation in it."
        )


def _shell(text: str) -> tuple[float, int]:
    """Parses a shell given as B:N into its b-value and number of directions."""
    try:
        b, count = text.split(":")
        return float(b), int(count)
    except ValueError:
        raise typer.BadParameter(
            f"{text!r} is not a shell: write it as B:N, such as 1000:32",
            param_hint="--shell",
        ) from None
