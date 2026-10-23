"""The `lar bv` command, which opens the viewer."""

from pathlib import Path
from typing import Annotated, Any

import typer

_SEVERAL = "Repeat the option for several files."

FilesArg = Annotated[
    list[Path] | None,
    typer.Argument(
        help="Files to open, sorted by extension into tractograms (.tck, .trk, "
        ".vtk, .fib), a session (.lar.json) and volumes (any other file). Other "
        ".json files are skipped.",
        show_default=False,
    ),
]

VolumesOpt = Annotated[
    list[Path] | None,
    typer.Option(
        "--volumes", "-v", help=f"Volume files, such as a T1 or an FA map. {_SEVERAL}"
    ),
]
OverlaysOpt = Annotated[
    list[Path] | None,
    typer.Option(
        "--overlays",
        "-o",
        help="Overlay files shown over the volumes, such as masks, maps and "
        f"parcellations. {_SEVERAL}",
    ),
]
TractsOpt = Annotated[
    list[Path] | None,
    typer.Option("--tracts", help=f"Tractogram files. {_SEVERAL}"),
]
FodsOpt = Annotated[
    list[Path] | None,
    typer.Option(
        "--fods",
        "-f",
        help="FOD files, drawn as glyphs in the 2D views, with their SH "
        "coefficients along the 4th axis. They are read in the tournier basis "
        f"unless another basis is chosen in the viewer. {_SEVERAL}",
    ),
]
PeaksOpt = Annotated[
    list[Path] | None,
    typer.Option(
        "--peaks",
        "-p",
        help="Peaks files as lar peaks writes them, drawn as lines in the 2D views. "
        f"{_SEVERAL}",
    ),
]
SurfacesOpt = Annotated[
    list[Path] | None,
    typer.Option(
        "--surfaces",
        help="Surface files, GIfTI (.gii) or FreeSurfer. A FreeSurfer surface such "
        "as lh.white is read together with its lh.curv and ../label/lh.cortex.label "
        f"when they exist. {_SEVERAL}",
    ),
]
MaxStreamlinesOpt = Annotated[
    int | None,
    typer.Option(
        "--max-streamlines",
        min=0,
        help="A tractogram with more streamlines is reduced to a random subset of "
        "this size when read, the same subset every time. Defaults to the read "
        "limit in the viewer's Settings (1,000,000 unless changed).",
    ),
]
SessionOpt = Annotated[
    Path | None,
    typer.Option(
        "--session",
        "-s",
        help="Session saved from the viewer (File > Save Session). It is opened "
        "first, and the files of the other options are added to it.",
    ),
]
PhantomOpt = Annotated[
    Path | None,
    typer.Option(
        "--phantom",
        help="Phantom file (.phantom.json) to open in the Phantom panel, to edit "
        "and simulate.",
    ),
]
SimulationOpt = Annotated[
    Path | None,
    typer.Option(
        "--simulation",
        help="Simulation folder written by lar simulate. The viewer opens its "
        "DWI, tissue masks, ground-truth peaks and bundles, and shows its phantom in "
        "the Phantom panel.",
    ),
]

McpOpt = Annotated[
    bool,
    typer.Option(
        "--mcp",
        help="Let AI agents control the viewer through the Model Context Protocol "
        "(MCP), served at http://127.0.0.1:<port>/mcp. The URL and the access token "
        "are printed.",
    ),
]
McpPortOpt = Annotated[
    int | None,
    typer.Option(
        "--mcp-port",
        min=1024,
        max=65535,
        help="Port for --mcp; giving it also turns on --mcp. Defaults to the port "
        "in the viewer's Settings (8765 unless changed).",
    ),
]


InstallLauncherOpt = Annotated[
    bool,
    typer.Option(
        "--install-launcher",
        help="Add a launcher for the viewer of this Larmor installation to the "
        "desktop (the application menu on Linux, ~/Applications on macOS, the "
        "Start menu on Windows), then exit.",
    ),
]
UninstallLauncherOpt = Annotated[
    bool,
    typer.Option(
        "--uninstall-launcher",
        help="Remove what --install-launcher added, then exit.",
    ),
]


def bv(
    files: FilesArg = None,
    volumes: VolumesOpt = None,
    overlays: OverlaysOpt = None,
    tracts: TractsOpt = None,
    fods: FodsOpt = None,
    peaks: PeaksOpt = None,
    surfaces: SurfacesOpt = None,
    max_streamlines: MaxStreamlinesOpt = None,
    session: SessionOpt = None,
    phantom: PhantomOpt = None,
    simulation: SimulationOpt = None,
    mcp: McpOpt = False,
    mcp_port: McpPortOpt = None,
    install_launcher: InstallLauncherOpt = False,
    uninstall_launcher: UninstallLauncherOpt = False,
) -> None:
    """Open the Larmor viewer, with the given files loaded."""
    if install_launcher or uninstall_launcher:
        viewing = (
            files,
            volumes,
            overlays,
            tracts,
            fods,
            peaks,
            surfaces,
            session,
            phantom,
            simulation,
        )
        if any(viewing) or max_streamlines is not None or mcp or mcp_port is not None:
            raise typer.BadParameter(
                "--install-launcher and --uninstall-launcher take no files or "
                "viewer options."
            )
        from larmor.viewer import launcher

        if uninstall_launcher:
            removed = launcher.uninstall_launcher()
            for path in removed:
                typer.echo(f"Removed {path}")
            if not removed:
                typer.echo("No launcher was installed.")
        if install_launcher:
            for path in launcher.install_launcher():
                typer.echo(f"Wrote {path}")
        return

    from larmor.viewer import view
    from larmor.viewer.loading import DEFAULT, sort_files

    given = sort_files(files or [])
    if len(given.sessions) + (session is not None) > 1:
        raise typer.BadParameter("Give one session: as a file or with --session.")

    # DEFAULT takes the read limit in the viewer's Settings.
    limit = DEFAULT if max_streamlines is None else max_streamlines
    # The MCP arguments are passed only when asked for.
    serve: dict[str, Any] = {}
    if mcp or mcp_port is not None:
        serve = {"mcp": True, "mcp_port": mcp_port}
    view(
        volumes=[*given.volumes, *(volumes or [])] or None,
        overlays=overlays,
        tracts=[*given.tracts, *(tracts or [])] or None,
        fods=fods,
        peaks=peaks,
        surfaces=surfaces,
        max_streamlines=limit,
        session=session or next(iter(given.sessions), None),
        phantom=phantom,
        simulation=simulation,
        **serve,
    )
