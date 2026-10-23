"""Entry point of the `lar` command line tool."""

from typing import Annotated

import typer

from larmor import __version__
from larmor.cli import (
    bv,
    connectome,
    convert,
    degibbs,
    denoise,
    dki,
    dti,
    fod,
    info,
    mask,
    peaks,
    sh_convert,
    simulate,
    track,
    tract_density,
    tract_edit,
    tract_profile,
    transform,
)
from larmor.cli.command import larmor_command

app = typer.Typer(
    help="Command line tools of Larmor, for diffusion MRI analysis, tractography "
    "and simulation.",
    no_args_is_help=True,
)
app.command(name="dti")(larmor_command(dti.dti))
app.command(name="dki")(larmor_command(dki.dki))
app.command(name="denoise")(larmor_command(denoise.denoise))
app.command(name="mask")(larmor_command(mask.mask))
app.command(name="degibbs")(larmor_command(degibbs.degibbs))
app.command(name="fod")(larmor_command(fod.fod))
app.command(name="info")(larmor_command(info.info))
app.command(name="convert")(larmor_command(convert.convert))
app.command(name="sh-convert")(larmor_command(sh_convert.sh_convert))
app.command(name="peaks")(larmor_command(peaks.peaks))
app.command(name="track")(larmor_command(track.track))
app.command(name="tract-edit")(larmor_command(tract_edit.tract_edit))
app.command(name="tract-profile")(larmor_command(tract_profile.tract_profile))
app.command(name="tract-density")(larmor_command(tract_density.tract_density))
app.command(name="connectome")(larmor_command(connectome.connectome))
app.command(name="transform")(larmor_command(transform.transform))
app.command(name="simulate")(larmor_command(simulate.simulate))
app.command(name="bv")(larmor_command(bv.bv))


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"larmor {__version__}")
        raise typer.Exit()


@app.callback()
def _options(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Options of `lar` itself."""


def main() -> None:
    """Run the `lar` command line application."""
    app()


if __name__ == "__main__":
    main()
