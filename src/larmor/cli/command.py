"""Adds the verbosity flags, logging and error handling to every command."""

import functools
import inspect
import logging
import sys
from collections.abc import Callable

import typer

from larmor._log import LarmorLogger
from larmor.cli.options import DebugOpt, InfoOpt, QuietOpt


class _StderrHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Writes each message to sys.stderr as it is at that moment.

    A plain StreamHandler would miss a sys.stderr replaced later, as the test
    runner does.
    """

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        super().emit(record)


_STDERR_HANDLER = _StderrHandler()
_STDERR_HANDLER.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

_VERBOSITY_PARAMS = [
    inspect.Parameter(
        name, inspect.Parameter.KEYWORD_ONLY, default=False, annotation=opt
    )
    for name, opt in (("quiet", QuietOpt), ("info", InfoOpt), ("debug", DebugOpt))
]


def larmor_command(command: Callable[..., None]) -> Callable[..., None]:
    """Adds the verbosity flags and the error handling that every command shares."""

    @functools.wraps(command)
    def wrapper(
        *args: object, quiet: bool, info: bool, debug: bool, **kwargs: object
    ) -> None:
        level = logging.WARNING
        if quiet:
            level = logging.ERROR
        if info:
            level = logging.INFO
        if debug:
            level = logging.DEBUG

        LarmorLogger.setLevel(level)
        LarmorLogger.addHandler(_STDERR_HANDLER)
        try:
            command(*args, **kwargs)
        except typer.Exit, typer.Abort, typer.TyperException:
            raise
        except Exception as e:
            err = str(e) or type(e).__name__
            LarmorLogger.error(err, exc_info=debug)
            raise typer.Exit(code=1) from None

    signature = inspect.signature(command)
    wrapper.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=[*signature.parameters.values(), *_VERBOSITY_PARAMS]
    )

    return wrapper
