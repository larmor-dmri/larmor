"""Reports the progress of the FOD fits to a caller other than the terminal."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar

# A function given the voxels fitted so far, the voxels to fit, and the stage.
type Report = Callable[[int, int, str], None]

_report: ContextVar[Report | None] = ContextVar("report", default=None)


@contextmanager
def fit_progress(report: Report) -> Generator[None]:
    """Calls `report` as the FOD fits run inside the block fit their voxels.

    It covers `fit` of SDModel, CSDModel and MSMTCSDModel run in the same thread.
    An exception raised by `report` stops the fit and passes through it, so a
    caller can cancel a fit from `report`.

    Args:
        report: Called with the voxels fitted so far, the voxels to fit, and the
            stage, once before the first voxel and after each chunk of voxels.
    """
    token = _report.set(report)
    try:
        yield
    finally:
        _report.reset(token)


def report_voxels(done: int, total: int) -> None:
    """Passes the voxels fitted so far to the `fit_progress` block, if any."""
    report = _report.get()
    if report is not None:
        report(done, total, "Fitting voxels")
