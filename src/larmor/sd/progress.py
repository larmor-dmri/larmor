"""Reports the progress of the FOD fits to a caller other than the terminal."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar

# A function given the voxels fitted so far, the voxels to fit, and the stage.
type Report = Callable[[int, int, str], None]

# The stage reported before the responses are estimated inside a fit.
RESPONSE_STAGE = "Estimating responses"

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


def report_stage(stage: str) -> None:
    """Passes a new stage, with no voxel count, to the `fit_progress` block, if any.

    It lets a caller cancel a fit from `report` before a long stage, such as the
    response estimation, starts.
    """
    report = _report.get()
    if report is not None:
        report(0, 0, stage)
