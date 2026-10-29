"""Reports the progress of tracking runs to a caller other than the terminal."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar

# A function given the count done so far, the count to reach (0 when not known),
# and the stage.
type Report = Callable[[int, int, str], None]

_report: ContextVar[Report | None] = ContextVar("report", default=None)


@contextmanager
def track_progress(report: Report) -> Generator[None]:
    """Calls `report` as the tracking runs inside the block go through their seeds.

    It covers `Tracker.track` run in the same thread. An exception raised by
    `report` stops the run and passes through it, so a caller can cancel a run from
    `report`; streamlines already written to a file stay in it.

    Args:
        report: Called once before the first batch of seeds and after each batch.
            With a target number of streamlines it is given the streamlines kept
            and the target; otherwise the seeds tried and the number of seeds, 0
            when the seeds have no fixed number.
    """
    token = _report.set(report)
    try:
        yield
    finally:
        _report.reset(token)


def report_tracking(done: int, total: int, stage: str) -> None:
    """Passes the run's progress to the `track_progress` block, if any."""
    report = _report.get()
    if report is not None:
        report(done, total, stage)
