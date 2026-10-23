"""The job runner: long jobs on their own thread, results on the main thread."""

import queue
import threading
from collections.abc import Callable

import pytest

from larmor.viewer.jobs import JobBusyError, JobRunner, JobStatus, Report

WAIT = 10.0  # seconds


class Recorder:
    """A runner whose posted calls run when the test drains them."""

    def __init__(self) -> None:
        self.posted: queue.Queue[Callable[[], None]] = queue.Queue()
        self.statuses: list[JobStatus | None] = []
        self.errors: list[str] = []
        self.results: list[object] = []
        self.jobs = JobRunner(
            self.posted.put, on_status=self.statuses.append, on_error=self.errors.append
        )

    def drain(self) -> None:
        """Runs posted calls, as the main thread would, until the job ends."""
        while self.jobs.busy:
            self.posted.get(timeout=WAIT)()


def test_progress_arrives_in_order_then_the_result() -> None:
    rec = Recorder()

    def work(report: Report) -> str:
        for i in range(3):
            report(i + 1, 3, "Signal")
        return "done"

    rec.jobs.start("Simulation", work, rec.results.append)
    rec.drain()
    assert rec.statuses == [
        JobStatus("Simulation", 0, 0, ""),
        JobStatus("Simulation", 1, 3, "Signal"),
        JobStatus("Simulation", 2, 3, "Signal"),
        JobStatus("Simulation", 3, 3, "Signal"),
        None,
    ]
    assert rec.results == ["done"]
    assert rec.errors == []


def test_a_cancelled_job_delivers_nothing() -> None:
    rec = Recorder()
    started, go, ended = threading.Event(), threading.Event(), threading.Event()

    def work(report: Report) -> str:
        try:
            started.set()
            go.wait(WAIT)
            report(1, 2, "")  # raises: cancelled meanwhile
            return "result"
        finally:
            ended.set()

    rec.jobs.start("Job", work, rec.results.append, rec.errors.append)
    assert started.wait(WAIT)
    rec.jobs.cancel()
    # Busy until its thread stops, so no second job writes beside it.
    with pytest.raises(JobBusyError, match="stopping"):
        rec.jobs.start("Second", lambda report: None, rec.results.append)
    go.set()
    assert ended.wait(WAIT)
    for _ in range(int(WAIT * 100)):
        if not rec.jobs.busy:
            break
        threading.Event().wait(0.01)
    assert not rec.jobs.busy
    while not rec.posted.empty():
        rec.posted.get()()
    assert rec.results == []
    assert rec.errors == []
    assert rec.statuses[-1] is None


def test_a_failing_job_is_reported_as_failed() -> None:
    rec = Recorder()

    def work(report: Report) -> None:
        raise ValueError("no tissue")

    rec.jobs.start("Simulation", work, rec.results.append)
    rec.drain()
    assert rec.results == []
    assert rec.errors == ["Simulation failed: no tissue"]
    assert rec.statuses[-1] is None


def test_a_second_job_is_refused_while_one_runs() -> None:
    rec = Recorder()
    go = threading.Event()
    rec.jobs.start("First", lambda report: go.wait(WAIT), rec.results.append)
    with pytest.raises(JobBusyError, match="First"):
        rec.jobs.start("Second", lambda report: None, rec.results.append)
    go.set()
    rec.drain()
    assert rec.results == [True]
    rec.jobs.start("Second", lambda report: 2, rec.results.append)
    rec.drain()
    assert rec.results == [True, 2]
