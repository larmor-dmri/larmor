"""Runs long jobs, such as a simulation, one at a time on a thread of their own."""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from larmor._log import LarmorLogger

# Runs a function on the main thread, later; callable from any thread.
type Post = Callable[[Callable[[], None]], None]
# Called by a job with the steps done, their total and the stage's name; raises
# JobCancelledError once the job is cancelled.
type Report = Callable[[int, int, str], None]


# The stage in which a job writes its outputs; from then on a cancel leaves it to
# finish and deliver them.
SAVING = "Saving"


class JobCancelledError(Exception):
    """Raised inside a job by its report once the job is cancelled."""


class JobBusyError(RuntimeError):
    """Raised when a job is started while another one runs."""


@dataclass(frozen=True)
class JobStatus:
    """Holds how far the running job is, for the status bar."""

    name: str  # such as "Simulation"
    done: int  # steps done
    total: int  # steps in all; 0 when not known yet
    stage: str  # what it does now, such as "Signal"; "" for nothing to say

    @property
    def fraction(self) -> float:
        """Returns the share of the steps done, 0 to 1; 0 while the total is unknown."""
        return min(self.done / self.total, 1.0) if self.total > 0 else 0.0


class JobRunner:
    """Runs one job at a time on a thread of its own, and refuses a second one.

    Every callback (status, done, failed) runs on the main thread, through
    `post`. A cancelled job's result and failure are dropped.
    """

    def __init__(
        self,
        post: Post,
        on_status: Callable[[JobStatus | None], None] = lambda status: None,
        on_error: Callable[[str], None] = lambda message: None,
    ) -> None:
        """Creates a runner that tells its status and the jobs' errors."""
        self._post = post
        self._on_status = on_status
        self._on_error = on_error
        self._generation = 0  # a cancel starts a new one; older jobs are dropped
        self._running: str | None = None  # the running job's name
        self._thread: threading.Thread | None = None  # the running job's thread
        # A cancelled job's thread, which runs on until its next report.
        self._cancelled: threading.Thread | None = None
        # The generation of a job that reached SAVING, which a cancel leaves be.
        self._saving: int | None = None
        self._lock = threading.Lock()  # between a report and a cancel

    @property
    def busy(self) -> bool:
        """Returns whether a job runs, or a cancelled one has not stopped yet."""
        return self._running is not None or self._stopping()

    def _stopping(self) -> bool:
        """Returns whether a cancelled job's thread still runs."""
        return self._cancelled is not None and self._cancelled.is_alive()

    def start[T](
        self,
        name: str,
        work: Callable[[Report], T],
        on_done: Callable[[T], None],
        on_failed: Callable[[str], None] | None = None,
    ) -> None:
        """Starts work on its own thread; on_done is told its result.

        Args:
            name: The job's name, shown with its progress.
            work: The job; it calls the report it is given with its progress,
                which raises JobCancelledError once the job is cancelled.
            on_done: Told the job's result, on the main thread.
            on_failed: Told why the job failed, on the main thread; the runner's
                error callback if None.

        Raises:
            JobBusyError: If a job runs already.
        """
        if self._running is not None:
            raise JobBusyError(f"{self._running} is running; wait or cancel it.")
        if self._stopping():
            # Two jobs at once could write the same files.
            raise JobBusyError("The cancelled job is still stopping; try again.")
        self._running = name
        generation = self._generation
        self._on_status(JobStatus(name, 0, 0, ""))
        thread = threading.Thread(
            target=self._work,
            args=(name, generation, work, on_done, on_failed),
            name="viewer-job",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    @property
    def saving(self) -> bool:
        """Returns whether the running job is saving, so it cannot be cancelled."""
        return self._running is not None and self._saving == self._generation

    def cancel(self, *, force: bool = False) -> None:
        """Stops the running job, which then delivers nothing.

        The job's thread ends at its next report; until then the runner stays
        busy and refuses a new job. A job that is saving is left to finish and
        deliver its result, unless force is given.

        Args:
            force: Drop even a job that is saving, as when its window closes.
        """
        with self._lock:
            if self.saving and not force:
                LarmorLogger.info(f"{self._running} is saving; it cannot be cancelled.")
                return
            self._generation += 1
        if self._running is not None:
            LarmorLogger.info(f"{self._running} was cancelled.")
            self._cancelled = self._thread
        self._running = None
        self._on_status(None)

    def stop(self) -> None:
        """Cancels the job and reports nothing more, as the window is closing."""
        self._on_status = lambda status: None
        self._on_error = lambda message: None
        self.cancel(force=True)

    # The job's thread

    def _work[T](
        self,
        name: str,
        generation: int,
        work: Callable[[Report], T],
        on_done: Callable[[T], None],
        on_failed: Callable[[str], None] | None,
    ) -> None:
        """Runs the job on its thread, and posts its result or failure."""

        def report(done: int, total: int, stage: str) -> None:
            """Posts the job's progress; raises JobCancelledError once cancelled."""
            with self._lock:
                if generation != self._generation:
                    raise JobCancelledError
                if stage == SAVING:
                    self._saving = generation
            status = JobStatus(name, done, total, stage)
            self._post(partial(self._status, generation, status))

        try:
            result = work(report)
        except JobCancelledError:
            return
        except Exception as e:  # any failure is reported, never a crash
            message = f"{name} failed: {str(e) or type(e).__name__}"
            self._post(partial(self._failed, generation, message, on_failed))
            return
        self._post(partial(self._done, generation, result, on_done))

    # The main thread

    def _status(self, generation: int, status: JobStatus) -> None:
        """Tells the status of a job that is still current."""
        if generation == self._generation:
            self._on_status(status)

    def _done[T](
        self, generation: int, result: T, on_done: Callable[[T], None]
    ) -> None:
        """Ends a current job and gives on_done its result."""
        if generation != self._generation:
            return
        self._finish()
        on_done(result)

    def _failed(
        self,
        generation: int,
        message: str,
        on_failed: Callable[[str], None] | None,
    ) -> None:
        """Ends a current job, logs its failure and tells it."""
        if generation != self._generation:
            return
        self._finish()
        LarmorLogger.error(message)
        (on_failed or self._on_error)(message)

    def _finish(self) -> None:
        """Ends the job so that a new one may start."""
        self._generation += 1  # its late status calls are dropped
        self._running = None
        self._on_status(None)
