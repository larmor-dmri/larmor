"""Runs slow work off the main thread, where only the newest request counts."""

from collections.abc import Callable, Iterable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from functools import partial
from weakref import WeakSet

from rendercanvas import BaseLoop, BaseRenderCanvas

from larmor._log import LarmorLogger

# One thread for all the views' slow work; a request that a newer one has replaced
# is skipped before it starts.
_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="larmor-viewer")


class Worker:
    """Runs one owner's requests in the background and hands the newest result back.

    The result is handed to `done` on the canvas's event loop, the main thread.
    An off-screen canvas has none, so its work runs at once, unless the canvas
    carries an event loop as `work_loop`.
    """

    def __init__(self, canvas: BaseRenderCanvas, pool: Executor = _POOL) -> None:
        """Creates a worker that runs on pool and hands results back to the canvas."""
        # rendercanvas keeps a canvas's loop on its class's group, with no public
        # getter; an off-screen canvas has none.
        group = getattr(canvas, "_rc_canvas_group", None)
        loop = getattr(canvas, "work_loop", None)
        if loop is None and group is not None:
            loop = group.get_loop()
        self._loop = loop
        self.canvas = canvas
        self._pool = pool
        self._generation = 0
        self._closed = False
        self._future: Future[None] | None = None  # the newest request's run
        # The newest request's result, as (generation, the call to `done`), until
        # it is handed back.
        self._outcome: tuple[int, Callable[[], None]] | None = None
        _WORKERS.add(self)

    @property
    def pending(self) -> bool:
        """Returns whether the newest request is still to be handed back."""
        future = self._future
        outcome = self._outcome
        # A result of an older request is never handed back, so it does not count.
        waiting = outcome is not None and outcome[0] == self._generation
        return not self._closed and (
            (future is not None and not future.done()) or waiting
        )

    def submit[T](
        self,
        work: Callable[[], T],
        done: Callable[[T], None],
        failed: Callable[[], None] | None = None,
        ended: Callable[[], None] | None = None,
    ) -> None:
        """Runs `work`, then `done` with its result, unless a newer request came.

        If `work` raises, the error is logged and `failed` is called instead;
        off screen, the error is raised again after it. `ended` is called once the
        request is over (run, failed or skipped). In the background it is called
        on the worker's thread, so it must be thread-safe.
        """
        self._generation += 1
        if self._loop is None:
            _run_now(work, done, failed, ended)
            return
        self._future = self._pool.submit(
            partial(self._run, self._generation, self._loop, work, done, failed, ended)
        )

    def _run[T](
        self,
        generation: int,
        loop: BaseLoop,
        work: Callable[[], T],
        done: Callable[[T], None],
        failed: Callable[[], None] | None,
        ended: Callable[[], None] | None,
    ) -> None:
        """Runs request generation on the pool's thread, and posts its outcome."""
        try:
            if generation != self._generation or self._closed:
                return
            try:
                result = work()
            except Exception:
                LarmorLogger.exception("Background work in the viewer failed.")
                if failed is not None:
                    loop.call_soon_threadsafe(self._fail, generation, failed)
                return
            if generation != self._generation or self._closed:
                return  # replaced or closed while it ran: nothing to hand back
            self._outcome = (generation, lambda: done(result))
            loop.call_soon_threadsafe(self._settle, generation)
        finally:
            if ended is not None:
                ended()

    def flush(self) -> None:
        """Waits for the newest request, and hands its result back now."""
        future = self._future
        if future is not None:
            future.result()
        self._settle(self._generation)

    def _settle(self, generation: int) -> None:
        """Hands back the result of request `generation`, if it is still the newest."""
        outcome = self._outcome
        if outcome is None or outcome[0] != generation:
            return
        self._outcome = None
        if generation == self._generation and not self._closed:
            outcome[1]()

    def _fail(self, generation: int, failed: Callable[[], None]) -> None:
        """Calls `failed` of request `generation`, if it is still the newest."""
        if generation == self._generation and not self._closed:
            failed()

    def cancel(self) -> None:
        """Drops the requests still queued, and the result of the one running."""
        self._generation += 1

    def close(self) -> None:
        """Drops every request still running or queued."""
        self._closed = True
        _WORKERS.discard(self)


# Every open worker, so that the work of some canvases can be finished.
_WORKERS: WeakSet[Worker] = WeakSet()


def _run_now[T](
    work: Callable[[], T],
    done: Callable[[T], None],
    failed: Callable[[], None] | None,
    ended: Callable[[], None] | None,
) -> None:
    """Runs a request at once, for a canvas without an event loop.

    If work raises, failed is called and the error is raised again.
    """
    try:
        try:
            result = work()
        except Exception:
            if failed is not None:
                failed()
            raise
        done(result)
    finally:
        if ended is not None:
            ended()


def flush_all() -> None:
    """Finishes the work of every open worker, and of the work it starts."""
    while pending := [w for w in list(_WORKERS) if w.pending]:
        for worker in pending:
            worker.flush()


def flush(canvases: Iterable[BaseRenderCanvas]) -> None:
    """Finishes the work of the canvases' workers, and of the work it starts."""
    targets = set(map(id, canvases))
    while True:
        pending = [w for w in list(_WORKERS) if id(w.canvas) in targets and w.pending]
        if not pending:
            return
        for worker in pending:
            worker.flush()
