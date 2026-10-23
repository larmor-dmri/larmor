import threading
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from types import SimpleNamespace
from typing import cast

import pytest
from rendercanvas import BaseRenderCanvas

from larmor.viewer.render import worker as worker_module
from larmor.viewer.render.worker import Worker


class _Loop:
    def __init__(self) -> None:
        self.calls: list[tuple[Callable[..., None], tuple[object, ...]]] = []

    def call_soon_threadsafe(self, f: Callable[..., None], *args: object) -> None:
        self.calls.append((f, args))

    def run(self) -> None:
        calls, self.calls = self.calls, []
        for f, args in calls:
            f(*args)


class _AtOnce(Executor):
    # Runs each request on submit, so the tests decide when the loop runs.
    def submit[**P, T](
        self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        future: Future[T] = Future()
        future.set_result(fn(*args, **kwargs))
        return future


def _canvas(loop: _Loop) -> BaseRenderCanvas:
    return cast(BaseRenderCanvas, SimpleNamespace(work_loop=loop))


def _flushed(canvas: BaseRenderCanvas, *workers: Worker) -> bool:
    # A flush that never ends fails the test instead of hanging the run; closing
    # the workers then stops its loop.
    thread = threading.Thread(target=worker_module.flush, args=([canvas],), daemon=True)
    thread.start()
    thread.join(2)
    ended = not thread.is_alive()
    for worker in workers:
        worker.close()
    thread.join(2)
    return ended


def test_flush_waits_and_hands_back_only_the_newest_result_at_once() -> None:
    loop = _Loop()
    release = threading.Event()
    worker = Worker(_canvas(loop), ThreadPoolExecutor(1))
    done: list[str] = []

    def first() -> str:
        release.wait(5)
        return "first"

    worker.submit(first, done.append)
    worker.submit(lambda: "second", done.append)
    assert worker.pending
    threading.Timer(0.05, release.set).start()
    worker.flush()
    assert done == ["second"]
    assert not worker.pending
    loop.run()
    assert done == ["second"]


@pytest.mark.parametrize("by", ["loop", "flush"])
def test_a_result_is_pending_until_it_is_handed_back(by: str) -> None:
    loop = _Loop()
    worker = Worker(_canvas(loop), _AtOnce())
    done: list[int] = []
    assert not worker.pending
    worker.submit(lambda: 1, done.append)
    assert worker.pending and done == []
    if by == "loop":
        loop.run()
    else:
        worker.flush()
    assert not worker.pending and done == [1]


def test_flushing_canvases_finishes_the_work_a_result_starts() -> None:
    loop = _Loop()
    canvas = _canvas(loop)
    first, second = Worker(canvas, _AtOnce()), Worker(canvas, _AtOnce())
    done: list[str] = []

    def chain(result: str) -> None:
        done.append(result)
        second.submit(lambda: "second", done.append)

    first.submit(lambda: "first", chain)
    assert _flushed(canvas)
    assert done == ["first", "second"]
    assert not first.pending and not second.pending


def test_flushing_leaves_the_workers_of_other_canvases() -> None:
    loop = _Loop()
    worker = Worker(_canvas(loop), _AtOnce())
    done: list[int] = []
    worker.submit(lambda: 1, done.append)
    assert _flushed(_canvas(loop))
    assert done == [] and worker.pending


def test_a_closed_worker_is_not_flushed() -> None:
    loop = _Loop()
    canvas = _canvas(loop)
    worker = Worker(canvas, _AtOnce())
    done: list[int] = []
    worker.submit(lambda: 1, done.append)
    worker.close()
    assert not worker.pending
    assert _flushed(canvas)
    loop.run()
    assert done == []


@pytest.mark.parametrize("then", ["cancel", "newer request fails"])
def test_flush_ends_when_the_waiting_result_is_no_longer_the_newest(
    then: str,
) -> None:
    loop = _Loop()
    canvas = _canvas(loop)
    worker = Worker(canvas, _AtOnce())
    done: list[int] = []
    worker.submit(lambda: 1, done.append)
    if then == "cancel":
        worker.cancel()
    else:

        def fail() -> int:
            raise RuntimeError("no result")

        worker.submit(fail, done.append)
    assert _flushed(canvas, worker)
    assert done == []


def test_an_older_run_ending_last_does_not_hide_the_newest_result() -> None:
    # Pools with more than one thread (tractograms, FODs) can run two requests
    # of one worker at the same time.
    loop = _Loop()
    pool = ThreadPoolExecutor(2)
    worker = Worker(_canvas(loop), pool)
    started, release, second_ended = (threading.Event() for _ in range(3))
    done: list[str] = []

    def first() -> str:
        started.set()
        release.wait(5)
        return "first"

    worker.submit(first, done.append)
    assert started.wait(5)
    worker.submit(lambda: "second", done.append, ended=second_ended.set)
    assert second_ended.wait(5)
    release.set()
    pool.shutdown(wait=True)
    loop.run()
    assert done == ["second"]


def test_a_failure_replaced_by_a_newer_request_is_not_handed_back() -> None:
    loop = _Loop()
    worker = Worker(_canvas(loop), _AtOnce())
    failed: list[str] = []
    done: list[str] = []

    def broken() -> str:
        raise RuntimeError("stale")

    worker.submit(broken, done.append, lambda: failed.append("first"))
    worker.submit(lambda: "second", done.append, lambda: failed.append("second"))
    loop.run()
    assert (done, failed) == (["second"], [])
    worker.submit(broken, done.append, lambda: failed.append("third"))
    loop.run()
    assert failed == ["third"]  # the newest request's failure still is
