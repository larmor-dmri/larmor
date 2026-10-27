"""Reads files on a worker thread and adds them to the viewer."""

import threading
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from larmor._log import LarmorLogger
from larmor.io import ReadCancelledError, file_kind
from larmor.viewer.model.item import Category, Item, ItemId
from larmor.viewer.model.matlab_import import MatlabPart, PlannedItem, read_part
from larmor.viewer.model.registration import Registration
from larmor.viewer.model.registry import MAX_STREAMLINES, item_type
from larmor.viewer.model.session import SUFFIX as SESSION_SUFFIX
from larmor.viewer.model.session import (
    SessionItem,
    SessionReport,
    missing_file,
    read_session,
)
from larmor.viewer.model.viewer import Viewer

# Runs a function on the main thread, later; callable from any thread.
type Post = Callable[[Callable[[], None]], None]
# Called on the main thread with the item a job added, or why it failed.
type Then = Callable[[Item | None, str | None], None]


class Default:
    """Marks an argument as not given, where None means "no limit"; see `DEFAULT`."""


DEFAULT = Default()


@dataclass(frozen=True)
class LoadStatus:
    """Holds what is being read, for the status bar while files load."""

    name: str  # the file's name
    number: int  # its place among the files queued since loading began, from 1
    total: int
    fraction: float  # share of the file read so far, 0 to 1


@dataclass(frozen=True)
class _Job:
    """Holds one file or object waiting to be read and added."""

    source: Path | object  # a file to read, or an object already in memory
    category: Category
    generation: int  # a cancel starts a new generation; older jobs are dropped
    number: int
    max_streamlines: int | None  # the read limit when the job was queued
    item_name: str | None = None  # the name in the scene tree; the file's if None
    then: Then | None = None  # told the outcome instead of on_error
    part: MatlabPart | None = None  # what to read inside a MATLAB file
    registration: Registration | None = None  # brought through it once read

    @property
    def name(self) -> str:
        """Returns the file's name, or the object's type."""
        if isinstance(self.source, Path):
            return self.source.name
        return f"{type(self.source).__name__} object"


class Loader:
    """Reads files one at a time on a worker thread; the main thread adds them.

    The worker only reads; every change to the viewer and every callback runs on
    the main thread, through `post`.
    """

    def __init__(
        self,
        viewer: Viewer,
        post: Post,
        on_status: Callable[[LoadStatus | None], None] = lambda status: None,
        on_error: Callable[[str], None] = lambda message: None,
        max_streamlines: int | None = MAX_STREAMLINES,
    ) -> None:
        """Creates a loader that adds what it reads to viewer."""
        self.viewer = viewer
        self.max_streamlines = max_streamlines
        self._post = post
        self._on_status = on_status
        self._on_error = on_error
        self._lock = threading.Lock()  # guards the queue and the worker thread
        self._jobs: deque[_Job] = deque()
        self._worker: threading.Thread | None = None
        self._generation = 0
        self._total = 0  # files queued since loading began
        self._finished = 0
        self._waiting: dict[int, _Job] = {}  # by id: the jobs told of a cancel

    @property
    def busy(self) -> bool:
        """Returns whether files are still being read or waiting to be."""
        return self._finished < self._total

    def open(
        self,
        sources: Iterable[str | Path | object],
        category: Category,
        max_streamlines: int | None | Default = DEFAULT,
        registration: Registration | None = None,
    ) -> None:
        """Queues files to be read, or objects to be added, as items of category.

        Objects wait their turn behind the files queued before them, so items
        appear in the order given. The read limit defaults to the loader's. Files
        are brought through `registration` once read, on the worker thread.
        """
        _check_category(category)
        was_busy = self.busy
        jobs = []
        limit = self.max_streamlines if max_streamlines is DEFAULT else max_streamlines
        assert not isinstance(limit, Default)
        for source in sources:
            if isinstance(source, (str, Path)):
                source = Path(source).absolute()  # not resolved: symlinks as given
            self._total += 1
            jobs.append(
                _Job(
                    source,
                    category,
                    self._generation,
                    self._total,
                    limit,
                    registration=registration if isinstance(source, Path) else None,
                )
            )
        self._queue(jobs, was_busy)

    def load(
        self,
        path: str | Path,
        category: Category,
        max_streamlines: int | None | Default = DEFAULT,
        *,
        then: Then,
    ) -> None:
        """Queues one file, as `open` does; then is told the item, or why it failed.

        A cancel, or the window closing, comes to then as a failure. The read
        limit defaults to the loader's.
        """
        _check_category(category)
        limit = self.max_streamlines if max_streamlines is DEFAULT else max_streamlines
        assert not isinstance(limit, Default)
        was_busy = self.busy
        self._total += 1
        job = _Job(
            Path(path).absolute(),
            category,
            self._generation,
            self._total,
            limit,
            then=then,
        )
        self._waiting[id(job)] = job
        self._queue([job], was_busy)

    def load_planned(
        self,
        path: str | Path,
        planned: list[PlannedItem],
        on_done: Callable[[list[Item], list[str]], None],
        max_streamlines: int | None | Default = DEFAULT,
    ) -> None:
        """Queues the items an import of a MATLAB file plans, read one by one.

        Once every one is read (or has failed), they get their first options and
        on_done is told the items added and the failures, one sentence each.
        """
        limit = self.max_streamlines if max_streamlines is DEFAULT else max_streamlines
        assert not isinstance(limit, Default)
        source = Path(path).absolute()
        added: list[Item | None] = [None] * len(planned)
        problems: list[str] = []
        waiting = len(planned)

        def then(index: int, item: Item | None, error: str | None) -> None:
            """Records one planned item; after the last, applies the import."""
            nonlocal waiting
            added[index] = item
            if error is not None:
                problems.append(error)
            waiting -= 1
            if waiting == 0:
                problems.extend(self.viewer._apply_import(planned, added))
                on_done([i for i in added if i is not None], problems)

        if not planned:
            on_done([], [])
            return
        was_busy = self.busy
        jobs = []
        for index, item in enumerate(planned):
            self._total += 1
            job = _Job(
                source,
                item.category,
                self._generation,
                self._total,
                limit,
                item.name,
                partial(then, index),
                item.part,
            )
            self._waiting[id(job)] = job
            jobs.append(job)
        self._queue(jobs, was_busy)

    def open_session(
        self,
        path: str | Path,
        on_done: Callable[[SessionReport], None],
        max_streamlines: int | None | Default = DEFAULT,
    ) -> None:
        """Replaces the scene with a session's, reading its files in the background.

        The session file itself is read at once, and the scene cleared; once
        every file is read (or has failed), the items get their options and the
        views their state, and on_done is told what happened. Cancel stops the
        session: the files read so far stay, with their default options. The
        read limit defaults to the loader's.

        Raises:
            SessionError: If the file is not a session this version reads; the
                scene is then unchanged.
            OSError: If the session file cannot be read.
        """
        session = read_session(Path(path))
        limit = self.max_streamlines if max_streamlines is DEFAULT else max_streamlines
        assert not isinstance(limit, Default)
        self.cancel()
        self.viewer.clear()
        problems = list(session.problems)
        ids: dict[ItemId, ItemId] = {}
        found = _locate_entries(session.items, problems)
        waiting = len(found)

        def finish() -> None:
            """Applies the session's options and views, and tells on_done."""
            problems.extend(self.viewer._apply_session(session, ids))
            for problem in problems:
                LarmorLogger.warning(problem)
            present = {item.id for item in self.viewer.scene.items}
            items = tuple(
                self.viewer.scene.item(i) for i in ids.values() if i in present
            )
            on_done(SessionReport(items, tuple(problems)))

        def then(saved: ItemId, item: Item | None, error: str | None) -> None:
            """Records one session item; after the last, finishes the session."""
            nonlocal waiting
            if item is not None:
                ids[saved] = item.id
            if error is not None:
                problems.append(error)
            waiting -= 1
            if waiting == 0:
                finish()

        if not found:
            finish()
            return
        was_busy = self.busy
        jobs = []
        for entry, source in found:
            self._total += 1
            jobs.append(
                _Job(
                    source,
                    entry.category,
                    self._generation,
                    self._total,
                    limit,
                    entry.name,
                    partial(then, entry.id),
                    entry.part,
                    entry.registration,
                )
            )
        self._queue(jobs, was_busy)

    def _queue(self, jobs: list[_Job], was_busy: bool) -> None:
        """Hands jobs to the worker thread, starting it if it is not running."""
        if not jobs:
            return
        if not was_busy:  # otherwise the bar shows the file being read
            self._status(jobs[0], 0.0)
        with self._lock:
            self._jobs.extend(jobs)
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._work, name="viewer-loader", daemon=True
                )
                self._worker.start()

    def cancel(self) -> None:
        """Stops the file being read and drops the files waiting."""
        self._generation += 1
        with self._lock:
            self._jobs.clear()
        self._total = self._finished = 0
        waiting, self._waiting = list(self._waiting.values()), {}
        for job in waiting:
            assert job.then is not None
            try:
                job.then(None, f"Loading {job.name} was cancelled.")
            except Exception as error:  # the others are still told
                LarmorLogger.error(f"Telling a cancelled load failed: {error}")
        self._on_status(None)

    def stop(self) -> None:
        """Cancels and reports nothing more, as the window is closing."""
        self._on_status = lambda status: None
        self._on_error = lambda message: None
        self.cancel()

    # The worker thread

    def _work(self) -> None:
        """Takes the queued jobs one by one until none is left."""
        while True:
            with self._lock:
                if not self._jobs:
                    self._worker = None
                    return
                job = self._jobs.popleft()
            if job.generation != self._generation:
                continue
            self._post(partial(self._status, job, 0.0))
            if not isinstance(job.source, Path):
                self._post(partial(self._read, job, job.source))
                continue
            self._read_file(job, job.source)

    def _read_file(self, job: _Job, path: Path) -> None:
        """Reads a job's file and posts its data, or why it failed."""

        def progress(done: int, total: int) -> None:
            """Posts how far the read is; raises ReadCancelledError once cancelled."""
            if job.generation != self._generation:
                raise ReadCancelledError
            fraction = done / total if total > 0 else 1.0
            self._post(partial(self._status, job, fraction))

        try:
            if job.part is not None:
                data = read_part(path, job.part, progress, job.max_streamlines)
            else:
                data = item_type(job.category).load(path, progress, job.max_streamlines)
            if job.registration is not None:
                data = job.registration.apply(data, job.category)
        except ReadCancelledError:
            return
        except Exception as e:  # any failure is reported, never a crash
            if job.generation == self._generation:
                self._post(partial(self._failed, job, e))
            return
        if job.generation == self._generation:  # else cancelled or stopped
            self._post(partial(self._read, job, data))

    # The main thread

    def _status(self, job: _Job, fraction: float) -> None:
        """Tells the status of a job that is still current."""
        if job.generation == self._generation:
            self._on_status(LoadStatus(job.name, job.number, self._total, fraction))

    def _read(self, job: _Job, data: object) -> None:
        """Adds a current job's data to the viewer and tells its callback."""
        if job.generation != self._generation:
            return
        self._waiting.pop(id(job), None)
        path = job.source if isinstance(job.source, Path) else None
        name = job.item_name or (path.name if path else None)
        try:
            item = self.viewer.add(
                data,
                job.category,
                name=name,
                source=path,
                part=job.part,
                registration=job.registration,
            )
        except Exception as e:
            self._failed(job, e)
            return
        try:
            if job.then is not None:
                job.then(item, None)
        finally:  # a failing callback must not leave the loader busy
            self._done()

    def _failed(self, job: _Job, error: Exception) -> None:
        """Logs why a current job failed and tells its callback or on_error."""
        if job.generation != self._generation:
            return
        self._waiting.pop(id(job), None)
        label = item_type(job.category).label.lower()
        what = job.source if isinstance(job.source, Path) else job.name
        message = f"Cannot open {what} as {label}: {_reason(error)}"
        LarmorLogger.error(message)
        try:
            if job.then is not None:
                job.then(None, message)
            else:
                self._on_error(message)
        finally:
            self._done()

    def _done(self) -> None:
        """Counts a finished job, and clears the status once none is left."""
        self._finished += 1
        if not self.busy:
            self._total = self._finished = 0
            self._on_status(None)


@dataclass(frozen=True)
class SortedFiles:
    """Files given without a category, sorted into what they open as."""

    volumes: list[Path]
    tracts: list[Path]
    sessions: list[Path]  # more than one is the caller's to refuse


def sort_files(paths: Iterable[str | Path]) -> SortedFiles:
    """Sorts files by their names: tractograms, sessions (.lar.json) and volumes.

    Any other .json file, such as a BIDS sidecar, is skipped and logged; a file
    of no known kind is taken as a volume, whose reading then reports it.

    Returns:
        SortedFiles: The volumes, tractograms and sessions among the files.
    """
    volumes: list[Path] = []
    tracts: list[Path] = []
    sessions: list[Path] = []
    for path in map(Path, paths):
        name = path.name.lower()  # extensions in any case, as the categories read
        if name.endswith(SESSION_SUFFIX):
            sessions.append(path)
            continue
        if name.endswith(".json"):
            LarmorLogger.info("Skipping %s: a .json file that is not a session.", path)
            continue
        try:
            kind = file_kind(name)
        except ValueError:
            kind = "volume"
        (tracts if kind == "tractogram" else volumes).append(path)
    return SortedFiles(volumes, tracts, sessions)


def _locate_entries(
    entries: Iterable[SessionItem], problems: list[str]
) -> list[tuple[SessionItem, Path]]:
    """Returns the session entries whose files exist, with their paths.

    A missing file is added to problems.
    """
    found = []
    for entry in entries:
        source = entry.locate()
        if source is None:
            problems.append(missing_file(entry))
        else:
            found.append((entry, source))
    return found


def _check_category(category: object) -> None:
    """Raises TypeError if category is not a Category."""
    if not isinstance(category, Category):
        raise TypeError(
            f"Expected a Category, such as Category.VOLUMES; got {category!r}."
        )


def _reason(error: Exception) -> str:
    """Returns why a read failed, in words, without the path an OSError repeats."""
    if isinstance(error, OSError) and error.strerror:
        return f"{error.strerror}."
    return str(error) or type(error).__name__
