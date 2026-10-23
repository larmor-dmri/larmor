import dataclasses
import logging
import queue
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import larmor.viewer
from larmor.io import Progress, save_tractogram, save_volume
from larmor.types import Tractogram, Volume
from larmor.viewer import Category, Viewer, loading
from larmor.viewer.loading import Loader, LoadStatus, SortedFiles, sort_files
from larmor.viewer.model.registry import ItemType, item_type

WAIT = 10.0  # seconds; generous, a tiny file reads in milliseconds

type Read = Callable[[Path, Progress | None, int | None], object]


class Recorder:
    """A loader whose posted calls run when the test drains them."""

    def __init__(self) -> None:
        self.posted: queue.Queue[Callable[[], None]] = queue.Queue()
        self.statuses: list[LoadStatus | None] = []
        self.errors: list[str] = []
        self.viewer = Viewer()
        self.loader = Loader(
            self.viewer,
            self.posted.put,
            on_status=self.statuses.append,
            on_error=self.errors.append,
        )

    def drain(self) -> None:
        """Runs posted calls, as the main thread would, until nothing is loading."""
        while self.loader.busy:
            self.posted.get(timeout=WAIT)()

    def names(self) -> list[tuple[str, Category]]:
        return [(i.name, i.category) for i in self.viewer.scene.items]


@pytest.fixture
def rec() -> Recorder:
    return Recorder()


def _volume_file(path: Path) -> Path:
    save_volume(Volume(np.zeros((2, 2, 2), np.float32), np.eye(4)), path)
    return path


def _tractogram_file(path: Path) -> Path:
    points = np.arange(12, dtype=np.float32).reshape(-1, 3)
    save_tractogram(Tractogram(points=points, offsets=np.array([0, 2, 4])), path)
    return path


def _fake_reads(monkeypatch: pytest.MonkeyPatch, read: Read) -> None:
    # Every category reads files with `read`; the rest of its entry is kept.
    def fake(category: Category) -> ItemType:
        return dataclasses.replace(item_type(category), load=read)

    monkeypatch.setattr(loading, "item_type", fake)


class Gate:
    """A read that waits for the test: `started` once it runs, then `go`."""

    def __init__(self, gated: str, result: object, *, report: bool = True) -> None:
        self.gated = gated
        self.result = result
        self.report = report
        self.fail = False
        self.started = threading.Event()
        self.go = threading.Event()
        self.reads: list[str] = []
        self.finished: list[str] = []

    def __call__(
        self, path: Path, progress: Progress | None, limit: int | None
    ) -> object:
        self.reads.append(path.name)
        if path.name == self.gated:
            self.started.set()
            assert self.go.wait(WAIT)
            if self.report and progress is not None:
                progress(1, 2)  # raises ReadCancelledError once cancelled
            if self.fail:
                raise ValueError("unreadable")
        self.finished.append(path.name)
        return self.result


def test_files_are_added_in_the_order_they_were_queued(
    rec: Recorder, tmp_path: Path
) -> None:
    a = _volume_file(tmp_path / "a.nii.gz")
    b = _volume_file(tmp_path / "b.nii")
    t = _tractogram_file(tmp_path / "t.tck")
    o = _volume_file(tmp_path / "o.nii.gz")
    rec.loader.open([a, b], Category.VOLUMES)
    rec.loader.open([t], Category.TRACTOGRAMS)
    rec.loader.open([str(o)], Category.OVERLAYS)
    rec.drain()
    assert rec.names() == [
        ("a.nii.gz", Category.VOLUMES),
        ("b.nii", Category.VOLUMES),
        ("t.tck", Category.TRACTOGRAMS),
        ("o.nii.gz", Category.OVERLAYS),
    ]
    assert [i.source for i in rec.viewer.scene.items] == [a, b, t, o]
    assert rec.errors == []


def test_open_refuses_a_category_name_given_as_text(rec: Recorder) -> None:
    with pytest.raises(TypeError, match="Category"):
        rec.loader.open(["a.nii"], "volumes")  # type: ignore[arg-type]
    assert not rec.loader.busy


def test_unreadable_files_are_reported_and_later_files_still_load(
    rec: Recorder, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    missing = tmp_path / "missing.nii.gz"
    bad = tmp_path / "bad.nii"
    bad.write_bytes(b"not a nifti file")
    good = _volume_file(tmp_path / "good.nii.gz")
    with caplog.at_level(logging.ERROR, logger="larmor"):
        rec.loader.open([missing, bad, good], Category.OVERLAYS)
        rec.drain()
    assert rec.names() == [("good.nii.gz", Category.OVERLAYS)]
    assert len(rec.errors) == 2
    assert rec.errors[0] == (
        f"Cannot open {missing} as overlays: No such file or directory."
    )
    assert rec.errors[1].startswith(f"Cannot open {bad} as overlays: ")
    assert "is not a NIfTI file" in rec.errors[1]
    assert [r.getMessage() for r in caplog.records] == rec.errors
    assert rec.statuses[-1] is None


def test_a_file_of_the_wrong_kind_for_its_category_is_reported(
    rec: Recorder, tmp_path: Path
) -> None:
    t = _tractogram_file(tmp_path / "t.tck")
    rec.loader.open([t], Category.VOLUMES)
    rec.drain()
    assert rec.viewer.scene.items == []
    assert len(rec.errors) == 1
    assert rec.errors[0].startswith(f"Cannot open {t} as volumes: ")


def test_cancel_stops_the_read_and_drops_the_queued_files(
    rec: Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate = Gate("slow.nii", Volume(np.zeros((2, 2, 2), np.float32), np.eye(4)))
    _fake_reads(monkeypatch, gate)
    rec.loader.open([tmp_path / "slow.nii", tmp_path / "queued.nii"], Category.VOLUMES)
    assert gate.started.wait(WAIT)
    rec.loader.cancel()
    assert not rec.loader.busy
    assert rec.statuses[-1] is None
    gate.go.set()
    # A file opened after the cancel loads; it is read after the cancelled one.
    rec.loader.open([tmp_path / "after.nii"], Category.VOLUMES)
    rec.drain()
    assert rec.names() == [("after.nii", Category.VOLUMES)]
    assert gate.reads == ["slow.nii", "after.nii"]
    assert gate.finished == ["after.nii"]
    assert rec.errors == []


@pytest.mark.parametrize("fails", [False, True])
def test_results_of_a_cancelled_read_are_dropped(
    fails: bool, rec: Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # This read never reports progress, so it runs to its end after the cancel.
    gate = Gate(
        "slow.nii", Volume(np.zeros((2, 2, 2), np.float32), np.eye(4)), report=False
    )
    gate.fail = fails
    _fake_reads(monkeypatch, gate)
    rec.loader.open([tmp_path / "slow.nii"], Category.VOLUMES)
    assert gate.started.wait(WAIT)
    rec.loader.cancel()
    gate.go.set()
    rec.loader.open([tmp_path / "after.nii"], Category.OVERLAYS)
    rec.drain()
    assert rec.names() == [("after.nii", Category.OVERLAYS)]
    assert rec.errors == []


def test_progress_is_reported_per_file_and_cleared_at_the_end(
    rec: Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def read(path: Path, progress: Progress | None, limit: int | None) -> object:
        assert progress is not None
        for done in (1, 4):
            progress(done, 4)
        progress(0, 0)  # a file of unknown size counts as read
        return Volume(np.zeros((2, 2, 2), np.float32), np.eye(4))

    _fake_reads(monkeypatch, read)
    rec.loader.open([tmp_path / "a.nii", tmp_path / "b.nii"], Category.VOLUMES)
    rec.drain()
    seen = [s for i, s in enumerate(rec.statuses) if i == 0 or s != rec.statuses[i - 1]]
    assert seen == [
        LoadStatus("a.nii", 1, 2, 0.0),
        LoadStatus("a.nii", 1, 2, 0.25),
        LoadStatus("a.nii", 1, 2, 1.0),
        LoadStatus("b.nii", 2, 2, 0.0),
        LoadStatus("b.nii", 2, 2, 0.25),
        LoadStatus("b.nii", 2, 2, 1.0),
        None,
    ]
    assert rec.errors == []
    assert len(rec.viewer.scene.items) == 2


def test_files_queued_while_loading_count_on_from_the_first(
    rec: Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate = Gate("a.nii", Volume(np.zeros((2, 2, 2), np.float32), np.eye(4)))
    _fake_reads(monkeypatch, gate)
    rec.loader.open([tmp_path / "a.nii"], Category.VOLUMES)
    assert gate.started.wait(WAIT)
    rec.loader.open([tmp_path / "b.nii"], Category.VOLUMES)
    gate.go.set()
    rec.drain()
    named = [s for s in rec.statuses if s is not None]
    assert {(s.name, s.number, s.total) for s in named if s.name == "b.nii"} == {
        ("b.nii", 2, 2)
    }
    assert rec.statuses[-1] is None
    assert len(rec.viewer.scene.items) == 2


def test_the_read_limit_reaches_the_reader(
    rec: Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    limits: list[int | None] = []

    def read(path: Path, progress: Progress | None, limit: int | None) -> object:
        limits.append(limit)
        return Volume(np.zeros((2, 2, 2), np.float32), np.eye(4))

    _fake_reads(monkeypatch, read)
    rec.loader.max_streamlines = 7
    rec.loader.open([tmp_path / "a.nii"], Category.VOLUMES)
    rec.drain()
    assert limits == [7]


def test_a_queued_object_follows_the_files_before_it(
    rec: Recorder, tmp_path: Path
) -> None:
    a = _volume_file(tmp_path / "a.nii.gz")
    b = _volume_file(tmp_path / "b.nii")
    volume = Volume(np.zeros((2, 2, 2), np.float32), np.eye(4))
    rec.loader.open([a, volume, b], Category.VOLUMES)
    rec.drain()
    first, second, third = rec.viewer.scene.items
    assert (first.source, third.source) == (a, b)
    assert second.data is volume
    assert second.source is None
    assert second.name == "Volume 2"
    assert LoadStatus("Volume object", 2, 3, 0.0) in rec.statuses
    assert rec.errors == []


def test_cancel_drops_a_queued_object(
    rec: Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    volume = Volume(np.zeros((2, 2, 2), np.float32), np.eye(4))
    gate = Gate("slow.nii", volume)
    _fake_reads(monkeypatch, gate)
    queued = Volume(np.zeros((2, 2, 2), np.float32), np.eye(4))
    rec.loader.open([tmp_path / "slow.nii", queued], Category.VOLUMES)
    assert gate.started.wait(WAIT)
    rec.loader.cancel()
    gate.go.set()
    rec.loader.open([tmp_path / "after.nii"], Category.VOLUMES)
    rec.drain()
    assert rec.names() == [("after.nii", Category.VOLUMES)]
    assert all(i.data is not queued for i in rec.viewer.scene.items)


def test_each_job_keeps_the_read_limit_it_was_queued_with(
    rec: Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate = Gate("a.tck", Volume(np.zeros((2, 2, 2), np.float32), np.eye(4)))
    limits: dict[str, int | None] = {}

    def read(path: Path, progress: Progress | None, limit: int | None) -> object:
        limits[path.name] = limit
        return gate(path, progress, limit)

    _fake_reads(monkeypatch, read)
    rec.loader.max_streamlines = 7
    rec.loader.open([tmp_path / "a.tck"], Category.TRACTOGRAMS, 5)
    assert gate.started.wait(WAIT)
    rec.loader.open([tmp_path / "b.tck"], Category.TRACTOGRAMS, None)
    rec.loader.open([tmp_path / "c.tck"], Category.TRACTOGRAMS)
    rec.loader.max_streamlines = 3  # too late for the jobs already queued
    gate.go.set()
    rec.drain()
    assert limits == {"a.tck": 5, "b.tck": None, "c.tck": 7}


def test_a_failing_session_callback_leaves_the_loader_idle(
    rec: Recorder, tmp_path: Path
) -> None:
    viewer = Viewer()
    viewer.load(_volume_file(tmp_path / "a.nii.gz"), Category.VOLUMES)
    viewer.save_session(tmp_path / "s.lar.json")

    def fail(report: object) -> None:
        raise RuntimeError("late")

    rec.loader.open_session(tmp_path / "s.lar.json", fail)
    with pytest.raises(RuntimeError, match="late"):
        rec.drain()
    assert not rec.loader.busy


def _join_worker() -> None:
    for t in threading.enumerate():
        if t.name == "viewer-loader":
            t.join(WAIT)
            assert not t.is_alive()


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("end", [Loader.cancel, Loader.stop])
def test_a_read_that_ends_after_a_cancel_posts_nothing(
    end: Callable[[Loader], None],
    fails: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # This read never reports progress, so it runs to its end after the cancel.
    gate = Gate(
        "slow.nii", Volume(np.zeros((2, 2, 2), np.float32), np.eye(4)), report=False
    )
    gate.fail = fails
    _fake_reads(monkeypatch, gate)
    posted: list[Callable[[], None]] = []
    loader = Loader(Viewer(), posted.append)
    loader.open([tmp_path / "slow.nii"], Category.VOLUMES)
    assert gate.started.wait(WAIT)
    end(loader)
    before = len(posted)
    gate.go.set()
    _join_worker()
    assert gate.reads == ["slow.nii"]
    assert len(posted) == before


def test_stop_reports_nothing_afterwards(rec: Recorder, tmp_path: Path) -> None:
    rec.loader.open([tmp_path / "missing.nii"], Category.VOLUMES)
    rec.statuses.clear()
    rec.loader.stop()
    rec.loader.open([tmp_path / "also_missing.nii"], Category.VOLUMES)
    rec.drain()
    assert rec.statuses == []
    assert rec.errors == []


def test_importing_the_viewer_package_loads_no_qt() -> None:
    # A fresh interpreter: this test process may have imported Qt already.
    heavy = ("PySide6", "pygfx", "wgpu", "rendercanvas", "PIL")
    code = (
        "import sys, larmor.viewer\n"
        "from larmor.viewer import view, wait, Viewer, Category\n"
        "from larmor.viewer import ViewKind, OptionName, Colormap, Convention\n"
        f"print(','.join(m for m in {heavy!r} if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_the_viewer_exports_its_enums() -> None:
    names = {
        "ViewKind",
        "OptionName",
        "Colormap",
        "Convention",
        "Category",
        "CameraPreset",
    }
    assert names <= set(larmor.viewer.__all__)


def test_view_runs_the_checks_before_importing_qt() -> None:
    code = (
        "import sys\n"
        "from larmor.viewer import system, view\n"
        "def refuse():\n"
        "    raise system.ViewerUnavailableError('no')\n"
        "system.check_viewer_can_start = refuse\n"
        "try:\n"
        "    view(volumes='t1.nii.gz')\n"
        "except system.ViewerUnavailableError:\n"
        "    print('refused', 'PySide6' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "refused False"


def test_real_files_report_their_progress(rec: Recorder, tmp_path: Path) -> None:
    v = _volume_file(tmp_path / "v.nii.gz")
    t = _tractogram_file(tmp_path / "t.tck")
    rec.loader.open([v], Category.VOLUMES)
    rec.loader.open([t], Category.TRACTOGRAMS)
    rec.drain()
    done = {(s.name, s.fraction) for s in rec.statuses if s is not None}
    assert {("v.nii.gz", 1.0), ("t.tck", 1.0)} <= done


@pytest.mark.parametrize(
    ("category", "name", "reads"),
    [
        (Category.VOLUMES, "T1.NII.GZ", True),
        (Category.OVERLAYS, "fa.Nii", True),
        (Category.TRACTOGRAMS, "CST.TCK", True),
        (Category.VOLUMES, "cst.tck", False),
        (Category.TRACTOGRAMS, "t1.nii.gz", False),
        (Category.OVERLAYS, "notes.txt", False),
    ],
)
def test_a_category_reads_files_by_extension_in_any_case(
    category: Category, name: str, reads: bool
) -> None:
    assert item_type(category).reads(Path(name)) is reads


@pytest.mark.parametrize("end", [Loader.cancel, Loader.stop])
def test_a_load_ended_before_its_read_finishes_is_told_it_was_cancelled(
    end: Callable[[Loader], None],
    rec: Recorder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gate = Gate("slow.nii", Volume(np.zeros((2, 2, 2), np.float32), np.eye(4)))
    _fake_reads(monkeypatch, gate)
    told: list[tuple[object, str | None]] = []
    rec.loader.load(
        tmp_path / "slow.nii",
        Category.VOLUMES,
        None,
        then=lambda i, e: told.append((i, e)),
    )
    assert gate.started.wait(WAIT)
    end(rec.loader)
    gate.go.set()
    _join_worker()
    assert told == [(None, "Loading slow.nii was cancelled.")]
    assert rec.viewer.scene.items == []


def test_cancel_tells_every_waiting_load_when_one_callback_fails(
    rec: Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate = Gate("slow.nii", Volume(np.zeros((2, 2, 2), np.float32), np.eye(4)))
    _fake_reads(monkeypatch, gate)
    told: list[str | None] = []

    def fail(item: object, error: str | None) -> None:
        raise RuntimeError("callback broke")

    rec.loader.load(tmp_path / "slow.nii", Category.VOLUMES, then=fail)
    rec.loader.load(
        tmp_path / "queued.nii", Category.VOLUMES, then=lambda i, e: told.append(e)
    )
    assert gate.started.wait(WAIT)
    rec.loader.cancel()
    gate.go.set()
    _join_worker()
    assert told == ["Loading queued.nii was cancelled."]
    assert rec.statuses[-1] is None


def test_sort_files_sorts_by_name_into_volumes_tracts_and_a_session() -> None:
    names = ["a.tck", "t1.nii.gz", "s.lar.json", "b.trk", "fa.mif", "notes.txt"]
    assert sort_files(names) == SortedFiles(
        volumes=[Path("t1.nii.gz"), Path("fa.mif"), Path("notes.txt")],
        tracts=[Path("a.tck"), Path("b.trk")],
        sessions=[Path("s.lar.json")],
    )


def test_sort_files_reads_extensions_in_any_case_as_the_categories_do() -> None:
    given = sort_files(["A.TCK", "T1.NII.GZ", "S.LAR.JSON"])
    assert given == SortedFiles(
        [Path("T1.NII.GZ")], [Path("A.TCK")], [Path("S.LAR.JSON")]
    )


def test_sort_files_keeps_every_session_for_the_caller_to_refuse() -> None:
    given = sort_files(["a.lar.json", "t1.nii", "b.lar.json"])
    assert given.sessions == [Path("a.lar.json"), Path("b.lar.json")]
    assert given.volumes == [Path("t1.nii")]


def test_sort_files_skips_a_json_that_is_not_a_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="larmor"):
        given = sort_files(["sub-01_dwi.nii.gz", "sub-01_dwi.json"])
    assert given == SortedFiles([Path("sub-01_dwi.nii.gz")], [], [])
    assert "sub-01_dwi.json" in caplog.text
