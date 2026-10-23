import dataclasses
import hashlib
import http.server
import os
import sys
import threading
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from larmor.data import (
    data_folder,
    fetch,
    fetch_cfin_multishell,
    fetch_ismrm2015,
    fetch_stanford_hardi,
)
from larmor.data.datasets import CFIN_MULTISHELL, ISMRM2015, STANFORD_HARDI
from larmor.data.fetch import NOTICE_FILE, Dataset, Member, RemoteFile, fetch_dataset

CONTENT = b"diffusion" * 5000


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serves CONTENT, whole at /whole and cut short at /cut."""

    requests: list[str] = []

    def do_GET(self) -> None:  # noqa: N802 (the name http.server calls)
        self.requests.append(self.path)
        self.send_response(200)
        self.send_header("Content-Length", str(len(CONTENT)))
        self.end_headers()
        if self.path == "/cut":
            self.wfile.write(CONTENT[:1000])  # then the connection closes
            self.close_connection = True
        else:
            self.wfile.write(CONTENT)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    _Handler.requests = []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _dataset(url: str, sha256: str | None = None) -> Dataset:
    digest = sha256 or hashlib.sha256(CONTENT).hexdigest()
    remote = RemoteFile("dwi", "dwi.nii", url, digest, len(CONTENT))
    return Dataset("test", "A test dataset", "page", "CC0-1.0", "Someone.", (remote,))


def test_downloads_once_into_the_cache(server: str, tmp_path: Path) -> None:
    dataset = _dataset(f"{server}/whole")
    paths = fetch_dataset(dataset, tmp_path)
    assert paths["dwi"].read_bytes() == CONTENT
    assert "CC0-1.0" in (tmp_path / "test" / NOTICE_FILE).read_text()
    assert fetch_dataset(dataset, tmp_path) == paths
    assert _Handler.requests == ["/whole"]
    assert {p.name for p in (tmp_path / "test").iterdir()} == {"dwi.nii", NOTICE_FILE}


def test_refuses_a_file_with_the_wrong_checksum(server: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="checksum"):
        fetch_dataset(_dataset(f"{server}/whole", "0" * 64), tmp_path)
    assert {p.name for p in (tmp_path / "test").iterdir()} == {NOTICE_FILE}


def test_an_interrupted_download_restarts_cleanly(
    server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch, "RETRY_PAUSE_SECONDS", 0.0)
    with pytest.raises(OSError, match="Run it again"):
        fetch_dataset(_dataset(f"{server}/cut"), tmp_path)
    assert _Handler.requests == ["/cut"] * fetch.ATTEMPTS
    assert {p.name for p in (tmp_path / "test").iterdir()} == {NOTICE_FILE}
    paths = fetch_dataset(_dataset(f"{server}/whole"), tmp_path)
    assert paths["dwi"].read_bytes() == CONTENT


def test_file_urls_work(tmp_path: Path) -> None:
    source = tmp_path / "source.nii"
    source.write_bytes(CONTENT)
    paths = fetch_dataset(_dataset(source.as_uri()), tmp_path / "cache")
    assert paths["dwi"].read_bytes() == CONTENT


def test_an_archive_is_extracted_then_deleted(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("data/Diffusion.nii.gz", CONTENT)
        archive.writestr("data/T1.nii.gz", b"t1")
        archive.writestr("data/fmap.nii.gz", b"not wanted")
    members = (
        Member("dwi", "dwi.nii.gz", "data/Diffusion.nii.gz", len(CONTENT)),
        Member("t1", "t1.nii.gz", "data/T1.nii.gz", 2),
    )
    remote = RemoteFile(
        "archive",
        "data.zip",
        source.as_uri(),
        hashlib.sha256(source.read_bytes()).hexdigest(),
        source.stat().st_size,
        members,
    )
    dataset = Dataset("test", "A test dataset", "page", "CC-BY", "Someone.", (remote,))
    paths = fetch_dataset(dataset, tmp_path / "cache")
    assert paths == {
        "dwi": tmp_path / "cache" / "test" / "dwi.nii.gz",
        "t1": tmp_path / "cache" / "test" / "t1.nii.gz",
    }
    assert paths["dwi"].read_bytes() == CONTENT
    assert {p.name for p in (tmp_path / "cache" / "test").iterdir()} == {
        "dwi.nii.gz",
        "t1.nii.gz",
        NOTICE_FILE,
    }
    source.unlink()  # the extracted files are enough: nothing is downloaded again
    assert fetch_dataset(dataset, tmp_path / "cache") == paths


def test_ismrm_bvecs_are_corrected_once_also_in_an_old_cache(tmp_path: Path) -> None:
    published = b"-0 -0.5 0.25 \n0 0.5 -0.75 \n1 0 0 \n"
    corrected = b"0 0.5 -0.25\n0 0.5 -0.75 \n1 0 0 \n"
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("data/Diffusion.bvecs", published)
    fix = ISMRM2015.files[0].members[2].fix
    assert fix is not None
    member = Member(
        "bvecs",
        "dwi.bvecs",
        "data/Diffusion.bvecs",
        len(corrected),
        fix,
        hashlib.sha256(corrected).hexdigest(),
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    remote = RemoteFile(
        "archive", "a.zip", source.as_uri(), digest, source.stat().st_size, (member,)
    )
    dataset = dataclasses.replace(ISMRM2015, files=(remote,))
    old = tmp_path / "cache" / dataset.name / "dwi.bvecs"
    old.parent.mkdir(parents=True)
    # A cache from before the correction, cut to the corrected file's size: only
    # its hash tells it apart.
    old.write_bytes(published[: len(corrected)])
    paths = fetch_dataset(dataset, tmp_path / "cache")
    assert paths["bvecs"].read_bytes() == corrected
    assert "Changes:" in (old.parent / NOTICE_FILE).read_text()
    source.unlink()  # a second call finds the corrected file and leaves it alone
    fetch_dataset(dataset, tmp_path / "cache")
    assert paths["bvecs"].read_bytes() == corrected


def test_a_failed_extraction_leaves_no_archive_and_no_partial_dataset(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("data/T1.nii.gz", b"t1")  # the DWI is missing
    members = (
        Member("t1", "t1.nii.gz", "data/T1.nii.gz", 2),
        Member("dwi", "dwi.nii.gz", "data/Diffusion.nii.gz", 10),
    )
    remote = RemoteFile(
        "archive",
        "data.zip",
        source.as_uri(),
        hashlib.sha256(source.read_bytes()).hexdigest(),
        source.stat().st_size,
        members,
    )
    dataset = Dataset("test", "A test dataset", "page", "CC-BY", "Someone.", (remote,))
    with pytest.raises(KeyError):
        fetch_dataset(dataset, tmp_path / "cache")
    assert {p.name for p in (tmp_path / "cache" / "test").iterdir()} == {NOTICE_FILE}


def test_the_variable_chooses_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LARMOR_DATA_DIR", str(tmp_path / "mine"))
    assert data_folder() == tmp_path / "mine"


@pytest.mark.parametrize(
    ("platform", "variable", "expected"),
    [
        ("darwin", None, "Library/Caches/larmor"),
        ("linux", None, ".cache/larmor"),
        ("linux", "XDG_CACHE_HOME", "xdg/larmor"),
        ("win32", None, "AppData/Local/larmor/Cache"),
        ("win32", "LOCALAPPDATA", "xdg/larmor/Cache"),
    ],
)
def test_the_platform_cache_folder(
    platform: str,
    variable: str | None,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    for name in ("HOME", "USERPROFILE"):  # the home folder on POSIX and on Windows
        monkeypatch.setenv(name, str(tmp_path))
    for name in ("XDG_CACHE_HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(name, raising=False)
    if variable is not None:
        monkeypatch.setenv(variable, str(tmp_path / "xdg"))
    assert data_folder() == tmp_path / expected


def test_stale_partial_files_are_removed(tmp_path: Path) -> None:
    folder = tmp_path / "test"
    folder.mkdir()
    stale, fresh = folder / "dwi.nii.1.part", folder / "dwi.nii.2.part"
    stale.write_bytes(b"old")
    fresh.write_bytes(b"new")
    two_days_ago = time.time() - 2 * 24 * 3600
    os.utime(stale, (two_days_ago, two_days_ago))
    source = tmp_path / "source.nii"
    source.write_bytes(CONTENT)
    fetch_dataset(_dataset(source.as_uri()), tmp_path)
    assert not stale.exists()
    assert fresh.exists()


def test_fetchers_return_their_files_from_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Files already in the cache, of the right size (and, when corrected, of the
    # right hash), are not downloaded again; the network is never touched.
    hashes = {
        m.name: m.sha256 for m in ISMRM2015.files[0].members if m.sha256 is not None
    }
    monkeypatch.setattr(fetch, "_sha256", lambda path: hashes[path.name])

    def offline(*args: object, **kwargs: object) -> None:
        raise AssertionError("the test tried to download")

    monkeypatch.setattr("urllib.request.urlopen", offline)
    for dataset in (CFIN_MULTISHELL, STANFORD_HARDI, ISMRM2015):
        for remote in dataset.files:
            for name, size in remote.outputs().values():
                path = tmp_path / dataset.name / name
                path.parent.mkdir(exist_ok=True)
                with path.open("wb") as file:
                    file.truncate(size)
    cfin = fetch_cfin_multishell(tmp_path)
    stanford = fetch_stanford_hardi(tmp_path)
    ismrm = fetch_ismrm2015(tmp_path)
    assert cfin == {
        "dwi": tmp_path / "cfin_multishell" / "dwi.nii",
        "bvals": tmp_path / "cfin_multishell" / "dwi.bval",
        "bvecs": tmp_path / "cfin_multishell" / "dwi.bvec",
        "t1": tmp_path / "cfin_multishell" / "t1.nii",
    }
    assert stanford == {
        "dwi": tmp_path / "stanford_hardi" / "dwi.nii.gz",
        "bvals": tmp_path / "stanford_hardi" / "dwi.bvals",
        "bvecs": tmp_path / "stanford_hardi" / "dwi.bvecs",
        "t1": tmp_path / "stanford_hardi" / "t1.nii.gz",
        "labels": tmp_path / "stanford_hardi" / "aparc-reduced.nii.gz",
    }
    assert ismrm == {
        "dwi": tmp_path / "ismrm2015" / "dwi.nii.gz",
        "bvals": tmp_path / "ismrm2015" / "dwi.bvals",
        "bvecs": tmp_path / "ismrm2015" / "dwi.bvecs",
        "t1": tmp_path / "ismrm2015" / "t1.nii.gz",
    }
