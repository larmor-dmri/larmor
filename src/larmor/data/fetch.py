"""Downloads example datasets once into a cache folder and checks them."""

import hashlib
import http.client
import os
import sys
import time
import urllib.request
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from larmor._log import LarmorLogger

# The environment variable that sets the cache folder.
DATA_DIR_VARIABLE = "LARMOR_DATA_DIR"
# The name of the cache folder in the user's cache folder.
CACHE_FOLDER = "larmor"
# The bytes read or written at a time.
CHUNK_BYTES = 1 << 20
# How long a download may wait for the server.
TIMEOUT_SECONDS = 60.0
# The number of tries of a download.
ATTEMPTS = 3
# The pause before a new try, times the number of the try.
RETRY_PAUSE_SECONDS = 2.0
# The suffix of a file still being written.
PART_SUFFIX = ".part"
# The age after which a partial file is from an interrupted download.
STALE_PART_SECONDS = 24 * 3600.0
# The file with the licence and citation, saved next to the files of a dataset.
NOTICE_FILE = "README.txt"


@dataclass(frozen=True)
class Member:
    """A file in a zip archive, with its key, its name in the cache, its path and size.

    `fix`, if given, corrects a known error of the file when it is extracted;
    `size` and `sha256` are then those of the corrected file.
    """

    key: str
    name: str
    path: str
    size: int
    fix: Callable[[bytes], bytes] | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class RemoteFile:
    """One file of a dataset, with its key, its name in the cache, its source and hash.

    A zip archive lists the members to extract under their own keys and names;
    the archive is then deleted.
    """

    key: str
    name: str
    url: str
    sha256: str
    size: int
    members: tuple[Member, ...] = ()

    def outputs(self) -> dict[str, tuple[str, int]]:
        """Returns the name and size of each file it leaves in the cache, by key."""
        if self.members:
            return {m.key: (m.name, m.size) for m in self.members}
        return {self.key: (self.name, self.size)}


@dataclass(frozen=True)
class Dataset:
    """A public dataset, with its files and the licence and citation it comes with.

    `changes` says how the files in the cache differ from the published ones, if
    they do (see `Member.fix`).
    """

    name: str
    title: str
    page: str
    licence: str
    citation: str
    files: tuple[RemoteFile, ...]
    changes: str = ""

    def notice(self) -> str:
        """Returns the text saved next to the files and logged on the first download."""
        changed = f"Changes: {self.changes}\n" if self.changes else ""
        return (
            f"{self.title}\n\nSource: {self.page}\nLicence: {self.licence}\n"
            f"{changed}Please cite: {self.citation}\n"
        )


def data_folder() -> Path:
    """Returns the folder that example datasets are downloaded into.

    It is the folder in the `LARMOR_DATA_DIR` environment variable when that is
    set, and otherwise `~/Library/Caches/larmor` on macOS, `~/.cache/larmor` (or
    `$XDG_CACHE_HOME/larmor`) on Linux and `%LOCALAPPDATA%/larmor/Cache` on
    Windows.

    Returns:
        The folder, which may not exist yet.
    """
    chosen = os.environ.get(DATA_DIR_VARIABLE)
    if chosen:
        return Path(chosen).expanduser()
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return base / CACHE_FOLDER / "Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / CACHE_FOLDER
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / CACHE_FOLDER


def fetch_dataset(
    dataset: Dataset, folder: str | Path | None = None
) -> dict[str, Path]:
    """Downloads the files of a dataset that are not in the cache yet.

    Args:
        dataset: The dataset.
        folder: The cache folder; None for `data_folder()`.

    Returns:
        The paths of the files, by key.
    """
    target = Path(folder) if folder is not None else data_folder()
    target = target / dataset.name
    target.mkdir(parents=True, exist_ok=True)
    _remove_stale_parts(target)
    missing = [f for f in dataset.files if not _is_cached(target, f)]
    if missing:
        LarmorLogger.info("%s", dataset.notice())
        (target / NOTICE_FILE).write_text(dataset.notice(), encoding="utf-8")
    for remote in missing:
        _download(remote, target / remote.name)
        if remote.members:
            _extract(remote, target)
    return {
        key: target / name
        for remote in dataset.files
        for key, (name, _) in remote.outputs().items()
    }


def _is_cached(folder: Path, remote: RemoteFile) -> bool:
    """Checks whether the outputs of `remote` are already in `folder`.

    The sizes are checked, and the hash of each member that has one, so that a
    copy from before a correction is extracted again.

    Returns:
        True if every output is there with its size and every checksummed member
        matches its hash.
    """
    for name, size in remote.outputs().values():
        path = folder / name
        if not path.is_file() or path.stat().st_size != size:
            return False
    for member in remote.members:
        if member.sha256 is not None and _sha256(folder / member.name) != member.sha256:
            return False
    return True


def _sha256(path: Path) -> str:
    """Returns the SHA-256 of the file at `path`, as hexadecimal."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _extract(remote: RemoteFile, folder: Path) -> None:
    """Extracts the members of the archive `remote` in `folder`, then deletes it.

    If an extraction fails, the members extracted so far are deleted too.
    """
    archive = folder / remote.name
    done: list[Path] = []
    try:
        with zipfile.ZipFile(archive) as opened:
            for member in remote.members:
                path = folder / member.name
                _extract_member(opened, member, path)
                done.append(path)
    except BaseException:
        for path in done:
            path.unlink(missing_ok=True)
        raise
    finally:
        archive.unlink(missing_ok=True)


def _extract_member(opened: zipfile.ZipFile, member: Member, path: Path) -> None:
    """Extracts `member` to `path`, corrected if it has a fix."""
    part = path.with_name(f"{path.name}.{uuid.uuid4().hex}{PART_SUFFIX}")
    try:
        with opened.open(member.path) as source, part.open("xb") as out:
            if member.fix is not None:
                out.write(member.fix(source.read()))
            else:
                while chunk := source.read(CHUNK_BYTES):
                    out.write(chunk)
        if member.sha256 is not None and _sha256(part) != member.sha256:
            raise ValueError(
                f"{member.name}, extracted from {member.path}, does not match its "
                "checksum; it was not saved."
            )
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


def _remove_stale_parts(folder: Path) -> None:
    """Deletes the partial files of interrupted downloads in `folder`."""
    now = time.time()
    for part in folder.glob(f"*{PART_SUFFIX}"):
        try:
            if now - part.stat().st_mtime > STALE_PART_SECONDS:
                part.unlink()
        except OSError:  # another process removed it, or still holds it on Windows
            continue


def _download(remote: RemoteFile, path: Path) -> None:
    """Downloads `remote` to `path`, trying again after a dropped connection."""
    LarmorLogger.info(
        "Downloading %s (%.1f MB) from %s", remote.name, remote.size / 1e6, remote.url
    )
    for attempt in range(1, ATTEMPTS + 1):
        try:
            _download_once(remote, path)
            return
        except (OSError, http.client.HTTPException) as error:
            if attempt == ATTEMPTS:
                raise OSError(
                    f"Could not download {remote.name} from {remote.url}: {error}. "
                    "Run it again to start anew."
                ) from error
            LarmorLogger.warning(
                "The download of %s failed (%s); starting it again.", remote.name, error
            )
            time.sleep(RETRY_PAUSE_SECONDS * attempt)


def _download_once(remote: RemoteFile, path: Path) -> None:
    """Downloads `remote` to a temporary file, checks it, then moves it to `path`."""
    # A unique name, so that two processes never share it; not mkstemp, whose
    # owner-only permissions the cached file would keep.
    part = path.with_name(f"{path.name}.{uuid.uuid4().hex}{PART_SUFFIX}")
    try:
        digest = hashlib.sha256()
        with part.open("xb") as out:
            with urllib.request.urlopen(remote.url, timeout=TIMEOUT_SECONDS) as answer:
                while chunk := answer.read(CHUNK_BYTES):
                    digest.update(chunk)
                    out.write(chunk)
            received = out.tell()
        # A dropped connection can end the stream early without an error.
        if received != remote.size:
            raise OSError(f"received {received} of {remote.size} bytes")
        if digest.hexdigest() != remote.sha256:
            raise ValueError(
                f"The download of {remote.name} from {remote.url} does not match its "
                f"checksum (SHA-256 {digest.hexdigest()}, expected {remote.sha256}); "
                "it was not saved. Try again later, or report it if it persists."
            )
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)
