"""Loads and saves volumes, tractograms and surfaces, in the format of the file name."""

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Literal

import numpy as np

from larmor._log import LarmorLogger
from larmor.io.formats import (
    read_freesurfer_surface,
    read_freesurfer_values,
    read_gifti_surface,
    read_gifti_values,
    read_mif,
    read_mif_header,
    read_nifti,
    read_nifti_header,
    read_tck,
    read_trk,
    read_vtk,
    write_freesurfer_surface,
    write_gifti_surface,
    write_mif,
    write_nifti,
    write_tck,
    write_trk,
    write_vtk,
)
from larmor.io.formats.freesurfer import CURV_MAGIC, TRIANGLE_MAGIC
from larmor.io.formats.mif import mih_data_file
from larmor.io.formats.tck import TckWriter, read_tck_chunks
from larmor.io.formats.trk import TrkWriter, read_trk_chunks
from larmor.io.mritoolkit import read_mritoolkit_grid, read_mritoolkit_tracts
from larmor.io.reader import Progress
from larmor.io.writer import BatchWriter
from larmor.types import GradientTable, Grid, Surface, Tractogram, Volume

#: The volume formats, and the file name endings that select each.
VOLUME_EXTENSIONS = {"nifti": (".nii", ".nii.gz"), "mif": (".mif", ".mif.gz", ".mih")}
#: The tractogram formats, and the file name endings that select each.
TRACTOGRAM_EXTENSIONS = {
    "tck": (".tck", ".tck.gz"),
    "trk": (".trk", ".trk.gz"),
    "vtk": (".vtk", ".vtk.gz", ".fib", ".fib.gz"),
}
#: The surface formats chosen by the file name ending; other surfaces are FreeSurfer's.
SURFACE_EXTENSIONS = {"gifti": (".gii",)}


def _unsupported(path: Path, kind: str, extensions: dict[str, tuple[str, ...]]) -> str:
    """Returns the error message for a `path` whose extension `kind` cannot read.

    `extensions` are those of `kind`. The message says when the file is of the other
    kind, a volume given for a tractogram or the reverse.
    """
    supported = ", ".join(ext for exts in extensions.values() for ext in exts)
    other = "tractogram" if kind == "volume" else "volume"
    other_extensions = TRACTOGRAM_EXTENSIONS if kind == "volume" else VOLUME_EXTENSIONS
    if any(path.name.endswith(exts) for exts in other_extensions.values()):
        return f"{path} is a {other}, not a {kind}; a {kind} is one of {supported}."
    return f"{path}: unsupported extension; a {kind} is one of {supported}."


def _volume_format(path: Path) -> Literal["nifti", "mif"]:
    """Returns the format of the volume file `path`, from the end of its name.

    Raises:
        ValueError: If the name ends in none of the supported extensions.
    """
    if path.name.endswith(VOLUME_EXTENSIONS["nifti"]):
        return "nifti"
    if path.name.endswith(VOLUME_EXTENSIONS["mif"]):
        return "mif"
    raise ValueError(_unsupported(path, "volume", VOLUME_EXTENSIONS))


def _tractogram_format(path: Path) -> Literal["tck", "trk", "vtk"]:
    """Returns the format of the tractogram file `path`, from the end of its name.

    Raises:
        ValueError: If the name ends in none of the supported extensions.
    """
    if path.name.endswith(TRACTOGRAM_EXTENSIONS["tck"]):
        return "tck"
    if path.name.endswith(TRACTOGRAM_EXTENSIONS["trk"]):
        return "trk"
    if path.name.endswith(TRACTOGRAM_EXTENSIONS["vtk"]):
        return "vtk"
    raise ValueError(_unsupported(path, "tractogram", TRACTOGRAM_EXTENSIONS))


def _surface_format(path: Path) -> Literal["gifti", "freesurfer"]:
    """Returns the format of the surface file `path`.

    A GIfTI file is known by its extension, a FreeSurfer file by its first bytes.

    Raises:
        ValueError: If the file is neither, or is a FreeSurfer per-vertex file.
    """
    if path.name.endswith(SURFACE_EXTENSIONS["gifti"]):
        return "gifti"
    with open(path, "rb") as f:
        magic: bytes = f.read(3)
    if magic == TRIANGLE_MAGIC:
        return "freesurfer"
    if magic == CURV_MAGIC:
        raise ValueError(
            f"{path} holds values per vertex (or is an old quad surface, which is "
            "not supported), not a mesh; pass values in values=."
        )
    raise ValueError(
        f"{path}: unsupported surface; expected a .gii file or a FreeSurfer surface "
        "(such as lh.white or lh.pial)."
    )


def load_surface(
    path: str | Path, *, values: str | Path | Sequence[str | Path] = ()
) -> Surface:
    """Reads a surface, and optionally files of values per vertex for it.

    Args:
        path: A GIfTI `.gii` file, or a FreeSurfer surface such as `lh.white`
            (recognised by its first bytes, since it has no extension).
        values: A file, or files, of values per vertex to add to `vertex_data`.
            A FreeSurfer file is named after the part after the hemisphere
            (`lh.thickness` gives "thickness"), a GIfTI file by the names inside it
            (an unnamed array by its file name, in the same way).

    Returns:
        The surface, in scanner coordinates when the file allows it.

    Raises:
        ValueError: If a file is not a supported surface or values file, a name
            appears twice, or a values file does not have one entry per vertex.
    """
    p = Path(path)
    if _surface_format(p) == "gifti":
        surface = read_gifti_surface(p)
    else:
        surface = read_freesurfer_surface(p)
    vertex_data = dict(surface.vertex_data)
    files = [values] if isinstance(values, (str, Path)) else values
    for file in files:
        loaded = _load_values(Path(file))
        if not loaded:
            raise ValueError(
                f"{file} holds no values per vertex (is it a surface, not values?)."
            )
        for name, array in loaded.items():
            if name in vertex_data:
                raise ValueError(f"{file}: the surface already has values {name!r}.")
            if array.ndim == 0 or len(array) != surface.num_vertices:
                raise ValueError(
                    f"{file} holds {array.shape[0] if array.ndim else 0} values per "
                    f"vertex for a surface of {surface.num_vertices} vertices."
                )
            vertex_data[name] = array
    return Surface(
        surface.vertices,
        surface.faces,
        reference=surface.reference,
        vertex_data=vertex_data,
    )


def _load_values(path: Path) -> dict[str, np.ndarray]:
    """Returns the values per vertex in the GIfTI or FreeSurfer file `path`, by name."""
    if path.name.endswith(SURFACE_EXTENSIONS["gifti"]):
        return read_gifti_values(path, unnamed=_stem(path))
    return {_stem(path): read_freesurfer_values(path)}


def _stem(path: Path) -> str:
    """Returns the name of the values in `path`, such as "thickness" for "lh.thickness".

    The hemisphere before the first dot and GIfTI endings such as ".shape.gii" are
    dropped.
    """
    name = path.name
    for ending in (".shape.gii", ".func.gii", ".label.gii", ".gii"):
        if name.endswith(ending):
            name = name[: -len(ending)]
            break
    return name.split(".", 1)[-1]


def save_surface(surface: Surface, path: str | Path) -> None:
    """Writes a surface, as GIfTI for a `.gii` name and FreeSurfer for any other name.

    A FreeSurfer surface holds the mesh only; write values per vertex with
    `write_freesurfer_values`, or save as `.gii` to keep them.

    Args:
        surface: The surface to write.
        path: The file to write.

    Raises:
        ValueError: If the name ends in a volume or tractogram extension.
    """
    p = Path(path)
    others = tuple(
        ext
        for extensions in (VOLUME_EXTENSIONS, TRACTOGRAM_EXTENSIONS)
        for exts in extensions.values()
        for ext in exts
    )
    if p.name.endswith(SURFACE_EXTENSIONS["gifti"]):
        write_gifti_surface(surface, p)
    elif p.name.endswith(others):
        raise ValueError(
            f"{p} is a volume or tractogram name; save a surface as .gii or as a "
            "FreeSurfer surface (such as lh.white)."
        )
    else:
        if surface.vertex_data:
            LarmorLogger.warning(
                "%s: a FreeSurfer surface holds the mesh only; values %s are not "
                "saved (save as .gii to keep them).",
                p,
                ", ".join(surface.vertex_data),
            )
        write_freesurfer_surface(surface, p)


def file_kind(path: str | Path) -> Literal["volume", "tractogram"]:
    """Returns whether a file is a volume or a tractogram, from its extension.

    Args:
        path: The file path; only its name is read.

    Returns:
        "volume" or "tractogram".

    Raises:
        ValueError: If the extension is neither.
    """
    name = Path(path).name
    if any(name.endswith(exts) for exts in VOLUME_EXTENSIONS.values()):
        return "volume"
    if any(name.endswith(exts) for exts in TRACTOGRAM_EXTENSIONS.values()):
        return "tractogram"
    volumes = ", ".join(e for exts in VOLUME_EXTENSIONS.values() for e in exts)
    tractograms = ", ".join(e for exts in TRACTOGRAM_EXTENSIONS.values() for e in exts)
    raise ValueError(
        f"{path}: unsupported extension; a volume is one of {volumes}, a tractogram "
        f"one of {tractograms}."
    )


def load_volume(path: str | Path, *, on_progress: Progress | None = None) -> Volume:
    """Reads a volume, in the format given by the file's extension.

    Args:
        path: A `.nii`, `.nii.gz`, `.mif`, `.mif.gz` or `.mih` file.
        on_progress: Called as the file is read, with the bytes read so far and the
            file's size on disk (compressed bytes for a gzip file). Raise
            ReadCancelledError in it to stop the read.

    Returns:
        The volume.

    Raises:
        ValueError: If the extension is not supported, or the file cannot be read
            as that format.
        ReadCancelledError: If on_progress raises it.
    """
    p = Path(path)
    if _volume_format(p) == "nifti":
        return read_nifti(p, on_progress=on_progress)
    return read_mif(p, on_progress=on_progress)


def load_tractogram(
    path: str | Path,
    *,
    reference: Grid | None = None,
    on_progress: Progress | None = None,
    max_streamlines: int | None = None,
    seed: int | None = None,
    space: str | None = None,
) -> Tractogram:
    """Reads a tractogram, in the format given by the file's extension.

    Args:
        path: A `.tck`, `.trk`, `.vtk` or `.fib` file, each also gzipped (`.gz`),
            or MRIToolkit's tractography in a `.mat` file.
        reference: The grid of the image the tractogram belongs to. A TRK uses it
            when it has no vox_to_ras and checks it otherwise; TCK and VTK files
            store no grid, so it becomes the tractogram's `reference` as given. A
            `.mat` file needs it to place its points.
        on_progress: Called as the file is read, with the bytes read so far and the
            file's size on disk (compressed bytes for a gzip file). Raise
            ReadCancelledError in it to stop the read.
        max_streamlines: If the file has more streamlines, keep a random subset of
            this size, chosen after the whole file is read. None keeps all.
        seed: Seed of the random subset; the same seed gives the same streamlines.
        space: For a VTK or `.fib` file only, "RAS" or "LPS", the space its points
            are in (see `read_vtk`). None follows the file's `SPACE=` tag, and RAS
            when it has none.

    Returns:
        The tractogram.

    Raises:
        ValueError: If the extension is not supported, the file cannot be read as
            that format, max_streamlines is negative, or `space` is given for a
            file that is not VTK.
        ReadCancelledError: If on_progress raises it.
    """
    if max_streamlines is not None:
        tractogram, _, _ = load_tractogram_subset(
            path,
            max_streamlines,
            seed=seed,
            reference=reference,
            on_progress=on_progress,
            space=space,
        )
        return tractogram
    p = Path(path)
    if space is not None and (
        p.name.endswith(".mat") or _tractogram_format(p) != "vtk"
    ):
        raise ValueError(
            f"{p}: space applies only to VTK and .fib files; other formats fix "
            "their own space."
        )
    if p.name.endswith(".mat"):
        return _load_mat_tracts(p, reference)
    fmt = _tractogram_format(p)
    if fmt == "trk":
        tractogram = read_trk(p, reference=reference, on_progress=on_progress)
    else:
        tractogram = (
            read_tck(p, on_progress=on_progress)
            if fmt == "tck"
            else read_vtk(p, space=space, on_progress=on_progress)
        )
        tractogram.reference = reference
    return tractogram


def _load_mat_tracts(path: Path, reference: Grid | None) -> Tractogram:
    """Reads MRIToolkit's tractography from the `.mat` file `path`, on `reference`."""
    if reference is None:
        raise ValueError(
            f"{path}: an MRIToolkit tractography file stores no grid; give the grid "
            "of the image it was tracked on as the reference."
        )
    return read_mritoolkit_tracts(path, reference)


def load_tractogram_subset(
    path: str | Path,
    max_streamlines: int,
    *,
    seed: int | None = None,
    reference: Grid | None = None,
    on_progress: Progress | None = None,
    space: str | None = None,
) -> tuple[Tractogram, np.ndarray, int]:
    """Reads a tractogram, keeping at most `max_streamlines`, and says which it kept.

    The file is read whole. The subset is the one `load_tractogram` keeps for the
    same seed. The returned indices match values stored per streamline of the whole
    file, such as a TSF or weights file, to the streamlines kept.

    Args:
        path: A `.tck`, `.trk`, `.vtk` or `.fib` file, each also gzipped (`.gz`).
        max_streamlines: The most streamlines to keep.
        seed: Seed of the random subset; the same seed gives the same streamlines.
        reference: As in `load_tractogram`.
        on_progress: As in `load_tractogram`.
        space: As in `load_tractogram`.

    Returns:
        The tractogram; the (S,) int64 indices of its streamlines in the file,
        ascending (all of them if none was dropped); and the number of
        streamlines in the file.

    Raises:
        ValueError: If the extension is not supported, the file cannot be read as
            that format, or max_streamlines is negative.
        ReadCancelledError: If on_progress raises it.
    """
    if max_streamlines < 0:
        raise ValueError(f"max_streamlines must be 0 or more, got {max_streamlines}.")
    whole = load_tractogram(
        path, reference=reference, on_progress=on_progress, space=space
    )
    total = whole.num_streamlines
    if total <= max_streamlines:
        return whole, np.arange(total, dtype=np.int64), total
    indices = whole.subsample_indices(max_streamlines, seed=seed)
    return whole[indices], indices, total


# Streamlines per batch of iter_tractogram.
DEFAULT_BATCH_SIZE = 10_000


def iter_tractogram(
    path: str | Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    reference: Grid | None = None,
) -> Iterator[Tractogram]:
    """Reads a tractogram batch by batch, for files too large to hold in memory.

    The streamlines are handed out in order, `batch_size` at a time, and a
    streamline is never split between batches. Joined in order, the batches are
    the tractogram `load_tractogram` returns, and each batch has its `point_data`,
    `streamline_data` and reference. A file without streamlines gives one empty
    batch.

    Args:
        path: An uncompressed `.tck` or `.trk` file.
        batch_size: The most streamlines per batch; every batch but the last has
            exactly this many.
        reference: As for `load_tractogram`.

    Returns:
        An iterator over the batches.

    Raises:
        ValueError: At the call, if the file is not `.tck` or `.trk`, is compressed,
            or batch_size is less than 1. While iterating, if the file cannot be
            read as its format.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be 1 or more, got {batch_size}.")
    p = Path(path)
    fmt = _tractogram_format(p)
    if fmt == "vtk" or p.name.endswith(".gz"):
        raise ValueError(
            f"{p}: only uncompressed .tck and .trk files can be read in batches; "
            "load it whole with load_tractogram."
        )
    if fmt == "trk":
        return _rebatch(read_trk_chunks(p, reference=reference), batch_size)
    return _rebatch(read_tck_chunks(p), batch_size, reference)


def _rebatch(
    pieces: Iterable[Tractogram], batch_size: int, reference: Grid | None = None
) -> Iterator[Tractogram]:
    """Yields `pieces` of whole streamlines regrouped into batches of `batch_size`.

    `reference`, when given, is set on every piece.
    """
    held: list[Tractogram] = []
    num_held = 0
    last = None
    for piece in pieces:
        if reference is not None:
            piece.reference = reference
        last = piece
        start = 0
        while num_held + piece.num_streamlines - start >= batch_size:
            stop = start + batch_size - num_held
            held.append(_streamline_range(piece, start, stop))
            # The pieces are dropped before the batch is handed out.
            batch = _join_pieces(held)
            held, num_held, start = [], 0, stop
            yield batch
        if start < piece.num_streamlines:
            held.append(_streamline_range(piece, start, piece.num_streamlines))
            num_held += piece.num_streamlines - start
    if held:
        yield _join_pieces(held)
    elif last is not None and last.num_streamlines == 0:
        yield last


def _streamline_range(tractogram: Tractogram, start: int, stop: int) -> Tractogram:
    """Returns streamlines `start` to `stop` of a tractogram, as views of its arrays."""
    first, end = tractogram.offsets[start], tractogram.offsets[stop]
    return Tractogram(
        tractogram.points[first:end],
        tractogram.offsets[start : stop + 1] - first,
        {name: v[first:end] for name, v in tractogram.point_data.items()},
        {name: v[start:stop] for name, v in tractogram.streamline_data.items()},
        tractogram.reference,
    )


def _join_pieces(pieces: list[Tractogram]) -> Tractogram:
    """Joins the streamlines of `pieces` of one file, in order, into one tractogram."""
    if len(pieces) == 1:
        # Copied, so that the batch does not keep the whole larger piece in memory.
        piece = pieces[0]
        return Tractogram(
            piece.points.copy(),
            piece.offsets.copy(),
            {name: v.copy() for name, v in piece.point_data.items()},
            {name: v.copy() for name, v in piece.streamline_data.items()},
            piece.reference,
        )
    counts = np.concatenate([np.diff(piece.offsets) for piece in pieces])
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    first = pieces[0]
    return Tractogram(
        np.concatenate([piece.points for piece in pieces]),
        offsets,
        {
            name: np.concatenate([piece.point_data[name] for piece in pieces])
            for name in first.point_data
        },
        {
            name: np.concatenate([piece.streamline_data[name] for piece in pieces])
            for name in first.streamline_data
        },
        first.reference,
    )


def load_header(path: str | Path) -> tuple[Grid, tuple[int, ...]]:
    """Reads only the header of a volume file, without the data.

    Args:
        path: A `.nii`, `.nii.gz`, `.mif`, `.mif.gz` or `.mih` file.

    Returns:
        The grid, and the full shape of the data: (X, Y, Z) or (X, Y, Z, N).

    Raises:
        ValueError: If the extension is not supported, or the file cannot be read
            as that format.
    """
    p = Path(path)
    if _volume_format(p) == "nifti":
        return read_nifti_header(p)
    return read_mif_header(p)


def load_grid(path: str | Path) -> Grid:
    """Reads only the grid of a file, without the data.

    Args:
        path: A volume file (`.nii`, `.nii.gz`, `.mif`, `.mif.gz` or `.mih`), or
            an MRIToolkit DTI or DKI fit (`.mat`).

    Returns:
        The grid, with the spatial shape (X, Y, Z) and the affine.

    Raises:
        ValueError: If the extension is not supported, or the file cannot be read
            as that format.
    """
    if Path(path).name.endswith(".mat"):
        return read_mritoolkit_grid(path)
    return load_header(path)[0]


def output_files(path: str | Path) -> list[Path]:
    """Returns every file that `save_volume` writes for `path`.

    A `.mih` header keeps its data in a `.dat` file of the same name.

    Args:
        path: A `.nii`, `.nii.gz`, `.mif`, `.mif.gz` or `.mih` file.

    Returns:
        The file itself, and for a `.mih` header also its `.dat` data file.

    Raises:
        ValueError: If the extension is not supported.
    """
    p = Path(path)
    _volume_format(p)
    return [p, mih_data_file(p)] if p.name.endswith(".mih") else [p]


def save_volume(
    volume: Volume, path: str | Path, *, gradients: GradientTable | None = None
) -> None:
    """Writes a volume, in the format given by the file's extension.

    Args:
        volume: The volume to write.
        path: A `.nii`, `.nii.gz`, `.mif`, `.mif.gz` or `.mih` file; a `.mih` also
            writes a `.dat` (see `output_files`).
        gradients: The volume's gradient table, written into the header. Only MIF
            files can hold one, and only for a 4D volume.

    Raises:
        ValueError: If the extension is not supported, `gradients` is given for a
            NIfTI file or a 3D volume, or does not have one entry per volume.
    """
    p = Path(path)
    if _volume_format(p) == "nifti":
        if gradients is not None:
            raise ValueError(
                f"{p}: a NIfTI file cannot hold a gradient table; write it as "
                "bvals/bvecs files instead, or save as .mif."
            )
        write_nifti(volume, p)
    else:
        write_mif(volume, p, gradients=gradients)


def save_tractogram(
    tractogram: Tractogram, path: str | Path, *, reference: Grid | None = None
) -> None:
    """Writes a tractogram, in the format given by the file's extension.

    Args:
        tractogram: The tractogram to write.
        path: A `.tck`, `.trk`, `.vtk` or `.fib` file, each also gzipped (`.gz`).
            VTK and FIB are written in binary.
        reference: The grid a TRK is written for; defaults to the tractogram's own
            `reference`. TCK and VTK do not use it.

    Raises:
        ValueError: If the extension is not supported, or the tractogram cannot be
            written as that format (a TRK without any reference grid).
    """
    p = Path(path)
    fmt = _tractogram_format(p)
    if fmt == "tck":
        write_tck(tractogram, p)
    elif fmt == "trk":
        write_trk(tractogram, p, reference=reference)
    else:
        write_vtk(tractogram, p)


def open_tractogram_writer(
    path: str | Path, *, reference: Grid | None = None
) -> BatchWriter:
    """Opens a tractogram file to be written batch by batch, by its extension.

    Use it as a context manager and call `write` for each batch; leaving the block
    normally finishes the file. The file is the same, byte for byte, as the one
    `save_tractogram` writes for all the streamlines at once.

    Args:
        path: A `.tck` or `.trk` file; an existing file is overwritten.
        reference: The grid a TRK is written for; TCK does not use it.

    Returns:
        A `BatchWriter` for the file's format.

    Raises:
        ValueError: If the format cannot be written in batches (compressed files
            and VTK), or a TRK has no reference grid.
    """
    p = Path(path)
    fmt = _tractogram_format(p)
    if fmt == "vtk":
        raise ValueError(
            f"Cannot write {p} batch by batch: VTK stores the number of points "
            "before them. Write a .tck or .trk."
        )
    if fmt == "tck":
        return TckWriter(p)
    if reference is None:
        raise ValueError(
            f"Cannot write {p}: TRK needs a reference grid; pass `reference` "
            "(for example volume.grid)."
        )
    return TrkWriter(p, reference)
