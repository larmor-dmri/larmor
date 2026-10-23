"""Editing tractogram files, from input files to an output file."""

import dataclasses
import re
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from larmor._log import LarmorLogger
from larmor.io import (
    TRACTOGRAM_EXTENSIONS,
    VOLUME_EXTENSIONS,
    BatchWriter,
    iter_tractogram,
    load_grid,
    load_tractogram,
    load_volume,
    open_tractogram_writer,
    read_streamline_weights,
    read_track_timestamp,
    read_tsf,
    save_tractogram,
    write_streamline_weights,
    write_tsf,
)
from larmor.io.formats import write_tck
from larmor.io.formats.tck import TckWriter, read_tck_count
from larmor.io.formats.trk import read_trk_contents
from larmor.io.load_save import DEFAULT_BATCH_SIZE
from larmor.tractography.editing import (
    EditCounts,
    EditOptions,
    concatenate_tractograms,
    edit,
    edit_batch,
)
from larmor.tractography.regions import Region, Sphere
from larmor.types import Grid, Tractogram

# One number of a sphere written as text.
_NUMBER = r"\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*"
# A sphere written as four comma-separated numbers, x,y,z,r.
_SPHERE = re.compile(",".join([_NUMBER] * 4))


def load_region(spec: str | Path) -> Region:
    """Reads a region given as text, a sphere `x,y,z,r` or the path of a mask image.

    Args:
        spec: Four comma-separated numbers, the centre in world coordinates (RAS+)
            and the radius, in mm; anything else is the path of a mask image
            (`.nii`, `.nii.gz`, `.mif`, `.mif.gz` or `.mih`).

    Returns:
        Region: A `Sphere`, or the mask as a `Volume`.

    Raises:
        ValueError: If the sphere's radius is negative, the text has commas but is
            neither four numbers nor an image's name, or the image cannot be read.
    """
    text = str(spec)
    if isinstance(spec, str) and _SPHERE.fullmatch(text):
        x, y, z, radius = (float(value) for value in text.split(","))
        return Sphere(np.array([x, y, z]), radius)
    image_extensions = tuple(e for exts in VOLUME_EXTENSIONS.values() for e in exts)
    if "," in text and not text.endswith(image_extensions):
        raise ValueError(
            f"Region {text!r}: expected a sphere as x,y,z,r (four numbers: the centre "
            "and the radius in mm) or the path of a mask image."
        )
    return load_volume(text)


def _input_format(path: Path) -> str:
    """Returns an input's format, a key of TRACTOGRAM_EXTENSIONS or "mat"."""
    return "mat" if path.name.endswith(".mat") else _tractogram_format(path)


def _tractogram_format(path: Path) -> str:
    """Returns the format key of TRACTOGRAM_EXTENSIONS that the file name ends with.

    Raises:
        ValueError: If no tractogram extension matches.
    """
    for fmt, extensions in TRACTOGRAM_EXTENSIONS.items():
        if path.name.endswith(extensions):
            return fmt
    supported = ", ".join(e for exts in TRACTOGRAM_EXTENSIONS.values() for e in exts)
    raise ValueError(
        f"{path}: unsupported extension; a tractogram is one of {supported}."
    )


def _tsf_output_path(output: str | Path, name: str) -> Path:
    """Returns the TSF file written for the `point_data` entry `name` next to an output.

    Only the tractogram extension is removed from the output name, so `out.tck`
    and the name `fa` give `out_fa.tsf`.

    Args:
        output: The output tractogram.
        name: The `point_data` entry.

    Returns:
        Path: `<output name>_<name>.tsf`, in the output's folder.
    """
    path = Path(output)
    extensions = sorted(
        (e for exts in TRACTOGRAM_EXTENSIONS.values() for e in exts),
        key=len,
        reverse=True,
    )
    stem = path.name
    for extension in extensions:
        if stem.endswith(extension):
            stem = stem.removesuffix(extension)
            break
    return path.with_name(f"{stem}_{name}.tsf")


def edit_files(
    inputs: Sequence[str | Path],
    output: str | Path,
    options: EditOptions,
    *,
    weights_in: str | Path | None = None,
    weights_out: str | Path | None = None,
    tsf_in: Sequence[str | Path] = (),
    tsf_out: Sequence[str] = (),
    reference: Grid | str | Path | None = None,
    force: bool = False,
    lazy: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[int, int]:
    """Edits tractogram files into one output file, as `edit` does in memory.

    The inputs are read and joined in order (see `concatenate_tractograms`), the
    weights and TSF values attached, the edit run and the result written with its
    weights and TSF files. Everything that can be checked from the paths and headers
    is checked before any streamline is read.

    With `lazy`, each input is read, edited and written in batches of `batch_size`
    streamlines, and reading stops once `number` streamlines are written. What the
    join keeps is decided on the first batch of every input. An error during the
    run leaves the output marked as incomplete, but not the `weights_out` file,
    which holds the weights written so far.

    The regions of `options` may be given as text (`x,y,z,r` or a mask image's
    path), read with `load_region`.

    The weights are `streamline_data["weight"]`, from `weights_in` or else from a
    TRK input's `weight` property, which `weights_in` replaces with a warning. Each
    `tsf_in` file becomes the `point_data` entry named after its stem; each
    `tsf_out` entry is written as `<output name>_<name>.tsf`, with only the
    tractogram extension removed from the output name (`out.tck` and `fa` give
    `out_fa.tsf`). A TCK output and the TSF files share one timestamp, that of a
    single TCK input that has one, else a new one.

    Args:
        inputs: The input tractograms, `.tck`, `.trk` or `.vtk`/`.fib` (each also
            gzipped), joined in this order.
        output: The output tractogram, of any of these formats.
        options: The selections and edits.
        weights_in: A weights file with one value per streamline of the single
            input. Defaults to None.
        weights_out: The file to write the weights of the output streamlines to.
            Defaults to None.
        tsf_in: TSF files with values per point of the single input. Defaults to
            none.
        tsf_out: Names of `point_data` entries to write as TSF files next to the
            output. Defaults to none.
        reference: The grid of a TRK output when the inputs do not all have one, or
            an image whose grid it is (a NIfTI or MIF file, or an MRIToolkit fit
            `.mat`); also passed to `load_tractogram`. Needed for MRIToolkit
            tractography inputs (`.mat`). Defaults to None.
        force: Whether to overwrite existing output files. Defaults to False.
        lazy: Whether to read, edit and write in batches. Defaults to False.
        batch_size: Streamlines per batch with `lazy`.

    Returns:
        tuple[int, int]: The number of streamlines read (fewer than in the inputs
            when `lazy` reading stops early at `number`) and written. The read
            count is that of the streamlines passed to the edit: the first batches
            of later inputs, read before the stop to decide the join, are not
            counted.

    Raises:
        FileExistsError: If an output file exists and `force` is False.
        FileNotFoundError: If the folder of an output file does not exist.
        ValueError: Before any work: if no input is given, an extension is not a
            tractogram's, `weights_in` or `tsf_in` is given with several inputs,
            two `tsf_in` files have the same stem, a weight limit or `weights_out`
            is given without weights, a `tsf_out` name is not provided by every
            input, a TRK output has no reference grid, or `lazy` is given with a
            VTK or compressed file, `subsample`, TSF files or an output that is
            also an input. During the work: if the weights or TSF values do not
            match the streamlines, a VTK input lacks the weights or TSF values
            asked for (VTK arrays are known only once read), or a file cannot be
            read or written.

    References:
        Follows MRtrix3's tckedit (version 3.0.8). Tournier, J.-D. et al. MRtrix3: A
        fast, flexible and open software framework for medical image processing and
        visualisation. NeuroImage 202, 116137 (2019).
        doi:10.1016/j.neuroimage.2019.116137
    """
    input_paths = [Path(p) for p in inputs]
    output_path = Path(output)
    tsf_in_paths = [Path(p) for p in tsf_in]
    # The outputs first, so an existing output is reported before any input is read.
    outputs = [output_path]
    outputs += [Path(weights_out)] if weights_out is not None else []
    outputs += [_tsf_output_path(output_path, name) for name in tsf_out]
    _check_outputs(outputs, force)
    _check_request(
        input_paths,
        output_path,
        options,
        weights_in=weights_in,
        weights_out=weights_out,
        tsf_in=tsf_in_paths,
        tsf_out=tsf_out,
        reference=reference,
        lazy=lazy,
    )
    options = _with_loaded_regions(options)
    if reference is not None and not isinstance(reference, Grid):
        reference = load_grid(reference)

    timestamp = None
    if len(input_paths) == 1 and _input_format(input_paths[0]) == "tck":
        timestamp = read_track_timestamp(input_paths[0])
    if timestamp is None:
        timestamp = repr(time.time())

    weights = None if weights_in is None else read_streamline_weights(weights_in)
    if lazy:
        return _edit_lazily(
            input_paths,
            output_path,
            options,
            weights,
            str(weights_in),
            weights_out,
            reference,
            timestamp,
            batch_size,
        )

    tractograms = [load_tractogram(p, reference=reference) for p in input_paths]
    t = (
        tractograms[0]
        if len(tractograms) == 1
        else concatenate_tractograms(tractograms)
    )
    del tractograms
    num_read = t.num_streamlines
    if weights is not None:
        _attach_weights(t, weights, str(weights_in))
    for path in tsf_in_paths:
        values, _ = read_tsf(path, tractogram=t)
        t.point_data[path.stem] = values
    _check_entries(t, options, weights_out, tsf_out)
    result = edit(t, options)
    del t

    if _tractogram_format(output_path) == "tck":
        write_tck(
            _without_side_files(
                result,
                weights_in is not None or weights_out is not None,
                [*tsf_out, *(path.stem for path in tsf_in_paths)],
            ),
            output_path,
            timestamp=timestamp,
        )
    else:
        save_tractogram(result, output_path, reference=reference)
    if weights_out is not None:
        write_streamline_weights(result.streamline_data["weight"], weights_out)
    for name in tsf_out:
        write_tsf(
            result.point_data[name],
            result.offsets,
            _tsf_output_path(output_path, name),
            timestamp=timestamp,
        )
    return num_read, result.num_streamlines


def _check_request(
    inputs: list[Path],
    output: Path,
    options: EditOptions,
    *,
    weights_in: str | Path | None,
    weights_out: str | Path | None,
    tsf_in: list[Path],
    tsf_out: Sequence[str],
    reference: Grid | str | Path | None,
    lazy: bool,
) -> None:
    """Checks everything that can be refused before reading streamlines.

    The arguments are those of `edit_files`, with the paths as `Path`.

    Raises:
        ValueError: If the request is refused, as listed in `edit_files`.
    """
    if not inputs:
        raise ValueError("Give at least one input tractogram.")
    formats = [_input_format(p) for p in inputs]
    output_format = _tractogram_format(output)
    if "mat" in formats and reference is None:
        raise ValueError(
            "An MRIToolkit tractography file (.mat) stores no grid; give reference: "
            "the fit it was tracked on (.mat), or a NIfTI or MIF file of its image."
        )
    if len(inputs) > 1 and (weights_in is not None or tsf_in):
        raise ValueError(
            "weights_in and tsf_in describe one input tractogram; join the inputs "
            "first to use them with several."
        )
    stems = [p.stem for p in tsf_in]
    repeated = sorted({stem for stem in stems if stems.count(stem) > 1})
    if repeated:
        raise ValueError(
            f"Several tsf_in files have the stem {repeated[0]!r}, which names the "
            "entry each one becomes."
        )

    # The entries (per point, per streamline) each input provides; a VTK's are
    # known only once it is read (None), so they are checked again then. Only an
    # entry every input has survives the join.
    known = _known_entries(inputs, formats)
    has_weights = weights_in is not None or _may_provide(known, "weight", True)
    if (options._selects_by_weight or weights_out is not None) and not has_weights:
        raise ValueError(_NO_WEIGHTS)
    missing = [
        n for n in tsf_out if n not in stems and not _may_provide(known, n, False)
    ]
    if missing:
        raise ValueError(_no_tsf_values(missing[0]))
    all_trk = all(fmt == "trk" for fmt in formats)
    if output_format == "trk" and reference is None and not all_trk:
        raise ValueError(
            f"Cannot write {output}: a TRK output needs a reference grid when not "
            "every input is a TRK; give reference."
        )

    if lazy:
        _check_lazy(inputs, output, formats, output_format, options, tsf_in, tsf_out)


def _known_entries(
    inputs: list[Path], formats: list[str]
) -> list[tuple[set[str], set[str]] | None]:
    """Returns the per-point and per-streamline entries each input provides.

    Returns:
        For each input, the names of its per-point and per-streamline entries, or
        None for any other format, whose entries are known only once it is read.
    """
    known: list[tuple[set[str], set[str]] | None] = []
    for path, fmt in zip(inputs, formats, strict=True):
        if fmt == "trk":
            scalars, properties, _ = read_trk_contents(path)
            known.append(
                (
                    {name for name, k in scalars if k == 1},
                    {name for name, k in properties if k == 1},
                )
            )
        else:
            known.append((set(), set()) if fmt == "tck" else None)
    return known


def _may_provide(
    known: list[tuple[set[str], set[str]] | None], name: str, per_streamline: bool
) -> bool:
    """Returns whether every input may have the entry `name`."""
    return all(k is None or name in k[per_streamline] for k in known)


def _check_lazy(
    inputs: list[Path],
    output: Path,
    formats: list[str],
    output_format: str,
    options: EditOptions,
    tsf_in: list[Path],
    tsf_out: Sequence[str],
) -> None:
    """Checks that a lazy edit reads and writes only what it can.

    Raises:
        ValueError: If a file is not an uncompressed TCK or TRK, the edit
            subsamples or uses TSF files, or the output is also an input.
    """
    compressed = [p for p in [*inputs, output] if p.name.endswith(".gz")]
    if {"vtk", "mat"} & set(formats) or output_format == "vtk" or compressed:
        raise ValueError(
            "Lazy editing reads and writes only uncompressed TCK and TRK files."
        )
    if options.subsample is not None:
        raise ValueError(
            "Lazy editing cannot subsample: it needs every selected streamline "
            "first. Use number or every, or edit without lazy."
        )
    if tsf_in or tsf_out:
        raise ValueError("Lazy editing does not read or write TSF files.")
    if any(p.resolve() == output.resolve() for p in inputs):
        raise ValueError(
            f"{output} is also an input; lazy editing would overwrite it while "
            "reading it."
        )


_NO_WEIGHTS = (
    "A weight limit or weights_out needs weights: give weights_in, or inputs that "
    "all have a 'weight' per streamline (a TRK property or a VTK cell array)."
)


def _no_tsf_values(name: str) -> str:
    """Returns the error message for a tsf_out name that no input provides."""
    return (
        f"No values named {name!r} to write as a TSF file: the names come from "
        "tsf_in and the values per point every input has (TRK scalars, VTK point "
        "arrays)."
    )


def _check_entries(
    tractogram: Tractogram,
    options: EditOptions,
    weights_out: str | Path | None,
    tsf_out: Sequence[str],
) -> None:
    """Raises ValueError if the tractogram lacks the weights or TSF values asked for."""
    if (options._selects_by_weight or weights_out is not None) and (
        "weight" not in tractogram.streamline_data
    ):
        raise ValueError(_NO_WEIGHTS)
    for name in tsf_out:
        if name not in tractogram.point_data or tractogram.point_data[name].ndim != 1:
            raise ValueError(_no_tsf_values(name))


def _without_side_files(
    tractogram: Tractogram, drop_weight: bool, point_names: Sequence[str]
) -> Tractogram:
    """Returns the tractogram without the entries read from or written to side files.

    Used for a TCK output, so that the TCK writer reports as dropped only the
    entries the inputs carried themselves. `drop_weight` drops the weights and
    `point_names` names the per-point entries to drop.
    """
    return Tractogram(
        tractogram.points,
        tractogram.offsets,
        {k: v for k, v in tractogram.point_data.items() if k not in point_names},
        {
            k: v
            for k, v in tractogram.streamline_data.items()
            if not (k == "weight" and drop_weight)
        },
    )


def _with_loaded_regions(options: EditOptions) -> EditOptions:
    """Returns the options with every region given as text or a path read."""

    def loaded(specs: Sequence[Region | str | Path]) -> list[Region]:
        return [
            load_region(spec) if isinstance(spec, (str, Path)) else spec
            for spec in specs
        ]

    return dataclasses.replace(
        options,
        include=loaded(options.include),
        include_ordered=loaded(options.include_ordered),
        exclude=loaded(options.exclude),
        within=loaded(options.within),
        mask=loaded(options.mask),
    )


def _check_outputs(paths: list[Path], force: bool) -> None:
    """Raises if an output's folder is missing, or it exists and force is False.

    Raises:
        FileNotFoundError: If the folder of an output does not exist.
        FileExistsError: If an output exists and `force` is False.
    """
    for path in paths:
        if not path.parent.is_dir():
            raise FileNotFoundError(f"Output folder {path.parent} does not exist.")
        if path.exists() and not force:
            raise FileExistsError(f"File {path} exists. Use --force to overwrite.")


def _attach_weights(
    tractogram: Tractogram, weights: np.ndarray, path: str, warn: bool = True
) -> None:
    """Stores the weights as `streamline_data["weight"]`, replacing a TRK property.

    `path` names the weights file in the messages; `warn` logs a warning when a
    property is replaced.

    Raises:
        ValueError: If there is not one weight per streamline.
    """
    if len(weights) != tractogram.num_streamlines:
        raise ValueError(
            f"{path} has {len(weights)} weights, but the tractogram has "
            f"{tractogram.num_streamlines} streamlines."
        )
    if warn and "weight" in tractogram.streamline_data:
        LarmorLogger.warning(
            "The weights of %s replace the input's 'weight' property.", path
        )
    tractogram.streamline_data["weight"] = weights


def _header_count(path: Path) -> int | None:
    """Returns the streamline count a TCK or TRK header records, or None if unknown."""
    if _tractogram_format(path) == "tck":
        return read_tck_count(path)
    count = read_trk_contents(path)[2]
    return count if count > 0 else None


def _conform(batch: Tractogram, template: Tractogram) -> Tractogram:
    """Returns the batch with the data entries, dtypes and reference of `template`."""
    return Tractogram(
        batch.points,
        batch.offsets,
        {
            name: batch.point_data[name].astype(values.dtype, copy=False)
            for name, values in template.point_data.items()
        },
        {
            name: batch.streamline_data[name].astype(values.dtype, copy=False)
            for name, values in template.streamline_data.items()
        },
        template.reference,
    )


def _edit_lazily(
    inputs: list[Path],
    output: Path,
    options: EditOptions,
    weights: np.ndarray | None,
    weights_path: str,
    weights_out: str | Path | None,
    reference: Grid | None,
    timestamp: str,
    batch_size: int,
) -> tuple[int, int]:
    """Runs `edit_files` batch by batch, after its checks.

    Returns:
        The number of streamlines read and written.

    Raises:
        ValueError: If the weights do not match the streamlines.
    """
    if weights is not None:
        _check_header_count(inputs[0], weights, weights_path)

    batches = [
        iter_tractogram(p, batch_size=batch_size, reference=reference) for p in inputs
    ]
    try:
        # One batch of every input decides what the join keeps.
        firsts: list[Tractogram | None] = [next(it) for it in batches]
        template = concatenate_tractograms([t for t in firsts if t is not None])
        if weights is not None and "weight" in template.streamline_data:
            LarmorLogger.warning(
                "The weights of %s replace the input's 'weight' property.",
                weights_path,
            )

        tck_output = _tractogram_format(output) == "tck"
        writer = _open_writer(output, tck_output, reference, template, timestamp)
        if weights_out is not None:
            write_streamline_weights(np.zeros(0, dtype=np.float32), weights_out)

        counts = EditCounts()
        num_read = num_written = 0
        with writer:
            for index, rest in enumerate(batches):
                first = firsts[index]
                firsts[index] = None
                assert first is not None
                for batch in _chain(first, rest):
                    if counts.reached_number(options):
                        break
                    batch = _conform(batch, template)
                    if weights is not None:
                        _attach_weights(
                            batch,
                            _weights_slice(
                                weights, num_read, batch.num_streamlines, weights_path
                            ),
                            weights_path,
                            warn=False,
                        )
                    num_read += batch.num_streamlines
                    result = edit_batch(batch, options, counts)
                    del batch
                    _write_result(writer, result, tck_output, weights, weights_out)
                    num_written += result.num_streamlines
            stopped_early = counts.reached_number(options)
            if weights is not None and not stopped_early and num_read != len(weights):
                raise ValueError(
                    f"{weights_path} has {len(weights)} weights, but the input has "
                    f"{num_read} streamlines."
                )
    finally:
        _close_all(batches)
    return num_read, num_written


def _check_header_count(path: Path, weights: np.ndarray, weights_path: str) -> None:
    """Checks the weights against the streamline count in the input's header.

    Raises:
        ValueError: If the header gives a count other than the number of weights.
    """
    count = _header_count(path)
    if count is not None and count != len(weights):
        raise ValueError(
            f"{weights_path} has {len(weights)} weights, but {path} has "
            f"{count} streamlines."
        )


def _open_writer(
    output: Path,
    tck_output: bool,
    reference: Grid | None,
    template: Tractogram,
    timestamp: str,
) -> BatchWriter:
    """Opens the batch writer of the output, on the reference or the input's grid."""
    if tck_output:
        return TckWriter(output, timestamp=timestamp)
    grid = reference if reference is not None else template.reference
    return open_tractogram_writer(output, reference=grid)


def _write_result(
    writer: BatchWriter,
    result: Tractogram,
    tck_output: bool,
    weights: np.ndarray | None,
    weights_out: str | Path | None,
) -> None:
    """Writes one edited batch, and appends its weights to weights_out if given.

    A TCK output drops the weight entry when weights are given or written out.
    """
    if tck_output:
        drop_weight = weights is not None or weights_out is not None
        writer.write(_without_side_files(result, drop_weight, ()))
    else:
        writer.write(result)
    if weights_out is not None:
        write_streamline_weights(
            result.streamline_data["weight"], weights_out, append=True
        )


def _close_all(batches: list[Iterator[Tractogram]]) -> None:
    """Closes each batch iterator that can be closed."""
    for it in batches:
        close = getattr(it, "close", None)
        if close is not None:
            close()


def _chain(first: Tractogram, rest: Iterator[Tractogram]) -> Iterator[Tractogram]:
    """Yields the first batch, then the remaining batches of the same input."""
    yield first
    yield from rest


def _weights_slice(
    weights: np.ndarray, start: int, count: int, path: str
) -> np.ndarray:
    """Returns the weights of `count` streamlines from `start`.

    Raises:
        ValueError: If the weights file `path` holds fewer.
    """
    if start + count > len(weights):
        raise ValueError(
            f"{path} has {len(weights)} weights, but the input has more streamlines."
        )
    return weights[start : start + count]
