"""The `lar convert` command, which converts images and tractograms."""

import typing
from enum import StrEnum
from pathlib import Path

import typer

from larmor._log import LarmorLogger
from larmor.cli.options import (
    BvecFrame,
    BvecFrameOpt,
    ForceOpt,
)
from larmor.cli.outputs import check_outputs
from larmor.io import (
    TRACTOGRAM_EXTENSIONS,
    VOLUME_EXTENSIONS,
    file_kind,
    has_mif_gradients,
    load_dwi,
    load_grid,
    load_tractogram,
    load_volume,
    read_matlab,
    read_mritoolkit_fit,
    read_mritoolkit_kind,
    read_mritoolkit_tracts,
    save_tractogram,
    save_volume,
    write_bvals_bvecs,
)

InputArg = typing.Annotated[
    Path,
    typer.Argument(
        help="Path to the image or tractogram to convert, or a MATLAB .mat file "
        "with an MRIToolkit result (a DTI or DKI fit, a GRL or mFOD result, or "
        "tractography).",
    ),
]

OutputArg = typing.Annotated[
    Path,
    typer.Argument(
        help="Path of the converted file, of the same kind (image or tractogram) "
        "as the input; its extension sets the format. A .mat input also writes "
        "files next to it, with the same extension. From a DTI or DKI fit, "
        "'out.nii.gz' receives the DWI, and out_fa, out_md, out_ad, out_rd, out_pe, "
        "out_dt and, for DKI, out_kt receive the maps. From a GRL or mFOD result, "
        "'out.nii.gz' receives the FOD in the tournier basis (out_fod1, out_fod2 "
        "and so on for several FODs), and out_tissue_fraction and out_rss receive "
        "the tissue fractions and the residual sum of squares.",
    ),
]

# Own --bvals and --bvecs, since a conversion does not need a gradient table.
InBValsOpt = typing.Annotated[
    str | None,
    typer.Option(
        "--bvals",
        help="b-values file of the input image, given with --bvecs. The gradient "
        "table goes into the header of a .mif output, and to --bvals-out and "
        "--bvecs-out if they are given.",
    ),
]
InBVecsOpt = typing.Annotated[
    str | None,
    typer.Option(
        "--bvecs",
        help="b-vectors file of the input image, given with --bvals.",
    ),
]

BValsOutOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--bvals-out",
        help="Write the b-values of the gradient table to this file, in FSL "
        "format. Give it with --bvecs-out.",
    ),
]
BVecsOutOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--bvecs-out",
        help="Write the b-vectors of the gradient table to this file, in FSL "
        "format. Give it with --bvals-out.",
    ),
]
ToRasOpt = typing.Annotated[
    bool,
    typer.Option(
        "--to-ras",
        help="Reorient an image to the closest RAS orientation by permuting and "
        "flipping its voxel axes; every voxel keeps its world position.",
    ),
]
ReferenceOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--reference",
        help="Image whose grid a .trk output is written for, or that a .trk input "
        "is checked against. It is needed for a .trk output unless the input is a "
        ".trk, and for MRIToolkit tracts and GRL or mFOD results (.mat), which "
        "store no grid; for these, give the fit (.mat) they came from, or a NIfTI "
        "or MIF file of its image.",
    ),
]


class VtkSpace(StrEnum):
    """Space of the points in a VTK or .fib file."""

    RAS = "RAS"
    LPS = "LPS"


VtkSpaceOpt = typing.Annotated[
    VtkSpace | None,
    typer.Option(
        "--vtk-space",
        help="Space of the points of a .vtk or .fib input, RAS or LPS (x and y "
        "negated); LPS is converted to RAS. Defaults to the SPACE= tag of the "
        "file's title line, or to RAS for a file without one.",
    ),
]


def _is_mif(path: Path) -> bool:
    """Returns whether path is a MIF image."""
    return path.name.endswith(VOLUME_EXTENSIONS["mif"])


def _is_trk(path: Path) -> bool:
    """Returns whether path is a TRK tractogram."""
    return path.name.endswith(TRACTOGRAM_EXTENSIONS["trk"])


def _is_vtk(path: Path) -> bool:
    """Returns whether path is a VTK or .fib tractogram."""
    return path.name.endswith(TRACTOGRAM_EXTENSIONS["vtk"])


def _convert_volume(
    input_path: Path,
    output_path: Path,
    bvals: str | None,
    bvecs: str | None,
    bvec_frame: BvecFrame,
    out_bvals: Path | None,
    out_bvecs: Path | None,
    to_ras: bool,
) -> None:
    """Converts an image, with its gradient table if it has one.

    The table comes from --bvals and --bvecs, or else from a MIF input's header.
    It goes into a MIF output's header and to --bvals-out and --bvecs-out, with a
    warning when it reaches neither.

    Raises:
        ValueError: If --bvals-out is given and there is no gradient table to write.
    """
    has_table = bvals is not None or bvecs is not None
    has_table = has_table or (_is_mif(input_path) and has_mif_gradients(input_path))
    if out_bvals is not None and not has_table:
        raise ValueError(
            f"{input_path} has no gradient table to write to --bvals-out and "
            "--bvecs-out: give --bvals and --bvecs, or a .mif with a table."
        )

    if has_table:
        volume, gtab = load_dwi(
            input_path, bvals=bvals, bvecs=bvecs, frame=bvec_frame.value
        )
    else:
        volume, gtab = load_volume(input_path), None
    # The table is in world coordinates, so reorienting the voxels leaves it as is.
    if to_ras:
        volume.to_closest_canonical()

    save_volume(volume, output_path, gradients=gtab if _is_mif(output_path) else None)
    if gtab is not None and out_bvals is not None and out_bvecs is not None:
        # After the image: FSL bvecs are written relative to the saved file.
        write_bvals_bvecs(gtab, out_bvals, out_bvecs, ref_path=output_path)
    elif gtab is not None and not _is_mif(output_path):
        LarmorLogger.warning(
            "%s cannot hold a gradient table and it was not written; give "
            "--bvals-out and --bvecs-out to keep it.",
            output_path,
        )


def _convert_tractogram(
    input_path: Path,
    output_path: Path,
    reference: Path | None,
    vtk_space: VtkSpace | None,
) -> None:
    """Converts a tractogram.

    A .trk output gets its grid from --reference, or else from the input .trk.
    """
    grid = load_grid(reference) if reference is not None else None
    space = None if vtk_space is None else vtk_space.value
    tractogram = load_tractogram(input_path, reference=grid, space=space)
    save_tractogram(tractogram, output_path)


def _split_extension(path: Path) -> tuple[Path, str]:
    """Returns the path without its volume extension, and the extension."""
    for extension in (e for exts in VOLUME_EXTENSIONS.values() for e in exts):
        if path.name.endswith(extension):
            return path.with_name(path.name.removesuffix(extension)), extension
    return path, ""


# The name of each kind of MRIToolkit result, and the kind of file it becomes.
_KIND_NAMES = {
    "fit": ("an MRIToolkit tensor fit", "volume"),
    "fod": ("an MRIToolkit FOD fit", "volume"),
    "tracts": ("MRIToolkit tractography", "tractogram"),
}


def _matlab_kind(input_path: Path, output_path: Path) -> str:
    """Returns the kind of result in the .mat file, checked against the output."""
    kind = read_mritoolkit_kind(input_path)
    if kind is None:
        raise ValueError(
            f"{input_path.name} holds no MRIToolkit fit, FOD fit or tractography."
        )
    name, wanted = _KIND_NAMES[kind]
    if file_kind(output_path) != wanted:
        extensions = VOLUME_EXTENSIONS if wanted == "volume" else TRACTOGRAM_EXTENSIONS
        listed = ", ".join(e for exts in extensions.values() for e in exts)
        raise ValueError(
            f"{input_path.name} is {name}; convert it to a {wanted} ({listed})."
        )
    return kind


def _matlab_outputs(
    input_path: Path, output_path: Path, kind: str, reference: Path | None
) -> list[Path]:
    """Returns the files a .mat conversion writes, the output first."""
    if kind != "fit" and reference is None:
        raise typer.BadParameter(
            f"needed for {input_path.name}: {_KIND_NAMES[kind][0]} stores no grid",
            param_hint="--reference",
        )
    if kind == "tracts":
        return [output_path]
    stem, extension = _split_extension(output_path)
    f = read_matlab(input_path)
    if kind == "fit":
        names = ["fa", "md", "ad", "rd", "pe", "dt", *(["kt"] if "KT" in f else [])]
        return [
            output_path,
            *(stem.with_name(f"{stem.name}_{n}{extension}") for n in names),
        ]
    # Imported here, so that starting lar does not load scipy.
    from larmor.sd import read_mritoolkit_fod_count

    n = read_mritoolkit_fod_count(f)
    fods = (
        [output_path]
        if n == 1
        else [stem.with_name(f"{stem.name}_fod{i}{extension}") for i in range(1, n + 1)]
    )
    extras = ("tissue_fraction", "rss")
    return [*fods, *(stem.with_name(f"{stem.name}_{e}{extension}") for e in extras)]


def _convert_matlab(
    input_path: Path,
    output_path: Path,
    kind: str,
    reference: Path | None,
    out_bvals: Path | None,
    out_bvecs: Path | None,
    outputs: list[Path],
) -> None:
    """Converts an MRIToolkit result: tracts, a fit's maps, or a GRL result's FODs."""
    if kind == "tracts":
        assert reference is not None
        save_tractogram(read_mritoolkit_tracts(input_path, reference), output_path)
        return
    f = read_matlab(input_path)
    if kind == "fit":
        fit = read_mritoolkit_fit(f)
        is_mif = _is_mif(output_path)
        save_volume(fit.dwi, output_path, gradients=fit.gradients if is_mif else None)
        if out_bvals is not None and out_bvecs is not None:
            write_bvals_bvecs(fit.gradients, out_bvals, out_bvecs, ref_path=output_path)
        maps = [fit.maps[n] for n in ("FA", "MD", "AD", "RD", "FE")] + [fit.dt]
        maps += [] if fit.kt is None else [fit.kt]
        for volume, path in zip(maps, outputs[1:], strict=True):
            save_volume(volume, path)
        return
    from larmor.sd import read_mritoolkit_fod

    assert reference is not None
    result = read_mritoolkit_fod(f, reference)
    volumes = [*result.fods, result.tissue_fractions]
    volumes += [] if result.rss is None else [result.rss]
    for volume, path in zip(volumes, outputs, strict=False):
        save_volume(volume, path)


def _convert_mat_input(
    input_path: Path,
    output_path: Path,
    bvals: str | None,
    bvecs: str | None,
    to_ras: bool,
    vtk_space: VtkSpace | None,
    out_bvals: Path | None,
    out_bvecs: Path | None,
    reference: Path | None,
    force: bool,
) -> None:
    """Checks the options for a .mat input and converts it."""
    given = [bvals, bvecs, vtk_space] + ([True] if to_ras else [])
    if any(o is not None for o in given):
        raise typer.BadParameter(
            "do not apply to a .mat input",
            param_hint="--bvals, --bvecs, --to-ras, --vtk-space",
        )
    if (out_bvals is None) != (out_bvecs is None):
        raise typer.BadParameter(
            "give both or neither", param_hint="--bvals-out, --bvecs-out"
        )
    kind = _matlab_kind(input_path, output_path)
    outputs = _matlab_outputs(input_path, output_path, kind, reference)
    written = outputs + [p for p in (out_bvals, out_bvecs) if p is not None]
    check_outputs(written, force)
    _convert_matlab(
        input_path, output_path, kind, reference, out_bvals, out_bvecs, outputs
    )


def _check_options(
    input_path: Path,
    output_path: Path,
    input_kind: str,
    bvals: str | None,
    bvecs: str | None,
    out_bvals: Path | None,
    out_bvecs: Path | None,
    reference: Path | None,
    to_ras: bool,
    vtk_space: VtkSpace | None,
) -> None:
    """Checks that the options fit the input and output files, before any reading."""
    if (bvals is None) != (bvecs is None):
        raise typer.BadParameter("give both or neither", param_hint="--bvals, --bvecs")
    if (out_bvals is None) != (out_bvecs is None):
        raise typer.BadParameter(
            "give both or neither", param_hint="--bvals-out, --bvecs-out"
        )
    if reference is not None and not (_is_trk(input_path) or _is_trk(output_path)):
        raise typer.BadParameter(
            "applies only to a .trk input or output", param_hint="--reference"
        )
    if vtk_space is not None and not _is_vtk(input_path):
        raise typer.BadParameter(
            "applies only to a .vtk or .fib input", param_hint="--vtk-space"
        )
    if to_ras and input_kind == "tractogram":
        raise typer.BadParameter("applies to images only", param_hint="--to-ras")
    gradient_options = (bvals, bvecs, out_bvals)
    if input_kind == "tractogram" and any(o is not None for o in gradient_options):
        raise typer.BadParameter(
            "apply to images only", param_hint="--bvals, --bvecs, --bvals-out"
        )
    if _is_trk(output_path) and not _is_trk(input_path) and reference is None:
        raise typer.BadParameter(
            f"needed to write {output_path.name}: a .trk stores points relative to "
            "an image grid",
            param_hint="--reference",
        )


def convert(
    input_path: InputArg,
    output_path: OutputArg,
    bvals: InBValsOpt = None,
    bvecs: InBVecsOpt = None,
    bvec_frame: BvecFrameOpt = BvecFrame.fsl,
    out_bvals: BValsOutOpt = None,
    out_bvecs: BVecsOutOpt = None,
    reference: ReferenceOpt = None,
    to_ras: ToRasOpt = False,
    vtk_space: VtkSpaceOpt = None,
    force: ForceOpt = False,
) -> None:
    """Convert an image or a tractogram to another file format of the same kind."""
    if input_path.name.lower().endswith(".mat"):
        _convert_mat_input(
            input_path,
            output_path,
            bvals,
            bvecs,
            to_ras,
            vtk_space,
            out_bvals,
            out_bvecs,
            reference,
            force,
        )
        return
    input_kind, output_kind = file_kind(input_path), file_kind(output_path)
    if input_kind != output_kind:
        raise ValueError(
            f"{input_path} is a {input_kind} and {output_path} a {output_kind}; "
            "lar convert converts between files of the same kind."
        )
    _check_options(
        input_path,
        output_path,
        input_kind,
        bvals,
        bvecs,
        out_bvals,
        out_bvecs,
        reference,
        to_ras,
        vtk_space,
    )

    outputs = [output_path, *(p for p in (out_bvals, out_bvecs) if p is not None)]
    check_outputs(outputs, force)

    if input_kind == "volume":
        _convert_volume(
            input_path,
            output_path,
            bvals,
            bvecs,
            bvec_frame,
            out_bvals,
            out_bvecs,
            to_ras,
        )
    else:
        _convert_tractogram(input_path, output_path, reference, vtk_space)
