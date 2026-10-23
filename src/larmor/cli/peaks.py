"""The `lar peaks` command, which finds the peaks of FODs."""

import typing
from enum import StrEnum
from pathlib import Path

import numpy as np
import typer

from larmor.cli.options import ForceOpt, SHBasis, ThreadsOpt
from larmor.cli.outputs import check_outputs
from larmor.cli.sh_input import read_layout
from larmor.io import load_volume, save_volume

FodArg = typing.Annotated[
    Path,
    typer.Argument(
        help="Path to the FOD image, with its spherical harmonic (SH) coefficients "
        "along the 4th axis. An FOD that stores all SH degrees, not only the even "
        "ones, is read as asymmetric."
    ),
]
OutputArg = typing.Annotated[
    Path,
    typer.Argument(
        help="Path of the peaks image. Each peak takes three volumes, the x, y and "
        "z of its direction in world coordinates (RAS), scaled by its amplitude, "
        "largest peak first; unused "
        "volumes are NaN. For an asymmetric FOD, a peak and its opposite are two "
        "peaks, and the sign of a direction matters."
    ),
]
MaskOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--mask",
        "-m",
        help="Only find peaks inside this mask; voxels outside it are NaN.",
    ),
]
BasisOpt = typing.Annotated[
    SHBasis,
    typer.Option(
        "--basis",
        help="SH basis the FOD is stored in, which the file does not record.",
    ),
]
NPeaksOpt = typing.Annotated[
    int | None,
    typer.Option(
        "--num",
        help="Maximum number of peaks per voxel. Defaults to 3, or to 6 for an "
        "asymmetric FOD.",
        show_default=False,
    ),
]
RelativeThresholdOpt = typing.Annotated[
    float | None,
    typer.Option(
        "--relative-threshold",
        help="Drop peaks whose amplitude is below this fraction, from 0 to 1, of "
        "the voxel's largest peak. Defaults to 0.2 for GRL and mFOD FODs and 0.1 "
        "for CSD and MSMT-CSD ones (see --fod-method).",
        show_default=False,
    ),
]


class FodMethod(StrEnum):
    """The method that fitted the FOD, which sets the default relative threshold."""

    GRL = "GRL"
    MFOD = "mFOD"
    CSD = "CSD"
    MSMT_CSD = "MSMT-CSD"


FodMethodOpt = typing.Annotated[
    FodMethod,
    typer.Option(
        "--fod-method",
        case_sensitive=False,
        help="Method that fitted the FOD, as given to lar fod --method, which the "
        "file does not record. It sets the default --relative-threshold. Give CSD "
        "for FODs from other software.",
    ),
]
ThresholdOpt = typing.Annotated[
    float,
    typer.Option("--threshold", help="Drop peaks whose amplitude is below this value."),
]


def peaks(
    fod_path: FodArg,
    output_path: OutputArg,
    mask_path: MaskOpt = None,
    basis: BasisOpt = SHBasis.tournier,
    num_peaks: NPeaksOpt = None,
    relative_threshold: RelativeThresholdOpt = None,
    fod_method: FodMethodOpt = FodMethod.GRL,
    threshold: ThresholdOpt = 0.0,
    force: ForceOpt = False,
    jobs: ThreadsOpt = -1,
) -> None:
    """Find the peaks (local maxima) of an FOD image in every voxel."""
    from larmor.sd import (
        PEAK_RELATIVE_THRESHOLD,
        SD_MODEL_PEAK_RELATIVE_THRESHOLD,
        find_peaks,
    )

    if num_peaks is not None and num_peaks < 1:
        raise typer.BadParameter("must be at least 1", param_hint="--num")
    if relative_threshold is None:
        relative_threshold = (
            SD_MODEL_PEAK_RELATIVE_THRESHOLD
            if fod_method in (FodMethod.GRL, FodMethod.MFOD)
            else PEAK_RELATIVE_THRESHOLD
        )
    if not 0 <= relative_threshold <= 1:
        raise typer.BadParameter(
            "must be between 0 and 1", param_hint="--relative-threshold"
        )
    if jobs == 0 or jobs < -1:
        raise typer.BadParameter("must be -1 (all) or at least 1", param_hint="--nt")

    check_outputs([output_path], force)

    fod = load_volume(fod_path)
    if fod.data.ndim != 4:
        raise typer.BadParameter(
            f"expected a 4D image of SH coefficients, got shape {fod.data.shape}",
            param_hint="FOD_PATH",
        )
    _, full = read_layout(fod_path, fod.data.shape[3])
    if num_peaks is None:
        num_peaks = 6 if full else 3
    mask = None if mask_path is None else load_volume(mask_path).data.astype(bool)

    found = find_peaks(
        fod.as_float(dtype=np.float64),
        mask=mask,
        basis=basis.value,
        num_peaks=num_peaks,
        relative_threshold=relative_threshold,
        threshold=threshold,
        n_jobs=jobs,
    )

    vectors = found.directions * found.amplitudes[..., None]
    data = vectors.reshape(*vectors.shape[:3], 3 * num_peaks).astype(np.float32)
    save_volume(fod.with_data(data), output_path)
