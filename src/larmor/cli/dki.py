"""The `lar dki` command, which fits DKI to multi-shell data and writes its maps."""

import typing
from enum import StrEnum

import numpy as np
import typer

from larmor._log import LarmorLogger
from larmor.cli.options import (
    BValsOpt,
    BvecFrame,
    BvecFrameOpt,
    BVecsOpt,
    DiffusionArg,
    ForceOpt,
    ThreadsOpt,
)
from larmor.cli.outputs import check_outputs, output_prefix, with_suffix
from larmor.io import load_dwi, load_volume, save_volume


class FitMethod(StrEnum):
    """Fit methods of `lar dki`, as those of `DKIModel`."""

    ols = "ols"
    wls = "wls"


MaskArg = typing.Annotated[
    str | None,
    typer.Option(
        "--mask",
        "-m",
        help="Brain mask; voxels outside it are not fitted. Without it, a mask is "
        "computed from the mean b = 0 image.",
    ),
]

FitOpt = typing.Annotated[
    FitMethod,
    typer.Option(
        "--fit",
        help="Fitting method. 'wls' is iterated weighted least squares, weighted "
        "by the predicted signal, and 'ols' is ordinary least squares.",
    ),
]

UnconstrainedOpt = typing.Annotated[
    bool,
    typer.Option(
        "--unconstrained",
        help="Fit without constraints. By default, the diffusivity D(n) and the "
        "kurtosis K(n) along every direction n are kept within D(n) >= 0 and "
        "0 <= K(n) <= 3 / (b_max D(n)).",
    ),
]

OutputArg = typing.Annotated[
    str,
    typer.Argument(
        help="Prefix of the output maps. 'out' writes the tensor maps of lar dti "
        "(out_fa, out_md, out_ad, out_rd, out_pe and out_npd) and the mean, axial "
        "and radial kurtosis and the kurtosis anisotropy to out_mk, out_ak, out_rk "
        "and out_ka, all as .nii.gz.",
    ),
]


# The DKIFit attributes saved, each to <prefix>_<name>.nii.gz.
METRICS = ("fa", "md", "ad", "rd", "pe", "npd", "mk", "ak", "rk", "ka")


def dki(
    volume: DiffusionArg,
    output: OutputArg,
    bvals: BValsOpt = None,
    bvecs: BVecsOpt = None,
    bvec_frame: BvecFrameOpt = BvecFrame.fsl,
    mask: MaskArg = None,
    fit: FitOpt = FitMethod.wls,
    unconstrained: UnconstrainedOpt = False,
    force: ForceOpt = False,
    jobs: ThreadsOpt = -1,
) -> None:
    """Fit diffusion kurtosis imaging (DKI) to multi-shell data and write its maps."""
    from larmor.dwi import DKIModel

    prefix = output_prefix(output)
    outputs = {name: with_suffix(prefix, f"_{name}.nii.gz") for name in METRICS}
    check_outputs(list(outputs.values()), force)

    image, gtab = load_dwi(volume, bvals=bvals, bvecs=bvecs, frame=bvec_frame.value)
    mask_volume = load_volume(mask) if mask is not None else None
    dki_fit = DKIModel(fit_method=fit.value, constrained=not unconstrained).fit(
        image, gtab, mask_volume, n_jobs=jobs
    )

    for name, path in outputs.items():
        values = getattr(dki_fit, name)
        # The boolean non-positive-definite flag is saved as 0 and 1.
        if values.dtype == bool:
            values = values.astype(np.uint8)
        save_volume(image.with_data(values), path)
        LarmorLogger.info(f"Saved {name.upper()} to {path}")
