"""The `lar dti` command, which fits the diffusion tensor and writes its maps."""

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
    """Fit methods of `lar dti`, as those of `DTIModel`."""

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
        help="Fitting method. 'ols' is ordinary least squares, and 'wls' is "
        "iterated weighted least squares, weighted by the predicted signal.",
    ),
]

OutputArg = typing.Annotated[
    str,
    typer.Argument(
        help="Prefix of the output maps. 'out' writes the fractional anisotropy "
        "to out_fa.nii.gz, the mean, axial and radial diffusivity in mm^2/s to "
        "out_md, out_ad and out_rd, the principal eigenvector to out_pe and a mask "
        "of the voxels whose tensor is not positive definite to out_npd, all as "
        ".nii.gz. The eigenvector is in world coordinates (RAS), not in the voxel "
        "frame of the image.",
    ),
]


# The DTIFit attributes saved, each to <prefix>_<name>.nii.gz.
METRICS = ("fa", "md", "ad", "rd", "pe", "npd")


def dti(
    volume: DiffusionArg,
    output: OutputArg,
    bvals: BValsOpt = None,
    bvecs: BVecsOpt = None,
    bvec_frame: BvecFrameOpt = BvecFrame.fsl,
    mask: MaskArg = None,
    fit: FitOpt = FitMethod.ols,
    force: ForceOpt = False,
    jobs: ThreadsOpt = -1,
) -> None:
    """Fit the diffusion tensor (DTI) in every voxel and write its maps."""
    # Imported here, so that starting lar and Tab completion do not load scipy.
    from larmor.dwi import DTIModel

    prefix = output_prefix(output)
    outputs = {name: with_suffix(prefix, f"_{name}.nii.gz") for name in METRICS}
    check_outputs(list(outputs.values()), force)

    image, gtab = load_dwi(volume, bvals=bvals, bvecs=bvecs, frame=bvec_frame.value)
    mask_volume = load_volume(mask) if mask is not None else None
    dti_fit = DTIModel(fit_method=fit.value).fit(image, gtab, mask_volume, n_jobs=jobs)

    for name, path in outputs.items():
        values = getattr(dti_fit, name)
        # The boolean non-positive-definite flag is saved as 0 and 1.
        if values.dtype == bool:
            values = values.astype(np.uint8)
        save_volume(image.with_data(values), path)
        LarmorLogger.info(f"Saved {name.upper()} to {path}")
