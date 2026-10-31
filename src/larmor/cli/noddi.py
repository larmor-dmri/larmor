"""The `lar noddi` command, which fits NODDI to multi-shell data and writes its maps."""

import logging
import typing

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

MaskArg = typing.Annotated[
    str | None,
    typer.Option(
        "--mask",
        "-m",
        help="Brain mask; voxels outside it are not fitted. Without it, a mask is "
        "computed from the mean b = 0 image.",
    ),
]

OutputArg = typing.Annotated[
    str,
    typer.Argument(
        help="Prefix of the output maps. 'out' writes the neurite density to "
        "out_v_ic.nii.gz, the orientation dispersion index to out_od, the Watson "
        "concentration to out_kappa, the free-water fraction to out_v_iso and the "
        "mean neurite direction to out_mu, all as .nii.gz. The direction is in "
        "world coordinates (RAS), not in the voxel frame of the image.",
    ),
]


# The NoddiFit attributes saved, each to <prefix>_<name>.nii.gz.
METRICS = ("v_ic", "od", "kappa", "v_iso", "mu")


def noddi(
    volume: DiffusionArg,
    output: OutputArg,
    bvals: BValsOpt = None,
    bvecs: BVecsOpt = None,
    bvec_frame: BvecFrameOpt = BvecFrame.fsl,
    mask: MaskArg = None,
    force: ForceOpt = False,
    jobs: ThreadsOpt = -1,
) -> None:
    """Fit NODDI (neurite density and orientation dispersion) and write its maps."""
    # Imported here, so that starting lar and Tab completion do not load scipy.
    from larmor.dwi import NoddiModel

    prefix = output_prefix(output)
    outputs = {name: with_suffix(prefix, f"_{name}.nii.gz") for name in METRICS}
    check_outputs(list(outputs.values()), force)

    image, gtab = load_dwi(volume, bvals=bvals, bvecs=bvecs, frame=bvec_frame.value)
    mask_volume = load_volume(mask) if mask is not None else None
    noddi_fit = NoddiModel().fit(
        image,
        gtab,
        mask_volume,
        n_jobs=jobs,
        progress=LarmorLogger.isEnabledFor(logging.WARNING),
    )

    for name, path in outputs.items():
        save_volume(image.with_data(getattr(noddi_fit, name)), path)
        LarmorLogger.info(f"Saved {name.upper()} to {path}")
