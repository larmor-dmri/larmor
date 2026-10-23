"""The `lar mask` command, which computes a brain mask from a DWI."""

import typing
from pathlib import Path

import numpy as np
import typer

from larmor.cli.options import (
    BValsOpt,
    BvecFrame,
    BvecFrameOpt,
    BVecsOpt,
    DiffusionArg,
    ForceOpt,
)
from larmor.cli.outputs import check_outputs
from larmor.io import load_dwi, save_volume

OutArg = typing.Annotated[str, typer.Argument(help="Path of the mask image to write.")]


def mask(
    volume: DiffusionArg,
    output: OutArg,
    bvals: BValsOpt = None,
    bvecs: BVecsOpt = None,
    bvec_frame: BvecFrameOpt = BvecFrame.fsl,
    force: ForceOpt = False,
) -> None:
    """Compute a binary brain mask from the mean b = 0 image of a DWI volume."""
    # Imported here, so that starting lar and Tab completion do not load scipy.
    from larmor.dwi import create_brain_mask, mean_b0

    out_path = Path(output)
    check_outputs([out_path], force)

    image, gtab = load_dwi(volume, bvals=bvals, bvecs=bvecs, frame=bvec_frame.value)
    brain_mask = create_brain_mask(mean_b0(image, gtab))
    save_volume(image.with_data(brain_mask.astype(np.uint8)), out_path)
