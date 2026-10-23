"""Loading a diffusion-weighted volume together with its gradient table."""

from pathlib import Path
from typing import Literal

from larmor._log import LarmorLogger
from larmor.io.bvals_bvecs import read_bvals_bvecs
from larmor.io.formats import read_mif_gradients
from larmor.io.formats.mif import has_mif_gradients
from larmor.io.load_save import VOLUME_EXTENSIONS, load_volume
from larmor.types import GradientTable, Volume


def load_dwi(
    path: str | Path,
    bvals: str | Path | None = None,
    bvecs: str | Path | None = None,
    *,
    frame: Literal["fsl", "world"] = "fsl",
) -> tuple[Volume, GradientTable]:
    """Loads a diffusion-weighted volume and its gradient table, ready for fitting.

    A MIF file can store the gradient table in its header (`dw_scheme`), and then
    no other file is needed. A NIfTI file, or a MIF file without a table, needs
    the bvals and bvecs files, which are never looked for by name. Files given
    for a MIF file that has a table override the table, with a warning.

    The arguments are checked in a fixed order: giving only one of the two files
    is reported first, then the volume is read, so a missing volume is reported
    before any problem with the gradients.

    Args:
        path: The volume file.
        bvals: The b-values file, or None.
        bvecs: The b-vectors file, in `frame`, or None. Give both files or neither.
        frame: The frame of the bvecs file: "fsl" (converted with the volume file's
            affine) or "world". A MIF header is always in world coordinates.

    Returns:
        The volume as read, and its gradient table in world coordinates.

    Raises:
        FileNotFoundError: If the volume or a gradient file does not exist.
        ValueError: If only one of the two files is given, there is no gradient
            table (no files, and no table in the header), a file cannot be read, or
            the table does not have one entry per volume.
    """
    p = Path(path)
    if (bvals is None) != (bvecs is None):
        raise ValueError("Give both the bvals and the bvecs file, or neither.")

    # Read the volume first, so that its errors come before those of the gradients.
    volume = load_volume(p)
    in_header = p.name.endswith(VOLUME_EXTENSIONS["mif"]) and has_mif_gradients(p)
    if bvals is not None and bvecs is not None:
        if in_header:
            LarmorLogger.warning(
                "%s has a gradient table in its header; the bvals and bvecs files "
                "given override it.",
                p,
            )
        gtab = read_bvals_bvecs(bvals, bvecs, ref_path=p, frame=frame)
    elif in_header:
        gtab = read_mif_gradients(p)
    else:
        raise ValueError(
            f"{p} has no gradient table: give the bvals and bvecs files (a NIfTI "
            "file never holds one, a MIF file only in its header)."
        )
    return volume, gtab
