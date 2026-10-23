"""Reads the SH layout of an FOD file, with a warning for an asymmetric FOD."""

from pathlib import Path

from larmor._log import LarmorLogger


def read_layout(path: str | Path, num_coeffs: int) -> tuple[int, bool]:
    """Returns the lmax and whether all degrees are stored; warns if they are."""
    # Imported here: larmor.sd loads scipy, which `lar --help` avoids.
    from larmor.sd import sh_layout

    lmax, full = sh_layout(num_coeffs)
    if full:
        LarmorLogger.warning(
            f"{path} has {num_coeffs} coefficients: read as an asymmetric FOD (all "
            f"degrees, lmax {lmax})."
        )
    return lmax, full
