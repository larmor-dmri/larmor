"""Phantom files that ship with Larmor."""

from pathlib import Path
from typing import Final

from larmor.simulation.phantom_file import PhantomFile, read_phantom

_FOLDER: Final = Path(__file__).parent / "phantoms"

#: The names of the bundled phantoms.
EXAMPLE_PHANTOMS: Final = ("crossing",)


def example_phantom(name: str = "crossing") -> PhantomFile:
    """Reads a phantom that ships with Larmor, with its scan.

    The names are in `EXAMPLE_PHANTOMS`. "crossing" is two straight bundles of
    34 mm crossing at 90 degrees, one along x and one along y, with different
    standard model parameters and T2, on a 40 x 40 x 40 grid of 1 mm voxels. It
    is scanned with b = 1000 and 2000 s/mm^2 (32 directions each) and two b = 0
    volumes.

    Args:
        name: The phantom's name. Defaults to "crossing".

    Returns:
        The phantom and its protocol.

    Raises:
        ValueError: If there is no bundled phantom of that name.
    """
    if name not in EXAMPLE_PHANTOMS:
        raise ValueError(
            f"There is no example phantom {name!r}; the examples are "
            f"{', '.join(EXAMPLE_PHANTOMS)}."
        )
    return read_phantom(_FOLDER / f"{name}.phantom.json")
