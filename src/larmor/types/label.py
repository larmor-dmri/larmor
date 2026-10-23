"""A label of a parcellation: its name and colour."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Label:
    """One entry of a lookup table, with a label's name and colour.

    Attributes:
        name: The label's name.
        rgb: The colour as (red, green, blue), each from 0 to 255.
    """

    name: str
    rgb: tuple[int, int, int]
