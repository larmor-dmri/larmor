"""Public example datasets, with utility functions to download them."""

from larmor.data.datasets import (
    fetch_cfin_multishell,
    fetch_ismrm2015,
    fetch_stanford_hardi,
)
from larmor.data.fetch import data_folder

__all__ = [
    "data_folder",
    "fetch_cfin_multishell",
    "fetch_ismrm2015",
    "fetch_stanford_hardi",
]
