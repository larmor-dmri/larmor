"""Crops the asymmetric ISMRM FOD to the block the peaks test reads.

The FOD comes from make_asym_fod.py (scilpy's asymmetric filtering of MRIToolkit's
fods_mrtrix.nii.gz). Run with Larmor's environment:
uv run python make_asym_ref.py SOURCE
"""

import sys
from pathlib import Path

import numpy as np

from larmor.io import load_volume, save_volume
from larmor.types import Volume

BLOCK = (slice(32, 36), slice(46, 50), slice(47, 49))

source = load_volume(sys.argv[1])
offset = np.eye(4)
offset[:3, 3] = [s.start for s in BLOCK]
block = np.ascontiguousarray(source.data[BLOCK], dtype=np.float32)
save_volume(
    Volume(block, source.affine @ offset), Path(__file__).parent / "fod_block.nii.gz"
)
