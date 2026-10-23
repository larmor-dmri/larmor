"""Writing a tractogram batch by batch
===================================

This example writes a tractogram in three batches, reads it back, checks that the
file is the same as writing all the streamlines at once, and shows the three batches
in the viewer.

A whole-brain tractography run can produce millions of streamlines, far more than
fit in memory at once. `open_tractogram_writer` writes a TCK or TRK file in pieces:
each batch of streamlines is appended as soon as it exists, and only that batch is
kept in memory. The streamline count goes into the header when the file is finished.
"""

import tempfile
from pathlib import Path

import numpy as np

from larmor.io import load_tractogram, open_tractogram_writer, save_tractogram
from larmor.types import Tractogram
from larmor.viewer import (
    CameraPreset,
    Category,
    Colouring,
    OptionName,
    Viewer,
    ViewKind,
)

# %%
# Batches of streamlines
# ----------------------
# Each batch is a `Tractogram`: the points of its streamlines one after the other,
# and the offsets where each streamline starts. Here there are three batches of
# gently curving streamlines. Each batch is a bundle of its own, as the seeds of one
# batch would give.

rng = np.random.default_rng(0)


def random_batch(num_streamlines: int, direction: np.ndarray) -> Tractogram:
    """Streamlines of 1 mm steps along a direction, each bending its own way."""
    lines = []
    for _ in range(num_streamlines):
        n = int(rng.integers(30, 60))
        t = np.arange(n, dtype=np.float32)[:, None]
        start = rng.normal(0.0, 3.0, 3) - direction * n / 2
        bend = rng.normal(0.0, 0.0015, 3)
        lines.append((start + t * direction + t**2 * bend).astype(np.float32))
    offsets = np.concatenate([[0], np.cumsum([len(line) for line in lines])])
    return Tractogram(np.concatenate(lines), offsets)


x, y, z = np.eye(3)
batches = [random_batch(n, d) for n, d in ((100, x), (250, y), (50, z))]

# %%
# Writing in batches
# ------------------
# The writer is a context manager: leaving the block finishes the file. If the block
# raises, as an interrupted run does, the file keeps a count that marks it as
# incomplete, and reading it gives a warning.

folder = Path(tempfile.mkdtemp())
with open_tractogram_writer(folder / "batched.tck") as writer:
    for batch in batches:
        writer.write(batch)

tractogram = load_tractogram(folder / "batched.tck")
print(f"streamlines written: {tractogram.num_streamlines}")
assert tractogram.num_streamlines == 400

# %%
# The same file as writing everything at once
# -------------------------------------------
# For comparison, the streamlines of all batches are joined and written at once with
# `save_tractogram`.

points = np.concatenate([b.points for b in batches])
lengths = np.concatenate([np.diff(b.offsets) for b in batches])
whole = Tractogram(points, np.concatenate([[0], np.cumsum(lengths)]))
save_tractogram(whole, folder / "whole.tck")

same = (folder / "batched.tck").read_bytes() == (folder / "whole.tck").read_bytes()
print(f"identical files: {same}")
assert same

# %%
# The batches in the viewer
# -------------------------
# A tractogram can be indexed with a slice, here to take each batch back out of the
# file, and the viewer gives each tractogram its own colour.

viewer = Viewer()
start = 0
for n, batch in enumerate(batches):
    end = start + batch.num_streamlines
    viewer.add(tractogram[start:end], Category.TRACTOGRAMS, name=f"batch {n + 1}")
    start = end
viewer.set(Category.TRACTOGRAMS, OptionName.COLOURING, Colouring.PER_TRACTOGRAM)
viewer.set_camera(ViewKind.THREE_D, CameraPreset.CORONAL)
viewer.rotate_camera(35.0, 25.0)
viewer.zoom(ViewKind.THREE_D, 0.7)
viewer.screenshot(ViewKind.THREE_D, "batches_3d.png", size=(500, 500), overlay=False)
