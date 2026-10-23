"""Structural connectomes
======================

This example builds a parcellation of four regions and streamlines between them,
assigns the streamlines' endpoints to regions, builds connectome matrices and saves
one with the region names. `lar connectome` does the same from files.

A structural connectome counts, for every pair of regions of a parcellation, the
streamlines that end in both (Hagmann et al. 2008). Each streamline's two endpoints
are first assigned to regions. The matrix then holds the number of streamlines
between each pair, the sum of their weights, or the mean of a value such as their
length.
"""

import tempfile
from pathlib import Path

import numpy as np

from larmor.tractography import (
    build_connectome,
    endpoint_nodes,
    streamline_lengths,
    write_connectome,
)
from larmor.types import Tractogram, Volume

# %%
# Regions and streamlines
# -----------------------
# The parcellation is a 1 mm image with four 6 mm cubes, labelled 1 to 4, at the
# corners of a square. The voxels between them are 0, which means no region. Voxel
# (i, j, k) is at world position (i, j, k) mm.

labels = np.zeros((30, 30, 6), np.uint8)
corners = {1: (2, 2), 2: (22, 2), 3: (2, 22), 4: (22, 22)}
for label, (x, y) in corners.items():
    labels[x : x + 6, y : y + 6, :] = label
parcellation = Volume(labels, np.eye(4))
names = ["frontal", "parietal", "temporal", "occipital"]

# Straight streamlines between region centres, some ending just short of a region.
rng = np.random.default_rng(2)
pairs = [(1, 2)] * 30 + [(1, 3)] * 20 + [(2, 4)] * 10 + [(3, 4)] * 5 + [(1, 4)] * 5
streamlines = []
for a, b in pairs:
    start = np.array([*corners[a], 0]) + 2.5 + rng.uniform(-2, 2, 3)
    end = np.array([*corners[b], 0]) + 2.5 + rng.uniform(-2, 2, 3)
    # Pull the end 0 to 4 mm back from the region along the streamline.
    direction = (end - start) / np.linalg.norm(end - start)
    end = end - rng.uniform(0, 4) * direction
    n = int(np.linalg.norm(end - start)) + 1
    streamlines.append(np.linspace(start, end, n).astype(np.float32))
offsets = np.concatenate([[0], np.cumsum([len(s) for s in streamlines])])
tracks = Tractogram(np.concatenate(streamlines), offsets)

# %%
# Assigning the endpoints
# -----------------------
# With a radius, which turns on the radial search, an endpoint outside every region
# takes the region of the nearest labelled voxel within that distance; the default
# radius is 4 mm. With `radius=None`, only the voxel the streamline ends in counts.

nodes = endpoint_nodes(tracks, parcellation, radius=4.0)
only_end_voxels = endpoint_nodes(tracks, parcellation, radius=None)
print(
    f"assigned at both ends: {np.all(nodes > 0, axis=1).sum()} with the radial "
    f"search, {np.all(only_end_voxels > 0, axis=1).sum()} with the end voxels only"
)

# %%
# The matrices
# ------------
# By default the matrix holds counts in its upper triangle; `symmetric` fills the
# lower one too. With `values`, each entry is the mean value of its streamlines.

counts = build_connectome(nodes, 4, symmetric=True)
lengths = build_connectome(nodes, 4, values=streamline_lengths(tracks), symmetric=True)
print("streamlines between regions:")
print(counts)
print(f"mean length frontal-parietal: {lengths[0, 1]:.1f} mm")
assert counts[0, 1] == 30 and counts[2, 3] == 5

# %%
# Saving with names
# -----------------
# A `.csv` file gets the names as its first row and column; read it back with
# pandas' `read_csv(path, index_col=0)` or with numpy as below.

path = Path(tempfile.mkdtemp()) / "connectome.csv"
write_connectome(counts, path, names=names)
print(path.read_text().splitlines()[0])
saved = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(1, 5))
assert np.array_equal(saved, counts)
