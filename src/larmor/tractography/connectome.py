"""Structural connectomes, the streamlines between the regions of a parcellation."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from larmor._threads import n_threads
from larmor.types import Label, Tractogram, Volume


def parcellation_labels(parcellation: Volume) -> np.ndarray:
    """Returns the labels of a parcellation, (X, Y, Z) int64 with 0 for no region.

    The scaled values are rounded to the nearest integer.

    Args:
        parcellation: A 3D image of region labels (or 4D with a single volume).

    Returns:
        The labels, one per voxel.

    Raises:
        ValueError: If the image is not 3D (or 4D with one volume), or has a negative
            or non-finite value.
    """
    data = parcellation.data
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(
            "A parcellation must be 3D (or 4D with a single volume), got shape "
            f"{parcellation.data.shape}."
        )
    if (
        np.issubdtype(data.dtype, np.integer)
        and parcellation.slope == 1.0
        and parcellation.intercept == 0.0
    ):
        labels = data.astype(np.int64)
    else:
        values = data.astype(np.float64) * parcellation.slope + parcellation.intercept
        if not np.all(np.isfinite(values)):
            raise ValueError("A parcellation's values must be finite.")
        labels = np.rint(values).astype(np.int64)
    if labels.size and labels.min() < 0:
        raise ValueError(
            f"A parcellation's labels must be 0 or more, got {labels.min()}."
        )
    return labels


def node_names(
    parcellation: Volume, *, lookup_table: Mapping[int, Label] | None = None
) -> list[str]:
    """Returns the names of the nodes of a connectome built on a parcellation.

    The nodes are the labels 1 to the largest label of the parcellation, so the
    length of the list is the number of nodes (`build_connectome`'s `num_nodes`).
    The matrix grows with the largest label, not with the number of regions, so a
    parcellation with sparse labels (such as FreeSurfer's, up to about 14,000)
    gives a row for every unused label unless it is relabelled first.

    Args:
        parcellation: A 3D image of integer region labels (see
            `parcellation_labels`).
        lookup_table: Names by label, as from `read_lookup_table`. Defaults to None,
            every node named by its label.

    Returns:
        One name per label from 1 to the largest; a label missing from the table is
        named by its number.

    Raises:
        ValueError: If the parcellation is not valid (see `parcellation_labels`).
    """
    num_nodes = int(parcellation_labels(parcellation).max(initial=0))
    table = lookup_table or {}
    return [
        table[label].name if label in table else str(label)
        for label in range(1, num_nodes + 1)
    ]


def endpoint_nodes(
    tractogram: Tractogram,
    parcellation: Volume,
    *,
    radius: float | None = 4.0,
    n_jobs: int = -1,
) -> np.ndarray:
    """Assigns the two endpoints of every streamline to regions of a parcellation.

    With `radius` None or 0, an endpoint's region is the label of its nearest voxel
    (0 outside the image). With a radius (the radial search), it is the label of the
    labelled voxel whose centre is nearest to the endpoint, if that centre is less
    than `radius` from it, and 0 if there is none. The radius applies to the
    endpoint's own voxel as well: with voxels larger than the radius, an endpoint
    near a voxel's corner may get no region although its own voxel is labelled.

    Args:
        tractogram: The streamlines, in world coordinates (mm).
        parcellation: A 3D image of integer region labels, 0 for no region (see
            `parcellation_labels`), on any grid.
        radius: The search radius in mm, or None to use only the voxel the endpoint
            is in. Defaults to 4.0.
        n_jobs: Number of threads for the radial search, at most the cores; -1 is
            all cores. Defaults to -1.

    Returns:
        (S, 2) int64: the regions of the first and the last point of each streamline,
        0 where an endpoint is in no region; both 0 for an empty streamline.

    Raises:
        ValueError: If `radius` is negative or not finite, or the parcellation is not
            valid (see `parcellation_labels`).

    References:
        Smith RE, Tournier JD, Calamante F, Connelly A. The effects of SIFT on the
        reproducibility and biological accuracy of the structural connectome.
        NeuroImage 104 (2015) 253-265 (the radial search). Follows MRtrix3's
        tck2connectome (version 3.0.8) with -assignment_radial_search and
        -assignment_end_voxels.
    """
    if radius is not None and not (np.isfinite(radius) and radius >= 0):
        raise ValueError(f"radius must be finite and not negative, got {radius}.")
    threads = n_threads(n_jobs)
    # Mapped in RAS axis order, so an endpoint halfway between two voxels goes to the
    # same one whatever the order of the image's axes.
    canonical = Volume(parcellation_labels(parcellation), parcellation.affine.copy())
    canonical.to_closest_canonical()
    labels = canonical.data
    nonempty = np.diff(tractogram.offsets) > 0
    first = tractogram.offsets[:-1][nonempty]
    last = tractogram.offsets[1:][nonempty] - 1
    ends = np.concatenate([tractogram.points[first], tractogram.points[last]])
    ends = ends.astype(np.float64)
    voxels, inside = _nearest_voxels(ends, canonical.affine, labels.shape)
    found = np.zeros(len(ends), dtype=np.int64)
    found[inside] = labels[tuple(voxels[inside].T)]
    if radius is not None and radius > 0:
        # The endpoint's own voxel is kept only when its centre is within the
        # radius; it wins over other voxels at the same distance.
        centres = voxels @ canonical.affine[:3, :3].T + canonical.affine[:3, 3]
        near = np.linalg.norm(ends - centres, axis=1) < radius
        found[~near] = 0
        missing = np.flatnonzero(found == 0)
        found[missing] = _nearest_labels(
            ends[missing], labels, canonical.affine, radius, threads
        )
    nodes = np.zeros((tractogram.num_streamlines, 2), dtype=np.int64)
    count = len(first)
    nodes[nonempty, 0] = found[:count]
    nodes[nonempty, 1] = found[count:]
    return nodes


def _nearest_voxels(
    points: np.ndarray, affine: np.ndarray, shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the (N, 3) voxels nearest to the points and (N,) whether each is inside.

    `affine` and `shape` are those of the image; a point outside gets voxel 0.
    """
    to_voxel = np.linalg.inv(affine)
    coordinates = points @ to_voxel[:3, :3].T + to_voxel[:3, 3]
    # NaN compares False, so it is outside.
    inside = np.all(
        (coordinates > -0.5) & (coordinates < np.array(shape) - 0.5), axis=1
    )
    coordinates[~inside] = 0.0
    # Halves away from zero; inside the image every coordinate is above -0.5.
    voxels = np.floor(np.abs(coordinates) + 0.5).astype(np.int64)
    return voxels, inside


def _nearest_labels(
    points: np.ndarray,
    labels: np.ndarray,
    affine: np.ndarray,
    radius: float,
    threads: int,
) -> np.ndarray:
    """Returns the label of the labelled voxel centre nearest to each point.

    A point with no labelled centre closer than `radius` gets 0.
    """
    result = np.zeros(len(points), dtype=np.int64)
    labelled = np.argwhere(labels > 0)
    if len(points) == 0 or len(labelled) == 0:
        return result
    centres = labelled @ affine[:3, :3].T + affine[:3, 3]
    tree = cKDTree(centres)
    distances, indices = tree.query(
        points, k=1, distance_upper_bound=radius, workers=threads
    )
    near = distances < radius
    result[near] = labels[tuple(labelled[indices[near]].T)]
    return result


def build_connectome(
    nodes: np.ndarray,
    num_nodes: int,
    *,
    weights: np.ndarray | None = None,
    values: np.ndarray | None = None,
    symmetric: bool = False,
    zero_diagonal: bool = False,
) -> np.ndarray:
    """Builds a connectome matrix from the regions of the streamlines' endpoints.

    Every streamline whose two endpoints are both in a region (nodes 1 to
    `num_nodes`) adds to the edge between its two regions, an entry of the upper
    triangle (row <= column); one that ends twice in the same region adds to the
    diagonal. Streamlines with an endpoint in no region (node 0) are left out.
    Without `values`, an edge holds the number of its streamlines, or the sum of
    their weights. With `values`, it holds their mean value, weighted by the weights
    if given, and 0 where no streamline connects the two regions.

    Args:
        nodes: (S, 2) the regions of each streamline's endpoints, as from
            `endpoint_nodes`.
        num_nodes: The number of regions, with one row per label from 1 to
            `num_nodes`; usually the largest label of the parcellation
            (`len(node_names(parcellation))`). Sparse labels give a row for every
            unused label.
        weights: (S,) one weight per streamline, such as SIFT2's. Defaults to None,
            every streamline counting 1.
        values: (S,) one value per streamline, such as its length
            (`streamline_lengths`), to average over each edge. Defaults to None.
        symmetric: Whether to fill the lower triangle with the upper one. Defaults
            to False, which leaves the lower triangle 0.
        zero_diagonal: Whether to set the diagonal (connections of a region to
            itself) to 0. Defaults to False.

    Returns:
        (num_nodes, num_nodes) float64; row and column i are region i + 1.

    Raises:
        ValueError: If `nodes` is not (S, 2) with labels from 0 to `num_nodes`, or
            `weights` or `values` does not have S finite values.

    References:
        Hagmann P, Cammoun L, Gigandet X, Meuli R, Honey CJ, Wedeen VJ, Sporns O.
        Mapping the structural core of human cerebral cortex. PLoS Biol 6 (2008)
        e159. Follows MRtrix3's tck2connectome (version 3.0.8) with the default
        -stat_edge sum, and -stat_edge mean with -scale_length or -scale_file.
    """
    nodes = np.asarray(nodes)
    if (
        nodes.ndim != 2
        or nodes.shape[1] != 2
        or not np.issubdtype(nodes.dtype, np.integer)
    ):
        raise ValueError(f"nodes must be an (S, 2) integer array, got {nodes.shape}.")
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be 0 or more, got {num_nodes}.")
    if nodes.size and (nodes.min() < 0 or nodes.max() > num_nodes):
        raise ValueError(
            f"nodes must be labels from 0 to num_nodes ({num_nodes}), got "
            f"{nodes.min()} to {nodes.max()}."
        )
    count = len(nodes)
    weight = _per_streamline(weights, count, "weights")
    value = _per_streamline(values, count, "values")
    assigned = np.all(nodes > 0, axis=1)
    row = nodes[assigned].min(axis=1) - 1
    column = nodes[assigned].max(axis=1) - 1
    edge = row * num_nodes + column
    size = num_nodes * num_nodes
    weight = np.ones(count) if weight is None else weight
    weight = weight[assigned]
    if value is None:
        flat = np.bincount(edge, weights=weight, minlength=size)
    else:
        total = np.bincount(edge, weights=weight * value[assigned], minlength=size)
        weight_sum = np.bincount(edge, weights=weight, minlength=size)
        flat = np.divide(total, weight_sum, out=np.zeros(size), where=weight_sum != 0)
    matrix: np.ndarray = flat.reshape(num_nodes, num_nodes)
    if symmetric:
        matrix = matrix + np.triu(matrix, 1).T
    if zero_diagonal:
        np.fill_diagonal(matrix, 0.0)
    return matrix


def _per_streamline(
    values: np.ndarray | None, count: int, name: str
) -> np.ndarray | None:
    """Returns `values` as (S,) float64, or None when they are None.

    Raises:
        ValueError: If there is not one finite value per streamline (`count`).
    """
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (count,) or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must have one finite value per streamline ({count}), got shape "
            f"{array.shape}."
        )
    return array


def write_connectome(
    matrix: np.ndarray, path: str | Path, *, names: Sequence[str] | None = None
) -> None:
    """Writes a connectome matrix as text, with the region names.

    A `.csv` file is comma-separated, with the names as its first row and first
    column and an empty first cell. Any other extension gives a matrix separated by
    spaces after one comment line `# names: <name> <name> ...`. Values are written
    with 10 significant digits.

    Args:
        matrix: (N, N) the connectome.
        path: The output file.
        names: N region names, without whitespace or commas, such as those of
            `read_lookup_table`. Defaults to None, the labels 1 to N.

    Raises:
        ValueError: If `matrix` is not square or `names` does not have one name per
            row without whitespace or commas.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"A connectome must be a square matrix, got {matrix.shape}.")
    count = matrix.shape[0]
    labels = [str(i) for i in range(1, count + 1)] if names is None else list(names)
    if len(labels) != count:
        raise ValueError(f"Expected {count} region names, got {len(labels)}.")
    if any(not name or "," in name or len(name.split()) != 1 for name in labels):
        raise ValueError(
            "Region names must be non-empty, without whitespace or commas."
        )
    rows = [[f"{v:.10g}" for v in row] for row in matrix.tolist()]
    if Path(path).suffix.lower() == ".csv":
        lines = [",".join(["", *labels])]
        lines += [
            ",".join([name, *row]) for name, row in zip(labels, rows, strict=True)
        ]
    else:
        lines = ["# names: " + " ".join(labels)]
        lines += [" ".join(row) for row in rows]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_node_assignments(nodes: np.ndarray, path: str | Path) -> None:
    """Writes the regions of each streamline's endpoints, one streamline per line.

    Each line holds the two labels separated by a space, in the order of the
    streamlines.

    Args:
        nodes: (S, 2) integer labels, as from `endpoint_nodes`.
        path: The output text file.

    Raises:
        ValueError: If `nodes` is not an (S, 2) integer array.
    """
    nodes = np.asarray(nodes)
    if (
        nodes.ndim != 2
        or nodes.shape[1] != 2
        or not np.issubdtype(nodes.dtype, np.integer)
    ):
        raise ValueError(f"nodes must be an (S, 2) integer array, got {nodes.shape}.")
    Path(path).write_text(
        "".join(f"{a} {b}\n" for a, b in nodes.tolist()), encoding="utf-8", newline="\n"
    )
