"""The white matter response of CSD, estimated from the data's single-fibre voxels."""

import functools
from dataclasses import dataclass
from math import sqrt

import numpy as np
import numpy.typing as npt
from scipy.ndimage import binary_dilation
from scipy.optimize import nnls
from scipy.spatial import ConvexHull
from scipy.special import eval_legendre, lpmv

from larmor import _core
from larmor._log import LarmorLogger
from larmor._threads import n_threads
from larmor.dwi.inputs import (
    as_mask,
    check_dwi,
    finite_rows,
    signal_chunks,
    signal_rows,
)
from larmor.sd.csd_solver import (
    CSDSolver,
    csd_problem,
    csd_shell,
    solve_csd,
    warn_unconverged,
)
from larmor.sd.rf import ZonalSHRF
from larmor.sd.sh import build_basis_matrix
from larmor.sd.sphere import get_theta_phi
from larmor.sd.tissue_responses import build_shell_response
from larmor.types import GradientTable, Volume

# The broad response the first iteration deconvolves with, as zonal SH l = 0, 2, 4.
INITIAL_RESPONSE = np.array([1.0, -1.0, 1.0])
# Highest degree of the response fitted in the first iteration.
INITIAL_LMAX = 4
# Highest degree of the FODs that select the voxels.
MAX_FOD_LMAX = 8


def build_csd_response(coefficients: npt.ArrayLike, gtab: GradientTable) -> ZonalSHRF:
    """Turns the rows of an MRtrix3 response file into a response for `CSDModel`.

    A response file holds only zonal SH coefficients, one row per shell, without the
    b-values. This function gives each row the b-value of a shell of the data, so the
    response can be passed to `CSDModel`. A single row is taken as the response on
    the shell CSD fits, the one with the highest b-value. Several rows are taken as
    one per shell in increasing b-value, as dwi2response writes them, with b0 first
    when the data have b0s.

    Args:
        coefficients: The response, one row per shell, as `read_mrtrix_response`
            returns it.
        gtab: The gradient table of the data the response belongs to.

    Returns:
        ZonalSHRF: The response with the b-value of each row.

    Raises:
        ValueError: If there are several rows and not one per shell of the data.
    """
    rows = np.atleast_2d(np.asarray(coefficients, dtype=np.float64))
    if len(rows) == 1:
        return ZonalSHRF(bvals=np.array([csd_shell(gtab)]), coefficients=rows)
    return build_shell_response(rows, gtab)


@dataclass(frozen=True)
class SingleFibreRF:
    """Holds the white matter response `estimate_csd_rf` returns and its voxels.

    Attributes:
        rf: The response on the shell CSD fits (the one with the highest b-value),
            as zonal SH coefficients.
        voxels: The single-fibre voxels of the last iteration, shape (X, Y, Z),
            bool.
        num_iter: Number of iterations run.
        converged: Whether the selection of voxels stopped changing; when False the
            estimate is that of the last iteration.
    """

    rf: ZonalSHRF
    voxels: np.ndarray
    num_iter: int
    converged: bool


@functools.cache
def _sphere() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Returns 2562 unit vectors of a four times subdivided icosahedron.

    Returns:
        The vertices, shape (2562, 3), and the indices of each vertex's 5 or 6
        neighbours, padded with its own index to 6, shape (2562, 6).
    """
    t = (1 + sqrt(5)) / 2
    corners = [
        np.roll((0.0, a, b * t), shift)
        for shift in range(3)
        for a in (1.0, -1.0)
        for b in (1.0, -1.0)
    ]
    pts = np.array(corners) / np.sqrt(1 + t**2)
    for _ in range(4):
        edges = {
            (min(i, j), max(i, j))
            for a, b, c in ConvexHull(pts).simplices
            for i, j in ((a, b), (b, c), (a, c))
        }
        mids = np.array([pts[i] + pts[j] for i, j in sorted(edges)])
        pts = np.vstack([pts, mids / np.linalg.norm(mids, axis=1, keepdims=True)])
    neighbours: list[set[int]] = [set() for _ in range(len(pts))]
    for a, b, c in ConvexHull(pts).simplices:
        neighbours[a] |= {int(b), int(c)}
        neighbours[b] |= {int(a), int(c)}
        neighbours[c] |= {int(a), int(b)}
    table = np.array(
        [sorted(n) + [i] * (6 - len(n)) for i, n in enumerate(neighbours)],
        dtype=np.int64,
    )
    return pts, table


def _two_largest_peaks(fods: np.ndarray, lmax: int, threads: int) -> np.ndarray:
    """Returns the amplitudes of each FOD's two largest peaks, 0 where it has fewer.

    The peaks are refined from the local maxima of the amplitudes on `_sphere`. An
    FOD whose l = 0 coefficient is not positive has none.

    Returns:
        Shape (n_voxels, 2), largest first.
    """
    vertices, neighbours = _sphere()
    theta, phi = get_theta_phi(vertices)
    on_sphere = np.ascontiguousarray(fods @ build_basis_matrix(theta, phi, lmax).T)
    _, amplitudes = _core.peaks_chunk(
        np.ascontiguousarray(fods),
        lmax,
        on_sphere,
        vertices,
        neighbours,
        2,
        0.0,
        0.0,
        threads,
    )
    peaks = np.nan_to_num(amplitudes, nan=0.0)
    peaks[fods[:, 0] <= 0] = 0.0
    return peaks


@functools.cache
def _hemisphere() -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    """Returns one direction of each antipodal pair of `_sphere`, with neighbours.

    Returns:
        The 1281 directions, shape (1281, 3); the neighbours of each, a direction
        next to another's opposite counting as its neighbour; and integration weights,
        shape (1281,), the least-squares weights that integrate every even SH up to
        degree 50 exactly over the sphere.
    """
    vertices, _ = _sphere()
    keep: list[int] = []
    seen: set[tuple[float, ...]] = set()
    for index, vertex in enumerate(np.round(vertices, 9)):
        if tuple(-vertex) not in seen:
            seen.add(tuple(vertex))
            keep.append(index)
    dirs = vertices[keep]
    n = len(dirs)
    neighbours: list[set[int]] = [set() for _ in range(n)]
    for triangle in ConvexHull(np.vstack([dirs, -dirs])).simplices:
        corners = {int(i) % n for i in triangle}
        for corner in corners:
            neighbours[corner] |= corners - {corner}
    theta, phi = get_theta_phi(dirs)
    integrals = build_basis_matrix(theta, phi, 50).T
    target = np.zeros(integrals.shape[0])
    # The integral of Y_00 over the sphere; every other one is 0.
    target[0] = 2 * sqrt(np.pi)
    weights = np.linalg.lstsq(integrals, target, rcond=None)[0]
    return dirs, [sorted(s) for s in neighbours], weights


def _first_lobe_direction(fod: np.ndarray, lmax: int) -> np.ndarray:
    """Returns the mean direction of the FOD's lobe around its largest amplitude.

    The positive amplitudes on the sphere are grouped into lobes from the largest
    down. The direction is the mean of the lobe's directions, each turned to the
    lobe's side and weighted by its amplitude and integration weight.

    Returns:
        A unit vector, shape (3,).
    """
    dirs, neighbours, weights = _hemisphere()
    theta, phi = get_theta_phi(dirs)
    amplitudes = build_basis_matrix(theta, phi, lmax) @ fod
    label = np.full(len(dirs), -1)
    means: list[np.ndarray] = []
    between: list[tuple[int, int]] = []

    def add(lobe: int, index: int) -> None:
        side = 1.0 if means[lobe] @ dirs[index] > 0 else -1.0
        means[lobe] = (
            means[lobe] + side * amplitudes[index] * weights[index] * dirs[index]
        )

    order = np.argsort(-amplitudes, kind="stable")
    for index in order[amplitudes[order] > 0]:
        adjacent = sorted({int(label[j]) for j in neighbours[index] if label[j] >= 0})
        if not adjacent:
            label[index] = len(means)
            means.append(amplitudes[index] * weights[index] * dirs[index])
        elif len(adjacent) == 1:
            label[index] = adjacent[0]
            add(adjacent[0], int(index))
        else:
            between.append((int(index), adjacent[0]))
    for index, lobe in between:
        add(lobe, index)
    direction: np.ndarray = means[0] / np.linalg.norm(means[0])
    return direction


def _zonal_basis(cosines: np.ndarray, lmax: int) -> np.ndarray:
    """Returns the zonal SH sqrt((2l + 1) / 4 pi) P_l(cos t), (n, lmax / 2 + 1)."""
    degrees = np.arange(0, lmax + 1, 2)
    return np.stack(
        [
            np.sqrt((2 * d + 1) / (4 * np.pi)) * eval_legendre(d, cosines)
            for d in degrees
        ],
        axis=-1,
    )


def _fit_zonal_response(
    signals: np.ndarray, bvecs: np.ndarray, fibres: np.ndarray, lmax: int
) -> np.ndarray:
    """Fits a zonal SH response to the signals of single-fibre voxels.

    Each voxel's measurements are placed at their angle to its fibre, and one set of
    coefficients is fitted to all of them by least squares, constrained so that the
    response is non-negative and does not decrease from the fibre to the plane across
    it (checked every degree from 0 to 90).

    Args:
        signals: The voxels' signals on the shell, shape (n_voxels, n_meas).
        bvecs: Unit gradient directions, shape (n_meas, 3).
        fibres: Unit fibre direction of each voxel, shape (n_voxels, 3).
        lmax: Highest degree of the response.

    Returns:
        The coefficients of l = 0, 2, ..., lmax.
    """
    n = lmax // 2 + 1
    gram = np.zeros((n, n))
    projected = np.zeros(n)
    for signal, fibre in zip(signals, fibres, strict=True):
        design = _zonal_basis(np.abs(bvecs @ fibre), lmax)
        gram += design.T @ design
        projected += design.T @ signal
    angles = np.radians(np.arange(91))
    degrees = np.arange(0, lmax + 1, 2)
    # d/dt of P_l(cos t) is P_l^1(cos t) with scipy's Condon-Shortley phase.
    slope = np.stack(
        [
            np.sqrt((2 * d + 1) / (4 * np.pi)) * lpmv(1, d, np.cos(angles))
            for d in degrees
        ],
        axis=-1,
    )
    constraints = np.vstack([_zonal_basis(np.cos(angles), lmax), slope])
    return _constrained_least_squares(gram, projected, constraints)


def _constrained_least_squares(
    gram: np.ndarray, projected: np.ndarray, constraints: np.ndarray
) -> np.ndarray:
    """Minimises x^T G x - 2 p^T x subject to C x >= 0, G positive definite.

    With G = L L^T and y = L^T x, this is the projection of z = L^-1 p onto the cone
    C L^-T y >= 0, solved as a least-distance problem through NNLS.

    References:
        Lawson CL, Hanson RJ. Solving Least Squares Problems. Prentice-Hall, 1974,
        chapter 23 (least distance programming).

    Returns:
        The solution x.
    """
    lower = np.linalg.cholesky(gram)
    z = np.linalg.solve(lower, projected)
    cone = np.linalg.solve(lower, constraints.T).T
    # Least distance: min ||w|| subject to cone (z + w) >= 0.
    n = len(z)
    stacked = np.vstack([cone.T, -(cone @ z)[None]])
    target = np.zeros(n + 1)
    target[-1] = 1.0
    weights, _ = nnls(stacked, target, maxiter=100 * stacked.shape[1])
    residual = stacked @ weights - target
    w = -residual[:n] / residual[n]
    solution: np.ndarray = np.linalg.solve(lower.T, z + w)
    return solution


def _top(cost: np.ndarray, count: int) -> np.ndarray:
    """Returns the voxels with the `count` largest finite costs, ties included."""
    finite = cost[np.isfinite(cost)]
    if len(finite) == 0:
        return np.zeros(cost.shape, dtype=bool)
    threshold = np.sort(finite)[-min(count, len(finite))]
    selected: np.ndarray = np.isfinite(cost) & (cost >= threshold)
    return selected


def _check_settings(
    num_voxels: int, candidates_count: int, dilate: int, max_iter: int, lmax: int
) -> None:
    """Checks the settings of `estimate_csd_rf`.

    Raises:
        ValueError: If a setting is out of range.
    """
    if num_voxels < 1 or candidates_count < num_voxels:
        raise ValueError(
            "num_voxels must be at least 1 and num_candidates at least num_voxels, "
            f"got {num_voxels} and {candidates_count}"
        )
    if dilate < 0 or max_iter < 2 or lmax < 2 or lmax % 2:
        raise ValueError(
            "dilate must be >= 0, max_iter >= 2 and lmax even and >= 2, got "
            f"{dilate}, {max_iter} and {lmax}"
        )


def estimate_csd_rf(
    volume: Volume,
    gtab: GradientTable,
    mask: Volume | np.ndarray,
    *,
    num_voxels: int = 300,
    num_candidates: int | None = None,
    dilate: int = 1,
    max_iter: int = 10,
    lmax: int = 10,
    n_jobs: int = -1,
) -> SingleFibreRF:
    """Estimates the white matter response of CSD from the data's single-fibre voxels.

    The response is the signal of a single fibre on the shell CSD fits, the one with
    the highest b-value. Each iteration deconvolves the candidate voxels, takes the
    num_voxels whose FODs have the most dominant single peak, and fits a new response
    to their signals, aligned with their fibres. The iterations stop when those
    voxels repeat.

    Args:
        volume: Diffusion-weighted volume, shape (X, Y, Z, N).
        gtab: Its gradient table.
        mask: The voxels to search (nonzero inside), on the volume's grid.
        num_voxels: Number of single-fibre voxels the response is fitted to.
            Defaults to 300.
        num_candidates: Number of best voxels the next iteration starts from, before
            growing. Defaults to None, which gives 10 * num_voxels.
        dilate: Voxels the candidates are grown by (6-connected) each iteration.
            Defaults to 1.
        max_iter: Maximum number of iterations, at least 2. Defaults to 10.
        lmax: Highest degree of the response, even. Defaults to 10.
        n_jobs: Number of threads; -1 uses every core. Defaults to -1.

    Returns:
        SingleFibreRF: The response on the highest shell, in the units of the signal,
        and the voxels it was fitted to.

    Raises:
        ValueError: If a setting is out of range, or the data have no
            diffusion-weighted shell.

    References:
        The defaults, the score and the fit of the response follow MRtrix3's
        dwi2response tournier and amp2response: Tournier JD, Smith R, Raffelt D, et
        al. MRtrix3: A fast, flexible and open software framework for medical image
        processing and visualisation. NeuroImage 202 (2019) 116137.
        doi:10.1016/j.neuroimage.2019.116137
    """
    candidates_count = 10 * num_voxels if num_candidates is None else num_candidates
    _check_settings(num_voxels, candidates_count, dilate, max_iter, lmax)
    threads = n_threads(n_jobs)
    check_dwi(volume, gtab)
    brain = as_mask(volume, mask)
    bval = csd_shell(gtab)
    on_shell = gtab.shell_index == len(gtab.shells) - 1
    bvecs = gtab.bvecs[on_shell]
    usable = np.zeros(brain.shape, dtype=bool)
    usable[brain] = finite_rows(volume, np.nonzero(brain), on_shell)

    response = INITIAL_RESPONSE
    candidates = brain & usable
    previous: np.ndarray | None = None
    selected = np.zeros(brain.shape, dtype=bool)
    converged = False
    for iteration in range(max_iter):
        fod_lmax = min(2 * (len(response) - 1), MAX_FOD_LMAX)
        fods = _candidate_fods(
            volume, candidates, on_shell, bvecs, response, fod_lmax, threads
        )
        peaks = _two_largest_peaks(fods, fod_lmax, threads)
        cost = np.full(brain.shape, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            cost[candidates] = (
                np.sqrt(peaks[:, 0]) * (1 - peaks[:, 1] / peaks[:, 0]) ** 2
            )
        selected = _top(cost, num_voxels)
        # The cost is finite only on candidates, and both masks list voxels in the
        # same order, so these are the selected voxels' FODs in the order of their
        # signals.
        fibres = np.array(
            [_first_lobe_direction(f, fod_lmax) for f in fods[selected[candidates]]]
        )
        response_lmax = INITIAL_LMAX if iteration == 0 else lmax
        selected_voxels = np.nonzero(selected)
        signals = signal_rows(volume, selected_voxels, slice(None), on_shell)
        response = _fit_zonal_response(signals, bvecs, fibres, response_lmax)
        LarmorLogger.info(
            f"Response iteration {iteration + 1}: {int(selected.sum())} voxels, "
            f"coefficients {np.array2string(response, precision=3)}"
        )
        if previous is not None and np.array_equal(selected, previous):
            converged = True
            break
        previous = selected
        candidates = _top(cost, candidates_count)
        if dilate:
            candidates = binary_dilation(candidates, iterations=dilate)
        candidates &= brain & usable
    if not converged:
        LarmorLogger.warning(
            f"The single-fibre voxels still changed after {max_iter} iterations; "
            "using the last estimate."
        )
    return SingleFibreRF(
        rf=ZonalSHRF(bvals=np.array([bval]), coefficients=response[None]),
        voxels=selected,
        num_iter=iteration + 1,
        converged=converged,
    )


def _candidate_fods(
    volume: Volume,
    candidates: np.ndarray,
    on_shell: np.ndarray,
    bvecs: np.ndarray,
    response: np.ndarray,
    lmax: int,
    threads: int,
) -> np.ndarray:
    """Returns the CSD FODs of the candidate voxels on the shell, read in chunks.

    Args:
        volume: The DWI.
        candidates: The voxels to fit, (X, Y, Z) bool.
        on_shell: The volumes of the shell, bool.
        bvecs: Their gradient directions, (n_meas, 3).
        response: The response's zonal SH coefficients on the shell.
        lmax: Highest degree of the FODs.
        threads: Number of threads.

    Returns:
        The FODs in the tournier basis, one row per candidate in mask order.
    """
    solver = CSDSolver()
    problem = csd_problem(bvecs, response, lmax, solver)
    voxels = np.nonzero(candidates)
    fods = np.empty((len(voxels[0]), problem.forward.shape[1]))
    unconverged = 0
    for rows, signals in signal_chunks(volume, voxels, on_shell):
        fods[rows], converged = solve_csd(signals, problem, threads)
        unconverged += int((~converged).sum())
    warn_unconverged(unconverged, solver.max_iter)
    return fods
