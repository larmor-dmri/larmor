"""
The compiled numerical code of Larmor.

Private: the public API in larmor.sd, larmor.dwi, larmor.tractography
and larmor.simulation checks the inputs and calls these functions.
"""

from collections.abc import Sequence
from typing import Annotated

import numpy
import numpy.typing
from numpy.typing import NDArray


def norms(v: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], n_threads: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
    """
    The norm of each row of a C-contiguous (n, 3) float64 array.

    Raises:
        ValueError: If the array is not C-contiguous, does not have 3 columns, or
            n_threads is 0.
    """

MAX_LMAX: int = 20

def peaks_chunk(coeffs: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], lmax: int, on_sphere: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], vertices: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], neighbours: Annotated[NDArray[numpy.int64], dict(shape=(None, None), device='cpu', writable=False)], n_peaks: int, relative_threshold: float, threshold: float, n_threads: int, full: bool = False) -> tuple[Annotated[NDArray[numpy.float64], dict(shape=(None, None, None))], Annotated[NDArray[numpy.float64], dict(shape=(None, None))]]:
    """
    Finds the peaks of every voxel of a chunk.

    coeffs is (n, C) in the tournier basis, with all degrees if full (an asymmetric
    FOD, whose peaks keep their sign and are not merged with their opposites).
    on_sphere (n, V) holds the amplitudes at the vertices (V, 3), and neighbours
    (V, 6) the 6 neighbours of each vertex. Returns directions (n, n_peaks, 3) and
    amplitudes (n, n_peaks), largest first, with z >= 0 unless full, and NaN where
    a voxel has fewer peaks.

    Raises:
        ValueError: If lmax is above 20, odd without full or does not match C, the
            shapes do not match, a neighbour is outside the sphere, n_peaks or
            n_threads is 0.
    """

def climb(sh: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu', writable=False)], lmax: int, start: tuple[float, float, float] | list[float] | numpy.typing.NDArray[numpy.float64], max_iters: int = 50, tol: float = 0.0001, max_step: float = 0.2, full: bool = False) -> tuple[Annotated[NDArray[numpy.float64], dict(shape=(None,))], float]:
    """
    Climbs from start to the peak whose basin it lies in.

    Returns the peak's direction (3,) and its amplitude, both NaN if the climb does
    not converge within max_iters.

    Raises:
        ValueError: If lmax is above 20, odd without full, or does not match sh.
    """

def sh_derivatives(sh: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu', writable=False)], lmax: int, theta: float, phi: float, full: bool = False) -> tuple[float, float, float, float, float, float]:
    """
    Returns the amplitude f of a series and its derivatives at (theta, phi).

    The values are f, df/dtheta, df/dphi / sin theta, d2f/dtheta2,
    d2f/dtheta dphi / sin theta and d2f/dphi2 / sin^2 theta.

    Raises:
        ValueError: If lmax is above 20, odd without full, or does not match sh.
    """

class FodSolver:
    """
    An FOD solver with the matrices it needs from one kernel, copied into C++.
    """

    @staticmethod
    def drl(k_t: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], k_t_k: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], nu: float, n_iter: int, init: float) -> FodSolver:
        """
        Makes a damped Richardson-Lucy solver, starting from init in every
        direction.

        k_t (c, m) is the kernel transposed and k_t_k (c, c) its product with the
        kernel.

        Raises:
            ValueError: If the shapes do not match.
        """

    @staticmethod
    def l2nnls(k: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], reg: float, tol_x: float) -> FodSolver:
        """
        Makes a non-negative least-squares solver with a ridge penalty on the FOD.

        k (m, c) is the weighted anisotropic kernel. Each voxel's FOD f minimises
        ||k f - s||^2 + reg^2 ||f||^2 with f >= 0, so reg = 0 is plain NNLS. A
        coefficient enters the solution only when its gradient exceeds tol_x.

        Raises:
            ValueError: If reg is negative or tol_x is not positive.
        """

    @staticmethod
    def lasso(k: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], reg: float, alpha: float, tol: float, max_iter: int) -> FodSolver:
        """
        Makes an elastic-net solver, which regresses each voxel's signal on the
        kernel's columns.

        k (m, c) is the weighted anisotropic kernel, whose columns are scaled to
        unit root mean square, giving k0; there is no intercept. The coefficients
        b minimise (1 / 2m) ||s - k0 b||^2 + reg (alpha ||b||_1 +
        (1 - alpha) / 2 ||b||^2), until no coefficient changes by more than tol
        relative to 1 + its size, or after max_iter sweeps. The FOD is b divided
        by the column scales, and it can be negative.

        Raises:
            ValueError: If reg is negative, alpha is not in (0, 1], tol is not
                positive or max_iter is below 1.
        """

    @property
    def n_dirs(self) -> int:
        """Number of FOD directions the solver returns."""

    @property
    def n_meas(self) -> int:
        """Number of measurements per voxel it expects."""

    @property
    def n_unconverged(self) -> int:
        """
        Number of solves so far that stopped at max_iter without converging;
        always 0 except for lasso.
        """

def fit_single_tissue(signals: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], solver: FodSolver, n_threads: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
    """
    Fits the FOD (n, c) of every row of signals (n, m), C-contiguous, with one
    solver.

    Raises:
        ValueError: If signals is not C-contiguous, does not have the solver's
            number of measurements, or n_threads is 0.
    """

def fit_csd(signals: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], initial: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], forward: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], constraint: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], norm_weight: float, threshold: float, max_iter: int, n_threads: int) -> tuple[Annotated[NDArray[numpy.float64], dict(shape=(None, None))], Annotated[NDArray[numpy.bool_], dict(shape=(None,))]]:
    """
    Fits every row of signals (n, m) by constrained spherical deconvolution.

    initial (c0, m) gives the starting FOD's first c0 coefficients, forward (m, c)
    is the forward model M and constraint (k, c) the weighted SH basis on the
    constraint directions. Each solve minimises ||M f - s||^2 + norm_weight ||f||^2
    plus the squared constraint rows where the FOD is below threshold, until that
    set of rows repeats or after max_iter solves. Returns the FODs (n, c) and
    whether each voxel's set repeated (n,).

    Raises:
        ValueError: If signals is not C-contiguous, the shapes do not match,
            max_iter is negative or n_threads is 0.
    """

def lsqnonneg(c: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], d: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu', writable=False)], tol_x: float | None = None) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
    """
    Solves one non-negative least-squares problem, min ||c x - d|| with x >= 0.

    c is (m, n) and d (m,). A coefficient enters the solution only when its
    gradient exceeds tol_x; None uses 10 * eps * norm(c, 1) * max(m, n).

    Raises:
        ValueError: If c is not C-contiguous or the shapes do not match.
    """

def ridge_l1_nnls(c: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], d: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu', writable=False)], ridge: float, l1: float, tol_x: float | None = None) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
    """
    Solves min ||c x - d||^2 + ridge^2 ||x||^2 + 2 l1 sum(x) with x >= 0.

    c is (m, n) and d (m,). Since x >= 0, the last term is an L1 penalty, which
    sets the coefficients the data do not support to zero; ridge = l1 = 0 is
    plain NNLS. A coefficient enters the solution only when its gradient
    exceeds tol_x; None uses 10 * eps * norm(c, 1) * max(m, n).

    Raises:
        ValueError: If c is not C-contiguous, the shapes do not match, ridge or
            l1 is negative or not finite, or l1 > 0 with ridge = 0.
    """

class DeconvKernel:
    """One resolution of a multi-tissue kernel, copied into C++."""

    def __init__(self, k_aniso: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], k_iso: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], w_k_iso: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], aniso_blocks: Sequence[tuple[int, int]]) -> None:
        """
        Copies the matrices of one kernel resolution.

        k_aniso is (m, ca), k_iso and w_k_iso (m, ci). aniso_blocks holds the
        (start, stop) columns of each anisotropic response function.

        Raises:
            ValueError: If the shapes do not match, or a block is empty or outside
                0..ca.
        """

def fit_multi_tissue(signals: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], weights: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu', writable=False)], lr: DeconvKernel, lr_solver: FodSolver, hr: DeconvKernel, hr_solver: FodSolver, max_outer: int, fraction_tol: float, final_tol_x: float, n_threads: int) -> tuple[Annotated[NDArray[numpy.float64], dict(shape=(None, None))], Annotated[NDArray[numpy.float64], dict(shape=(None, None))]]:
    """
    Fits the FODs and tissue fractions of every row of signals (n, m). The
    shell weights (m,) apply to the FOD fits only.

    Returns the high-resolution FODs (n, ca_hr) and the fractions
    (n, n_aniso_rfs + ci), summing to 1.

    Raises:
        ValueError: If signals is not C-contiguous, the shapes do not match, a
            solver does not match its kernel, or n_threads is 0.
    """

class LsiDesign:
    """
    A least-squares problem with inequality constraints, copied into C++ and prepared for many data vectors.
    """

    def __init__(self, a: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], g: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], ridge: float) -> None:
        """
        Prepares min ||a x - s||^2 + ridge^2 ||x||^2 subject to g x >= 0.

        a is (m, n) and g (k, n).

        Raises:
            ValueError: If the shapes do not match, ridge is negative, or [a; ridge I]
                does not have full column rank.
        """

def fit_lsi(signals: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], design: LsiDesign, n_threads: int) -> tuple[Annotated[NDArray[numpy.float64], dict(shape=(None, None))], Annotated[NDArray[numpy.bool_], dict(shape=(None,))]]:
    """
    Solves the constrained least-squares problem of every row of signals (v, m).

    Returns the solutions (v, n) and whether each voxel's NNLS converged (v,).

    Raises:
        ValueError: If signals is not C-contiguous, does not have the design's
            number of measurements, or n_threads is 0.
    """

def fit_log_signal(x: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], g: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], log_s: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], weighted: bool, max_iter: int, tol: float, n_threads: int) -> tuple[Annotated[NDArray[numpy.float64], dict(shape=(None, None))], Annotated[NDArray[numpy.bool_], dict(shape=(None,))]]:
    """
    Fits each row of log_s (v, n) by least squares on the design x (n, p).

    With weighted=True the fit is weighted by the predicted signal, iterated until
    no predicted log signal changes by more than tol or for max_iter weighted
    fits. A voxel whose solution has g @ p < 0 in some row of g (m, p) is fitted
    again subject to g @ p >= 0; m = 0 means no constraints.

    Returns the parameters (v, p) and whether each voxel broke a constraint (v,).

    Raises:
        ValueError: If an array is not C-contiguous, the shapes do not match,
            max_iter is negative or n_threads is 0.
    """

def fit_noddi(signals: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], mu: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], bvals: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu', writable=False)], bvecs: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], is_b0: Annotated[NDArray[numpy.bool_], dict(shape=(None,), device='cpu', writable=False)], v_ic_grid: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu', writable=False)], kappa_grid: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu', writable=False)], d_par: float, d_iso: float, ridge: float, l1: float, n_threads: int) -> tuple[Annotated[NDArray[numpy.float64], dict(shape=(None, None))], Annotated[NDArray[numpy.bool_], dict(shape=(None,))]]:
    """
    Fits NODDI to each voxel's signal, a row of signals (v, n), from the direction
    mu (v, 3).

    The start comes from a dictionary of NODDI signals over v_ic_grid and
    kappa_grid, solved with the ridge and l1 weights; the fit is the NODDI
    Matlab Toolbox's maximum-likelihood fit with Rician noise, with its model.
    d_par and d_iso are in mm^2/s and bvals in s/mm^2.

    Returns v_ic, kappa, v_iso and the fitted direction of each voxel (v, 6), and
    whether each voxel was fitted (v,).

    Raises:
        ValueError: If an array is not C-contiguous, the shapes do not match, a
            grid is empty, ridge is not positive, l1 is negative or n_threads is 0.
    """

def noddi_signal(bvals: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu', writable=False)], bvecs: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], v_ic: float, kappa: float, v_iso: float, mu: Sequence[float], d_par: float, d_iso: float) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
    """
    Returns the normalised NODDI signal of each volume, as the NODDI Matlab
    Toolbox's WatsonSHStickTortIsoV_B0 model computes it, for the neurite density
    v_ic, Watson concentration kappa and free-water fraction v_iso about the
    direction mu. bvals (n,) are in s/mm^2, bvecs (n, 3) unit directions, and
    d_par and d_iso in mm^2/s.

    Raises:
        ValueError: If an array is not C-contiguous or the shapes do not match.
    """

def denoise_mppca(image: Annotated[NDArray[numpy.float32], dict(shape=(None, None, None, None), device='cpu', writable=False)], voxels: Annotated[NDArray[numpy.int64], dict(shape=(None,), device='cpu', writable=False)], patch: Sequence[int], n_threads: int) -> tuple[Annotated[NDArray[numpy.float64], dict(shape=(None, None))], Annotated[NDArray[numpy.float64], dict(shape=(None,))], Annotated[NDArray[numpy.int64], dict(shape=(None,))]]:
    """
    Denoises by MP-PCA the voxels at flat indices voxels (k,) int64 of a
    C-contiguous float32 image (X, Y, Z, M).

    Each voxel gets its own patch of patch = (px, py, pz) voxels, shifted to stay
    inside the grid at its edges.

    Returns the denoised signals (k, M), the noise level sigma (k,) and the
    number of signal components (k,) of each voxel; NaN signals and sigma
    where the patch holds a value that is not finite.

    Raises:
        ValueError: If an array is not C-contiguous, the image has no volumes,
            the patch is not 1 to the grid's size along an axis, an index is
            outside the grid or n_threads is 0.
    """

def remove_gibbs_ringing(data: Annotated[NDArray[numpy.float32], dict(shape=(None, None, None, None), device='cpu', writable=False)], out: Annotated[NDArray[numpy.float32], dict(shape=(None, None, None, None), device='cpu')], slice_axis: int, n_shifts: int, min_window: int, max_window: int, n_threads: int) -> None:
    """
    Removes Gibbs ringing from each slice of data (x, y, z, v) into out.

    The slices are the planes across slice_axis (0, 1 or 2), in every volume v.
    Each is unringed by local sub-voxel shifts in steps of 1 / (2 n_shifts)
    voxels, chosen by the total variation between neighbours min_window to
    max_window + 1 voxels away on each side. Both arrays are C-contiguous float32
    of one shape.

    Raises:
        ValueError: If an array is not C-contiguous, the shapes differ,
            slice_axis is not 0, 1 or 2, n_shifts < 1, the window is not
            0 <= min_window <= max_window, or n_threads is 0.
    """

class DetParams:
    """
    Parameters of deterministic tracking.

    lmax comes from the FOD.
    """

    def __init__(self, cutoff: float, max_angle: float) -> None:
        """
        Checks and keeps the parameters.

        cutoff is the FOD amplitude below which a streamline stops, and max_angle
        the largest turn between steps, in degrees.

        Raises:
            ValueError: If cutoff is negative or not finite, or max_angle is not in
                (0, 180].
        """

class ProbParams:
    """
    Parameters of probabilistic tracking.

    lmax comes from the FOD; the rejection-sampling bound is calibrated from it.
    """

    def __init__(self, cutoff: float, max_angle: float, trials: int = 1000, attempts: int = 1000) -> None:
        """
        Checks and keeps the parameters.

        cutoff is the FOD amplitude a candidate direction must exceed, max_angle the
        half-angle of the cone candidates are drawn in, in degrees, trials the number
        of candidates per step before NO_DIRECTION, and attempts the number of random
        directions tried for a seed's first direction.

        Raises:
            ValueError: If cutoff is negative or not finite, max_angle is not above
                0.001 rad or above 180, or trials or attempts is 0.
        """

class TestAlgorithm:
    """
    A dummy algorithm for testing the tracking loop.

    step is "straight", "quarter_turn" (90 degrees about z each
    step), "wander" (a random walk that draws three numbers per
    step), or "refuse" with stop, an EndStatus from 1 to 6 that
    every step returns. The first-direction search draws a random
    unit vector and fails where the seed's x is above
    search_refuses_past_x.
    """

    def __init__(self, step: str, stop: int | None = None, search_refuses_past_x: float = float('inf')) -> None:
        """
        Builds the dummy algorithm.

        Raises:
            ValueError: If step is unknown, or stop is missing for "refuse", given
                otherwise, or not in 1..6.
        """

def track_batch(fod: Annotated[NDArray[numpy.float32], dict(shape=(None, None, None, None), device='cpu', writable=False)], fod_inv_affine: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], mask: tuple[Annotated[NDArray[numpy.float32], dict(shape=(None, None, None), device='cpu', writable=False)], Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)]] | None, seeds: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], directions: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)] | None, indices: Annotated[NDArray[numpy.int64], dict(shape=(None,), device='cpu', writable=False)], run_seed: int, step_size: float, max_steps: int, bidirectional: bool, algorithm: DetParams | ProbParams | TestAlgorithm, positions: Annotated[NDArray[numpy.float32], dict(shape=(None, None, None), device='cpu')], n_threads: int, full: bool = False) -> tuple[Annotated[NDArray[numpy.int64], dict(shape=(None,))], Annotated[NDArray[numpy.uint8], dict(shape=(None, None))], Annotated[NDArray[numpy.float64], dict(shape=(None, None))], dict[str, float]]:
    """
    Tracks a batch of seeds in parallel into the rows of positions.

    fod is (X, Y, Z, C) float32 in the tournier basis, C-contiguous, with all SH
    degrees if full (an asymmetric FOD), else the even ones. mask, if any, is an
    (X, Y, Z) float32 array and its (4, 4) world-to-voxel matrix. seeds are (n, 3)
    world positions, directions (n, 3) unit vectors followed as given, or None for
    the algorithm's first-direction search, and indices (n,) the global indices that
    key the random numbers. positions is (batch, max_steps + 1, 3) float32 with
    batch >= n, C-contiguous; row i receives seed i's points, and rows past n are
    untouched.

    Returns each seed's number of points (n,), 0 where it did not start, the
    EndStatus of the start and the end of its row (n, 2), the direction followed,
    given or found (n, 3), and the batch's sampling counters summed over its
    streamlines: sampled_steps, accepted_index_sum, truncations, max_truncation and
    calibration_directions, all 0 for algorithms that do not sample.

    Raises:
        ValueError: If an array is not C-contiguous or the shapes do not match, an
            index is negative, step_size or max_steps is not positive, C is not a
            whole series up to lmax 20 in the layout full says (det, prob), or
            n_threads is 0.
    """

def resample_streamlines(points: Annotated[NDArray[numpy.float32], dict(shape=(None, None), device='cpu', writable=False)], offsets: Annotated[NDArray[numpy.int64], dict(shape=(None,), device='cpu', writable=False)], step_size: float, num_points: int, n_threads: int, positions: bool = True) -> tuple[Annotated[NDArray[numpy.float32], dict(shape=(None, None))], Annotated[NDArray[numpy.int64], dict(shape=(None,))], Annotated[NDArray[numpy.int64], dict(shape=(None,))], Annotated[NDArray[numpy.float64], dict(shape=(None,))]]:
    """
    Resamples every streamline to step_size (mm) or to num_points points.

    points is (P, 3) float32 and offsets (S + 1,) int64, as in a Tractogram. Give
    step_size > 0 with num_points 0, or num_points >= 2 with step_size 0.
    Streamlines of fewer than 2 points are copied. Returns the new points (Q, 3)
    float32 and offsets (S + 1,) int64, and with positions, for each new point
    the index of the input point that starts its segment (Q,) int64 and its
    fraction along the segment (Q,) float64; without, both are empty.

    Raises:
        ValueError: If the arrays are not C-contiguous or do not match, the offsets
            are not valid, both or neither of step_size and num_points are given,
            the output would be too large to index, or n_threads is 0.
    """

def add_track_density(points: Annotated[NDArray[numpy.float32], dict(shape=(None, None), device='cpu', writable=False)], offsets: Annotated[NDArray[numpy.int64], dict(shape=(None,), device='cpu', writable=False)], world_to_voxel: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], shape: Annotated[NDArray[numpy.int64], dict(shape=(None,), device='cpu', writable=False)], upsample_ratio: int, max_spacing: float, ends_only: bool, weights: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu', writable=False)], density: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu')], n_threads: int) -> None:
    """
    Adds the track density of a tractogram to density, in place.

    points is (P, 3) float32 and offsets (S + 1,) int64, as in a Tractogram.
    world_to_voxel (3, 4) float64 maps world coordinates (mm) to voxel
    coordinates of a grid of the given (3,) int64 shape, and density (N,) float64
    holds its voxels in C order. Each streamline is upsampled along its spline by
    upsample_ratio, or, with max_spacing > 0, each segment by the smallest ratio
    that brings its pieces below max_spacing. It then adds its weight (1 when
    weights is empty, otherwise (S,) float64) once to every voxel it passes
    through, or with ends_only to the voxel of each of its two endpoints.

    Raises:
        ValueError: If the arrays are not C-contiguous or do not match, the offsets
            are not valid, upsample_ratio is below 1, max_spacing is negative or
            not finite, or n_threads is 0.
    """

def compartment_model_codes() -> dict[str, int]:
    """
    Returns the code of each compartment model in the engine, by the name
    larmor.simulation.CompartmentModel gives it.
    """

def compartment_attenuation(models: Annotated[NDArray[numpy.int64], dict(shape=(None,), device='cpu', writable=False)], parameters: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], bvals: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu', writable=False)], cosines: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu', writable=False)], small_delta_ms: float, big_delta_ms: float) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
    """
    Returns the attenuation of each compartment (rows) for each gradient (columns).

    models holds the codes 0 stick, 1 zeppelin, 2 ball, 3 powder-averaged stick,
    4 sphere, 5 Watson-dispersed sticks, 6 cylinder and 7 dot. parameters (C, 4)
    holds the diffusivity and perpendicular diffusivity in mm^2/s, the radius in um
    and the Watson kappa. The b-values are in s/mm^2, and cosines holds g . n per
    gradient.
    """

def sphere_roots() -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
    """
    Returns the positive roots x_m of j1'(x) = 0 the sphere model sums over.
    """

def cylinder_roots() -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
    """
    Returns the positive roots x_m of J1'(x) = 0 the cylinder model sums over.
    """

def relaxation_weights(proton_density: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu', writable=False)], t1_ms: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu', writable=False)], decay_ms: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu', writable=False)], tr_ms: float, te_ms: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu', writable=False)]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
    """
    Returns the relaxation weight of each compartment (rows) at each echo time
    (columns).

    The weight is PD (1 - exp(-TR / T1)) exp(-TE / T_decay), with times in ms.
    """

def simulate_instant(models: Annotated[NDArray[numpy.int64], dict(shape=(None,), device='cpu', writable=False)], parameters: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], tissue: Annotated[NDArray[numpy.int64], dict(shape=(None,), device='cpu', writable=False)], fraction: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu', writable=False)], relaxation: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], bvals: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu', writable=False)], bvecs: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], small_delta_ms: float, big_delta_ms: float, tissue_fractions: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], offsets: Annotated[NDArray[numpy.int64], dict(shape=(None,), device='cpu', writable=False)], fibre_tissue: Annotated[NDArray[numpy.int64], dict(shape=(None,), device='cpu', writable=False)], fibre_weight: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu', writable=False)], fibre_direction: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device='cpu', writable=False)], out: Annotated[NDArray[numpy.float32], dict(shape=(None, None, None), device='cpu')], n_threads: int) -> None:
    """
    Writes the noise-free signal of each voxel into out, (V, E, N) float32.

    Raises:
        ValueError: If the shapes do not fit together, an index is out of range,
            or n_threads is 0.
    """
