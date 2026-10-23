// Constrained spherical deconvolution of one shell's signal into an FOD in spherical harmonics.
#pragma once

#include "common/types.hpp"

namespace larmor::sd {

/// The matrices of a CSD fit, shared by every voxel, for m measurements, c SH coefficients of
/// the FOD and k constraint directions.
struct CsdProblem {
    /// Maps a signal to the first c0 <= c coefficients of the starting FOD, (c0, m); the others
    /// start at 0.
    Matrix initial;
    /// The forward model M: the signal of each SH coefficient of the FOD, (m, c).
    Matrix forward;
    /// The SH basis on the constraint directions, times the constraint's weight, (k, c).
    Matrix constraint;
    /// M^T M plus the weight of the penalty on ||f||^2 on its diagonal, (c, c).
    Matrix gram;
    /// A direction is constrained when its weighted amplitude is below this.
    double threshold = 0.0;
    /// Maximum number of constrained solves per voxel.
    Index max_iterations = 0;
};

/// Builds a CsdProblem, computing its Gram matrix from forward and norm_weight.
CsdProblem make_csd_problem(Matrix initial, Matrix forward, Matrix constraint, double norm_weight,
                            double threshold, Index max_iterations);

/// The CSD solution of one signal and whether it settled.
struct CsdResult {
    /// The FOD's SH coefficients, (c,).
    Vector fod;
    /// False when max_iterations solves ran out before the set of constrained directions
    /// repeated; the FOD is then that of the last solve. True for max_iterations = 0.
    bool converged = true;
};

/// Returns the CSD solution for one signal (m,).
///
/// Starting from the linear estimate `initial * signal`, each iteration penalises the constraint
/// directions where the FOD is below the threshold and solves again, until that set of
/// directions repeats or after max_iterations solves.
///
/// References:
///   Tournier JD, Calamante F, Connelly A. Robust determination of the fibre orientation
///   distribution in diffusion MRI: non-negativity constrained super-resolved spherical
///   deconvolution. NeuroImage 35 (2007) 1459-1472.
CsdResult solve_csd(const CsdProblem& problem, const VectorView& signal);

} // namespace larmor::sd
