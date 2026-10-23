// Linear least-squares fits of the log signal of diffusion-weighted voxels.
#pragma once

#include <Eigen/LU>

#include "common/types.hpp"

namespace larmor::dwi {

/// A column vector of booleans, one per voxel.
using Flags = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

/// How each voxel is fitted.
struct LogFitSettings {
    /// Whether the fit is weighted, with weights from the predicted signal.
    bool weighted = false;
    /// Maximum number of weighted fits after the first, ordinary one.
    Index max_reweightings = 0;
    /// Reweighting stops once no predicted log signal changes by more than this.
    double log_signal_tolerance = 0.0;
};

/// A design matrix X (N, P) of the log signal and constraints G (M, P) on its parameters p,
/// G p >= 0, prepared once for every voxel.
///
/// The fits work on X with its columns scaled to unit norm and on G with the same column scales;
/// the parameters are scaled back at the end. M = 0 means no constraints.
struct LogSignalDesign {
    /// The norm of each column of X, 1 for a zero column, (P,).
    Vector column_scales;
    /// X with columns of unit norm, (N, P).
    Matrix scaled_design;
    /// The pseudo-inverse of scaled_design, (P, N).
    Matrix pseudo_inverse;
    /// G with the columns divided by column_scales, (M, P).
    Matrix scaled_constraints;

    /// Returns the number of measurements, N.
    [[nodiscard]] Index n_measurements() const { return scaled_design.rows(); }
    /// Returns the number of parameters, P.
    [[nodiscard]] Index n_parameters() const { return scaled_design.cols(); }
    /// Returns whether there are constraints.
    [[nodiscard]] bool is_constrained() const { return scaled_constraints.rows() > 0; }
};

/// Prepares a design matrix (N, P), of full column rank, and constraints (M, P), M >= 0.
LogSignalDesign make_log_signal_design(const MatrixView& design, const MatrixView& constraints);

/// Buffers one thread reuses for every voxel it fits.
struct LogFitScratch {
    /// The scaled design with each row multiplied by its weight, (N, P).
    Matrix weighted_design;
    /// The matrix of the weighted normal equations; (P, P).
    Matrix normal_matrix;
    /// Its LU decomposition.
    Eigen::PartialPivLU<Matrix> normal_lu;
};

/// Makes the scratch buffers of a thread for one design.
LogFitScratch make_log_fit_scratch(const LogSignalDesign& design);

/// Fits the log signal of a block of voxels, one per row of block_log_signals (b, N), writing
/// their parameters to block_parameters (b, P) and whether each broke a constraint to
/// block_constrained (b,).
///
/// The weighted fit weights each measurement by the square of the signal the fit predicts. It
/// starts from the ordinary fit and stops once no predicted log signal changes by more than
/// log_signal_tolerance, or after max_reweightings weighted fits. A voxel whose weighted normal
/// equations are singular or whose weighted fit is not finite keeps the ordinary fit.
///
/// With constraints, a voxel whose solution has G p < 0 in some row is fitted again subject to
/// G p >= 0, with the weights of its unconstrained solution, through a least-distance problem
/// solved by non-negative least squares. Where no parameters meet the constraints, the weighted
/// design does not have full rank, the solve does not converge or its result breaks a
/// constraint by more than rounding, the voxel keeps its unconstrained solution.
///
/// References:
///   Veraart J, Sijbers J, Sunaert S, Leemans A, Jeurissen B. Weighted linear least squares
///   estimation of diffusion MRI parameters: strengths, limitations, and pitfalls. NeuroImage
///   81 (2013) 335-346.
///   Lawson CL, Hanson RJ. Solving Least Squares Problems. Prentice-Hall, 1974 (chapter 23).
void fit_log_signal_block(const LogSignalDesign& design, const LogFitSettings& settings,
                          const MatrixView& block_log_signals, LogFitScratch& scratch,
                          Eigen::Ref<Matrix> block_parameters, Eigen::Ref<Flags> block_constrained);

} // namespace larmor::dwi
