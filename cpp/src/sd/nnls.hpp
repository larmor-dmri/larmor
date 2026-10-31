// Non-negative least squares by the Lawson-Hanson active-set method.
#pragma once

#include "common/types.hpp"

namespace larmor::sd {

/// Returns the gradient tolerance scaled to a design matrix, 10 eps norm1(C) max(rows, columns),
/// where norm1(C) is the largest sum of absolute values of a column.
double default_gradient_tolerance(const MatrixView& design);

/// Solves min ||C x - d|| subject to x >= 0, for a design matrix C (rows, columns) with any
/// number of columns, d (rows,) and x (columns,).
///
/// The method ends when no gradient C^T (d - C x) exceeds `gradient_tolerance`. When active
/// columns are linearly dependent or nearly so, the dependent ones get zero coefficients, so x
/// stays bounded.
///
/// References:
///   Lawson CL, Hanson RJ. Solving Least Squares Problems. Prentice-Hall, 1974.
Vector solve_nnls(const MatrixView& design, const VectorView& data, double gradient_tolerance);

/// The solution of an NNLS problem and whether the method ended by its own rule.
struct NnlsResult {
    /// The solution x, (columns,).
    Vector solution;
    /// False when the method stopped at its limit of inner steps, returning its last
    /// least-squares fit, which may have negative coefficients.
    bool converged = true;
};

/// Runs solve_nnls with a limit on the total number of inner steps, the steps that move x back
/// into x >= 0, and says whether the method converged. solve_nnls uses 3 * columns steps.
NnlsResult solve_nnls_with_limit(const MatrixView& design, const VectorView& data,
                                 double gradient_tolerance, Index max_inner_steps);

/// A kernel prepared once for the ridge-regularised NNLS of every voxel.
struct RidgeDesign {
    /// The kernel K, (m, c).
    Matrix kernel;
    /// K^T K + ridge^2 I, (c, c).
    Matrix gram;
    /// The weight of the ridge.
    double ridge = 0.0;
    /// Whether the least-squares fits solve the normal equations with the Gram matrix instead of a
    /// QR decomposition of the stacked matrix: with no ridge, or a ridge that keeps them well
    /// conditioned.
    bool normal_equations = false;
};

/// Removes the column at `position` from the lower Cholesky factor L of the Gram matrix of
/// n_columns columns, so that L L^T is the Gram matrix of the others, in the same order.
///
/// Only the top-left (n_columns - 1) square of `factor` is valid afterwards.
void remove_cholesky_column(Matrix& factor, Index n_columns, Index position);

/// Prepares a kernel for solve_ridge_nnls.
RidgeDesign make_ridge_design(const MatrixView& kernel, double ridge);

/// Solves min ||K x - s||^2 + ridge^2 ||x||^2 + 2 l1 sum(x) subject to x >= 0.
///
/// Since x >= 0, the last term is an L1 penalty. The steps and result are those of solve_nnls on
/// the stacked matrix [K; ridge I] and data [s; -l1 / ridge]. l1 > 0 needs ridge > 0, and throws
/// std::invalid_argument otherwise.
Vector solve_ridge_nnls(const RidgeDesign& design, const VectorView& signal, double l1,
                        double gradient_tolerance);

} // namespace larmor::sd
