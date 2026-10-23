// Elastic-net regression, least squares with a mix of L1 and L2 penalties, by coordinate descent.
#pragma once

#include "common/types.hpp"

namespace larmor::sd {

/// The design matrix with each column divided by its root mean square, computed once per kernel
/// for every voxel.
struct StandardizedDesign {
    /// The scaled columns, (m, c), each with (1 / m) ||x_j||^2 = 1. A column of zeros stays as
    /// it is, and out of the fit.
    Matrix columns;
    /// The root mean square of each original column, sqrt((1 / m) ||x_j||^2), or 1 for a
    /// column of zeros, (c,).
    Vector scales;
    /// Whether each column has a non-zero value; a column of zeros always gets a coefficient
    /// of 0.
    Eigen::Array<bool, Eigen::Dynamic, 1> nonzero;
};

/// Divides each column of design (m, c) by its root mean square, without centring it.
StandardizedDesign standardize_columns(const MatrixView& design);

/// Settings of the elastic net.
struct ElasticNetSettings {
    /// The overall weight of the penalty, lambda.
    double penalty = 0.1;
    /// The share of the penalty on the L1 norm, alpha in (0, 1]; the rest is on half the
    /// squared L2 norm.
    double l1_ratio = 0.5;
    /// Coordinate descent stops when no coefficient changes by more than this, relative to
    /// 1 + its size, over a sweep of every column.
    double relative_tolerance = 1e-4;
    /// Maximum number of sweeps, full or over the non-zero coefficients only.
    Index max_sweeps = 100000;
};

/// The coefficients of one elastic-net solve and whether it converged.
struct ElasticNetResult {
    /// The coefficients of the original (unscaled) columns, (c,).
    Vector coefficients;
    /// Whether a full sweep changed no coefficient by more than the tolerance before
    /// max_sweeps; when false, the coefficients are those of the last sweep.
    bool converged = true;
};

/// Solves the elastic net for one right-hand side, without an intercept.
///
/// With X0 the scaled columns and d the data, b minimises
///   (1 / (2 m)) ||d - X0 b||^2 + penalty * (l1_ratio ||b||_1 + (1 - l1_ratio) / 2 ||b||^2)
/// and the result is b divided by the column scales. When the penalty is at least
/// max |X0^T d| / (m * l1_ratio), the result is all zero. The coefficients are not constrained
/// to be non-negative.
///
/// References:
///   Zou H, Hastie T. Regularization and variable selection via the elastic net. J R Stat Soc
///   B 67 (2005) 301-320.
///   Friedman J, Hastie T, Tibshirani R. Regularization paths for generalized linear models
///   via coordinate descent. J Stat Softw 33 (2010) 1-22.
ElasticNetResult solve_elastic_net(const StandardizedDesign& design, const VectorView& data,
                                   const ElasticNetSettings& settings);

} // namespace larmor::sd
