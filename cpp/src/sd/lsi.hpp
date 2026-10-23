// Least squares with linear inequality constraints, through least-distance programming and NNLS.
#pragma once

#include "common/types.hpp"

namespace larmor::sd {

/// The problem min ||A x - s||^2 + ridge^2 ||x||^2 subject to G x >= 0, prepared once for many
/// data vectors s.
///
/// The problem becomes the least-distance problem min ||z|| subject to E z >= f, with
/// [A; ridge I] P = Q R (a QR decomposition with column pivoting), z = R P^T x - Q^T [s; 0],
/// E = G P R^-1 and f = -E Q^T [s; 0]. Only f depends on s.
struct LsiDesign {
    /// The rows of Q that multiply the measurements, transposed, (n, m): Q^T [s; 0] is
    /// q_transposed s.
    Matrix q_transposed;
    /// P R^-1, (n, n): x = back_substitution (z + Q^T [s; 0]).
    Matrix back_substitution;
    /// E = G P R^-1, (k, n), each row scaled to unit length.
    Matrix constraints;
    /// E E^T, (k, k).
    Matrix constraint_gram;
    /// The sum of absolute values of each row of E, (k,).
    Vector constraint_abs_sums;
};

/// Prepares the problem for design A (m, n) and constraints G (k, n).
///
/// Throws std::invalid_argument when the shapes do not match, or when [A; ridge I] does not have
/// full column rank, so that the solution would not be unique.
LsiDesign make_lsi_design(const MatrixView& design, const MatrixView& constraints, double ridge);

/// The solution of one data vector and whether its NNLS ended by its own rule.
struct LsiResult {
    /// The solution x, (n,).
    Vector solution;
    /// False when the NNLS stopped at its limit of inner steps (3 per constraint): x is then
    /// the last least-squares fit and may break a constraint.
    bool converged = true;
};

/// Solves the problem for one data vector s (m,).
///
/// The least-distance problem is solved by NNLS, with its least-squares fits solved from a Gram
/// matrix; a voxel where that is ill-conditioned is solved again by solve_nnls on the explicit
/// matrix.
///
/// References:
///   Lawson CL, Hanson RJ. Solving Least Squares Problems. Prentice-Hall, 1974 (chapter 23).
LsiResult solve_lsi(const LsiDesign& design, const VectorView& data);

/// Runs solve_lsi with the QR-based NNLS only, the fallback of an ill-conditioned voxel.
LsiResult solve_lsi_by_qr(const LsiDesign& design, const VectorView& data);

} // namespace larmor::sd
