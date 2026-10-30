#include "sd/lsi.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/QR>

#include "sd/nnls.hpp"

namespace larmor::sd {

LsiDesign make_lsi_design(const MatrixView& design, const MatrixView& constraints, double ridge) {
    const Index n_measurements = design.rows();
    const Index n_parameters = design.cols();
    if (constraints.cols() != n_parameters) {
        throw std::invalid_argument("the constraints have " + std::to_string(constraints.cols()) +
                                    " columns but the design has " + std::to_string(n_parameters));
    }
    if (!(ridge >= 0.0)) {
        throw std::invalid_argument("the ridge must be at least 0");
    }

    const bool has_ridge = ridge > 0.0;
    Matrix stacked = Matrix::Zero(n_measurements + (has_ridge ? n_parameters : 0), n_parameters);
    stacked.topRows(n_measurements) = design;
    if (has_ridge) {
        stacked.bottomRows(n_parameters).diagonal().setConstant(ridge);
    }

    Eigen::ColPivHouseholderQR<Matrix> qr(stacked);
    // A pivot below max(rows, columns) * eps of the largest is rounding error.
    qr.setThreshold(static_cast<double>(std::max(stacked.rows(), stacked.cols())) *
                    std::numeric_limits<double>::epsilon());
    if (stacked.rows() < n_parameters || qr.rank() < n_parameters) {
        throw std::invalid_argument(
            "the design does not have full column rank: the measurements cannot determine all " +
            std::to_string(n_parameters) + " parameters");
    }

    const Matrix thin_q = qr.householderQ() * Matrix::Identity(stacked.rows(), n_parameters);
    const Matrix r_inverse = qr.matrixR()
                                 .topLeftCorner(n_parameters, n_parameters)
                                 .triangularView<Eigen::Upper>()
                                 .solve(Matrix::Identity(n_parameters, n_parameters));
    const Matrix back_substitution = qr.colsPermutation() * r_inverse;
    Matrix projected = constraints * back_substitution;
    // A positive scale of a constraint does not change the problem; unit rows keep the NNLS
    // columns comparable.
    for (Index row = 0; row < projected.rows(); ++row) {
        const double norm = projected.row(row).norm();
        if (norm > 0.0) {
            projected.row(row) /= norm;
        }
    }
    Matrix gram = projected * projected.transpose();
    Vector abs_sums = projected.cwiseAbs().rowwise().sum();
    return {.q_transposed = thin_q.topRows(n_measurements).transpose(),
            .back_substitution = back_substitution,
            .constraints = std::move(projected),
            .constraint_gram = std::move(gram),
            .constraint_abs_sums = std::move(abs_sums)};
}

namespace {

// A Cholesky pivot below this share of its diagonal entry counts as a dependent column.
constexpr double MIN_RELATIVE_PIVOT = 1e-12;

// The Gram matrix of the least-distance NNLS, H = E E^T + f f^T, read entry by entry.
struct LdpGram {
    const Matrix& constraint_gram;
    const Vector& bounds;

    // Returns the entry (row, column) of H.
    double operator()(Index row, Index column) const {
        return constraint_gram(row, column) + bounds(row) * bounds(column);
    }
};

// The Cholesky factor L of a Gram matrix on its active columns, G_AA = L L^T, the columns in the
// order they joined.
class GrowingCholesky {
public:
    explicit GrowingCholesky(const LdpGram& gram_entries, Index n_columns)
        : gram(gram_entries), factor(n_columns, n_columns) {}

    // Adds a column; returns false when its pivot is not clearly positive.
    bool append(Index column) {
        const auto position = static_cast<Index>(columns.size());
        Vector cross(position);
        for (Index row = 0; row < position; ++row) {
            cross(row) = gram(columns[static_cast<std::size_t>(row)], column);
        }
        factor.topLeftCorner(position, position).triangularView<Eigen::Lower>().solveInPlace(cross);
        const double pivot = gram(column, column) - cross.squaredNorm();
        if (!(pivot > MIN_RELATIVE_PIVOT * gram(column, column))) {
            return false;
        }
        factor.row(position).head(position) = cross.transpose();
        factor(position, position) = std::sqrt(pivot);
        columns.push_back(column);
        return true;
    }

    // Removes the columns no longer active from the factor, keeping the others in their order.
    void keep(const std::vector<bool>& is_active) {
        for (std::size_t position = columns.size(); position-- > 0;) {
            if (!is_active[static_cast<std::size_t>(columns[position])]) {
                remove_cholesky_column(factor, static_cast<Index>(columns.size()),
                                       static_cast<Index>(position));
                columns.erase(columns.begin() + static_cast<std::ptrdiff_t>(position));
            }
        }
    }

    // Returns the solution of G_AA z = rhs_A, spread into a vector of all columns.
    [[nodiscard]] Vector solve(const Vector& rhs) const {
        const auto n_active = static_cast<Index>(columns.size());
        Vector active(n_active);
        for (Index position = 0; position < n_active; ++position) {
            active(position) = rhs(columns[static_cast<std::size_t>(position)]);
        }
        const auto lower = factor.topLeftCorner(n_active, n_active).triangularView<Eigen::Lower>();
        lower.solveInPlace(active);
        lower.transpose().solveInPlace(active);
        Vector full = Vector::Zero(rhs.size());
        for (Index position = 0; position < n_active; ++position) {
            full(columns[static_cast<std::size_t>(position)]) = active(position);
        }
        return full;
    }

private:
    LdpGram gram;
    Matrix factor;
    std::vector<Index> columns;
};

// Returns the NNLS gradient f - H u = f (1 - f.u) - E E^T u, over the non-zero coefficients only.
Vector ldp_gradients(const LsiDesign& design, const Vector& bounds, const Vector& solution) {
    Vector gradients = bounds * (1.0 - bounds.dot(solution));
    for (Index column = 0; column < solution.size(); ++column) {
        if (solution(column) != 0.0) {
            // E E^T is symmetric, and its rows are contiguous.
            gradients -= solution(column) * design.constraint_gram.row(column).transpose();
        }
    }
    return gradients;
}

// Returns the inactive column with the largest gradient, the first on a tie; -1 if none.
Index steepest_inactive(const Vector& gradients, const std::vector<bool>& is_active) {
    Index steepest = -1;
    for (Index column = 0; column < gradients.size(); ++column) {
        if (!is_active[static_cast<std::size_t>(column)] &&
            (steepest < 0 || gradients(column) > gradients(steepest))) {
            steepest = column;
        }
    }
    return steepest;
}

// How far solution may move towards candidate before an active coefficient turns negative, and
// that coefficient's column (-1 when none does).
struct BlockingStep {
    double step = std::numeric_limits<double>::infinity();
    Index column = -1;
};

// Returns the blocking step of solution towards candidate.
BlockingStep blocking_step(const Vector& solution, const Vector& candidate,
                           const std::vector<bool>& is_active) {
    BlockingStep blocking;
    for (Index column = 0; column < solution.size(); ++column) {
        if (!is_active[static_cast<std::size_t>(column)] || candidate(column) > 0.0) {
            continue;
        }
        const double distance = solution(column) - candidate(column);
        const double step = distance > 0.0 ? solution(column) / distance : 0.0;
        if (step < blocking.step) {
            blocking = {.step = step, .column = column};
        }
    }
    return blocking;
}

// Solves the NNLS min ||[E^T; f^T] u - (0, ..., 0, 1)|| over u >= 0 by the Lawson-Hanson steps on
// the Gram matrix H = E E^T + f f^T. Returns none when a Cholesky pivot fails or the inner steps
// run out.
std::optional<Vector> ldp_multipliers(const LsiDesign& design, const Vector& bounds) {
    const Index n_columns = bounds.size();
    // solve_nnls's default tolerance, 10 eps norm1(C) max(rows, columns), for C = [E^T; f^T].
    const double norm1 = (design.constraint_abs_sums + bounds.cwiseAbs()).maxCoeff();
    const double tolerance =
        10.0 * std::numeric_limits<double>::epsilon() * norm1 *
        static_cast<double>(std::max(design.back_substitution.rows() + 1, n_columns));
    const Index max_inner_steps = 3 * n_columns;

    std::vector<bool> is_active(static_cast<std::size_t>(n_columns), false);
    GrowingCholesky cholesky(LdpGram{.constraint_gram = design.constraint_gram, .bounds = bounds},
                             n_columns);
    Vector solution = Vector::Zero(n_columns);
    Index n_inner_steps = 0;
    while (true) {
        const Vector gradients = ldp_gradients(design, bounds, solution);
        const Index entering = steepest_inactive(gradients, is_active);
        if (entering < 0 || gradients(entering) <= tolerance) {
            return solution;
        }
        is_active[static_cast<std::size_t>(entering)] = true;
        if (!cholesky.append(entering)) {
            return std::nullopt;
        }
        Vector candidate = cholesky.solve(bounds);
        for (BlockingStep blocking = blocking_step(solution, candidate, is_active);
             blocking.column >= 0; blocking = blocking_step(solution, candidate, is_active)) {
            if (++n_inner_steps > max_inner_steps) {
                return std::nullopt;
            }
            solution += blocking.step * (candidate - solution);
            solution(blocking.column) = 0.0;
            // The blocking coefficient leaves, and so does any that rounding left near zero.
            for (Index column = 0; column < n_columns; ++column) {
                if (column == blocking.column || std::fabs(solution(column)) < tolerance) {
                    is_active[static_cast<std::size_t>(column)] = false;
                    solution(column) = 0.0;
                }
            }
            cholesky.keep(is_active);
            candidate = cholesky.solve(bounds);
        }
        solution = candidate;
    }
}

// Solves the problem for one data vector, with the Cholesky-based NNLS unless only_qr, and the
// QR-based one when it fails.
LsiResult solve_lsi_by(const LsiDesign& design, const VectorView& data, bool only_qr) {
    const Index n_parameters = design.back_substitution.rows();
    const Vector rotated = design.q_transposed * data;
    // E z >= f is solved as E z' >= f / scale with z = scale z'. A violated constraint
    // shows in the NNLS gradient as violation * (1 - f.u), and 1 - f.u is small when the
    // bounds are large, so bounds of order one keep the stopping rule's tolerance at the
    // scale of the constraints.
    const Vector raw_bounds = -(design.constraints * rotated);
    const double scale = raw_bounds.cwiseAbs().maxCoeff();
    if (!(scale > 0.0)) {
        // f = 0: z = 0 is the closest feasible point, so x is the unconstrained solution.
        return {.solution = design.back_substitution * rotated, .converged = true};
    }
    const Vector bounds = raw_bounds / scale;

    Vector multipliers;
    bool converged = true;
    const std::optional<Vector> fast = only_qr ? std::nullopt : ldp_multipliers(design, bounds);
    if (fast) {
        multipliers = *fast;
    } else {
        // The NNLS on the explicit matrix [E^T; f^T], by QR, for an ill-conditioned voxel.
        Matrix nnls_design(n_parameters + 1, design.constraints.rows());
        nnls_design.topRows(n_parameters) = design.constraints.transpose();
        nnls_design.row(n_parameters) = bounds.transpose();
        Vector target = Vector::Zero(n_parameters + 1);
        target(n_parameters) = 1.0;
        NnlsResult fallback =
            solve_nnls_with_limit(nnls_design, target, default_gradient_tolerance(nnls_design),
                                  3 * design.constraints.rows());
        multipliers = std::move(fallback.solution);
        converged = fallback.converged;
    }
    // The residual r = [E^T u; f^T u - 1], and z = -r_head / r_last.
    const Vector distance =
        scale * (design.constraints.transpose() * multipliers) / (1.0 - bounds.dot(multipliers));
    return {.solution = design.back_substitution * (distance + rotated), .converged = converged};
}

} // namespace

LsiResult solve_lsi(const LsiDesign& design, const VectorView& data) {
    return solve_lsi_by(design, data, false);
}

LsiResult solve_lsi_by_qr(const LsiDesign& design, const VectorView& data) {
    return solve_lsi_by(design, data, true);
}

} // namespace larmor::sd
