#include "sd/nnls.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

#include <Eigen/QR>

namespace larmor::sd {

namespace {

// Whether each column is in the active set, free to be non-zero.
using ActiveSet = Eigen::Array<bool, Eigen::Dynamic, 1>;

// Returns C^T (d - C x), how much each column would lower the error if its coefficient grew.
Vector gradient(const MatrixView& design, const VectorView& data, const Vector& solution) {
    return design.transpose() * (data - design * solution);
}

// Returns the inactive column with the largest gradient, the first on a tie, or none.
std::optional<Index> steepest_inactive_column(const Vector& gradients, const ActiveSet& is_active) {
    std::optional<Index> steepest;
    for (Index column = 0; column < gradients.size(); ++column) {
        if (!is_active(column) && (!steepest || gradients(column) > gradients(*steepest))) {
            steepest = column;
        }
    }
    return steepest;
}

// Returns the indices of the active columns, in increasing order.
std::vector<Index> active_indices(const ActiveSet& is_active) {
    std::vector<Index> indices;
    for (Index column = 0; column < is_active.size(); ++column) {
        if (is_active(column)) {
            indices.push_back(column);
        }
    }
    return indices;
}

// Returns the least-squares coefficients of data on the columns of active_design, (k,).
//
// The columns after the numerical rank of the pivoted QR are dependent on the earlier ones and
// get zero coefficients.
Vector least_squares(const Matrix& active_design, const VectorView& data) {
    Eigen::ColPivHouseholderQR<Matrix> qr(active_design);
    // A pivot below max(rows, columns) * eps of the largest is rounding error, so a column equal
    // to others up to rounding counts as dependent.
    qr.setThreshold(static_cast<double>(std::max(active_design.rows(), active_design.cols())) *
                    std::numeric_limits<double>::epsilon());
    const Index rank = qr.rank();

    // R11 b = (Q^T d) on the first rank pivots, in pivot order.
    Vector rotated = data;
    rotated.applyOnTheLeft(qr.householderQ().setLength(rank).adjoint());
    const Vector pivot_coefficients = qr.matrixR()
                                          .topLeftCorner(rank, rank)
                                          .triangularView<Eigen::Upper>()
                                          .solve(rotated.head(rank));
    Vector coefficients = Vector::Zero(active_design.cols());
    for (Index pivot = 0; pivot < rank; ++pivot) {
        coefficients(qr.colsPermutation().indices()(pivot)) = pivot_coefficients(pivot);
    }
    return coefficients;
}

// Spreads the coefficients of the active columns into a vector of all n_columns, zeros elsewhere.
Vector scatter(const Vector& coefficients, const std::vector<Index>& indices, Index n_columns) {
    Vector full = Vector::Zero(n_columns);
    for (std::size_t position = 0; position < indices.size(); ++position) {
        full(indices[position]) = coefficients(static_cast<Index>(position));
    }
    return full;
}

// A step from solution towards candidate, and the active column whose coefficient it brings
// to zero.
struct FeasibleStep {
    double step = 0.0;
    Index blocking_column = 0;
};

// Returns how far solution may move towards candidate before an active coefficient turns
// negative, the smallest solution / (solution - candidate) over the active columns with
// candidate <= 0, the first on a tie. A column at zero in both gives a step of zero.
FeasibleStep largest_feasible_step(const Vector& solution, const Vector& candidate,
                                   const ActiveSet& is_active) {
    FeasibleStep feasible{.step = std::numeric_limits<double>::infinity(), .blocking_column = 0};
    for (Index column = 0; column < solution.size(); ++column) {
        if (is_active(column) && candidate(column) <= 0.0) {
            const double distance = solution(column) - candidate(column);
            const double step = distance > 0.0 ? solution(column) / distance : 0.0;
            if (step < feasible.step) {
                feasible = {.step = step, .blocking_column = column};
            }
        }
    }
    return feasible;
}

} // namespace

double default_gradient_tolerance(const MatrixView& design) {
    const double norm1 = design.size() == 0 ? 0.0 : design.cwiseAbs().colwise().sum().maxCoeff();
    return 10.0 * std::numeric_limits<double>::epsilon() * norm1 *
           static_cast<double>(std::max(design.rows(), design.cols()));
}

namespace {

// Runs the Lawson-Hanson active-set loop. gradient_at(x) gives C^T (d - C x) and fit_on(active)
// the least-squares fit on the active columns, zero elsewhere.
template <class GradientAt, class FitOn>
NnlsResult lawson_hanson(Index n_columns, double gradient_tolerance, Index max_inner_steps,
                         GradientAt gradient_at, FitOn fit_on) {
    ActiveSet is_active = ActiveSet::Constant(n_columns, false);
    Vector solution = Vector::Zero(n_columns);
    Vector gradients = gradient_at(solution);
    Index n_inner_steps = 0;

    while (true) {
        const std::optional<Index> entering = steepest_inactive_column(gradients, is_active);
        if (!entering || gradients(*entering) <= gradient_tolerance) {
            break;
        }
        is_active(*entering) = true;
        Vector candidate = fit_on(is_active);

        while ((is_active && candidate.array() <= 0.0).any()) {
            if (++n_inner_steps > max_inner_steps) {
                // Too many inner steps: the last least-squares fit is the answer.
                return {.solution = candidate, .converged = false};
            }
            const FeasibleStep feasible = largest_feasible_step(solution, candidate, is_active);
            solution += feasible.step * (candidate - solution);
            // The blocking coefficient is zero by construction; rounding must not keep it
            // active, which would happen with a zero tolerance.
            solution(feasible.blocking_column) = 0.0;
            is_active(feasible.blocking_column) = false;
            is_active = is_active && solution.array().abs() >= gradient_tolerance;
            candidate = fit_on(is_active);
        }
        solution = candidate;
        gradients = gradient_at(solution);
    }
    return {.solution = solution, .converged = true};
}

// Returns the inner-step limit of solve_nnls and solve_ridge_nnls.
Index default_inner_steps(Index n_columns) {
    return 3 * n_columns;
}

// The largest bound on the condition number of the Gram matrix, (||K||_F^2 + ridge^2) / ridge^2,
// for which the ridge fits use the normal equations.
constexpr double MAX_NORMAL_EQUATIONS_CONDITION = 1e8;

// The Cholesky factor L of the Gram matrix on the active columns, G_AA = L L^T, with the columns
// in the order they joined the factor.
class ActiveCholesky {
public:
    explicit ActiveCholesky(const Matrix& gram_matrix)
        : gram(gram_matrix), factor(gram_matrix.rows(), gram_matrix.rows()),
          in_factor(ActiveSet::Constant(gram_matrix.rows(), false)) {}

    // Brings the factor to the columns of is_active, appending a column that joined and
    // rebuilding it if any column left. Returns false when a pivot is not positive, the Gram
    // matrix being singular on those columns.
    bool update(const ActiveSet& is_active) {
        if ((in_factor && !is_active).any()) {
            std::vector<Index> kept;
            for (const Index column : columns) {
                if (is_active(column)) {
                    kept.push_back(column);
                }
            }
            columns.clear();
            in_factor.setConstant(false);
            for (const Index column : kept) {
                if (!append(column)) {
                    return false;
                }
            }
        }
        for (Index column = 0; column < is_active.size(); ++column) {
            if (is_active(column) && !in_factor(column) && !append(column)) {
                return false;
            }
        }
        return true;
    }

    // Returns the solution of G_AA z = rhs_A, spread into a vector of all columns.
    [[nodiscard]] Vector solve(const Vector& rhs) const {
        const auto n_active = static_cast<Index>(columns.size());
        Vector active_rhs(n_active);
        for (Index position = 0; position < n_active; ++position) {
            active_rhs(position) = rhs(columns[static_cast<std::size_t>(position)]);
        }
        const auto lower = factor.topLeftCorner(n_active, n_active).triangularView<Eigen::Lower>();
        lower.solveInPlace(active_rhs);
        lower.transpose().solveInPlace(active_rhs);
        return scatter(active_rhs, columns, rhs.size());
    }

private:
    // Adds a row to the factor for column, L w = G_A,column, then sqrt(G_column,column - w.w).
    bool append(Index column) {
        const auto position = static_cast<Index>(columns.size());
        Vector cross(position);
        for (Index row = 0; row < position; ++row) {
            cross(row) = gram(columns[static_cast<std::size_t>(row)], column);
        }
        factor.topLeftCorner(position, position).triangularView<Eigen::Lower>().solveInPlace(cross);
        const double pivot = gram(column, column) - cross.squaredNorm();
        if (!(pivot > 0.0)) {
            return false;
        }
        factor.row(position).head(position) = cross.transpose();
        factor(position, position) = std::sqrt(pivot);
        columns.push_back(column);
        in_factor(column) = true;
        return true;
    }

    const Matrix& gram;
    Matrix factor;              // lower triangular, its top-left corner holds L
    std::vector<Index> columns; // the active columns, in factor order
    ActiveSet in_factor;
};

} // namespace

Vector solve_nnls(const MatrixView& design, const VectorView& data, double gradient_tolerance) {
    return solve_nnls_with_limit(design, data, gradient_tolerance,
                                 default_inner_steps(design.cols()))
        .solution;
}

NnlsResult solve_nnls_with_limit(const MatrixView& design, const VectorView& data,
                                 double gradient_tolerance, Index max_inner_steps) {
    const Index n_columns = design.cols();
    const auto fit_on = [&](const ActiveSet& is_active) {
        const std::vector<Index> indices = active_indices(is_active);
        if (indices.empty()) {
            return Vector::Zero(n_columns).eval();
        }
        const Matrix active_design = design(Eigen::placeholders::all, indices);
        return scatter(least_squares(active_design, data), indices, n_columns);
    };
    return lawson_hanson(
        n_columns, gradient_tolerance, max_inner_steps,
        [&](const Vector& solution) { return gradient(design, data, solution); }, fit_on);
}

RidgeDesign make_ridge_design(const MatrixView& kernel, double ridge) {
    Matrix gram = kernel.transpose() * kernel;
    gram.diagonal().array() += ridge * ridge;
    const double ridge_squared = ridge * ridge;
    const bool normal_equations =
        ridge_squared > 0.0 &&
        kernel.squaredNorm() + ridge_squared <= MAX_NORMAL_EQUATIONS_CONDITION * ridge_squared;
    return {.kernel = kernel,
            .gram = std::move(gram),
            .ridge = ridge,
            .normal_equations = normal_equations};
}

Vector solve_ridge_nnls(const RidgeDesign& design, const VectorView& signal,
                        double gradient_tolerance) {
    const Index n_measurements = design.kernel.rows();
    const Index n_columns = design.kernel.cols();
    // C^T d, with C = [K; ridge I] and d = [s; 0].
    const Vector correlations = design.kernel.transpose() * signal;

    // C^T (d - C x) = K^T s - (K^T K + ridge^2 I) x, over the non-zero coefficients only.
    const auto gradient_at = [&](const Vector& solution) {
        Vector gradients = correlations;
        for (Index column = 0; column < n_columns; ++column) {
            if (solution(column) != 0.0) {
                gradients -= solution(column) * design.gram.col(column);
            }
        }
        return gradients;
    };

    // The active columns of C: the kernel's, and under them ridge I on those columns only (the
    // ridge rows of inactive columns are zero there and do not change the fit).
    const auto fit_by_qr = [&](const ActiveSet& is_active) {
        const std::vector<Index> indices = active_indices(is_active);
        const auto n_active = static_cast<Index>(indices.size());
        if (n_active == 0) {
            return Vector::Zero(n_columns).eval();
        }
        const bool has_ridge = design.ridge > 0.0;
        Matrix active_design = Matrix::Zero(n_measurements + (has_ridge ? n_active : 0), n_active);
        active_design.topRows(n_measurements) = design.kernel(Eigen::placeholders::all, indices);
        Vector data = Vector::Zero(active_design.rows());
        data.head(n_measurements) = signal;
        if (has_ridge) {
            active_design.bottomRows(n_active).diagonal().setConstant(design.ridge);
        }
        return scatter(least_squares(active_design, data), indices, n_columns);
    };
    if (!design.normal_equations) {
        return lawson_hanson(n_columns, gradient_tolerance, default_inner_steps(n_columns),
                             gradient_at, fit_by_qr)
            .solution;
    }

    // (K_A^T K_A + ridge^2 I) x_A = K_A^T s; QR from the first fit whose factor fails.
    ActiveCholesky cholesky(design.gram);
    bool factor_failed = false;
    const auto fit_on = [&](const ActiveSet& is_active) {
        factor_failed = factor_failed || !cholesky.update(is_active);
        return factor_failed ? fit_by_qr(is_active) : cholesky.solve(correlations);
    };
    return lawson_hanson(n_columns, gradient_tolerance, default_inner_steps(n_columns), gradient_at,
                         fit_on)
        .solution;
}

} // namespace larmor::sd
