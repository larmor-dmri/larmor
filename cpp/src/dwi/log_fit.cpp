#include "dwi/log_fit.hpp"

#include <limits>
#include <utility>

#include <Eigen/QR>

#include "sd/nnls.hpp"

namespace larmor::dwi {

namespace {

// How far below zero the last NNLS residual must be for the least-distance problem to have a
// solution.
constexpr double FEASIBILITY_MARGIN = 1e-12;
// Maximum number of inner steps of the NNLS solve per constraint.
constexpr Index INNER_STEPS_PER_CONSTRAINT = 50;
// How far a constrained solution may break a constraint and still meet it, relative to the
// largest term of the voxel's constraints at the solution and at its unconstrained start.
constexpr double CONSTRAINT_TOLERANCE = 1e-9;

// Runs the weighted least-squares iteration from the ordinary fit and returns the scaled
// parameters (P,), or `ordinary` where a weighted fit is singular or not finite.
//
// The weights S^2 are taken relative to the largest predicted signal of the voxel, so that exp
// cannot overflow; a common factor does not change the fit.
Vector reweighted_fit(const LogSignalDesign& design, const LogFitSettings& settings,
                      const VectorView& log_signal, const Vector& ordinary,
                      LogFitScratch& scratch) {
    if (!ordinary.allFinite()) {
        return ordinary;
    }
    const Matrix& scaled_design = design.scaled_design;
    Vector parameters = ordinary;
    Vector predicted = scaled_design * parameters;
    for (Index iteration = 0; iteration < settings.max_reweightings; ++iteration) {
        const Vector weights = (2.0 * (predicted.array() - predicted.maxCoeff())).exp().matrix();
        scratch.weighted_design.noalias() = weights.asDiagonal() * scaled_design;
        scratch.normal_matrix.noalias() = scaled_design.transpose() * scratch.weighted_design;
        const Vector right_hand_side = scratch.weighted_design.transpose() * log_signal;
        // LU with partial pivoting. Normal equations are singular when weights underflow to
        // zero: a pivot is then exactly zero, and the weighted fit has no unique solution.
        scratch.normal_lu.compute(scratch.normal_matrix);
        if ((scratch.normal_lu.matrixLU().diagonal().array() == 0.0).any()) {
            return ordinary;
        }
        const Vector next = scratch.normal_lu.solve(right_hand_side);
        if (!next.allFinite()) {
            return ordinary;
        }
        const Vector next_predicted = scaled_design * next;
        const double change = (next_predicted - predicted).cwiseAbs().maxCoeff();
        parameters = next;
        predicted = next_predicted;
        if (!(change > settings.log_signal_tolerance)) {
            break;
        }
    }
    return parameters;
}

// Returns whether scaled parameters (P,) break a constraint, G p < 0 in some row.
bool breaks_a_constraint(const LogSignalDesign& design, const Vector& parameters) {
    return (design.scaled_constraints * parameters).minCoeff() < 0.0;
}

// Returns whether scaled parameters (P,), computed from the unconstrained `start` (P,), meet
// every constraint, G p >= 0, up to rounding.
bool meets_the_constraints(const LogSignalDesign& design, const Vector& parameters,
                           const Vector& start) {
    const Vector values = design.scaled_constraints * parameters;
    const double size =
        (design.scaled_constraints.cwiseAbs() * (parameters.cwiseAbs() + start.cwiseAbs()))
            .maxCoeff();
    return values.minCoeff() >= -CONSTRAINT_TOLERANCE * size;
}

// Fits a voxel again subject to the constraints and returns its scaled parameters (P,).
//
// `parameters` is its unconstrained solution, which breaks a constraint, and `root_weights` (N,)
// the square roots of its weights. Where no point meets the constraints or the solve fails,
// `parameters` is returned unchanged.
Vector constrained_fit(const LogSignalDesign& design, const VectorView& log_signal,
                       const Vector& root_weights, const Vector& parameters) {
    const Index n_parameters = design.n_parameters();
    const Index n_constraints = design.scaled_constraints.rows();

    // W^(1/2) X = Q R, and the unconstrained solution p_free = R^-1 Q^T W^(1/2) y.
    const Matrix weighted_design = root_weights.asDiagonal() * design.scaled_design;
    const Eigen::HouseholderQR<Matrix> qr(weighted_design);
    Vector rotated = root_weights.cwiseProduct(log_signal);
    rotated.applyOnTheLeft(qr.householderQ().adjoint());
    const auto upper = qr.matrixQR().topRows(n_parameters).triangularView<Eigen::Upper>();
    // Weights that underflow to zero on most measurements leave the weighted design without full
    // rank: a diagonal element of R is then zero up to rounding, and there is no unique solution.
    const Vector diagonal = qr.matrixQR().diagonal().cwiseAbs();
    const double rank_tolerance = static_cast<double>(weighted_design.rows()) *
                                  std::numeric_limits<double>::epsilon() * diagonal.maxCoeff();
    if (!(diagonal.minCoeff() > rank_tolerance)) {
        return parameters;
    }
    const Vector free_solution = upper.solve(rotated.head(n_parameters));

    // With z = R (p - p_free): min ||z|| subject to A z >= c, A = G R^-1 and c = -G p_free. The
    // NNLS matrix stacks A^T (P, M), computed as R^-T G^T, over c^T.
    Matrix least_distance(n_parameters + 1, n_constraints);
    least_distance.topRows(n_parameters) =
        upper.transpose().solve(design.scaled_constraints.transpose());
    least_distance.row(n_parameters) = -(design.scaled_constraints * free_solution).transpose();
    Vector unit = Vector::Zero(n_parameters + 1);
    unit(n_parameters) = 1.0;

    const sd::NnlsResult multipliers = sd::solve_nnls_with_limit(
        least_distance, unit, sd::default_gradient_tolerance(least_distance),
        INNER_STEPS_PER_CONSTRAINT * n_constraints);
    if (!multipliers.converged) {
        return parameters;
    }
    const Vector residual = least_distance * multipliers.solution - unit;
    if (residual(n_parameters) >= -FEASIBILITY_MARGIN) {
        return parameters;
    }
    const Vector distance = -residual.head(n_parameters) / residual(n_parameters);
    Vector constrained = free_solution + upper.solve(distance);
    if (!constrained.allFinite() || !meets_the_constraints(design, constrained, free_solution)) {
        return parameters;
    }
    return constrained;
}

} // namespace

LogSignalDesign make_log_signal_design(const MatrixView& design, const MatrixView& constraints) {
    Vector column_scales = design.colwise().norm().transpose();
    for (Index column = 0; column < column_scales.size(); ++column) {
        if (column_scales(column) == 0.0) {
            column_scales(column) = 1.0;
        }
    }
    const Matrix inverse_scales = column_scales.cwiseInverse().asDiagonal();
    Matrix scaled_design = design * inverse_scales;
    Matrix pseudo_inverse =
        Eigen::CompleteOrthogonalDecomposition<Matrix>(scaled_design).pseudoInverse();
    Matrix scaled_constraints = constraints * inverse_scales;
    return {.column_scales = std::move(column_scales),
            .scaled_design = std::move(scaled_design),
            .pseudo_inverse = std::move(pseudo_inverse),
            .scaled_constraints = std::move(scaled_constraints)};
}

LogFitScratch make_log_fit_scratch(const LogSignalDesign& design) {
    const Index n_parameters = design.n_parameters();
    return {.weighted_design = Matrix(design.n_measurements(), n_parameters),
            .normal_matrix = Matrix(n_parameters, n_parameters),
            .normal_lu = Eigen::PartialPivLU<Matrix>(n_parameters)};
}

void fit_log_signal_block(const LogSignalDesign& design, const LogFitSettings& settings,
                          const MatrixView& block_log_signals, LogFitScratch& scratch,
                          Eigen::Ref<Matrix> block_parameters,
                          Eigen::Ref<Flags> block_constrained) {
    for (Index voxel = 0; voxel < block_log_signals.rows(); ++voxel) {
        const Vector log_signal = block_log_signals.row(voxel).transpose();
        const Vector ordinary = design.pseudo_inverse * log_signal;
        Vector parameters = settings.weighted
                                ? reweighted_fit(design, settings, log_signal, ordinary, scratch)
                                : ordinary;

        const bool constrain = design.is_constrained() && parameters.allFinite() &&
                               breaks_a_constraint(design, parameters);
        if (constrain) {
            Vector root_weights = Vector::Ones(design.n_measurements());
            if (settings.weighted) {
                const Vector predicted = design.scaled_design * parameters;
                root_weights = (predicted.array() - predicted.maxCoeff()).exp().matrix();
            }
            parameters = constrained_fit(design, log_signal, root_weights, parameters);
        }
        block_parameters.row(voxel) = parameters.cwiseQuotient(design.column_scales).transpose();
        block_constrained(voxel) = constrain;
    }
}

} // namespace larmor::dwi
