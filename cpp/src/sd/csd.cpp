#include "sd/csd.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>

namespace larmor::sd {

CsdProblem make_csd_problem(Matrix initial, Matrix forward, Matrix constraint, double norm_weight,
                            double threshold, Index max_iterations) {
    Matrix gram = forward.transpose() * forward;
    gram.diagonal().array() += norm_weight;
    return CsdProblem{.initial = std::move(initial),
                      .forward = std::move(forward),
                      .constraint = std::move(constraint),
                      .gram = std::move(gram),
                      .threshold = threshold,
                      .max_iterations = max_iterations};
}

CsdResult solve_csd(const CsdProblem& problem, const VectorView& signal) {
    const Index n_coefficients = problem.forward.cols();
    Vector fod = Vector::Zero(n_coefficients);
    fod.head(problem.initial.rows()) = problem.initial * signal;
    const Vector projected = problem.forward.transpose() * signal;

    std::vector<Index> constrained;
    std::vector<Index> previous;
    bool first = true;
    Matrix system(n_coefficients, n_coefficients);
    Matrix rows(problem.constraint.rows(), n_coefficients);
    Eigen::LLT<Matrix> cholesky(n_coefficients);
    bool converged = problem.max_iterations == 0;
    for (Index iteration = 0; iteration < problem.max_iterations; ++iteration) {
        const Vector amplitudes = problem.constraint * fod;
        constrained.clear();
        for (Index direction = 0; direction < amplitudes.size(); ++direction) {
            if (amplitudes(direction) < problem.threshold) {
                constrained.push_back(direction);
            }
        }
        // The same set gives the same solve, so the FOD is final.
        if (!first && constrained == previous) {
            converged = true;
            break;
        }
        first = false;

        system = problem.gram;
        const auto n_constrained = static_cast<Index>(constrained.size());
        if (n_constrained > 0) {
            for (Index row = 0; row < n_constrained; ++row) {
                rows.row(row) = problem.constraint.row(constrained[static_cast<std::size_t>(row)]);
            }
            const auto active = rows.topRows(n_constrained);
            system.noalias() += active.transpose() * active;
        }
        cholesky.compute(system);
        fod = cholesky.solve(projected);
        std::swap(previous, constrained);
    }
    return CsdResult{.fod = std::move(fod), .converged = converged};
}

} // namespace larmor::sd
