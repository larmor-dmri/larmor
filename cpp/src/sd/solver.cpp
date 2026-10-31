#include "sd/solver.hpp"

#include <utility>

#include "sd/nnls.hpp"

namespace larmor::sd {

void DrlSolver::solve_block(const MatrixView& block_signals, Eigen::Ref<Matrix> block_fods) const {
    block_fods.setConstant(initial_amplitude);
    damped_richardson_lucy(kernel, settings, block_signals, block_fods);
}

void L2NnlsSolver::solve_block(const MatrixView& block_signals,
                               Eigen::Ref<Matrix> block_fods) const {
    for (Index voxel = 0; voxel < block_signals.rows(); ++voxel) {
        const Vector signal = block_signals.row(voxel).transpose();
        block_fods.row(voxel) =
            solve_ridge_nnls(design, signal, 0.0, gradient_tolerance).transpose();
    }
}

void ElasticNetSolver::solve_block(const MatrixView& block_signals,
                                   Eigen::Ref<Matrix> block_fods) const {
    Index n_unconverged = 0;
    for (Index voxel = 0; voxel < block_signals.rows(); ++voxel) {
        const Vector signal = block_signals.row(voxel).transpose();
        const ElasticNetResult result = solve_elastic_net(design, signal, settings);
        block_fods.row(voxel) = result.coefficients.transpose();
        n_unconverged += result.converged ? 0 : 1;
    }
    if (n_unconverged > 0) {
        unconverged->fetch_add(n_unconverged);
    }
}

Index FodSolver::n_directions() const {
    return std::visit([](const auto& solver) { return solver.n_directions(); }, method);
}

Index FodSolver::n_measurements() const {
    return std::visit([](const auto& solver) { return solver.n_measurements(); }, method);
}

Index FodSolver::n_unconverged() const {
    if (const auto* elastic_net = std::get_if<ElasticNetSolver>(&method)) {
        return elastic_net->unconverged->load();
    }
    return 0;
}

void FodSolver::solve_block(const MatrixView& block_signals, Eigen::Ref<Matrix> block_fods) const {
    std::visit([&](const auto& solver) { solver.solve_block(block_signals, block_fods); }, method);
}

L2NnlsSolver make_l2nnls_solver(const MatrixView& kernel, double ridge, double gradient_tolerance) {
    return L2NnlsSolver{.design = make_ridge_design(kernel, ridge),
                        .gradient_tolerance = gradient_tolerance};
}

} // namespace larmor::sd
