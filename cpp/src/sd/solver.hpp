// The FOD solvers, behind one entry point that the single- and multi-tissue fits call.
#pragma once

#include <atomic>
#include <memory>
#include <utility>
#include <variant>

#include "common/types.hpp"
#include "sd/drl.hpp"
#include "sd/elastic_net.hpp"
#include "sd/nnls.hpp"

namespace larmor::sd {

/// Damped Richardson-Lucy, starting from the same value in every direction.
struct DrlSolver {
    /// The matrices of the kernel.
    RlKernel kernel;
    /// The iteration settings.
    DrlSettings settings;
    /// The starting FOD amplitude of every direction.
    double initial_amplitude = 0.0;

    /// Returns the number of FOD directions, c.
    [[nodiscard]] Index n_directions() const { return kernel.transposed.rows(); }

    /// Returns the number of measurements per voxel, m.
    [[nodiscard]] Index n_measurements() const { return kernel.transposed.cols(); }

    /// Fills block_fods (b, c) with the starting amplitude and runs damped Richardson-Lucy on
    /// block_signals (b, m).
    void solve_block(const MatrixView& block_signals, Eigen::Ref<Matrix> block_fods) const;
};

/// Non-negative least squares with a ridge penalty, where each voxel's FOD f minimises
/// ||K f - s||^2 + ridge^2 ||f||^2 subject to f >= 0.
struct L2NnlsSolver {
    /// The kernel with its Gram matrix, prepared once for every voxel.
    RidgeDesign design;
    /// A coefficient enters the NNLS solution only when its gradient exceeds this.
    double gradient_tolerance;

    /// Returns the number of FOD directions, c.
    [[nodiscard]] Index n_directions() const { return design.kernel.cols(); }
    /// Returns the number of measurements per voxel, m.
    [[nodiscard]] Index n_measurements() const { return design.kernel.rows(); }

    /// Fits the FOD of each voxel of a block. Row v of block_signals (b, m) is that voxel's
    /// signal, and its FOD (c,) goes to row v of block_fods (b, c).
    void solve_block(const MatrixView& block_signals, Eigen::Ref<Matrix> block_fods) const;
};

/// Elastic-net regression of each voxel's signal on the kernel's columns, scaled to unit root
/// mean square, without an intercept. The FOD is not constrained to be non-negative.
struct ElasticNetSolver {
    /// The kernel's columns, scaled, with their scales.
    StandardizedDesign design;
    /// The penalty and stopping settings.
    ElasticNetSettings settings;
    /// The number of solves so far that stopped at max_sweeps without converging, shared by
    /// every copy of the solver and every thread.
    std::shared_ptr<std::atomic<Index>> unconverged = std::make_shared<std::atomic<Index>>(0);

    /// Returns the number of FOD directions, c.
    [[nodiscard]] Index n_directions() const { return design.columns.cols(); }
    /// Returns the number of measurements per voxel, m.
    [[nodiscard]] Index n_measurements() const { return design.columns.rows(); }

    /// Fits the FOD of each voxel of a block. Row v of block_signals (b, m) is that voxel's
    /// signal, and its FOD (c,) goes to row v of block_fods (b, c).
    void solve_block(const MatrixView& block_signals, Eigen::Ref<Matrix> block_fods) const;
};

/// A way of solving for the FODs of a block of voxels, one of the methods above.
class FodSolver {
public:
    /// One of the solver methods.
    using Method = std::variant<DrlSolver, L2NnlsSolver, ElasticNetSolver>;

    /// Wraps a solver method.
    explicit FodSolver(Method chosen) : method(std::move(chosen)) {}

    /// Returns the number of FOD directions the solver returns, c.
    [[nodiscard]] Index n_directions() const;

    /// Returns the number of measurements per voxel the solver expects, m.
    [[nodiscard]] Index n_measurements() const;

    /// Solves for the FOD of each row of block_signals, (b, m), into the same row of
    /// block_fods, (b, c).
    void solve_block(const MatrixView& block_signals, Eigen::Ref<Matrix> block_fods) const;

    /// Returns the number of solves so far that stopped at their iteration limit; only the
    /// elastic net counts them, and the other methods return 0.
    [[nodiscard]] Index n_unconverged() const;

private:
    Method method;
};

/// Builds the solver from the weighted anisotropic kernel K, (m, c); a ridge of 0 is plain NNLS.
L2NnlsSolver make_l2nnls_solver(const MatrixView& kernel, double ridge, double gradient_tolerance);

} // namespace larmor::sd
