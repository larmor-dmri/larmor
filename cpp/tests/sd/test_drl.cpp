#include <cmath>

#include <doctest/doctest.h>

#include "common/types.hpp"
#include "sd/drl.hpp"

namespace {

using larmor::Index;
using larmor::Matrix;

constexpr Index N_MEASUREMENTS = 12;
constexpr Index N_DIRECTIONS = 8;
constexpr Index N_VOXELS = 5;

// With N_SHORT iterations one voxel is still iterating at the end; with N_LONG every voxel
// converges, at different iterations.
constexpr Index N_SHORT = 1000;
constexpr Index N_LONG = 5000;

// A small positive kernel, (m, c).
Matrix toy_kernel() {
    Matrix kernel(N_MEASUREMENTS, N_DIRECTIONS);
    for (Index row = 0; row < N_MEASUREMENTS; ++row) {
        for (Index column = 0; column < N_DIRECTIONS; ++column) {
            kernel(row, column) = 1.0 + static_cast<double>((row * 7 + column * 3) % 11) / 11.0;
        }
    }
    return kernel;
}

// Signals made from different FODs, so the voxels converge at different iterations.
Matrix toy_signals(const Matrix& kernel) {
    Matrix fods(N_VOXELS, N_DIRECTIONS);
    for (Index voxel = 0; voxel < N_VOXELS; ++voxel) {
        for (Index direction = 0; direction < N_DIRECTIONS; ++direction) {
            fods(voxel, direction) = direction == voxel % N_DIRECTIONS
                                         ? 1.0 + static_cast<double>(voxel)
                                         : 0.05 * static_cast<double>(1 + (voxel + direction) % 3);
        }
    }
    return fods * kernel.transpose() / static_cast<double>(N_MEASUREMENTS);
}

// Fits signals as one block, starting every voxel from 1 / c.
Matrix fit(const Matrix& kernel, const Matrix& signals, Index n_iterations) {
    const larmor::sd::RlKernel rl_kernel{.transposed = kernel.transpose(),
                                         .gram = kernel.transpose() * kernel};
    const larmor::sd::DrlSettings settings{.damping_threshold = 0.05, .n_iterations = n_iterations};
    Matrix fods = Matrix::Constant(signals.rows(), N_DIRECTIONS, 1.0 / N_DIRECTIONS);
    larmor::sd::damped_richardson_lucy(rl_kernel, settings, signals, fods);
    return fods;
}

} // namespace

TEST_CASE("drl: a block gives what each voxel gives alone") {
    const Matrix kernel = toy_kernel();
    const Matrix signals = toy_signals(kernel);
    for (const Index n_iterations : {N_SHORT, N_LONG}) {
        const Matrix together = fit(kernel, signals, n_iterations);
        for (Index voxel = 0; voxel < N_VOXELS; ++voxel) {
            const Matrix alone = fit(kernel, signals.middleRows(voxel, 1), n_iterations);
            // Equal to rounding: a product over 5 rows and one over 1 row can sum in different
            // orders. A packing bug differs by far more.
            for (Index direction = 0; direction < N_DIRECTIONS; ++direction) {
                CHECK(std::fabs(together(voxel, direction) - alone(0, direction)) <=
                      1e-12 * std::fabs(alone(0, direction)));
            }
        }
    }
}

TEST_CASE("drl: the test data converge at different iterations") {
    // A voxel that has converged gives the same FOD for any larger number of iterations; one
    // that has not keeps changing. Some of each means rows leave the block while others are
    // still iterating, which is what the test above needs to exercise.
    const Matrix kernel = toy_kernel();
    const Matrix signals = toy_signals(kernel);
    const Matrix short_fit = fit(kernel, signals, N_SHORT);
    const Matrix long_fit = fit(kernel, signals, N_LONG);
    Index n_converged = 0;
    for (Index voxel = 0; voxel < N_VOXELS; ++voxel) {
        n_converged += short_fit.row(voxel) == long_fit.row(voxel) ? 1 : 0;
    }
    CHECK(n_converged > 0);
    CHECK(n_converged < N_VOXELS);
}

TEST_CASE("drl: the FOD explains the signal") {
    // dRL stops once a step changes the FOD by less than 1e-4 of the signal's maximum, and
    // this small kernel is badly conditioned, so the fit is close but not exact: numpy's
    // solver leaves the same residuals, 1.4% at worst.
    const Matrix kernel = toy_kernel();
    const Matrix signals = toy_signals(kernel);
    const Matrix predicted = fit(kernel, signals, N_LONG) * kernel.transpose();
    for (Index voxel = 0; voxel < N_VOXELS; ++voxel) {
        const double error =
            (predicted.row(voxel) - signals.row(voxel)).cwiseAbs().sum() / signals.row(voxel).sum();
        CHECK(error < 2e-2);
    }
}

TEST_CASE("drl: the FOD stays above the floor") {
    const Matrix kernel = toy_kernel();
    CHECK((fit(kernel, toy_signals(kernel), N_LONG).array() >= 1e-6).all());
}
