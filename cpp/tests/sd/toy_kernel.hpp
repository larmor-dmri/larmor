// A small multi-tissue kernel and a dRL solver on it, shared by the fraction and multi-tissue
// tests: 12 measurements on three shells, one anisotropic response function and two
// isotropic decays.
#pragma once

#include <cmath>

#include "common/types.hpp"
#include "sd/fractions.hpp"
#include "sd/solver.hpp"

namespace larmor::sd::testing {

inline constexpr Index N_MEASUREMENTS = 12;
inline constexpr Index N_ISOTROPIC = 2;

/// The shell weight of each measurement.
inline Vector toy_weights() {
    Vector weights(N_MEASUREMENTS);
    for (Index measurement = 0; measurement < N_MEASUREMENTS; ++measurement) {
        weights(measurement) = 1.0 + static_cast<double>(measurement % 3) * 0.5;
    }
    return weights;
}

/// A kernel with n_anisotropic columns of one response function and N_ISOTROPIC decays.
inline TissueKernel toy_kernel(Index n_anisotropic) {
    TissueKernel kernel{
        .anisotropic = Matrix(N_MEASUREMENTS, n_anisotropic),
        .isotropic = Matrix(N_MEASUREMENTS, N_ISOTROPIC),
        .weighted_isotropic = Matrix(N_MEASUREMENTS, N_ISOTROPIC),
        .anisotropic_blocks = {{.begin = 0, .end = n_anisotropic}},
    };
    for (Index measurement = 0; measurement < N_MEASUREMENTS; ++measurement) {
        const auto shell = static_cast<double>(measurement / 4);
        for (Index column = 0; column < n_anisotropic; ++column) {
            kernel.anisotropic(measurement, column) =
                std::exp(-0.3 * shell *
                         (1.0 + static_cast<double>((measurement * 5 + column * 3) % 7) / 3.0));
        }
        for (Index tissue = 0; tissue < N_ISOTROPIC; ++tissue) {
            kernel.isotropic(measurement, tissue) =
                std::exp(-shell * (0.4 + 1.2 * static_cast<double>(tissue)));
        }
    }
    kernel.weighted_isotropic = toy_weights().asDiagonal() * kernel.isotropic;
    return kernel;
}

/// A dRL solver on the weighted anisotropic columns of kernel, starting near zero.
inline FodSolver toy_drl(const TissueKernel& kernel, double damping_threshold, Index n_iterations) {
    const Matrix weighted = toy_weights().asDiagonal() * kernel.anisotropic;
    return FodSolver(DrlSolver{
        .kernel = {.transposed = weighted.transpose(), .gram = weighted.transpose() * weighted},
        .settings = {.damping_threshold = damping_threshold, .n_iterations = n_iterations},
        .initial_amplitude = 1e-6,
    });
}

} // namespace larmor::sd::testing
