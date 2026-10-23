#include <cmath>
#include <utility>

#include <doctest/doctest.h>

#include "common/types.hpp"
#include "sd/multi_tissue.hpp"
#include "toy_kernel.hpp"

namespace {

using larmor::Index;
using larmor::Matrix;
using larmor::sd::testing::N_ISOTROPIC;
using larmor::sd::testing::N_MEASUREMENTS;

constexpr Index N_VOXELS = 5;
constexpr Index N_LOW_RESOLUTION = 6;
constexpr Index N_HIGH_RESOLUTION = 10;

// Voxels mixing one anisotropic direction with the two isotropic decays in different
// proportions.
Matrix toy_signals(const larmor::sd::TissueKernel& high_resolution) {
    Matrix signals(N_VOXELS, N_MEASUREMENTS);
    for (Index voxel = 0; voxel < N_VOXELS; ++voxel) {
        const double white_matter = static_cast<double>(voxel + 1) / N_VOXELS;
        for (Index measurement = 0; measurement < N_MEASUREMENTS; ++measurement) {
            signals(voxel, measurement) =
                white_matter *
                    high_resolution.anisotropic(measurement, (2 * voxel) % N_HIGH_RESOLUTION) +
                (1.0 - white_matter) * 0.7 * high_resolution.isotropic(measurement, 0) +
                (1.0 - white_matter) * 0.3 * high_resolution.isotropic(measurement, 1);
        }
    }
    return signals;
}

// The FODs and fractions of signals, fitted as one block.
std::pair<Matrix, Matrix> fit(const Matrix& signals) {
    const larmor::sd::TissueKernel low_kernel = larmor::sd::testing::toy_kernel(N_LOW_RESOLUTION);
    const larmor::sd::TissueKernel high_kernel = larmor::sd::testing::toy_kernel(N_HIGH_RESOLUTION);
    const larmor::sd::FodSolver low_solver = larmor::sd::testing::toy_drl(low_kernel, 0.05, 199);
    const larmor::sd::FodSolver high_solver = larmor::sd::testing::toy_drl(high_kernel, 0.05, 599);
    const larmor::sd::LoopSettings settings{
        .max_rounds = 50, .fraction_tolerance = 1e-3, .final_gradient_tolerance = 1e-2};
    Matrix fods(signals.rows(), N_HIGH_RESOLUTION);
    Matrix fractions(signals.rows(), 1 + N_ISOTROPIC);
    larmor::sd::fit_multi_tissue_block(
        signals, larmor::sd::testing::toy_weights(), {.kernel = low_kernel, .solver = low_solver},
        {.kernel = high_kernel, .solver = high_solver}, settings, fods, fractions);
    return {fods, fractions};
}

} // namespace

TEST_CASE("multi-tissue: a block gives what each voxel gives alone") {
    const Matrix signals = toy_signals(larmor::sd::testing::toy_kernel(N_HIGH_RESOLUTION));
    const auto [fods, fractions] = fit(signals);
    for (Index voxel = 0; voxel < N_VOXELS; ++voxel) {
        const auto [voxel_fods, voxel_fractions] = fit(signals.middleRows(voxel, 1));
        // Equal to rounding: products over different numbers of rows can sum in different
        // orders. A packing bug differs by far more.
        for (Index direction = 0; direction < N_HIGH_RESOLUTION; ++direction) {
            CHECK(std::fabs(fods(voxel, direction) - voxel_fods(0, direction)) <=
                  1e-12 * std::fabs(voxel_fods(0, direction)));
        }
        for (Index tissue = 0; tissue < 1 + N_ISOTROPIC; ++tissue) {
            CHECK(std::fabs(fractions(voxel, tissue) - voxel_fractions(0, tissue)) <= 1e-12);
        }
    }
}

TEST_CASE("multi-tissue: the fractions are non-negative and sum to one") {
    const auto [fods, fractions] =
        fit(toy_signals(larmor::sd::testing::toy_kernel(N_HIGH_RESOLUTION)));
    CHECK((fractions.array() >= 0.0).all());
    for (Index voxel = 0; voxel < N_VOXELS; ++voxel) {
        CHECK(std::fabs(fractions.row(voxel).sum() - 1.0) < 1e-12);
    }
}

TEST_CASE("multi-tissue: more white matter in the signal gives a larger white matter fraction") {
    const auto [fods, fractions] =
        fit(toy_signals(larmor::sd::testing::toy_kernel(N_HIGH_RESOLUTION)));
    for (Index voxel = 1; voxel < N_VOXELS; ++voxel) {
        CHECK(fractions(voxel, 0) > fractions(voxel - 1, 0));
    }
}
