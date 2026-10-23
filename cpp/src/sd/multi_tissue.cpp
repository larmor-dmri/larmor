#include "sd/multi_tissue.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

namespace larmor::sd {

namespace {

// Keeps the normalisation of the final fractions finite when they are all zero.
constexpr double FRACTION_SUM_OFFSET = 1e-16;

// Returns the weighted signal (m,) minus the isotropic tissues' weighted signal at the given
// fractions (ci,), clipped at zero.
Vector anisotropic_signal(const VectorView& weighted_signal, const MatrixView& weighted_isotropic,
                          const VectorView& isotropic_fractions) {
    const Vector remainder = weighted_signal - weighted_isotropic * isotropic_fractions;
    return remainder.unaryExpr([](double value) { return std::fmax(value, 0.0); });
}

} // namespace

void fit_multi_tissue_block(const MatrixView& block_signals, const VectorView& weights,
                            const Resolution& low_resolution, const Resolution& high_resolution,
                            const LoopSettings& settings, Eigen::Ref<Matrix> block_fods,
                            Eigen::Ref<Matrix> block_fractions) {
    const Index n_voxels = block_signals.rows();
    const Index n_measurements = block_signals.cols();
    const Index n_isotropic = low_resolution.kernel.n_isotropic();
    const Index n_loop_fractions = 1 + n_isotropic;

    const Matrix weighted_signals = block_signals * weights.asDiagonal();
    Matrix isotropic_fractions = Matrix::Zero(n_voxels, n_isotropic);
    // The fractions of each voxel at its previous round; none at first.
    Matrix previous_fractions =
        Matrix::Constant(n_voxels, n_loop_fractions, std::numeric_limits<double>::infinity());
    // The voxels still in the loop, and those that stay in it for the next round.
    std::vector<Index> looping(static_cast<std::size_t>(n_voxels));
    std::iota(looping.begin(), looping.end(), Index{0});
    std::vector<Index> still_looping;

    for (Index round = 0; round < settings.max_rounds && !looping.empty(); ++round) {
        const auto n_looping = static_cast<Index>(looping.size());
        Matrix anisotropic_signals(n_looping, n_measurements);
        for (Index row = 0; row < n_looping; ++row) {
            const Index voxel = looping[static_cast<std::size_t>(row)];
            anisotropic_signals.row(row) =
                anisotropic_signal(weighted_signals.row(voxel).transpose(),
                                   low_resolution.kernel.weighted_isotropic,
                                   isotropic_fractions.row(voxel).transpose())
                    .transpose();
        }
        Matrix low_resolution_fods(n_looping, low_resolution.kernel.n_anisotropic());
        low_resolution.solver.solve_block(anisotropic_signals, low_resolution_fods);

        still_looping.clear();
        for (Index row = 0; row < n_looping; ++row) {
            const Index voxel = looping[static_cast<std::size_t>(row)];
            const Vector fractions = compute_fractions(
                low_resolution_fods.row(row).transpose(), low_resolution.kernel,
                block_signals.row(voxel).transpose(), Grouping::Lumped, std::nullopt);
            // The isotropic fractions are kept even when the voxel converges, so it leaves
            // the loop with those of its last round.
            isotropic_fractions.row(voxel) = fractions.tail(n_isotropic).transpose();
            const bool converged =
                ((fractions.transpose() - previous_fractions.row(voxel)).array().abs() <
                 settings.fraction_tolerance)
                    .all();
            if (!converged) {
                previous_fractions.row(voxel) = fractions.transpose();
                still_looping.push_back(voxel);
            }
        }
        std::swap(looping, still_looping);
    }

    Matrix anisotropic_signals(n_voxels, n_measurements);
    for (Index voxel = 0; voxel < n_voxels; ++voxel) {
        anisotropic_signals.row(voxel) =
            anisotropic_signal(weighted_signals.row(voxel).transpose(),
                               high_resolution.kernel.weighted_isotropic,
                               isotropic_fractions.row(voxel).transpose())
                .transpose();
    }
    high_resolution.solver.solve_block(anisotropic_signals, block_fods);

    for (Index voxel = 0; voxel < n_voxels; ++voxel) {
        const Vector fractions =
            compute_fractions(block_fods.row(voxel).transpose(), high_resolution.kernel,
                              block_signals.row(voxel).transpose(), Grouping::PerBlock,
                              settings.final_gradient_tolerance);
        const double total = fractions.sum() + FRACTION_SUM_OFFSET;
        block_fractions.row(voxel) = (fractions / total).transpose();
    }
}

} // namespace larmor::sd
