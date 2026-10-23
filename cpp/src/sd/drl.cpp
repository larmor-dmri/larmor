#include "sd/drl.hpp"

#include <cmath>
#include <numeric>
#include <vector>

#include "common/maths.hpp"

namespace larmor::sd {

namespace {

// FOD amplitudes never go below this value, so a multiplicative update can bring them back.
constexpr double FOD_FLOOR = 1e-6;

// Keeps the update finite where K^T K f is zero.
constexpr double DENOMINATOR_OFFSET = 1e-16;

// A voxel has converged when its FOD changes in total by no more than this share of its
// largest signal value.
constexpr double CONVERGENCE_SHARE = 1e-4;

// Returns the sample standard deviation of values, dividing by n - 1.
double sample_standard_deviation(const Vector& values) {
    const double mean = values.mean();
    const double sum_of_squares = (values.array() - mean).square().sum();
    return std::sqrt(sum_of_squares / static_cast<double>(values.size() - 1));
}

// Copies row `from` of the matrix onto row `to`.
void move_row(Matrix& matrix, Index from, Index to) {
    if (from != to) {
        matrix.row(to) = matrix.row(from);
    }
}

} // namespace

void damped_richardson_lucy(const RlKernel& kernel, const DrlSettings& settings,
                            const MatrixView& block_signals, Eigen::Ref<Matrix> block_fods) {
    const Index n_voxels = block_signals.rows();
    const Index n_directions = kernel.gram.rows();

    // Per voxel: how strongly the damping applies, and the FOD change below which it has
    // converged.
    Vector damping_strength(n_voxels);
    Vector convergence_threshold(n_voxels);
    for (Index voxel = 0; voxel < n_voxels; ++voxel) {
        const Vector signal = block_signals.row(voxel).transpose();
        damping_strength(voxel) = std::fmax(1.0 - 4.0 * sample_standard_deviation(signal), 0.0);
        convergence_threshold(voxel) =
            CONVERGENCE_SHARE * signal.maxCoeff<Eigen::PropagateNumbers>();
    }
    const double threshold_8 = power_8(settings.damping_threshold);

    // The voxels still iterating are the first n_active rows of these matrices; voxel_of_row
    // gives each row's voxel in the block. A voxel that converges is replaced by the last row.
    Matrix kernel_signals = block_signals * kernel.transposed.transpose(); // K^T s
    Matrix fods = block_fods;
    Matrix previous_fods = block_fods;
    Matrix predicted(n_voxels, n_directions); // K^T K f
    std::vector<Index> voxel_of_row(static_cast<std::size_t>(n_voxels));
    std::iota(voxel_of_row.begin(), voxel_of_row.end(), Index{0});
    Index n_active = n_voxels;

    for (Index iteration = 0; iteration < settings.n_iterations && n_active > 0; ++iteration) {
        predicted.topRows(n_active).noalias() = fods.topRows(n_active) * kernel.gram;

        for (Index row = n_active - 1; row >= 0; --row) {
            const Index voxel = voxel_of_row[static_cast<std::size_t>(row)];
            double change = 0.0;
            for (Index direction = 0; direction < n_directions; ++direction) {
                const double amplitude = fods(row, direction);
                const double amplitude_8 = power_8(amplitude);
                const double damping = 1.0 - damping_strength(voxel) *
                                                 (1.0 - amplitude_8 / (amplitude_8 + threshold_8));
                const double ratio = (kernel_signals(row, direction) - predicted(row, direction)) /
                                     (predicted(row, direction) + DENOMINATOR_OFFSET);
                const double updated = std::fmax(amplitude * (1.0 + damping * ratio), FOD_FLOOR);
                fods(row, direction) = updated;
                change += std::fabs(updated - previous_fods(row, direction));
            }
            if (change <= convergence_threshold(voxel)) {
                block_fods.row(voxel) = fods.row(row);
                --n_active;
                move_row(fods, n_active, row);
                move_row(previous_fods, n_active, row);
                move_row(kernel_signals, n_active, row);
                voxel_of_row[static_cast<std::size_t>(row)] =
                    voxel_of_row[static_cast<std::size_t>(n_active)];
            } else {
                previous_fods.row(row) = fods.row(row);
            }
        }
    }
    for (Index row = 0; row < n_active; ++row) {
        block_fods.row(voxel_of_row[static_cast<std::size_t>(row)]) = fods.row(row);
    }
}

} // namespace larmor::sd
