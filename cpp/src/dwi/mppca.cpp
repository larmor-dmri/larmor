#include "dwi/mppca.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace larmor::dwi {

namespace {

// Returns the first voxel of the patch along one axis, centred on `position` and shifted to
// stay inside the grid.
Index patch_start(Index position, Index patch_length, Index grid_length) {
    return std::clamp<Index>(position - patch_length / 2, 0, grid_length - patch_length);
}

// Copies the signals of the patch around the voxel at flat index `centre` into scratch.patch,
// one row per voxel, and returns the row of the centre voxel.
Index load_patch(const ImageRowsView& image, const PatchGrid& grid, std::int64_t centre,
                 MppcaScratch& scratch) {
    const auto [size_x, size_y, size_z] = grid.grid_shape;
    const auto [patch_x, patch_y, patch_z] = grid.patch_shape;
    const Index x = centre / (size_y * size_z);
    const Index y = (centre / size_z) % size_y;
    const Index z = centre % size_z;
    const Index start_x = patch_start(x, patch_x, size_x);
    const Index start_y = patch_start(y, patch_y, size_y);
    const Index start_z = patch_start(z, patch_z, size_z);

    Index row = 0;
    for (Index offset_x = 0; offset_x < patch_x; ++offset_x) {
        for (Index offset_y = 0; offset_y < patch_y; ++offset_y) {
            for (Index offset_z = 0; offset_z < patch_z; ++offset_z) {
                const Index voxel = ((start_x + offset_x) * size_y + start_y + offset_y) * size_z +
                                    start_z + offset_z;
                scratch.patch.row(row) = image.row(voxel).cast<double>();
                ++row;
            }
        }
    }
    return ((x - start_x) * patch_y + y - start_y) * patch_z + z - start_z;
}

} // namespace

NoiseSplit split_marchenko_pastur(const VectorView& ascending_eigenvalues, Index n_larger) {
    const Index n_smaller = ascending_eigenvalues.size();
    const Vector eigenvalues = ascending_eigenvalues.cwiseMax(0.0);
    // noise_sums(count) is the sum of the count smallest eigenvalues.
    Vector noise_sums(n_smaller + 1);
    noise_sums(0) = 0.0;
    for (Index count = 0; count < n_smaller; ++count) {
        noise_sums(count + 1) = noise_sums(count) + eigenvalues(count);
    }
    // p signal components leave the n_smaller - p smallest eigenvalues as noise.
    for (Index n_signal = 0; n_signal < n_smaller; ++n_signal) {
        const Index n_noise = n_smaller - n_signal;
        const double mean = noise_sums(n_noise) / static_cast<double>(n_noise);
        const double ratio = static_cast<double>(n_noise) / static_cast<double>(n_larger);
        const double range_variance =
            (eigenvalues(n_noise - 1) - eigenvalues(0)) / (4.0 * std::sqrt(ratio));
        if (mean >= range_variance) {
            return {.n_signal = n_signal, .noise_variance = mean};
        }
    }
    return {.n_signal = n_smaller, .noise_variance = 0.0};
}

MppcaScratch make_mppca_scratch(Index n_patch_voxels, Index n_volumes) {
    const Index n_smaller = std::min(n_patch_voxels, n_volumes);
    return {.patch = Matrix(n_patch_voxels, n_volumes),
            .gram = Matrix(n_smaller, n_smaller),
            .eigen_solver = Eigen::SelfAdjointEigenSolver<Matrix>(n_smaller),
            .eigenvalues = Vector(n_smaller)};
}

void denoise_mppca_block(const ImageRowsView& image, const PatchGrid& grid,
                         const VoxelIndicesView& block_centres, MppcaScratch& scratch,
                         Eigen::Ref<Matrix> block_denoised, Eigen::Ref<Vector> block_noise,
                         Eigen::Ref<Counts> block_n_signal) {
    const Index n_volumes = image.cols();
    const Index n_patch_voxels = grid.n_patch_voxels();
    const bool volumes_are_smaller = n_volumes <= n_patch_voxels;
    const Index n_larger = std::max(n_volumes, n_patch_voxels);
    constexpr double NOT_A_NUMBER = std::numeric_limits<double>::quiet_NaN();

    for (Index item = 0; item < block_centres.size(); ++item) {
        const Index centre_row = load_patch(image, grid, block_centres(item), scratch);
        if (!scratch.patch.allFinite()) {
            block_denoised.row(item).setConstant(NOT_A_NUMBER);
            block_noise(item) = NOT_A_NUMBER;
            block_n_signal(item) = 0;
            continue;
        }
        // scratch.patch is X^T, (N, M).
        if (volumes_are_smaller) {
            scratch.gram.noalias() = scratch.patch.transpose() * scratch.patch;
        } else {
            scratch.gram.noalias() = scratch.patch * scratch.patch.transpose();
        }
        scratch.eigen_solver.compute(scratch.gram, Eigen::ComputeEigenvectors);
        scratch.eigenvalues = scratch.eigen_solver.eigenvalues() / static_cast<double>(n_larger);
        const NoiseSplit split = split_marchenko_pastur(scratch.eigenvalues, n_larger);

        // The signal components are the last n_signal eigenvectors (largest eigenvalues).
        const auto signal_vectors = scratch.eigen_solver.eigenvectors().rightCols(split.n_signal);
        const Vector centre_signal = scratch.patch.row(centre_row).transpose();
        if (volumes_are_smaller) {
            // U_P U_P^T x, with U_P the signal vectors in the space of the volumes.
            block_denoised.row(item) =
                (signal_vectors * (signal_vectors.transpose() * centre_signal)).transpose();
        } else {
            // The centre's column of X V_P V_P^T, with V_P in the space of the patch voxels.
            const Vector weights = signal_vectors * signal_vectors.row(centre_row).transpose();
            block_denoised.row(item) = (scratch.patch.transpose() * weights).transpose();
        }
        block_noise(item) = std::sqrt(split.noise_variance);
        block_n_signal(item) = split.n_signal;
    }
}

} // namespace larmor::dwi
