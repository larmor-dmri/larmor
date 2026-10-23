// Denoising of diffusion-weighted images by MP-PCA.
#pragma once

#include <array>
#include <cstdint>

#include <Eigen/Eigenvalues>

#include "common/types.hpp"

namespace larmor::dwi {

/// A 4D float32 image (X, Y, Z, M), read only, seen as a matrix with one row per voxel.
/// Row (x * Y + y) * Z + z holds the M volumes of voxel (x, y, z).
using ImageRowsView =
    Eigen::Ref<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

/// Flat voxel indices, (x * Y + y) * Z + z, read only.
using VoxelIndicesView = Eigen::Ref<const Eigen::Matrix<std::int64_t, Eigen::Dynamic, 1>>;

/// A column vector of integer counts, one per voxel.
using Counts = Eigen::Matrix<std::int64_t, Eigen::Dynamic, 1>;

/// The image grid and the patch shape.
struct PatchGrid {
    /// The image grid (X, Y, Z).
    std::array<Index, 3> grid_shape{};
    /// The patch width along each axis, at least 1 and at most the grid's.
    std::array<Index, 3> patch_shape{};

    /// The number of voxels in a patch, N.
    [[nodiscard]] Index n_patch_voxels() const {
        return patch_shape[0] * patch_shape[1] * patch_shape[2];
    }
};

/// The split of a patch's eigenvalues into signal and noise.
struct NoiseSplit {
    /// The number of signal components, P.
    Index n_signal = 0;
    /// The noise variance sigma^2, in squared signal units.
    double noise_variance = 0.0;
};

/// Finds the number of signal components P and the noise variance sigma^2 from the eigenvalues
/// (m,), in ascending order, of X X^T / n for a patch matrix X (m, n) with m <= n = n_larger.
///
/// The m - p smallest eigenvalues of pure noise follow the Marchenko-Pastur distribution, of mean
/// sigma^2 and width 4 sqrt(gamma) sigma^2 with gamma = (m - p) / n. P is the smallest p at which
/// the mean of the m - p smallest eigenvalues is at least the variance their range implies,
/// (lambda_(p+1) - lambda_m) / (4 sqrt(gamma)), counting eigenvalues from the largest; sigma^2 is
/// then that mean. Negative eigenvalues count as zero. When no p qualifies, P = m and
/// sigma^2 = 0.
///
/// References:
///   Veraart J, Novikov DS, Christiaens D, Ades-aron B, Sijbers J, Fieremans E. Denoising of
///   diffusion MRI using random matrix theory. NeuroImage 142 (2016) 394-406 (equations 10-12).
NoiseSplit split_marchenko_pastur(const VectorView& ascending_eigenvalues, Index n_larger);

/// Buffers one thread reuses for every patch it denoises.
struct MppcaScratch {
    /// The patch, one row per voxel, (N, M).
    Matrix patch;
    /// The smaller Gram matrix of the patch, (m, m), m = min(M, N).
    Matrix gram;
    /// The eigendecomposition of the Gram matrix.
    Eigen::SelfAdjointEigenSolver<Matrix> eigen_solver;
    /// The eigenvalues of the Gram matrix divided by max(M, N), ascending, (m,).
    Vector eigenvalues;
};

/// Makes the scratch buffers of a thread for patches of n_patch_voxels voxels and n_volumes
/// volumes.
MppcaScratch make_mppca_scratch(Index n_patch_voxels, Index n_volumes);

/// Denoises the voxels block_centres (b,) of an image, writing each one's denoised signal to a
/// row of block_denoised (b, M), its noise level sigma to block_noise (b,) and its number of
/// signal components to block_n_signal (b,).
///
/// Each voxel gets its own patch, a box of patch_shape voxels centred on it and shifted to stay
/// inside the grid. The patch matrix X (M, N) holds the signals of its N voxels in its columns,
/// with no mean removed. The eigenvalues of its smaller Gram matrix, divided by max(M, N), are
/// split into signal and noise with split_marchenko_pastur; the denoised signal is the voxel's
/// column of X projected onto the P signal components, and sigma the square root of the noise
/// variance. Only the voxel itself is written, so voxels are independent. A patch with a value
/// that is not finite gives NaN for its voxel.
///
/// References:
///   Veraart J, Novikov DS, Christiaens D, Ades-aron B, Sijbers J, Fieremans E. Denoising of
///   diffusion MRI using random matrix theory. NeuroImage 142 (2016) 394-406.
void denoise_mppca_block(const ImageRowsView& image, const PatchGrid& grid,
                         const VoxelIndicesView& block_centres, MppcaScratch& scratch,
                         Eigen::Ref<Matrix> block_denoised, Eigen::Ref<Vector> block_noise,
                         Eigen::Ref<Counts> block_n_signal);

} // namespace larmor::dwi
