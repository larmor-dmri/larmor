#include <cmath>
#include <cstdint>
#include <limits>

#include <doctest/doctest.h>

#include "common/types.hpp"
#include "dwi/mppca.hpp"

namespace {

using larmor::Index;
using larmor::Matrix;
using larmor::Vector;
using larmor::dwi::Counts;
using larmor::dwi::PatchGrid;

using FloatRows = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VoxelIndices = Eigen::Matrix<std::int64_t, Eigen::Dynamic, 1>;

// A noise-free image of rank 1 up to float32 rounding on a 4x5x6 grid: voxel v has the signal
// amplitude(v) * profile, with an amplitude that differs from voxel to voxel and a profile over
// n_volumes volumes.
FloatRows rank_one_image(Index n_volumes) {
    FloatRows image(4 * 5 * 6, n_volumes);
    for (Index voxel = 0; voxel < image.rows(); ++voxel) {
        const double amplitude = 100.0 + 3.0 * static_cast<double>(voxel);
        for (Index volume = 0; volume < n_volumes; ++volume) {
            const double profile = std::exp(-0.1 * static_cast<double>(volume));
            image(voxel, volume) = static_cast<float>(amplitude * profile);
        }
    }
    return image;
}

struct Denoised {
    Matrix signals;
    Vector noise;
    Counts n_signal;
};

Denoised denoise(const FloatRows& image, const PatchGrid& grid, const VoxelIndices& centres) {
    auto scratch = larmor::dwi::make_mppca_scratch(grid.n_patch_voxels(), image.cols());
    Denoised result{.signals = Matrix(centres.size(), image.cols()),
                    .noise = Vector(centres.size()),
                    .n_signal = Counts(centres.size())};
    larmor::dwi::denoise_mppca_block(image, grid, centres, scratch, result.signals, result.noise,
                                     result.n_signal);
    return result;
}

} // namespace

TEST_CASE("mppca: equal eigenvalues are all noise, one large one is signal") {
    const Vector equal = Vector::Constant(10, 2.5);
    const auto all_noise = larmor::dwi::split_marchenko_pastur(equal, 40);
    CHECK(all_noise.n_signal == 0);
    CHECK(all_noise.noise_variance == doctest::Approx(2.5));

    Vector one_large = equal;
    one_large(9) = 100.0;
    const auto one_signal = larmor::dwi::split_marchenko_pastur(one_large, 40);
    CHECK(one_signal.n_signal == 1);
    CHECK(one_signal.noise_variance == doctest::Approx(2.5));
}

TEST_CASE("mppca: a rank-one image comes back unchanged, at the edges too") {
    // The corner, an edge voxel and an inner voxel, with fewer volumes than patch voxels (27) and
    // with more (40).
    const VoxelIndices centres{{0, 5 * 6 * 3 + 5, (2 * 5 + 2) * 6 + 3}};
    for (const Index n_volumes : {Index{12}, Index{40}}) {
        const FloatRows image = rank_one_image(n_volumes);
        const PatchGrid grid{.grid_shape = {4, 5, 6}, .patch_shape = {3, 3, 3}};
        const Denoised result = denoise(image, grid, centres);
        for (Index item = 0; item < centres.size(); ++item) {
            const Vector expected = image.row(centres(item)).cast<double>().transpose();
            const Vector found = result.signals.row(item).transpose();
            CHECK((found - expected).cwiseAbs().maxCoeff() <= 1e-6 * expected.maxCoeff());
            CHECK(result.noise(item) <= 1e-6 * expected.maxCoeff());
        }
    }
}

TEST_CASE("mppca: a patch with a NaN gives NaN for its voxel only") {
    FloatRows image = rank_one_image(12);
    image(0, 3) = std::numeric_limits<float>::quiet_NaN();
    const PatchGrid grid{.grid_shape = {4, 5, 6}, .patch_shape = {3, 3, 3}};
    // Voxel 1 has voxel 0 in its patch; voxel (3, 4, 5) does not.
    const VoxelIndices centres{{1, (3 * 5 + 4) * 6 + 5}};
    const Denoised result = denoise(image, grid, centres);
    CHECK(result.signals.row(0).array().isNaN().all());
    CHECK(std::isnan(result.noise(0)));
    CHECK(result.signals.row(1).allFinite());
}
