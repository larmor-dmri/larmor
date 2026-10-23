#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <doctest/doctest.h>

#include "common/types.hpp"
#include "streamlines/density.hpp"

namespace {

using larmor::Index;
using larmor::streamlines::DensitySettings;
using larmor::streamlines::Point;
using larmor::streamlines::VoxelGrid;

using Points = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Offsets = Eigen::Matrix<std::int64_t, Eigen::Dynamic, 1>;

// A grid of 1 mm voxels whose voxel (i, j, k) is centred at world (i, j, k).
VoxelGrid unit_grid(Index nx, Index ny, Index nz) {
    VoxelGrid grid;
    grid.shape = {nx, ny, nz};
    grid.world_to_voxel.leftCols<3>().setIdentity();
    return grid;
}

std::vector<Index> voxels_of(const Points& points, const VoxelGrid& grid,
                             const DensitySettings& settings) {
    std::vector<Index> scratch;
    std::vector<Index> out;
    larmor::streamlines::streamline_voxels(points, grid, settings, scratch, out);
    return out;
}

} // namespace

TEST_CASE("nearest_voxel rounds halves away from zero and refuses points outside") {
    const VoxelGrid grid = unit_grid(4, 3, 2);
    CHECK(larmor::streamlines::nearest_voxel(grid, Point(1.5F, 0.0F, 0.0F)) == 2 * 3 * 2);
    CHECK(larmor::streamlines::nearest_voxel(grid, Point(0.0F, 1.0F, 1.4F)) == 1 * 2 + 1);
    CHECK(larmor::streamlines::nearest_voxel(grid, Point(-0.4F, 0.0F, 0.0F)) == 0);
    CHECK(larmor::streamlines::nearest_voxel(grid, Point(-0.5F, 0.0F, 0.0F)) == -1);
    CHECK(larmor::streamlines::nearest_voxel(grid, Point(3.5F, 0.0F, 0.0F)) == -1);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    CHECK(larmor::streamlines::nearest_voxel(grid, Point(nan, 0.0F, 0.0F)) == -1);
}

TEST_CASE("a streamline counts each voxel once, and upsampling fills the gaps") {
    const VoxelGrid grid = unit_grid(10, 1, 1);
    // Out along x with 3 mm steps, then back over the same voxels.
    Points points(4, 3);
    points << 0.0F, 0.0F, 0.0F, 3.0F, 0.0F, 0.0F, 6.0F, 0.0F, 0.0F, 3.0F, 0.0F, 0.0F;
    CHECK(voxels_of(points, grid, {.upsample_ratio = 1}) == std::vector<Index>{0, 3, 6});
    // A ratio of 3 adds points 1 mm apart on the straight parts.
    const std::vector<Index> upsampled = voxels_of(points, grid, {.upsample_ratio = 3});
    CHECK(upsampled == std::vector<Index>{0, 1, 2, 3, 4, 5, 6});
}

TEST_CASE("ends_only keeps both endpoints, even in one voxel") {
    const VoxelGrid grid = unit_grid(5, 1, 1);
    Points points(3, 3);
    points << 1.1F, 0.0F, 0.0F, 3.0F, 0.0F, 0.0F, 0.9F, 0.0F, 0.0F;
    CHECK(voxels_of(points, grid, {.ends_only = true}) == std::vector<Index>{1, 1});
    const Points single = points.topRows(1);
    CHECK(voxels_of(single, grid, {.ends_only = true}) == std::vector<Index>{1});
}

TEST_CASE("add_track_density gives the same sums on any number of threads") {
    const VoxelGrid grid = unit_grid(20, 20, 1);
    constexpr Index N_STREAMLINES = 3000;
    Points points(N_STREAMLINES * 3, 3);
    Offsets offsets(N_STREAMLINES + 1);
    Eigen::VectorXd weights(N_STREAMLINES);
    for (Index streamline = 0; streamline < N_STREAMLINES; ++streamline) {
        const auto shift = static_cast<float>(streamline % 17);
        for (Index point = 0; point < 3; ++point) {
            points.row(streamline * 3 + point) << shift * 0.9F + static_cast<float>(point) * 1.7F,
                static_cast<float>(streamline % 13) * 1.3F, 0.0F;
        }
        offsets(streamline) = streamline * 3;
        weights(streamline) = 0.1 + 1e-3 * static_cast<double>(streamline);
    }
    offsets(N_STREAMLINES) = N_STREAMLINES * 3;
    const DensitySettings settings{.upsample_ratio = 4};
    Eigen::VectorXd single = Eigen::VectorXd::Zero(400);
    Eigen::VectorXd several = Eigen::VectorXd::Zero(400);
    larmor::streamlines::add_track_density(points, offsets, grid, settings, weights, single, 1);
    larmor::streamlines::add_track_density(points, offsets, grid, settings, weights, several, 4);
    CHECK(single.sum() > 0.0);
    CHECK((single.array() == several.array()).all());
}
