#include "streamlines/density.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "common/parallel.hpp"
#include "common/types.hpp"
#include "streamlines/spline.hpp"

namespace larmor::streamlines {

namespace {

// Streamlines per block of the parallel loop.
constexpr Index DENSITY_BLOCK_SIZE = 256;
// Streamlines whose voxels are held at a time, to bound the memory of the buffers.
constexpr Index STREAMLINES_PER_ROUND = 1 << 16;

// Returns the spline weights at k / ratio for k = 1 to ratio - 1, the same for every segment.
std::vector<HermiteWeights> upsampling_weights(Index ratio) {
    std::vector<HermiteWeights> weights;
    for (Index step = 1; step < ratio; ++step) {
        weights.push_back(hermite_weights(static_cast<float>(step) / static_cast<float>(ratio)));
    }
    return weights;
}

// Appends the voxel of point to voxels unless it is outside or the same as the last one added.
void add_voxel(const VoxelGrid& grid, const Point& point, std::vector<Index>& voxels) {
    const Index voxel = nearest_voxel(grid, point);
    if (voxel >= 0 && (voxels.empty() || voxels.back() != voxel)) {
        voxels.push_back(voxel);
    }
}

// Returns the number of pieces a segment of this length is split into, the smallest whole number
// that brings the length divided by it below max_spacing.
Index segment_ratio(float length, double max_spacing) {
    // Rounded first, so that a length a float32 hair off a whole ratio gets that ratio.
    const double ratio = static_cast<double>(length) / max_spacing;
    return static_cast<Index>(std::floor(std::round(ratio * 1000.0) / 1000.0)) + 1;
}

// Appends to `voxels` the voxels of every point of the upsampled streamline (at least 2 points).
// With a positive max_spacing each segment gets its own ratio; otherwise `weights` holds the
// spline weights of the fixed ratio.
void upsampled_voxels(const PointsView& points, const VoxelGrid& grid,
                      const std::vector<HermiteWeights>& weights, double max_spacing,
                      std::vector<Index>& voxels) {
    const ExtendedPoints extended(points);
    const Index n_points = points.rows();
    for (Index segment = 0; segment + 1 < n_points; ++segment) {
        const Point p0 = extended(segment);
        const Point p1 = extended(segment + 1);
        const Point p2 = extended(segment + 2);
        const Point p3 = extended(segment + 3);
        add_voxel(grid, p1, voxels);
        if (max_spacing > 0.0) {
            const Index ratio = segment_ratio((p2 - p1).norm(), max_spacing);
            for (Index step = 1; step < ratio; ++step) {
                const HermiteWeights weight =
                    hermite_weights(static_cast<float>(step) / static_cast<float>(ratio));
                add_voxel(grid, weight.w0 * p0 + weight.w1 * p1 + weight.w2 * p2 + weight.w3 * p3,
                          voxels);
            }
            continue;
        }
        for (const HermiteWeights& step : weights) {
            add_voxel(grid, step.w0 * p0 + step.w1 * p1 + step.w2 * p2 + step.w3 * p3, voxels);
        }
    }
    add_voxel(grid, points.row(n_points - 1).transpose(), voxels);
}

// Appends the voxels of one streamline to `out`, as streamline_voxels, with precomputed
// upsampling weights.
void voxels_with_weights(const PointsView& points, const VoxelGrid& grid,
                         const DensitySettings& settings,
                         const std::vector<HermiteWeights>& weights, std::vector<Index>& scratch,
                         std::vector<Index>& out) {
    const Index n_points = points.rows();
    if (n_points == 0) {
        return;
    }
    if (settings.ends_only) {
        for (const Index end : {Index{0}, n_points - 1}) {
            const Index voxel = nearest_voxel(grid, points.row(end).transpose());
            if (voxel >= 0) {
                out.push_back(voxel);
            }
            if (n_points == 1) {
                break;
            }
        }
        return;
    }
    scratch.clear();
    if (n_points == 1) {
        add_voxel(grid, points.row(0).transpose(), scratch);
    } else {
        upsampled_voxels(points, grid, weights, settings.max_spacing, scratch);
    }
    std::ranges::sort(scratch);
    const auto duplicates = std::ranges::unique(scratch);
    out.insert(out.end(), scratch.begin(), duplicates.begin());
}

// The voxels found for one block of streamlines, and how many each streamline has.
struct BlockVoxels {
    std::vector<Index> voxels;
    std::vector<Index> counts;
};

} // namespace

Index nearest_voxel(const VoxelGrid& grid, const Point& point) {
    Index flat_index = 0;
    for (Index axis = 0; axis < 3; ++axis) {
        const auto row = grid.world_to_voxel.row(axis);
        double coordinate = static_cast<double>(point.x()) * row(0);
        coordinate += static_cast<double>(point.y()) * row(1);
        coordinate += static_cast<double>(point.z()) * row(2);
        coordinate += row(3);
        const auto size = grid.shape.at(static_cast<std::size_t>(axis));
        if (std::isnan(coordinate) || coordinate <= -0.5 ||
            coordinate >= static_cast<double>(size) - 0.5) {
            return -1;
        }
        flat_index = flat_index * size + static_cast<Index>(std::round(coordinate));
    }
    return flat_index;
}

void streamline_voxels(const PointsView& points, const VoxelGrid& grid,
                       const DensitySettings& settings, std::vector<Index>& scratch,
                       std::vector<Index>& out) {
    const std::vector<HermiteWeights> weights = upsampling_weights(settings.upsample_ratio);
    voxels_with_weights(points, grid, settings, weights, scratch, out);
}

void add_track_density(const PointsView& points, const OffsetsView& offsets, const VoxelGrid& grid,
                       const DensitySettings& settings,
                       const Eigen::Ref<const Eigen::VectorXd>& weights,
                       Eigen::Ref<Eigen::VectorXd> density, int n_threads) {
    const std::vector<HermiteWeights> spline_weights = upsampling_weights(settings.upsample_ratio);
    const Index n_streamlines = offsets.size() - 1;
    for (Index round_begin = 0; round_begin < n_streamlines; round_begin += STREAMLINES_PER_ROUND) {
        const Index round_size = std::min(STREAMLINES_PER_ROUND, n_streamlines - round_begin);
        const Index n_blocks = (round_size + DENSITY_BLOCK_SIZE - 1) / DENSITY_BLOCK_SIZE;
        std::vector<BlockVoxels> blocks(static_cast<std::size_t>(n_blocks));
        parallel_for_blocks(
            round_size, DENSITY_BLOCK_SIZE, n_threads, [] { return std::vector<Index>{}; },
            [&](Index begin, Index end, std::vector<Index>& scratch) {
                BlockVoxels& block = blocks[static_cast<std::size_t>(begin / DENSITY_BLOCK_SIZE)];
                for (Index index = begin; index < end; ++index) {
                    const Index streamline = round_begin + index;
                    const Index first = offsets(streamline);
                    const Index n_points = offsets(streamline + 1) - first;
                    const auto before = static_cast<Index>(block.voxels.size());
                    voxels_with_weights(points.middleRows(first, n_points), grid, settings,
                                        spline_weights, scratch, block.voxels);
                    block.counts.push_back(static_cast<Index>(block.voxels.size()) - before);
                }
            });
        // Added in streamline order on this thread, so the sums are the same for any n_threads.
        Index streamline = round_begin;
        for (const BlockVoxels& block : blocks) {
            std::size_t position = 0;
            for (const Index count : block.counts) {
                const double weight = weights.size() == 0 ? 1.0 : weights(streamline);
                for (Index voxel = 0; voxel < count; ++voxel) {
                    density(block.voxels[position]) += weight;
                    ++position;
                }
                ++streamline;
            }
        }
    }
}

} // namespace larmor::streamlines
