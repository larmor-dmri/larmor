// Track density maps, which count the streamlines that pass through each voxel of a grid.
#pragma once

#include <array>
#include <vector>

#include <Eigen/Core>

#include "common/types.hpp"
#include "streamlines/spline.hpp"

namespace larmor::streamlines {

/// A voxel grid.
///
/// Voxel (i, j, k) has the flat index (i shape[1] + j) shape[2] + k, the order of a numpy array.
struct VoxelGrid {
    /// The grid shape.
    std::array<Index, 3> shape{};
    /// The affine map from world coordinates (mm) to voxel coordinates, without its last row.
    Eigen::Matrix<double, 3, 4> world_to_voxel = Eigen::Matrix<double, 3, 4>::Zero();
};

/// How streamlines are mapped to voxels.
struct DensitySettings {
    /// The number of pieces each segment is split into, when max_spacing is 0.
    Index upsample_ratio = 1;
    /// When positive, each segment is split into the fewest equal pieces, in spline parameter,
    /// such that the segment's straight length divided by their number is below this (mm);
    /// upsample_ratio is then not used.
    double max_spacing = 0.0;
    /// Whether only the two endpoints of each streamline are mapped.
    bool ends_only = false;
};

/// Returns the flat index of the voxel nearest to point, or -1 when that voxel is outside the
/// grid or the point is NaN.
Index nearest_voxel(const VoxelGrid& grid, const Point& point);

/// Appends to out the voxels of one streamline inside the grid, each once, in increasing order.
///
/// The streamline is first upsampled along the spline of the resampling, and every point is
/// mapped to its nearest voxel. With ends_only, only the first and the last point are mapped,
/// without upsampling, and both are kept even when they fall in the same voxel. scratch holds
/// the voxels while they are sorted.
///
/// References:
///   Calamante F, Tournier JD, Jackson GD, Connelly A. Track-density imaging (TDI):
///   super-resolution white matter imaging using whole-brain track-density mapping. NeuroImage 53
///   (2010) 1233-1243. MRtrix3, tckmap (version 3.0.8).
void streamline_voxels(const PointsView& points, const VoxelGrid& grid,
                       const DensitySettings& settings, std::vector<Index>& scratch,
                       std::vector<Index>& out);

/// Adds the track density of a tractogram (points (P, 3), offsets (S + 1)) to density.
///
/// density holds one value per voxel of grid, in flat order. Each streamline adds its weight
/// (weights(s), or 1 when weights is empty) to each of its voxels from streamline_voxels. The
/// result does not depend on n_threads. The caller checks that the offsets start at 0, never
/// decrease and end at P, and that weights is empty or has S entries.
void add_track_density(const PointsView& points, const OffsetsView& offsets, const VoxelGrid& grid,
                       const DensitySettings& settings,
                       const Eigen::Ref<const Eigen::VectorXd>& weights,
                       Eigen::Ref<Eigen::VectorXd> density, int n_threads);

} // namespace larmor::streamlines
