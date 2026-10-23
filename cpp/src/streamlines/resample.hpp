// Resampling of streamlines along a spline through their points, to a number of points or to a
// step size.
#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "common/types.hpp"
#include "streamlines/spline.hpp"

namespace larmor::streamlines {

/// The resampled points of one or more streamlines, and where each lies on its input.
///
/// Point k lies on the segment from input point segments[k] to segments[k] + 1, a fraction
/// fractions[k] (0 to 1) of the way along the spline parameter. An input point is its segment at
/// fraction 0, or the last segment at fraction 1 for a last point.
struct ResampledPoints {
    /// The resampled points.
    std::vector<Point> points;
    /// The input segment of each point.
    std::vector<Index> segments;
    /// The position of each point along its segment.
    std::vector<double> fractions;

    /// Appends one point and its position.
    void push_back(const Point& point, Index segment, double fraction);

    /// Returns the number of points held.
    [[nodiscard]] Index size() const { return static_cast<Index>(points.size()); }

    /// Reverses the order of the points from index first on.
    void reverse_from(Index first);
};

/// How to resample.
struct ResampleSettings {
    /// The number of points of each streamline, or 0 to resample to step_size.
    Index num_points = 0;
    /// The distance between consecutive points (mm).
    float step_size = 0.0F;
    /// Whether each new point's segment and fraction are kept too.
    bool positions = true;
};

/// Appends num_points points of one streamline of at least 2 points, at equal arc lengths.
///
/// Output point k is at arc length L k / (num_points - 1) along the polyline of length L, taken
/// on the cubic Hermite spline (tension 0.1) of its segment at the fraction of the segment's
/// length that reaches it (0.5 on a segment of length 0). The first and last points are the
/// input's. Segments are local to the streamline.
///
/// out is a ResampledPoints, or any type with the same push_back, size and reverse_from.
///
/// References:
///   MRtrix3, tckresample -num_points (Resampling::FixedNumPoints, version 3.0.8).
///   Tournier JD, Smith R, Raffelt D, et al. MRtrix3: A fast, flexible and open software framework
///   for medical image processing and visualisation. NeuroImage 202 (2019) 116137.
///   doi:10.1016/j.neuroimage.2019.116137
template <typename Sink>
void resample_num_points(const PointsView& points, Index num_points, Sink& out);

/// Appends one streamline of at least 2 points resampled to step_size (mm).
///
/// The middle input point (index n / 2) is kept, and from it each new point is found on the
/// spline of resample_num_points at straight-line distance step_size from the previous one, out
/// to each end. Each pass ends on the end point itself, which can be closer than step_size. A
/// point that is not finite ends its pass at the end point. The points run from the first input
/// point to the last. Segments are local; out is as for resample_num_points.
///
/// References:
///   MRtrix3, tckresample -step_size (Resampling::FixedStepSize, version 3.0.8).
///   Tournier JD, Smith R, Raffelt D, et al. MRtrix3: A fast, flexible and open software framework
///   for medical image processing and visualisation. NeuroImage 202 (2019) 116137.
///   doi:10.1016/j.neuroimage.2019.116137
template <typename Sink>
void resample_step_size(const PointsView& points, float step_size, Sink& out);

extern template void resample_num_points<ResampledPoints>(const PointsView& points,
                                                          Index num_points, ResampledPoints& out);
extern template void resample_step_size<ResampledPoints>(const PointsView& points, float step_size,
                                                         ResampledPoints& out);

/// A resampled tractogram; segments and fractions are empty unless positions were asked for.
struct ResampledTractogram {
    /// The points, (Q, 3).
    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> points;
    /// The offsets of the streamlines, (S + 1).
    Eigen::Matrix<std::int64_t, Eigen::Dynamic, 1> offsets;
    /// The input segment of each point, as an index into the input points; (Q,).
    Eigen::Matrix<std::int64_t, Eigen::Dynamic, 1> segments;
    /// The position of each point along its segment; (Q,).
    Eigen::Matrix<double, Eigen::Dynamic, 1> fractions;
};

/// Resamples every streamline of a tractogram (points (P, 3), offsets (S + 1)) on n_threads.
///
/// Streamlines of fewer than 2 points are copied as they are, each point its own segment at
/// fraction 0. The result does not depend on n_threads. The caller checks that the offsets start
/// at 0, never decrease and end at P.
ResampledTractogram resample_tractogram(const PointsView& points, const OffsetsView& offsets,
                                        const ResampleSettings& settings, int n_threads);

} // namespace larmor::streamlines
