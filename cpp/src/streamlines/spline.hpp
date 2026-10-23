// Views of a tractogram's points and offsets, and the cubic Hermite spline through a streamline's
// points.
#pragma once

#include <cmath>
#include <cstdint>

#include <Eigen/Core>

#include "common/types.hpp"

namespace larmor::streamlines {

/// A streamline point in float32 (mm).
using Point = Eigen::Vector3f;

/// The (n, 3) float32 points of one streamline or of a whole tractogram, read only.
using PointsView = Eigen::Ref<const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>>;

/// A tractogram's (S + 1) int64 offsets, read only.
using OffsetsView = Eigen::Ref<const Eigen::Matrix<std::int64_t, Eigen::Dynamic, 1>>;

/// Half the spline's tension of 0.1.
constexpr float HALF_TENSION = 0.5F * 0.1F;

/// The weights of the four control points of the spline at one parameter.
struct HermiteWeights {
    /// One weight per control point, p0 to p3.
    float w0;
    float w1;
    float w2;
    float w3;
};

/// Returns the weights of the four control points of the cubic Hermite spline at mu in [0, 1].
inline HermiteWeights hermite_weights(float mu) {
    const float mu2 = mu * mu;
    const float mu3 = mu * mu2;
    // The fused multiply-adds in w1 and w2 round as the compiled reference tool does.
    return {.w0 = (0.5F - HALF_TENSION) * (2.0F * mu2 - mu3 - mu),
            .w1 = std::fma(-(2.5F + HALF_TENSION), mu2, std::fma(1.5F + HALF_TENSION, mu3, 1.0F)),
            .w2 = std::fma(-(1.5F + HALF_TENSION), mu3,
                           std::fma(2.0F + 2.0F * HALF_TENSION, mu2, (0.5F - HALF_TENSION) * mu)),
            .w3 = (0.5F - HALF_TENSION) * (mu3 - mu2)};
}

/// Returns the point at mu on the spline from p1 to p2, whose outer control points are p0 and p3.
inline Point spline_point(float mu, const Point& p0, const Point& p1, const Point& p2,
                          const Point& p3) {
    const HermiteWeights weights = hermite_weights(mu);
    return weights.w0 * p0 + weights.w1 * p1 + weights.w2 * p2 + weights.w3 * p3;
}

/// The n input points of a streamline (n >= 2) with one mirrored point before the first and one
/// after the last, indexed 0 to n + 1: index k + 1 is input point k.
struct ExtendedPoints {
    /// The input points.
    const PointsView& points;
    /// The mirrored point before the first.
    Point before;
    /// The mirrored point after the last.
    Point after;

    explicit ExtendedPoints(const PointsView& input)
        : points(input), before(mirror(input, 0, 1)),
          after(mirror(input, input.rows() - 1, input.rows() - 2)) {}

    /// Returns the extended point at index (0 to n + 1).
    [[nodiscard]] Point operator()(Index index) const {
        if (index == 0) {
            return before;
        }
        if (index == points.rows() + 1) {
            return after;
        }
        return points.row(index - 1).transpose();
    }

    /// Returns the reflection of input point `neighbour` through input point `end`.
    static Point mirror(const PointsView& input, Index end, Index neighbour) {
        const Point end_point = input.row(end).transpose();
        const Point neighbour_point = input.row(neighbour).transpose();
        return end_point + (end_point - neighbour_point);
    }
};

} // namespace larmor::streamlines
