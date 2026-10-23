#include <cmath>
#include <cstdint>

#include <Eigen/Core>
#include <doctest/doctest.h>

#include "common/types.hpp"
#include "streamlines/resample.hpp"

namespace {

using larmor::Index;
using larmor::streamlines::Point;
using larmor::streamlines::ResampledPoints;
using larmor::streamlines::ResampleSettings;

using Points = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;

// n points along x, spacing apart, starting at the origin.
Points straight_line(Index n_points, float spacing) {
    Points points = Points::Zero(n_points, 3);
    for (Index index = 0; index < n_points; ++index) {
        points(index, 0) = spacing * static_cast<float>(index);
    }
    return points;
}

// A bent streamline: a quarter circle of radius 10 mm in 12 points.
Points quarter_circle() {
    constexpr Index N_CIRCLE_POINTS = 12;
    Points points(N_CIRCLE_POINTS, 3);
    for (Index index = 0; index < N_CIRCLE_POINTS; ++index) {
        const double angle =
            1.5707963267948966 * static_cast<double>(index) / (N_CIRCLE_POINTS - 1);
        points.row(index) << static_cast<float>(10.0 * std::cos(angle)),
            static_cast<float>(10.0 * std::sin(angle)), 0.0F;
    }
    return points;
}

float gap(const ResampledPoints& resampled, Index index) {
    const auto first = static_cast<std::size_t>(index);
    return (resampled.points[first + 1] - resampled.points[first]).norm();
}

} // namespace

TEST_CASE("num_points on a straight line gives equal gaps and keeps both ends") {
    const Points points = straight_line(5, 1.0F);
    ResampledPoints resampled;
    larmor::streamlines::resample_num_points(points, 9, resampled);
    REQUIRE(resampled.size() == 9);
    CHECK(resampled.points.front() == Point(0.0F, 0.0F, 0.0F));
    CHECK(resampled.points.back().x() == doctest::Approx(4.0).epsilon(1e-6));
    for (Index index = 0; index + 1 < resampled.size(); ++index) {
        CHECK(gap(resampled, index) == doctest::Approx(0.5).epsilon(1e-5));
    }
    CHECK(resampled.segments.front() == 0);
    CHECK(resampled.fractions.front() == 0.0);
}

TEST_CASE("step_size keeps the gaps at the step except next to the ends") {
    const Points points = quarter_circle();
    constexpr float STEP = 0.7F;
    ResampledPoints resampled;
    larmor::streamlines::resample_step_size(points, STEP, resampled);
    REQUIRE(resampled.size() > 3);
    CHECK(resampled.points.front() == Point(points.row(0).transpose()));
    CHECK(resampled.points.back() == Point(points.row(points.rows() - 1).transpose()));
    for (Index index = 1; index + 2 < resampled.size(); ++index) {
        CHECK(gap(resampled, index) == doctest::Approx(STEP).epsilon(2e-3));
    }
    CHECK(gap(resampled, 0) <= STEP);
    CHECK(gap(resampled, resampled.size() - 2) <= STEP);
    // Segments never decrease along the output, and each point lies between its two inputs.
    for (Index index = 0; index + 1 < resampled.size(); ++index) {
        const auto item = static_cast<std::size_t>(index);
        CHECK(resampled.segments[item] <= resampled.segments[item + 1]);
        CHECK(resampled.fractions[item] >= 0.0);
        CHECK(resampled.fractions[item] <= 1.0);
    }
}

TEST_CASE("step_size on two points keeps both ends") {
    const Points points = straight_line(2, 3.5F);
    ResampledPoints resampled;
    larmor::streamlines::resample_step_size(points, 1.0F, resampled);
    REQUIRE(resampled.size() == 5);
    CHECK(resampled.points.front() == Point(0.0F, 0.0F, 0.0F));
    CHECK(resampled.points.back() == Point(3.5F, 0.0F, 0.0F));
    CHECK(resampled.segments.back() == 0);
    CHECK(resampled.fractions.back() == 1.0);
}

TEST_CASE("resample_tractogram copies short streamlines and does not depend on threads") {
    const Points circle = quarter_circle();
    // Streamlines of 12, 0, 1 and 12 points, repeated past one block.
    constexpr Index N_REPEATS = 300;
    Points points(N_REPEATS * 25, 3);
    Eigen::Matrix<std::int64_t, Eigen::Dynamic, 1> offsets(N_REPEATS * 4 + 1);
    offsets(0) = 0;
    Index n_points = 0;
    for (Index repeat = 0; repeat < N_REPEATS; ++repeat) {
        points.middleRows(n_points, 12) = circle;
        points.row(n_points + 12) << 1.0F, 2.0F, 3.0F;
        points.middleRows(n_points + 13, 12) = circle.colwise().reverse();
        offsets(repeat * 4 + 1) = n_points + 12;
        offsets(repeat * 4 + 2) = n_points + 12;
        offsets(repeat * 4 + 3) = n_points + 13;
        offsets(repeat * 4 + 4) = n_points + 25;
        n_points += 25;
    }
    const ResampleSettings settings{.num_points = 0, .step_size = 0.9F};
    const auto one_thread = larmor::streamlines::resample_tractogram(points, offsets, settings, 1);
    const auto four_threads =
        larmor::streamlines::resample_tractogram(points, offsets, settings, 4);
    CHECK(one_thread.points == four_threads.points);
    CHECK(one_thread.offsets == four_threads.offsets);
    CHECK(one_thread.segments == four_threads.segments);
    CHECK(one_thread.fractions == four_threads.fractions);
    // The empty streamline stays empty and the single point is copied as it is.
    CHECK(one_thread.offsets(2) == one_thread.offsets(1));
    CHECK(one_thread.offsets(3) == one_thread.offsets(2) + 1);
    CHECK(one_thread.points.row(one_thread.offsets(2)) == Eigen::RowVector3f(1.0F, 2.0F, 3.0F));
    CHECK(one_thread.segments(one_thread.offsets(2)) == 12);
}
