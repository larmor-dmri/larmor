#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <tuple>
#include <vector>

#include <doctest/doctest.h>

#include "common/maths.hpp"
#include "sh/harmonics.hpp"
#include "tracking/calibrate.hpp"

namespace {

using larmor::Index;
using larmor::Vector;
using larmor::Vector3;
using larmor::tracking::calibrate;
using larmor::tracking::Calibration;
using larmor::tracking::direction_grid;

double to_radians(double degrees) {
    return degrees * (std::numbers::pi / 180.0);
}

bool contains_axis(const std::vector<Vector3>& grid) {
    return std::ranges::any_of(
        grid, [](const Vector3& direction) { return direction == Vector3::UnitZ(); });
}

} // namespace

TEST_CASE("direction_grid is a centre and hexagonal rings") {
    CHECK(direction_grid(0.1, 1.0).size() == 1);
    // Rings at exactly the radius sit on a rounding edge, so the radius is a hair past them: the
    // centre, 6 at distance 1, then 6 at sqrt(3) and 6 at 2.
    CHECK(direction_grid(1.0 + 1e-9, 1.0).size() == 7);
    CHECK(direction_grid(2.0 + 1e-9, 1.0).size() == 19);
    const std::vector<Vector3> grid = direction_grid(0.3, 0.1);
    CHECK(contains_axis(grid));
    for (const Vector3& direction : grid) {
        CHECK(std::abs(larmor::norm3(direction) - 1.0) < 1e-12);
        CHECK(std::acos(direction.z()) <= 0.3 + 1e-12);
    }
}

// tckgen 3.0.8 -algorithm iFOD1 -info on an lmax-8 FOD: "rejection sampling will use N
// directions with a ratio of R" for each maximum angle (degrees). MRtrix calibrates in float, so
// the ratio agrees to about 1e-4.
TEST_CASE("the calibration matches tckgen") {
    const std::array<std::tuple<double, std::size_t, double>, 5> cases{{
        {15.0, 7, 1.21581316},
        {30.0, 7, 2.44004321},
        {45.0, 13, 4.48055172},
        {60.0, 19, 3.26650047},
        {90.0, 37, 4.24406433},
    }};
    for (const auto& [angle, n_directions, ratio] : cases) {
        CAPTURE(angle);
        const Calibration calibration = calibrate(8, false, to_radians(angle));
        CHECK(calibration.grid.size() == n_directions);
        CHECK(std::abs(calibration.ratio - ratio) < 1e-4 * ratio);
        CHECK(contains_axis(calibration.grid));
    }
}

// The bound, the largest amplitude on the grid times the ratio, is above the one-sided reference
// lobe everywhere in the cone, whichever way the cone points.
TEST_CASE("a full calibration bounds its one-sided lobe") {
    const Index lmax = 8;
    Vector fibre = Vector::Zero(larmor::sh::n_coeffs(lmax, true));
    for (Index degree = 0; degree <= lmax; ++degree) {
        fibre(larmor::sh::coefficient_index(degree, 0, true)) =
            std::sqrt(static_cast<double>(2 * degree + 1) / (4.0 * std::numbers::pi));
    }
    for (const double angle : {15.0, 30.0, 45.0, 60.0, 90.0}) {
        CAPTURE(angle);
        const double max_angle = to_radians(angle);
        const Calibration calibration = calibrate(lmax, true, max_angle);
        CHECK(std::isfinite(calibration.ratio));
        CHECK(calibration.ratio >= 1.0);
        CHECK(contains_axis(calibration.grid));
        const std::vector<Vector3> cone = direction_grid(max_angle, 0.01);
        for (const double tilt : {0.0, 5.0, 10.0, 20.0, 40.0}) {
            CAPTURE(tilt);
            const double tilt_rad = to_radians(tilt);
            const Vector3 previous(std::sin(tilt_rad), 0.0, std::cos(tilt_rad));
            const auto amplitude_at = [&](const Vector3& direction) {
                return larmor::sh::amplitude(fibre, lmax, true,
                                             larmor::rotate_to(previous, direction));
            };
            double largest_on_grid = 0.0;
            for (const Vector3& direction : calibration.grid) {
                largest_on_grid = std::fmax(largest_on_grid, amplitude_at(direction));
            }
            double top = 0.0;
            for (const Vector3& direction : cone) {
                top = std::fmax(top, amplitude_at(direction));
            }
            CHECK(top <= calibration.ratio * largest_on_grid);
        }
    }
}
