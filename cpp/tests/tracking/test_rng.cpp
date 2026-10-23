#include <array>
#include <cmath>
#include <numbers>

#include <doctest/doctest.h>

#include "common/maths.hpp"
#include "tracking/rng.hpp"

namespace {

using larmor::Vector3;
using larmor::tracking::Rng;

double degrees_to_radians(double degrees) {
    return degrees * std::numbers::pi / 180.0;
}

} // namespace

// Values from the numpy twin, uniform_array(42, [7, 7, 7], [0, 1, 2**63]).
TEST_CASE("streamline draws match numpy") {
    Rng rng = Rng::streamline(42, 7);
    CHECK(rng.key() == 15985126342852841771ULL);
    CHECK(rng.uniform() == 0.8940814753778417);
    CHECK(rng.uniform() == 0.5331080354362024);
}

TEST_CASE("search draws from its own range") {
    const Rng rng = Rng::streamline(42, 7);
    Rng search = rng.search();
    CHECK(search.uniform() == 0.45263234128120544);
    CHECK(rng.counter() == 0);
}

TEST_CASE("uniform is in the unit interval") {
    Rng rng = Rng::streamline(0, 0);
    bool all_inside = true;
    for (int draw = 0; draw < 10000; ++draw) {
        const double value = rng.uniform();
        all_inside = all_inside && value >= 0.0 && value < 1.0;
    }
    CHECK(all_inside);
}

TEST_CASE("unit_vector has unit norm and uses two draws") {
    Rng rng = Rng::streamline(3, 11);
    bool all_unit = true;
    for (int draw = 0; draw < 1000; ++draw) {
        all_unit = all_unit && std::abs(larmor::norm3(rng.unit_vector()) - 1.0) < 1e-12;
    }
    CHECK(all_unit);
    CHECK(rng.counter() == 2000);
}

TEST_CASE("unit_vector_in_cone covers the cap evenly") {
    // On a uniform cap of half-angle a, cos theta is uniform on [cos a, 1].
    const std::array<double, 3> angles{degrees_to_radians(15.0), degrees_to_radians(60.0),
                                       degrees_to_radians(120.0)};
    for (const double angle : angles) {
        CAPTURE(angle);
        const double sin_bound = angle > std::numbers::pi / 2.0 ? 1.0 : std::sin(angle);
        Rng rng = Rng::streamline(5, 9);
        const int n_draws = 100000;
        double sum = 0.0;
        bool all_in_cap = true;
        for (int draw = 0; draw < n_draws; ++draw) {
            const Vector3 direction = rng.unit_vector_in_cone(angle, sin_bound);
            all_in_cap = all_in_cap && std::abs(larmor::norm3(direction) - 1.0) < 1e-12 &&
                         direction.z() >= std::cos(angle) - 1e-12;
            sum += direction.z();
        }
        CHECK(all_in_cap);
        const double mean = sum / n_draws;
        const double expected = (1.0 + std::cos(angle)) / 2.0;
        // The standard error of the mean is (1 - cos a) / sqrt(12 n).
        const double standard_error = (1.0 - std::cos(angle)) / std::sqrt(12.0 * n_draws);
        CHECK(std::abs(mean - expected) < 5.0 * standard_error);
    }
}
