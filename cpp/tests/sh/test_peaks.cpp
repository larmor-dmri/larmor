#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <vector>

#include <doctest/doctest.h>

#include "common/types.hpp"
#include "sh/harmonics.hpp"
#include "sh/peaks.hpp"

namespace {

using larmor::Index;
using larmor::Vector;
using larmor::Vector3;
using larmor::sh::ClimbSettings;
using larmor::sh::coefficient_index;
using larmor::sh::n_coeffs;
using larmor::sh::Peak;

constexpr Index LMAX = 8;

double taper(Index degree) {
    const auto degree_f = static_cast<double>(degree);
    return std::sqrt(2.0 * degree_f + 1.0) * std::exp(-degree_f * (degree_f + 1.0) / 30.0);
}

// A single fibre along z: order-0 terms only, tapered with the degree so that the lobe has no
// side rings (equal weights would leave a ring at about 50 degrees).
Vector fibre_z() {
    Vector coefficients = Vector::Zero(n_coeffs(LMAX, false));
    for (Index degree = 0; degree <= LMAX; degree += 2) {
        coefficients(coefficient_index(degree, 0, false)) = taper(degree);
    }
    return coefficients;
}

// The same fibre with all degrees: odd degrees weighted by odd_weight, so 0 gives the symmetric
// fibre, 1 a one-sided lobe along +z and -1 one along -z.
Vector full_lobe(double odd_weight) {
    Vector coefficients = Vector::Zero(n_coeffs(LMAX, true));
    for (Index degree = 0; degree <= LMAX; ++degree) {
        const double weight = degree % 2 == 1 ? odd_weight : 1.0;
        coefficients(coefficient_index(degree, 0, true)) = weight * taper(degree);
    }
    return coefficients;
}

double amplitude_at(const Vector& coefficients, bool full, const Vector3& direction) {
    return larmor::sh::derivatives(coefficients, LMAX, full,
                                   std::acos(std::clamp(direction.z(), -1.0, 1.0)),
                                   std::atan2(direction.y(), direction.x()))
        .value;
}

// n_points points spread evenly on the sphere (a Fibonacci spiral), each with its 6 nearest
// points as neighbours.
larmor::sh::Sphere fibonacci_sphere(Index n_points) {
    const double golden = std::numbers::pi * (3.0 - std::sqrt(5.0));
    larmor::sh::Sphere sphere;
    for (Index point = 0; point < n_points; ++point) {
        const double z =
            1.0 - 2.0 * (static_cast<double>(point) + 0.5) / static_cast<double>(n_points);
        const double radius = std::sqrt(1.0 - z * z);
        const double angle = golden * static_cast<double>(point);
        sphere.vertices.emplace_back(radius * std::cos(angle), radius * std::sin(angle), z);
    }
    for (Index point = 0; point < n_points; ++point) {
        const Vector3& here = sphere.vertices[static_cast<std::size_t>(point)];
        std::vector<Index> by_distance;
        for (Index other = 0; other < n_points; ++other) {
            if (other != point) {
                by_distance.push_back(other);
            }
        }
        std::ranges::stable_sort(by_distance, [&](Index first, Index second) {
            return here.dot(sphere.vertices[static_cast<std::size_t>(first)]) >
                   here.dot(sphere.vertices[static_cast<std::size_t>(second)]);
        });
        std::array<Index, 6> neighbours{};
        std::copy_n(by_distance.begin(), 6, neighbours.begin());
        sphere.neighbours.push_back(neighbours);
    }
    return sphere;
}

struct SearchResult {
    Vector directions; // n_peaks * 3
    Vector amplitudes; // n_peaks
};

SearchResult search(const Vector& coefficients, bool full, Index n_peaks) {
    const larmor::sh::Sphere sphere = fibonacci_sphere(400);
    Vector on_sphere(static_cast<Index>(sphere.vertices.size()));
    for (Index vertex = 0; vertex < on_sphere.size(); ++vertex) {
        on_sphere(vertex) =
            amplitude_at(coefficients, full, sphere.vertices[static_cast<std::size_t>(vertex)]);
    }
    const larmor::sh::PeakSettings settings{.n_peaks = n_peaks,
                                            .relative_threshold = 0.1,
                                            .threshold = 0.0,
                                            .climb_settings = ClimbSettings{}};
    SearchResult result{.directions = Vector::Zero(n_peaks * 3),
                        .amplitudes = Vector::Zero(n_peaks)};
    std::vector<Peak> found;
    larmor::sh::voxel_peaks(coefficients, LMAX, full, on_sphere, sphere, settings, found,
                            result.directions, result.amplitudes);
    return result;
}

// The peak a climb found, or a NaN peak if it found none, so that every check on it fails.
Peak found_or_nan(const std::optional<Peak>& peak) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return peak.value_or(Peak{.direction = Vector3::Constant(nan), .amplitude = nan});
}

Vector3 start_20_degrees_from(double z_sign) {
    const double off = 20.0 * std::numbers::pi / 180.0;
    return {std::sin(off) * 0.6, std::sin(off) * 0.8, z_sign * std::cos(off)};
}

} // namespace

TEST_CASE("climb reaches the fibre axis") {
    const Peak peak = found_or_nan(
        larmor::sh::climb(fibre_z(), LMAX, false, start_20_degrees_from(1.0), ClimbSettings{}));
    CHECK(peak.direction.z() > 1.0 - 1e-9);
}

TEST_CASE("climb from a stationary point stays there") {
    const Vector3 pole = Vector3::UnitZ();
    const Peak peak =
        found_or_nan(larmor::sh::climb(fibre_z(), LMAX, false, pole, ClimbSettings{}));
    CHECK(peak.direction == pole);
    CHECK(std::abs(peak.amplitude - amplitude_at(fibre_z(), false, pole)) < 1e-14);
}

TEST_CASE("climb out of iterations gives nothing") {
    const ClimbSettings settings{.max_iters = 1};
    CHECK_FALSE(
        larmor::sh::climb(fibre_z(), LMAX, false, Vector3(0.5, 0.0, 0.866), settings).has_value());
}

TEST_CASE("climb on a full series reaches the lobe") {
    const Peak peak = found_or_nan(larmor::sh::climb(full_lobe(-1.0), LMAX, true,
                                                     start_20_degrees_from(-1.0), ClimbSettings{}));
    CHECK(peak.direction.z() < -1.0 + 1e-9);
}

TEST_CASE("one fibre gives one peak along +z") {
    const SearchResult result = search(fibre_z(), false, 3);
    CHECK(result.directions(2) > 1.0 - 1e-9);
    CHECK(std::abs(result.amplitudes(0) - amplitude_at(fibre_z(), false, Vector3::UnitZ())) < 1e-9);
    CHECK(result.directions.tail(6).array().isNaN().all());
    CHECK(result.amplitudes.tail(2).array().isNaN().all());
}

TEST_CASE("NaN coefficients give no peak") {
    const Vector coefficients =
        Vector::Constant(n_coeffs(LMAX, false), std::numeric_limits<double>::quiet_NaN());
    const SearchResult result = search(coefficients, false, 2);
    CHECK(result.directions.array().isNaN().all());
    CHECK(result.amplitudes.array().isNaN().all());
}

TEST_CASE("MAX_LMAX fits the series used here") {
    CHECK(n_coeffs(larmor::sh::MAX_LMAX, false) > n_coeffs(LMAX, false));
}

TEST_CASE("a symmetric fibre in full gives a peak and its opposite") {
    const SearchResult result = search(full_lobe(0.0), true, 3);
    const double first_z = result.directions(2);
    const double second_z = result.directions(5);
    CHECK(std::fmax(first_z, second_z) == std::abs(first_z));
    CHECK(std::abs(first_z) > 1.0 - 1e-9);
    CHECK(std::abs(first_z + second_z) < 1e-9);
    CHECK(std::abs(result.amplitudes(0) - result.amplitudes(1)) < 1e-9);
    CHECK(std::isnan(result.amplitudes(2)));
}

TEST_CASE("a one-sided lobe gives one peak with its sign") {
    for (const double sign : {1.0, -1.0}) {
        CAPTURE(sign);
        const Vector coefficients = full_lobe(sign);
        const SearchResult result = search(coefficients, true, 3);
        CHECK(sign * result.directions(2) > 1.0 - 1e-9);
        const double top = amplitude_at(coefficients, true, Vector3(0.0, 0.0, sign));
        CHECK(std::abs(result.amplitudes(0) - top) < 1e-9);
        CHECK(std::isnan(result.amplitudes(1)));
        CHECK(std::isnan(result.amplitudes(2)));
    }
}
