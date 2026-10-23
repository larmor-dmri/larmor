#include <array>
#include <cmath>
#include <numbers>
#include <utility>

#include <doctest/doctest.h>

#include "common/types.hpp"
#include "sh/harmonics.hpp"

namespace {

using larmor::Index;
using larmor::Vector;
using larmor::Vector3;
using larmor::sh::coefficient_index;
using larmor::sh::Derivatives;
using larmor::sh::MAX_LMAX;
using larmor::sh::n_coeffs;
using std::numbers::pi;

// A fixed, irregular series of the even degrees up to lmax.
Vector series(Index lmax) {
    Vector coefficients(n_coeffs(lmax, false));
    for (Index position = 0; position < coefficients.size(); ++position) {
        coefficients(position) = static_cast<double>(position * 37 % 17 - 8) / 9.0;
    }
    return coefficients;
}

// A fixed, irregular series of all degrees up to lmax.
Vector full_series(Index lmax) {
    Vector coefficients(n_coeffs(lmax, true));
    for (Index position = 0; position < coefficients.size(); ++position) {
        coefficients(position) = static_cast<double>(position * 29 % 13 - 6) / 7.0;
    }
    return coefficients;
}

Derivatives at(const Vector& coefficients, Index lmax, bool full, double theta, double phi) {
    return larmor::sh::derivatives(coefficients, lmax, full, theta, phi);
}

// The unit vector at elevation theta and azimuth phi.
Vector3 unit(double theta, double phi) {
    return {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};
}

constexpr std::array<std::pair<double, double>, 5> POINTS{
    {{0.0, 0.0}, {0.3, 0.2}, {1.2, -2.0}, {2.5, 3.0}, {pi, 0.0}}};

// Checks the derivatives at (theta, phi) against central differences with step 1e-5: truncation
// (about step^2) and rounding (about eps / step) both stay far below the tolerance of 1e-6, while
// a wrong term would be off by order 1.
void check_against_finite_differences(const Vector& coefficients, Index lmax, bool full,
                                      double theta, double phi) {
    const double step = 1e-5;
    const Derivatives here = at(coefficients, lmax, full, theta, phi);
    const Derivatives theta_up = at(coefficients, lmax, full, theta + step, phi);
    const Derivatives theta_down = at(coefficients, lmax, full, theta - step, phi);
    const Derivatives phi_up = at(coefficients, lmax, full, theta, phi + step);
    const Derivatives phi_down = at(coefficients, lmax, full, theta, phi - step);
    const double sin_theta = std::sin(theta);
    const auto close = [](double value, double expected) {
        return std::abs(value - expected) < 1e-6 * (1.0 + std::abs(expected));
    };
    CAPTURE(theta);
    CAPTURE(phi);
    CHECK(close(here.d_theta, (theta_up.value - theta_down.value) / (2.0 * step)));
    CHECK(close(here.d_phi, (phi_up.value - phi_down.value) / (2.0 * step) / sin_theta));
    CHECK(close(here.d_theta_theta, (theta_up.d_theta - theta_down.d_theta) / (2.0 * step)));
    CHECK(close(here.d_theta_phi, (phi_up.d_theta - phi_down.d_theta) / (2.0 * step) / sin_theta));
    CHECK(close(here.d_phi_phi, (phi_up.d_phi - phi_down.d_phi) / (2.0 * step) / sin_theta));
}

} // namespace

TEST_CASE("Legendre values match their closed forms") {
    for (const double theta : {0.1, 0.7, 1.5, 2.9}) {
        CAPTURE(theta);
        const auto legendre = larmor::sh::Legendre::at_elevation(4, theta);
        const double x = std::cos(theta);
        CHECK(std::abs(legendre.values(0, 0) - 1.0 / std::sqrt(4.0 * pi)) < 1e-14);
        const double p20 = std::sqrt(5.0 / (4.0 * pi)) * (3.0 * x * x - 1.0) / 2.0;
        CHECK(std::abs(legendre.values(2, 0) - p20) < 1e-14);
        CHECK(legendre.values(1, 2) == 0.0);
    }
}

TEST_CASE("a constant series is flat") {
    Vector coefficients = Vector::Zero(n_coeffs(8, false));
    coefficients(0) = 2.0;
    for (const auto& [theta, phi] : {std::pair{0.3, 0.2}, {1.2, -2.0}, {2.5, 3.0}}) {
        const Derivatives here = at(coefficients, 8, false, theta, phi);
        CHECK(std::abs(here.value - 2.0 / std::sqrt(4.0 * pi)) < 1e-14);
        for (const double derivative :
             {here.d_theta, here.d_phi, here.d_theta_theta, here.d_theta_phi, here.d_phi_phi}) {
            CHECK(std::abs(derivative) < 1e-13);
        }
    }
}

TEST_CASE("derivatives match finite differences") {
    const Vector coefficients = series(8);
    for (const auto& [theta, phi] : {std::pair{0.4, 0.3}, {1.1, 2.0}, {2.3, -1.2}}) {
        check_against_finite_differences(coefficients, 8, false, theta, phi);
    }
}

TEST_CASE("full derivatives match finite differences") {
    const Vector coefficients = full_series(7);
    for (const auto& [theta, phi] : {std::pair{0.4, 0.3}, {1.1, 2.0}, {2.3, -1.2}}) {
        check_against_finite_differences(coefficients, 7, true, theta, phi);
    }
}

TEST_CASE("amplitude matches the value of derivatives") {
    for (const Index lmax : {Index{0}, Index{2}, Index{8}, MAX_LMAX}) {
        const Vector coefficients = series(lmax);
        for (const auto& [theta, phi] : POINTS) {
            CAPTURE(lmax);
            CAPTURE(theta);
            const double value = larmor::sh::amplitude(coefficients, lmax, false, unit(theta, phi));
            CHECK(std::abs(value - at(coefficients, lmax, false, theta, phi).value) < 1e-12);
        }
    }
}

TEST_CASE("full amplitude matches the value of derivatives") {
    for (const Index lmax : {Index{0}, Index{1}, Index{2}, Index{7}, Index{8}, MAX_LMAX}) {
        const Vector coefficients = full_series(lmax);
        for (const auto& [theta, phi] : POINTS) {
            CAPTURE(lmax);
            CAPTURE(theta);
            const double value = larmor::sh::amplitude(coefficients, lmax, true, unit(theta, phi));
            CHECK(std::abs(value - at(coefficients, lmax, true, theta, phi).value) < 1e-12);
        }
    }
}

TEST_CASE("the full index orders by degree then order") {
    Index expected = 0;
    for (Index degree = 0; degree <= MAX_LMAX; ++degree) {
        for (Index order = -degree; order <= degree; ++order) {
            CHECK(coefficient_index(degree, order, true) == expected);
            ++expected;
        }
    }
    CHECK(expected == n_coeffs(MAX_LMAX, true));
    CHECK(larmor::sh::MAX_COEFFS == n_coeffs(MAX_LMAX, true));
}

TEST_CASE("degree 1 matches its closed forms") {
    // At degree 1, order 0 is c cos theta, and orders 1 and -1 are -c sin theta times cos phi and
    // sin phi, with c = sqrt(3 / (4 pi)).
    const double normalisation = std::sqrt(3.0 / (4.0 * pi));
    for (const auto& [theta, phi] : {std::pair{0.7, 0.4}, {2.2, -1.3}}) {
        const std::array<double, 3> expected{-normalisation * std::sin(theta) * std::sin(phi),
                                             normalisation * std::cos(theta),
                                             -normalisation * std::sin(theta) * std::cos(phi)};
        for (Index order = -1; order <= 1; ++order) {
            CAPTURE(order);
            Vector coefficients = Vector::Zero(n_coeffs(1, true));
            coefficients(coefficient_index(1, order, true)) = 1.0;
            const double target = expected.at(static_cast<std::size_t>(order + 1));
            CHECK(std::abs(at(coefficients, 1, true, theta, phi).value - target) < 1e-14);
            CHECK(std::abs(larmor::sh::amplitude(coefficients, 1, true, unit(theta, phi)) -
                           target) < 1e-14);
        }
    }
}

TEST_CASE("a symmetric series written in full gives the same values") {
    const Index lmax = 8;
    const Vector even = series(lmax);
    Vector full = Vector::Zero(n_coeffs(lmax, true));
    for (Index degree = 0; degree <= lmax; degree += 2) {
        for (Index order = -degree; order <= degree; ++order) {
            full(coefficient_index(degree, order, true)) =
                even(coefficient_index(degree, order, false));
        }
    }
    for (const auto& [theta, phi] : POINTS) {
        CAPTURE(theta);
        const Derivatives from_even = at(even, lmax, false, theta, phi);
        const Derivatives from_full = at(full, lmax, true, theta, phi);
        CHECK(std::abs(from_even.value - from_full.value) < 1e-12);
        CHECK(std::abs(from_even.d_theta - from_full.d_theta) < 1e-12);
        CHECK(std::abs(from_even.d_phi - from_full.d_phi) < 1e-12);
        CHECK(std::abs(from_even.d_theta_theta - from_full.d_theta_theta) < 1e-12);
        const Vector3 direction = unit(theta, phi);
        CHECK(std::abs(larmor::sh::amplitude(even, lmax, false, direction) -
                       larmor::sh::amplitude(full, lmax, true, direction)) < 1e-12);
    }
}

TEST_CASE("odd degrees change sign at the opposite direction") {
    // Y(degree, order)(-d) = (-1)^degree Y(degree, order)(d): a series of odd degrees only is
    // antisymmetric.
    const Index lmax = 7;
    Vector coefficients = full_series(lmax);
    for (Index degree = 0; degree <= lmax; degree += 2) {
        for (Index order = -degree; order <= degree; ++order) {
            coefficients(coefficient_index(degree, order, true)) = 0.0;
        }
    }
    for (const auto& [theta, phi] : {std::pair{0.3, 0.2}, {1.2, -2.0}, {2.5, 3.0}}) {
        CAPTURE(theta);
        const double here = at(coefficients, lmax, true, theta, phi).value;
        const double opposite = at(coefficients, lmax, true, pi - theta, phi + pi).value;
        CHECK(std::abs(here) > 1e-4);
        CHECK(std::abs(here + opposite) < 1e-12);
        const Vector3 direction = unit(theta, phi);
        CHECK(std::abs(larmor::sh::amplitude(coefficients, lmax, true, direction) +
                       larmor::sh::amplitude(coefficients, lmax, true, -direction)) < 1e-12);
    }
}
