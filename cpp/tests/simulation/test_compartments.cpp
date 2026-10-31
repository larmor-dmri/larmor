#include <array>
#include <cmath>
#include <numbers>

#include <doctest/doctest.h>

#include "common/maths.hpp"
#include "simulation/compartments/compartments.hpp"

namespace {

using larmor::simulation::PulseTiming;

} // namespace

TEST_CASE("compartments: the sphere roots solve j1'(x) = 0 and satisfy the sum rule") {
    const auto& roots = larmor::simulation::sphere_roots();
    CHECK(roots[0] == doctest::Approx(2.0816).epsilon(1e-4));
    CHECK(roots[1] == doctest::Approx(5.9404).epsilon(1e-4));
    CHECK(roots[2] == doctest::Approx(9.2058).epsilon(1e-4));
    CHECK(roots[3] == doctest::Approx(12.4044).epsilon(1e-4));
    double sum = 0.0;
    for (const double x : roots) {
        CHECK(std::abs((x * x - 2.0) * std::sin(x) + 2.0 * x * std::cos(x)) < 1e-9 * x * x);
        sum += 1.0 / (x * x * (x * x - 2.0));
    }
    CHECK(sum == doctest::Approx(0.1).epsilon(1e-6));
}

TEST_CASE("compartments: the sphere gives the reference values at R = 6 um") {
    const PulseTiming timing{.small_delta_ms = 15.0, .big_delta_ms = 30.0};
    const double diffusivity = 3e-3;
    CHECK(larmor::simulation::sphere(1000.0, diffusivity, 6.0, timing) ==
          doctest::Approx(0.9176).epsilon(2e-4));
    CHECK(larmor::simulation::sphere(3000.0, diffusivity, 6.0, timing) ==
          doctest::Approx(0.7727).epsilon(2e-4));
    CHECK(larmor::simulation::sphere(6000.0, diffusivity, 6.0, timing) ==
          doctest::Approx(0.5971).epsilon(2e-4));
    CHECK(larmor::simulation::sphere(10000.0, diffusivity, 6.0, timing) ==
          doctest::Approx(0.4234).epsilon(2e-4));
    CHECK(larmor::simulation::sphere(0.0, diffusivity, 6.0, timing) == 1.0);
}

TEST_CASE(
    "compartments: the powder-averaged stick matches its series on both sides of the switch") {
    for (const double x : {0.999e-4, 1.001e-4}) {
        const double series = 1.0 - x / 3.0 + x * x / 10.0;
        CHECK(larmor::simulation::powder_stick(x, 1.0) == doctest::Approx(series).epsilon(1e-11));
    }
    CHECK(larmor::simulation::powder_stick(0.0, 1e-3) == 1.0);
}

TEST_CASE("compartments: the relaxation weight is PD (1 - exp(-TR/T1)) exp(-TE/T2)") {
    const double weight = larmor::simulation::relaxation_weight(0.7, 800.0, 80.0, 3000.0, 60.0);
    CHECK(weight == doctest::Approx(0.7 * (1.0 - std::exp(-3000.0 / 800.0)) * std::exp(-0.75)));
}

TEST_CASE("compartments: the cylinder roots solve J1'(x) = 0 and satisfy the sum rules") {
    const auto& roots = larmor::simulation::cylinder_roots();
    CHECK(roots[0] == doctest::Approx(1.841184).epsilon(1e-6));
    CHECK(roots[1] == doctest::Approx(5.331443).epsilon(1e-6));
    CHECK(roots[2] == doctest::Approx(8.536316).epsilon(1e-6));
    double eighth = 0.0;
    double long_pulse = 0.0;
    for (const double x : roots) {
        eighth += 1.0 / (x * x * (x * x - 1.0));
        long_pulse += 1.0 / (x * x * x * x * (x * x - 1.0));
    }
    CHECK(eighth == doctest::Approx(1.0 / 8.0).epsilon(1e-6));
    CHECK(long_pulse == doctest::Approx(7.0 / 192.0).epsilon(1e-12));
}

TEST_CASE("compartments: exp(-y) I0(y) has its reference values on both sides of y = 25") {
    // The references are scipy.special.i0e.
    CHECK(larmor::scaled_bessel_i0(0.0) == 1.0);
    CHECK(larmor::scaled_bessel_i0(1.0) == doctest::Approx(0.4657596075936404).epsilon(1e-14));
    CHECK(larmor::scaled_bessel_i0(24.999) == doctest::Approx(0.08019839425680447).epsilon(1e-14));
    CHECK(larmor::scaled_bessel_i0(25.0) == doctest::Approx(0.08019677354743669).epsilon(1e-14));
    CHECK(larmor::scaled_bessel_i0(40.0) == doctest::Approx(0.06327827987523532).epsilon(1e-14));
}

TEST_CASE("compartments: the dot is 1 and the Watson sticks span the powder stick to the stick") {
    using larmor::simulation::Compartment;
    using larmor::simulation::Model;
    const PulseTiming timing{.small_delta_ms = 15.0, .big_delta_ms = 30.0};
    const Compartment dot{.model = Model::Dot};
    CHECK(larmor::simulation::attenuation(dot, 3000.0, 0.4, timing) == 1.0);
    for (const double cosine : {0.0, 0.5, 1.0}) {
        CHECK(larmor::simulation::watson_stick(2000.0, 1.7e-3, 0.0, cosine) ==
              doctest::Approx(larmor::simulation::powder_stick(2000.0, 1.7e-3)).epsilon(1e-10));
        CHECK(larmor::simulation::watson_stick(2000.0, 1.7e-3, 1e6, cosine) ==
              doctest::Approx(larmor::simulation::stick(2000.0, 1.7e-3, cosine)).epsilon(1e-4));
    }
}

TEST_CASE("compartments: the angular profiles give the attenuation at any cosine") {
    using larmor::simulation::Compartment;
    using larmor::simulation::Model;
    const PulseTiming timing{.small_delta_ms = 15.0, .big_delta_ms = 30.0};
    const std::array<Compartment, 5> compartments{{
        {.model = Model::Stick, .diffusivity = 2e-3},
        {.model = Model::Zeppelin, .diffusivity = 1.7e-3, .perpendicular_diffusivity = 0.5e-3},
        {.model = Model::Cylinder, .diffusivity = 2e-3, .radius_um = 3.0},
        {.model = Model::WatsonStick, .diffusivity = 1.7e-3, .kappa = 0.5},
        {.model = Model::WatsonStick, .diffusivity = 1.7e-3, .kappa = 64.0},
    }};
    for (const Compartment& compartment : compartments) {
        for (const double b_value : {0.0, 1000.0, 10000.0}) {
            const auto profile = larmor::simulation::angular_profile(compartment, b_value, timing);
            for (const double cosine : {-1.0, -0.37, 0.0, 0.123, 0.5, 0.9, 0.999, 1.0}) {
                CHECK(profile(cosine) == doctest::Approx(larmor::simulation::attenuation(
                                                             compartment, b_value, cosine, timing))
                                             .epsilon(1e-7));
            }
        }
    }
}

TEST_CASE("compartments: the Watson profile holds at high b wherever the signal is not tiny") {
    using larmor::simulation::Compartment;
    using larmor::simulation::Model;
    const PulseTiming timing{.small_delta_ms = 15.0, .big_delta_ms = 30.0};
    const Compartment sticks{.model = Model::WatsonStick, .diffusivity = 3e-3, .kappa = 64.0};
    const double b_value = 20000.0;
    const auto profile = larmor::simulation::angular_profile(sticks, b_value, timing);
    int compared = 0;
    for (int step = 0; step <= 200; ++step) {
        const double cosine = step / 200.0;
        const double exact = larmor::simulation::attenuation(sticks, b_value, cosine, timing);
        if (exact > 1e-6) {
            CHECK(profile(cosine) == doctest::Approx(exact).epsilon(1e-6));
            ++compared;
        }
    }
    CHECK(compared > 50);
}
