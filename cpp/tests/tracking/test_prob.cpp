#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>
#include <variant>
#include <vector>

#include <doctest/doctest.h>

#include "common/maths.hpp"
#include "sh/harmonics.hpp"
#include "tracking/algorithm.hpp"
#include "tracking/calibrate.hpp"
#include "tracking/grid.hpp"
#include "tracking/prob.hpp"
#include "tracking/rng.hpp"

namespace {

using larmor::Index;
using larmor::Vector;
using larmor::Vector3;
using larmor::tracking::Calibration;
using larmor::tracking::Field;
using larmor::tracking::GridShape;
using larmor::tracking::InverseAffine;
using larmor::tracking::ProbabilisticAlgorithm;
using larmor::tracking::Rng;
using larmor::tracking::SamplingStats;
using larmor::tracking::StepResult;
using larmor::tracking::StopReason;

constexpr Index LMAX = 8;
constexpr GridShape SHAPE{3, 3, 3};

Vector3 position() {
    return {1.0, 1.0, 1.0};
}

double to_radians(double degrees) {
    return degrees * (std::numbers::pi / 180.0);
}

// A single fibre along z: the even degrees, or every degree (a lobe along +z only).
Vector fibre(bool full) {
    Vector coefficients = Vector::Zero(larmor::sh::n_coeffs(LMAX, full));
    for (Index degree = 0; degree <= LMAX; degree += full ? 1 : 2) {
        const auto degree_value = static_cast<double>(degree);
        coefficients(larmor::sh::coefficient_index(degree, 0, full)) =
            std::sqrt(2.0 * degree_value + 1.0) *
            std::exp(-degree_value * (degree_value + 1.0) / 30.0);
    }
    return coefficients;
}

// A 3x3x3 volume with the same coefficients in every voxel, as float32, C order.
std::vector<float> volume(const Vector& coefficients) {
    std::vector<float> values;
    values.reserve(static_cast<std::size_t>(27 * coefficients.size()));
    for (int voxel = 0; voxel < 27; ++voxel) {
        for (Index index = 0; index < coefficients.size(); ++index) {
            values.push_back(static_cast<float>(coefficients(index)));
        }
    }
    return values;
}

// The coefficients as the step sees them, after the float32 round trip.
Vector stored(const Vector& coefficients) {
    return coefficients.cast<float>().cast<double>();
}

Field field(const std::vector<float>& values, bool full) {
    return {values.data(), SHAPE, larmor::sh::n_coeffs(LMAX, full), InverseAffine::Identity()};
}

Vector3 tilted(double degrees) {
    const double angle = to_radians(degrees);
    return {std::sin(angle), 0.0, std::cos(angle)};
}

double dot(const Vector3& first, const Vector3& second) {
    return first.x() * second.x() + first.y() * second.y() + first.z() * second.z();
}

std::optional<Vector3> direction_of(const StepResult& step) {
    if (const auto* direction = std::get_if<Vector3>(&step)) {
        return *direction;
    }
    return std::nullopt;
}

} // namespace

TEST_CASE("prob steps stay in the cone without truncation on a single fibre") {
    const std::vector<float> values = volume(fibre(false));
    const Field fod = field(values, false);
    const ProbabilisticAlgorithm prob(0.1, 30.0, LMAX, false, 1000, 1000);
    Rng rng = Rng::streamline(1, 2);
    SamplingStats stats;
    const double cos_max = std::cos(to_radians(30.0));
    bool all_in_cone = true;
    for (const Vector3& previous : {Vector3(Vector3::UnitZ()), tilted(10.0), tilted(25.0)}) {
        for (int step = 0; step < 2000; ++step) {
            const std::optional<Vector3> direction =
                direction_of(prob.next_direction(fod, position(), previous, rng, stats));
            all_in_cone = all_in_cone && direction.has_value() &&
                          std::abs(larmor::norm3(*direction) - 1.0) < 1e-12 &&
                          dot(*direction, previous) >= cos_max - 1e-12;
        }
    }
    CHECK(all_in_cone);
    CHECK(stats.sampled_steps == 6000);
    CHECK(stats.truncations == 0);
    CHECK(stats.max_truncation == 0.0);
}

// The accepted directions follow the FOD within the cone: their mean equals the amplitude-weighted
// mean direction over the cone, estimated here from uniform cone samples with another stream of
// random numbers.
TEST_CASE("prob's sampled directions follow the FOD in the cone") {
    const Vector coefficients = fibre(false);
    const std::vector<float> values = volume(coefficients);
    const Field fod = field(values, false);
    const ProbabilisticAlgorithm prob(0.0, 30.0, LMAX, false, 1000, 1000);
    const Vector3 previous = tilted(20.0);
    const int n_steps = 40000;

    Rng rng = Rng::streamline(3, 4);
    SamplingStats stats;
    Vector3 mean = Vector3::Zero();
    for (int step = 0; step < n_steps; ++step) {
        const std::optional<Vector3> direction =
            direction_of(prob.next_direction(fod, position(), previous, rng, stats));
        REQUIRE(direction.has_value());
        if (direction.has_value()) {
            mean += *direction / static_cast<double>(n_steps);
        }
    }

    const Vector series = stored(coefficients);
    const double max_angle = to_radians(30.0);
    Rng reference = Rng::streamline(5, 6);
    Vector3 weighted = Vector3::Zero();
    double total = 0.0;
    for (int sample = 0; sample < 400000; ++sample) {
        const Vector3 direction = larmor::rotate_to(
            previous, reference.unit_vector_in_cone(max_angle, std::sin(max_angle)));
        const double amplitude =
            std::fmax(larmor::sh::amplitude(series, LMAX, false, direction), 0.0);
        weighted += amplitude * direction;
        total += amplitude;
    }
    const Vector3 expected = weighted / total;
    // The sampled mean leans from the cone's axis towards the fibre.
    CHECK(mean.x() < previous.x() - 0.02);
    CHECK((mean - expected).cwiseAbs().maxCoeff() < 4e-3);
}

TEST_CASE("prob below the cutoff stops only when it runs out of trials") {
    const Vector coefficients = fibre(false);
    const std::vector<float> values = volume(coefficients);
    const double peak = larmor::sh::amplitude(stored(coefficients), LMAX, false, Vector3::UnitZ());
    const ProbabilisticAlgorithm prob(1.01 * peak, 15.0, LMAX, false, 50, 1000);
    Rng rng = Rng::streamline(0, 0);
    SamplingStats stats;
    const StepResult step =
        prob.next_direction(field(values, false), position(), Vector3::UnitZ(), rng, stats);
    CHECK(step == StepResult(StopReason::NoDirection));
    CHECK(stats.sampled_steps == 1);
    CHECK(stats.accepted_index_sum == 0);
}

TEST_CASE("prob on an FOD without positive amplitude in the cone is LowFod") {
    const std::vector<float> values = volume(Vector::Zero(larmor::sh::n_coeffs(LMAX, false)));
    const ProbabilisticAlgorithm prob(0.0, 15.0, LMAX, false, 1000, 1000);
    Rng rng = Rng::streamline(0, 0);
    SamplingStats stats;
    const StepResult step =
        prob.next_direction(field(values, false), position(), Vector3::UnitZ(), rng, stats);
    CHECK(step == StepResult(StopReason::LowFod));
    CHECK(stats == SamplingStats{});
}

TEST_CASE("prob counts a bound too tight as truncations") {
    const std::vector<float> values = volume(fibre(false));
    const Field fod = field(values, false);
    Calibration calibration = larmor::tracking::calibrate(LMAX, false, to_radians(30.0));
    calibration.ratio = 0.5;
    const ProbabilisticAlgorithm prob(0.0, 30.0, LMAX, false, 1000, 1000, calibration);
    Rng rng = Rng::streamline(7, 8);
    SamplingStats stats;
    bool all_accepted = true;
    for (int step = 0; step < 500; ++step) {
        all_accepted =
            all_accepted &&
            direction_of(prob.next_direction(fod, position(), Vector3::UnitZ(), rng, stats))
                .has_value();
    }
    CHECK(all_accepted);
    CHECK(stats.truncations > 0);
    CHECK(stats.max_truncation > 1.0);
    CHECK(stats.max_truncation <= 2.0 + 1e-12);
}

TEST_CASE("prob's first direction is any direction above the cutoff") {
    const Vector coefficients = fibre(false);
    const Vector series = stored(coefficients);
    const std::vector<float> values = volume(coefficients);
    const Field fod = field(values, false);
    const double cutoff = 0.5 * larmor::sh::amplitude(series, LMAX, false, Vector3::UnitZ());
    const ProbabilisticAlgorithm prob(cutoff, 15.0, LMAX, false, 1000, 1000);
    for (std::uint64_t index = 0; index < 20; ++index) {
        Rng rng = Rng::streamline(0, index);
        const std::optional<Vector3> direction = prob.first_direction(fod, position(), rng);
        REQUIRE(direction.has_value());
        if (direction.has_value()) {
            CHECK(larmor::sh::amplitude(series, LMAX, false, *direction) > cutoff);
        }
    }
    const ProbabilisticAlgorithm hopeless(4.0 * cutoff, 15.0, LMAX, false, 1000, 1000);
    Rng rng = Rng::streamline(0, 0);
    CHECK_FALSE(hopeless.first_direction(fod, position(), rng).has_value());
}

TEST_CASE("prob on a one-sided lobe follows only its side, under the bound") {
    const Vector coefficients = fibre(true);
    const std::vector<float> values = volume(coefficients);
    const Field fod = field(values, true);
    const double peak = larmor::sh::amplitude(stored(coefficients), LMAX, true, Vector3::UnitZ());
    // Above the ringing of the truncated series, 2% of the peak along -z.
    const ProbabilisticAlgorithm prob(0.05 * peak, 30.0, LMAX, true, 1000, 1000);
    Rng rng = Rng::streamline(1, 2);
    SamplingStats stats;
    const double cos_max = std::cos(to_radians(30.0));
    bool all_in_cone = true;
    for (const Vector3& previous : {Vector3(Vector3::UnitZ()), tilted(10.0), tilted(25.0)}) {
        for (int step = 0; step < 2000; ++step) {
            const std::optional<Vector3> direction =
                direction_of(prob.next_direction(fod, position(), previous, rng, stats));
            all_in_cone = all_in_cone && direction.has_value() &&
                          dot(*direction, previous) >= cos_max - 1e-12;
        }
    }
    CHECK(all_in_cone);
    CHECK(stats.truncations == 0);
    for (const Vector3& previous : {Vector3(-Vector3::UnitZ()), tilted(160.0)}) {
        CHECK_FALSE(
            direction_of(prob.next_direction(fod, position(), previous, rng, stats)).has_value());
    }
    const ProbabilisticAlgorithm half_peak(0.5 * peak, 30.0, LMAX, true, 1000, 1000);
    for (std::uint64_t index = 0; index < 20; ++index) {
        Rng search = Rng::streamline(0, index);
        const std::optional<Vector3> direction = half_peak.first_direction(fod, position(), search);
        REQUIRE(direction.has_value());
        if (direction.has_value()) {
            CHECK(direction->z() > 0.0);
        }
    }
}
