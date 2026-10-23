#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>
#include <variant>
#include <vector>

#include <doctest/doctest.h>

#include "sh/harmonics.hpp"
#include "tracking/algorithm.hpp"
#include "tracking/det.hpp"
#include "tracking/grid.hpp"
#include "tracking/rng.hpp"

namespace {

using larmor::Index;
using larmor::Vector;
using larmor::Vector3;
using larmor::tracking::DeterministicAlgorithm;
using larmor::tracking::Field;
using larmor::tracking::GridShape;
using larmor::tracking::InverseAffine;
using larmor::tracking::Rng;
using larmor::tracking::SamplingStats;
using larmor::tracking::StepResult;
using larmor::tracking::StopReason;

constexpr Index LMAX = 8;
constexpr GridShape SHAPE{3, 3, 3};

Vector3 position() {
    return {1.0, 1.0, 1.0};
}

// The coefficients of a single fibre along z, of the even degrees or of every degree (a lobe
// along +z only), as float32.
std::vector<float> fibre_coefficients(bool full) {
    std::vector<float> coefficients(static_cast<std::size_t>(larmor::sh::n_coeffs(LMAX, full)),
                                    0.0F);
    for (Index degree = 0; degree <= LMAX; degree += full ? 1 : 2) {
        const auto degree_value = static_cast<double>(degree);
        coefficients[static_cast<std::size_t>(larmor::sh::coefficient_index(degree, 0, full))] =
            static_cast<float>(std::sqrt(2.0 * degree_value + 1.0) *
                               std::exp(-degree_value * (degree_value + 1.0) / 30.0));
    }
    return coefficients;
}

// A 3x3x3 volume with the same coefficients in every voxel, C order.
std::vector<float> volume(const std::vector<float>& coefficients) {
    std::vector<float> values;
    values.reserve(27 * coefficients.size());
    for (int voxel = 0; voxel < 27; ++voxel) {
        values.insert(values.end(), coefficients.begin(), coefficients.end());
    }
    return values;
}

// The even-degree fibre written in the full layout, odd degrees zero.
std::vector<float> in_full(const std::vector<float>& even) {
    std::vector<float> full(static_cast<std::size_t>(larmor::sh::n_coeffs(LMAX, true)), 0.0F);
    for (Index degree = 0; degree <= LMAX; degree += 2) {
        for (Index order = -degree; order <= degree; ++order) {
            full[static_cast<std::size_t>(larmor::sh::coefficient_index(degree, order, true))] =
                even[static_cast<std::size_t>(larmor::sh::coefficient_index(degree, order, false))];
        }
    }
    return full;
}

Field field(const std::vector<float>& values, bool full) {
    return {values.data(), SHAPE, larmor::sh::n_coeffs(LMAX, full), InverseAffine::Identity()};
}

double peak_amplitude(const std::vector<float>& coefficients, bool full) {
    Vector series(static_cast<Index>(coefficients.size()));
    for (Index index = 0; index < series.size(); ++index) {
        series(index) = static_cast<double>(coefficients[static_cast<std::size_t>(index)]);
    }
    return larmor::sh::derivatives(series, LMAX, full, 0.0, 0.0).value;
}

Vector3 tilted(double degrees) {
    const double angle = degrees * (std::numbers::pi / 180.0);
    return {std::sin(angle), 0.0, std::cos(angle)};
}

bool close(const std::optional<Vector3>& first, const Vector3& second) {
    return first.has_value() && (*first - second).cwiseAbs().maxCoeff() < 1e-6;
}

std::optional<Vector3> direction_of(const StepResult& step) {
    if (const auto* direction = std::get_if<Vector3>(&step)) {
        return *direction;
    }
    return std::nullopt;
}

} // namespace

TEST_CASE("det steps to the nearest peak") {
    const std::vector<float> values = volume(fibre_coefficients(false));
    const Field fod = field(values, false);
    const DeterministicAlgorithm det(0.1, 60.0, LMAX, false);
    Rng rng = Rng::streamline(0, 0);
    SamplingStats stats;
    CHECK(close(direction_of(det.next_direction(fod, position(), tilted(20.0), rng, stats)),
                Vector3::UnitZ()));
    // The FOD is antipodally symmetric: from below, the peak is -z.
    CHECK(close(direction_of(det.next_direction(fod, position(), tilted(160.0), rng, stats)),
                -Vector3::UnitZ()));
}

TEST_CASE("det below the cutoff is LowFod") {
    const std::vector<float> coefficients = fibre_coefficients(false);
    const std::vector<float> values = volume(coefficients);
    const DeterministicAlgorithm det(1.01 * peak_amplitude(coefficients, false), 60.0, LMAX, false);
    Rng rng = Rng::streamline(0, 0);
    SamplingStats stats;
    const StepResult step =
        det.next_direction(field(values, false), position(), Vector3::UnitZ(), rng, stats);
    CHECK(step == StepResult(StopReason::LowFod));
}

TEST_CASE("det turning more than the maximum angle is SharpTurn") {
    const std::vector<float> values = volume(fibre_coefficients(false));
    const Field fod = field(values, false);
    Rng rng = Rng::streamline(0, 0);
    SamplingStats stats;
    const DeterministicAlgorithm narrow(0.1, 10.0, LMAX, false);
    CHECK(narrow.next_direction(fod, position(), tilted(30.0), rng, stats) ==
          StepResult(StopReason::SharpTurn));
    const DeterministicAlgorithm any_turn(0.1, 180.0, LMAX, false);
    CHECK(direction_of(any_turn.next_direction(fod, position(), tilted(30.0), rng, stats))
              .has_value());
}

TEST_CASE("det's first direction climbs from one random direction") {
    const std::vector<float> coefficients = fibre_coefficients(false);
    const std::vector<float> values = volume(coefficients);
    const Field fod = field(values, false);
    const DeterministicAlgorithm det(0.1, 60.0, LMAX, false);
    Rng rng = Rng::streamline(0, 0);
    const std::optional<Vector3> found = det.first_direction(fod, position(), rng);
    CHECK((close(found, Vector3::UnitZ()) || close(found, -Vector3::UnitZ())));
    const DeterministicAlgorithm hopeless(1.01 * peak_amplitude(coefficients, false), 60.0, LMAX,
                                          false);
    Rng again = Rng::streamline(0, 0);
    CHECK_FALSE(hopeless.first_direction(fod, position(), again).has_value());
}

TEST_CASE("det on a symmetric fibre in full steps as on the symmetric one") {
    const std::vector<float> even = fibre_coefficients(false);
    const std::vector<float> even_values = volume(even);
    const std::vector<float> full_values = volume(in_full(even));
    const Field even_fod = field(even_values, false);
    const Field full_fod = field(full_values, true);
    const DeterministicAlgorithm even_det(0.1, 60.0, LMAX, false);
    const DeterministicAlgorithm full_det(0.1, 60.0, LMAX, true);
    for (const double degrees : {0.0, 20.0, 50.0, 130.0, 160.0, 180.0}) {
        CAPTURE(degrees);
        Rng rng = Rng::streamline(0, 0);
        SamplingStats stats;
        const StepResult even_step =
            even_det.next_direction(even_fod, position(), tilted(degrees), rng, stats);
        const StepResult full_step =
            full_det.next_direction(full_fod, position(), tilted(degrees), rng, stats);
        CHECK(even_step == full_step);
    }
    for (std::uint64_t index = 0; index < 20; ++index) {
        Rng even_rng = Rng::streamline(0, index);
        Rng full_rng = Rng::streamline(0, index);
        CHECK(even_det.first_direction(even_fod, position(), even_rng) ==
              full_det.first_direction(full_fod, position(), full_rng));
    }
}

TEST_CASE("det on a one-sided lobe has a direction only on its side") {
    const std::vector<float> coefficients = fibre_coefficients(true);
    const std::vector<float> values = volume(coefficients);
    const Field fod = field(values, true);
    // Above the ringing of the truncated series, 2% of the peak along -z.
    const DeterministicAlgorithm det(0.05 * peak_amplitude(coefficients, true), 60.0, LMAX, true);
    Rng rng = Rng::streamline(0, 0);
    SamplingStats stats;
    CHECK(close(direction_of(det.next_direction(fod, position(), tilted(20.0), rng, stats)),
                Vector3::UnitZ()));
    // From below there is no lobe to follow; the step does not flip to +z.
    CHECK_FALSE(
        direction_of(det.next_direction(fod, position(), tilted(160.0), rng, stats)).has_value());
    // One climb from a random direction: it finds +z, or nothing from the far side.
    bool any_found = false;
    for (std::uint64_t index = 0; index < 20; ++index) {
        Rng search = Rng::streamline(0, index);
        const std::optional<Vector3> found = det.first_direction(fod, position(), search);
        if (found.has_value()) {
            any_found = true;
            CHECK(close(found, Vector3::UnitZ()));
        }
    }
    CHECK(any_found);
}
