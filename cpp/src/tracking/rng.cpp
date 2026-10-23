#include "tracking/rng.hpp"

#include <cmath>
#include <numbers>

namespace larmor::tracking {

namespace {

// SplitMix64's golden-ratio increment.
constexpr std::uint64_t GAMMA = 0x9E3779B97F4A7C15;
// The two multipliers of SplitMix64's finaliser.
constexpr std::uint64_t MULTIPLIER_1 = 0xBF58476D1CE4E5B9;
constexpr std::uint64_t MULTIPLIER_2 = 0x94D049BB133111EB;
// The counter at which a first-direction search starts.
constexpr std::uint64_t SEARCH_COUNTER = std::uint64_t{1} << 63;
// 2^-53, which turns the top 53 bits of an output into a number in [0, 1).
constexpr double TO_UNIT = 1.0 / static_cast<double>(std::uint64_t{1} << 53);

// Applies SplitMix64's finaliser. Unsigned products wrap modulo 2^64.
std::uint64_t mix(std::uint64_t value) {
    value = (value ^ (value >> 30)) * MULTIPLIER_1;
    value = (value ^ (value >> 27)) * MULTIPLIER_2;
    return value ^ (value >> 31);
}

} // namespace

Rng::Rng(std::uint64_t key, std::uint64_t counter)
    : generator_key(key), generator_counter(counter) {}

// The key is mix(mix(run_seed) + index).
Rng Rng::streamline(std::uint64_t run_seed, std::uint64_t index) {
    return {mix(mix(run_seed) + index), 0};
}

Rng Rng::search() const {
    return {generator_key, SEARCH_COUNTER};
}

// The k-th output is mix(key + (k + 1) * GAMMA).
double Rng::uniform() {
    const std::uint64_t bits = mix(generator_key + (generator_counter + 1) * GAMMA);
    ++generator_counter;
    return static_cast<double>(bits >> 11) * TO_UNIT;
}

Vector3 Rng::unit_vector() {
    const double z = 2.0 * uniform() - 1.0;
    const double phi = 2.0 * std::numbers::pi * uniform();
    const double radius = std::sqrt(1.0 - z * z);
    return {radius * std::cos(phi), radius * std::sin(phi), z};
}

// A candidate angle t = max_angle_rad * u is kept when sin_bound * u' <= sin t, for fresh
// uniform draws u and u'.
Vector3 Rng::unit_vector_in_cone(double max_angle_rad, double sin_bound) {
    const double phi = 2.0 * std::numbers::pi * uniform();
    double theta = 0.0;
    while (true) {
        theta = max_angle_rad * uniform();
        if (sin_bound * uniform() <= std::sin(theta)) {
            break;
        }
    }
    const double sin_theta = std::sin(theta);
    const double cos_theta = std::cos(theta);
    const double sin_phi = std::sin(phi);
    const double cos_phi = std::cos(phi);
    return {sin_theta * cos_phi, sin_theta * sin_phi, cos_theta};
}

} // namespace larmor::tracking
