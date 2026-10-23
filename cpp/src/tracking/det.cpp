#include "tracking/det.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <numbers>

#include <Eigen/Core>

#include "sh/harmonics.hpp"

namespace larmor::tracking {
namespace {
// The number of random directions a seed climbs from before it is given up.
constexpr int MAX_ATTEMPTS_PER_SEED = 20;

// A buffer for the FOD's SH coefficients at a position, of the largest size; only its first
// n_channels entries are written and read, so it is left uninitialised.
using CoefficientBuffer = std::array<double, static_cast<std::size_t>(sh::MAX_COEFFS)>;

// Returns a view of the first n_channels coefficients of buffer.
Eigen::Map<const Vector> channels(const CoefficientBuffer& buffer, Index n_channels) {
    return {buffer.data(), n_channels};
}

} // namespace

DeterministicAlgorithm::DeterministicAlgorithm(double cutoff, double max_angle_deg, Index lmax,
                                               bool full)
    : fod_cutoff(cutoff),
      cos_max_angle(max_angle_deg >= 180.0 ? -std::numeric_limits<double>::infinity()
                                           : std::cos(max_angle_deg * (std::numbers::pi / 180.0))),
      fod_lmax(lmax), full_series(full) {}

std::optional<Vector3>
DeterministicAlgorithm::first_direction(const Field& fod, const Vector3& position, Rng& rng) const {
    CoefficientBuffer coefficients;
    fod.trilinear(position, coefficients);

    for (int iter = 0; iter < MAX_ATTEMPTS_PER_SEED; ++iter) {
        const std::optional<sh::Peak> peak =
            sh::climb(channels(coefficients, fod.n_channels()), fod_lmax, full_series,
                      rng.unit_vector(), climb_settings);

        if (peak.has_value() && peak->amplitude >= fod_cutoff) {
            return peak->direction;
        }
    }
    return std::nullopt;
}

StepResult DeterministicAlgorithm::next_direction(const Field& fod, Vector3 position,
                                                  Vector3 previous_direction, Rng& /*rng*/,
                                                  SamplingStats& /*stats*/) const {
    CoefficientBuffer coefficients;
    fod.trilinear(position, coefficients);
    const std::optional<sh::Peak> peak =
        sh::climb(channels(coefficients, fod.n_channels()), fod_lmax, full_series,
                  previous_direction, climb_settings);
    if (!peak.has_value()) {
        return StopReason::NoDirection;
    }
    const Vector3& direction = peak->direction;
    const double cos_turn = direction.x() * previous_direction.x() +
                            direction.y() * previous_direction.y() +
                            direction.z() * previous_direction.z();
    if (peak->amplitude < fod_cutoff) {
        return StopReason::LowFod;
    }
    if (cos_turn < cos_max_angle) {
        return StopReason::SharpTurn;
    }
    return direction;
}

} // namespace larmor::tracking
