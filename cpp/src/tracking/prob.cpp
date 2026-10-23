#include "tracking/prob.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <utility>

#include <Eigen/Core>

#include "common/maths.hpp"
#include "sh/harmonics.hpp"

namespace larmor::tracking {

namespace {

// A buffer for the FOD's SH coefficients at a position, of the largest size; only its first
// n_channels entries are written and read, so it is left uninitialised.
using CoefficientBuffer = std::array<double, static_cast<std::size_t>(sh::MAX_COEFFS)>;

// Converts degrees to radians.
double to_radians(double degrees) {
    return degrees * (std::numbers::pi / 180.0);
}

} // namespace

ProbabilisticAlgorithm::ProbabilisticAlgorithm(double cutoff, double max_angle_deg, Index lmax,
                                               bool full, Index trials, Index attempts)
    : ProbabilisticAlgorithm(cutoff, max_angle_deg, lmax, full, trials, attempts,
                             calibrate(lmax, full, to_radians(max_angle_deg))) {}

ProbabilisticAlgorithm::ProbabilisticAlgorithm(double cutoff, double max_angle_deg, Index lmax,
                                               bool full, Index trials, Index attempts,
                                               Calibration calibration)
    : fod_cutoff(cutoff), max_angle_rad(to_radians(max_angle_deg)),
      sin_bound(max_angle_rad >= std::numbers::pi / 2.0 ? 1.0 : std::sin(max_angle_rad)),
      fod_lmax(lmax), full_series(full), n_trials(trials), n_attempts(attempts),
      bound_calibration(std::move(calibration)) {}

Index ProbabilisticAlgorithm::calibration_directions() const {
    return static_cast<Index>(bound_calibration.grid.size());
}

std::optional<Vector3>
ProbabilisticAlgorithm::first_direction(const Field& fod, const Vector3& position, Rng& rng) const {
    CoefficientBuffer buffer;
    fod.trilinear(position, buffer);
    const Eigen::Map<const Vector> coefficients(buffer.data(), fod.n_channels());
    for (Index attempt = 0; attempt < n_attempts; ++attempt) {
        const Vector3 direction = rng.unit_vector();
        if (sh::amplitude(coefficients, fod_lmax, full_series, direction) > fod_cutoff) {
            return direction;
        }
    }
    return std::nullopt;
}

StepResult ProbabilisticAlgorithm::next_direction(const Field& fod, Vector3 position,
                                                  Vector3 previous_direction, Rng& rng,
                                                  SamplingStats& stats) const {
    CoefficientBuffer buffer;
    fod.trilinear(position, buffer);
    const Eigen::Map<const Vector> coefficients(buffer.data(), fod.n_channels());

    // std::fmax skips a NaN amplitude, so largest is never NaN.
    double largest = 0.0;
    for (const Vector3& grid_direction : bound_calibration.grid) {
        largest = std::fmax(largest, sh::amplitude(coefficients, fod_lmax, full_series,
                                                   rotate_to(previous_direction, grid_direction)));
    }
    if (largest <= 0.0) {
        return StopReason::LowFod;
    }
    const double bound = largest * bound_calibration.ratio;
    ++stats.sampled_steps;
    for (Index trial = 0; trial < n_trials; ++trial) {
        const Vector3 candidate =
            rotate_to(previous_direction, rng.unit_vector_in_cone(max_angle_rad, sin_bound));
        const double amplitude = sh::amplitude(coefficients, fod_lmax, full_series, candidate);
        if (std::isnan(amplitude) || amplitude <= fod_cutoff) {
            continue;
        }
        if (amplitude > bound) {
            ++stats.truncations;
            stats.max_truncation = std::fmax(stats.max_truncation, amplitude / bound);
        }
        if (rng.uniform() < amplitude / bound) {
            stats.accepted_index_sum += static_cast<std::uint64_t>(trial);
            const double length = norm3(candidate);
            return Vector3(candidate.x() / length, candidate.y() / length, candidate.z() / length);
        }
    }
    return StopReason::NoDirection;
}

} // namespace larmor::tracking
