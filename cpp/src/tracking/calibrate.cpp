#include "tracking/calibrate.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <utility>

#include <Eigen/Core>

#include "sh/harmonics.hpp"

namespace larmor::tracking {

namespace {

// sqrt(3) / 2, the height of a row of the hexagonal lattice.
constexpr double SQRT_3_OVER_2 = 0.866025403784439;
// The step, in radians, at which the reference fibre's amplitude is sampled.
constexpr double ELEVATION_STEP_RAD = 0.001;

// Converts a double to an index, saturating at the ends of the range of Index; NaN gives 0.
Index saturating_index(double value) {
    if (std::isnan(value)) {
        return 0;
    }
    constexpr auto LIMIT = static_cast<double>(std::numeric_limits<Index>::max());
    if (value >= LIMIT) {
        return std::numeric_limits<Index>::max();
    }
    if (value <= -LIMIT) {
        return std::numeric_limits<Index>::min();
    }
    return static_cast<Index>(value);
}

} // namespace

// Builds the hexagonal grid of directions; see the header.
std::vector<Vector3> direction_grid(double max_angle_rad, double spacing_rad) {
    const double radius_in_steps = max_angle_rad / spacing_rad;
    const double max_radius_squared = radius_in_steps * radius_in_steps;
    const Index extent = saturating_index(std::ceil(radius_in_steps));
    std::vector<Vector3> grid;
    for (Index i_lattice = -extent; i_lattice <= extent; ++i_lattice) {
        for (Index j_lattice = -extent; j_lattice <= extent; ++j_lattice) {
            const double x = static_cast<double>(i_lattice) + 0.5 * static_cast<double>(j_lattice);
            const double y = SQRT_3_OVER_2 * static_cast<double>(j_lattice);
            const double radius_squared = x * x + y * y;
            if (radius_squared > max_radius_squared) {
                continue;
            }
            const double angle = spacing_rad * std::sqrt(radius_squared);
            const double scale = angle > 0.0 ? std::sin(angle) / std::sqrt(radius_squared) : 0.0;
            grid.emplace_back(scale * x, scale * y, std::cos(angle));
        }
    }
    return grid;
}

// Computes the calibration by trying grid spacings from the lobe of a reference fibre along +z.
Calibration calibrate(Index lmax, bool full, double max_angle_rad) {
    Vector fibre = Vector::Zero(sh::n_coeffs(lmax, full));
    for (Index degree = 0; degree <= lmax; degree += full ? 1 : 2) {
        fibre(sh::coefficient_index(degree, 0, full)) =
            std::sqrt(static_cast<double>(2 * degree + 1) / (4.0 * std::numbers::pi));
    }

    // The fibre's amplitude from +z outwards, up to the cone's edge or the first amplitude that
    // is not positive (kept).
    std::vector<std::pair<double, double>> amplitudes;
    double elevation = 0.0;
    while (elevation < max_angle_rad) {
        const double amplitude = sh::amplitude(
            fibre, lmax, full, Vector3(std::sin(elevation), 0.0, std::cos(elevation)));
        amplitudes.emplace_back(elevation, amplitude);
        if (!std::isfinite(amplitude) || amplitude <= 0.0) {
            break;
        }
        elevation += ELEVATION_STEP_RAD;
    }
    if (amplitudes.empty()) {
        return {.grid = {}, .ratio = std::numeric_limits<double>::quiet_NaN()};
    }

    const double zero = amplitudes.back().first;
    const double on_axis = amplitudes.front().second;
    double best = std::numeric_limits<double>::infinity();
    double theta = std::numeric_limits<double>::quiet_NaN();
    double ratio = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t sample = 1; sample < amplitudes.size(); ++sample) {
        const auto [sample_elevation, sample_amplitude] = amplitudes[sample];
        const double n_cone = max_angle_rad * max_angle_rad * (1.0 + on_axis / sample_amplitude) /
                              (2.0 * zero * zero);
        const double n_evaluations =
            n_cone + static_cast<double>(direction_grid(max_angle_rad + sample_elevation,
                                                        std::numbers::sqrt3 * sample_elevation)
                                             .size());
        if (n_evaluations > 0.0 && n_evaluations < best) {
            best = n_evaluations;
            theta = sample_elevation;
            ratio = on_axis / sample_amplitude;
        }
    }
    return {.grid = direction_grid(max_angle_rad + theta, std::numbers::sqrt3 * theta),
            .ratio = ratio};
}

} // namespace larmor::tracking
