#include "sh/peaks.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "common/maths.hpp"
#include "sh/harmonics.hpp"

namespace larmor::sh {

namespace {

// Two peaks closer than this in cosine are the same peak.
constexpr double DOT_THRESHOLD = 0.99;

// Returns the dot product of two 3-vectors.
double dot3(const Vector3& first, const Vector3& second) {
    return first.x() * second.x() + first.y() * second.y() + first.z() * second.z();
}

// Returns whether `vertex` has a positive amplitude and no neighbour above it.
bool is_local_max(const VectorView& on_sphere, const std::array<Index, 6>& neighbours,
                  Index vertex) {
    const double here = on_sphere(vertex);
    return here > 0.0 && std::ranges::all_of(neighbours, [&](Index neighbour) {
               return on_sphere(neighbour) <= here;
           });
}

} // namespace

std::optional<Peak> climb(const VectorView& coefficients, Index lmax, bool full,
                          const Vector3& start, const ClimbSettings& settings) {
    Vector3 direction = start;
    for (Index iteration = 0; iteration < settings.max_iters; ++iteration) {
        const double theta = std::acos(std::clamp(direction.z(), -1.0, 1.0));
        const double phi = std::atan2(direction.y(), direction.x());
        const Derivatives here = derivatives(coefficients, lmax, full, theta, phi);

        // The unit vector along the gradient, in the (theta, phi) tangent frame.
        const double gradient_norm =
            std::sqrt(here.d_theta * here.d_theta + here.d_phi * here.d_phi);
        double along_theta = 0.0;
        double along_phi = 0.0;
        if (gradient_norm != 0.0) {
            along_theta = here.d_theta / gradient_norm;
            along_phi = here.d_phi / gradient_norm;
        }

        const double slope = along_theta * here.d_theta + along_phi * here.d_phi;
        const double curvature = along_theta * along_theta * here.d_theta_theta +
                                 2.0 * along_theta * along_phi * here.d_theta_phi +
                                 along_phi * along_phi * here.d_phi_phi;
        double step = curvature != 0.0 ? std::abs(slope / curvature) : 0.0;
        step = std::fmin(step, settings.max_step);

        const double sin_theta = std::sin(theta);
        const double cos_theta = std::cos(theta);
        const double sin_phi = std::sin(phi);
        const double cos_phi = std::cos(phi);
        const Vector3 theta_tangent(cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta);
        const Vector3 phi_tangent(-sin_phi, cos_phi, 0.0);
        for (Index axis = 0; axis < 3; ++axis) {
            direction(axis) +=
                step * (along_theta * theta_tangent(axis) + along_phi * phi_tangent(axis));
        }
        const double norm = norm3(direction);
        for (Index axis = 0; axis < 3; ++axis) {
            direction(axis) /= norm;
        }
        if (step < settings.tolerance) {
            return Peak{.direction = direction, .amplitude = here.value};
        }
    }
    return std::nullopt;
}

void voxel_peaks(const VectorView& coefficients, Index lmax, bool full, const VectorView& on_sphere,
                 const Sphere& sphere, const PeakSettings& settings, std::vector<Peak>& found,
                 Eigen::Ref<Vector> directions, Eigen::Ref<Vector> amplitudes) {
    directions.setConstant(std::numeric_limits<double>::quiet_NaN());
    amplitudes.setConstant(std::numeric_limits<double>::quiet_NaN());
    if (on_sphere.size() == 0 || !std::isfinite(on_sphere(0))) {
        return;
    }
    found.clear();

    const auto n_vertices = static_cast<Index>(sphere.neighbours.size());
    for (Index vertex = 0; vertex < n_vertices; ++vertex) {
        if (!is_local_max(on_sphere, sphere.neighbours[static_cast<std::size_t>(vertex)], vertex)) {
            continue;
        }
        const std::optional<Peak> peak =
            climb(coefficients, lmax, full, sphere.vertices[static_cast<std::size_t>(vertex)],
                  settings.climb_settings);
        // Written this way so that a NaN amplitude is dropped too.
        if (!peak || !(peak->amplitude > 0.0)) {
            continue;
        }
        const auto same_peak = [&](const Peak& other) {
            const double cosine = dot3(peak->direction, other.direction);
            return (full ? cosine : std::abs(cosine)) > DOT_THRESHOLD;
        };
        const auto match = std::ranges::find_if(found, same_peak);
        if (match == found.end()) {
            found.push_back(*peak);
        } else if (peak->amplitude > match->amplitude) {
            *match = *peak;
        }
    }
    if (found.empty()) {
        return;
    }
    // Stable, so peaks of equal amplitude keep the order they were found in.
    std::ranges::stable_sort(found, [](const Peak& first, const Peak& second) {
        return first.amplitude > second.amplitude;
    });
    const double largest = found.front().amplitude;
    const Index n_kept = std::min(settings.n_peaks, static_cast<Index>(found.size()));
    for (Index rank = 0; rank < n_kept; ++rank) {
        const Peak& peak = found[static_cast<std::size_t>(rank)];
        if (peak.amplitude < settings.threshold ||
            peak.amplitude < settings.relative_threshold * largest) {
            break;
        }
        const bool flip = !full && peak.direction.z() < 0.0;
        for (Index axis = 0; axis < 3; ++axis) {
            directions(3 * rank + axis) = flip ? -peak.direction(axis) : peak.direction(axis);
        }
        amplitudes(rank) = peak.amplitude;
    }
}

} // namespace larmor::sh
