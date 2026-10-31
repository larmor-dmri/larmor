#include "common/maths.hpp"

#include <cmath>
#include <numbers>

namespace larmor {

namespace {

// y below which exp(-y) I0(y) sums its power series, and above which its asymptotic series.
constexpr double BESSEL_SERIES_LIMIT = 25.0;
// Relative size of the last term at which a series stops.
constexpr double SERIES_TOLERANCE = 1e-17;
// Maximum number of terms of a series.
constexpr int MAX_SERIES_TERMS = 200;

} // namespace

double norm3(const Vector3& vector) {
    return std::sqrt(vector.x() * vector.x() + vector.y() * vector.y() + vector.z() * vector.z());
}

Vector3 rotate_to(const Vector3& reference, const Vector3& direction) {
    const double xy_norm = std::sqrt(reference.x() * reference.x() + reference.y() * reference.y());
    if (xy_norm == 0.0) {
        return reference.z() < 0.0 ? Vector3(-direction) : direction;
    }

    // horizontal is the unit vector of reference projected on the xy plane; tilted is it turned
    // by the rotation.
    const Vector3 horizontal(reference.x() / xy_norm, reference.y() / xy_norm, 0.0);
    const Vector3 tilted(reference.z() * horizontal.x(), reference.z() * horizontal.y(), -xy_norm);
    const double along_z = direction.z();
    const double along_horizontal = direction.x() * horizontal.x() + direction.y() * horizontal.y();

    return {
        direction.x() + along_z * reference.x() + along_horizontal * (tilted.x() - horizontal.x()),
        direction.y() + along_z * reference.y() + along_horizontal * (tilted.y() - horizontal.y()),
        direction.z() + along_z * (reference.z() - 1.0) +
            along_horizontal * (tilted.z() - horizontal.z()),
    };
}

double scaled_bessel_i0(double y) {
    if (y < BESSEL_SERIES_LIMIT) {
        const double quarter_squared = 0.25 * y * y;
        double term = 1.0;
        double sum = 1.0;
        for (int order = 1; order < MAX_SERIES_TERMS && term > SERIES_TOLERANCE * sum; ++order) {
            term *= quarter_squared / (static_cast<double>(order) * order);
            sum += term;
        }
        return std::exp(-y) * sum;
    }
    // 1 / sqrt(2 pi y) sum_k c_k / y^k with c_k = c_{k-1} (2k - 1)^2 / (8 k); the terms fall
    // until k is about 2 y, far beyond the precision needed.
    double term = 1.0;
    double sum = 1.0;
    for (int order = 1; order < MAX_SERIES_TERMS && term > SERIES_TOLERANCE * sum; ++order) {
        term *= (2.0 * order - 1.0) * (2.0 * order - 1.0) / (8.0 * order * y);
        sum += term;
    }
    return sum / std::sqrt(2.0 * std::numbers::pi * y);
}

} // namespace larmor
