#include "common/maths.hpp"

#include <cmath>

namespace larmor {

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

} // namespace larmor
