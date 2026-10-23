// Small numerical helpers shared by the numerical modules.
#pragma once

#include "common/types.hpp"

namespace larmor {

/// Returns the Euclidean norm of a 3-vector.
double norm3(const Vector3& vector);

/// Rotates `direction` by the rotation that takes +z to `reference`, a unit vector.
///
/// When `reference` is -z, `direction` is negated.
///
/// References:
///   MRtrix3, the function rotate_direction used by its streamline tracking.
///   Tournier JD, Smith R, Raffelt D, et al. MRtrix3: A fast, flexible and open software framework
///   for medical image processing and visualisation. NeuroImage 202 (2019) 116137.
///   doi:10.1016/j.neuroimage.2019.116137
Vector3 rotate_to(const Vector3& reference, const Vector3& direction);

/// Returns x to the power 8, by three squarings.
constexpr double power_8(double x) {
    const double x2 = x * x;
    const double x4 = x2 * x2;
    return x4 * x4;
}

} // namespace larmor
