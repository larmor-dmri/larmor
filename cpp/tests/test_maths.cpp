#include <array>
#include <cmath>

#include <doctest/doctest.h>

#include "common/maths.hpp"

namespace {

using larmor::Vector3;

bool close(const Vector3& first, const Vector3& second) {
    return (first - second).cwiseAbs().maxCoeff() < 1e-12;
}

Vector3 unit(const Vector3& vector) {
    return vector / larmor::norm3(vector);
}

} // namespace

TEST_CASE("norm3 of a Pythagorean triple") {
    CHECK(larmor::norm3(Vector3(3.0, 4.0, 12.0)) == 13.0);
}

TEST_CASE("norm3 of zero is zero") {
    CHECK(larmor::norm3(Vector3::Zero()) == 0.0);
}

TEST_CASE("rotate_to takes +z to the reference") {
    const std::array<Vector3, 3> references{Vector3(1.0, 2.0, 3.0), Vector3(-1.0, 0.5, -2.0),
                                            Vector3(0.0, 1.0, 0.0)};
    for (const Vector3& reference : references) {
        const Vector3 unit_reference = unit(reference);
        CHECK(close(larmor::rotate_to(unit_reference, Vector3::UnitZ()), unit_reference));
    }
    CHECK(close(larmor::rotate_to(Vector3::UnitZ(), Vector3::UnitX()), Vector3::UnitX()));
    CHECK(close(larmor::rotate_to(-Vector3::UnitZ(), Vector3::UnitZ()), -Vector3::UnitZ()));
}

TEST_CASE("rotate_to preserves lengths and angles") {
    const Vector3 reference = unit(Vector3(0.3, -0.8, 0.5));
    const Vector3 first = unit(Vector3(0.2, 0.1, 0.9));
    const Vector3 second = unit(Vector3(-0.7, 0.4, 0.3));
    const Vector3 first_rotated = larmor::rotate_to(reference, first);
    const Vector3 second_rotated = larmor::rotate_to(reference, second);
    CHECK(std::abs(larmor::norm3(first_rotated) - 1.0) < 1e-12);
    CHECK(std::abs(first_rotated.dot(second_rotated) - first.dot(second)) < 1e-12);
    // The angle from +z becomes the angle from the reference.
    CHECK(std::abs(first_rotated.dot(reference) - first.z()) < 1e-12);
}

TEST_CASE("power_8 is three squarings") {
    CHECK(larmor::power_8(2.0) == 256.0);
    CHECK(larmor::power_8(-1.5) == 25.62890625);
    const double x = 0.7071067811865476;
    const double x2 = x * x;
    const double x4 = x2 * x2;
    CHECK(larmor::power_8(x) == x4 * x4);
}
