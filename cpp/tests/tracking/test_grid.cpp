#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <doctest/doctest.h>

#include "tracking/grid.hpp"

namespace {

using larmor::Index;
using larmor::Vector3;
using larmor::tracking::Field;
using larmor::tracking::Grid;
using larmor::tracking::GridShape;
using larmor::tracking::InverseAffine;
using larmor::tracking::Mask;

constexpr GridShape LINEAR_SHAPE{3, 4, 5};

// Linear in x, y and z, so trilinear interpolation is exact inside the grid; the second channel is
// the negative of the first. C order, (3, 4, 5, 2).
std::vector<float> linear_field() {
    std::vector<float> values;
    for (Index x = 0; x < 3; ++x) {
        for (Index y = 0; y < 4; ++y) {
            for (Index z = 0; z < 5; ++z) {
                const auto value = static_cast<float>(x + 10 * y + 100 * z);
                values.push_back(value);
                values.push_back(-value);
            }
        }
    }
    return values;
}

} // namespace

TEST_CASE("contains includes the half-voxel edges") {
    const Grid grid(LINEAR_SHAPE, InverseAffine::Identity());
    CHECK(grid.contains(Vector3(-0.5, -0.5, -0.5)));
    CHECK(grid.contains(Vector3(2.5, 3.5, 4.5)));
    CHECK_FALSE(grid.contains(Vector3(-0.5 - 1e-9, 0.0, 0.0)));
    CHECK_FALSE(grid.contains(Vector3(0.0, 3.5 + 1e-9, 0.0)));
    CHECK_FALSE(grid.contains(Vector3(0.0, 0.0, std::numeric_limits<double>::quiet_NaN())));
}

TEST_CASE("contains is false on a grid with an empty axis") {
    const Grid grid(GridShape{3, 0, 5}, InverseAffine::Identity());
    CHECK_FALSE(grid.contains(Vector3(0.0, -0.5, 0.0)));
}

TEST_CASE("to_voxel applies the translation") {
    InverseAffine inverse = InverseAffine::Identity();
    inverse(0, 3) = 2.0;
    inverse(1, 1) = 0.5;
    const Vector3 voxel = Grid(LINEAR_SHAPE, inverse).to_voxel(Vector3(1.0, 4.0, 3.0));
    CHECK(voxel == Vector3(3.0, 2.0, 3.0));
}

TEST_CASE("interpolation_axis clamps both neighbours and keeps the weight") {
    const auto inside = larmor::tracking::interpolation_axis(1.25, 4);
    CHECK(inside.lower == 1);
    CHECK(inside.upper == 2);
    CHECK(inside.upper_weight == 0.25);
    const auto below = larmor::tracking::interpolation_axis(-0.4, 4);
    CHECK(below.lower == 0);
    CHECK(below.upper == 0);
    CHECK(below.upper_weight == doctest::Approx(0.6));
    const auto above = larmor::tracking::interpolation_axis(3.4, 4);
    CHECK(above.lower == 3);
    CHECK(above.upper == 3);
    const auto not_a_number =
        larmor::tracking::interpolation_axis(std::numeric_limits<double>::quiet_NaN(), 4);
    CHECK(not_a_number.lower == 0);
    CHECK(not_a_number.upper == 0);
}

TEST_CASE("trilinear at a voxel centre is the voxel") {
    const std::vector<float> values = linear_field();
    const Field field(values.data(), LINEAR_SHAPE, 2, InverseAffine::Identity());
    std::array<double, 2> out{};
    field.trilinear(Vector3(2.0, 1.0, 3.0), out);
    CHECK(out == std::array<double, 2>{312.0, -312.0});
}

TEST_CASE("trilinear between voxels is linear") {
    const std::vector<float> values = linear_field();
    const Field field(values.data(), LINEAR_SHAPE, 2, InverseAffine::Identity());
    std::array<double, 2> out{};
    field.trilinear(Vector3(0.5, 1.25, 0.75), out);
    CHECK(std::abs(out[0] - 88.0) < 1e-12);
    CHECK(std::abs(out[1] + 88.0) < 1e-12);
}

TEST_CASE("trilinear clamps past the edge") {
    const std::vector<float> values = linear_field();
    const Field field(values.data(), LINEAR_SHAPE, 2, InverseAffine::Identity());
    std::array<double, 2> out{};
    field.trilinear(Vector3(-0.4, 0.0, 0.0), out);
    CHECK(out[0] == 0.0);
    field.trilinear(Vector3(2.4, 3.0, 4.0), out);
    CHECK(out[0] == 432.0);
}

TEST_CASE("trilinear writes only the channels") {
    const std::vector<float> values = linear_field();
    const Field field(values.data(), LINEAR_SHAPE, 2, InverseAffine::Identity());
    std::array<double, 3> out{7.0, 7.0, 7.0};
    field.trilinear(Vector3(1.0, 1.0, 1.0), out);
    CHECK(out == std::array<double, 3>{111.0, -111.0, 7.0});
}

TEST_CASE("mask is inside over the whole of its nearest voxel") {
    std::vector<float> values(27, 0.0F);
    values[(1 * 3 + 1) * 3 + 1] = 0.5F;
    const Mask mask(values.data(), GridShape{3, 3, 3}, InverseAffine::Identity());
    CHECK(mask.inside(Vector3(1.0, 1.0, 1.0)));
    CHECK(mask.inside(Vector3(1.45, 0.55, 1.45)));
    CHECK_FALSE(mask.inside(Vector3(1.5, 1.0, 1.0)));
    CHECK_FALSE(mask.inside(Vector3(1.0, 1.6, 1.0)));
}

TEST_CASE("mask is outside off its own grid") {
    const std::vector<float> values(27, 1.0F);
    InverseAffine inverse = InverseAffine::Identity();
    inverse(0, 3) = -10.0;
    const Mask mask(values.data(), GridShape{3, 3, 3}, inverse);
    CHECK(mask.inside(Vector3(10.0, 0.0, 0.0)));
    CHECK_FALSE(mask.inside(Vector3(0.0, 0.0, 0.0)));
}
