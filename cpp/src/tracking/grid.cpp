#include "tracking/grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

namespace larmor::tracking {

namespace {

// Converts a whole-numbered double to an index clamped to [0, last]; NaN gives 0.
Index clamped_index(double index, Index last) {
    if (!(index > 0.0)) {
        return 0;
    }
    if (index >= static_cast<double>(last)) {
        return last;
    }
    return static_cast<Index>(index);
}

} // namespace

Grid::Grid(const GridShape& shape, InverseAffine inverse_affine)
    : grid_shape(shape), inverse(std::move(inverse_affine)) {}

// Each coordinate is summed from the x term to the translation.
Vector3 Grid::to_voxel(const Vector3& position) const {
    Vector3 voxel;
    for (Index axis = 0; axis < 3; ++axis) {
        voxel(axis) = inverse(axis, 0) * position.x() + inverse(axis, 1) * position.y() +
                      inverse(axis, 2) * position.z() + inverse(axis, 3);
    }
    return voxel;
}

bool Grid::contains(const Vector3& position) const {
    const Vector3 voxel = to_voxel(position);
    for (Index axis = 0; axis < 3; ++axis) {
        const auto size = grid_shape.at(static_cast<std::size_t>(axis));
        const double coordinate = voxel(axis);
        // Written so that a NaN coordinate is off the axis.
        const bool on_axis = -0.5 <= coordinate && coordinate <= static_cast<double>(size) - 0.5;
        if (size < 1 || !on_axis) {
            return false;
        }
    }
    return true;
}

InterpolationAxis interpolation_axis(double coordinate, Index size) {
    const double lower = std::floor(coordinate);
    const double upper_weight = coordinate - lower;
    const Index last = size - 1;
    return {.lower = clamped_index(lower, last),
            .upper = clamped_index(lower + 1.0, last),
            .upper_weight = upper_weight};
}

Field::Field(const float* values, const GridShape& shape, Index n_channels,
             const InverseAffine& inverse_affine)
    : data(values), channels(n_channels), field_grid(shape, inverse_affine) {}

// The corners are summed in the order x, then y, then z, lower before upper.
void Field::trilinear(const Vector3& position, std::span<double> values) const {
    const GridShape& shape = field_grid.shape();
    const Index n_y = shape[1];
    const Index n_z = shape[2];
    const Vector3 voxel = field_grid.to_voxel(position);
    const InterpolationAxis along_x = interpolation_axis(voxel.x(), shape[0]);
    const InterpolationAxis along_y = interpolation_axis(voxel.y(), n_y);
    const InterpolationAxis along_z = interpolation_axis(voxel.z(), n_z);

    const std::span<double> out = values.first(static_cast<std::size_t>(channels));
    std::ranges::fill(out, 0.0);
    const std::array<std::pair<Index, double>, 2> corners_x{
        {{along_x.lower, 1.0 - along_x.upper_weight}, {along_x.upper, along_x.upper_weight}}};
    const std::array<std::pair<Index, double>, 2> corners_y{
        {{along_y.lower, 1.0 - along_y.upper_weight}, {along_y.upper, along_y.upper_weight}}};
    const std::array<std::pair<Index, double>, 2> corners_z{
        {{along_z.lower, 1.0 - along_z.upper_weight}, {along_z.upper, along_z.upper_weight}}};
    for (const auto& [x, weight_x] : corners_x) {
        for (const auto& [y, weight_y] : corners_y) {
            for (const auto& [z, weight_z] : corners_z) {
                const double weight = weight_x * weight_y * weight_z;
                const float* corner = data + ((x * n_y + y) * n_z + z) * channels;
                for (Index channel = 0; channel < channels; ++channel) {
                    out[static_cast<std::size_t>(channel)] +=
                        weight * static_cast<double>(corner[channel]);
                }
            }
        }
    }
}

Mask::Mask(const float* values, const GridShape& shape, const InverseAffine& inverse_affine)
    : data(values), mask_grid(shape, inverse_affine) {}

bool Mask::inside(const Vector3& position) const {
    const Vector3 voxel = mask_grid.to_voxel(position);
    const GridShape& shape = mask_grid.shape();
    Index flat_index = 0;
    for (Index axis = 0; axis < 3; ++axis) {
        const auto size = shape.at(static_cast<std::size_t>(axis));
        const double coordinate = voxel(axis);
        // Written so that a NaN coordinate is off the axis.
        const bool on_axis = coordinate > -0.5 && coordinate < static_cast<double>(size) - 0.5;
        if (!on_axis) {
            return false;
        }
        flat_index = flat_index * size + static_cast<Index>(std::round(coordinate));
    }
    return static_cast<double>(data[flat_index]) >= MASK_THRESHOLD;
}

} // namespace larmor::tracking
