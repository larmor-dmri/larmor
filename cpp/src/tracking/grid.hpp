// Images on a voxel grid for tracking, read at world positions (mm).
#pragma once

#include <array>
#include <span>

#include <Eigen/Core>

#include "common/types.hpp"

namespace larmor::tracking {

/// A (4, 4) world-to-voxel matrix, stored row by row.
using InverseAffine = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;

/// The number of voxels of a 3D grid along x, y and z.
using GridShape = std::array<Index, 3>;

/// The mask value from which a position in its voxel counts as inside the mask.
constexpr double MASK_THRESHOLD = 0.5;

/// A voxel grid, with its shape and the matrix that maps world positions to voxel coordinates.
///
/// Voxel coordinates are fractional indices, with the centre of voxel (i, j, k) at (i, j, k). The
/// image covers half a voxel beyond its outermost centres, from -0.5 to size - 0.5 along each axis.
class Grid {
public:
    Grid(const GridShape& shape, InverseAffine inverse_affine);

    /// Returns the voxel coordinates of a world position.
    [[nodiscard]] Vector3 to_voxel(const Vector3& position) const;

    /// Returns whether a world position lies on the image, within half a voxel of its outermost
    /// centres.
    ///
    /// The edges count as inside; a NaN coordinate, or a grid with an empty axis, does not.
    [[nodiscard]] bool contains(const Vector3& position) const;

    /// Returns the grid's shape.
    [[nodiscard]] const GridShape& shape() const { return grid_shape; }

private:
    GridShape grid_shape;
    InverseAffine inverse;
};

/// The two neighbouring voxel indices along one axis and the weight of the upper one.
struct InterpolationAxis {
    /// The lower index, floor(coordinate), clamped to the image.
    Index lower;
    /// The next index, clamped to the image.
    Index upper;
    /// The weight of upper, coordinate - floor(coordinate).
    double upper_weight;
};

/// Returns the neighbours of a voxel coordinate along an axis of size voxels, clamped to the
/// image.
///
/// Both indices are clamped to [0, size - 1], so a coordinate between the outermost centre and
/// the edge of the image reads the outermost voxel twice. upper_weight does not change when an
/// index is clamped. A NaN coordinate gives index 0. size must be at least 1.
InterpolationAxis interpolation_axis(double coordinate, Index size);

/// An image of n_channels float32 values per voxel, such as the SH coefficients of an FOD.
///
/// The values are an (X, Y, Z, n_channels) array stored with the channel varying fastest, then
/// z, y and x, which the field reads without a copy; it must outlive the field.
class Field {
public:
    Field(const float* values, const GridShape& shape, Index n_channels,
          const InverseAffine& inverse_affine);

    /// Interpolates the image trilinearly at a world position into the first n_channels entries
    /// of values.
    ///
    /// The 8 surrounding voxels are clamped to the image (interpolation_axis). Entries of values
    /// past n_channels are left as they are. values must hold at least n_channels entries.
    void trilinear(const Vector3& position, std::span<double> values) const;

    /// Returns the number of values per voxel.
    [[nodiscard]] Index n_channels() const { return channels; }

    /// Returns the image's grid.
    [[nodiscard]] const Grid& grid() const { return field_grid; }

private:
    const float* data;
    Index channels;
    Grid field_grid;
};

/// The tracking mask: a float32 image on its own grid, read at the nearest voxel.
///
/// The values are an (X, Y, Z) array, stored with z varying fastest, which the mask reads without
/// a copy; it must outlive the mask.
class Mask {
public:
    Mask(const float* values, const GridShape& shape, const InverseAffine& inverse_affine);

    /// Returns whether a world position is inside the mask.
    ///
    /// It is when the value of the position's nearest voxel is at least MASK_THRESHOLD, so a
    /// voxel of the mask is inside over all of its cube. Voxel coordinates are rounded with
    /// halves away from zero; a coordinate at most -0.5 or at least the axis size minus 0.5, or
    /// NaN, is outside.
    [[nodiscard]] bool inside(const Vector3& position) const;

private:
    const float* data;
    Grid mask_grid;
};

} // namespace larmor::tracking
