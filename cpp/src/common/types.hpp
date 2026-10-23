// The array and index types every numerical module uses.
#pragma once

#include <Eigen/Core>

namespace larmor {

/// A size or an index into an array, signed as in Eigen.
using Index = Eigen::Index;

// Matrices are stored row by row, like numpy arrays, so a numpy array maps onto one without a
// copy.

/// A matrix of doubles, stored row by row.
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// A Matrix or a block of one, read only and without a copy.
using MatrixView = Eigen::Ref<const Matrix>;

/// A column vector of doubles.
using Vector = Eigen::VectorXd;

/// A Vector or a segment of one, read only and without a copy.
using VectorView = Eigen::Ref<const Vector>;

/// A point or direction in world coordinates (mm).
using Vector3 = Eigen::Vector3d;

} // namespace larmor
