// Numpy arrays passed between Python and the numerical library, and the checks the bindings make
// on them.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include <Eigen/Core>
#include <nanobind/ndarray.h>

#include "common/types.hpp"

namespace larmor::python {

namespace nb = nanobind;

/// A numpy array passed in from Python that C++ only reads, such as the b-vectors.
///
/// Bind it with .noconvert(), so that an array of another dtype is a TypeError instead of a silent
/// copy, and check it with require_contiguous.
template <typename Scalar, std::size_t n_dims>
using InArray = nb::ndarray<const Scalar, nb::ndim<n_dims>, nb::device::cpu>;

/// A numpy array passed in from Python that C++ writes to, such as a buffer the caller reuses.
///
/// A read-only array is refused.
template <typename Scalar, std::size_t n_dims>
using InOutArray = nb::ndarray<Scalar, nb::ndim<n_dims>, nb::device::cpu>;

/// A numpy array returned to Python.
template <typename Scalar, std::size_t n_dims>
using NumpyArray = nb::ndarray<nb::numpy, Scalar, nb::ndim<n_dims>>;

/// A matrix of any element type, stored row by row.
template <typename Scalar>
using RowMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// A column vector of any element type.
template <typename Scalar> using ColumnVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

/// Raises ValueError("n_threads must be at least 1") for a thread count below 1.
inline void require_threads(int n_threads) {
    if (n_threads < 1) {
        throw nb::value_error("n_threads must be at least 1");
    }
}

/// Returns the length of `axis` of `array`.
template <typename Array> Index extent(const Array& array, Index axis) {
    return static_cast<Index>(array.shape(static_cast<std::size_t>(axis)));
}

/// Returns the number of elements of `array`.
template <typename Array> Index n_elements(const Array& array) {
    return static_cast<Index>(array.size());
}

/// Returns the shape of `array` as "[2, 3]", for error messages.
template <typename Array> std::string shape_string(const Array& array) {
    std::string text = "[";
    for (Index axis = 0; axis < static_cast<Index>(array.ndim()); ++axis) {
        if (axis > 0) {
            text += ", ";
        }
        text += std::to_string(extent(array, axis));
    }
    return text + "]";
}

/// Returns whether `array` is C-contiguous, by numpy's rule: its last axis varies fastest and no
/// axis skips elements.
template <typename Array> bool is_contiguous(const Array& array) {
    if (array.size() == 0) {
        return true;
    }
    std::int64_t expected_stride = 1;
    for (Index axis = static_cast<Index>(array.ndim()) - 1; axis >= 0; --axis) {
        const auto unsigned_axis = static_cast<std::size_t>(axis);
        const auto length = static_cast<std::int64_t>(array.shape(unsigned_axis));
        if (length != 1 && array.stride(unsigned_axis) != expected_stride) {
            return false;
        }
        expected_stride *= length;
    }
    return true;
}

/// Raises ValueError("<name> must be C-contiguous") unless array is.
template <typename Array>
void require_contiguous(const Array& array, const std::string& name = "the array") {
    if (!is_contiguous(array)) {
        throw nb::value_error((name + " must be C-contiguous").c_str());
    }
}

/// Raises ValueError("expected <n> columns, got <m>") unless the 2-D array has n_columns columns.
template <typename Array> void require_columns(const Array& array, Index n_columns) {
    const Index n_found = extent(array, 1);
    if (n_found != n_columns) {
        throw nb::value_error(
            ("expected " + std::to_string(n_columns) + " columns, got " + std::to_string(n_found))
                .c_str());
    }
}

/// Returns a C-contiguous 2D array as an Eigen matrix, without a copy.
///
/// Checks contiguity first. The view is read-only for an InArray and writable for an InOutArray.
template <typename Scalar, typename... ArrayOptions>
auto matrix_view(const nb::ndarray<Scalar, ArrayOptions...>& array) {
    static_assert(nb::ndarray<Scalar, ArrayOptions...>::Config::N == 2,
                  "matrix_view needs a 2-D array");
    require_contiguous(array);
    using Plain = RowMatrix<std::remove_const_t<Scalar>>;
    using Mapped = std::conditional_t<std::is_const_v<Scalar>, const Plain, Plain>;
    return Eigen::Map<Mapped>(array.data(), extent(array, 0), extent(array, 1));
}

/// Returns a C-contiguous array as a matrix with one row per item along the first axis, without a
/// copy.
///
/// A (batch, steps, 3) array becomes a (batch, steps * 3) matrix. Checks contiguity first.
template <typename Scalar, typename... ArrayOptions>
auto rows_view(const nb::ndarray<Scalar, ArrayOptions...>& array) {
    require_contiguous(array);
    using Plain = RowMatrix<std::remove_const_t<Scalar>>;
    using Mapped = std::conditional_t<std::is_const_v<Scalar>, const Plain, Plain>;
    const Index n_rows = array.ndim() == 0 ? 1 : extent(array, 0);
    const Index n_columns = n_rows == 0 ? 0 : n_elements(array) / n_rows;
    return Eigen::Map<Mapped>(array.data(), n_rows, n_columns);
}

/// Returns every element of a C-contiguous array as one column vector, without a copy.
///
/// Checks contiguity first.
template <typename Scalar, typename... ArrayOptions>
auto flat_view(const nb::ndarray<Scalar, ArrayOptions...>& array) {
    require_contiguous(array);
    using Plain = ColumnVector<std::remove_const_t<Scalar>>;
    using Mapped = std::conditional_t<std::is_const_v<Scalar>, const Plain, Plain>;
    return Eigen::Map<Mapped>(array.data(), n_elements(array));
}

/// Hands an Eigen vector or matrix to numpy as an array of the given shape, without a copy.
///
/// The numpy array takes ownership of the values. The shape must hold as many elements as
/// values. Call it only while holding the Python lock, not inside a gil_scoped_release.
template <std::size_t n_dims, typename Plain>
NumpyArray<typename Plain::Scalar, n_dims> to_numpy(Plain&& values,
                                                    const std::array<Index, n_dims>& shape) {
    static_assert(!std::is_lvalue_reference_v<Plain>, "to_numpy takes its values by move");
    std::array<std::size_t, n_dims> unsigned_shape{};
    Index n_shape = 1;
    for (std::size_t axis = 0; axis < n_dims; ++axis) {
        unsigned_shape.at(axis) = static_cast<std::size_t>(shape.at(axis));
        n_shape *= shape.at(axis);
    }
    if (n_shape != values.size()) {
        throw nb::value_error("to_numpy: the shape does not match the number of values");
    }
    auto owned = std::make_unique<Plain>(std::forward<Plain>(values));
    auto* data = owned->data();
    const nb::capsule owner(owned.get(), [](void* pointer) noexcept {
        std::unique_ptr<Plain>(static_cast<Plain*>(pointer)).reset();
    });
    // The capsule owns the values from here on.
    static_cast<void>(owned.release());
    return NumpyArray<typename Plain::Scalar, n_dims>(data, n_dims, unsigned_shape.data(), owner);
}

/// Hands an Eigen column vector to numpy as a 1D array, without a copy; see the general to_numpy.
template <typename Scalar> NumpyArray<Scalar, 1> to_numpy(ColumnVector<Scalar>&& values) {
    const std::array<Index, 1> shape{values.size()};
    return to_numpy<1>(std::move(values), shape);
}

/// Hands an Eigen matrix to numpy as a 2D array, without a copy; see the general to_numpy.
template <typename Scalar> NumpyArray<Scalar, 2> to_numpy(RowMatrix<Scalar>&& values) {
    const std::array<Index, 2> shape{values.rows(), values.cols()};
    return to_numpy<2>(std::move(values), shape);
}

} // namespace larmor::python
