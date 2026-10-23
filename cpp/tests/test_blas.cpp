#include <cmath>

#include <doctest/doctest.h>

#include "common/types.hpp"

// A product large enough that Eigen sends it to the BLAS library when the build uses one: checks
// that products give the right numbers with whichever BLAS, or none, the build uses.
TEST_CASE("a large matrix product matches the sum it stands for") {
    const larmor::Index n_rows = 200;
    const larmor::Index n_inner = 150;
    const larmor::Index n_columns = 120;
    larmor::Matrix left(n_rows, n_inner);
    larmor::Matrix right(n_inner, n_columns);
    for (larmor::Index row = 0; row < n_rows; ++row) {
        for (larmor::Index inner = 0; inner < n_inner; ++inner) {
            left(row, inner) = std::sin(static_cast<double>(row + 2 * inner));
        }
    }
    for (larmor::Index inner = 0; inner < n_inner; ++inner) {
        for (larmor::Index column = 0; column < n_columns; ++column) {
            right(inner, column) = std::cos(static_cast<double>(3 * inner - column));
        }
    }

    const larmor::Matrix product = left * right;

    double largest_error = 0.0;
    for (larmor::Index row = 0; row < n_rows; ++row) {
        for (larmor::Index column = 0; column < n_columns; ++column) {
            double sum = 0.0;
            for (larmor::Index inner = 0; inner < n_inner; ++inner) {
                sum += left(row, inner) * right(inner, column);
            }
            largest_error = std::fmax(largest_error, std::abs(product(row, column) - sum));
        }
    }
    // Only the order of the sums differs: a few eps of the terms' size times n_inner.
    CHECK(largest_error < 1e-12);
}
