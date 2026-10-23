#include <cmath>
#include <cstdint>
#include <stdexcept>

#include <Eigen/QR>
#include <doctest/doctest.h>

#include "common/types.hpp"
#include "sd/lsi.hpp"

namespace {

using larmor::Index;
using larmor::Matrix;
using larmor::Vector;

// A full-rank (6, 3) matrix.
Matrix tall() {
    Matrix design(6, 3);
    for (Index row = 0; row < 6; ++row) {
        for (Index column = 0; column < 3; ++column) {
            design(row, column) = 1.0 + static_cast<double>((row * 5 + column * 3) % 7) / 7.0 +
                                  (row == column ? 1.0 : 0.0);
        }
    }
    return design;
}

} // namespace

TEST_CASE("lsi: a feasible least-squares solution is returned unchanged") {
    const Matrix design = tall();
    const Vector truth{{0.5, 1.5, 2.0}};
    const Matrix constraints = Matrix::Identity(3, 3);
    const auto prepared = larmor::sd::make_lsi_design(design, constraints, 0.0);
    const larmor::sd::LsiResult found = larmor::sd::solve_lsi(prepared, design * truth);
    CHECK(found.converged);
    CHECK((found.solution - truth).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("lsi: identity constraints give the NNLS solution") {
    // min ||x - s|| with x >= 0 clips the negative entries.
    const auto prepared =
        larmor::sd::make_lsi_design(Matrix::Identity(3, 3), Matrix::Identity(3, 3), 0.0);
    const Vector found = larmor::sd::solve_lsi(prepared, Vector{{2.0, -1.0, 3.0}}).solution;
    CHECK((found - Vector{{2.0, 0.0, 3.0}}).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("lsi: a general constraint holds with equality at the solution") {
    // min ||x - (0.5, 0.5)|| subject to x0 + x1 <= 0: the projection onto that half-plane is
    // the origin.
    Matrix constraints(1, 2);
    constraints << -1.0, -1.0;
    const auto prepared = larmor::sd::make_lsi_design(Matrix::Identity(2, 2), constraints, 0.0);
    const Vector found = larmor::sd::solve_lsi(prepared, Vector{{0.5, 0.5}}).solution;
    CHECK(std::fabs(found(0)) < 1e-12);
    CHECK(std::fabs(found(1)) < 1e-12);
}

TEST_CASE("lsi: the QR fallback gives the Cholesky path's solution, duplicate rows or not") {
    // Ten constraints on four parameters, several active at the solution, one of them twice.
    Matrix design(12, 4);
    Matrix constraints(10, 4);
    Vector data(12);
    std::uint64_t state = 11;
    const auto next = [&state] {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(state >> 11U) * 0x1.0p-53 - 0.5;
    };
    for (Index row = 0; row < 12; ++row) {
        for (Index column = 0; column < 4; ++column) {
            design(row, column) = next() + (row == column ? 2.0 : 0.0);
        }
        data(row) = next();
    }
    for (Index row = 0; row < 10; ++row) {
        for (Index column = 0; column < 4; ++column) {
            constraints(row, column) = next();
        }
    }
    Matrix duplicated(11, 4);
    duplicated.topRows(10) = constraints;
    duplicated.row(10) = constraints.row(0);

    const auto prepared = larmor::sd::make_lsi_design(design, constraints, 0.0);
    const auto fast = larmor::sd::solve_lsi(prepared, data);
    const auto by_qr = larmor::sd::solve_lsi_by_qr(prepared, data);
    const auto with_duplicate =
        larmor::sd::solve_lsi_by_qr(larmor::sd::make_lsi_design(design, duplicated, 0.0), data);
    CHECK(fast.converged);
    CHECK(by_qr.converged);
    CHECK(with_duplicate.converged);
    CHECK((constraints * fast.solution).minCoeff() > -1e-12);
    // Some constraints are active, so the constrained fit is not the plain least squares.
    const Vector unconstrained = design.colPivHouseholderQr().solve(data);
    CHECK((constraints * unconstrained).minCoeff() < -1e-3);
    CHECK((fast.solution - by_qr.solution).cwiseAbs().maxCoeff() < 1e-12);
    CHECK((fast.solution - with_duplicate.solution).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("lsi: a rank-deficient design is refused unless a ridge is given") {
    Matrix design(3, 2);
    design << 1.0, 2.0, 2.0, 4.0, 3.0, 6.0;
    const Matrix constraints = Matrix::Identity(2, 2);
    CHECK_THROWS_AS(larmor::sd::make_lsi_design(design, constraints, 0.0), std::invalid_argument);
    CHECK_NOTHROW(larmor::sd::make_lsi_design(design, constraints, 1e-3));
}
