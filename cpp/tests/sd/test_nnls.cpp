#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <doctest/doctest.h>

#include "common/types.hpp"
#include "sd/nnls.hpp"

namespace {

using larmor::Index;
using larmor::Matrix;
using larmor::Vector;

// A full-rank (6, 3) matrix with positive entries.
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

void check_close(const Vector& found, const Vector& expected, double tolerance) {
    REQUIRE(found.size() == expected.size());
    for (Index entry = 0; entry < found.size(); ++entry) {
        CHECK(std::fabs(found(entry) - expected(entry)) <= tolerance);
    }
}

Vector solve(const Matrix& design, const Vector& data) {
    return larmor::sd::solve_nnls(design, data, larmor::sd::default_gradient_tolerance(design));
}

// Uniform numbers in [0, 1) from a fixed linear congruential sequence, the same on every
// platform.
struct PseudoRandom {
    std::uint64_t state = 7;

    double next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(state >> 11U) * 0x1.0p-53;
    }
};

} // namespace

TEST_CASE("nnls: the identity clips the negative entries") {
    const Matrix design = Matrix::Identity(3, 3);
    const Vector data{{2.0, -1.0, 3.0}};
    check_close(solve(design, data), Vector{{2.0, 0.0, 3.0}}, 1e-15);
}

TEST_CASE("nnls: a positive least-squares solution is returned unchanged") {
    const Matrix design = tall();
    const Vector truth{{0.5, 1.5, 2.0}};
    check_close(solve(design, design * truth), truth, 1e-12);
}

TEST_CASE("nnls: a coefficient the data push negative is held at zero") {
    // The data are the first column minus a little of the third: the best non-negative fit
    // uses the first column only, and the gradient along the third is negative.
    const Matrix design = tall();
    const Vector data = design.col(0) - 0.3 * design.col(2);
    const Vector solution = solve(design, data);
    CHECK(solution(2) == 0.0);
    CHECK((solution.array() >= 0.0).all());
    // Optimality (KKT): the gradient C^T (d - C x) is zero on the free columns and <= 0 on the
    // ones held at zero.
    const Vector gradients = design.transpose() * (data - design * solution);
    for (Index column = 0; column < 3; ++column) {
        if (solution(column) > 0.0) {
            CHECK(std::fabs(gradients(column)) < 1e-12);
        } else {
            CHECK(gradients(column) <= 1e-12);
        }
    }
}

TEST_CASE("nnls: a tolerance above every gradient gives zero") {
    const Matrix design = tall();
    const Vector data = design * Vector::Ones(3);
    CHECK(larmor::sd::solve_nnls(design, data, 1e6) == Vector::Zero(3));
}

TEST_CASE("nnls: a column of zeros never enters") {
    Matrix design = tall();
    design.col(1).setZero();
    const Vector solution = solve(design, design * Vector{{1.0, 5.0, 2.0}});
    CHECK(solution(1) == 0.0);
    CHECK(std::fabs(solution(0) - 1.0) <= 1e-12);
    CHECK(std::fabs(solution(2) - 2.0) <= 1e-12);
}

TEST_CASE("nnls: any number of columns") {
    // More columns than the Python interface allows: a wide positive combination of 20
    // columns of a (60, 20) matrix is recovered.
    Matrix design(60, 20);
    for (Index row = 0; row < 60; ++row) {
        for (Index column = 0; column < 20; ++column) {
            design(row, column) = std::exp(-0.05 * static_cast<double>(row) *
                                           (1.0 + static_cast<double>(column) / 4.0)) +
                                  (row == column ? 1.0 : 0.0);
        }
    }
    const Vector truth = Vector::LinSpaced(20, 0.5, 2.0);
    check_close(solve(design, design * truth), truth, 1e-10);
}

TEST_CASE("nnls: dependent columns give a bounded solution that fits as well") {
    // A (6, 4) matrix whose last column repeats the first exactly, or up to noise of 1e-12.
    const Matrix distinct = tall();
    for (const double noise : {0.0, 1e-12}) {
        CAPTURE(noise);
        Matrix design(6, 4);
        design.leftCols(3) = distinct;
        for (Index row = 0; row < 6; ++row) {
            design(row, 3) = distinct(row, 0) + noise * static_cast<double>(row % 2 == 0 ? 1 : -1);
        }
        // Data the distinct columns cannot fit exactly, so the residual is not zero.
        const Vector data{{3.0, 1.0, 4.0, 1.0, 5.0, 9.0}};
        const Vector solution = solve(design, data);
        const Vector reference = solve(distinct, data);

        CHECK(solution.allFinite());
        CHECK((solution.array() >= 0.0).all());
        CHECK(solution.cwiseAbs().maxCoeff() <= 10.0 * reference.cwiseAbs().maxCoeff());
        // KKT: the gradient is about zero on the free columns and <= 0 on those held at zero.
        const Vector gradients = design.transpose() * (data - design * solution);
        for (Index column = 0; column < 4; ++column) {
            if (solution(column) > 0.0) {
                CHECK(std::fabs(gradients(column)) < 1e-10);
            } else {
                CHECK(gradients(column) <= 1e-10);
            }
        }
        // The fit is as good as the full-rank fit on the distinct columns.
        CHECK((design * solution - data).norm() <=
              (distinct * reference - data).norm() * (1.0 + 1e-10));
    }
}

TEST_CASE("nnls: an exactly duplicated column with a zero tolerance") {
    // With a zero tolerance the duplicate can join the active set; it must get a zero
    // coefficient instead of a huge one cancelling its twin. Many problems, as only some
    // orders of rounding expose it.
    PseudoRandom uniform;
    for (int problem = 0; problem < 200; ++problem) {
        CAPTURE(problem);
        Matrix distinct(30, 4);
        for (Index row = 0; row < 30; ++row) {
            for (Index column = 0; column < 4; ++column) {
                distinct(row, column) = uniform.next();
            }
        }
        Matrix design(30, 5);
        design.leftCols(4) = distinct;
        design.col(4) = distinct.col(1);
        Vector data = design * Vector{{0.5, 1.0, 0.0, 0.3, 1.0}};
        for (Index row = 0; row < 30; ++row) {
            data(row) += 0.05 * (uniform.next() - 0.5);
        }

        const Vector solution = larmor::sd::solve_nnls(design, data, 0.0);
        const Vector reference = larmor::sd::solve_nnls(distinct, data, 0.0);
        CHECK(solution.allFinite());
        CHECK((solution.array() >= 0.0).all());
        CHECK((design * solution - data).norm() <=
              (distinct * reference - data).norm() * (1.0 + 1e-8));
    }
}

TEST_CASE("nnls: the default tolerance scales with the largest column sum") {
    const Matrix design{{1.0, -4.0}, {2.0, 1.0}, {-3.0, 0.5}};
    const double expected = 10.0 * std::numeric_limits<double>::epsilon() * 6.0 * 3.0;
    CHECK(larmor::sd::default_gradient_tolerance(design) == expected);
}

TEST_CASE("nnls: the step limit is reported, and the default one gives solve_nnls") {
    // Correlated columns of both signs and data from coefficients of both signs, so that some
    // problems need inner steps.
    PseudoRandom uniform;
    int n_stopped = 0;
    for (int problem = 0; problem < 50; ++problem) {
        CAPTURE(problem);
        Matrix design(12, 8);
        for (Index row = 0; row < 12; ++row) {
            for (Index column = 0; column < 8; ++column) {
                design(row, column) = uniform.next() - 0.5;
            }
        }
        Vector coefficients(8);
        for (Index column = 0; column < 8; ++column) {
            coefficients(column) = uniform.next() - 0.4;
        }
        const Vector data = design * coefficients;
        const double tolerance = larmor::sd::default_gradient_tolerance(design);

        const auto full = larmor::sd::solve_nnls_with_limit(design, data, tolerance, 24);
        CHECK(full.converged);
        CHECK(full.solution == larmor::sd::solve_nnls(design, data, tolerance));
        const auto stopped = larmor::sd::solve_nnls_with_limit(design, data, tolerance, 0);
        if (!stopped.converged) {
            ++n_stopped;
        } else {
            CHECK(stopped.solution == full.solution);
        }
    }
    CHECK(n_stopped > 0);
}

TEST_CASE("nnls: the ridge solve matches plain NNLS on the stacked matrix") {
    // A wide (10, 40) kernel and signals partly outside its cone: more columns enter than there
    // are rows and some leave again. A ridge of 0.1 uses the normal equations, one of 1e-7 the QR.
    PseudoRandom random;
    Matrix kernel(10, 40);
    for (Index row = 0; row < 10; ++row) {
        for (Index column = 0; column < 40; ++column) {
            kernel(row, column) = random.next();
        }
    }
    for (const double ridge : {0.1, 1e-7}) {
        for (const double l1 : {0.0, 0.01}) {
            const larmor::sd::RidgeDesign design = larmor::sd::make_ridge_design(kernel, ridge);
            CHECK(design.normal_equations == (ridge == 0.1));
            Matrix stacked = Matrix::Zero(50, 40);
            stacked.topRows(10) = kernel;
            stacked.bottomRows(40).diagonal().setConstant(ridge);
            for (int voxel = 0; voxel < 5; ++voxel) {
                Vector signal(10);
                for (Index row = 0; row < 10; ++row) {
                    signal(row) = 2.0 * random.next() - 0.5;
                }
                Vector data = Vector::Zero(50);
                data.head(10) = signal;
                data.tail(40).setConstant(-l1 / ridge);
                check_close(larmor::sd::solve_ridge_nnls(design, signal, l1, 1e-12),
                            larmor::sd::solve_nnls(stacked, data, 1e-12), 1e-9);
            }
        }
    }
}

TEST_CASE("nnls: with no ridge, the normal equations give plain NNLS") {
    // A tall (30, 8) kernel, whose Gram matrix is well conditioned.
    PseudoRandom random;
    Matrix kernel(30, 8);
    for (Index row = 0; row < 30; ++row) {
        for (Index column = 0; column < 8; ++column) {
            kernel(row, column) = random.next();
        }
    }
    const larmor::sd::RidgeDesign design = larmor::sd::make_ridge_design(kernel, 0.0);
    CHECK(design.normal_equations);
    for (int voxel = 0; voxel < 5; ++voxel) {
        Vector signal(30);
        for (Index row = 0; row < 30; ++row) {
            signal(row) = 2.0 * random.next() - 0.5;
        }
        check_close(larmor::sd::solve_ridge_nnls(design, signal, 0.0, 1e-12),
                    larmor::sd::solve_nnls(kernel, signal, 1e-12), 1e-9);
    }
}

TEST_CASE("nnls: removing a column keeps the Cholesky factor of the others") {
    PseudoRandom random;
    Matrix columns(8, 5);
    for (Index row = 0; row < 8; ++row) {
        for (Index column = 0; column < 5; ++column) {
            columns(row, column) = random.next();
        }
    }
    for (Index position = 0; position < 5; ++position) {
        Matrix factor = Matrix(columns.transpose() * columns).llt().matrixL();
        larmor::sd::remove_cholesky_column(factor, 5, position);
        Matrix kept(8, 4);
        for (Index column = 0, out = 0; column < 5; ++column) {
            if (column != position) {
                kept.col(out++) = columns.col(column);
            }
        }
        const Matrix lower = factor.topLeftCorner(4, 4).triangularView<Eigen::Lower>();
        CHECK((lower * lower.transpose() - kept.transpose() * kept).cwiseAbs().maxCoeff() <= 1e-12);
        CHECK((lower.diagonal().array() > 0.0).all());
    }
}

TEST_CASE("nnls: exception thrown if l1 > 0 and ridge == 0") {
    PseudoRandom random;
    Matrix kernel(10, 40);
    for (Index row = 0; row < 10; ++row) {
        for (Index column = 0; column < 40; ++column) {
            kernel(row, column) = random.next();
        }
    }
    Vector signal(10);
    for (Index row = 0; row < 10; ++row) {
        signal(row) = 2.0 * random.next() - 0.5;
    }
    CHECK_THROWS_AS(larmor::sd::solve_ridge_nnls(larmor::sd::make_ridge_design(kernel, 0.0), signal,
                                                 0.5, 1e-12),
                    std::invalid_argument);
}
