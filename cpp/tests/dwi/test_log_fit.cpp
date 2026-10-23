#include <cmath>

#include <doctest/doctest.h>

#include "common/types.hpp"
#include "dwi/log_fit.hpp"

namespace {

using larmor::Index;
using larmor::Matrix;
using larmor::Vector;
using larmor::dwi::Flags;
using larmor::dwi::LogFitSettings;

constexpr Index N_DIRECTIONS = 30;
constexpr Index N_B0 = 2;

// The log-signal design of a diffusion tensor: columns 1 and -b g_i g_j (xx, xy, xz, yy, yz,
// zz, off-diagonals doubled), on N_B0 b0s and N_DIRECTIONS directions at b = 1000 and 2000.
Matrix tensor_design() {
    Matrix design = Matrix::Zero(N_B0 + 2 * N_DIRECTIONS, 7);
    for (Index row = 0; row < design.rows(); ++row) {
        design(row, 0) = 1.0;
    }
    for (Index direction = 0; direction < N_DIRECTIONS; ++direction) {
        // A spiral over the sphere.
        const double z = 1.0 - (2.0 * static_cast<double>(direction) + 1.0) / N_DIRECTIONS;
        const double radius = std::sqrt(1.0 - z * z);
        const double azimuth = 2.399963 * static_cast<double>(direction);
        const double x = radius * std::cos(azimuth);
        const double y = radius * std::sin(azimuth);
        for (Index shell = 0; shell < 2; ++shell) {
            const double b_value = 1000.0 * static_cast<double>(shell + 1);
            const Index row = N_B0 + shell * N_DIRECTIONS + direction;
            design.row(row).tail(6) << x * x, 2 * x * y, 2 * x * z, y * y, 2 * y * z, z * z;
            design.row(row).tail(6) *= -b_value;
        }
    }
    return design;
}

// log S0 and xx, xy, xz, yy, yz, zz of a prolate tensor (mm^2/s).
Vector tensor_parameters() {
    return Vector{{std::log(1000.0), 1.2e-3, 0.3e-3, -0.1e-3, 0.6e-3, 0.05e-3, 0.4e-3}};
}

struct BlockFit {
    Matrix parameters;
    Flags constrained;
};

BlockFit fit(const Matrix& design, const Matrix& constraints, const Matrix& log_signals,
             bool weighted) {
    const auto prepared = larmor::dwi::make_log_signal_design(design, constraints);
    auto scratch = larmor::dwi::make_log_fit_scratch(prepared);
    const LogFitSettings settings{
        .weighted = weighted, .max_reweightings = 20, .log_signal_tolerance = 1e-6};
    BlockFit result{.parameters = Matrix(log_signals.rows(), design.cols()),
                    .constrained = Flags(log_signals.rows())};
    larmor::dwi::fit_log_signal_block(prepared, settings, log_signals, scratch, result.parameters,
                                      result.constrained);
    return result;
}

void check_close(const Vector& found, const Vector& expected, double relative_tolerance) {
    REQUIRE(found.size() == expected.size());
    for (Index entry = 0; entry < found.size(); ++entry) {
        CHECK(std::fabs(found(entry) - expected(entry)) <=
              relative_tolerance * expected.cwiseAbs().maxCoeff());
    }
}

} // namespace

TEST_CASE("log fit: a noise-free signal gives its parameters, weighted or not") {
    const Matrix design = tensor_design();
    const Matrix log_signals = (design * tensor_parameters()).transpose();
    for (const bool weighted : {false, true}) {
        const BlockFit result = fit(design, Matrix(0, 7), log_signals, weighted);
        check_close(result.parameters.row(0).transpose(), tensor_parameters(), 1e-12);
        CHECK_FALSE(result.constrained(0));
    }
}

// One constraint, xx <= 1e-3, i.e. 1e-3 log S0 / log(1000) - xx >= 0 on voxels whose log S0 is
// log(1000).
Matrix xx_bound() {
    Matrix constraints = Matrix::Zero(1, 7);
    constraints(0, 0) = 1e-3 / std::log(1000.0);
    constraints(0, 1) = -1.0;
    return constraints;
}

TEST_CASE("log fit: a weighted fit whose weights underflow keeps the ordinary fit") {
    // Predicted diffusion-weighted signals of exp(-420) and less give weights S^2 that underflow
    // to 0 against the b0s: the weighted normal equations are singular, and so is the weighted
    // design of the constrained fit, which this voxel's xx of 0.42 needs.
    const Matrix design = tensor_design();
    const Vector broken{{std::log(1000.0), 0.42, 0.0, 0.0, 0.42, 0.0, 0.42}};
    const Matrix log_signals = (design * broken).transpose();
    const BlockFit ordinary = fit(design, Matrix(0, 7), log_signals, false);
    for (const Matrix& constraints : {Matrix(0, 7), xx_bound()}) {
        const BlockFit weighted = fit(design, constraints, log_signals, true);
        CHECK(weighted.parameters.allFinite());
        CHECK(weighted.parameters == ordinary.parameters);
    }
}

// The minimiser of sum_i w_i (x_i^T p - y_i)^2 subject to c^T p = 0, from the KKT system
// [X^T W X, c; c^T, 0] [p; multiplier] = [X^T W y; 0], with the columns of X scaled to unit norm
// so that the system is well conditioned.
Vector equality_constrained_fit(const Matrix& design, const Vector& log_signal,
                                const Vector& weights, const Vector& constraint) {
    const Index n_parameters = design.cols();
    const Vector scales = design.colwise().norm().transpose();
    const Matrix scaled = design * scales.cwiseInverse().asDiagonal();
    Matrix system = Matrix::Zero(n_parameters + 1, n_parameters + 1);
    system.topLeftCorner(n_parameters, n_parameters) =
        scaled.transpose() * weights.asDiagonal() * scaled;
    system.topRightCorner(n_parameters, 1) = constraint.cwiseQuotient(scales);
    system.bottomLeftCorner(1, n_parameters) = constraint.cwiseQuotient(scales).transpose();
    Vector right_hand_side = Vector::Zero(n_parameters + 1);
    right_hand_side.head(n_parameters) = scaled.transpose() * weights.cwiseProduct(log_signal);
    const Vector solution = system.fullPivLu().solve(right_hand_side);
    return solution.head(n_parameters).cwiseQuotient(scales);
}

TEST_CASE("log fit: only a voxel that breaks a constraint is fitted again, onto it") {
    // xx <= 1e-3: the first voxel meets it, the second (xx = 1.2e-3) does not.
    const Matrix design = tensor_design();
    const Matrix constraints = xx_bound();
    Vector inside = tensor_parameters();
    inside(1) = 0.8e-3;
    Matrix log_signals(2, design.rows());
    log_signals.row(0) = (design * inside).transpose();
    log_signals.row(1) = (design * tensor_parameters()).transpose();
    // Noise, so that the constrained optimum is not on the line through the truth.
    for (Index column = 0; column < design.rows(); ++column) {
        log_signals(1, column) += 0.01 * std::sin(3.7 * static_cast<double>(column));
    }

    for (const bool weighted : {false, true}) {
        CAPTURE(weighted);
        const BlockFit constrained = fit(design, constraints, log_signals, weighted);
        const BlockFit plain = fit(design, Matrix(0, 7), log_signals, weighted);
        CHECK_FALSE(constrained.constrained(0));
        CHECK(constrained.parameters.row(0) == plain.parameters.row(0));
        CHECK(constrained.constrained(1));
        const double value = (constraints * constrained.parameters.row(1).transpose())(0);
        CHECK(std::fabs(value) < 1e-12);
        CHECK(constrained.parameters(1, 1) < plain.parameters(1, 1));

        // With its one constraint active, the solution is the equality-constrained minimiser
        // of the same sum of squares, weighted by the squared signal of the plain fit.
        Vector weights = Vector::Ones(design.rows());
        if (weighted) {
            const Vector predicted = design * plain.parameters.row(1).transpose();
            weights = (2.0 * (predicted.array() - predicted.maxCoeff())).exp().matrix();
        }
        const Vector expected = equality_constrained_fit(design, log_signals.row(1).transpose(),
                                                         weights, constraints.row(0).transpose());
        check_close(constrained.parameters.row(1).transpose(), expected, 1e-9);
    }
}
