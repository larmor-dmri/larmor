#include "sd/elastic_net.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace larmor::sd {

namespace {

// Returns value moved towards 0 by threshold, or 0 if it is within threshold of 0.
double soft_threshold(double value, double threshold) {
    if (value > threshold) {
        return value - threshold;
    }
    if (value < -threshold) {
        return value + threshold;
    }
    return 0.0;
}

} // namespace

StandardizedDesign standardize_columns(const MatrixView& design) {
    const auto n_rows = static_cast<double>(design.rows());
    Vector scales = (design.colwise().squaredNorm() / n_rows).cwiseSqrt().transpose();
    Eigen::Array<bool, Eigen::Dynamic, 1> nonzero = scales.array() > 0.0;
    for (Index column = 0; column < scales.size(); ++column) {
        if (!nonzero(column)) {
            scales(column) = 1.0;
        }
    }
    Matrix scaled = design;
    scaled.array().rowwise() /= scales.transpose().array();
    return {.columns = std::move(scaled), .scales = std::move(scales), .nonzero = nonzero};
}

namespace {

// The state of one coordinate descent: the coefficients on the standardised columns and the
// residual they leave.
class CoordinateDescent {
public:
    CoordinateDescent(const StandardizedDesign& standardized, const VectorView& data,
                      const ElasticNetSettings& settings)
        // Starts from zero coefficients, so the residual is the data.
        : design(standardized), row_count(static_cast<double>(standardized.columns.rows())),
          l1_threshold(settings.penalty * settings.l1_ratio),
          // Scaled columns have (1 / m) ||x_j||^2 = 1, so each update divides by this.
          shrink(1.0 + settings.penalty * (1.0 - settings.l1_ratio)), residual(data),
          coefficients(Vector::Zero(standardized.columns.cols())) {}

    // Returns whether some coefficient can be non-zero, which is when the largest correlation of
    // a column with the data exceeds the L1 threshold.
    [[nodiscard]] bool has_nonzero_solution() const {
        const Vector correlations = design.columns.transpose() * residual / row_count;
        return (correlations.array().abs() * design.nonzero.cast<double>()).maxCoeff() >
               l1_threshold;
    }

    // Updates every non-zero column once; returns the largest relative change.
    double sweep_all() {
        double largest_change = 0.0;
        for (Index column = 0; column < coefficients.size(); ++column) {
            if (design.nonzero(column)) {
                largest_change = std::max(largest_change, update(column));
            }
        }
        return largest_change;
    }

    // Updates the given columns once; returns the largest relative change.
    double sweep(const std::vector<Index>& columns) {
        double largest_change = 0.0;
        for (const Index column : columns) {
            largest_change = std::max(largest_change, update(column));
        }
        return largest_change;
    }

    // Returns the columns whose coefficient is non-zero.
    [[nodiscard]] std::vector<Index> nonzero_columns() const {
        std::vector<Index> nonzero;
        for (Index column = 0; column < coefficients.size(); ++column) {
            if (coefficients(column) != 0.0) {
                nonzero.push_back(column);
            }
        }
        return nonzero;
    }

    // Returns the coefficients of the original, unscaled columns.
    [[nodiscard]] Vector unscaled_coefficients() const {
        return coefficients.cwiseQuotient(design.scales);
    }

private:
    // Moves one coefficient to its minimiser with the others fixed; returns its change relative
    // to 1 + its old size.
    double update(Index column) {
        const double old_value = coefficients(column);
        const double correlation = design.columns.col(column).dot(residual) / row_count + old_value;
        const double new_value = soft_threshold(correlation, l1_threshold) / shrink;
        if (new_value == old_value) {
            return 0.0;
        }
        residual -= (new_value - old_value) * design.columns.col(column);
        coefficients(column) = new_value;
        return std::abs(new_value - old_value) / (1.0 + std::abs(old_value));
    }

    const StandardizedDesign& design;
    double row_count;
    double l1_threshold;
    double shrink;
    Vector residual;
    Vector coefficients;
};

} // namespace

ElasticNetResult solve_elastic_net(const StandardizedDesign& design, const VectorView& data,
                                   const ElasticNetSettings& settings) {
    CoordinateDescent descent(design, data, settings);
    // At or above the penalty where no column correlates enough, every coefficient is zero.
    if (!descent.has_nonzero_solution()) {
        return {.coefficients = Vector::Zero(design.columns.cols()), .converged = true};
    }

    // A sweep over every column finds the coefficients that can be non-zero; sweeps over those
    // alone then converge them, and another full sweep checks that no other column enters.
    // Only the full sweep that changes nothing beyond the tolerance ends the descent.
    Index sweeps = 0;
    bool converged = false;
    while (sweeps < settings.max_sweeps) {
        ++sweeps;
        if (descent.sweep_all() < settings.relative_tolerance) {
            converged = true;
            break;
        }
        const std::vector<Index> nonzero = descent.nonzero_columns();
        while (sweeps < settings.max_sweeps) {
            ++sweeps;
            if (descent.sweep(nonzero) < settings.relative_tolerance) {
                break;
            }
        }
    }
    return {.coefficients = descent.unscaled_coefficients(), .converged = converged};
}

} // namespace larmor::sd
