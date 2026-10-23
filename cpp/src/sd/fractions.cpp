#include "sd/fractions.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "sd/nnls.hpp"

namespace larmor::sd {

namespace {

// FOD values this close below the median, relative to it, count as equal to it and survive
// the median threshold.
constexpr double MEDIAN_TIE_TOLERANCE = 1e-9;

// Orders numbers by value with NaN after everything, so that sorting is well defined.
bool less_nan_last(double left, double right) {
    return std::isnan(right) ? !std::isnan(left) : left < right;
}

} // namespace

Index n_groups(Grouping grouping, const TissueKernel& kernel) {
    return grouping == Grouping::Lumped ? 1 : static_cast<Index>(kernel.anisotropic_blocks.size());
}

double median(const VectorView& values) {
    std::vector<double> sorted(values.begin(), values.end());
    if (sorted.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const auto middle = sorted.begin() + static_cast<std::ptrdiff_t>(sorted.size() / 2);
    std::nth_element(sorted.begin(), middle, sorted.end(), less_nan_last);
    if (sorted.size() % 2 == 1) {
        return *middle;
    }
    const double lower = *std::max_element(sorted.begin(), middle, less_nan_last);
    return (lower + *middle) / 2.0;
}

Vector compute_fractions(const VectorView& fod, const TissueKernel& kernel,
                         const VectorView& signal, Grouping grouping,
                         std::optional<double> gradient_tolerance) {
    const double threshold = median(fod) * (1.0 - MEDIAN_TIE_TOLERANCE);
    const Vector thresholded = (fod.array() < threshold).select(0.0, fod);

    const ColumnRange all_columns{.begin = 0, .end = kernel.n_anisotropic()};
    const std::span<const ColumnRange> groups =
        grouping == Grouping::Lumped ? std::span<const ColumnRange>(&all_columns, 1)
                                     : std::span<const ColumnRange>(kernel.anisotropic_blocks);
    const auto n_group_columns = static_cast<Index>(groups.size());

    // The columns of the fit: one per group, then the isotropic tissues.
    Matrix columns(kernel.anisotropic.rows(), n_group_columns + kernel.n_isotropic());
    for (Index group = 0; group < n_group_columns; ++group) {
        const ColumnRange& range = groups[static_cast<std::size_t>(group)];
        Vector group_signal = kernel.anisotropic.middleCols(range.begin, range.size()) *
                              thresholded.segment(range.begin, range.size());
        if (group_signal.size() > 0) {
            const double largest = group_signal.maxCoeff<Eigen::PropagateNumbers>();
            if (largest > 0.0) {
                group_signal /= largest;
            }
        }
        columns.col(group) = group_signal;
    }
    columns.rightCols(kernel.n_isotropic()) = kernel.isotropic;

    const double tolerance =
        gradient_tolerance ? *gradient_tolerance : default_gradient_tolerance(columns);
    return solve_nnls(columns, signal, tolerance);
}

} // namespace larmor::sd
