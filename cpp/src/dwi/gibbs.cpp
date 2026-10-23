#include "dwi/gibbs.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numbers>
#include <utility>

#include <pocketfft_hdronly.h>

namespace larmor::dwi {

namespace {

// Returns the index of `position` in a line of `length` samples, wrapping around at both ends.
Index wrap(Index position, Index length) {
    const Index index = position % length;
    return index < 0 ? index + length : index;
}

// Returns the signed frequency of FFT bin `bin` of a line of `length` samples: `bin` up to
// length / 2, bin - length above.
double signed_frequency(Index bin, Index length) {
    return static_cast<double>(2 * bin <= length ? bin : bin - length);
}

// Transforms `values` (n_rows, n_columns) in place by the FFT, along each row or in 2D.
// The inverse transform is scaled so that it undoes the forward one.
void fft(ComplexMatrix& values, bool both_axes, bool forward) {
    const auto n_rows = static_cast<std::size_t>(values.rows());
    const auto n_columns = static_cast<std::size_t>(values.cols());
    const auto element = static_cast<std::ptrdiff_t>(sizeof(std::complex<double>));
    const pocketfft::stride_t strides{static_cast<std::ptrdiff_t>(n_columns) * element, element};
    const pocketfft::shape_t axes = both_axes ? pocketfft::shape_t{0, 1} : pocketfft::shape_t{1};
    const auto n_transformed = static_cast<double>(both_axes ? n_rows * n_columns : n_columns);
    const double scale = forward ? 1.0 : 1.0 / n_transformed;
    pocketfft::c2c<double>({n_rows, n_columns}, strides, strides, axes, forward, values.data(),
                           values.data(), scale);
}

// Returns the cosine term of the split weights for bin k of n, 1 + cos(2 pi k / n).
double weight_term(Index bin, Index length) {
    return 1.0 + std::cos(2.0 * std::numbers::pi * static_cast<double>(bin) /
                          static_cast<double>(length));
}

// Computes the shifted copies of two lines of `image` into lines.shifted_pair, packed as
// real + i imaginary; a negative second_line packs zeros as the second line.
void shift_pair(const Eigen::Ref<Matrix>& image, Index first_line, Index second_line,
                LineScratch& lines) {
    const Index length = image.cols();
    for (Index x = 0; x < length; ++x) {
        lines.packed(0, x) = {image(first_line, x), second_line >= 0 ? image(second_line, x) : 0.0};
    }
    fft(lines.packed, false, true);
    for (Index tried = 0; tried < lines.phases.rows(); ++tried) {
        lines.shifted_pair.row(tried) = lines.packed.row(0).cwiseProduct(lines.phases.row(tried));
    }
    fft(lines.shifted_pair, false, false);
}

// Finds, for each voxel of the line in lines.shifted, the shift with the least ringing and
// stores it in lines.best_shift.
void find_best_shifts(const GibbsSettings& settings, LineScratch& lines) {
    const Index length = lines.shifted.cols();
    const Index padding = settings.max_window + 1;
    lines.least_ringing.setConstant(std::numeric_limits<double>::infinity());
    lines.best_shift.setZero();
    for (Index tried = 0; tried < lines.shifted.rows(); ++tried) {
        // differences(x + padding) = |g(x) - g(x + 1)|, so that the sums below need no
        // wrapping.
        for (Index padded = 0; padded < lines.differences.size(); ++padded) {
            const Index x = wrap(padded - padding, length);
            lines.differences(padded) =
                std::fabs(lines.shifted(tried, x) - lines.shifted(tried, wrap(x + 1, length)));
        }
        for (Index x = 0; x < length; ++x) {
            double right = 0.0;
            double left = 0.0;
            for (Index offset = settings.min_window; offset <= settings.max_window; ++offset) {
                right += lines.differences(x + padding + offset);
                left += lines.differences(x + padding - offset - 1);
            }
            // Strictly less: of equal shifts, the one tried first is kept.
            const double ringing = std::fmin(left, right);
            if (ringing < lines.least_ringing(x)) {
                lines.least_ringing(x) = ringing;
                lines.best_shift(x) = static_cast<int>(tried);
            }
        }
    }
}

// Writes each voxel of the line in lines.shifted at its best shift, interpolated linearly back
// onto the grid, to row `line` of `image`.
void interpolate_back(const GibbsSettings& settings, const LineScratch& lines,
                      Eigen::Ref<Matrix>& image, Index line) {
    const Index length = image.cols();
    const double shift_unit = 1.0 / static_cast<double>(2 * settings.n_shifts);
    for (Index x = 0; x < length; ++x) {
        const Index best = lines.best_shift(x);
        const double shift = static_cast<double>(lines.steps(best)) * shift_unit;
        const double value = lines.shifted(best, x);
        if (shift > 0.0) {
            image(line, x) =
                (1.0 - shift) * value + shift * lines.shifted(best, wrap(x - 1, length));
        } else {
            image(line, x) =
                (1.0 + shift) * value - shift * lines.shifted(best, wrap(x + 1, length));
        }
    }
}

} // namespace

LineScratch make_line_scratch(const GibbsSettings& settings, Index length) {
    const Index n_tried = 2 * settings.n_shifts + 1;
    LineScratch lines{.steps = Eigen::VectorXi(n_tried),
                      .phases = ComplexMatrix(n_tried, length),
                      .packed = ComplexMatrix(1, length),
                      .shifted_pair = ComplexMatrix(n_tried, length),
                      .shifted = Matrix(n_tried, length),
                      .differences = Vector(length + 2 * (settings.max_window + 1)),
                      .least_ringing = Vector(length),
                      .best_shift = Eigen::VectorXi(length)};
    lines.steps(0) = 0;
    for (Index step = 1; step <= settings.n_shifts; ++step) {
        lines.steps(step) = static_cast<int>(step);
        lines.steps(settings.n_shifts + step) = -static_cast<int>(step);
    }
    for (Index tried = 0; tried < n_tried; ++tried) {
        const double shift_voxels =
            static_cast<double>(lines.steps(tried)) / static_cast<double>(2 * settings.n_shifts);
        for (Index bin = 0; bin < length; ++bin) {
            const double angle = 2.0 * std::numbers::pi * signed_frequency(bin, length) *
                                 shift_voxels / static_cast<double>(length);
            lines.phases(tried, bin) = {std::cos(angle), std::sin(angle)};
        }
        if (length % 2 == 0 && lines.steps(tried) != 0) {
            lines.phases(tried, length / 2) = 0.0;
        }
    }
    return lines;
}

GibbsScratch make_gibbs_scratch(const GibbsSettings& settings, Index n_rows, Index n_columns) {
    ComplexMatrix packed_weights(n_rows, n_columns);
    for (Index row = 0; row < n_rows; ++row) {
        const double row_term = weight_term(row, n_rows);
        for (Index column = 0; column < n_columns; ++column) {
            const double column_term = weight_term(column, n_columns);
            const double total = row_term + column_term;
            // The frequency highest along both axes belongs to neither image.
            packed_weights(row, column) =
                total > 0.0 ? std::complex<double>{row_term / total, column_term / total} : 0.0;
        }
    }
    return GibbsScratch{.row_lines = make_line_scratch(settings, n_columns),
                        .column_lines = make_line_scratch(settings, n_rows),
                        .packed_weights = std::move(packed_weights),
                        .spectrum = ComplexMatrix(n_rows, n_columns),
                        .along_rows = Matrix(n_rows, n_columns),
                        .along_columns = Matrix(n_columns, n_rows)};
}

void remove_ringing_along_rows(const GibbsSettings& settings, Eigen::Ref<Matrix> image,
                               LineScratch& lines) {
    const Index n_lines = image.rows();
    for (Index first_line = 0; first_line < n_lines; first_line += 2) {
        const Index second_line = first_line + 1 < n_lines ? first_line + 1 : -1;
        shift_pair(image, first_line, second_line, lines);
        for (const Index line : {first_line, second_line}) {
            if (line < 0) {
                continue;
            }
            const bool is_second = line == second_line;
            for (Index tried = 0; tried < lines.shifted.rows(); ++tried) {
                for (Index x = 0; x < image.cols(); ++x) {
                    const std::complex<double> pair = lines.shifted_pair(tried, x);
                    lines.shifted(tried, x) = is_second ? pair.imag() : pair.real();
                }
            }
            find_best_shifts(settings, lines);
            interpolate_back(settings, lines, image, line);
        }
    }
}

void remove_gibbs_ringing_slice(const GibbsSettings& settings, Eigen::Ref<Matrix> slice,
                                GibbsScratch& scratch) {
    scratch.spectrum = slice.cast<std::complex<double>>();
    fft(scratch.spectrum, true, true);
    // Both weights are real and even in each frequency, so each split image is real, and the
    // inverse transform of their packed sum gives one as the real part and one as the imaginary.
    scratch.spectrum = scratch.spectrum.cwiseProduct(scratch.packed_weights);
    fft(scratch.spectrum, true, false);
    scratch.along_rows = scratch.spectrum.real();
    scratch.along_columns = scratch.spectrum.imag().transpose();

    remove_ringing_along_rows(settings, scratch.along_rows, scratch.row_lines);
    remove_ringing_along_rows(settings, scratch.along_columns, scratch.column_lines);
    slice = scratch.along_rows + scratch.along_columns.transpose();
}

} // namespace larmor::dwi
