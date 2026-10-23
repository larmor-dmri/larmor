// Removal of Gibbs ringing from 2D slices by local sub-voxel shifts.
#pragma once

#include <complex>

#include "common/types.hpp"

namespace larmor::dwi {

/// A matrix of complex doubles, stored row by row.
using ComplexMatrix =
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// The shifts tried and the window over which the ringing is measured.
struct GibbsSettings {
    /// Shifts tried on each side of zero, in steps of 1 / (2 n_shifts) voxels.
    Index n_shifts = 20;
    /// The ringing next to a voxel is measured from its neighbour min_window voxels away on
    /// each side.
    Index min_window = 1;
    /// The window ends max_window + 1 voxels away on each side.
    Index max_window = 3;
};

/// Buffers for the lines of one length, reused for every slice.
struct LineScratch {
    /// Shift of each row of phases, in units of 1 / (2 n_shifts) voxels: 0, 1, ..., n_shifts,
    /// -1, ..., -n_shifts. Ties go to the earlier row.
    Eigen::VectorXi steps;
    /// The phase factor that shifts each frequency bin (columns, in FFT order) by each shift
    /// (rows), exp(2 pi i f s / n) with f the bin's signed frequency; (2 n_shifts + 1, n).
    ComplexMatrix phases;
    /// Two lines packed as one complex line, real + i imaginary, then its spectrum; (1, n).
    ComplexMatrix packed;
    /// The shifted copies of a packed pair; (2 n_shifts + 1, n).
    ComplexMatrix shifted_pair;
    /// The shifted copies of one line; (2 n_shifts + 1, n).
    Matrix shifted;
    /// The absolute differences of neighbouring samples of one shifted copy, padded on each
    /// side by max_window + 1 samples that wrap around the line; (n + 2 (max_window + 1),).
    Vector differences;
    /// The least ringing found so far at each voxel; (n,).
    Vector least_ringing;
    /// The row of the shift that gave it; (n,).
    Eigen::VectorXi best_shift;
};

/// Makes the buffers and the phase factors for lines of `length` samples.
///
/// For an even length the highest frequency, n / 2, has no unique shifted form: the shifted
/// copies leave it out and the unshifted copy keeps it, so a line whose best shift is zero comes
/// out unchanged.
LineScratch make_line_scratch(const GibbsSettings& settings, Index length);

/// Buffers one thread reuses for every slice of one shape.
struct GibbsScratch {
    /// Buffers for lines along a row, of n_columns samples.
    LineScratch row_lines;
    /// Buffers for lines along a column, of n_rows samples.
    LineScratch column_lines;
    /// The weights of the two images the spectrum is split into, one complex number per
    /// frequency. The real part weights the image unringed along its rows, the imaginary part
    /// the image unringed along its columns; (n_rows, n_columns).
    ComplexMatrix packed_weights;
    /// The slice's spectrum, then the two split images packed as real + i imaginary;
    /// (n_rows, n_columns).
    ComplexMatrix spectrum;
    /// The image unringed along its rows; (n_rows, n_columns).
    Matrix along_rows;
    /// The image unringed along its columns, transposed; (n_columns, n_rows).
    Matrix along_columns;
};

/// Makes the scratch of a thread for slices of n_rows x n_columns voxels.
///
/// Its weights split the slice's spectrum into an image unringed along its rows, with weight
/// w_r / (w_r + w_c), and one unringed along its columns, with weight w_c / (w_r + w_c), where
/// w_r = 1 + cos(2 pi k_r / n_rows) and w_c = 1 + cos(2 pi k_c / n_columns). The weights sum to
/// 1, except at the frequency that is the highest along both axes when both sizes are even, where
/// both are 0.
GibbsScratch make_gibbs_scratch(const GibbsSettings& settings, Index n_rows, Index n_columns);

/// Removes the ringing along each row of `image` (n_lines, n), in place; `lines` is the
/// scratch for lines of n samples.
///
/// Each line is shifted by multiplying its spectrum with the phases in `lines`. Each voxel takes
/// the shift with the least total variation over the window on its left or on its right, and its
/// value is interpolated linearly back onto the grid. Two lines are shifted together as the real
/// and imaginary parts of one complex line.
///
/// References:
///   Kellner E, Dhital B, Kiselev VG, Reisert M. Gibbs-ringing artifact removal based on local
///   subvoxel-shifts. Magn Reson Med 76 (2016) 1574-1581.
void remove_ringing_along_rows(const GibbsSettings& settings, Eigen::Ref<Matrix> image,
                               LineScratch& lines);

/// Removes Gibbs ringing from one slice (n_rows, n_columns), in place.
///
/// The slice is split into two images by the scratch's weights; one is unringed along its rows
/// and the other along its columns, and the slice becomes their sum. A slice with a NaN or
/// infinite value comes out all NaN.
///
/// References:
///   Kellner E, Dhital B, Kiselev VG, Reisert M. Gibbs-ringing artifact removal based on local
///   subvoxel-shifts. Magn Reson Med 76 (2016) 1574-1581.
void remove_gibbs_ringing_slice(const GibbsSettings& settings, Eigen::Ref<Matrix> slice,
                                GibbsScratch& scratch);

} // namespace larmor::dwi
