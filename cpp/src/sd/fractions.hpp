// The tissue kernel of a multi-tissue fit and the fit of each tissue's signal fraction.
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "common/types.hpp"

namespace larmor::sd {

/// The columns [begin, end) of a matrix.
struct ColumnRange {
    Index begin = 0;
    Index end = 0;

    /// Returns the number of columns, end - begin.
    [[nodiscard]] Index size() const { return end - begin; }
};

/// The signal of every tissue along every FOD direction, one row per measurement.
///
/// An anisotropic response function gives one column per FOD direction, and its columns form
/// one of anisotropic_blocks; an isotropic one gives a single column. weighted_isotropic has
/// each measurement multiplied by its shell weight, for the FOD solvers, which fit the weighted
/// signal; the fraction fits use the unweighted one.
struct TissueKernel {
    Matrix anisotropic;                          ///< (m, ca)
    Matrix isotropic;                            ///< (m, ci)
    Matrix weighted_isotropic;                   ///< (m, ci)
    std::vector<ColumnRange> anisotropic_blocks; ///< within 0..ca, one per response function

    /// Returns the number of anisotropic columns, ca.
    [[nodiscard]] Index n_anisotropic() const { return anisotropic.cols(); }

    /// Returns the number of isotropic tissues, ci.
    [[nodiscard]] Index n_isotropic() const { return isotropic.cols(); }
};

/// How the anisotropic part of the signal enters a fraction fit.
enum class Grouping : std::uint8_t {
    /// One column for all anisotropic tissues together.
    Lumped,
    /// One column per anisotropic response function.
    PerBlock,
};

/// Returns the number of anisotropic columns of a fraction fit with grouping on kernel.
Index n_groups(Grouping grouping, const TissueKernel& kernel);

/// Returns the median of values, the mean of the two middle values for an even length.
double median(const VectorView& values);

/// Fits the signal fraction of each tissue in one voxel from its FOD (ca,) and its unweighted
/// signal (m,).
///
/// The FOD values below their median are set to zero. Each anisotropic group's signal from that
/// FOD, scaled to a maximum of 1, is fitted together with the isotropic columns by non-negative
/// least squares. The fractions come back with the groups first, then the isotropic tissues.
/// gradient_tolerance is the NNLS tolerance below which a column stays at zero; none scales it
/// to the columns.
///
/// References:
///   Guo F, Leemans A, Viergever MA, Dell'Acqua F, De Luca A. Generalized Richardson-Lucy
///   (GRL) for analyzing multi-shell diffusion MRI data. NeuroImage 218 (2020) 116948.
///   doi:10.1016/j.neuroimage.2020.116948
///   De Luca A, Guo F, Froeling M, Leemans A. Spherical deconvolution with tissue-specific
///   response functions and multi-shell diffusion MRI to estimate multiple fiber orientation
///   distributions (mFODs). NeuroImage 222 (2020) 117206. doi:10.1016/j.neuroimage.2020.117206
///   MRIToolkit (MRTTrack), A. De Luca, https://github.com/delucaal/MRIToolkit
Vector compute_fractions(const VectorView& fod, const TissueKernel& kernel,
                         const VectorView& signal, Grouping grouping,
                         std::optional<double> gradient_tolerance);

} // namespace larmor::sd
