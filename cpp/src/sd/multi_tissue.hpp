// The multi-tissue fit of a block of voxels, with an FOD per anisotropic response function and
// the signal fraction of every tissue.
#pragma once

#include "common/types.hpp"
#include "sd/fractions.hpp"
#include "sd/solver.hpp"

namespace larmor::sd {

/// One resolution of the fit: a kernel and the FOD solver prepared from it.
struct Resolution {
    /// The kernel.
    const TissueKernel& kernel;
    /// The FOD solver prepared from it.
    const FodSolver& solver;
};

/// Settings of the loop that alternates FOD and fraction fits.
struct LoopSettings {
    /// Maximum number of rounds of the low-resolution loop.
    Index max_rounds = 0;
    /// A voxel leaves the loop when no fraction changes by this much from one round to the next.
    double fraction_tolerance = 0.0;
    /// The NNLS gradient tolerance of the final fraction fit, below which a fraction is zero.
    double final_gradient_tolerance = 0.0;
};

/// Runs the multi-tissue fit of a block of voxels, one per row of block_signals (b, m),
/// unweighted; the shell weights (m,) apply to the FOD fits.
///
/// Each voxel alternates low-resolution FOD and fraction fits, with all anisotropic tissues as
/// one column, until no fraction changes by fraction_tolerance or more, or after max_rounds
/// rounds. With its last isotropic fractions, it then fits the high-resolution FOD, written to
/// block_fods (b, ca), and the final fractions, one per anisotropic response function then one
/// per isotropic tissue, normalised to sum to 1, written to block_fractions (b, n_blocks + ci).
///
/// References:
///   Guo F, Leemans A, Viergever MA, Dell'Acqua F, De Luca A. Generalized Richardson-Lucy
///   (GRL) for analyzing multi-shell diffusion MRI data. NeuroImage 218 (2020) 116948.
///   doi:10.1016/j.neuroimage.2020.116948
///   MRIToolkit (MRTTrack), A. De Luca, https://github.com/delucaal/MRIToolkit
void fit_multi_tissue_block(const MatrixView& block_signals, const VectorView& weights,
                            const Resolution& low_resolution, const Resolution& high_resolution,
                            const LoopSettings& settings, Eigen::Ref<Matrix> block_fods,
                            Eigen::Ref<Matrix> block_fractions);

} // namespace larmor::sd
