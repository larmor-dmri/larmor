// Damped Richardson-Lucy deconvolution of a block of voxels.
#pragma once

#include "common/types.hpp"

namespace larmor::sd {

/// The kernel K as Richardson-Lucy uses it, for c FOD directions and m measurements.
struct RlKernel {
    Matrix transposed; ///< K^T, (c, m)
    Matrix gram;       ///< K^T K, (c, c)
};

/// Settings of damped Richardson-Lucy.
struct DrlSettings {
    /// The FOD amplitude nu below which updates are damped.
    double damping_threshold = 0.0;
    /// Maximum number of iterations per voxel.
    Index n_iterations = 0;
};

/// Runs damped Richardson-Lucy on a block of voxels, one per row of block_signals (b, m),
/// starting from the FODs in block_fods (b, c) and writing the result there.
///
/// The update of amplitudes small compared with nu is damped, the more so the flatter the
/// voxel's signal. Amplitudes are kept above a small floor. A voxel stops once its FOD changes
/// in total by no more than 1e-4 of its largest signal value.
///
/// References:
///   Dell'Acqua F, Scifo P, Rizzo G, et al. A modified damped Richardson-Lucy algorithm to
///   reduce isotropic background effects in spherical deconvolution. NeuroImage 49 (2010)
///   1446-1458.
void damped_richardson_lucy(const RlKernel& kernel, const DrlSettings& settings,
                            const MatrixView& block_signals, Eigen::Ref<Matrix> block_fods);

} // namespace larmor::sd
