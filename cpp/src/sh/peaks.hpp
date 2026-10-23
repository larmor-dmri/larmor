// Peaks of FODs stored as SH series.
#pragma once

#include <array>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "common/types.hpp"

namespace larmor::sh {

/// The settings of the climb to a peak.
struct ClimbSettings {
    /// Maximum number of steps.
    Index max_iters = 50;
    /// The climb stops at a step shorter than this, in radians.
    double tolerance = 1e-4;
    /// The longest step, in radians.
    double max_step = 0.2;
};

/// A peak of an SH series.
struct Peak {
    /// Unit direction.
    Vector3 direction;
    /// Amplitude.
    double amplitude = 0.0;
};

/// Climbs from start, a unit vector, to the peak of the SH series whose basin it lies in.
///
/// coefficients holds n_coeffs(lmax, full) values in the tournier basis. Each step is a Newton
/// step along the gradient on the sphere, at most max_step. The climb returns the direction
/// after the first step below tolerance, with the amplitude before that step, or nothing if it
/// has not stopped within max_iters steps.
///
/// References:
///   MRtrix3, Math::SH::get_peak (core/math/SH.h, version 3.0.8).
///   Tournier JD, Smith R, Raffelt D, et al. MRtrix3: A fast, flexible and open software framework
///   for medical image processing and visualisation. NeuroImage 202 (2019) 116137.
///   doi:10.1016/j.neuroimage.2019.116137
std::optional<Peak> climb(const VectorView& coefficients, Index lmax, bool full,
                          const Vector3& start, const ClimbSettings& settings);

/// The sphere searched for local maxima.
struct Sphere {
    /// Unit vertices.
    std::vector<Vector3> vertices;
    /// The indices of the 6 neighbours of each vertex; a vertex with 5 lists itself as the sixth.
    std::vector<std::array<Index, 6>> neighbours;
};

/// The settings of the peak search of one voxel.
struct PeakSettings {
    /// Maximum number of peaks kept.
    Index n_peaks = 3;
    /// Peaks below this fraction of the largest are dropped.
    double relative_threshold = 0.1;
    /// Peaks below this amplitude are dropped.
    double threshold = 0.0;
    /// The settings of each climb.
    ClimbSettings climb_settings;
};

/// Finds the peaks of one voxel and writes them, largest first, to directions (n_peaks * 3, peak
/// after peak) and amplitudes (n_peaks).
///
/// on_sphere (V) holds the voxel's amplitudes at the sphere's V vertices, and each local maximum
/// with a positive amplitude starts a climb. Peaks closer than 0.99 in cosine are one peak, in
/// |cos| for a symmetric series and in signed cos for a full one. Unused entries are NaN, and all
/// of them are when on_sphere[0] is not finite. The peaks of a symmetric series are turned to
/// z >= 0; those of a full one keep their sign. found is scratch, reused across voxels.
///
/// References:
///   MRtrix3, the command sh2peaks, with the climb of Math::SH::get_peak.
///   Tournier JD, Smith R, Raffelt D, et al. MRtrix3: A fast, flexible and open software framework
///   for medical image processing and visualisation. NeuroImage 202 (2019) 116137.
///   doi:10.1016/j.neuroimage.2019.116137
void voxel_peaks(const VectorView& coefficients, Index lmax, bool full, const VectorView& on_sphere,
                 const Sphere& sphere, const PeakSettings& settings, std::vector<Peak>& found,
                 Eigen::Ref<Vector> directions, Eigen::Ref<Vector> amplitudes);

} // namespace larmor::sh
