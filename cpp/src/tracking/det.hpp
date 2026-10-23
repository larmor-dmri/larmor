// Deterministic tracking, which follows the FOD peak nearest to the current direction.
#pragma once

#include <optional>

#include "common/types.hpp"
#include "sh/peaks.hpp"
#include "tracking/algorithm.hpp"
#include "tracking/grid.hpp"
#include "tracking/rng.hpp"

namespace larmor::tracking {

/// Deterministic tracking, in which each step follows the peak of the FOD nearest to the
/// previous direction.
///
/// The FOD's SH coefficients (tournier basis, even degrees, or all degrees if full) are
/// interpolated trilinearly and climbed from the previous direction to a peak. The streamline
/// stops with NoDirection if the climb does not converge, LowFod if the peak's amplitude is below
/// the cutoff, and SharpTurn if the peak is further than the maximum angle from the previous
/// direction (never at 180 degrees). A seed without a direction climbs from up to 20 random
/// directions and starts along the first peak whose amplitude is at least the cutoff. The FOD
/// must have n_coeffs(lmax, full) channels, with lmax at most sh::MAX_LMAX.
///
/// References:
///   MRtrix3, the command tckgen with -algorithm SD_STREAM.
///   Tournier JD, Smith R, Raffelt D, et al. MRtrix3: A fast, flexible and open software framework
///   for medical image processing and visualisation. NeuroImage 202 (2019) 116137.
///   doi:10.1016/j.neuroimage.2019.116137
///   Tournier JD, Calamante F, Gadian DG, Connelly A. Direct estimation of the fiber orientation
///   density function from diffusion-weighted MRI data using spherical deconvolution. NeuroImage
///   23 (2004) 1176-1185. doi:10.1016/j.neuroimage.2004.07.037
class DeterministicAlgorithm final : public TrackingAlgorithm {
public:
    /// Makes the algorithm with an FOD cutoff, a maximum turn per step in degrees, and the FOD's
    /// lmax and whether it has all degrees.
    DeterministicAlgorithm(double cutoff, double max_angle_deg, Index lmax, bool full);

    [[nodiscard]] std::optional<Vector3> first_direction(const Field& fod, const Vector3& position,
                                                         Rng& rng) const override;

    [[nodiscard]] StepResult next_direction(const Field& fod, Vector3 position,
                                            Vector3 previous_direction, Rng& rng,
                                            SamplingStats& stats) const override;

private:
    double fod_cutoff;
    // The cosine of the maximum angle; -infinity at 180 degrees, so that no turn is too sharp,
    // even with a dot product rounded below -1.
    double cos_max_angle;
    Index fod_lmax;
    // Whether the FOD has all degrees rather than the even ones.
    bool full_series;
    sh::ClimbSettings climb_settings;
};

} // namespace larmor::tracking
