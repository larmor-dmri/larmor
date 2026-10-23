// Probabilistic tracking, which draws each direction in proportion to the FOD amplitude.
#pragma once

#include <optional>

#include "common/types.hpp"
#include "tracking/algorithm.hpp"
#include "tracking/calibrate.hpp"
#include "tracking/grid.hpp"
#include "tracking/rng.hpp"

namespace larmor::tracking {

/// Probabilistic tracking: each step draws its direction in the cone around the previous one with
/// a probability proportional to the FOD amplitude, by rejection sampling.
///
/// The FOD's SH coefficients (tournier basis, even degrees, or all degrees if full) are
/// interpolated trilinearly. The bound of the rejection sampling is the largest amplitude on the
/// calibration grid, turned to the previous direction, times the calibration's ratio; if that
/// amplitude is not positive, the streamline stops with LowFod. Then up to `trials` candidates are
/// drawn uniformly in the cone of half-angle max_angle around the previous direction; a candidate
/// whose amplitude is NaN or not above the cutoff is skipped, and one with amplitude a is
/// accepted with probability a / bound. If none is accepted, the streamline stops with
/// NoDirection. A candidate above the bound is counted as a truncation. A seed without a
/// direction tries up to `attempts` random directions and starts along the first whose amplitude
/// is above the cutoff. The FOD must have n_coeffs(lmax, full) channels, with lmax at most
/// sh::MAX_LMAX.
///
/// References:
///   MRtrix3, the command tckgen with -algorithm iFOD1.
///   Tournier JD, Smith R, Raffelt D, et al. MRtrix3: A fast, flexible and open software framework
///   for medical image processing and visualisation. NeuroImage 202 (2019) 116137.
///   doi:10.1016/j.neuroimage.2019.116137
///   Tournier JD, Calamante F, Connelly A. MRtrix: Diffusion tractography in crossing fiber
///   regions. International Journal of Imaging Systems and Technology 22 (2012) 53-66.
///   doi:10.1002/ima.22005
class ProbabilisticAlgorithm final : public TrackingAlgorithm {
public:
    /// Makes the algorithm with an FOD cutoff, the cone's half-angle in degrees, the FOD's lmax
    /// and whether it has all degrees, the candidates per step and the attempts per seed.
    ///
    /// It calibrates the bound (calibrate), which can take a while for a narrow cone.
    ProbabilisticAlgorithm(double cutoff, double max_angle_deg, Index lmax, bool full, Index trials,
                           Index attempts);

    /// Makes the same algorithm with a given calibration instead of its own.
    ProbabilisticAlgorithm(double cutoff, double max_angle_deg, Index lmax, bool full, Index trials,
                           Index attempts, Calibration calibration);

    /// Returns the number of directions the bound is taken over.
    [[nodiscard]] Index calibration_directions() const;

    [[nodiscard]] std::optional<Vector3> first_direction(const Field& fod, const Vector3& position,
                                                         Rng& rng) const override;

    [[nodiscard]] StepResult next_direction(const Field& fod, Vector3 position,
                                            Vector3 previous_direction, Rng& rng,
                                            SamplingStats& stats) const override;

private:
    double fod_cutoff;
    double max_angle_rad;
    // The largest sin theta in the cone: sin(max_angle_rad) up to pi / 2, else 1.
    double sin_bound;
    Index fod_lmax;
    bool full_series;
    Index n_trials;
    Index n_attempts;
    Calibration bound_calibration;
};

} // namespace larmor::tracking
