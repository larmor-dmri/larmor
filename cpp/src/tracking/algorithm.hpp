// The interface between the step loop and a tracking algorithm.
#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include "common/types.hpp"
#include "tracking/grid.hpp"
#include "tracking/rng.hpp"

namespace larmor::tracking {

/// Why one end of a streamline stopped.
///
/// The values are the codes of Python's EndStatus. Its code 0 (NONE), an end that was not
/// tracked, is an empty std::optional<StopReason> here.
enum class StopReason : std::uint8_t {
    MaxLength = 1,
    LeftVolume = 2,
    LeftMask = 3,
    LowFod = 4,
    SharpTurn = 5,
    NoDirection = 6,
};

/// Returns the EndStatus code of an end, or 0 for an end that was not tracked.
std::uint8_t end_code(std::optional<StopReason> end);

/// What a sampling algorithm counts while it tracks.
///
/// An algorithm that does not sample leaves every count at zero. The counts are sums and a
/// maximum, so the totals of a batch do not depend on which thread tracked which streamline.
///
/// References:
///   MRtrix3, the command tckgen.
///   Tournier JD, Smith R, Raffelt D, et al. MRtrix3: A fast, flexible and open software framework
///   for medical image processing and visualisation. NeuroImage 202 (2019) 116137.
///   doi:10.1016/j.neuroimage.2019.116137
struct SamplingStats {
    /// Steps that drew candidate directions.
    std::uint64_t sampled_steps = 0;
    /// Over the accepted steps, the sum of the accepted candidate's index, from 0.
    std::uint64_t accepted_index_sum = 0;
    /// Candidates whose amplitude was above the rejection-sampling bound.
    std::uint64_t truncations = 0;
    /// The largest amplitude / bound among those candidates; 0 if there were none.
    double max_truncation = 0.0;

    /// Adds the counts of other, keeping the larger of the two maxima.
    void add(const SamplingStats& other);

    bool operator==(const SamplingStats&) const = default;
};

/// The direction to follow next, or why the streamline stops here.
using StepResult = std::variant<Vector3, StopReason>;

/// A tracking algorithm, which finds the direction a streamline starts in and chooses each next
/// one.
///
/// The step loop (track_one_way) stops a streamline where a point lands off the image or the mask
/// or at the maximum length; the algorithm stops it where the FOD is too low, the turn too sharp
/// or no direction is found. An algorithm is shared by all threads, so it keeps any
/// per-streamline state in its arguments.
///
/// References:
///   MRtrix3, the command tckgen.
///   Tournier JD, Smith R, Raffelt D, et al. MRtrix3: A fast, flexible and open software framework
///   for medical image processing and visualisation. NeuroImage 202 (2019) 116137.
///   doi:10.1016/j.neuroimage.2019.116137
class TrackingAlgorithm {
public:
    TrackingAlgorithm() = default;
    TrackingAlgorithm(const TrackingAlgorithm&) = default;
    TrackingAlgorithm& operator=(const TrackingAlgorithm&) = default;
    TrackingAlgorithm(TrackingAlgorithm&&) = default;
    TrackingAlgorithm& operator=(TrackingAlgorithm&&) = default;
    virtual ~TrackingAlgorithm() = default;

    /// Returns the direction a streamline starts in from a seed at position, or none if the
    /// search fails.
    ///
    /// Called only for a seed without a given direction, with the generator of the streamline's
    /// search (Rng::search).
    [[nodiscard]] virtual std::optional<Vector3>
    first_direction(const Field& fod, const Vector3& position, Rng& rng) const = 0;

    /// Returns the direction of the next step from position, after a step along
    /// previous_direction, or the reason the streamline stops here.
    ///
    /// stats accumulates the algorithm's counts over the streamline.
    [[nodiscard]] virtual StepResult next_direction(const Field& fod, Vector3 position,
                                                    Vector3 previous_direction, Rng& rng,
                                                    SamplingStats& stats) const = 0;
};

} // namespace larmor::tracking
