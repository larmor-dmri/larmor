// The step loop of streamline tracking, for one seed or a batch of seeds in parallel.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include <Eigen/Core>

#include "common/types.hpp"
#include "tracking/algorithm.hpp"
#include "tracking/grid.hpp"
#include "tracking/rng.hpp"

namespace larmor::tracking {

/// What every streamline of a run shares.
struct TrackingSetup {
    /// The FOD's SH coefficients.
    Field fod;
    /// The tracking mask, if any.
    std::optional<Mask> mask;
    /// The step size, in mm.
    double step_size_mm;
    /// The most steps a streamline takes, over both directions; it has at most max_steps + 1
    /// points with the seed.
    Index max_steps;
    /// Whether each seed is tracked in both directions.
    bool bidirectional;
    /// The run's seed, which keys every streamline's random numbers (Rng::streamline).
    std::uint64_t run_seed;
};

/// A seed to track from.
struct Seed {
    /// Its world position.
    Vector3 position;
    /// The direction to start in, if given.
    std::optional<Vector3> direction;
    /// Its global index, which keys its random numbers.
    std::uint64_t index;
};

/// The points of one streamline, (max_steps + 1, 3) float32, one point per row.
using StreamlinePoints = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;

/// The points of one streamline inside a larger buffer, written in place.
using StreamlineRow = Eigen::Map<StreamlinePoints>;

/// A batch's buffer of points, (batch, (max_steps + 1) * 3) float32, stored row by row; row i
/// holds the points of seed i.
using PositionsMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// What happened to one seed.
struct Track {
    /// Points written to the row: 0 when the seed did not start.
    Index n_points;
    /// Why the start and the end of the row stopped; empty for an end that was not tracked.
    std::array<std::optional<StopReason>, 2> ends;
    /// The direction the streamline started in, given or found; zero when it did not start.
    Vector3 direction;
    /// The algorithm's counts over the streamline.
    SamplingStats stats;
};

/// What tracking in one direction gives.
struct HalfResult {
    /// Why it stopped.
    StopReason stop;
    /// The number of points in the row after it.
    Index n_points;
    /// The algorithm's counts on the way.
    SamplingStats stats;
};

/// Tracks from start along direction until a stop, appending each new point to points after the
/// n_points already there.
///
/// Each step moves step_size_mm along the current direction. A new point off the FOD's image
/// (LeftVolume) or outside the mask (LeftMask) is not written, so the last point is the last one
/// inside. A streamline that reaches max_steps + 1 points stops with MaxLength and keeps its last
/// point. Otherwise the algorithm chooses the next direction or stops, and a refused step adds no
/// point. n_points must be at most max_steps, and points must have max_steps + 1 rows.
///
/// References:
///   MRtrix3, the command tckgen.
///   Tournier JD, Smith R, Raffelt D, et al. MRtrix3: A fast, flexible and open software framework
///   for medical image processing and visualisation. NeuroImage 202 (2019) 116137.
///   doi:10.1016/j.neuroimage.2019.116137
HalfResult track_one_way(const TrackingSetup& setup, const TrackingAlgorithm& algorithm,
                         const Vector3& start, const Vector3& direction, Index n_points,
                         StreamlineRow& points, Rng& rng);

/// Tracks one seed into points, a (max_steps + 1, 3) row, in one or both directions.
///
/// A seed off the FOD's image or outside the mask does not start (0 points, no ends), whatever its
/// direction. A seed without a direction asks the algorithm for one, and does not start if none
/// is found. Bidirectionally, the row runs from the end of the first half, through the seed, to
/// the end of the second; if the first half used up max_steps, the second is not started and its
/// end is MaxLength. One way, the row starts at the seed and its first end is empty. Rows past
/// the points written are left as they are.
///
/// References:
///   MRtrix3, the command tckgen.
///   Tournier JD, Smith R, Raffelt D, et al. MRtrix3: A fast, flexible and open software framework
///   for medical image processing and visualisation. NeuroImage 202 (2019) 116137.
///   doi:10.1016/j.neuroimage.2019.116137
Track track_seed(const TrackingSetup& setup, const TrackingAlgorithm& algorithm, const Seed& seed,
                 StreamlineRow points);

/// Tracks a batch of seeds in parallel; seed i writes only row i of positions.
///
/// The result does not depend on the number of threads. positions must have at least as many
/// rows as seeds and (max_steps + 1) * 3 columns; rows past the seeds are left as they are.
/// Returns each seed's Track, in seed order.
std::vector<Track> track_batch(const TrackingSetup& setup, const TrackingAlgorithm& algorithm,
                               std::span<const Seed> seeds, Eigen::Ref<PositionsMatrix> positions,
                               int n_threads);

} // namespace larmor::tracking
