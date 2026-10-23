#include "tracking/tracker.hpp"

#include <cstddef>
#include <optional>
#include <utility>
#include <variant>

#include "common/parallel.hpp"

namespace larmor::tracking {

namespace {

// Seeds per block of a batch: small, so that threads share the long streamlines evenly.
constexpr Index SEED_BLOCK_SIZE = 4;

// Writes position to the given row of points, as float32.
void write_point(StreamlineRow& points, Index row, const Vector3& position) {
    points(row, 0) = static_cast<float>(position.x());
    points(row, 1) = static_cast<float>(position.y());
    points(row, 2) = static_cast<float>(position.z());
}

// Reverses the order of the first n_points rows in place.
void reverse_points(StreamlineRow& points, Index n_points) {
    for (Index row = 0; row < n_points / 2; ++row) {
        const Index mirror = n_points - 1 - row;
        for (Index axis = 0; axis < 3; ++axis) {
            std::swap(points(row, axis), points(mirror, axis));
        }
    }
}

// Returns whether a position is off the FOD's image or outside the mask.
bool off_the_seed_region(const TrackingSetup& setup, const Vector3& position) {
    return !setup.fod.grid().contains(position) ||
           (setup.mask.has_value() && !setup.mask->inside(position));
}

} // namespace

HalfResult track_one_way(const TrackingSetup& setup, const TrackingAlgorithm& algorithm,
                         const Vector3& start, const Vector3& direction, Index n_points,
                         StreamlineRow& points, Rng& rng) {
    Vector3 position = start;
    Vector3 current_direction = direction;
    SamplingStats stats;
    while (true) {
        position = Vector3(position.x() + setup.step_size_mm * current_direction.x(),
                           position.y() + setup.step_size_mm * current_direction.y(),
                           position.z() + setup.step_size_mm * current_direction.z());
        // A point off the image or outside the mask ends the streamline without being written.
        if (!setup.fod.grid().contains(position)) {
            return {.stop = StopReason::LeftVolume, .n_points = n_points, .stats = stats};
        }
        if (setup.mask.has_value() && !setup.mask->inside(position)) {
            return {.stop = StopReason::LeftMask, .n_points = n_points, .stats = stats};
        }
        write_point(points, n_points, position);
        ++n_points;
        if (n_points == setup.max_steps + 1) {
            return {.stop = StopReason::MaxLength, .n_points = n_points, .stats = stats};
        }
        const StepResult step =
            algorithm.next_direction(setup.fod, position, current_direction, rng, stats);
        if (const auto* stop = std::get_if<StopReason>(&step)) {
            return {.stop = *stop, .n_points = n_points, .stats = stats};
        }
        current_direction = std::get<Vector3>(step);
    }
}

Track track_seed(const TrackingSetup& setup, const TrackingAlgorithm& algorithm, const Seed& seed,
                 StreamlineRow points) {
    const Track no_start{.n_points = 0,
                         .ends = {std::nullopt, std::nullopt},
                         .direction = Vector3::Zero(),
                         .stats = SamplingStats{}};
    if (off_the_seed_region(setup, seed.position)) {
        return no_start;
    }
    Rng rng = Rng::streamline(setup.run_seed, seed.index);
    Vector3 direction;
    if (seed.direction.has_value()) {
        direction = *seed.direction;
    } else {
        Rng search = rng.search();
        const std::optional<Vector3> found =
            algorithm.first_direction(setup.fod, seed.position, search);
        if (!found.has_value()) {
            return no_start;
        }
        direction = *found;
    }

    write_point(points, 0, seed.position);
    const HalfResult first =
        track_one_way(setup, algorithm, seed.position, direction, 1, points, rng);
    if (!setup.bidirectional) {
        return {.n_points = first.n_points,
                .ends = {std::nullopt, first.stop},
                .direction = direction,
                .stats = first.stats};
    }

    reverse_points(points, first.n_points);
    if (first.n_points == setup.max_steps + 1) {
        return {.n_points = first.n_points,
                .ends = {first.stop, StopReason::MaxLength},
                .direction = direction,
                .stats = first.stats};
    }
    const Vector3 opposite(-direction.x(), -direction.y(), -direction.z());
    const HalfResult second =
        track_one_way(setup, algorithm, seed.position, opposite, first.n_points, points, rng);
    SamplingStats stats = first.stats;
    stats.add(second.stats);
    return {.n_points = second.n_points,
            .ends = {first.stop, second.stop},
            .direction = direction,
            .stats = stats};
}

std::vector<Track> track_batch(const TrackingSetup& setup, const TrackingAlgorithm& algorithm,
                               std::span<const Seed> seeds, Eigen::Ref<PositionsMatrix> positions,
                               int n_threads) {
    const auto n_seeds = static_cast<Index>(seeds.size());
    const Index n_rows = setup.max_steps + 1;
    std::vector<Track> tracks(seeds.size());
    parallel_for_blocks(n_seeds, SEED_BLOCK_SIZE, n_threads, [&](Index begin, Index end) {
        for (Index item = begin; item < end; ++item) {
            const auto unsigned_item = static_cast<std::size_t>(item);
            tracks[unsigned_item] =
                track_seed(setup, algorithm, seeds[unsigned_item],
                           StreamlineRow(positions.row(item).data(), n_rows, 3));
        }
    });
    return tracks;
}

} // namespace larmor::tracking
