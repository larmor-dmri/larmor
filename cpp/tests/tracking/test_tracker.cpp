#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <doctest/doctest.h>

#include "tracking/algorithm.hpp"
#include "tracking/grid.hpp"
#include "tracking/rng.hpp"
#include "tracking/testing.hpp"
#include "tracking/tracker.hpp"

namespace {

using larmor::Index;
using larmor::Vector3;
using larmor::tracking::Field;
using larmor::tracking::GridShape;
using larmor::tracking::InverseAffine;
using larmor::tracking::Mask;
using larmor::tracking::PositionsMatrix;
using larmor::tracking::Rng;
using larmor::tracking::SamplingStats;
using larmor::tracking::Seed;
using larmor::tracking::StepResult;
using larmor::tracking::StopReason;
using larmor::tracking::StreamlinePoints;
using larmor::tracking::StreamlineRow;
using larmor::tracking::TestAlgorithm;
using larmor::tracking::TestStep;
using larmor::tracking::Track;
using larmor::tracking::TrackingAlgorithm;
using larmor::tracking::TrackingSetup;

using Ends = std::array<std::optional<StopReason>, 2>;

constexpr GridShape CUBE_SHAPE{10, 10, 10};
constexpr float NOT_A_NUMBER = std::numeric_limits<float>::quiet_NaN();
Vector3 along_x() {
    return Vector3::UnitX();
}

Vector3 centre() {
    return {5.0, 5.0, 5.0};
}

TestAlgorithm algorithm(TestStep rule, StopReason refuse_with = StopReason::LowFod) {
    return {rule, refuse_with, std::numeric_limits<double>::infinity()};
}

// A 10-voxel cube of ones (the grid ends at -0.5 and 9.5), one channel.
std::vector<float> cube() {
    std::vector<float> values(1000, 1.0F);
    return values;
}

// The cube's mask with 1 where lower <= x <= upper.
std::vector<float> slab_mask(Index lower, Index upper) {
    std::vector<float> values(1000, 0.0F);
    for (Index x = lower; x <= upper; ++x) {
        for (Index yz = 0; yz < 100; ++yz) {
            values[static_cast<std::size_t>(x * 100 + yz)] = 1.0F;
        }
    }
    return values;
}

// 1 mm steps on the cube, run seed 0.
TrackingSetup setup(const std::vector<float>& fod, const std::vector<float>* mask, Index max_steps,
                    bool bidirectional) {
    std::optional<Mask> tracking_mask;
    if (mask != nullptr) {
        tracking_mask.emplace(mask->data(), CUBE_SHAPE, InverseAffine::Identity());
    }
    return {.fod = Field(fod.data(), CUBE_SHAPE, 1, InverseAffine::Identity()),
            .mask = tracking_mask,
            .step_size_mm = 1.0,
            .max_steps = max_steps,
            .bidirectional = bidirectional,
            .run_seed = 0};
}

Seed seed_at(const Vector3& position, std::optional<Vector3> direction, std::uint64_t index = 0) {
    return {.position = position, .direction = direction, .index = index};
}

// Tracks one seed into a NaN row; returns the track and its points.
std::pair<Track, StreamlinePoints> run(const TrackingSetup& tracking,
                                       const TrackingAlgorithm& steps, const Seed& seed) {
    StreamlinePoints points = StreamlinePoints::Constant(tracking.max_steps + 1, 3, NOT_A_NUMBER);
    const Track track = larmor::tracking::track_seed(
        tracking, steps, seed, StreamlineRow(points.data(), points.rows(), 3));
    return {track, points};
}

std::vector<float> xs(const StreamlinePoints& points, Index n_points) {
    std::vector<float> values;
    values.reserve(static_cast<std::size_t>(n_points));
    for (Index row = 0; row < n_points; ++row) {
        values.push_back(points(row, 0));
    }
    return values;
}

bool all_nan(const Eigen::Ref<const PositionsMatrix>& values) {
    return values.array().isNaN().all();
}

// Counts how often it is asked for a direction; goes straight on.
class Counting final : public TrackingAlgorithm {
public:
    [[nodiscard]] std::optional<Vector3> first_direction(const Field& /*fod*/,
                                                         const Vector3& /*position*/,
                                                         Rng& /*rng*/) const override {
        return std::nullopt;
    }

    [[nodiscard]] StepResult next_direction(const Field& /*fod*/, Vector3 /*position*/,
                                            Vector3 previous_direction, Rng& /*rng*/,
                                            SamplingStats& /*stats*/) const override {
        calls.fetch_add(1, std::memory_order_relaxed);
        return previous_direction;
    }

    [[nodiscard]] int n_calls() const { return calls.load(); }

private:
    mutable std::atomic<int> calls{0};
};

std::vector<Seed> batch_seeds() {
    std::vector<Seed> seeds;
    seeds.reserve(20);
    for (std::uint64_t index = 0; index < 20; ++index) {
        seeds.push_back(seed_at(Vector3(2.0 + 0.3 * static_cast<double>(index), 5.0, 5.0),
                                std::nullopt, index));
    }
    return seeds;
}

std::pair<std::vector<Track>, PositionsMatrix> batch(int n_threads) {
    const std::vector<float> fod = cube();
    const TrackingSetup tracking = setup(fod, nullptr, 8, true);
    PositionsMatrix positions = PositionsMatrix::Constant(23, 27, NOT_A_NUMBER);
    const std::vector<Seed> seeds = batch_seeds();
    std::vector<Track> tracks = larmor::tracking::track_batch(tracking, algorithm(TestStep::Wander),
                                                              seeds, positions, n_threads);
    return {std::move(tracks), std::move(positions)};
}

} // namespace

TEST_CASE("a bidirectional line ends at the last points on the image") {
    const std::vector<float> fod = cube();
    const auto [track, points] = run(setup(fod, nullptr, 100, true), algorithm(TestStep::Straight),
                                     seed_at(centre(), along_x()));
    // The points at x = 10 and -1 are off the image and are not written.
    CHECK(track.n_points == 10);
    CHECK(track.ends == Ends{StopReason::LeftVolume, StopReason::LeftVolume});
    std::vector<float> expected;
    for (int x = 9; x >= 0; --x) {
        expected.push_back(static_cast<float>(x));
    }
    CHECK(xs(points, 10) == expected);
    CHECK((points.topRows(10).rightCols(2).array() == 5.0F).all());
}

TEST_CASE("one way starts at the seed and leaves its end untracked") {
    const std::vector<float> fod = cube();
    const auto [track, points] = run(setup(fod, nullptr, 100, false), algorithm(TestStep::Straight),
                                     seed_at(centre(), along_x()));
    CHECK(track.n_points == 5);
    CHECK(track.ends == Ends{std::nullopt, StopReason::LeftVolume});
    CHECK(xs(points, 5) == std::vector<float>{5.0F, 6.0F, 7.0F, 8.0F, 9.0F});
    CHECK(std::isnan(points(5, 0)));
}

TEST_CASE("max_steps counts the steps of the whole streamline") {
    const std::vector<float> fod = cube();
    // The first half uses them all: the second is not started.
    const auto [track, points] = run(setup(fod, nullptr, 4, true), algorithm(TestStep::Straight),
                                     seed_at(centre(), along_x()));
    CHECK(track.n_points == 5);
    CHECK(track.ends == Ends{StopReason::MaxLength, StopReason::MaxLength});
    CHECK(xs(points, 5) == std::vector<float>{9.0F, 8.0F, 7.0F, 6.0F, 5.0F});
    // The first half leaves the image, the second uses up what is left.
    const auto [second_track, second_points] = run(
        setup(fod, nullptr, 7, true), algorithm(TestStep::Straight), seed_at(centre(), along_x()));
    CHECK(second_track.n_points == 8);
    CHECK(second_track.ends == Ends{StopReason::LeftVolume, StopReason::MaxLength});
    CHECK(xs(second_points, 8) ==
          std::vector<float>{9.0F, 8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F});
}

TEST_CASE("the mask ends a streamline at its last point inside, at both ends") {
    const std::vector<float> fod = cube();
    const std::vector<float> mask = slab_mask(3, 7);
    const auto [track, points] = run(setup(fod, &mask, 100, true), algorithm(TestStep::Straight),
                                     seed_at(centre(), along_x()));
    CHECK(track.ends == Ends{StopReason::LeftMask, StopReason::LeftMask});
    CHECK(xs(points, track.n_points) == std::vector<float>{7.0F, 6.0F, 5.0F, 4.0F, 3.0F});
}

TEST_CASE("the edge is checked before the mask and the mask before the length") {
    const std::vector<float> fod = cube();
    // Point 10 leaves both the image and the mask, and is the last step allowed.
    const std::vector<float> everywhere = slab_mask(0, 9);
    const auto [track, points] = run(setup(fod, &everywhere, 5, false),
                                     algorithm(TestStep::Straight), seed_at(centre(), along_x()));
    CHECK(track.ends[1] == StopReason::LeftVolume);
    // Point 9 leaves the mask and is the last step allowed.
    const std::vector<float> mask = slab_mask(0, 8);
    const auto [masked_track, masked_points] = run(
        setup(fod, &mask, 4, false), algorithm(TestStep::Straight), seed_at(centre(), along_x()));
    CHECK(masked_track.ends[1] == StopReason::LeftMask);
}

TEST_CASE("a seed off the grid or outside the mask does not start") {
    const std::vector<float> fod = cube();
    std::vector<float> mask(1000, 0.0F);
    mask[555] = 1.0F;
    const TrackingSetup tracking = setup(fod, &mask, 100, true);
    for (const Vector3& position : {Vector3(-1.0, 5.0, 5.0), Vector3(2.0, 2.0, 2.0)}) {
        const auto [track, points] =
            run(tracking, algorithm(TestStep::Straight), seed_at(position, along_x()));
        CHECK(track.n_points == 0);
        CHECK(track.ends == Ends{std::nullopt, std::nullopt});
        CHECK(points.array().isNaN().all());
    }
}

TEST_CASE("a refused step adds no point and records its stop") {
    const std::vector<float> fod = cube();
    const auto [track, points] =
        run(setup(fod, nullptr, 100, true), algorithm(TestStep::Refuse, StopReason::LowFod),
            seed_at(centre(), along_x()));
    CHECK(track.n_points == 3);
    CHECK(track.ends == Ends{StopReason::LowFod, StopReason::LowFod});
    CHECK(xs(points, 3) == std::vector<float>{6.0F, 5.0F, 4.0F});
}

TEST_CASE("the first step follows the seed, then the algorithm its previous step") {
    const std::vector<float> fod = cube();
    const auto [track, points] =
        run(setup(fod, nullptr, 4, false), algorithm(TestStep::QuarterTurn),
            seed_at(centre(), along_x()));
    StreamlinePoints expected(5, 3);
    expected << 5.0F, 5.0F, 5.0F, 6.0F, 5.0F, 5.0F, 6.0F, 6.0F, 5.0F, 5.0F, 6.0F, 5.0F, 5.0F, 5.0F,
        5.0F;
    CHECK(points == expected);
}

TEST_CASE("steps that end a half do not ask for a next direction") {
    const std::vector<float> fod = cube();
    const std::array<std::tuple<Index, bool, int>, 4> cases{
        {{100, true, 4 + 5}, {100, false, 4}, {4, true, 3}, {7, true, 4 + 2}}};
    for (const auto& [max_steps, bidirectional, n_calls] : cases) {
        CAPTURE(max_steps);
        const Counting counting;
        run(setup(fod, nullptr, max_steps, bidirectional), counting, seed_at(centre(), along_x()));
        CHECK(counting.n_calls() == n_calls);
    }
}

TEST_CASE("a seed without a direction starts along the direction found") {
    const std::vector<float> fod = cube();
    const auto [track, points] = run(setup(fod, nullptr, 3, false), algorithm(TestStep::Straight),
                                     seed_at(centre(), std::nullopt, 4));
    const Vector3 expected = Rng::streamline(0, 4).search().unit_vector();
    CHECK(track.direction == expected);
    CHECK(track.n_points == 4);
}

TEST_CASE("a seed where the search fails does not start") {
    const std::vector<float> fod = cube();
    const TrackingSetup tracking = setup(fod, nullptr, 100, true);
    const TestAlgorithm refusing(TestStep::Straight, StopReason::LowFod, 4.0);
    const auto [track, points] = run(tracking, refusing, seed_at(centre(), std::nullopt));
    CHECK(track.n_points == 0);
    CHECK(points.array().isNaN().all());
    // A given direction is followed even where the search would refuse.
    const auto [given_track, given_points] = run(tracking, refusing, seed_at(centre(), along_x()));
    CHECK(given_track.n_points > 0);
}

TEST_CASE("steps draw the same numbers whether the direction was given or found") {
    const std::vector<float> fod = cube();
    const TrackingSetup tracking = setup(fod, nullptr, 8, true);
    const TestAlgorithm wander = algorithm(TestStep::Wander);
    const Seed searched = seed_at(centre(), std::nullopt, 2);
    const auto [found, found_points] = run(tracking, wander, searched);
    const Seed given = seed_at(centre(), found.direction, 2);
    const auto [again, given_points] = run(tracking, wander, given);
    CHECK(found.n_points == again.n_points);
    CHECK(found.ends == again.ends);
    const Index n_points = found.n_points;
    CHECK(found_points.topRows(n_points) == given_points.topRows(n_points));
    // The walk does turn: a straight line would hide a shifted counter.
    const auto first_step = (found_points.row(1) - found_points.row(0)).eval();
    const auto second_step = (found_points.row(2) - found_points.row(1)).eval();
    CHECK(first_step != second_step);
}

TEST_CASE("each seed fills its own row and later rows are untouched") {
    const auto [tracks, positions] = batch(4);
    CHECK(tracks.size() == 20);
    const std::vector<float> fod = cube();
    const auto [alone, points] = run(setup(fod, nullptr, 8, true), algorithm(TestStep::Wander),
                                     seed_at(Vector3(2.0 + 0.3 * 7.0, 5.0, 5.0), std::nullopt, 7));
    CHECK(tracks[7].n_points == alone.n_points);
    const Index n_values = alone.n_points * 3;
    const Eigen::Map<const Eigen::RowVectorXf> written(points.data(), n_values);
    CHECK(positions.row(7).head(n_values) == written);
    CHECK(all_nan(positions.bottomRows(3)));
}

TEST_CASE("a batch does not depend on the number of threads") {
    const auto [one, one_positions] = batch(1);
    const auto [many, many_positions] = batch(4);
    bool same_tracks = one.size() == many.size();
    for (std::size_t item = 0; same_tracks && item < one.size(); ++item) {
        same_tracks =
            one[item].n_points == many[item].n_points && one[item].ends == many[item].ends;
    }
    CHECK(same_tracks);
    // NaN != NaN, so compare the bits.
    const auto n_bytes = static_cast<std::size_t>(one_positions.size()) * sizeof(float);
    CHECK(std::memcmp(one_positions.data(), many_positions.data(), n_bytes) == 0);
}
