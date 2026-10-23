#include <array>
#include <cstdint>
#include <limits>
#include <optional>

#include <doctest/doctest.h>

#include "tracking/algorithm.hpp"

namespace {

using larmor::tracking::SamplingStats;
using larmor::tracking::StopReason;

} // namespace

// The codes of Python's EndStatus.
TEST_CASE("end codes match EndStatus") {
    CHECK(larmor::tracking::end_code(std::nullopt) == 0);
    const std::array<StopReason, 6> stops{StopReason::MaxLength, StopReason::LeftVolume,
                                          StopReason::LeftMask,  StopReason::LowFod,
                                          StopReason::SharpTurn, StopReason::NoDirection};
    for (std::size_t code = 0; code < stops.size(); ++code) {
        CHECK(larmor::tracking::end_code(stops.at(code)) == code + 1);
    }
}

TEST_CASE("stats add sums the counts and keeps the larger maximum") {
    SamplingStats total{
        .sampled_steps = 1, .accepted_index_sum = 2, .truncations = 3, .max_truncation = 0.5};
    total.add(SamplingStats{
        .sampled_steps = 10, .accepted_index_sum = 20, .truncations = 30, .max_truncation = 0.25});
    CHECK(total == SamplingStats{.sampled_steps = 11,
                                 .accepted_index_sum = 22,
                                 .truncations = 33,
                                 .max_truncation = 0.5});
    total.add(SamplingStats{.max_truncation = std::numeric_limits<double>::quiet_NaN()});
    CHECK(total.max_truncation == 0.5);
}
