#include "tracking/algorithm.hpp"

#include <cmath>

namespace larmor::tracking {

std::uint8_t end_code(std::optional<StopReason> end) {
    return end.has_value() ? static_cast<std::uint8_t>(*end) : std::uint8_t{0};
}

void SamplingStats::add(const SamplingStats& other) {
    sampled_steps += other.sampled_steps;
    accepted_index_sum += other.accepted_index_sum;
    truncations += other.truncations;
    max_truncation = std::fmax(max_truncation, other.max_truncation);
}

} // namespace larmor::tracking
