// A dummy tracking algorithm with fixed rules, for testing the step loop without an FOD.
#pragma once

#include <cstdint>
#include <optional>

#include "common/types.hpp"
#include "tracking/algorithm.hpp"
#include "tracking/grid.hpp"
#include "tracking/rng.hpp"

namespace larmor::tracking {

/// The rule a TestAlgorithm steps by.
enum class TestStep : std::uint8_t {
    /// Keeps the previous direction.
    Straight,
    /// Refuses every step with a fixed stop reason.
    Refuse,
    /// Turns 90 degrees about z: (x, y, z) becomes (-y, x, 0).
    QuarterTurn,
    /// A random walk: adds 0.5 * (u - 0.5) to each component of the previous direction, for three
    /// uniform draws u in the order x, y, z, and normalises.
    Wander,
};

/// A dummy algorithm that steps by a fixed rule and ignores the FOD.
///
/// The first-direction search draws a random unit vector (two draws) and fails where the seed's x
/// is above search_refuses_past_x, so a test can make seeds start or not by their position.
class TestAlgorithm final : public TrackingAlgorithm {
public:
    /// Makes an algorithm stepping by rule; refuse_with is the stop of TestStep::Refuse and
    /// unused otherwise.
    TestAlgorithm(TestStep rule, StopReason refuse_with, double search_refuses_past_x);

    [[nodiscard]] std::optional<Vector3> first_direction(const Field& fod, const Vector3& position,
                                                         Rng& rng) const override;

    [[nodiscard]] StepResult next_direction(const Field& fod, Vector3 position,
                                            Vector3 previous_direction, Rng& rng,
                                            SamplingStats& stats) const override;

private:
    TestStep step_rule;
    StopReason refusal;
    double refuses_past_x;
};

} // namespace larmor::tracking
