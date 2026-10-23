#include "tracking/testing.hpp"

#include "common/maths.hpp"

namespace larmor::tracking {

TestAlgorithm::TestAlgorithm(TestStep rule, StopReason refuse_with, double search_refuses_past_x)
    : step_rule(rule), refusal(refuse_with), refuses_past_x(search_refuses_past_x) {}

std::optional<Vector3> TestAlgorithm::first_direction(const Field& /*fod*/, const Vector3& position,
                                                      Rng& rng) const {
    const Vector3 direction = rng.unit_vector();
    if (position.x() <= refuses_past_x) {
        return direction;
    }
    return std::nullopt;
}

StepResult TestAlgorithm::next_direction(const Field& /*fod*/, Vector3 /*position*/,
                                         Vector3 previous_direction, Rng& rng,
                                         SamplingStats& /*stats*/) const {
    switch (step_rule) {
    case TestStep::Straight:
        return previous_direction;
    case TestStep::Refuse:
        return refusal;
    case TestStep::QuarterTurn:
        return Vector3(-previous_direction.y(), previous_direction.x(), 0.0);
    case TestStep::Wander: {
        // Three statements, so the draws go to x, y and z in that order.
        const double x = previous_direction.x() + 0.5 * (rng.uniform() - 0.5);
        const double y = previous_direction.y() + 0.5 * (rng.uniform() - 0.5);
        const double z = previous_direction.z() + 0.5 * (rng.uniform() - 0.5);
        const double norm = norm3(Vector3(x, y, z));
        return Vector3(x / norm, y / norm, z / norm);
    }
    }
    return refusal;
}

} // namespace larmor::tracking
