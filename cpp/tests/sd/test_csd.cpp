#include <doctest/doctest.h>

#include "common/types.hpp"
#include "sd/csd.hpp"

namespace {

using larmor::Index;
using larmor::Matrix;
using larmor::Vector;

// Weight of the toy constraint rows.
constexpr double WEIGHT = 10.0;

// Three coefficients measured directly (M = I, no norm penalty), with the second and third
// constrained, so a signal with a negative second coefficient has a known solution.
larmor::sd::CsdProblem toy_problem(Index max_iterations) {
    Matrix constraint = Matrix::Zero(2, 3);
    constraint(0, 1) = WEIGHT;
    constraint(1, 2) = WEIGHT;
    return larmor::sd::make_csd_problem(Matrix::Identity(3, 3), Matrix::Identity(3, 3), constraint,
                                        0.0, 0.0, max_iterations);
}

Vector toy_signal() {
    Vector signal(3);
    signal << 1.0, -0.5, 0.2;
    return signal;
}

} // namespace

TEST_CASE("csd: no iteration gives the linear start") {
    const larmor::sd::CsdResult result = larmor::sd::solve_csd(toy_problem(0), toy_signal());
    CHECK(result.converged);
    CHECK(result.fod.isApprox(toy_signal()));
}

TEST_CASE("csd: the negative coefficient is penalised and the set settles") {
    const larmor::sd::CsdResult result = larmor::sd::solve_csd(toy_problem(50), toy_signal());
    // Only the second direction is constrained: (1 + w^2) f_1 = s_1.
    CHECK(result.converged);
    CHECK(result.fod(0) == doctest::Approx(1.0));
    CHECK(result.fod(1) == doctest::Approx(-0.5 / (1.0 + WEIGHT * WEIGHT)));
    CHECK(result.fod(2) == doctest::Approx(0.2));
}

TEST_CASE("csd: running out of iterations is reported") {
    // One solve leaves no iteration to see that the set repeats.
    CHECK_FALSE(larmor::sd::solve_csd(toy_problem(1), toy_signal()).converged);
    CHECK(larmor::sd::solve_csd(toy_problem(2), toy_signal()).converged);
}
