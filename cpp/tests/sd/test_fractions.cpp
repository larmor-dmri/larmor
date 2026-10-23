#include <cmath>
#include <optional>

#include <doctest/doctest.h>

#include "common/types.hpp"
#include "sd/fractions.hpp"
#include "toy_kernel.hpp"

namespace {

using larmor::Index;
using larmor::Vector;
using larmor::sd::Grouping;

void check_close(const Vector& found, const Vector& expected, double tolerance) {
    REQUIRE(found.size() == expected.size());
    for (Index entry = 0; entry < found.size(); ++entry) {
        CHECK(std::fabs(found(entry) - expected(entry)) < tolerance);
    }
}

} // namespace

TEST_CASE("fractions: the median of odd and even lengths") {
    CHECK(larmor::sd::median(Vector{{3.0, 1.0, 2.0}}) == 2.0);
    CHECK(larmor::sd::median(Vector{{4.0, 1.0, 3.0, 2.0}}) == 2.5);
}

TEST_CASE("fractions: a pure isotropic signal goes to its isotropic fraction") {
    const larmor::sd::TissueKernel kernel = larmor::sd::testing::toy_kernel(6);
    const Vector fod = Vector::Constant(6, 1e-6);
    const Vector signal = kernel.isotropic.col(1);
    const Vector fractions =
        larmor::sd::compute_fractions(fod, kernel, signal, Grouping::Lumped, std::nullopt);
    check_close(fractions, Vector{{0.0, 0.0, 1.0}}, 1e-10);
}

TEST_CASE("fractions: the median threshold drops the values below the median") {
    // Sorted, the FOD is 0.001, 0.002, 0.003, 0.004, 0.5, 1.0: the median is 0.0035, so only
    // directions 3, 4 and 5 are left in the anisotropic column.
    const larmor::sd::TissueKernel kernel = larmor::sd::testing::toy_kernel(6);
    const Vector fod{{0.001, 0.002, 0.003, 1.0, 0.5, 0.004}};
    const Vector kept = 1.0 * kernel.anisotropic.col(3) + 0.5 * kernel.anisotropic.col(4) +
                        0.004 * kernel.anisotropic.col(5);
    const Vector signal = 2.0 * kept / kept.maxCoeff();
    const Vector fractions =
        larmor::sd::compute_fractions(fod, kernel, signal, Grouping::Lumped, std::nullopt);
    check_close(fractions, Vector{{2.0, 0.0, 0.0}}, 1e-10);
}
