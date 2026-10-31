#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <utility>
#include <vector>

#include <doctest/doctest.h>

#include "common/types.hpp"
#include "dwi/noddi.hpp"
#include "simulation/compartments/compartments.hpp"

namespace {

using larmor::Index;
using larmor::Matrix;
using larmor::Vector;
using larmor::Vector3;

constexpr double D_PAR = 1.7e-3;

// Returns n unit directions spread over the half sphere, by the golden spiral.
Matrix spiral_directions(Index count) {
    Matrix directions(count, 3);
    const double golden = std::numbers::pi * (3.0 - std::sqrt(5.0));
    for (Index index = 0; index < count; ++index) {
        const double z = 1.0 - (static_cast<double>(index) + 0.5) / static_cast<double>(count);
        const double radius = std::sqrt(1.0 - z * z);
        const double angle = golden * static_cast<double>(index);
        directions.row(index) << radius * std::cos(angle), radius * std::sin(angle), z;
    }
    return directions;
}

// A two-shell table: 6 b0s, 30 directions at b = 1000 and 60 at b = 2000 s/mm^2.
larmor::dwi::NoddiProtocol two_shells(const larmor::dwi::NoddiSettings& settings) {
    const Index n_volumes = 96;
    Vector bvals = Vector::Zero(n_volumes);
    Matrix bvecs = Matrix::Zero(n_volumes, 3);
    std::vector<bool> is_b0(static_cast<std::size_t>(n_volumes), false);
    for (std::size_t volume = 0; volume < 6; ++volume) {
        is_b0.at(volume) = true;
    }
    bvals.segment(6, 30).setConstant(1000.0);
    bvals.segment(36, 60).setConstant(2000.0);
    bvecs.middleRows(6, 30) = spiral_directions(30);
    bvecs.middleRows(36, 60) = spiral_directions(60);
    return larmor::dwi::make_noddi_protocol(bvals, bvecs, is_b0, settings);
}

larmor::dwi::NoddiSettings fit_settings() {
    larmor::dwi::NoddiSettings settings;
    for (int atom = 0; atom < 12; ++atom) {
        settings.v_ic_grid.push_back(0.1 + 0.9 * atom / 11.0);
        settings.kappa_grid.push_back(20.0 * atom / 11.0);
    }
    settings.ridge = std::sqrt(0.001);
    settings.l1 = 0.5;
    return settings;
}

} // namespace

TEST_CASE("noddi: the toolbox's Watson stick matches the exact integral") {
    // The series stops at degree 12, so the gap grows with kappa and b; these bounds sit above
    // the measured ones (3e-4 at kappa 64 and b = 3000).
    const auto settings = fit_settings();
    const Matrix directions = spiral_directions(40);
    for (const double bval : {1000.0, 3000.0}) {
        Vector bvals = Vector::Constant(40, bval);
        const auto protocol = larmor::dwi::make_noddi_protocol(
            bvals, directions, std::vector<bool>(40, false), settings);
        for (const double kappa : {0.0, 0.05, 0.1001, 0.3, 2.0, 16.0, 29.9, 30.1, 64.0}) {
            const Vector series =
                larmor::dwi::noddi_signal(protocol, settings, 1.0, kappa, 0.0, Vector3::UnitZ());
            for (Index volume = 0; volume < 40; ++volume) {
                const double exact =
                    larmor::simulation::watson_stick(bval, D_PAR, kappa, directions(volume, 2));
                CAPTURE(bval);
                CAPTURE(kappa);
                CHECK(std::abs(series(volume) - exact) < (kappa > 16.0 ? 1e-3 : 1e-4));
            }
        }
    }
}

TEST_CASE("noddi: the whole signal matches Zhang et al.'s model") {
    // Eq. 1 with the exact Watson stick, the zeppelin of Eqs. 4 to 6 and tau1 of Eq. 7, the last
    // integrated by the midpoint rule.
    const auto settings = fit_settings();
    const auto protocol = two_shells(settings);
    const Vector3 mu = Vector3(0.3, -0.5, 0.8).normalized();
    for (const double kappa : {0.5, 3.0, 12.0}) {
        double weight_sum = 0.0;
        double moment = 0.0;
        const int n_steps = 100000;
        for (int step = 0; step < n_steps; ++step) {
            const double cosine = (step + 0.5) / n_steps;
            const double weight = std::exp(kappa * cosine * cosine);
            weight_sum += weight;
            moment += weight * cosine * cosine;
        }
        const double tau = moment / weight_sum;
        const double v_ic = 0.6;
        const double v_iso = 0.15;
        const double along = D_PAR - D_PAR * v_ic * (1.0 - tau);
        const double across = D_PAR - D_PAR * v_ic * (1.0 + tau) / 2.0;
        const Vector signal = larmor::dwi::noddi_signal(protocol, settings, v_ic, kappa, v_iso, mu);
        for (Index volume = 0; volume < protocol.bvals.size(); ++volume) {
            const double bval = protocol.bvals(volume);
            const double cosine = protocol.bvecs.row(volume).dot(mu);
            const double sticks = larmor::simulation::watson_stick(bval, D_PAR, kappa, cosine);
            const double zeppelin = std::exp(-bval * (across + (along - across) * cosine * cosine));
            const double expected = (1.0 - v_iso) * (v_ic * sticks + (1.0 - v_ic) * zeppelin) +
                                    v_iso * std::exp(-bval * 3.0e-3);
            CAPTURE(kappa);
            CHECK(std::abs(signal(volume) - expected) < 1e-4);
        }
    }
}

TEST_CASE("noddi: the Watson coefficients join up where the toolbox changes formula") {
    // Just above kappa = 0.1 the closed form of degrees 10 and 12 loses digits to cancellation, as
    // in the toolbox; the stick test above bounds what that does to the signal.
    for (const auto& [edge, n_terms] :
         {std::pair{0.1, std::size_t{5}}, std::pair{30.0, std::size_t{7}}}) {
        const auto below = larmor::dwi::watson_sh_coefficients(edge * (1.0 - 1e-9));
        const auto above = larmor::dwi::watson_sh_coefficients(edge * (1.0 + 1e-9));
        for (std::size_t term = 0; term < n_terms; ++term) {
            CAPTURE(edge);
            CAPTURE(term);
            // The fits above kappa = 30 are the toolbox's, good to about 1e-3 of the value.
            CHECK(std::abs(below.at(term) - above.at(term)) <=
                  2e-3 * std::abs(below.at(term)) + 1e-7);
        }
    }
}

TEST_CASE("noddi: the averaged zeppelin follows the tortuosity model") {
    // Zhang et al. 2012, Eqs. 5 and 6: d_par' = d - d v (1 - tau1) and d_perp' = d - d v (1 +
    // tau1) / 2, with tau1 the mean squared cosine of the Watson density (Eq. 7), integrated here
    // by the midpoint rule.
    for (const double kappa : {0.0, 1e-6, 0.3, 4.0, 25.0}) {
        double weight_sum = 0.0;
        double moment = 0.0;
        const int n_steps = 200000;
        for (int step = 0; step < n_steps; ++step) {
            const double cosine = (step + 0.5) / n_steps;
            const double weight = std::exp(kappa * cosine * cosine);
            weight_sum += weight;
            moment += weight * cosine * cosine;
        }
        const double tau = moment / weight_sum;
        for (const double v_ic : {0.2, 0.7}) {
            const auto [along, across] =
                larmor::dwi::watson_hindered_diffusivities(D_PAR, D_PAR * (1.0 - v_ic), kappa);
            CAPTURE(kappa);
            CHECK(along == doctest::Approx(D_PAR - D_PAR * v_ic * (1.0 - tau)).epsilon(1e-8));
            CHECK(across ==
                  doctest::Approx(D_PAR - D_PAR * v_ic * (1.0 + tau) / 2.0).epsilon(1e-8));
        }
    }
}

TEST_CASE("noddi: a noise-free voxel's parameters and direction are recovered") {
    const auto settings = fit_settings();
    const auto protocol = two_shells(settings);
    const Vector3 mu = Vector3(0.3, -0.5, 0.8).normalized();
    // The start's direction is off by a few degrees, as a DTI direction can be.
    const Vector3 start = Vector3(0.35, -0.45, 0.82).normalized();
    for (const auto& [v_ic, kappa, v_iso] :
         std::vector<std::array<double, 3>>{{0.5, 3.0, 0.2}, {0.8, 12.0, 0.05}, {0.3, 0.8, 0.4}}) {
        const Vector signal =
            900.0 * larmor::dwi::noddi_signal(protocol, settings, v_ic, kappa, v_iso, mu);
        const auto fit = larmor::dwi::fit_noddi_voxel(protocol, settings, signal, start);
        CAPTURE(v_ic);
        CHECK(fit.fitted);
        CHECK(fit.v_ic == doctest::Approx(v_ic).epsilon(1e-3));
        CHECK(fit.kappa == doctest::Approx(kappa).epsilon(1e-2));
        CHECK(fit.v_iso == doctest::Approx(v_iso).epsilon(1e-3));
        CHECK(std::abs(fit.mu.dot(mu)) > std::cos(0.1 * std::numbers::pi / 180.0));
    }
}

TEST_CASE("noddi: a voxel without a positive b0 is not fitted") {
    const auto settings = fit_settings();
    const auto protocol = two_shells(settings);
    Vector signal =
        900.0 * larmor::dwi::noddi_signal(protocol, settings, 0.5, 3.0, 0.2, Vector3::UnitZ());
    signal.head(6).setConstant(-1.0);
    const auto fit = larmor::dwi::fit_noddi_voxel(protocol, settings, signal, Vector3::UnitZ());
    CHECK_FALSE(fit.fitted);
    CHECK(fit.v_ic == 0.0);
    CHECK(fit.v_iso == 0.0);
}

TEST_CASE("noddi: a starting direction that is zero or not finite is not fitted") {
    const auto settings = fit_settings();
    const auto protocol = two_shells(settings);
    const Vector signal =
        900.0 * larmor::dwi::noddi_signal(protocol, settings, 0.5, 3.0, 0.2, Vector3::UnitZ());
    for (const Vector3& start : {Vector3::Zero().eval(), Vector3(std::nan(""), 0.0, 1.0),
                                 Vector3(std::numeric_limits<double>::infinity(), 0.0, 0.0)}) {
        CHECK_FALSE(larmor::dwi::fit_noddi_voxel(protocol, settings, signal, start).fitted);
    }
}
