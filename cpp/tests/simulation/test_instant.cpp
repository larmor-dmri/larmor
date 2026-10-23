#include <cmath>

#include <doctest/doctest.h>

#include "common/types.hpp"
#include "simulation/instant/instant.hpp"

namespace {

using larmor::Index;
using larmor::Matrix;
using larmor::Vector;
using larmor::simulation::Model;

using FloatRows = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

} // namespace

TEST_CASE("instant: a voxel sums its tissues' unoriented signal and its fibres' oriented one") {
    // Tissue 0: a stick (0.6) and a ball (0.4); tissue 1: a ball.
    larmor::simulation::TissueCompartments tissues;
    tissues.compartments = {{.model = Model::Stick, .diffusivity = 2e-3},
                            {.model = Model::Ball, .diffusivity = 1e-3},
                            {.model = Model::Ball, .diffusivity = 3e-3}};
    tissues.tissue = Eigen::VectorX<Index>{{0, 0, 1}};
    tissues.fraction = Vector{{0.6, 0.4, 1.0}};
    tissues.relaxation = Matrix{{0.5}, {0.25}, {1.0}};
    tissues.n_tissues = 2;

    larmor::simulation::Acquisition acquisition;
    acquisition.bvals = Vector{{0.0, 1000.0}};
    acquisition.bvecs = Matrix{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    const auto model = larmor::simulation::make_instant_model(tissues, acquisition);

    // One voxel: 0.5 of tissue 0 along x, 0.2 of tissue 1, the rest empty.
    const Matrix fractions{{0.5, 0.2}};
    const Eigen::VectorX<Index> offsets{{0, 1}};
    const Eigen::VectorX<Index> fibre_tissue{{0}};
    const Vector weight{{0.5}};
    const Matrix direction{{1.0, 0.0, 0.0}};
    const larmor::simulation::VoxelContents contents{.tissue_fractions = fractions,
                                                     .offsets = offsets,
                                                     .fibre_tissue = fibre_tissue,
                                                     .fibre_weight = weight,
                                                     .fibre_direction = direction};
    FloatRows out(1, 2);
    larmor::simulation::simulate_voxels(model, contents, 0, 1, out);

    const double expected_b0 = 0.5 * (0.6 * 0.5 + 0.4 * 0.25) + 0.2 * 1.0;
    const double expected_b1000 =
        0.5 * (0.6 * 0.5 * std::exp(-2.0) + 0.4 * 0.25 * std::exp(-1.0)) + 0.2 * std::exp(-3.0);
    CHECK(out(0, 0) == doctest::Approx(expected_b0).epsilon(1e-6));
    CHECK(out(0, 1) == doctest::Approx(expected_b1000).epsilon(1e-6));
}
