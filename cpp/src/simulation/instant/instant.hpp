// The instant engine, which sums the closed-form signals of the compartments in each voxel.
#pragma once

#include <vector>

#include <Eigen/Core>

#include "common/types.hpp"
#include "simulation/compartments/compartments.hpp"

namespace larmor::simulation {

/// The tissues' compartments, flattened.
struct TissueCompartments {
    /// Each compartment's model and parameters; (C,).
    std::vector<Compartment> compartments;
    /// The tissue each compartment belongs to, 0 to n_tissues - 1; (C,).
    Eigen::VectorX<Index> tissue;
    /// Each compartment's fraction within its tissue; (C,).
    Vector fraction;
    /// Each compartment's relaxation weight at each echo; (C, E).
    Matrix relaxation;
    /// The number of tissues, T.
    Index n_tissues = 0;
};

/// The acquisition: one b-value and unit direction per volume, and the pulse timings.
struct Acquisition {
    /// b-values in s/mm^2; (N,).
    Vector bvals;
    /// Unit gradient directions; (N, 3).
    Matrix bvecs;
    /// The pulse timings.
    PulseTiming timing;
};

/// One oriented compartment of a tissue, with its fraction times relaxation weight per echo.
struct OrientedTerm {
    /// The compartment's model and parameters.
    Compartment compartment;
    /// Its fraction times its relaxation weight at each echo; (E,).
    Vector coefficient;
    /// The attenuation's dependence on the fibre direction at each volume's b-value; (N,).
    std::vector<AngularProfile> profiles;
};

/// The parts of the signal that every voxel shares, computed once.
struct InstantModel {
    /// The acquisition.
    Acquisition acquisition;
    /// The number of echoes, E.
    Index n_echoes = 0;
    /// The signal of each tissue's unoriented compartments, with echo e and volume n at column
    /// e N + n; (T, E N).
    Matrix unoriented;
    /// The oriented compartments of each tissue; (T,) lists.
    std::vector<std::vector<OrientedTerm>> oriented;
};

/// Returns the relaxation weight of each compartment at each echo time; (C, E).
/// Times are in ms; decay is T2 for a spin echo and T2* for a gradient echo.
Matrix relaxation_weights(const VectorView& proton_density, const VectorView& t1_ms,
                          const VectorView& decay_ms, double tr_ms, const VectorView& te_ms);

/// Computes the parts of the signal that every voxel shares.
InstantModel make_instant_model(const TissueCompartments& tissues, const Acquisition& acquisition);

/// The voxels' contents, for a block of voxels.
struct VoxelContents {
    /// Each voxel's fraction of each tissue, which weights the tissue's unoriented compartments;
    /// (V, T).
    Eigen::Ref<const Matrix> tissue_fractions;
    /// Where each voxel's fibres start: voxel v has fibres offsets(v) to offsets(v + 1) - 1;
    /// (V + 1,).
    Eigen::Ref<const Eigen::VectorX<Index>> offsets;
    /// Each fibre's tissue; (F,).
    Eigen::Ref<const Eigen::VectorX<Index>> fibre_tissue;
    /// Each fibre's share of its voxel, which weights its tissue's oriented compartments; (F,).
    VectorView fibre_weight;
    /// Each fibre's unit direction; (F, 3).
    Eigen::Ref<const Matrix> fibre_direction;
};

/// Computes the signal of voxels begin to end - 1 of contents and writes it to the same rows of
/// out, with echo e and volume n at column e N + n.
void simulate_voxels(
    const InstantModel& model, const VoxelContents& contents, Index begin, Index end,
    Eigen::Ref<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> out);

} // namespace larmor::simulation
