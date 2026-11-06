#include "simulation/instant/instant.hpp"

#include <cstddef>
#include <map>
#include <utility>
#include <vector>

namespace larmor::simulation {

Matrix relaxation_weights(const VectorView& proton_density, const VectorView& t1_ms,
                          const VectorView& decay_ms, double tr_ms, const VectorView& te_ms) {
    Matrix weights(proton_density.size(), te_ms.size());
    for (Index row = 0; row < weights.rows(); ++row) {
        for (Index echo = 0; echo < weights.cols(); ++echo) {
            weights(row, echo) = relaxation_weight(proton_density(row), t1_ms(row), decay_ms(row),
                                                   tr_ms, te_ms(echo));
        }
    }
    return weights;
}

InstantModel make_instant_model(const TissueCompartments& tissues, const Acquisition& acquisition) {
    InstantModel model;
    model.acquisition = acquisition;
    model.n_echoes = tissues.relaxation.cols();
    const Index n_volumes = acquisition.bvals.size();
    model.unoriented = Matrix::Zero(tissues.n_tissues, model.n_echoes * n_volumes);
    model.oriented.resize(static_cast<std::size_t>(tissues.n_tissues));

    for (std::size_t index = 0; index < tissues.compartments.size(); ++index) {
        const Compartment& compartment = tissues.compartments[index];
        const auto row = static_cast<Index>(index);
        const Index tissue = tissues.tissue(row);
        const Vector coefficient = tissues.fraction(row) * tissues.relaxation.row(row).transpose();
        if (is_oriented(compartment.model)) {
            // Volumes of one b-value share a profile, which is slow to build for Watson sticks.
            std::map<double, AngularProfile> by_bval;
            std::vector<AngularProfile> profiles;
            profiles.reserve(static_cast<std::size_t>(n_volumes));
            for (Index volume = 0; volume < n_volumes; ++volume) {
                const double bval = acquisition.bvals(volume);
                auto found = by_bval.find(bval);
                if (found == by_bval.end()) {
                    found =
                        by_bval
                            .emplace(bval, angular_profile(compartment, bval, acquisition.timing))
                            .first;
                }
                profiles.push_back(found->second);
            }
            model.oriented[static_cast<std::size_t>(tissue)].push_back(
                {.compartment = compartment,
                 .coefficient = coefficient,
                 .profiles = std::move(profiles)});
            continue;
        }
        for (Index volume = 0; volume < n_volumes; ++volume) {
            const double signal =
                attenuation(compartment, acquisition.bvals(volume), 0.0, acquisition.timing);
            for (Index echo = 0; echo < model.n_echoes; ++echo) {
                model.unoriented(tissue, echo * n_volumes + volume) += coefficient(echo) * signal;
            }
        }
    }
    return model;
}

void simulate_voxels(
    const InstantModel& model, const VoxelContents& contents, Index begin, Index end,
    Eigen::Ref<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> out) {
    const Acquisition& acquisition = model.acquisition;
    const Index n_volumes = acquisition.bvals.size();
    Vector signal(model.n_echoes * n_volumes);
    for (Index voxel = begin; voxel < end; ++voxel) {
        signal.noalias() =
            model.unoriented.transpose() * contents.tissue_fractions.row(voxel).transpose();
        for (Index fibre = contents.offsets(voxel); fibre < contents.offsets(voxel + 1); ++fibre) {
            const auto& terms =
                model.oriented[static_cast<std::size_t>(contents.fibre_tissue(fibre))];
            const double weight = contents.fibre_weight(fibre);
            if (terms.empty() || weight == 0.0) {
                continue;
            }
            const Vector3 direction = contents.fibre_direction.row(fibre).transpose();
            for (Index volume = 0; volume < n_volumes; ++volume) {
                const double cosine = acquisition.bvecs.row(volume).dot(direction.transpose());
                for (const OrientedTerm& term : terms) {
                    const double attenuated =
                        weight * term.profiles[static_cast<std::size_t>(volume)](cosine);
                    for (Index echo = 0; echo < model.n_echoes; ++echo) {
                        signal(echo * n_volumes + volume) += term.coefficient(echo) * attenuated;
                    }
                }
            }
        }
        out.row(voxel) = signal.cast<float>().transpose();
    }
}

} // namespace larmor::simulation
