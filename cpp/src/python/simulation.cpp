// Bindings of the simulation, with the entry points of the engines still to come.

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <Python.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>

#include "common/parallel.hpp"
#include "common/types.hpp"
#include "python/arrays.hpp"
#include "python/bindings.hpp"
#include "simulation/compartments/compartments.hpp"
#include "simulation/instant/instant.hpp"

namespace nb = nanobind;

namespace larmor::python {

namespace {

// Voxels per block of the parallel loop.
constexpr Index SIMULATION_BLOCK_SIZE = 256;

// A float32 matrix, stored row by row.
using FloatRows = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Raises ValueError unless `found` equals `expected`.
void require_length(Index found, Index expected, const std::string& name) {
    if (found != expected) {
        throw nb::value_error((name + " must have " + std::to_string(expected) + " entries, got " +
                               std::to_string(found))
                                  .c_str());
    }
}

// Reads the compartments from their model codes (C,) and parameters (C, 4).
std::vector<simulation::Compartment> read_compartments(const InArray<std::int64_t, 1>& models_array,
                                                       const InArray<double, 2>& parameters_array) {
    const auto models = flat_view(models_array);
    const auto parameters = matrix_view(parameters_array);
    require_columns(parameters_array, 4);
    require_length(parameters.rows(), models.size(), "parameters");
    std::vector<simulation::Compartment> compartments;
    compartments.reserve(static_cast<std::size_t>(models.size()));
    for (Index row = 0; row < models.size(); ++row) {
        if (models(row) < 0 || models(row) >= simulation::N_MODELS) {
            throw nb::value_error(
                ("unknown compartment model " + std::to_string(models(row))).c_str());
        }
        compartments.push_back({.model = static_cast<simulation::Model>(models(row)),
                                .diffusivity = parameters(row, 0),
                                .perpendicular_diffusivity = parameters(row, 1),
                                .radius_um = parameters(row, 2),
                                .kappa = parameters(row, 3)});
    }
    return compartments;
}

// Returns the attenuation of each compartment for each gradient.
NumpyArray<double, 2> compartment_attenuation(const InArray<std::int64_t, 1>& models_array,
                                              const InArray<double, 2>& parameters_array,
                                              const InArray<double, 1>& bvals_array,
                                              const InArray<double, 1>& cosines_array,
                                              double small_delta_ms, double big_delta_ms) {
    const auto compartments = read_compartments(models_array, parameters_array);
    const auto bvals = flat_view(bvals_array);
    const auto cosines = flat_view(cosines_array);
    require_length(cosines.size(), bvals.size(), "cosines");
    const simulation::PulseTiming timing{.small_delta_ms = small_delta_ms,
                                         .big_delta_ms = big_delta_ms};
    Matrix result(static_cast<Index>(compartments.size()), bvals.size());
    {
        const nb::gil_scoped_release release;
        for (Index row = 0; row < result.rows(); ++row) {
            for (Index column = 0; column < result.cols(); ++column) {
                result(row, column) =
                    simulation::attenuation(compartments[static_cast<std::size_t>(row)],
                                            bvals(column), cosines(column), timing);
            }
        }
    }
    return to_numpy(std::move(result));
}

// Returns a fixed array of roots as a numpy array.
template <std::size_t n_roots>
NumpyArray<double, 1> roots_array(const std::array<double, n_roots>& roots) {
    Vector result(static_cast<Index>(roots.size()));
    for (std::size_t index = 0; index < roots.size(); ++index) {
        result(static_cast<Index>(index)) = roots.at(index);
    }
    return to_numpy(std::move(result));
}

// Returns the roots the sphere model sums over.
NumpyArray<double, 1> sphere_roots() {
    return roots_array(simulation::sphere_roots());
}

// Returns the roots the cylinder model sums over.
NumpyArray<double, 1> cylinder_roots() {
    return roots_array(simulation::cylinder_roots());
}

// Returns the relaxation weight of each compartment at each echo time.
NumpyArray<double, 2> relaxation_weights(const InArray<double, 1>& proton_density,
                                         const InArray<double, 1>& t1_ms,
                                         const InArray<double, 1>& decay_ms, double tr_ms,
                                         const InArray<double, 1>& te_ms) {
    const auto density = flat_view(proton_density);
    const auto t1 = flat_view(t1_ms);
    const auto decay = flat_view(decay_ms);
    const auto te = flat_view(te_ms);
    require_length(t1.size(), density.size(), "t1_ms");
    require_length(decay.size(), density.size(), "decay_ms");
    return to_numpy(simulation::relaxation_weights(density, t1, decay, tr_ms, te));
}

// Writes the noise-free signal of each voxel into out_array.
void simulate_instant(
    const InArray<std::int64_t, 1>& models_array, const InArray<double, 2>& parameters_array,
    const InArray<std::int64_t, 1>& tissue_array, const InArray<double, 1>& fraction_array,
    const InArray<double, 2>& relaxation_array, const InArray<double, 1>& bvals_array,
    const InArray<double, 2>& bvecs_array, double small_delta_ms, double big_delta_ms,
    const InArray<double, 2>& tissue_fractions_array, const InArray<std::int64_t, 1>& offsets_array,
    const InArray<std::int64_t, 1>& fibre_tissue_array,
    const InArray<double, 1>& fibre_weight_array, const InArray<double, 2>& fibre_direction_array,
    const InOutArray<float, 3>& out_array, int n_threads) {
    simulation::TissueCompartments tissues;
    tissues.compartments = read_compartments(models_array, parameters_array);
    const auto n_compartments = static_cast<Index>(tissues.compartments.size());
    const auto tissue = flat_view(tissue_array);
    const auto fraction = flat_view(fraction_array);
    const auto relaxation = matrix_view(relaxation_array);
    require_length(tissue.size(), n_compartments, "tissue");
    require_length(fraction.size(), n_compartments, "fraction");
    require_length(relaxation.rows(), n_compartments, "relaxation");

    const auto tissue_fractions = matrix_view(tissue_fractions_array);
    const Index n_tissues = tissue_fractions.cols();
    if (n_compartments > 0 && (tissue.minCoeff() < 0 || tissue.maxCoeff() >= n_tissues)) {
        throw nb::value_error(
            ("compartment tissues must be in [0, " + std::to_string(n_tissues) + ")").c_str());
    }
    tissues.tissue = tissue.cast<Index>();
    tissues.fraction = fraction;
    tissues.relaxation = relaxation;
    tissues.n_tissues = n_tissues;

    simulation::Acquisition acquisition;
    acquisition.bvals = flat_view(bvals_array);
    acquisition.bvecs = matrix_view(bvecs_array);
    require_columns(bvecs_array, 3);
    require_length(acquisition.bvecs.rows(), acquisition.bvals.size(), "bvecs");
    acquisition.timing = {.small_delta_ms = small_delta_ms, .big_delta_ms = big_delta_ms};

    const Index n_voxels = tissue_fractions.rows();
    const auto offsets = flat_view(offsets_array);
    const auto fibre_tissue = flat_view(fibre_tissue_array);
    const auto fibre_weight = flat_view(fibre_weight_array);
    const auto fibre_direction = matrix_view(fibre_direction_array);
    require_columns(fibre_direction_array, 3);
    const Index n_fibres = fibre_tissue.size();
    require_length(offsets.size(), n_voxels + 1, "offsets");
    require_length(fibre_weight.size(), n_fibres, "fibre_weight");
    require_length(fibre_direction.rows(), n_fibres, "fibre_direction");
    if (offsets(0) != 0 || offsets(n_voxels) != n_fibres ||
        (n_voxels > 0 && (offsets.tail(n_voxels).array() < offsets.head(n_voxels).array()).any())) {
        throw nb::value_error("offsets must start at 0, never decrease and end at the fibre count");
    }
    if (n_fibres > 0 && (fibre_tissue.minCoeff() < 0 || fibre_tissue.maxCoeff() >= n_tissues)) {
        throw nb::value_error(
            ("fibre tissues must be in [0, " + std::to_string(n_tissues) + ")").c_str());
    }

    require_contiguous(out_array, "out");
    const Index n_echoes = relaxation.cols();
    const Index n_volumes = acquisition.bvals.size();
    if (extent(out_array, 0) != n_voxels || extent(out_array, 1) != n_echoes ||
        extent(out_array, 2) != n_volumes) {
        throw nb::value_error(("out must have shape [" + std::to_string(n_voxels) + ", " +
                               std::to_string(n_echoes) + ", " + std::to_string(n_volumes) +
                               "], got " + shape_string(out_array))
                                  .c_str());
    }
    require_threads(n_threads);
    Eigen::Map<FloatRows> out(out_array.data(), n_voxels, n_echoes * n_volumes);

    const Eigen::VectorX<Index> offsets_index = offsets.cast<Index>();
    const Eigen::VectorX<Index> fibre_tissue_index = fibre_tissue.cast<Index>();
    {
        const nb::gil_scoped_release release;
        const simulation::InstantModel model = simulation::make_instant_model(tissues, acquisition);
        const simulation::VoxelContents contents{.tissue_fractions = tissue_fractions,
                                                 .offsets = offsets_index,
                                                 .fibre_tissue = fibre_tissue_index,
                                                 .fibre_weight = fibre_weight,
                                                 .fibre_direction = fibre_direction};
        parallel_for_blocks(n_voxels, SIMULATION_BLOCK_SIZE, n_threads,
                            [&](Index begin, Index end) {
                                simulation::simulate_voxels(model, contents, begin, end, out);
                            });
    }
}

// Returns the code of each compartment model by its name.
std::map<std::string, int> compartment_model_codes() {
    return {{"stick", static_cast<int>(simulation::Model::Stick)},
            {"zeppelin", static_cast<int>(simulation::Model::Zeppelin)},
            {"ball", static_cast<int>(simulation::Model::Ball)},
            {"powder_stick", static_cast<int>(simulation::Model::PowderStick)},
            {"sphere", static_cast<int>(simulation::Model::Sphere)},
            {"watson_stick", static_cast<int>(simulation::Model::WatsonStick)},
            {"cylinder", static_cast<int>(simulation::Model::Cylinder)},
            {"dot", static_cast<int>(simulation::Model::Dot)}};
}

} // namespace

void register_simulation(nb::module_& module) {
    module.def("compartment_model_codes", &compartment_model_codes,
               "Returns the code of each compartment model in the engine, by the name\n"
               "larmor.simulation.CompartmentModel gives it.\n");
    module.def("compartment_attenuation", &compartment_attenuation, nb::arg("models").noconvert(),
               nb::arg("parameters").noconvert(), nb::arg("bvals").noconvert(),
               nb::arg("cosines").noconvert(), nb::arg("small_delta_ms"), nb::arg("big_delta_ms"),
               "Returns the attenuation of each compartment (rows) for each gradient (columns).\n"
               "\n"
               "models holds the codes 0 stick, 1 zeppelin, 2 ball, 3 powder-averaged stick,\n"
               "4 sphere, 5 Watson-dispersed sticks, 6 cylinder and 7 dot. parameters (C, 4)\n"
               "holds the diffusivity and perpendicular diffusivity in mm^2/s, the radius in um\n"
               "and the Watson kappa. The b-values are in s/mm^2, and cosines holds g . n per\n"
               "gradient.\n");
    module.def("sphere_roots", &sphere_roots,
               "Returns the positive roots x_m of j1'(x) = 0 the sphere model sums over.\n");
    module.def("cylinder_roots", &cylinder_roots,
               "Returns the positive roots x_m of J1'(x) = 0 the cylinder model sums over.\n");
    module.def("relaxation_weights", &relaxation_weights, nb::arg("proton_density").noconvert(),
               nb::arg("t1_ms").noconvert(), nb::arg("decay_ms").noconvert(), nb::arg("tr_ms"),
               nb::arg("te_ms").noconvert(),
               "Returns the relaxation weight of each compartment (rows) at each echo time\n"
               "(columns).\n"
               "\n"
               "The weight is PD (1 - exp(-TR / T1)) exp(-TE / T_decay), with times in ms.\n");
    module.def("simulate_instant", &simulate_instant, nb::arg("models").noconvert(),
               nb::arg("parameters").noconvert(), nb::arg("tissue").noconvert(),
               nb::arg("fraction").noconvert(), nb::arg("relaxation").noconvert(),
               nb::arg("bvals").noconvert(), nb::arg("bvecs").noconvert(),
               nb::arg("small_delta_ms"), nb::arg("big_delta_ms"),
               nb::arg("tissue_fractions").noconvert(), nb::arg("offsets").noconvert(),
               nb::arg("fibre_tissue").noconvert(), nb::arg("fibre_weight").noconvert(),
               nb::arg("fibre_direction").noconvert(), nb::arg("out").noconvert(),
               nb::arg("n_threads"),
               "Writes the noise-free signal of each voxel into out, (V, E, N) float32.\n"
               "\n"
               "Raises:\n"
               "    ValueError: If the shapes do not fit together, an index is out of range,\n"
               "        or n_threads is 0.\n");
}

} // namespace larmor::python
