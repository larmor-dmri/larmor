// Bindings of the spherical-deconvolution code.

#include <cmath>
#include <cstddef>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "common/parallel.hpp"
#include "common/types.hpp"
#include "python/arrays.hpp"
#include "python/bindings.hpp"
#include "sd/csd.hpp"
#include "sd/drl.hpp"
#include "sd/fractions.hpp"
#include "sd/lsi.hpp"
#include "sd/multi_tissue.hpp"
#include "sd/nnls.hpp"
#include "sd/solver.hpp"

namespace nb = nanobind;

namespace larmor::python {

namespace {

// Voxels per block of the parallel loops.
constexpr Index BLOCK_SIZE = 128;

// The (start, stop) columns of each anisotropic block, as Python passes them.
using BlockPairs = std::vector<std::pair<Index, Index>>;

// Copies a 2D array of any strides into a Matrix.
Matrix copy_matrix(const InArray<double, 2>& array) {
    const auto elements = array.view();
    Matrix copy(extent(array, 0), extent(array, 1));
    for (Index row = 0; row < copy.rows(); ++row) {
        for (Index column = 0; column < copy.cols(); ++column) {
            copy(row, column) =
                elements(static_cast<std::size_t>(row), static_cast<std::size_t>(column));
        }
    }
    return copy;
}

// Copies a 1D array of any stride into a Vector.
Vector copy_vector(const InArray<double, 1>& array) {
    const auto elements = array.view();
    Vector copy(extent(array, 0));
    for (Index index = 0; index < copy.size(); ++index) {
        copy(index) = elements(static_cast<std::size_t>(index));
    }
    return copy;
}

// Returns a shape as "(5, 12)", for error messages.
std::string shape_tuple(Index n_rows, Index n_columns) {
    return "(" + std::to_string(n_rows) + ", " + std::to_string(n_columns) + ")";
}

// Returns the anisotropic blocks as "[(0, 3), (3, 6)]", for error messages.
std::string blocks_string(const BlockPairs& blocks) {
    std::string text = "[";
    for (std::size_t position = 0; position < blocks.size(); ++position) {
        if (position > 0) {
            text += ", ";
        }
        text += shape_tuple(blocks[position].first, blocks[position].second);
    }
    return text + "]";
}

// Raises ValueError("<name> must be at least 0") for a negative count.
void require_non_negative(Index count, const std::string& name) {
    if (count < 0) {
        throw nb::value_error((name + " must be at least 0").c_str());
    }
}

// Makes the ridge-penalised NNLS solver of a kernel.
sd::FodSolver l2nnls_solver(const InArray<double, 2>& kernel_array, double ridge,
                            double gradient_tolerance) {
    if (!(ridge >= 0.0)) {
        throw nb::value_error("reg must be at least 0");
    }
    if (!(gradient_tolerance > 0.0)) {
        throw nb::value_error("tol_x must be positive");
    }
    const Matrix kernel = copy_matrix(kernel_array);
    return sd::FodSolver(sd::make_l2nnls_solver(kernel, ridge, gradient_tolerance));
}

// Makes the elastic-net solver of a kernel.
sd::FodSolver elastic_net_solver(const InArray<double, 2>& kernel_array, double penalty,
                                 double l1_ratio, double relative_tolerance, Index max_sweeps) {
    if (!(penalty >= 0.0)) {
        throw nb::value_error("reg must be at least 0");
    }
    if (!(l1_ratio > 0.0) || !(l1_ratio <= 1.0)) {
        throw nb::value_error("alpha must be in (0, 1]");
    }
    if (!(relative_tolerance > 0.0)) {
        throw nb::value_error("tol must be positive");
    }
    if (max_sweeps < 1) {
        throw nb::value_error("max_iter must be at least 1");
    }
    const Matrix kernel = copy_matrix(kernel_array);
    return sd::FodSolver(sd::ElasticNetSolver{
        .design = sd::standardize_columns(kernel),
        .settings = {.penalty = penalty,
                     .l1_ratio = l1_ratio,
                     .relative_tolerance = relative_tolerance,
                     .max_sweeps = max_sweeps},
    });
}

// Makes the damped Richardson-Lucy solver of a kernel.
sd::FodSolver drl_solver(const InArray<double, 2>& transposed_array,
                         const InArray<double, 2>& gram_array, double damping_threshold,
                         Index n_iterations, double initial_amplitude) {
    Matrix transposed = copy_matrix(transposed_array);
    Matrix gram = copy_matrix(gram_array);
    const Index n_directions = transposed.rows();
    if (gram.rows() != n_directions || gram.cols() != n_directions) {
        throw nb::value_error(("shapes do not match: k_t " +
                               shape_tuple(transposed.rows(), transposed.cols()) + ", k_t_k " +
                               shape_tuple(gram.rows(), gram.cols()))
                                  .c_str());
    }
    require_non_negative(n_iterations, "n_iter");
    return sd::FodSolver(sd::DrlSolver{
        .kernel = {.transposed = std::move(transposed), .gram = std::move(gram)},
        .settings = {.damping_threshold = damping_threshold, .n_iterations = n_iterations},
        .initial_amplitude = initial_amplitude,
    });
}

// Fits the FOD of every voxel with one solver.
NumpyArray<double, 2> fit_single_tissue(const InArray<double, 2>& signals_array,
                                        const sd::FodSolver& solver, int n_threads) {
    const auto signals = matrix_view(signals_array); // checks contiguity
    if (signals.cols() != solver.n_measurements()) {
        throw nb::value_error(("signals " + shape_tuple(signals.rows(), signals.cols()) +
                               " do not have the solver's " +
                               std::to_string(solver.n_measurements()) + " measurements")
                                  .c_str());
    }
    require_threads(n_threads);

    Matrix fods(signals.rows(), solver.n_directions());
    {
        const nb::gil_scoped_release release;
        parallel_for_blocks(signals.rows(), BLOCK_SIZE, n_threads, [&](Index begin, Index end) {
            solver.solve_block(signals.middleRows(begin, end - begin),
                               fods.middleRows(begin, end - begin));
        });
    }
    return to_numpy(std::move(fods));
}

// Fits the FOD of every voxel by constrained spherical deconvolution.
std::pair<NumpyArray<double, 2>, NumpyArray<bool, 1>>
fit_csd(const InArray<double, 2>& signals_array, const InArray<double, 2>& initial_array,
        const InArray<double, 2>& forward_array, const InArray<double, 2>& constraint_array,
        double norm_weight, double threshold, Index max_iterations, int n_threads) {
    const auto signals = matrix_view(signals_array); // checks contiguity
    const sd::CsdProblem problem =
        sd::make_csd_problem(copy_matrix(initial_array), copy_matrix(forward_array),
                             copy_matrix(constraint_array), norm_weight, threshold, max_iterations);
    const Index n_measurements = problem.forward.rows();
    const Index n_coefficients = problem.forward.cols();
    if (signals.cols() != n_measurements || problem.initial.cols() != n_measurements ||
        problem.initial.rows() > n_coefficients || problem.constraint.cols() != n_coefficients) {
        throw nb::value_error(
            ("shapes do not match: signals " + shape_tuple(signals.rows(), signals.cols()) +
             ", initial " + shape_tuple(problem.initial.rows(), problem.initial.cols()) +
             ", forward " + shape_tuple(n_measurements, n_coefficients) + ", constraint " +
             shape_tuple(problem.constraint.rows(), problem.constraint.cols()))
                .c_str());
    }
    require_non_negative(max_iterations, "max_iter");
    require_threads(n_threads);

    Matrix fods(signals.rows(), n_coefficients);
    ColumnVector<bool> converged(signals.rows());
    {
        const nb::gil_scoped_release release;
        parallel_for_blocks(signals.rows(), BLOCK_SIZE, n_threads, [&](Index begin, Index end) {
            for (Index voxel = begin; voxel < end; ++voxel) {
                sd::CsdResult result = sd::solve_csd(problem, signals.row(voxel).transpose());
                fods.row(voxel) = result.fod.transpose();
                converged(voxel) = result.converged;
            }
        });
    }
    return {to_numpy(std::move(fods)), to_numpy(std::move(converged))};
}

// Solves one non-negative least-squares problem.
NumpyArray<double, 1> lsqnonneg(const InArray<double, 2>& design_array,
                                const InArray<double, 1>& data_array,
                                std::optional<double> gradient_tolerance) {
    const auto design = matrix_view(design_array); // checks contiguity
    const Vector data = copy_vector(data_array);
    if (data.size() != design.rows()) {
        throw nb::value_error(("c must be (m, n) and d (m,): got " +
                               shape_tuple(design.rows(), design.cols()) + " and " +
                               std::to_string(data.size()))
                                  .c_str());
    }
    Vector solution;
    {
        const nb::gil_scoped_release release;
        const double tolerance =
            gradient_tolerance ? *gradient_tolerance : sd::default_gradient_tolerance(design);
        solution = sd::solve_nnls(design, data, tolerance);
    }
    return to_numpy(std::move(solution));
}

// Solves one non-negative least-squares problem with a ridge and an L1 penalty.
NumpyArray<double, 1> ridge_l1_nnls(const InArray<double, 2>& design_array,
                                    const InArray<double, 1>& data_array, double ridge, double l1,
                                    std::optional<double> gradient_tolerance) {
    const auto design = matrix_view(design_array); // checks contiguity
    const Vector data = copy_vector(data_array);
    if (data.size() != design.rows()) {
        throw nb::value_error(("c must be (m, n) and d (m,): got " +
                               shape_tuple(design.rows(), design.cols()) + " and " +
                               std::to_string(data.size()))
                                  .c_str());
    }
    // Written so that NaN fails too.
    if (!std::isfinite(ridge) || ridge < 0.0 || !std::isfinite(l1) || l1 < 0.0) {
        throw nb::value_error("ridge and l1 must be finite and >= 0.");
    }
    if (l1 > 0.0 && ridge == 0.0) {
        throw nb::value_error("l1 > 0 needs ridge > 0.");
    }
    Vector solution;
    {
        const nb::gil_scoped_release release;
        const double tolerance =
            gradient_tolerance ? *gradient_tolerance : sd::default_gradient_tolerance(design);
        const sd::RidgeDesign ridge_design = sd::make_ridge_design(design, ridge);
        solution = sd::solve_ridge_nnls(ridge_design, data, l1, tolerance);
    }
    return to_numpy(std::move(solution));
}

// Constructs a TissueKernel in place from copies of the kernel matrices.
void construct_kernel(sd::TissueKernel* kernel, const InArray<double, 2>& anisotropic_array,
                      const InArray<double, 2>& isotropic_array,
                      const InArray<double, 2>& weighted_isotropic_array,
                      const BlockPairs& blocks) {
    sd::TissueKernel copied{
        .anisotropic = copy_matrix(anisotropic_array),
        .isotropic = copy_matrix(isotropic_array),
        .weighted_isotropic = copy_matrix(weighted_isotropic_array),
        .anisotropic_blocks = {},
    };
    const Index n_measurements = copied.anisotropic.rows();
    const Index n_anisotropic = copied.n_anisotropic();
    const Index n_isotropic = copied.n_isotropic();
    if (copied.isotropic.rows() != n_measurements ||
        copied.weighted_isotropic.rows() != n_measurements ||
        copied.weighted_isotropic.cols() != n_isotropic) {
        throw nb::value_error("the kernel matrices' shapes do not match");
    }
    bool blocks_valid = !blocks.empty();
    for (const auto& [start, stop] : blocks) {
        blocks_valid = blocks_valid && start >= 0 && start < stop && stop <= n_anisotropic;
    }
    if (!blocks_valid) {
        throw nb::value_error(("aniso_blocks must be non-empty ranges within 0.." +
                               std::to_string(n_anisotropic) + ", got " + blocks_string(blocks))
                                  .c_str());
    }
    for (const auto& [start, stop] : blocks) {
        copied.anisotropic_blocks.push_back({.begin = start, .end = stop});
    }
    new (kernel) sd::TissueKernel(std::move(copied));
}

// Raises ValueError unless solver has as many directions as kernel has anisotropic columns,
// and n_measurements measurements.
void require_matching_solver(const std::string& name, const sd::TissueKernel& kernel,
                             const sd::FodSolver& solver, Index n_measurements) {
    if (solver.n_directions() != kernel.n_anisotropic() ||
        solver.n_measurements() != n_measurements) {
        throw nb::value_error(("the " + name + " solver (" + std::to_string(solver.n_directions()) +
                               " directions, " + std::to_string(solver.n_measurements()) +
                               " measurements) does not match its kernel (" +
                               std::to_string(kernel.n_anisotropic()) + " directions, " +
                               std::to_string(n_measurements) + " measurements)")
                                  .c_str());
    }
}

// Fits the FODs and tissue fractions of every voxel with a multi-tissue kernel.
std::pair<NumpyArray<double, 2>, NumpyArray<double, 2>> fit_multi_tissue(
    const InArray<double, 2>& signals_array, const InArray<double, 1>& weights_array,
    const sd::TissueKernel& low_resolution_kernel, const sd::FodSolver& low_resolution_solver,
    const sd::TissueKernel& high_resolution_kernel, const sd::FodSolver& high_resolution_solver,
    Index max_rounds, double fraction_tolerance, double final_gradient_tolerance, int n_threads) {
    const auto signals = matrix_view(signals_array); // checks contiguity
    const Vector weights = copy_vector(weights_array);
    const Index n_measurements = high_resolution_kernel.anisotropic.rows();
    if (signals.cols() != n_measurements || weights.size() != n_measurements ||
        low_resolution_kernel.anisotropic.rows() != n_measurements) {
        throw nb::value_error(("signals " + shape_tuple(signals.rows(), signals.cols()) +
                               ", weights " + std::to_string(weights.size()) +
                               " and the kernels' " + std::to_string(n_measurements) +
                               " measurements do not match")
                                  .c_str());
    }
    if (low_resolution_kernel.n_isotropic() != high_resolution_kernel.n_isotropic()) {
        throw nb::value_error("the two kernels have different numbers of isotropic tissues");
    }
    require_matching_solver("low-resolution", low_resolution_kernel, low_resolution_solver,
                            n_measurements);
    require_matching_solver("high-resolution", high_resolution_kernel, high_resolution_solver,
                            n_measurements);
    require_threads(n_threads);
    require_non_negative(max_rounds, "max_outer");

    const sd::Resolution low_resolution{.kernel = low_resolution_kernel,
                                        .solver = low_resolution_solver};
    const sd::Resolution high_resolution{.kernel = high_resolution_kernel,
                                         .solver = high_resolution_solver};
    const sd::LoopSettings settings{.max_rounds = max_rounds,
                                    .fraction_tolerance = fraction_tolerance,
                                    .final_gradient_tolerance = final_gradient_tolerance};
    const Index n_voxels = signals.rows();
    const Index n_fractions = static_cast<Index>(high_resolution_kernel.anisotropic_blocks.size()) +
                              high_resolution_kernel.n_isotropic();
    Matrix fods(n_voxels, high_resolution_kernel.n_anisotropic());
    Matrix fractions(n_voxels, n_fractions);
    {
        const nb::gil_scoped_release release;
        parallel_for_blocks(n_voxels, BLOCK_SIZE, n_threads, [&](Index begin, Index end) {
            const Index n_block = end - begin;
            sd::fit_multi_tissue_block(signals.middleRows(begin, n_block), weights, low_resolution,
                                       high_resolution, settings, fods.middleRows(begin, n_block),
                                       fractions.middleRows(begin, n_block));
        });
    }
    return {to_numpy(std::move(fods)), to_numpy(std::move(fractions))};
}

// Constructs an LsiDesign in place from copies of its matrices.
void construct_lsi_design(sd::LsiDesign* design, const InArray<double, 2>& design_array,
                          const InArray<double, 2>& constraints_array, double ridge) {
    const Matrix matrix = copy_matrix(design_array);
    const Matrix constraints = copy_matrix(constraints_array);
    new (design) sd::LsiDesign(sd::make_lsi_design(matrix, constraints, ridge));
}

// Solves the constrained least-squares problem of every voxel.
std::pair<NumpyArray<double, 2>, NumpyArray<bool, 1>>
fit_lsi(const InArray<double, 2>& signals_array, const sd::LsiDesign& design, int n_threads) {
    const auto signals = matrix_view(signals_array); // checks contiguity
    if (signals.cols() != design.q_transposed.cols()) {
        throw nb::value_error(("signals " + shape_tuple(signals.rows(), signals.cols()) +
                               " do not have the design's " +
                               std::to_string(design.q_transposed.cols()) + " measurements")
                                  .c_str());
    }
    require_threads(n_threads);

    Matrix solutions(signals.rows(), design.back_substitution.rows());
    ColumnVector<bool> converged(signals.rows());
    {
        const nb::gil_scoped_release release;
        parallel_for_blocks(signals.rows(), BLOCK_SIZE, n_threads, [&](Index begin, Index end) {
            for (Index voxel = begin; voxel < end; ++voxel) {
                const Vector signal = signals.row(voxel).transpose();
                sd::LsiResult result = sd::solve_lsi(design, signal);
                solutions.row(voxel) = result.solution.transpose();
                converged(voxel) = result.converged;
            }
        });
    }
    return {to_numpy(std::move(solutions)), to_numpy(std::move(converged))};
}

} // namespace

void register_sd(nb::module_& module) {
    nb::class_<sd::FodSolver>(
        module, "FodSolver",
        "An FOD solver with the matrices it needs from one kernel, copied into C++.")
        .def_static("drl", &drl_solver, nb::arg("k_t").noconvert(), nb::arg("k_t_k").noconvert(),
                    nb::arg("nu"), nb::arg("n_iter"), nb::arg("init"),
                    "Makes a damped Richardson-Lucy solver, starting from init in every\n"
                    "direction.\n"
                    "\n"
                    "k_t (c, m) is the kernel transposed and k_t_k (c, c) its product with the\n"
                    "kernel.\n"
                    "\n"
                    "Raises:\n"
                    "    ValueError: If the shapes do not match.\n")
        .def_static("l2nnls", &l2nnls_solver, nb::arg("k").noconvert(), nb::arg("reg"),
                    nb::arg("tol_x"),
                    "Makes a non-negative least-squares solver with a ridge penalty on the FOD.\n"
                    "\n"
                    "k (m, c) is the weighted anisotropic kernel. Each voxel's FOD f minimises\n"
                    "||k f - s||^2 + reg^2 ||f||^2 with f >= 0, so reg = 0 is plain NNLS. A\n"
                    "coefficient enters the solution only when its gradient exceeds tol_x.\n"
                    "\n"
                    "Raises:\n"
                    "    ValueError: If reg is negative or tol_x is not positive.\n")
        .def_static("lasso", &elastic_net_solver, nb::arg("k").noconvert(), nb::arg("reg"),
                    nb::arg("alpha"), nb::arg("tol"), nb::arg("max_iter"),
                    "Makes an elastic-net solver, which regresses each voxel's signal on the\n"
                    "kernel's columns.\n"
                    "\n"
                    "k (m, c) is the weighted anisotropic kernel, whose columns are scaled to\n"
                    "unit root mean square, giving k0; there is no intercept. The coefficients\n"
                    "b minimise (1 / 2m) ||s - k0 b||^2 + reg (alpha ||b||_1 +\n"
                    "(1 - alpha) / 2 ||b||^2), until no coefficient changes by more than tol\n"
                    "relative to 1 + its size, or after max_iter sweeps. The FOD is b divided\n"
                    "by the column scales, and it can be negative.\n"
                    "\n"
                    "Raises:\n"
                    "    ValueError: If reg is negative, alpha is not in (0, 1], tol is not\n"
                    "        positive or max_iter is below 1.\n")
        .def_prop_ro("n_dirs", &sd::FodSolver::n_directions,
                     "Number of FOD directions the solver returns.")
        .def_prop_ro("n_meas", &sd::FodSolver::n_measurements,
                     "Number of measurements per voxel it expects.")
        .def_prop_ro("n_unconverged", &sd::FodSolver::n_unconverged,
                     "Number of solves so far that stopped at max_iter without converging;\n"
                     "always 0 except for lasso.");

    module.def("fit_single_tissue", &fit_single_tissue, nb::arg("signals").noconvert(),
               nb::arg("solver"), nb::arg("n_threads"),
               "Fits the FOD (n, c) of every row of signals (n, m), C-contiguous, with one\n"
               "solver.\n"
               "\n"
               "Raises:\n"
               "    ValueError: If signals is not C-contiguous, does not have the solver's\n"
               "        number of measurements, or n_threads is 0.\n");

    module.def("fit_csd", &fit_csd, nb::arg("signals").noconvert(), nb::arg("initial").noconvert(),
               nb::arg("forward").noconvert(), nb::arg("constraint").noconvert(),
               nb::arg("norm_weight"), nb::arg("threshold"), nb::arg("max_iter"),
               nb::arg("n_threads"),
               "Fits every row of signals (n, m) by constrained spherical deconvolution.\n"
               "\n"
               "initial (c0, m) gives the starting FOD's first c0 coefficients, forward (m, c)\n"
               "is the forward model M and constraint (k, c) the weighted SH basis on the\n"
               "constraint directions. Each solve minimises ||M f - s||^2 + norm_weight ||f||^2\n"
               "plus the squared constraint rows where the FOD is below threshold, until that\n"
               "set of rows repeats or after max_iter solves. Returns the FODs (n, c) and\n"
               "whether each voxel's set repeated (n,).\n"
               "\n"
               "Raises:\n"
               "    ValueError: If signals is not C-contiguous, the shapes do not match,\n"
               "        max_iter is negative or n_threads is 0.\n");

    module.def("lsqnonneg", &lsqnonneg, nb::arg("c").noconvert(), nb::arg("d").noconvert(),
               nb::arg("tol_x") = nb::none(),
               "Solves one non-negative least-squares problem, min ||c x - d|| with x >= 0.\n"
               "\n"
               "c is (m, n) and d (m,). A coefficient enters the solution only when its\n"
               "gradient exceeds tol_x; None uses 10 * eps * norm(c, 1) * max(m, n).\n"
               "\n"
               "Raises:\n"
               "    ValueError: If c is not C-contiguous or the shapes do not match.\n");

    module.def("ridge_l1_nnls", &ridge_l1_nnls, nb::arg("c").noconvert(), nb::arg("d").noconvert(),
               nb::arg("ridge"), nb::arg("l1"), nb::arg("tol_x") = nb::none(),
               "Solves min ||c x - d||^2 + ridge^2 ||x||^2 + 2 l1 sum(x) with x >= 0.\n"
               "\n"
               "c is (m, n) and d (m,). Since x >= 0, the last term is an L1 penalty, which\n"
               "sets the coefficients the data do not support to zero; ridge = l1 = 0 is\n"
               "plain NNLS. A coefficient enters the solution only when its gradient\n"
               "exceeds tol_x; None uses 10 * eps * norm(c, 1) * max(m, n).\n"
               "\n"
               "Raises:\n"
               "    ValueError: If c is not C-contiguous, the shapes do not match, ridge or\n"
               "        l1 is negative or not finite, or l1 > 0 with ridge = 0.\n");

    nb::class_<sd::TissueKernel>(module, "DeconvKernel",
                                 "One resolution of a multi-tissue kernel, copied into C++.")
        .def("__init__", &construct_kernel, nb::arg("k_aniso").noconvert(),
             nb::arg("k_iso").noconvert(), nb::arg("w_k_iso").noconvert(), nb::arg("aniso_blocks"),
             "Copies the matrices of one kernel resolution.\n"
             "\n"
             "k_aniso is (m, ca), k_iso and w_k_iso (m, ci). aniso_blocks holds the\n"
             "(start, stop) columns of each anisotropic response function.\n"
             "\n"
             "Raises:\n"
             "    ValueError: If the shapes do not match, or a block is empty or outside\n"
             "        0..ca.\n");

    module.def("fit_multi_tissue", &fit_multi_tissue, nb::arg("signals").noconvert(),
               nb::arg("weights").noconvert(), nb::arg("lr"), nb::arg("lr_solver"), nb::arg("hr"),
               nb::arg("hr_solver"), nb::arg("max_outer"), nb::arg("fraction_tol"),
               nb::arg("final_tol_x"), nb::arg("n_threads"),
               "Fits the FODs and tissue fractions of every row of signals (n, m). The\n"
               "shell weights (m,) apply to the FOD fits only.\n"
               "\n"
               "Returns the high-resolution FODs (n, ca_hr) and the fractions\n"
               "(n, n_aniso_rfs + ci), summing to 1.\n"
               "\n"
               "Raises:\n"
               "    ValueError: If signals is not C-contiguous, the shapes do not match, a\n"
               "        solver does not match its kernel, or n_threads is 0.\n");

    nb::class_<sd::LsiDesign>(module, "LsiDesign",
                              "A least-squares problem with inequality constraints, copied into "
                              "C++ and prepared for many data vectors.")
        .def("__init__", &construct_lsi_design, nb::arg("a").noconvert(), nb::arg("g").noconvert(),
             nb::arg("ridge"),
             "Prepares min ||a x - s||^2 + ridge^2 ||x||^2 subject to g x >= 0.\n"
             "\n"
             "a is (m, n) and g (k, n).\n"
             "\n"
             "Raises:\n"
             "    ValueError: If the shapes do not match, ridge is negative, or [a; ridge I]\n"
             "        does not have full column rank.\n");

    module.def("fit_lsi", &fit_lsi, nb::arg("signals").noconvert(), nb::arg("design"),
               nb::arg("n_threads"),
               "Solves the constrained least-squares problem of every row of signals (v, m).\n"
               "\n"
               "Returns the solutions (v, n) and whether each voxel's NNLS converged (v,).\n"
               "\n"
               "Raises:\n"
               "    ValueError: If signals is not C-contiguous, does not have the design's\n"
               "        number of measurements, or n_threads is 0.\n");
}

} // namespace larmor::python
