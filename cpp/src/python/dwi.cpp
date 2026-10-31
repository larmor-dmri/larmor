// Bindings of the log-signal fits of DTI and DKI, the NODDI fit, MP-PCA denoising and Gibbs-ringing
// removal.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>

#include "common/parallel.hpp"
#include "common/types.hpp"
#include "dwi/gibbs.hpp"
#include "dwi/log_fit.hpp"
#include "dwi/mppca.hpp"
#include "dwi/noddi.hpp"
#include "python/arrays.hpp"
#include "python/bindings.hpp"

namespace nb = nanobind;

namespace larmor::python {

namespace {

// Voxels per block of the parallel loop of the fits.
constexpr Index BLOCK_SIZE = 64;
// Voxels per block of the parallel loop of the denoising.
constexpr Index DENOISE_BLOCK_SIZE = 8;
// Voxels per block of the parallel loop of the NODDI fit.
constexpr Index NODDI_BLOCK_SIZE = 16;

// Fits the log signal of every voxel and returns the parameters and the constrained flags.
std::pair<NumpyArray<double, 2>, NumpyArray<bool, 1>>
fit_log_signal(const InArray<double, 2>& design_array, const InArray<double, 2>& constraints_array,
               const InArray<double, 2>& log_signals_array, bool weighted, Index max_reweightings,
               double log_signal_tolerance, int n_threads) {
    const auto design = matrix_view(design_array); // checks contiguity
    const auto constraints = matrix_view(constraints_array);
    const auto log_signals = matrix_view(log_signals_array);
    if (design.rows() < design.cols() || constraints.cols() != design.cols() ||
        log_signals.cols() != design.rows()) {
        throw nb::value_error(
            ("x must be (n, p) with n >= p, g (m, p) and log_s (v, n): got x (" +
             std::to_string(design.rows()) + ", " + std::to_string(design.cols()) + "), g (" +
             std::to_string(constraints.rows()) + ", " + std::to_string(constraints.cols()) +
             "), log_s (" + std::to_string(log_signals.rows()) + ", " +
             std::to_string(log_signals.cols()) + ")")
                .c_str());
    }
    if (max_reweightings < 0) {
        throw nb::value_error("max_iter must be at least 0");
    }
    require_threads(n_threads);

    const dwi::LogFitSettings settings{.weighted = weighted,
                                       .max_reweightings = max_reweightings,
                                       .log_signal_tolerance = log_signal_tolerance};
    const Index n_voxels = log_signals.rows();
    Matrix parameters(n_voxels, design.cols());
    dwi::Flags constrained(n_voxels);
    {
        const nb::gil_scoped_release release;
        const dwi::LogSignalDesign prepared = dwi::make_log_signal_design(design, constraints);
        parallel_for_blocks(
            n_voxels, BLOCK_SIZE, n_threads, [&] { return dwi::make_log_fit_scratch(prepared); },
            [&](Index begin, Index end, dwi::LogFitScratch& scratch) {
                const Index n_block = end - begin;
                dwi::fit_log_signal_block(
                    prepared, settings, log_signals.middleRows(begin, n_block), scratch,
                    parameters.middleRows(begin, n_block), constrained.segment(begin, n_block));
            });
    }
    return {to_numpy(std::move(parameters)), to_numpy(std::move(constrained))};
}

// Returns "(rows, columns)", for error messages.
std::string shape_tuple(Index rows, Index columns) {
    return "(" + std::to_string(rows) + ", " + std::to_string(columns) + ")";
}

// Fits NODDI to every voxel and returns the parameters and which voxels were fitted.
std::pair<NumpyArray<double, 2>, NumpyArray<bool, 1>>
fit_noddi(const InArray<double, 2>& signals_array, const InArray<double, 2>& directions_array,
          const InArray<double, 1>& bvals_array, const InArray<double, 2>& bvecs_array,
          const InArray<bool, 1>& is_b0_array, const InArray<double, 1>& v_ic_grid_array,
          const InArray<double, 1>& kappa_grid_array, double d_par, double d_iso, double ridge,
          double l1, int n_threads) {
    const auto signals = matrix_view(signals_array); // checks contiguity
    const auto directions = matrix_view(directions_array);
    const auto bvals = flat_view(bvals_array);
    const auto bvecs = matrix_view(bvecs_array);
    const auto is_b0 = flat_view(is_b0_array);
    const auto v_ic_grid = flat_view(v_ic_grid_array);
    const auto kappa_grid = flat_view(kappa_grid_array);
    const Index n = bvals.size();
    if (signals.cols() != n || bvecs.rows() != n || bvecs.cols() != 3 || is_b0.size() != n ||
        directions.rows() != signals.rows() || directions.cols() != 3) {
        throw nb::value_error(
            ("signals must be (v, n), mu (v, 3), bvals (n,), bvecs (n, 3) and is_b0 (n,): got "
             "signals " +
             shape_tuple(signals.rows(), signals.cols()) + ", mu " +
             shape_tuple(directions.rows(), directions.cols()) + ", bvals (" + std::to_string(n) +
             ",), bvecs " + shape_tuple(bvecs.rows(), bvecs.cols()) + ", is_b0 (" +
             std::to_string(is_b0.size()) + ",)")
                .c_str());
    }
    if (v_ic_grid.size() == 0 || kappa_grid.size() == 0) {
        throw nb::value_error("the v_ic and kappa grids must not be empty");
    }
    if (!(ridge > 0.0) || !(l1 >= 0.0)) {
        throw nb::value_error("ridge must be positive and l1 at least 0");
    }
    require_threads(n_threads);

    const dwi::NoddiSettings settings{
        .d_par = d_par,
        .d_iso = d_iso,
        .v_ic_grid = std::vector<double>(v_ic_grid.begin(), v_ic_grid.end()),
        .kappa_grid = std::vector<double>(kappa_grid.begin(), kappa_grid.end()),
        .ridge = ridge,
        .l1 = l1};
    const Index n_voxels = signals.rows();
    Matrix parameters = Matrix::Zero(n_voxels, 6);
    dwi::Flags fitted(n_voxels);
    {
        const nb::gil_scoped_release release;
        const dwi::NoddiProtocol protocol = dwi::make_noddi_protocol(
            Vector(bvals), Matrix(bvecs), std::vector<bool>(is_b0.begin(), is_b0.end()), settings);
        parallel_for_blocks(
            n_voxels, NODDI_BLOCK_SIZE, n_threads, [] { return 0; },
            [&](Index begin, Index end, int& /*scratch*/) {
                for (Index voxel = begin; voxel < end; ++voxel) {
                    const Vector signal = signals.row(voxel).transpose();
                    const Vector3 mu = directions.row(voxel).transpose();
                    const dwi::NoddiVoxelFit fit =
                        dwi::fit_noddi_voxel(protocol, settings, signal, mu);
                    parameters.row(voxel) << fit.v_ic, fit.kappa, fit.v_iso, fit.mu.x(), fit.mu.y(),
                        fit.mu.z();
                    fitted(voxel) = fit.fitted;
                }
            });
    }
    return {to_numpy(std::move(parameters)), to_numpy(std::move(fitted))};
}

// Returns the normalised NODDI signal of every volume.
NumpyArray<double, 1> noddi_signal(const InArray<double, 1>& bvals_array,
                                   const InArray<double, 2>& bvecs_array, double v_ic, double kappa,
                                   double v_iso, const std::array<double, 3>& mu, double d_par,
                                   double d_iso) {
    const auto bvals = flat_view(bvals_array); // checks contiguity
    const auto bvecs = matrix_view(bvecs_array);
    if (bvecs.rows() != bvals.size() || bvecs.cols() != 3) {
        throw nb::value_error(("bvals must be (n,) and bvecs (n, 3): got (" +
                               std::to_string(bvals.size()) + ",) and " +
                               shape_tuple(bvecs.rows(), bvecs.cols()))
                                  .c_str());
    }
    const dwi::NoddiSettings settings{.d_par = d_par, .d_iso = d_iso};
    const dwi::NoddiProtocol protocol = dwi::make_noddi_protocol(
        Vector(bvals), Matrix(bvecs), std::vector<bool>(static_cast<std::size_t>(bvals.size())),
        settings);
    Vector signal = dwi::noddi_signal(protocol, settings, v_ic, kappa, v_iso,
                                      Vector3(mu[0], mu[1], mu[2]).normalized());
    return to_numpy(std::move(signal));
}

// The denoised signals, the noise levels and the numbers of signal components.
using DenoisedArrays =
    std::tuple<NumpyArray<double, 2>, NumpyArray<double, 1>, NumpyArray<std::int64_t, 1>>;

// Denoises the voxels at the given flat indices by MP-PCA.
DenoisedArrays denoise_mppca(const InArray<float, 4>& image_array,
                             const InArray<std::int64_t, 1>& centres_array,
                             const std::array<Index, 3>& patch_shape, int n_threads) {
    require_contiguous(image_array, "image");
    const auto centres = flat_view(centres_array); // checks contiguity
    const std::array<Index, 3> grid_shape{extent(image_array, 0), extent(image_array, 1),
                                          extent(image_array, 2)};
    const Index n_volumes = extent(image_array, 3);
    if (n_volumes < 1) {
        throw nb::value_error("the image must have at least one volume");
    }
    for (Index axis = 0; axis < 3; ++axis) {
        const auto unsigned_axis = static_cast<std::size_t>(axis);
        if (patch_shape.at(unsigned_axis) < 1 ||
            patch_shape.at(unsigned_axis) > grid_shape.at(unsigned_axis)) {
            throw nb::value_error(("the patch must have 1 to " +
                                   std::to_string(grid_shape.at(unsigned_axis)) +
                                   " voxels along axis " + std::to_string(axis) + ", got " +
                                   std::to_string(patch_shape.at(unsigned_axis)))
                                      .c_str());
        }
    }
    const Index n_grid_voxels = grid_shape[0] * grid_shape[1] * grid_shape[2];
    if (centres.size() > 0 && (centres.minCoeff() < 0 || centres.maxCoeff() >= n_grid_voxels)) {
        throw nb::value_error(
            ("voxel indices must be in [0, " + std::to_string(n_grid_voxels) + ")").c_str());
    }
    require_threads(n_threads);

    const Eigen::Map<const RowMatrix<float>> image(image_array.data(), n_grid_voxels, n_volumes);
    const dwi::PatchGrid grid{.grid_shape = grid_shape, .patch_shape = patch_shape};
    const Index n_centres = centres.size();
    Matrix denoised(n_centres, n_volumes);
    Vector noise(n_centres);
    dwi::Counts n_signal(n_centres);
    {
        const nb::gil_scoped_release release;
        parallel_for_blocks(
            n_centres, DENOISE_BLOCK_SIZE, n_threads,
            [&] { return dwi::make_mppca_scratch(grid.n_patch_voxels(), n_volumes); },
            [&](Index begin, Index end, dwi::MppcaScratch& scratch) {
                const Index n_block = end - begin;
                dwi::denoise_mppca_block(image, grid, centres.segment(begin, n_block), scratch,
                                         denoised.middleRows(begin, n_block),
                                         noise.segment(begin, n_block),
                                         n_signal.segment(begin, n_block));
            });
    }
    return {to_numpy(std::move(denoised)), to_numpy(std::move(noise)),
            to_numpy(std::move(n_signal))};
}

// Removes Gibbs ringing from every slice of data_array into out_array.
void remove_gibbs_ringing(const InArray<float, 4>& data_array,
                          const InOutArray<float, 4>& out_array, int slice_axis, Index n_shifts,
                          Index min_window, Index max_window, int n_threads) {
    require_contiguous(data_array, "data");
    require_contiguous(out_array, "out");
    std::array<Index, 4> shape{};
    for (Index axis = 0; axis < 4; ++axis) {
        shape.at(static_cast<std::size_t>(axis)) = extent(data_array, axis);
        if (extent(out_array, axis) != extent(data_array, axis)) {
            throw nb::value_error(("out must have the shape of data, " + shape_string(data_array) +
                                   ", got " + shape_string(out_array))
                                      .c_str());
        }
    }
    if (slice_axis < 0 || slice_axis > 2) {
        throw nb::value_error("slice_axis must be 0, 1 or 2");
    }
    if (n_shifts < 1 || min_window < 0 || max_window < min_window) {
        throw nb::value_error("need n_shifts >= 1 and 0 <= min_window <= max_window");
    }
    require_threads(n_threads);

    // Element (i, j, k, v) of the C-contiguous (X, Y, Z, V) arrays is at
    // i * strides[0] + j * strides[1] + k * strides[2] + v.
    const std::array<Index, 3> strides{shape[1] * shape[2] * shape[3], shape[2] * shape[3],
                                       shape[3]};
    // The slice's rows run along the first in-plane axis, its columns along the second.
    const int row_axis = slice_axis == 0 ? 1 : 0;
    const int column_axis = slice_axis == 2 ? 1 : 2;
    const auto at = [](const auto& values, int axis) {
        return values.at(static_cast<std::size_t>(axis));
    };
    const Index n_rows = at(shape, row_axis);
    const Index n_columns = at(shape, column_axis);
    const Index n_volumes = shape[3];
    const Index n_slices = at(shape, slice_axis) * n_volumes;
    const float* data = data_array.data();
    float* out = out_array.data();
    const dwi::GibbsSettings settings{
        .n_shifts = n_shifts, .min_window = min_window, .max_window = max_window};

    struct SliceScratch {
        dwi::GibbsScratch gibbs;
        Matrix slice;
    };
    {
        const nb::gil_scoped_release release;
        parallel_for_blocks(
            n_slices, 1, n_threads,
            [&] {
                return SliceScratch{.gibbs = dwi::make_gibbs_scratch(settings, n_rows, n_columns),
                                    .slice = Matrix(n_rows, n_columns)};
            },
            [&](Index begin, Index end, SliceScratch& scratch) {
                for (Index item = begin; item < end; ++item) {
                    const Index first =
                        (item / n_volumes) * at(strides, slice_axis) + item % n_volumes;
                    const Index row_stride = at(strides, row_axis);
                    const Index column_stride = at(strides, column_axis);
                    for (Index row = 0; row < n_rows; ++row) {
                        for (Index column = 0; column < n_columns; ++column) {
                            scratch.slice(row, column) = static_cast<double>(
                                data[first + row * row_stride + column * column_stride]);
                        }
                    }
                    dwi::remove_gibbs_ringing_slice(settings, scratch.slice, scratch.gibbs);
                    for (Index row = 0; row < n_rows; ++row) {
                        for (Index column = 0; column < n_columns; ++column) {
                            out[first + row * row_stride + column * column_stride] =
                                static_cast<float>(scratch.slice(row, column));
                        }
                    }
                }
            });
    }
}

} // namespace

void register_dwi(nb::module_& module) {
    module.def("fit_log_signal", &fit_log_signal, nb::arg("x").noconvert(),
               nb::arg("g").noconvert(), nb::arg("log_s").noconvert(), nb::arg("weighted"),
               nb::arg("max_iter"), nb::arg("tol"), nb::arg("n_threads"),
               "Fits each row of log_s (v, n) by least squares on the design x (n, p).\n"
               "\n"
               "With weighted=True the fit is weighted by the predicted signal, iterated until\n"
               "no predicted log signal changes by more than tol or for max_iter weighted\n"
               "fits. A voxel whose solution has g @ p < 0 in some row of g (m, p) is fitted\n"
               "again subject to g @ p >= 0; m = 0 means no constraints.\n"
               "\n"
               "Returns the parameters (v, p) and whether each voxel broke a constraint (v,).\n"
               "\n"
               "Raises:\n"
               "    ValueError: If an array is not C-contiguous, the shapes do not match,\n"
               "        max_iter is negative or n_threads is 0.\n");
    module.def("fit_noddi", &fit_noddi, nb::arg("signals").noconvert(), nb::arg("mu").noconvert(),
               nb::arg("bvals").noconvert(), nb::arg("bvecs").noconvert(),
               nb::arg("is_b0").noconvert(), nb::arg("v_ic_grid").noconvert(),
               nb::arg("kappa_grid").noconvert(), nb::arg("d_par"), nb::arg("d_iso"),
               nb::arg("ridge"), nb::arg("l1"), nb::arg("n_threads"),
               "Fits NODDI to each voxel's signal, a row of signals (v, n), from the direction\n"
               "mu (v, 3).\n"
               "\n"
               "The start comes from a dictionary of NODDI signals over v_ic_grid and\n"
               "kappa_grid, solved with the ridge and l1 weights; the fit is the NODDI\n"
               "Matlab Toolbox's maximum-likelihood fit with Rician noise, with its model.\n"
               "d_par and d_iso are in mm^2/s and bvals in s/mm^2.\n"
               "\n"
               "Returns v_ic, kappa, v_iso and the fitted direction of each voxel (v, 6), and\n"
               "whether each voxel was fitted (v,).\n"
               "\n"
               "Raises:\n"
               "    ValueError: If an array is not C-contiguous, the shapes do not match, a\n"
               "        grid is empty, ridge is not positive, l1 is negative or n_threads is 0.\n");
    module.def("noddi_signal", &noddi_signal, nb::arg("bvals").noconvert(),
               nb::arg("bvecs").noconvert(), nb::arg("v_ic"), nb::arg("kappa"), nb::arg("v_iso"),
               nb::arg("mu"), nb::arg("d_par"), nb::arg("d_iso"),
               "Returns the normalised NODDI signal of each volume, as the NODDI Matlab\n"
               "Toolbox's WatsonSHStickTortIsoV_B0 model computes it, for the neurite density\n"
               "v_ic, Watson concentration kappa and free-water fraction v_iso about the\n"
               "direction mu. bvals (n,) are in s/mm^2, bvecs (n, 3) unit directions, and\n"
               "d_par and d_iso in mm^2/s.\n"
               "\n"
               "Raises:\n"
               "    ValueError: If an array is not C-contiguous or the shapes do not match.\n");
    module.def("denoise_mppca", &denoise_mppca, nb::arg("image").noconvert(),
               nb::arg("voxels").noconvert(), nb::arg("patch"), nb::arg("n_threads"),
               "Denoises by MP-PCA the voxels at flat indices voxels (k,) int64 of a\n"
               "C-contiguous float32 image (X, Y, Z, M).\n"
               "\n"
               "Each voxel gets its own patch of patch = (px, py, pz) voxels, shifted to stay\n"
               "inside the grid at its edges.\n"
               "\n"
               "Returns the denoised signals (k, M), the noise level sigma (k,) and the\n"
               "number of signal components (k,) of each voxel; NaN signals and sigma\n"
               "where the patch holds a value that is not finite.\n"
               "\n"
               "Raises:\n"
               "    ValueError: If an array is not C-contiguous, the image has no volumes,\n"
               "        the patch is not 1 to the grid's size along an axis, an index is\n"
               "        outside the grid or n_threads is 0.\n");
    module.def("remove_gibbs_ringing", &remove_gibbs_ringing, nb::arg("data").noconvert(),
               nb::arg("out").noconvert(), nb::arg("slice_axis"), nb::arg("n_shifts"),
               nb::arg("min_window"), nb::arg("max_window"), nb::arg("n_threads"),
               "Removes Gibbs ringing from each slice of data (x, y, z, v) into out.\n"
               "\n"
               "The slices are the planes across slice_axis (0, 1 or 2), in every volume v.\n"
               "Each is unringed by local sub-voxel shifts in steps of 1 / (2 n_shifts)\n"
               "voxels, chosen by the total variation between neighbours min_window to\n"
               "max_window + 1 voxels away on each side. Both arrays are C-contiguous float32\n"
               "of one shape.\n"
               "\n"
               "Raises:\n"
               "    ValueError: If an array is not C-contiguous, the shapes differ,\n"
               "        slice_axis is not 0, 1 or 2, n_shifts < 1, the window is not\n"
               "        0 <= min_window <= max_window, or n_threads is 0.\n");
}

} // namespace larmor::python
