// Bindings of the spherical-harmonics and peak code.

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>

#include "common/parallel.hpp"
#include "common/types.hpp"
#include "python/arrays.hpp"
#include "python/bindings.hpp"
#include "sh/harmonics.hpp"
#include "sh/peaks.hpp"

namespace nb = nanobind;

namespace larmor::python {

namespace {

// Voxels per block of the parallel loop.
constexpr Index PEAKS_BLOCK_SIZE = 16;

// An SH series of at most MAX_COEFFS coefficients, copied out of a numpy array.
using Coefficients = Eigen::Matrix<double, Eigen::Dynamic, 1, 0, sh::MAX_COEFFS, 1>;

// Refuses an lmax the fixed-size tables cannot hold, an odd lmax for a symmetric series, or a
// number of coefficients that does not match the layout.
void check_lmax(Index lmax, bool full, Index n_coeffs_given) {
    if (lmax < 0 || lmax > sh::MAX_LMAX || (!full && lmax % 2 != 0)) {
        throw nb::value_error(("lmax must be at most " + std::to_string(sh::MAX_LMAX) +
                               ", and even unless full, got " + std::to_string(lmax))
                                  .c_str());
    }
    const Index n_expected = sh::n_coeffs(lmax, full);
    if (n_expected != n_coeffs_given) {
        throw nb::value_error(("lmax " + std::to_string(lmax) + " has " +
                               std::to_string(n_expected) + " coefficients, got " +
                               std::to_string(n_coeffs_given))
                                  .c_str());
    }
}

// Returns the shape of a 2D array as "(2, 3)", for error messages.
template <typename Array> std::string shape_tuple(const Array& array) {
    return "(" + std::to_string(extent(array, 0)) + ", " + std::to_string(extent(array, 1)) + ")";
}

// Copies a 1D array of any strides into a series; the caller checks its length first.
Coefficients copy_series(const InArray<double, 1>& series_array) {
    Coefficients series(extent(series_array, 0));
    for (Index position = 0; position < series.size(); ++position) {
        series(position) = series_array(static_cast<std::size_t>(position));
    }
    return series;
}

// Returns the amplitude of a series and its derivatives at (theta, phi).
std::tuple<double, double, double, double, double, double>
sh_derivatives(const InArray<double, 1>& series_array, Index lmax, double theta, double phi,
               bool full) {
    check_lmax(lmax, full, extent(series_array, 0));
    const Coefficients series = copy_series(series_array);
    sh::Derivatives result;
    {
        const nb::gil_scoped_release release;
        result = sh::derivatives(series, lmax, full, theta, phi);
    }
    return {result.value,         result.d_theta,     result.d_phi,
            result.d_theta_theta, result.d_theta_phi, result.d_phi_phi};
}

// Climbs from `start` to the nearest peak and returns its direction and amplitude.
std::pair<NumpyArray<double, 1>, double> climb(const InArray<double, 1>& series_array, Index lmax,
                                               const std::array<double, 3>& start, Index max_iters,
                                               double tolerance, double max_step, bool full) {
    check_lmax(lmax, full, extent(series_array, 0));
    if (max_iters < 0) {
        throw nb::value_error("max_iters must be at least 0");
    }
    const Coefficients series = copy_series(series_array);
    const sh::ClimbSettings settings{
        .max_iters = max_iters, .tolerance = tolerance, .max_step = max_step};
    std::optional<sh::Peak> peak;
    {
        const nb::gil_scoped_release release;
        peak = sh::climb(series, lmax, full, Vector3(start[0], start[1], start[2]), settings);
    }
    Vector direction = Vector::Constant(3, std::numeric_limits<double>::quiet_NaN());
    double amplitude = std::numeric_limits<double>::quiet_NaN();
    if (peak) {
        direction = peak->direction;
        amplitude = peak->amplitude;
    }
    return {to_numpy(std::move(direction)), amplitude};
}

// Finds the peaks of every voxel of a chunk.
std::pair<NumpyArray<double, 3>, NumpyArray<double, 2>>
peaks_chunk(const InArray<double, 2>& coeffs_array, Index lmax,
            const InArray<double, 2>& on_sphere_array, const InArray<double, 2>& vertices_array,
            const InArray<std::int64_t, 2>& neighbours_array, Index n_peaks,
            double relative_threshold, double threshold, int n_threads, bool full) {
    const auto coeffs = matrix_view(coeffs_array); // checks contiguity
    const auto on_sphere = matrix_view(on_sphere_array);
    check_lmax(lmax, full, coeffs.cols());
    const Index n_vertices = extent(vertices_array, 0);
    if (extent(vertices_array, 1) != 3 || extent(neighbours_array, 0) != n_vertices ||
        extent(neighbours_array, 1) != 6 || on_sphere.rows() != coeffs.rows() ||
        on_sphere.cols() != n_vertices) {
        throw nb::value_error(("shapes do not match: coeffs " + shape_tuple(coeffs_array) +
                               ", on_sphere " + shape_tuple(on_sphere_array) + ", vertices " +
                               shape_tuple(vertices_array) + ", neighbours " +
                               shape_tuple(neighbours_array))
                                  .c_str());
    }

    // The sphere is copied, so vertices and neighbours may have any strides.
    sh::Sphere sphere;
    sphere.vertices.reserve(static_cast<std::size_t>(n_vertices));
    sphere.neighbours.reserve(static_cast<std::size_t>(n_vertices));
    for (Index vertex = 0; vertex < n_vertices; ++vertex) {
        const auto row = static_cast<std::size_t>(vertex);
        sphere.vertices.emplace_back(vertices_array(row, 0), vertices_array(row, 1),
                                     vertices_array(row, 2));
        std::array<Index, 6> neighbours{};
        for (std::size_t slot = 0; slot < neighbours.size(); ++slot) {
            const std::int64_t neighbour = neighbours_array(row, slot);
            if (neighbour < 0 || neighbour >= n_vertices) {
                throw nb::value_error("a neighbour index is outside the sphere");
            }
            neighbours.at(slot) = neighbour;
        }
        sphere.neighbours.push_back(neighbours);
    }
    if (n_peaks < 1) {
        throw nb::value_error("n_peaks must be at least 1");
    }
    require_threads(n_threads);

    const sh::PeakSettings settings{.n_peaks = n_peaks,
                                    .relative_threshold = relative_threshold,
                                    .threshold = threshold,
                                    .climb_settings = sh::ClimbSettings{}};
    const Index n_voxels = coeffs.rows();
    Matrix directions(n_voxels, n_peaks * 3);
    Matrix amplitudes(n_voxels, n_peaks);
    {
        const nb::gil_scoped_release release;
        parallel_for_blocks(
            n_voxels, PEAKS_BLOCK_SIZE, n_threads, [] { return std::vector<sh::Peak>{}; },
            [&](Index begin, Index end, std::vector<sh::Peak>& found) {
                for (Index voxel = begin; voxel < end; ++voxel) {
                    sh::voxel_peaks(coeffs.row(voxel).transpose(), lmax, full,
                                    on_sphere.row(voxel).transpose(), sphere, settings, found,
                                    directions.row(voxel).transpose(),
                                    amplitudes.row(voxel).transpose());
                }
            });
    }
    const std::array<Index, 3> directions_shape{n_voxels, n_peaks, 3};
    return {to_numpy<3>(std::move(directions), directions_shape), to_numpy(std::move(amplitudes))};
}

} // namespace

void register_sh(nb::module_& module) {
    module.attr("MAX_LMAX") = sh::MAX_LMAX;

    module.def("peaks_chunk", &peaks_chunk, nb::arg("coeffs").noconvert(), nb::arg("lmax"),
               nb::arg("on_sphere").noconvert(), nb::arg("vertices").noconvert(),
               nb::arg("neighbours").noconvert(), nb::arg("n_peaks"), nb::arg("relative_threshold"),
               nb::arg("threshold"), nb::arg("n_threads"), nb::arg("full") = false,
               "Finds the peaks of every voxel of a chunk.\n"
               "\n"
               "coeffs is (n, C) in the tournier basis, with all degrees if full (an asymmetric\n"
               "FOD, whose peaks keep their sign and are not merged with their opposites).\n"
               "on_sphere (n, V) holds the amplitudes at the vertices (V, 3), and neighbours\n"
               "(V, 6) the 6 neighbours of each vertex. Returns directions (n, n_peaks, 3) and\n"
               "amplitudes (n, n_peaks), largest first, with z >= 0 unless full, and NaN where\n"
               "a voxel has fewer peaks.\n"
               "\n"
               "Raises:\n"
               "    ValueError: If lmax is above 20, odd without full or does not match C, the\n"
               "        shapes do not match, a neighbour is outside the sphere, n_peaks or\n"
               "        n_threads is 0.\n");

    module.def("climb", &climb, nb::arg("sh").noconvert(), nb::arg("lmax"), nb::arg("start"),
               nb::arg("max_iters") = 50, nb::arg("tol") = 1e-4, nb::arg("max_step") = 0.2,
               nb::arg("full") = false,
               // start takes any sequence of three numbers, a numpy array too, which the
               // default signature leaves out.
               nb::sig("def climb("
                       "sh: ndarray[dtype=float64, shape=(*), device='cpu', writable=False], "
                       "lmax: int, "
                       "start: tuple[float, float, float] | list[float] | "
                       "numpy.typing.NDArray[numpy.float64], "
                       "max_iters: int = 50, tol: float = 0.0001, max_step: float = 0.2, "
                       "full: bool = False) -> "
                       "tuple[numpy.ndarray[dtype=float64, shape=(*)], float]"),
               "Climbs from start to the peak whose basin it lies in.\n"
               "\n"
               "Returns the peak's direction (3,) and its amplitude, both NaN if the climb does\n"
               "not converge within max_iters.\n"
               "\n"
               "Raises:\n"
               "    ValueError: If lmax is above 20, odd without full, or does not match sh.\n");

    module.def("sh_derivatives", &sh_derivatives, nb::arg("sh").noconvert(), nb::arg("lmax"),
               nb::arg("theta"), nb::arg("phi"), nb::arg("full") = false,
               "Returns the amplitude f of a series and its derivatives at (theta, phi).\n"
               "\n"
               "The values are f, df/dtheta, df/dphi / sin theta, d2f/dtheta2,\n"
               "d2f/dtheta dphi / sin theta and d2f/dphi2 / sin^2 theta.\n"
               "\n"
               "Raises:\n"
               "    ValueError: If lmax is above 20, odd without full, or does not match sh.\n");
}

} // namespace larmor::python
