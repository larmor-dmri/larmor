// Bindings of the streamline operations.

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include <Eigen/Core>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include "common/types.hpp"
#include "python/arrays.hpp"
#include "python/bindings.hpp"
#include "streamlines/density.hpp"
#include "streamlines/resample.hpp"

namespace nb = nanobind;

namespace larmor::python {

namespace {

// The points, offsets, segments and fractions of a resampled tractogram.
using ResampledArrays = std::tuple<NumpyArray<float, 2>, NumpyArray<std::int64_t, 1>,
                                   NumpyArray<std::int64_t, 1>, NumpyArray<double, 1>>;

// Refuses offsets that do not start at 0, decrease, or do not end at n_points.
template <typename Offsets> void check_offsets(const Offsets& offsets, Index n_points) {
    if (offsets.size() < 1 || offsets(0) != 0 || offsets(offsets.size() - 1) != n_points) {
        throw nb::value_error("offsets must start at 0 and end at the number of points");
    }
    for (Index index = 1; index < offsets.size(); ++index) {
        if (offsets(index) < offsets(index - 1)) {
            throw nb::value_error("offsets must never decrease");
        }
    }
}

// Refuses a num_points whose output would have too many coordinates to index.
template <typename Offsets> void check_output_size(const Offsets& offsets, Index num_points) {
    constexpr Index MAX_POINTS = std::numeric_limits<Index>::max() / 3;
    Index n_out = 0;
    for (Index streamline = 0; streamline + 1 < offsets.size(); ++streamline) {
        const Index n_points = offsets(streamline + 1) - offsets(streamline);
        const Index count = n_points < 2 ? n_points : num_points;
        if (count > MAX_POINTS - n_out) {
            throw nb::value_error("num_points is too large: the output would have more points "
                                  "than an int64 can index");
        }
        n_out += count;
    }
}

// Resamples every streamline of a tractogram.
ResampledArrays resample_streamlines(const InArray<float, 2>& points_array,
                                     const InArray<std::int64_t, 1>& offsets_array,
                                     double step_size, Index num_points, int n_threads,
                                     bool positions) {
    require_contiguous(points_array, "points");
    require_columns(points_array, 3);
    const auto offsets = flat_view(offsets_array); // checks contiguity
    const Index n_points = extent(points_array, 0);
    check_offsets(offsets, n_points);
    if ((num_points > 0) == (step_size > 0.0)) {
        throw nb::value_error("give either step_size > 0 or num_points >= 2");
    }
    if (num_points != 0 && num_points < 2) {
        throw nb::value_error("num_points must be at least 2");
    }
    if (num_points == 0 && !std::isfinite(static_cast<float>(step_size))) {
        throw nb::value_error("step_size must be finite");
    }
    if (num_points > 0) {
        check_output_size(offsets, num_points);
    }
    require_threads(n_threads);

    const Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> points(
        points_array.data(), n_points, 3);
    const streamlines::ResampleSettings settings{.num_points = num_points,
                                                 .step_size = static_cast<float>(step_size),
                                                 .positions = positions};
    streamlines::ResampledTractogram result;
    {
        const nb::gil_scoped_release release;
        result = streamlines::resample_tractogram(points, offsets, settings, n_threads);
    }
    const std::array<Index, 2> points_shape{result.points.rows(), 3};
    return {to_numpy<2>(std::move(result.points), points_shape),
            to_numpy(std::move(result.offsets)), to_numpy(std::move(result.segments)),
            to_numpy(std::move(result.fractions))};
}

// Adds the track density of a tractogram to density_array, in place.
void add_track_density(const InArray<float, 2>& points_array,
                       const InArray<std::int64_t, 1>& offsets_array,
                       const InArray<double, 2>& world_to_voxel_array,
                       const InArray<std::int64_t, 1>& shape_array, Index upsample_ratio,
                       double max_spacing, bool ends_only, const InArray<double, 1>& weights_array,
                       const InOutArray<double, 1>& density_array, int n_threads) {
    require_contiguous(points_array, "points");
    require_columns(points_array, 3);
    const auto offsets = flat_view(offsets_array); // checks contiguity
    const Index n_points = extent(points_array, 0);
    check_offsets(offsets, n_points);
    const auto world_to_voxel = matrix_view(world_to_voxel_array);
    if (world_to_voxel.rows() != 3 || world_to_voxel.cols() != 4) {
        throw nb::value_error("world_to_voxel must be a (3, 4) matrix");
    }
    const auto shape = flat_view(shape_array);
    if (shape.size() != 3 || shape.minCoeff() < 1) {
        throw nb::value_error("shape must be three sizes of at least 1");
    }
    const auto density = flat_view(density_array);
    if (density.size() != shape.prod()) {
        throw nb::value_error("density must have one value per voxel of shape");
    }
    const auto weights = flat_view(weights_array);
    if (weights.size() != 0 && weights.size() != offsets.size() - 1) {
        throw nb::value_error("weights must be empty or have one value per streamline");
    }
    if (upsample_ratio < 1) {
        throw nb::value_error("upsample_ratio must be at least 1");
    }
    if (!(max_spacing >= 0.0) || !std::isfinite(max_spacing)) {
        throw nb::value_error("max_spacing must be a finite value of at least 0");
    }
    require_threads(n_threads);

    const Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> points(
        points_array.data(), n_points, 3);
    streamlines::VoxelGrid grid;
    grid.world_to_voxel = world_to_voxel;
    for (Index axis = 0; axis < 3; ++axis) {
        grid.shape.at(static_cast<std::size_t>(axis)) = shape(axis);
    }
    const streamlines::DensitySettings settings{
        .upsample_ratio = upsample_ratio, .max_spacing = max_spacing, .ends_only = ends_only};
    const nb::gil_scoped_release release;
    streamlines::add_track_density(points, offsets, grid, settings, weights, density, n_threads);
}

} // namespace

void register_streamlines(nb::module_& module) {
    module.def("resample_streamlines", &resample_streamlines, nb::arg("points").noconvert(),
               nb::arg("offsets").noconvert(), nb::arg("step_size"), nb::arg("num_points"),
               nb::arg("n_threads"), nb::arg("positions") = true,
               "Resamples every streamline to step_size (mm) or to num_points points.\n"
               "\n"
               "points is (P, 3) float32 and offsets (S + 1,) int64, as in a Tractogram. Give\n"
               "step_size > 0 with num_points 0, or num_points >= 2 with step_size 0.\n"
               "Streamlines of fewer than 2 points are copied. Returns the new points (Q, 3)\n"
               "float32 and offsets (S + 1,) int64, and with positions, for each new point\n"
               "the index of the input point that starts its segment (Q,) int64 and its\n"
               "fraction along the segment (Q,) float64; without, both are empty.\n"
               "\n"
               "Raises:\n"
               "    ValueError: If the arrays are not C-contiguous or do not match, the offsets\n"
               "        are not valid, both or neither of step_size and num_points are given,\n"
               "        the output would be too large to index, or n_threads is 0.\n");
    module.def("add_track_density", &add_track_density, nb::arg("points").noconvert(),
               nb::arg("offsets").noconvert(), nb::arg("world_to_voxel").noconvert(),
               nb::arg("shape").noconvert(), nb::arg("upsample_ratio"), nb::arg("max_spacing"),
               nb::arg("ends_only"), nb::arg("weights").noconvert(), nb::arg("density").noconvert(),
               nb::arg("n_threads"),
               "Adds the track density of a tractogram to density, in place.\n"
               "\n"
               "points is (P, 3) float32 and offsets (S + 1,) int64, as in a Tractogram.\n"
               "world_to_voxel (3, 4) float64 maps world coordinates (mm) to voxel\n"
               "coordinates of a grid of the given (3,) int64 shape, and density (N,) float64\n"
               "holds its voxels in C order. Each streamline is upsampled along its spline by\n"
               "upsample_ratio, or, with max_spacing > 0, each segment by the smallest ratio\n"
               "that brings its pieces below max_spacing. It then adds its weight (1 when\n"
               "weights is empty, otherwise (S,) float64) once to every voxel it passes\n"
               "through, or with ends_only to the voxel of each of its two endpoints.\n"
               "\n"
               "Raises:\n"
               "    ValueError: If the arrays are not C-contiguous or do not match, the offsets\n"
               "        are not valid, upsample_ratio is below 1, max_spacing is negative or\n"
               "        not finite, or n_threads is 0.\n");
}

} // namespace larmor::python
