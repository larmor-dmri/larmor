// Bindings of the tracking code, with one call per batch of seeds.
//
// The algorithm's parameters come as a small Python object whose type picks the algorithm.

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include "common/types.hpp"
#include "python/arrays.hpp"
#include "python/bindings.hpp"
#include "sh/harmonics.hpp"
#include "tracking/algorithm.hpp"
#include "tracking/det.hpp"
#include "tracking/grid.hpp"
#include "tracking/prob.hpp"
#include "tracking/testing.hpp"
#include "tracking/tracker.hpp"

namespace nb = nanobind;

namespace larmor::python {

namespace {

using tracking::StopReason;
using tracking::TestAlgorithm;
using tracking::TestStep;
using tracking::TrackingAlgorithm;

// The tracking mask, if any, as its values and its world-to-voxel matrix.
using MaskArgument = std::optional<std::tuple<InArray<float, 3>, InArray<double, 2>>>;

// Returns a number as an error message shows it, in the shortest plain digits that give it back,
// or as "NaN", "inf" or "-inf".
std::string number_text(double value) {
    if (std::isnan(value)) {
        return "NaN";
    }
    if (std::isinf(value)) {
        return value > 0.0 ? "inf" : "-inf";
    }
    std::array<char, 512> buffer{};
    const auto [end, error] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                            std::chars_format::fixed);
    if (error != std::errc()) {
        return std::to_string(value);
    }
    return {buffer.data(), end};
}

// Returns a string in double quotes, with quotes, backslashes and control characters escaped.
std::string quoted(const std::string& text) {
    std::string result = "\"";
    for (const char character : text) {
        switch (character) {
        case '"':
            result += "\\\"";
            break;
        case '\\':
            result += "\\\\";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            result += character;
        }
    }
    return result + "\"";
}

// Returns the shape of a 2D array as "(rows, columns)".
template <typename Array> std::string pair_text(const Array& array) {
    return "(" + std::to_string(extent(array, 0)) + ", " + std::to_string(extent(array, 1)) + ")";
}

// Raises OverflowError for a stop that does not fit a uint8, before the step is checked.
void require_stop_byte(std::optional<std::int64_t> stop) {
    if (stop.has_value() &&
        (*stop < 0 || std::cmp_greater(*stop, std::numeric_limits<std::uint8_t>::max()))) {
        PyErr_SetString(PyExc_OverflowError, "out of range integral type conversion attempted");
        throw nb::python_error();
    }
}

// Returns the stop reason of an EndStatus code from 1 to 6.
StopReason stop_from_code(std::int64_t code) {
    if (code < 1 || code > 6) {
        throw nb::value_error(
            ("a stop is an EndStatus from 1 to 6, got " + std::to_string(code)).c_str());
    }
    return static_cast<StopReason>(code);
}

// Makes the test algorithm from its step name and stop.
TestAlgorithm make_test_algorithm(const std::string& step, std::optional<std::int64_t> stop,
                                  double search_refuses_past_x) {
    require_stop_byte(stop);
    const double past_x = search_refuses_past_x;
    if (!stop.has_value()) {
        if (step == "straight") {
            return {TestStep::Straight, StopReason::LowFod, past_x};
        }
        if (step == "quarter_turn") {
            return {TestStep::QuarterTurn, StopReason::LowFod, past_x};
        }
        if (step == "wander") {
            return {TestStep::Wander, StopReason::LowFod, past_x};
        }
    } else if (step == "refuse") {
        return {TestStep::Refuse, stop_from_code(*stop), past_x};
    }
    const std::string stop_text =
        stop.has_value() ? "Some(" + std::to_string(*stop) + ")" : std::string("None");
    throw nb::value_error(("expected straight, quarter_turn, wander, or refuse with a stop; got " +
                           quoted(step) + " with stop " + stop_text)
                              .c_str());
}

// Copies a (4, 4) world-to-voxel matrix of any memory layout.
tracking::InverseAffine inverse_affine(const InArray<double, 2>& matrix, const std::string& name) {
    if (extent(matrix, 0) != 4 || extent(matrix, 1) != 4) {
        throw nb::value_error((name + " must be (4, 4), got " + pair_text(matrix)).c_str());
    }
    tracking::InverseAffine inverse;
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            inverse(static_cast<Index>(row), static_cast<Index>(column)) = matrix(row, column);
        }
    }
    return inverse;
}

// Returns the shape of a voxel grid.
tracking::GridShape grid_shape(Index n_x, Index n_y, Index n_z) {
    return {n_x, n_y, n_z};
}

// Returns the lmax of an SH series of n_coefficients, for the algorithms that evaluate the FOD.
//
// Raises ValueError unless the series is whole, lmax is at most sh::MAX_LMAX, and lmax is even
// for a symmetric series.
Index lmax_of(Index n_coefficients, bool full) {
    const auto root = [](Index value) {
        return static_cast<Index>(std::sqrt(static_cast<double>(value)));
    };
    const Index lmax = full ? std::max<Index>(root(n_coefficients) - 1, 0)
                            : std::max<Index>(root(8 * n_coefficients + 1) - 3, 0) / 2;
    if (lmax > sh::MAX_LMAX || (!full && lmax % 2 != 0)) {
        throw nb::value_error(("lmax must be at most " + std::to_string(sh::MAX_LMAX) +
                               ", and even unless full, got " + std::to_string(lmax))
                                  .c_str());
    }
    if (sh::n_coeffs(lmax, full) != n_coefficients) {
        throw nb::value_error(("lmax " + std::to_string(lmax) + " has " +
                               std::to_string(sh::n_coeffs(lmax, full)) + " coefficients, got " +
                               std::to_string(n_coefficients))
                                  .c_str());
    }
    return lmax;
}

// The parameters of deterministic tracking.
struct DetParams {
    // The FOD amplitude below which a streamline stops.
    double cutoff;
    // The largest turn between steps, in degrees.
    double max_angle_deg;
};

// The parameters of probabilistic tracking.
struct ProbParams {
    // The FOD amplitude a candidate direction must exceed.
    double cutoff;
    // The half-angle of the cone candidates are drawn in, in degrees.
    double max_angle_deg;
    // Candidates per step before the streamline stops.
    Index trials;
    // Random directions tried for a seed's first direction.
    Index attempts;
};

// Raises ValueError unless cutoff is non-negative and finite.
void require_cutoff(double cutoff) {
    if (!std::isfinite(cutoff) || cutoff < 0.0) {
        throw nb::value_error(
            ("the cutoff must be non-negative and finite, got " + number_text(cutoff)).c_str());
    }
}

// Checks and returns the parameters of deterministic tracking.
DetParams make_det_params(double cutoff, double max_angle) {
    require_cutoff(cutoff);
    // Written so that a NaN angle is refused.
    const bool angle_in_range = max_angle > 0.0 && max_angle <= 180.0;
    if (!angle_in_range) {
        throw nb::value_error(
            ("the maximum angle must be in (0, 180], got " + number_text(max_angle)).c_str());
    }
    return {.cutoff = cutoff, .max_angle_deg = max_angle};
}

// Checks and returns the parameters of probabilistic tracking.
ProbParams make_prob_params(double cutoff, double max_angle, Index trials, Index attempts) {
    require_cutoff(cutoff);
    // The calibration samples the cone every 0.001 rad and needs two samples. Written so that a
    // NaN angle is refused.
    const bool angle_in_range =
        max_angle * (std::numbers::pi / 180.0) > 0.001 && max_angle <= 180.0;
    if (!angle_in_range) {
        throw nb::value_error(("the maximum angle must be above 0.001 rad (0.057 degrees) and at "
                               "most 180, got " +
                               number_text(max_angle))
                                  .c_str());
    }
    if (trials < 1 || attempts < 1) {
        throw nb::value_error(("trials and attempts must be at least 1, got " +
                               std::to_string(trials) + " and " + std::to_string(attempts))
                                  .c_str());
    }
    return {.cutoff = cutoff, .max_angle_deg = max_angle, .trials = trials, .attempts = attempts};
}

// The algorithm a batch runs.
struct BatchAlgorithm {
    // The algorithm.
    std::unique_ptr<const TrackingAlgorithm> algorithm;
    // The number of directions its calibration sampled, 0 for algorithms without one.
    std::uint64_t calibration_directions = 0;
};

// Raises TypeError unless params is the parameters of an algorithm track_batch runs.
void require_algorithm_params(nb::handle params) {
    if (!nb::isinstance<DetParams>(params) && !nb::isinstance<ProbParams>(params) &&
        !nb::isinstance<TestAlgorithm>(params)) {
        throw nb::type_error(
            ("expected DetParams, ProbParams or TestAlgorithm for algorithm, got " +
             std::string(nb::inst_name(params).c_str()))
                .c_str());
    }
}

// Builds the algorithm of a batch from its parameters, whose type picks it.
//
// The algorithms that evaluate the FOD check its series first. Call it while holding the Python
// lock; it releases the lock during the calibration of the probabilistic algorithm.
BatchAlgorithm make_batch_algorithm(nb::handle params, Index n_coefficients, bool full) {
    if (nb::isinstance<DetParams>(params)) {
        const auto& det = nb::cast<const DetParams&>(params);
        const Index lmax = lmax_of(n_coefficients, full);
        return {.algorithm = std::make_unique<tracking::DeterministicAlgorithm>(
                    det.cutoff, det.max_angle_deg, lmax, full),
                .calibration_directions = 0};
    }
    if (nb::isinstance<ProbParams>(params)) {
        const ProbParams prob = nb::cast<const ProbParams&>(params);
        const Index lmax = lmax_of(n_coefficients, full);
        std::unique_ptr<tracking::ProbabilisticAlgorithm> algorithm;
        {
            const nb::gil_scoped_release release;
            algorithm = std::make_unique<tracking::ProbabilisticAlgorithm>(
                prob.cutoff, prob.max_angle_deg, lmax, full, prob.trials, prob.attempts);
        }
        const auto n_directions = static_cast<std::uint64_t>(algorithm->calibration_directions());
        return {.algorithm = std::move(algorithm), .calibration_directions = n_directions};
    }
    return {.algorithm = std::make_unique<TestAlgorithm>(nb::cast<const TestAlgorithm&>(params)),
            .calibration_directions = 0};
}

// Tracks a batch of seeds into the rows of `positions`.
nb::tuple track_batch(const InArray<float, 4>& fod, const InArray<double, 2>& fod_inv_affine,
                      const MaskArgument& mask, const InArray<double, 2>& seeds,
                      const std::optional<InArray<double, 2>>& directions,
                      const InArray<std::int64_t, 1>& indices, std::uint64_t run_seed,
                      double step_size, Index max_steps, bool bidirectional,
                      nb::handle algorithm_params, const InOutArray<float, 3>& positions,
                      int n_threads, bool full) {
    require_algorithm_params(algorithm_params);
    require_contiguous(fod);
    const tracking::InverseAffine fod_inverse = inverse_affine(fod_inv_affine, "fod_inv_affine");
    std::optional<tracking::Mask> tracking_mask;
    if (mask.has_value()) {
        const auto& [mask_values, mask_inverse] = *mask;
        require_contiguous(mask_values);
        tracking_mask.emplace(
            mask_values.data(),
            grid_shape(extent(mask_values, 0), extent(mask_values, 1), extent(mask_values, 2)),
            inverse_affine(mask_inverse, "mask inverse affine"));
    }

    const Index n_seeds = extent(seeds, 0);
    const bool directions_mismatch = directions.has_value() && (extent(*directions, 0) != n_seeds ||
                                                                extent(*directions, 1) != 3);
    if (extent(seeds, 1) != 3 || directions_mismatch || extent(indices, 0) != n_seeds) {
        const std::string directions_text =
            directions.has_value() ? "Some(" + pair_text(*directions) + ")" : std::string("None");
        throw nb::value_error(("expected seeds and directions (n, 3) and indices (n,): got seeds " +
                               pair_text(seeds) + ", directions " + directions_text + ", indices " +
                               std::to_string(extent(indices, 0)))
                                  .c_str());
    }
    for (Index item = 0; item < n_seeds; ++item) {
        if (indices(static_cast<std::size_t>(item)) < 0) {
            throw nb::value_error("seed indices must be non-negative");
        }
    }
    if (!std::isfinite(step_size) || step_size <= 0.0 || max_steps <= 0) {
        throw nb::value_error(("need a positive step size and max_steps, got " +
                               number_text(step_size) + " and " + std::to_string(max_steps))
                                  .c_str());
    }

    require_contiguous(positions, "positions");
    if (extent(positions, 0) < n_seeds || extent(positions, 1) != max_steps + 1 ||
        extent(positions, 2) != 3) {
        throw nb::value_error(("positions must be (at least " + std::to_string(n_seeds) + ", " +
                               std::to_string(max_steps + 1) + ", 3), got (" +
                               std::to_string(extent(positions, 0)) + ", " +
                               std::to_string(extent(positions, 1)) + ", " +
                               std::to_string(extent(positions, 2)) + ")")
                                  .c_str());
    }
    require_threads(n_threads);

    std::vector<tracking::Seed> batch;
    batch.reserve(static_cast<std::size_t>(n_seeds));
    for (Index row = 0; row < n_seeds; ++row) {
        const auto item = static_cast<std::size_t>(row);
        std::optional<Vector3> direction;
        if (directions.has_value()) {
            const auto& given = *directions;
            direction = Vector3(given(item, 0), given(item, 1), given(item, 2));
        }
        batch.push_back({.position = Vector3(seeds(item, 0), seeds(item, 1), seeds(item, 2)),
                         .direction = direction,
                         .index = static_cast<std::uint64_t>(indices(item))});
    }
    const tracking::TrackingSetup setup{
        .fod =
            tracking::Field(fod.data(), grid_shape(extent(fod, 0), extent(fod, 1), extent(fod, 2)),
                            extent(fod, 3), fod_inverse),
        .mask = tracking_mask,
        .step_size_mm = step_size,
        .max_steps = max_steps,
        .bidirectional = bidirectional,
        .run_seed = run_seed};
    auto buffer = rows_view(positions);
    auto rows = buffer.topRows(n_seeds);

    const BatchAlgorithm batch_algorithm =
        make_batch_algorithm(algorithm_params, extent(fod, 3), full);
    std::vector<tracking::Track> tracks;
    {
        const nb::gil_scoped_release release;
        tracks = tracking::track_batch(setup, *batch_algorithm.algorithm, batch, rows, n_threads);
    }

    ColumnVector<std::int64_t> counts(n_seeds);
    RowMatrix<std::uint8_t> status(n_seeds, 2);
    RowMatrix<double> followed(n_seeds, 3);
    tracking::SamplingStats total;
    for (Index item = 0; item < n_seeds; ++item) {
        const tracking::Track& track = tracks[static_cast<std::size_t>(item)];
        counts(item) = static_cast<std::int64_t>(track.n_points);
        status(item, 0) = tracking::end_code(track.ends[0]);
        status(item, 1) = tracking::end_code(track.ends[1]);
        followed.row(item) = track.direction.transpose();
        total.add(track.stats);
    }
    nb::dict stats;
    stats["sampled_steps"] = total.sampled_steps;
    stats["accepted_index_sum"] = total.accepted_index_sum;
    stats["truncations"] = total.truncations;
    stats["max_truncation"] = total.max_truncation;
    stats["calibration_directions"] = batch_algorithm.calibration_directions;
    return nb::make_tuple(to_numpy(std::move(counts)), to_numpy(std::move(status)),
                          to_numpy(std::move(followed)), stats);
}

} // namespace

void register_tracking(nb::module_& module) {
    nb::class_<DetParams>(module, "DetParams",
                          "Parameters of deterministic tracking.\n"
                          "\n"
                          "lmax comes from the FOD.\n")
        .def(
            "__init__",
            [](DetParams* self, double cutoff, double max_angle) {
                new (self) DetParams(make_det_params(cutoff, max_angle));
            },
            nb::arg("cutoff"), nb::arg("max_angle"),
            "Checks and keeps the parameters.\n"
            "\n"
            "cutoff is the FOD amplitude below which a streamline stops, and max_angle\n"
            "the largest turn between steps, in degrees.\n"
            "\n"
            "Raises:\n"
            "    ValueError: If cutoff is negative or not finite, or max_angle is not in\n"
            "        (0, 180].\n");

    nb::class_<ProbParams>(module, "ProbParams",
                           "Parameters of probabilistic tracking.\n"
                           "\n"
                           "lmax comes from the FOD; the rejection-sampling bound is calibrated "
                           "from it.\n")
        .def(
            "__init__",
            [](ProbParams* self, double cutoff, double max_angle, Index trials, Index attempts) {
                new (self) ProbParams(make_prob_params(cutoff, max_angle, trials, attempts));
            },
            nb::arg("cutoff"), nb::arg("max_angle"), nb::arg("trials") = 1000,
            nb::arg("attempts") = 1000,
            "Checks and keeps the parameters.\n"
            "\n"
            "cutoff is the FOD amplitude a candidate direction must exceed, max_angle the\n"
            "half-angle of the cone candidates are drawn in, in degrees, trials the number\n"
            "of candidates per step before NO_DIRECTION, and attempts the number of random\n"
            "directions tried for a seed's first direction.\n"
            "\n"
            "Raises:\n"
            "    ValueError: If cutoff is negative or not finite, max_angle is not above\n"
            "        0.001 rad or above 180, or trials or attempts is 0.\n");

    nb::class_<TestAlgorithm>(module, "TestAlgorithm",
                              "A dummy algorithm for testing the tracking loop.\n"
                              "\n"
                              "step is \"straight\", \"quarter_turn\" (90 degrees about z each\n"
                              "step), \"wander\" (a random walk that draws three numbers per\n"
                              "step), or \"refuse\" with stop, an EndStatus from 1 to 6 that\n"
                              "every step returns. The first-direction search draws a random\n"
                              "unit vector and fails where the seed's x is above\n"
                              "search_refuses_past_x.\n")
        .def(
            "__init__",
            [](TestAlgorithm* self, const std::string& step, std::optional<std::int64_t> stop,
               double search_refuses_past_x) {
                new (self) TestAlgorithm(make_test_algorithm(step, stop, search_refuses_past_x));
            },
            nb::arg("step"), nb::arg("stop") = nb::none(),
            nb::arg("search_refuses_past_x") = std::numeric_limits<double>::infinity(),
            "Builds the dummy algorithm.\n"
            "\n"
            "Raises:\n"
            "    ValueError: If step is unknown, or stop is missing for \"refuse\", given\n"
            "        otherwise, or not in 1..6.\n");

    module.def(
        "track_batch", &track_batch, nb::arg("fod").noconvert(),
        nb::arg("fod_inv_affine").noconvert(), nb::arg("mask").noconvert().none(),
        nb::arg("seeds").noconvert(), nb::arg("directions").noconvert().none(),
        nb::arg("indices").noconvert(), nb::arg("run_seed"), nb::arg("step_size"),
        nb::arg("max_steps"), nb::arg("bidirectional"), nb::arg("algorithm"),
        nb::arg("positions").noconvert(), nb::arg("n_threads"), nb::arg("full") = false,
        // The signature names the types of algorithm and of the result, which the C++ types
        // leave out; it becomes the line in _core.pyi.
        nb::sig("def track_batch("
                "fod: ndarray[dtype=float32, shape=(*, *, *, *), device='cpu', writable=False], "
                "fod_inv_affine: ndarray[dtype=float64, shape=(*, *), device='cpu', "
                "writable=False], "
                "mask: tuple[ndarray[dtype=float32, shape=(*, *, *), device='cpu', "
                "writable=False], ndarray[dtype=float64, shape=(*, *), device='cpu', "
                "writable=False]] | None, "
                "seeds: ndarray[dtype=float64, shape=(*, *), device='cpu', writable=False], "
                "directions: ndarray[dtype=float64, shape=(*, *), device='cpu', "
                "writable=False] | None, "
                "indices: ndarray[dtype=int64, shape=(*), device='cpu', writable=False], "
                "run_seed: int, step_size: float, max_steps: int, bidirectional: bool, "
                "algorithm: DetParams | ProbParams | TestAlgorithm, "
                "positions: ndarray[dtype=float32, shape=(*, *, *), device='cpu'], "
                "n_threads: int, full: bool = False) -> "
                "tuple[numpy.ndarray[dtype=int64, shape=(*)], "
                "numpy.ndarray[dtype=uint8, shape=(*, *)], "
                "numpy.ndarray[dtype=float64, shape=(*, *)], dict[str, float]]"),
        "Tracks a batch of seeds in parallel into the rows of positions.\n"
        "\n"
        "fod is (X, Y, Z, C) float32 in the tournier basis, C-contiguous, with all SH\n"
        "degrees if full (an asymmetric FOD), else the even ones. mask, if any, is an\n"
        "(X, Y, Z) float32 array and its (4, 4) world-to-voxel matrix. seeds are (n, 3)\n"
        "world positions, directions (n, 3) unit vectors followed as given, or None for\n"
        "the algorithm's first-direction search, and indices (n,) the global indices that\n"
        "key the random numbers. positions is (batch, max_steps + 1, 3) float32 with\n"
        "batch >= n, C-contiguous; row i receives seed i's points, and rows past n are\n"
        "untouched.\n"
        "\n"
        "Returns each seed's number of points (n,), 0 where it did not start, the\n"
        "EndStatus of the start and the end of its row (n, 2), the direction followed,\n"
        "given or found (n, 3), and the batch's sampling counters summed over its\n"
        "streamlines: sampled_steps, accepted_index_sum, truncations, max_truncation and\n"
        "calibration_directions, all 0 for algorithms that do not sample.\n"
        "\n"
        "Raises:\n"
        "    ValueError: If an array is not C-contiguous or the shapes do not match, an\n"
        "        index is negative, step_size or max_steps is not positive, C is not a\n"
        "        whole series up to lmax 20 in the layout full says (det, prob), or\n"
        "        n_threads is 0.\n");
}

} // namespace larmor::python
