#include "streamlines/resample.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "common/parallel.hpp"
#include "common/types.hpp"
#include "streamlines/spline.hpp"

namespace larmor::streamlines {

namespace {

// The bisection stops once its bracket is shorter than this fraction of the step size.
constexpr float BISECTION_TOLERANCE = 0.001F;
// Streamlines per block of the parallel loops.
constexpr Index RESAMPLE_BLOCK_SIZE = 256;

// Returns the distance between two points.
float distance(const Point& first, const Point& second) {
    const Point difference = first - second;
    return std::sqrt(difference.x() * difference.x() + difference.y() * difference.y() +
                     difference.z() * difference.z());
}

// Appends input point `index` of a streamline, at its own position.
template <typename Sink> void push_input_point(const PointsView& points, Index index, Sink& out) {
    const Index segment = std::min(index, points.rows() - 2);
    out.push_back(points.row(index).transpose(), segment, static_cast<double>(index - segment));
}

// The outcome of one bisection.
struct Bisection {
    // The point taken.
    Point point;
    // Its spline parameter.
    float mu = 0.0F;
    // The bracket left.
    float mu_lower = 0.0F;
    float mu_upper = 1.0F;
    // Whether the bracket shrank to the tolerance.
    bool converged = false;
};

// Bisects mu in [mu_lower, 1] on the spline through the four control points for the point at
// step_size from previous, until the bracket's two points are within step_size / 1000.
//
// When the bracket can no longer shrink in float32, the point taken is its upper end and the
// result is not converged.
Bisection bisect(const std::array<Point, 4>& controls, const Point& previous, float step_size,
                 float mu_lower) {
    const float tolerance = BISECTION_TOLERANCE * step_size;
    Bisection result{.point = controls[1], .mu = mu_lower, .mu_lower = mu_lower};
    Point p_lower = controls[1];
    Point p_upper = controls[2];
    bool first_try = true;
    while (true) {
        const float next_mu = 0.5F * (result.mu_lower + result.mu_upper);
        if (!first_try && next_mu == result.mu) {
            result.point = p_upper;
            result.mu = result.mu_upper;
            return result;
        }
        first_try = false;
        result.mu = next_mu;
        result.point = spline_point(next_mu, controls[0], controls[1], controls[2], controls[3]);
        if (distance(result.point, previous) < step_size) {
            result.mu_lower = next_mu;
            p_lower = result.point;
        } else {
            result.mu_upper = next_mu;
            p_upper = result.point;
        }
        if (!(distance(p_upper, p_lower) > tolerance)) {
            result.converged = true;
            return result;
        }
    }
}

// Runs one pass of resample_step_size from the extended point `middle` to the extended end point
// (1 or n), in `direction` -1 or +1.
template <typename Sink>
void step_pass(const PointsView& points, const ExtendedPoints& extended, Index middle,
               Index direction, float step_size, Sink& out) {
    const Index end = direction < 0 ? 1 : points.rows();
    Index index = middle;
    float mu_lower = 0.0F;
    Point previous = extended(middle);
    while (index != end) {
        while (index != end && distance(previous, extended(index + direction)) < step_size) {
            index += direction;
            mu_lower = 0.0F;
        }
        if (index == end) {
            break;
        }
        const Bisection found =
            bisect({extended(index - direction), extended(index), extended(index + direction),
                    extended(index + 2 * direction)},
                   previous, step_size, mu_lower);
        mu_lower = found.mu_lower;
        if (!found.converged) {
            // The next search starts beyond the point taken, so the pass always moves on; with
            // no room left on this segment, it moves to the next one.
            if (!(found.mu_upper > found.mu_lower)) {
                index += direction;
                mu_lower = 0.0F;
                continue;
            }
            mu_lower = found.mu_upper;
        }
        // A point that is not finite comes from points that are not: the pass ends there.
        if (!found.point.allFinite()) {
            break;
        }
        // Extended index k is input point k - 1; a backward segment runs from input point
        // index - 1 back to index - 2.
        if (direction > 0) {
            out.push_back(found.point, index - 1, static_cast<double>(found.mu));
        } else {
            out.push_back(found.point, index - 2, 1.0 - static_cast<double>(found.mu));
        }
        previous = found.point;
    }
    // A pass that starts at its end (forward, on 2 points) adds nothing; the others end on it.
    if (middle != end) {
        push_input_point(points, end - 1, out);
    }
}

// One block's resampled points, and their positions if the output keeps them.
//
// Segments are offset by input_start, set for each streamline, to index the whole tractogram.
struct BlockBuffer {
    bool positions = false;
    Index input_start = 0;
    std::vector<Point> points;
    std::vector<Index> segments;
    std::vector<double> fractions;

    [[nodiscard]] Index size() const { return static_cast<Index>(points.size()); }

    void push_back(const Point& point, Index segment, double fraction) {
        points.push_back(point);
        if (positions) {
            segments.push_back(input_start + segment);
            fractions.push_back(fraction);
        }
    }

    void reverse_from(Index first) {
        const auto offset = static_cast<std::ptrdiff_t>(first);
        std::reverse(points.begin() + offset, points.end());
        if (positions) {
            std::reverse(segments.begin() + offset, segments.end());
            std::reverse(fractions.begin() + offset, fractions.end());
        }
    }
};

// Writes one streamline's points straight into its rows of the output, from row first on, and
// its positions if the output keeps them; segments are offset to index the whole tractogram.
struct OutputWriter {
    ResampledTractogram& result;
    Index first;
    Index input_start;
    bool positions;
    Index count = 0;

    [[nodiscard]] Index size() const { return count; }

    void push_back(const Point& point, Index segment, double fraction) {
        const Index row = first + count;
        result.points.row(row) = point.transpose();
        if (positions) {
            result.segments(row) = input_start + segment;
            result.fractions(row) = fraction;
        }
        ++count;
    }

    void reverse_from(Index from) {
        for (Index low = first + from, high = first + count - 1; low < high; ++low, --high) {
            result.points.row(low).swap(result.points.row(high));
            if (positions) {
                std::swap(result.segments(low), result.segments(high));
                std::swap(result.fractions(low), result.fractions(high));
            }
        }
    }
};

// Resamples one streamline of the tractogram into out; one of fewer than 2 points is copied.
template <typename Sink>
void resample_streamline(const PointsView& points, Index start, Index n_points,
                         const ResampleSettings& settings, Sink& out) {
    if (n_points < 2) {
        for (Index index = 0; index < n_points; ++index) {
            out.push_back(points.row(start + index).transpose(), index, 0.0);
        }
    } else if (settings.num_points > 0) {
        resample_num_points(points.middleRows(start, n_points), settings.num_points, out);
    } else {
        resample_step_size(points.middleRows(start, n_points), settings.step_size, out);
    }
}

} // namespace

void ResampledPoints::push_back(const Point& point, Index segment, double fraction) {
    points.push_back(point);
    segments.push_back(segment);
    fractions.push_back(fraction);
}

void ResampledPoints::reverse_from(Index first) {
    const auto offset = static_cast<std::ptrdiff_t>(first);
    std::reverse(points.begin() + offset, points.end());
    std::reverse(segments.begin() + offset, segments.end());
    std::reverse(fractions.begin() + offset, fractions.end());
}

template <typename Sink>
void resample_num_points(const PointsView& points, Index num_points, Sink& out) {
    const Index n_points = points.rows();
    const auto segment_length = [&](Index index) {
        if (index >= n_points - 1) {
            return 0.0F;
        }
        return distance(points.row(index + 1).transpose(), points.row(index).transpose());
    };
    float length = 0.0F;
    for (Index index = 0; index < n_points - 1; ++index) {
        length += segment_length(index);
    }
    const ExtendedPoints extended(points);

    float cumulative = 0.0F;
    Index index = 0;
    float current_length = segment_length(0);
    for (Index output_index = 0; output_index < num_points; ++output_index) {
        const float target =
            length * static_cast<float>(output_index) / static_cast<float>(num_points - 1);
        while (index < n_points && cumulative + current_length < target) {
            cumulative += current_length;
            ++index;
            current_length = segment_length(index);
        }
        if (index >= n_points - 1) {
            push_input_point(points, n_points - 1, out);
            continue;
        }
        const float mu = current_length != 0.0F ? (target - cumulative) / current_length : 0.5F;
        out.push_back(spline_point(mu, extended(index), extended(index + 1), extended(index + 2),
                                   extended(index + 3)),
                      index, static_cast<double>(mu));
    }
}

template <typename Sink>
void resample_step_size(const PointsView& points, float step_size, Sink& out) {
    const ExtendedPoints extended(points);
    const Index middle = (points.rows() + 2) / 2;
    const Index first = out.size();
    step_pass(points, extended, middle, -1, step_size, out);
    out.reverse_from(first);
    push_input_point(points, middle - 1, out);
    step_pass(points, extended, middle, 1, step_size, out);
}

template void resample_num_points<ResampledPoints>(const PointsView& points, Index num_points,
                                                   ResampledPoints& out);
template void resample_step_size<ResampledPoints>(const PointsView& points, float step_size,
                                                  ResampledPoints& out);

namespace {

// Sizes the output arrays from the summed offsets.
void allocate(ResampledTractogram& result, bool positions) {
    const Index n_out = result.offsets(result.offsets.size() - 1);
    result.points.resize(n_out, 3);
    if (positions) {
        result.segments.resize(n_out);
        result.fractions.resize(n_out);
    }
}

// Turns counts in offsets(1..S) into offsets.
void sum_counts(ResampledTractogram& result) {
    result.offsets(0) = 0;
    for (Index streamline = 1; streamline < result.offsets.size(); ++streamline) {
        result.offsets(streamline) += result.offsets(streamline - 1);
    }
}

// Resamples every streamline to num_points, written straight into its rows of the output.
void resample_to_num_points(const PointsView& points, const OffsetsView& offsets,
                            const ResampleSettings& settings, int n_threads,
                            ResampledTractogram& result) {
    const Index n_streamlines = offsets.size() - 1;
    for (Index streamline = 0; streamline < n_streamlines; ++streamline) {
        const Index n_points = offsets(streamline + 1) - offsets(streamline);
        result.offsets(streamline + 1) = n_points < 2 ? n_points : settings.num_points;
    }
    sum_counts(result);
    allocate(result, settings.positions);
    parallel_for_blocks(n_streamlines, RESAMPLE_BLOCK_SIZE, n_threads, [&](Index begin, Index end) {
        for (Index streamline = begin; streamline < end; ++streamline) {
            const Index start = offsets(streamline);
            OutputWriter writer{.result = result,
                                .first = result.offsets(streamline),
                                .input_start = start,
                                .positions = settings.positions};
            resample_streamline(points, start, offsets(streamline + 1) - start, settings, writer);
        }
    });
}

// Resamples every streamline to step_size, each block into its own buffer, then copies the
// buffers into the output once the counts are known.
void resample_to_step_size(const PointsView& points, const OffsetsView& offsets,
                           const ResampleSettings& settings, int n_threads,
                           ResampledTractogram& result) {
    const Index n_streamlines = offsets.size() - 1;
    const Index n_blocks = (n_streamlines + RESAMPLE_BLOCK_SIZE - 1) / RESAMPLE_BLOCK_SIZE;
    std::vector<BlockBuffer> blocks(static_cast<std::size_t>(n_blocks));
    parallel_for_blocks(n_streamlines, RESAMPLE_BLOCK_SIZE, n_threads, [&](Index begin, Index end) {
        BlockBuffer& block = blocks[static_cast<std::size_t>(begin / RESAMPLE_BLOCK_SIZE)];
        block.positions = settings.positions;
        for (Index streamline = begin; streamline < end; ++streamline) {
            const Index start = offsets(streamline);
            const Index n_before = block.size();
            block.input_start = start;
            resample_streamline(points, start, offsets(streamline + 1) - start, settings, block);
            result.offsets(streamline + 1) = block.size() - n_before;
        }
    });
    sum_counts(result);
    allocate(result, settings.positions);
    parallel_for_blocks(n_blocks, 1, n_threads, [&](Index begin, Index end) {
        for (Index block_index = begin; block_index < end; ++block_index) {
            BlockBuffer& block = blocks[static_cast<std::size_t>(block_index)];
            const Index first = result.offsets(block_index * RESAMPLE_BLOCK_SIZE);
            for (Index entry = 0; entry < block.size(); ++entry) {
                const auto item = static_cast<std::size_t>(entry);
                result.points.row(first + entry) = block.points[item].transpose();
                if (settings.positions) {
                    result.segments(first + entry) = block.segments[item];
                    result.fractions(first + entry) = block.fractions[item];
                }
            }
            block = BlockBuffer{}; // frees the buffer once copied
        }
    });
}

} // namespace

ResampledTractogram resample_tractogram(const PointsView& points, const OffsetsView& offsets,
                                        const ResampleSettings& settings, int n_threads) {
    ResampledTractogram result;
    result.offsets.resize(offsets.size());
    if (settings.num_points > 0) {
        resample_to_num_points(points, offsets, settings, n_threads, result);
    } else {
        resample_to_step_size(points, offsets, settings, n_threads, result);
    }
    return result;
}

} // namespace larmor::streamlines
