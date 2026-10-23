#include <cmath>
#include <numbers>

#include <doctest/doctest.h>

#include "common/types.hpp"
#include "dwi/gibbs.hpp"

namespace {

using larmor::Index;
using larmor::Matrix;
using larmor::dwi::GibbsSettings;

// A box of height 1 from x = length / 4 + 0.3 to 3 length / 4 + 0.3, between sampling points,
// kept to the frequencies below the line's highest (a cosine series, no window), as an
// acquisition samples it: it rings.
double band_limited_box(double x, Index length) {
    const Index n_frequencies = length / 2 - 1;
    const double width = 0.5;
    double value = width;
    for (Index frequency = 1; frequency <= n_frequencies; ++frequency) {
        const auto wave_number = static_cast<double>(frequency);
        const double phase =
            2.0 * std::numbers::pi * wave_number * ((x - 0.3) / static_cast<double>(length) - 0.5);
        value += 2.0 * std::sin(std::numbers::pi * wave_number * width) /
                 (std::numbers::pi * wave_number) * std::cos(phase);
    }
    return value;
}

} // namespace

TEST_CASE("gibbs: lines unringed together or one at a time give the same result") {
    // Lines are shifted in pairs; an odd number of lines leaves the last one alone. The lines
    // are irregular, so that no two shifts ring equally up to rounding.
    const GibbsSettings settings{};
    for (const Index length : {32, 31}) {
        const Index n_lines = 3;
        Matrix lines(n_lines, length);
        for (Index line = 0; line < n_lines; ++line) {
            for (Index x = 0; x < length; ++x) {
                const auto position = static_cast<double>(x + 7 * line);
                lines(line, x) = band_limited_box(static_cast<double>(x), length) +
                                 0.1 * std::sin(0.37 * position * position);
            }
        }
        auto scratch = larmor::dwi::make_line_scratch(settings, length);
        Matrix together = lines;
        larmor::dwi::remove_ringing_along_rows(settings, together, scratch);
        for (Index line = 0; line < n_lines; ++line) {
            Matrix alone = lines.row(line);
            larmor::dwi::remove_ringing_along_rows(settings, alone, scratch);
            CHECK((alone.row(0) - together.row(line)).cwiseAbs().maxCoeff() < 1e-12);
        }
    }
}

TEST_CASE("gibbs: the ringing of a band-limited box is removed, its edges kept") {
    const Index length = 64;
    const Index n_rows = 4;
    Matrix slice(n_rows, length);
    for (Index row = 0; row < n_rows; ++row) {
        for (Index x = 0; x < length; ++x) {
            slice(row, x) = band_limited_box(static_cast<double>(x), length);
        }
    }
    const GibbsSettings settings{};
    auto scratch = larmor::dwi::make_gibbs_scratch(settings, n_rows, length);
    Matrix unringed = slice;
    larmor::dwi::remove_gibbs_ringing_slice(settings, unringed, scratch);

    // Inside the box, four voxels or more from its edges, the input oscillates around 1 by over
    // a per cent; the output is flat to a tenth of that.
    double input_ringing = 0.0;
    double output_ringing = 0.0;
    for (Index x = 20; x < 45; ++x) {
        input_ringing = std::fmax(input_ringing, std::fabs(slice(0, x) - 1.0));
        output_ringing = std::fmax(output_ringing, std::fabs(unringed(0, x) - 1.0));
    }
    CHECK(input_ringing > 0.01);
    CHECK(output_ringing < 0.1 * input_ringing);
    // The edges stay where they were: the jump across each edge is kept.
    CHECK(unringed(0, 18) - unringed(0, 14) > 0.9);
    CHECK(unringed(0, 46) - unringed(0, 50) > 0.9);
    // Every row is the same line, so every row comes out the same.
    for (Index row = 1; row < n_rows; ++row) {
        CHECK((unringed.row(row) - unringed.row(0)).cwiseAbs().maxCoeff() < 1e-12);
    }
}
