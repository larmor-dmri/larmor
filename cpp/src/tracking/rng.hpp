// Random numbers for tracking, which do not depend on which thread tracks a streamline or when.
#pragma once

#include <cstdint>

#include "common/types.hpp"

namespace larmor::tracking {

/// The random numbers of one streamline, made from a key and a counter.
///
/// The key comes from the run's seed and the streamline's global index, and the k-th draw is
/// SplitMix64's output for the key and the counter k, so it depends on (seed, index, k) only. The
/// streamline's steps draw from counter 0 up; its first-direction search draws from its own
/// range, from 2^63, so that tracking a seed again with the direction it found gives the same
/// streamline.
///
/// References:
///   Steele GL, Lea D, Flood CH. Fast splittable pseudorandom number generators. Proceedings of
///   OOPSLA 2014, 453-472. doi:10.1145/2660193.2660195
///   Vigna S. splitmix64.c, the reference implementation of SplitMix64, whose finaliser
///   constants are used here. https://prng.di.unimi.it/splitmix64.c
class Rng {
public:
    /// Returns the generator of streamline index in a run with seed run_seed, at counter 0.
    static Rng streamline(std::uint64_t run_seed, std::uint64_t index);

    /// Returns the generator of the same streamline's first-direction search, with the same key
    /// and the counter at 2^63.
    [[nodiscard]] Rng search() const;

    /// Returns a number drawn uniformly from [0, 1), a multiple of 2^-53, and advances the
    /// counter by one.
    double uniform();

    /// Returns a direction drawn uniformly on the unit sphere, from two draws.
    ///
    /// z is uniform on [-1, 1) and the azimuth phi uniform on [0, 2 pi); the direction is
    /// (r cos phi, r sin phi, z) with r = sqrt(1 - z * z).
    Vector3 unit_vector();

    /// Returns a direction drawn uniformly on the cap of half-angle max_angle_rad around +z.
    ///
    /// The polar angle is drawn by rejection, with a density proportional to its sine. sin_bound
    /// is the largest sine on [0, max_angle_rad]: sin(max_angle_rad) up to pi / 2, else 1.
    ///
    /// References:
    ///   MRtrix3, the function random_direction used by its streamline tracking.
    ///   Tournier JD, Smith R, Raffelt D, et al. MRtrix3: A fast, flexible and open software
    ///   framework for medical image processing and visualisation. NeuroImage 202 (2019) 116137.
    ///   doi:10.1016/j.neuroimage.2019.116137
    Vector3 unit_vector_in_cone(double max_angle_rad, double sin_bound);

    /// Returns the key.
    [[nodiscard]] std::uint64_t key() const { return generator_key; }

    /// Returns the counter of the next draw.
    [[nodiscard]] std::uint64_t counter() const { return generator_counter; }

private:
    Rng(std::uint64_t key, std::uint64_t counter);

    std::uint64_t generator_key;
    std::uint64_t generator_counter;
};

} // namespace larmor::tracking
