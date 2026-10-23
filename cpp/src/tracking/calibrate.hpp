// The calibration of the rejection sampling of probabilistic tracking.
#pragma once

#include <vector>

#include "common/types.hpp"

namespace larmor::tracking {

/// The directions the rejection-sampling bound is taken over and the ratio applied to it.
///
/// The largest FOD amplitude on the grid, turned to the previous direction, times the ratio
/// bounds the amplitude everywhere in the cone.
struct Calibration {
    /// The directions, around +z.
    std::vector<Vector3> grid;
    /// The factor the largest amplitude on the grid is multiplied by.
    double ratio = 0.0;
};

/// Returns a hexagonal grid of unit directions around +z, out to the angle max_angle_rad, with
/// neighbouring directions about spacing_rad apart.
///
/// A hexagonal lattice in the plane is mapped to the sphere, each point along its own azimuth at
/// a polar angle proportional to its distance from the origin; the origin is +z.
///
/// References:
///   MRtrix3, the rejection-sampling calibration of tckgen's probabilistic algorithms
///   (src/dwi/tractography/algorithms/calibrator.h).
///   Tournier JD, Smith R, Raffelt D, et al. MRtrix3: A fast, flexible and open software framework
///   for medical image processing and visualisation. NeuroImage 202 (2019) 116137.
///   doi:10.1016/j.neuroimage.2019.116137
std::vector<Vector3> direction_grid(double max_angle_rad, double spacing_rad);

/// Returns the rejection-sampling calibration for an FOD of degree lmax (all degrees if full)
/// and a cone of half-angle max_angle_rad.
///
/// The grid spacing and the ratio come from the lobe of a reference fibre along +z; of the
/// spacings tried, the one with the fewest expected amplitude evaluations per step is kept.
/// max_angle_rad must be above 0.001.
///
/// References:
///   MRtrix3, the rejection-sampling calibration of tckgen's probabilistic algorithms
///   (src/dwi/tractography/algorithms/calibrator.h).
///   Tournier JD, Smith R, Raffelt D, et al. MRtrix3: A fast, flexible and open software framework
///   for medical image processing and visualisation. NeuroImage 202 (2019) 116137.
///   doi:10.1016/j.neuroimage.2019.116137
Calibration calibrate(Index lmax, bool full, double max_angle_rad);

} // namespace larmor::tracking
