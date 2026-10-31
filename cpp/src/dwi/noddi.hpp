// The NODDI model as the NODDI Matlab Toolbox computes it, and its fit to each voxel: a start from
// a dictionary of NODDI signals, then the toolbox's maximum-likelihood fit with Rician noise.
#pragma once

#include <array>
#include <vector>

#include "common/types.hpp"

namespace larmor::dwi {

/// The number of even Legendre terms in the toolbox's Watson stick, degrees 0 to 12.
constexpr int NODDI_TERMS = 7;

/// One value per even Legendre degree 0, 2, ..., 12.
using NoddiTerms = std::array<double, NODDI_TERMS>;

/// Returns the coefficients of the Watson density of concentration kappa >= 0 in the even
/// spherical harmonics, as the toolbox's WatsonSHCoeff computes them: in closed form up to
/// kappa = 30, by its fits in ln(kappa) above, and by series up to kappa = 0.1.
NoddiTerms watson_sh_coefficients(double kappa);

/// Returns the integrals over u in [-1, 1] of exp(-x u^2) P_2i(u), i = 0 to 6, for x = b d >= 0,
/// as the toolbox's LegendreGaussianIntegral computes them.
NoddiTerms legendre_gaussian_integrals(double x);

/// Returns the parallel and perpendicular diffusivities of a cylindrically symmetric tensor with
/// diffusivities d_par and d_perp, averaged over a Watson density of concentration kappa, as the
/// toolbox's WatsonHinderedDiffusionCoeff computes them.
std::array<double, 2> watson_hindered_diffusivities(double d_par, double d_perp, double kappa);

/// The fixed values of the model and of the fit.
struct NoddiSettings {
    /// The diffusivity along the neurites, mm^2/s.
    double d_par = 1.7e-3;
    /// The diffusivity of free water, mm^2/s.
    double d_iso = 3.0e-3;
    /// The neurite densities of the dictionary's atoms.
    std::vector<double> v_ic_grid;
    /// The Watson concentrations of the dictionary's atoms.
    std::vector<double> kappa_grid;
    /// The ridge of the dictionary's sparse solve, the square root of its lambda.
    double ridge = 0.0;
    /// The L1 weight of the dictionary's sparse solve.
    double l1 = 0.0;
};

/// A gradient table prepared once for the fits of every voxel.
struct NoddiProtocol {
    /// The b-value of each volume, s/mm^2, (N,).
    Vector bvals;
    /// The unit gradient direction of each volume, (N, 3).
    Matrix bvecs;
    /// Whether each volume is a b0.
    std::vector<bool> is_b0;
    /// legendre_gaussian_integrals(b d_par) of each volume, (N, NODDI_TERMS).
    Matrix integrals;
    /// exp(-b d_iso) of each volume, (N,).
    Vector free_water;
};

/// Prepares a gradient table: b-values (N,), unit directions (N, 3) and the b0 flags.
NoddiProtocol make_noddi_protocol(const VectorView& bvals, const MatrixView& bvecs,
                                  const std::vector<bool>& is_b0, const NoddiSettings& settings);

/// Returns the normalised NODDI signal of every volume for the neurite density v_ic, Watson
/// concentration kappa and free-water fraction v_iso about the unit direction mu, as the toolbox's
/// WatsonSHStickTortIsoV_B0 model computes it, (N,).
Vector noddi_signal(const NoddiProtocol& protocol, const NoddiSettings& settings, double v_ic,
                    double kappa, double v_iso, const Vector3& mu);

/// The fit of one voxel.
struct NoddiVoxelFit {
    /// The neurite density.
    double v_ic = 0.0;
    /// The Watson concentration.
    double kappa = 0.0;
    /// The free-water fraction.
    double v_iso = 0.0;
    /// The fitted mean direction, a unit vector, or zero when the voxel is not fitted.
    Vector3 mu = Vector3::Zero();
    /// False for a voxel without a positive mean b0 or with fewer than six positive
    /// diffusion-weighted measurements, which is not fitted and gets zeros.
    bool fitted = true;
};

/// Fits one voxel's signal (N,) from the starting direction mu, a unit vector.
///
/// The start comes from the dictionary of NODDI signals about mu, solved as AMICO does; the
/// parameters and the direction are then fitted by maximum likelihood with Rician noise, as the
/// toolbox's two gradient-descent stages do. A voxel whose dictionary solve finds no tissue gets
/// v_ic = kappa = 0 and the dictionary's v_iso, without the second step.
NoddiVoxelFit fit_noddi_voxel(const NoddiProtocol& protocol, const NoddiSettings& settings,
                              const VectorView& signal, const Vector3& mu);

} // namespace larmor::dwi
