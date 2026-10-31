// Computes the NODDI signal and fits it to each voxel, from a start found in a dictionary of NODDI
// signals, by maximum likelihood with Rician noise.
//
// References: Zhang et al., NeuroImage 61 (2012) 1000-1016; NODDI Matlab Toolbox 1.05; Daducci et
// al., NeuroImage 105 (2015) 32-44.
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
/// spherical harmonics: by series up to kappa = 0.1, in closed form up to kappa = 30, and by fits
/// in ln(kappa) above.
NoddiTerms watson_sh_coefficients(double kappa);

/// Returns the integrals over u in [-1, 1] of exp(-x u^2) P_2i(u), i = 0 to 6, for x = b d >= 0.
NoddiTerms legendre_gaussian_integrals(double x);

/// Returns the parallel and perpendicular diffusivities of a cylindrically symmetric tensor with
/// diffusivities d_par and d_perp, averaged over a Watson density of concentration kappa.
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
/// concentration kappa and free-water fraction v_iso about the unit direction mu, (N,).
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
    /// False for a voxel that is not fitted and gets zeros: one with a signal that is not finite,
    /// without a positive mean b0, with fewer than six positive diffusion-weighted measurements,
    /// or with a starting direction that is zero or not finite.
    bool fitted = true;
};

/// Fits one voxel's signal (N,) from the starting direction mu, which is normalised.
///
/// The start comes from the dictionary of NODDI signals about mu; the parameters and the direction
/// are then fitted by maximum likelihood with Rician noise, first the fractions and then all of
/// them. A voxel whose dictionary solve finds no tissue gets v_ic = kappa = 0 and the
/// dictionary's v_iso, without the second step.
NoddiVoxelFit fit_noddi_voxel(const NoddiProtocol& protocol, const NoddiSettings& settings,
                              const VectorView& signal, const Vector3& mu);

} // namespace larmor::dwi
