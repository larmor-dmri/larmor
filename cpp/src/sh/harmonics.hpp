// Real spherical harmonics in the tournier basis, and the amplitude of an SH series on the
// sphere with its derivatives.
//
// A series holds the even degrees of a symmetric FOD, or all degrees (full) of an asymmetric one.
#pragma once

#include <Eigen/Core>

#include "common/types.hpp"

namespace larmor::sh {

/// The largest lmax supported.
constexpr Index MAX_LMAX = 20;

/// Returns the number of coefficients of a series up to lmax.
constexpr Index n_coeffs(Index lmax, bool full) {
    return full ? (lmax + 1) * (lmax + 1) : (lmax + 1) * (lmax + 2) / 2;
}

/// The number of coefficients of the largest series, all degrees up to MAX_LMAX.
constexpr Index MAX_COEFFS = n_coeffs(MAX_LMAX, true);

/// Returns the position of the coefficient of (degree, order) in a series, with order from
/// -degree to degree.
constexpr Index coefficient_index(Index degree, Index order, bool full) {
    return full ? degree * degree + degree + order : degree * (degree + 1) / 2 + order;
}

/// The normalised associated Legendre values P(degree, order) at one elevation, with the
/// Condon-Shortley sign, and their derivatives in theta.
///
/// They are normalised so that P(degree, order) * sqrt(2) * cos(order * phi) is an orthonormal
/// real harmonic.
struct Legendre {
    /// A table indexed by (degree, order).
    using Table = Eigen::Matrix<double, MAX_LMAX + 1, MAX_LMAX + 1, Eigen::RowMajor>;

    /// Computes the table up to lmax at the elevation theta, in radians from +z.
    static Legendre at_elevation(Index lmax, double theta);

    /// Computes the table up to lmax from cos theta and sin theta (sin theta >= 0).
    static Legendre from_cos_sin(Index lmax, double cos_theta, double sin_theta);

    /// Returns dP(degree, order) / dtheta, for 0 <= order <= degree <= lmax.
    [[nodiscard]] double d_theta(Index degree, Index order) const;

    /// Returns d2P(degree, order) / dtheta2, for 0 <= order <= degree <= lmax.
    [[nodiscard]] double d2_theta(Index degree, Index order) const;

    /// P(degree, order)(cos theta) for 0 <= order <= degree <= lmax, and 0 elsewhere.
    Table values = Table::Zero();
};

/// The amplitude of an SH series at one direction and its derivatives on the sphere.
///
/// The phi-derivatives are divided by sin theta once per phi, so that every first derivative is
/// per radian on the sphere and every second derivative per square radian.
struct Derivatives {
    double value = 0.0;         // f
    double d_theta = 0.0;       // df/dtheta
    double d_phi = 0.0;         // df/dphi / sin theta
    double d_theta_theta = 0.0; // d2f/dtheta2
    double d_theta_phi = 0.0;   // d2f/dtheta dphi / sin theta
    double d_phi_phi = 0.0;     // d2f/dphi2 / sin^2 theta
};

/// Computes the amplitude of an SH series and its derivatives on the sphere at elevation theta
/// (from +z) and azimuth phi (from +x), in radians.
///
/// coefficients holds at least n_coeffs(lmax, full) values in the tournier basis, with
/// lmax <= MAX_LMAX. Where sin theta < 1e-4, df/dphi / sin theta takes its limit at the pole and
/// the mixed and phi-phi derivatives are 0.
///
/// References:
///   MRtrix3, Math::SH::derivatives (core/math/SH.h, version 3.0.8).
///   Tournier JD, Smith R, Raffelt D, et al. MRtrix3: A fast, flexible and open software framework
///   for medical image processing and visualisation. NeuroImage 202 (2019) 116137.
///   doi:10.1016/j.neuroimage.2019.116137
Derivatives derivatives(const VectorView& coefficients, Index lmax, bool full, double theta,
                        double phi);

/// Returns the amplitude of an SH series along the unit vector `direction`.
///
/// coefficients and lmax are as for derivatives.
double amplitude(const VectorView& coefficients, Index lmax, bool full, const Vector3& direction);

} // namespace larmor::sh
