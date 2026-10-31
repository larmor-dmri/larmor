// Closed-form signal attenuations and relaxation weights of the tissue compartments.
//
// Units: b in s/mm^2, diffusivities in mm^2/s, radii in um, times in ms. The cosine is the one
// between the unit gradient direction and the unit fibre direction.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "common/types.hpp"

namespace larmor::simulation {

/// The compartment models. The values are the codes the Python side passes.
enum class Model : std::uint8_t {
    Stick = 0,
    Zeppelin = 1,
    Ball = 2,
    PowderStick = 3,
    Sphere = 4,
    WatsonStick = 5,
    Cylinder = 6,
    Dot = 7,
};

/// The number of models.
constexpr int N_MODELS = 8;

/// The diffusion gradient timings of a pulsed-gradient spin echo, in ms.
struct PulseTiming {
    /// delta, the duration of each gradient pulse.
    double small_delta_ms = 0.0;
    /// Delta, the time between the starts of the two pulses.
    double big_delta_ms = 0.0;
};

/// One compartment's model and parameters.
struct Compartment {
    /// The model.
    Model model = Model::Ball;
    /// The diffusivity (along the fibre for the stick, the zeppelin, the Watson-dispersed sticks
    /// and the cylinder, which also uses it inside), mm^2/s; unused by the dot.
    double diffusivity = 0.0;
    /// The zeppelin's diffusivity across the fibre, mm^2/s; unused by the other models.
    double perpendicular_diffusivity = 0.0;
    /// The radius of the sphere or the cylinder, um; unused by the other models.
    double radius_um = 0.0;
    /// The Watson concentration kappa >= 0 of the dispersed sticks; unused by the other models.
    double kappa = 0.0;
};

/// Returns whether the model's attenuation depends on the fibre direction.
constexpr bool is_oriented(Model model) {
    return model == Model::Stick || model == Model::Zeppelin || model == Model::WatsonStick ||
           model == Model::Cylinder;
}

/// Returns the attenuation of a stick, exp(-b D cosine^2).
double stick(double b_value, double diffusivity, double cosine);

/// Returns the attenuation of a zeppelin, an axially symmetric tensor,
/// exp(-b (D_perp + (D_par - D_perp) cosine^2)).
double zeppelin(double b_value, double parallel, double perpendicular, double cosine);

/// Returns the attenuation of a ball, isotropic free diffusion, exp(-b D).
double ball(double b_value, double diffusivity);

/// Returns the attenuation of a stick averaged over all directions,
/// sqrt(pi) / (2 sqrt(x)) erf(sqrt(x)) with x = b D.
double powder_stick(double b_value, double diffusivity);

/// The number of roots in sphere_roots.
constexpr int N_SPHERE_ROOTS = 4000;

/// Returns the first N_SPHERE_ROOTS positive roots x_m of j1'(x) = 0, with j1 the spherical
/// Bessel function of order 1 (2.0816, 5.9404, 9.2058, ...), computed on the first call.
const std::array<double, N_SPHERE_ROOTS>& sphere_roots();

/// Returns the attenuation of restricted diffusion in an impermeable sphere, in the Gaussian
/// phase approximation for rectangular pulses:
///
///   ln E = -2 gamma^2 G^2 sum_m 1 / (a_m^2 (a_m^2 R^2 - 2))
///          [2 delta / (a_m^2 D) - (2 + e^{-a_m^2 D (Delta - delta)} - 2 e^{-a_m^2 D delta}
///           - 2 e^{-a_m^2 D Delta} + e^{-a_m^2 D (Delta + delta)}) / (a_m^2 D)^2]
///
/// with a_m = x_m / R and gamma^2 G^2 = b / (delta^2 (Delta - delta / 3)). Large spheres tend to
/// free diffusion; the result is accurate up to radii of about 2 mm. Returns 1 for b = 0 or
/// D = 0.
double sphere(double b_value, double diffusivity, double radius_um, const PulseTiming& timing);

/// Returns the attenuation of sticks with diffusivity D dispersed about the fibre direction mu by
/// a Watson density (Zhang et al. 2012, Eqs. 2 and 3):
///
///   E = integral over the sphere of f(n) exp(-b D (g . n)^2) dn,
///   f(n) = M(1/2, 3/2, kappa)^-1 exp(kappa (mu . n)^2),
///
/// with M the confluent hypergeometric function and dn normalised to 1 over the sphere. The
/// integral is reduced to one dimension and computed by quadrature. kappa = 0 gives the
/// powder-averaged stick and kappa -> infinity the stick.
double watson_stick(double b_value, double diffusivity, double kappa, double cosine);

/// The number of roots in cylinder_roots.
constexpr int N_CYLINDER_ROOTS = 4000;

/// Returns the first N_CYLINDER_ROOTS positive roots x_m of J1'(x) = 0, with J1 the Bessel
/// function of the first kind of order 1 (1.8412, 5.3314, 8.5363, ...), computed on the first
/// call.
const std::array<double, N_CYLINDER_ROOTS>& cylinder_roots();

/// Returns the attenuation of restricted diffusion across an impermeable cylinder of radius R, in
/// the Gaussian phase approximation for rectangular pulses, and free diffusion along it (Van
/// Gelderen et al. 1994; the form of De Santis et al. 2016, Eq. 3):
///
///   ln E_perp = -2 gamma^2 G_perp^2 sum_m [2 D a_m^2 delta - 2 + 2 e^{-D a_m^2 delta}
///               + 2 e^{-D a_m^2 Delta} - e^{-D a_m^2 (Delta - delta)} - e^{-D a_m^2 (Delta +
///               delta)}] / (D^2 a_m^6 (R^2 a_m^2 - 1)),
///
/// with a_m = x_m / R, the roots of J1'(a_m R) = 0, and the gradient across the fibre
/// gamma^2 G_perp^2 = b (1 - cosine^2) / (delta^2 (Delta - delta / 3)). E is
/// exp(-b D cosine^2) E_perp.
double cylinder(double b_value, double diffusivity, double radius_um, double cosine,
                const PulseTiming& timing);

/// Returns the attenuation of a compartment for one gradient. The cosine is ignored by the models
/// without orientation, and the pulse timings are used only by the sphere and the cylinder.
double attenuation(const Compartment& compartment, double b_value, double cosine,
                   const PulseTiming& timing);

/// An oriented compartment's attenuation for one b-value and timing, as a function of the
/// cosine between the gradient and the fibre.
///
/// The stick, the zeppelin and the cylinder keep ln E = offset + slope cosine^2. The
/// Watson-dispersed sticks keep E on an even grid of cosine^2 and interpolate it with a cubic.
struct AngularProfile {
    /// ln E at cosine 0.
    double offset = 0.0;
    /// The change of ln E per unit of cosine^2.
    double slope = 0.0;
    /// E at cosine^2 = k / (size - 1); empty for the models with ln E linear in cosine^2.
    std::vector<double> table;

    /// Returns E at this cosine.
    [[nodiscard]] double operator()(double cosine) const;
};

/// The number of points of an AngularProfile's table.
constexpr int N_PROFILE_POINTS = 513;

/// Returns the profile of an oriented compartment at one b-value and timing.
AngularProfile angular_profile(const Compartment& compartment, double b_value,
                               const PulseTiming& timing);

/// Returns the relaxation weight PD (1 - exp(-TR / T1)) exp(-TE / T_decay), where T_decay is T2
/// for a spin echo and T2* for a gradient echo. Times are in ms.
double relaxation_weight(double proton_density, double t1_ms, double decay_ms, double tr_ms,
                         double te_ms);

} // namespace larmor::simulation
