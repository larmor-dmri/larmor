#include "simulation/compartments/compartments.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>

#include "common/maths.hpp"

namespace larmor::simulation {

namespace {

// b D below which the powder-averaged stick uses its Taylor series,
// 1 - x / 3 + x^2 / 10 - x^3 / 42.
constexpr double POWDER_SERIES_LIMIT = 1e-4;

// sum_m 1 / (x_m^4 (x_m^2 - 2)) over all sphere roots.
constexpr double SPHERE_LONG_PULSE_SUM = 4.0 / 175.0;
// Minimum number of terms of the sphere's and the cylinder's series.
constexpr std::size_t MIN_SPHERE_TERMS = 60;
// The series then run until a_m^2 D delta >= MIN_SPHERE_DECAY^2, where the exponentials are
// negligible.
constexpr double MIN_SPHERE_DECAY = 6.0;

// Number of bisection steps for a sphere root.
constexpr int BISECTION_STEPS = 60;

// sum_m 1 / (x_m^4 (x_m^2 - 1)) over all cylinder roots.
constexpr double CYLINDER_LONG_PULSE_SUM = 7.0 / 192.0;
// Number of cylinder roots refined by Newton's method; McMahon's expansion alone gives the rest.
constexpr int N_NEWTON_ROOTS = 200;
// Number of Newton steps per cylinder root.
constexpr int NEWTON_STEPS = 8;

// The Watson integrals in u = 1 - t stop at u = WATSON_CUTOFF / rate, where the integrand is
// below exp(-WATSON_CUTOFF).
constexpr double WATSON_CUTOFF = 40.0;
// Number of panels of the Watson integrals, [0, U 2^-(P-1)], ..., [U / 2, U].
constexpr int WATSON_PANELS = 12;
// Number of Gauss-Legendre points per panel.
constexpr int GAUSS_POINTS = 16;

// Returns (x^2 - 2) sin x + 2 x cos x, which is zero where j1'(x) = 0.
double sphere_root_equation(double x) {
    return (x * x - 2.0) * std::sin(x) + 2.0 * x * std::cos(x);
}

// Finds the sphere roots by bisection; root m lies in ((m - 1/2) pi, m pi).
std::array<double, N_SPHERE_ROOTS> find_sphere_roots() {
    std::array<double, N_SPHERE_ROOTS> roots{};
    for (int root_number = 1; root_number <= N_SPHERE_ROOTS; ++root_number) {
        double low = (root_number - 0.5) * std::numbers::pi;
        double high = root_number * std::numbers::pi;
        const double low_value = sphere_root_equation(low);
        for (int step = 0; step < BISECTION_STEPS; ++step) {
            const double middle = 0.5 * (low + high);
            if ((sphere_root_equation(middle) > 0.0) == (low_value > 0.0)) {
                low = middle;
            } else {
                high = middle;
            }
        }
        roots.at(static_cast<std::size_t>(root_number - 1)) = 0.5 * (low + high);
    }
    return roots;
}

// The Bessel function J1 and its first two derivatives at one point.
struct BesselJ1 {
    // J1(x).
    double value;
    // J1'(x).
    double first;
    // J1''(x).
    double second;
};

// Returns J1(x) and its derivatives from J1(x) = (1/pi) int_0^pi cos(tau - x sin tau) dtau, by
// the trapezoidal rule.
BesselJ1 bessel_j1(double x) {
    const int points = static_cast<int>(1.1 * x) + 64;
    const double step = std::numbers::pi / points;
    double value = 0.0;
    double first = 0.0;
    double second = 0.0;
    for (int point = 0; point <= points; ++point) {
        const double tau = point * step;
        const double weight = (point == 0 || point == points) ? 0.5 : 1.0;
        const double phase = tau - x * std::sin(tau);
        const double sine = std::sin(tau);
        value += weight * std::cos(phase);
        first += weight * sine * std::sin(phase);
        second -= weight * sine * sine * std::cos(phase);
    }
    const double scale = step / std::numbers::pi;
    return {.value = value * scale, .first = first * scale, .second = second * scale};
}

// Returns McMahon's expansion of the m-th root of J1'(x) = 0, with beta = (m - 1/4) pi and
// mu = 4.
double mcmahon_root(int root_number) {
    const double beta = (root_number - 0.25) * std::numbers::pi;
    const double eight_beta = 8.0 * beta;
    const double mu = 4.0;
    return beta - (mu + 3.0) / eight_beta -
           4.0 * (7.0 * mu * mu + 82.0 * mu - 9.0) / (3.0 * std::pow(eight_beta, 3)) -
           32.0 * (83.0 * mu * mu * mu + 2075.0 * mu * mu - 3039.0 * mu + 3537.0) /
               (15.0 * std::pow(eight_beta, 5));
}

// Finds the cylinder roots from McMahon's expansion, refining the first N_NEWTON_ROOTS by
// Newton's method.
std::array<double, N_CYLINDER_ROOTS> find_cylinder_roots() {
    std::array<double, N_CYLINDER_ROOTS> roots{};
    for (int root_number = 1; root_number <= N_CYLINDER_ROOTS; ++root_number) {
        double x = mcmahon_root(root_number);
        if (root_number <= N_NEWTON_ROOTS) {
            for (int step = 0; step < NEWTON_STEPS; ++step) {
                const BesselJ1 j1 = bessel_j1(x);
                x -= j1.first / j1.second;
            }
        }
        roots.at(static_cast<std::size_t>(root_number - 1)) = x;
    }
    return roots;
}

// The nodes and weights of a Gauss-Legendre rule on [-1, 1].
struct GaussLegendre {
    std::array<double, GAUSS_POINTS> nodes{};
    std::array<double, GAUSS_POINTS> weights{};
};

// Computes the Gauss-Legendre rule on [-1, 1], with the roots of P_n found by Newton's method.
GaussLegendre make_gauss_legendre() {
    GaussLegendre rule;
    const int degree = GAUSS_POINTS;
    for (int node = 0; node < degree; ++node) {
        double x = std::cos(std::numbers::pi * (node + 0.75) / (degree + 0.5));
        double derivative = 1.0;
        for (int step = 0; step < 100; ++step) {
            double previous = 1.0;
            double current = x;
            for (int order = 2; order <= degree; ++order) {
                const double next =
                    ((2.0 * order - 1.0) * x * current - (order - 1.0) * previous) / order;
                previous = current;
                current = next;
            }
            derivative = degree * (x * current - previous) / (x * x - 1.0);
            const double change = current / derivative;
            x -= change;
            if (std::abs(change) < 1e-16) {
                break;
            }
        }
        rule.nodes.at(static_cast<std::size_t>(node)) = x;
        rule.weights.at(static_cast<std::size_t>(node)) =
            2.0 / ((1.0 - x * x) * derivative * derivative);
    }
    return rule;
}

// Returns the Gauss-Legendre rule, computed on the first call.
const GaussLegendre& gauss_legendre() {
    static const GaussLegendre rule = make_gauss_legendre();
    return rule;
}

// Returns int_0^1 exp(rate (t^2 - 1)) exp(-y) I0(y) dt with y = spread (1 - t^2) / 2, integrated
// in u = 1 - t on panels that shrink towards u = 0, where the integrand peaks.
double watson_integral(double rate, double spread) {
    const double end = rate > WATSON_CUTOFF ? WATSON_CUTOFF / rate : 1.0;
    const auto& rule = gauss_legendre();
    double total = 0.0;
    double high = end;
    for (int panel = 0; panel < WATSON_PANELS; ++panel) {
        const double low = panel == WATSON_PANELS - 1 ? 0.0 : 0.5 * high;
        const double middle = 0.5 * (low + high);
        const double half = 0.5 * (high - low);
        for (std::size_t node = 0; node < rule.nodes.size(); ++node) {
            const double distance = middle + half * rule.nodes.at(node);
            const double sine_squared = distance * (2.0 - distance);
            const double bessel =
                spread > 0.0 ? scaled_bessel_i0(0.5 * spread * sine_squared) : 1.0;
            total += half * rule.weights.at(node) * std::exp(-rate * sine_squared) * bessel;
        }
        high = low;
    }
    return total;
}

// Returns ln E_perp of the cylinder for a gradient wholly across it, computed in SI units.
double cylinder_log_across(double b_value, double diffusivity, double radius_um,
                           const PulseTiming& timing) {
    if (b_value <= 0.0 || diffusivity <= 0.0) {
        return 0.0;
    }
    const double b_si = b_value * 1e6;      // s/m^2
    const double d_si = diffusivity * 1e-6; // m^2/s
    const double radius = radius_um * 1e-6; // m
    const double delta = timing.small_delta_ms * 1e-3;
    const double big_delta = timing.big_delta_ms * 1e-3;
    const double gamma_g_squared = b_si / (delta * delta * (big_delta - delta / 3.0));

    // The 2 D a^2 delta part of the bracket sums in closed form, as in the sphere.
    const double radius_squared = radius * radius;
    double sum = 2.0 * delta * radius_squared * radius_squared / d_si * CYLINDER_LONG_PULSE_SUM;
    const auto& roots = cylinder_roots();
    const double needed = MIN_SPHERE_DECAY * radius / std::sqrt(d_si * delta);
    for (std::size_t index = 0; index < roots.size(); ++index) {
        const double x = roots.at(index);
        if (index >= MIN_SPHERE_TERMS && x > needed) {
            break;
        }
        const double alpha_squared = x * x / radius_squared;
        const double rate = alpha_squared * d_si;
        const double decays =
            2.0 - 2.0 * std::exp(-rate * delta) - 2.0 * std::exp(-rate * big_delta) +
            std::exp(-rate * (big_delta - delta)) + std::exp(-rate * (big_delta + delta));
        sum -= decays / (rate * rate * alpha_squared * (x * x - 1.0));
    }
    return -2.0 * gamma_g_squared * sum;
}

} // namespace

double stick(double b_value, double diffusivity, double cosine) {
    return std::exp(-b_value * diffusivity * cosine * cosine);
}

double zeppelin(double b_value, double parallel, double perpendicular, double cosine) {
    return std::exp(-b_value * (perpendicular + (parallel - perpendicular) * cosine * cosine));
}

double ball(double b_value, double diffusivity) {
    return std::exp(-b_value * diffusivity);
}

double powder_stick(double b_value, double diffusivity) {
    const double x = b_value * diffusivity;
    if (x < POWDER_SERIES_LIMIT) {
        return 1.0 - x / 3.0 + x * x / 10.0 - x * x * x / 42.0;
    }
    const double root = std::sqrt(x);
    return std::sqrt(std::numbers::pi) / (2.0 * root) * std::erf(root);
}

const std::array<double, N_SPHERE_ROOTS>& sphere_roots() {
    static const std::array<double, N_SPHERE_ROOTS> roots = find_sphere_roots();
    return roots;
}

double sphere(double b_value, double diffusivity, double radius_um, const PulseTiming& timing) {
    if (b_value <= 0.0 || diffusivity <= 0.0) {
        return 1.0;
    }
    const double b_si = b_value * 1e6;      // s/m^2
    const double d_si = diffusivity * 1e-6; // m^2/s
    const double radius = radius_um * 1e-6; // m
    const double delta = timing.small_delta_ms * 1e-3;
    const double big_delta = timing.big_delta_ms * 1e-3;
    const double gamma_g_squared = b_si / (delta * delta * (big_delta - delta / 3.0));

    // The 2 delta / (a^2 D) part of the bracket sums in closed form, so the series
    // keeps only the decay part, whose terms fall as x^-8 once a^2 D delta is large.
    const double radius_squared = radius * radius;
    double sum = 2.0 * delta * radius_squared * radius_squared / d_si * SPHERE_LONG_PULSE_SUM;
    const auto& roots = sphere_roots();
    const double needed = MIN_SPHERE_DECAY * radius / std::sqrt(d_si * delta);
    for (std::size_t index = 0; index < roots.size(); ++index) {
        const double x = roots.at(index);
        if (index >= MIN_SPHERE_TERMS && x > needed) {
            break;
        }
        const double alpha_squared = x * x / radius_squared;
        const double rate = alpha_squared * d_si;
        const double decays = 2.0 + std::exp(-rate * (big_delta - delta)) -
                              2.0 * std::exp(-rate * delta) - 2.0 * std::exp(-rate * big_delta) +
                              std::exp(-rate * (big_delta + delta));
        sum -= decays / (rate * rate * alpha_squared * (x * x - 2.0));
    }
    return std::exp(-2.0 * gamma_g_squared * sum);
}

double watson_stick(double b_value, double diffusivity, double kappa, double cosine) {
    const double x = std::max(b_value * diffusivity, 0.0);
    const double c2 = std::min(cosine * cosine, 1.0);
    // The 2 x 2 block of A in the plane of mu and g has trace kappa - x and determinant
    // -kappa x (1 - c^2); each eigenvalue comes from the root that does not cancel.
    const double trace = kappa - x;
    const double determinant = -kappa * x * (1.0 - c2);
    const double root = std::sqrt(trace * trace - 4.0 * determinant);
    double upper = 0.0;
    double lower = 0.0;
    if (trace >= 0.0) {
        upper = 0.5 * (trace + root);
        lower = upper > 0.0 ? determinant / upper : 0.0;
    } else {
        lower = 0.5 * (trace - root);
        upper = determinant / lower;
    }
    return std::exp(upper - kappa) * watson_integral(upper, -lower) / watson_integral(kappa, 0.0);
}

const std::array<double, N_CYLINDER_ROOTS>& cylinder_roots() {
    static const std::array<double, N_CYLINDER_ROOTS> roots = find_cylinder_roots();
    return roots;
}

double cylinder(double b_value, double diffusivity, double radius_um, double cosine,
                const PulseTiming& timing) {
    const double c2 = std::min(cosine * cosine, 1.0);
    return std::exp(-b_value * diffusivity * c2 +
                    (1.0 - c2) * cylinder_log_across(b_value, diffusivity, radius_um, timing));
}

double AngularProfile::operator()(double cosine) const {
    const double c2 = std::min(cosine * cosine, 1.0);
    if (table.empty()) {
        return std::exp(offset + slope * c2);
    }
    const auto last = static_cast<Index>(table.size()) - 1;
    const double position = c2 * static_cast<double>(last);
    const Index start = std::clamp(static_cast<Index>(position) - 1, Index{0}, last - 3);
    const double at = position - static_cast<double>(start);
    // Lagrange's cubic through the points start to start + 3, at `at` from the first.
    const auto value = [&](Index offset_from_start) {
        return table[static_cast<std::size_t>(start + offset_from_start)];
    };
    return -value(0) * (at - 1.0) * (at - 2.0) * (at - 3.0) / 6.0 +
           value(1) * at * (at - 2.0) * (at - 3.0) / 2.0 -
           value(2) * at * (at - 1.0) * (at - 3.0) / 2.0 +
           value(3) * at * (at - 1.0) * (at - 2.0) / 6.0;
}

AngularProfile angular_profile(const Compartment& compartment, double b_value,
                               const PulseTiming& timing) {
    AngularProfile profile;
    switch (compartment.model) {
    case Model::Stick:
        profile.slope = -b_value * compartment.diffusivity;
        break;
    case Model::Zeppelin:
        profile.offset = -b_value * compartment.perpendicular_diffusivity;
        profile.slope =
            -b_value * (compartment.diffusivity - compartment.perpendicular_diffusivity);
        break;
    case Model::Cylinder:
        profile.offset =
            cylinder_log_across(b_value, compartment.diffusivity, compartment.radius_um, timing);
        profile.slope = -b_value * compartment.diffusivity - profile.offset;
        break;
    case Model::WatsonStick:
        profile.table.resize(N_PROFILE_POINTS);
        for (int point = 0; point < N_PROFILE_POINTS; ++point) {
            const double c2 = static_cast<double>(point) / (N_PROFILE_POINTS - 1);
            profile.table[static_cast<std::size_t>(point)] =
                watson_stick(b_value, compartment.diffusivity, compartment.kappa, std::sqrt(c2));
        }
        break;
    case Model::Ball:
    case Model::PowderStick:
    case Model::Sphere:
    case Model::Dot:
        profile.offset = std::log(attenuation(compartment, b_value, 0.0, timing));
        break;
    }
    return profile;
}

double attenuation(const Compartment& compartment, double b_value, double cosine,
                   const PulseTiming& timing) {
    switch (compartment.model) {
    case Model::Stick:
        return stick(b_value, compartment.diffusivity, cosine);
    case Model::Zeppelin:
        return zeppelin(b_value, compartment.diffusivity, compartment.perpendicular_diffusivity,
                        cosine);
    case Model::Ball:
        return ball(b_value, compartment.diffusivity);
    case Model::PowderStick:
        return powder_stick(b_value, compartment.diffusivity);
    case Model::Sphere:
        return sphere(b_value, compartment.diffusivity, compartment.radius_um, timing);
    case Model::WatsonStick:
        return watson_stick(b_value, compartment.diffusivity, compartment.kappa, cosine);
    case Model::Cylinder:
        return cylinder(b_value, compartment.diffusivity, compartment.radius_um, cosine, timing);
    case Model::Dot:
        return 1.0;
    }
    return 1.0;
}

double relaxation_weight(double proton_density, double t1_ms, double decay_ms, double tr_ms,
                         double te_ms) {
    return proton_density * (1.0 - std::exp(-tr_ms / t1_ms)) * std::exp(-te_ms / decay_ms);
}

} // namespace larmor::simulation
