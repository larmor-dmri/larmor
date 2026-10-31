// The model, noise level and fit follow the NODDI Matlab Toolbox 1.05 (Zhang et al. 2012); the
// start follows AMICO (Daducci et al. 2015).
#include "dwi/noddi.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numbers>

#include "common/maths.hpp"
#include "sd/nnls.hpp"

namespace larmor::dwi {

namespace {

constexpr double PI = std::numbers::pi;
constexpr double SQRT_PI = 1.0 / std::numbers::inv_sqrtpi;

// x from which erfi(x) is taken as exp(x^2) / (x sqrt(pi)).
constexpr double ERFI_ASYMPTOTIC = 5.7;
// Relative size of the last term at which the erfi series stops.
constexpr double SERIES_TOLERANCE = 1e-17;
// Maximum number of terms of the erfi series.
constexpr int MAX_SERIES_TERMS = 400;
// kappa above which the Watson coefficients come from fits in ln(kappa / 30).
constexpr double WATSON_FIT_KAPPA = 30.0;
// kappa up to which the Watson coefficients come from their series.
constexpr double WATSON_SERIES_KAPPA = 0.1;
// x = b d up to which the Legendre-Gaussian integrals come from their series.
constexpr double INTEGRAL_SERIES_X = 0.05;
// kappa below which the Watson-averaged diffusivities come from their series.
constexpr double HINDERED_SERIES_KAPPA = 1e-5;

// The noise level sigma is the spread of the b0s, at least MIN_SNR times their mean, divided by
// SIGMA_SCALING.
constexpr double MIN_SNR = 0.02;
constexpr double SIGMA_SCALING = 100.0;
// The largest kappa of the fit.
constexpr double KAPPA_MAX = 64.0;
// The fit's parameter is kappa divided by this.
constexpr double KAPPA_SCALE = 10.0;
// Bounds of the encoded fractions and angles.
constexpr double FRACTION_UPPER = 1e5;
constexpr double ANGLE_LIMIT = 100.0;
// Argument above which log I0(x) is taken as x - log(2 pi x) / 2.
constexpr double LOG_BESSEL_ASYMPTOTIC = 700.0;
// Most iterations of each stage.
constexpr int MAX_ITERATIONS = 1000;
// Largest change of an encoded parameter at which a stage stops.
constexpr double STEP_TOLERANCE = 1e-6;
// Relative change of the objective below which a stage stops.
constexpr double OBJECTIVE_TOLERANCE = 1e-12;
// Line search: sufficient-decrease constant, and the number of halvings.
constexpr double ARMIJO = 1e-4;
constexpr int MAX_HALVINGS = 40;
// Largest change of an encoded parameter in the first step of a stage.
constexpr double FIRST_STEP = 0.1;
// How far inside [0, 1] the fit starts the fractions.
constexpr double START_MARGIN = 0.01;
// Fewest positive diffusion-weighted measurements the fit needs.
constexpr Index MIN_DIFFUSION_WEIGHTED = 6;

// Returns erfi(x) for x >= 0: its power series below ERFI_ASYMPTOTIC, the leading asymptotic term
// above.
double erfi(double x) {
    if (x >= ERFI_ASYMPTOTIC) {
        return std::exp(x * x) / x / SQRT_PI;
    }
    // erfi(x) = 2 / sqrt(pi) sum_n x^(2n + 1) / (n! (2n + 1)); every term is positive.
    const double x2 = x * x;
    double power = x;
    double sum = x;
    for (int index = 1; index < MAX_SERIES_TERMS; ++index) {
        power *= x2 / index;
        const double term = power / (2.0 * index + 1.0);
        sum += term;
        if (term <= SERIES_TOLERANCE * sum) {
            break;
        }
    }
    return 2.0 / SQRT_PI * sum;
}

// Returns P_0(x), P_2(x), ..., P_12(x), the even Legendre polynomials, by their recurrence.
NoddiTerms even_legendre(double x) {
    NoddiTerms even{};
    double previous = 1.0;
    double current = x;
    even[0] = 1.0;
    for (int degree = 1; degree < 2 * (NODDI_TERMS - 1); ++degree) {
        const double next = ((2.0 * degree + 1.0) * x * current - degree * previous) / (degree + 1);
        previous = current;
        current = next;
        if ((degree + 1) % 2 == 0) {
            even.at(static_cast<std::size_t>((degree + 1) / 2)) = current;
        }
    }
    return even;
}

// Returns log I0(x) for x >= 0.
double log_bessel_i0(double x) {
    if (x >= LOG_BESSEL_ASYMPTOTIC) {
        return x - 0.5 * std::log(2.0 * PI * x);
    }
    return std::log(scaled_bessel_i0(x)) + x;
}

// The coefficients of the fits of the Watson coefficients of degrees 2 to 12 above
// kappa = 30, in powers 0 to 6 of ln(kappa / 30).
constexpr std::array<std::array<double, 7>, NODDI_TERMS - 1> WATSON_FITS{{
    {7.52308, 0.411538, -0.214588, 0.0784091, -0.023981, 0.00731537, -0.0026467},
    {8.93718, 1.62147, -0.733421, 0.191568, -0.0202906, -0.00779095, 0.00574847},
    {8.87905, 3.35689, -1.15935, 0.0673053, 0.121857, -0.066642, 0.0180215},
    {7.84352, 5.03178, -1.0193, -0.426362, 0.328816, -0.0688176, -0.0229398},
    {6.30113, 6.09914, -0.16088, -1.05578, 0.338069, 0.0937157, -0.106935},
    {4.65678, 6.30069, 1.13754, -1.38393, -0.0134758, 0.331686, -0.105954},
}};

// The Watson coefficients above kappa = 30, from the fits.
NoddiTerms watson_fitted(double kappa) {
    NoddiTerms coefficients{};
    coefficients[0] = 2.0 * SQRT_PI;
    const double log_ratio = std::log(kappa) - std::log(WATSON_FIT_KAPPA);
    for (std::size_t term = 1; term < coefficients.size(); ++term) {
        double power = 1.0;
        double value = 0.0;
        for (const double coefficient : WATSON_FITS.at(term - 1)) {
            value += coefficient * power;
            power *= log_ratio;
        }
        coefficients.at(term) = value;
    }
    return coefficients;
}

// The Watson coefficients for kappa up to 0.1, from their series.
NoddiTerms watson_series(double kappa) {
    const double k2 = kappa * kappa;
    const double k3 = k2 * kappa;
    const double k4 = k3 * kappa;
    const double k5 = k4 * kappa;
    const double k6 = k5 * kappa;
    return {2.0 * SQRT_PI,
            (4.0 / 3.0 * kappa + 8.0 / 63.0 * k2) * std::sqrt(PI / 5.0),
            (8.0 / 21.0 * k2 + 32.0 / 693.0 * k3) * (SQRT_PI * 0.2),
            (16.0 / 693.0 * k3 + 32.0 / 10395.0 * k4) * std::sqrt(PI / 13.0),
            32.0 / 19305.0 * k4 * std::sqrt(PI / 17.0),
            64.0 * std::sqrt(PI / 21.0) * k5 / 692835.0,
            128.0 * SQRT_PI * k6 / 152108775.0};
}

// The Watson coefficients for 0.1 < kappa <= 30, in closed form.
NoddiTerms watson_closed_form(double kappa) {
    const double sk = std::sqrt(kappa);
    const double sk2 = sk * kappa;
    const double sk3 = sk2 * kappa;
    const double sk4 = sk3 * kappa;
    const double sk5 = sk4 * kappa;
    const double sk6 = sk5 * kappa;
    const double k2 = kappa * kappa;
    const double k3 = k2 * kappa;
    const double k4 = k3 * kappa;
    const double k5 = k4 * kappa;
    const double k6 = k5 * kappa;
    const double erfik = erfi(sk);
    const double ierfik = 1.0 / erfik;
    const double ek = std::exp(kappa);
    const double dawsonk = 0.5 * SQRT_PI * erfik / ek;

    NoddiTerms coefficients{};
    coefficients[0] = 2.0 * SQRT_PI;
    coefficients[1] =
        std::sqrt(5.0) * (3.0 * sk - (3.0 + 2.0 * kappa) * dawsonk) * ek * ierfik / kappa;
    coefficients[2] = 0.375 *
                      ((105.0 + 60.0 * kappa + 12.0 * k2) * dawsonk - 105.0 * sk + 10.0 * sk2) *
                      ek / k2 * ierfik;
    coefficients[3] = ((-3465.0 - 1890.0 * kappa - 420.0 * k2 - 40.0 * k3) * dawsonk + 3465.0 * sk -
                       420.0 * sk2 + 84.0 * sk3) *
                      std::sqrt(13.0 * PI) / 64.0 / k3 / dawsonk;
    coefficients[4] =
        std::sqrt(17.0) *
        ((675675.0 + 360360.0 * kappa + 83160.0 * k2 + 10080.0 * k3 + 560.0 * k4) * dawsonk -
         675675.0 * sk + 90090.0 * sk2 - 23100.0 * sk3 + 744.0 * sk4) *
        ek / 512.0 / k4 * ierfik;
    coefficients[5] =
        std::sqrt(21.0 * PI) *
        ((-43648605.0 - 22972950.0 * kappa - 5405400.0 * k2 - 720720.0 * k3 - 55440.0 * k4 -
          2016.0 * k5) *
             dawsonk +
         43648605.0 * sk - 6126120.0 * sk2 + 1729728.0 * sk3 - 82368.0 * sk4 + 5104.0 * sk5) /
        4096.0 / k5 / dawsonk;
    coefficients[6] = 5.0 *
                      ((7027425405.0 + 3666482820.0 * kappa + 872972100.0 * k2 + 122522400.0 * k3 +
                        10810800.0 * k4 + 576576.0 * k5 + 14784.0 * k6) *
                           dawsonk -
                       7027425405.0 * sk + 1018467450.0 * sk2 - 302630328.0 * sk3 +
                       17153136.0 * sk4 - 1553552.0 * sk5 + 25376.0 * sk6) *
                      ek / 16384.0 / k6 * ierfik;
    return coefficients;
}

// The parts of the signal that depend on the direction, for every volume: g . mu, and
// sqrt((2 l + 1) / (4 pi)) P_l(g . mu) of each even degree l.
struct DirectionTerms {
    Vector cosines;
    Matrix harmonics;
};

DirectionTerms direction_terms(const NoddiProtocol& protocol, const Vector3& mu) {
    const Index n_volumes = protocol.bvals.size();
    DirectionTerms terms{.cosines = Vector(n_volumes), .harmonics = Matrix(n_volumes, NODDI_TERMS)};
    for (Index volume = 0; volume < n_volumes; ++volume) {
        const double cosine = std::clamp(protocol.bvecs.row(volume).dot(mu), -1.0, 1.0);
        terms.cosines(volume) = cosine;
        const NoddiTerms legendre = even_legendre(cosine);
        for (int term = 0; term < NODDI_TERMS; ++term) {
            terms.harmonics(volume, term) =
                std::sqrt((term + 0.25) / PI) * legendre.at(static_cast<std::size_t>(term));
        }
    }
    return terms;
}

// Returns the Watson sticks' signal of every volume. A value that is not positive, from the
// series' cut at degree 12, becomes 0.1 times the smallest positive one.
Vector stick_signal(const NoddiProtocol& protocol, const DirectionTerms& terms, double kappa) {
    const NoddiTerms coefficients = watson_sh_coefficients(kappa);
    const Index n_volumes = protocol.bvals.size();
    Vector signal(n_volumes);
    for (Index volume = 0; volume < n_volumes; ++volume) {
        double sum = 0.0;
        for (int term = 0; term < NODDI_TERMS; ++term) {
            sum += protocol.integrals(volume, term) *
                   coefficients.at(static_cast<std::size_t>(term)) * terms.harmonics(volume, term);
        }
        signal(volume) = 0.5 * sum;
    }
    double smallest = std::numeric_limits<double>::infinity();
    for (Index volume = 0; volume < n_volumes; ++volume) {
        if (signal(volume) > 0.0) {
            smallest = std::min(smallest, signal(volume));
        }
    }
    for (Index volume = 0; volume < n_volumes; ++volume) {
        if (!(signal(volume) > 0.0)) {
            signal(volume) = 0.1 * smallest;
        }
    }
    return signal;
}

// Returns the zeppelin's signal of every volume, with the tortuous perpendicular diffusivity
// d (1 - v_ic) averaged over the Watson density.
Vector zeppelin_signal(const NoddiProtocol& protocol, const NoddiSettings& settings,
                       const DirectionTerms& terms, double v_ic, double kappa) {
    const auto [along, across] =
        watson_hindered_diffusivities(settings.d_par, settings.d_par * (1.0 - v_ic), kappa);
    const Index n_volumes = protocol.bvals.size();
    Vector signal(n_volumes);
    for (Index volume = 0; volume < n_volumes; ++volume) {
        const double c2 = terms.cosines(volume) * terms.cosines(volume);
        signal(volume) = std::exp(-protocol.bvals(volume) * (across + (along - across) * c2));
    }
    return signal;
}

// Returns the dictionary about mu: one column per (v_ic, kappa) of the grids, v_ic outer, and
// free water last, (N, n_v_ic n_kappa + 1).
Matrix noddi_dictionary(const NoddiProtocol& protocol, const NoddiSettings& settings,
                        const DirectionTerms& terms) {
    const Index n_volumes = protocol.bvals.size();
    const auto n_kappa = static_cast<Index>(settings.kappa_grid.size());
    const auto n_atoms = static_cast<Index>(settings.v_ic_grid.size()) * n_kappa;
    Matrix dictionary(n_volumes, n_atoms + 1);
    std::vector<Vector> sticks;
    sticks.reserve(settings.kappa_grid.size());
    for (const double kappa : settings.kappa_grid) {
        sticks.push_back(stick_signal(protocol, terms, kappa));
    }
    Index column = 0;
    for (const double v_ic : settings.v_ic_grid) {
        for (std::size_t index = 0; index < sticks.size(); ++index) {
            const double kappa = settings.kappa_grid.at(index);
            dictionary.col(column) =
                v_ic * sticks.at(index) +
                (1.0 - v_ic) * zeppelin_signal(protocol, settings, terms, v_ic, kappa);
            ++column;
        }
    }
    dictionary.col(n_atoms) = protocol.free_water;
    return dictionary;
}

// The start of the fit: the dictionary estimate, or no tissue.
struct NoddiStart {
    double v_ic = 0.0;
    double kappa = 0.0;
    double v_iso = 0.0;
    bool has_tissue = false;
};

// Solves the dictionary: free water from a plain NNLS, the tissue atoms' support
// from a non-negative L1 + L2 solve of the rest, their weights from a plain NNLS on the support,
// and v_ic and kappa as the weighted means of the atoms' values.
NoddiStart dictionary_start(const NoddiSettings& settings, const Matrix& dictionary,
                            const Vector& normalised) {
    const Index n_atoms = dictionary.cols() - 1;
    const auto solve = [](const Matrix& design, const Vector& data, double ridge, double l1) {
        const sd::RidgeDesign prepared = sd::make_ridge_design(design, ridge);
        return sd::solve_ridge_nnls(prepared, data, l1, sd::default_gradient_tolerance(design));
    };
    NoddiStart start;
    start.v_iso = solve(dictionary, normalised, 0.0, 0.0)(n_atoms);
    const Matrix tissue = dictionary.leftCols(n_atoms);
    const Vector without_water = normalised - start.v_iso * dictionary.col(n_atoms);
    Vector weights = solve(tissue, without_water, settings.ridge, settings.l1);

    std::vector<Index> support;
    for (Index atom = 0; atom < n_atoms; ++atom) {
        if (weights(atom) > 0.0) {
            support.push_back(atom);
        }
    }
    if (support.empty()) {
        return start;
    }
    const Matrix kept = tissue(Eigen::placeholders::all, support);
    const Vector refitted = solve(kept, without_water, 0.0, 0.0);
    double total = 0.0;
    const auto n_kappa = static_cast<Index>(settings.kappa_grid.size());
    for (std::size_t index = 0; index < support.size(); ++index) {
        const double weight = refitted(static_cast<Index>(index));
        const Index atom = support.at(index);
        total += weight;
        start.v_ic += weight * settings.v_ic_grid.at(static_cast<std::size_t>(atom / n_kappa));
        start.kappa += weight * settings.kappa_grid.at(static_cast<std::size_t>(atom % n_kappa));
    }
    if (!(total > 0.0)) {
        start.v_ic = 0.0;
        start.kappa = 0.0;
        return start;
    }
    start.v_ic /= total;
    start.kappa /= total;
    start.has_tissue = true;
    return start;
}

// The five fitted parameters, encoded: asin(sqrt(v_ic)), sqrt(kappa / 10),
// asin(sqrt(v_iso)), and the polar and azimuthal angles of mu.
using Encoded = Eigen::Matrix<double, 5, 1>;

Encoded encode(double v_ic, double kappa, double v_iso, const Vector3& mu) {
    Encoded x;
    x << std::asin(std::sqrt(std::clamp(v_ic, 0.0, 1.0))),
        std::sqrt(std::min(kappa, KAPPA_MAX) / KAPPA_SCALE),
        std::asin(std::sqrt(std::clamp(v_iso, 0.0, 1.0))), std::acos(std::clamp(mu.z(), -1.0, 1.0)),
        std::atan2(mu.y(), mu.x());
    return x;
}

Vector3 direction_of(const Encoded& x) {
    return {std::cos(x(4)) * std::sin(x(3)), std::sin(x(4)) * std::sin(x(3)), std::cos(x(3))};
}

// The measurements a voxel's fit uses, its positive ones.
struct Measurements {
    NoddiProtocol protocol;
    Vector signal;
    double s0 = 0.0;
    double sigma = 0.0;
};

// Returns the voxel's positive measurements with their S0 and noise level, or false when there is
// no positive b0 or too few positive diffusion-weighted measurements.
bool positive_measurements(const NoddiProtocol& protocol, const VectorView& signal,
                           Measurements& out) {
    std::vector<Index> kept;
    std::vector<double> b0s;
    Index n_weighted = 0;
    for (Index volume = 0; volume < signal.size(); ++volume) {
        if (signal(volume) > 0.0) {
            kept.push_back(volume);
            if (protocol.is_b0.at(static_cast<std::size_t>(volume))) {
                b0s.push_back(signal(volume));
            } else {
                ++n_weighted;
            }
        }
    }
    if (b0s.empty() || n_weighted < MIN_DIFFUSION_WEIGHTED) {
        return false;
    }
    out.protocol.bvals = protocol.bvals(kept);
    out.protocol.bvecs = protocol.bvecs(kept, Eigen::placeholders::all);
    out.protocol.integrals = protocol.integrals(kept, Eigen::placeholders::all);
    out.protocol.free_water = protocol.free_water(kept);
    out.signal = signal(kept);
    double mean = 0.0;
    for (const double b0 : b0s) {
        mean += b0;
    }
    mean /= static_cast<double>(b0s.size());
    // The spread about the mean, over n rather than n - 1.
    double spread = 0.0;
    for (const double b0 : b0s) {
        spread += (b0 - mean) * (b0 - mean);
    }
    spread = std::sqrt(spread / static_cast<double>(b0s.size()));
    out.s0 = mean;
    out.sigma = std::max(spread, MIN_SNR * mean) / SIGMA_SCALING;
    return true;
}

// Returns the negative Rician log-likelihood of the measurements for the encoded parameters. Like
// the toolbox, it adds log E where the Rician density has log m.
double negative_log_likelihood(const Measurements& data, const NoddiSettings& settings,
                               const Encoded& x) {
    const double v_ic = std::sin(x(0)) * std::sin(x(0));
    const double kappa = KAPPA_SCALE * x(1) * x(1);
    const double v_iso = std::sin(x(2)) * std::sin(x(2));
    const DirectionTerms terms = direction_terms(data.protocol, direction_of(x));
    const Vector sticks = stick_signal(data.protocol, terms, kappa);
    const Vector zeppelin = zeppelin_signal(data.protocol, settings, terms, v_ic, kappa);
    const double variance = data.sigma * data.sigma;
    double total = 0.0;
    for (Index volume = 0; volume < data.signal.size(); ++volume) {
        const double tissue = v_ic * sticks(volume) + (1.0 - v_ic) * zeppelin(volume);
        const double model =
            data.s0 * ((1.0 - v_iso) * tissue + v_iso * data.protocol.free_water(volume));
        const double measured = data.signal(volume);
        total += -2.0 * std::log(data.sigma) -
                 (model * model + measured * measured) / (2.0 * variance) + std::log(model) +
                 log_bessel_i0(measured * model / variance);
    }
    return std::isfinite(total) ? -total : std::numeric_limits<double>::infinity();
}

// Returns the lower bounds of the encoded parameters.
Encoded lower_bounds() {
    Encoded lower;
    lower << 0.0, 0.0, 0.0, -ANGLE_LIMIT, -ANGLE_LIMIT;
    return lower;
}

// Returns the upper bounds of the encoded parameters.
Encoded upper_bounds() {
    Encoded upper;
    upper << FRACTION_UPPER, std::sqrt(KAPPA_MAX / KAPPA_SCALE), FRACTION_UPPER, ANGLE_LIMIT,
        ANGLE_LIMIT;
    return upper;
}

using Hessian = Eigen::Matrix<double, 5, 5>;
using Objective = std::function<double(const Encoded&)>;
// Whether each encoded parameter is fitted or held.
using FreeParameters = std::array<bool, 5>;

// Returns the forward-difference gradient of the objective at point, where it has the given value;
// the parameters that are not free get 0.
Encoded forward_gradient(const Objective& objective, const FreeParameters& free,
                         const Encoded& point, double value) {
    const Encoded upper = upper_bounds();
    Encoded gradient = Encoded::Zero();
    for (Index param = 0; param < 5; ++param) {
        if (!free.at(static_cast<std::size_t>(param))) {
            continue;
        }
        double step = std::sqrt(std::numeric_limits<double>::epsilon()) *
                      std::max(std::abs(point(param)), 1.0);
        if (point(param) + step > upper(param)) {
            step = -step;
        }
        Encoded moved = point;
        moved(param) += step;
        gradient(param) = (objective(moved) - value) / step;
    }
    return gradient;
}

// Returns the quasi-Newton direction from x, holding the parameters that are not free and those at
// a bound that the gradient pushes out of. When that is no descent direction, the inverse Hessian
// is reset, restarted is set, and the direction is the steepest descent of the free parameters.
Encoded descent_direction(Hessian& inverse_hessian, const Encoded& gradient, const Encoded& x,
                          const FreeParameters& free, bool& restarted) {
    const Encoded lower = lower_bounds();
    const Encoded upper = upper_bounds();
    Encoded direction = -inverse_hessian * gradient;
    for (Index param = 0; param < 5; ++param) {
        const bool held = !free.at(static_cast<std::size_t>(param)) ||
                          (x(param) <= lower(param) && gradient(param) > 0.0) ||
                          (x(param) >= upper(param) && gradient(param) < 0.0);
        if (held) {
            direction(param) = 0.0;
        }
    }
    if (direction.dot(gradient) < 0.0) {
        return direction;
    }
    inverse_hessian.setIdentity();
    restarted = true;
    direction = -gradient;
    for (Index param = 0; param < 5; ++param) {
        if (!free.at(static_cast<std::size_t>(param))) {
            direction(param) = 0.0;
        }
    }
    return direction;
}

// A point found by the line search, the objective there, and the fraction of the step taken.
struct LineStep {
    Encoded point;
    double value = 0.0;
    double length = 1.0;
    bool accepted = false;
};

// Halves the step along direction from x, projected onto the bounds, until the objective falls
// enough below value (the Armijo condition).
LineStep line_search(const Objective& objective, const Encoded& x, double value,
                     const Encoded& gradient, const Encoded& direction) {
    const Encoded lower = lower_bounds();
    const Encoded upper = upper_bounds();
    LineStep step{.point = x, .value = value, .length = 1.0, .accepted = false};
    for (int halving = 0; halving < MAX_HALVINGS; ++halving) {
        step.point = (x + step.length * direction).cwiseMax(lower).cwiseMin(upper);
        step.value = objective(step.point);
        if (step.value <= value + ARMIJO * gradient.dot(step.point - x)) {
            step.accepted = true;
            return step;
        }
        step.length *= 0.5;
    }
    return step;
}

// Updates the inverse Hessian by BFGS from a step and the change of the gradient over it, scaled
// first when it is still the identity. Returns false, changing nothing, when the curvature along
// the step is not positive.
bool update_inverse_hessian(Hessian& inverse_hessian, const Encoded& step,
                            const Encoded& difference, bool scale) {
    const double curvature = step.dot(difference);
    if (!(curvature > 0.0)) {
        return false;
    }
    if (scale) {
        inverse_hessian *= curvature / difference.squaredNorm();
    }
    const double rho = 1.0 / curvature;
    const Hessian identity = Hessian::Identity();
    inverse_hessian = (identity - rho * step * difference.transpose()) * inverse_hessian *
                          (identity - rho * difference * step.transpose()) +
                      rho * step * step.transpose();
    return true;
}

// Minimises the objective over the free encoded parameters within the bounds, from x, by a
// quasi-Newton method (BFGS) with forward-difference gradients and a backtracking line search.
// Returns the objective at the result.
double minimise(const Objective& objective, const FreeParameters& free, Encoded& x) {
    double value = objective(x);
    if (!std::isfinite(value)) {
        return value;
    }
    Encoded gradient = forward_gradient(objective, free, x, value);
    Hessian inverse_hessian = Hessian::Identity();
    bool first = true;
    for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        Encoded direction = descent_direction(inverse_hessian, gradient, x, free, first);
        const double largest = direction.cwiseAbs().maxCoeff();
        if (largest == 0.0) {
            break;
        }
        if (first) {
            direction *= FIRST_STEP / largest;
        }
        const LineStep next = line_search(objective, x, value, gradient, direction);
        if (!next.accepted) {
            break;
        }
        const Encoded step = next.point - x;
        const double change = value - next.value;
        x = next.point;
        value = next.value;
        // A step shortened by the line search is no sign of convergence; only a full one is.
        const bool full_step = next.length == 1.0 && !first;
        if ((full_step && step.cwiseAbs().maxCoeff() < STEP_TOLERANCE) ||
            change <= OBJECTIVE_TOLERANCE * std::abs(value)) {
            break;
        }
        const Encoded next_gradient = forward_gradient(objective, free, x, value);
        if (update_inverse_hessian(inverse_hessian, step, next_gradient - gradient, first)) {
            first = false;
        }
        gradient = next_gradient;
    }
    return value;
}

} // namespace

NoddiTerms watson_sh_coefficients(double kappa) {
    if (kappa > WATSON_FIT_KAPPA) {
        return watson_fitted(kappa);
    }
    if (kappa > WATSON_SERIES_KAPPA) {
        return watson_closed_form(kappa);
    }
    return watson_series(kappa);
}

NoddiTerms legendre_gaussian_integrals(double x) {
    if (x <= INTEGRAL_SERIES_X) {
        const double x2 = x * x;
        const double x3 = x2 * x;
        const double x4 = x3 * x;
        const double x5 = x4 * x;
        const double x6 = x5 * x;
        return {2.0 - 2.0 * x / 3.0 + x2 / 5.0 - x3 / 21.0 + x4 / 108.0,
                -4.0 * x / 15.0 + 4.0 * x2 / 35.0 - 2.0 * x3 / 63.0 + 2.0 * x4 / 297.0,
                8.0 * x2 / 315.0 - 8.0 * x3 / 693.0 + 4.0 * x4 / 1287.0,
                -16.0 * x3 / 9009.0 + 16.0 * x4 / 19305.0,
                32.0 * x4 / 328185.0,
                -64.0 * x5 / 14549535.0,
                128.0 * x6 / 760543875.0};
    }
    // I_i = integral of u^(2i) exp(-x u^2) over [-1, 1], by recurrence.
    std::array<double, NODDI_TERMS> moments{};
    const double root = std::sqrt(x);
    moments[0] = SQRT_PI * std::erf(root) / root;
    const double minus_exp = -std::exp(-x);
    for (std::size_t power = 1; power < moments.size(); ++power) {
        moments.at(power) =
            (minus_exp + (static_cast<double>(power) - 0.5) * moments.at(power - 1)) / x;
    }
    const auto& mom = moments;
    return {mom[0],
            -0.5 * mom[0] + 1.5 * mom[1],
            0.375 * mom[0] - 3.75 * mom[1] + 4.375 * mom[2],
            -0.3125 * mom[0] + 6.5625 * mom[1] - 19.6875 * mom[2] + 14.4375 * mom[3],
            0.2734375 * mom[0] - 9.84375 * mom[1] + 54.140625 * mom[2] - 93.84375 * mom[3] +
                50.2734375 * mom[4],
            (-63.0 * mom[0] + 3465.0 * mom[1] - 30030.0 * mom[2] + 90090.0 * mom[3] -
             109395.0 * mom[4] + 46189.0 * mom[5]) /
                256.0,
            (231.0 * mom[0] - 18018.0 * mom[1] + 225225.0 * mom[2] - 1021020.0 * mom[3] +
             2078505.0 * mom[4] - 1939938.0 * mom[5] + 676039.0 * mom[6]) /
                1024.0};
}

std::array<double, 2> watson_hindered_diffusivities(double d_par, double d_perp, double kappa) {
    const double difference = d_par - d_perp;
    if (kappa < HINDERED_SERIES_KAPPA) {
        const double mean_sum = d_par + 2.0 * d_perp;
        const double k2 = kappa * kappa;
        return {mean_sum / 3.0 + 4.0 * difference * kappa / 45.0 + 8.0 * difference * k2 / 945.0,
                mean_sum / 3.0 - 2.0 * difference * kappa / 45.0 - 4.0 * difference * k2 / 945.0};
    }
    const double root = std::sqrt(kappa);
    const double dawson = 0.5 * std::exp(-kappa) * SQRT_PI * erfi(root);
    const double factor = root / dawson;
    return {(-difference + 2.0 * d_perp * kappa + difference * factor) / (2.0 * kappa),
            (difference + 2.0 * (d_par + d_perp) * kappa - difference * factor) / (4.0 * kappa)};
}

NoddiProtocol make_noddi_protocol(const VectorView& bvals, const MatrixView& bvecs,
                                  const std::vector<bool>& is_b0, const NoddiSettings& settings) {
    const Index n_volumes = bvals.size();
    NoddiProtocol protocol{.bvals = bvals,
                           .bvecs = bvecs,
                           .is_b0 = is_b0,
                           .integrals = Matrix(n_volumes, NODDI_TERMS),
                           .free_water = Vector(n_volumes)};
    for (Index volume = 0; volume < n_volumes; ++volume) {
        const NoddiTerms integrals = legendre_gaussian_integrals(bvals(volume) * settings.d_par);
        for (int term = 0; term < NODDI_TERMS; ++term) {
            protocol.integrals(volume, term) = integrals.at(static_cast<std::size_t>(term));
        }
        protocol.free_water(volume) = std::exp(-bvals(volume) * settings.d_iso);
    }
    return protocol;
}

Vector noddi_signal(const NoddiProtocol& protocol, const NoddiSettings& settings, double v_ic,
                    double kappa, double v_iso, const Vector3& mu) {
    const DirectionTerms terms = direction_terms(protocol, mu);
    const Vector tissue = v_ic * stick_signal(protocol, terms, kappa) +
                          (1.0 - v_ic) * zeppelin_signal(protocol, settings, terms, v_ic, kappa);
    return (1.0 - v_iso) * tissue + v_iso * protocol.free_water;
}

NoddiVoxelFit fit_noddi_voxel(const NoddiProtocol& protocol, const NoddiSettings& settings,
                              const VectorView& signal, const Vector3& mu) {
    NoddiVoxelFit fit;
    double b0_sum = 0.0;
    Index n_b0 = 0;
    for (Index volume = 0; volume < signal.size(); ++volume) {
        if (protocol.is_b0.at(static_cast<std::size_t>(volume))) {
            b0_sum += signal(volume);
            ++n_b0;
        }
    }
    const double b0_mean = n_b0 > 0 ? b0_sum / static_cast<double>(n_b0) : 0.0;
    Measurements data;
    const double mu_norm = mu.norm();
    if (!(b0_mean > 0.0) || !signal.allFinite() || !std::isfinite(mu_norm) || mu_norm == 0.0 ||
        !positive_measurements(protocol, signal, data)) {
        fit.fitted = false;
        return fit;
    }
    const Vector3 start_mu = mu / mu_norm;

    const Matrix dictionary =
        noddi_dictionary(protocol, settings, direction_terms(protocol, start_mu));
    const NoddiStart start = dictionary_start(settings, dictionary, signal / b0_mean);
    fit.v_iso = start.v_iso;
    fit.mu = start_mu;
    if (!start.has_tissue) {
        return fit;
    }

    const auto objective = [&](const Encoded& x) {
        return negative_log_likelihood(data, settings, x);
    };
    // v = sin^2(u) has no slope at v = 0 and 1, where a fit could not move, so the start keeps
    // the fractions just inside.
    const Encoded initial =
        encode(std::clamp(start.v_ic, START_MARGIN, 1.0 - START_MARGIN), start.kappa,
               std::clamp(start.v_iso, START_MARGIN, 1.0 - START_MARGIN), start_mu);
    const double initial_value = objective(initial);
    // Two stages: the fractions with kappa and mu held, then everything.
    Encoded x = initial;
    minimise(objective, {true, false, true, false, false}, x);
    double value = minimise(objective, {true, true, true, true, true}, x);
    // A fraction that ends at 0 or 1 cannot move again, so the second stage runs once more with it
    // just inside, and the better of the two fits is kept.
    Encoded moved = x;
    for (const Index fraction : {Index{0}, Index{2}}) {
        const double share = std::sin(x(fraction)) * std::sin(x(fraction));
        if (share < START_MARGIN || share > 1.0 - START_MARGIN) {
            moved(fraction) =
                std::asin(std::sqrt(std::clamp(share, START_MARGIN, 1.0 - START_MARGIN)));
        }
    }
    if (moved != x) {
        const double moved_value = minimise(objective, {true, true, true, true, true}, moved);
        if (moved_value < value) {
            x = moved;
            value = moved_value;
        }
    }
    if (!(value < initial_value)) {
        x = initial;
    }
    fit.v_ic = std::sin(x(0)) * std::sin(x(0));
    fit.kappa = KAPPA_SCALE * x(1) * x(1);
    fit.v_iso = std::sin(x(2)) * std::sin(x(2));
    fit.mu = direction_of(x);
    return fit;
}

} // namespace larmor::dwi
