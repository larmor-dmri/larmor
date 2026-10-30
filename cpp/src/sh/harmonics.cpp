#include "sh/harmonics.hpp"

#include <cmath>
#include <numbers>

namespace larmor::sh {

namespace {

using std::numbers::pi;
using std::numbers::sqrt2;

// Returns base to the power exponent >= 0, by binary exponentiation.
double power_int(double base, Index exponent) {
    double result = 1.0;
    while (true) {
        if (exponent % 2 == 1) {
            result *= base;
        }
        exponent /= 2;
        if (exponent == 0) {
            break;
        }
        base *= base;
    }
    return result;
}

// Returns the ladder factor linking P(degree, order) to P(degree, order + 1); order <= degree.
double up(Index degree, Index order) {
    return std::sqrt(static_cast<double>((degree - order) * (degree + order + 1)));
}

// Returns the ladder factor linking P(degree, order) to P(degree, order - 1); order <= degree.
double down(Index degree, Index order) {
    return std::sqrt(static_cast<double>((degree + order) * (degree - order + 1)));
}

// The constants of the Legendre recurrence up to MAX_LMAX, which depend only on degree and order.
struct RecurrenceConstants {
    // P(order, order) / sin^order theta.
    Eigen::Matrix<double, MAX_LMAX + 1, 1> diagonal =
        Eigen::Matrix<double, MAX_LMAX + 1, 1>::Zero();
    // sqrt((4 degree^2 - 1) / (degree^2 - order^2)), for degree > order.
    Legendre::Table factor = Legendre::Table::Zero();

    RecurrenceConstants() {
        // ratio is the product of (2 k - 1) / (2 k) for k = 1 to order.
        double ratio = 1.0;
        for (Index order = 0; order <= MAX_LMAX; ++order) {
            const auto order_f = static_cast<double>(order);
            if (order > 0) {
                ratio *= (2.0 * order_f - 1.0) / (2.0 * order_f);
            }
            const double sign = order % 2 == 1 ? -1.0 : 1.0;
            diagonal(order) = sign * std::sqrt((2.0 * order_f + 1.0) / (4.0 * pi) * ratio);
            for (Index degree = order + 1; degree <= MAX_LMAX; ++degree) {
                const auto degree_f = static_cast<double>(degree);
                factor(degree, order) = std::sqrt((4.0 * degree_f * degree_f - 1.0) /
                                                  (degree_f * degree_f - order_f * order_f));
            }
        }
    }
};

const RecurrenceConstants& recurrence_constants() {
    static const RecurrenceConstants constants;
    return constants;
}

// Writes P(degree, order)(cos theta) for 0 <= order <= degree <= lmax into values, leaving the
// other entries as they were.
void fill_legendre(Index lmax, double cos_theta, double sin_theta, Legendre::Table& values) {
    const RecurrenceConstants& constants = recurrence_constants();
    for (Index order = 0; order <= lmax; ++order) {
        const auto order_f = static_cast<double>(order);
        values(order, order) = constants.diagonal(order) * power_int(sin_theta, order);
        if (order < lmax) {
            values(order + 1, order) =
                std::sqrt(2.0 * order_f + 3.0) * cos_theta * values(order, order);
        }
        for (Index degree = order + 2; degree <= lmax; ++degree) {
            values(degree, order) =
                constants.factor(degree, order) *
                (cos_theta * values(degree - 1, order) -
                 values(degree - 2, order) / constants.factor(degree - 1, order));
        }
    }
}

} // namespace

Legendre Legendre::at_elevation(Index lmax, double theta) {
    return from_cos_sin(lmax, std::cos(theta), std::sin(theta));
}

Legendre Legendre::from_cos_sin(Index lmax, double cos_theta, double sin_theta) {
    Legendre legendre;
    fill_legendre(lmax, cos_theta, sin_theta, legendre.values);
    return legendre;
}

double Legendre::d_theta(Index degree, Index order) const {
    if (degree == 0) {
        return 0.0;
    }
    if (order == 0) {
        return up(degree, 0) * values(degree, 1);
    }
    const double above = order < degree ? up(degree, order) * values(degree, order + 1) : 0.0;
    return (above - down(degree, order) * values(degree, order - 1)) / 2.0;
}

double Legendre::d2_theta(Index degree, Index order) const {
    if (degree == 0) {
        return 0.0;
    }
    if (order == 0) {
        const double up_zero = up(degree, 0);
        return (up_zero * up(degree, 1) * values(degree, 2) -
                up_zero * up_zero * values(degree, 0)) /
               2.0;
    }
    const double up_here = up(degree, order);
    const double down_here = down(degree, order);
    double total = -(up_here * up_here + down_here * down_here) * values(degree, order);
    if (order == 1) {
        // P(degree, -1) = -P(degree, 1) in this normalisation.
        total -= down(degree, 1) * down(degree, 0) * values(degree, 1);
    } else {
        total += down(degree, order) * down(degree, order - 1) * values(degree, order - 2);
    }
    if (order + 2 <= degree) {
        total += up(degree, order) * up(degree, order + 1) * values(degree, order + 2);
    }
    return total / 4.0;
}

Derivatives derivatives(const VectorView& coefficients, Index lmax, bool full, double theta,
                        double phi) {
    const Legendre legendre = Legendre::at_elevation(lmax, theta);
    const double sin_theta = std::sin(theta);
    const bool at_pole = sin_theta < 1e-4;

    Derivatives result;
    const Index degree_step = full ? 1 : 2;
    for (Index degree = 0; degree <= lmax; degree += degree_step) {
        const double zonal = coefficients(coefficient_index(degree, 0, full));
        result.value += zonal * legendre.values(degree, 0);
        result.d_theta += zonal * legendre.d_theta(degree, 0);
        result.d_theta_theta += zonal * legendre.d2_theta(degree, 0);
        for (Index order = 1; order <= degree; ++order) {
            const auto order_f = static_cast<double>(order);
            const double cos_coefficient = coefficients(coefficient_index(degree, order, full));
            const double sin_coefficient = coefficients(coefficient_index(degree, -order, full));
            const double sin_order_phi = std::sin(order_f * phi);
            const double cos_order_phi = std::cos(order_f * phi);
            // The angular factor of the pair of orders +order and -order, and its phi-derivative
            // divided by order.
            const double angular =
                sqrt2 * (cos_coefficient * cos_order_phi + sin_coefficient * sin_order_phi);
            const double angular_d_phi =
                sqrt2 * (sin_coefficient * cos_order_phi - cos_coefficient * sin_order_phi);
            const double legendre_value = legendre.values(degree, order);
            const double legendre_d_theta = legendre.d_theta(degree, order);
            result.value += angular * legendre_value;
            result.d_theta += angular * legendre_d_theta;
            result.d_theta_theta += angular * legendre.d2_theta(degree, order);

            if (at_pole) {
                result.d_phi += angular_d_phi * legendre_d_theta;
            } else {
                result.d_phi += order_f * angular_d_phi * legendre_value;
                result.d_theta_phi += order_f * angular_d_phi * legendre_d_theta;
                result.d_phi_phi -= order_f * order_f * angular * legendre_value;
            }
        }
    }
    if (!at_pole) {
        result.d_phi /= sin_theta;
        result.d_theta_phi /= sin_theta;
        result.d_phi_phi /= sin_theta * sin_theta;
    }
    return result;
}

double amplitude(const VectorView& coefficients, Index lmax, bool full, const Vector3& direction) {
    const double xy_norm = std::sqrt(direction.x() * direction.x() + direction.y() * direction.y());
    // Only the entries up to lmax are written and read, so the table is not zeroed first.
    Legendre::Table legendre;
    fill_legendre(lmax, direction.z(), xy_norm, legendre);
    double cos_phi = 1.0;
    double sin_phi = 0.0;
    if (xy_norm > 0.0) {
        cos_phi = direction.x() / xy_norm;
        sin_phi = direction.y() / xy_norm;
    }

    const Index degree_step = full ? 1 : 2;
    double value = 0.0;
    for (Index degree = 0; degree <= lmax; degree += degree_step) {
        value += coefficients(coefficient_index(degree, 0, full)) * legendre(degree, 0);
    }
    double cos_order_phi = 1.0;
    double sin_order_phi = 0.0;
    for (Index order = 1; order <= lmax; ++order) {
        const double next_cos = cos_order_phi * cos_phi - sin_order_phi * sin_phi;
        const double next_sin = sin_order_phi * cos_phi + cos_order_phi * sin_phi;
        cos_order_phi = next_cos;
        sin_order_phi = next_sin;
        // The degrees >= order: all of them if full, else the even ones.
        const Index first_degree = full || order % 2 == 0 ? order : order + 1;
        for (Index degree = first_degree; degree <= lmax; degree += degree_step) {
            value += sqrt2 * legendre(degree, order) *
                     (coefficients(coefficient_index(degree, order, full)) * cos_order_phi +
                      coefficients(coefficient_index(degree, -order, full)) * sin_order_phi);
        }
    }
    return value;
}

} // namespace larmor::sh
