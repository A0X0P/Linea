
/**
 * @file Utilities.hpp
 * @author A.N. Prosper
 * @date January 25th 2026
 * @brief Low-level numerical helper utilities.
 *
 * Contains:
 *   - Floating-point comparison helper
 *   - Integer exponentiation helper
 */

#ifndef LINEA_UTILITIES_H
#define LINEA_UTILITIES_H

#include "Concepts.hpp"
#include <cmath>
#include <limits>

namespace Linea {

/**
 * @brief Relative + absolute floating-point comparison.
 *
 * Returns true if:
 *
 *   |a - b| <= ε_abs + ε_rel * max(|a|, |b|)
 *
 * where:
 *   ε_abs = 10 * machine epsilon
 *   ε_rel = sqrt(machine epsilon)
 *
 * @tparam N Floating-point type.
 * @param a First value.
 * @param b Second value.
 *
 * @return True if approximately equal.
 *
 * @note Intended for testing and general numerical tolerance checks.
 *       Not suitable for rigorous backward error analysis.
 */

// Floating point equality comparison
template <RealType N> static bool floating_point_equality(N a, N b) noexcept {
  N abs_eps = std::numeric_limits<N>::epsilon() * 10;
  N rel_eps = std::sqrt(std::numeric_limits<N>::epsilon());
  return std::fabs(a - b) <=
         abs_eps + rel_eps * std::max(std::fabs(a), std::fabs(b));
}

/**
 * @brief Computes integer exponentiation using exponentiation by squaring.
 *
 * Computes:
 *   base^exp
 *
 * Time complexity:
 *   O(log exp)
 *
 * @tparam I Integral type.
 * @param base Base value.
 * @param exp Non-negative exponent.
 *
 * @return base raised to exp.
 *
 * @note constexpr and noexcept.
 */

// Integer power helper
template <IntegralType I>
constexpr I integer_pow(I base, unsigned int exp) noexcept {
  I result = 1;
  while (exp > 0) {
    if (exp & 1)
      result *= base;
    exp >>= 1;
    if (exp)
      base *= base;
  }
  return result;
}

} // namespace Linea

#endif // LINEA_UTILITIES_H