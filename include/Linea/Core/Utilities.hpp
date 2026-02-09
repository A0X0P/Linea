// created by : A.N. Prosper
// date : january 25th 2026
// time : 13:21

#ifndef LINEA_UTILITIES_H
#define LINEA_UTILITIES_H

#include "Concepts.hpp"
#include <cmath>
#include <limits>

namespace Linea {

// Floating point equality comparison
template <RealType N> static bool floating_point_equality(N a, N b) noexcept {
  N abs_eps = std::numeric_limits<N>::epsilon() * 10;
  N rel_eps = std::sqrt(std::numeric_limits<N>::epsilon());
  return std::fabs(a - b) <=
         abs_eps + rel_eps * std::max(std::fabs(a), std::fabs(b));
}

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