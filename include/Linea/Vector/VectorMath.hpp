// created by : A.N. Prosper
// date : january 25th 2026
// time : 09:58

#ifndef LINEA_VECTOR_MATH_H
#define LINEA_VECTOR_MATH_H

#include "Vector.hpp"
#include <cmath>
#include <stdexcept>
#include <string>

namespace Linea {

// Element-wise sine
template <RealType N> inline Vector<N> sin(const Vector<N> &vec) {
  const std::size_t n = vec.data.size();
  Vector<N> result(n);
  const N *RESTRICT in = vec.raw();
  N *RESTRICT out = result.raw();
  for (std::size_t i = 0; i < n; ++i)
    out[i] = std::sin(in[i]);
  return result;
}

// Element-wise cosine
template <RealType N> inline Vector<N> cos(const Vector<N> &vec) {
  const std::size_t n = vec.data.size();
  Vector<N> result(n);
  const N *RESTRICT in = vec.raw();
  N *RESTRICT out = result.raw();
  for (std::size_t i = 0; i < n; ++i)
    out[i] = std::cos(in[i]);
  return result;
}

// Element-wise tangent
template <RealType N> inline Vector<N> tan(const Vector<N> &vec) {
  const std::size_t n = vec.data.size();
  Vector<N> result(n);
  const N *RESTRICT in = vec.raw();
  N *RESTRICT out = result.raw();
  constexpr N eps = N(1e-12);
  for (std::size_t i = 0; i < n; ++i) {
    if (std::abs(std::cos(in[i])) < eps)
      throw std::domain_error("tan undefined for element " +
                              std::to_string(in[i]));
    out[i] = std::tan(in[i]);
  }
  return result;
}

// Element-wise square root
template <RealType N> inline Vector<N> sqrt(const Vector<N> &vec) {
  const std::size_t n = vec.data.size();
  Vector<N> result(n);
  const N *RESTRICT in = vec.raw();
  N *RESTRICT out = result.raw();
  for (std::size_t i = 0; i < n; ++i) {
    if (in[i] < N(0))
      throw std::domain_error("sqrt undefined for element " +
                              std::to_string(in[i]));
    out[i] = std::sqrt(in[i]);
  }
  return result;
}

// Element-wise logarithm
template <RealType N> inline Vector<N> log(const Vector<N> &vec) {
  const std::size_t n = vec.data.size();
  Vector<N> result(n);
  const N *RESTRICT in = vec.raw();
  N *RESTRICT out = result.raw();
  for (std::size_t i = 0; i < n; ++i) {
    if (in[i] <= N(0))
      throw std::domain_error("log undefined for element " +
                              std::to_string(in[i]));
    out[i] = std::log(in[i]);
  }
  return result;
}

// Element-wise exponential
template <RealType N> inline Vector<N> exp(const Vector<N> &vec) {
  const std::size_t n = vec.data.size();
  Vector<N> result(n);
  const N *RESTRICT in = vec.raw();
  N *RESTRICT out = result.raw();
  for (std::size_t i = 0; i < n; ++i)
    out[i] = std::exp(in[i]);
  return result;
}

} // namespace Linea

#endif // LINEA_VECTOR_MATH_H
