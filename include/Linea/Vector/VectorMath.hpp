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

template <RealType N> Vector<N> sin(const Vector<N> &vec) {
  const std::size_t n = vec.size();
  Vector<N> result(n);
  const N *RESTRICT in = vec.raw();
  N *RESTRICT out = result.raw();
  for (std::size_t i = 0; i < n; ++i)
    out[i] = std::sin(in[i]);
  return result;
}

template <RealType N> Vector<N> cos(const Vector<N> &vec) {
  const std::size_t n = vec.size();
  Vector<N> result(n);
  const N *RESTRICT in = vec.raw();
  N *RESTRICT out = result.raw();
  for (std::size_t i = 0; i < n; ++i)
    out[i] = std::cos(in[i]);
  return result;
}

template <DomainCheck Check, RealType N> Vector<N> tan(const Vector<N> &vec) {
  Vector<N> result(vec.size());

  auto *RESTRICT out = result.raw();
  const auto *RESTRICT in = vec.raw();
  const std::size_t n = vec.size();

  using std::cos;
  using std::tan;
  constexpr N eps = N(1e-12);

  if constexpr (Check == DomainCheck::Enable) {
    for (std::size_t i = 0; i < n; ++i) {
      if (std::abs(cos(in[i])) < eps)
        throw std::domain_error("tan undefined for element " +
                                std::to_string(i));
      out[i] = tan(in[i]);
    }
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = tan(in[i]);
    }
  }

  return result;
}

template <DomainCheck Check, RealType N> Vector<N> sqrt(const Vector<N> &vec) {
  Vector<N> result(vec.size());

  auto *RESTRICT out = result.raw();
  const auto *RESTRICT in = vec.raw();
  const std::size_t n = vec.size();

  using std::sqrt;

  if constexpr (Check == DomainCheck::Enable) {
    for (std::size_t i = 0; i < n; ++i) {
      if (in[i] < N{0})
        throw std::domain_error("sqrt undefined for element " +
                                std::to_string(i));
      out[i] = sqrt(in[i]);
    }
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = sqrt(in[i]);
    }
  }

  return result;
}

template <DomainCheck Check, RealType N> Vector<N> log(const Vector<N> &vec) {
  Vector<N> result(vec.size());

  auto *RESTRICT out = result.raw();
  const auto *RESTRICT in = vec.raw();
  const std::size_t n = vec.size();

  using std::log;

  if constexpr (Check == DomainCheck::Enable) {
    for (std::size_t i = 0; i < n; ++i) {
      if (in[i] <= N{0})
        throw std::domain_error("log undefined for element " +
                                std::to_string(i));
      out[i] = log(in[i]);
    }
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = log(in[i]);
    }
  }

  return result;
}

template <RealType N> Vector<N> exp(const Vector<N> &vec) {
  const std::size_t n = vec.size();
  Vector<N> result(n);
  const N *RESTRICT in = vec.raw();
  N *RESTRICT out = result.raw();
  for (std::size_t i = 0; i < n; ++i)
    out[i] = std::exp(in[i]);
  return result;
}

} // namespace Linea

#endif // LINEA_VECTOR_MATH_H
