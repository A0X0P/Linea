/**
 * @file VectorMath.hpp
 * @author A.N. Prosper
 * @date January 25th 2026
 * @brief Element-wise mathematical functions for Linea::Vector.
 *
 * This header provides high-performance, element-wise mathematical
 * operations for the Linea::Vector class.
 *
 * Each function applies a corresponding <cmath> operation to every
 * element of the input vector and returns a new vector containing
 * the results.
 *
 * Provided operations:
 *  - Trigonometric: sin, cos, tan
 *  - Exponential & logarithmic: exp, log
 *  - Root: sqrt
 *
 * Domain checking for selected functions is controlled at compile time
 * via the DomainCheck enum:
 *
 *   - DomainCheck::Enable  → Performs runtime validation and throws
 *                            std::domain_error on invalid input.
 *   - DomainCheck::Disable → Skips validation for maximum performance.
 *
 * Design goals:
 *  - Zero abstraction overhead
 *  - Cache-friendly linear traversal
 *  - RESTRICT-qualified raw pointer access
 *  - Compile-time domain validation branching (if constexpr)
 *
 * @tparam N Real numeric type satisfying RealType.
 *
 * @note All returned vectors preserve the input size.
 * @note Functions are non-mutating and allocate a new result vector.
 * @note Intended for real-valued vectors only.
 *
 * @ingroup Vector
 */

#ifndef LINEA_VECTOR_MATH_H
#define LINEA_VECTOR_MATH_H

#include "Vector.hpp"
#include <cmath>
#include <stdexcept>
#include <string>

namespace Linea {

/**
 * @brief Computes the element-wise sine of a vector.
 *
 * Applies `std::sin` to each element of the input vector and
 * returns a new vector containing the results.
 *
 * @tparam N Real numeric type of the vector elements.
 * @param vec Input vector.
 * @return A new vector where each element is `sin(vec[i])`.
 *
 * @note The size of the returned vector matches the input vector.
 */

template <RealType N> Vector<N> sin(const Vector<N> &vec) {
  const std::size_t n = vec.size();
  Vector<N> result(n);
  const N *RESTRICT in = vec.raw();
  N *RESTRICT out = result.raw();
  for (std::size_t i = 0; i < n; ++i)
    out[i] = std::sin(in[i]);
  return result;
}

/**
 * @brief Computes the element-wise cosine of a vector.
 *
 * Applies `std::cos` to each element of the input vector and
 * returns a new vector containing the results.
 *
 * @tparam N Real numeric type of the vector elements.
 * @param vec Input vector.
 * @return A new vector where each element is `cos(vec[i])`.
 *
 * @note The size of the returned vector matches the input vector.
 */

template <RealType N> Vector<N> cos(const Vector<N> &vec) {
  const std::size_t n = vec.size();
  Vector<N> result(n);
  const N *RESTRICT in = vec.raw();
  N *RESTRICT out = result.raw();
  for (std::size_t i = 0; i < n; ++i)
    out[i] = std::cos(in[i]);
  return result;
}

/**
 * @brief Computes the element-wise tangent of a vector.
 *
 * Applies `std::tan` to each element of the input vector.
 * Domain checking can be enabled or disabled at compile time.
 *
 * @tparam Check Controls domain checking behavior.
 *         - DomainCheck::Enable  → checks that cos(x) is not near zero
 *         - DomainCheck::Disable → no domain checking
 * @tparam N Real numeric type of the vector elements.
 *
 * @param vec Input vector.
 * @return A new vector where each element is `tan(vec[i])`.
 *
 * @throws std::domain_error If domain checking is enabled and
 *         `cos(vec[i])` is near zero.
 *
 * @note The size of the returned vector matches the input vector.
 */

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

/**
 * @brief Computes the element-wise square root of a vector.
 *
 * Applies `std::sqrt` to each element of the input vector.
 * Domain checking can be enabled or disabled at compile time.
 *
 * @tparam Check Controls domain checking behavior.
 *         - DomainCheck::Enable  → checks that elements are non-negative
 *         - DomainCheck::Disable → no domain checking
 * @tparam N Real numeric type of the vector elements.
 *
 * @param vec Input vector.
 * @return A new vector where each element is `sqrt(vec[i])`.
 *
 * @throws std::domain_error If domain checking is enabled and
 *         any element is negative.
 *
 * @note The size of the returned vector matches the input vector.
 */

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

/**
 * @brief Computes the element-wise natural logarithm of a vector.
 *
 * Applies `std::log` to each element of the input vector.
 * Domain checking can be enabled or disabled at compile time.
 *
 * @tparam Check Controls domain checking behavior.
 *         - DomainCheck::Enable  → checks that elements are strictly positive
 *         - DomainCheck::Disable → no domain checking
 * @tparam N Real numeric type of the vector elements.
 *
 * @param vec Input vector.
 * @return A new vector where each element is `log(vec[i])`.
 *
 * @throws std::domain_error If domain checking is enabled and
 *         any element is less than or equal to zero.
 *
 * @note The size of the returned vector matches the input vector.
 */

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

/**
 * @brief Computes the element-wise exponential of a vector.
 *
 * Applies `std::exp` to each element of the input vector and
 * returns a new vector containing the results.
 *
 * @tparam N Real numeric type of the vector elements.
 * @param vec Input vector.
 * @return A new vector where each element is `exp(vec[i])`.
 *
 * @note The size of the returned vector matches the input vector.
 */

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
