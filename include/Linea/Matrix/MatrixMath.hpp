/**
 * @file MatrixMath.hpp
 * @author A.N. Prosper
 * @date January 25th 2026
 * @brief Element-wise scalar mathematical operations for Linea::Matrix.
 *
 * @details
 * This header defines free-function, element-wise scalar transformations
 * over dense matrices whose scalar domain satisfies:
 *
 *      T ∈ ℤ ∪ ℝ
 *
 * according to the Linea concept system:
 *
 *      NumericType  = IntegralType ∪ RealType
 *
 * However, all transcendental functions in this file require:
 *
 *      T ∈ ℝ
 *
 * since trigonometric, logarithmic, and exponential functions are only
 * defined over real scalar fields.
 *
 * ---------------- Mathematical Semantics ---------------- 
 *
 * For A ∈ ℝ^{m×n} and scalar function f : ℝ → ℝ,
 *
 *      B = f(A)
 *
 * is defined element-wise as:
 *
 *      B_{ij} = f(A_{ij})
 *
 * These are **component-wise transformations**, not matrix functions
 * in the operator-theoretic sense. For example:
 *
 *      exp(A) ≠ matrix exponential e^A
 *
 * but instead:
 *
 *      (exp(A))_{ij} = e^{A_{ij}}
 *
 * ---------------- Domain Checking Policy ---------------- 
 *
 * Functions that may have restricted scalar domains are parameterized
 * by:
 *
 *      DomainCheck::Enable
 *      DomainCheck::Disable
 *
 * When enabled:
 *      Runtime validation ensures mathematical correctness.
 *
 * When disabled:
 *      No validation is performed (maximum performance mode).
 *
 * ---------------- Complexity ---------------- 
 *
 * All operations run in:
 *
 *      O(m · n)
 *
 * with contiguous row-major traversal.
 *
 * ---------------- Memory Model ---------------- 
 * 
 * - Single allocation for output matrix
 * - No intermediate temporaries
 * - SIMD-friendly linear access
 *
 * ---------------- Exception Safety ---------------- 
 * 
 * - Strong guarantee
 * - std::domain_error for invalid scalar domains
 * - std::invalid_argument for invalid exponent
 *
 * @warning
 * Disabling domain checking may produce NaN or undefined results
 * depending on the underlying standard library implementation.
 */

#ifndef LINEA_MATRIX_MATH_H
#define LINEA_MATRIX_MATH_H

#include "Matrix.hpp"
#include <cmath>
#include <stdexcept>
#include <string>

namespace Linea {

/**
 * @brief Computes element-wise sine.
 *
 * Mathematical definition:
 * \f[
 * B_{ij} = \sin(A_{ij})
 * \f]
 *
 * Domain:
 *     A_{ij} ∈ ℝ
 *
 * @tparam N RealType scalar.
 * @param matrix Input matrix A ∈ ℝ^{m×n}.
 * @return Matrix B ∈ ℝ^{m×n}.
 *
 * @complexity O(m·n)
 */

template <RealType N> Matrix<N> sin(const Matrix<N> &matrix) {
  Matrix<N> result(matrix.nrows(), matrix.ncols());
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = matrix.raw();
  const std::size_t n = matrix.size();

  using std::sin;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = sin(a[i]);
  }
  return result;
}

/**
 * @brief Computes element-wise cosine.
 *
 * Mathematical definition:
 * \f[
 * B_{ij} = \cos(A_{ij})
 * \f]
 *
 * @tparam N RealType scalar.
 * @param matrix Input matrix.
 * @return Matrix with cosine applied element-wise.
 *
 * @complexity O(m·n)
 */

template <RealType N> Matrix<N> cos(const Matrix<N> &matrix) {
  Matrix<N> result(matrix.nrows(), matrix.ncols());
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = matrix.raw();
  const std::size_t n = matrix.size();

  using std::cos;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = cos(a[i]);
  }
  return result;
}

/**
 * @brief Computes element-wise tangent.
 *
 * Mathematical definition:
 * \f[
 * B_{ij} = \tan(A_{ij})
 * \f]
 *
 * Domain restriction:
 * \f[
 * \cos(A_{ij}) \neq 0
 * \f]
 *
 * If DomainCheck::Enable:
 *     Verifies |cos(A_{ij})| > ε
 *
 * @tparam Check Domain checking policy.
 * @tparam N RealType scalar.
 * @param matrix Input matrix.
 *
 * @throws std::domain_error If tangent undefined and checking enabled.
 *
 * @complexity O(m·n)
 */

template <DomainCheck Check, RealType N>
Matrix<N> tan(const Matrix<N> &matrix) {
  Matrix<N> result(matrix.nrows(), matrix.ncols());
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = matrix.raw();
  const std::size_t n = matrix.size();

  using std::cos;
  using std::tan;
  constexpr N eps = N(1e-12);

  if constexpr (Check == DomainCheck::Enable) {
    for (std::size_t i = 0; i < n; ++i) {
      if (std::abs(cos(a[i])) < eps)
        throw std::domain_error("tan undefined for element " +
                                std::to_string(i));
      out[i] = tan(a[i]);
    }
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = tan(a[i]);
    }
  }

  return result;
}

/**
 * @brief Computes element-wise square root.
 *
 * Mathematical definition:
 * \f[
 * B_{ij} = \sqrt{A_{ij}}
 * \f]
 *
 * Domain restriction:
 * \f[
 * A_{ij} \ge 0
 * \f]
 *
 * @tparam Check Domain checking policy.
 * @tparam N RealType scalar.
 * @param matrix Input matrix.
 *
 * @throws std::domain_error If negative element detected and checking enabled.
 *
 * @complexity O(m·n)
 */

template <DomainCheck Check, RealType N>
Matrix<N> sqrt(const Matrix<N> &matrix) {
  Matrix<N> result(matrix.nrows(), matrix.ncols());
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = matrix.raw();
  const std::size_t n = matrix.size();

  using std::sqrt;

  if constexpr (Check == DomainCheck::Enable) {
    for (std::size_t i = 0; i < n; ++i) {
      if (a[i] < N{0})
        throw std::domain_error("sqrt undefined for element " +
                                std::to_string(i));
      out[i] = sqrt(a[i]);
    }
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = sqrt(a[i]);
    }
  }
  return result;
}

/**
 * @brief Computes element-wise natural logarithm.
 *
 * Mathematical definition:
 * \f[
 * B_{ij} = \ln(A_{ij})
 * \f]
 *
 * Domain restriction:
 * \f[
 * A_{ij} > 0
 * \f]
 *
 * @tparam Check Domain checking policy.
 * @tparam N RealType scalar.
 * @param matrix Input matrix.
 *
 * @throws std::domain_error If non-positive element detected.
 *
 * @complexity O(m·n)
 */

template <DomainCheck Check, RealType N>
Matrix<N> log(const Matrix<N> &matrix) {
  Matrix<N> result(matrix.nrows(), matrix.ncols());
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = matrix.raw();
  const std::size_t n = matrix.size();

  using std::log;

  if constexpr (Check == DomainCheck::Enable) {
    for (std::size_t i = 0; i < n; ++i) {
      if (a[i] <= N{0})
        throw std::domain_error("log undefined for element " +
                                std::to_string(i));
      out[i] = log(a[i]);
    }
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = log(a[i]);
    }
  }
  return result;
}

/**
 * @brief Computes element-wise exponential.
 *
 * Mathematical definition:
 * \f[
 * B_{ij} = e^{A_{ij}}
 * \f]
 *
 * @tparam N RealType scalar.
 * @param matrix Input matrix.
 * @return Matrix with exponential applied element-wise.
 *
 * @complexity O(m·n)
 */

template <RealType N> Matrix<N> exp(const Matrix<N> &matrix) {
  Matrix<N> result(matrix.nrows(), matrix.ncols());
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = matrix.raw();
  const std::size_t n = matrix.size();

  using std::exp;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = exp(a[i]);
  }
  return result;
}

/**
 * @brief Computes element-wise integer exponentiation.
 *
 * Mathematical definition:
 * \f[
 * B_{ij} = A_{ij}^{k}
 * \f]
 * where k ∈ ℕ₀.
 *
 * @tparam U IntegralType scalar.
 * @param matrix Input matrix.
 * @param exponent Non-negative integer exponent.
 *
 * @throws std::invalid_argument If exponent < 0.
 *
 * @complexity O(m·n · log(k))
 */

template <IntegralType U> Matrix<U> pow(const Matrix<U> &matrix, int exponent) {
  if (exponent < 0) {
    throw std::invalid_argument("Negative exponent not supported");
  }
  if (exponent == 0) {
    Matrix<U> result(matrix.nrows(), matrix.ncols());
    std::fill(result.begin(), result.end(), U{1});
    return result;
  }
  Matrix<U> result(matrix.nrows(), matrix.ncols());
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = matrix.raw();
  const std::size_t n = matrix.size();

  for (std::size_t i = 0; i < n; ++i) {
    out[i] = integer_pow(a[i], static_cast<unsigned int>(exponent));
  }
  return result;
}

/**
 * @brief Computes element-wise real exponentiation.
 *
 * Mathematical definition:
 * \f[
 * B_{ij} = A_{ij}^{\alpha}
 * \f]
 * where α ∈ ℝ.
 *
 * @tparam U RealType scalar.
 * @param matrix Input matrix.
 * @param exponent Real exponent.
 *
 * @complexity O(m·n)
 */

template <RealType U> Matrix<U> pow(const Matrix<U> &matrix, U exponent) {
  Matrix<U> result(matrix.nrows(), matrix.ncols());
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = matrix.raw();
  const std::size_t n = matrix.size();

  using std::pow;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = pow(a[i], exponent);
  }
  return result;
}

} // namespace Linea

#endif // LINEA_MATRIX_MATH_H