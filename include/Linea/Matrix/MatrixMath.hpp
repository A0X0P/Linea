// created by : A.N. Prosper
// date : january 25th 2026
// time : 9:56

#ifndef LINEA_MATRIX_MATH_H
#define LINEA_MATRIX_MATH_H

#include "Matrix.hpp"
#include <cmath>
#include <stdexcept>

namespace Linea {

// Element-wise sine
template <RealType N> Matrix<N> sin(const Matrix<N> &matrix) {
  Matrix<N> result(matrix.nrows(), matrix.ncols());
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = matrix.raw();
  const std::size_t n = matrix.data.size();

  using std::sin;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = sin(a[i]);
  }
  return result;
}

// Element-wise cosine
template <RealType N> Matrix<N> cos(const Matrix<N> &matrix) {
  Matrix<N> result(matrix.nrows(), matrix.ncols());
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = matrix.raw();
  const std::size_t n = matrix.data.size();

  using std::cos;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = cos(a[i]);
  }
  return result;
}

// Element-wise tangent
template <RealType N> Matrix<N> tan(const Matrix<N> &matrix) {
  Matrix<N> result(matrix.nrows(), matrix.ncols());
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = matrix.raw();
  const std::size_t n = matrix.data.size();

  using std::cos;
  using std::tan;
  constexpr N eps = N(1e-12);
  for (std::size_t i = 0; i < n; ++i) {

    out[i] = tan(a[i]);
  }
  return result;
}

// Element-wise square root
template <RealType N> Matrix<N> sqrt(const Matrix<N> &matrix) {
  Matrix<N> result(matrix.nrows(), matrix.ncols());
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = matrix.raw();
  const std::size_t n = matrix.data.size();

  using std::sqrt;
  for (std::size_t i = 0; i < n; ++i) {

    out[i] = sqrt(a[i]);
  }
  return result;
}

// Element-wise natural logarithm
template <RealType N> Matrix<N> log(const Matrix<N> &matrix) {
  Matrix<N> result(matrix.nrows(), matrix.ncols());
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = matrix.raw();
  const std::size_t n = matrix.data.size();

  using std::log;
  for (std::size_t i = 0; i < n; ++i) {

    out[i] = log(a[i]);
  }
  return result;
}

// Element-wise exponential
template <RealType N> Matrix<N> exp(const Matrix<N> &matrix) {
  Matrix<N> result(matrix.nrows(), matrix.ncols());
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = matrix.raw();
  const std::size_t n = matrix.data.size();

  using std::exp;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = exp(a[i]);
  }
  return result;
}

// Element-wise power (integral type)
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
  const std::size_t n = matrix.data.size();

  for (std::size_t i = 0; i < n; ++i) {
    out[i] = integer_pow(a[i], static_cast<unsigned int>(exponent));
  }
  return result;
}

// Element-wise power (real type)
template <RealType U> Matrix<U> pow(const Matrix<U> &matrix, U exponent) {
  Matrix<U> result(matrix.nrows(), matrix.ncols());
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = matrix.raw();
  const std::size_t n = matrix.data.size();

  using std::pow;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = pow(a[i], exponent);
  }
  return result;
}

} // namespace Linea

#endif // LINEA_MATRIX_MATH_H