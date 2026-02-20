/**
 * @file MatrixOperations.hpp
 * @author A.N. Prosper
 * @date January 25th 2026
 * @brief Arithmetic and comparison operators for Linea::Matrix.
 *
 * @details
 * This file implements algebraic operations over the dense matrix
 * container Linea::Matrix<M>.
 *
 * It provides:
 *
 *   - Equality and inequality comparison
 *   - Element-wise addition and subtraction
 *   - Unary negation
 *   - Scalar multiplication
 *   - Matrix-matrix multiplication
 *   - Hadamard (element-wise) product
 *   - Element-wise division
 *   - Bounds-checked element access
 *
 *
 * ----------------- Algebraic Model -----------------
 *
 *
 * For matrices:
 *
 *      A ∈ 𝔽^{m×n}
 *      B ∈ 𝔽^{m×n}
 *
 * where:
 *
 *      𝔽 = ℤ ∪ ℝ
 *
 * The following operations are defined:
 *
 *   A + B      → element-wise addition
 *   A - B      → element-wise subtraction
 *   -A         → additive inverse
 *   A * B      → matrix multiplication
 *   A ⊙ B     → Hadamard product
 *   A / B      → element-wise division
 *
 * Type promotion between heterogeneous scalar types is handled via:
 *
 *      Numeric<M, T>
 *
 * ensuring a mathematically valid common scalar domain.
 *
 * ----------------- Complexity -----------------
 *
 * Addition/Subtraction:       O(m·n)
 * Scalar multiplication:      O(m·n)
 * Hadamard product:           O(m·n)
 * Element-wise division:      O(m·n)
 * Matrix multiplication:      O(m·n·p)
 *
 * ----------------- Numerical Semantics -----------------
 *
 * For RealType<M>:
 *   Equality comparison uses tolerance-based floating comparison.
 *
 * For IntegralType<M>:
 *   Equality comparison is exact.
 *
 * Division performs explicit zero checks.
 *
 *
 * ----------------- Exception Safety -----------------
 *
 * - Dimension mismatches → std::invalid_argument
 * - Division by zero     → std::domain_error
 * - Index out of range   → std::out_of_range
 *
 * All operations provide strong exception safety.
 */

#ifndef LINEA_MATRIX_OPERATIONS_H
#define LINEA_MATRIX_OPERATIONS_H

#include "Matrix.hpp"

namespace Linea {

/**
 * @brief Equality comparison.
 *
 * Returns true if:
 *
 *   - Dimensions match
 *   - All elements compare equal
 *
 * For IntegralType<M>:
 *      Exact equality.
 *
 * For RealType<M>:
 *      Uses floating_point_equality():
 *
 *          |a - b| ≤ ε_abs + ε_rel · max(|a|, |b|)
 *
 * @param other Matrix to compare.
 * @return True if matrices are equal.
 *
 * @complexity O(m·n)
 */

template <NumericType M>
bool Matrix<M>::operator==(const Matrix<M> &other) const noexcept {
  if (row != other.row || column != other.column)
    return false;

  if constexpr (IntegralType<M>) {
    for (std::size_t i = 0; i < data.size(); ++i)
      if (data[i] != other.data[i])
        return false;
  } else if constexpr (RealType<M>) {
    for (std::size_t i = 0; i < data.size(); ++i)
      if (!floating_point_equality(data[i], other.data[i]))
        return false;
  }

  return true;
}

/**
 * @brief Inequality comparison.
 *
 * Defined as:
 *
 *      !(A == B)
 *
 * @param other Matrix to compare.
 * @return True if matrices differ.
 *
 * @complexity O(m·n)
 */

template <NumericType M>
bool Matrix<M>::operator!=(const Matrix<M> &other) const noexcept {
  return !(*this == other);
}

/**
 * @brief Element-wise matrix addition.
 *
 * Mathematical definition:
 *
 *      C = A + B
 *      C_{ij} = A_{ij} + B_{ij}
 *
 * Requires identical dimensions.
 *
 * @param other Matrix of same dimensions.
 * @return Result matrix.
 *
 * @throws std::invalid_argument if dimensions differ.
 *
 * @complexity O(m·n)
 */

template <NumericType M>
Matrix<M> Matrix<M>::operator+(const Matrix<M> &other) const {
  if ((this->row != other.row) || (this->column != other.column)) {
    throw std::invalid_argument(
        "Matrix addition requires identical dimensions.");
  }
  Matrix<M> result(this->row, this->column);

  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();
  const auto *RESTRICT in = other.raw();
  for (std::size_t i = 0; i < data.size(); ++i) {
    out[i] = a[i] + in[i];
  }
  return result;
}

/**
 * @brief Heterogeneous element-wise addition.
 *
 * Promotes scalar type via:
 *
 *      Numeric<M, T>
 *
 * Mathematical definition:
 *
 *      C_{ij} = A_{ij} + B_{ij}
 *
 * @tparam T Other scalar type.
 * @param other Matrix with compatible dimensions.
 * @return Matrix with promoted scalar type.
 *
 * @throws std::invalid_argument if dimensions differ.
 *
 * @complexity O(m·n)
 */

template <NumericType M>
template <NumericType T>
Matrix<Numeric<M, T>> Matrix<M>::operator+(const Matrix<T> &other) const {
  if ((this->row != other.row) || (this->column != other.column)) {
    throw std::invalid_argument(
        "Matrix addition requires identical dimensions.");
  }
  Matrix<Numeric<M, T>> result(this->row, this->column);

  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();
  const auto *RESTRICT in = other.raw();
  for (std::size_t i = 0; i < data.size(); ++i) {
    out[i] = a[i] + in[i];
  }
  return result;
}

/**
 * @brief Element-wise matrix subtraction.
 *
 * Mathematical definition:
 *
 *      C = A - B
 *      C_{ij} = A_{ij} - B_{ij}
 *
 * Requires identical dimensions.
 *
 * @param other Matrix of same dimensions.
 * @return Result matrix.
 *
 * @throws std::invalid_argument if dimensions differ.
 *
 * @complexity O(m·n)
 */

template <NumericType M>
Matrix<M> Matrix<M>::operator-(const Matrix<M> &other) const {
  if ((this->row != other.row) || (this->column != other.column)) {
    throw std::invalid_argument(
        "Matrix subtraction requires identical dimensions.");
  }
  Matrix<M> result(this->row, this->column);

  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();
  const auto *RESTRICT in = other.raw();
  for (std::size_t i = 0; i < data.size(); ++i) {
    out[i] = a[i] - in[i];
  }
  return result;
}

/**
 * @brief Heterogeneous element-wise subtraction.
 *
 * Promotes scalar type via:
 *
 *      Numeric<M, T>
 *
 * Mathematical definition:
 *
 *      C_{ij} = A_{ij} - B_{ij}
 *
 * @tparam T Other scalar type.
 * @param other Matrix with compatible dimensions.
 * @return Matrix with promoted scalar type.
 *
 * @throws std::invalid_argument if dimensions differ.
 *
 * @complexity O(m·n)
 */

template <NumericType M>
template <NumericType T>
Matrix<Numeric<M, T>> Matrix<M>::operator-(const Matrix<T> &other) const {
  if ((this->row != other.row) || (this->column != other.column)) {
    throw std::invalid_argument(
        "Matrix subtraction requires identical dimensions.");
  }
  Matrix<Numeric<M, T>> result(this->row, this->column);

  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();
  const auto *RESTRICT in = other.raw();
  for (std::size_t i = 0; i < data.size(); ++i) {
    out[i] = a[i] - in[i];
  }
  return result;
}

/**
 * @brief Unary negation.
 *
 * Mathematical definition:
 *
 *      B_{ij} = -A_{ij}
 *
 * @return Negated matrix.
 *
 * @complexity O(m·n)
 */

template <NumericType M> Matrix<M> Matrix<M>::operator-() const {
  Matrix<M> result(row, column);

  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();

  for (std::size_t i = 0; i < data.size(); ++i) {
    out[i] = -a[i];
  }
  return result;
}

/**
 * @brief Right scalar multiplication.
 *
 * Mathematical definition:
 *
 *      B = A * s
 *      B_{ij} = A_{ij} · s
 *
 * Scalar type promoted via Numeric<M, S>.
 *
 * @tparam S Scalar type.
 * @param scalar Multiplier.
 * @return Result matrix.
 *
 * @complexity O(m·n)
 */

template <NumericType M>
template <NumericType S>
Matrix<Numeric<M, S>> Matrix<M>::operator*(S scalar) const {
  Matrix<Numeric<M, S>> result(row, column);

  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();

  for (std::size_t i = 0; i < data.size(); ++i) {
    out[i] = a[i] * scalar;
  }
  return result;
}

/**
 * @brief Left scalar multiplication.
 *
 * Defined as:
 *
 *      s * A
 *
 * Equivalent to:
 *
 *      A * s
 *
 * @complexity O(m·n)
 */

template <NumericType S, NumericType M>
Matrix<Numeric<M, S>> operator*(S scalar, const Matrix<M> &matrix) {
  return matrix * scalar;
}

/**
 * @brief Matrix multiplication.
 *
 * For:
 *
 *      A ∈ 𝔽^{m×n}
 *      B ∈ 𝔽^{n×p}
 *
 * Result:
 *
 *      C ∈ 𝔽^{m×p}
 *
 * Mathematical definition:
 *
 *      C_{ij} = Σ_{k=0}^{n-1} A_{ik} B_{kj}
 *
 * @param other Right-hand matrix.
 * @return Product matrix.
 *
 * @throws std::invalid_argument if inner dimensions mismatch.
 *
 * @complexity O(m·n·p)
 */

template <NumericType M>
Matrix<M> Matrix<M>::operator*(const Matrix<M> &other) const {
  if (this->column != other.row) {
    throw std::invalid_argument("A.column must equal B.row");
  }

  Matrix<M> result(row, other.column);

  const std::size_t m = row;
  const std::size_t n = column;
  const std::size_t p = other.ncols();

  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();
  const auto *RESTRICT b = other.raw();

  M sum{};
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t k = 0; k < n; ++k) {
      sum = a[i * n + k];
      for (std::size_t j = 0; j < p; ++j) {
        out[i * p + j] += sum * b[k * p + j];
      }
    }
  }
  return result;
}

/**
 * @brief Heterogeneous matrix multiplication.
 *
 * Scalar type promoted via Numeric<M, T>.
 *
 * Same mathematical definition:
 *
 *      C_{ij} = Σ A_{ik} B_{kj}
 *
 * @complexity O(m·n·p)
 */

template <NumericType M>
template <NumericType T>
Matrix<Numeric<M, T>> Matrix<M>::operator*(const Matrix<T> &other) const {
  if (this->column != other.row) {
    throw std::invalid_argument("A.column must equal B.row");
  }

  Matrix<Numeric<M, T>> result(row, other.column);

  const std::size_t m = row;
  const std::size_t n = column;
  const std::size_t p = other.ncols();

  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();
  const auto *RESTRICT b = other.raw();

  Numeric<M, T> sum{};
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t k = 0; k < n; ++k) {
      sum = a[i * n + k];
      for (std::size_t j = 0; j < p; ++j) {
        out[i * p + j] += sum * b[k * p + j];
      }
    }
  }
  return result;
}

/**
 * @brief Hadamard (element-wise) product.
 *
 * Mathematical definition:
 *
 *      C_{ij} = A_{ij} · B_{ij}
 *
 * Requires identical dimensions.
 *
 * @throws std::invalid_argument if dimensions differ.
 *
 * @complexity O(m·n)
 */

template <NumericType M>
Matrix<M> Matrix<M>::hadamard_product(const Matrix<M> &other) const {
  if ((this->row != other.row) || (this->column != other.column)) {
    throw std::invalid_argument(
        "Hadamard product requires identical dimensions.");
  }
  Matrix<M> result(this->row, this->column);

  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();
  const auto *RESTRICT in = other.raw();
  for (std::size_t i = 0; i < data.size(); ++i) {
    out[i] = a[i] * in[i];
  }
  return result;
}

/**
 * @brief Heterogeneous Hadamard (element-wise) product.
 *
 * Promotes scalar type via:
 *
 *      Numeric<M, T>
 *
 * Mathematical definition:
 *
 *      C_{ij} = A_{ij} · B_{ij}
 *
 * @tparam T Other scalar type.
 * @param other Matrix with compatible dimensions.
 * @return Matrix with promoted scalar type.
 *
 * @throws std::invalid_argument if dimensions differ.
 *
 * @complexity O(m·n)
 */

template <NumericType M>
template <NumericType T>
Matrix<Numeric<T, M>>
Matrix<M>::hadamard_product(const Matrix<T> &other) const {
  if ((this->row != other.row) || (this->column != other.column)) {
    throw std::invalid_argument(
        "Hadamard product requires identical dimensions.");
  }
  Matrix<Numeric<T, M>> result(this->row, this->column);

  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();
  const auto *RESTRICT in = other.raw();
  for (std::size_t i = 0; i < data.size(); ++i) {
    out[i] = a[i] * in[i];
  }
  return result;
}

/**
 * @brief Element-wise division.
 *
 * Mathematical definition:
 *
 *      C_{ij} = A_{ij} / B_{ij}
 *
 * Requires identical dimensions.
 *
 * Performs explicit zero checks:
 *
 *      B_{ij} ≠ 0
 *
 * @throws std::invalid_argument if dimensions differ.
 * @throws std::domain_error if division by zero detected.
 *
 * @complexity O(m·n)
 */

template <NumericType M>
Matrix<M> Matrix<M>::operator/(const Matrix<M> &other) const {
  if ((this->row != other.row) || (this->column != other.column)) {
    throw std::invalid_argument(
        "Matrix division requires identical dimensions.");
  }
  Matrix<M> result(this->row, this->column);

  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();
  const auto *RESTRICT in = other.raw();
  for (std::size_t i = 0; i < data.size(); ++i) {
    if (in[i] == M{0}) {
      throw std::domain_error("Division by zero.");
    }
    out[i] = a[i] / in[i];
  }
  return result;
}

/**
 * @brief Heterogeneous Element-wise Division.
 *
 * Promotes scalar type via:
 *
 *      Numeric<M, T>
 *
 * Mathematical definition:
 *
 *      C_{ij} = A_{ij} / B_{ij}
 *
 * Requires identical dimensions.
 *
 * Performs explicit zero checks:
 *
 *      B_{ij} ≠ 0
 *
 * @tparam T Other scalar type.
 * @param other Matrix with compatible dimensions.
 * @return Matrix with promoted scalar type.
 *
 * @throws std::invalid_argument if dimensions differ.
 * @throws std::domain_error if division by zero detected.
 *
 * @complexity O(m·n)
 */

template <NumericType M>
template <NumericType T>
Matrix<Numeric<T, M>> Matrix<M>::operator/(const Matrix<T> &other) const {
  if ((this->row != other.row) || (this->column != other.column)) {
    throw std::invalid_argument(
        "Matrix division requires identical dimensions.");
  }
  Matrix<Numeric<T, M>> result(this->row, this->column);

  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();
  const auto *RESTRICT in = other.raw();
  for (std::size_t i = 0; i < data.size(); ++i) {
    if (in[i] == T{0}) {
      throw std::domain_error("Division by zero.");
    }
    out[i] = a[i] / in[i];
  }
  return result;
}

/**
 * @brief Bounds-checked element access.
 *
 * Accesses:
 *
 *      A(i, j)
 *
 * Storage mapping:
 *
 *      data[i * column + j]
 *
 * @param i Row index.
 * @param j Column index.
 * @return Reference to element.
 *
 * @throws std::out_of_range if indices invalid.
 *
 * @complexity O(1)
 */

template <NumericType M>
M &Matrix<M>::operator()(std::size_t i, std::size_t j) {
  if (i >= row || j >= column) {
    throw std::out_of_range("Matrix index is out of range.");
  }
  return data[i * column + j];
}

/**
 * @brief Bounds-checked element access.
 *
 * Accesses:
 *
 *      A(i, j)
 *
 * Storage mapping:
 *
 *      data[i * column + j]
 *
 * @param i Row index.
 * @param j Column index.
 * @return Const reference(read-only) to element.
 *
 * @throws std::out_of_range if indices invalid.
 *
 * @complexity O(1)
 */

template <NumericType M>
const M &Matrix<M>::operator()(std::size_t i, std::size_t j) const {
  if (i >= row || j >= column)
    throw std::out_of_range("Matrix index out of range");
  return data[i * column + j];
}

} // namespace Linea

#endif // LINEA_MATRIX_OPERATIONS_H