// created by : A.N. Prosper
// date : january 25th 2026
// time : 9:57

#ifndef LINEA_MATRIX_OPERATIONS_H
#define LINEA_MATRIX_OPERATIONS_H

#include "Matrix.hpp"

namespace Linea {

// Equality operator
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

// Inequality operator
template <NumericType M>
bool Matrix<M>::operator!=(const Matrix<M> &other) const noexcept {
  return !(*this == other);
}

// Addition (same type)
template <NumericType M>
Matrix<M> Matrix<M>::operator+(const Matrix<M> &other) const {
  if ((this->row != other.row) || (this->column != other.column)) {
    throw std::invalid_argument(
        "Matrix addition requires identical dimensions.");
  }
  Matrix<M> result(this->row, this->column);

  auto *out = result.data.data();
  const auto *a = data.data();
  const auto *in = other.data.data();
  for (std::size_t i = 0; i < data.size(); ++i) {
    out[i] = a[i] + in[i];
  }
  return result;
}

// Addition (different type)
template <NumericType M>
template <NumericType T>
Matrix<Numeric<M, T>> Matrix<M>::operator+(const Matrix<T> &other) const {
  if ((this->row != other.row) || (this->column != other.column)) {
    throw std::invalid_argument(
        "Matrix addition requires identical dimensions.");
  }
  Matrix<Numeric<M, T>> result(this->row, this->column);

  auto *out = result.data.data();
  const auto *a = data.data();
  const auto *in = other.data.data();
  for (std::size_t i = 0; i < data.size(); ++i) {
    out[i] = a[i] + in[i];
  }
  return result;
}

// Subtraction (same type)
template <NumericType M>
Matrix<M> Matrix<M>::operator-(const Matrix<M> &other) const {
  if ((this->row != other.row) || (this->column != other.column)) {
    throw std::invalid_argument(
        "Matrix subtraction requires identical dimensions.");
  }
  Matrix<M> result(this->row, this->column);

  auto *out = result.data.data();
  const auto *a = data.data();
  const auto *in = other.data.data();
  for (std::size_t i = 0; i < data.size(); ++i) {
    out[i] = a[i] - in[i];
  }
  return result;
}

// Subtraction (different type)
template <NumericType M>
template <NumericType T>
Matrix<Numeric<M, T>> Matrix<M>::operator-(const Matrix<T> &other) const {
  if ((this->row != other.row) || (this->column != other.column)) {
    throw std::invalid_argument(
        "Matrix subtraction requires identical dimensions.");
  }
  Matrix<Numeric<M, T>> result(this->row, this->column);

  auto *out = result.data.data();
  const auto *a = data.data();
  const auto *in = other.data.data();
  for (std::size_t i = 0; i < data.size(); ++i) {
    out[i] = a[i] - in[i];
  }
  return result;
}

// Unary minus
template <NumericType M> Matrix<M> Matrix<M>::operator-() const {
  Matrix<M> result(row, column);

  auto *out = result.data.data();
  const auto *a = data.data();

  for (std::size_t i = 0; i < data.size(); ++i) {
    out[i] = -a[i];
  }
  return result;
}

// Scalar multiplication
template <NumericType M>
template <NumericType S>
Matrix<Numeric<M, S>> Matrix<M>::operator*(S scalar) const {
  Matrix<Numeric<M, S>> result(row, column);

  auto *out = result.data.data();
  const auto *a = data.data();

  for (std::size_t i = 0; i < data.size(); ++i) {
    out[i] = a[i] * scalar;
  }
  return result;
}

// Left scalar multiplication
template <NumericType S, NumericType M>
Matrix<Numeric<M, S>> operator*(S scalar, const Matrix<M> &matrix) {
  return matrix * scalar;
}

// Matrix multiplication (same type)
template <NumericType M>
Matrix<M> Matrix<M>::operator*(const Matrix<M> &other) const {
  if (this->column != other.row) {
    throw std::invalid_argument("A.column must equal B.row");
  }

  Matrix<M> result(row, other.column);
  M sum{};
  for (std::size_t i = 0; i < row; ++i) {
    for (std::size_t k = 0; k < other.row; ++k) {
      sum = (*this)(i, k);
      for (std::size_t j = 0; j < other.column; ++j) {
        result(i, j) += sum * other(k, j);
      }
    }
  }
  return result;
}

// Matrix multiplication (different type)
template <NumericType M>
template <NumericType T>
Matrix<Numeric<M, T>> Matrix<M>::operator*(const Matrix<T> &other) const {
  if (this->column != other.row) {
    throw std::invalid_argument("A.column must equal B.row");
  }

  Matrix<Numeric<M, T>> result(row, other.column);
  Numeric<M, T> sum{};
  for (std::size_t i = 0; i < row; ++i) {
    for (std::size_t k = 0; k < other.row; ++k) {
      sum = (*this)(i, k);
      for (std::size_t j = 0; j < other.column; ++j) {
        result(i, j) += sum * other(k, j);
      }
    }
  }
  return result;
}

// Hadamard product (same type)
template <NumericType M>
Matrix<M> Matrix<M>::Hadamard_product(const Matrix<M> &other) const {
  if ((this->row != other.row) || (this->column != other.column)) {
    throw std::invalid_argument(
        "Hadamard product requires identical dimensions.");
  }
  Matrix<M> result(this->row, this->column);

  auto *out = result.data.data();
  const auto *a = data.data();
  const auto *in = other.data.data();
  for (std::size_t i = 0; i < data.size(); ++i) {
    out[i] = a[i] * in[i];
  }
  return result;
}

// Hadamard product (different type)
template <NumericType M>
template <NumericType T>
Matrix<Numeric<T, M>>
Matrix<M>::Hadamard_product(const Matrix<T> &other) const {
  if ((this->row != other.row) || (this->column != other.column)) {
    throw std::invalid_argument(
        "Hadamard product requires identical dimensions.");
  }
  Matrix<Numeric<T, M>> result(this->row, this->column);

  auto *out = result.data.data();
  const auto *a = data.data();
  const auto *in = other.data.data();
  for (std::size_t i = 0; i < data.size(); ++i) {
    out[i] = a[i] * in[i];
  }
  return result;
}

// Division (same type)
template <NumericType M>
Matrix<M> Matrix<M>::operator/(const Matrix<M> &other) const {
  if ((this->row != other.row) || (this->column != other.column)) {
    throw std::invalid_argument(
        "Matrix division requires identical dimensions.");
  }
  Matrix<M> result(this->row, this->column);

  auto *out = result.data.data();
  const auto *a = data.data();
  const auto *in = other.data.data();
  for (std::size_t i = 0; i < data.size(); ++i) {
    if (in[i] == M{0}) {
      throw std::domain_error("Division by zero.");
    }
    out[i] = a[i] / in[i];
  }
  return result;
}

// Division (different type)
template <NumericType M>
template <NumericType T>
Matrix<Numeric<T, M>> Matrix<M>::operator/(const Matrix<T> &other) const {
  if ((this->row != other.row) || (this->column != other.column)) {
    throw std::invalid_argument(
        "Matrix division requires identical dimensions.");
  }
  Matrix<Numeric<T, M>> result(this->row, this->column);

  auto *out = result.data.data();
  const auto *a = data.data();
  const auto *in = other.data.data();
  for (std::size_t i = 0; i < data.size(); ++i) {
    if (in[i] == T{0}) {
      throw std::domain_error("Division by zero.");
    }
    out[i] = a[i] / in[i];
  }
  return result;
}

// Index operators
template <NumericType M>
M &Matrix<M>::operator()(std::size_t i, std::size_t j) {
  if (i >= row || j >= column) {
    throw std::out_of_range("Matrix index is out of range.");
  }
  return data[i * column + j];
}

template <NumericType M>
const M &Matrix<M>::operator()(std::size_t i, std::size_t j) const {
  if (i >= row || j >= column)
    throw std::out_of_range("Matrix index out of range");
  return data[i * column + j];
}

} // namespace Linea

#endif // LINEA_MATRIX_OPERATIONS_H