// created by : A.N. Prosper
// date : january 25th 2026
// time : 13:56

#ifndef LINEA_VECTOR_OPERATIONS_H
#define LINEA_VECTOR_OPERATIONS_H

#include "Vector.hpp"

namespace Linea {

// Addition (same type)
template <NumericType V>
Vector<V> Vector<V>::operator+(const Vector<V> &other) const {
  if (data.size() != other.data.size()) {
    throw std::invalid_argument("element-wise addition requires same size");
  }
  const std::size_t n = data.size();
  Vector<V> result(n);
  for (std::size_t i = 0; i < n; ++i) {
    result[i] = data[i] + other[i];
  }
  return result;
}

// Addition (different type)
template <NumericType V>
template <NumericType T>
Vector<Numeric<T, V>> Vector<V>::operator+(const Vector<T> &other) const {
  if (data.size() != other.data.size()) {
    throw std::invalid_argument("element-wise addition requires same size");
  }
  const std::size_t n = data.size();
  Vector<Numeric<T, V>> result(n);
  for (std::size_t i = 0; i < n; ++i) {
    result[i] = data[i] + other[i];
  }
  return result;
}

// Subtraction (same type)
template <NumericType V>
Vector<V> Vector<V>::operator-(const Vector<V> &other) const {
  if (data.size() != other.data.size()) {
    throw std::invalid_argument("element-wise subtraction requires same size");
  }
  const std::size_t n = data.size();
  Vector<V> result(n);
  for (std::size_t i = 0; i < n; ++i) {
    result[i] = data[i] - other[i];
  }
  return result;
}

// Subtraction (different type)
template <NumericType V>
template <NumericType T>
Vector<Numeric<T, V>> Vector<V>::operator-(const Vector<T> &other) const {
  if (data.size() != other.data.size()) {
    throw std::invalid_argument("element-wise subtraction requires same size");
  }
  const std::size_t n = data.size();
  Vector<Numeric<T, V>> result(n);
  for (std::size_t i = 0; i < n; ++i) {
    result[i] = data[i] - other[i];
  }
  return result;
}

// Hadamard product (same type)
template <NumericType V>
Vector<V> Vector<V>::hadamard(const Vector<V> &other) const {
  if (data.size() != other.data.size()) {
    throw std::invalid_argument("hadamard product requires same size");
  }

  const std::size_t n = data.size();
  Vector<V> result(n);
  for (std::size_t i = 0; i < n; ++i) {
    result[i] = data[i] * other[i];
  }
  return result;
}

// Hadamard product (different type)
template <NumericType V>
template <NumericType T>
Vector<Numeric<T, V>> Vector<V>::hadamard(const Vector<T> &other) const {
  if (data.size() != other.data.size()) {
    throw std::invalid_argument("hadamard product requires same size");
  }

  const std::size_t n = data.size();
  Vector<Numeric<T, V>> result(n);
  for (std::size_t i = 0; i < n; ++i) {
    result[i] = data[i] * other[i];
  }
  return result;
}

// Scalar multiplication
template <NumericType V>
template <NumericType S>
Vector<Numeric<V, S>> Vector<V>::operator*(S scalar) const {
  Vector<Numeric<V, S>> result(data.size());
  for (std::size_t i = 0; i < data.size(); ++i) {
    result[i] = data[i] * scalar;
  }
  return result;
}

// Left scalar multiplication
template <NumericType S, NumericType V>
Vector<Numeric<V, S>> operator*(S scalar, const Vector<V> &vector) {
  return vector * scalar;
}

// Dot product
template <NumericType V> V Vector<V>::dot(const Vector<V> &other) const {
  if (data.size() != other.data.size()) {
    throw std::invalid_argument("dot product requires same size");
  }

  const std::size_t n = data.size();
  V sum = V{0};
  for (std::size_t i = 0; i < n; ++i) {
    sum += data[i] * other[i];
  }
  return sum;
}

// Outer product
template <NumericType V>
Matrix<V> Vector<V>::outer(const Vector<V> &other) const {
  const std::size_t rows = data.size();
  const std::size_t cols = other.data.size();

  Matrix<V> result(rows, cols);

  const V *data_ptr = data.data();
  const V *other_ptr = other.data.data();
  V *result_ptr = result.data.data();

  for (std::size_t i = 0; i < rows; ++i) {
    const V data_ptr_idx = data_ptr[i];
    V *row_ptr = result_ptr + i * cols;

    for (std::size_t j = 0; j < cols; ++j) {
      row_ptr[j] = data_ptr_idx * other_ptr[j];
    }
  }

  return result;
}

// Reshape
template <NumericType V>
Matrix<V> Vector<V>::reshape(std::size_t rows, std::size_t columns) const {
  if (rows * columns != data.size()) {
    throw std::invalid_argument("reshape dimensions do not match vector size");
  }

  Matrix<V> result(rows, columns);
  std::copy(data.begin(), data.end(), result.begin());
  return result;
}

// Concatenation (same type)
template <NumericType V>
Vector<V> Vector<V>::join(const Vector<V> &other) const {
  const std::size_t this_size = data.size();
  const std::size_t other_size = other.data.size();
  Vector<V> result(this_size + other_size);

  std::copy(data.data(), data.data() + this_size, result.data.data());
  std::copy(other.data.data(), other.data.data() + other_size,
            result.data.data() + this_size);

  return result;
}

// Concatenation (different type)
template <NumericType V>
template <NumericType T>
Vector<Numeric<T, V>> Vector<V>::join(const Vector<T> &other) const {
  const std::size_t this_size = data.size();
  const std::size_t other_size = other.data.size();
  Vector<Numeric<T, V>> result(this_size + other_size);

  for (std::size_t i = 0; i < this_size; ++i) {
    result[i] = data[i];
  }

  for (std::size_t i = 0; i < other_size; ++i) {
    result[this_size + i] = other[i];
  }

  return result;
}

} // namespace Linea

#endif // LINEA_VECTOR_OPERATIONS_H