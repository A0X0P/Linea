
/**
 * @file VectorOperations.hpp
 * @author A.N. Prosper
 * @date January 25th 2026
 * @brief Implements arithmetic, geometric, and structural operations
 *        for the Linea::Vector class.
 *
 * This file provides:
 *
 * Arithmetic operations:
 *   - Element-wise addition and subtraction
 *   - Hadamard (element-wise) product
 *   - Scalar multiplication
 *   - Dot product
 *
 * Geometric operations:
 *   - Euclidean distance
 *   - Angle between vectors
 *   - Vector norms (L1, L2, L∞, Lp)
 *   - Outer product
 *
 * Structural operations:
 *   - Reshape into matrix
 *   - Concatenation (join)
 *   - Segment extraction
 *
 * All size-sensitive operations validate dimensional compatibility
 * and throw std::invalid_argument or std::out_of_range when violated.
 *
 * @note Norms and distance return double for numerical stability.
 */

#ifndef LINEA_VECTOR_OPERATIONS_H
#define LINEA_VECTOR_OPERATIONS_H

#include "Vector.hpp"

namespace Linea {

/**
 * @brief Element-wise addition of two vectors of the same type.
 *
 * Computes a new vector where each element is the sum of the
 * corresponding elements of the two vectors.
 *
 * @param other Vector to add.
 * @return A new vector containing element-wise sums.
 *
 * @throws std::invalid_argument If vector sizes differ.
 *
 * @note Both vectors must have identical size.
 */

template <NumericType V>
Vector<V> Vector<V>::operator+(const Vector<V> &other) const {
  if (data.size() != other.data.size()) {
    throw std::invalid_argument("element-wise addition requires same size");
  }
  const std::size_t n = data.size();
  Vector<V> result(n);
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();
  const auto *RESTRICT b = other.raw();
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = a[i] + b[i];
  }
  return result;
}

/**
 * @brief Element-wise addition of two vectors of different numeric types.
 *
 * The resulting vector element type is deduced using `Numeric<T, V>`.
 *
 * @tparam T Numeric type of the other vector.
 * @param other Vector to add.
 * @return A new vector containing element-wise sums with promoted type.
 *
 * @throws std::invalid_argument If vector sizes differ.
 */

template <NumericType V>
template <NumericType T>
Vector<Numeric<T, V>> Vector<V>::operator+(const Vector<T> &other) const {
  if (data.size() != other.data.size()) {
    throw std::invalid_argument("element-wise addition requires same size");
  }
  const std::size_t n = data.size();
  Vector<Numeric<T, V>> result(n);
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();
  const auto *RESTRICT b = other.raw();
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = a[i] + b[i];
  }
  return result;
}

/**
 * @brief Element-wise subtraction of two vectors of the same type.
 *
 * Computes a new vector where each element is the difference
 * between corresponding elements.
 *
 * @param other Vector to subtract.
 * @return A new vector containing element-wise differences.
 *
 * @throws std::invalid_argument If vector sizes differ.
 */

template <NumericType V>
Vector<V> Vector<V>::operator-(const Vector<V> &other) const {
  if (data.size() != other.data.size()) {
    throw std::invalid_argument("element-wise subtraction requires same size");
  }
  const std::size_t n = data.size();
  Vector<V> result(n);
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();
  const auto *RESTRICT b = other.raw();
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = a[i] - b[i];
  }
  return result;
}

/**
 * @brief Element-wise subtraction of two vectors of different numeric types.
 *
 * The resulting vector element type is deduced using `Numeric<T, V>`.
 *
 * @tparam T Numeric type of the other vector.
 * @param other Vector to subtract.
 * @return A new vector containing element-wise differences with promoted type.
 *
 * @throws std::invalid_argument If vector sizes differ.
 */

template <NumericType V>
template <NumericType T>
Vector<Numeric<T, V>> Vector<V>::operator-(const Vector<T> &other) const {
  if (data.size() != other.data.size()) {
    throw std::invalid_argument("element-wise subtraction requires same size");
  }
  const std::size_t n = data.size();
  Vector<Numeric<T, V>> result(n);
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();
  const auto *RESTRICT b = other.raw();
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = a[i] - b[i];
  }
  return result;
}

/**
 * @brief Computes the Hadamard (element-wise) product.
 *
 * Each element of the resulting vector is the product of the
 * corresponding elements of the two vectors.
 *
 * @param other Vector to multiply element-wise.
 * @return A new vector containing element-wise products.
 *
 * @throws std::invalid_argument If vector sizes differ.
 */

template <NumericType V>
Vector<V> Vector<V>::hadamard(const Vector<V> &other) const {
  if (data.size() != other.data.size()) {
    throw std::invalid_argument("hadamard product requires same size");
  }

  const std::size_t n = data.size();
  Vector<V> result(n);
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();
  const auto *RESTRICT b = other.raw();
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = a[i] * b[i];
  }
  return result;
}

/**
 * @brief Computes the Hadamard (element-wise) product for vectors
 *        of different numeric types.
 *
 * The resulting vector element type is deduced using `Numeric<T, V>`.
 *
 * @tparam T Numeric type of the other vector.
 * @param other Vector to multiply element-wise.
 * @return A new vector containing element-wise products with promoted type.
 *
 * @throws std::invalid_argument If vector sizes differ.
 */

template <NumericType V>
template <NumericType T>
Vector<Numeric<T, V>> Vector<V>::hadamard(const Vector<T> &other) const {
  if (data.size() != other.data.size()) {
    throw std::invalid_argument("hadamard product requires same size");
  }

  const std::size_t n = data.size();
  Vector<Numeric<T, V>> result(n);
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();
  const auto *RESTRICT b = other.raw();
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = a[i] * b[i];
  }
  return result;
}

/**
 * @brief Multiplies each element of the vector by a scalar.
 *
 * @tparam S Scalar numeric type.
 * @param scalar Value to multiply each element by.
 * @return A new vector containing scaled elements.
 *
 * @note Resulting element type is deduced using `Numeric<V, S>`.
 */

template <NumericType V>
template <NumericType S>
Vector<Numeric<V, S>> Vector<V>::operator*(S scalar) const {
  Vector<Numeric<V, S>> result(data.size());
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();

  for (std::size_t i = 0; i < data.size(); ++i) {
    out[i] = a[i] * scalar;
  }
  return result;
}

/**
 * @brief Multiplies a scalar by a vector (left scalar multiplication).
 *
 * Equivalent to `vector * scalar`.
 *
 * @tparam S Scalar numeric type.
 * @tparam V Vector numeric type.
 * @param scalar Scalar value.
 * @param vector Vector to scale.
 * @return A new scaled vector.
 */

template <NumericType S, NumericType V>
Vector<Numeric<V, S>> operator*(S scalar, const Vector<V> &vector) {
  return vector * scalar;
}

/**
 * @brief Computes the dot (inner) product of two vectors.
 *
 * Calculates the sum of element-wise products.
 *
 * @param other Vector to compute dot product with.
 * @return The scalar dot product.
 *
 * @throws std::invalid_argument If vector sizes differ.
 */

template <NumericType V> V Vector<V>::dot(const Vector<V> &other) const {
  if (data.size() != other.data.size()) {
    throw std::invalid_argument("dot product requires same size");
  }

  const std::size_t n = data.size();
  V sum = V{0};

  const auto *RESTRICT a = this->raw();
  const auto *RESTRICT b = other.raw();
  for (std::size_t i = 0; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

/**
 * @brief Computes the Euclidean distance between two vectors.
 *
 * Defined as:
 *     sqrt( Σ (a_i - b_i)^2 )
 *
 * @param a First vector.
 * @param b Second vector.
 * @return Euclidean distance as double.
 *
 * @throws std::invalid_argument If vector sizes differ.
 */

template <NumericType T>
double distance(const Vector<T> &a, const Vector<T> &b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("distance requires same size vectors");
  }

  double sum = 0;
  const auto *RESTRICT _a = a.raw();
  const auto *RESTRICT _b = b.raw();
  for (std::size_t i = 0; i < a.size(); ++i) {
    double dis = _a[i] - _b[i];
    sum += dis * dis;
  }
  return std::sqrt(sum);
};

/**
 * @brief Computes the angle between two vectors in radians.
 *
 * Uses the relation:
 *     acos( dot(a,b) / (||a|| * ||b||) )
 *
 * The cosine value is clamped to [-1, 1] to improve numerical stability.
 *
 * @param a First vector.
 * @param b Second vector.
 * @return Angle in radians.
 *
 * @throws std::invalid_argument If vector sizes differ.
 */

template <NumericType T> double angle(const Vector<T> &a, const Vector<T> &b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("angle requires same size vectors");
  }

  auto cosθ = (a.dot(b)) / (a.norm_L2() * b.norm_L2());
  cosθ = std::clamp(cosθ, double{-1}, double{1});
  return std::acos(cosθ);
};

/**
 * @brief Computes a vector norm based on the specified type.
 *
 * Supported norms:
 * - VectorNorm::One      → L1 norm
 * - VectorNorm::Two      → L2 norm
 * - VectorNorm::Infinity → Maximum norm
 * - VectorNorm::P        → General Lp norm
 *
 * @param type Norm type.
 * @param p Parameter for Lp norm (ignored otherwise).
 * @return Computed norm as double.
 */

template <NumericType V>
double Vector<V>::norm(VectorNorm type, double p) const {
  switch (type) {
  case VectorNorm::One:
    return norm_L1();
  case VectorNorm::Two:
    return norm_L2();
  case VectorNorm::Infinity:
    return norm_Lmax();
  case VectorNorm::P:
    return norm_Lp(p);
  default:
    return norm_L2();
  }
}

/**
 * @brief Computes the L1 norm (Manhattan norm).
 *
 * Defined as:
 *     Σ |x_i|
 *
 * @return L1 norm as double.
 */

template <NumericType V> double Vector<V>::norm_L1() const {
  return std::accumulate(data.begin(), data.end(), V{0},
                         [](double sum, V value) {
                           return sum + static_cast<double>(std::abs(value));
                         });
}

/**
 * @brief Computes the L2 norm (Euclidean norm).
 *
 * Defined as:
 *     sqrt( Σ x_i^2 )
 *
 * @return L2 norm as double.
 */

template <NumericType V> double Vector<V>::norm_L2() const {
  return std::sqrt(
      std::accumulate(data.begin(), data.end(), 0.0, [](double sum, V value) {
        double d = static_cast<double>(value);
        return sum + d * d;
      }));
}

/**
 * @brief Computes the infinity norm (maximum absolute value).
 *
 * Defined as:
 *     max |x_i|
 *
 * @return Maximum absolute element.
 *
 * @throws std::domain_error If the vector is empty.
 */

template <NumericType V> double Vector<V>::norm_Lmax() const {
  if (data.empty())
    throw std::domain_error("norm_Lmax: empty vector");
  double max_elem = std::abs(static_cast<double>(data[0]));
  for (std::size_t i = 1; i < data.size(); ++i)
    max_elem = std::max(max_elem, std::abs(static_cast<double>(data[i])));
  return max_elem;
}

/**
 * @brief Computes the Lp norm.
 *
 * Defined as:
 *     ( Σ |x_i|^p )^(1/p)
 *
 * @param p Norm order (must be >= 1).
 * @return Lp norm as double.
 *
 * @throws std::invalid_argument If p < 1.
 */

template <NumericType V> double Vector<V>::norm_Lp(double p) const {
  if (p < 1.0)
    throw std::invalid_argument("Lp norm requires p >= 1");
  double sum =
      std::accumulate(data.begin(), data.end(), 0.0, [p](double acc, V value) {
        return acc + std::pow(std::abs(static_cast<double>(value)), p);
      });
  return std::pow(sum, 1.0 / p);
}

/**
 * @brief Computes the outer product of two vectors.
 *
 * Produces a matrix where:
 *     result(i,j) = this[i] * other[j]
 *
 * @param other Vector to form outer product with.
 * @return A matrix of size (this->size() × other.size()).
 */

template <NumericType V>
Matrix<V> Vector<V>::outer(const Vector<V> &other) const {
  const std::size_t rows = data.size();
  const std::size_t cols = other.data.size();

  Matrix<V> result(rows, cols);

  const V *RESTRICT data_ptr = this->raw();
  const V *RESTRICT other_ptr = other.raw();
  V *RESTRICT result_ptr = result.raw();

  for (std::size_t i = 0; i < rows; ++i) {
    const V data_ptr_idx = data_ptr[i];
    V *row_ptr = result_ptr + i * cols;

    for (std::size_t j = 0; j < cols; ++j) {
      row_ptr[j] = data_ptr_idx * other_ptr[j];
    }
  }

  return result;
}

/**
 * @brief Reshapes the vector into a matrix.
 *
 * Elements are copied in row-major order.
 *
 * @param rows Number of rows.
 * @param columns Number of columns.
 * @return Matrix with specified dimensions.
 *
 * @throws std::invalid_argument If rows * columns != vector size.
 */

template <NumericType V>
Matrix<V> Vector<V>::reshape(std::size_t rows, std::size_t columns) const {
  if (rows * columns != data.size()) {
    throw std::invalid_argument("reshape dimensions do not match vector size");
  }

  Matrix<V> result(rows, columns);
  std::copy(data.begin(), data.end(), result.begin());
  return result;
}

/**
 * @brief Concatenates two vectors of the same type.
 *
 * Appends the elements of `other` to the end of this vector.
 *
 * @param other Vector to append.
 * @return A new vector containing both sequences.
 */

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

/**
 * @brief Concatenates two vectors of different numeric types.
 *
 * Resulting element type is deduced using `Numeric<T, V>`.
 *
 * @tparam T Numeric type of the other vector.
 * @param other Vector to append.
 * @return A new concatenated vector with promoted type.
 */

template <NumericType V>
template <NumericType T>
Vector<Numeric<T, V>> Vector<V>::join(const Vector<T> &other) const {
  const std::size_t this_size = data.size();
  const std::size_t other_size = other.data.size();
  Vector<Numeric<T, V>> result(this_size + other_size);
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();
  const auto *RESTRICT b = other.raw();
  for (std::size_t i = 0; i < this_size; ++i) {
    out[i] = a[i];
  }

  for (std::size_t i = 0; i < other_size; ++i) {
    out[this_size + i] = b[i];
  }

  return result;
}

/**
 * @brief Extracts a contiguous segment of the vector.
 *
 * @param start Starting index.
 * @param length Number of elements to extract.
 * @return A new vector containing the specified segment.
 *
 * @throws std::out_of_range If the requested range exceeds vector bounds.
 */

template <NumericType V>
Vector<V> Vector<V>::segment(std::size_t start, std::size_t length) const {
  if (start + length > size()) {
    throw std::out_of_range("Vector::segment out of range");
  }

  Vector<V> result(length);
  auto *RESTRICT out = result.raw();
  const auto *RESTRICT a = this->raw();
  for (std::size_t i = 0; i < length; ++i) {
    out[i] = a[start + i];
  }
  return result;
}

} // namespace Linea

#endif // LINEA_VECTOR_OPERATIONS_H