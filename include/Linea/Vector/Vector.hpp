/**
 * @file Vector.hpp
 * @author A.N. Prosper
 * @date December 19th 2025
 * @brief Generic dense vector container with linear algebra utilities.
 *
 * @details
 * This file defines the class template Linea::Vector<V>, a contiguous,
 * dynamically-sized dense vector constrained by the NumericType concept.
 *
 * The vector supports:
 * - Size construction (zero or constant initialization)
 * - Initializer-list construction
 * - Explicit type-safe casting
 * - Random vector generation
 * - Element access with bounds checking
 * - Dot product
 * - Norm computations (L1, L2, L∞, Lp)
 * - Distance and angle operations
 * - Outer product construction
 * - Reshaping into Matrix
 * - Concatenation
 * - Hadamard product
 *
 * Mathematical Representation:
 *
 * A vector v ∈ 𝔽ⁿ is defined as:
 *
 *      v = [ v₀, v₁, …, vₙ₋₁ ]
 *
 * Storage layout:
 *
 *      index(i) = i
 *
 * so that:
 *
 *      v_i ↦ data[i]
 *
 * Memory Model:
 * - Contiguous std::vector<V>
 * - No padding
 * - Stable address unless reallocated
 * - raw() provides direct pointer access
 *
 * Type Requirements:
 * - V must satisfy NumericType
 * - Certain operations require:
 *      • RealType<V>   (norms, angle, distance)
 *
 * Exception Safety:
 * - Strong guarantee for constructors
 * - Bounds-checked element access throws std::invalid_argument
 * - Dimension mismatches throw std::invalid_argument
 *
 * Thread Safety:
 * - Distinct Vector objects are thread-safe.
 * - random() uses thread_local RNG.
 *
 * Complexity Model:
 * - Construction: O(n)
 * - Element access: O(1)
 * - Dot product: O(n)
 * - Norm computation: O(n)
 * - Outer product: O(n·m)
 *
 * Design Goals:
 * - Deterministic performance
 * - Zero hidden allocations beyond std::vector
 * - Clear mathematical semantics
 * - STL interoperability
 *
 * @note
 * This is a dense vector implementation. Sparse optimizations are not included.
 *
 * @warning
 * Numerical stability depends on algorithmic use (e.g., angle computation
 * may suffer from floating-point precision loss).
 */

#ifndef LINEA_VECTOR_H
#define LINEA_VECTOR_H
#include "../Core/Concepts.hpp"
#include "../Core/PlatformMacros.hpp"
#include "../Core/Types.hpp"
#include "../Core/Utilities.hpp"
#include "Vector3D.hpp"
#include <algorithm>
#include <initializer_list>
#include <random>
#include <stdexcept>
#include <vector>

namespace Linea {

template <NumericType M> class Matrix;

/**
 * @class Vector
 * @brief Dense, contiguous vector over a numeric scalar field.
 *
 * @tparam V Scalar type satisfying NumericType
 *         (IntegralType ∪ RealType).
 *
 * @details
 *
 * Vector<V> models a finite-dimensional vector:
 *
 *      v ∈ 𝔽ⁿ
 *
 * where:
 *
 *      𝔽 = ℤ  ∪  ℝ
 *
 * depending on V.
 *
 * ---------------   Mathematical Model    ------------------
 *
 * A vector is defined as:
 *
 *      v = (v₀, v₁, …, vₙ₋₁)
 *
 * Element access obeys:
 *
 *      v(i) = v_i
 *
 * Storage mapping:
 *
 *      v_i ↦ data[i]
 *
 * ensuring contiguous memory.
 *
 * ---------------    Memory Representation    ------------------
 *
 * Internally:
 *
 *      std::vector<V> data;
 *
 * Properties:
 * - Contiguous storage
 * - No hidden allocations beyond std::vector
 * - Random access iterator support
 *
 * ---------------    Dimensional Invariants    ------------------
 *
 * For all instances:
 *
 *      size() ≥ 0
 *      data.size() == size()
 *
 * ---------------    Algebraic Interpretation    ------------------
 *
 * If V ∈ ℝ:
 *      Vector<V> forms a finite-dimensional real vector space.
 *
 * If V ∈ ℤ:
 *      Vector<V> forms a free ℤ-module.
 *
 * Inner product:
 *
 *      ⟨v, w⟩ = Σ vᵢ wᵢ
 *
 * Norms:
 *
 *      ||v||₁   = Σ |vᵢ|
 *      ||v||₂   = sqrt(Σ vᵢ²)
 *      ||v||∞   = max |vᵢ|
 *      ||v||ₚ   = (Σ |vᵢ|ᵖ)^(1/p)
 *
 * ---------------   Design Goals    ------------------
 *
 * - Deterministic O(n) linear algebra
 * - Explicit domain control
 * - No hidden temporaries
 * - Clear mathematical behavior
 */

template <NumericType V> class Vector {

private:
  std::vector<V> data;

public:
  template <NumericType U> friend class Vector;

  // Constructors

  /**
   * @brief Constructs vector of given size filled with constant value.
   *
   * v_i = value  for all i.
   *
   * @param size Number of elements.
   * @param value Initialization value.
   *
   * @complexity O(n)
   */

  Vector(std::size_t size, V value) : data(size, value) {};

  /**
   * @brief Constructs zero-initialized vector of given size.
   *
   * v_i = V{0}
   *
   * @param size Number of elements.
   *
   * @complexity O(n)
   */

  explicit Vector(std::size_t size) : data(size, V{0}) {};

  /**
   * @brief Constructs vector from initializer list.
   *
   * Example:
   * @code
   * Vector<double> v{1,2,3};
   * @endcode
   *
   * @complexity O(n)
   */

  Vector(std::initializer_list<V> list) : data(list) {};

  /**
   * @brief Explicit construction from std::vector.
   *
   * @param v Source container.
   *
   * @complexity O(n)
   */

  explicit Vector(const std::vector<V> &v) : data(v) {};

  /**
   * @brief Constructs vector with random values in [min_range, max_range].
   *
   * Uses:
   * - uniform_int_distribution for integral types.
   * - uniform_real_distribution for floating types.
   *
   * @param size Number of elements.
   * @param min_range Lower bound.
   * @param max_range Upper bound.
   *
   * @return Randomly initialized vector.
   *
   * @complexity O(n)
   */

  static Vector<V> random(std::size_t size, V min_range, V max_range) {
    auto low = std::min(min_range, max_range);
    auto high = std::max(min_range, max_range);

    Vector<V> result(size);
    auto *RESTRICT out = result.raw();
    static thread_local std::mt19937 engine(std::random_device{}());

    if constexpr (std::is_integral_v<V>) {
      std::uniform_int_distribution<V> distribute(low, high);
      for (std::size_t i{}; i < size; ++i) {
        out[i] = distribute(engine);
      }

    } else if constexpr (std::is_floating_point_v<V>) {
      std::uniform_real_distribution<V> distribute(low, high);
      for (std::size_t i{}; i < size; ++i) {
        out[i] = distribute(engine);
      }
    }

    return result;
  }

  /**
   * @brief Constructs vector from Vector3D.
   *
   * Copies x, y, z components.
   *
   * @complexity O(1)
   */

  Vector(const Vector3D<V> &v);

  /**
   * @brief Explicit casting constructor.
   *
   * Performs element-wise static_cast:
   *
   *      v_i^(V) = static_cast<V>(u_i^(U))
   *
   * @tparam U Source numeric type.
   * @param vector Source vector.
   *
   * @complexity O(n)
   */

  template <NumericType U>
  explicit Vector(const Vector<U> &vector) : data(vector.size()) {
    for (std::size_t i = 0; i < data.size(); ++i) {
      auto *RESTRICT out = this->raw();
      auto *RESTRICT in = vector.raw();
      out[i] = static_cast<V>(in[i]);
    }
  }

  /**
   * @brief Returns number of elements.
   *
   * @return Vector length.
   */

  std::size_t size() const noexcept { return data.size(); };

  /**
   * @brief Returns const reference to underlying storage.
   *
   * @return const std::vector<V>&
   */

  const std::vector<V> &data_ref() const & noexcept { return data; };

  /**
   * @brief Bounds-checked element access (mutable).
   *
   * @param index Position.
   *
   * @return Reference to element.
   *
   * @throws std::invalid_argument If index out of range.
   */

  V &operator[](std::size_t index) {
    if (index >= data.size()) {
      throw std::invalid_argument("vector index out of range");
    }
    return data[index];
  };

  /**
   * @brief Bounds-checked element access (const).
   *
   * @param index Position.
   *
   * @return Const reference.
   *
   * @throws std::invalid_argument If index out of range.
   */

  const V &operator[](std::size_t index) const {
    if (index >= data.size()) {
      throw std::invalid_argument("vector index out of range");
    }
    return data[index];
  };

  /**
   * @brief Returns raw pointer to contiguous storage.
   *
   * @return Pointer to first element.
   */

  [[nodiscard]] inline V *raw() noexcept { return data.data(); }
  [[nodiscard]] inline const V *raw() const noexcept { return data.data(); }

  /**
   * @brief Equality comparison.
   *
   * For integral types:
   *      exact equality
   *
   * For floating types:
   *      uses floating_point_equality()
   *
   * @return True if vectors are element-wise equal.
   */

  bool operator==(const Vector<V> &other) const noexcept {
    if (data.size() != other.size())
      return false;

    auto *RESTRICT out = this->raw();
    auto *RESTRICT in = other.raw();

    if constexpr (IntegralType<V>) {
      for (std::size_t i = 0; i < data.size(); ++i)
        if (out[i] != in[i])
          return false;
    } else if constexpr (RealType<V>) {
      for (std::size_t i = 0; i < data.size(); ++i)
        if (!floating_point_equality(out[i], in[i]))
          return false;
    }

    return true;
  }

  bool operator!=(const Vector<V> &other) const noexcept {
    return !(*this == other);
  }

  // Arithmetic operations - declared here, defined in VectorOperations.h
  Vector<V> operator+(const Vector<V> &other) const;
  template <NumericType T>
  Vector<Numeric<T, V>> operator+(const Vector<T> &other) const;

  Vector<V> operator-(const Vector<V> &other) const;
  template <NumericType T>
  Vector<Numeric<T, V>> operator-(const Vector<T> &other) const;

  Vector<V> hadamard(const Vector<V> &other) const;
  template <NumericType T>
  Vector<Numeric<T, V>> hadamard(const Vector<T> &other) const;

  template <NumericType S> Vector<Numeric<V, S>> operator*(S scalar) const;
  Vector<V> operator*(V scalar) const;

  V dot(const Vector<V> &other) const;

  template <NumericType T>
  friend double distance(const Vector<T> &a, const Vector<T> &b);

  template <NumericType T>
  friend double angle(const Vector<T> &a, const Vector<T> &b);

  double norm(VectorNorm type = VectorNorm::Two, double p = 2.0) const;

private:
  double norm_L1() const;
  double norm_L2() const;
  double norm_Lmax() const;
  double norm_Lp(double p) const;

public:
  Matrix<V> outer(const Vector<V> &other) const;
  Matrix<V> reshape(std::size_t rows, std::size_t columns) const;

  // Concatenation
  Vector<V> join(const Vector<V> &other) const;
  template <NumericType T>
  Vector<Numeric<T, V>> join(const Vector<T> &other) const;

  Vector<V> operator|(const Vector<V> &other) const { return join(other); }
  template <NumericType T>
  Vector<Numeric<T, V>> operator|(const Vector<T> &other) const {
    return join(other);
  }

  // Extraction
  Vector<V> segment(std::size_t start, std::size_t length) const;

  // Iterators
  using iterator = typename std::vector<V>::iterator;
  using const_iterator = typename std::vector<V>::const_iterator;
  using reverse_iterator = typename std::vector<V>::reverse_iterator;
  using const_reverse_iterator =
      typename std::vector<V>::const_reverse_iterator;

  iterator begin() noexcept { return data.begin(); }
  const_iterator begin() const noexcept { return data.begin(); }
  const_iterator cbegin() const noexcept { return data.cbegin(); }

  reverse_iterator rbegin() noexcept { return data.rbegin(); }
  const_reverse_iterator rbegin() const noexcept { return data.rbegin(); }
  const_reverse_iterator crbegin() const noexcept { return data.crbegin(); }

  iterator end() noexcept { return data.end(); }
  const_iterator end() const noexcept { return data.end(); }
  const_iterator cend() const noexcept { return data.cend(); }

  reverse_iterator rend() noexcept { return data.rend(); }
  const_reverse_iterator rend() const noexcept { return data.rend(); }
  const_reverse_iterator crend() const noexcept { return data.crend(); }

  // Free functions - declared here, defined in VectorMath.hpp

  template <RealType N> Vector<N> sin(const Vector<N> &);
  template <RealType N> Vector<N> cos(const Vector<N> &);
  template <DomainCheck Check = DomainCheck::Enable, RealType N>
  Vector<N> tan(const Vector<N> &);
  template <DomainCheck Check = DomainCheck::Enable, RealType N>
  Vector<N> sqrt(const Vector<N> &);
  template <DomainCheck Check = DomainCheck::Enable, RealType N>
  Vector<N> log(const Vector<N> &);
  template <RealType N> Vector<N> exp(const Vector<N> &);
};

// Implementation of Vector3D constructor that depends on Vector
template <NumericType T> Vector3D<T>::Vector3D(const Vector<T> &v) {
  if (v.size() != 3) {
    throw std::invalid_argument("Vector3D requires size == 3");
  }
  std::copy(v.begin(), v.end(), data.begin());
}

// Implementation of Vector constructor that depends on Vector3D
template <NumericType V>
Vector<V>::Vector(const Vector3D<V> &v)
    : data(v.getdata().begin(), v.getdata().end()) {}

} // namespace Linea

#endif // LINEA_VECTOR_H