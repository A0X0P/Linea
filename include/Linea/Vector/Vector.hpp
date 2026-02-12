// created by : A.N. Prosper
// date : january 19th 2026
// time : 23:02

#ifndef LINEA_VECTOR_H
#define LINEA_VECTOR_H
#include "../Core/Concepts.hpp"
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

template <NumericType V> class Vector {

private:
  std::vector<V> data;

public:
  template <NumericType U> friend class Vector;

  // Constructors

  // size with fill value
  Vector(std::size_t size, V value) : data(size, value) {};

  // size-only (zero-initialized)
  explicit Vector(std::size_t size) : data(size, V{0}) {};

  // initializer-list construction
  Vector(std::initializer_list<V> list) : data(list) {};

  // std::vector copy construction
  explicit Vector(const std::vector<V> &v) : data(v) {};

  // random initialization
  static Vector<V> rand_fill(std::size_t size, V min_range, V max_range) {
    auto low = std::min(min_range, max_range);
    auto high = std::max(min_range, max_range);

    Vector<V> result(size);
    static thread_local std::mt19937 engine(std::random_device{}());

    if constexpr (std::is_integral_v<V>) {
      std::uniform_int_distribution<V> distribute(low, high);
      for (std::size_t i{}; i < size; ++i) {
        result[i] = distribute(engine);
      }

    } else if constexpr (std::is_floating_point_v<V>) {
      std::uniform_real_distribution<V> distribute(low, high);
      for (std::size_t i{}; i < size; ++i) {
        result[i] = distribute(engine);
      }
    }

    return result;
  }

  // vector3D initialization
  Vector(const Vector3D<V> &v);

  // casting
  template <NumericType U>
  explicit Vector(const Vector<U> &vector) : data(vector.size()) {
    for (std::size_t i = 0; i < data.size(); ++i) {
      data.data()[i] = static_cast<V>(vector.data.data()[i]);
    }
  }

  // Getters
  std::size_t size() const noexcept { return data.size(); };
  const std::vector<V> &data_ref() const & noexcept { return data; };

  // Element access
  V &operator[](std::size_t index) {
    if (index >= data.size()) {
      throw std::invalid_argument("vector index out of range");
    }
    return data[index];
  };

  const V &operator[](std::size_t index) const {
    if (index >= data.size()) {
      throw std::invalid_argument("vector index out of range");
    }
    return data[index];
  };

  V &operator()(std::size_t index);
  const V &operator()(std::size_t index) const;

  // Comparison operators
  bool operator==(const Vector<V> &other) const noexcept {
    if (data.size() != other.size())
      return false;

    if constexpr (IntegralType<V>) {
      for (std::size_t i = 0; i < data.size(); ++i)
        if (data[i] != other.data[i])
          return false;
    } else if constexpr (RealType<V>) {
      for (std::size_t i = 0; i < data.size(); ++i)
        if (!floating_point_equality(data[i], other.data[i]))
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

  // Friends

  template <RealType N> friend Vector<N> sin(const Vector<N> &);
  template <RealType N> friend Vector<N> cos(const Vector<N> &);
  template <RealType N> friend Vector<N> tan(const Vector<N> &);
  template <RealType N> friend Vector<N> sqrt(const Vector<N> &);
  template <RealType N> friend Vector<N> log(const Vector<N> &);
  template <RealType N> friend Vector<N> exp(const Vector<N> &);
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