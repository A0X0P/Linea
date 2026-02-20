
/**
 * @file Vector3D.hpp
 * @author A.N. Prosper
 * @date January 25th 2026
 * @brief Fixed-size 3D vector with geometric operations.
 *
 * Provides:
 *      - Cross product
 *      - Scalar triple product
 *      - Vector triple product
 *
 * Designed for:
 *      - Computational geometry
 *      - Physics
 *      - Graphics
 */

#ifndef LINEA_VECTOR3D_H
#define LINEA_VECTOR3D_H

#include "../Core/Concepts.hpp"
#include <array>

namespace Linea {

template <NumericType V> class Vector;

/**
 * @class Vector3D
 *
 * Represents a 3D vector:
 *
 *      v = (x, y, z)
 *
 * Storage:
 *      std::array<T,3>
 *
 * All operations are constexpr where possible.
 */

template <NumericType T> class Vector3D {

private:
  std::array<T, 3> data{};

public:
  // Constructors

  // std::array initialization
  constexpr explicit Vector3D(const std::array<T, 3> &arr) : data(arr) {}

  // list initialization
  constexpr Vector3D(T x, T y, T z) : data{x, y, z} {}

  // Linea::Vector initialization
  explicit Vector3D(const Vector<T> &v);

  // Element Access
  constexpr T &operator[](std::size_t i) noexcept { return data[i]; }
  constexpr const T &operator[](std::size_t i) const noexcept {
    return data[i];
  }

  /**
   * @brief Cross product.
   *
   * For vectors a and b:
   *
   *      a × b =
   *      (a₂b₃ − a₃b₂,
   *       a₃b₁ − a₁b₃,
   *       a₁b₂ − a₂b₁)
   *
   * Result orthogonal to both inputs.
   */

  constexpr Vector3D<T> cross(const Vector3D<T> &other) const {
    return Vector3D<T>{data[1] * other[2] - data[2] * other[1],
                       data[2] * other[0] - data[0] * other[2],
                       data[0] * other[1] - data[1] * other[0]};
  }

  /**
   * @brief Scalar triple product.
   *
   * Computes:
   *
   *      a · (b × c)
   *
   * Geometric meaning:
   *      Signed volume of parallelepiped.
   */

  constexpr T scalar_triple_product(const Vector3D &b,
                                    const Vector3D &c) const {
    const Vector3D result = b.cross(c);
    return data[0] * result[0] + data[1] * result[1] + data[2] * result[2];
  }

  // Vector Triple Product
  //(a × (b × c))
  constexpr Vector3D vector_triple_product(const Vector3D &b,
                                           const Vector3D &c) const {
    return this->cross(b.cross(c));
  }

  // Raw Access
  constexpr const std::array<T, 3> &data_ref() const noexcept { return data; }
};

} // namespace Linea

#endif // LINEA_VECTOR3D_H
