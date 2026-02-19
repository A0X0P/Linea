/**
 * @file Concepts.hpp
 * @author A.N. Prosper
 * @date January 25th 2026
 * @brief Compile-time numeric constraints for the Linea engine.
 *
 * @details
 * This header defines C++20 concepts that restrict admissible scalar
 * types for matrices, vectors, and numerical algorithms.
 *
 * Mathematical domain:
 *
 *   NumericType = ℤ ∪ ℝ
 *
 * Character types and boolean are explicitly excluded from integral
 * classification to prevent semantic ambiguity in arithmetic contexts.
 *
 * All core containers in Linea are constrained by these concepts.
 */

#ifndef LINEA_CONCEPTS_H
#define LINEA_CONCEPTS_H

#include <type_traits>

namespace Linea {

/**
 * @concept RealType
 * @brief Restricts a type to floating-point types.
 *
 * Equivalent to:
 *     std::is_floating_point_v<T>
 *
 * Used in algorithms requiring:
 *   - Division
 *   - Machine epsilon
 *   - Norm computations
 *   - Decomposition algorithms
 */

template <typename T>
concept RealType = std::is_floating_point_v<T>;

/**
 * @concept IntegralType
 * @brief Restricts a type to arithmetic integer types
 *        excluding boolean and character types.
 *
 * Requirements:
 *   - std::is_integral_v<T>
 *   - sizeof(T) >= sizeof(int)
 *
 * Excludes:
 *   - bool
 *   - char and all character encodings
 *
 * Rationale:
 * Prevents accidental use of encoding types as numeric scalars.
 */

template <typename T>
concept IntegralType =
    std::is_integral_v<T> && !std::is_same_v<T, bool> &&
    !std::is_same_v<T, char> && !std::is_same_v<T, signed char> &&
    !std::is_same_v<T, unsigned char> && !std::is_same_v<T, wchar_t> &&
    !std::is_same_v<T, char8_t> && !std::is_same_v<T, char16_t> &&
    !std::is_same_v<T, char32_t> && (sizeof(T) >= sizeof(int));

/**
 * @concept NumericType
 * @brief Represents admissible scalar types for Linea containers.
 *
 * Defined as:
 *     IntegralType<T> || RealType<T>
 *
 * Mathematical domain:
 *     T ∈ ℤ ∪ ℝ
 */

template <typename T>
concept NumericType = IntegralType<T> || RealType<T>;

} // namespace Linea

#endif // LINEA_CONCEPTS_H
