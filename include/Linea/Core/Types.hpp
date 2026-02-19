
/**
 * @file Types.hpp
 * @author A.N. Prosper
 * @date January 25th 2026
 * @brief Global type utilities and enumerations.
 *
 * Defines:
 *   - Scalar promotion rules
 *   - Norm classifications
 *   - Decomposition compute modes
 *   - Diagonal selection
 *   - Compile-time Domain control for math functions
 */

#ifndef LINEA_TYPES_H
#define LINEA_TYPES_H

#include "Concepts.hpp"
#include <type_traits>

namespace Linea {

/**
 * @brief Determines the common promoted scalar type
 *        between two numeric types.
 *
 * Example:
 *   Numeric<int, double> -> double
 *
 * Used in mixed-type arithmetic operations.
 */

template <NumericType T, NumericType S>
using Numeric = std::common_type_t<T, S>;

/**
 * @enum MatrixNorm
 * @brief Supported matrix norm types.
 *
 * Frobenius:
 *   ||A||_F = sqrt( Σ_ij |a_ij|^2 )
 *
 * One:
 *   ||A||_1 = max_j Σ_i |a_ij|
 *
 * Infinity:
 *   ||A||_∞ = max_i Σ_j |a_ij|
 *
 * Spectral:
 *   ||A||_2 = σ_max(A)
 */

enum class MatrixNorm { Frobenius, One, Infinity, Spectral };

/**
 * @enum VectorNorm
 * @brief Supported vector norm types.
 *
 * One:
 *   ||x||_1 = Σ_i |x_i|
 *
 * Two:
 *   ||x||_2 = sqrt( Σ_i x_i^2 )
 *
 * Infinity:
 *   ||x||_∞ = max_i |x_i|
 *
 * P:
 *   ||x||_p = ( Σ_i |x_i|^p )^(1/p)
 */

enum class VectorNorm { One, Two, Infinity, P };

/**
 * @enum Diagonal
 * @brief Diagonal selection type.
 *
 * Major:
 *   Elements where i = j
 *
 * Minor:
 *   Elements where i + j = n - 1
 */

enum class Diagonal { Major, Minor };

/**
 * @enum ComputeMode
 * @brief Specifies decomposition computation mode.
 *
 * Thin:
 *   Economy-size decomposition.
 *
 * Full:
 *   Complete orthogonal basis.
 */

enum class ComputeMode { Thin, Full };

/**
 * @brief Compile-time control for domain checking in mathematical functions.
 *
 */

enum class DomainCheck {
  Enable /**< Enable runtime domain validation (throws std::domain_error on
            invalid input) */
  ,
  Disable /**< Disable runtime domain validation (behavior follows standard
             library, no checks) */
};

} // namespace Linea
#endif // LINEA_TYPES_H
