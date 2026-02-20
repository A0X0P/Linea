
/**
 * @file Linea.hpp
 * @author A.N. Prosper
 * @date January 26th 2026
 * @brief Master umbrella header for the Linea linear algebra engine.
 *
 * This header provides a single-entry include point for the entire
 * Linea library. Including this file makes all major components
 * of the engine available:
 *
 *  - Core utilities and type traits
 *  - Vector types and element-wise math
 *  - Matrix types and operations
 *  - Matrix decompositions
 *  - Stream-based I/O helpers
 *
 * Modules included:
 *
 *  Core:
 *    - Concepts
 *    - Platform macros
 *    - Fundamental types
 *    - Utility helpers
 *
 *  Vector:
 *    - Vector (dynamic vector)
 *    - Vector3D (fixed 3D vector)
 *    - VectorMath (element-wise math functions)
 *    - VectorOperations (algebraic operations)
 *
 *  Matrix:
 *    - Matrix (dense matrix class)
 *    - MatrixMath (element-wise math)
 *    - MatrixOperations (algebraic operations)
 *    - MatrixUtilities (helpers and transformations)
 *
 *  Decompositions:
 *    - Cholesky
 *    - HouseholderQR
 *    - LU
 *    - SVD
 *
 *  IO:
 *    - StreamOperations
 *
 * Design Goals:
 *  - Header-only implementation
 *  - Compile-time safety via concepts
 *  - Zero-overhead abstractions
 *  - High-performance numerical kernels
 *
 * Usage:
 * @code
 *   #include <Linea/Linea.hpp>
 * @endcode
 *
 * @note This header is intended for convenience. For faster compilation
 *       in large projects, include only the specific headers required.
 *
 * @ingroup Linea
 */

#ifndef LINEA_H
#define LINEA_H

// Core
#include "Linea/Core/Concepts.hpp"
#include "Linea/Core/PlatformMacros.hpp"
#include "Linea/Core/Types.hpp"
#include "Linea/Core/Utilities.hpp"

// Vector
#include "Linea/Vector/Vector.hpp"
#include "Linea/Vector/Vector3D.hpp"
#include "Linea/Vector/VectorMath.hpp"
#include "Linea/Vector/VectorOperations.hpp"

// Matrix
#include "Linea/Matrix/Matrix.hpp"
#include "Linea/Matrix/MatrixMath.hpp"
#include "Linea/Matrix/MatrixOperations.hpp"
#include "Linea/Matrix/MatrixUtilities.hpp"

// Decomposition

#include "Linea/Decompositions/Cholesky.hpp"
#include "Linea/Decompositions/HouseholderQR.hpp"
#include "Linea/Decompositions/LU.hpp"
#include "Linea/Decompositions/SVD.hpp"

// IO
#include "Linea/IO/StreamOperations.hpp"

#endif // LINEA_H