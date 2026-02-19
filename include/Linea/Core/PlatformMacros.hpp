

/**
 * @file PlatformMacros.hpp
 * @author A.N. Prosper
 * @date Febuary 3rd 2026
 * @brief Compiler abstraction utilities.
 *
 * @details
 * This file provides portability macros for performance-critical optimizations.
 *
 * Currently defines:
 *   RESTRICT — a compiler-specific restrict qualifier abstraction.
 *
 * The RESTRICT macro informs the compiler that pointers do not alias,
 * enabling improved vectorization and instruction scheduling.
 */

#ifndef PLATFORM_MACROS_H
#define PLATFORM_MACROS_H

/**
 * @def RESTRICT
 * @brief Compiler-specific restrict qualifier.
 *
 * Expands to:
 *   - __restrict__ (GCC/Clang)
 *   - __restrict (MSVC)
 *   - empty otherwise
 *
 * Used in performance-critical loops involving raw pointer arithmetic.
 */

#if defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#elif defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

#endif // PLATFORM_MACROS_H
