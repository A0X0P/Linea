
/**
 * @file MatrixFormating.hpp
 * @author A.N. Prosper
 * @date January 28th 2026
 * @brief Formatting configuration for matrix stream output.
 *
 * Defines presentation parameters used when displaying matrices
 * through Linea::IO::display().
 *
 * This structure does not affect internal matrix storage,
 * only stream formatting.
 *
 * Default Behavior:
 *      - Fixed-point notation
 *      - Precision = 2
 *      - Width = 2
 *      - Dimensions hidden
 */

#pragma once

#include <cstddef>

namespace Linea::IO {

/**
 * @struct MatrixFormat
 *
 * Controls textual representation of Matrix<T>.
 *
 * Members:
 *      precision       Number of digits after decimal point.
 *      width           Minimum field width per element.
 *      scientific      If true → scientific notation.
 *                      Otherwise → fixed notation.
 *      show_dimensions If true, prints:
 *                          "Matrix dimensions: RxC"
 *
 * Notes:
 *      - Width applies to each element individually.
 *      - Formatting state is restored after display().
 */

struct MatrixFormat {

  std::size_t precision = 2;
  std::size_t width = 2;
  bool scientific = false;
  bool show_dimensions = false;
};

}; // namespace Linea::IO