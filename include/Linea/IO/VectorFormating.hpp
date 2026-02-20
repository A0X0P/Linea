
/**
 * @file VectorFormating.hpp
 * @author A.N. Prosper
 * @date January 28th 2026
 * @brief Formatting configuration for vector stream output.
 *
 * Defines presentation parameters used by Linea::IO::display()
 * when printing vectors.
 */

#pragma once

#include <cstddef>

namespace Linea::IO {

/**
 * @struct VectorFormat
 *
 * Controls textual representation of Vector<T>.
 *
 * Members:
 *      precision   Number of digits after decimal point.
 *      width       Minimum field width per element.
 *      scientific  Enables scientific notation.
 *      horizontal  If true:
 *                      [a, b, c]
 *                  If false:
 *                      [a]
 *                      [b]
 *                      [c]
 *
 * Stream state is preserved after output.
 */

struct VectorFormat {

  std::size_t precision = 2;
  std::size_t width = 2;
  bool scientific = false;
  bool horizontal = true;
};

}; // namespace Linea::IO
