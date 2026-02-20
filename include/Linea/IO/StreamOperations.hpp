
/**
 * @file StreamOperations.hpp
 * @author A.N. Prosper
 * @date January 28th 2026
 * @brief Stream output utilities for Matrix and Vector.
 *
 * Provides formatted output via:
 *
 *      display(std::ostream&, const Matrix<T>&, MatrixFormat)
 *      display(std::ostream&, const Vector<T>&, VectorFormat)
 *
 * Design Goals:
 *      - Non-intrusive (no operator<< override required)
 *      - Formatting configurable per-call
 *      - Stream state preservation
 *
 * Complexity:
 *      Matrix: O(n*m)
 *      Vector: O(n)
 */

#pragma once

#include <iomanip>
#include <ostream>

#include "../Matrix/Matrix.hpp"
#include "../Vector/Vector.hpp"
#include "MatrixFormating.hpp"
#include "VectorFormating.hpp"

namespace Linea::IO {

/**
 * @brief Displays a matrix using configurable formatting.
 *
 * Format:
 *      [ a11, a12, ... ]
 *      [ a21, a22, ... ]
 *
 * If show_dimensions = true:
 *      Prints matrix size before content.
 *
 * Stream Safety:
 *      - Saves original flags and precision
 *      - Restores them before returning
 *
 * @tparam F Numeric scalar type.
 * @param os Output stream.
 * @param matrix Matrix to display.
 * @param fmt Formatting configuration.
 * @return Reference to output stream.
 *
 * Time Complexity:
 *      O(nrows * ncols)
 */

template <NumericType F>
std::ostream &display(std::ostream &os, const Matrix<F> &matrix,
                      const MatrixFormat &fmt = {}) {
  auto flags = os.flags();
  auto prec = os.precision();

  if (fmt.show_dimensions) {
    os << "Matrix dimensions: " << matrix.nrows() << "x" << matrix.ncols()
       << "\n";
  }

  if (fmt.scientific) {
    os << std::scientific;
  } else {
    os << std::fixed;
  }

  const std::size_t R = matrix.nrows();
  const std::size_t C = matrix.ncols();

  os << std::setprecision(fmt.precision) << std::right << std::showpoint;
  for (std::size_t i{}; i < R; i++) {
    os << "[";
    for (std::size_t j{}; j < C; j++) {
      os << std::setw(fmt.width) << matrix(i, j);
      if (j < C - 1) {
        os << ", ";
      }
    }
    os << "]";
    if (i < R - 1) {
      os << "\n";
    }
  }

  os.flags(flags);
  os.precision(prec);
  return os;
}

/**
 * @brief Displays a vector using configurable formatting.
 *
 * Horizontal mode:
 *      [ v1, v2, v3 ]
 *
 * Vertical mode:
 *      [ v1 ]
 *      [ v2 ]
 *      [ v3 ]
 *
 * Stream Safety:
 *      Preserves original formatting state.
 *
 * @tparam F Numeric scalar type.
 * @param os Output stream.
 * @param vector Vector to display.
 * @param fmt Formatting configuration.
 * @return Reference to output stream.
 *
 * Time Complexity:
 *      O(n)
 */

template <NumericType F>
std::ostream &display(std::ostream &os, const Vector<F> &vector,
                      const VectorFormat &fmt = {}) {
  auto flags = os.flags();
  auto prec = os.precision();

  if (fmt.scientific) {
    os << std::scientific;
  } else {
    os << std::fixed;
  }

  if (fmt.horizontal) {
    os << std::setprecision(fmt.precision) << std::right << std::showpoint
       << "[";
    for (std::size_t i = 0; i < vector.size(); ++i) {
      os << std::setw(fmt.width) << vector[i];
      if (i + 1 < vector.size()) {
        os << ", ";
      }
    }
    os << "]\n";
  } else {
    os << std::setprecision(fmt.precision) << std::right << std::showpoint;
    for (std::size_t i = 0; i < vector.size(); ++i) {
      os << "[" << std::setw(fmt.width) << vector[i] << "]\n";
    }
  }

  os.flags(flags);
  os.precision(prec);
  return os;
}

}; // namespace Linea::IO