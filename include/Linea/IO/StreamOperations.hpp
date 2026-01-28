// created by : A.N. Prosper
// date : January 28th 2026
// time : 16:21

#pragma once

#include <iomanip>
#include <ostream>

#include "../Matrix/Matrix.hpp"
#include "../Vector/Vector.hpp"
#include "MatrixFormating.hpp"
#include "VectorFormating.hpp"

namespace Linea::IO {

template <NumericType F>
std::ostream &display(std::ostream &os, const Matrix<F> &matrix,
                      MatrixFormat fmt = {}) {
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

template <NumericType F>
std::ostream &display(std::ostream &os, const Vector<F> &vector,
                      VectorFormat fmt = {}) {
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
      os << std::setw(fmt.width) << vector[i] << '\n';
    }
  }

  os.flags(flags);
  os.precision(prec);
  return os;
}

}; // namespace Linea::IO