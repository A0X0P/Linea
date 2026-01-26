// created by : A.N. Prosper
// date : January 24th 2026
// time : 20:11

#ifndef LINEA_CHOLESKY_H
#define LINEA_CHOLESKY_H

#include "../Matrix/Matrix.hpp"
#include <cmath>
#include <stdexcept>

namespace Linea {

template <RealType C> struct CholeskyResult {
  Matrix<C> L_matrix;
};

template <RealType C> class Cholesky {
private:
  Matrix<C> data;

public:
  Cholesky(const Matrix<C> &matrix) : data(matrix) {
    if (matrix.nrows() != matrix.ncols()) {
      throw std::invalid_argument(
          "Cholesky decomposition requires a square matrix.");
    }
  }

  CholeskyResult<C> decompose() {
    int n = data.nrows();
    Matrix<C> L(n, n);

    for (int i = 0; i < n; i++) {
      for (int j = 0; j <= i; j++) {
        C sum = 0;
        for (int k = 0; k < j; k++) {
          sum += L(i, k) * L(j, k);
        }

        if (i == j) {
          C val = data(i, i) - sum;
          if (val <= 0) {
            throw std::runtime_error("Matrix is not positive-definite.");
          }
          using std::sqrt;
          L(i, j) = sqrt(val);
        } else {
          L(i, j) = (data(i, j) - sum) / L(j, j);
        }
      }
    }
    return CholeskyResult<C>{.L_matrix = L};
  }
};

} // namespace Linea
#endif