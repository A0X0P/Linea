// created by : A.N. Prosper
// date : January 24th 2026
// time : 20:09

#ifndef LINEA_SVD_HPP
#define LINEA_SVD_HPP

#include "../Core/Concepts.hpp"
#include "../Core/Types.hpp"
#include "../Matrix/Matrix.hpp"
#include "../Vector/Vector.hpp"
#include <cmath>

namespace Linea::Decompositions {

template <RealType Tp> class SVD {
private:
  std::size_t row, col;
  Tp tolerance_;
  std::size_t maxIterations_;
  Matrix<Tp> U_matrix;
  Matrix<Tp> V_matrix;
  Vector<Tp> singular_values;

public:
  SVD(const Matrix<Tp> &matrix, ComputeMode mode = ComputeMode::Thin,
      Tp tolerance = 1e-6, std::size_t maxIters = 100)
      : row(matrix.nrows()), col(matrix.ncols()), tolerance_(tolerance),
        maxIterations_(maxIters), U_matrix(row, row), V_matrix(col, col),
        singular_values(std::min(row, col)) {

    compute(matrix, mode);
  }

  const Matrix<Tp> &U() const noexcept { return U_matrix; }
  const Matrix<Tp> &V() const noexcept { return V_matrix; }
  const Vector<Tp> &singularValues() const noexcept { return singular_values; }

private:
  //--- Golub–Kahan bidiagonal + QR iteration ---//

  void compute(const Matrix<Tp> &matrix, ComputeMode mode) {
    std::size_t kmax = std::min(row, col);
    Matrix<Tp> B = matrix;

    U_matrix = Matrix<Tp>::Identity(row);
    V_matrix = Matrix<Tp>::Identity(col);
    singular_values = Vector<Tp>(kmax);

    // --- Bidiagonalization ---//
    for (std::size_t k = 0; k < kmax; ++k) {

      // Left Householder (column k)
      Tp sigma = 0;
      for (std::size_t i = k; i < row; ++i)
        sigma += B(i, k) * B(i, k);

      if (sigma > tolerance_) {
        Tp x0 = B(k, k);
        Tp normx = std::sqrt(sigma);
        Tp vk = (x0 >= 0) ? x0 + normx : x0 - normx;
        Tp beta = Tp(2) / (sigma - x0 * x0 + vk * vk);

        B(k, k) = -((x0 >= 0) ? normx : -normx);
        for (std::size_t i = k + 1; i < row; ++i)
          B(i, k) /= vk;

        // Apply to B
        for (std::size_t j = k + 1; j < col; ++j) {
          Tp dot = B(k, j);
          for (std::size_t i = k + 1; i < row; ++i)
            dot += B(i, k) * B(i, j);
          dot *= beta;

          B(k, j) -= dot;
          for (std::size_t i = k + 1; i < row; ++i)
            B(i, j) -= B(i, k) * dot;
        }

        // Accumulate U
        for (std::size_t j = 0; j < row; ++j) {
          Tp dot = U_matrix(k, j);
          for (std::size_t i = k + 1; i < row; ++i)
            dot += B(i, k) * U_matrix(i, j);
          dot *= beta;

          U_matrix(k, j) -= dot;
          for (std::size_t i = k + 1; i < row; ++i)
            U_matrix(i, j) -= B(i, k) * dot;
        }
      }

      if (k + 1 >= col)
        continue;

      // Right Householder (row k)
      sigma = 0;
      for (std::size_t j = k + 1; j < col; ++j)
        sigma += B(k, j) * B(k, j);

      if (sigma > tolerance_) {
        Tp x0 = B(k, k + 1);
        Tp normx = std::sqrt(sigma);
        Tp vk = (x0 >= 0) ? x0 + normx : x0 - normx;
        Tp beta = Tp(2) / (sigma - x0 * x0 + vk * vk);

        B(k, k + 1) = -((x0 >= 0) ? normx : -normx);
        for (std::size_t j = k + 2; j < col; ++j)
          B(k, j) /= vk;

        for (std::size_t i = k + 1; i < row; ++i) {
          Tp dot = B(i, k + 1);
          for (std::size_t j = k + 2; j < col; ++j)
            dot += B(k, j) * B(i, j);
          dot *= beta;

          B(i, k + 1) -= dot;
          for (std::size_t j = k + 2; j < col; ++j)
            B(i, j) -= B(k, j) * dot;
        }

        // Accumulate V
        for (std::size_t j = 0; j < col; ++j) {
          Tp dot = V_matrix(k + 1, j);
          for (std::size_t i = k + 2; i < col; ++i)
            dot += B(k, i) * V_matrix(i, j);
          dot *= beta;

          V_matrix(k + 1, j) -= dot;
          for (std::size_t i = k + 2; i < row; ++i)
            V_matrix(i, j) -= B(k, i) * dot;
        }
      }
    }

    // --- QR iteration on bidiagonal ---//
    for (std::size_t iter = 0; iter < maxIterations_; ++iter) {
      bool converged = true;
      for (std::size_t i = 0; i + 1 < kmax; ++i)
        if (std::abs(B(i, i + 1)) > tolerance_) {
          converged = false;
        }

      if (converged) {
        break;
      }

      for (std::size_t i = 0; i + 1 < kmax; ++i) {
        Tp a = B(i, i);
        Tp b = B(i, i + 1);
        Tp r = std::hypot(a, b);
        Tp c = a / r;
        Tp s = -b / r;

        B(i, i) = r;
        B(i, i + 1) = 0;

        for (std::size_t j = i + 1; j < kmax; ++j) {
          Tp t1 = c * B(j, i) - s * B(j, i + 1);
          Tp t2 = s * B(j, i) + c * B(j, i + 1);
          B(j, i) = t1;
          B(j, i + 1) = t2;
        }
      }
    }

    for (std::size_t i = 0; i < kmax; ++i) {
      singular_values[i] = std::abs(B(i, i));
    }
  }
};
} // namespace Linea::Decompositions

#endif
