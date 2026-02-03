// created by : A.N. Prosper
// date : January 24th 2026
// time : 20:09

#ifndef LINEA_SVD_HPP
#define LINEA_SVD_HPP

#include "../Core/Concepts.hpp"
#include "../Core/PlatformMacros.hpp"
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
  void compute(const Matrix<Tp> &matrix, ComputeMode /*mode*/) {
    const std::size_t kmax = std::min(row, col);

    Matrix<Tp> B = matrix;

    U_matrix = Matrix<Tp>::Identity(row);
    V_matrix = Matrix<Tp>::Identity(col);
    singular_values = Vector<Tp>(kmax);

    Tp *RESTRICT Bp = B.raw();
    Tp *RESTRICT Up = U_matrix.raw();
    Tp *RESTRICT Vp = V_matrix.raw();

    // --- Bidiagonalization ---
    for (std::size_t k = 0; k < kmax; ++k) {

      // Left Householder
      Tp sigma = 0;
      for (std::size_t i = k; i < row; ++i) {
        Tp v = Bp[i * col + k];
        sigma += v * v;
      }

      if (sigma > tolerance_) {
        Tp x0 = Bp[k * col + k];
        Tp normx = std::sqrt(sigma);
        Tp vk = (x0 >= 0) ? x0 + normx : x0 - normx;
        Tp beta = Tp(2) / (sigma - x0 * x0 + vk * vk);

        Bp[k * col + k] = -((x0 >= 0) ? normx : -normx);
        for (std::size_t i = k + 1; i < row; ++i)
          Bp[i * col + k] /= vk;

        for (std::size_t j = k + 1; j < col; ++j) {
          Tp dot = Bp[k * col + j];
          for (std::size_t i = k + 1; i < row; ++i)
            dot += Bp[i * col + k] * Bp[i * col + j];
          dot *= beta;

          Bp[k * col + j] -= dot;
          for (std::size_t i = k + 1; i < row; ++i)
            Bp[i * col + j] -= Bp[i * col + k] * dot;
        }

        // Accumulate U
        for (std::size_t j = 0; j < row; ++j) {
          Tp dot = Up[k * row + j];
          for (std::size_t i = k + 1; i < row; ++i)
            dot += Bp[i * col + k] * Up[i * row + j];
          dot *= beta;

          Up[k * row + j] -= dot;
          for (std::size_t i = k + 1; i < row; ++i)
            Up[i * row + j] -= Bp[i * col + k] * dot;
        }
      }

      if (k + 1 >= col)
        continue;

      // Right Householder
      sigma = 0;
      for (std::size_t j = k + 1; j < col; ++j) {
        Tp v = Bp[k * col + j];
        sigma += v * v;
      }

      if (sigma > tolerance_) {
        Tp x0 = Bp[k * col + k + 1];
        Tp normx = std::sqrt(sigma);
        Tp vk = (x0 >= 0) ? x0 + normx : x0 - normx;
        Tp beta = Tp(2) / (sigma - x0 * x0 + vk * vk);

        Bp[k * col + k + 1] = -((x0 >= 0) ? normx : -normx);
        for (std::size_t j = k + 2; j < col; ++j)
          Bp[k * col + j] /= vk;

        for (std::size_t i = k + 1; i < row; ++i) {
          Tp dot = Bp[i * col + k + 1];
          for (std::size_t j = k + 2; j < col; ++j)
            dot += Bp[k * col + j] * Bp[i * col + j];
          dot *= beta;

          Bp[i * col + k + 1] -= dot;
          for (std::size_t j = k + 2; j < col; ++j)
            Bp[i * col + j] -= Bp[k * col + j] * dot;
        }

        // Accumulate V
        for (std::size_t j = 0; j < col; ++j) {
          Tp dot = Vp[(k + 1) * col + j];
          for (std::size_t i = k + 2; i < col; ++i)
            dot += Bp[k * col + i] * Vp[i * col + j];
          dot *= beta;

          Vp[(k + 1) * col + j] -= dot;
          for (std::size_t i = k + 2; i < col; ++i)
            Vp[i * col + j] -= Bp[k * col + i] * dot;
        }
      }
    }

    // --- QR iteration ---
    for (std::size_t iter = 0; iter < maxIterations_; ++iter) {
      bool converged = true;
      for (std::size_t i = 0; i + 1 < kmax; ++i)
        if (std::abs(Bp[i * col + i + 1]) > tolerance_)
          converged = false;

      if (converged)
        break;

      for (std::size_t i = 0; i + 1 < kmax; ++i) {
        Tp a = Bp[i * col + i];
        Tp b = Bp[i * col + i + 1];
        Tp r = std::hypot(a, b);
        Tp c = a / r;
        Tp s = -b / r;

        Bp[i * col + i] = r;
        Bp[i * col + i + 1] = 0;

        for (std::size_t j = i + 1; j < kmax; ++j) {
          Tp t1 = c * Bp[j * col + i] - s * Bp[j * col + i + 1];
          Tp t2 = s * Bp[j * col + i] + c * Bp[j * col + i + 1];
          Bp[j * col + i] = t1;
          Bp[j * col + i + 1] = t2;
        }
      }
    }

    for (std::size_t i = 0; i < kmax; ++i)
      singular_values[i] = std::abs(Bp[i * col + i]);
  }
};

} // namespace Linea::Decompositions

#endif
