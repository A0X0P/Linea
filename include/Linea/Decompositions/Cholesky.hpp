// created by : A.N. Prosper
// date : January 24th 2026
// time : 20:11

#ifndef LINEA_CHOLESKY_HPP
#define LINEA_CHOLESKY_HPP

#include "../Matrix/Matrix.hpp"
#include <cmath>
#include <stdexcept>

namespace Linea::Decompositions {

template <RealType C> class Cholesky {
private:
  Matrix<C> L_;

public:
  explicit Cholesky(const Matrix<C> &A) : L_(Matrix<C>(A.nrows(), A.ncols())) {
    if (A.nrows() != A.ncols()) {
      throw std::invalid_argument("Cholesky requires square matrix");
    }
    decompose(A);
  }

  // Access L
  const Matrix<C> &L() const { return L_; }

  // --- Solve Ax = b  (A = LLᵀ) ---//
  Vector<C> solve(const Vector<C> &b) const {
    Vector<C> y = forward_substitution(b);
    return backward_substitution(y);
  }

  // -- Least squares: min ||Ax - b||₂ Solves (AᵀA)x = Aᵀb via Cholesky --//
  static Vector<C> least_squares(const Matrix<C> &A, const Vector<C> &b) {
    Matrix<C> At = A.Transpose();
    Matrix<C> normal = At * A;
    Vector<C> rhs = At * b;

    Cholesky<C> chol(normal);
    return chol.solve(rhs);
  }

private:
  void decompose(const Matrix<C> &A) {
    std::size_t n = A.nrows();
    L_ = Matrix<C>(n, n, C(0));

    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        C sum = 0;
        for (std::size_t k = 0; k < j; ++k)
          sum += L_(i, k) * L_(j, k);

        if (i == j) {
          C val = A(i, i) - sum;
          if (val <= 0)
            throw std::runtime_error("Matrix not positive-definite");
          L_(i, j) = std::sqrt(val);
        } else {
          L_(i, j) = (A(i, j) - sum) / L_(j, j);
        }
      }
    }
  }

  // Forward solve: Ly = b
  Vector<C> forward_substitution(const Vector<C> &b) const {
    std::size_t n = L_.nrows();
    Vector<C> y(n);

    for (std::size_t i = 0; i < n; ++i) {
      C sum = 0;
      for (std::size_t j = 0; j < i; ++j)
        sum += L_(i, j) * y(j);
      y(i) = (b(i) - sum) / L_(i, i);
    }
    return y;
  }

  // Backward solve: Lᵀx = y
  Vector<C> backward_substitution(const Vector<C> &y) const {
    std::size_t n = L_.nrows();
    Vector<C> x(n);

    for (int i = int(n) - 1; i >= 0; --i) {
      C sum = 0;
      for (std::size_t j = i + 1; j < n; ++j)
        sum += L_(j, i) * x(j);
      x(i) = (y(i) - sum) / L_(i, i);
    }
    return x;
  }
};

} // namespace Linea::Decompositions
#endif