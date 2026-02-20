
/**
 * @file Cholesky.hpp
 * @author A.N. Prosper
 * @date January 24th 2026
 * @brief Cholesky factorization for symmetric positive-definite matrices.
 *
 * Computes the factorization:
 *
 *      A = L Lᵀ
 *
 * where:
 *      - A ∈ ℝ^{n×n}
 *      - L is lower triangular with positive diagonal entries
 *
 * Requirements:
 *      - A must be symmetric positive-definite (SPD)
 *
 * Time Complexity:
 *      O(n³ / 3)
 *
 * Space Complexity:
 *      O(n²)
 *
 * Numerical Properties:
 *      - Backward stable for well-conditioned SPD matrices
 *      - Fails if matrix is not positive-definite
 *
 * Applications:
 *      - Solving linear systems
 *      - Least squares via normal equations
 *      - Gaussian process regression
 */

#ifndef LINEA_CHOLESKY_HPP
#define LINEA_CHOLESKY_HPP

#include "../Core/PlatformMacros.hpp"
#include "../Matrix/Matrix.hpp"
#include <cmath>
#include <stdexcept>

namespace Linea::Decompositions {

/**
 * @tparam C Floating-point scalar type.
 *
 * @class Cholesky
 *
 * Performs dense Cholesky decomposition using the classical
 * outer-product algorithm.
 *
 * The decomposition overwrites no external data.
 *
 * Exception Safety:
 *      - Throws std::invalid_argument if matrix not square
 *      - Throws std::runtime_error if matrix not SPD
 */

template <RealType C> class Cholesky {
private:
  Matrix<C> L_;

public:
  explicit Cholesky(const Matrix<C> &A) : L_(A.nrows(), A.ncols()) {
    if (A.nrows() != A.ncols()) {
      throw std::invalid_argument("Cholesky requires square matrix");
    }
    decompose(A);
  }

  /**
   * @brief Returns a const reference to the lower triangular matrix L.
   *
   * Provides read-only access to the lower triangular factor obtained
   * from the Cholesky decomposition.
   *
   * For a symmetric positive-definite matrix A, the decomposition satisfies:
   *
   *      A = L * Lᵀ
   *
   * where L is a lower triangular matrix.
   *
   * @return Const reference to the internally stored lower triangular matrix L.
   *
   * @note The returned reference remains valid as long as the Cholesky
   *       object exists and is not modified.
   */
  const Matrix<C> &L() const { return L_; }

  /**
   * @brief Solves Ax = b using stored factorization.
   *
   * Performs:
   *      1. Forward substitution: Ly = b
   *      2. Backward substitution: Lᵀx = y
   *
   * Complexity:
   *      O(n²)
   *
   * @param b Right-hand side vector.
   * @return Solution vector x.
   */

  Vector<C> solve(const Vector<C> &b) const {
    Vector<C> y = forward_substitution(b);
    return backward_substitution(y);
  }

  /**
   * @brief Solves least squares problem:
   *
   *      min ||Ax - b||₂
   *
   * using normal equations:
   *
   *      (AᵀA)x = Aᵀb
   *
   * Warning:
   *      Normal equations square the condition number:
   *      κ(AᵀA) = κ(A)²
   *
   * Prefer QR for ill-conditioned problems.
   */

  static Vector<C> least_squares(const Matrix<C> &A, const Vector<C> &b) {
    Matrix<C> At = A.transpose();
    Matrix<C> normal = At * A;
    Vector<C> rhs = At * b;

    Cholesky<C> chol(normal);
    return chol.solve(rhs);
  }

private:
  /**
   * @brief Computes the Cholesky factorization of a symmetric
   *        positive-definite matrix A.
   *
   * Performs an in-place Cholesky decomposition:
   *
   *      A = L Lᵀ
   *
   * where L is a lower triangular matrix with strictly positive diagonal.
   *
   * The algorithm follows the classical left-looking formulation:
   *
   *      L(i,j) = (A(i,j) - Σ_{k=0}^{j-1} L(i,k)L(j,k)) / L(j,j),  i > j
   *      L(i,i) = sqrt(A(i,i) - Σ_{k=0}^{i-1} L(i,k)^2)
   *
   * Memory is accessed through raw contiguous storage for improved
   * cache locality and to enable compiler vectorization (via RESTRICT).
   *
   * @param A Input matrix. Must be symmetric positive-definite.
   *
   * @throws std::runtime_error
   *         If the matrix is not positive-definite (i.e., a non-positive
   *         pivot is encountered during factorization).
   *
   * @note Time complexity: O(n^3 / 3)
   * @note No pivoting is performed.
   * @note Only the lower triangular part of A is referenced.
   */

  void decompose(const Matrix<C> &A) {
    const std::size_t n = A.nrows();

    L_ = Matrix<C>(n, n, C(0));

    C *RESTRICT Lp = L_.raw();
    const C *RESTRICT Ap = A.raw();

    const std::size_t lda = A.ncols();

    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {

        C sum = 0;
        for (std::size_t k = 0; k < j; ++k) {
          sum += Lp[i * n + k] * Lp[j * n + k];
        }

        if (i == j) {
          C val = Ap[i * lda + i] - sum;
          if (val <= C(0))
            throw std::runtime_error("Matrix not positive-definite");
          Lp[i * n + j] = std::sqrt(val);
        } else {
          Lp[i * n + j] = (Ap[i * lda + j] - sum) / Lp[j * n + j];
        }
      }
    }
  }

  /**
   * @brief Solves the lower triangular system L y = b.
   *
   * Performs forward substitution using the previously computed
   * Cholesky factor L.
   *
   * Algorithm:
   *
   *      y(i) = (b(i) - Σ_{j=0}^{i-1} L(i,j) y(j)) / L(i,i)
   *
   * @param b Right-hand side vector.
   * @return Solution vector y.
   *
   * @note Time complexity: O(n^2)
   * @note Assumes L is non-singular with strictly positive diagonal.
   */

  Vector<C> forward_substitution(const Vector<C> &b) const {
    const std::size_t n = L_.nrows();
    Vector<C> y(n);

    const C *RESTRICT Lp = L_.raw();

    for (std::size_t i = 0; i < n; ++i) {
      C sum = 0;
      for (std::size_t j = 0; j < i; ++j) {
        sum += Lp[i * n + j] * y(j);
      }
      y(i) = (b(i) - sum) / Lp[i * n + i];
    }
    return y;
  }

  /**
   * @brief Solves the upper triangular system Lᵀ x = y.
   *
   * Performs backward substitution on the transpose of the
   * Cholesky factor.
   *
   * Algorithm:
   *
   *      x(i) = (y(i) - Σ_{j=i+1}^{n-1} L(j,i) x(j)) / L(i,i)
   *
   * @param y Right-hand side vector (typically output of forward_substitution).
   * @return Solution vector x.
   *
   * @note Time complexity: O(n^2)
   * @note Assumes L has strictly positive diagonal entries.
   */

  Vector<C> backward_substitution(const Vector<C> &y) const {
    const std::size_t n = L_.nrows();
    Vector<C> x(n);

    const C *RESTRICT Lp = L_.raw();

    for (int i = int(n) - 1; i >= 0; --i) {
      C sum = 0;
      for (std::size_t j = i + 1; j < n; ++j) {
        sum += Lp[j * n + i] * x(j);
      }
      x(i) = (y(i) - sum) / Lp[i * n + i];
    }
    return x;
  }
};

} // namespace Linea::Decompositions
#endif